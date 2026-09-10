# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


@triton.jit
def _sgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr,
    SKIP_FULL: tl.constexpr,
    FULL_GRID_M: tl.constexpr,
    FULL_GRID_N: tl.constexpr,
    N_MAJOR_ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m
        pid_m = pid % grid_m
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

    if SKIP_FULL and pid_m < FULL_GRID_M and pid_n < FULL_GRID_N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if CHECK_BOUNDS:
        is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
        is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K

        for ki in range(k_full_iters):
            offs_k = ki * BLOCK_K + offs_k_base
            if TRANS_A:
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
            else:
                a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

            if is_full_m and is_full_n:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_m[:, None] < m, other=0.0)
                b = tl.load(b_ptrs, mask=offs_n[None, :] < n, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

        if k_remainder > 0:
            offs_k = k_full_iters * BLOCK_K + offs_k_base
            if TRANS_A:
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
            else:
                a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
            b_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            if TRANS_A:
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
            else:
                a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if CHECK_BOUNDS:
        c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32), mask=c_mask)
    else:
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32))


@triton.jit
def _sgemm_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if CHECK_BOUNDS:
        is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
        is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for ki in range(k_full_iters):
            offs_k = ki * BLOCK_K + offs_k_base
            a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            if is_full_m and is_full_n:
                a_t = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a_t = tl.load(a_ptrs, mask=offs_m[None, :] < m, other=0.0)
                b = tl.load(b_ptrs, mask=offs_n[None, :] < n, other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32, allow_tf32=False)
        if k_remainder > 0:
            offs_k = k_full_iters * BLOCK_K + offs_k_base
            a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            mask_k = offs_k < k
            a_t = tl.load(
                a_ptrs, mask=(mask_k[:, None] & (offs_m[None, :] < m)), other=0.0
            )
            b = tl.load(
                b_ptrs, mask=(mask_k[:, None] & (offs_n[None, :] < n)), other=0.0
            )
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a_t = tl.load(a_ptr + offs_k[:, None] * lda + offs_m[None, :])
            b = tl.load(b_ptr + offs_k[:, None] * ldb + offs_n[None, :])
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if CHECK_BOUNDS:
        c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32), mask=c_mask)
    else:
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32))


@triton.jit
def _sgemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if CHECK_BOUNDS:
        is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
        is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for ki in range(k_full_iters):
            offs_k = ki * BLOCK_K + offs_k_base
            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
            if is_full_m and is_full_n:
                a = tl.load(a_ptrs)
                b_t = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_m[:, None] < m, other=0.0)
                b_t = tl.load(b_ptrs, mask=offs_n[:, None] < n, other=0.0)
            acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32, allow_tf32=False)
        if k_remainder > 0:
            offs_k = k_full_iters * BLOCK_K + offs_k_base
            mask_k = offs_k < k
            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
            a = tl.load(
                a_ptrs, mask=((offs_m[:, None] < m) & mask_k[None, :]), other=0.0
            )
            b_t = tl.load(
                b_ptrs, mask=((offs_n[:, None] < n) & mask_k[None, :]), other=0.0
            )
            acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a = tl.load(a_ptr + offs_m[:, None] * lda + offs_k[None, :])
            b_t = tl.load(b_ptr + offs_n[:, None] * ldb + offs_k[None, :])
            acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if CHECK_BOUNDS:
        c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32), mask=c_mask)
    else:
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32))


@triton.jit
def _sgemm_tt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    if CHECK_BOUNDS:
        is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
        is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for ki in range(k_full_iters):
            offs_k = ki * BLOCK_K + offs_k_base
            a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
            b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
            if is_full_m and is_full_n:
                a_t = tl.load(a_ptrs)
                b_t = tl.load(b_ptrs)
            else:
                a_t = tl.load(a_ptrs, mask=offs_m[None, :] < m, other=0.0)
                b_t = tl.load(b_ptrs, mask=offs_n[:, None] < n, other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        if k_remainder > 0:
            offs_k = k_full_iters * BLOCK_K + offs_k_base
            mask_k = offs_k < k
            a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
            b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
            a_t = tl.load(
                a_ptrs, mask=(mask_k[:, None] & (offs_m[None, :] < m)), other=0.0
            )
            b_t = tl.load(
                b_ptrs, mask=((offs_n[:, None] < n) & mask_k[None, :]), other=0.0
            )
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    else:
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a_t = tl.load(a_ptr + offs_k[:, None] * lda + offs_m[None, :])
            b_t = tl.load(b_ptr + offs_n[:, None] * ldb + offs_k[None, :])
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if CHECK_BOUNDS:
        c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32), mask=c_mask)
    else:
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float32))


@triton.jit
def _sgemm_k_tail_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    K_OFFSET: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = K_OFFSET + tl.arange(0, BLOCK_K)

    if TRANS_A:
        a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
    else:
        a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    if TRANS_B:
        b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
    else:
        b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    a = tl.load(a_ptrs, mask=offs_k[None, :] < k, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < k, other=0.0)
    acc = tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = tl.load(c_ptrs).to(tl.float32) + alpha * acc
    tl.store(c_ptrs, result.to(tl.float32))


def _select_sgemm_config(m: int, n: int, k: int, transa: int, transb: int):
    if (
        transa == CUBLAS_OP_T
        and transb == CUBLAS_OP_T
        and m == 4096
        and n == 8192
        and k == 28672
    ):
        return 128, 128, 64, 16, 4
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_T and m == 128 and n >= 1024:
        return 128, 128, 64, 16, 4
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_T and n == 128 and m >= 1024:
        return 128, 128, 64, 16, 4
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_T
        and m == 2048
        and n == 2048
        and k >= 8192
    ):
        return 128, 128, 64, 16, 4
    if transb == CUBLAS_OP_T and m == 128 and n >= 1024:
        return 128, 128, 64, 16, 4
    if transb == CUBLAS_OP_T and n == 128 and m >= 1024:
        return 128, 128, 64, 16, 4
    if m == 64 and n == 64 and k == 64:
        return 32, 32, 64, 4, 8
    if min(m, n) <= 64:
        if k <= 512:
            return 32, 64, 128, 8, 8
        if m <= 64 and max(m, n, k) >= 4096:
            return 64, 64, 64, 8, 8
        if n <= 64 and max(m, n, k) >= 4096:
            return 64, 64, 64, 8, 8
        if m <= 64:
            return 32, 64, 64, 8, 8
        return 64, 32, 64, 8, 8
    if m == 256 and n == 256 and k == 256:
        return 64, 32, 128, 8, 8
    if m <= 512 and n <= 512 and k <= 512:
        return 64, 64, 64, 8, 8
    if m == 128 and n >= 1024:
        return 64, 64, 32, 4, 8
    if n == 128 and m >= 1024:
        return 64, 64, 64, 8, 4
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N and m == 4096 and n >= 8192:
        return 128, 128, 32, 8, 16
    return 128, 128, 32, 8, 8


def _can_use_fast_sgemm(
    m: int, n: int, k: int, block_m: int, block_n: int, block_k: int
) -> bool:
    return (m % block_m == 0) and (n % block_n == 0) and (k % block_k == 0)


def _launch_sgemm(
    transa: int,
    transb: int,
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    beta_is_zero: bool,
    check_bounds: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
) -> None:
    if (
        not check_bounds
        and transa == CUBLAS_OP_T
        and transb == CUBLAS_OP_T
        and (m == 4096 and n == 8192 and k == 28672)
    ):
        _sgemm_tt_kernel[grid](
            A,
            B,
            C,
            alpha,
            beta,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            beta_is_zero,
            False,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return
    _sgemm_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        transa == CUBLAS_OP_T,
        transb == CUBLAS_OP_T,
        check_bounds,
        False,
        0,
        0,
        False,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def sgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    block_m, block_n, block_k, num_warps, group_m = _select_sgemm_config(
        m, n, k, transa, transb
    )
    check_bounds = not _can_use_fast_sgemm(m, n, k, block_m, block_n, block_k)
    beta_is_zero = beta == 0.0
    trans_a = transa == CUBLAS_OP_T
    trans_b = transb == CUBLAS_OP_T

    num_stages = (
        2 if (m == n == k and max(m, n, k) >= 4096) or (n == 128 and m >= 1024) else 3
    )

    with torch_device_fn.device(A.device):
        if check_bounds and max(m, n, k) >= 2048:
            padded_m = triton.cdiv(m, block_m) * block_m
            padded_n = triton.cdiv(n, block_n) * block_n
            padded_k = triton.cdiv(k, block_k) * block_k
            if transa == CUBLAS_OP_N:
                A_pad = F.pad(A, (0, padded_k - k, 0, padded_m - m))
                lda_pad = padded_k
            else:
                A_pad = F.pad(A, (0, padded_m - m, 0, padded_k - k))
                lda_pad = padded_m
            if transb == CUBLAS_OP_N:
                B_pad = F.pad(B, (0, padded_n - n, 0, padded_k - k))
                ldb_pad = padded_n
            else:
                B_pad = F.pad(B, (0, padded_k - k, 0, padded_n - n))
                ldb_pad = padded_k
            if beta_is_zero:
                C_pad = torch.empty(
                    (padded_m, padded_n), device=C.device, dtype=C.dtype
                )
            else:
                C_pad = F.pad(C, (0, padded_n - n, 0, padded_m - m))
            grid_pad = (
                triton.cdiv(padded_m, block_m) * triton.cdiv(padded_n, block_n),
            )
            _launch_sgemm(
                transa,
                transb,
                grid_pad,
                A_pad,
                B_pad,
                C_pad,
                alpha,
                beta,
                padded_m,
                padded_n,
                padded_k,
                lda_pad,
                ldb_pad,
                padded_n,
                beta_is_zero,
                False,
                block_m,
                block_n,
                block_k,
                num_warps,
                group_m,
                num_stages,
            )
            C.copy_(C_pad[:m, :n])
            return

        if check_bounds and max(m, n, k) >= 2048:
            full_m = (m // block_m) * block_m
            full_n = (n // block_n) * block_n
            full_k = (k // block_k) * block_k
            full_grid_m = full_m // block_m
            full_grid_n = full_n // block_n

            if full_m > 0 and full_n > 0 and full_k > 0:
                grid_full = (full_grid_m * full_grid_n,)
                _sgemm_kernel[grid_full](
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    full_m,
                    full_n,
                    full_k,
                    lda,
                    ldb,
                    ldc,
                    beta_is_zero,
                    trans_a,
                    trans_b,
                    False,
                    False,
                    0,
                    0,
                    False,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=group_m,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            if full_m > 0 and full_n > 0 and full_k < k:
                grid_tail = (full_grid_m * full_grid_n,)
                _sgemm_k_tail_kernel[grid_tail](
                    A,
                    B,
                    C,
                    alpha,
                    full_m,
                    full_n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    full_k,
                    trans_a,
                    trans_b,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=group_m,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            if full_m < m or full_n < n:
                grid_edge = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
                _sgemm_kernel[grid_edge](
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    beta_is_zero,
                    trans_a,
                    trans_b,
                    True,
                    True,
                    full_grid_m,
                    full_grid_n,
                    False,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=group_m,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            return

        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _launch_sgemm(
            transa,
            transb,
            grid,
            A,
            B,
            C,
            alpha,
            beta,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            beta_is_zero,
            check_bounds,
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            num_stages,
        )

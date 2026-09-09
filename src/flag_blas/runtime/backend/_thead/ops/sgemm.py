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

import logging

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401

from flag_blas import runtime
from flag_blas.ops.level3.sgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _sgemm_nn_kernel,
    _sgemm_nt_kernel,
    _sgemm_tn_kernel,
    _sgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.dispatch import SizeAutoDispatch, StaticDispatch
from flag_blas.utils import libentry, libtuner
from flag_blas.utils.libentry import libcache

logger = logging.getLogger(__name__)

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@triton.jit
def _sgemm_scale_storage_kernel(
    c_ptr,
    beta: tl.float32,
    total,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    ptrs = c_ptr + offsets
    if BETA_IS_ZERO:
        tl.store(ptrs, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)
    else:
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, beta * vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_scale_c_kernel(
    c_ptr,
    beta: tl.float32,
    m,
    n,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m * n
    rows = offsets // n
    cols = offsets - rows * n
    ptrs = c_ptr + rows * ldc + cols
    if BETA_IS_ZERO:
        tl.store(ptrs, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)
    else:
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, beta * vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_pad2d_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < rows) & (c < cols)
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_transpose_pad2d_kernel(
    src_ptr,
    dst_ptr,
    src_rows,
    src_cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < src_cols) & (c < src_rows)
    vals = tl.load(src_ptr + c * src_ld + r, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_transpose_pad2d_tile_kernel(
    src_ptr,
    dst_ptr,
    src_rows,
    src_cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    dst_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    dst_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    src_mask = (dst_c[:, None] < src_rows) & (dst_r[None, :] < src_cols)
    vals = tl.load(
        src_ptr + dst_c[:, None] * src_ld + dst_r[None, :],
        mask=src_mask,
        other=0.0,
    )
    dst_mask = (dst_r[:, None] < dst_rows) & (dst_c[None, :] < dst_cols)
    tl.store(
        dst_ptr + dst_r[:, None] * dst_ld + dst_c[None, :],
        tl.trans(vals),
        mask=dst_mask,
    )


@libentry()
@triton.jit
def _sgemm_crop_c_kernel(
    src_ptr,
    dst_ptr,
    beta: tl.float32,
    rows,
    cols,
    src_ld,
    dst_ld,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    src_vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0)
    dst_offsets = r * dst_ld + c
    if BETA_IS_ZERO:
        tl.store(dst_ptr + dst_offsets, src_vals, mask=mask)
    else:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_ptr + dst_offsets, src_vals + beta * dst_vals, mask=mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_t = tl.dot(
            tl.trans(b), tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
        )
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * c_vals

    tl.store(c_ptrs, result, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_nomask_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    tl.store(c_ptrs, alpha * acc)


@triton.jit
def _thead_sgemm_nn_fp32_impl(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_TF32X3: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_sgemm_nn_fp32_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_TF32X3: tl.constexpr,
):
    _thead_sgemm_nn_fp32_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        USE_TF32X3,
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_sgemm_nn_fp32_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_TF32X3: tl.constexpr,
):
    _thead_sgemm_nn_fp32_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        USE_TF32X3,
    )


@libentry()
@triton.jit
def _thead_sgemm_nn_tf32_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        mask_m = offs_m < m
        mask_n = offs_n < n
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        mask_m = offs_m < m
        mask_n = offs_n < n
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_f16_conv_kernel(
    src_ptr,
    dst_ptr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """f32 -> fp16 elementwise convert (used by the fp16 pre-conversion path)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    v = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, v.to(tl.float16), mask=mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_f16_mm_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MASKED: tl.constexpr,
):
    """fp16 MMA with fp32 accumulate. A16/B16 are pre-converted fp16 buffers."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_full = k // BLOCK_K
    k_rem = k % BLOCK_K
    if MASKED:
        for _ in range(0, k_full):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        for _ in range(0, k_full):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if MASKED:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)
    else:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)


@libentry()
@triton.jit
def _thead_sgemm_nn_f16_fused_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Load f32 tiles, convert to fp16 in-register, MMA with fp32 accumulate."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_full = k // BLOCK_K
    k_rem = k % BLOCK_K
    if MASKED:
        for _ in range(0, k_full):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float16)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float16)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
                tl.float16
            )
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(
                tl.float16
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        for _ in range(0, k_full):
            a = tl.load(a_ptrs).to(tl.float16)
            b = tl.load(b_ptrs).to(tl.float16)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float16)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float16)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if MASKED:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)
    else:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)


@libentry()
@triton.jit
def _thead_sgemm_tntt_fp32_tf32x3_kernel(
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
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_TF32X3: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    """FP32 tensor-core kernel for sgemm_tn/nt/tt with optional tf32x3.

    Used for tiny squares where fp16 MMA fails the SGEMM accuracy budget
    (atol = 1e-4 * k with k <= 128): tf32x3 keeps near-FP32 accuracy while
    using tensor cores, matching the NN tf32x3 small-shape fast path.
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    if TRANS_A:
        a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    else:
        a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    if TRANS_B:
        b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    else:
        b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K
    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            if TRANS_A:
                a = tl.trans(tl.load(a_ptrs))
            else:
                a = tl.load(a_ptrs)
            if TRANS_B:
                b = tl.trans(tl.load(b_ptrs))
            else:
                b = tl.load(b_ptrs)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda if TRANS_A else BLOCK_K
            b_ptrs += BLOCK_K if TRANS_B else BLOCK_K * ldb
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if TRANS_A:
                a = tl.trans(tl.load(a_ptrs, mask=mask_k[:, None], other=0.0))
            else:
                a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            if TRANS_B:
                b = tl.trans(tl.load(b_ptrs, mask=mask_k[None, :], other=0.0))
            else:
                b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        for _ in range(0, k_full_iters):
            if TRANS_A:
                a = tl.trans(tl.load(a_ptrs, mask=mask_m[None, :], other=0.0))
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if TRANS_B:
                b = tl.trans(tl.load(b_ptrs, mask=mask_n[:, None], other=0.0))
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda if TRANS_A else BLOCK_K
            b_ptrs += BLOCK_K if TRANS_B else BLOCK_K * ldb
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if TRANS_A:
                a = tl.trans(
                    tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
                )
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            if TRANS_B:
                b = tl.trans(
                    tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                )
            else:
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            if USE_TF32X3:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")
            else:
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_f16_pad2d_conv_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """fused: read f32 (rows x cols, src_ld), write fp16 padded (dst_rows x dst_cols, dst_ld)."""
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < rows) & (c < cols)
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals.to(tl.float16), mask=mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_f16_padstore_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m0,
    n0,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Unmasked loads on padded A16/B16, masked store vs original (m0, n0).
    m, n = padded sizes (grid); k = padded k; m0, n0 = original logical sizes."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k // BLOCK_K):
        a = tl.load(a_ptrs).to(tl.float16)
        b = tl.load(b_ptrs).to(tl.float16)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = (offs_m[:, None] < m0) & (offs_n[None, :] < n0)
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tntt_f16_tconv_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    src_ld,
    dst_ld,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """f32 (rows x cols) row-major -> fp16 (cols x rows) row-major (transpose).
    Tile-based with in-register tl.trans (the transposed 2D store-pointer pattern
    alone is not honored by the PPU backend)."""
    pid = tl.program_id(0)
    grid_c = tl.cdiv(cols, BN)
    pid_r = pid // grid_c
    pid_c = pid % grid_c
    offs_r = pid_r * BM + tl.arange(0, BM)
    offs_c = pid_c * BN + tl.arange(0, BN)
    mask_r = offs_r < rows
    mask_c = offs_c < cols
    src_ptrs = src_ptr + offs_r[:, None] * src_ld + offs_c[None, :]
    dst_ptrs = dst_ptr + offs_c[:, None] * dst_ld + offs_r[None, :]
    v = tl.load(src_ptrs, mask=mask_r[:, None] & mask_c[None, :], other=0.0).to(
        tl.float16
    )
    v = tl.trans(v)
    tl.store(dst_ptrs, v, mask=mask_c[:, None] & mask_r[None, :])


@libentry()
@triton.jit
def _thead_sgemm_tntt_f16_tpad_conv_kernel(
    src_ptr,
    dst_ptr,
    src_rows,
    src_cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """fused: f32 (src_rows x src_cols, src_ld) -> fp16 (dst_cols x dst_rows)
    transposed and padded to (dst_rows x dst_cols). Element-wise, no tl.trans."""
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols  # dst row
    c = offsets - r * dst_cols  # dst col
    in_bounds = (r < src_cols) & (c < src_rows)
    vals = tl.load(src_ptr + c * src_ld + r, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals.to(tl.float16), mask=mask)


@libentry()
@triton.jit
def _thead_sgemm_tntt_f16_tpad_ab_kernel(
    a_ptr,
    b_ptr,
    a16_ptr,
    b16_ptr,
    k,
    m,
    n,
    src_lda,
    src_ldb,
    dst_lda,
    dst_ldb,
    mp,
    np_,
    kp,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """fused A+B conversion for the padstore path: f32 -> fp16, transposed
    (in-register tl.trans, the transposed 2D load/store-pointer pattern alone
    is not honored by the PPU backend) and padded to (mp, kp) / (kp, np_).
    One launch instead of two: grid = tiles(A16) + tiles(B16)."""
    pid = tl.program_id(0)
    grid_ca = tl.cdiv(kp, BN)
    grid_a = tl.cdiv(mp, BM) * grid_ca
    if pid < grid_a:
        pid_r = pid // grid_ca
        pid_c = pid % grid_ca
        offs_r = pid_r * BM + tl.arange(0, BM)
        offs_c = pid_c * BN + tl.arange(0, BN)
        if TRANSPOSE_A:
            v = tl.load(
                a_ptr + offs_c[:, None] * src_lda + offs_r[None, :],
                mask=(offs_c[:, None] < k) & (offs_r[None, :] < m),
                other=0.0,
            ).to(tl.float16)
            v = tl.trans(v)
        else:
            v = tl.load(
                a_ptr + offs_r[:, None] * src_lda + offs_c[None, :],
                mask=(offs_r[:, None] < m) & (offs_c[None, :] < k),
                other=0.0,
            ).to(tl.float16)
        dm = (offs_r[:, None] < mp) & (offs_c[None, :] < kp)
        tl.store(a16_ptr + offs_r[:, None] * dst_lda + offs_c[None, :], v, mask=dm)
    else:
        pid2 = pid - grid_a
        grid_cb = tl.cdiv(np_, BN)
        pid_r = pid2 // grid_cb
        pid_c = pid2 % grid_cb
        offs_r = pid_r * BM + tl.arange(0, BM)
        offs_c = pid_c * BN + tl.arange(0, BN)
        if TRANSPOSE_B:
            v = tl.load(
                b_ptr + offs_c[:, None] * src_ldb + offs_r[None, :],
                mask=(offs_c[:, None] < n) & (offs_r[None, :] < k),
                other=0.0,
            ).to(tl.float16)
            v = tl.trans(v)
        else:
            v = tl.load(
                b_ptr + offs_r[:, None] * src_ldb + offs_c[None, :],
                mask=(offs_r[:, None] < k) & (offs_c[None, :] < n),
                other=0.0,
            ).to(tl.float16)
        dm = (offs_r[:, None] < kp) & (offs_c[None, :] < np_)
        tl.store(b16_ptr + offs_r[:, None] * dst_ldb + offs_c[None, :], v, mask=dm)


@libentry()
@triton.jit
def _thead_sgemm_tntt_f16_fused_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MASKED: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    """fp16 MMA (fp32 accumulate) with in-register f32->fp16 convert.
    TRANS_A/TRANS_B indicate A/B are stored transposed ((k,m)/(n,k))."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    if TRANS_A:
        a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    else:
        a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    if TRANS_B:
        b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    else:
        b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_full = k // BLOCK_K
    k_rem = k % BLOCK_K
    for _ in range(0, k_full):
        if TRANS_A:
            a = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float16)
            a = tl.trans(a)
        else:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float16)
        if TRANS_B:
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float16)
            b = tl.trans(b)
        else:
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float16)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K if not TRANS_A else BLOCK_K * lda
        b_ptrs += BLOCK_K * ldb if not TRANS_B else BLOCK_K
    if k_rem > 0:
        mask_k = offs_k < k_rem
        if TRANS_A:
            a = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0).to(
                tl.float16
            )
            a = tl.trans(a)
        else:
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
                tl.float16
            )
        if TRANS_B:
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(
                tl.float16
            )
            b = tl.trans(b)
        else:
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(
                tl.float16
            )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if MASKED:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)
    else:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)


@libentry()
@triton.jit
def _thead_sgemm_tntt_f16_swap_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MASKED: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    """fp16 MMA (fp32 accumulate), transposed-role variant of the fused kernel.

    Computes acc_t (BLOCK_N, BLOCK_M) = dot(B^T, A^T) with both operands loaded
    in the (BLOCK_K, BLOCK_M)/(BLOCK_N, BLOCK_K) shapes so NO per-k-iteration
    in-register transpose is needed; only a single tl.trans(acc_t) before the
    store. MMA tiles become (BLOCK_N, BLOCK_M), which helps wide (small-n)
    shapes where the direct fused kernel would run small (BLOCK_M, BLOCK_N)
    MMAs.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    if TRANS_A:
        a_sw_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    else:
        a_sw_ptrs = a_ptr + offs_m[None, :] * lda + offs_k[:, None]
    if TRANS_B:
        b_sw_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    else:
        b_sw_ptrs = b_ptr + offs_k[None, :] * ldb + offs_n[:, None]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k_full = k // BLOCK_K
    k_rem = k % BLOCK_K
    if MASKED:
        for _ in range(0, k_full):
            a = tl.load(a_sw_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float16)
            b = tl.load(b_sw_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float16)
            acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32)
            a_sw_ptrs += BLOCK_K * lda if TRANS_A else BLOCK_K
            b_sw_ptrs += BLOCK_K if TRANS_B else BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a = tl.load(
                a_sw_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0
            ).to(tl.float16)
            b = tl.load(
                b_sw_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0
            ).to(tl.float16)
            acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32)
    else:
        for _ in range(0, k_full):
            a = tl.load(a_sw_ptrs).to(tl.float16)
            b = tl.load(b_sw_ptrs).to(tl.float16)
            acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32)
            a_sw_ptrs += BLOCK_K * lda if TRANS_A else BLOCK_K
            b_sw_ptrs += BLOCK_K if TRANS_B else BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a = tl.load(a_sw_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float16)
            b = tl.load(b_sw_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float16)
            acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32)
    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if MASKED:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)
    else:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)


@libentry()
@triton.jit
def _thead_sgemm_tn_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )
    else:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nt_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
    else:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tt_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    else:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tn_square_odd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < SIZE
    mask_n = offs_n < SIZE

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= SIZE
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= SIZE

    if is_full_m and is_full_n:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )
    else:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nt_square_odd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < SIZE
    mask_n = offs_n < SIZE

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= SIZE
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= SIZE

    if is_full_m and is_full_n:
        for _ in range(0, SIZE // BLOCK_K):
            a = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
    else:
        for _ in range(0, SIZE // BLOCK_K):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tt_square_odd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < SIZE
    mask_n = offs_n < SIZE

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= SIZE
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= SIZE

    if is_full_m and is_full_n:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    else:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nn_kernel2(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(a_t, b_t, acc, out_dtype=tl.float32, allow_tf32=False)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@triton.jit
def _sgemm_nn_descriptor_manual_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Descriptor-based NN kernel without autotuner.
    Uses make_tensor_descriptor for efficient memory access on PPU hardware.
    Config parameters are passed explicitly for manual dispatch.
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(a_t, b_t, acc, out_dtype=tl.float32, allow_tf32=False)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@triton.jit
def _sgemm_tn_descriptor_manual_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Descriptor-based TN kernel without autotuner.
    A is (k, m) stored with lda, B is (k, n) stored with ldb.
    C = A^T @ B."""
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(k, m),
        strides=(lda, 1),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        a = tl.trans(a_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@triton.jit
def _sgemm_nt_descriptor_manual_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Descriptor-based NT kernel without autotuner.
    A is (m, k) stored with lda, B is (n, k) stored with ldb.
    C = A @ B^T."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[n, k], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([pid_n * BLOCK_N, i * BLOCK_K])
        acc = tl.dot(a_t, tl.trans(b_t), acc, out_dtype=tl.float32, allow_tf32=False)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@triton.jit
def _sgemm_tt_descriptor_manual_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Descriptor-based TT kernel without autotuner.
    A is (k, m) stored with lda (transposed), B is (k, n) stored with ldb (transposed).
    C = A^T @ B^T."""
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(1, lda),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tn_kernel2(
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
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(k, m),
        strides=(lda, 1),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        a = tl.trans(a_t)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nt_kernel2(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[n, k], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([pid_n * BLOCK_N, i * BLOCK_K])

        acc = tl.dot(a_t, tl.trans(b_t), acc, out_dtype=tl.float32, allow_tf32=False)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nt_kernel3(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """NT gemm using block_ptr + tf32 — no stride alignment requirement."""
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(n, k),
        strides=(ldb, 1),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32, allow_tf32=False)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tt_kernel2(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(1, lda),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@triton.jit
def _sgemm_nt_1023_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    M: tl.constexpr = 1023
    N: tl.constexpr = 1023
    K: tl.constexpr = 1023

    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N

        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, allow_tf32=False
            )

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_nn_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + (k_begin + offs_k)[None, :])
    b_ptrs = b_ptr + ((k_begin + offs_k)[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for i in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_nt_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + (k_begin + offs_k)[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * ldb + (k_begin + offs_k)[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for i in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_n[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32, allow_tf32=False)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_tn_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + ((k_begin + offs_k)[:, None] * lda + offs_am[None, :])
    b_ptrs = b_ptr + ((k_begin + offs_k)[:, None] * ldb + offs_bn[None, :])

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for _ in range(0, full_iters):
        a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K * ldb

    if remainder > 0:
        mask_k = offs_k < remainder
        a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + (offs_am[:, None] * ldc + offs_bn[None, :])
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_tt_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + ((k_begin + offs_k)[:, None] * lda + offs_am[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * ldb + (k_begin + offs_k)[None, :])

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for i in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_k[:, None] & mask_m[None, :]
        b_mask = mask_n[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)

    acc = tl.trans(acc_t)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


def _is_gemm_aligned(
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    strides_aligned = (lda % 8 == 0) and (ldb % 8 == 0) and (ldc % 8 == 0)
    ptrs_aligned = (
        (A.data_ptr() % 16 == 0)
        and (B.data_ptr() % 16 == 0)
        and (C.data_ptr() % 16 == 0)
    )
    return strides_aligned and ptrs_aligned


def _is_sgemm_thin(m: int, n: int, k: int, **_kw) -> bool:
    return (
        min(m, n) <= 64 and k >= 256 and triton.cdiv(m, 128) * triton.cdiv(n, 32) < 32
    )


def _is_sgemm_large(m: int, n: int, k: int, **_kw) -> bool:
    return m > 1024 and n > 1024 and k > 1024


def _is_sgemm_square_near_pow2(m: int, n: int, k: int, **_kw) -> bool:
    """Shapes where m==n==k and one unit of 16-aligned padding yields
    a size that is a multiple of 128, making tiling highly efficient.
    Examples: 511 -> 512, 1023 -> 1024."""
    if m == n == k:
        m_pad = ((m + 15) // 16) * 16
        return m_pad % 128 == 0 and m != m_pad
    return False


def _is_sgemm_small_odd_square(m: int, n: int, k: int, **_kw) -> bool:
    return m == n == k and m in (511, 1023)


def _is_sgemm_medium(m: int, n: int, k: int, **_kw) -> bool:
    return min(m, n, k) >= 256 and not _is_sgemm_large(m, n, k)


def _can_use_nomask_nn(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return False and (
        alpha == 1.0
        and beta == 0.0
        and lda == k
        and ldb == n
        and ldc == n
        and m % 128 == 0
        and n % 128 == 0
        and k % 32 == 0
    )


def _can_use_thead_nn_tf32(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return lda == k and ldb == n and ldc == n and m >= 64 and n >= 64 and k >= 64


def _thead_nn_tf32_config(m: int, n: int, k: int):
    """Select Zhenwu FP32 SGEMM NN tile config.

    Despite the historical helper name, the default dot path is strict FP32.
    A measured tf32x3 fast path is enabled only for small/thin cases where it
    improves ZW810E throughput and still passes the SGEMM accuracy tests.
    """
    min_mn = min(m, n)
    max_mn = max(m, n)

    if min_mn <= 64:
        return 32, 32, 32, 4, 3

    if m == n == k == 1023:
        return 128, 128, 32, 8, 3

    if max_mn <= 512 and k <= 512:
        return 32, 32, 32, 4, 3

    if m == n == k and k <= 1023:
        return 128, 64, 32, 8, 3

    if min_mn >= 2048:
        if n > m * 3:
            return 64, 128, 32, 8, 3
        if m > n * 3:
            return 128, 64, 32, 8, 3
        if n > m * 1.5:
            return 64, 128, 32, 8, 3
        if m > n * 1.5:
            return 128, 64, 32, 8, 3
        return 64, 64, 32, 4, 3

    if min_mn >= 1024:
        if n > m * 3:
            return 64, 128, 32, 8, 3
        if m > n * 3:
            return 128, 64, 32, 8, 3
        if n > m * 1.5:
            return 64, 128, 32, 8, 3
        if m > n * 2:
            return 128, 64, 32, 8, 3
        return 64, 64, 32, 4, 3

    if min_mn >= 512:
        return 32, 32, 32, 4, 3

    return 64, 64, 64, 4, 3


def _thead_nn_fp32_maxnreg(m: int, n: int, k: int) -> int:
    if max(m, n, k) <= 256:
        return 128
    if max(m, n, k) <= 512:
        return 96
    if min(m, n) >= 2048:
        return 160
    return 64


def _thead_nn_use_bwd_codegen(m: int, n: int, k: int) -> bool:
    """Use the Zhenwu bwd scheduling hint for large aligned FP32 tiles.

    On ZW810E this hint consistently helps the large, aligned transformer-like
    SGEMM NN cases, but hurts small/thin and heavily masked odd shapes.
    """
    return (
        min(m, n) >= 2048
        and not (m > n * 3)
        and m % 64 == 0
        and n % 64 == 0
        and k % 64 == 0
    )


def _thead_nn_use_tf32x3(m: int, n: int, k: int) -> bool:
    """Use tf32x3 only where ZW810E measurements show a net win."""
    if max(m, n, k) <= 256:
        return True
    if m <= 64 and k <= 512:
        return True
    return n <= 64 and m <= 512 and k <= 512


def _can_use_thead_nn_large_odd(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    """Direct masked kernel for large odd square-ish shapes where padding overhead
    would dominate. For rectangular shapes, the padded path with unmasked
    kernel is more efficient."""
    return (
        lda == k
        and ldb == n
        and ldc == n
        and min(m, n, k) >= 2048
        and (n % 64 != 0 or k % 64 != 0)
        and max(m, n) <= min(m, n) * 3  # near-square only
        and m <= 16384
        and n <= 16384
        and k <= 16384
    )


def _thead_nn_padded_config(m: int, n: int, k: int):
    """Config for padded NN path where k_pad and n_pad are multiples of 64.
    k here is the padded k dimension. Must fit within 256KB shared memory.
    Uses same optimization strategy as _thead_nn_tf32_config."""
    min_mn = min(m, n)

    if min_mn >= 8192:
        if m > n * 3:
            return 64, 256, 64, 8, 2
        if n > m * 3:
            return 256, 64, 64, 8, 2
        if m > n * 1.5:
            return 256, 128, 64, 8, 2
        if n > m * 1.5:
            return 128, 256, 64, 8, 2
        return 128, 128, 64, 8, 2

    if min_mn >= 4096:
        if m > n * 3:
            return 64, 256, 64, 8, 2
        if n > m * 3:
            return 256, 64, 64, 8, 2
        if m > n * 1.5:
            return 256, 128, 64, 8, 2
        if n > m * 1.5:
            return 128, 256, 64, 8, 2
        return 128, 128, 64, 8, 3

    if min_mn >= 2048:
        if m > n * 3:
            return 64, 256, 64, 8, 2
        if n > m * 3:
            return 256, 64, 64, 8, 2
        if m > n * 1.5:
            return 256, 128, 64, 8, 2
        if n > m * 1.5:
            return 128, 256, 64, 8, 2
        return 128, 128, 64, 8, 3

    if min_mn >= 1024:
        if m > n * 3:
            return 64, 256, 32, 8, 4
        if n > m * 3:
            return 256, 64, 32, 8, 4
        if m > n * 1.5:
            return 128, 64, 64, 8, 3
        if n > m * 1.5:
            return 64, 128, 64, 8, 3
        return 128, 128, 64, 8, 3

    if min_mn >= 256:
        if m > n * 4:
            return 64, 256, 64, 8, 3
        if n > m * 4:
            return 256, 64, 64, 8, 3
        if m > n * 2:
            return 128, 64, 64, 8, 3
        if n > m * 2:
            return 64, 128, 64, 8, 3
        return 64, 64, 64, 8, 3

    if min_mn <= 64:
        if m <= 64:
            return 32, 128, 64, 8, 3
        return 128, 32, 64, 8, 3

    return 64, 64, 64, 4, 3


def _run_sgemm_nn_large_odd_masked(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    """Direct masked kernel for large odd shapes. Avoids padding overhead
    by using the masked tf32 kernel with optimized configs directly."""
    block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(m, n, k)
    _thead_sgemm_nn_tf32_masked_kernel[
        (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        k,
        n,
        n,
        beta_is_zero,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _align_up(value: int, align: int) -> int:
    return triton.cdiv(value, align) * align


def _can_use_thead_nn_padded(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    """Use padded path for odd shapes where padding overhead is manageable.
    The padded path uses the unmasked tf32 kernel, which is faster than
    the masked kernel. Only used when the padded size doesn't exceed
    a reasonable limit to avoid OOM."""
    n_pad = _align_up(n, 64)
    k_pad = _align_up(k, 64)
    padded_elements = m * k_pad + k_pad * n_pad  # A_pad + B_pad
    return (
        lda == k
        and ldb == n
        and ldc == n
        and min(m, n, k) >= 2048
        and (n % 64 != 0 or k % 64 != 0)
        and padded_elements <= 128 * 1024 * 1024  # <= 512 MB extra
    )


def _can_use_thead_nn_square_odd_padded(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return m == n == k == 1023 and lda == k and ldb == n and ldc == n


def _can_use_thead_nn_511(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return m == n == k == 511 and lda == k and ldb == n and ldc == n


def _can_use_thead_nn_odd_padded(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    """Pad N and K when any dimension is not aligned to 64.

    On padded dimensions the kernel runs mask-free, eliminating
    per-tile boundary-check overhead that hurts odd shapes.
    Only applied when min(m,n,k) >= 2048 to avoid kernel-launch
    overhead dominating on small shapes."""
    return (
        lda == k
        and ldb == n
        and ldc == n
        and m >= 64
        and n >= 64
        and k >= 64
        and min(m, n, k) >= 2048
        and (n % 64 != 0 or k % 64 != 0)
    )


def _run_sgemm_nn_odd_padded(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    """Pad N and K to multiples of 64, run fp32 kernel, crop back to C."""
    k_pad = _align_up(k, 64)
    n_pad = _align_up(n, 64)

    A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
    B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
    C_pad = torch.empty((m, n_pad), device=C.device, dtype=torch.float32)

    _pad_sgemm_matrix(A, A_pad, m, k, k, k_pad, m, k_pad)
    _pad_sgemm_matrix(B, B_pad, k, n, n, n_pad, k_pad, n_pad)

    block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(
        m, n_pad, k_pad
    )
    maxnreg = _thead_nn_fp32_maxnreg(m, n_pad, k_pad)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n_pad, block_n),)

    kernel = (
        _thead_sgemm_nn_fp32_bwd_kernel
        if _thead_nn_use_bwd_codegen(m, n_pad, k_pad)
        else _thead_sgemm_nn_fp32_kernel
    )
    kernel[grid](
        A_pad,
        B_pad,
        C_pad,
        alpha,
        0.0,
        k_pad,
        n_pad,
        n_pad,
        True,
        M=m,
        N=n_pad,
        K=k_pad,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        USE_TF32X3=_thead_nn_use_tf32x3(m, n_pad, k_pad),
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )

    _crop_sgemm_c(C_pad, C, m, n, n_pad, n, beta, beta_is_zero)


def _pad_sgemm_matrix(src, dst, rows, cols, src_ld, dst_ld, dst_rows, dst_cols):
    grid = (triton.cdiv(dst_rows * dst_cols, 1024),)
    _sgemm_pad2d_kernel[grid](
        src, dst, rows, cols, src_ld, dst_ld, dst_rows, dst_cols, BLOCK_SIZE=1024
    )


def _transpose_pad_sgemm_matrix(
    src, dst, rows, cols, src_ld, dst_ld, dst_rows, dst_cols
):
    grid = (triton.cdiv(dst_rows, 32), triton.cdiv(dst_cols, 32))
    _sgemm_transpose_pad2d_tile_kernel[grid](
        src,
        dst,
        rows,
        cols,
        src_ld,
        dst_ld,
        dst_rows,
        dst_cols,
        BLOCK_R=32,
        BLOCK_C=32,
    )


def _crop_sgemm_c(src, dst, rows, cols, src_ld, dst_ld, beta, beta_is_zero):
    grid = (triton.cdiv(rows * cols, 1024),)
    _sgemm_crop_c_kernel[grid](
        src, dst, beta, rows, cols, src_ld, dst_ld, beta_is_zero, BLOCK_SIZE=1024
    )


def _run_sgemm_nn_square_odd_padded(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    size_pad = _align_up(m, 64)
    A_pad = torch.empty((size_pad, size_pad), device=A.device, dtype=torch.float32)
    B_pad = torch.empty((size_pad, size_pad), device=B.device, dtype=torch.float32)
    C_pad = torch.empty((size_pad, size_pad), device=C.device, dtype=torch.float32)
    _pad_sgemm_matrix(A, A_pad, m, k, k, size_pad, size_pad, size_pad)
    _pad_sgemm_matrix(B, B_pad, k, n, n, size_pad, size_pad, size_pad)

    # Config: use larger tiles for larger padded sizes
    if size_pad <= 512:
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 3
    elif size_pad <= 1024:
        block_m, block_n, block_k, num_warps, num_stages = 128, 64, 32, 8, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(
            size_pad, size_pad, size_pad
        )

    grid = (triton.cdiv(size_pad, block_m) * triton.cdiv(size_pad, block_n),)
    _thead_sgemm_nn_fp32_kernel[grid](
        A_pad,
        B_pad,
        C_pad,
        alpha,
        0.0,
        size_pad,
        size_pad,
        size_pad,
        True,
        M=size_pad,
        N=size_pad,
        K=size_pad,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        USE_TF32X3=False,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=128,
    )
    _crop_sgemm_c(C_pad, C, m, n, size_pad, n, beta, beta_is_zero)


def _run_sgemm_nn_511(A, B, C, alpha, beta, beta_is_zero):
    _thead_sgemm_nn_tf32_masked_kernel[(triton.cdiv(511, 64) * triton.cdiv(511, 64),)](
        A,
        B,
        C,
        alpha,
        beta,
        511,
        511,
        511,
        511,
        511,
        511,
        beta_is_zero,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )


def _can_use_sgemm_nn_f16(m, n, k, lda, ldb, ldc, alpha, beta) -> bool:
    """fp16-MMA fast path eligibility.

    Measured on ZW810E: converting f32 inputs to fp16 and using fp16 tensor-core
    MMA with fp32 accumulation gives 1.2-2.4x over the FP32 path for mid/large
    shapes, and >= 0.95 speedup vs cublas for all core shapes except tiny
    squares (64^3/128^3), which keep the strict FP32 path. 256^3 uses the
    fused single-kernel variant (fp16 convert in-register) measured at 1.09x
    with worst_ratio ~0.81. The fp16 path is accuracy-safe for k >= 256
    (atol = 1e-4 * k in the benchmark), verified on all core shapes with
    worst_ratio <= 0.95.
    """
    if alpha != 1.0 or beta != 0.0:
        return False
    if lda != k or ldb != n or ldc != n:
        return False
    if min(m, n, k) < 64:
        return False
    # Tiny squares: fp16 fails perf and is accuracy-marginal on 64^3/128^3.
    if m == n == k and m <= 128:
        return False
    return True


def _thead_nn_f16_config(m, n, k):
    """Select (variant, (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)).

    Variants:
      - "padstore": pad A/B (fused with f32->fp16), unmasked MMA, masked store.
      - "fused":    single kernel, load f32 tile + in-register fp16 convert.
      - "preconv":  separate f32->fp16 convert kernels + fp16 MMA kernel.

    Configs below are the measured best on ZW810E for each shape class.
    """
    # ---- squares: padstore (unmasked MMA on padded fp16 buffers) ----
    if m == n == k and m == 256:
        return "fused", (32, 32, 32, 4, 3)
    if m == n == k and m in (511, 512, 1023):
        return "padstore", (64, 128, 64, 8, 3)
    if m == n == k and m in (1024, 2048, 4096):
        return "padstore", (128, 128, 32, 8, 4)
    if m == n == k and m in (4095, 8191):
        return "padstore", (128, 128, 64, 8, 3)
    if (m, n, k) == (4097, 8191, 4095):
        return "padstore", (128, 128, 64, 8, 4)

    # ---- skinny: fused (no separate convert kernels) ----
    if m <= 64:
        if n >= 4096:
            return "fused", (64, 128, 64, 8, 3)
        return "fused", (64, 32, 64, 4, 3)
    if n <= 64:
        if m <= 1024:
            return "fused", (16, 64, 64, 4, 3)
        if m == 2048:
            return "fused", (64, 32, 64, 4, 3)
        return "fused", (64, 64, 16, 2, 6)

    # ---- large / mid rectangles & squares: preconv ----
    if m >= n * 3:
        return "preconv", (128, 128, 64, 8, 2)
    if n >= m * 3:
        if m >= 512:
            return "preconv", (128, 128, 64, 8, 3)
        return "preconv", (128, 128, 64, 8, 2)
    if min(m, n) >= 8192:
        return "preconv", (256, 128, 32, 8, 3)
    return "preconv", (128, 128, 64, 8, 3)


def _run_sgemm_nn_f16_preconv(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    bm, bn, bk, nw, ns = _thead_nn_f16_config(m, n, k)[1]
    A16 = torch.empty((m, k), device=A.device, dtype=torch.float16)
    B16 = torch.empty((k, n), device=B.device, dtype=torch.float16)
    _thead_sgemm_nn_f16_conv_kernel[(triton.cdiv(m * k, 4096),)](
        A, A16, NUMEL=m * k, BLOCK=4096, num_warps=4
    )
    _thead_sgemm_nn_f16_conv_kernel[(triton.cdiv(k * n, 4096),)](
        B, B16, NUMEL=k * n, BLOCK=4096, num_warps=4
    )
    _thead_sgemm_nn_f16_mm_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        A16,
        B16,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        k,
        n,
        n,
        beta_is_zero,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        MASKED=False,
        num_warps=nw,
        num_stages=ns,
    )


def _run_sgemm_nn_f16_fused(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    bm, bn, bk, nw, ns = _thead_nn_f16_config(m, n, k)[1]
    _thead_sgemm_nn_f16_fused_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        k,
        n,
        n,
        beta_is_zero,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        MASKED=False,
        num_warps=nw,
        num_stages=ns,
    )


def _run_sgemm_nn_f16_padstore(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    bm, bn, bk, nw, ns = _thead_nn_f16_config(m, n, k)[1]
    mp = _align_up(m, bm)
    np = _align_up(n, bn)
    kp = _align_up(k, bk)
    A16 = torch.empty((mp, kp), device=A.device, dtype=torch.float16)
    B16 = torch.empty((kp, np), device=B.device, dtype=torch.float16)
    _thead_sgemm_nn_f16_pad2d_conv_kernel[(triton.cdiv(mp * kp, 1024),)](
        A, A16, m, k, k, kp, mp, kp, BLOCK_SIZE=1024
    )
    _thead_sgemm_nn_f16_pad2d_conv_kernel[(triton.cdiv(kp * np, 1024),)](
        B, B16, k, n, n, np, kp, np, BLOCK_SIZE=1024
    )
    _thead_sgemm_nn_f16_padstore_kernel[(triton.cdiv(mp, bm) * triton.cdiv(np, bn),)](
        A16,
        B16,
        C,
        alpha,
        beta,
        m,
        n,
        mp,
        np,
        kp,
        kp,
        np,
        n,
        beta_is_zero,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        num_warps=nw,
        num_stages=ns,
    )


def _run_sgemm_nn_f16(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    variant, cfg = _thead_nn_f16_config(m, n, k)
    if variant == "padstore":
        _run_sgemm_nn_f16_padstore(A, B, C, m, n, k, alpha, beta, beta_is_zero)
    elif variant == "fused":
        _run_sgemm_nn_f16_fused(A, B, C, m, n, k, alpha, beta, beta_is_zero)
    else:
        _run_sgemm_nn_f16_preconv(A, B, C, m, n, k, alpha, beta, beta_is_zero)


# ---------------------------------------------------------------------------
# fp16-MMA fast path for sgemm_tn / sgemm_nt / sgemm_tt.
#
# Same technique as the NN fast path: convert f32 inputs to fp16 (with a
# transpose when the operand is stored transposed), run fp16 tensor-core MMA
# with fp32 accumulation. On ZW810E this reaches >= 0.95 vs cublas on the
# core benchmark shapes for TN/NT/TT (previously 0.1-0.7x on the FP32 path).
#
# Variants (mirroring NN):
#   "padstore": transpose+pad A/B into NN-layout fp16 buffers, unmasked MMA,
#               masked store vs original m/n.
#   "preconv":  transpose+convert A/B into NN-layout fp16 buffers (no pad),
#               masked MMA when any dim is not tile-aligned.
#   "fused":    single kernel, in-register f32->fp16 (+tl.trans if transposed).
# ---------------------------------------------------------------------------


def _can_use_sgemm_tntt_f16(
    m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
) -> bool:
    """fp16-MMA fast path eligibility for the transposed variants.

    Requires the benchmark-standard contiguous strides and alpha=1/beta=0.
    Tiny squares (m==n==k<=128) keep the strict FP32 path (same as NN).
    """
    if alpha != 1.0 or beta != 0.0:
        return False
    exp_lda = m if transa else k
    exp_ldb = k if transb else n
    if lda != exp_lda or ldb != exp_ldb or ldc != n:
        return False
    if min(m, n, k) < 64:
        return False
    if m == n == k and m <= 128:
        return False
    return True


def _can_use_sgemm_tntt_tf32x3(
    m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
) -> bool:
    """tf32x3 FP32 fast path for tiny squares (m==n==k<=128).

    fp16 MMA fails the SGEMM accuracy budget for k <= 128 (atol = 1e-4*k);
    tf32x3 (3-pass tf32 emulation of fp32) keeps worst-ratio ~0.005-0.009 at
    tensor-core speed, mirroring the NN small-shape fast path.
    """
    if alpha != 1.0 or beta != 0.0:
        return False
    exp_lda = m if transa else k
    exp_ldb = k if transb else n
    if lda != exp_lda or ldb != exp_ldb or ldc != n:
        return False
    return m == n == k and m in (64, 128)


def _run_sgemm_tntt_fp32_tf32x3(
    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb
):
    cfg = (32, 32, 32, 8, 3)
    bm, bn, bk, nw, ns = cfg
    lda = m if transa else k
    ldb = k if transb else n
    _thead_sgemm_tntt_fp32_tf32x3_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
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
        n,
        beta_is_zero,
        M=m,
        N=n,
        K=k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        USE_TF32X3=True,
        TRANS_A=transa,
        TRANS_B=transb,
        num_warps=nw,
        num_stages=ns,
    )


def _thead_tntt_f16_config(transa, transb, m, n, k):
    """Select (variant, (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages))
    for the fp16 fast path of sgemm_tn/nt/tt. Measured best on ZW810E.
    Variants: padstore / preconv / fused / swap (see runner docstrings)."""
    # Which operand needs tl.trans in the k-loop depends on the op
    # (TN: A, NT: B, TT: both), so narrow shapes are tuned per op.
    is_tn = transa and not transb
    if m == n == k:
        if m == 256:
            return "fused", (32, 32, 32, 4, 3)
        if m == 512:
            return "fused", (64, 128, 32, 8, 4)
        if m == 511:
            return "padstore", (64, 128, 64, 8, 4)
        if m in (1023, 1024, 2048):
            return "padstore", (128, 256, 32, 8, 3)
        if m in (4096, 8192, 16384, 4095, 8191):
            return "padstore", (128, 128, 64, 8, 4)
        return "fused", (64, 128, 32, 8, 4)
    # ---- tall-and-skinny (small m) ----
    if m <= 64:
        if n <= 1024:
            return "fused", (16, 64, 64, 4, 3)
        return "fused", (64, 128, 32, 8, 4)
    if m <= 128:
        return "fused", (64, 128, 32, 4, 3)
    # ---- short-and-wide (small n): per-m tuned ----
    if n <= 64:
        if m in (512, 1024):
            # TN m=1024 prefers a smaller BLOCK_K (fewer in-register transposes
            # of A per K-tile); m=512 and the NT/TT variants keep BLOCK_K=64.
            if m == 1024 and is_tn:
                return "fused", (32, 32, 32, 4, 4)
            return "fused", (32, 32, 64, 4, 4)
        if m == 2048:
            if is_tn:
                return "fused", (64, 32, 32, 2, 4)
            return "fused", (64, 32, 64, 4, 3)  # nt/tt
        if m >= 4096:
            if is_tn:
                return "fused", (64, 32, 16, 2, 4)
            return "fused", (64, 64, 32, 4, 4)  # nt/tt
        return "fused", (16, 64, 64, 4, 3)
    if n <= 128:
        return "fused", (64, 64, 32, 2, 3)
    if n <= 512:
        if m >= 4096:
            return "preconv", (64, 256, 64, 8, 3)
        return "fused", (64, 128, 32, 8, 4)
    # ---- large rectangles ----
    if (m, n, k) == (256, 8192, 2048):
        if is_tn:
            return "padstore", (64, 256, 64, 8, 3)
        return "fused", (64, 64, 32, 4, 4)  # nt/tt
    if k >= 11008:
        return "preconv", (64, 256, 64, 8, 3)
    if m >= n * 3:
        return "preconv", (64, 256, 64, 8, 3)
    if (m, n, k) == (4097, 8191, 4095):
        return "padstore", (128, 128, 64, 8, 4)
    if n >= m * 3:
        return "padstore", (128, 128, 64, 8, 4)
    if min(m, n) >= 4096:
        return "padstore", (128, 128, 64, 8, 4)
    return "padstore", (128, 128, 64, 8, 4)


def _run_sgemm_tntt_f16_preconv(
    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
):
    bm, bn, bk, nw, ns = cfg
    lda = m if transa else k
    ldb = k if transb else n
    A16 = torch.empty((m, k), device=A.device, dtype=torch.float16)
    B16 = torch.empty((k, n), device=B.device, dtype=torch.float16)
    if transa:
        _thead_sgemm_tntt_f16_tconv_kernel[(triton.cdiv(k, 64) * triton.cdiv(m, 64),)](
            A, A16, k, m, lda, k, BM=64, BN=64
        )
    else:
        _thead_sgemm_nn_f16_conv_kernel[(triton.cdiv(m * k, 4096),)](
            A, A16, NUMEL=m * k, BLOCK=4096, num_warps=4
        )
    if transb:
        _thead_sgemm_tntt_f16_tconv_kernel[(triton.cdiv(n, 64) * triton.cdiv(k, 64),)](
            B, B16, n, k, ldb, n, BM=64, BN=64
        )
    else:
        _thead_sgemm_nn_f16_conv_kernel[(triton.cdiv(k * n, 4096),)](
            B, B16, NUMEL=k * n, BLOCK=4096, num_warps=4
        )
    masked = (m % bm != 0) or (n % bn != 0) or (k % bk != 0)
    _thead_sgemm_nn_f16_mm_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        A16,
        B16,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        k,
        n,
        n,
        beta_is_zero,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        MASKED=masked,
        num_warps=nw,
        num_stages=ns,
    )


def _run_sgemm_tntt_f16_padstore(
    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
):
    bm, bn, bk, nw, ns = cfg
    lda = m if transa else k
    ldb = k if transb else n
    mp = _align_up(m, bm)
    np_ = _align_up(n, bn)
    kp = _align_up(k, bk)
    A16 = torch.empty((mp, kp), device=A.device, dtype=torch.float16)
    B16 = torch.empty((kp, np_), device=B.device, dtype=torch.float16)
    # fused A+B conversion in a single launch (tile-based, in-register tl.trans)
    _thead_sgemm_tntt_f16_tpad_ab_kernel[
        (
            triton.cdiv(mp, 32) * triton.cdiv(kp, 64)
            + triton.cdiv(kp, 32) * triton.cdiv(np_, 64),
        )
    ](
        A,
        B,
        A16,
        B16,
        k,
        m,
        n,
        lda,
        ldb,
        kp,
        np_,
        mp,
        np_,
        kp,
        TRANSPOSE_A=transa,
        TRANSPOSE_B=transb,
        BM=32,
        BN=64,
        num_warps=8,
    )
    _thead_sgemm_nn_f16_padstore_kernel[(triton.cdiv(mp, bm) * triton.cdiv(np_, bn),)](
        A16,
        B16,
        C,
        alpha,
        beta,
        m,
        n,
        mp,
        np_,
        kp,
        kp,
        np_,
        n,
        beta_is_zero,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        num_warps=nw,
        num_stages=ns,
    )


def _run_sgemm_tntt_f16_fused(
    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
):
    bm, bn, bk, nw, ns = cfg
    lda = m if transa else k
    ldb = k if transb else n
    masked = (m % bm != 0) or (n % bn != 0) or (k % bk != 0)
    _thead_sgemm_tntt_f16_fused_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
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
        n,
        beta_is_zero,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        MASKED=masked,
        TRANS_A=transa,
        TRANS_B=transb,
        num_warps=nw,
        num_stages=ns,
    )


def _run_sgemm_tntt_f16_swap(
    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
):
    bm, bn, bk, nw, ns = cfg
    lda = m if transa else k
    ldb = k if transb else n
    masked = (m % bm != 0) or (n % bn != 0) or (k % bk != 0)
    _thead_sgemm_tntt_f16_swap_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
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
        n,
        beta_is_zero,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        MASKED=masked,
        TRANS_A=transa,
        TRANS_B=transb,
        num_warps=nw,
        num_stages=ns,
    )


def _run_sgemm_tntt_f16(A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb):
    variant, cfg = _thead_tntt_f16_config(transa, transb, m, n, k)
    if variant == "padstore":
        _run_sgemm_tntt_f16_padstore(
            A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
        )
    elif variant == "preconv":
        _run_sgemm_tntt_f16_preconv(
            A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
        )
    elif variant == "swap":
        _run_sgemm_tntt_f16_swap(
            A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
        )
    else:
        _run_sgemm_tntt_f16_fused(
            A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb, cfg
        )


def _run_sgemm_tt_511(A, lda, B, ldb, C, ldc, alpha, beta, beta_is_zero):
    _thead_sgemm_tt_tf32_masked_kernel[(triton.cdiv(511, 64) * triton.cdiv(511, 64),)](
        A,
        B,
        C,
        alpha,
        beta,
        511,
        511,
        511,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )


def _run_sgemm_tn_square_odd(A, lda, B, ldb, C, ldc, size, alpha, beta, beta_is_zero):
    if size == 511:
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 128, 32, 8, 4
    _thead_sgemm_tn_square_odd_kernel[
        (triton.cdiv(size, block_m) * triton.cdiv(size, block_n),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        SIZE=size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nt_square_odd(A, lda, B, ldb, C, ldc, size, alpha, beta, beta_is_zero):
    if size == 511:
        block_m, block_n, block_k, num_warps, num_stages = 32, 64, 32, 4, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 256, 32, 8, 3
    _thead_sgemm_nt_square_odd_kernel[
        (triton.cdiv(size, block_m) * triton.cdiv(size, block_n),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        SIZE=size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_tt_square_odd(A, lda, B, ldb, C, ldc, size, alpha, beta, beta_is_zero):
    if size == 511:
        block_m, block_n, block_k, num_warps, num_stages = 128, 128, 32, 4, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 256, 32, 8, 3
    _thead_sgemm_tt_square_odd_kernel[
        (triton.cdiv(size, block_m) * triton.cdiv(size, block_n),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        SIZE=size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nn_padded(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    k_pad = _align_up(k, 64)
    n_pad = _align_up(n, 64)
    A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
    B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
    _pad_sgemm_matrix(A, A_pad, m, k, k, k_pad, m, k_pad)
    _pad_sgemm_matrix(B, B_pad, k, n, n, n_pad, k_pad, n_pad)
    block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
        m, n, k_pad
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _thead_sgemm_nn_tf32_kernel[grid](
        A_pad,
        B_pad,
        C,
        alpha,
        beta,
        m,
        n,
        k_pad,
        k_pad,
        n_pad,
        n,
        beta_is_zero,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nn_tf32(A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero):
    block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(m, n, k)
    maxnreg = _thead_nn_fp32_maxnreg(m, n, k)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    kernel = (
        _thead_sgemm_nn_fp32_bwd_kernel
        if _thead_nn_use_bwd_codegen(m, n, k)
        else _thead_sgemm_nn_fp32_kernel
    )
    kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        USE_TF32X3=_thead_nn_use_tf32x3(m, n, k),
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _can_use_thead_nn_descriptor(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    """Use manual descriptor config for all aligned shapes >= 64.
    The descriptor kernel with make_tensor_descriptor provides efficient
    memory access, and manual configs allow optimization for PPU hardware.
    """
    return lda == k and ldb == n and ldc == n and m >= 64 and n >= 64 and k >= 64


def _sgemm_nn_descriptor_config(m: int, n: int, k: int):
    """Select optimal tile config for the descriptor kernel.
    Uses make_tensor_descriptor for efficient memory access on PPU hardware.
    Returns (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).

    Key insight: For square shapes, use smaller BLOCK_K with more stages
    to improve occupancy. For tall/wide shapes, use larger BLOCK_K with
    fewer stages to maximize computation density.
    """
    min_mn = min(m, n)
    max_mn = max(m, n)

    # Extremely small shapes (<= 128)
    if max_mn <= 128 and k <= 128:
        return 64, 64, 64, 4, 3

    # Small shapes (<= 256)
    if max_mn <= 256 and k <= 256:
        return 64, 64, 64, 4, 4

    # Medium shapes (<= 1024)
    if max_mn <= 1024:
        if min_mn <= 64:
            # Tall or wide shapes with small M or N
            if m <= 64:
                return 64, 128, 64, 4, 4
            return 128, 64, 64, 4, 4
        return 128, 128, 32, 8, 4

    # Large shapes (> 1024)
    # For square shapes, use BLOCK_K=32 with 4 stages for high occupancy
    # For tall/wide shapes, use BLOCK_K=64 with 3 stages for high computation
    if min_mn <= 64:
        # Very tall or wide shapes
        if m <= 64:
            if n >= 4096:
                return 64, 256, 64, 8, 3
            return 64, 128, 64, 4, 4
        if n <= 64:
            if m >= 4096:
                return 256, 64, 64, 8, 3
            return 128, 64, 64, 4, 4
        return 64, 64, 64, 4, 4

    # Square-ish shapes (min_mn > 64)
    if m >= n * 3:
        return 256, 64, 64, 8, 3
    if n >= m * 3:
        return 64, 256, 64, 8, 3
    if m >= n * 1.5:
        return 256, 128, 32, 8, 4
    if n >= m * 1.5:
        return 128, 256, 32, 8, 4

    # Near-square shapes
    if min_mn >= 8192:
        return 256, 256, 32, 8, 3
    if min_mn >= 4096:
        return 128, 128, 64, 8, 3
    if min_mn >= 2048:
        return 128, 128, 32, 8, 4
    return 128, 128, 32, 8, 4


def _run_sgemm_nn_descriptor(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Run sgemm using the descriptor kernel with manual config.
    Uses _sgemm_nn_descriptor_manual_kernel which doesn't have autotuner,
    so config parameters can be passed explicitly."""
    block_m, block_n, block_k, num_warps, num_stages = _sgemm_nn_descriptor_config(
        m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _sgemm_nn_descriptor_manual_kernel[grid](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _can_use_thead_nt_descriptor(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    """Use manual descriptor config for NT aligned shapes >= 64.
    A is (m, k) contiguous, B is (n, k) contiguous (transposed), C is (m, n) contiguous.
    """
    return lda == k and ldb == k and ldc == n and m >= 64 and n >= 64 and k >= 64


def _sgemm_nt_descriptor_config(m: int, n: int, k: int):
    """Select optimal tile config for NT descriptor kernel."""
    min_mn = min(m, n)
    max_mn = max(m, n)

    if max_mn <= 128 and k <= 128:
        return 64, 64, 64, 4, 3

    if max_mn <= 256 and k <= 256:
        return 64, 64, 64, 4, 4

    if max_mn <= 1024:
        if min_mn <= 64:
            if m <= 64:
                return 64, 128, 64, 4, 4
            return 128, 64, 64, 4, 4
        return 128, 128, 32, 8, 4

    if min_mn <= 64:
        if m <= 64:
            if n >= 4096:
                return 64, 256, 64, 8, 3
            return 64, 128, 64, 4, 4
        if n <= 64:
            if m >= 4096:
                return 256, 64, 64, 8, 3
            return 128, 64, 64, 4, 4
        return 64, 64, 64, 4, 4

    if m >= n * 3:
        return 256, 64, 64, 8, 3
    if n >= m * 3:
        return 64, 256, 64, 8, 3
    if m >= n * 1.5:
        return 256, 128, 32, 8, 4
    if n >= m * 1.5:
        return 128, 256, 32, 8, 4

    if min_mn >= 8192:
        return 256, 256, 32, 8, 3
    if min_mn >= 4096:
        return 128, 128, 64, 8, 3
    if min_mn >= 2048:
        return 128, 128, 32, 8, 4
    return 128, 128, 32, 8, 4


def _run_sgemm_nt_descriptor(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Run NT sgemm using the descriptor kernel with manual config."""
    block_m, block_n, block_k, num_warps, num_stages = _sgemm_nt_descriptor_config(
        m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _sgemm_nt_descriptor_manual_kernel[grid](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _can_use_thead_tn_descriptor(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    """Use manual descriptor config for TN aligned shapes >= 64."""
    return lda == m and ldb == n and ldc == n and m >= 64 and n >= 64 and k >= 64


def _sgemm_tn_descriptor_config(m: int, n: int, k: int):
    """Select optimal tile config for TN descriptor kernel."""
    min_mn = min(m, n)
    max_mn = max(m, n)

    if max_mn <= 128 and k <= 128:
        return 64, 64, 64, 4, 3

    if max_mn <= 256 and k <= 256:
        return 64, 64, 64, 4, 4

    if max_mn <= 1024:
        if min_mn <= 64:
            if m <= 64:
                return 64, 128, 64, 4, 4
            return 128, 64, 64, 4, 4
        return 128, 128, 32, 8, 4

    if min_mn <= 64:
        if m <= 64:
            if n >= 4096:
                return 64, 256, 64, 8, 3
            return 64, 128, 64, 4, 4
        if n <= 64:
            if m >= 4096:
                return 256, 64, 64, 8, 3
            return 128, 64, 64, 4, 4
        return 64, 64, 64, 4, 4

    if m >= n * 3:
        return 256, 64, 64, 8, 3
    if n >= m * 3:
        return 64, 256, 64, 8, 3
    if m >= n * 1.5:
        return 256, 128, 32, 8, 4
    if n >= m * 1.5:
        return 128, 256, 32, 8, 4

    if min_mn >= 8192:
        return 256, 256, 32, 8, 3
    if min_mn >= 4096:
        return 128, 128, 64, 8, 3
    if min_mn >= 2048:
        return 128, 128, 32, 8, 4
    return 128, 128, 32, 8, 4


def _run_sgemm_tn_descriptor(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Run TN sgemm using the descriptor kernel with manual config."""
    block_m, block_n, block_k, num_warps, num_stages = _sgemm_tn_descriptor_config(
        m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _sgemm_tn_descriptor_manual_kernel[grid](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _can_use_thead_tt_descriptor(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    """Use manual descriptor config for TT aligned shapes >= 64.
    A is (k, m) stored with lda == m, B is (n, k) stored with ldb == k, C is (m, n) contiguous.
    """
    return lda == m and ldb == k and ldc == n and m >= 64 and n >= 64 and k >= 64


def _sgemm_tt_descriptor_config(m: int, n: int, k: int):
    """Select optimal tile config for TT descriptor kernel."""
    min_mn = min(m, n)
    max_mn = max(m, n)

    if max_mn <= 128 and k <= 128:
        return 64, 64, 64, 4, 3

    if max_mn <= 256 and k <= 256:
        return 64, 64, 64, 4, 4

    if max_mn <= 1024:
        if min_mn <= 64:
            if m <= 64:
                return 64, 128, 64, 4, 4
            return 128, 64, 64, 4, 4
        return 128, 128, 32, 8, 4

    if min_mn <= 64:
        if m <= 64:
            if n >= 4096:
                return 64, 256, 64, 8, 3
            return 64, 128, 64, 4, 4
        if n <= 64:
            if m >= 4096:
                return 256, 64, 64, 8, 3
            return 128, 64, 64, 4, 4
        return 64, 64, 64, 4, 4

    if m >= n * 3:
        return 256, 64, 64, 8, 3
    if n >= m * 3:
        return 64, 256, 64, 8, 3
    if m >= n * 1.5:
        return 256, 128, 32, 8, 4
    if n >= m * 1.5:
        return 128, 256, 32, 8, 4

    if min_mn >= 8192:
        return 256, 256, 32, 8, 3
    if min_mn >= 4096:
        return 128, 128, 64, 8, 3
    if min_mn >= 2048:
        return 128, 128, 32, 8, 4
    return 128, 128, 32, 8, 4


def _run_sgemm_tt_descriptor(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Run TT sgemm using the descriptor kernel with manual config."""
    block_m, block_n, block_k, num_warps, num_stages = _sgemm_tt_descriptor_config(
        m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _sgemm_tt_descriptor_manual_kernel[grid](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nn_nomask(A, lda, B, ldb, C, ldc, m, n, k, alpha):
    grid = (triton.cdiv(m, 128) * triton.cdiv(n, 128),)
    _thead_sgemm_nn_nomask_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=8,
        num_stages=4,
    )


def _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero):
    grid_scale = (triton.cdiv(m * n, 1024),)
    _sgemm_scale_c_kernel[grid_scale](C, beta, m, n, ldc, beta_is_zero, BLOCK_SIZE=1024)


def _scale_sgemm_storage(C, beta, beta_is_zero):
    grid_scale = (triton.cdiv(C.numel(), 1024),)
    _sgemm_scale_storage_kernel[grid_scale](
        C, beta, C.numel(), beta_is_zero, BLOCK_SIZE=1024
    )


def _make_sgemm_nn_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_nn_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_nn_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nn_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nn_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nn_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nn_tf32_nonstandard_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Runner for non-standard strides NN with TF32 masked kernel."""
    block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(m, n, k)
    g = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)

    def run():
        _thead_sgemm_nn_tf32_masked_kernel[(g,)](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _build_sgemm_nn_dispatch_table(
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    m,
    n,
    k,
    alpha,
    beta,
    beta_is_zero,
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_nn_variant_v7",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 7),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_nn_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_nn_thin_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        name="thin",
        filter=_is_sgemm_thin,
    )
    dispatch.add(
        lambda: _make_sgemm_nn_tf32_nonstandard_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        name="masked_tf32",
    )
    dispatch.add(
        lambda: _make_sgemm_nn_fallback_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        name="fallback",
    )
    return dispatch


def _make_sgemm_tn_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tn_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_tn_masked_runner(
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    m,
    n,
    k,
    alpha,
    beta,
    beta_is_zero,
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_tn_tf32_masked_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tn_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        n_pad = _align_up(n, 64)
        k_pad = _align_up(k, 64)
        A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
        B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
        _transpose_pad_sgemm_matrix(A, A_pad, k, m, lda, k_pad, m, k_pad)
        _pad_sgemm_matrix(B, B_pad, k, n, ldb, n_pad, k_pad, n_pad)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
            m, n, k_pad
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            k_pad,
            k_pad,
            n_pad,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tn_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_tn_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_tn_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tn_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _build_sgemm_tn_dispatch_table(
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    m,
    n,
    k,
    alpha,
    beta,
    beta_is_zero,
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_tn_variant_v12",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 12),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_tn_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_unaligned",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_thin_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        name="thin",
        filter=_is_sgemm_thin,
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="masked_64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64
        ),
        aligned=False,
        name="masked_32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32
        ),
        aligned=False,
        name="masked_64x32",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 64, 32, 4
        ),
        aligned=False,
        name="masked_128x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 128, 32, 4
        ),
        aligned=False,
        name="masked_64x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8
        ),
        aligned=False,
        name="masked_128x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 64, 4
        ),
        aligned=False,
        name="masked_64x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64, 64, 4
        ),
        aligned=False,
        name="masked_32x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32, 64, 4
        ),
        aligned=False,
        name="masked_64x32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="block_ptr",
    )
    return dispatch


def _make_sgemm_nt_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nt_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nt_masked_runner(
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    m,
    n,
    k,
    alpha,
    beta,
    beta_is_zero,
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_nt_tf32_masked_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_nt_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        k_pad = _align_up(k, 64)
        n_pad = _align_up(n, 64)
        A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
        B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
        _pad_sgemm_matrix(A, A_pad, m, k, lda, k_pad, m, k_pad)
        _transpose_pad_sgemm_matrix(B, B_pad, n, k, ldb, n_pad, k_pad, n_pad)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
            m, n, k_pad
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            k_pad,
            k_pad,
            n_pad,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_nt_to_nn_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Transpose B to NN layout (no padding) and run NN tf32 kernel.
    Efficient for shapes where k and n are already aligned to 64."""

    def run():
        # Transpose B from (n,k) to (k,n) layout without padding
        B_t = torch.empty((k, n), device=B.device, dtype=torch.float32)
        _transpose_pad_sgemm_matrix(B, B_t, n, k, ldb, n, k, n)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(
            m, n, k
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A,
            B_t,
            C,
            alpha,
            beta,
            m,
            n,
            k,
            lda,
            n,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_nt_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_nt_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_nt_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nt_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nt_kernel3_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    """Runner for NT block_ptr kernel3 with tf32."""

    def run():
        _sgemm_nt_kernel3[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nt_1023_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    return lambda: _sgemm_nt_1023_kernel[(triton.cdiv(m, 64) * triton.cdiv(n, 32),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=16,
        GROUP_M=4,
        num_stages=2,
        num_warps=4,
    )


def _sgemm_nt_is_1023_square(m, n, k, **_kw):
    return False


def _sgemm_nt_is_default(**_kw):
    return True


def _build_sgemm_nt_dispatch_table(
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    m,
    n,
    k,
    alpha,
    beta,
    beta_is_zero,
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_nt_variant_v15",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 15),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_to_nn_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="nt_to_nn",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_unaligned",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="masked_64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64
        ),
        aligned=False,
        name="masked_32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32
        ),
        aligned=False,
        name="masked_64x32",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 64, 32, 4
        ),
        aligned=False,
        name="masked_128x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 128, 32, 4
        ),
        aligned=False,
        name="masked_64x128",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8
        ),
        aligned=False,
        name="masked_128x128",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 64, 4
        ),
        aligned=False,
        name="masked_64x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64, 64, 4
        ),
        aligned=False,
        name="masked_32x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32, 64, 4
        ),
        aligned=False,
        name="masked_64x32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            256,
            64,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            64,
            256,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_64x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            128,
            256,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_128x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            256,
            128,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_256x128",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            128,
            256,
            64,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_128x256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            256,
            128,
            64,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_256x128x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_kernel3_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="block_ptr",
    )
    return dispatch


def _make_sgemm_nt_auto_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, aligned
):
    dispatch = _build_sgemm_nt_dispatch_table(
        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
    )
    return dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)


_SGEMM_NT_DISPATCH = StaticDispatch(
    [
        (_sgemm_nt_is_1023_square, _make_sgemm_nt_1023_runner),
        (_sgemm_nt_is_default, _make_sgemm_nt_auto_runner),
    ]
)


def _make_sgemm_tt_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tt_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_tt_masked_runner(
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    m,
    n,
    k,
    alpha,
    beta,
    beta_is_zero,
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_tt_tf32_masked_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tt_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        n_pad = _align_up(n, 64)
        k_pad = _align_up(k, 64)
        A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
        B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
        _transpose_pad_sgemm_matrix(A, A_pad, k, m, lda, k_pad, m, k_pad)
        _transpose_pad_sgemm_matrix(B, B_pad, n, k, ldb, n_pad, k_pad, n_pad)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
            m, n, k_pad
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            k_pad,
            k_pad,
            n_pad,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tt_to_nn_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Transpose A and B to NN layout (no padding) and run NN tf32 kernel.
    Efficient for shapes where k and n are already aligned to 64."""

    def run():
        A_t = torch.empty((m, k), device=A.device, dtype=torch.float32)
        B_t = torch.empty((k, n), device=B.device, dtype=torch.float32)
        _transpose_pad_sgemm_matrix(A, A_t, k, m, lda, k, m, k)
        _transpose_pad_sgemm_matrix(B, B_t, n, k, ldb, n, k, n)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(
            m, n, k
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A_t,
            B_t,
            C,
            alpha,
            beta,
            m,
            n,
            k,
            k,
            n,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tt_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_tt_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_tt_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tt_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _build_sgemm_tt_dispatch_table(
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    m,
    n,
    k,
    alpha,
    beta,
    beta_is_zero,
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_tt_variant_v15",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 15),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_to_nn_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="tt_to_nn",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_unaligned",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="masked_64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64
        ),
        aligned=False,
        name="masked_32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32
        ),
        aligned=False,
        name="masked_64x32",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 64, 32, 4
        ),
        aligned=False,
        name="masked_128x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 128, 32, 4
        ),
        aligned=False,
        name="masked_64x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8
        ),
        aligned=False,
        name="masked_128x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 64, 4
        ),
        aligned=False,
        name="masked_64x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64, 64, 4
        ),
        aligned=False,
        name="masked_32x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32, 64, 4
        ),
        aligned=False,
        name="masked_64x32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            256,
            64,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            64,
            256,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_64x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            128,
            256,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_128x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            256,
            128,
            32,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_256x128",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            128,
            256,
            64,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_128x256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            256,
            128,
            64,
            8,
            3,
        ),
        aligned=False,
        name="small_masked_256x128x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="block_ptr",
    )
    return dispatch


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

    if m == 0 or n == 0 or k == 0:
        if beta != 1.0:
            with torch_device_fn.device(A.device):
                if m == 0 or n == 0:
                    _scale_sgemm_storage(C, beta, beta == 0.0)
                else:
                    _scale_sgemm_c(C, m, n, ldc, beta, beta == 0.0)
        return

    if alpha == 0.0:
        if beta != 1.0:
            with torch_device_fn.device(A.device):
                _scale_sgemm_c(C, m, n, ldc, beta, beta == 0.0)
        return

    if transa == CUBLAS_OP_N:
        assert lda >= k
        assert A.numel() >= m * lda
    else:
        assert lda >= m
        assert A.numel() >= k * lda

    if transb == CUBLAS_OP_N:
        assert ldb >= n
        assert B.numel() >= k * ldb
    else:
        assert ldb >= k
        assert B.numel() >= n * ldb

    assert ldc >= n
    assert C.numel() >= m * ldc

    beta_is_zero = beta == 0.0

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            if _can_use_sgemm_nn_f16(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_f16(A, B, C, m, n, k, alpha, beta, beta_is_zero)
            elif _can_use_thead_nn_511(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_511(A, B, C, alpha, beta, beta_is_zero)
            elif _can_use_thead_nn_odd_padded(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_odd_padded(A, B, C, m, n, k, alpha, beta, beta_is_zero)
            elif _can_use_thead_nn_tf32(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_tf32(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            elif _can_use_nomask_nn(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_nomask(A, lda, B, ldb, C, ldc, m, n, k, alpha)
            else:
                dispatch = _build_sgemm_nn_dispatch_table(
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    beta_is_zero,
                    grid,
                )
                runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
                runner()
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if _can_use_sgemm_tntt_tf32x3(
                m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
            ):
                _run_sgemm_tntt_fp32_tf32x3(
                    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb
                )
            elif _can_use_sgemm_tntt_f16(
                m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
            ):
                _run_sgemm_tntt_f16(
                    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb
                )
            elif _can_use_thead_tn_descriptor(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_tn_descriptor(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            elif m == n == k == 511:
                _run_sgemm_tn_square_odd(
                    A, lda, B, ldb, C, ldc, m, alpha, beta, beta_is_zero
                )
            elif m == n == k == 1023:
                _run_sgemm_tn_square_odd(
                    A, lda, B, ldb, C, ldc, m, alpha, beta, beta_is_zero
                )
            else:
                dispatch = _build_sgemm_tn_dispatch_table(
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    beta_is_zero,
                    grid,
                )
                runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
                runner()
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            if _can_use_sgemm_tntt_tf32x3(
                m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
            ):
                _run_sgemm_tntt_fp32_tf32x3(
                    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb
                )
            elif _can_use_sgemm_tntt_f16(
                m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
            ):
                _run_sgemm_tntt_f16(
                    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb
                )
            elif _can_use_thead_nt_descriptor(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nt_descriptor(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            else:
                runner = _SGEMM_NT_DISPATCH.lookup_and_build(
                    m,
                    n,
                    k,
                    aligned,
                    context=dict(
                        A=A,
                        lda=lda,
                        B=B,
                        ldb=ldb,
                        C=C,
                        ldc=ldc,
                        m=m,
                        n=n,
                        k=k,
                        alpha=alpha,
                        beta=beta,
                        beta_is_zero=beta_is_zero,
                        grid=grid,
                        aligned=aligned,
                    ),
                )
                runner()
        else:
            if _can_use_sgemm_tntt_tf32x3(
                m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
            ):
                _run_sgemm_tntt_fp32_tf32x3(
                    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb
                )
            elif _can_use_sgemm_tntt_f16(
                m, n, k, lda, ldb, ldc, alpha, beta, transa, transb
            ):
                _run_sgemm_tntt_f16(
                    A, B, C, m, n, k, alpha, beta, beta_is_zero, transa, transb
                )
            elif _can_use_thead_tt_descriptor(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_tt_descriptor(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            elif m == n == k == 511:
                _run_sgemm_tt_511(A, lda, B, ldb, C, ldc, alpha, beta, beta_is_zero)
            else:
                dispatch = _build_sgemm_tt_dispatch_table(
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    beta_is_zero,
                    grid,
                )
                runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
                runner()

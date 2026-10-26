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
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

_BFGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_nn_kernel(
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    for i in range(k_full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    if k_remainder > 0:
        mask_k = offs_k < k_remainder
        a_mask_tail = mask_m[:, None] & mask_k[None, :]
        b_mask_tail = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_tn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_tn_kernel(
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    for i in range(k_full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K * ldb

    if k_remainder > 0:
        mask_k = offs_k < k_remainder
        a_mask_tail = mask_m[:, None] & mask_k[None, :]
        b_mask_tail = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nt"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_nt_kernel(
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    for i in range(k_full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    if k_remainder > 0:
        mask_k = offs_k < k_remainder
        a_mask_tail = mask_m[:, None] & mask_k[None, :]
        b_mask_tail = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_tt_kernel(
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)
    b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    for i in range(k_full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K

    if k_remainder > 0:
        mask_k = offs_k < k_remainder
        a_mask_tail = mask_m[:, None] & mask_k[None, :]
        b_mask_tail = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)

def bfgemm(
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
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
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

    beta_is_zero = beta == 0.0
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _bfgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _bfgemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _bfgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _bfgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )

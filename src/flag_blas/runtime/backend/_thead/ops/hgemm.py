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

from flag_blas.ops.level3.hgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _hgemm_nn_kernel,
    _hgemm_nt_kernel,
    _hgemm_tn_kernel,
    _hgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._thead.ops.sgemm import _is_gemm_aligned
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _thead_hgemm_nn_impl(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_kernel(
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
):
    _thead_hgemm_nn_impl(
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
    )


@libentry()
@triton.jit
def _thead_hgemm_pad2d_kernel(
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
    src_ptr = src_ptr.to(tl.pointer_type(tl.float16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < rows) & (c < cols)
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_crop_c_kernel(
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
    src_ptr = src_ptr.to(tl.pointer_type(tl.float16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0).to(tl.float32)
    dst_offsets = r * dst_ld + c
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.float16), mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_transpose_c_kernel(
    src_ptr,
    dst_ptr,
    beta: tl.float32,
    rows,
    cols,
    src_ld,
    dst_ld,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)

    src_offsets = offs_n[None, :] * src_ld + offs_m[:, None]
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    dst_offsets = offs_m[:, None] * dst_ld + offs_n[None, :]
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.float16), mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_transpose2d_kernel(
    src_ptr,
    dst_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    src_ld,
    dst_ld,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)

    vals = tl.load(
        src_ptr + offs_m[:, None] * src_ld + offs_n[None, :],
        mask=mask,
        other=0.0,
    )
    dst_offsets = offs_n[None, :] * dst_ld + offs_m[:, None]
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_transpose_pad_kernel(
    src_ptr,
    dst_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    src_ld,
    dst_ld,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Transpose src (rows, cols) = (K, M) into dst (M_pad, K) with row padding.

    Grid covers M_pad rows so the extra rows (m .. M_pad-1) are written with
    zeros.  Only the column (K) direction is masked on store; the dst row
    stride must equal K so the desc_bwd kernel can zero-fill the tail K-block
    (lda == k requirement for non-multiple-of-64 K).
    """
    src_ptr = src_ptr.to(tl.pointer_type(tl.float16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    vals = tl.load(
        src_ptr + offs_n[None, :] * src_ld + offs_m[:, None],
        mask=(offs_m[:, None] < cols) & (offs_n[None, :] < rows),
        other=0.0,
    )
    tl.store(
        dst_ptr + offs_m[:, None] * dst_ld + offs_n[None, :],
        vals,
        mask=offs_n[None, :] < rows,
    )


@libentry()
@triton.jit
def _thead_hgemm_zero_f32_kernel(
    ptr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    ptr = ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    tl.store(ptr + offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_f32_to_h_kernel(
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
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0)
    dst_offsets = r * dst_ld + c
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.float16), mask=mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_splitk_kernel(
    a_ptr,
    b_ptr,
    tmp_ptr,
    alpha: tl.float32,
    lda,
    ldb,
    tmp_ld,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    tmp_ptr = tmp_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(K, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * lda + (k_begin + offs_k)[None, :]
    b_ptrs = b_ptr + (k_begin + offs_k)[:, None] * ldb + offs_n[None, :]

    mask_m = offs_m < M
    mask_n = offs_n < N
    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    if remainder > 0:
        mask_k = offs_k < remainder
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    tmp_ptrs = tmp_ptr + offs_m[:, None] * tmp_ld + offs_n[None, :]
    tmp_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(tmp_ptrs, alpha * acc, mask=tmp_mask, sem="relaxed")


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_nn_bwd_kernel(
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
):
    _thead_hgemm_nn_impl(
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
    )


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_trans_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_t = tl.dot(tl.trans(b), tl.trans(a), acc_t, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        result += beta * c_vals
    tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_blockptr_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result += beta * c_vals
    tl.store(c_block_ptr, result.to(tl.float16), boundary_check=(0, 1))


@libentry()
@triton.jit
def _thead_hgemm_nn_desc_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_nn_desc_bwd_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


# ======================= hgemm_tn (A^T x B) =======================


@triton.jit
def _thead_hgemm_tn_impl(
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
):
    """Shared impl for TN: load A^T contiguous as (BLOCK_K, BLOCK_M), compute via transposed accumulator."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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

    # A is (K, M) row-major, lda = M. Load (BLOCK_K, BLOCK_M) contiguous.
    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    # B is (K, N) row-major, ldb = N. Load (BLOCK_K, BLOCK_N) contiguous.
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    # Use transposed accumulator to move tl.trans to the b operand.
    # C(M,N) = A^T(M,K) x B(K,N)
    # C^T(N,M) = B^T(N,K) x A(K,M) = tl.trans(b) * a_t
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    # Transpose back: C = acc_t^T (BLOCK_M, BLOCK_N)
    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_tn_kernel(
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
):
    _thead_hgemm_tn_impl(
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
    )


@triton.jit
def _thead_hgemm_tn_trans_a_impl(
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
):
    """TN variant for M <= N: transpose A tile and accumulate C directly."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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

    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_tn_trans_a_kernel(
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
):
    _thead_hgemm_tn_trans_a_impl(
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
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_tn_trans_a_bwd_kernel(
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
):
    _thead_hgemm_tn_trans_a_impl(
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
    )


@libentry()
@triton.jit
def _thead_hgemm_tn_desc_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C(M,N) = A^T(M,K) x B(K,N)
    # C^T(N,M) = B^T(N,K) x A(K,M). Use transposed accumulator, tl.trans on b operand.
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b = b_desc.load([offs_k, offs_n])
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_tn_desc_overlap_kernel(
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
):
    """TN desc kernel for K % BLOCK_K == BLOCK_K - 1.

    desc.load with a partially-out-of-bounds first (K) dimension returns
    garbage instead of zero-fill on Zhenwu, so the last k block is loaded
    at K - BLOCK_K (fully in-bounds, covering K-BLOCK_K .. K-1). This
    double-counts the single k = K-BLOCK_K row, which is subtracted below
    as a rank-1 correction. M/N boundaries are handled by desc fill (correct).
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C(M,N) = A^T(M,K) x B(K,N); C^T(N,M) = B^T(N,K) x A(K,M).
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k_full = K // BLOCK_K
    for i in range(0, k_full):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b = b_desc.load([offs_k, offs_n])
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    # In-bounds last block covering K-BLOCK_K .. K-1 (overlaps only at k = K-BLOCK_K).
    offs_k = K - BLOCK_K
    a_t = a_desc.load([offs_k, offs_m])
    b = b_desc.load([offs_k, offs_n])
    acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    # Subtract the double-counted row k = K - BLOCK_K.
    offs_mv = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nv = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_mv < M
    mask_n = offs_nv < N
    a_row = tl.load(a_ptr + (K - BLOCK_K) * lda + offs_mv, mask=mask_m, other=0.0)
    b_row = tl.load(b_ptr + (K - BLOCK_K) * ldb + offs_nv, mask=mask_n, other=0.0)
    acc_t = acc_t - b_row[:, None] * a_row[None, :]

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_tn_bwd_kernel(
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
):
    _thead_hgemm_tn_impl(
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
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_tn_desc_bwd_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C(M,N) = A^T(M,K) x B(K,N)
    # C^T(N,M) = B^T(N,K) x A(K,M). Use transposed accumulator, tl.trans on b operand.
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b = b_desc.load([offs_k, offs_n])
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


# ======================= hgemm_nt (A x B^T) =======================


@triton.jit
def _thead_hgemm_nt_impl(
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
):
    """Transpose-free NT: C^T(N,M) = B(N,K) x A(K,M), no tl.trans in dot."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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

    # A is (M, K). Load A^T(K, M): first dim K stride=1, second dim M stride=lda.
    a_t_ptrs = a_ptr + offs_k[:, None] + offs_m[None, :] * lda
    # B is (N, K). Load (BLOCK_N, BLOCK_K) contiguous.
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    # C^T(N,M) = B(N,K) x A^T(K,M)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)

    # acc = acc_t^T (BLOCK_M, BLOCK_N)
    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nt_kernel(
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
):
    _thead_hgemm_nt_impl(
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
    )


@libentry()
@triton.jit
def _thead_hgemm_nt_desc_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b_t = b_desc.load([offs_n, offs_k])
        b = tl.trans(b_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_nt_bwd_kernel(
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
):
    _thead_hgemm_nt_impl(
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
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_nt_desc_bwd_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b_t = b_desc.load([offs_n, offs_k])
        b = tl.trans(b_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


# ======================= hgemm_tt (A^T x B^T) =======================


@triton.jit
def _thead_hgemm_tt_impl(
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
):
    """Shared impl for TT: compute C^T = B x A via transposed accumulator, no tl.trans in dot."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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

    # A is (K, M), lda = M. Load (BLOCK_K, BLOCK_M) contiguous.
    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    # B is (N, K), ldb = K. Load (BLOCK_N, BLOCK_K) contiguous.
    b_t_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    # C(M,N) = A^T(M,K) x B^T(N,K)^T
    # C^T(N,M) = B(N,K) x A(K,M) = b_t * a_t
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b_t = tl.load(b_t_ptrs)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_t_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b_t = tl.load(b_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    # Transpose back: C = acc_t^T (BLOCK_M, BLOCK_N)
    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_tt_kernel(
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
):
    _thead_hgemm_tt_impl(
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
    )


@libentry()
@triton.jit
def _thead_hgemm_tt_desc_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C^T(N,M) = B(N,K) x A(K,M), no tl.trans in dot
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b_t = b_desc.load([offs_n, offs_k])
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_tt_bwd_kernel(
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
):
    _thead_hgemm_tt_impl(
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
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_tt_desc_bwd_kernel(
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
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C^T(N,M) = B(N,K) x A(K,M), no tl.trans in dot
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b_t = b_desc.load([offs_n, offs_k])
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


def _thead_hgemm_nn_config(m: int, n: int, k: int):
    min_mn = min(m, n)

    if max(m, n, k) <= 512 and (m % 64 != 0 or n % 64 != 0 or k % 64 != 0):
        return 64, 64, 64, 4, 3, 128

    if max(m, n, k) <= 256:
        return 64, 64, 64, 4, 3, 128

    if _thead_hgemm_nn_use_desc_bwd_narrow(m, n, k):
        return 128, 256, 64, 8, 3, 160

    if _thead_hgemm_nn_use_desc_bwd(m, n, k):
        return 128, 256, 64, 8, 3, 160

    if _thead_hgemm_nn_use_large_bwd(m, n, k):
        return 128, 128, 64, 4, 3, 128

    if min_mn <= 64:
        return 64, 64, 128, 4, 3, 128

    if min_mn == 128 and max(m, n) >= 4096 and k <= 1024:
        return 128, 128, 64, 8, 3, 128

    if min_mn == 256 and max(m, n) >= 4096:
        if m <= n:
            return 128, 64, 64, 4, 3, 128
        return 128, 128, 128, 8, 3, 128

    if min_mn >= 2048:
        return 128, 128, 128, 8, 3, 128

    if min_mn >= 512:
        if max(m, n) >= 4096:
            return 128, 128, 128, 8, 3, 128
        return 128, 64, 64, 4, 4, 128

    return 128, 128, 64, 8, 3, 128


def _thead_hgemm_nn_use_large_bwd(m: int, n: int, k: int) -> bool:
    return k <= 8192 and max(m, n) >= 8192


def _thead_hgemm_nn_use_bwd(m: int, n: int, k: int) -> bool:
    return _thead_hgemm_nn_use_large_bwd(
        m, n, k
    ) and not _thead_hgemm_nn_use_desc_bwd_narrow(m, n, k)


def _thead_hgemm_nn_use_desc_bwd_narrow(m: int, n: int, k: int) -> bool:
    return 256 <= min(m, n) < 1024 and max(m, n) >= 8192 and k <= 8192


def _thead_hgemm_nn_use_desc_bwd(m: int, n: int, k: int) -> bool:
    return (
        _thead_hgemm_nn_use_desc_bwd_narrow(m, n, k)
        or min(m, n) >= 1024
        or (min(m, n) >= 512 and 512 < max(m, n, k) <= 1024)
    )


def _thead_hgemm_nn_use_trans(m: int, n: int, k: int) -> bool:
    return False


def _thead_hgemm_nn_use_blockptr(m: int, n: int, k: int) -> bool:
    return False


def _thead_hgemm_nn_use_desc(m: int, n: int, k: int, aligned: bool) -> bool:
    return not _thead_hgemm_nn_use_bwd(m, n, k) and not _thead_hgemm_nn_use_desc_bwd(
        m, n, k
    )


def _round_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _thead_hgemm_nn_should_pad(m: int, n: int, k: int) -> bool:
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return m * n * k >= 2048 * 2048 * 2048 and extra <= original * 1.08


def _thead_hgemm_nn_use_splitk(m: int, n: int, k: int) -> bool:
    return False


def _thead_hgemm_nn_splitk_config(m: int, n: int, k: int):
    if min(m, n) <= 64:
        if m <= n:
            return 64, 64, 64, 4, 3, min(triton.cdiv(k, 256), 8)
        return 64, 64, 64, 4, 3, min(triton.cdiv(k, 256), 8)
    if min(m, n) <= 128:
        return 128, 128, 64, 8, 3, min(triton.cdiv(k, 256), 8)
    return 128, 128, 64, 8, 3, min(triton.cdiv(k, 512), 4)


def _thead_hgemm_tn_config(m: int, n: int, k: int):
    """T-Head TN config.

    TN has a different codegen profile from NN because one dot operand is
    transposed in-register. Avoid the NN 128x256 tile on Zhenwu: it is fast for
    NN but consistently poor for direct TN.
    """
    if max(m, n, k) <= 512 and (m % 64 != 0 or n % 64 != 0 or k % 64 != 0):
        return 64, 64, 64, 4, 3, 128

    min_mn = min(m, n)
    if max(m, n, k) <= 512:
        return 64, 64, 64, 4, 3, 128

    if min_mn <= 64:
        if k >= 1024:
            return 64, 64, 128, 4, 3, 128
        if m <= n:
            if n >= 4096 and k >= 2048:
                return 64, 128, 64, 4, 3, 128
            return 64, 64, 64, 4, 3, 128
        if m >= 4096 and k >= 2048:
            return 128, 64, 64, 4, 3, 128
        return 64, 64, 64, 4, 3, 128

    if min_mn == 128 and max(m, n) >= 4096 and k <= 1024:
        return 64, 128, 64, 4, 3, 128

    if m <= n:
        if m == n and m == 1024:
            return 128, 128, 32, 4, 3, 128
        if m == 2048 and n == 2048 and k == 2048:
            return 128, 128, 32, 4, 3, 128
        if m == n and m <= 1024:
            return 128, 128, 64, 8, 3, 128
        return 128, 128, 64, 4, 3, 128

    return 128, 128, 64, 4, 3, 128


def _thead_hgemm_tn_use_trans_a(m: int, n: int, k: int) -> bool:
    return m <= n or (n == 128 and m >= 4096 and k <= 1024)


def _thead_hgemm_tn_desc_overlap_config(m: int, n: int, k: int):
    """Config for the small-shape TN desc-overlap kernel (tuned on 511)."""
    return 64, 128, 64, 8, 3, 128


def _thead_hgemm_tn_use_desc_overlap(m: int, n: int, k: int) -> bool:
    """Use the desc-overlap TN kernel for small non-aligned shapes.

    desc.load fills out-of-bounds rows of the last (partial) K block with
    garbage on Zhenwu, which the overlap kernel avoids by loading the tail
    block fully in-bounds. Only beneficial for small shapes: for larger
    shapes the materialize (transpose + NN) path is faster, since its dot
    has no in-register transpose.
    """
    if max(m, n, k) > 512:
        return False
    _, _, block_k, _, _, _ = _thead_hgemm_tn_desc_overlap_config(m, n, k)
    return k % block_k == block_k - 1 and k >= 2 * block_k


def _thead_hgemm_nt_config(m: int, n: int, k: int):
    """Reuse NN config for NT variant."""
    return _thead_hgemm_nn_config(m, n, k)


def _thead_hgemm_tt_config(m: int, n: int, k: int):
    """T-Head TT config.

    TT uses a transposed accumulator and has a different sweet spot from NN.
    The NN 128x256 tile is competitive for large-K cases but loses on 2048
    square and is not better on the core large shapes.
    """
    min_mn = min(m, n)

    if max(m, n, k) <= 512:
        return 64, 64, 64, 4, 3, 128

    if min_mn <= 64:
        return 64, 64, 128, 4, 3, 128

    if min_mn == 128 and max(m, n) >= 4096 and k <= 1024:
        return 64, 128, 64, 4, 3, 128

    if min_mn >= 1024:
        return 128, 128, 64, 4, 3, 128

    return _thead_hgemm_nn_config(m, n, k)


def _can_use_thead_hgemm_nn(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return lda == k and ldb == n and ldc == n and m >= 16 and n >= 16 and k >= 16


def _can_use_thead_hgemm_tn(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    # A^T: A is (K, M), lda = M; B: (K, N), ldb = N
    return lda == m and ldb == n and ldc == n and m >= 16 and n >= 16 and k >= 16


def _can_use_thead_hgemm_nt(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    # A: (M, K), lda = K; B^T: B is (N, K), ldb = K
    return lda == k and ldb == k and ldc == n and m >= 16 and n >= 16 and k >= 16


def _can_use_thead_hgemm_tt(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    # A^T: A is (K, M), lda = M; B^T: B is (N, K), ldb = K
    return lda == m and ldb == k and ldc == n and m >= 16 and n >= 16 and k >= 16


def _thead_hgemm_is_tile_aligned(
    m: int, n: int, k: int, block_m: int, block_n: int, block_k: int
) -> bool:
    """Check if all dimensions are multiples of their block sizes (no partial tiles)."""
    return m % block_m == 0 and n % block_n == 0 and k % block_k == 0


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_tn_cola_kernel(
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
    FULL: tl.constexpr,
):
    """TN (A^T x B) kernel loading A blocks column-wise.

    A is (K, M) row-major with lda=M; the (BLOCK_M, BLOCK_K) block is read with
    the M dimension contiguous, so the A^T transpose happens implicitly in the
    load addressing and tl.dot sees a normal (BLOCK_M, BLOCK_K) x (BLOCK_K,
    BLOCK_N) pair. FULL selects the fully-aligned, branch-free fast path.
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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

    a_ptrs = a_ptr + offs_m[:, None] + offs_k[None, :] * lda
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    k_full = K // BLOCK_K
    k_rem = K % BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if FULL:
        for _ in range(0, k_full):
            a_block = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a_block = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
        result = alpha * acc
        c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16))
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full):
            a_block = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
        result = alpha * acc
        c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
        c_mask = mask_m[:, None] & mask_n[None, :]
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


def _thead_hgemm_tn_cola_config(m: int, n: int, k: int):
    """Config for the colA TN kernel, tuned on Zhenwu for large aligned shapes."""
    return 128, 128, 64, 8, 4, 112


def _thead_hgemm_tn_use_cola(m: int, n: int, k: int) -> bool:
    """Use the column-loaded (colA) TN kernel for aligned shapes where the
    direct trans-a TN kernel is measurably slower.

    The colA kernel avoids both tl.trans in the dot and the materialize
    (transpose + NN) overhead; it is the fastest option measured on Zhenwu for
    the 2048-square family and large-K shapes.
    """
    if m % 128 != 0 or n % 128 != 0 or k % 64 != 0:
        return False
    if m == n and m == k == 2048:
        return True
    if m == 2048 and n == 2048 and k == 16384:
        return True
    if m == 2048 and n == 4096 and k == 11008:
        return True
    return False


def _run_thead_hgemm_tn_cola(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        maxnreg,
    ) = _thead_hgemm_tn_cola_config(m, n, k)
    _thead_hgemm_tn_cola_kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
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
        GROUP_M=4,
        FULL=True,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _run_thead_hgemm_tn(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    if _thead_hgemm_tn_use_cola(m, n, k):
        _run_thead_hgemm_tn_cola(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        )
        return
    if _thead_hgemm_tn_use_desc_overlap(m, n, k):
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            maxnreg,
        ) = _thead_hgemm_tn_desc_overlap_config(m, n, k)
        _thead_hgemm_tn_desc_overlap_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
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
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
        return

    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_hgemm_tn_config(
        m, n, k
    )
    if _thead_hgemm_tn_use_trans_a(m, n, k):
        kernel = _thead_hgemm_tn_trans_a_kernel
        kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
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
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
        return

    tile_aligned = _thead_hgemm_is_tile_aligned(m, n, k, block_m, block_n, block_k)
    if _thead_hgemm_nn_use_desc_bwd(m, n, k) and tile_aligned:
        kernel = _thead_hgemm_tn_desc_bwd_kernel
    elif _thead_hgemm_nn_use_bwd(m, n, k):
        kernel = _thead_hgemm_tn_bwd_kernel
    elif tile_aligned:
        kernel = _thead_hgemm_tn_desc_kernel
    else:
        kernel = _thead_hgemm_tn_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
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
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _run_thead_hgemm_nt(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_hgemm_nt_config(
        m, n, k
    )
    tile_aligned = _thead_hgemm_is_tile_aligned(m, n, k, block_m, block_n, block_k)
    if _thead_hgemm_nn_use_desc_bwd(m, n, k) and tile_aligned:
        kernel = _thead_hgemm_nt_desc_bwd_kernel
    elif _thead_hgemm_nn_use_bwd(m, n, k):
        kernel = _thead_hgemm_nt_bwd_kernel
    elif tile_aligned:
        kernel = _thead_hgemm_nt_desc_kernel
    else:
        kernel = _thead_hgemm_nt_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
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
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _run_thead_hgemm_tt(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_hgemm_tt_config(
        m, n, k
    )
    tile_aligned = _thead_hgemm_is_tile_aligned(m, n, k, block_m, block_n, block_k)
    if _thead_hgemm_nn_use_desc_bwd(m, n, k) and tile_aligned:
        kernel = _thead_hgemm_tt_desc_bwd_kernel
    elif _thead_hgemm_nn_use_bwd(m, n, k):
        kernel = _thead_hgemm_tt_bwd_kernel
    elif tile_aligned:
        kernel = _thead_hgemm_tt_desc_kernel
    else:
        kernel = _thead_hgemm_tt_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
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
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _thead_hgemm_tn_should_pad(m: int, n: int, k: int) -> bool:
    """Pad TN when non-aligned. Uses same strategy as NT/TT with lower threshold."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return m * n * k >= 256 * 256 * 256 and extra <= original * 1.15


def _thead_hgemm_tn_should_materialize(m: int, n: int, k: int) -> bool:
    """Materialize A^T (or B^T) and use NN (or TT) kernel.

    Direct TN kernel uses tl.trans which is slow (0.77-0.81x of NN).
    By materializing the cheaper transpose, we get near 1.0x NN performance.
    - If M <= N: materialize A^T, use NN kernel
    - If N < M: materialize B^T, use TT transpose-free kernel

    Extended to non-aligned shapes where padding + materialization enables
    the desc_bwd kernel, avoiding both tl.trans and masked loads.
    """
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        # On aligned square/large-K shapes the direct trans-a TN kernel avoids
        # the copy overhead and is faster than materializing A.T. Keep
        # materialization for skinny cases where the copied operand is small
        # enough and the NN/TT kernels recover more compute throughput.
        if m == n and m >= 8192:
            return True
        if min(m, n) <= 64:
            return False
        if m != n and max(m, n) >= 8 * min(m, n) and min(m, n) >= 1024:
            # Semi-skinny aligned shapes (e.g. 2048x16384, 16384x2048,
            # 32768x1024): the transpose is cheap relative to the GEMM and the
            # transpose-free NN/TT kernels beat the direct trans-a TN kernel.
            return True
        if m <= n:
            return m <= 512 and n >= 4096 and k >= 2048
        return n <= 512 and m >= 4096 and k >= 2048

    # For non-aligned shapes: check if padded version benefits from desc_bwd
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    if m * n * k < 256 * 256 * 256 or extra > original * 1.15:
        return False
    # Materialize if padded version would use desc_bwd (e.g. (1023,1023,1023))
    if _thead_hgemm_nn_use_desc_bwd(m_pad, n_pad, k_pad):
        return True
    # For small non-aligned shapes: materialization avoids tl.trans even if
    # desc_bwd isn't available on padded dims (e.g. (511,511,511))
    if max(m, n, k) <= 2048:
        return True
    return False


def _thead_hgemm_nt_should_pad(m: int, n: int, k: int) -> bool:
    """Pad NT when non-aligned and overhead is reasonable."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return m * n * k >= 256 * 256 * 256 and extra <= original * 1.15


def _thead_hgemm_tn_use_narrow_materialize(m: int, n: int, k: int) -> bool:
    """TN materialize-A variant that avoids the padA/padB copies.

    A^T is transposed directly into a row-padded (M_pad, K) buffer with
    lda == K, then the desc_bwd NN kernel runs on unpadded B (N = n, ldb = n)
    into a row/col-padded C (ldc = n_pad) which is cropped.  The desc kernel
    zero-fills the tail K-block only when lda == K, so this path is only valid
    when K is not a multiple of 64 (i.e. when padding is otherwise needed).
    Only used for shapes where the desc_bwd kernel is selected (min(m, n) >=
    1024) and the plain aligned path is not applicable.
    """
    if m > n:
        return False
    if max(m, n, k) <= 1024:
        return False
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    if k % 64 == 0:
        return False
    if min(m, n) < 1024:
        return False
    return True


def _thead_hgemm_nt_should_materialize(m: int, n: int, k: int) -> bool:
    """Materialize B^T for odd NT cases where padding wastes too much time."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    if max(m, n, k) <= 1024:
        return True
    return m != n and m * n * k >= 256 * 256 * 256


def _thead_hgemm_tt_should_pad(m: int, n: int, k: int) -> bool:
    """Pad TT when non-aligned and overhead is reasonable."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return m * n * k >= 256 * 256 * 256 and extra <= original * 1.15


def _thead_hgemm_tt_should_materialize(m: int, n: int, k: int) -> bool:
    """Materialize C^T for odd TT cases where padding overhead dominates."""
    if m % 64 == 0:
        return False
    if m == n and n == k:
        return max(m, n, k) <= 1024
    if m != n and m * n * k >= 256 * 256 * 256:
        return True
    return False


def _thead_hgemm_transpose2d(src, rows: int, cols: int, src_ld: int):
    if rows <= 512 and cols <= 512:
        block_m, block_n = 64, 16
    elif rows <= 1024 and cols <= 1024:
        block_m, block_n = 32, 32
    else:
        block_m, block_n = 16, 64

    dst = torch.empty((cols, rows), dtype=src.dtype, device=src.device)
    _thead_hgemm_transpose2d_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    ](
        src,
        dst,
        rows,
        cols,
        src_ld,
        rows,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return dst


def _run_thead_hgemm_nt_materialized(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    B_T = _thead_hgemm_transpose2d(B, n, k, ldb)
    if _thead_hgemm_nn_should_pad(m, n, k):
        _run_thead_hgemm_nn_padded(
            A, lda, B_T, n, C, ldc, m, n, k, alpha, beta, beta_is_zero
        )
    else:
        _run_thead_hgemm_nn(
            A,
            lda,
            B_T,
            n,
            C,
            ldc,
            m,
            n,
            k,
            alpha,
            beta,
            beta_is_zero,
            True,
        )


def _run_thead_hgemm_tn_materialize_a(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    A_T = _thead_hgemm_transpose2d(A, k, m, lda)
    _run_thead_hgemm_nn(
        A_T,
        k,
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
        True,
    )


def _run_thead_hgemm_tn_materialize_a_narrow(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """TN materialize-A without padA/padB: transpose into (M_pad, K) with
    lda == K, run desc_bwd NN with unpadded B, crop padded C."""
    m_pad = _round_up(m, 128)
    n_pad = _round_up(n, 64)
    A_pad = torch.empty((m_pad, k), dtype=A.dtype, device=A.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    if max(m, k) <= 1024:
        block_m, block_n = 32, 32
    else:
        block_m, block_n = 16, 64
    _thead_hgemm_transpose_pad_kernel[
        (triton.cdiv(m_pad, block_m), triton.cdiv(k, block_n))
    ](
        A,
        A_pad,
        k,
        m,
        lda,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    _run_thead_hgemm_nn(
        A_pad,
        k,
        B,
        ldb,
        C_pad,
        n_pad,
        m_pad,
        n,
        k,
        alpha,
        0.0,
        True,
        True,
    )
    pad_block = 1024
    _thead_hgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def _run_thead_hgemm_tn_materialize_b(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    B_T = _thead_hgemm_transpose2d(B, k, n, ldb)
    _run_thead_hgemm_tt(
        A,
        lda,
        B_T,
        k,
        C,
        ldc,
        m,
        n,
        k,
        alpha,
        beta,
        beta_is_zero,
        True,
    )


def _run_thead_hgemm_tt_materialized(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    C_T = torch.empty((n, m), dtype=C.dtype, device=C.device)
    if _thead_hgemm_nn_should_pad(n, m, k):
        _run_thead_hgemm_nn_padded(B, ldb, A, lda, C_T, m, n, m, k, alpha, 0.0, True)
    else:
        _run_thead_hgemm_nn(B, ldb, A, lda, C_T, m, n, m, k, alpha, 0.0, True, True)

    if m <= 512 and n <= 512:
        block_m, block_n = 8, 64
    elif m * n >= 2048 * 2048 and m != n:
        block_m, block_n = 16, 128
    else:
        block_m, block_n = 16, 64
    _thead_hgemm_transpose_c_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        C_T,
        C,
        beta,
        m,
        n,
        m,
        ldc,
        beta_is_zero,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )


def _run_thead_hgemm_tn_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Pad A(K,M) and B(K,N) to multiples of 64, then crop result."""
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    # A is (K, M) with lda=M
    A_pad = torch.empty((k_pad, m_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((k_pad, n_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_hgemm_pad2d_kernel[(triton.cdiv(k_pad * m_pad, pad_block),)](
        A, A_pad, k, m, lda, m_pad, k_pad, m_pad, BLOCK_SIZE=pad_block
    )
    _thead_hgemm_pad2d_kernel[(triton.cdiv(k_pad * n_pad, pad_block),)](
        B, B_pad, k, n, ldb, n_pad, k_pad, n_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_hgemm_tn(
        A_pad,
        m_pad,
        B_pad,
        n_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    _thead_hgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def _run_thead_hgemm_nt_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Pad A(M,K) and B(N,K) to multiples of 64, then crop result."""
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((m_pad, k_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((n_pad, k_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_hgemm_pad2d_kernel[(triton.cdiv(m_pad * k_pad, pad_block),)](
        A, A_pad, m, k, lda, k_pad, m_pad, k_pad, BLOCK_SIZE=pad_block
    )
    _thead_hgemm_pad2d_kernel[(triton.cdiv(n_pad * k_pad, pad_block),)](
        B, B_pad, n, k, ldb, k_pad, n_pad, k_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_hgemm_nt(
        A_pad,
        k_pad,
        B_pad,
        k_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    _thead_hgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def _run_thead_hgemm_tt_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Pad A(K,M) and B(N,K) to multiples of 64, then crop result."""
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((k_pad, m_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((n_pad, k_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_hgemm_pad2d_kernel[(triton.cdiv(k_pad * m_pad, pad_block),)](
        A, A_pad, k, m, lda, m_pad, k_pad, m_pad, BLOCK_SIZE=pad_block
    )
    _thead_hgemm_pad2d_kernel[(triton.cdiv(n_pad * k_pad, pad_block),)](
        B, B_pad, n, k, ldb, k_pad, n_pad, k_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_hgemm_tt(
        A_pad,
        m_pad,
        B_pad,
        k_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    _thead_hgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def _run_thead_hgemm_nn(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_hgemm_nn_config(
        m, n, k
    )
    kernel = _thead_hgemm_nn_kernel
    if _thead_hgemm_nn_use_desc_bwd(m, n, k):
        kernel = _thead_hgemm_nn_desc_bwd_kernel
    elif _thead_hgemm_nn_use_bwd(m, n, k):
        kernel = _thead_hgemm_nn_bwd_kernel
    elif _thead_hgemm_nn_use_desc(m, n, k, aligned):
        kernel = _thead_hgemm_nn_desc_kernel
    elif _thead_hgemm_nn_use_trans(m, n, k):
        kernel = _thead_hgemm_nn_trans_kernel
    elif _thead_hgemm_nn_use_blockptr(m, n, k):
        kernel = _thead_hgemm_nn_blockptr_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
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
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _run_thead_hgemm_nn_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((m_pad, k_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((k_pad, n_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_hgemm_pad2d_kernel[(triton.cdiv(m_pad * k_pad, pad_block),)](
        A, A_pad, m, k, lda, k_pad, m_pad, k_pad, BLOCK_SIZE=pad_block
    )
    _thead_hgemm_pad2d_kernel[(triton.cdiv(k_pad * n_pad, pad_block),)](
        B, B_pad, k, n, ldb, n_pad, k_pad, n_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_hgemm_nn(
        A_pad,
        k_pad,
        B_pad,
        n_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    _thead_hgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def _run_thead_hgemm_nn_splitk(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        split_k,
    ) = _thead_hgemm_nn_splitk_config(m, n, k)
    tmp = torch.empty((m, n), dtype=torch.float32, device=C.device)
    block = 1024
    _thead_hgemm_zero_f32_kernel[(triton.cdiv(m * n, block),)](
        tmp, m * n, BLOCK_SIZE=block
    )
    _thead_hgemm_nn_splitk_kernel[
        (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), split_k)
    ](
        A,
        B,
        tmp,
        alpha,
        lda,
        ldb,
        n,
        M=m,
        N=n,
        K=k,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _thead_hgemm_f32_to_h_kernel[(triton.cdiv(m * n, block),)](
        tmp, C, beta, m, n, n, ldc, beta_is_zero, BLOCK_SIZE=block
    )


def hgemm(
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
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert C.dtype == torch.float16
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

    aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
    with torch_device_fn.device(A.device):
        if (
            transa == CUBLAS_OP_N
            and transb == CUBLAS_OP_N
            and _can_use_thead_hgemm_nn(m, n, k, lda, ldb, ldc, alpha, beta)
        ):
            if _thead_hgemm_nn_should_pad(m, n, k):
                _run_thead_hgemm_nn_padded(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            elif _thead_hgemm_nn_use_splitk(m, n, k):
                _run_thead_hgemm_nn_splitk(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            else:
                _run_thead_hgemm_nn(
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
                    aligned,
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _hgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if _can_use_thead_hgemm_tn(m, n, k, lda, ldb, ldc, alpha, beta):
                if _thead_hgemm_tn_use_desc_overlap(m, n, k):
                    # Small non-aligned TN: single fused kernel, avoids the
                    # materialize (transpose + NN) overhead.
                    _run_thead_hgemm_tn(
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
                        aligned,
                    )
                elif _thead_hgemm_tn_should_materialize(m, n, k):
                    if m <= n:
                        # M <= N: Materialize A^T as (M, K) and use fast NN kernel.
                        # A is (K, M) with lda=M, A^T will be (M, K) with lda=K.
                        if max(m, n, k) <= 1024:
                            _run_thead_hgemm_tn_materialize_a(
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
                            )
                        elif _thead_hgemm_nn_should_pad(
                            m, n, k
                        ) or _thead_hgemm_nt_should_pad(m, n, k):
                            if _thead_hgemm_tn_use_narrow_materialize(m, n, k):
                                _run_thead_hgemm_tn_materialize_a_narrow(
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
                                )
                            else:
                                A_T = _thead_hgemm_transpose2d(A, k, m, lda)
                                _run_thead_hgemm_nn_padded(
                                    A_T,
                                    k,
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
                                )
                        else:
                            _run_thead_hgemm_tn_materialize_a(
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
                            )
                    else:
                        # N < M: Materialize B^T as (N, K) and use TT transpose-free kernel.
                        # B is (K, N) with ldb=N, B^T will be (N, K) with lda_BT=K.
                        # TT kernel computes C(M,N) = A^T(K,M) x B_T^T(K,N) = (M,K)x(K,N).
                        if max(m, n, k) <= 1024:
                            _run_thead_hgemm_tn_materialize_b(
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
                            )
                        elif _thead_hgemm_nn_should_pad(
                            m, n, k
                        ) or _thead_hgemm_tt_should_pad(m, n, k):
                            B_T = _thead_hgemm_transpose2d(B, k, n, ldb)
                            _run_thead_hgemm_tt_padded(
                                A,
                                lda,
                                B_T,
                                k,
                                C,
                                ldc,
                                m,
                                n,
                                k,
                                alpha,
                                beta,
                                beta_is_zero,
                            )
                        else:
                            _run_thead_hgemm_tn_materialize_b(
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
                            )
                elif _thead_hgemm_tn_should_pad(m, n, k):
                    _run_thead_hgemm_tn_padded(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                else:
                    _run_thead_hgemm_tn(
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
                        aligned,
                    )
            else:
                _hgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            if _can_use_thead_hgemm_nt(m, n, k, lda, ldb, ldc, alpha, beta):
                if _thead_hgemm_nt_should_materialize(m, n, k):
                    _run_thead_hgemm_nt_materialized(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                elif _thead_hgemm_nt_should_pad(m, n, k):
                    _run_thead_hgemm_nt_padded(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                else:
                    _run_thead_hgemm_nt(
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
                        aligned,
                    )
            else:
                _hgemm_nt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        else:
            if _can_use_thead_hgemm_tt(m, n, k, lda, ldb, ldc, alpha, beta):
                if _thead_hgemm_tt_should_materialize(m, n, k):
                    _run_thead_hgemm_tt_materialized(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                elif _thead_hgemm_tt_should_pad(m, n, k):
                    _run_thead_hgemm_tt_padded(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                else:
                    _run_thead_hgemm_tt(
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
                        aligned,
                    )
            else:
                _hgemm_tt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )


__all__ = ["hgemm"]

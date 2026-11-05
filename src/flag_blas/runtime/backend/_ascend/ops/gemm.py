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

"""Ascend 910B specialized GEMM (level-3) implementations.

All kernels below follow a loop pattern that is known to be compiled and
executed correctly by the Ascend (flagtree) Triton backend:

    for kk in range(0, k, BLOCK_K):
        km = (kk + offs_k) < k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & km[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=km[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        ...

The "split into full iterations + remainder tail block" pattern used by the
generic level-3 kernels produces wrong results on Ascend, so it is avoided
here entirely.
"""

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


# ---------------------------------------------------------------------------
# sgemm (fp32)
# ---------------------------------------------------------------------------
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
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    if TRANS_A == 0:
        a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
        a_adv = BLOCK_K
    else:
        # Load A^T in its contiguous (K, M) layout and transpose in-kernel.
        # Indexing A as (K, M) rows keeps the inner dim (M) contiguous, which
        # the Ascend compiler turns into fast linear loads; the strided
        # (M, K) indexing used before compiled to a ~190x slower kernel.
        a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
        a_adv = BLOCK_K * lda
    if TRANS_B == 0:
        b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        b_adv = BLOCK_K * ldb
    else:
        # Same treatment for B: load B^T as (N, K) rows (inner K contiguous)
        # and transpose back to (K, N) for the dot.
        b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        b_adv = BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, k, BLOCK_K):
        km = (kk + offs_k) < k
        if TRANS_A == 0:
            a = tl.load(a_ptrs, mask=mask_m[:, None] & km[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=km[:, None] & mask_m[None, :], other=0.0)
            a = tl.trans(a)
        if TRANS_B == 0:
            b = tl.load(b_ptrs, mask=km[:, None] & mask_n[None, :], other=0.0)
        else:
            b = tl.load(b_ptrs, mask=mask_n[:, None] & km[None, :], other=0.0)
            b = tl.trans(b)
        acc = tl.dot(a, b, acc)
        a_ptrs += a_adv
        b_ptrs += b_adv

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    # NOTE: alpha * acc on the accumulator is expensive on Ascend (it moves
    # the dot result off the cube fast path), so special-case alpha == 1.0.
    if BETA_IS_ZERO:
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, acc, mask=c_mask)
        else:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, acc + beta * c_vals, mask=c_mask)
        else:
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


# ---------------------------------------------------------------------------
# Padding helpers (odd-dimension GEMMs)
# ---------------------------------------------------------------------------
# Odd leading dimensions (e.g. 1023) break 64B cache-line alignment of every
# matrix row and force masked loads on the boundary tiles; on Ascend this is
# worth 26% (stride) to 61% (masked boundary) of the kernel time.  Copying the
# operands into buffers whose strides are multiples of the tile size lets the
# GEMM run fully unmasked and aligned.  The copy kernels below are plain
# Triton (no torch ops) and every element of the padded region is explicitly
# written, so the destination buffers never need pre-zeroing.
COPY_BM = 64
COPY_BN = 512


@triton.jit
def _pad2d_kernel(src, dst, src_rows, src_cols, dst_rows, dst_cols,
                  GRID_N: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Copy a (src_rows, src_cols) row-major matrix into the top-left of a
    (dst_rows, dst_cols) buffer.  Every dst element within the padded region
    is explicitly zeroed; the grid must cover the whole destination."""
    pid = tl.program_id(0)
    pid_m = pid // GRID_N
    pid_n = pid % GRID_N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_valid = offs_m < src_rows
    n_valid = offs_n < src_cols
    val = tl.load(src + offs_m[:, None] * src_cols + offs_n[None, :],
                  mask=m_valid[:, None] & n_valid[None, :], other=0.0)
    m_dst = offs_m < dst_rows
    n_dst = offs_n < dst_cols
    tl.store(dst + offs_m[:, None] * dst_cols + offs_n[None, :], val,
             mask=m_dst[:, None] & n_dst[None, :])


@triton.jit
def _slice2d_kernel(src, dst, rows, cols, src_cols, dst_cols,
                    GRID_N: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Copy the (rows, cols) top-left block of src into dst."""
    pid = tl.program_id(0)
    pid_m = pid // GRID_N
    pid_n = pid % GRID_N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_valid = offs_m < rows
    n_valid = offs_n < cols
    val = tl.load(src + offs_m[:, None] * src_cols + offs_n[None, :],
                  mask=m_valid[:, None] & n_valid[None, :], other=0.0)
    tl.store(dst + offs_m[:, None] * dst_cols + offs_n[None, :], val,
             mask=m_valid[:, None] & n_valid[None, :])


_PAD_WORKSPACE = {}


def _pad_workspace(device, mp, np_, kp, dtype):
    key = (str(device), mp, np_, kp, str(dtype))
    buf = _PAD_WORKSPACE.get(key)
    if buf is None:
        buf = (
            torch.empty(mp * kp, dtype=dtype, device=device),
            torch.empty(kp * np_, dtype=dtype, device=device),
            torch.empty(mp * np_, dtype=dtype, device=device),
        )
        _PAD_WORKSPACE[key] = buf
    return buf


def _should_pad(m, n, k, block_m, block_n, block_k):
    # Padding only helps when the copy overhead is a small fraction of the
    # GEMM; tiny shapes (longest axis <= 512, e.g. 511^3) lose more on the
    # copies than they win on alignment, so run them unpadded.
    if max(m, n, k) <= 512:
        return False
    return (m % block_m != 0) or (n % block_n != 0) or (k % block_k != 0)


def _sgemm_padded(
    transa, transb, m, n, k, alpha, beta, A, B, C, ldc, beta_is_zero,
    bm, bn, bk, nw, gm,
):
    mp = triton.cdiv(m, bm) * bm
    np_ = triton.cdiv(n, bn) * bn
    kp = triton.cdiv(k, bk) * bk
    bm2, bn2, bk2, nw2, gm2 = _select_gemm_config(transa, transb, mp, np_, kp)
    Ap, Bp, Cp = _pad_workspace(A.device, mp, np_, kp, A.dtype)
    with torch_device_fn.device(A.device):
        # A occupies the top-left (m, k) [or (k, m) when transposed] block.
        a_rows = m if transa == 0 else k
        a_cols = k if transa == 0 else m
        a_dst_rows = mp if transa == 0 else kp
        a_dst_cols = kp if transa == 0 else mp
        a_gn = triton.cdiv(a_dst_cols, COPY_BN)
        _pad2d_kernel[((triton.cdiv(a_dst_rows, COPY_BM) * a_gn),)](
            A, Ap, a_rows, a_cols, a_dst_rows, a_dst_cols, GRID_N=a_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        b_rows = k if transb == 0 else n
        b_cols = n if transb == 0 else k
        b_dst_rows = kp if transb == 0 else np_
        b_dst_cols = np_ if transb == 0 else kp
        b_gn = triton.cdiv(b_dst_cols, COPY_BN)
        _pad2d_kernel[((triton.cdiv(b_dst_rows, COPY_BM) * b_gn),)](
            B, Bp, b_rows, b_cols, b_dst_rows, b_dst_cols, GRID_N=b_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        if not beta_is_zero:
            # Cp is a reused workspace, so seed the (m, n) block with the real
            # C for the beta*C term (the padding region stays 0).
            c_gn = triton.cdiv(np_, COPY_BN)
            _pad2d_kernel[((triton.cdiv(mp, COPY_BM) * c_gn),)](
                C, Cp, m, n, mp, np_, GRID_N=c_gn,
                BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        # Fully aligned GEMM: no masks anywhere (mp/np_/kp are tile multiples).
        grid = (triton.cdiv(mp, bm2) * triton.cdiv(np_, bn2),)
        _sgemm_kernel[grid](
            Ap, Bp, Cp, alpha, beta, mp, np_, kp,
            kp if transa == 0 else mp,
            np_ if transb == 0 else kp,
            np_,
            TRANS_A=transa, TRANS_B=transb, BETA_IS_ZERO=beta_is_zero,
            ALPHA_IS_ONE=(alpha == 1.0),
            BLOCK_M=bm2, BLOCK_N=bn2, BLOCK_K=bk2, GROUP_M=gm2,
            num_warps=nw2,
        )
        # Copy the valid (m, n) block back into C.
        c_gn = triton.cdiv(n, COPY_BN)
        _slice2d_kernel[((triton.cdiv(m, COPY_BM) * c_gn),)](
            Cp, C, m, n, np_, ldc, GRID_N=c_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)


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

    beta_is_zero = beta == 0.0
    block_m, block_n, block_k, num_warps, group_m = _select_gemm_config(
        transa, transb, m, n, k
    )

    if _should_pad(m, n, k, block_m, block_n, block_k):
        _sgemm_padded(
            transa, transb, m, n, k, alpha, beta, A, B, C, ldc,
            beta_is_zero, block_m, block_n, block_k, num_warps, group_m,
        )
        return

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    with torch_device_fn.device(A.device):
        _sgemm_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
            TRANS_A=transa, TRANS_B=transb, BETA_IS_ZERO=beta_is_zero,
            ALPHA_IS_ONE=(alpha == 1.0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m, num_warps=num_warps,
        )


# ---------------------------------------------------------------------------
# hgemm (fp16)
# ---------------------------------------------------------------------------
@triton.jit
def _hgemm_kernel(
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
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    if TRANS_A == 0:
        a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
        a_adv = BLOCK_K
    else:
        # Load A^T in its contiguous (K, M) layout and transpose in-kernel.
        # Indexing A as (K, M) rows keeps the inner dim (M) contiguous, which
        # the Ascend compiler turns into fast linear loads; the strided
        # (M, K) indexing used before compiled to a ~190x slower kernel.
        a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
        a_adv = BLOCK_K * lda
    if TRANS_B == 0:
        b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        b_adv = BLOCK_K * ldb
    else:
        # Same treatment for B: load B^T as (N, K) rows (inner K contiguous)
        # and transpose back to (K, N) for the dot.
        b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        b_adv = BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, k, BLOCK_K):
        km = (kk + offs_k) < k
        if TRANS_A == 0:
            a = tl.load(a_ptrs, mask=mask_m[:, None] & km[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=km[:, None] & mask_m[None, :], other=0.0)
            a = tl.trans(a)
        if TRANS_B == 0:
            b = tl.load(b_ptrs, mask=km[:, None] & mask_n[None, :], other=0.0)
        else:
            b = tl.load(b_ptrs, mask=mask_n[:, None] & km[None, :], other=0.0)
            b = tl.trans(b)
        acc = tl.dot(a, b, acc)
        a_ptrs += a_adv
        b_ptrs += b_adv

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    # NOTE: alpha * acc on the accumulator is expensive on Ascend (it moves
    # the dot result off the cube fast path), so special-case alpha == 1.0.
    if BETA_IS_ZERO:
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
        else:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, (acc + beta * c_vals).to(tl.float16), mask=c_mask)
        else:
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


def _hgemm_padded(
    transa, transb, m, n, k, alpha, beta, A, B, C, ldc, beta_is_zero,
    bm, bn, bk, nw, gm,
):
    """Pad A/B/C up to tile multiples, run the unmasked hgemm kernel, then
    slice the valid (m, n) block back into C.  Odd shapes (e.g. 1023^3,
    4095^3, 8191^3) lose a lot to masked loads in the plain kernel; the two
    extra copies are cheap by comparison."""
    mp = triton.cdiv(m, bm) * bm
    np_ = triton.cdiv(n, bn) * bn
    kp = triton.cdiv(k, bk) * bk
    bm2, bn2, bk2, nw2, gm2 = _select_gemm_config_f16(transa, transb, mp, np_, kp)
    Ap, Bp, Cp = _pad_workspace(A.device, mp, np_, kp, A.dtype)
    with torch_device_fn.device(A.device):
        a_rows = m if transa == 0 else k
        a_cols = k if transa == 0 else m
        a_dst_rows = mp if transa == 0 else kp
        a_dst_cols = kp if transa == 0 else mp
        a_gn = triton.cdiv(a_dst_cols, COPY_BN)
        _pad2d_kernel[((triton.cdiv(a_dst_rows, COPY_BM) * a_gn),)](
            A, Ap, a_rows, a_cols, a_dst_rows, a_dst_cols, GRID_N=a_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        b_rows = k if transb == 0 else n
        b_cols = n if transb == 0 else k
        b_dst_rows = kp if transb == 0 else np_
        b_dst_cols = np_ if transb == 0 else kp
        b_gn = triton.cdiv(b_dst_cols, COPY_BN)
        _pad2d_kernel[((triton.cdiv(b_dst_rows, COPY_BM) * b_gn),)](
            B, Bp, b_rows, b_cols, b_dst_rows, b_dst_cols, GRID_N=b_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        if not beta_is_zero:
            # Cp is a fresh/reused workspace, so seed the (m, n) block with
            # the real C for the beta*C term (the padding region stays 0).
            c_gn = triton.cdiv(np_, COPY_BN)
            _pad2d_kernel[((triton.cdiv(mp, COPY_BM) * c_gn),)](
                C, Cp, m, n, mp, np_, GRID_N=c_gn,
                BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        # Fully aligned GEMM: no masks anywhere (mp/np_/kp are tile multiples).
        grid = (triton.cdiv(mp, bm2) * triton.cdiv(np_, bn2),)
        _hgemm_kernel[grid](
            Ap, Bp, Cp, alpha, beta, mp, np_, kp,
            kp if transa == 0 else mp,
            np_ if transb == 0 else kp,
            np_,
            TRANS_A=transa, TRANS_B=transb, BETA_IS_ZERO=beta_is_zero,
            ALPHA_IS_ONE=(alpha == 1.0),
            BLOCK_M=bm2, BLOCK_N=bn2, BLOCK_K=bk2, GROUP_M=gm2,
            num_warps=nw2,
        )
        # Copy the valid (m, n) block back into C.
        c_gn = triton.cdiv(n, COPY_BN)
        _slice2d_kernel[((triton.cdiv(m, COPY_BM) * c_gn),)](
            Cp, C, m, n, np_, ldc, GRID_N=c_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)


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
    block_m, block_n, block_k, num_warps, group_m = _select_gemm_config_f16(
        transa, transb, m, n, k
    )

    if _should_pad(m, n, k, block_m, block_n, block_k):
        _hgemm_padded(
            transa, transb, m, n, k, alpha, beta, A, B, C, ldc,
            beta_is_zero, block_m, block_n, block_k, num_warps, group_m,
        )
        return

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    with torch_device_fn.device(A.device):
        _hgemm_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
            TRANS_A=transa, TRANS_B=transb, BETA_IS_ZERO=beta_is_zero,
            ALPHA_IS_ONE=(alpha == 1.0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m, num_warps=num_warps,
        )


# ---------------------------------------------------------------------------
# bfgemm (bf16)
# ---------------------------------------------------------------------------
@triton.jit
def _bfgemm_kernel(
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
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    if TRANS_A == 0:
        a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
        a_adv = BLOCK_K
    else:
        # Load A^T in its contiguous (K, M) layout and transpose in-kernel.
        # Indexing A as (K, M) rows keeps the inner dim (M) contiguous, which
        # the Ascend compiler turns into fast linear loads; the strided
        # (M, K) indexing used before compiled to a ~190x slower kernel.
        a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
        a_adv = BLOCK_K * lda
    if TRANS_B == 0:
        b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        b_adv = BLOCK_K * ldb
    else:
        # Same treatment for B: load B^T as (N, K) rows (inner K contiguous)
        # and transpose back to (K, N) for the dot.
        b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        b_adv = BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, k, BLOCK_K):
        km = (kk + offs_k) < k
        if TRANS_A == 0:
            a = tl.load(a_ptrs, mask=mask_m[:, None] & km[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=km[:, None] & mask_m[None, :], other=0.0)
            a = tl.trans(a)
        if TRANS_B == 0:
            b = tl.load(b_ptrs, mask=km[:, None] & mask_n[None, :], other=0.0)
        else:
            b = tl.load(b_ptrs, mask=mask_n[:, None] & km[None, :], other=0.0)
            b = tl.trans(b)
        acc = tl.dot(a, b, acc)
        a_ptrs += a_adv
        b_ptrs += b_adv

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    # NOTE: alpha * acc on the accumulator is expensive on Ascend (it moves
    # the dot result off the cube fast path), so special-case alpha == 1.0.
    if BETA_IS_ZERO:
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
        else:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, (acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)
        else:
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


def _bfgemm_padded(
    transa, transb, m, n, k, alpha, beta, A, B, C, ldc, beta_is_zero,
    bm, bn, bk, nw, gm,
):
    """Pad A/B/C up to tile multiples, run the unmasked bfgemm kernel, then
    slice the valid (m, n) block back into C.

    Same treatment as ``_hgemm_padded``: odd shapes (511^3, 1023^3, 4095^3,
    8191^3, ...) lose a lot to masked loads in the plain kernel, while the two
    extra copies are cheap by comparison.
    """
    mp = triton.cdiv(m, bm) * bm
    np_ = triton.cdiv(n, bn) * bn
    kp = triton.cdiv(k, bk) * bk
    bm2, bn2, bk2, nw2, gm2 = _select_gemm_config_f16(transa, transb, mp, np_, kp)
    Ap, Bp, Cp = _pad_workspace(A.device, mp, np_, kp, A.dtype)
    with torch_device_fn.device(A.device):
        a_rows = m if transa == 0 else k
        a_cols = k if transa == 0 else m
        a_dst_rows = mp if transa == 0 else kp
        a_dst_cols = kp if transa == 0 else mp
        a_gn = triton.cdiv(a_dst_cols, COPY_BN)
        _pad2d_kernel[((triton.cdiv(a_dst_rows, COPY_BM) * a_gn),)](
            A, Ap, a_rows, a_cols, a_dst_rows, a_dst_cols, GRID_N=a_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        b_rows = k if transb == 0 else n
        b_cols = n if transb == 0 else k
        b_dst_rows = kp if transb == 0 else np_
        b_dst_cols = np_ if transb == 0 else kp
        b_gn = triton.cdiv(b_dst_cols, COPY_BN)
        _pad2d_kernel[((triton.cdiv(b_dst_rows, COPY_BM) * b_gn),)](
            B, Bp, b_rows, b_cols, b_dst_rows, b_dst_cols, GRID_N=b_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        if not beta_is_zero:
            # Cp is a reused workspace, so seed the (m, n) block with the real
            # C for the beta*C term (the padding region stays 0).
            c_gn = triton.cdiv(np_, COPY_BN)
            _pad2d_kernel[((triton.cdiv(mp, COPY_BM) * c_gn),)](
                C, Cp, m, n, mp, np_, GRID_N=c_gn,
                BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)
        # Fully aligned GEMM: no masks anywhere (mp/np_/kp are tile multiples).
        grid = (triton.cdiv(mp, bm2) * triton.cdiv(np_, bn2),)
        _bfgemm_kernel[grid](
            Ap, Bp, Cp, alpha, beta, mp, np_, kp,
            kp if transa == 0 else mp,
            np_ if transb == 0 else kp,
            np_,
            TRANS_A=transa, TRANS_B=transb, BETA_IS_ZERO=beta_is_zero,
            ALPHA_IS_ONE=(alpha == 1.0),
            BLOCK_M=bm2, BLOCK_N=bn2, BLOCK_K=bk2, GROUP_M=gm2,
            num_warps=nw2,
        )
        # Copy the valid (m, n) block back into C.
        c_gn = triton.cdiv(n, COPY_BN)
        _slice2d_kernel[((triton.cdiv(m, COPY_BM) * c_gn),)](
            Cp, C, m, n, np_, ldc, GRID_N=c_gn,
            BLOCK_M=COPY_BM, BLOCK_N=COPY_BN)


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
    # bf16 shares the fp16 tile table: both operands are 16-bit, so the larger
    # (128, 256, 256)-class tiles that fit the 128KB Unified Buffer for fp16
    # apply here too.  The fp32 table in _select_gemm_config leaves ~30% on the
    # table for the large cubic shapes.
    block_m, block_n, block_k, num_warps, group_m = _select_gemm_config_f16(
        transa, transb, m, n, k
    )

    if _should_pad(m, n, k, block_m, block_n, block_k):
        _bfgemm_padded(
            transa, transb, m, n, k, alpha, beta, A, B, C, ldc,
            beta_is_zero, block_m, block_n, block_k, num_warps, group_m,
        )
        return

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    with torch_device_fn.device(A.device):
        _bfgemm_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
            TRANS_A=transa, TRANS_B=transb, BETA_IS_ZERO=beta_is_zero,
            ALPHA_IS_ONE=(alpha == 1.0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m, num_warps=num_warps,
        )


# ---------------------------------------------------------------------------
# dgemm (fp64)
#
# The Ascend BiShengHIR pipeline cannot handle fp64 at all (even a plain fp64
# load/store fails to compile).  We therefore emulate fp64 GEMM with a hi/lo
# fp32 split:
#
#     x ~= x_hi + x_lo       (x_hi = fl32(x), x_lo = fl32(x - x_hi))
#     A@B ~= Ah@Bh + Ah@Bl + Al@Bh + Al@Bl
#
# Every product is computed with fp32 tl.dot, so the four GEMMs together
# capture ~48 bits of the 53-bit fp64 mantissa.  The hi/lo split and the final
# alpha/beta epilogue run on the host in fp64 (torch_npu supports fp64
# elementwise ops, only fp64 matmul is missing).
# ---------------------------------------------------------------------------
@triton.jit
def _dgemm_split_kernel(
    ah_ptr,
    al_ptr,
    bh_ptr,
    bl_ptr,
    s_hi_ptr,
    s_lo_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    lds,
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
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    if TRANS_A == 0:
        ah_ptrs = ah_ptr + offs_m[:, None] * lda + offs_k[None, :]
        al_ptrs = al_ptr + offs_m[:, None] * lda + offs_k[None, :]
        a_adv = BLOCK_K
    else:
        ah_ptrs = ah_ptr + offs_k[:, None] * lda + offs_m[None, :]
        al_ptrs = al_ptr + offs_k[:, None] * lda + offs_m[None, :]
        a_adv = BLOCK_K * lda
    if TRANS_B == 0:
        bh_ptrs = bh_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        bl_ptrs = bl_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        b_adv = BLOCK_K * ldb
    else:
        bh_ptrs = bh_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        bl_ptrs = bl_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        b_adv = BLOCK_K

    acc_hi = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # Ah @ Bh
    acc_lo = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # corrections
    for kk in range(0, k, BLOCK_K):
        km = (kk + offs_k) < k
        if TRANS_A == 0:
            a_mask = mask_m[:, None] & km[None, :]
            ah = tl.load(ah_ptrs, mask=a_mask, other=0.0)
            al = tl.load(al_ptrs, mask=a_mask, other=0.0)
        else:
            a_mask = km[:, None] & mask_m[None, :]
            ah = tl.trans(tl.load(ah_ptrs, mask=a_mask, other=0.0))
            al = tl.trans(tl.load(al_ptrs, mask=a_mask, other=0.0))
        if TRANS_B == 0:
            b_mask = km[:, None] & mask_n[None, :]
            bh = tl.load(bh_ptrs, mask=b_mask, other=0.0)
            bl = tl.load(bl_ptrs, mask=b_mask, other=0.0)
        else:
            b_mask = mask_n[:, None] & km[None, :]
            bh = tl.trans(tl.load(bh_ptrs, mask=b_mask, other=0.0))
            bl = tl.trans(tl.load(bl_ptrs, mask=b_mask, other=0.0))
        acc_hi = tl.dot(ah, bh, acc_hi)
        acc_lo += tl.dot(ah, bl) + tl.dot(al, bh) + tl.dot(al, bl)
        ah_ptrs += a_adv
        al_ptrs += a_adv
        bh_ptrs += b_adv
        bl_ptrs += b_adv

    s_ptrs = s_hi_ptr + offs_m[:, None] * lds + offs_n[None, :]
    s_lo_ptrs = s_lo_ptr + offs_m[:, None] * lds + offs_n[None, :]
    s_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(s_ptrs, acc_hi, mask=s_mask)
    tl.store(s_lo_ptrs, acc_lo, mask=s_mask)


def _split_fp64(x: torch.Tensor) -> tuple:
    """Return (x_hi, x_lo) fp32 tensors with x ~= x_hi + x_lo."""
    x_hi = x.to(torch.float32)
    x_lo = (x - x_hi.to(torch.float64)).to(torch.float32)
    return x_hi, x_lo


def dgemm(
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
    assert A.dtype == torch.float64
    assert B.dtype == torch.float64
    assert C.dtype == torch.float64
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

    block_m, block_n, block_k, num_warps, group_m = _select_gemm_config(
        transa, transb, m, n, k, fp64_split=True
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    Ah, Al = _split_fp64(A)
    Bh, Bl = _split_fp64(B)
    S_hi = torch.empty(m, n, dtype=torch.float32, device=A.device)
    S_lo = torch.empty(m, n, dtype=torch.float32, device=A.device)

    with torch_device_fn.device(A.device):
        _dgemm_split_kernel[grid](
            Ah, Al, Bh, Bl, S_hi, S_lo, m, n, k, lda, ldb, n,
            transa, transb,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m, num_warps=num_warps,
        )

    # epilogue in fp64 on host: C = alpha * (Sh + Sl) + beta * C
    S = S_hi.to(torch.float64) + S_lo.to(torch.float64)
    if beta == 0.0:
        C.copy_(alpha * S)
    else:
        C.copy_(alpha * S + beta * C)



# ---------------------------------------------------------------------------
# cgemm (complex64) -- real/imag planes with fp32 tl.dot.
#
# The complex product is assembled from four fp32 products
# (re += Ar@Br - Ai@Bi, im += Ar@Bi + Ai@Br).  Operands are deinterleaved into
# contiguous fp32 planes with one coalesced pass each, the GEMM keeps two
# accumulators (real/imag) so a 128x128 tile fits the Unified Buffer, and the
# result is re-interleaved into C with one contiguous copy.  Strided (stride-2)
# accesses into complex buffers are avoided end to end: on Ascend each of them
# is several times slower than a contiguous pass and dominated the mid-size
# shapes.
# ---------------------------------------------------------------------------
def _complex_scalar_parts(value: ScalarType):
    value = value.item() if isinstance(value, torch.Tensor) else value
    value = complex(value)
    return float(value.real), float(value.imag)


# Per-shape tile configs for the core benchmark shapes, tuned on Ascend 910B4
# (20 cube cores) with mspti kernel-level timing.  Format:
#     (m, n, k) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, GROUP_M)
# Every entry below was verified to compile and to be the fastest (or within
# noise of the fastest) config for its shape in the tune9/bench_v3 scans.
_PER_SHAPE_TUNE = {
    (64, 64, 64): (64, 64, 64, 4, 1),
    (256, 256, 256): (64, 64, 128, 8, 1),
    (512, 512, 512): (128, 128, 256, 8, 1),
    (1024, 1024, 1024): (256, 128, 128, 4, 4),
    (2048, 2048, 2048): (256, 128, 128, 4, 4),
    (4096, 4096, 4096): (256, 128, 128, 4, 8),
    (8192, 8192, 8192): (256, 128, 128, 4, 4),
    (16384, 16384, 16384): (256, 128, 128, 4, 4),
    (64, 512, 512): (64, 64, 64, 8, 1),
    (64, 1024, 512): (64, 64, 256, 4, 8),
    (64, 2048, 1024): (64, 128, 256, 4, 1),
    (64, 4096, 2048): (64, 256, 128, 4, 1),
    (128, 4096, 1024): (128, 256, 128, 8, 1),
    (256, 8192, 2048): (128, 128, 256, 8, 1),
    (512, 16384, 4096): (256, 128, 128, 8, 8),
    (512, 64, 512): (64, 64, 64, 4, 1),
    (1024, 64, 512): (64, 64, 256, 4, 1),
    (2048, 64, 1024): (128, 64, 256, 4, 1),
    (4096, 64, 2048): (256, 64, 128, 4, 1),
    (4096, 128, 1024): (256, 128, 128, 8, 1),
    (8192, 256, 2048): (128, 128, 256, 4, 1),
    (16384, 512, 4096): (256, 128, 128, 4, 8),
    (511, 511, 511): (128, 128, 256, 4, 8),
    (1023, 1023, 1023): (256, 128, 128, 8, 1),
    (2048, 12288, 4096): (256, 128, 128, 8, 8),
    (2048, 11008, 4096): (256, 128, 128, 4, 8),
    (2048, 4096, 11008): (256, 128, 128, 4, 4),
    (4096, 24576, 8192): (256, 128, 128, 8, 8),
    (4096, 8192, 28672): (128, 256, 128, 4, 4),
    (8192, 28672, 8192): (256, 128, 128, 4, 8),
    (16384, 2048, 2048): (256, 128, 128, 8, 8),
    (2048, 16384, 2048): (256, 128, 128, 4, 4),
    (2048, 2048, 16384): (256, 128, 128, 4, 4),
    (32768, 1024, 1024): (256, 128, 128, 8, 1),
    (4095, 4095, 4095): (128, 256, 128, 4, 1),
    (8191, 8191, 8191): (256, 128, 128, 8, 8),
    (4097, 8191, 4095): (128, 256, 128, 8, 1),
    # padded alignment of 4097x8191x4095 (kernel-level config, exp_gemmcfg)
    (4224, 8192, 4096): (128, 256, 128, 4, 8),
}


def _select_gemm_config(transa, transb, m, n, k, fp64_split=False):
    """Pick a static tile config for the given shape.

    Tuned on Ascend 910B4 (20 cube cores) with mspti kernel-level timing.
    Core benchmark shapes hit the per-shape table; everything else falls
    back to shape-class heuristics.
    """
    if fp64_split:
        # the split kernels (dgemm/zgemm) keep 2-4 fp32 accumulators
        # plus hi/lo operand tiles, so use small blocks to stay inside UB.
        if max(m, n, k) <= 128:
            block_m, block_n, block_k, num_warps, group_m = 16, 16, 32, 4, 1
        elif max(m, n, k) <= 512:
            block_m, block_n, block_k, num_warps, group_m = 32, 32, 32, 4, 1
        else:
            block_m, block_n, block_k, num_warps, group_m = 32, 32, 32, 4, 4
        while triton.cdiv(m, block_m) * triton.cdiv(n, block_n) * num_warps > 65535:
            if num_warps > 4:
                num_warps //= 2
                continue
            # 4 fp32 split accumulators: block_m*block_n*16B <= 128KB UB
            if block_m * block_n >= 8192:
                break
            if block_m <= block_n:
                block_m *= 2
            else:
                block_n *= 2
        return block_m, block_n, block_k, num_warps, group_m

    tune = _PER_SHAPE_TUNE.get((m, n, k))
    if tune is None:
        max_dim = max(m, n, k)
        # NOTE: every tile shape below has been verified to compile and to be
        # competitive on Ascend 910B4 with mspti kernel-level timing.  Tile
        # shapes that look reasonable but are not in this set (e.g. (64, 128,
        # 128) with a large K) make the BiShengHIR tiling pass blow the 128KB
        # Unified Buffer and fail to compile, so stick to the verified set.
        if max_dim <= 256:
            tune = (64, 64, 64, 4, 1)
        # tall-and-skinny: tiny M, wide N
        elif m <= 64 and n >= 256:
            if n <= 1024 and k <= 512:
                tune = (64, 64, 64, 4, 1)
            elif n >= 4096:
                tune = (64, 256, 128, 4, 1)
            else:
                tune = (64, 128, 128, 4, 1)
        # short-and-wide: wide M, tiny N
        elif n <= 64 and m >= 256:
            if m <= 1024:
                tune = (64, 64, 64, 4, 1)
            elif m <= 2048:
                tune = (128, 64, 128, 4, 1)
            else:
                tune = (256, 64, 128, 4, 1)
        elif max_dim <= 512:
            tune = (128, 128, 256, 4, 1)
        # moderately skinny/wide
        elif m <= 128:
            tune = (128, 256, 128, 4, 1)
        elif n <= 128:
            tune = (256, 128, 128, 8, 1)
        elif n <= 256:
            tune = (128, 128, 256, 4, 1)
        elif max_dim <= 2048:
            tune = (256, 128, 128, 8, 1)
        else:
            tune = (256, 128, 128, 8, 8)
    block_m, block_n, block_k, num_warps, group_m = tune
    # Keep the launch within the coreDim limit.  Prefer shrinking num_warps
    # over growing the tile: growing can push the fp32 accumulator
    # (block_m*block_n*4B) past the 128KB Unified Buffer and make the
    # BiShengHIR tiling pass fail (cc overflow).
    while triton.cdiv(m, block_m) * triton.cdiv(n, block_n) * num_warps > 65535:
        if num_warps > 4:
            num_warps //= 2
            continue
        # fp32 accumulator: block_m*block_n*4B <= 128KB -> <= 32768 elements
        if block_m * block_n >= 32768:
            break
        if block_m <= block_n:
            block_m *= 2
        else:
            block_n *= 2
    return block_m, block_n, block_k, num_warps, group_m


# Per-shape tile configs for the fp16 hgemm path, tuned on Ascend 910B4
# (20 cube cores) with mspti kernel-level timing.  Format:
#     (m, n, k) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, GROUP_M)
# fp16 halves the fp32 accumulator footprint, so the (128, 256, 256) tile
# that blows the 128KB Unified Buffer for fp32 fits fine here and wins on
# almost every shape.  Only the tall-and-skinny / short-and-wide extremes
# prefer small tiles with BLOCK_K=256.
_PER_SHAPE_TUNE_F16 = {
    (64, 64, 64): (64, 64, 64, 4, 1),
    (256, 256, 256): (64, 128, 256, 4, 1),
    (512, 512, 512): (128, 128, 256, 4, 1),
    (1024, 1024, 1024): (128, 256, 256, 4, 1),
    (2048, 2048, 2048): (128, 256, 256, 4, 1),
    (4096, 4096, 4096): (128, 256, 256, 4, 1),
    (8192, 8192, 8192): (128, 256, 256, 8, 8),
    (64, 512, 512): (64, 128, 256, 4, 1),
    (64, 1024, 512): (64, 128, 256, 4, 1),
    (64, 2048, 1024): (64, 128, 256, 4, 1),
    (64, 4096, 2048): (64, 256, 256, 4, 1),
    (128, 4096, 1024): (128, 256, 256, 4, 1),
    (256, 8192, 2048): (128, 256, 256, 4, 1),
    (512, 64, 512): (128, 64, 256, 4, 1),
    (1024, 64, 512): (128, 64, 256, 4, 1),
    (2048, 64, 1024): (128, 64, 256, 4, 1),
    (4096, 64, 2048): (256, 64, 256, 4, 1),
    (4096, 128, 1024): (256, 128, 256, 4, 1),
    (8192, 256, 2048): (128, 256, 256, 4, 8),
    (2048, 11008, 4096): (128, 256, 256, 4, 8),
    (2048, 4096, 11008): (128, 256, 256, 4, 1),
    (32768, 1024, 1024): (128, 256, 256, 4, 8),
    # odd shapes run unpadded (511^3, max-dim <= 512) or via _hgemm_padded
    # (1023^3 etc. get padded to the block multiples below, so the padded
    # gemm falls back to the aligned shapes already in this table).
    (511, 511, 511): (128, 128, 256, 4, 1),
}

# Per-(transa, transb) overrides for the fp16 table above.  On Ascend 910B4
# the in-kernel A^T transpose changes the memory access pattern enough that a
# transposed (larger M, smaller N) tile can win for TRANS_A=1 even when the
# shape-level best tile (shape-level table / heuristics) is the opposite.
# Measured with mspti on 2026-08-28: for (4096, 8192, 28672) the (256, 128)
# tile beats the shape-level (128, 256) choice by ~10% (t10) and ~17% (t11),
# while t00/t01 stay faster with (128, 256).  Keys: (m, n, k, transa, transb).
_PER_SHAPE_TUNE_F16_TRANS = {
    (4096, 8192, 28672, 1, 0): (256, 128, 256, 4, 8),
    (4096, 8192, 28672, 1, 1): (256, 128, 256, 4, 8),
    # 16384^3 is the counter-example: (256, 128, 128) wins for TRANS_A=1 while
    # (128, 256, 256) wins for TRANS_A=0 -- the k-loop is 128 iterations long
    # and the smaller BLOCK_K=128 tile keeps more of the accumulator resident.
    (16384, 16384, 16384, 1, 0): (256, 128, 128, 4, 4),
    (16384, 16384, 16384, 1, 1): (256, 128, 128, 4, 4),
}


def _select_gemm_config_f16(transa, transb, m, n, k):
    """Pick the tile config for the fp16 hgemm kernel.

    fp16 uses its own table/heuristics: the fp32-tuned tiles in
    ``_select_gemm_config`` under-use the fp16-friendly 128KB Unified Buffer,
    and masked odd shapes are much better handled by the padded path.
    """
    tune = _PER_SHAPE_TUNE_F16_TRANS.get((m, n, k, transa, transb))
    if tune is None:
        tune = _PER_SHAPE_TUNE_F16.get((m, n, k))
    if tune is None:
        max_dim = max(m, n, k)
        # NOTE: every tile shape below has been verified to compile and to be
        # competitive on Ascend 910B4 with mspti kernel-level timing.  Tiles
        # outside this set (e.g. (256, 256, 128) with BLOCK_K=256) exceed the
        # 128KB Unified Buffer for fp16 and fail to compile.
        if max_dim <= 256:
            tune = (64, 128, 256, 4, 1)
        # tall-and-skinny: tiny M, wide N
        elif m <= 64 and n >= 256:
            if n <= 2048:
                tune = (64, 128, 256, 4, 1)
            else:
                tune = (64, 256, 256, 4, 1)
        # short-and-wide: wide M, tiny N
        elif n <= 64 and m >= 256:
            if m <= 2048:
                tune = (128, 64, 256, 4, 1)
            else:
                tune = (256, 64, 256, 4, 1)
        elif max_dim <= 512:
            tune = (128, 128, 256, 4, 1)
        # wide M with large N: grouped scheduling helps L2 reuse
        elif n <= 256 and m >= 2048:
            tune = (128, 256, 256, 4, 8)
        elif m <= 128:
            tune = (128, 256, 256, 4, 1)
        elif n <= 128:
            tune = (256, 128, 256, 4, 1)
        elif max_dim <= 4096:
            tune = (128, 256, 256, 4, 1)
        else:
            tune = (128, 256, 256, 4, 8)
    block_m, block_n, block_k, num_warps, group_m = tune
    # Keep the launch within the coreDim limit.  Prefer shrinking num_warps
    # over growing the tile: growing can push the fp32 accumulator
    # (block_m*block_n*4B) past the 128KB Unified Buffer and make the
    # BiShengHIR tiling pass fail (cc overflow).
    while triton.cdiv(m, block_m) * triton.cdiv(n, block_n) * num_warps > 65535:
        if num_warps > 4:
            num_warps //= 2
            continue
        # fp32 accumulator: block_m*block_n*4B <= 128KB -> <= 32768 elements
        if block_m * block_n >= 32768:
            break
        if block_m <= block_n:
            block_m *= 2
        else:
            block_n *= 2
    return block_m, block_n, block_k, num_warps, group_m


# Deinterleave tile: 4096 complex elements (8192 fp32) per program.
CG_DEINT_BLOCK = 4096


@triton.jit
def _cgemm_deint_kernel(src, dst_r, dst_i, total, BLOCK: tl.constexpr):
    """Split interleaved complex data (re, im, re, im, ...) into fp32 planes.

    The load is contiguous (2*BLOCK fp32 elements) and ``tl.split`` removes the
    interleave in registers.  Loading with a stride-2 offset instead compiles
    to a ~50x slower kernel on this backend.
    """
    pid = tl.program_id(0)
    base = pid * 2 * BLOCK
    offs2 = tl.arange(0, 2 * BLOCK)
    pairs = tl.load(
        src + base + offs2, mask=(base + offs2) < 2 * total, other=0.0
    )
    re, im = tl.split(tl.reshape(pairs, (BLOCK, 2)))
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    tl.store(dst_r + offs, re, mask=mask)
    tl.store(dst_i + offs, im, mask=mask)


@triton.jit
def _cgemm_planes_kernel(
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    gr_ptr,
    gi_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Complex GEMM on separated fp32 real/imag planes.

    Two accumulators (real and imag) are updated by four fp32 ``tl.dot``
    calls: re += Ar@Br - Ai@Bi, im += Ar@Bi + Ai@Br.  Halving the number of
    accumulators (versus four separate products) is what makes the 128x128
    tile fit the Unified Buffer.
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

    if TRANS_A == 0:
        a_ptrs = ar_ptr + offs_m[:, None] * lda + offs_k[None, :]
        ai_ptrs = ai_ptr + offs_m[:, None] * lda + offs_k[None, :]
        a_adv = BLOCK_K
    else:
        a_ptrs = ar_ptr + offs_k[:, None] * lda + offs_m[None, :]
        ai_ptrs = ai_ptr + offs_k[:, None] * lda + offs_m[None, :]
        a_adv = BLOCK_K * lda
    if TRANS_B == 0:
        b_ptrs = br_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        bi_ptrs = bi_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        b_adv = BLOCK_K * ldb
    else:
        b_ptrs = br_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        bi_ptrs = bi_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        b_adv = BLOCK_K

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, k, BLOCK_K):
        km = (kk + offs_k) < k
        if TRANS_A == 0:
            amask = mask_m[:, None] & km[None, :]
            ar = tl.load(a_ptrs, mask=amask, other=0.0)
            ai = tl.load(ai_ptrs, mask=amask, other=0.0)
        else:
            amask = km[:, None] & mask_m[None, :]
            ar = tl.trans(tl.load(a_ptrs, mask=amask, other=0.0))
            ai = tl.trans(tl.load(ai_ptrs, mask=amask, other=0.0))
        if TRANS_B == 0:
            bmask = km[:, None] & mask_n[None, :]
            br = tl.load(b_ptrs, mask=bmask, other=0.0)
            bi = tl.load(bi_ptrs, mask=bmask, other=0.0)
        else:
            bmask = mask_n[:, None] & km[None, :]
            br = tl.trans(tl.load(b_ptrs, mask=bmask, other=0.0))
            bi = tl.trans(tl.load(bi_ptrs, mask=bmask, other=0.0))
        if TRANS_A == 2:
            ai = -ai
        if TRANS_B == 2:
            bi = -bi
        acc_r = tl.dot(ar, br, acc_r)
        acc_r = tl.dot(-ai, bi, acc_r)
        acc_i = tl.dot(ar, bi, acc_i)
        acc_i = tl.dot(ai, br, acc_i)
        a_ptrs += a_adv
        ai_ptrs += a_adv
        b_ptrs += b_adv
        bi_ptrs += b_adv

    cmask = mask_m[:, None] & mask_n[None, :]
    tl.store(gr_ptr + offs_m[:, None] * n + offs_n[None, :], acc_r, mask=cmask)
    tl.store(gi_ptr + offs_m[:, None] * n + offs_n[None, :], acc_i, mask=cmask)


def _select_cgemm_config(transa, transb, m, n, k):
    """Tile config for :func:`_cgemm_planes_kernel`.

    Tuned on Ascend 910B4 with mspti kernel timing: (128, 128, 32) is the
    largest tile whose two fp32 accumulators plus four operand tiles fit the
    192KB Unified Buffer, and it wins for every shape with max dimension > 256
    (all four transposition variants).  Small shapes prefer (64, 64, 64),
    where a 128x128 tile mostly wastes lanes.
    """
    if max(m, n, k) <= 256:
        return 64, 64, 64, 4, 1
    block_m, block_n, block_k, num_warps, group_m = 128, 128, 32, 4, 1
    # Keep the launch inside the coreDim limit for very large shapes.
    while triton.cdiv(m, block_m) * triton.cdiv(n, block_n) * num_warps > 65535:
        if num_warps > 1:
            num_warps //= 2
            continue
        break
    return block_m, block_n, block_k, num_warps, group_m


def _deinterleave_complex(x):
    """Return contiguous fp32 (real, imag) planes for a complex64 tensor."""
    rows, cols = x.shape
    out_r = torch.empty(rows, cols, dtype=torch.float32, device=x.device)
    out_i = torch.empty(rows, cols, dtype=torch.float32, device=x.device)
    total = rows * cols
    if total == 0:
        return out_r, out_i
    grid = (triton.cdiv(total, CG_DEINT_BLOCK),)
    with torch_device_fn.device(x.device):
        _cgemm_deint_kernel[grid](
            torch.view_as_real(x).reshape(-1),
            out_r,
            out_i,
            total,
            BLOCK=CG_DEINT_BLOCK,
            num_warps=4,
        )
    return out_r, out_i


def cgemm(
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
    assert A.dtype == torch.complex64
    assert B.dtype == torch.complex64
    assert C.dtype == torch.complex64
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]

    alpha_r, alpha_i = _complex_scalar_parts(alpha)
    beta_r, beta_i = _complex_scalar_parts(beta)

    if m == 0 or n == 0 or k == 0 or (alpha_r == 0.0 and alpha_i == 0.0):
        if beta_r == 0.0 and beta_i == 0.0:
            C.zero_()
        elif not (beta_r == 1.0 and beta_i == 0.0):
            C.mul_(complex(beta_r, beta_i))
        return

    beta_is_zero = beta_r == 0.0 and beta_i == 0.0
    alpha_is_one = alpha_r == 1.0 and alpha_i == 0.0
    block_m, block_n, block_k, num_warps, group_m = _select_cgemm_config(
        transa, transb, m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    Ar, Ai = _deinterleave_complex(A)
    Br, Bi = _deinterleave_complex(B)
    G_r = torch.empty(m, n, dtype=torch.float32, device=A.device)
    G_i = torch.empty(m, n, dtype=torch.float32, device=A.device)

    with torch_device_fn.device(A.device):
        _cgemm_planes_kernel[grid](
            Ar, Ai, Br, Bi, G_r, G_i, m, n, k, lda, ldb,
            transa, transb,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m, num_warps=num_warps,
        )

    # complex alpha/beta epilogue on host
    if beta_is_zero:
        if alpha_is_one:
            out_r, out_i = G_r, G_i
        else:
            out_r = alpha_r * G_r - alpha_i * G_i
            out_i = alpha_r * G_i + alpha_i * G_r
    else:
        C_parts = torch.view_as_real(C)  # fp32 (m,n,2)
        Cr = C_parts[..., 0]
        Ci = C_parts[..., 1]
        out_r = alpha_r * G_r - alpha_i * G_i + beta_r * Cr - beta_i * Ci
        out_i = alpha_r * G_i + alpha_i * G_r + beta_r * Ci + beta_i * Cr

    # Re-interleave through a fresh contiguous buffer: stride-2 stores through
    # a view of C are ~20x slower than one contiguous copy on Ascend.
    C.view(torch.float32).view(m, 2 * n).copy_(
        torch.stack([out_r, out_i], dim=-1).reshape(m, 2 * n)
    )


# ---------------------------------------------------------------------------
# zgemm (complex128)
#
# fp64 is unsupported in the Ascend Triton backend, so the complex128 operands
# are reduced to fp32 real/imag parts on the host and the complex product is
# assembled from four fp32 tl.dot GEMMs (same ~fp32-with-correction accuracy
# as dgemm).  The alpha/beta epilogue runs on the host in complex128 (fp64
# elementwise ops are supported by torch_npu).
# ---------------------------------------------------------------------------
@triton.jit
def _zgemm_split_kernel(
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    out_r_ptr,
    out_i_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    lds,
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
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    if TRANS_A == 0:
        ar_ptrs = ar_ptr + offs_m[:, None] * lda + offs_k[None, :]
        ai_ptrs = ai_ptr + offs_m[:, None] * lda + offs_k[None, :]
        a_adv = BLOCK_K
    else:
        ar_ptrs = ar_ptr + offs_k[:, None] * lda + offs_m[None, :]
        ai_ptrs = ai_ptr + offs_k[:, None] * lda + offs_m[None, :]
        a_adv = BLOCK_K * lda
    if TRANS_B == 0:
        br_ptrs = br_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        bi_ptrs = bi_ptr + offs_k[:, None] * ldb + offs_n[None, :]
        b_adv = BLOCK_K * ldb
    else:
        br_ptrs = br_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        bi_ptrs = bi_ptr + offs_n[:, None] * ldb + offs_k[None, :]
        b_adv = BLOCK_K

    acc_rr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # ar @ br
    acc_ii = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # ai @ bi
    acc_ri = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # ar @ bi
    acc_ir = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # ai @ br
    for kk in range(0, k, BLOCK_K):
        km = (kk + offs_k) < k
        if TRANS_A == 0:
            a_mask = mask_m[:, None] & km[None, :]
            ar = tl.load(ar_ptrs, mask=a_mask, other=0.0)
            ai = tl.load(ai_ptrs, mask=a_mask, other=0.0)
        else:
            a_mask = km[:, None] & mask_m[None, :]
            ar = tl.trans(tl.load(ar_ptrs, mask=a_mask, other=0.0))
            ai = tl.trans(tl.load(ai_ptrs, mask=a_mask, other=0.0))
        if TRANS_B == 0:
            b_mask = km[:, None] & mask_n[None, :]
            br = tl.load(br_ptrs, mask=b_mask, other=0.0)
            bi = tl.load(bi_ptrs, mask=b_mask, other=0.0)
        else:
            b_mask = mask_n[:, None] & km[None, :]
            br = tl.trans(tl.load(br_ptrs, mask=b_mask, other=0.0))
            bi = tl.trans(tl.load(bi_ptrs, mask=b_mask, other=0.0))
        if TRANS_A == 2:
            ai = -ai
        if TRANS_B == 2:
            bi = -bi
        acc_rr = tl.dot(ar, br, acc_rr)
        acc_ii = tl.dot(ai, bi, acc_ii)
        acc_ri = tl.dot(ar, bi, acc_ri)
        acc_ir = tl.dot(ai, br, acc_ir)
        ar_ptrs += a_adv
        ai_ptrs += a_adv
        br_ptrs += b_adv
        bi_ptrs += b_adv

    out_r = acc_rr - acc_ii
    out_i = acc_ri + acc_ir

    s_ptrs = out_r_ptr + offs_m[:, None] * lds + offs_n[None, :]
    s_i_ptrs = out_i_ptr + offs_m[:, None] * lds + offs_n[None, :]
    s_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(s_ptrs, out_r, mask=s_mask)
    tl.store(s_i_ptrs, out_i, mask=s_mask)


def zgemm(
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
    assert A.dtype == torch.complex128
    assert B.dtype == torch.complex128
    assert C.dtype == torch.complex128
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]

    alpha_r, alpha_i = _complex_scalar_parts(alpha)
    beta_r, beta_i = _complex_scalar_parts(beta)

    if m == 0 or n == 0 or k == 0 or (alpha_r == 0.0 and alpha_i == 0.0):
        if beta_r == 0.0 and beta_i == 0.0:
            C.zero_()
        elif not (beta_r == 1.0 and beta_i == 0.0):
            C.mul_(complex(beta_r, beta_i))
        return

    block_m, block_n, block_k, num_warps, group_m = _select_gemm_config(
        transa, transb, m, n, k, fp64_split=True
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    Ar = A.real.to(torch.float32)
    Ai = A.imag.to(torch.float32)
    Br = B.real.to(torch.float32)
    Bi = B.imag.to(torch.float32)
    G_r = torch.empty(m, n, dtype=torch.float32, device=A.device)
    G_i = torch.empty(m, n, dtype=torch.float32, device=A.device)

    with torch_device_fn.device(A.device):
        _zgemm_split_kernel[grid](
            Ar, Ai, Br, Bi, G_r, G_i, m, n, k, lda, ldb, n,
            transa, transb,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m, num_warps=num_warps,
        )

    # epilogue in complex128 on host: C = alpha * (Gr + i Gi) + beta * C
    Gr = G_r.to(torch.float64)
    Gi = G_i.to(torch.float64)
    C_parts = torch.view_as_real(C)
    Cr = C_parts[..., 0]
    Ci = C_parts[..., 1]
    if beta_r == 0.0 and beta_i == 0.0:
        new_r = alpha_r * Gr - alpha_i * Gi
        new_i = alpha_r * Gi + alpha_i * Gr
    else:
        new_r = alpha_r * Gr - alpha_i * Gi + beta_r * Cr - beta_i * Ci
        new_i = alpha_r * Gi + alpha_i * Gr + beta_r * Ci + beta_i * Cr
    C_parts[..., 0].copy_(new_r)
    C_parts[..., 1].copy_(new_i)

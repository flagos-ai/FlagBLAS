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

"""Ascend 910B hgemm (fp16 in, fp32 accumulate): kernel + padded path."""

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

from .gemm_common import (
    COPY_BM,
    COPY_BN,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _pad2d_kernel,
    _pad_workspace,
    _select_gemm_config_f16,
    _should_pad,
    _slice2d_kernel,
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


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

"""Ascend 910B dgemm (fp64 emulated with an fp32 hi/lo split)."""

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

from .gemm_common import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _select_gemm_config,
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


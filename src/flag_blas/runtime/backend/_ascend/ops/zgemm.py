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

"""Ascend 910B zgemm (complex128 reduced to four fp32 GEMMs)."""

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

from .gemm_common import (
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _complex_scalar_parts,
    _select_gemm_config,
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


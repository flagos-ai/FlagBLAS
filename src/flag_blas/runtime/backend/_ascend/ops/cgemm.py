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

"""Ascend 910B cgemm (complex64): fp32 real/imag plane decomposition."""

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
)

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


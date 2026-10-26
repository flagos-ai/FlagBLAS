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

"""MThreads (MUSA) sgemm implementation.

All transposition variants (``NN``/``TN``/``NT``/``TT``) use pure Triton
kernels via two fast paths:

* **tf32x3** (``_sgemm_nn_tf32x3_kernel``): for small products
  (``m*n*k <= 512**3``). ``tl.dot(..., input_precision="tf32x3")`` recovers
  close-to-fp32 accuracy with three tf32 passes. The kernel is mask-free and
  requires the host to pad ``m/n/k`` up to multiples of the tile size.
* **fp16 hi/lo split** (``_sgemm_nn_2acc_kernel``): for large products.
  ``A``/``B`` are split on the host into fp16 ``hi`` + ``lo`` parts (the low
  part scaled by 4096 to stay in fp16's narrow exponent range) with the
  elementwise ``_split_a_kernel``/``_split_b_kernel``. Three fast fp16 MMAs
  (``hi@hi`` + ``hi@lo`` + ``lo@hi``) approximate fp32 with an error around
  ``2^-22``, while running at fp16 throughput (100+ TFLOPS).

The MMAs always run in the canonical row-major ``(m, k) x (k, n)`` layout.
The host normalises transposed operands (``TN``: ``A`` stored as ``(k, m)``;
``NT``: ``B`` stored as ``(n, k)``) by folding the transpose into the
elementwise copy/split pass: the fp16 split kernels transpose through
tiled ``tl.trans`` passes (``_split_a_kernel_t``/``_split_b_kernel_t``),
the tf32x3 path copies through a transposed view. Both paths avoid masks by
padding on the host and re-using cached device buffers. The MThreads compiler
lowers fp32 ``tl.dot`` with IEEE precision to a ~0.9 TFLOPS path, which is why
no variant uses it.

The tile configuration is picked heuristically from the shape by
``_pick_nn_config``.
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.sgemm import CUBLAS_OP_N, CUBLAS_OP_T, ScalarType
from flag_blas.runtime import torch_device_fn

# fp16 hi/lo split scaling: low parts are stored as (a - hi) * S in fp16 so
# they do not underflow to subnormals for inputs of order 1.
_FP16_SPLIT_S = 4096.0
_FP16_SPLIT_INV_S = 1.0 / _FP16_SPLIT_S
_SPLIT_BLOCK = 1024
# Tile shape for the transposing split kernels (see ``_split_a_kernel_t``).
_SPLIT_BI = 32
_SPLIT_BJ = 64
# Products at or below this size use the tf32x3 single-kernel path.
_TF32_LIMIT = 512**3
# For transposed tf32x3 shapes at or below this product the mask-free kernel
# reads the operands transposed directly (strided loads stay cache-resident),
# skipping the padded copies that dominate tiny-shape latency.
_DIRECT_TRANS_LIMIT = 256**3


def _pad_to(v, mul):
    return ((v + mul - 1) // mul) * mul


def _pick_nn_config(m, n, k):
    """Heuristic tile selection validated against the core GEMM shapes."""
    if m * n * k <= _TF32_LIMIT:
        if min(m, n) <= 256:
            return "tf32x3", 32, 32, 64, 4, 2
        return "tf32x3", 16, 64, 32, 4, 2
    if n >= 2 * m or n <= 256 or m <= 128:
        return "fp16x3", 128, 256, 64, 16, 2
    return "fp16x3", 256, 128, 64, 16, 2


# ---------------------------------------------------------------------------
# Host-side split kernels: fp32 A/B -> fp16 hi/lo (low part scaled by S).
# The grid covers the *padded* size so padding lanes are written with 0 on
# every call, which keeps the cached buffers safe for re-use.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["rows", "cols", "rows_orig", "cols_orig", "lda"])
def _split_a_kernel(a_ptr, ahi_ptr, alo_ptr, rows, cols, rows_orig, cols_orig,
                    lda, lda_hi, S: tl.constexpr, BLOCK: tl.constexpr):
    # ``rows``/``cols`` are the *padded* A dims (pm, pk): the grid covers the
    # whole padded buffer so every lane is stored on each call (0 where
    # out-of-range), which keeps the cached buffers safe for re-use. Loads are
    # masked to the original (m, k) shape to avoid out-of-bounds reads of A.
    # (Transposed operands go through ``_split_a_kernel_t`` instead.)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = offs // cols
    j = offs % cols
    a = tl.load(a_ptr + i * lda + j, mask=(i < rows_orig) & (j < cols_orig), other=0.0)
    hi = a.to(tl.float16)
    lo = ((a - hi.to(tl.float32)) * S).to(tl.float16)
    tl.store(ahi_ptr + i * lda_hi + j, hi)
    tl.store(alo_ptr + i * lda_hi + j, lo)


@triton.jit(do_not_specialize=["rows", "cols", "rows_orig", "cols_orig", "ldb"])
def _split_b_kernel(b_ptr, bhi_ptr, blo_ptr, rows, cols, rows_orig, cols_orig,
                    ldb, ldb_hi, S: tl.constexpr, BLOCK: tl.constexpr):
    # Same padding-lane coverage as ``_split_a_kernel`` (see above).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = offs // cols
    j = offs % cols
    b = tl.load(b_ptr + i * ldb + j, mask=(i < rows_orig) & (j < cols_orig), other=0.0)
    hi = b.to(tl.float16)
    lo = ((b - hi.to(tl.float32)) * S).to(tl.float16)
    tl.store(bhi_ptr + i * ldb_hi + j, hi)
    tl.store(blo_ptr + i * ldb_hi + j, lo)


# ---------------------------------------------------------------------------
# Transposing split kernels: the source holds the operand transposed (A'^T =
# (k, m) with leading dim ``lda``; B'^T = (n, k) with leading dim ``ldb``),
# so element (i, j) of the padded destination sits at j * lda + i. A 1D
# ``offs // cols`` decomposition would read with stride ``lda`` (fully
# uncoalesced, ~4-6x slower on tall shapes), so each program instead loads a
# (BLOCK_J, BLOCK_I) source tile coalesced along ``i``, transposes it in
# registers, and stores the (BLOCK_I, BLOCK_J) hi/lo tiles coalesced along
# ``j``. The grid covers the whole padded buffer, so padding lanes are
# written with 0 on every call (keeps the cached buffers safe for re-use).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["rows", "cols", "rows_orig", "cols_orig", "lda"])
def _split_a_kernel_t(a_ptr, ahi_ptr, alo_ptr, rows, cols, rows_orig, cols_orig,
                      lda, lda_hi, S: tl.constexpr,
                      BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    mask = (i[None, :] < rows_orig) & (j[:, None] < cols_orig)
    a = tl.load(a_ptr + j[:, None] * lda + i[None, :], mask=mask, other=0.0)
    a = tl.trans(a)
    hi = a.to(tl.float16)
    lo = ((a - hi.to(tl.float32)) * S).to(tl.float16)
    dst = ahi_ptr + i[:, None] * lda_hi + j[None, :]
    tl.store(dst, hi)
    tl.store(alo_ptr + i[:, None] * lda_hi + j[None, :], lo)


@triton.jit(do_not_specialize=["rows", "cols", "rows_orig", "cols_orig", "ldb"])
def _split_b_kernel_t(b_ptr, bhi_ptr, blo_ptr, rows, cols, rows_orig, cols_orig,
                      ldb, ldb_hi, S: tl.constexpr,
                      BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    mask = (i[None, :] < rows_orig) & (j[:, None] < cols_orig)
    b = tl.load(b_ptr + j[:, None] * ldb + i[None, :], mask=mask, other=0.0)
    b = tl.trans(b)
    hi = b.to(tl.float16)
    lo = ((b - hi.to(tl.float32)) * S).to(tl.float16)
    dst = bhi_ptr + i[:, None] * ldb_hi + j[None, :]
    tl.store(dst, hi)
    tl.store(blo_ptr + i[:, None] * ldb_hi + j[None, :], lo)


# ---------------------------------------------------------------------------
# NN fast path 1: tf32x3, mask-free (m/n/k must be padded by the host).
# With ``TRANS_A``/``TRANS_B`` the corresponding operand is read transposed
# directly (A'[i, kk] = A[kk, i] at kk * lda + i; B'[kk, j] = B[j, kk] at
# j * ldb + kk). Strided loads are only used for tiny cache-resident shapes;
# larger transposed shapes go through the padded-buffer path instead.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["m", "n", "k"])
def _sgemm_nn_tf32x3_kernel(
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, k, BLOCK_K):
        rk = start_k + offs_k
        if TRANS_A:
            a = tl.load(a_ptr + (ram[:, None] + rk[None, :] * lda))
        else:
            a = tl.load(a_ptr + (ram[:, None] * lda + rk[None, :]))
        if TRANS_B:
            b = tl.load(b_ptr + (rk[:, None] + rbn[None, :] * ldb))
        else:
            b = tl.load(b_ptr + (rk[:, None] * ldb + rbn[None, :]))
        acc += tl.dot(a, b, out_dtype=tl.float32, input_precision="tf32x3")

    acc = alpha * acc

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    if BETA_IS_ZERO:
        # Keep a runtime-scalar dependency (`+ beta`, beta == 0 at runtime) so
        # the MThreads backend does not miscompile the plain `alpha * acc`
        # store for non-aligned shapes.
        tl.store(c_ptrs, acc + beta)
    else:
        c_vals = tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, acc + beta * c_vals)


# ---------------------------------------------------------------------------
# NN fast path 2: fp16 hi/lo split with three MMAs, mask-free.
# The low parts are already scaled by S on the host, so the correction term
# is (acc2 * INV_S) folded into a single expression (scaling acc2 on its own
# triggers a slow path in the MThreads compiler).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["m", "n", "k"])
def _sgemm_nn_2acc_kernel(
    ahi_ptr,
    alo_ptr,
    bhi_ptr,
    blo_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    INV_S: tl.constexpr,
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, k, BLOCK_K):
        rk = start_k + offs_k
        ahi = tl.load(ahi_ptr + (ram[:, None] * lda + rk[None, :]))
        alo = tl.load(alo_ptr + (ram[:, None] * lda + rk[None, :]))
        bhi = tl.load(bhi_ptr + (rk[:, None] * ldb + rbn[None, :]))
        blo = tl.load(blo_ptr + (rk[:, None] * ldb + rbn[None, :]))
        acc1 += tl.dot(ahi, bhi, out_dtype=tl.float32)
        acc2 += tl.dot(ahi, blo, out_dtype=tl.float32)
        acc2 += tl.dot(alo, bhi, out_dtype=tl.float32)

    acc = alpha * (acc1 + acc2 * INV_S)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    if BETA_IS_ZERO:
        tl.store(c_ptrs, acc + beta)
    else:
        c_vals = tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, acc + beta * c_vals)


# Cached device buffers for the fast paths, keyed by (device, kind). The
# buffers grow monotonically; padding lanes are re-written (zeroed) on every
# call by the split kernels / the mask-free MM kernels.
_sgemm_bufs = {}


def _get_buf(device, kind, pm, pn, pk):
    """Return cached input buffers big enough for the padded shape.

    The output buffer is NOT cached: the MThreads compiler substitutes the
    row stride of the destination tensor for the explicit ``ldc`` on tile
    stores, so ``c_out`` must have row stride exactly ``pn``. A per-call
    exact-sized allocation is cheap (caching allocator) and avoids that
    miscompile for shapes whose padded width differs from a reused buffer.
    """
    global _sgemm_bufs
    key = (device, kind)
    entry = _sgemm_bufs.get(key)
    if kind == "fp16x3":
        if entry is None or entry[1] < pm or entry[2] < pn or entry[3] < pk:
            bufs = (
                torch.zeros(pm, pk, dtype=torch.float16, device=device),
                torch.zeros(pm, pk, dtype=torch.float16, device=device),
                torch.zeros(pk, pn, dtype=torch.float16, device=device),
                torch.zeros(pk, pn, dtype=torch.float16, device=device),
            )
            _sgemm_bufs[key] = (bufs, pm, pn, pk)
        ahi, alo, bhi, blo = _sgemm_bufs[key][0]
        return ahi, alo, bhi, blo, torch.empty(pm, pn, dtype=torch.float32, device=device)
    if entry is None or entry[1] < pm or entry[2] < pn or entry[3] < pk:
        bufs = (
            torch.zeros(pm, pk, dtype=torch.float32, device=device),
            torch.zeros(pk, pn, dtype=torch.float32, device=device),
        )
        _sgemm_bufs[key] = (bufs, pm, pn, pk)
    a, b = _sgemm_bufs[key][0]
    return a, b, torch.empty(pm, pn, dtype=torch.float32, device=device)


def _fill_padded(dst, src, rows, cols, mul, trans):
    """Copy the logical (rows, cols) operand ``src`` into the padded buffer
    ``dst`` (both dims padded to multiples of ``mul``).

    With ``trans`` the source is stored transposed (cols, rows); the transpose
    is folded into the copy. When neither dimension needs padding the whole
    buffer is overwritten, so the zero-fill is skipped.
    """
    if not trans:
        dst.zero_()
        dst[:rows, :cols] = src
    elif rows % mul == 0 and cols % mul == 0:
        dst[:rows, :cols] = src.t()
    else:
        dst.zero_()
        dst[:rows, :cols] = src.t()


def _sgemm_fast(A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                trans_a=False, trans_b=False):
    """Fast path for every transposition variant.

    With ``trans_a``/``trans_b`` the corresponding operand holds the
    transposed matrix (``A``: logical ``(k, m)`` with leading dim ``lda``;
    ``B``: logical ``(n, k)`` with leading dim ``ldb``). The transpose is
    normalised on the host -- the tf32x3 path copies through a transposed
    view into the padded buffers, the fp16 split kernels fold it into their
    elementwise pass -- so the MMA kernels always see the canonical row-major
    ``(m, k) x (k, n)`` operands.
    """
    kind, bm, bn, bk, nw, ns = _pick_nn_config(m, n, k)
    device = A.device

    if kind == "tf32x3":
        # No padding + tiny product: run the mask-free kernel directly on the
        # operands (transposed reads are cache-resident at this size), which
        # avoids the padded copies that dominate tiny-shape latency.
        if (m % 64 == 0 and n % 64 == 0 and k % 64 == 0
                and (m * n * k <= _DIRECT_TRANS_LIMIT
                     or (not trans_a and not trans_b))):
            a, b, c_out = A, B, C
            lda_, ldb_, ldc_ = lda, ldb, ldc
            pm, pn, pk = m, n, k
            copyout = None
            trans_a_k, trans_b_k = trans_a, trans_b
        else:
            pm, pn, pk = _pad_to(m, 64), _pad_to(n, 64), _pad_to(k, 64)
            a, b, c_out = _get_buf(device, "tf32x3", pm, pn, pk)
            _fill_padded(a, A, m, k, 64, trans_a)
            _fill_padded(b, B, k, n, 64, trans_b)
            if not beta_is_zero:
                c_out[:m, :n] = C
            # The cached input buffers may be larger than the current padded
            # shape, so the MMA addressing must use their *actual* row strides,
            # not the current pm/pn/pk.
            lda_, ldb_, ldc_ = a.stride(0), b.stride(0), c_out.stride(0)
            copyout = C
            # The padded buffers already hold the canonical row-major layout
            # (the transpose was folded by _fill_padded), so the kernel must
            # not transpose them again.
            trans_a_k, trans_b_k = False, False

        grid = (triton.cdiv(pm, bm) * triton.cdiv(pn, bn),)
        _sgemm_nn_tf32x3_kernel[grid](
            a, b, c_out, alpha, beta, pm, pn, pk, lda_, ldb_, ldc_, beta_is_zero,
            trans_a_k, trans_b_k, bm, bn, bk, 8, num_warps=nw, num_stages=ns,
        )
        if copyout is not None:
            copyout[:m, :n].copy_(c_out[:m, :n])
        return

    # fp16 hi/lo split.
    pm, pn, pk = _pad_to(m, 256), _pad_to(n, 256), _pad_to(k, 256)
    ahi, alo, bhi, blo, c_out = _get_buf(device, "fp16x3", pm, pn, pk)
    if not beta_is_zero:
        c_out[:m, :n] = C
    grid_a = (_pad_to(pm * pk, _SPLIT_BLOCK) // _SPLIT_BLOCK,)
    grid_b = (_pad_to(pk * pn, _SPLIT_BLOCK) // _SPLIT_BLOCK,)
    if trans_a:
        _split_a_kernel_t[(triton.cdiv(pm, _SPLIT_BI), triton.cdiv(pk, _SPLIT_BJ))](
            A, ahi, alo, pm, pk, m, k, lda, ahi.stride(0), _FP16_SPLIT_S,
            _SPLIT_BI, _SPLIT_BJ,
        )
    else:
        _split_a_kernel[grid_a](
            A, ahi, alo, pm, pk, m, k, lda, ahi.stride(0), _FP16_SPLIT_S,
            BLOCK=_SPLIT_BLOCK,
        )
    if trans_b:
        _split_b_kernel_t[(triton.cdiv(pk, _SPLIT_BI), triton.cdiv(pn, _SPLIT_BJ))](
            B, bhi, blo, pk, pn, k, n, ldb, bhi.stride(0), _FP16_SPLIT_S,
            _SPLIT_BI, _SPLIT_BJ,
        )
    else:
        _split_b_kernel[grid_b](
            B, bhi, blo, pk, pn, k, n, ldb, bhi.stride(0), _FP16_SPLIT_S,
            BLOCK=_SPLIT_BLOCK,
        )

    grid = (triton.cdiv(pm, bm) * triton.cdiv(pn, bn),)
    _sgemm_nn_2acc_kernel[grid](
        ahi, alo, bhi, blo, c_out, alpha, beta, pm, pn, pk,
        ahi.stride(0), bhi.stride(0), c_out.stride(0),
        _FP16_SPLIT_INV_S, beta_is_zero, bm, bn, bk, 8, num_warps=nw, num_stages=ns,
    )
    C[:m, :n].copy_(c_out[:m, :n])


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

    with torch_device_fn.device(A.device):
        _sgemm_fast(
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            trans_a=(transa == CUBLAS_OP_T), trans_b=(transb == CUBLAS_OP_T),
        )

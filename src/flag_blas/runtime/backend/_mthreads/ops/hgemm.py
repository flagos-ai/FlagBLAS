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

"""MThreads (MUSA) hgemm implementation.

All transposition variants (``NN``/``TN``/``NT``/``TT``) share a single
mask-free fp16 MMA kernel (``_hgemm_nn_kernel``). The operands are already
fp16, so no hi/lo split is needed: ``tl.dot`` accumulates directly into fp32
and the result is rounded back to fp16, matching the cuBLAS ``gemmEx``
reference semantics (fp16 in, fp32 accumulate, fp16 out).

The transpose is normalised on the host, exactly like the mthreads ``sgemm``
fast path:

* unpadded shapes whose product is small (or that are not transposed) run the
  kernel directly on ``A``/``B``/``C`` with ``TRANS_A``/``TRANS_B`` selecting
  transposed reads -- no copies at all;
* everything else is copied into cached padded fp16 buffers through a
  transposed view (``_fill_padded``), so the kernel always sees the canonical
  row-major ``(m, k) x (k, n)`` operands.

The tile configuration is picked heuristically from the shape by
``_pick_config``.
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.hgemm import CUBLAS_OP_N, CUBLAS_OP_T, ScalarType
from flag_blas.runtime import torch_device_fn

# Products at or below this size use the small-tile config.
_SMALL_LIMIT = 512**3

# Tile/copy-kernel sizes for the padded-buffer copies (see ``_copy_block``).
# The row-copy kernel writes one BLOCK-wide chunk of a row per program, so it
# vectorizes the odd-width row boundaries that make torch's strided ``copy_``
# ~2.5x slower on non-aligned shapes (e.g. 4095x4095). Copies wider than
# ``_COPY_KERNEL_MAX`` fall back to torch, which is faster for large aligned
# blocks. Transposed copies always use torch's ``src.t()``: on MUSA it reaches
# ~2x the throughput of a hand-written Triton transpose tile.
_COPY_BLOCK = 1024
_COPY_KERNEL_MAX = 8192

# ``-mtgpu-opt-level=1`` enables the MUSA backend's ILP scheduling pass. On
# MTT S5000 it speeds up the big-tile (128x128 / 256x256) MMA kernels by
# 8-17% and the transposed-A split-K kernels by 8-15%, but regresses the
# small 64x64 tile (fewer instructions to schedule) and the non-transposed
# split-K path, so it is applied selectively per launch (the backend accepts
# ``llc_options`` as a per-launch option with its own cache key).
_LLC_OPT = "-mtgpu-opt-level=1"


def _pad_to(v, mul):
    return ((v + mul - 1) // mul) * mul


def _pick_config(m, n, k, trans_a=False, trans_b=False):
    """Heuristic tile selection validated against the core GEMM shapes.

    Returns ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M)``.

    The GPU needs a dense enough output grid to stay busy, so the tile size is
    chosen from the number of blocks it produces rather than from the shape
    side alone: a 256x256 tile only wins on big dense shapes (e.g. 4096^3,
    2048x12288x4096), while mid-size shapes like 1024^3 need a 64x64 tile
    (16x16 256-tiles underfill the GPU). Odd shapes (1023^3, 511^3) also want
    the small tile so the padded grid wastes little compute. ``GROUP_M=4`` is
    uniformly faster than 8 for the 256-tile on MTT S5000.

    Transposed variants change the cost profile: the 64x64 tile wins on most
    transposed shapes, but the big 256x256 tile is still needed when one side
    is very large (>= 8192) and the other is at least 256, or when both sides
    are >= 2048. When a 256x256 grid would produce fewer than ~128 output
    tiles it underfills the 60-SM MTT S5000; there the 128x128 tile (16 warps)
    is at least as fast as 256x256 and matches the non-transposed heuristic
    (verified neutral on 2048^3, 2048x2048x16384, 256x8192x2048 and
    8192x256x2048).
    """
    if m * n * k <= _SMALL_LIMIT:  # 512³: launch/occupancy bound, tiny tiles
        return 32, 32, 64, 4, 2, 2
    if trans_a or trans_b:
        if min(m, n) >= 256 and triton.cdiv(m, 256) * triton.cdiv(n, 256) >= 128:
            return 256, 256, 64, 16, 2, 4
        if min(m, n) >= 128 and triton.cdiv(m, 128) * triton.cdiv(n, 128) >= 128:
            return 128, 128, 64, 16, 2, 8
        return 64, 64, 64, 4, 2, 8
    if min(m, n) >= 256 and triton.cdiv(m, 256) * triton.cdiv(n, 256) >= 128:
        return 256, 256, 64, 16, 2, 4
    if min(m, n) >= 64 and triton.cdiv(m, 128) * triton.cdiv(n, 128) >= 128:
        # 16 warps 一致快于 4 warps(与转置分支参数对齐;实测 2048^3 +7%,
        # 512x16384x4096 +18%)。例外:行数 m 极小、网格恰为 128 的 skinny
        # 形状(GROUP_M 沿 m 分组的 L2 优化失效)仍用 4 warps(实测
        # 256x8192x2048 / 512x4096x4096 / 256x8192x4096 上 nw4 更快)。
        if m < 768 and m < n and triton.cdiv(m, 128) * triton.cdiv(n, 128) == 128:
            return 128, 128, 64, 4, 2, 8
        return 128, 128, 64, 16, 2, 8
    return 64, 64, 64, 4, 2, 8


# ---------------------------------------------------------------------------
# Fast padded-buffer copies. torch's ``copy_`` into a strided destination
# (padded row stride, odd width) drops to ~300 GB/s; these kernels keep the
# row stores coalesced so a 4095x4095 fill takes ~90us instead of ~200us.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["rows", "cols", "s0", "ss0"])
def _copy_rows_kernel(src, dst, rows, cols, s0, ss0, BLOCK: tl.constexpr):
    """Copy a row-major (rows, cols) block; each program writes one
    BLOCK-wide chunk of one row (so any ``cols`` is covered)."""
    pid = tl.program_id(0)
    row = pid // tl.cdiv(cols, BLOCK)
    j0 = (pid % tl.cdiv(cols, BLOCK)) * BLOCK
    offs = j0 + tl.arange(0, BLOCK)
    m = offs < cols
    v = tl.load(src + row * ss0 + offs, mask=m)
    tl.store(dst + row * s0 + offs, v, mask=m)


def _copy_block(dst, src, rows, cols, dst_s0, src_s0, trans=False):
    """Copy the logical (rows, cols) block from ``src`` into ``dst``.

    With ``trans`` the source is stored transposed (cols, rows); the transpose
    is folded into the copy (torch's ``src.t()`` is ~2x faster than a Triton
    transpose tile on MUSA). Non-contiguous row copies (odd widths or a padded
    destination row stride) go through the Triton row-copy kernel; fully
    contiguous blocks keep torch's ``copy_``, which is faster there.
    """
    if trans:
        dst[:rows, :cols] = src.t()
    elif dst_s0 != cols or src_s0 != cols:
        grid = (rows * triton.cdiv(cols, _COPY_BLOCK),)
        _copy_rows_kernel[grid](src, dst, rows, cols, dst_s0, src_s0, _COPY_BLOCK)
    else:
        # The source buffer may be padded larger than (rows, cols); slice it
        # so shapes whose rows are below the tile height (e.g. m < BLOCK_M)
        # copy correctly instead of broadcasting the whole padded buffer.
        dst[:rows, :cols] = src[:rows, :cols]


# ---------------------------------------------------------------------------
# Single fp16 MMA kernel, mask-free (m/n/k must be padded by the host).
# With ``TRANS_A``/``TRANS_B`` the corresponding operand is read transposed
# directly (A is stored ``(k, m)`` row-major, B ``(n, k)`` row-major): the
# tile is loaded coalesced in the transposed shape and ``tl.trans`` moves it
# into the MMA operand layout. fp16 inputs accumulate into fp32 and are
# rounded back to fp16 on store.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["m", "n", "k"])
def _hgemm_nn_kernel(
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

    if TRANS_A:
        # A is stored (K, M) row-major (lda = M): load the (BLOCK_K, BLOCK_M)
        # tile coalesced along the contiguous M dimension and transpose at the
        # register level; a strided per-element load would be ~2.7x slower.
        a_ptrs = a_ptr + (offs_k[:, None] * lda + ram[None, :])
    else:
        a_ptrs = a_ptr + (ram[:, None] * lda + offs_k[None, :])
    if TRANS_B:
        # B is stored (N, K) row-major (ldb = K): load (BLOCK_N, BLOCK_K)
        # coalesced along K, then transpose.
        b_ptrs = b_ptr + (rbn[:, None] * ldb + offs_k[None, :])
    else:
        b_ptrs = b_ptr + (offs_k[:, None] * ldb + rbn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if TRANS_A:
            a = tl.trans(a)
        if TRANS_B:
            b = tl.trans(b)
        # Three-operand accumulate form (a*b+acc in one MMA chain) is faster
        # on MTT S5000 than ``acc += tl.dot(a, b)``.
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * (lda if TRANS_A else 1)
        b_ptrs += BLOCK_K * (1 if TRANS_B else ldb)

    acc = alpha * acc

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    if BETA_IS_ZERO:
        # Keep a runtime-scalar dependency (`+ beta`, beta == 0 at runtime) so
        # the MThreads backend does not miscompile the plain store for
        # non-aligned shapes (same trick as the mthreads sgemm path).
        tl.store(c_ptrs, (acc + beta).to(tl.float16))
    else:
        c_vals = tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, (acc + beta * c_vals).to(tl.float16))


@triton.jit
def _hgemm_splitk_kernel(
    a_ptr,
    b_ptr,
    p_ptr,
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
    K_SPLITS: tl.constexpr,
):
    """Split-K GEMM over an (m, n) output tile writing an fp32 partial sum
    into the plane ``p_ptr + split_k_id * m * n``.

    Only used for k-deep non-transposed shapes whose m x n grid is too sparse
    to fill the machine (see ``_hgemm_fast``); the epilogue is a separate
    reduce kernel so the partial sums never go through the slow MUSA atomics.
    """
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

    k_per_split = k // K_SPLITS
    k_start = pid_k * k_per_split
    ks = k_start + offs_k
    if TRANS_A:
        a_ptrs = a_ptr + (ks[:, None] * lda + ram[None, :])
        a_step = BLOCK_K * lda
    else:
        a_ptrs = a_ptr + (ram[:, None] * lda + ks[None, :])
        a_step = BLOCK_K
    if TRANS_B:
        b_ptrs = b_ptr + (rbn[:, None] * ldb + ks[None, :])
        b_step = BLOCK_K
    else:
        b_ptrs = b_ptr + (ks[:, None] * ldb + rbn[None, :])
        b_step = BLOCK_K * ldb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(k_start, k_start + k_per_split, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if TRANS_A:
            a = tl.trans(a)
        if TRANS_B:
            b = tl.trans(b)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += a_step
        b_ptrs += b_step

    p_ptrs = p_ptr + pid_k * (m * n) + offs_am[:, None] * n + offs_bn[None, :]
    tl.store(p_ptrs, acc)


@triton.jit
def _hgemm_splitk_reduce_kernel(
    p_ptr,
    c_ptr,
    m,
    n,
    alpha: tl.float32,
    beta: tl.float32,
    BETA_IS_ZERO: tl.constexpr,
    K_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum the K_SPLITS fp32 partial planes into C = alpha * sum + beta * C."""
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = offs_m[:, None] * n + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ks in tl.static_range(K_SPLITS):
        acc += tl.load(p_ptr + ks * (m * n) + offs).to(tl.float32)
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        c = tl.load(c_ptr + offs).to(tl.float32)
        out = alpha * acc + beta * c
    tl.store(c_ptr + offs, out.to(tl.float16))


# Cached padded fp16 input buffers, keyed by device. The buffers grow
# monotonically; padding lanes are re-written (zeroed) on every call by
# ``_fill_padded``.
_hgemm_bufs = {}


def _get_buf(device, pm, pn, pk):
    """Return cached padded fp16 input buffers big enough for the shape.

    The output tensor is never cached: the MThreads compiler substitutes the
    row stride of the destination tensor for the explicit ``ldc`` on tile
    stores, so a padded output must be a freshly allocated buffer whose row
    stride is exactly ``pn`` (see ``_hgemm_fast``).
    """
    global _hgemm_bufs
    entry = _hgemm_bufs.get(device)
    if entry is None or entry[1] < pm or entry[2] < pn or entry[3] < pk:
        bufs = (
            torch.zeros(pm, pk, dtype=torch.float16, device=device),
            torch.zeros(pk, pn, dtype=torch.float16, device=device),
        )
        _hgemm_bufs[device] = (bufs, pm, pn, pk)
    a, b = _hgemm_bufs[device][0]
    return a, b


def _fill_padded(dst, src, rows, cols, mul_r, mul_c, trans):
    """Copy the logical (rows, cols) operand ``src`` into the padded buffer
    ``dst`` (rows padded to a multiple of ``mul_r``, cols to ``mul_c``).

    With ``trans`` the source is stored transposed (cols, rows); the transpose
    is folded into the copy. Only the actual padding lanes are zeroed (the
    kernel multiplies them, so they must read 0); zeroing the whole cached
    buffer would be expensive when the buffer is bigger than the current pad.
    """
    if rows % mul_r or cols % mul_c:
        pm = _pad_to(rows, mul_r)
        pk = _pad_to(cols, mul_c)
        if pm > rows:
            dst[rows:pm, :pk].zero_()
        if pk > cols:
            dst[:rows, cols:pk].zero_()
    _copy_block(dst, src, rows, cols, dst.stride(0), src.stride(0), trans)


def _should_split_k(m, n, k):
    """Whether the k-deep shape benefits from the 256x256 split-K path.

    Split-K with a 256x256 tile wins when the 256^2 output grid underfills the
    60-SM MTT S5000 (``grid256 < 160``, fewer than ~2.7 waves) while still
    producing enough tiles after the split to fill the machine
    (``2 * grid256 >= 30``), and k is long enough that the extra fp32
    partial-sum pass amortises. Measured 5-20% faster on 2048^2x16384 and
    2048x4096x11008 across all four transposition variants; neutral on dense
    grids (4096x8192x28672) and negative on short-k shapes (the reduce pass
    dominates).
    """
    if k % 2 or k < 2 * max(m, n):
        return False
    grid256 = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    return 2 * grid256 >= 30 and grid256 < 160


def _hgemm_fast(A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                trans_a=False, trans_b=False):
    """Fast path for every transposition variant.

    With ``trans_a``/``trans_b`` the corresponding operand holds the
    transposed matrix (``A``: logical ``(k, m)`` with leading dim ``lda``;
    ``B``: logical ``(n, k)`` with leading dim ``ldb``). Tile-divisible
    shapes run the kernel directly on the operands (transposed operands are
    read through the kernel's coalesced ``tl.trans`` path); only shapes that
    need padding are copied into the padded buffers first.
    """
    bm, bn, bk, nw, ns, group_m = _pick_config(m, n, k, trans_a, trans_b)
    device = A.device

    # k-deep shapes whose 256^2 output grid underfills the machine: the 256x256
    # tile plus two k-splits beats the direct path by 5-20% (see
    # ``_should_split_k``). The kernel is mask-free, so m/n/k must be
    # tile-divisible; ragged shapes fall through to the padded path below.
    if _should_split_k(m, n, k) and m % 256 == 0 and n % 256 == 0 and k % 64 == 0:
        p = torch.empty(2, m, n, dtype=torch.float32, device=device)
        grid = (triton.cdiv(m, 256) * triton.cdiv(n, 256), 2)
        # GROUP_M=4 plus the backend ILP scheduler is 3-4% faster than the
        # previous GROUP_M=8 across all transposition variants on k-deep
        # shapes (interleaved A/B: 2048^2x16384 144.9 vs 140.5 TF,
        # 2048x4096x11008 169.2 vs 162.6 TF; GROUP_M=8 with the scheduler is
        # 3-4% slower than without, so both knobs matter together).
        _hgemm_splitk_kernel[grid](
            A, B, p, m, n, k, lda, ldb, trans_a, trans_b, 256, 256, 64, 4, 2,
            num_warps=16, num_stages=2, llc_options=_LLC_OPT,
        )
        _hgemm_splitk_reduce_kernel[(grid[0],)](
            p, C, m, n, alpha, beta, beta_is_zero, 2, 256, 256, num_warps=16,
        )
        return

    # Tile-divisible shapes run the mask-free kernel directly on the operands
    # (and write straight into C), avoiding the padded copies that dominate
    # small-shape latency. The tile must divide the shape so the mask-free
    # kernel never reads out of bounds.
    if m % bm == 0 and n % bn == 0 and k % bk == 0:
        a, b, c_out = A, B, C
        lda_, ldb_, ldc_ = lda, ldb, ldc
        pm, pn, pk = m, n, k
        copyout = None
        trans_a_k, trans_b_k = trans_a, trans_b
    else:
        # Normalise each operand independently: only operands that actually
        # need it (transposed, or a non-tile-divisible side) are copied into
        # the padded buffers; the rest are read in place by the kernel. The k
        # dimension is shared by both operands, so ``k % bk != 0`` forces both
        # into the buffers. A tile-divisible output goes straight into ``C``
        # (whose row stride equals the explicit ``ldc``), skipping the padded
        # output buffer and the copy-back.
        kpad = k % bk != 0
        a_pad = trans_a or kpad or m % bm != 0
        b_pad = trans_b or kpad or n % bn != 0
        c_pad = m % bm != 0 or n % bn != 0
        pm, pn, pk = _pad_to(m, bm), _pad_to(n, bn), _pad_to(k, bk)
        a, b = _get_buf(device, pm, pn, pk)
        lda_, ldb_ = lda, ldb
        if a_pad:
            _fill_padded(a, A, m, k, bm, bk, trans_a)
            # The cached input buffers may be larger than the current padded
            # shape, so the MMA addressing must use their *actual* row stride.
            lda_ = a.stride(0)
        else:
            a = A
        if b_pad:
            _fill_padded(b, B, k, n, bk, bn, trans_b)
            ldb_ = b.stride(0)
        else:
            b = B
        if c_pad:
            c_out = torch.empty(pm, pn, dtype=torch.float16, device=device)
            if not beta_is_zero:
                c_out[:m, :n] = C
            ldc_ = c_out.stride(0)
            copyout = C
        else:
            c_out = C
            ldc_ = ldc
            copyout = None
        pm, pn = (m, n) if not c_pad else (pm, pn)
        # Every operand the kernel reads is now canonical row-major (the
        # transpose was folded by _fill_padded), so the kernel never
        # transposes again.
        trans_a_k, trans_b_k = False, False

    grid = (triton.cdiv(pm, bm) * triton.cdiv(pn, bn),)
    kw = {}
    # The 128x128 / 256x256 MMA kernels are 8-17% faster with the backend ILP
    # scheduler; the small 64x64 tile regresses (too few instructions for the
    # scheduler to help), so it keeps the default codegen.
    if bm >= 128 and bn >= 128:
        kw["llc_options"] = _LLC_OPT
    _hgemm_nn_kernel[grid](
        a, b, c_out, alpha, beta, pm, pn, pk, lda_, ldb_, ldc_, beta_is_zero,
        trans_a_k, trans_b_k, bm, bn, bk, group_m, num_warps=nw, num_stages=ns,
        **kw,
    )
    if copyout is not None:
        _copy_block(copyout, c_out, m, n, copyout.stride(0), c_out.stride(0))


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

    with torch_device_fn.device(A.device):
        _hgemm_fast(
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            trans_a=(transa == CUBLAS_OP_T), trans_b=(transb == CUBLAS_OP_T),
        )

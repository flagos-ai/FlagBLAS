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
# The row-copy kernel runs one program per row and copies BLOCK-wide unmasked
# chunks (vectorized; only the odd-width row tail is masked), avoiding the
# scalar path that makes torch's strided ``copy_`` ~2.5x slower on non-aligned
# shapes (e.g. 4095x4095). Copies wider than
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
#
# The full option set (ILP scheduler + burst/fop combine + max-ILP
# instruction scheduler) adds another 3-6% on top of ``-mtgpu-opt-level=1``
# alone for the dense 128x128/256x256 tiles; very large shapes regress ~1%,
# so ``_hgemm_fast`` falls back to the single option there.
_LLC_OPT = "-mtgpu-opt-level=1"
_LLC_OPT_FULL = "-mtgpu-opt-level=1 -misched=mtgpu-max-ilp -mtgpu-combine-instr-with-burst=1 -mtgpu-combine-fop-instr=1"
# Above this many FLOP the full option set regresses vs ``_LLC_OPT``.
_LLC_FULL_FLOP_LIMIT = 1 << 40  # ~1.1e12


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
            grid256 = triton.cdiv(m, 256) * triton.cdiv(n, 256)
            # 转置瘦高（512x16384x4096 / 16384x512x4096）用 128x128 ns1 快
            # 4-15%（与 bfgemm 相同实测）；grid256==256 的 1024x16384x2048
            # 仍偏好 256x256，条件比非转置分支更严。
            if grid256 <= 128 and min(m, n) <= 1024:
                return 128, 128, 64, 16, 1, 4
            return 256, 256, 64, 16, 2, 4
        if m == n and m >= 2048:
            # 方形中尺寸转置（2048^3）：num_stages=1 + 16 warps 比 nw4 ns2 快
            # 12-25%（与 bfgemm 相同实测）。nw4 ns1 会触发标量退化。
            return 128, 128, 64, 16, 1, 4
        if min(m, n) >= 128 and triton.cdiv(m, 128) * triton.cdiv(n, 128) >= 128:
            # 4 warps 稳定快于 16 warps(与 bfgemm 相同实测)。
            return 128, 128, 64, 4, 2, 8
        # 极端 skinny 的 64x64 网格（64x4096x2048 / 4096x64x2048，
        # grid64==64）8 warps 快 2-6%（launch/occupancy bound）。
        if (m == 64 or n == 64) and triton.cdiv(m, 64) * triton.cdiv(n, 64) == 64:
            return 64, 64, (128 if k % 128 == 0 else 64), 8, 2, 8
        # 一侧恰为 128 的 skinny 转置形状（128x4096x1024 / 4096x128x1024）
        # 用 BLOCK_K=128 快 5-20%：k 迭代减半（1024: 16->8 次），小 tile 的
        # 循环开销占比大。稠密 64x64 网格（1024^3）上 bk=128 因共享内存翻倍
        # 掉占用率反而 +47%，所以仅限此 m/n==128 且 k 足够长的窄分支。
        if k % 128 == 0 and (m == 128 or n == 128) and max(m, n) >= 1024:
            return 64, 64, 128, 4, 2, 8
        return 64, 64, 64, 4, 2, 8
    if min(m, n) >= 256 and triton.cdiv(m, 256) * triton.cdiv(n, 256) >= 128:
        grid256 = triton.cdiv(m, 256) * triton.cdiv(n, 256)
        # num_stages=1 + BLOCK_K=128 (mask-free 需要 k%128==0) 在窄/中型
        # 128x128 网格上比 256x256 快 4-14%（实测 512x16384x4096 447->410us、
        # 1024x16384x2048 400->373us、512x4096x4096 155->120us）：256-tile 会
        # 使 B 侧流量翻倍或网格过稀。稠密方形（4096^3/8192^3）与两侧都大的
        # 形状（2048x12288x4096）仍偏好 256x256。
        if grid256 <= 256 and min(m, n) <= 1024:
            return 128, 128, (128 if k % 128 == 0 else 64), 16, 1, 4
        return 256, 256, 64, 16, 2, 4
    if min(m, n) >= 64 and triton.cdiv(m, 128) * triton.cdiv(n, 128) >= 128:
        # num_stages=1 比 ns2 快 13-27%（实测 2048^3 150->117us、
        # 8192x256x2048 89->71us、512x4096x4096 163->120us）：128x128 tile
        # 减半 smem 双缓冲、提高占用率，网格足够密可无需流水线掩盖加载延迟。
        # k%128==0 时 BLOCK_K=128 再省 8-10% 迭代。旧 grid128==128 skinny 的
        # nw4 特判已被超越（nw16 ns1 在该处同样更快）。
        return 128, 128, (128 if k % 128 == 0 else 64), 16, 1, 4
    # 极端 skinny 的 64x64 网格（64x4096x2048 / 4096x64x2048，
    # grid64==64）8 warps 快 2-6%（launch/occupancy bound）。
    if (m == 64 or n == 64) and triton.cdiv(m, 64) * triton.cdiv(n, 64) == 64:
        return 64, 64, (128 if k % 128 == 0 else 64), 8, 2, 8
    return 64, 64, 64, 4, 2, 8


# ---------------------------------------------------------------------------
# Fast padded-buffer copies. torch's ``copy_`` into a strided destination
# (padded row stride, odd width) drops to ~300 GB/s; these kernels keep the
# row stores coalesced so a 4095x4095 fill takes ~90us instead of ~200us.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["rows", "cols", "s0", "ss0"])
def _copy_rows_kernel(src, dst, rows, cols, s0, ss0,
                      BLOCK: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    """Copy a row-major (rows, cols) block; one program per ``ROWS_PER_PROG``
    rows.

    Full ``BLOCK``-wide chunks are copied without a mask (keeps the loads
    vectorized on the MUSA backend, which collapses to a scalar path when any
    ``mask=`` is present); only the odd-width row tail is masked. Copying
    several rows per program raises the achieved bandwidth on wide matrices
    (8191^2: 342us at 1 row/prog vs 296us at 4, ~900 GB/s)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n_full = cols // BLOCK
    rem = cols % BLOCK
    for r in tl.static_range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + r
        if row < rows:
            for i in range(n_full):
                j = i * BLOCK
                v = tl.load(src + row * ss0 + j + offs)
                tl.store(dst + row * s0 + j + offs, v)
            if rem:
                j = n_full * BLOCK
                m = offs < rem
                v = tl.load(src + row * ss0 + j + offs, mask=m)
                tl.store(dst + row * s0 + j + offs, v, mask=m)


@triton.jit(do_not_specialize=[
    "a_rows", "a_cols", "b_rows", "b_cols", "a_s0", "a_ss0", "b_s0", "b_ss0",
])
def _copy_rows_pair_kernel(
    a_src, a_dst, b_src, b_dst,
    a_rows, a_cols, b_rows, b_cols,
    a_s0, a_ss0, b_s0, b_ss0,
    A_ROWS: tl.constexpr, A_BLOCK: tl.constexpr, B_BLOCK: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """Copy two row-major blocks in a single launch: programs in
    ``[0, cdiv(A_ROWS, ROWS_PER_PROG))`` cover the first block (ROWS_PER_PROG
    rows per program), the rest the second. Used to fill both padded GEMM
    operands at once (each launch costs ~5us of the small padded shapes'
    latency). Full chunks are unmasked to keep the copies vectorized; only the
    odd-width row tail is masked. Row guards handle non-divisible row counts."""
    pid = tl.program_id(0)
    a_progs = tl.cdiv(A_ROWS, ROWS_PER_PROG)
    offs_a = tl.arange(0, A_BLOCK)
    offs_b = tl.arange(0, B_BLOCK)
    n_full_a = a_cols // A_BLOCK
    rem_a = a_cols % A_BLOCK
    n_full_b = b_cols // B_BLOCK
    rem_b = b_cols % B_BLOCK
    if pid < a_progs:
        for r in tl.static_range(ROWS_PER_PROG):
            row = pid * ROWS_PER_PROG + r
            if row < a_rows:
                for i in range(n_full_a):
                    j = i * A_BLOCK
                    v = tl.load(a_src + row * a_ss0 + j + offs_a)
                    tl.store(a_dst + row * a_s0 + j + offs_a, v)
                if rem_a:
                    j = n_full_a * A_BLOCK
                    m = offs_a < rem_a
                    v = tl.load(a_src + row * a_ss0 + j + offs_a, mask=m)
                    tl.store(a_dst + row * a_s0 + j + offs_a, v, mask=m)
    else:
        bpid = pid - a_progs
        for r in tl.static_range(ROWS_PER_PROG):
            row = bpid * ROWS_PER_PROG + r
            if row < b_rows:
                for i in range(n_full_b):
                    j = i * B_BLOCK
                    v = tl.load(b_src + row * b_ss0 + j + offs_b)
                    tl.store(b_dst + row * b_s0 + j + offs_b, v)
                if rem_b:
                    j = n_full_b * B_BLOCK
                    m = offs_b < rem_b
                    v = tl.load(b_src + row * b_ss0 + j + offs_b, mask=m)
                    tl.store(b_dst + row * b_s0 + j + offs_b, v, mask=m)


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
        # Narrow matrices (<= ~1K cols) copy fastest with a 512-wide chunk per
        # program; wide ones prefer 2048 (measured 511^2: 512b 5.3us vs 1024b
        # 5.7us, 8191^2: 2048b 309us vs 1024b 325us).
        block = 512 if cols <= 1024 else 2048
        # Copy-out into ``C`` (aligned destination rows) benefits from row
        # batching on wide matrices (8191^2: 360->293us at rpp=4; 4095^2:
        # ~neutral; 1023^2 prefers rpp=2: 9.0us). Narrow rows (511^2) regress
        # at rpp>=2 in the full op, so they keep 1 row per program. Fills into
        # padded buffers (unaligned destination rows) keep 1 row per program
        # (8191^2 regresses to 263us at rpp>=2 vs 252us at rpp=1).
        rpp = 1 if cols < 1024 else (8 if cols > 4096 else (4 if cols >= 4096 else 2))
        if dst_s0 != cols:
            rpp = 1
        _copy_rows_kernel[(triton.cdiv(rows, rpp),)](
            src, dst, rows, cols, dst_s0, src_s0, block, rpp)
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
# Cached padded output buffer (see ``_get_cbuf``).
_hgemm_cbufs = {}
# Cached padded buffers in the transposed operands' natural layout (see
# ``_get_buf_t``).
_hgemm_bufs_t = {}


def _get_buf(device, pm, pn, pk):
    """Return cached padded fp16 input buffers big enough for the shape."""
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


def _get_buf_t(device, pm, pn, pk):
    """Return cached padded fp16 buffers in the transposed operands' natural
    layout: A ``(pk, pm)`` (kernel reads it with ``TRANS_A``, lda=pm) and B
    ``(pn, pk)`` (``TRANS_B``, ldb=pk).

    Keeping transposed operands in their natural layout avoids the torch
    ``src.t()`` transpose during the fill, which costs ~9us more than the row
    copy on the small padded shapes (511^3 TT fill 27->18us, 1023^3 TT
    34->26us); the kernel's ``tl.trans``-based transposed read is neutral
    there. The K-padding lanes (A's rows ``k:pk``, B's cols ``k:pk``) are
    zeroed by the caller.
    """
    global _hgemm_bufs_t
    entry = _hgemm_bufs_t.get(device)
    if entry is None or entry[1] < pm or entry[2] < pn or entry[3] < pk:
        bufs = (
            torch.zeros(pk, pm, dtype=torch.float16, device=device),
            torch.zeros(pn, pk, dtype=torch.float16, device=device),
        )
        _hgemm_bufs_t[device] = (bufs, pm, pn, pk)
    a_t, b_t = _hgemm_bufs_t[device][0]
    return a_t, b_t


def _get_cbuf(device, pm, pn):
    """Return a cached padded fp16 output buffer of exact shape (pm, pn).

    The MThreads compiler substitutes the row stride of the destination tensor
    for the explicit ``ldc`` on tile stores, so the padded output stride must
    be exactly ``pn``; caching a buffer of exactly that stride keeps that
    property while avoiding a per-call ``torch.empty``.
    """
    global _hgemm_cbufs
    entry = _hgemm_cbufs.get(device)
    if entry is None or entry[1] != pm or entry[2] != pn:
        buf = torch.empty(pm, pn, dtype=torch.float16, device=device)
        _hgemm_cbufs[device] = (buf, pm, pn)
    return _hgemm_cbufs[device][0]


def _fill_padded(dst, src, rows, cols, mul_r, mul_c, trans,
                 zero_rows=True, zero_cols=True):
    """Copy the logical (rows, cols) operand ``src`` into the padded buffer
    ``dst`` (rows padded to a multiple of ``mul_r``, cols to ``mul_c``).

    With ``trans`` the source is stored transposed (cols, rows); the transpose
    is folded into the copy. Only the actual padding lanes that the MMA reads
    are zeroed (the kernel multiplies them, so they must read 0); zeroing the
    whole cached buffer would be expensive when the buffer is bigger than the
    current pad. For A only the padded *columns* (the K lanes) matter and for
    B only the padded *rows* (the K lanes) -- the other padding lanes only
    contribute to C lanes that the copy-out discards, so the caller skips
    zeroing them (``zero_rows``/``zero_cols``).
    """
    if rows % mul_r or cols % mul_c:
        pm = _pad_to(rows, mul_r)
        pk = _pad_to(cols, mul_c)
        if zero_rows and pm > rows:
            dst[rows:pm, :pk].zero_()
        if zero_cols and pk > cols:
            dst[:rows, cols:pk].zero_()
    _copy_block(dst, src, rows, cols, dst.stride(0), src.stride(0), trans)


def _fill_padded_pair(a_dst, A, b_dst, B, m, n, k, bm, bn, bk):
    """Copy both padded operands with a single Triton launch.

    Only usable when both copies go through the row-copy kernel (untransposed,
    non-contiguous operands); transposed or contiguous copies use torch and
    cannot be merged. Zeroing matches the ``_fill_padded`` A/B split: only the
    padded K lanes (A's extra columns, B's extra rows) are zeroed -- the other
    padding lanes only feed C lanes the copy-out discards.
    """
    pm, pk = _pad_to(m, bm), _pad_to(k, bk)
    pn = _pad_to(n, bn)
    if pk > k:
        a_dst[:m, k:pk].zero_()
        b_dst[k:pk, :pn].zero_()
    a_block = 512 if k <= 1024 else 2048
    b_block = 512 if n <= 1024 else 2048
    # Padded targets (unaligned rows) copy fastest with one row per program
    # (see ``_copy_block``), so the pair kernel always uses ROWS_PER_PROG=1.
    _copy_rows_pair_kernel[(m + k,)](
        A, a_dst, B, b_dst, m, k, k, n,
        a_dst.stride(0), A.stride(0), b_dst.stride(0), B.stride(0),
        m, a_block, b_block, 1,
    )


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
        # 方形 k 深（m==n 且 k%128==0）的 NN/TN/TT 用 128x128 tile +
        # BLOCK_K=128 + num_stages=1 比 256x256 BK=64 ns=2 快 3-5%（实测
        # 2048^2x16384 NN 902->874 / TN 899->857us），TT 中性；NT（仅 B
        # 转置）回退 ~2% 故保留大 tile。非方形（2048x4096x11008）用 128x128
        # 全面回退 +6~13%，同样保留 256x256。
        square128 = (m == n and k % 128 == 0 and not (trans_b and not trans_a))
        bm = bn = 128 if square128 else 256
        bk = 128 if square128 else 64
        ns = 1 if square128 else 2
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), 2)
        # GROUP_M=4 plus the backend ILP scheduler is 3-4% faster than the
        # previous GROUP_M=8 across all transposition variants on k-deep
        # shapes (interleaved A/B: 2048^2x16384 144.9 vs 140.5 TF,
        # 2048x4096x11008 169.2 vs 162.6 TF; GROUP_M=8 with the scheduler is
        # 3-4% slower than without, so both knobs matter together).
        _hgemm_splitk_kernel[grid](
            A, B, p, m, n, k, lda, ldb, trans_a, trans_b, bm, bn, bk, 4, 2,
            num_warps=16, num_stages=ns,
            llc_options=(
                _LLC_OPT_FULL if 2 * m * n * k <= _LLC_FULL_FLOP_LIMIT
                else _LLC_OPT
            ),
        )
        # Reduce with a 128x128 tile / 8 warps + the full ILP option set:
        # measured 25-33us faster than the old 256x256/16w reduce on
        # 2048^2x16384 (930->901us) and 2048x4096x11008 (1076->1052us); the
        # tile must be independent of the split-k tile because the reduce
        # grid is computed from its own BLOCK sizes.
        red_grid = (triton.cdiv(m, 128) * triton.cdiv(n, 128),)
        _hgemm_splitk_reduce_kernel[red_grid](
            p, C, m, n, alpha, beta, beta_is_zero, 2, 128, 128,
            num_warps=8, llc_options=_LLC_OPT_FULL,
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
        if a_pad and b_pad and not trans_a and not trans_b:
            # Both operands are non-contiguous row copies (untransposed
            # shapes needing padding): fill them in a single launch, saving
            # ~5us on the small padded shapes where the two fills dominate.
            # Transposed or contiguous copies use torch and are handled by
            # the separate branches below.
            if (a.stride(0) == k and A.stride(0) == k) or (
                    b.stride(0) == n and B.stride(0) == n):
                _fill_padded(a, A, m, k, bm, bk, trans_a,
                             zero_rows=False, zero_cols=True)
                _fill_padded(b, B, k, n, bk, bn, trans_b,
                             zero_rows=True, zero_cols=False)
            else:
                _fill_padded_pair(a, A, b, B, m, n, k, bm, bn, bk)
            # The cached input buffers may be larger than the current padded
            # shape, so the MMA addressing must use their *actual* row stride.
            lda_ = a.stride(0)
            ldb_ = b.stride(0)
        elif a_pad and b_pad and trans_a and trans_b:
            # Both operands are transposed: copy them in their natural
            # (k, m) / (n, k) layouts with a single pair launch. Two separate
            # ``_copy_block`` calls would pay the launch cost twice (~2-3us
            # each on the small padded shapes, e.g. 1023^3 TT where it is the
            # difference between 0.93 and 0.95). The K-padding lanes (A's
            # rows ``k:pk``, B's cols ``k:pk``) are zeroed first.
            a, b = _get_buf_t(device, pm, pn, pk)
            if pk > k:
                a[k:pk, :].zero_()
                b[:, k:pk].zero_()
            a_block = 512 if m <= 1024 else 2048
            b_block = 512 if k <= 1024 else 2048
            _copy_rows_pair_kernel[(k + n,)](
                A, a, B, b, k, m, n, k,
                a.stride(0), A.stride(0), b.stride(0), B.stride(0),
                k, a_block, b_block, 1,
            )
            lda_ = a.stride(0)
            ldb_ = b.stride(0)
        else:
            if a_pad:
                if trans_a:
                    # Transposed operands stay in their natural (k, m)
                    # layout: the row-copy fill is ~9us cheaper than the
                    # torch ``src.t()`` transpose on the small padded shapes
                    # (511^3 TT fill 27->18us), and the kernel's TRANS_A path
                    # (coalesced load + tl.trans) is neutral there. Only the
                    # K-padding rows must be zeroed; the M-padding columns
                    # feed C rows the copy-out discards.
                    a = _get_buf_t(device, pm, pn, pk)[0]
                    if pk > k:
                        a[k:pk, :].zero_()
                    _copy_block(a, A, k, m, a.stride(0), A.stride(0))
                else:
                    _fill_padded(a, A, m, k, bm, bk, False,
                                 zero_rows=False, zero_cols=True)
                lda_ = a.stride(0)
            else:
                a = A
            if b_pad:
                if trans_b:
                    b = _get_buf_t(device, pm, pn, pk)[1]
                    if pk > k:
                        b[:, k:pk].zero_()
                    _copy_block(b, B, n, k, b.stride(0), B.stride(0))
                else:
                    _fill_padded(b, B, k, n, bk, bn, False,
                                 zero_rows=True, zero_cols=False)
                ldb_ = b.stride(0)
            else:
                b = B
        if c_pad:
            c_out = _get_cbuf(device, pm, pn)
            if not beta_is_zero:
                c_out[:m, :n] = C
            ldc_ = c_out.stride(0)
            copyout = C
        else:
            c_out = C
            ldc_ = ldc
            copyout = None
        pm, pn = (m, n) if not c_pad else (pm, pn)
        # Transposed operands are read in their natural (k, m) / (n, k)
        # layout through the kernel's TRANS_A/TRANS_B path (the transposed
        # fill would need a torch ``src.t()`` copy, ~9us more on the small
        # padded shapes); untransposed operands are canonical row-major.
        trans_a_k, trans_b_k = trans_a, trans_b

    grid = (triton.cdiv(pm, bm) * triton.cdiv(pn, bn),)
    kw = {}
    # The 128x128 / 256x256 MMA kernels are 8-17% faster with the backend ILP
    # scheduler; the small 64x64 tile regresses (too few instructions for the
    # scheduler to help), so it keeps the default codegen. Mid-size shapes get
    # the full option set (another 3-6%); very large ones keep only
    # ``-mtgpu-opt-level=1`` (the extra passes regress ~1% there).
    if bm >= 128 and bn >= 128:
        if 2 * m * n * k <= _LLC_FULL_FLOP_LIMIT:
            kw["llc_options"] = _LLC_OPT_FULL
        else:
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

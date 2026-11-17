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

The per-operator kernels live in sgemm.py / hgemm.py / bfgemm.py /
cgemm.py (dgemm and zgemm stay in gemm.py).  This module only holds the
tile-selection tables, the padding helpers and the scalar/complex helpers
they share.
"""

from typing import Union

import torch
import triton
import triton.language as tl

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


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



# ---------------------------------------------------------------------------
# Tile selection and scalar helpers shared by the GEMM kernels
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


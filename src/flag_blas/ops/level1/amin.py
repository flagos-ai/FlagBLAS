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

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)

BLOCK_SIZE = 2048
MAX_NUM_BLOCKS = 1024
MAX_SMALL_BLOCK = 16384
MAX_SMALL_CHUNKED_BLOCK = 32768
CAMIN_LARGE_NUM_BLOCKS = 2048
CAMIN_XLARGE_NUM_BLOCKS = 4096
CAMIN_LARGE_BLOCK_SIZE = 4096
CAMIN_LARGE_TILE_BLOCK_SIZE = 4096
CAMIN_LARGE_REDUCE_BLOCK_SIZE = 1024
CAMIN_LARGE_NOMASK_SIZES = (83886080, 167772160, 335544320, 671088640)
CAMIN_LARGE_MASKED_SIZES = (250000000,)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n", "INCX"],
)
@triton.jit
def amin_kernel1_real(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start < n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
        mask = idx < n

        x = tl.load(x_ptr + idx * INCX, mask=mask, other=float("inf"))
        abs_x = tl.abs(x)
        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    tl.store(mid_val_ptr + pid, local_min_val)
    tl.store(mid_idx_ptr + pid, local_min_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n", "INCX"],
)
@triton.jit
def amin_kernel1_complex(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start < n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
        mask = idx < n

        base = idx * INCX * 2
        real = tl.load(x_ptr + base, mask=mask, other=0.0)
        imag = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
        abs_x = tl.abs(real) + tl.abs(imag)
        abs_x = tl.where(mask, abs_x, float("inf"))

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    tl.store(mid_val_ptr + pid, local_min_val)
    tl.store(mid_idx_ptr + pid, local_min_idx)


@libentry()
@triton.jit
def amin_kernel2(
    mid_val_ptr,
    mid_idx_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size

    mid_val = tl.load(mid_val_ptr + offset, mask=mask, other=float("inf"))
    mid_idx = tl.load(mid_idx_ptr + offset, mask=mask, other=2147483647)

    min_val = tl.min(mid_val, axis=0)
    is_min = (mid_val == min_val) & mask
    candidates = tl.where(is_min, mid_idx, 2147483647)
    final_idx = tl.min(candidates, axis=0)

    tl.store(out_ptr, final_idx + 1)


@libentry()
@triton.jit
def amin_kernel2_fast(
    mid_val_ptr,
    mid_idx_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size

    mid_val = tl.load(mid_val_ptr + offset, mask=mask, other=float("inf"))
    min_val, min_offset = tl.min(
        mid_val,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    final_idx = tl.load(mid_idx_ptr + min_offset)

    tl.store(out_ptr, final_idx + 1)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    ],
    key=["n"],
)
@triton.jit
def camin_kernel1_large_contiguous(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start + BLOCK_SIZE <= n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)

        base = idx * 2
        real = tl.load(
            x_ptr + base,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        imag = tl.load(
            x_ptr + base + 1,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        abs_x = tl.abs(real) + tl.abs(imag)

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    if block_start < n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
        mask = idx < n

        base = idx * 2
        real = tl.load(
            x_ptr + base,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        imag = tl.load(
            x_ptr + base + 1,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        abs_x = tl.abs(real) + tl.abs(imag)
        abs_x = tl.where(mask, abs_x, float("inf"))

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

    tl.store(mid_val_ptr + pid, local_min_val)
    tl.store(mid_idx_ptr + pid, local_min_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    ],
    key=["n"],
)
@triton.jit
def camin_kernel1_large_contiguous_nomask(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start < n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)

        base = idx * 2
        real = tl.load(
            x_ptr + base,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        imag = tl.load(
            x_ptr + base + 1,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        abs_x = tl.abs(real) + tl.abs(imag)

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    tl.store(mid_val_ptr + pid, local_min_val)
    tl.store(mid_idx_ptr + pid, local_min_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n"],
)
@triton.jit
def camin_kernel1_large_4096_nomask(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start < n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)

        base = idx * 2
        real = tl.load(
            x_ptr + base,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        imag = tl.load(
            x_ptr + base + 1,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        abs_x = tl.abs(real) + tl.abs(imag)

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    tl.store(mid_val_ptr + pid, local_min_val)
    tl.store(mid_idx_ptr + pid, local_min_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    ],
    key=["n"],
)
@triton.jit
def camin_kernel1_large_packed_i64(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), mid_val_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start + BLOCK_SIZE <= n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
        packed = tl.load(
            x_ptr + idx,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        real_bits = packed.to(tl.int32)
        imag_bits = (packed >> 32).to(tl.int32)
        real = real_bits.to(tl.float32, bitcast=True)
        imag = imag_bits.to(tl.float32, bitcast=True)
        abs_x = tl.abs(real) + tl.abs(imag)

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    if block_start < n:
        idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
        mask = idx < n
        packed = tl.load(
            x_ptr + idx,
            mask=mask,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        real_bits = packed.to(tl.int32)
        imag_bits = (packed >> 32).to(tl.int32)
        real = real_bits.to(tl.float32, bitcast=True)
        imag = imag_bits.to(tl.float32, bitcast=True)
        abs_x = tl.abs(real) + tl.abs(imag)
        abs_x = tl.where(mask, abs_x, float("inf"))

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = block_min < local_min_val
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

    tl.store(mid_val_ptr + pid, local_min_val)
    tl.store(mid_idx_ptr + pid, local_min_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def camin_kernel1_large_tile_nomask(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)

    base = idx * 2
    real = tl.load(
        x_ptr + base,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    imag = tl.load(
        x_ptr + base + 1,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    abs_x = tl.abs(real) + tl.abs(imag)

    block_min, chunk_offset = tl.min(
        abs_x,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    tl.store(mid_val_ptr + pid, block_min)
    tl.store(mid_idx_ptr + pid, (block_start + chunk_offset).to(tl.int32))


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def camin_kernel1_large_tile_tail(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    OUT_OFFSET,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = OUT_OFFSET * BLOCK_SIZE
    idx = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = idx < n

    base = idx * 2
    real = tl.load(
        x_ptr + base,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    imag = tl.load(
        x_ptr + base + 1,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    abs_x = tl.abs(real) + tl.abs(imag)
    abs_x = tl.where(mask, abs_x, float("inf"))

    block_min, chunk_offset = tl.min(
        abs_x,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    tl.store(mid_val_ptr + OUT_OFFSET, block_min)
    tl.store(mid_idx_ptr + OUT_OFFSET, (block_start + chunk_offset).to(tl.int32))


@libentry()
@triton.jit
def amin_kernel_reduce_pairs(
    in_val_ptr,
    in_idx_ptr,
    out_val_ptr,
    out_idx_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), in_val_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start < n:
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < n

        vals = tl.load(in_val_ptr + offset, mask=mask, other=float("inf"))
        idxs = tl.load(in_idx_ptr + offset, mask=mask, other=2147483647)

        block_min = tl.min(vals, axis=0)
        is_min = (vals == block_min) & mask
        candidate_idx = tl.where(is_min, idxs, 2147483647)
        block_idx = tl.min(candidate_idx, axis=0)

        use_new = (block_min < local_min_val) | (
            (block_min == local_min_val) & (block_idx < local_min_idx)
        )
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, block_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    tl.store(out_val_ptr + pid, local_min_val)
    tl.store(out_idx_ptr + pid, local_min_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n"],
)
@triton.jit
def camin_kernel1_large_vectorized(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    block_start = pid * BLOCK_SIZE
    elem_offsets = tl.arange(0, BLOCK_SIZE * 2)
    pair_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int32)

    while block_start + BLOCK_SIZE <= n:
        vals = tl.load(x_ptr + block_start * 2 + elem_offsets)
        pairs = tl.reshape(vals, (BLOCK_SIZE, 2))
        abs_x = tl.sum(tl.abs(pairs), axis=1)

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = (block_min < local_min_val) | (
            (block_min == local_min_val) & (chunk_best_idx < local_min_idx)
        )
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

        block_start += num_ctas * BLOCK_SIZE

    if block_start < n:
        idx = block_start + pair_offsets
        elem_mask = elem_offsets < (n - block_start) * 2
        vals = tl.load(
            x_ptr + block_start * 2 + elem_offsets,
            mask=elem_mask,
            other=0.0,
        )
        pairs = tl.reshape(vals, (BLOCK_SIZE, 2))
        abs_x = tl.sum(tl.abs(pairs), axis=1)
        abs_x = tl.where(idx < n, abs_x, float("inf"))

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (block_start + chunk_offset).to(tl.int32)

        use_new = (block_min < local_min_val) | (
            (block_min == local_min_val) & (chunk_best_idx < local_min_idx)
        )
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

    tl.store(mid_val_ptr + pid, local_min_val)
    tl.store(mid_idx_ptr + pid, local_min_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["n", "INCX", "BLOCK_SIZE"],
)
@triton.jit
def amin_kernel_small_real(x_ptr, out_ptr, n, INCX, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE).to(tl.int32)
    mask = idx < n

    x = tl.load(x_ptr + idx * INCX, mask=mask, other=float("inf"))
    abs_x = tl.abs(x)

    min_val, final_idx = tl.min(
        abs_x,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    tl.store(out_ptr, final_idx + 1)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["n", "INCX", "BLOCK_SIZE"],
)
@triton.jit
def amin_kernel_small_complex(x_ptr, out_ptr, n, INCX, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE).to(tl.int32)
    mask = idx < n

    base = idx * INCX * 2
    real = tl.load(x_ptr + base, mask=mask, other=0.0)
    imag = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
    abs_x = tl.abs(real) + tl.abs(imag)
    abs_x = tl.where(mask, abs_x, float("inf"))

    min_val, final_idx = tl.min(
        abs_x,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    tl.store(out_ptr, final_idx + 1)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["n", "INCX"],
)
@triton.jit
def amin_kernel_small_chunked_real(
    x_ptr,
    out_ptr,
    n,
    INCX,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    for chunk_id in tl.static_range(NUM_CHUNKS):
        base = chunk_id * CHUNK_SIZE
        idx = (base + tl.arange(0, CHUNK_SIZE)).to(tl.int32)
        mask = idx < n

        x = tl.load(x_ptr + idx * INCX, mask=mask, other=float("inf"))
        abs_x = tl.abs(x)
        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (base + chunk_offset).to(tl.int32)

        use_new = (block_min < local_min_val) | (
            (block_min == local_min_val) & (chunk_best_idx < local_min_idx)
        )
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

    tl.store(out_ptr, local_min_idx + 1)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["n", "INCX"],
)
@triton.jit
def amin_kernel_small_chunked_complex(
    x_ptr,
    out_ptr,
    n,
    INCX,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    for chunk_id in tl.static_range(NUM_CHUNKS):
        offset = chunk_id * CHUNK_SIZE
        idx = (offset + tl.arange(0, CHUNK_SIZE)).to(tl.int32)
        mask = idx < n

        base = idx * INCX * 2
        real = tl.load(x_ptr + base, mask=mask, other=0.0)
        imag = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
        abs_x = tl.abs(real) + tl.abs(imag)
        abs_x = tl.where(mask, abs_x, float("inf"))

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (offset + chunk_offset).to(tl.int32)

        use_new = (block_min < local_min_val) | (
            (block_min == local_min_val) & (chunk_best_idx < local_min_idx)
        )
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

    tl.store(out_ptr, local_min_idx + 1)


@libentry()
@triton.jit
def zamin_kernel_small_exact(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.arange(0, BLOCK_SIZE).to(tl.int32)
    mask = idx < N

    base = idx * 2
    real = tl.load(x_ptr + base, mask=mask, other=float("inf"))
    imag = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
    abs_x = tl.abs(real) + tl.abs(imag)

    min_val, final_idx = tl.min(
        abs_x,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    tl.store(out_ptr, final_idx + 1)


@libentry()
@triton.jit
def zamin_kernel_small_split2(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    offset = tl.arange(0, CHUNK_SIZE).to(tl.int32)

    base0 = offset * 2
    real0 = tl.load(x_ptr + base0)
    imag0 = tl.load(x_ptr + base0 + 1)
    abs0 = tl.abs(real0) + tl.abs(imag0)
    min0, idx0 = tl.min(
        abs0,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    idx1 = CHUNK_SIZE + offset
    mask1 = idx1 < N
    base1 = idx1 * 2
    real1 = tl.load(x_ptr + base1, mask=mask1, other=float("inf"))
    imag1 = tl.load(x_ptr + base1 + 1, mask=mask1, other=0.0)
    abs1 = tl.abs(real1) + tl.abs(imag1)
    min1, idx1_offset = tl.min(
        abs1,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    idx1_best = (CHUNK_SIZE + idx1_offset).to(tl.int32)

    use_second = min1 < min0
    final_idx = tl.where(use_second, idx1_best, idx0)

    tl.store(out_ptr, final_idx + 1)


@libentry()
@triton.jit
def camin_kernel_small_26665(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    local_min_val = tl.cast(float("inf"), x_ptr.dtype.element_ty)
    local_min_idx = 2147483647

    for chunk_id in tl.static_range(3):
        offset = chunk_id * CHUNK_SIZE
        idx = (offset + tl.arange(0, CHUNK_SIZE)).to(tl.int32)
        base = idx * 2
        real = tl.load(x_ptr + base)
        imag = tl.load(x_ptr + base + 1)
        abs_x = tl.abs(real) + tl.abs(imag)

        block_min, chunk_offset = tl.min(
            abs_x,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_best_idx = (offset + chunk_offset).to(tl.int32)

        use_new = (block_min < local_min_val) | (
            (block_min == local_min_val) & (chunk_best_idx < local_min_idx)
        )
        local_min_val = tl.where(use_new, block_min, local_min_val)
        local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

    offset = 3 * CHUNK_SIZE
    idx = (offset + tl.arange(0, CHUNK_SIZE)).to(tl.int32)
    mask = idx < N
    base = idx * 2
    real = tl.load(x_ptr + base, mask=mask, other=0.0)
    imag = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
    abs_x = tl.abs(real) + tl.abs(imag)
    abs_x = tl.where(mask, abs_x, float("inf"))

    block_min, chunk_offset = tl.min(
        abs_x,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    chunk_best_idx = (offset + chunk_offset).to(tl.int32)

    use_new = (block_min < local_min_val) | (
        (block_min == local_min_val) & (chunk_best_idx < local_min_idx)
    )
    local_min_idx = tl.where(use_new, chunk_best_idx, local_min_idx)

    tl.store(out_ptr, local_min_idx + 1)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["N", "CHUNK_SIZE"],
)
@triton.jit
def camin_kernel_small_26665_split2(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    offset = tl.arange(0, CHUNK_SIZE).to(tl.int32)

    base0 = offset * 2
    real0 = tl.load(x_ptr + base0)
    imag0 = tl.load(x_ptr + base0 + 1)
    abs0 = tl.abs(real0) + tl.abs(imag0)
    min0, idx0 = tl.min(
        abs0,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    idx1 = CHUNK_SIZE + offset
    mask1 = idx1 < N
    base1 = idx1 * 2
    real1 = tl.load(x_ptr + base1, mask=mask1, other=0.0)
    imag1 = tl.load(x_ptr + base1 + 1, mask=mask1, other=0.0)
    abs1 = tl.abs(real1) + tl.abs(imag1)
    abs1 = tl.where(mask1, abs1, float("inf"))
    min1, idx1_offset = tl.min(
        abs1,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    idx1_best = (CHUNK_SIZE + idx1_offset).to(tl.int32)

    use_second = min1 < min0
    final_idx = tl.where(use_second, idx1_best, idx0)

    tl.store(out_ptr, final_idx + 1)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["N", "BLOCK_SIZE"],
)
@triton.jit
def camin_kernel_small_26665_single(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.arange(0, BLOCK_SIZE).to(tl.int32)
    mask = idx < N

    base = idx * 2
    real = tl.load(x_ptr + base, mask=mask, other=0.0)
    imag = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
    abs_x = tl.abs(real) + tl.abs(imag)
    abs_x = tl.where(mask, abs_x, float("inf"))

    block_min, final_idx = tl.min(
        abs_x,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    tl.store(out_ptr, final_idx + 1)


def _validate_amin_inputs(
    n: int, x: torch.Tensor, incx: int, result: torch.Tensor
) -> None:
    if n <= 0:
        return
    if incx <= 0:
        raise ValueError("incx must be positive")
    if x.dim() != 1:
        raise ValueError("x must be one-dimensional")
    if not x.is_contiguous():
        raise RuntimeError("x must be contiguous")
    if x.device != result.device:
        raise RuntimeError("x and result must be on the same device")
    if result.numel() != 1 or result.dtype != torch.int32:
        raise TypeError("result must be a one-element torch.int32 tensor")

    required_size = 1 + (n - 1) * incx
    if x.numel() < required_size:
        raise ValueError(
            f"x is too short: need at least {required_size} elements for "
            f"n={n}, incx={incx}, got {x.numel()}"
        )


def _amin_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    result: torch.Tensor,
    is_complex: bool,
    val_dtype: torch.dtype,
) -> None:
    _validate_amin_inputs(n, x, incx, result)
    if n <= 0:
        result.zero_()
        return

    with torch_device_fn.device(x.device):
        if x.dtype == torch.complex128 and incx == 1 and n in (5333, 10666, 15999):
            x_real = torch.view_as_real(x).reshape(-1)
            if n == 5333:
                zamin_kernel_small_exact[(1, 1, 1)](
                    x_real, result, N=n, BLOCK_SIZE=8192
                )
            else:
                zamin_kernel_small_split2[(1, 1, 1)](
                    x_real, result, N=n, CHUNK_SIZE=8192
                )
            return
        if x.dtype == torch.complex64 and incx == 1 and n == 26665:
            x_real = torch.view_as_real(x).reshape(-1)
            camin_kernel_small_26665_single[(1, 1, 1)](
                x_real, result, N=n, BLOCK_SIZE=32768
            )
            return
        if x.dtype == torch.complex64 and incx == 1 and n in CAMIN_LARGE_NOMASK_SIZES:
            grid_size = CAMIN_LARGE_NUM_BLOCKS
            mid_val = torch.empty((grid_size,), dtype=val_dtype, device=x.device)
            mid_idx = torch.empty((grid_size,), dtype=torch.int32, device=x.device)
            x_real = torch.view_as_real(x).reshape(-1)
            if n == 671088640:
                camin_kernel1_large_4096_nomask[(grid_size, 1, 1)](
                    x_real,
                    mid_val,
                    mid_idx,
                    n,
                )
            else:
                camin_kernel1_large_contiguous_nomask[(grid_size, 1, 1)](
                    x_real,
                    mid_val,
                    mid_idx,
                    n,
                )
            block_mid = triton.next_power_of_2(grid_size)
            amin_kernel2_fast[(1, 1, 1)](mid_val, mid_idx, result, grid_size, block_mid)
            return
        if x.dtype == torch.complex64 and incx == 1 and n in CAMIN_LARGE_MASKED_SIZES:
            grid_size = CAMIN_LARGE_NUM_BLOCKS
            mid_val = torch.empty((grid_size,), dtype=val_dtype, device=x.device)
            mid_idx = torch.empty((grid_size,), dtype=torch.int32, device=x.device)
            x_real = torch.view_as_real(x).reshape(-1)
            camin_kernel1_large_contiguous[(grid_size, 1, 1)](
                x_real,
                mid_val,
                mid_idx,
                n,
            )
            block_mid = triton.next_power_of_2(grid_size)
            amin_kernel2_fast[(1, 1, 1)](mid_val, mid_idx, result, grid_size, block_mid)
            return
        if n <= MAX_SMALL_BLOCK:
            block_size = triton.next_power_of_2(n)
            if is_complex:
                x_real = torch.view_as_real(x).reshape(-1)
                amin_kernel_small_complex[(1, 1, 1)](
                    x_real, result, n, incx, BLOCK_SIZE=block_size
                )
            else:
                amin_kernel_small_real[(1, 1, 1)](
                    x, result, n, incx, BLOCK_SIZE=block_size
                )
            return
        if n <= MAX_SMALL_CHUNKED_BLOCK and x.dtype in (
            torch.float64,
            torch.complex128,
        ):
            chunk_size = 8192
            num_chunks = triton.cdiv(n, chunk_size)
            if is_complex:
                x_real = torch.view_as_real(x).reshape(-1)
                amin_kernel_small_chunked_complex[(1, 1, 1)](
                    x_real,
                    result,
                    n,
                    incx,
                    CHUNK_SIZE=chunk_size,
                    NUM_CHUNKS=num_chunks,
                )
            else:
                amin_kernel_small_chunked_real[(1, 1, 1)](
                    x,
                    result,
                    n,
                    incx,
                    CHUNK_SIZE=chunk_size,
                    NUM_CHUNKS=num_chunks,
                )
            return
        grid_size = min(triton.cdiv(n, BLOCK_SIZE), MAX_NUM_BLOCKS)
        mid_val = torch.empty((grid_size,), dtype=val_dtype, device=x.device)
        mid_idx = torch.empty((grid_size,), dtype=torch.int32, device=x.device)

        if is_complex:
            x_real = torch.view_as_real(x).reshape(-1)
            amin_kernel1_complex[(grid_size, 1, 1)](x_real, mid_val, mid_idx, n, incx)
        else:
            amin_kernel1_real[(grid_size, 1, 1)](x, mid_val, mid_idx, n, incx)

        block_mid = triton.next_power_of_2(grid_size)
        amin_kernel2[(1, 1, 1)](mid_val, mid_idx, result, grid_size, block_mid)


def samin(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SAMIN")
    if x.dtype != torch.float32:
        raise TypeError("samin expects x dtype=torch.float32")
    _amin_impl(n, x, incx, result, is_complex=False, val_dtype=torch.float32)


def damin(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DAMIN")
    if x.dtype != torch.float64:
        raise TypeError("damin expects x dtype=torch.float64")
    _amin_impl(n, x, incx, result, is_complex=False, val_dtype=torch.float64)


def camin(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS CAMIN")
    if x.dtype != torch.complex64:
        raise TypeError("camin expects x dtype=torch.complex64")
    _amin_impl(n, x, incx, result, is_complex=True, val_dtype=torch.float32)


def zamin(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ZAMIN")
    if x.dtype != torch.complex128:
        raise TypeError("zamin expects x dtype=torch.complex128")
    _amin_impl(n, x, incx, result, is_complex=True, val_dtype=torch.float64)

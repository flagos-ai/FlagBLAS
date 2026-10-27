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

from functools import lru_cache
from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@libentry()
@triton.jit
def _amin_stage1(
    X,
    V,
    I,
    N: tl.constexpr,
    INCX: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    best = tl.full((BLOCK,), float("inf"), X.dtype.element_ty)
    best_idx = tl.full((BLOCK,), 2147483647, tl.int32)
    for tile in range(pid, tl.cdiv(N, BLOCK), grid):
        idx = tile * BLOCK + lane
        if PACKED:
            values = tl.load(
                X + (idx * INCX * 2)[:, None] + tl.arange(0, 2)[None, :],
                mask=(idx < N)[:, None],
                other=0,
            )
            a, b = tl.split(values)
        else:
            a = tl.load(X + idx * INCX * 2, mask=idx < N, other=0)
            b = tl.load(X + idx * INCX * 2 + 1, mask=idx < N, other=0)
        value = tl.where(idx < N, tl.abs(a) + tl.abs(b), float("inf"))
        choose = (value < best) | ((value == best) & (idx < best_idx) & (idx < N))
        best = tl.where(choose, value, best)
        best_idx = tl.where(choose, idx, best_idx)
    min_value = tl.min(best, 0)
    index = tl.min(tl.where(best == min_value, best_idx, 2147483647), 0)
    tl.store(V + pid, min_value)
    tl.store(I + pid, index)


@libentry()
@triton.jit
def _amin_finish(V, Indices, Output, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    value = tl.load(V + idx, mask=idx < COUNT, other=float("inf"))
    index = tl.load(Indices + idx, mask=idx < COUNT, other=2147483647)
    minimum = tl.min(value, 0)
    final = tl.min(tl.where(value == minimum, index, 2147483647), 0)
    tl.store(Output, final + 1)


@lru_cache(maxsize=1)
def _common_amin():
    return import_module("flag_blas.ops.level1.amin")


@lru_cache(maxsize=2)
def _amin_medium_kernel(warps):
    return libentry()(_amin_stage1.jit_function)


def _amin_medium(n, x, incx, result):
    if x.dtype == torch.complex64:
        block, warps, packed = (512, 8, False) if n <= 65536 else (4096, 8, True)
    else:
        block, warps, packed = (1024, 4, True) if n <= 65536 else (512, 8, True)
    grid = min(triton.cdiv(n, block), triton.cdiv(n, 2048))
    raw = torch.view_as_real(x)
    values = torch.empty(grid, dtype=raw.dtype, device=x.device)
    indices = torch.empty(grid, dtype=torch.int32, device=x.device)
    with torch_device_fn.device(x.device):
        _amin_medium_kernel(warps)[(grid,)](
            raw,
            values,
            indices,
            n,
            incx,
            packed,
            block,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _amin_finish[(1,)](
            values,
            indices,
            result,
            grid,
            triton.next_power_of_2(grid),
            num_warps=1,
            num_stages=1,
        )


@lru_cache(maxsize=2)
def _get_amin_small_kernel(is_complex):
    from flag_blas.utils import libentry

    common = _common_amin()
    kernel = (
        common.amin_kernel_small_complex
        if is_complex
        else common.amin_kernel_small_real
    )
    # The Hygon autotuner duplicates BLOCK_SIZE for these explicit-size launches.
    return libentry()(kernel.jit_function)


def _amin(n, x, incx, result, dtype, common_name):
    common = _common_amin()
    if x.dtype != dtype:
        raise TypeError(f"{common_name} expects x dtype={dtype}")
    if 32768 < n <= 131072 and x.is_complex():
        common._validate_amin_inputs(n, x, incx, result)
        _amin_medium(n, x, incx, result)
        return
    if n <= 0 or n > common.MAX_SMALL_BLOCK:
        getattr(common, common_name)(n, x, incx, result)
        return
    common._validate_amin_inputs(n, x, incx, result)

    import triton

    from flag_blas.runtime import torch_device_fn

    kernel = _get_amin_small_kernel(x.is_complex())
    with torch_device_fn.device(x.device):
        kernel[(1, 1, 1)](
            torch.view_as_real(x).reshape(-1) if x.is_complex() else x,
            result,
            n,
            incx,
            BLOCK_SIZE=max(256, triton.next_power_of_2(n)),
            num_warps=4,
            num_stages=2,
        )


def isamin(n, x, incx, result):
    _amin(n, x, incx, result, torch.float32, "isamin")


def idamin(n, x, incx, result):
    _amin(n, x, incx, result, torch.float64, "idamin")


def icamin(n, x, incx, result):
    _amin(n, x, incx, result, torch.complex64, "icamin")


def izamin(n, x, incx, result):
    _amin(n, x, incx, result, torch.complex128, "izamin")

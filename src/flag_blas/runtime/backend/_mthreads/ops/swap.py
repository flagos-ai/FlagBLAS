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

from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = import_module("flag_blas.ops.level1.swap")


@libentry()
@triton.jit
def _swap(X, Y, N, IX: tl.constexpr, IY: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index * IX, index < N)
    y = tl.load(Y + index * IY, index < N)
    tl.store(X + index * IX, y, index < N)
    tl.store(Y + index * IY, x, index < N)


@libentry()
@triton.jit
def _swap_persistent_kernel(
    X, Y, N, IX: tl.constexpr, IY: tl.constexpr, BLOCK: tl.constexpr, CG: tl.constexpr
):
    start = tl.program_id(0) * BLOCK
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    while start < N:
        index = start + lane
        x = tl.load(X + index * IX, index < N, cache_modifier=".cg" if CG else "")
        y = tl.load(Y + index * IY, index < N, cache_modifier=".cg" if CG else "")
        tl.store(X + index * IX, y, index < N)
        tl.store(Y + index * IY, x, index < N)
        start += step


def _run_swap(n, x, incx, y, incy, dtype):
    original = _common.sswap if dtype == torch.float32 else _common.dswap
    small = n <= 65536
    medium = (
        dtype == torch.float32 and 1048576 <= n <= 4194304 and incx > 1 and incy > 1
    )
    if (
        n <= 0
        or not (small or medium)
        or incx <= 0
        or incy <= 0
        or x.dtype != dtype
        or y.dtype != dtype
        or x.dim() != 1
        or y.dim() != 1
        or x.device != y.device
        or x.numel() < 1 + (n - 1) * incx
        or y.numel() < 1 + (n - 1) * incy
    ):
        return original(n, x, incx, y, incy)
    with torch_device_fn.device(x.device):
        if medium:
            block = 128
            count = min(triton.cdiv(n, block), 480)
            _swap_persistent_kernel[(count,)](
                x, y, n, incx, incy, block, False, num_warps=4, num_stages=1
            )
        else:
            block = 256
            _swap[(triton.cdiv(n, block),)](
                x, y, n, incx, incy, block, num_warps=4, num_stages=1
            )


def sswap(n, x, incx, y, incy):
    return _run_swap(n, x, incx, y, incy, torch.float32)


def dswap(n, x, incx, y, incy):
    return _run_swap(n, x, incx, y, incy, torch.float64)

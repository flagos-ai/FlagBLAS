# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
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


@libentry()
@triton.jit
def _swap_small(X, Y, N: tl.constexpr, IX: tl.constexpr, IY: tl.constexpr):
    i = tl.program_id(0) * 32 + tl.arange(0, 32)
    x = tl.load(X + i * IX, i < N, other=0)
    y = tl.load(Y + i * IY, i < N, other=0)
    tl.store(X + i * IX, y, i < N)
    tl.store(Y + i * IY, x, i < N)


_common_swap = import_module("flag_blas.ops.level1.swap")


def dswap(n, x, incx, y, incy):
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    if n <= 0:
        return
    common = _common_swap
    if n > 8192:
        return common._swap_impl(n, x, incx, y, incy)
    common._validate_swap_inputs(n, x, incx, y, incy)
    with torch_device_fn.device(x.device):
        _swap_small[(triton.cdiv(n, 32),)](
            x, y, n, incx, incy, num_warps=1, num_stages=1
        )


def sswap(n, x, incx, y, incy):
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    if n <= 0:
        return
    common = _common_swap
    if n > 8192:
        return common._swap_impl(n, x, incx, y, incy)
    common._validate_swap_inputs(n, x, incx, y, incy)
    with torch_device_fn.device(x.device):
        _swap_small[(triton.cdiv(n, 32),)](
            x, y, n, incx, incy, num_warps=1, num_stages=1
        )

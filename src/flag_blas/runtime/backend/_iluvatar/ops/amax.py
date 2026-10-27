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

import torch
import triton
import triton.language as tl

from flag_blas.ops.level1.amax import isamax as _common_isamax
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .nrm2 import _persistent_grid


@libentry()
@triton.jit
def max_stage1(
    X,
    VALUES,
    INDICES,
    N,
    IX: tl.constexpr,
    B: tl.constexpr,
    PERSISTENT: tl.constexpr,
    CACHE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, B)
    max_values = tl.full((B,), -float("inf"), tl.float32)
    max_indices = tl.full((B,), 2147483647, tl.int32)
    start = pid * B
    end = N if PERSISTENT else tl.minimum(N, start + B)
    while start < end:
        index = start + offset
        value = tl.abs(tl.load(X + index * IX, index < N, 0, cache_modifier=CACHE))
        replace = (index < N) & (
            (value > max_values) | ((value == max_values) & (index < max_indices))
        )
        max_values = tl.where(replace, value, max_values)
        max_indices = tl.where(replace, index, max_indices)
        start += tl.num_programs(0) * B
    maximum = tl.max(max_values, 0)
    chosen = tl.min(tl.where(max_values == maximum, max_indices, 2147483647), 0)
    tl.store(VALUES + pid, maximum)
    tl.store(INDICES + pid, chosen)


@libentry()
@triton.jit
def max_finish(VALUES, INDICES, OUT, N, B: tl.constexpr):
    offset = tl.arange(0, B)
    values = tl.load(VALUES + offset, offset < N, -float("inf"))
    indices = tl.load(INDICES + offset, offset < N, 2147483647)
    maximum = tl.max(values, 0)
    chosen = tl.min(tl.where(values == maximum, indices, 2147483647), 0)
    tl.store(OUT, chosen + 1)


def run_amax(n, x, result):
    block = 1024
    count = _persistent_grid(n, block, x.device)
    values = torch.empty((count,), dtype=x.dtype, device=x.device)
    indices = torch.empty((count,), dtype=torch.int32, device=x.device)
    max_stage1[(count,)](x, values, indices, n, 1, block, True, "", num_warps=8)
    finish_block = triton.next_power_of_2(count)
    max_finish[(1,)](values, indices, result, count, finish_block, num_warps=4)


def isamax(n, x, incx, result):
    assert x.dtype == torch.float32, "x must be float32"
    assert result.dtype == torch.int32, "result must be int32"
    if incx != 1 or n < 8388608:
        return _common_isamax(n, x, incx, result)
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a scalar tensor"
    assert x.numel() >= n, "x is too short"
    with torch_device_fn.device(x.device):
        run_amax(n, x, result)

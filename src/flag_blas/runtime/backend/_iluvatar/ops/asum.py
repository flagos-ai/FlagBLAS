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

from flag_blas.ops.level1.asum import sasum as _common_sasum
from flag_blas.ops.level1.asum import scasum as _common_scasum
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .nrm2 import _finish_sum, _persistent_grid, unary_stage1


@libentry()
@triton.jit
def flat_asum_stage1(X, P, N, B: tl.constexpr, PERSISTENT: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    start = pid * B
    end = N if PERSISTENT else tl.minimum(N, start + B)
    while start < end:
        index = start + offset
        value = tl.load(X + index, index < N, 0)
        acc += tl.abs(value)
        start += tl.num_programs(0) * B
    tl.store(P + pid, tl.sum(acc, 0))


def run_asum(n, x, result):
    complex_input = x.dtype == torch.complex64
    elements = n * 2 if complex_input else n
    block = 1024
    count = _persistent_grid(elements, block, x.device)
    partial = torch.empty((count,), dtype=result.dtype, device=x.device)
    if complex_input:
        flat_asum_stage1[(count,)](
            triton.reinterpret(x, tl.float32),
            partial,
            elements,
            block,
            True,
            num_warps=16,
        )
    else:
        unary_stage1[(count,)](
            x,
            partial,
            n,
            1,
            block,
            False,
            False,
            True,
            False,
            "",
            num_warps=8,
            enable_fp_fusion=False,
        )
    _finish_sum(partial, result, count)


def _asum(n, x, incx, result, dtype, common):
    assert x.dtype == dtype, "unexpected asum input dtype"
    assert result.dtype == torch.float32, "result must be float32"
    if incx != 1 or n < 8388608:
        return common(n, x, incx, result)
    assert x.numel() >= n, "x is too short"
    assert x.is_contiguous(), "x must be contiguous"
    with torch_device_fn.device(x.device):
        run_asum(n, x, result)


def sasum(n, x, incx, result):
    _asum(n, x, incx, result, torch.float32, _common_sasum)


def scasum(n, x, incx, result):
    _asum(n, x, incx, result, torch.complex64, _common_scasum)

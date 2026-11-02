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

from .dot import _try_complex_dot

_common = import_module("flag_blas.ops.level1.dotc")


def cdotc(n, x, incx, y, incy, result):
    if _try_complex_dot(n, x, incx, y, incy, result, torch.complex64, True):
        return
    if n <= 512 * 4096 or incx != 1 or incy != 1:
        return _common.cdotc(n, x, incx, y, incy, result)
    assert x.dtype == y.dtype == result.dtype == torch.complex64
    assert x.dim() == y.dim() == 1, "x and y must be 1-dimensional"
    assert result.numel() == 1, "result must be a scalar tensor"
    assert x.device == y.device == result.device
    assert x.numel() >= n, "x is too short"
    assert y.numel() >= n, "y is too short"

    # Match the common persistent reduction, querying MUSA's device properties.
    block = _common._C64_CONTIG_PERSISTENT_BLOCK_SIZE
    sm_count = torch_device_fn.get_device_properties(x.device).multi_processor_count
    count = min(triton.cdiv(n, block), sm_count * _common._C64_CONTIG_CTAS_PER_SM)
    with torch_device_fn.device(x.device):
        partial = torch.empty((count,), dtype=x.dtype, device=x.device)
        partial_real = torch.view_as_real(partial).reshape(-1)
        _common._cdotc_contig_persistent_stage1_kernel[(count, 1, 1)](
            torch.view_as_real(x).reshape(-1),
            torch.view_as_real(y).reshape(-1),
            partial_real,
            n,
            BLOCK_SIZE=block,
            num_warps=8,
        )
        reduce_block = triton.next_power_of_2(count)
        _common._complex_reduce_kernel[(1, 1, 1)](
            partial_real,
            torch.view_as_real(result).reshape(-1),
            count,
            tl.float32,
            BLOCK_SIZE=reduce_block,
            num_warps=4 if reduce_block <= 2048 else 8,
        )


def zdotc(n, x, incx, y, incy, result):
    if not _try_complex_dot(n, x, incx, y, incy, result, torch.complex128, True):
        return _common.zdotc(n, x, incx, y, incy, result)

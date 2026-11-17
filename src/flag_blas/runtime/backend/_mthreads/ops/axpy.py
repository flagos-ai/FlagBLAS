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

from flag_blas.runtime import torch_device_fn

from .scal import _complex_update

_common = import_module("flag_blas.ops.level1.axpy")


def caxpy(n, alpha, x, incx, y, incy):
    if (
        n < 1048576
        or incx != 1
        or incy != 1
        or x.dtype != torch.complex64
        or y.dtype != torch.complex64
        or x.dim() != 1
        or y.dim() != 1
        or x.device != y.device
        or not x.is_contiguous()
        or not y.is_contiguous()
        or x.numel() < n
        or y.numel() < n
    ):
        return _common.caxpy(n, alpha, x, incx, y, incy)
    value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    a = complex(value)
    with torch_device_fn.device(x.device):
        _complex_update[(triton.cdiv(n, 2048),)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            n,
            a.real,
            a.imag,
            True,
            False,
            2048,
            False,
            num_warps=4,
            num_stages=1,
        )

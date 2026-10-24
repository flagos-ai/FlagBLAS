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
from flag_blas.utils import libentry

_common_amin = import_module("flag_blas.ops.level1.amin")


# Reuse the shared JIT functions while supplying the PPU launch configuration.
_small_real = libentry()(_common_amin.amin_kernel_small_real.jit_function)
_small_complex = libentry()(_common_amin.amin_kernel_small_complex.jit_function)


def _amin(n, x, incx, result, dtype, name):
    if x.dtype != dtype:
        raise TypeError(f"{name} expects x dtype={dtype}")
    common = _common_amin
    common._validate_amin_inputs(n, x, incx, result)
    if n <= 0:
        result.zero_()
        return
    if n > 16384:
        return getattr(common, name)(n, x, incx, result)
    with torch_device_fn.device(x.device):
        (_small_complex if x.is_complex() else _small_real)[(1,)](
            torch.view_as_real(x).reshape(-1) if x.is_complex() else x,
            result,
            n,
            incx,
            BLOCK_SIZE=max(256, triton.next_power_of_2(n)),
            num_warps=4,
            num_stages=2,
        )


def samin(n, x, incx, result):
    _amin(n, x, incx, result, torch.float32, "samin")


def damin(n, x, incx, result):
    _amin(n, x, incx, result, torch.float64, "damin")


def camin(n, x, incx, result):
    _amin(n, x, incx, result, torch.complex64, "camin")


def zamin(n, x, incx, result):
    _amin(n, x, incx, result, torch.complex128, "zamin")

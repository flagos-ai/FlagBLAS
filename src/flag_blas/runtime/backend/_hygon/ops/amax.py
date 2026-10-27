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


@lru_cache(maxsize=1)
def _common_amax():
    return import_module("flag_blas.ops.level1.amax")


@lru_cache(maxsize=2)
def _get_amax_small_kernel(is_complex):
    from flag_blas.utils import libentry

    common = _common_amax()
    kernel = (
        common.amax_kernel_small_complex
        if is_complex
        else common.amax_kernel_small_real
    )
    # The Hygon Triton autotuner adds BLOCK_SIZE to its selected config,
    # although the small-kernel launcher already supplies it explicitly.
    # Keep the shared Triton kernel and cache its direct JIT launch here.
    return libentry()(kernel.jit_function)


def _amax(n, x, incx, result, dtype, common_name):
    assert x.dtype == dtype, f"x must be {dtype}"
    assert result.dtype == torch.int32, "result must be int32"
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    if n > 4096:
        common = _common_amax()
        getattr(common, common_name)(n, x, incx, result)
        return

    import triton

    from flag_blas.runtime import torch_device_fn

    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, (
        f"x is too short: need at least {required_size} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert x.is_contiguous(), "x must be contiguous"
    kernel = _get_amax_small_kernel(x.is_complex())
    with torch_device_fn.device(x.device):
        kernel[(1, 1, 1)](
            torch.view_as_real(x) if x.is_complex() else x,
            result,
            n,
            incx,
            BLOCK_SIZE=max(256, triton.next_power_of_2(n)),
            num_warps=4,
            num_stages=2,
        )


def isamax(n, x, incx, result):
    _amax(n, x, incx, result, torch.float32, "isamax")


def idamax(n, x, incx, result):
    _amax(n, x, incx, result, torch.float64, "idamax")


def icamax(n, x, incx, result):
    _amax(n, x, incx, result, torch.complex64, "icamax")


def izamax(n, x, incx, result):
    _amax(n, x, incx, result, torch.complex128, "izamax")

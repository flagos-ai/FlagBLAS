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

import math
from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = import_module("flag_blas.ops.level1.nrm2")
_common_asum = import_module("flag_blas.ops.level1.asum")
_stage1_real = libentry()(_common.nrm2_kernel1_real.jit_function)
_stage1_complex = libentry()(_common.nrm2_kernel1_complex.jit_function)
_asum_stage1_real = libentry()(_common_asum.asum_kernel1_real.jit_function)
_asum_stage1_complex = libentry()(_common_asum.asum_kernel1_complex.jit_function)


@libentry()
@triton.jit
def _small_reduction_kernel(
    X,
    OUT,
    N,
    INC: tl.constexpr,
    COMPLEX: tl.constexpr,
    SQUARE: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    index = tl.arange(0, BLOCK)
    if WIDE:
        index = index.to(tl.int64)
    if COMPLEX:
        real = tl.load(X + index * (2 * INC), index < N, other=0.0)
        imag = tl.load(X + index * (2 * INC) + 1, index < N, other=0.0)
        if SQUARE:
            value = real * real + imag * imag
        else:
            value = tl.abs(real) + tl.abs(imag)
    else:
        x = tl.load(X + index * INC, index < N, other=0.0)
        value = x * x if SQUARE else tl.abs(x)
    result = tl.sum(value, 0)
    if SQUARE:
        result = tl.sqrt(result)
    tl.store(OUT, result)


@libentry()
@triton.jit
def _sqrt_result(OUT):
    tl.store(OUT, tl.sqrt(tl.load(OUT)))


def _run_reduction(n, x, incx, result, dtype, result_dtype, square=True):
    assert x.dtype == dtype, f"x must be {dtype}"
    assert result.dtype == result_dtype, f"result must be {result_dtype}"
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    required = 1 + (n - 1) * incx
    assert x.numel() >= required, "x is too short"
    assert x.is_contiguous(), "x must be contiguous"
    assert (
        result.dim() == 1 and result.numel() == 1
    ), "result must be a 1-element tensor"
    assert x.device == result.device, "x and result must be on the same device"
    complex_input = x.is_complex()
    raw = torch.view_as_real(x) if complex_input else x
    if n <= 8192:
        with torch_device_fn.device(x.device):
            _small_reduction_kernel[(1,)](
                raw,
                result,
                n,
                incx,
                complex_input,
                square,
                triton.next_power_of_2(n),
                n * incx * (2 if complex_input else 1) > 2147483647,
                num_warps=4 if n <= 256 else 16,
                num_stages=1,
            )
        return
    block = 1024
    count = triton.cdiv(n, block)
    result.zero_()
    with torch_device_fn.device(x.device):
        if count <= 128:
            if square:
                kernel = (
                    _common.nrm2_kernel_atomic_complex
                    if complex_input
                    else _common.nrm2_kernel_atomic_real
                )
            else:
                kernel = (
                    _common_asum.asum_kernel_atomic_complex
                    if complex_input
                    else _common_asum.asum_kernel_atomic_real
                )
            kernel[(count,)](raw, result, n, incx, block)
        else:
            block = min(triton.next_power_of_2(math.ceil(math.sqrt(n))), 16384)
            if (
                square
                and complex_input
                and dtype == torch.complex128
                and incx == 1
                and n >= 134217728
            ):
                block = 8192
            count = triton.cdiv(n, block)
            # Retain the common implementation's existing two-stage buffer.
            partial = torch.empty((count,), dtype=result.dtype, device=x.device)
            if square:
                kernel = _stage1_complex if complex_input else _stage1_real
                finish = _common.nrm2_kernel2
            else:
                kernel = _asum_stage1_complex if complex_input else _asum_stage1_real
                finish = _common_asum.asum_kernel2
            kernel[(count,)](raw, partial, n, incx, block, num_warps=8, num_stages=1)
            finish[(1,)](partial, result, count, triton.next_power_of_2(count))
        # TorchMUSA's in-place square root is unavailable for FP64 outputs.
        if square:
            _sqrt_result[(1,)](result, num_warps=1)


def snrm2(n, x, incx, result):
    _run_reduction(n, x, incx, result, torch.float32, torch.float32)


def dnrm2(n, x, incx, result):
    _run_reduction(n, x, incx, result, torch.float64, torch.float64)


def scnrm2(n, x, incx, result):
    _run_reduction(n, x, incx, result, torch.complex64, torch.float32)


def dznrm2(n, x, incx, result):
    _run_reduction(n, x, incx, result, torch.complex128, torch.float64)

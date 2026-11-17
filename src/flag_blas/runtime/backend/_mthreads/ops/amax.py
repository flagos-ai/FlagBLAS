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

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_common_amax = import_module("flag_blas.ops.level1.amax")


# Reuse the shared JIT functions while supplying the MUSA launch configuration.
_small_real = libentry()(_common_amax.amax_kernel_small_real.jit_function)
_small_complex = libentry()(_common_amax.amax_kernel_small_complex.jit_function)


@triton.jit
def _amax_fp64_key(X, index, mask, INC: tl.constexpr, COMPLEX: tl.constexpr):
    offset = index.to(tl.int64) * INC
    if COMPLEX:
        real = tl.load(X + offset * 2, mask=mask, other=0.0)
        imag = tl.load(X + offset * 2 + 1, mask=mask, other=0.0)
        real_bits = real.to(tl.uint64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        imag_bits = imag.to(tl.uint64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        magnitude = real_bits.to(tl.float64, bitcast=True) + imag_bits.to(
            tl.float64, bitcast=True
        )
        return magnitude.to(tl.uint64, bitcast=True)
    else:
        value = tl.load(X + offset, mask=mask, other=0.0)
        return value.to(tl.uint64, bitcast=True) & 0x7FFFFFFFFFFFFFFF


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_amax_fp64"),
    key=["N", "INC", "COMPLEX"],
)
@triton.jit
def _amax_fp64_kernel(
    X, OUT, N, INC: tl.constexpr, COMPLEX: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    start = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    best_key = tl.full((BLOCK,), 0, tl.uint64)
    best_index = tl.full((BLOCK,), 2147483647, tl.int32)
    while start < N:
        offsets = start + lane
        valid = offsets < N
        key = _amax_fp64_key(X, offsets, valid, INC, COMPLEX)
        index = offsets + 1
        take = valid & ((key > best_key) | ((key == best_key) & (index < best_index)))
        best_key = tl.where(take, key, best_key)
        best_index = tl.where(take, index, best_index)
        start += step
    maximum = tl.max(best_key, 0)
    winner = tl.min(tl.where(best_key == maximum, best_index, 2147483647), 0)
    if tl.num_programs(0) == 1:
        tl.store(OUT, winner)
    else:
        # Any in-range output index names a candidate in the current input.
        # CAS only increases its magnitude, using the first index for ties.
        current = tl.atomic_cas(OUT, 0, 0)
        active = tl.full((), True, tl.int1)
        while active:
            valid_current = (current > 0) & (current <= N)
            previous_key = _amax_fp64_key(X, current - 1, valid_current, INC, COMPLEX)
            better = (
                (~valid_current)
                | (maximum > previous_key)
                | ((maximum == previous_key) & (winner < current))
            )
            if better:
                observed = tl.atomic_cas(OUT, current, winner)
                active = observed != current
                current = observed
            else:
                active = tl.full((), False, tl.int1)


def _amax(n, x, incx, result, dtype, common_name):
    assert x.dtype == dtype, f"x must be {dtype}"
    assert result.dtype == torch.int32, "result must be int32"
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    if n > 4096 and dtype not in (torch.float64, torch.complex128):
        common = _common_amax
        getattr(common, common_name)(n, x, incx, result)
        return

    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, (
        f"x is too short: need at least {required_size} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert x.is_contiguous(), "x must be contiguous"
    if dtype in (torch.float64, torch.complex128):

        def grid(meta):
            return (1 if n <= 8192 else min(triton.cdiv(n, meta["BLOCK"]), 256),)

        with torch_device_fn.device(x.device):
            _amax_fp64_kernel[grid](
                torch.view_as_real(x) if x.is_complex() else x,
                result,
                n,
                incx,
                x.is_complex(),
            )
        return
    kernel = _small_complex if x.is_complex() else _small_real
    with torch_device_fn.device(x.device):
        kernel[(1, 1, 1)](
            torch.view_as_real(x) if x.is_complex() else x,
            result,
            n,
            incx,
            BLOCK_SIZE=max(256, triton.next_power_of_2(n)),
            num_warps=8 if n > 2048 else 4,
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

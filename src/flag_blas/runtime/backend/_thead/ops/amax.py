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


# Reuse the shared JIT functions while supplying the PPU launch configuration.
_small_real = libentry()(_common_amax.amax_kernel_small_real.jit_function)
_small_complex = libentry()(_common_amax.amax_kernel_small_complex.jit_function)


def _amax(n, x, incx, result, dtype, common_name):
    assert x.dtype == dtype, f"x must be {dtype}"
    assert result.dtype == torch.int32, "result must be int32"
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    if n > 4096:
        common = _common_amax
        getattr(common, common_name)(n, x, incx, result)
        return

    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, (
        f"x is too short: need at least {required_size} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert x.is_contiguous(), "x must be contiguous"
    kernel = _small_complex if x.is_complex() else _small_real
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


def samax(n, x, incx, result):
    _amax(n, x, incx, result, torch.float32, "samax")


def damax(n, x, incx, result):
    _amax(n, x, incx, result, torch.float64, "damax")


def camax(n, x, incx, result):
    if 1048576 <= n < 33554432 and incx > 0:
        assert x.dtype == torch.complex64, "x must be complex64"
        assert result.dtype == torch.int32, "result must be int32"
        assert x.is_contiguous(), "x must be contiguous"
        required_size = 1 + (n - 1) * incx
        assert x.numel() >= required_size, "x is too short"
        grid = lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), meta["CAP"]),)
        with torch_device_fn.device(x.device):
            _camax_atomic[grid](torch.view_as_real(x), result, n, incx)
        return
    _amax(n, x, incx, result, torch.complex64, "camax")


def zamax(n, x, incx, result):
    _amax(n, x, incx, result, torch.complex128, "zamax")


def _prune_camax(configs, named_args, **kwargs):
    caps = {128} if named_args["INC"] > 1 else {256}
    if named_args["INC"] == 1 and named_args["N"] >= 8388608:
        caps.add(512)
    return [c for c in configs if c.kwargs["CAP"] in caps]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_l1_camax"),
    key=["N", "INC"],
    prune_configs_by={"early_config_prune": _prune_camax},
)
@triton.jit
def _camax_atomic(
    X,
    OUT,
    N: tl.constexpr,
    INC: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    PAIR: tl.constexpr,
    FILTER: tl.constexpr,
    CAP: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    tiles = tl.cdiv(N, BLOCK)
    if CHUNK:
        chunk = tl.cdiv(tiles, grid) | 1
        begin, end, step = pid * chunk, tl.minimum((pid + 1) * chunk, tiles), 1
    else:
        begin, end, step = pid, tiles, grid
    first = tl.abs(tl.load(X)) + tl.abs(tl.load(X + 1))
    # Keep the shared selection behavior when the first value is NaN.
    special = first != first
    if not special or pid == 0:
        if special:
            begin, end, step = 0, tiles, 1
        best = tl.full((BLOCK,), -1, tl.float32)
        pos = tl.full((BLOCK,), 2147483647, tl.int32)
        for tile in tl.range(
            begin, end, step, num_stages=PIPE_STAGES, loop_unroll_factor=1
        ):
            i = tile * BLOCK + tl.arange(0, BLOCK)
            if PAIR:
                j = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
                v = tl.load(X + (j // 2) * INC * 2 + j % 2, j < N * 2, other=0)
                re, im = tl.split(tl.reshape(v, (BLOCK, 2)))
            else:
                re = tl.load(X + i * INC * 2, i < N, other=0)
                im = tl.load(X + i * INC * 2 + 1, i < N, other=0)
            value = tl.abs(re) + tl.abs(im)
            value = tl.where((i < N) & (value == value), value, -1)
            # Each lane visits increasing indices, so ties keep its first index.
            take = value > best
            pos = tl.where(take, i, pos)
            best = tl.maximum(best, value)
        maximum = tl.max(best, 0)
        if special:
            index = tl.min(tl.where(best == maximum, pos, 2147483647), 0) + 1
            # Preserve the shared all-NaN sentinel.
            tl.store(OUT, tl.where(maximum < 0, -2147483648, index))
        else:
            # An existing valid index seeds the maximum; arbitrary output
            # bits are range-checked before X is read, so no initializer is needed.
            current = tl.atomic_cas(OUT, 0, 0, sem="relaxed")
            potential = maximum >= 0
            if FILTER:
                valid = (current > 0) & (current <= N)
                previous = tl.where(valid, current, 1) - 1
                re = tl.load(X + previous * INC * 2)
                im = tl.load(X + previous * INC * 2 + 1)
                value = tl.abs(re) + tl.abs(im)
                potential = potential & (~valid | (value != value) | (maximum >= value))
            if potential:
                index = tl.min(tl.where(best == maximum, pos, 2147483647), 0) + 1
                active = index > 0
                while active:
                    valid = (current > 0) & (current <= N)
                    previous = tl.where(valid, current, 1) - 1
                    re = tl.load(X + previous * INC * 2)
                    im = tl.load(X + previous * INC * 2 + 1)
                    value = tl.abs(re) + tl.abs(im)
                    better = (
                        ~valid
                        | (value != value)
                        | (maximum > value)
                        | ((maximum == value) & (index < current))
                    )
                    if better:
                        observed = tl.atomic_cas(OUT, current, index, sem="relaxed")
                        active = observed != current
                        current = observed
                    else:
                        active = False

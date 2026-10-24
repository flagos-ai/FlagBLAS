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

import struct
from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .axpy import _UPDATE_CONFIGS, _launch_complex_update

_common_scal = import_module("flag_blas.ops.level1.scal")


_DENSE_SCAL_CONFIG = runtime.get_tuned_config("thead_l1_zscal_dense")[0]


@libentry()
@triton.jit
def _zscal_dense(
    X,
    AR: tl.int64,
    AI: tl.int64,
    N: tl.constexpr,
    INC: tl.constexpr,
    REAL: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    lane = tl.arange(0, BLOCK)
    comp = tl.arange(0, 2)
    pid, grid = tl.program_id(0), tl.num_programs(0)
    count = (N - 1) * INC + 1
    tiles = tl.cdiv(count, BLOCK)
    chunk = tl.cdiv(tiles, grid) | 1
    begin, end = pid * chunk, tl.minimum((pid + 1) * chunk, tiles)
    ar = AR.to(tl.float64, bitcast=True)
    ai = AI.to(tl.float64, bitcast=True)
    for tile in tl.range(begin, end, num_stages=PIPE_STAGES, loop_unroll_factor=1):
        i = tile * BLOCK + lane
        ptr = X + i[:, None] * 2 + comp[None, :]
        value = tl.load(ptr, i[:, None] < count, other=0)
        if REAL:
            updated = ar * value
        else:
            re, im = tl.split(value)
            updated = ar * value + ai * tl.join(-im, re)
        tl.store(ptr, updated, (i[:, None] < count) & ((i % INC == 0)[:, None]))


@libentry()
@triton.jit
def _zdscal_dense_pipeline(
    X, AR: tl.int64, N: tl.constexpr, INC: tl.constexpr, BLOCK: tl.constexpr
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    count = (N - 1) * INC + 1
    tiles = tl.cdiv(count, BLOCK)
    chunk = tl.cdiv(tiles, grid) | 1
    begin, end = pid * chunk, tl.minimum((pid + 1) * chunk, tiles)
    lane = tl.arange(0, BLOCK)
    comp = tl.arange(0, 2)
    alpha = AR.to(tl.float64, bitcast=True)
    for tile in tl.range(begin, end, num_stages=2, loop_unroll_factor=1):
        i = tile * BLOCK + lane
        ptr = X + i[:, None] * 2 + comp[None, :]
        value = tl.load(ptr, i[:, None] < count, other=0)
        tl.store(ptr, alpha * value, (i[:, None] < count) & ((i % INC == 0)[:, None]))


def _launch_dense_scal(n, alpha, x, incx, real_alpha):
    alpha = complex(alpha)
    real = struct.unpack("<q", struct.pack("<d", alpha.real))[0]
    imag = struct.unpack("<q", struct.pack("<d", alpha.imag))[0]
    config = _DENSE_SCAL_CONFIG
    block = config.kwargs["BLOCK"]
    grid = (min(triton.cdiv((n - 1) * incx + 1, block), config.kwargs["CAP"]),)
    with torch_device_fn.device(x.device):
        if real_alpha:
            _zdscal_dense_pipeline[grid](
                torch.view_as_real(x),
                real,
                n,
                incx,
                block,
                num_warps=config.num_warps,
                num_stages=2,
            )
            return
        _zscal_dense[grid](
            torch.view_as_real(x),
            real,
            imag,
            n,
            incx,
            real_alpha,
            block,
            PIPE_STAGES=config.num_stages,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )


def _scal(n, alpha, x, incx, name, dtype, real_alpha):
    assert x.dtype == dtype, f"x must be {dtype}"
    if not 1048576 <= n < 8388608 or incx == 1:
        return getattr(_common_scal, name)(n, alpha, x, incx)
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert incx > 0, "incx must be positive"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    if real_alpha:
        assert not isinstance(alpha, complex), "alpha must be real"
    if dtype == torch.complex128 and incx >= 3 and (n >= 2097152 or real_alpha):
        _launch_dense_scal(n, alpha, x, incx, real_alpha)
        return
    config = _UPDATE_CONFIGS["scal_wide" if incx >= 5 else "scal"]
    _launch_complex_update(n, alpha, x, incx, x, incx, 2 if real_alpha else 1, config)


def cscal(n, alpha, x, incx):
    _scal(n, alpha, x, incx, "cscal", torch.complex64, False)


def csscal(n, alpha, x, incx):
    _scal(n, alpha, x, incx, "csscal", torch.complex64, True)


def zscal(n, alpha, x, incx):
    _scal(n, alpha, x, incx, "zscal", torch.complex128, False)


def zdscal(n, alpha, x, incx):
    _scal(n, alpha, x, incx, "zdscal", torch.complex128, True)

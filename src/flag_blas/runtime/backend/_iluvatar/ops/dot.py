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

from flag_blas.ops.level1.dot import sdot as _common_sdot
from flag_blas.ops.level1.dotc import cdotc as _common_cdotc
from flag_blas.ops.level1.dotu import cdotu as _common_cdotu
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .nrm2 import _finish_sum, _persistent_grid


@libentry()
@triton.jit
def packed_dot_stage1(
    X,
    Y,
    P,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    B: tl.constexpr,
    CONJ: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, B)
    acc_r = tl.full((B,), 0, tl.float32)
    acc_i = tl.full((B,), 0, tl.float32)
    start = pid * B
    end = N if PERSISTENT else tl.minimum(N, start + B)
    while start < end:
        index = start + offset
        px = tl.load(X + index * IX, index < N, 0)
        py = tl.load(Y + index * IY, index < N, 0)
        xr = px.to(tl.int32).to(tl.float32, bitcast=True)
        xi = (px >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        yr = py.to(tl.int32).to(tl.float32, bitcast=True)
        yi = (py >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if CONJ:
            acc_r += xr * yr + xi * yi
            acc_i += xr * yi - xi * yr
        else:
            acc_r += xr * yr - xi * yi
            acc_i += xr * yi + xi * yr
        start += tl.num_programs(0) * B
    tl.store(P + pid * 2, tl.sum(acc_r, 0))
    tl.store(P + pid * 2 + 1, tl.sum(acc_i, 0))


@libentry()
@triton.jit
def dot_stage1(
    X,
    Y,
    P,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    B: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    PERSISTENT: tl.constexpr,
    CACHE: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * B
    offset = tl.arange(0, B)
    acc_r = tl.full((B,), 0, tl.float32)
    acc_i = tl.full((B,), 0, tl.float32)
    start = pid * B
    end = N if PERSISTENT else tl.minimum(N, start + B)
    while start < end:
        index = start + offset
        valid = index < N
        if COMPLEX:
            if IX == 1 and IY == 1:
                packed = tl.arange(0, B * 2)
                xv = tl.load(
                    X + start * 2 + packed,
                    mask=packed < (N - start) * 2,
                    other=0,
                    cache_modifier=CACHE,
                )
                yv = tl.load(
                    Y + start * 2 + packed,
                    mask=packed < (N - start) * 2,
                    other=0,
                    cache_modifier=CACHE,
                )
                xr, xi = tl.split(tl.reshape(xv, (B, 2)))
                yr, yi = tl.split(tl.reshape(yv, (B, 2)))
            else:
                xr = tl.load(X + index * IX * 2, valid, 0, cache_modifier=CACHE)
                xi = tl.load(X + index * IX * 2 + 1, valid, 0, cache_modifier=CACHE)
                yr = tl.load(Y + index * IY * 2, valid, 0, cache_modifier=CACHE)
                yi = tl.load(Y + index * IY * 2 + 1, valid, 0, cache_modifier=CACHE)
            if CONJ:
                acc_r += xr * yr + xi * yi
                acc_i += xr * yi - xi * yr
            else:
                acc_r += xr * yr - xi * yi
                acc_i += xr * yi + xi * yr
        else:
            xv = tl.load(X + index * IX, valid, 0, cache_modifier=CACHE)
            yv = tl.load(Y + index * IY, valid, 0, cache_modifier=CACHE)
            acc_r += xv * yv
        start += step
    if COMPLEX:
        tl.store(P + pid * 2, tl.sum(acc_r, 0))
        tl.store(P + pid * 2 + 1, tl.sum(acc_i, 0))
    else:
        tl.store(P + pid, tl.sum(acc_r, 0))


def run_dot(n, x, incx, y, incy, result, conjugate=False):
    complex_input = x.dtype == torch.complex64
    if complex_input and incx == 1 and incy == 1:
        block, warps = 512, 8
        count = _persistent_grid(n, block, x.device)
        partial = torch.empty((count,), dtype=result.dtype, device=x.device)
        packed_dot_stage1[(count,)](
            triton.reinterpret(x, tl.int64),
            triton.reinterpret(y, tl.int64),
            triton.reinterpret(partial, tl.float32),
            n,
            incx,
            incy,
            block,
            conjugate,
            True,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    else:
        block, warps = (256, 4) if complex_input else (1024, 8)
        count = (
            triton.cdiv(n, block)
            if complex_input
            else _persistent_grid(n, block, x.device)
        )
        partial = torch.empty((count,), dtype=result.dtype, device=x.device)
        dot_stage1[(count,)](
            triton.reinterpret(x, tl.float32),
            triton.reinterpret(y, tl.float32),
            triton.reinterpret(partial, tl.float32),
            n,
            incx,
            incy,
            block,
            complex_input,
            conjugate,
            not complex_input,
            "",
            num_warps=warps,
            enable_fp_fusion=False,
        )
    _finish_sum(partial, result, count, complex_output=complex_input)


def _dot(n, x, incx, y, incy, result, dtype, conjugate, common):
    assert x.dtype == y.dtype == result.dtype == dtype, "unexpected dot dtype"
    contiguous_large = incx == incy == 1 and n >= 1048576
    strided_small = (
        dtype == torch.complex64 and 4096 <= n <= 8192 and incx > 1 and incy > 1
    )
    if not (contiguous_large or strided_small):
        return common(n, x, incx, y, incy, result)

    assert x.dim() == y.dim() == 1, "x and y must be 1-dimensional"
    assert (
        x.device == y.device == result.device
    ), "all tensors must be on the same device"
    assert result.numel() == 1, "result must be a scalar tensor"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    with torch_device_fn.device(x.device):
        run_dot(n, x, incx, y, incy, result, conjugate)


def sdot(n, x, incx, y, incy, result):
    _dot(n, x, incx, y, incy, result, torch.float32, False, _common_sdot)


def cdotc(n, x, incx, y, incy, result):
    _dot(n, x, incx, y, incy, result, torch.complex64, True, _common_cdotc)


def cdotu(n, x, incx, y, incy, result):
    _dot(n, x, incx, y, incy, result, torch.complex64, False, _common_cdotu)

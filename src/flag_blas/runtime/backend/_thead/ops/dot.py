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

from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


@libentry()
@triton.jit
def _complex_dot_single(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK * 2)
    xv = tl.load(X + (i // 2) * INCX * 2 + i % 2, i < N * 2, other=0)
    yv = tl.load(Y + (i // 2) * INCY * 2 + i % 2, i < N * 2, other=0)
    xr, xi = tl.split(tl.reshape(xv, (BLOCK, 2)))
    yr, yi = tl.split(tl.reshape(yv, (BLOCK, 2)))
    if CONJ:
        real = xr * yr + xi * yi
        imag = xr * yi - xi * yr
    else:
        real = xr * yr - xi * yi
        imag = xr * yi + xi * yr
    tl.store(OUT, tl.sum(real, 0))
    tl.store(OUT + 1, tl.sum(imag, 0))


@libentry()
@triton.jit
def _zero_result(OUT, COUNT: tl.constexpr):
    i = tl.arange(0, 32)
    tl.store(OUT + i, 0, i < COUNT)


@libentry()
@triton.jit
def _complex_dot_atomic(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    PAIR: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr = 128,
    PIPE_STAGES: tl.constexpr = 1,
    LOOP_UNROLL: tl.constexpr = 1,
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    tiles = tl.cdiv(N, BLOCK)
    if CHUNK:
        chunk = tl.cdiv(tiles, grid) | 1
        begin, end, step = pid * chunk, tl.minimum((pid + 1) * chunk, tiles), 1
    else:
        begin, end, step = pid, tiles, grid
    lane = tl.arange(0, BLOCK)
    re = tl.full((BLOCK,), 0, OUT.dtype.element_ty)
    im = tl.full((BLOCK,), 0, OUT.dtype.element_ty)
    for tile in tl.range(
        begin, end, step, num_stages=PIPE_STAGES, loop_unroll_factor=LOOP_UNROLL
    ):
        i = tile * BLOCK + lane
        if PAIR:
            j = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            xv = tl.load(X + (j // 2) * IX * 2 + j % 2, j < 2 * N, other=0)
            yv = tl.load(Y + (j // 2) * IY * 2 + j % 2, j < 2 * N, other=0)
            xr, xi = tl.split(tl.reshape(xv, (BLOCK, 2)))
            yr, yi = tl.split(tl.reshape(yv, (BLOCK, 2)))
        else:
            xr = tl.load(X + i * IX * 2, i < N, other=0)
            xi = tl.load(X + i * IX * 2 + 1, i < N, other=0)
            yr = tl.load(Y + i * IY * 2, i < N, other=0)
            yi = tl.load(Y + i * IY * 2 + 1, i < N, other=0)
        if CONJ:
            re += xr * yr + xi * yi
            im += xr * yi - xi * yr
        else:
            re += xr * yr - xi * yi
            im += xr * yi + xi * yr
    tl.atomic_add(OUT, tl.sum(re, 0), sem="relaxed")
    tl.atomic_add(OUT + 1, tl.sum(im, 0), sem="relaxed")


_complex_dot_strided = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("thead_l1_cdot_strided"),
        key=["N", "IX", "IY", "CONJ"],
        reset_to_zero=["O"],
        warmup=20,
        rep=50,
    )(_complex_dot_atomic.jit_function)
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_l1_zdot_pipelined"),
    key=["N", "IX", "IY", "CONJ"],
    reset_to_zero=["O"],
    warmup=20,
    rep=50,
)
@triton.jit
def _complex_zdot_pipelined(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    PAIR: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr = 128,
    PIPE_STAGES: tl.constexpr = 1,
    LOOP_UNROLL: tl.constexpr = 1,
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    tiles = tl.cdiv(N, BLOCK)
    if CHUNK:
        chunk = tl.cdiv(tiles, grid) | 1
        begin, end, step = pid * chunk, tl.minimum((pid + 1) * chunk, tiles), 1
    else:
        begin, end, step = pid, tiles, grid
    lane = tl.arange(0, BLOCK)
    re = tl.full((BLOCK,), 0, OUT.dtype.element_ty)
    im = tl.full((BLOCK,), 0, OUT.dtype.element_ty)
    for tile in tl.range(
        begin, end, step, num_stages=PIPE_STAGES, loop_unroll_factor=LOOP_UNROLL
    ):
        i = tile * BLOCK + lane
        if PAIR:
            j = tile * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            xv = tl.load(X + (j // 2) * IX * 2 + j % 2, j < 2 * N, other=0)
            yv = tl.load(Y + (j // 2) * IY * 2 + j % 2, j < 2 * N, other=0)
            xr, xi = tl.split(tl.reshape(xv, (BLOCK, 2)))
            yr, yi = tl.split(tl.reshape(yv, (BLOCK, 2)))
        else:
            xr = tl.load(X + i * IX * 2, i < N, other=0)
            xi = tl.load(X + i * IX * 2 + 1, i < N, other=0)
            yr = tl.load(Y + i * IY * 2, i < N, other=0)
            yi = tl.load(Y + i * IY * 2 + 1, i < N, other=0)
        if CONJ:
            re += xr * yr + xi * yi
            im += xr * yi - xi * yr
        else:
            re += xr * yr - xi * yi
            im += xr * yi + xi * yr
    tl.atomic_add(OUT, tl.sum(re, 0), sem="relaxed")
    tl.atomic_add(OUT + 1, tl.sum(im, 0), sem="relaxed")


_common_dotu = import_module("flag_blas.ops.level1.dotu")
_common_dotc = import_module("flag_blas.ops.level1.dotc")


def _run_dot(n, x, incx, y, incy, result, conj):
    double = x.dtype == torch.complex128
    atomic = 1048576 <= n < 8388608 if double else n >= 1048576
    if (
        (n > 8192 and not atomic)
        or (double and not atomic)
        or not x.is_contiguous()
        or not y.is_contiguous()
    ):
        common = _common_dotc if conj else _common_dotu
        impl = common._dotc_impl if conj else common._dotu_impl
        return impl(n, x, incx, y, incy, result)
    if n <= 0:
        result.zero_()
        return
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert result.numel() == 1, "result must have exactly one element"
    assert (
        x.device == y.device == result.device
    ), "all tensors must be on the same device"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    with torch_device_fn.device(x.device):
        if atomic:
            contiguous = incx == 1 and incy == 1
            block = 256 if double and not contiguous else 1024
            chunk = n < 8388608
            if not chunk:
                block = 4096 if contiguous else 1024
            raw = torch.view_as_real(result)
            _zero_result[(1,)](raw, 2, num_warps=1)
            if double and not contiguous:
                grid = lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), meta["CAP"]),)
                _complex_zdot_pipelined[grid](
                    torch.view_as_real(x),
                    torch.view_as_real(y),
                    raw,
                    n,
                    incx,
                    incy,
                    conj,
                    enable_fp_fusion=False,
                )
                return
            if not double and not contiguous and n < 8388608:
                grid = lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), meta["CAP"]),)
                _complex_dot_strided[grid](
                    torch.view_as_real(x),
                    torch.view_as_real(y),
                    raw,
                    n,
                    incx,
                    incy,
                    conj,
                    enable_fp_fusion=False,
                )
                return
            _complex_dot_atomic[(min(triton.cdiv(n, block), 128),)](
                torch.view_as_real(x),
                torch.view_as_real(y),
                raw,
                n,
                incx,
                incy,
                conj,
                block,
                contiguous,
                chunk,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return
        block = triton.next_power_of_2(n)
        warps = 1 if n <= 128 else (8 if block >= 8192 else 4)
        _complex_dot_single[(1,)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            torch.view_as_real(result),
            n,
            incx,
            incy,
            conj,
            block,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )


def cdotu(n, x, incx, y, incy, result):
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    assert result.dtype == torch.complex64, "result must be complex64"
    _run_dot(n, x, incx, y, incy, result, False)


def cdotc(n, x, incx, y, incy, result):
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    assert result.dtype == torch.complex64, "result must be complex64"
    _run_dot(n, x, incx, y, incy, result, True)


def zdotu(n, x, incx, y, incy, result):
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    assert result.dtype == torch.complex128, "result must be complex128"
    _run_dot(n, x, incx, y, incy, result, False)


def zdotc(n, x, incx, y, incy, result):
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    assert result.dtype == torch.complex128, "result must be complex128"
    _run_dot(n, x, incx, y, incy, result, True)

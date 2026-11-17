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
from flag_blas.utils import libentry

_common = import_module("flag_blas.ops.level1.dot")


@triton.jit
def _dot_terms(
    X,
    Y,
    index,
    mask,
    IX: tl.constexpr,
    IY: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    PAIRED: tl.constexpr,
):
    if COMPLEX:
        if PAIRED:
            component = tl.arange(0, 2)
            x_pair = tl.load(
                X + index[:, None] * (2 * IX) + component[None, :],
                mask[:, None],
                other=0.0,
            )
            y_pair = tl.load(
                Y + index[:, None] * (2 * IY) + component[None, :],
                mask[:, None],
                other=0.0,
            )
            xr, xi = tl.split(x_pair)
            yr, yi = tl.split(y_pair)
        else:
            xr = tl.load(X + 2 * index * IX, mask, other=0.0)
            xi = tl.load(X + 2 * index * IX + 1, mask, other=0.0)
            yr = tl.load(Y + 2 * index * IY, mask, other=0.0)
            yi = tl.load(Y + 2 * index * IY + 1, mask, other=0.0)
        if CONJ:
            return xr * yr + xi * yi, xr * yi - xi * yr
        else:
            return xr * yr - xi * yi, xr * yi + xi * yr
    else:
        x = tl.load(X + index * IX, mask, other=0.0)
        y = tl.load(Y + index * IY, mask, other=0.0)
        return x * y, tl.full(index.shape, 0.0, X.dtype.element_ty)


@libentry()
@triton.jit
def _dot_v2_kernel(
    X,
    Y,
    OUT,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    ATOMIC: tl.constexpr,
    PERSISTENT: tl.constexpr,
    WIDE: tl.constexpr,
    PAIRED: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    start = tl.program_id(0) * BLOCK
    if PERSISTENT:
        real = tl.zeros((BLOCK,), X.dtype.element_ty)
        imag = tl.zeros((BLOCK,), X.dtype.element_ty)
        step = tl.num_programs(0) * BLOCK
        while start < N:
            index = start + lane
            if WIDE:
                index = index.to(tl.int64)
            rv, iv = _dot_terms(X, Y, index, index < N, IX, IY, COMPLEX, CONJ, PAIRED)
            real += rv
            imag += iv
            start += step
    else:
        index = start + lane
        if WIDE:
            index = index.to(tl.int64)
        real, imag = _dot_terms(X, Y, index, index < N, IX, IY, COMPLEX, CONJ, PAIRED)
    if ATOMIC:
        tl.atomic_add(OUT, tl.sum(real, 0), sem="relaxed")
        if COMPLEX:
            tl.atomic_add(OUT + 1, tl.sum(imag, 0), sem="relaxed")
    else:
        tl.store(OUT, tl.sum(real, 0))
        if COMPLEX:
            tl.store(OUT + 1, tl.sum(imag, 0))


def _launch_small_dot(
    n,
    x,
    incx,
    y,
    incy,
    result,
    conj=False,
    block=512,
    warps=4,
    atomic=True,
    cap=256,
    paired=True,
):
    if n <= 0:
        result.zero_()
        return
    is_complex = x.is_complex()
    count = min(triton.cdiv(n, block), cap) if atomic else 1
    if not atomic:
        block = triton.next_power_of_2(n)
    output = torch.view_as_real(result) if is_complex else result
    wide = n * max(incx, incy) * (2 if is_complex else 1) > 2147483647
    with torch_device_fn.device(x.device):
        if count > 1:
            _dot_zero_kernel[(1,)](output, is_complex, num_warps=1)
        _dot_v2_kernel[(count,)](
            torch.view_as_real(x) if is_complex else x,
            torch.view_as_real(y) if is_complex else y,
            output,
            n,
            incx,
            incy,
            is_complex,
            conj,
            block,
            count > 1,
            n > count * block,
            wide,
            paired,
            num_warps=warps,
            num_stages=1,
        )


@libentry()
@triton.jit
def _dot_zero_kernel(OUT, COMPLEX: tl.constexpr):
    tl.store(OUT, 0.0)
    if COMPLEX:
        tl.store(OUT + 1, 0.0)


@triton.jit
def _load_pair(X, start, N, INC: tl.constexpr, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    # Flat component loads retain vectorized access for interleaved complex data.
    elements = tl.arange(0, BLOCK * 2)
    logical = start + elements // 2
    if WIDE:
        logical = logical.to(tl.int64)
    if INC == 1:
        base = start.to(tl.int64) if WIDE else start
        pointer = X + base * 2 + elements
    else:
        pointer = X + logical * (2 * INC) + elements % 2
    values = tl.load(pointer, logical < N, other=0.0)
    return tl.split(tl.reshape(values, (BLOCK, 2)))


@libentry()
@triton.jit
def _complex_dot_kernel(
    X,
    Y,
    OUT,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    PERSISTENT: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK
    if PERSISTENT:
        step = tl.num_programs(0) * BLOCK
        # Independent products avoid extra FP64 operations inside the loop.
        rr = tl.zeros((BLOCK,), X.dtype.element_ty)
        ii = tl.zeros((BLOCK,), X.dtype.element_ty)
        ri = tl.zeros((BLOCK,), X.dtype.element_ty)
        ir = tl.zeros((BLOCK,), X.dtype.element_ty)
        while start < N:
            xr, xi = _load_pair(X, start, N, IX, BLOCK, WIDE)
            yr, yi = _load_pair(Y, start, N, IY, BLOCK, WIDE)
            rr = tl.fma(xr, yr, rr)
            ii = tl.fma(xi, yi, ii)
            ri = tl.fma(xr, yi, ri)
            ir = tl.fma(xi, yr, ir)
            start += step
    else:
        xr, xi = _load_pair(X, start, N, IX, BLOCK, WIDE)
        yr, yi = _load_pair(Y, start, N, IY, BLOCK, WIDE)
        rr = xr * yr
        ii = xi * yi
        ri = xr * yi
        ir = xi * yr
    if CONJ:
        real = tl.sum(rr + ii, 0)
        imag = tl.sum(ri - ir, 0)
    else:
        real = tl.sum(rr - ii, 0)
        imag = tl.sum(ri + ir, 0)
    if ATOMIC:
        tl.atomic_add(OUT, real, sem="relaxed")
        tl.atomic_add(OUT + 1, imag, sem="relaxed")
    else:
        tl.store(OUT, real)
        tl.store(OUT + 1, imag)


def _launch_complex_dot(
    n, x, incx, y, incy, result, conj=True, block=512, warps=8, cap=256
):
    if n <= 0:
        result.zero_()
        return
    count = min(triton.cdiv(n, block), cap)
    output = torch.view_as_real(result)
    wide = n * max(incx, incy) * 2 > 2147483647
    with torch_device_fn.device(x.device):
        if count > 1:
            _dot_zero_kernel[(1,)](output, True, num_warps=1)
        _complex_dot_kernel[(count,)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            output,
            n,
            incx,
            incy,
            conj,
            block,
            wide,
            n > count * block,
            count > 1,
            num_warps=warps,
            num_stages=1,
        )


@libentry()
@triton.jit
def _complex_dot_component_kernel(
    X,
    Y,
    OUT,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    part = tl.program_id(0)
    xr, xi = _load_pair(X, tl.full((), 0, tl.int32), N, IX, BLOCK, WIDE)
    yr, yi = _load_pair(Y, tl.full((), 0, tl.int32), N, IY, BLOCK, WIDE)
    if part == 0:
        value = xr * yr + xi * yi if CONJ else xr * yr - xi * yi
    else:
        value = xr * yi - xi * yr if CONJ else xr * yi + xi * yr
    tl.store(OUT + part, tl.sum(value, 0))


def _launch_dot_components(n, x, ix, y, iy, out, conj=True, warps=16):
    with torch_device_fn.device(x.device):
        _complex_dot_component_kernel[(2,)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            torch.view_as_real(out),
            n,
            ix,
            iy,
            conj,
            triton.next_power_of_2(n),
            n * max(ix, iy) * 2 > 2147483647,
            num_warps=warps,
            num_stages=1,
        )


def _eligible(n, x, incx, y, incy, result, dtype):
    return (
        n > 256
        and incx > 0
        and incy > 0
        and x.dtype == y.dtype == result.dtype == dtype
        and x.dim() == y.dim() == 1
        and result.numel() == 1
        and x.device == y.device == result.device
        and x.numel() >= 1 + (n - 1) * incx
        and y.numel() >= 1 + (n - 1) * incy
    )


def _try_complex_dot(n, x, incx, y, incy, result, dtype, conj):
    if not _eligible(n, x, incx, y, incy, result, dtype):
        return False
    if n <= 2048:
        _launch_dot_components(n, x, incx, y, incy, result, conj, 16)
    elif n <= 4096 or (dtype == torch.complex64 and n <= 8192):
        _launch_complex_dot(n, x, incx, y, incy, result, conj, 512, 8, 256)
    elif dtype == torch.complex64 and n >= 1048576 and (incx != 1 or incy != 1):
        _launch_complex_dot(n, x, incx, y, incy, result, conj, 512, 8, 480)
    elif dtype == torch.complex128 and n > 131072:
        _launch_complex_dot(n, x, incx, y, incy, result, conj, 512, 8, 256)
    else:
        return False
    return True


def _run_real_dot(n, x, incx, y, incy, result, dtype):
    if (
        dtype == torch.float32
        and n >= 16777216
        and incx == incy == 2
        and _eligible(n, x, incx, y, incy, result, dtype)
    ):
        _launch_small_dot(n, x, incx, y, incy, result, False, 2048, 8, True, 480)
        return
    if n <= 131072 and _eligible(n, x, incx, y, incy, result, dtype):
        direct = n <= (2048 if dtype == torch.float32 else 1024)
        _launch_small_dot(
            n,
            x,
            incx,
            y,
            incy,
            result,
            False,
            256 if dtype == torch.float64 and n <= 4096 else 512,
            16 if direct else 4 if dtype == torch.float64 and n <= 4096 else 8,
            not direct,
        )

        return
    original = _common.sdot if dtype == torch.float32 else _common.ddot
    return original(n, x, incx, y, incy, result)


def sdot(n, x, incx, y, incy, result):
    return _run_real_dot(n, x, incx, y, incy, result, torch.float32)


def ddot(n, x, incx, y, incy, result):
    return _run_real_dot(n, x, incx, y, incy, result, torch.float64)

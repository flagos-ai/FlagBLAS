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

_UPDATE_CONFIGS = {
    name: runtime.get_tuned_config("thead_l1_update_" + name)[0]
    for name in (
        "contiguous",
        "strided",
        "strided_small",
        "strided_double",
        "large",
        "scal",
        "scal_wide",
    )
}


@libentry()
@triton.jit
def _complex_update(
    X,
    Y,
    AR: tl.int64,
    AI: tl.int64,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    tiles = tl.cdiv(N * 2, BLOCK)
    if CHUNK:
        chunk = tl.cdiv(tiles, grid) | 1
        start, end, step = pid * chunk, tl.minimum((pid + 1) * chunk, tiles), 1
    else:
        start, end, step = pid, tiles, grid
    ar = AR.to(tl.float64, bitcast=True).to(X.dtype.element_ty)
    ai = AI.to(tl.float64, bitcast=True).to(X.dtype.element_ty)
    lane = tl.arange(0, BLOCK)
    for tile in range(start, end, step):
        j = tile * BLOCK + lane
        xv = tl.load(X + (j // 2) * IX * 2 + j % 2, j < N * 2, other=0)
        if MODE == 2:
            result = ar * xv
        else:
            re, im = tl.split(tl.reshape(xv, (BLOCK // 2, 2)))
            cross = tl.reshape(tl.join(-im, re), (BLOCK,))
            result = ar * xv + ai * cross
        if MODE == 0:
            result += tl.load(Y + (j // 2) * IY * 2 + j % 2, j < N * 2, other=0)
        tl.store(Y + (j // 2) * IY * 2 + j % 2, result, j < N * 2)


@libentry()
@triton.jit
def _complex_update_paired(
    X,
    Y,
    AR: tl.int64,
    AI: tl.int64,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    LOOP_UNROLL: tl.constexpr,
    REGISTER_PREFETCH: tl.constexpr,
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    tiles = tl.cdiv(N, BLOCK)
    chunk = tl.cdiv(tiles, grid) | 1
    begin, end = pid * chunk, tl.minimum((pid + 1) * chunk, tiles)
    lane = tl.arange(0, BLOCK)
    comp = tl.arange(0, 2)
    ar = AR.to(tl.float64, bitcast=True).to(X.dtype.element_ty)
    ai = AI.to(tl.float64, bitcast=True).to(X.dtype.element_ty)
    if REGISTER_PREFETCH:
        first = begin * BLOCK + lane
        xv = tl.load(
            X + first[:, None] * IX * 2 + comp[None, :],
            first[:, None] < N,
            other=0,
        )
    for tile in tl.range(
        begin, end, num_stages=PIPE_STAGES, loop_unroll_factor=LOOP_UNROLL
    ):
        i = tile * BLOCK + lane
        if REGISTER_PREFETCH:
            nxt = (tile + 1) * BLOCK + lane
            next_x = tl.load(
                X + nxt[:, None] * IX * 2 + comp[None, :],
                (nxt[:, None] < N) & (tile + 1 < end),
                other=0,
            )
        else:
            xv = tl.load(
                X + i[:, None] * IX * 2 + comp[None, :],
                i[:, None] < N,
                other=0,
            )
        re, im = tl.split(xv)
        updated = ar * xv + ai * tl.join(-im, re)
        yp = Y + i[:, None] * IY * 2 + comp[None, :]
        updated += tl.load(yp, i[:, None] < N, other=0)
        tl.store(yp, updated, i[:, None] < N)
        if REGISTER_PREFETCH:
            xv = next_x


_PAIRED_CONFIGS = {
    name: runtime.get_tuned_config("thead_l1_update_paired_" + name)[0]
    for name in ("strided", "strided_equal", "double", "double_small")
}


_common_axpy = import_module("flag_blas.ops.level1.axpy")


def _launch_complex_update(n, alpha, x, incx, y, incy, mode, config):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    alpha = complex(alpha)
    real = struct.unpack("<q", struct.pack("<d", alpha.real))[0]
    imag = struct.unpack("<q", struct.pack("<d", alpha.imag))[0]
    block = config.kwargs["BLOCK"]
    cap = config.kwargs["CAP"]
    with torch_device_fn.device(x.device):
        _complex_update[(min(triton.cdiv(n * 2, block), cap),)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            real,
            imag,
            n,
            incx,
            incy,
            mode,
            block,
            True,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )


def _launch_paired_axpy(n, alpha, x, incx, y, incy, config):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    alpha = complex(alpha)
    real = struct.unpack("<q", struct.pack("<d", alpha.real))[0]
    imag = struct.unpack("<q", struct.pack("<d", alpha.imag))[0]
    block = config.kwargs["BLOCK"]
    with torch_device_fn.device(x.device):
        grid = (min(triton.cdiv(n, block), config.kwargs["CAP"]),)
        _complex_update_paired[grid](
            torch.view_as_real(x),
            torch.view_as_real(y),
            real,
            imag,
            n,
            incx,
            incy,
            block,
            config.num_stages,
            config.kwargs["LOOP_UNROLL"],
            config.kwargs["REGISTER_PREFETCH"],
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )


def _axpy(n, alpha, x, incx, y, incy, double):
    large = not double and 8388608 <= n < 16777216 and incx == incy == 1
    use = 1048576 <= n < 8388608 or large
    paired_double = double and 1048576 <= n < 8388608 and incx == incy and incx > 1
    if (
        not use
        or (double and incx == incy and not paired_double)
        or not x.is_contiguous()
        or not y.is_contiguous()
    ):
        return getattr(_common_axpy, "zaxpy" if double else "caxpy")(
            n, alpha, x, incx, y, incy
        )
    assert x.device == y.device, "x and y must be on the same device"
    assert x.dim() == 1 and y.dim() == 1, "x and y must be 1-dimensional"
    assert incx > 0 and incy > 0, "incx and incy must be positive"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    if paired_double:
        config = _PAIRED_CONFIGS["double_small" if n < 2097152 else "double"]
        _launch_paired_axpy(n, alpha, x, incx, y, incy, config)
        return
    if not double and 2097152 <= n < 8388608 and (incx != 1 or incy != 1):
        config = _PAIRED_CONFIGS["strided_equal" if incx == incy else "strided"]
        _launch_paired_axpy(n, alpha, x, incx, y, incy, config)
        return
    if large:
        kind = "large"
    elif incx == incy == 1:
        kind = "contiguous"
    elif double:
        kind = "strided_double"
    else:
        kind = "strided_small" if n < 2097152 else "strided"
    _launch_complex_update(n, alpha, x, incx, y, incy, 0, _UPDATE_CONFIGS[kind])


def caxpy(n, alpha, x, incx, y, incy):
    assert x.dtype == y.dtype == torch.complex64, "x and y must be complex64"
    _axpy(n, alpha, x, incx, y, incy, False)


def zaxpy(n, alpha, x, incx, y, incy):
    assert x.dtype == y.dtype == torch.complex128, "x and y must be complex128"
    _axpy(n, alpha, x, incx, y, incy, True)

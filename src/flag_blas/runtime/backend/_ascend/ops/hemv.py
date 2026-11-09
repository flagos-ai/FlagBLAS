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

"""Row-major complex64 HEMV kernels for Ascend, without a global workspace.

Small matrices use a row-owned reduction. Larger matrices read the stored
triangle once and atomically accumulate its two Hermitian contributions.
The imaginary part of the diagonal is ignored before any multiplication.
"""
from typing import Union

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as _hemv_ext

from flag_blas.ops.level2.hemv import _check_common, _complex_scalars, _strided_y
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .hpmv import (
    _FAST_LAUNCH_SUPPORTED,
    _HOOK_CHAIN_TYPE,
    _NPU_LAUNCHER,
    _current_device,
    _current_raw_stream,
    _DevicePointer,
)
from .hpmv import _launch as _launch_fused

ScalarType = Union[float, int, complex, torch.Tensor]
_HEMV_KERNEL_CACHE = {}


@libentry()
@triton.jit
def _chemv_fused(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    # One program owns each output strip. For the reflected half, read the
    # stored tile in row-major order and conjugate its contribution.
    rb = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    floats = tl.arange(0, 2 * BLOCK)
    parity = floats & 1
    sign = tl.where(parity == 0, 1.0, -1.0)
    acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
    for cb in range(triton.cdiv(N, BLOCK)):
        direct = rb >= cb if UPLO == 0 else rb <= cb
        sr = tl.where(direct, rb, cb)
        sc = tl.where(direct, cb, rb)
        rows = sr * BLOCK + lanes
        cols = sc * BLOCK + (floats >> 1)
        triangle = (
            cols[None, :] <= rows[:, None]
            if UPLO == 0
            else cols[None, :] >= rows[:, None]
        )
        av = tl.load(
            A + rows[:, None] * (2 * LDA) + sc * BLOCK * 2 + floats[None, :],
            (rows[:, None] < N) & (cols[None, :] < N) & triangle,
            0.0,
        )
        diagonal = rows[:, None] == cols[None, :]
        av = tl.where(diagonal & (parity[None, :] != 0), 0.0, av)
        xoff = (
            cb * BLOCK * 2 + floats
            if INCX == 1
            else (cb * BLOCK + (floats >> 1)) * (2 * INCX) + parity
        )
        xv = tl.load(X + xoff, cb * BLOCK * 2 + floats < N * 2, 0.0)
        if direct:
            real = tl.sum(av * (xv * sign)[None, :], 1)
            imag = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
            acc += tl.where(
                parity == 0,
                tl.gather(real, floats >> 1, 0),
                tl.gather(imag, floats >> 1, 0),
            )
        if (not direct) or rb == cb:
            # The real diagonal was included in the direct contribution.
            mirror_av = tl.where(diagonal, 0.0, av)
            xr = tl.gather(xv, lanes * 2, 0)
            xi = tl.gather(xv, lanes * 2 + 1, 0)
            mirror_r = tl.sum(mirror_av * xr[:, None], 0)
            mirror_i = tl.sum(mirror_av * xi[:, None], 0)
            mirror = mirror_r * sign + tl.gather(mirror_i, floats ^ 1, 0)
            acc += mirror
    result = AR * acc + tl.where(parity == 0, -AI, AI) * tl.gather(acc, floats ^ 1, 0)
    yoff = (
        rb * BLOCK * 2 + floats
        if INCY == 1
        else (rb * BLOCK + (floats >> 1)) * (2 * INCY) + parity
    )
    valid = rb * BLOCK * 2 + floats < N * 2
    if not BETA_ZERO:
        yv = tl.load(Y + yoff, valid, 0.0)
        result += BR * yv + tl.where(parity == 0, -BI, BI) * tl.gather(
            yv, floats ^ 1, 0
        )
    tl.store(Y + yoff, result, valid)


@triton.jit
def _hemv_rows_kernel(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    REDUCE: tl.constexpr,
):
    first = tl.program_id(0) * BM
    rows = first + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    if (
        N * LDA * 2 >= 2147483648
        or N * IX * 2 >= 2147483648
        or N * IY * 2 >= 2147483648
    ):
        rows = rows.to(tl.int64)
        ks = ks.to(tl.int64)
    lane = tl.arange(0, 2)
    if REDUCE:
        dr = tl.full((BM,), 0, tl.float32)
        di = tl.full((BM,), 0, tl.float32)
    else:
        dr = tl.full((BM, BK), 0, tl.float32)
        di = tl.full((BM, BK), 0, tl.float32)
    if UPLO == 0:
        (db, de) = (0, tl.minimum(first + BM, N))
        (mb, me) = (first, N)
    else:
        (db, de) = (first, N)
        (mb, me) = (0, tl.minimum(first + BM, N))
    for start in range(db, de, BK):
        cols = start + ks
        xp = tl.load(X + cols[:, None] * IX * 2 + lane[None, :], cols[:, None] < N, 0)
        (xr, xi) = tl.split(xp)
        ap = tl.load(
            A
            + (rows[:, None] * LDA + cols[None, :])[:, :, None] * 2
            + lane[None, None, :],
            ((rows[:, None] < N) & (cols[None, :] < N))[:, :, None],
            0,
        )
        (ar, ai) = tl.split(ap)
        if UPLO == 0:
            valid = rows[:, None] >= cols[None, :]
        else:
            valid = rows[:, None] <= cols[None, :]
        ar = tl.where(valid, ar, 0)
        ai = tl.where(valid, ai, 0)
        ai = tl.where(rows[:, None] == cols[None, :], 0.0, ai)
        vr = ar * xr[None, :] - ai * xi[None, :]
        vi = ar * xi[None, :] + ai * xr[None, :]
        if REDUCE:
            dr += tl.sum(vr, 1)
            di += tl.sum(vi, 1)
        else:
            dr += vr
            di += vi
    # Release direct accumulators before allocating the mirrored tile state.
    if REDUCE:
        sr, si = dr, di
    else:
        sr, si = tl.sum(dr, 1), tl.sum(di, 1)
    if REDUCE:
        mr = tl.full((BM,), 0, tl.float32)
        mi = tl.full((BM,), 0, tl.float32)
    else:
        mr = tl.full((BK, BM), 0, tl.float32)
        mi = tl.full((BK, BM), 0, tl.float32)
    for start in range(mb, me, BK):
        cols = start + ks
        xp = tl.load(X + cols[:, None] * IX * 2 + lane[None, :], cols[:, None] < N, 0)
        (xr, xi) = tl.split(xp)
        ap = tl.load(
            A
            + (cols[:, None] * LDA + rows[None, :])[:, :, None] * 2
            + lane[None, None, :],
            ((cols[:, None] < N) & (rows[None, :] < N))[:, :, None],
            0,
        )
        (ar, ai) = tl.split(ap)
        if UPLO == 0:
            valid = cols[:, None] > rows[None, :]
        else:
            valid = cols[:, None] < rows[None, :]
        ar = tl.where(valid, ar, 0)
        ai = tl.where(valid, ai, 0)
        vr = ar * xr[:, None] + ai * xi[:, None]
        vi = ar * xi[:, None] - ai * xr[:, None]
        if REDUCE:
            mr += tl.sum(vr, 0)
            mi += tl.sum(vi, 0)
        else:
            mr += vr
            mi += vi
    if REDUCE:
        sr += mr
        si += mi
    else:
        sr += tl.sum(mr, 0)
        si += tl.sum(mi, 0)
    yr = AR * sr - AI * si
    yi = AR * si + AI * sr
    if not ZERO:
        yp = tl.load(Y + rows[:, None] * IY * 2 + lane[None, :], rows[:, None] < N, 0)
        (oldr, oldi) = tl.split(yp)
        yr += BR * oldr - BI * oldi
        yi += BR * oldi + BI * oldr
    tl.store(
        Y + rows[:, None] * IY * 2 + lane[None, :], tl.join(yr, yi), rows[:, None] < N
    )


@triton.jit
def _hemv_panel_kernel(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    COL_FIRST: tl.constexpr,
):
    if REVERSE:
        first = (tl.cdiv(N, BM) - 1 - tl.program_id(0)) * BM
    else:
        first = tl.program_id(0) * BM
    rows = first + tl.arange(0, BM)
    k = tl.arange(0, BK)
    if (
        N * LDA * 2 >= 2147483648
        or N * IX * 2 >= 2147483648
        or N * IY * 2 >= 2147483648
    ):
        rows = rows.to(tl.int64)
        k = k.to(tl.int64)
    lane = tl.arange(0, 2)
    xp = tl.load(X + rows[:, None] * IX * 2 + lane[None, :], rows[:, None] < N, 0)
    xrr, xri = tl.split(xp)
    rr = tl.full((BM,), 0, tl.float32)
    ri = tl.full((BM,), 0, tl.float32)
    diag_begin = first // BK * BK
    diag_end = tl.cdiv(first + BM, BK) * BK
    if UPLO == 0:
        begin, end = 0, diag_begin
    else:
        begin, end = diag_end, N
    for start in range(begin, end, BK):
        cols = start + k
        xc = tl.load(X + cols[:, None] * IX * 2 + lane[None, :], cols[:, None] < N, 0)
        xcr, xci = tl.split(xc)
        off = (rows[:, None] * LDA + cols[None, :]) * 2
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        ap = tl.load(A + off[:, :, None] + lane[None, None, :], mask[:, :, None], 0)
        ar, ai = tl.split(ap)
        if COL_FIRST:
            cr = tl.sum(ar * xrr[:, None] + ai * xri[:, None], 0)
            ci = tl.sum(ar * xri[:, None] - ai * xrr[:, None], 0)
            yr = AR * cr - AI * ci
            yi = AR * ci + AI * cr
            tl.atomic_add(
                Y + cols[:, None] * IY * 2 + lane[None, :],
                tl.join(yr, yi),
                cols[:, None] < N,
                sem="relaxed",
            )
        rr += tl.sum(ar * xcr[None, :] - ai * xci[None, :], 1)
        ri += tl.sum(ar * xci[None, :] + ai * xcr[None, :], 1)
        if not COL_FIRST:
            cr = tl.sum(ar * xrr[:, None] + ai * xri[:, None], 0)
            ci = tl.sum(ar * xri[:, None] - ai * xrr[:, None], 0)
            yr = AR * cr - AI * ci
            yi = AR * ci + AI * cr
            tl.atomic_add(
                Y + cols[:, None] * IY * 2 + lane[None, :],
                tl.join(yr, yi),
                cols[:, None] < N,
                sem="relaxed",
            )
    for start in range(diag_begin, diag_end, BK):
        cols = start + k
        xc = tl.load(X + cols[:, None] * IX * 2 + lane[None, :], cols[:, None] < N, 0)
        xcr, xci = tl.split(xc)
        off = (rows[:, None] * LDA + cols[None, :]) * 2
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        ap = tl.load(A + off[:, :, None] + lane[None, None, :], mask[:, :, None], 0)
        ar, ai = tl.split(ap)
        if UPLO == 0:
            valid = rows[:, None] >= cols[None, :]
        else:
            valid = rows[:, None] <= cols[None, :]
        ar, ai = tl.where(valid, ar, 0), tl.where(valid, ai, 0)
        ai = tl.where(rows[:, None] == cols[None, :], 0.0, ai)
        rr += tl.sum(ar * xcr[None, :] - ai * xci[None, :], 1)
        ri += tl.sum(ar * xci[None, :] + ai * xcr[None, :], 1)
        ar = tl.where(rows[:, None] != cols[None, :], ar, 0)
        ai = tl.where(rows[:, None] != cols[None, :], ai, 0)
        cr = tl.sum(ar * xrr[:, None] + ai * xri[:, None], 0)
        ci = tl.sum(ar * xri[:, None] - ai * xrr[:, None], 0)
        yr = AR * cr - AI * ci
        yi = AR * ci + AI * cr
        tl.atomic_add(
            Y + cols[:, None] * IY * 2 + lane[None, :],
            tl.join(yr, yi),
            cols[:, None] < N,
            sem="relaxed",
        )
    yr = AR * rr - AI * ri
    yi = AR * ri + AI * rr
    tl.atomic_add(
        Y + rows[:, None] * IY * 2 + lane[None, :],
        tl.join(yr, yi),
        rows[:, None] < N,
        sem="relaxed",
    )


@triton.jit
def _hemv_cached_panel_kernel(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    COL_FIRST: tl.constexpr,
):
    if REVERSE:
        first = (tl.cdiv(N, BM) - 1 - tl.program_id(0)) * BM
    else:
        first = tl.program_id(0) * BM
    rows = first + tl.arange(0, BM)
    k = tl.arange(0, BK)
    if (
        N * LDA * 2 >= 2147483648
        or N * IX * 2 >= 2147483648
        or N * IY * 2 >= 2147483648
    ):
        rows = rows.to(tl.int64)
        k = k.to(tl.int64)
    lane = tl.arange(0, 2)
    # Keep packed X in on-chip UB; dispatch bounds this storage to 64 KiB.
    xn = tl.arange(0, 2 * triton.next_power_of_2(N))
    fullx = tl.load(X + xn, xn < N * 2, 0)
    xrows = tl.reshape(
        _hemv_ext.extract_slice(fullx, [first * 2], [BM * 2], [1]), (BM, 2)
    )
    xrr, xri = tl.split(xrows)
    rr = tl.full((BM,), 0, tl.float32)
    ri = tl.full((BM,), 0, tl.float32)
    diag_begin = first // BK * BK
    diag_end = tl.cdiv(first + BM, BK) * BK
    if UPLO == 0:
        begin, end = 0, diag_begin
    else:
        begin, end = diag_end, N
    count = tl.cdiv(end - begin, BK)
    # Stagger atomic destinations while visiting every off-diagonal tile once.
    shift = (tl.program_id(0) * 7) % tl.maximum(count, 1)
    for step in range(count):
        start = begin + ((step + shift) % tl.maximum(count, 1)) * BK
        cols = start + k
        xc = tl.reshape(
            _hemv_ext.extract_slice(fullx, [start * 2], [BK * 2], [1]), (BK, 2)
        )
        xcr, xci = tl.split(xc)
        off = (rows[:, None] * LDA + cols[None, :]) * 2
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        ap = tl.load(A + off[:, :, None] + lane[None, None, :], mask[:, :, None], 0)
        ar, ai = tl.split(ap)
        if COL_FIRST:
            cr = tl.sum(ar * xrr[:, None] + ai * xri[:, None], 0)
            ci = tl.sum(ar * xri[:, None] - ai * xrr[:, None], 0)
            yr = AR * cr - AI * ci
            yi = AR * ci + AI * cr
            tl.atomic_add(
                Y + cols[:, None] * IY * 2 + lane[None, :],
                tl.join(yr, yi),
                cols[:, None] < N,
                sem="relaxed",
            )
        rr += tl.sum(ar * xcr[None, :] - ai * xci[None, :], 1)
        ri += tl.sum(ar * xci[None, :] + ai * xcr[None, :], 1)
        if not COL_FIRST:
            cr = tl.sum(ar * xrr[:, None] + ai * xri[:, None], 0)
            ci = tl.sum(ar * xri[:, None] - ai * xrr[:, None], 0)
            yr = AR * cr - AI * ci
            yi = AR * ci + AI * cr
            tl.atomic_add(
                Y + cols[:, None] * IY * 2 + lane[None, :],
                tl.join(yr, yi),
                cols[:, None] < N,
                sem="relaxed",
            )
    for start in range(diag_begin, diag_end, BK):
        cols = start + k
        xc = tl.reshape(
            _hemv_ext.extract_slice(fullx, [start * 2], [BK * 2], [1]), (BK, 2)
        )
        xcr, xci = tl.split(xc)
        off = (rows[:, None] * LDA + cols[None, :]) * 2
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        ap = tl.load(A + off[:, :, None] + lane[None, None, :], mask[:, :, None], 0)
        ar, ai = tl.split(ap)
        if UPLO == 0:
            valid = rows[:, None] >= cols[None, :]
        else:
            valid = rows[:, None] <= cols[None, :]
        ar, ai = tl.where(valid, ar, 0), tl.where(valid, ai, 0)
        ai = tl.where(rows[:, None] == cols[None, :], 0.0, ai)
        rr += tl.sum(ar * xcr[None, :] - ai * xci[None, :], 1)
        ri += tl.sum(ar * xci[None, :] + ai * xcr[None, :], 1)
        ar = tl.where(rows[:, None] != cols[None, :], ar, 0)
        ai = tl.where(rows[:, None] != cols[None, :], ai, 0)
        cr = tl.sum(ar * xrr[:, None] + ai * xri[:, None], 0)
        ci = tl.sum(ar * xri[:, None] - ai * xrr[:, None], 0)
        yr = AR * cr - AI * ci
        yi = AR * ci + AI * cr
        tl.atomic_add(
            Y + cols[:, None] * IY * 2 + lane[None, :],
            tl.join(yr, yi),
            cols[:, None] < N,
            sem="relaxed",
        )
    yr = AR * rr - AI * ri
    yi = AR * ri + AI * rr
    tl.atomic_add(
        Y + rows[:, None] * IY * 2 + lane[None, :],
        tl.join(yr, yi),
        rows[:, None] < N,
        sem="relaxed",
    )


@triton.jit
def _hemv_scale_kernel(
    Y,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    if N * IY * 2 >= 2147483648:
        rows = rows.to(tl.int64)
    lanes = tl.arange(0, 2)
    yr = tl.full((BM,), 0, tl.float32)
    yi = tl.full((BM,), 0, tl.float32)
    if not ZERO:
        # Load and store complex pairs together to keep GM transfers contiguous.
        packed = tl.load(
            Y + rows[:, None] * IY * 2 + lanes[None, :], rows[:, None] < N, 0
        )
        oldr, oldi = tl.split(packed)
        yr = BR * oldr - BI * oldi
        yi = BR * oldi + BI * oldr
    tl.store(
        Y + rows[:, None] * IY * 2 + lanes[None, :], tl.join(yr, yi), rows[:, None] < N
    )


def _launch_cached(kernel, grid, tensors, scalars, constants, multibuffer=True):
    key = (
        kernel,
        tensors[0].device.index,
        constants,
        scalars,
        multibuffer,
        tuple(t.data_ptr() % 16 for t in tensors),
    )
    compiled = _HEMV_KERNEL_CACHE.get(key)
    if compiled is None:
        real = tuple(torch.view_as_real(t) for t in tensors)
        options = {}
        if (
            kernel in (_complex_packed_panel_kernel, _hemv_packed_diagonal_kernel)
            and constants[0] >= 4095
        ):
            # Reuse local buffers in the large packed reductions. This does
            # not allocate a global workspace or change the arithmetic.
            options["enable_ubuf_saving"] = True
        compiled = kernel.warmup(
            *real,
            *scalars,
            *constants,
            grid=grid,
            num_warps=1,
            num_stages=1,
            multibuffer=multibuffer,
            **options,
        )
        assert int(getattr(compiled.metadata, "workspace_size", 0)) == 0
        if len(_HEMV_KERNEL_CACHE) >= 512:
            _HEMV_KERNEL_CACHE.clear()
        _HEMV_KERNEL_CACHE[key] = compiled
    run = compiled.run
    knobs = triton.knobs.runtime
    hooks = (knobs.launch_enter_hook, knobs.launch_exit_hook)
    has_hooks = any(
        h is not None and (type(h) is not _HOOK_CHAIN_TYPE or bool(h.calls))
        for h in hooks
    )
    launch_grid = (*grid, *(1 for _ in range(3 - len(grid))))
    if (
        not _FAST_LAUNCH_SUPPORTED
        or type(run) is not _NPU_LAUNCHER
        or has_hooks
        or getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
        or any(
            type(t) is not torch.Tensor or t.is_conj() or t.is_neg() for t in tensors
        )
    ):
        real = tuple(torch.view_as_real(t) for t in tensors)
        compiled[launch_grid](*real, *scalars, *constants)
        return
    ptrs = []
    for t in tensors:
        ptr = _DevicePointer(t.data_ptr())
        ptr.tensor = t
        ptrs.append(ptr)
    run(
        *launch_grid,
        _current_raw_stream(tensors[0].device.index),
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *ptrs,
        *scalars,
        *constants,
    )


@triton.jit
def _hemv_packed_fused_kernel(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    REDUCE: tl.constexpr,
):
    rb = tl.program_id(0)
    ks = tl.arange(0, BM)
    lane = tl.arange(0, 2)
    if (
        N * LDA * 2 >= 2147483648
        or N * IX * 2 >= 2147483648
        or N * IY * 2 >= 2147483648
    ):
        ks = ks.to(tl.int64)
    rr = tl.full((BM,), 0, tl.float32)
    ri = tl.full((BM,), 0, tl.float32)
    for cb in range(triton.cdiv(N, BM)):
        direct = rb >= cb if UPLO == 0 else rb <= cb
        sr = tl.where(direct, rb, cb)
        sc = tl.where(direct, cb, rb)
        rows = sr * BM + ks
        cols = sc * BM + ks
        ap = tl.load(
            A
            + (rows[:, None] * LDA + cols[None, :])[:, :, None] * 2
            + lane[None, None, :],
            ((rows[:, None] < N) & (cols[None, :] < N))[:, :, None],
            0,
        )
        ar, ai = tl.split(ap)
        if rb == cb:
            if UPLO == 0:
                valid = rows[:, None] >= cols[None, :]
            else:
                valid = rows[:, None] <= cols[None, :]
            ar = tl.where(valid, ar, 0)
            ai = tl.where(valid & (rows[:, None] != cols[None, :]), ai, 0)
        xs = cb * BM + ks
        xp = tl.load(X + xs[:, None] * IX * 2 + lane[None, :], xs[:, None] < N, 0)
        xr, xi = tl.split(xp)
        if direct:
            rr += tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
            ri += tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
        if not direct or rb == cb:
            if rb == cb:
                ar = tl.where(rows[:, None] != cols[None, :], ar, 0)
            rr += tl.sum(ar * xr[:, None] + ai * xi[:, None], 0)
            ri += tl.sum(ar * xi[:, None] - ai * xr[:, None], 0)
    rows = rb * BM + ks
    yr = AR * rr - AI * ri
    yi = AR * ri + AI * rr
    if not ZERO:
        yp = tl.load(Y + rows[:, None] * IY * 2 + lane[None, :], rows[:, None] < N, 0)
        oldr, oldi = tl.split(yp)
        yr += BR * oldr - BI * oldi
        yi += BR * oldi + BI * oldr
    tl.store(
        Y + rows[:, None] * IY * 2 + lane[None, :], tl.join(yr, yi), rows[:, None] < N
    )


@triton.jit
def _hemv_triangle_kernel(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    COL_FIRST: tl.constexpr,
):
    side: tl.constexpr = triton.cdiv(N, BM)
    tiles: tl.constexpr = side * (side + 1) // 2
    ks = tl.arange(0, BM)
    lane = tl.arange(0, 2)
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        major = ((tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5).to(tl.int32)
        major = tl.where(major * (major + 1) // 2 > tile, major - 1, major)
        major = tl.where((major + 1) * (major + 2) // 2 <= tile, major + 1, major)
        minor = tile - major * (major + 1) // 2
        if UPLO == 0:
            rb, cb = major, minor
        else:
            rb, cb = minor, major
        rows = rb * BM + ks
        cols = cb * BM + ks
        ap = tl.load(
            A
            + (rows[:, None] * LDA + cols[None, :])[:, :, None] * 2
            + lane[None, None, :],
            ((rows[:, None] < N) & (cols[None, :] < N))[:, :, None],
            0,
        )
        ar, ai = tl.split(ap)
        if rb == cb:
            if UPLO == 0:
                valid = rows[:, None] >= cols[None, :]
            else:
                valid = rows[:, None] <= cols[None, :]
            ar = tl.where(valid, ar, 0)
            ai = tl.where(valid & (rows[:, None] != cols[None, :]), ai, 0)
        xc = tl.load(X + cols[:, None] * IX * 2 + lane[None, :], cols[:, None] < N, 0)
        xr, xi = tl.split(xc)
        rr = tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
        ri = tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
        yr = AR * rr - AI * ri
        yi = AR * ri + AI * rr
        tl.atomic_add(
            Y + rows[:, None] * IY * 2 + lane[None, :],
            tl.join(yr, yi),
            rows[:, None] < N,
            sem="relaxed",
        )
        if rb == cb:
            ar = tl.where(rows[:, None] != cols[None, :], ar, 0)
        xp = tl.load(X + rows[:, None] * IX * 2 + lane[None, :], rows[:, None] < N, 0)
        xr, xi = tl.split(xp)
        cr = tl.sum(ar * xr[:, None] + ai * xi[:, None], 0)
        ci = tl.sum(ar * xi[:, None] - ai * xr[:, None], 0)
        yr = AR * cr - AI * ci
        yi = AR * ci + AI * cr
        tl.atomic_add(
            Y + cols[:, None] * IY * 2 + lane[None, :],
            tl.join(yr, yi),
            cols[:, None] < N,
            sem="relaxed",
        )


def run_chemv(uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy):
    # Use Python grid arithmetic: constexpr dispatch can starve small launches.
    beta_zero = br == 0 and bi == 0
    if 3072 <= n <= 16384 and lda == n and incx == incy == 1:
        return _run_complex_packed(
            uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy, True
        )
    # Preserve the original fused path for tiny and strided small matrices.
    if n <= 512 or ((incx != 1 or incy != 1) and n <= 2048):
        if n <= 192:
            block = 16
        elif n <= 256:
            block = 32 if uplo == 0 else 16
        elif n <= 512:
            block = 32 if uplo == 0 else 64
        else:
            block = 32
        return _launch_fused(
            _chemv_fused,
            ((n + block - 1) // block,),
            (A, x, y),
            (ar, ai, br, bi),
            (n, lda, incx, incy, uplo, block, beta_zero),
        )
    # Bound strided atomic vector tiles to the local-memory budget.
    if incx != 1 or incy != 1:
        kernel, bm, bk, mb = _hemv_panel_kernel, 8, 32, False
    elif n <= 1024:
        block = 32 if n <= 512 or n % 64 else 64
        return _launch_cached(
            _hemv_packed_fused_kernel,
            ((n + block - 1) // block,),
            (A, x, y),
            (ar, ai, br, bi),
            (n, lda, incx, incy, uplo, beta_zero, block, block, False),
            False,
        )
    elif n <= 2048:
        return _launch_cached(
            _hemv_rows_kernel,
            ((n + 63) // 64,),
            (A, x, y),
            (ar, ai, br, bi),
            (n, lda, incx, incy, uplo, beta_zero, 64, 64, False),
            True,
        )
    elif n < 4096:
        kernel, bm, bk, mb = _hemv_triangle_kernel, 64, 64, False
    elif n <= 8192 and n % 128 == 0:
        kernel, bm, bk, mb = _hemv_cached_panel_kernel, 32, 128, False
    elif n < 8192:
        kernel, bm, bk, mb = _hemv_panel_kernel, 16, 256, False
    else:
        kernel, bm, bk, mb = _hemv_panel_kernel, 32, 128, True
    if br != 1 or bi != 0:
        _launch_cached(
            _hemv_scale_kernel,
            ((n + 255) // 256,),
            (y,),
            (br, bi),
            (n, incy, beta_zero, 256),
        )
    if kernel is _hemv_triangle_kernel:
        side = (n + bm - 1) // bm
        grid = (min(40, side * (side + 1) // 2),)
    else:
        grid = ((n + bm - 1) // bm,)
    _launch_cached(
        kernel,
        grid,
        (A, x, y),
        (ar, ai),
        (n, lda, incx, incy, uplo, bm, bk, uplo == 0, uplo == 1),
        mb,
    )


def chemv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return
    if type(alpha) is complex and type(beta) is complex:
        ar, ai, br, bi = alpha.real, alpha.imag, beta.real, beta.imag
    else:
        ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0 and ai == 0:
        y_view = _strided_y(y, n, incy)
        if br == 0 and bi == 0:
            y_view.zero_()
        elif br != 1 or bi != 0:
            y_view.mul_(complex(br, bi))
        return
    if A.device.index != _current_device():
        with torch_device_fn.device(A.device):
            return run_chemv(uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy)
    return run_chemv(uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy)


@triton.jit
def _complex_packed_panel_kernel(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    HERM: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
):
    block = tl.program_id(0)
    first = (tl.cdiv(N, BM) - 1 - block) * BM if REVERSE else block * BM
    rows = first + tl.arange(0, BM)
    f = tl.arange(0, 2 * BK)
    parity = f & 1
    sign = tl.where(parity == 0, 1.0, -1.0)
    rlane = tl.arange(0, 2)
    rowx = tl.load(X + rows[:, None] * 2 + rlane[None, :], rows[:, None] < N, 0.0)
    (xrr, xri) = tl.split(rowx)
    PRESCALE_X: tl.constexpr = not HERM and N >= 4095
    if PRESCALE_X:
        # Reuse alpha*x for every mirror tile of a symmetric row panel.
        scaled_real = AR * xrr - AI * xri
        xri = AR * xri + AI * xrr
        xrr = scaled_real
    rr = tl.full((BM,), 0.0, tl.float32)
    ri = tl.full((BM,), 0.0, tl.float32)
    db = first // BK * BK
    de = tl.cdiv(first + BM, BK) * BK
    (begin, end) = (0, db) if UPLO == 0 else (de, N)
    count = tl.cdiv(end - begin, BK)
    shift = block * 7 % tl.maximum(count, 1)
    for step in range(count):
        start = begin + (step + shift) % tl.maximum(count, 1) * BK
        cols = start + (f >> 1)
        xv = tl.load(X + start * 2 + f, start * 2 + f < N * 2, 0)
        av = tl.load(
            A + rows[:, None] * (2 * LDA) + start * 2 + f[None, :],
            (rows[:, None] < N) & (start * 2 + f[None, :] < N * 2),
            0.0,
        )
        rr += tl.sum(av * (xv * sign)[None, :], 1)
        ri += tl.sum(av * tl.gather(xv, f ^ 1, 0)[None, :], 1)
        m0 = tl.sum(av * xrr[:, None], 0)
        m1 = tl.sum(av * xri[:, None], 0)
        if HERM:
            m = m0 * sign + tl.gather(m1, f ^ 1, 0)
        else:
            m = m0 - sign * tl.gather(m1, f ^ 1, 0)
        if PRESCALE_X:
            out = m
        else:
            out = AR * m + tl.where(parity == 0, -AI, AI) * tl.gather(m, f ^ 1, 0)
        tl.atomic_add(Y + start * 2 + f, out, start * 2 + f < N * 2, sem="relaxed")
    for start in range(db, de, BK):
        cols = start + (f >> 1)
        xv = tl.load(X + start * 2 + f, start * 2 + f < N * 2, 0)
        valid = (
            rows[:, None] >= cols[None, :]
            if UPLO == 0
            else rows[:, None] <= cols[None, :]
        )
        av = tl.load(
            A + rows[:, None] * (2 * LDA) + start * 2 + f[None, :],
            (rows[:, None] < N) & (start * 2 + f[None, :] < N * 2),
            0.0,
        )
        av = tl.where(valid, av, 0.0)
        diag = rows[:, None] == cols[None, :]
        if HERM:
            av = tl.where(diag & (parity[None, :] != 0), 0.0, av)
        rr += tl.sum(av * (xv * sign)[None, :], 1)
        ri += tl.sum(av * tl.gather(xv, f ^ 1, 0)[None, :], 1)
        av = tl.where(diag, 0.0, av)
        m0 = tl.sum(av * xrr[:, None], 0)
        m1 = tl.sum(av * xri[:, None], 0)
        if HERM:
            m = m0 * sign + tl.gather(m1, f ^ 1, 0)
        else:
            m = m0 - sign * tl.gather(m1, f ^ 1, 0)
        if PRESCALE_X:
            out = m
        else:
            out = AR * m + tl.where(parity == 0, -AI, AI) * tl.gather(m, f ^ 1, 0)
        tl.atomic_add(Y + start * 2 + f, out, start * 2 + f < N * 2, sem="relaxed")
    yr = AR * rr - AI * ri
    yi = AR * ri + AI * rr
    lanes = tl.arange(0, 2)
    tl.atomic_add(
        Y + rows[:, None] * IY * 2 + lanes[None, :],
        tl.join(yr, yi),
        rows[:, None] < N,
        sem="relaxed",
    )


@triton.jit
def _hemv_packed_diagonal_kernel(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    HERM: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
):
    block = tl.program_id(0)
    first = (tl.cdiv(N, BM) - 1 - block) * BM if REVERSE else block * BM
    rows = first + tl.arange(0, BM)
    f = tl.arange(0, 2 * BK)
    parity = f & 1
    sign = tl.where(parity == 0, 1.0, -1.0)
    rlane = tl.arange(0, 2)
    rowx = tl.load(X + rows[:, None] * 2 + rlane[None, :], rows[:, None] < N, 0.0)
    (xrr, xri) = tl.split(rowx)
    rr = tl.full((BM,), 0.0, tl.float32)
    ri = tl.full((BM,), 0.0, tl.float32)
    db = first
    de = first + BM
    (begin, end) = (0, db) if UPLO == 0 else (de, N)
    count = tl.cdiv(end - begin, BK)
    shift = block * 7 % tl.maximum(count, 1)
    for step in range(count):
        start = begin + (step + shift) % tl.maximum(count, 1) * BK
        xv = tl.load(X + start * 2 + f, start * 2 + f < end * 2, 0)
        av = tl.load(
            A + rows[:, None] * (2 * LDA) + start * 2 + f[None, :],
            (rows[:, None] < N) & (start * 2 + f[None, :] < end * 2),
            0.0,
        )
        rr += tl.sum(av * (xv * sign)[None, :], 1)
        ri += tl.sum(av * tl.gather(xv, f ^ 1, 0)[None, :], 1)
        m0 = tl.sum(av * xrr[:, None], 0)
        m1 = tl.sum(av * xri[:, None], 0)
        if HERM:
            m = m0 * sign + tl.gather(m1, f ^ 1, 0)
        else:
            m = m0 - sign * tl.gather(m1, f ^ 1, 0)
        out = AR * m + tl.where(parity == 0, -AI, AI) * tl.gather(m, f ^ 1, 0)
        tl.atomic_add(Y + start * 2 + f, out, start * 2 + f < end * 2, sem="relaxed")
    df = tl.arange(0, 2 * BM)
    dp = df & 1
    dsign = tl.where(dp == 0, 1.0, -1.0)
    dc = first + (df >> 1)
    dx = tl.load(X + first * 2 + df, first * 2 + df < N * 2, 0.0)
    da = tl.load(
        A + rows[:, None] * (LDA * 2) + first * 2 + df[None, :],
        (rows[:, None] < N) & (first * 2 + df[None, :] < N * 2),
        0.0,
    )
    valid = rows[:, None] >= dc[None, :] if UPLO == 0 else rows[:, None] <= dc[None, :]
    da = tl.where(valid, da, 0.0)
    diagonal = rows[:, None] == dc[None, :]
    if HERM:
        da = tl.where(diagonal & (dp[None, :] != 0), 0.0, da)
    rr += tl.sum(da * (dx * dsign)[None, :], 1)
    ri += tl.sum(da * tl.gather(dx, df ^ 1, 0)[None, :], 1)
    da = tl.where(diagonal, 0.0, da)
    m0 = tl.sum(da * xrr[:, None], 0)
    m1 = tl.sum(da * xri[:, None], 0)
    if HERM:
        m = m0 * dsign + tl.gather(m1, df ^ 1, 0)
    else:
        m = m0 - dsign * tl.gather(m1, df ^ 1, 0)
    out = AR * m + tl.where(dp == 0, -AI, AI) * tl.gather(m, df ^ 1, 0)
    tl.atomic_add(Y + first * 2 + df, out, first * 2 + df < N * 2, sem="relaxed")
    yr = AR * rr - AI * ri
    yi = AR * ri + AI * rr
    lanes = tl.arange(0, 2)
    tl.atomic_add(
        Y + rows[:, None] * IY * 2 + lanes[None, :],
        tl.join(yr, yi),
        rows[:, None] < N,
        sem="relaxed",
    )


def _run_complex_packed(uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy, hermitian):
    # This path keeps the matrix interleaved and uses contiguous float addresses.
    # Reduce complex pairs before mixing their real and imaginary components.
    if A.device.index != _current_device():
        with torch_device_fn.device(A.device):
            return _run_complex_packed(
                uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy, hermitian
            )
    if br != 1 or bi != 0:
        _launch_cached(
            _hemv_scale_kernel,
            ((n + 255) // 256,),
            (y,),
            (br, bi),
            (n, incy, br == 0 and bi == 0, 256),
        )
    bm, bk = (
        (16, 256)
        if (not hermitian and n >= 4095) or n <= 4096 or n % 128
        else (32, 128)
    )
    kernel = (
        _hemv_packed_diagonal_kernel
        if hermitian and n <= 4096
        else _complex_packed_panel_kernel
    )
    _launch_cached(
        kernel,
        ((n + bm - 1) // bm,),
        (A, x, y),
        (ar, ai),
        (n, lda, incx, incy, uplo, hermitian, bm, bk, uplo == 0),
        False,
    )

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

"""PPU triangular solve with parallel diagonal inverses and ordered row tiles."""

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.trsv import _check_trsv
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@triton.jit
def _dense_offset(row, col, lda, TRANS: tl.constexpr):
    if TRANS:
        return col * lda + row
    return row * lda + col


@libentry()
@triton.jit
def _trsv_inverse_kernel(
    A,
    D,
    State,
    n,
    lda,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    GATHER: tl.constexpr,
):
    if tl.program_id(0) == 0:
        tl.store(State + tl.arange(0, 2), 0)
    offsets = tl.arange(0, BLOCK)
    indices = tl.program_id(0) * BLOCK + offsets
    rows = indices if FORWARD else n - 1 - indices
    valid = indices < n
    addr = _dense_offset(rows[:, None], rows[None, :], lda, TRANS)
    mask = valid[:, None] & valid[None, :] & (offsets[:, None] > offsets[None, :])
    diagonal = _dense_offset(rows, rows, lda, TRANS)
    if COMPLEX:
        mr = tl.load(A + 2 * addr, mask, other=0)
        mi = tl.load(A + 2 * addr + 1, mask, other=0)
        if CONJ:
            mi = -mi
        if not UNIT:
            dr = tl.load(A + 2 * diagonal, valid, other=1)
            di = tl.load(A + 2 * diagonal + 1, valid, other=0)
            if CONJ:
                di = -di
            denom = dr * dr + di * di
            ir, ii = dr / denom, -di / denom
            scaled = mr * ir[:, None] - mi * ii[:, None]
            mi = mr * ii[:, None] + mi * ir[:, None]
            mr = scaled
        else:
            ir = tl.full((BLOCK,), 1, mr.dtype)
            ii = tl.full((BLOCK,), 0, mr.dtype)
        xr = tl.where(offsets[:, None] == offsets[None, :], ir[:, None], 0)
        xi = tl.where(offsets[:, None] == offsets[None, :], ii[:, None], 0)
    else:
        mr = tl.load(A + addr, mask, other=0)
        if not UNIT:
            inverse = 1.0 / tl.load(A + diagonal, valid, other=1)
            mr *= inverse[:, None]
        else:
            inverse = tl.full((BLOCK,), 1, mr.dtype)
        xr = tl.where(offsets[:, None] == offsets[None, :], inverse[:, None], 0)
    # Solve all identity columns together. Only the selected triangle and
    # explicit diagonals are read, including for padded final tiles.
    for pivot in range(BLOCK):
        if GATHER:
            ar = tl.gather(mr, tl.full((BLOCK, 1), pivot, tl.int32), 1)
            pr = tl.gather(xr, tl.full((1, BLOCK), pivot, tl.int32), 0)
        else:
            ar = tl.sum(tl.where(offsets[None, :] == pivot, mr, 0), 1)[:, None]
            pr = tl.sum(tl.where(offsets[:, None] == pivot, xr, 0), 0)[None, :]
        if COMPLEX:
            if GATHER:
                ai = tl.gather(mi, tl.full((BLOCK, 1), pivot, tl.int32), 1)
                pi = tl.gather(xi, tl.full((1, BLOCK), pivot, tl.int32), 0)
            else:
                ai = tl.sum(tl.where(offsets[None, :] == pivot, mi, 0), 1)[:, None]
                pi = tl.sum(tl.where(offsets[:, None] == pivot, xi, 0), 0)[None, :]
            xr -= ar * pr - ai * pi
            xi -= ar * pi + ai * pr
        else:
            xr -= ar * pr
    out = tl.program_id(0) * BLOCK * BLOCK + offsets[:, None] * BLOCK + offsets[None, :]
    if COMPLEX:
        tl.store(D + 2 * out, xr)
        tl.store(D + 2 * out + 1, xi)
    else:
        tl.store(D + out, xr)


@libentry()
@triton.jit
def _trsv_wavefront_kernel(
    A,
    X,
    D,
    State,
    n,
    lda,
    incx,
    TRANS: tl.constexpr,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Assign tiles in execution order, not program-id order: every predecessor
    # has started before a dependent CTA can wait for it.
    tile = tl.atomic_add(State, 1, sem="relaxed")
    offsets = tl.arange(0, BLOCK)
    history = tl.arange(0, BLOCK * CHUNK)
    base = tile * BLOCK
    indices = base + offsets
    rows = indices if FORWARD else n - 1 - indices
    valid = indices < n
    if COMPLEX:
        xr = tl.load(X + 2 * rows * incx, valid, other=0)
        xi = tl.load(X + 2 * rows * incx + 1, valid, other=0)
    else:
        xr = tl.load(X + rows * incx, valid, other=0)
    for start in range(0, base, BLOCK * CHUNK):
        end = tl.minimum(start + BLOCK * CHUNK, base)
        # PPU atomic_add(..., 0) polling can repeatedly observe a stale value.
        # Reload progress and the published solution through volatile loads.
        while tl.load(State + 1, volatile=True) < end:
            pass
        previous = start + history
        cols = previous if FORWARD else n - 1 - previous
        mask = valid[:, None] & (previous[None, :] < end)
        address = _dense_offset(rows[:, None], cols[None, :], lda, TRANS)
        if COMPLEX:
            ar = tl.load(A + 2 * address, mask, other=0)
            ai = tl.load(A + 2 * address + 1, mask, other=0)
            if CONJ:
                ai = -ai
            br = tl.load(X + 2 * cols * incx, previous < end, other=0, volatile=True)
            bi = tl.load(
                X + 2 * cols * incx + 1, previous < end, other=0, volatile=True
            )
            xr -= tl.sum(ar * br[None, :] - ai * bi[None, :], 1)
            xi -= tl.sum(ar * bi[None, :] + ai * br[None, :], 1)
        else:
            ar = tl.load(A + address, mask, other=0)
            br = tl.load(X + cols * incx, previous < end, other=0, volatile=True)
            xr -= tl.sum(ar * br[None, :], 1)
    inv_offset = tile * BLOCK * BLOCK + offsets[:, None] * BLOCK + offsets[None, :]
    if COMPLEX:
        dr = tl.load(D + 2 * inv_offset)
        di = tl.load(D + 2 * inv_offset + 1)
        yr = tl.sum(dr * xr[None, :] - di * xi[None, :], 1)
        yi = tl.sum(dr * xi[None, :] + di * xr[None, :], 1)
        tl.store(X + 2 * rows * incx, yr, valid)
        tl.store(X + 2 * rows * incx + 1, yi, valid)
    else:
        inverse = tl.load(D + inv_offset)
        result = tl.sum(inverse * xr[None, :], 1)
        tl.store(X + rows * incx, result, valid)
    tl.debug_barrier()
    tl.atomic_xchg(State + 1, base + BLOCK, sem="release")


def _config(n, dtype):
    # Offline PPU-ZW810E tuning. In-place solves must restore X between
    # autotune trials; use the measured configuration without runtime trials.
    if n <= 64:
        return (8 if dtype.is_complex else 16), 1, 1, False
    if n <= 256:
        return (8, 1, 1, False) if dtype.is_complex else (16, 4, 1, False)
    if dtype == torch.float32 and n >= 4096:
        return 64, 4, 2, True
    return 32, (8 if dtype == torch.complex128 else 4), 1, True


def _solve(dtype, uplo, trans, diag, n, A, lda, x, incx):
    assert A.dtype == dtype == x.dtype
    complex_data = dtype.is_complex
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=complex_data)
    if n == 0:
        return
    block, warps, chunk, gather = _config(n, dtype)
    forward = (uplo == 0) != (trans != 0)
    tiles = triton.cdiv(n, block)
    with torch_device_fn.device(A.device):
        inverse = torch.empty((tiles, block, block), dtype=dtype, device=A.device)
        state = torch.empty((2,), dtype=torch.int32, device=A.device)
        matrix = torch.view_as_real(A) if complex_data else A
        vector = torch.view_as_real(x) if complex_data else x
        inverse_data = torch.view_as_real(inverse) if complex_data else inverse
        # Stream ordering makes the inverse blocks and reset counters visible
        # before the dependent solve starts. Buffers belong to this call.
        _trsv_inverse_kernel[(tiles,)](
            matrix,
            inverse_data,
            state,
            n,
            lda,
            TRANS=trans != 0,
            UNIT=diag == 1,
            FORWARD=forward,
            COMPLEX=complex_data,
            CONJ=trans == 2,
            BLOCK=block,
            GATHER=gather,
            num_warps=warps,
        )
        _trsv_wavefront_kernel[(tiles,)](
            matrix,
            vector,
            inverse_data,
            state,
            n,
            lda,
            incx,
            TRANS=trans != 0,
            FORWARD=forward,
            COMPLEX=complex_data,
            CONJ=trans == 2,
            BLOCK=block,
            CHUNK=chunk,
            num_warps=warps,
        )


def strsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a single-precision triangular system in-place."""
    _solve(torch.float32, uplo, trans, diag, n, A, lda, x, incx)


def dtrsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a double-precision triangular system in-place."""
    _solve(torch.float64, uplo, trans, diag, n, A, lda, x, incx)


def ctrsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a complex single-precision triangular system in-place."""
    _solve_ctrsv(uplo, trans, diag, n, A, lda, x, incx)


def ztrsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a complex double-precision triangular system in-place."""
    _solve_ztrsv(uplo, trans, diag, n, A, lda, x, incx)


@libentry()
@triton.jit
def _ctrsv_inverse_packed_kernel(
    A,
    D,
    State,
    n,
    lda,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    GATHER: tl.constexpr,
):
    if tl.program_id(0) == 0:
        tl.store(State + tl.arange(0, 2), 0)
    offsets = tl.arange(0, BLOCK)
    indices = tl.program_id(0) * BLOCK + offsets
    rows = indices if FORWARD else n - 1 - indices
    valid = indices < n
    addr = _dense_offset(rows[:, None], rows[None, :], lda, TRANS)
    mask = valid[:, None] & valid[None, :] & (offsets[:, None] > offsets[None, :])
    diagonal = _dense_offset(rows, rows, lda, TRANS)
    if COMPLEX:
        mr = tl.load(A + 2 * addr, mask, other=0)
        mi = tl.load(A + 2 * addr + 1, mask, other=0)
        if CONJ:
            mi = -mi
        if not UNIT:
            dr = tl.load(A + 2 * diagonal, valid, other=1)
            di = tl.load(A + 2 * diagonal + 1, valid, other=0)
            if CONJ:
                di = -di
            denom = dr * dr + di * di
            ir, ii = dr / denom, -di / denom
            scaled = mr * ir[:, None] - mi * ii[:, None]
            mi = mr * ii[:, None] + mi * ir[:, None]
            mr = scaled
        else:
            ir = tl.full((BLOCK,), 1, mr.dtype)
            ii = tl.full((BLOCK,), 0, mr.dtype)
        xr = tl.where(offsets[:, None] == offsets[None, :], ir[:, None], 0)
        xi = tl.where(offsets[:, None] == offsets[None, :], ii[:, None], 0)
    else:
        mr = tl.load(A + addr, mask, other=0)
        if not UNIT:
            inverse = 1.0 / tl.load(A + diagonal, valid, other=1)
            mr *= inverse[:, None]
        else:
            inverse = tl.full((BLOCK,), 1, mr.dtype)
        xr = tl.where(offsets[:, None] == offsets[None, :], inverse[:, None], 0)
    # Solve all identity columns together. Only the selected triangle and
    # explicit diagonals are read, including for padded final tiles.
    for pivot in range(BLOCK):
        if GATHER:
            ar = tl.gather(mr, tl.full((BLOCK, 1), pivot, tl.int32), 1)
            pr = tl.gather(xr, tl.full((1, BLOCK), pivot, tl.int32), 0)
        else:
            ar = tl.sum(tl.where(offsets[None, :] == pivot, mr, 0), 1)[:, None]
            pr = tl.sum(tl.where(offsets[:, None] == pivot, xr, 0), 0)[None, :]
        if COMPLEX:
            if GATHER:
                ai = tl.gather(mi, tl.full((BLOCK, 1), pivot, tl.int32), 1)
                pi = tl.gather(xi, tl.full((1, BLOCK), pivot, tl.int32), 0)
            else:
                ai = tl.sum(tl.where(offsets[None, :] == pivot, mi, 0), 1)[:, None]
                pi = tl.sum(tl.where(offsets[:, None] == pivot, xi, 0), 0)[None, :]
            xr -= ar * pr - ai * pi
            xi -= ar * pi + ai * pr
        else:
            xr -= ar * pr
    out = (
        tl.program_id(0) * (BLOCK * (BLOCK - 1) // 2)
        + offsets[:, None] * (offsets[:, None] - 1) // 2
        + offsets[None, :]
    )
    if COMPLEX:
        tl.store(D + 2 * out, xr, mask=offsets[:, None] > offsets[None, :])
        tl.store(D + 2 * out + 1, xi, mask=offsets[:, None] > offsets[None, :])
    else:
        tl.store(D + out, xr, mask=offsets[:, None] > offsets[None, :])


@libentry()
@triton.jit
def _ctrsv_wavefront_packed_kernel(
    A,
    X,
    D,
    State,
    n,
    lda,
    incx,
    TRANS: tl.constexpr,
    FORWARD: tl.constexpr,
    CONJ: tl.constexpr,
    UNIT: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    tile = tl.atomic_add(State, 1, sem="relaxed")
    o = tl.arange(0, BLOCK)
    hist = tl.arange(0, BLOCK * CHUNK)
    a64 = A.to(tl.pointer_type(tl.int64))
    x64 = X.to(tl.pointer_type(tl.int64))
    d64 = D.to(tl.pointer_type(tl.int64))
    base = tile * BLOCK
    idx = base + o
    rows = idx if FORWARD else n - 1 - idx
    valid = idx < n
    xv = tl.load(x64 + rows * incx, valid, 0)
    xr = xv.to(tl.int32).to(tl.float32, bitcast=True)
    xi = (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    for start in range(0, base, BLOCK * CHUNK):
        end = tl.minimum(start + BLOCK * CHUNK, base)
        prev = start + hist
        cols = prev if FORWARD else n - 1 - prev
        if TRANS:
            addr = cols[:, None] * lda + rows[None, :]
            mask = (prev[:, None] < end) & valid[None, :]
        else:
            addr = rows[:, None] * lda + cols[None, :]
            mask = valid[:, None] & (prev[None, :] < end)
        # The matrix is immutable: start fetching it before waiting for X.
        av = tl.load(a64 + addr, mask, 0)
        ar = av.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (av >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if CONJ:
            ai = -ai
        while tl.load(State + 1, volatile=True) < end:
            pass
        bv = tl.load(x64 + cols * incx, prev < end, 0, volatile=True)
        br = bv.to(tl.int32).to(tl.float32, bitcast=True)
        bi = (bv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if TRANS:
            xr -= tl.sum(ar * br[:, None] - ai * bi[:, None], 0)
            xi -= tl.sum(ar * bi[:, None] + ai * br[:, None], 0)
        else:
            xr -= tl.sum(ar * br[None, :] - ai * bi[None, :], 1)
            xi -= tl.sum(ar * bi[None, :] + ai * br[None, :], 1)
    if PACKED:
        off = (
            tile * (BLOCK * (BLOCK - 1) // 2)
            + o[:, None] * (o[:, None] - 1) // 2
            + o[None, :]
        )
        dv = tl.load(d64 + off, o[:, None] > o[None, :], 0)
    else:
        off = tile * BLOCK * BLOCK + o[:, None] * BLOCK + o[None, :]
        dv = tl.load(d64 + off)
    dr = dv.to(tl.int32).to(tl.float32, bitcast=True)
    di = (dv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    if PACKED:
        # The stored inverse omits diagonals and the zero upper triangle.
        if UNIT:
            ir = tl.full((BLOCK,), 1.0, tl.float32)
            ii = tl.full((BLOCK,), 0.0, tl.float32)
        else:
            diag = tl.load(a64 + rows * lda + rows, valid, 0)
            dr0 = diag.to(tl.int32).to(tl.float32, bitcast=True)
            di0 = (diag >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            dr0 = tl.where(valid, dr0, 1.0)
            if CONJ:
                di0 = -di0
            denom = dr0 * dr0 + di0 * di0
            ir, ii = dr0 / denom, -di0 / denom
        isdiag = o[:, None] == o[None, :]
        dr = tl.where(isdiag, ir[:, None], dr)
        di = tl.where(isdiag, ii[:, None], di)
    yr = tl.sum(dr * xr[None, :] - di * xi[None, :], 1)
    yi = tl.sum(dr * xi[None, :] + di * xr[None, :], 1)
    real_bits = yr.to(tl.uint32, bitcast=True).to(tl.uint64)
    imag_bits = yi.to(tl.uint32, bitcast=True).to(tl.uint64)
    out = (real_bits | (imag_bits << 32)).to(tl.int64, bitcast=True)
    tl.store(x64 + rows * incx, out, valid)
    tl.debug_barrier()
    tl.atomic_xchg(State + 1, base + BLOCK, sem="release")


_CTRSV_CONFIGS = {
    name: runtime.get_tuned_config("thead_ctrsv_" + name)[0]
    for name in ("small", "forward", "backward", "trans")
}


def _solve_ctrsv(uplo, trans, diag, n, A, lda, x, incx):
    if n <= 256:
        return _solve(torch.complex64, uplo, trans, diag, n, A, lda, x, incx)
    assert A.dtype == torch.complex64 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    forward = (uplo == 0) != (trans != 0)
    packed_inverse = trans == 0 and forward
    if trans != 0:
        config = _CTRSV_CONFIGS["trans"]
    elif n <= 2048:
        config = _CTRSV_CONFIGS["small"]
    elif forward:
        config = _CTRSV_CONFIGS["forward"]
    else:
        config = _CTRSV_CONFIGS["backward"]
    block = config.kwargs["BLOCK"]
    chunk = config.kwargs["CHUNK"]
    tiles = triton.cdiv(n, block)
    inverse_size = block * (block - 1) // 2 if packed_inverse else block * block
    with torch_device_fn.device(A.device):
        inverse = torch.empty((tiles, inverse_size), dtype=A.dtype, device=A.device)
        state = torch.empty((2,), dtype=torch.int32, device=A.device)
        matrix = torch.view_as_real(A)
        vector = torch.view_as_real(x)
        inverse_data = torch.view_as_real(inverse)
        inverse_kernel = (
            _ctrsv_inverse_packed_kernel if packed_inverse else _trsv_inverse_kernel
        )
        inverse_kernel[(tiles,)](
            matrix,
            inverse_data,
            state,
            n,
            lda,
            TRANS=trans != 0,
            UNIT=diag == 1,
            FORWARD=forward,
            COMPLEX=True,
            CONJ=trans == 2,
            BLOCK=block,
            GATHER=True,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _ctrsv_wavefront_packed_kernel[(tiles,)](
            matrix,
            vector,
            inverse_data,
            state,
            n,
            lda,
            incx,
            TRANS=trans != 0,
            FORWARD=forward,
            CONJ=trans == 2,
            UNIT=diag == 1,
            PACKED=packed_inverse,
            BLOCK=block,
            CHUNK=chunk,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )


@libentry()
@triton.jit
def _ztrsv_wavefront_kernel(
    A,
    X,
    D,
    State,
    n,
    lda,
    incx,
    TRANS: tl.constexpr,
    FORWARD: tl.constexpr,
    CONJ: tl.constexpr,
    UNIT: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Claim tiles in execution order so every dependency has started.
    tile = tl.atomic_add(State, 1, sem="relaxed")
    o = tl.arange(0, BLOCK)
    h = tl.arange(0, BLOCK * CHUNK)
    part = tl.arange(0, 2)
    base = tile * BLOCK
    idx = base + o
    rows = idx if FORWARD else n - 1 - idx
    valid = idx < n
    xv = tl.load(X + (rows * incx)[:, None] * 2 + part[None, :], valid[:, None], 0)
    xr, xi = tl.split(xv)
    for start in range(0, base, BLOCK * CHUNK):
        end = tl.minimum(start + BLOCK * CHUNK, base)
        prev = start + h
        cols = prev if FORWARD else n - 1 - prev
        if TRANS:
            # Interleave real/imaginary components along contiguous outputs.
            f = tl.arange(0, BLOCK * 2)
            r2 = base + f // 2
            rphys = r2 if FORWARD else n - 1 - r2
            addr = cols[:, None] * lda + rphys[None, :]
            mask = (prev[:, None] < end) & (r2[None, :] < n)
            av = tl.load(A + addr * 2 + f[None, :] % 2, mask, 0)
            if CONJ:
                av = tl.where(f[None, :] % 2 == 0, av, -av)
            swap = tl.gather(
                av, tl.broadcast_to((f ^ 1)[None, :], (BLOCK * CHUNK, BLOCK * 2)), 1
            )
            while tl.load(State + 1, volatile=True) < end:
                pass
            bv = tl.load(
                X + (cols * incx)[:, None] * 2 + part[None, :],
                (prev < end)[:, None],
                0,
                volatile=True,
            )
            br, bi = tl.split(bv)
            prod = av * br[:, None] + swap * bi[:, None] * tl.where(
                f[None, :] % 2 == 0, -1.0, 1.0
            )
            result = tl.sum(prod, 0).reshape((BLOCK, 2))
            rr, ri = tl.split(result)
            xr -= rr
            xi -= ri
        else:
            addr = rows[:, None] * lda + cols[None, :]
            mask = valid[:, None] & (prev[None, :] < end)
            # A is immutable; fetch it while predecessor tiles publish X.
            av = tl.load(
                A + addr[:, :, None] * 2 + part[None, None, :], mask[:, :, None], 0
            )
            ar, ai = tl.split(av)
            if CONJ:
                ai = -ai
            while tl.load(State + 1, volatile=True) < end:
                pass
            bv = tl.load(
                X + (cols * incx)[:, None] * 2 + part[None, :],
                (prev < end)[:, None],
                0,
                volatile=True,
            )
            br, bi = tl.split(bv)
            xr -= tl.sum(ar * br[None, :] - ai * bi[None, :], 1)
            xi -= tl.sum(ar * bi[None, :] + ai * br[None, :], 1)
    # Compact inverse blocks contain only the strict lower triangle.
    if PACKED:
        off = (
            tile * (BLOCK * (BLOCK - 1) // 2)
            + o[:, None] * (o[:, None] - 1) // 2
            + o[None, :]
        )
        mask = o[:, None] > o[None, :]
        dv = tl.load(D + off[:, :, None] * 2 + part[None, None, :], mask[:, :, None], 0)
    else:
        off = tile * BLOCK * BLOCK + o[:, None] * BLOCK + o[None, :]
        dv = tl.load(D + off[:, :, None] * 2 + part[None, None, :])
    dr, di = tl.split(dv)
    if PACKED:
        if UNIT:
            ir = tl.full((BLOCK,), 1.0, tl.float64)
            ii = tl.full((BLOCK,), 0.0, tl.float64)
        else:
            diag = tl.load(
                A + (rows * lda + rows)[:, None] * 2 + part[None, :], valid[:, None], 0
            )
            ar, ai = tl.split(diag)
            ar = tl.where(valid, ar, 1.0)
            if CONJ:
                ai = -ai
            den = ar * ar + ai * ai
            ir, ii = ar / den, -ai / den
        isdiag = o[:, None] == o[None, :]
        dr = tl.where(isdiag, ir[:, None], dr)
        di = tl.where(isdiag, ii[:, None], di)
    yr = tl.sum(dr * xr[None, :] - di * xi[None, :], 1)
    yi = tl.sum(dr * xi[None, :] + di * xr[None, :], 1)
    tl.store(
        X + (rows * incx)[:, None] * 2 + part[None, :], tl.join(yr, yi), valid[:, None]
    )
    tl.debug_barrier()
    tl.atomic_xchg(State + 1, base + BLOCK, sem="release")


_ZTRSV_CONFIGS = {
    name: runtime.get_tuned_config("thead_ztrsv_" + name)[0]
    for name in ("normal", "trans")
}


def _solve_ztrsv(uplo, trans, diag, n, A, lda, x, incx):
    if n <= 64 or (trans != 0 and n <= 2048):
        return _solve(torch.complex128, uplo, trans, diag, n, A, lda, x, incx)
    assert A.dtype == torch.complex128 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    forward = (uplo == 0) != (trans != 0)
    packed_inverse = trans == 0 and forward
    config = _ZTRSV_CONFIGS["trans" if trans != 0 else "normal"]
    block = config.kwargs["BLOCK"]
    chunk = config.kwargs["CHUNK"]
    tiles = triton.cdiv(n, block)
    inverse_size = block * (block - 1) // 2 if packed_inverse else block * block
    with torch_device_fn.device(A.device):
        inverse = torch.empty((tiles, inverse_size), dtype=A.dtype, device=A.device)
        state = torch.empty((2,), dtype=torch.int32, device=A.device)
        matrix = torch.view_as_real(A)
        vector = torch.view_as_real(x)
        inverse_data = torch.view_as_real(inverse)
        # The existing compact inverse builder also supports complex128.
        inverse_kernel = (
            _ctrsv_inverse_packed_kernel if packed_inverse else _trsv_inverse_kernel
        )
        inverse_kernel[(tiles,)](
            matrix,
            inverse_data,
            state,
            n,
            lda,
            TRANS=trans != 0,
            UNIT=diag == 1,
            FORWARD=forward,
            COMPLEX=True,
            CONJ=trans == 2,
            BLOCK=block,
            GATHER=True,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _ztrsv_wavefront_kernel[(tiles,)](
            matrix,
            vector,
            inverse_data,
            state,
            n,
            lda,
            incx,
            TRANS=trans != 0,
            FORWARD=forward,
            CONJ=trans == 2,
            UNIT=diag == 1,
            PACKED=packed_inverse,
            BLOCK=block,
            CHUNK=chunk,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

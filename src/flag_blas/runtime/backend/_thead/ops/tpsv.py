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

from flag_blas.ops.level2.tpsv import (
    CUBLAS_OP_C,
    _check_common,
    _real_tpsv_panel_ranges,
    _row_major_tpsv_args,
    _tpsv_packed_offset,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@libentry()
@triton.jit
def _tpsv_panel_kernel(
    AP,
    X,
    n,
    start,
    end,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    CONJ: tl.constexpr,
    COMPLEX: tl.constexpr,
    INCX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Keep the RHS in registers and update one column in parallel per pivot.
    # A 1-D layout avoids a shared-memory transpose at every substitution step.
    offsets = tl.arange(0, BLOCK)
    rows = start + offsets
    valid = rows < end
    if COMPLEX:
        xr = tl.load(X + 2 * rows * INCX, valid, other=0)
        xi = tl.load(X + 2 * rows * INCX + 1, valid, other=0)
    else:
        xr = tl.load(X + rows * INCX, valid, other=0)
    if not UNIT:
        # Normalize all rows together, outside the sequential pivot loop.
        diagonal = _tpsv_packed_offset(rows, rows, n, UPLO)
        if COMPLEX:
            dr = tl.load(AP + 2 * diagonal, valid, other=1)
            di = tl.load(AP + 2 * diagonal + 1, valid, other=0)
            if CONJ:
                di = -di
            denominator = dr * dr + di * di
            inverse_r = dr / denominator
            inverse_i = -di / denominator
            scaled_xr = xr * inverse_r - xi * inverse_i
            xi = xr * inverse_i + xi * inverse_r
            xr = scaled_xr
        else:
            dr = tl.load(AP + diagonal, valid, other=1)
            inverse = 1.0 / dr
            xr *= inverse

    for step in range(BLOCK):
        if FORWARD:
            pivot = step
            remaining = offsets > pivot
        else:
            pivot = BLOCK - 1 - step
            remaining = offsets < pivot
        pr = tl.sum(tl.where(offsets == pivot, xr, 0), 0)
        if COMPLEX:
            pi = tl.sum(tl.where(offsets == pivot, xi, 0), 0)
        if TRANS == 0:
            packed = _tpsv_packed_offset(rows, start + pivot, n, UPLO)
        else:
            packed = _tpsv_packed_offset(start + pivot, rows, n, UPLO)
        mask = valid & remaining & (start + pivot < end)
        if COMPLEX:
            ar = tl.load(AP + 2 * packed, mask, other=0)
            ai = tl.load(AP + 2 * packed + 1, mask, other=0)
            if CONJ:
                ai = -ai
            if not UNIT:
                scaled_ar = ar * inverse_r - ai * inverse_i
                ai = ar * inverse_i + ai * inverse_r
                ar = scaled_ar
            xr = tl.where(remaining, xr - ar * pr + ai * pi, xr)
            xi = tl.where(remaining, xi - ar * pi - ai * pr, xi)
        else:
            ar = tl.load(AP + packed, mask, other=0)
            if not UNIT:
                ar *= inverse
            xr = tl.where(remaining, xr - ar * pr, xr)

    if COMPLEX:
        tl.store(X + 2 * rows * INCX, xr, valid)
        tl.store(X + 2 * rows * INCX + 1, xi, valid)
    else:
        tl.store(X + rows * INCX, xr, valid)


@libentry()
@triton.jit
def _tpsv_update_kernel(
    AP,
    X,
    n,
    row_base,
    row_count,
    start,
    end,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    COMPLEX: tl.constexpr,
    INCX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    local_rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = row_base + local_rows
    cols = start + tl.arange(0, BLOCK_N)
    row_mask = local_rows < row_count
    col_mask = cols < end
    if TRANS == 0:
        packed = _tpsv_packed_offset(rows[:, None], cols[None, :], n, UPLO)
    else:
        packed = _tpsv_packed_offset(cols[None, :], rows[:, None], n, UPLO)
    mask = row_mask[:, None] & col_mask[None, :]
    if COMPLEX:
        ar = tl.load(AP + 2 * packed, mask, other=0)
        ai = tl.load(AP + 2 * packed + 1, mask, other=0)
        if CONJ:
            ai = -ai
        xr = tl.load(X + 2 * cols * INCX, col_mask, other=0)
        xi = tl.load(X + 2 * cols * INCX + 1, col_mask, other=0)
        update_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
        update_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
        rhs_r = tl.load(X + 2 * rows * INCX, row_mask, other=0)
        rhs_i = tl.load(X + 2 * rows * INCX + 1, row_mask, other=0)
        tl.store(X + 2 * rows * INCX, rhs_r - update_r, row_mask)
        tl.store(X + 2 * rows * INCX + 1, rhs_i - update_i, row_mask)
    else:
        matrix = tl.load(AP + packed, mask, other=0)
        solved = tl.load(X + cols * INCX, col_mask, other=0)
        update = tl.sum(matrix * solved[None, :], 1)
        rhs = tl.load(X + rows * INCX, row_mask, other=0)
        tl.store(X + rows * INCX, rhs - update, row_mask)


def _solve(uplo, trans, diag, n, AP, x, incx, complex_data):
    uplo, trans, conj = _row_major_tpsv_args(uplo, trans)
    forward = (uplo == 0) != (trans != 0)
    block_n = min(64, triton.next_power_of_2(n))
    block_m = 64
    matrix = torch.view_as_real(AP) if complex_data else AP
    vector = torch.view_as_real(x) if complex_data else x
    # Stream ordering provides the dependencies without cross-program spin
    # waits, which can deadlock on T-Head. Autotuning would need to restore X
    # before every trial because the solve modifies its RHS in place.
    with torch_device_fn.device(AP.device):
        for start, end in _real_tpsv_panel_ranges(n, block_n, forward):
            _tpsv_panel_kernel[(1,)](
                matrix,
                vector,
                n,
                start,
                end,
                UPLO=uplo,
                TRANS=trans,
                UNIT=diag == 1,
                FORWARD=forward,
                CONJ=conj,
                COMPLEX=complex_data,
                INCX=incx,
                BLOCK=block_n,
                num_warps=1,
            )
            row_base = end if forward else 0
            row_count = n - end if forward else start
            if row_count:
                _tpsv_update_kernel[(triton.cdiv(row_count, block_m),)](
                    matrix,
                    vector,
                    n,
                    row_base,
                    row_count,
                    start,
                    end,
                    UPLO=uplo,
                    TRANS=trans,
                    CONJ=conj,
                    COMPLEX=complex_data,
                    INCX=incx,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                )
    return x


def _real_tpsv(dtype, uplo, trans, diag, n, AP, x, incx):
    assert trans != CUBLAS_OP_C
    _check_common(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype is dtype and x.dtype is dtype
    if n == 0:
        return x

    return _solve(uplo, trans, diag, n, AP, x, incx, False)


def _complex_tpsv(dtype, uplo, trans, diag, n, AP, x, incx):
    _check_common(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype is dtype and x.dtype is dtype
    if n == 0:
        return x

    return _solve(uplo, trans, diag, n, AP, x, incx, True)


def stpsv(uplo, trans, diag, n, AP, x, incx):
    return _real_tpsv(torch.float32, uplo, trans, diag, n, AP, x, incx)


def dtpsv(uplo, trans, diag, n, AP, x, incx):
    return _real_tpsv(torch.float64, uplo, trans, diag, n, AP, x, incx)


def ctpsv(uplo, trans, diag, n, AP, x, incx):
    return _complex_tpsv(torch.complex64, uplo, trans, diag, n, AP, x, incx)


def ztpsv(uplo, trans, diag, n, AP, x, incx):
    return _complex_tpsv(torch.complex128, uplo, trans, diag, n, AP, x, incx)

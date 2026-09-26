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

from flag_blas.ops.level2.tbsv import (
    CUBLAS_DIAG_UNIT,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    _check_tbsv,
    _complex_tbsv_kernel,
    _real_tbsv_kernel,
    _row_major_tbsv_args,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@triton.jit
def _band_offset(row, col, k, lda, UPLO: tl.constexpr, TRANS: tl.constexpr):
    if TRANS:
        row, col = col, row
    if UPLO == 0:
        return row * lda + k + col - row
    return row * lda + col - row


@libentry()
@triton.jit
def _tbsv_panel_kernel(
    A,
    X,
    n,
    k,
    lda,
    incx,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program owns the whole solve: no cross-program spin-wait flags.
    # Panels retain their RHS in registers; earlier panels contribute through
    # a parallel banded matrix-vector reduction before local substitution.
    offsets = tl.arange(0, BLOCK)
    history = tl.arange(0, BLOCK_K)
    FORWARD: tl.constexpr = (UPLO == 0) != TRANS
    for base in range(0, n, BLOCK):
        indices = base + offsets
        rows = indices if FORWARD else n - 1 - indices
        valid = indices < n
        if COMPLEX:
            xr = tl.load(X + 2 * rows * incx, valid, other=0)
            xi = tl.load(X + 2 * rows * incx + 1, valid, other=0)
        else:
            xr = tl.load(X + rows * incx, valid, other=0)
        for kb in range(0, tl.minimum(k, base), BLOCK_K):
            previous = base - 1 - kb - history
            cols = previous if FORWARD else n - 1 - previous
            distance = indices[:, None] - previous[None, :]
            mask = valid[:, None] & (previous[None, :] >= 0) & (distance <= k)
            address = _band_offset(
                rows[:, None], cols[None, :], k, lda, UPLO, TRANS
            )
            if COMPLEX:
                ar = tl.load(A + 2 * address, mask, other=0)
                ai = tl.load(A + 2 * address + 1, mask, other=0)
                if CONJ:
                    ai = -ai
                br = tl.load(X + 2 * cols * incx, previous >= 0, other=0)
                bi = tl.load(X + 2 * cols * incx + 1, previous >= 0, other=0)
                xr -= tl.sum(ar * br[None, :] - ai * bi[None, :], 1)
                xi -= tl.sum(ar * bi[None, :] + ai * br[None, :], 1)
            else:
                ar = tl.load(A + address, mask, other=0)
                br = tl.load(X + cols * incx, previous >= 0, other=0)
                xr -= tl.sum(ar * br[None, :], 1)

        if not UNIT:
            diagonal = _band_offset(rows, rows, k, lda, UPLO, TRANS)
            if COMPLEX:
                dr = tl.load(A + 2 * diagonal, valid, other=1)
                di = tl.load(A + 2 * diagonal + 1, valid, other=0)
                if CONJ:
                    di = -di
                denominator = dr * dr + di * di
                inverse_r = dr / denominator
                inverse_i = -di / denominator
                scaled_r = xr * inverse_r - xi * inverse_i
                xi = xr * inverse_i + xi * inverse_r
                xr = scaled_r
            else:
                inverse = 1.0 / tl.load(A + diagonal, valid, other=1)
                xr *= inverse

        for pivot in range(tl.minimum(BLOCK, n - base)):
            pr = tl.sum(tl.where(offsets == pivot, xr, 0), 0)
            if COMPLEX:
                pi = tl.sum(tl.where(offsets == pivot, xi, 0), 0)
            col = base + pivot if FORWARD else n - 1 - base - pivot
            remaining = (offsets > pivot) & (offsets - pivot <= k)
            mask = valid & remaining
            address = _band_offset(rows, col, k, lda, UPLO, TRANS)
            if COMPLEX:
                ar = tl.load(A + 2 * address, mask, other=0)
                ai = tl.load(A + 2 * address + 1, mask, other=0)
                if CONJ:
                    ai = -ai
                if not UNIT:
                    scaled_r = ar * inverse_r - ai * inverse_i
                    ai = ar * inverse_i + ai * inverse_r
                    ar = scaled_r
                xr = tl.where(remaining, xr - ar * pr + ai * pi, xr)
                xi = tl.where(remaining, xi - ar * pi - ai * pr, xi)
            else:
                ar = tl.load(A + address, mask, other=0)
                if not UNIT:
                    ar *= inverse
                xr = tl.where(remaining, xr - ar * pr, xr)

        if COMPLEX:
            tl.store(X + 2 * rows * incx, xr, valid)
            tl.store(X + 2 * rows * incx + 1, xi, valid)
        else:
            tl.store(X + rows * incx, xr, valid)
        # The next panel can read values stored by another warp in this CTA.
        tl.debug_barrier()


def _panel_solve(uplo, trans, diag, n, k, A, lda, x, incx):
    complex_data = x.dtype.is_complex
    # A warp-sized panel limits serial pivot work. Four warps distribute the
    # history reduction without the register pressure of larger panels.
    with torch_device_fn.device(A.device):
        _tbsv_panel_kernel[(1,)](
            torch.view_as_real(A) if complex_data else A,
            torch.view_as_real(x) if complex_data else x,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans != CUBLAS_OP_N,
            UNIT=diag == CUBLAS_DIAG_UNIT,
            COMPLEX=complex_data,
            CONJ=trans == CUBLAS_OP_C,
            BLOCK=32,
            BLOCK_K=min(128, triton.next_power_of_2(max(k, 1))),
            num_warps=4,
        )


def _real_tbsv(
    dtype: torch.dtype,
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == dtype == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return

    if k >= 4:
        return _panel_solve(uplo, trans, diag, n, k, A, lda, x, incx)

    uplo, trans, _ = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    # Multi-program TBSV kernels use a spin-wait flag that can deadlock on T-Head.
    with torch_device_fn.device(A.device):
        _real_tbsv_kernel[(1,)](
            A,
            x,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
        )


def _complex_tbsv(
    dtype: torch.dtype,
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == dtype == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return

    if k >= 4:
        return _panel_solve(uplo, trans, diag, n, k, A, lda, x, incx)

    uplo, trans, conj = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0

    with torch_device_fn.device(A.device):
        _complex_tbsv_kernel[(1,)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans,
            UNIT=unit,
            CONJ=conj,
        )


def stbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a single-precision triangular banded system in-place."""
    _real_tbsv(torch.float32, uplo, trans, diag, n, k, A, lda, x, incx)


def dtbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a double-precision triangular banded system in-place."""
    _real_tbsv(torch.float64, uplo, trans, diag, n, k, A, lda, x, incx)


def ctbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a complex single-precision triangular banded system in-place."""
    _complex_tbsv(torch.complex64, uplo, trans, diag, n, k, A, lda, x, incx)


def ztbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a complex double-precision triangular banded system in-place."""
    _complex_tbsv(torch.complex128, uplo, trans, diag, n, k, A, lda, x, incx)

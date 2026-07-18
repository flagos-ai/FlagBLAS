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

import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUBLAS_OP_C,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]


_TRMV_KEY = ["n", "mode_key"]
_TRMV_RESTORE = ["x_ptr"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strmv"),
    key=_TRMV_KEY,
    restore_value=_TRMV_RESTORE,
)
@triton.jit
def strmv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Pre-diagonal interior tiles (j<=rows pattern only)
    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            if TRANS == 0:
                a_off = rows[:, None] + j[None, :] * LDA
            else:
                a_off = j[None, :] + rows[:, None] * LDA
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(
                a_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            x_vals = tl.load(
                xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    # Diagonal tiles (need triangular mask)
    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
            a_off = rows[:, None] + j[None, :] * LDA
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            a_off = j[None, :] + rows[:, None] * LDA
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_vals = tl.load(
            a_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        x_vals = tl.load(
            xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    # Post-diagonal interior tiles (j>=rows pattern only)
    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            if TRANS == 0:
                a_off = rows[:, None] + j[None, :] * LDA
            else:
                a_off = j[None, :] + rows[:, None] * LDA
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(
                a_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            x_vals = tl.load(
                xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dtrmv"),
    key=_TRMV_KEY,
    restore_value=_TRMV_RESTORE,
)
@triton.jit
def dtrmv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    offs_k = tl.arange(0, BLOCK_K)

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j_mask = j < n
            if TRANS == 0:
                a_off = rows[:, None] + j[None, :] * LDA
            else:
                a_off = j[None, :] + rows[:, None] * LDA
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
            x_vals = tl.load(xin_ptr + j, mask=j_mask, other=0.0)
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
            a_off = rows[:, None] + j[None, :] * LDA
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            a_off = j[None, :] + rows[:, None] * LDA
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(xin_ptr + j, mask=j_mask, other=0.0)
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j_mask = j < n
            if TRANS == 0:
                a_off = rows[:, None] + j[None, :] * LDA
            else:
                a_off = j[None, :] + rows[:, None] * LDA
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
            x_vals = tl.load(xin_ptr + j, mask=j_mask, other=0.0)
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctrmv"),
    key=_TRMV_KEY,
    restore_value=_TRMV_RESTORE,
)
@triton.jit
def ctrmv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)
    acc_i_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j_mask = j < n
            if TRANS == 0:
                a_off = (rows[:, None] + j[None, :] * LDA) * 2
            else:
                a_off = (j[None, :] + rows[:, None] * LDA) * 2
            mask = row_mask[:, None] & j_mask[None, :]
            ar = tl.load(
                a_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                a_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + j * 2, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + j * 2 + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r_2d += ar * xr[None, :] - ai * xi[None, :]
            acc_i_2d += ar * xi[None, :] + ai * xr[None, :]
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
            a_off = (rows[:, None] + j[None, :] * LDA) * 2
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            a_off = (j[None, :] + rows[:, None] * LDA) * 2
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first")
        ai = tl.load(
            a_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        xr = tl.load(
            xin_ptr + j * 2, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        xi = tl.load(
            xin_ptr + j * 2 + 1,
            mask=j_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        if CONJ:
            ai = -ai
        acc_r_2d += ar * xr[None, :] - ai * xi[None, :]
        acc_i_2d += ar * xi[None, :] + ai * xr[None, :]

    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j_mask = j < n
            if TRANS == 0:
                a_off = (rows[:, None] + j[None, :] * LDA) * 2
            else:
                a_off = (j[None, :] + rows[:, None] * LDA) * 2
            mask = row_mask[:, None] & j_mask[None, :]
            ar = tl.load(
                a_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                a_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + j * 2, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + j * 2 + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r_2d += ar * xr[None, :] - ai * xi[None, :]
            acc_i_2d += ar * xi[None, :] + ai * xr[None, :]

    acc_r = tl.sum(acc_r_2d, axis=1)
    acc_i = tl.sum(acc_i_2d, axis=1)

    if UNIT:
        acc_r += tl.load(
            xin_ptr + rows * 2,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        acc_i += tl.load(
            xin_ptr + rows * 2 + 1,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_last",
        )

    x_off = rows * INCX * 2
    tl.store(x_ptr + x_off, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off + 1, acc_i, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ztrmv"),
    key=_TRMV_KEY,
    restore_value=_TRMV_RESTORE,
)
@triton.jit
def ztrmv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    offs_k = tl.arange(0, BLOCK_K)

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j_mask = j < n
            if TRANS == 0:
                a_off = (rows[:, None] + j[None, :] * LDA) * 2
            else:
                a_off = (j[None, :] + rows[:, None] * LDA) * 2
            mask = row_mask[:, None] & j_mask[None, :]
            ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
            ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
            xr = tl.load(xin_ptr + j * 2, mask=j_mask, other=0.0)
            xi = tl.load(xin_ptr + j * 2 + 1, mask=j_mask, other=0.0)
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
            a_off = (rows[:, None] + j[None, :] * LDA) * 2
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            a_off = (j[None, :] + rows[:, None] * LDA) * 2
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(xin_ptr + j * 2, mask=j_mask, other=0.0)
        xi = tl.load(xin_ptr + j * 2 + 1, mask=j_mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j_mask = j < n
            if TRANS == 0:
                a_off = (rows[:, None] + j[None, :] * LDA) * 2
            else:
                a_off = (j[None, :] + rows[:, None] * LDA) * 2
            mask = row_mask[:, None] & j_mask[None, :]
            ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
            ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
            xr = tl.load(xin_ptr + j * 2, mask=j_mask, other=0.0)
            xi = tl.load(xin_ptr + j * 2 + 1, mask=j_mask, other=0.0)
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off = rows * INCX * 2
    tl.store(x_ptr + x_off, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off + 1, acc_i, mask=row_mask)


def _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok):
    assert A.is_contiguous() and x.is_contiguous()
    assert A.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
    allowed = (
        [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
        if complex_ok
        else [CUBLAS_OP_N, CUBLAS_OP_T]
    )
    assert trans in allowed
    assert diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    assert incx > 0
    assert n >= 0
    assert lda >= max(1, n)
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= n * lda


def _mode_key(uplo, trans, unit):
    return (uplo << 4) | (trans << 2) | unit


def strmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        xin = x.as_strided((n,), (incx,)).clone()

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)

        strmv_kernel[grid](
            A,
            xin,
            x,
            n,
            lda,
            incx,
            _mode_key(uplo, trans_flag, unit),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
        )


def dtrmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float64 == x.dtype
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        xin = x.as_strided((n,), (incx,)).clone()

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)

        dtrmv_kernel[grid](
            A,
            xin,
            x,
            n,
            lda,
            incx,
            _mode_key(uplo, trans_flag, unit),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
        )


def ctrmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0

    with torch_device_fn.device(A.device):
        xin = x.as_strided((n,), (incx,)).clone()
        A_real = torch.view_as_real(A)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)

        ctrmv_kernel[grid](
            A_real,
            xin_real,
            x_real,
            n,
            lda,
            incx,
            _mode_key(uplo, trans_flag, unit) | (conj << 8),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
        )


def ztrmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex128 == x.dtype
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0

    with torch_device_fn.device(A.device):
        xin = x.as_strided((n,), (incx,)).clone()
        A_real = torch.view_as_real(A)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)

        ztrmv_kernel[grid](
            A_real,
            xin_real,
            x_real,
            n,
            lda,
            incx,
            _mode_key(uplo, trans_flag, unit) | (conj << 8),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
        )

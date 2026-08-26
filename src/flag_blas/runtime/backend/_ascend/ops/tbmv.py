import importlib

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT, CUBLAS_OP_N
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = importlib.import_module("flag_blas.ops.level2.tbmv")


@libentry()
@triton.jit
def stbmv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n
    xin = tl.load(xin_ptr + rows, mask=row_mask, other=0.0)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, k + 1, BAND_TILE):
        r = r_base + r_off
        r_mask = r <= k
        if UPLO == 1:
            if TRANS == 0:
                j = rows[:, None] + (k - r[None, :])
                col = j
            else:
                j = rows[:, None] - (k - r[None, :])
                col = rows[:, None] + tl.zeros((1, BAND_TILE), tl.int32)
            diag = r[None, :] == k
        else:
            if TRANS == 0:
                j = rows[:, None] - r[None, :]
                col = j
            else:
                j = rows[:, None] + r[None, :]
                col = rows[:, None] + tl.zeros((1, BAND_TILE), tl.int32)
            diag = r[None, :] == 0
        valid = row_mask[:, None] & r_mask[None, :] & (j >= 0) & (j < n)
        if UNIT:
            valid &= ~diag
        safe_j = tl.where(valid, j, 0)
        safe_col = tl.where(valid, col, 0)
        av = tl.load(a_ptr + r[None, :] + safe_col * LDA, mask=valid, other=0.0)
        xv = tl.load(xin_ptr + safe_j, mask=valid, other=0.0)
        acc += tl.sum(av * xv, axis=1)
    if UNIT:
        acc += xin
    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@libentry()
@triton.jit
def ctbmv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n
    xin_r = tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
    xin_i = tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)
    acc_r = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, k + 1, BAND_TILE):
        r = r_base + r_off
        r_mask = r <= k
        if UPLO == 1:
            if TRANS == 0:
                j = rows[:, None] + (k - r[None, :])
                col = j
            else:
                j = rows[:, None] - (k - r[None, :])
                col = rows[:, None] + tl.zeros((1, BAND_TILE), tl.int32)
            diag = r[None, :] == k
        else:
            if TRANS == 0:
                j = rows[:, None] - r[None, :]
                col = j
            else:
                j = rows[:, None] + r[None, :]
                col = rows[:, None] + tl.zeros((1, BAND_TILE), tl.int32)
            diag = r[None, :] == 0
        valid = row_mask[:, None] & r_mask[None, :] & (j >= 0) & (j < n)
        if UNIT:
            valid &= ~diag
        safe_j = tl.where(valid, j, 0)
        safe_col = tl.where(valid, col, 0)
        a_off = (r[None, :] + safe_col * LDA) * 2
        x_off = safe_j * 2
        ar = tl.load(a_ptr + a_off, mask=valid, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=valid, other=0.0)
        xr = tl.load(xin_ptr + x_off, mask=valid, other=0.0)
        xi = tl.load(xin_ptr + x_off + 1, mask=valid, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)
    if UNIT:
        acc_r += xin_r
        acc_i += xin_i
    out = rows * INCX * 2
    tl.store(x_ptr + out, acc_r, mask=row_mask)
    tl.store(x_ptr + out + 1, acc_i, mask=row_mask)


def stbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.float32 == x.dtype
    _common._check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return x
    uplo, trans, _ = _common._row_major_tbmv_args(uplo, trans)
    with torch_device_fn.device(A.device):
        xin = x.as_strided((n,), (incx,)).clone()
        stbmv_kernel[(triton.cdiv(n, 32),)](
            A,
            xin,
            x,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=int(trans != CUBLAS_OP_N),
            UNIT=int(diag == CUBLAS_DIAG_UNIT),
            BLOCK_M=32,
            BAND_TILE=8,
            num_warps=1,
        )
    return x


def ctbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.complex64 == x.dtype
    _common._check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return x
    uplo, trans, conj = _common._row_major_tbmv_args(uplo, trans)
    with torch_device_fn.device(A.device):
        xin = x.as_strided((n,), (incx,)).clone()
        ctbmv_kernel[(triton.cdiv(n, 32),)](
            torch.view_as_real(A),
            torch.view_as_real(xin),
            torch.view_as_real(x),
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=int(trans != CUBLAS_OP_N),
            UNIT=int(diag == CUBLAS_DIAG_UNIT),
            CONJ=conj,
            BLOCK_M=32,
            BAND_TILE=8,
            num_warps=1,
        )
    return x


__all__ = ["stbmv", "ctbmv"]

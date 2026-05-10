import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_DIAG_UNIT = 1


_TBSV_KEY = ["n", "k_bucket", "mode_key"]
_TBSV_RESTORE = ["x_ptr"]


def _band_bucket(k: int) -> int:
    if k <= 1:
        return 1
    b = 1
    while b < k and b < 1024:
        b <<= 1
    return b


def _mode_key(uplo: int, trans: int, unit: int) -> int:
    return (uplo << 4) | (trans << 2) | unit


# --------------------------------------------------------------------------
# Kernel — Iluvatar-tuned
# --------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
)
@triton.jit
def stbsv_kernel(
    a_ptr, x_ptr,
    n, k, LDA, INCX,
    k_bucket, mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)

    # ----------------------------------------------------------------------
    # Lower / NoTrans : forward substitution
    # a_off = d + j*LDA
    # ----------------------------------------------------------------------
    if (UPLO == 0) and (TRANS == 0):
        for j in tl.range(0, n):
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j + d
                m = (d <= k) & (i < n)
                a_off = d + j * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                xv = xv - av * xj
                tl.store(x_ptr + i * INCX, xv, mask=m)

    # ----------------------------------------------------------------------
    # Upper / NoTrans : back substitution -> backward propagation
    # ----------------------------------------------------------------------
    elif (UPLO == 1) and (TRANS == 0):
        for jc in tl.range(0, n):
            j = n - 1 - jc
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + k + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j - d
                m = (d <= k) & (i >= 0)
                a_off = (k - d) + j * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                xv = xv - av * xj
                tl.store(x_ptr + i * INCX, xv, mask=m)

    # ----------------------------------------------------------------------
    # Upper / Trans : forward substitution,dot-product
    # a_off = (k-d) + j*LDA
    # ----------------------------------------------------------------------
    elif (UPLO == 1) and (TRANS == 1):
        for j in tl.range(0, n):
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j - d
                m = (d <= k) & (i >= 0)
                a_off = (k - d) + j * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                acc += av * xv
            s = tl.sum(acc, axis=0)
            xj = tl.load(x_ptr + j * INCX) - s
            if not UNIT:
                ajj = tl.load(a_ptr + k + j * LDA)
                xj = xj / ajj
            tl.store(x_ptr + j * INCX, xj)

    # ----------------------------------------------------------------------
    # Lower / Trans : back substitution,dot-product
    # a_off = d + j*LDA
    # ----------------------------------------------------------------------
    else:
        for jc in tl.range(0, n):
            j = n - 1 - jc
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j + d
                m = (d <= k) & (i < n)
                a_off = d + j * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                acc += av * xv
            s = tl.sum(acc, axis=0)
            xj = tl.load(x_ptr + j * INCX) - s
            if not UNIT:
                ajj = tl.load(a_ptr + j * LDA)
                xj = xj / ajj
            tl.store(x_ptr + j * INCX, xj)


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------
def _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok):
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
    assert n >= 0 and k >= 0
    assert lda >= k + 1
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= n * lda


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def stbsv(
    uplo: int, trans: int, diag: int, n: int, k: int,
    A: torch.Tensor, lda: int,
    x: torch.Tensor, incx: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        grid = (1,)
        stbsv_kernel[grid](
            A, x,
            n, k, lda, incx,
            _band_bucket(k + 1), _mode_key(uplo, trans_flag, unit),
            UPLO=uplo, TRANS=trans_flag, UNIT=unit,
        )
import importlib

import torch

from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT, CUBLAS_OP_N
from flag_blas.runtime import torch_device_fn

_common = importlib.import_module("flag_blas.ops.level2.tbsv")


def stbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.float32 == x.dtype
    _common._check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    uplo, trans, _ = _common._row_major_tbsv_args(uplo, trans)
    with torch_device_fn.device(A.device):
        _common._real_tbsv_kernel[(1,)](
            A,
            x,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=int(trans != CUBLAS_OP_N),
            UNIT=int(diag == CUBLAS_DIAG_UNIT),
        )


def ctbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.complex64 == x.dtype
    _common._check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return
    uplo, trans, conj = _common._row_major_tbsv_args(uplo, trans)
    with torch_device_fn.device(A.device):
        _common._complex_tbsv_kernel[(1,)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans,
            UNIT=int(diag == CUBLAS_DIAG_UNIT),
            CONJ=conj,
        )

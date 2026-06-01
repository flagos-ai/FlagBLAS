"""
BLAS Level 1, Level 2, and Level 3 operations
"""

from flag_blas.ops.level1.abs import cabs, dabs, sabs, zabs
from flag_blas.ops.level1.amax import camax, damax, samax, zamax
from flag_blas.ops.level1.amin import camin, damin, samin, zamin
from flag_blas.ops.level1.asum import dasum, dzasum, sasum, scasum
from flag_blas.ops.level1.axpy import caxpy, daxpy, saxpy, zaxpy
from flag_blas.ops.level1.copy import ccopy, dcopy, scopy, zcopy
from flag_blas.ops.level1.dot import ddot, sdot
from flag_blas.ops.level1.dotc import cdotc, zdotc
from flag_blas.ops.level1.dotu import cdotu, zdotu
from flag_blas.ops.level1.nrm2 import dnrm2, dznrm2, scnrm2, snrm2
from flag_blas.ops.level1.rot import crot, drot, srot, zrot
from flag_blas.ops.level1.scal import cscal, csscal, dscal, sscal, zdscal, zscal
from flag_blas.ops.level1.swap import cswap, dswap, sswap, zswap
from flag_blas.ops.level2._constants import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.ops.level2.gbmv import cgbmv, dgbmv, sgbmv, zgbmv
from flag_blas.ops.level2.gemv import (
    bfgemv,
    cgemv,
    dgemv,
    fp8_gemv,
    hgemv,
    sgemv,
    zgemv,
)
from flag_blas.ops.level2.hbmv import chbmv, zhbmv
from flag_blas.ops.level2.hemv import chemv, zhemv
from flag_blas.ops.level2.sbmv import dsbmv, ssbmv
from flag_blas.ops.level2.spmv import dspmv, sspmv
from flag_blas.ops.level2.symv import csymv, dsymv, ssymv, zsymv
from flag_blas.ops.level2.tbmv import ctbmv, dtbmv, stbmv, ztbmv
from flag_blas.ops.level2.tbsv import stbsv
from flag_blas.ops.level2.tpmv import ctpmv, dtpmv, stpmv, ztpmv
from flag_blas.ops.level2.trmv import ctrmv, dtrmv, strmv, ztrmv
from flag_blas.ops.level3.gemm import bfgemm, fp8gemm, hgemm, sgemm
from flag_blas.ops.level3.group_gemm import group_gemm, group_mm

__all__ = [
    # amax
    "samax",
    "damax",
    "camax",
    "zamax",
    # amin
    "samin",
    "damin",
    "camin",
    "zamin",
    # asum
    "dasum",
    "dzasum",
    "sasum",
    "scasum",
    # dot
    "sdot",
    "ddot",
    # dotc
    "cdotc",
    "zdotc",
    # dotu
    "cdotu",
    "zdotu",
    # nrm2
    "snrm2",
    "dnrm2",
    "scnrm2",
    "dznrm2",
    # axpy
    "caxpy",
    "daxpy",
    "saxpy",
    "zaxpy",
    # scal
    "zscal",
    "cscal",
    "dscal",
    "sscal",
    "csscal",
    "zdscal",
    # rot
    "srot",
    "drot",
    "crot",
    "zrot",
    # gemv
    "sgemv",
    "dgemv",
    "cgemv",
    "zgemv",
    "hgemv",
    "bfgemv",
    "fp8_gemv",
    # gbmv
    "sgbmv",
    "dgbmv",
    "cgbmv",
    "zgbmv",
    # sbmv
    "ssbmv",
    "dsbmv",
    # spmv
    "sspmv",
    "dspmv",
    # hbmv
    "chbmv",
    "zhbmv",
    # symv
    "ssymv",
    "dsymv",
    "csymv",
    "zsymv",
    # trmv
    "strmv",
    "dtrmv",
    "ctrmv",
    "ztrmv",
    # tbmv
    "stbmv",
    "dtbmv",
    "ctbmv",
    "ztbmv",
    # tpmv
    "stpmv",
    "dtpmv",
    "ctpmv",
    "ztpmv",
    # hemv
    "chemv",
    "zhemv",
    "CUBLAS_DIAG_NON_UNIT",
    "CUBLAS_DIAG_UNIT",
    "CUBLAS_FILL_MODE_LOWER",
    "CUBLAS_FILL_MODE_UPPER",
    "CUBLAS_OP_N",
    "CUBLAS_OP_T",
    "CUBLAS_OP_C",
    # gemm
    "sgemm",
    "hgemm",
    "bfgemm",
    "fp8gemm",
    "group_mm",
    "group_gemm",
    # abs
    "sabs",
    "dabs",
    "cabs",
    "zabs",
    # copy
    "scopy",
    "dcopy",
    "ccopy",
    "zcopy",
    # swap
    "sswap",
    "dswap",
    "cswap",
    "zswap",
    # tbsv
    "stbsv",
]

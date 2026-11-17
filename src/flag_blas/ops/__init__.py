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

"""
BLAS Level 1, Level 2, and Level 3 operations
"""

from flag_blas.ops.level1.amax import icamax, idamax, isamax, izamax
from flag_blas.ops.level1.amin import icamin, idamin, isamin, izamin
from flag_blas.ops.level1.asum import dasum, dzasum, sasum, scasum
from flag_blas.ops.level1.axpy import caxpy, daxpy, saxpy, zaxpy
from flag_blas.ops.level1.copy import ccopy, dcopy, scopy, zcopy
from flag_blas.ops.level1.dot import ddot, sdot
from flag_blas.ops.level1.dotc import cdotc, zdotc
from flag_blas.ops.level1.dotu import cdotu, zdotu
from flag_blas.ops.level1.nrm2 import dnrm2, dznrm2, scnrm2, snrm2
from flag_blas.ops.level1.rot import csrot, drot, srot, zdrot
from flag_blas.ops.level1.rotg import crotg, drotg, srotg, zrotg
from flag_blas.ops.level1.rotm import drotm, srotm
from flag_blas.ops.level1.rotmg import drotmg, srotmg
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
from flag_blas.ops.level2.ger import cgerc, cgeru, dger, sger, zgerc, zgeru
from flag_blas.ops.level2.hbmv import chbmv, zhbmv
from flag_blas.ops.level2.hemv import chemv, zhemv
from flag_blas.ops.level2.her import cher, zher
from flag_blas.ops.level2.her2 import cher2, zher2
from flag_blas.ops.level2.hpmv import chpmv, zhpmv
from flag_blas.ops.level2.hpr import chpr, zhpr
from flag_blas.ops.level2.hpr2 import chpr2, zhpr2
from flag_blas.ops.level2.sbmv import dsbmv, ssbmv
from flag_blas.ops.level2.spmv import dspmv, sspmv
from flag_blas.ops.level2.spr import dspr, sspr
from flag_blas.ops.level2.spr2 import dspr2, sspr2
from flag_blas.ops.level2.symv import csymv, dsymv, ssymv, zsymv
from flag_blas.ops.level2.syr import csyr, dsyr, ssyr, zsyr
from flag_blas.ops.level2.syr2 import dsyr2, ssyr2
from flag_blas.ops.level2.tbmv import ctbmv, dtbmv, stbmv, ztbmv
from flag_blas.ops.level2.tbsv import ctbsv, dtbsv, stbsv, ztbsv
from flag_blas.ops.level2.tpmv import ctpmv, dtpmv, stpmv, ztpmv
from flag_blas.ops.level2.tpsv import ctpsv, dtpsv, stpsv, ztpsv
from flag_blas.ops.level2.trmv import ctrmv, dtrmv, strmv, ztrmv
from flag_blas.ops.level2.trsv import ctrsv, dtrsv, strsv, ztrsv
from flag_blas.ops.level3.bfgemm import bfgemm
from flag_blas.ops.level3.cgemm import cgemm
from flag_blas.ops.level3.dgemm import dgemm
from flag_blas.ops.level3.group_gemm import (
    group_bfgemm,
    group_hgemm,
    group_mm,
    group_tf32gemm,
)
from flag_blas.ops.level3.hgemm import hgemm
from flag_blas.ops.level3.sgemm import sgemm
from flag_blas.ops.level3.trmm import (
    CUBLAS_SIDE_LEFT,
    CUBLAS_SIDE_RIGHT,
    ctrmm,
    dtrmm,
    strmm,
    ztrmm,
)
from flag_blas.ops.level3.zgemm import zgemm

__all__ = [
    # amax
    "isamax",
    "idamax",
    "icamax",
    "izamax",
    # amin
    "isamin",
    "idamin",
    "icamin",
    "izamin",
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
    "csrot",
    "zdrot",
    "srotg",
    "drotg",
    "crotg",
    "zrotg",
    "srotm",
    "drotm",
    "srotmg",
    "drotmg",
    # gemv
    "sgemv",
    "dgemv",
    "cgemv",
    "zgemv",
    "hgemv",
    "bfgemv",
    "fp8_gemv",
    # ger
    "sger",
    "dger",
    "cgeru",
    "cgerc",
    "zgeru",
    "zgerc",
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
    # spr
    "sspr",
    "dspr",
    # spr2
    "sspr2",
    "dspr2",
    # hbmv
    "chbmv",
    "zhbmv",
    # symv
    "ssymv",
    "dsymv",
    "csymv",
    "zsymv",
    # syr2
    "ssyr2",
    "dsyr2",
    # "csyr2",  # disabled: no direct SciPy/CPU BLAS SYR2 reference
    # "zsyr2",  # disabled: no direct SciPy/CPU BLAS SYR2 reference
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
    # her2
    "cher2",
    "zher2",
    # hpmv
    "chpmv",
    "zhpmv",
    # hpr
    "chpr",
    "zhpr",
    # hpr2
    "chpr2",
    "zhpr2",
    "CUBLAS_DIAG_NON_UNIT",
    "CUBLAS_DIAG_UNIT",
    "CUBLAS_FILL_MODE_LOWER",
    "CUBLAS_FILL_MODE_UPPER",
    "CUBLAS_OP_N",
    "CUBLAS_OP_T",
    "CUBLAS_OP_C",
    "CUBLAS_SIDE_LEFT",
    "CUBLAS_SIDE_RIGHT",
    # gemm
    "sgemm",
    "dgemm",
    "cgemm",
    "zgemm",
    "hgemm",
    "bfgemm",
    "group_mm",
    "group_hgemm",
    "group_bfgemm",
    "group_tf32gemm",
    "strmm",
    "dtrmm",
    "ctrmm",
    "ztrmm",
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
    # trsv
    "strsv",
    "dtrsv",
    "ctrsv",
    "ztrsv",
    "stpsv",
    "dtpsv",
    "ctpsv",
    "ztpsv",
    "dtbsv",
    "ctbsv",
    "ztbsv",
    "ssyr",
    "dsyr",
    "csyr",
    "zsyr",
    "cher",
    "zher",
]

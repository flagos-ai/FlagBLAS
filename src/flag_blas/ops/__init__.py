"""
BLAS Level 1, Level 2, and Level 3 operations
"""

from flag_blas.ops.level1.amax import (
    camax,
    damax,
    samax,
    zamax,
)
from flag_blas.ops.level1.asum import (
    dasum,
    dzasum,
    sasum,
    scasum,
)
from flag_blas.ops.level1.nrm2 import (
    snrm2,
    dnrm2,
    scnrm2,
    dznrm2
)
from flag_blas.ops.level1.axpy import (
    caxpy,
    daxpy,
    saxpy,
    zaxpy,
)
from flag_blas.ops.level1.rot import (
    srot,
    drot,
    crot,
    zrot,
)
from flag_blas.ops.level1.scal import (
    zscal,
    cscal,
    dscal,
    sscal,
    csscal,
    zdscal
)
from flag_blas.ops.level2._constants import (
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUBLAS_OP_C,
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
)
from flag_blas.ops.level2.gemv import (
    sgemv,
    dgemv,
    cgemv,
    zgemv,
    hgemv,
    bfgemv,
    fp8_gemv,
)
from flag_blas.ops.level2.gbmv import (
    sgbmv,
    dgbmv,
    cgbmv,
    zgbmv,
)
from flag_blas.ops.level2.symv import (
    ssymv,
    dsymv,
    csymv,
    zsymv,
)
from flag_blas.ops.level3.gemm import (
    sgemm,
    hgemm,
    bfgemm,
    fp8gemm,
)

from flag_blas.ops.level1.abs import (
    sabs,
    dabs,
    cabs,
    zabs,
)
from flag_blas.ops.level1.copy import (
    scopy,
    dcopy,
    ccopy,
    zcopy,
)

__all__ = [
    # amax
    "samax",
    "damax",
    "camax",
    "zamax",
    # asum
    "dasum",
    "dzasum",
    "sasum",
    "scasum",
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
    # symv
    "ssymv",
    "dsymv",
    "csymv",
    "zsymv",
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
]

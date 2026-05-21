import ctypes
import ctypes.util
import math

import cupy as cp
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor
from .conftest import TO_CPU


def load_cublas():
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)
    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = load_cublas()


def cublas_trmv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if n == 0:
        return
    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStrmv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtrmv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtrmv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtrmv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtrmv_v2 execution failed with error code: {status}")


def cpu_trmv_reference(uplo, trans, diag, n, A, lda, x, incx):
    ref_x = to_cpu_blas_tensor(x)
    if n == 0:
        return ref_x

    ref_A = to_cpu_blas_tensor(A)
    func = cpu_blas.ztrmv if ref_A.dtype.is_complex else cpu_blas.dtrmv
    xout = func(
        ref_A[:n, :n].T.numpy(),
        ref_x.numpy(),
        incx=incx,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        trans=trans,
        diag=diag,
        overwrite_x=1,
    )
    return torch.from_numpy(xout)


def trmv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if TO_CPU:
        return cpu_trmv_reference(uplo, trans, diag, n, A, lda, x, incx)

    ref_x = x.clone()
    cublas_trmv_reference(uplo, trans, diag, n, A, lda, ref_x, incx)
    return ref_x


TRMV_SIZES = [
    0,
    1,
    8,
    31,
    64,
    127,
    128,
    192,
    256,
    384,
    512,
    768,
    1023,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    10000,
    16384,
]
TRMV_STRIDE_SIZES = [64, 127, 256]
INCS = [1, 2, 3]
LDA_EXTRAS = [0, 2]
LDA_EXTRAS_STRIDE = [0, 1]


def make_triangular(n, lda, uplo, diag, dtype, device):
    """Fill the active triangle with random values and poison everything else
    (the ignored triangle, the diagonal when UNIT, and the lda-vs-n padding
    region) with NaN. A correct kernel masks those cells out; any stray read
    propagates NaN and fails assert_close. cuBLAS and flag_blas both skip
    those cells by spec, so the poison is safe for the reference."""
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    A = torch.randn((n, lda), dtype=dtype, device=device)
    if not dtype.is_complex:
        A = A * 0.1
    rows = torch.arange(lda, device=device).view(1, lda)
    cols = torch.arange(n, device=device).view(n, 1)
    unit = diag == CUBLAS_DIAG_UNIT

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = rows <= cols
        if unit:
            valid &= rows != cols
    else:
        valid = (rows >= cols) & (rows < n)
        if unit:
            valid &= rows != cols

    if dtype.is_complex:
        torch.view_as_real(A).masked_fill_(~valid.unsqueeze(-1), float("nan"))
    else:
        A.masked_fill_(~valid, float("nan"))
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _trmv_tol(dtype, n):
    K = max(1, n)
    if dtype == torch.float32:
        return min(max(1e-5, 1e-6 * math.sqrt(K)), 1e-3)
    if dtype == torch.float64:
        return min(max(1e-13, 1e-14 * math.sqrt(K)), 1e-11)
    if dtype == torch.complex64:
        return min(max(2e-5, 8e-6 * math.sqrt(K)), 2e-3)
    if dtype == torch.complex128:
        return min(max(2e-13, 2e-14 * math.sqrt(K)), 2e-11)
    raise ValueError(f"Unsupported dtype {dtype}")


UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


@pytest.mark.strmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_strmv(n, uplo, trans, diag, lda_extra):
    dtype = torch.float32
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ops.strmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.strmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_strmv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.float32
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ops.strmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_dtrmv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ops.dtrmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_dtrmv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ops.dtrmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ctrmv(n, uplo, trans, diag, lda_extra):
    dtype = torch.complex64
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ops.ctrmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ctrmv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.complex64
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ops.ctrmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ztrmv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ops.ztrmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ztrmv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ops.ztrmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)

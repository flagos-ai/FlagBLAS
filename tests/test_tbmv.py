import ctypes
import ctypes.util
import math
import pytest
import torch
import cupy as cp
import flag_blas
from flag_blas.ops import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUBLAS_OP_C,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
)


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


def cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0:
        return
    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStbmv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtbmv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtbmv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtbmv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtbmv_v2 execution failed with error code: {status}")


TBMV_SIZES = [
    0,
    1,
    31,
    256,
    4096,
    16384,
]
TBMV_STRIDE_SIZES = [64, 256]
TBMV_KS = [0, 1, 16, 256]
INCS = [1, 2]
LDA_EXTRAS = [0, 2]
LDA_EXTRAS_STRIDE = [0, 1]


def make_triangular_banded(n, k, lda, uplo, diag, dtype, device):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    A = torch.randn((n, lda), dtype=dtype, device=device)
    cols = torch.arange(lda, device=device).view(1, lda)
    j = torch.arange(n, device=device).view(n, 1)
    unit = diag == CUBLAS_DIAG_UNIT

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = (cols >= torch.clamp(k - j, min=0)) & (cols <= k)
        if unit:
            valid &= cols != k
    else:
        valid = cols <= torch.clamp(n - 1 - j, max=k)
        if unit:
            valid &= cols != 0

    if dtype.is_complex:
        torch.view_as_real(A).masked_fill_(~valid.unsqueeze(-1), float("nan"))
    else:
        A.masked_fill_(~valid, float("nan"))
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


def _effective_k(n, k):
    return min(k, max(0, n - 1))


def _tbmv_tol(dtype, k):
    K = max(1, k + 1)
    if dtype == torch.float32:
        return min(max(1e-5, 2e-6 * math.sqrt(K)), 1e-3)
    if dtype == torch.float64:
        return min(max(1e-13, 1e-14 * math.sqrt(K)), 1e-11)
    if dtype == torch.complex64:
        return min(max(2e-5, 2e-6 * math.sqrt(K)), 2e-3)
    if dtype == torch.complex128:
        return min(max(2e-13, 2e-14 * math.sqrt(K)), 2e-11)
    raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.stbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_stbmv(n, k, uplo, trans, diag, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.float32
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, 1)
    flag_blas.ops.stbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.stbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_stbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.float32
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    flag_blas.ops.stbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.dtbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_dtbmv(n, k, uplo, trans, diag, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, 1)
    flag_blas.ops.dtbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.dtbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_dtbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    flag_blas.ops.dtbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.ctbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ctbmv(n, k, uplo, trans, diag, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.complex64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, 1)
    flag_blas.ops.ctbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.ctbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ctbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.complex64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    flag_blas.ops.ctbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.ztbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ztbmv(n, k, uplo, trans, diag, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, 1)
    flag_blas.ops.ztbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.ztbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ztbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    flag_blas.ops.ztbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _tbmv_tol(dtype, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)

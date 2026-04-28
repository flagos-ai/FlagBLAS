import ctypes
import ctypes.util
import math
import pytest
import torch
import cupy as cp
import flag_blas
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

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

class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

def cublas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype

    if dtype == torch.float32:
        func = _cublas.cublasSsymv_v2
        alpha_c = ctypes.c_float(alpha)
        beta_c  = ctypes.c_float(beta)
    elif dtype == torch.float64:
        func = _cublas.cublasDsymv_v2
        alpha_c = ctypes.c_double(alpha)
        beta_c  = ctypes.c_double(beta)
    elif dtype == torch.complex64:
        func = _cublas.cublasCsymv_v2
        alpha_c = cuComplex(alpha.real, alpha.imag)
        beta_c  = cuComplex(beta.real, beta.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZsymv_v2
        alpha_c = cuDoubleComplex(alpha.real, alpha.imag)
        beta_c  = cuDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy)
    )
    if status != 0:
        raise RuntimeError(f"cublasXsymv_v2 execution failed with error code: {status}")

SYMV_SIZES = [
    1, 2, 15, 32, 63, 64, 65, 127, 128, 192, 255, 256, 384, 512, 768, 1024,
    1023, 1025, 1536, 2048, 3072, 4095, 4096, 6144, 8192, 9999, 10000, 12288,
    16384,
]

SYMV_STRIDE_SIZES = [64, 127, 256]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]
def create_symv_data(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    data = torch.randn(n, n, dtype=dtype, device=device)
    A[:, :n] = data
    return A.contiguous()

def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _symv_tol(dtype, n):
    K = max(1, n)
    if dtype == torch.float32:
        return min(max(1e-5, 2e-6 * math.sqrt(K)), 1e-3)
    if dtype == torch.float64:
        return min(max(1e-13, 1e-14 * math.sqrt(K)), 1e-11)
    if dtype == torch.complex64:
        return min(max(2e-5, 4e-6 * math.sqrt(K)), 2e-3)
    if dtype == torch.complex128:
        return min(max(2e-13, 2e-14 * math.sqrt(K)), 2e-11)
    raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.ssymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_ssymv(n, uplo, beta):
    dtype, alpha = torch.float32, 1.5
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, 1, beta, ref_y, 1)
    flag_blas.ops.ssymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.ssymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_ssymv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.float32, 2.0, 0.5
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    flag_blas.ops.ssymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.ssymv
def test_ssymv_alpha_zero():
    n, lda = 256, 258
    dtype = torch.float32
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y_ref, 1)
    flag_blas.ops.ssymv(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y, 1)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * 2.0)

@pytest.mark.ssymv
def test_ssymv_beta_zero():
    n, lda = 256, 256
    dtype = torch.float32
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, ref_y_nan, 1)
    flag_blas.ops.ssymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.ssymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_zero, 1)
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.dsymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_dsymv(n, uplo, beta):
    check_fp64_support()
    dtype, alpha = torch.float64, 1.5
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, 1, beta, ref_y, 1)
    flag_blas.ops.dsymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.dsymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dsymv_stride(n, uplo, incx, incy):
    check_fp64_support()
    dtype, alpha, beta = torch.float64, 2.0, 0.5
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    flag_blas.ops.dsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.dsymv
def test_dsymv_alpha_zero():
    check_fp64_support()
    n, lda = 256, 258
    dtype = torch.float64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y_ref, 1)
    flag_blas.ops.dsymv(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y, 1)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * 2.0)

@pytest.mark.dsymv
def test_dsymv_beta_zero():
    check_fp64_support()
    n, lda = 256, 256
    dtype = torch.float64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, ref_y_nan, 1)
    flag_blas.ops.dsymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.dsymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_zero, 1)
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.csymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_csymv(n, uplo, beta):
    dtype, alpha = torch.complex64, 1.5 + 0.5j
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, 1, beta, ref_y, 1)
    flag_blas.ops.csymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.csymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_csymv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    flag_blas.ops.csymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.csymv
def test_csymv_alpha_zero():
    n, lda = 256, 258
    dtype = torch.complex64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0+1.0j, y_ref, 1)
    flag_blas.ops.csymv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0+1.0j, y, 1)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * (2.0+1.0j))

@pytest.mark.csymv
def test_csymv_beta_zero():
    n, lda = 256, 256
    dtype = torch.complex64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_LOWER, n, 1.0+0.5j, A, lda, x, 1, 0.0j, ref_y_nan, 1)
    flag_blas.ops.csymv(CUBLAS_FILL_MODE_LOWER, n, 1.0+0.5j, A, lda, x, 1, 0.0j, y_nan, 1)
    flag_blas.ops.csymv(CUBLAS_FILL_MODE_LOWER, n, 1.0+0.5j, A, lda, x, 1, 0.0j, y_zero, 1)
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.zsymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_zsymv(n, uplo, beta):
    check_fp64_support()
    dtype, alpha = torch.complex128, 1.5 + 0.5j
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, 1, beta, ref_y, 1)
    flag_blas.ops.zsymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.zsymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zsymv_stride(n, uplo, incx, incy):
    check_fp64_support()
    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    flag_blas.ops.zsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    tol = _symv_tol(dtype, n)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)

@pytest.mark.zsymv
def test_zsymv_alpha_zero():
    check_fp64_support()
    n, lda = 256, 258
    dtype = torch.complex128
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0+1.0j, y_ref, 1)
    flag_blas.ops.zsymv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0+1.0j, y, 1)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * (2.0+1.0j))

@pytest.mark.zsymv
def test_zsymv_beta_zero():
    check_fp64_support()
    n, lda = 256, 256
    dtype = torch.complex128
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_symv_reference(CUBLAS_FILL_MODE_LOWER, n, 1.0+0.5j, A, lda, x, 1, 0.0j, ref_y_nan, 1)
    flag_blas.ops.zsymv(CUBLAS_FILL_MODE_LOWER, n, 1.0+0.5j, A, lda, x, 1, 0.0j, y_nan, 1)
    flag_blas.ops.zsymv(CUBLAS_FILL_MODE_LOWER, n, 1.0+0.5j, A, lda, x, 1, 0.0j, y_zero, 1)
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.parametrize(
    "dtype, op, alpha, beta",
    [
        (torch.float32, flag_blas.ops.ssymv, 1.5, 0.5),
        (torch.float64, flag_blas.ops.dsymv, 1.5, 0.5),
        (torch.complex64, flag_blas.ops.csymv, 1.5 + 0.5j, 0.5 + 0.25j),
        (torch.complex128, flag_blas.ops.zsymv, 1.5 + 0.5j, 0.5 + 0.25j),
    ],
)
def test_symv_n_zero(dtype, op, alpha, beta):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()

    A = torch.empty((0, 2), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, A, 2, x, 1, beta, y, 1)
    assert y.numel() == 0


@pytest.mark.parametrize(
    "dtype, op, alpha, beta, uplo",
    [
        (torch.float32, flag_blas.ops.ssymv, 1.25, 0.5, CUBLAS_FILL_MODE_UPPER),
        (torch.float32, flag_blas.ops.ssymv, 1.25, 0.5, CUBLAS_FILL_MODE_LOWER),
        (torch.float64, flag_blas.ops.dsymv, 1.25, 0.5, CUBLAS_FILL_MODE_UPPER),
        (torch.float64, flag_blas.ops.dsymv, 1.25, 0.5, CUBLAS_FILL_MODE_LOWER),
        (torch.complex64, flag_blas.ops.csymv, 1.25 + 0.5j, 0.5 + 0.25j, CUBLAS_FILL_MODE_UPPER),
        (torch.complex64, flag_blas.ops.csymv, 1.25 + 0.5j, 0.5 + 0.25j, CUBLAS_FILL_MODE_LOWER),
        (torch.complex128, flag_blas.ops.zsymv, 1.25 + 0.5j, 0.5 + 0.25j, CUBLAS_FILL_MODE_UPPER),
        (torch.complex128, flag_blas.ops.zsymv, 1.25 + 0.5j, 0.5 + 0.25j, CUBLAS_FILL_MODE_LOWER),
    ],
)
def test_symv_ignored_triangle(dtype, op, alpha, beta, uplo):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()

    n = 64
    lda = n + 3
    A_clean = create_symv_data(n, lda, dtype, flag_blas.device)
    A_dirty = A_clean.clone()
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_clean = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_dirty = y_clean.clone()

    tri_upper = torch.triu_indices(n, n, offset=1, device=flag_blas.device)
    tri_lower = torch.tril_indices(n, n, offset=-1, device=flag_blas.device)
    dirty_vals = torch.full(
        (tri_upper.shape[1],),
        complex(float("nan"), float("nan")) if dtype.is_complex else float("nan"),
        dtype=dtype,
        device=flag_blas.device,
    )
    if uplo == CUBLAS_FILL_MODE_UPPER:
        A_dirty[tri_upper[0], tri_upper[1]] = dirty_vals
    else:
        A_dirty[tri_lower[0], tri_lower[1]] = dirty_vals[: tri_lower.shape[1]]

    op(uplo, n, alpha, A_clean, lda, x, 1, beta, y_clean, 1)
    op(uplo, n, alpha, A_dirty, lda, x, 1, beta, y_dirty, 1)

    torch.testing.assert_close(y_dirty, y_clean)

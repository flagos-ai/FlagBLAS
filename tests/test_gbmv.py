import ctypes
import ctypes.util
import math
import pytest
import torch
import cupy as cp
import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C


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


def cublas_gbmv_reference(trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    dtype = AB.dtype

    if dtype == torch.float32:
        func = _cublas.cublasSgbmv_v2
        alpha_c = ctypes.c_float(alpha)
        beta_c = ctypes.c_float(beta)
    elif dtype == torch.float64:
        func = _cublas.cublasDgbmv_v2
        alpha_c = ctypes.c_double(alpha)
        beta_c = ctypes.c_double(beta)
    elif dtype == torch.complex64:
        func = _cublas.cublasCgbmv_v2
        alpha_c = cuComplex(alpha.real, alpha.imag)
        beta_c = cuComplex(beta.real, beta.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZgbmv_v2
        alpha_c = cuDoubleComplex(alpha.real, alpha.imag)
        beta_c = cuDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(kl),
        ctypes.c_int(ku),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(AB.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXgbmv_v2 execution failed with error code: {status}")


GBMV_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (63, 63),
    (127, 127),
    (4095, 4095),
    (1024, 4096),
    (4096, 1024),
    (127, 255),
    (4096, 4096),
    (1, 65536),
    (65536, 64),
]

GBMV_STRIDE_SHAPES = [(64, 128), (128, 64), (256, 256)]

GBMV_BANDS = [
    (0, 0),
    (1, 1),
    (2, 5),
    (10, 0),
    (0, 10),
    (32, 32),
]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def create_banded_data(m, n, kl, ku, lda, dtype, device):
    A_dense = torch.randn(m, n, dtype=dtype, device=device)

    AB = torch.zeros((n, lda), dtype=dtype, device=device)
    for d in range(-ku, kl + 1):
        j_min = max(0, -d)
        j_max = min(n, m - d)
        if j_min < j_max:
            j_idx = torch.arange(j_min, j_max, device=device)
            i_idx = j_idx + d
            AB[j_idx, ku + d] = A_dense[i_idx, j_idx]

    return AB.contiguous()


def get_effective_bands(m, n, kl, ku):
    actual_kl = min(kl, max(0, m - 1))
    actual_ku = min(ku, max(0, n - 1))
    is_truncated = (actual_kl != kl) or (actual_ku != ku)
    return actual_kl, actual_ku, is_truncated


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _gbmv_tol(dtype, kl, ku):
    K = max(1, kl + ku + 1)
    if dtype == torch.float32:
        return min(max(1e-5, 1e-6 * math.sqrt(K)), 1e-3)
    if dtype == torch.float64:
        return min(max(1e-13, 1e-14 * math.sqrt(K)), 1e-11)
    if dtype == torch.complex64:
        return min(max(2e-5, 2e-6 * math.sqrt(K)), 2e-3)
    if dtype == torch.complex128:
        return min(max(2e-13, 2e-14 * math.sqrt(K)), 2e-11)
    raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.sgbmv
@pytest.mark.parametrize("m,n", GBMV_SHAPES)
@pytest.mark.parametrize("kl,ku", GBMV_BANDS)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_sgbmv(m, n, kl, ku, trans, beta):
    actual_kl, actual_ku, is_truncated = get_effective_bands(m, n, kl, ku)
    if is_truncated and max(kl, ku) > max(m, n):
        pytest.skip("Skipping redundant wide-band test.")

    dtype, alpha = torch.float32, 1.5
    lda = actual_kl + actual_ku + 1 + 2

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, ref_y, 1
    )
    flag_blas.ops.sgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.sgbmv
@pytest.mark.parametrize("m,n", GBMV_STRIDE_SHAPES)
@pytest.mark.parametrize("kl,ku", [(2, 2), (10, 5)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_sgbmv_stride(m, n, kl, ku, trans, incx, incy):
    actual_kl, actual_ku, _ = get_effective_bands(m, n, kl, ku)
    dtype, alpha, beta = torch.float32, 2.0, 0.5
    lda = actual_kl + actual_ku + 1

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, ref_y, incy
    )
    flag_blas.ops.sgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.sgbmv
def test_sgbmv_alpha_zero():
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.float32
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(m, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_gbmv_reference(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y_ref, 1)
    flag_blas.ops.sgbmv(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y, 1)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * 2.0)


@pytest.mark.sgbmv
def test_sgbmv_beta_zero():
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.float32
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, ref_y_nan, 1
    )
    flag_blas.ops.sgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.sgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_zero, 1)
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.dgbmv
@pytest.mark.parametrize("m,n", GBMV_SHAPES)
@pytest.mark.parametrize("kl,ku", GBMV_BANDS)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_dgbmv(m, n, kl, ku, trans, beta):
    check_fp64_support()
    actual_kl, actual_ku, is_truncated = get_effective_bands(m, n, kl, ku)
    if is_truncated and max(kl, ku) > max(m, n):
        pytest.skip("Skipping redundant wide-band test.")

    dtype, alpha = torch.float64, 1.5
    lda = actual_kl + actual_ku + 1

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, ref_y, 1
    )
    flag_blas.ops.dgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.dgbmv
@pytest.mark.parametrize("m,n", GBMV_STRIDE_SHAPES)
@pytest.mark.parametrize("kl,ku", [(2, 2), (10, 5)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dgbmv_stride(m, n, kl, ku, trans, incx, incy):
    check_fp64_support()
    actual_kl, actual_ku, _ = get_effective_bands(m, n, kl, ku)
    dtype, alpha, beta = torch.float64, 2.0, 0.5
    lda = actual_kl + actual_ku + 1

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, ref_y, incy
    )
    flag_blas.ops.dgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.dgbmv
def test_dgbmv_alpha_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.float64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(m, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_gbmv_reference(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y_ref, 1)
    flag_blas.ops.dgbmv(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y, 1)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * 2.0)


@pytest.mark.dgbmv
def test_dgbmv_beta_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.float64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, ref_y_nan, 1
    )
    flag_blas.ops.dgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.dgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_zero, 1)
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.cgbmv
@pytest.mark.parametrize("m,n", GBMV_SHAPES)
@pytest.mark.parametrize("kl,ku", GBMV_BANDS)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_cgbmv(m, n, kl, ku, trans, beta):
    actual_kl, actual_ku, is_truncated = get_effective_bands(m, n, kl, ku)
    if is_truncated and max(kl, ku) > max(m, n):
        pytest.skip("Skipping redundant wide-band test.")

    dtype, alpha = torch.complex64, 1.5 + 0.5j
    lda = actual_kl + actual_ku + 1

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, ref_y, 1
    )
    flag_blas.ops.cgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.cgbmv
@pytest.mark.parametrize("m,n", GBMV_STRIDE_SHAPES)
@pytest.mark.parametrize("kl,ku", [(2, 2), (10, 5)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_cgbmv_stride(m, n, kl, ku, trans, incx, incy):
    actual_kl, actual_ku, _ = get_effective_bands(m, n, kl, ku)
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j
    lda = actual_kl + actual_ku + 1

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, ref_y, incy
    )
    flag_blas.ops.cgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.cgbmv
def test_cgbmv_alpha_zero():
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.complex64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(m, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y_ref, 1
    )
    flag_blas.ops.cgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * (2.0 + 1.0j))


@pytest.mark.cgbmv
def test_cgbmv_beta_zero():
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.complex64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, ref_y_nan, 1
    )
    flag_blas.ops.cgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.ops.cgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_zero, 1
    )
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.zgbmv
@pytest.mark.parametrize("m,n", GBMV_SHAPES)
@pytest.mark.parametrize("kl,ku", GBMV_BANDS)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_zgbmv(m, n, kl, ku, trans, beta):
    check_fp64_support()
    actual_kl, actual_ku, is_truncated = get_effective_bands(m, n, kl, ku)
    if is_truncated and max(kl, ku) > max(m, n):
        pytest.skip("Skipping redundant wide-band test.")

    dtype, alpha = torch.complex128, 1.5 + 0.5j
    lda = actual_kl + actual_ku + 1

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, ref_y, 1
    )
    flag_blas.ops.zgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.zgbmv
@pytest.mark.parametrize("m,n", GBMV_STRIDE_SHAPES)
@pytest.mark.parametrize("kl,ku", [(2, 2), (10, 5)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zgbmv_stride(m, n, kl, ku, trans, incx, incy):
    check_fp64_support()
    actual_kl, actual_ku, _ = get_effective_bands(m, n, kl, ku)
    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j
    lda = actual_kl + actual_ku + 1

    AB = create_banded_data(m, n, actual_kl, actual_ku, lda, dtype, flag_blas.device)
    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = y.clone()

    cublas_gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, ref_y, incy
    )
    flag_blas.ops.zgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    tol = _gbmv_tol(dtype, actual_kl, actual_ku)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.zgbmv
def test_zgbmv_alpha_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.complex128
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(m, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    cublas_gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y_ref, 1
    )
    flag_blas.ops.zgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(y, y_orig * (2.0 + 1.0j))


@pytest.mark.zgbmv
def test_zgbmv_beta_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.complex128
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = y_nan.clone()

    cublas_gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, ref_y_nan, 1
    )
    flag_blas.ops.zgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.ops.zgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_zero, 1
    )
    torch.testing.assert_close(y_nan, ref_y_nan)
    torch.testing.assert_close(y_nan, y_zero)

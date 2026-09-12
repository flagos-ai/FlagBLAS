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

import ctypes
import ctypes.util

import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

if flag_blas.vendor_name in {"hygon", "mthreads"}:
    from .vendor_blas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )
elif flag_blas.vendor_name not in {"ascend", "mthreads"}:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
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


_cublas = (
    None
    if flag_blas.vendor_name in {"ascend", "hygon", "mthreads"}
    else load_cublas()
)


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def row_to_column_hbmv(A, n, k, lda, uplo):
    """Convert row-major Hermitian-band storage to BLAS column-major storage."""
    column_A = torch.zeros((n, lda), dtype=A.dtype, device=A.device)
    for d in range(k + 1):
        count = n - d
        if count <= 0:
            continue
        if uplo == CUBLAS_FILL_MODE_UPPER:
            column_A[d:, k - d] = A[:count, d]
        else:
            column_A[:count, d] = A[d:, k - d]
    return column_A


def hipblas_hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return y

    column_A = row_to_column_hbmv(A, n, k, lda, uplo)
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else complex(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else complex(beta)

    if A.dtype == torch.complex64:
        symbol = "hipblasChbmv_v2"
        alpha_value = HipComplex(alpha.real, alpha.imag)
        beta_value = HipComplex(beta.real, beta.imag)
    elif A.dtype == torch.complex128:
        symbol = "hipblasZhbmv_v2"
        alpha_value = HipDoubleComplex(alpha.real, alpha.imag)
        beta_value = HipDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS HBMV: {A.dtype}")

    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    library, handle = get_hipblas_context(column_A)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    check_hipblas_status(
        function(
            handle,
            hip_uplo,
            n,
            k,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(column_A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        symbol,
    )
    return y


def cublas_hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return

    column_A = row_to_column_hbmv(A, n, k, lda, uplo)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    dtype = A.dtype

    if dtype == torch.complex64:
        func = _cublas.cublasChbmv_v2
        alpha_c = cuComplex(alpha.real, alpha.imag)
        beta_c = cuComplex(beta.real, beta.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZhbmv_v2
        alpha_c = cuDoubleComplex(alpha.real, alpha.imag)
        beta_c = cuDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(column_A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXhbmv_v2 execution failed with error code: {status}")


def cpu_hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return to_cpu_blas_tensor(y)

    ref_A = to_cpu_blas_tensor(row_to_column_hbmv(A, n, k, lda, uplo))
    ref_x = to_cpu_blas_tensor(x)
    if beta == 0 and incy == 1:
        ref_y = torch.empty(y.shape, dtype=torch.complex128)
    else:
        ref_y = to_cpu_blas_tensor(y)

    yout = cpu_blas.zhbmv(
        k,
        alpha,
        ref_A.numpy().T,
        ref_x.numpy(),
        incx=incx,
        beta=beta,
        y=ref_y.numpy(),
        incy=incy,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    if TO_CPU:
        return cpu_hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)

    ref_y = y.clone()
    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        hipblas_hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, ref_y, incy)
    else:
        cublas_hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, ref_y, incy)
    return ref_y


HBMV_SIZES = [
    1,
    2,
    15,
    32,
    63,
    64,
    65,
    127,
    128,
    255,
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
]
HBMV_STRIDE_SIZES = [64, 127, 256]
HBMV_KS = [0, 1, 3, 4, 16, 32, 63, 64, 128, 256]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def hbmv_randn(*shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        normalized = (
            tuple(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (tuple, torch.Size))
            else shape
        )
        values = torch.randn((*normalized, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(*shape, dtype=dtype, device=device)


def make_hermitian_banded(n, k, lda, uplo, dtype, device):
    A = hbmv_randn((n, lda), dtype=dtype, device=device)
    diag_col = 0 if uplo == CUBLAS_FILL_MODE_UPPER else k
    torch.view_as_real(A)[:, diag_col, 1].zero_()
    return A


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _effective_k(n, k):
    return min(k, max(0, n - 1))


def _band_reduce_dim(n, k):
    return max(1, min(2 * k + 1, n))


@pytest.mark.chbmv
@pytest.mark.parametrize("n", HBMV_SIZES)
@pytest.mark.parametrize("k", HBMV_KS)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.5 + 0.25j])
def test_accuracy_chbmv(n, k, uplo, beta):
    k = _effective_k(n, k)
    dtype, alpha = torch.complex64, 1.5 + 0.5j
    lda = k + 1 + 2

    A = make_hermitian_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hbmv_reference(uplo, n, k, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.chbmv(uplo, n, k, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_band_reduce_dim(n, k))


@pytest.mark.chbmv
@pytest.mark.parametrize("n", HBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32, 64])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_chbmv_stride(n, k, uplo, incx, incy):
    k = _effective_k(n, k)
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j
    lda = k + 1

    A = make_hermitian_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = hbmv_randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = hbmv_randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.chbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_band_reduce_dim(n, k))


@pytest.mark.chbmv
def test_chbmv_alpha_zero():
    n, k = 256, 8
    lda = k + 1 + 2
    dtype = torch.complex64
    A = make_hermitian_banded(
        n, k, lda, CUBLAS_FILL_MODE_UPPER, dtype, flag_blas.device
    )
    x = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    y_ref = hbmv_reference(
        CUBLAS_FILL_MODE_UPPER, n, k, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y_ref, 1
    )
    flag_blas.chbmv(CUBLAS_FILL_MODE_UPPER, n, k, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.chbmv
def test_chbmv_beta_zero():
    n, k = 256, 16
    lda = k + 1
    dtype = torch.complex64
    A = make_hermitian_banded(
        n, k, lda, CUBLAS_FILL_MODE_LOWER, dtype, flag_blas.device
    )
    x = hbmv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full(
        (n,), complex(float("nan"), float("nan")), dtype=dtype, device=flag_blas.device
    )
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hbmv_reference(
        CUBLAS_FILL_MODE_LOWER, n, k, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.chbmv(
        CUBLAS_FILL_MODE_LOWER, n, k, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.chbmv(
        CUBLAS_FILL_MODE_LOWER, n, k, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=_band_reduce_dim(n, k))
    blas_assert_close(
        y_nan, to_reference(y_zero), dtype, reduce_dim=_band_reduce_dim(n, k)
    )


@pytest.mark.zhbmv
@pytest.mark.parametrize("n", HBMV_SIZES)
@pytest.mark.parametrize("k", HBMV_KS)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.5 + 0.25j])
def test_accuracy_zhbmv(n, k, uplo, beta):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype, alpha = torch.complex128, 1.5 + 0.5j
    lda = k + 1 + 2

    A = make_hermitian_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hbmv_reference(uplo, n, k, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.zhbmv(uplo, n, k, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_band_reduce_dim(n, k))


@pytest.mark.zhbmv
@pytest.mark.parametrize("n", HBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32, 64])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zhbmv_stride(n, k, uplo, incx, incy):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j
    lda = k + 1

    A = make_hermitian_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = hbmv_randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = hbmv_randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hbmv_reference(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.zhbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_band_reduce_dim(n, k))


@pytest.mark.zhbmv
def test_zhbmv_alpha_zero():
    check_fp64_support()
    n, k = 256, 8
    lda = k + 1 + 2
    dtype = torch.complex128
    A = make_hermitian_banded(
        n, k, lda, CUBLAS_FILL_MODE_UPPER, dtype, flag_blas.device
    )
    x = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hbmv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig, y_ref = y.clone(), y.clone()

    y_ref = hbmv_reference(
        CUBLAS_FILL_MODE_UPPER, n, k, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y_ref, 1
    )
    flag_blas.zhbmv(CUBLAS_FILL_MODE_UPPER, n, k, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.zhbmv
def test_zhbmv_beta_zero():
    check_fp64_support()
    n, k = 256, 16
    lda = k + 1
    dtype = torch.complex128
    A = make_hermitian_banded(
        n, k, lda, CUBLAS_FILL_MODE_LOWER, dtype, flag_blas.device
    )
    x = hbmv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full(
        (n,), complex(float("nan"), float("nan")), dtype=dtype, device=flag_blas.device
    )
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hbmv_reference(
        CUBLAS_FILL_MODE_LOWER, n, k, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.zhbmv(
        CUBLAS_FILL_MODE_LOWER, n, k, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.zhbmv(
        CUBLAS_FILL_MODE_LOWER, n, k, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=_band_reduce_dim(n, k))
    blas_assert_close(
        y_nan, to_reference(y_zero), dtype, reduce_dim=_band_reduce_dim(n, k)
    )


@pytest.mark.parametrize(
    "dtype, op, alpha, beta",
    [
        (torch.complex64, flag_blas.chbmv, 1.5 + 0.5j, 0.5 + 0.25j),
        (torch.complex128, flag_blas.zhbmv, 1.5 + 0.5j, 0.5 + 0.25j),
    ],
)
def test_hbmv_n_zero(dtype, op, alpha, beta):
    if dtype == torch.complex128:
        check_fp64_support()

    A = torch.empty((0, 2), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    op(CUBLAS_FILL_MODE_UPPER, 0, 0, alpha, A, 1, x, 1, beta, y, 1)
    assert y.numel() == 0

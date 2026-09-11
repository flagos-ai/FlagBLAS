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


def _row_to_column_hpmv(AP, uplo):
    column_AP = torch.empty_like(AP)
    column_AP.copy_(AP.conj())
    column_uplo = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    return column_AP, column_uplo


def hipblas_hpmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy):
    if n == 0:
        return y

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else complex(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else complex(beta)

    if AP.dtype == torch.complex64:
        symbol = "hipblasChpmv_v2"
        alpha_value = HipComplex(alpha.real, alpha.imag)
        beta_value = HipComplex(beta.real, beta.imag)
    elif AP.dtype == torch.complex128:
        symbol = "hipblasZhpmv_v2"
        alpha_value = HipDoubleComplex(alpha.real, alpha.imag)
        beta_value = HipDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS HPMV: {AP.dtype}")

    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    library, handle = get_hipblas_context(AP)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
            ctypes.byref(alpha_value),
            ctypes.c_void_p(AP.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        symbol,
    )
    return y


def cublas_hpmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy):
    if n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    dtype = AP.dtype

    if dtype == torch.complex64:
        func = _cublas.cublasChpmv_v2
        alpha_c = cuComplex(alpha.real, alpha.imag)
        beta_c = cuComplex(beta.real, beta.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZhpmv_v2
        alpha_c = cuDoubleComplex(alpha.real, alpha.imag)
        beta_c = cuDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXhpmv_v2 execution failed with error code: {status}")


def cpu_hpmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy):
    if n == 0:
        return to_cpu_blas_tensor(y)

    ref_AP = to_cpu_blas_tensor(AP)
    ref_x = to_cpu_blas_tensor(x)
    if beta == 0 and incy == 1:
        ref_y = torch.empty(y.shape, dtype=torch.complex128)
    else:
        ref_y = to_cpu_blas_tensor(y)

    yout = cpu_blas.zhpmv(
        n,
        alpha,
        ref_AP.numpy(),
        ref_x.numpy(),
        incx=incx,
        beta=beta,
        y=ref_y.numpy(),
        incy=incy,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def hpmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy):
    column_AP, column_uplo = _row_to_column_hpmv(AP, uplo)
    if TO_CPU:
        return cpu_hpmv_reference(
            column_uplo, n, alpha, column_AP, x, incx, beta, y, incy
        )

    ref_y = y.clone()
    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        hipblas_hpmv_reference(
            column_uplo, n, alpha, column_AP, x, incx, beta, ref_y, incy
        )
    else:
        cublas_hpmv_reference(
            column_uplo, n, alpha, column_AP, x, incx, beta, ref_y, incy
        )
    return ref_y


HPMV_SIZES = [
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
    8191,
    8192,
    12288,
    16384,
]
HPMV_STRIDE_SIZES = [64, 127, 256]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def hpmv_randn(*shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        normalized = (
            tuple(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (tuple, torch.Size))
            else shape
        )
        values = torch.randn((*normalized, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(*shape, dtype=dtype, device=device)


def make_hermitian_packed(n, dtype, device):
    return hpmv_randn(n * (n + 1) // 2, dtype=dtype, device=device)


def _diag_packed_offsets(n, uplo, device):
    k = torch.arange(n, dtype=torch.int64, device=device)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        return k * n - k * (k - 1) // 2
    return k * (k + 1) // 2 + k


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _hpmv_reduce_dim(n):
    return max(1, n)


@pytest.mark.chpmv
@pytest.mark.parametrize("n", HPMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_chpmv(n, uplo, beta):
    dtype, alpha = torch.complex64, 1.5 + 0.5j

    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hpmv_reference(uplo, n, alpha, AP, x, 1, beta, y, 1)
    flag_blas.chpmv(uplo, n, alpha, AP, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_hpmv_reduce_dim(n))


@pytest.mark.chpmv
@pytest.mark.parametrize("n", HPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_chpmv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j

    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = hpmv_randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hpmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy)
    flag_blas.chpmv(uplo, n, alpha, AP, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_hpmv_reduce_dim(n))


@pytest.mark.chpmv
def test_chpmv_alpha_zero():
    n = 256
    dtype = torch.complex64
    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()

    y_ref = hpmv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0j, AP, x, 1, 2.0 + 1.0j, y, 1)
    flag_blas.chpmv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, AP, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=_hpmv_reduce_dim(n))
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.chpmv
def test_chpmv_beta_zero():
    n = 256
    dtype = torch.complex64
    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hpmv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, AP, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.chpmv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, AP, x, 1, 0.0j, y_nan, 1)
    flag_blas.chpmv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, AP, x, 1, 0.0j, y_zero, 1)
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=_hpmv_reduce_dim(n))
    blas_assert_close(
        y_nan, to_reference(y_zero), dtype, reduce_dim=_hpmv_reduce_dim(n)
    )


@pytest.mark.zhpmv
@pytest.mark.parametrize("n", HPMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_zhpmv(n, uplo, beta):
    check_fp64_support()
    dtype, alpha = torch.complex128, 1.5 + 0.5j

    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hpmv_reference(uplo, n, alpha, AP, x, 1, beta, y, 1)
    flag_blas.zhpmv(uplo, n, alpha, AP, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_hpmv_reduce_dim(n))


@pytest.mark.zhpmv
@pytest.mark.parametrize("n", HPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zhpmv_stride(n, uplo, incx, incy):
    check_fp64_support()
    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j

    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = hpmv_randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hpmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy)
    flag_blas.zhpmv(uplo, n, alpha, AP, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_hpmv_reduce_dim(n))


@pytest.mark.zhpmv
def test_zhpmv_alpha_zero():
    check_fp64_support()
    n = 256
    dtype = torch.complex128
    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()

    y_ref = hpmv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0j, AP, x, 1, 2.0 + 1.0j, y, 1)
    flag_blas.zhpmv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, AP, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=_hpmv_reduce_dim(n))
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.zhpmv
def test_zhpmv_beta_zero():
    check_fp64_support()
    n = 256
    dtype = torch.complex128
    AP = make_hermitian_packed(n, dtype, flag_blas.device)
    x = hpmv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hpmv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, AP, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.zhpmv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, AP, x, 1, 0.0j, y_nan, 1)
    flag_blas.zhpmv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, AP, x, 1, 0.0j, y_zero, 1)
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=_hpmv_reduce_dim(n))
    blas_assert_close(
        y_nan, to_reference(y_zero), dtype, reduce_dim=_hpmv_reduce_dim(n)
    )


@pytest.mark.parametrize(
    "dtype, op, alpha, beta",
    [
        (torch.complex64, flag_blas.chpmv, 1.5 + 0.5j, 0.5 + 0.25j),
        (torch.complex128, flag_blas.zhpmv, 1.5 + 0.5j, 0.5 + 0.25j),
    ],
)
def test_hpmv_n_zero(dtype, op, alpha, beta):
    if dtype == torch.complex128:
        check_fp64_support()

    AP = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, AP, x, 1, beta, y, 1)
    assert y.numel() == 0


@pytest.mark.parametrize(
    "dtype, op, alpha, beta, uplo",
    [
        (torch.complex64, flag_blas.chpmv, 1.25 + 0.5j, 0.5 + 0.25j, m)
        for m in FILL_MODES
    ]
    + [
        (torch.complex128, flag_blas.zhpmv, 1.25 + 0.5j, 0.5 + 0.25j, m)
        for m in FILL_MODES
    ],
)
def test_hpmv_diagonal_imag_ignored(dtype, op, alpha, beta, uplo):
    if dtype == torch.complex128:
        check_fp64_support()

    n = 128
    AP_clean = make_hermitian_packed(n, dtype, flag_blas.device)
    diag_off = _diag_packed_offsets(n, uplo, flag_blas.device)
    AP_clean_parts = torch.view_as_real(AP_clean)
    real_part = AP_clean_parts[diag_off, 0].clone()
    AP_clean_parts[diag_off, 1].zero_()

    AP_dirty = AP_clean.clone()
    diag_imag_noise = hpmv_randn(n, dtype=dtype, device=flag_blas.device).imag
    AP_dirty_parts = torch.view_as_real(AP_dirty)
    AP_dirty_parts[diag_off, 0] = real_part
    AP_dirty_parts[diag_off, 1] = diag_imag_noise

    x = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y_clean = hpmv_randn(n, dtype=dtype, device=flag_blas.device)
    y_dirty = y_clean.clone()

    op(uplo, n, alpha, AP_clean, x, 1, beta, y_clean, 1)
    op(uplo, n, alpha, AP_dirty, x, 1, beta, y_dirty, 1)

    blas_assert_close(y_dirty, to_reference(y_clean), dtype, reduce_dim=n)

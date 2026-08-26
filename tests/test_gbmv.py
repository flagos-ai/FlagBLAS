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
from flag_blas.ops import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T

if flag_blas.vendor_name == "hygon":
    from .hipblas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
from .conftest import TO_CPU

IS_ASCEND = flag_blas.vendor_name == "ascend"
IS_HYGON = flag_blas.vendor_name == "hygon"

if not IS_ASCEND and not IS_HYGON:
    import cupy as cp


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


_cublas = None if IS_ASCEND or IS_HYGON else load_cublas()


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def hipblas_gbmv_reference(trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return

    column_AB = row_to_column_band(AB, m, n, kl, ku, lda)
    if AB.dtype == torch.float32:
        symbol = "hipblasSgbmv"
        scalar_type = ctypes.c_float
        alpha_value = scalar_type(alpha)
        beta_value = scalar_type(beta)
    elif AB.dtype == torch.float64:
        symbol = "hipblasDgbmv"
        scalar_type = ctypes.c_double
        alpha_value = scalar_type(alpha)
        beta_value = scalar_type(beta)
    elif AB.dtype == torch.complex64:
        symbol = "hipblasCgbmv_v2"
        scalar_type = HipComplex
        alpha_value = scalar_type(alpha.real, alpha.imag)
        beta_value = scalar_type(beta.real, beta.imag)
    elif AB.dtype == torch.complex128:
        symbol = "hipblasZgbmv_v2"
        scalar_type = HipDoubleComplex
        alpha_value = scalar_type(alpha.real, alpha.imag)
        beta_value = scalar_type(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS GBMV: {AB.dtype}")

    hip_trans = {
        CUBLAS_OP_N: 111,
        CUBLAS_OP_T: 112,
        CUBLAS_OP_C: 113,
    }[trans]
    library, handle = get_hipblas_context(column_AB)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(scalar_type),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(scalar_type),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    check_hipblas_status(
        function(
            handle,
            hip_trans,
            m,
            n,
            kl,
            ku,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(column_AB.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        symbol,
    )


def cublas_gbmv_reference(trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return

    column_AB = row_to_column_band(AB, m, n, kl, ku, lda)
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
        ctypes.c_void_p(column_AB.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXgbmv_v2 execution failed with error code: {status}")


def cpu_gbmv_reference(trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return to_cpu_blas_tensor(y)

    if beta == 0 and incy == 1:
        ref_dtype = torch.complex128 if y.is_complex() else torch.float64
        ref_y = torch.empty(y.shape, dtype=ref_dtype)
    else:
        ref_y = to_cpu_blas_tensor(y)
    ref_AB = row_to_column_band(to_cpu_blas_tensor(AB), m, n, kl, ku, lda)
    ref_x = to_cpu_blas_tensor(x)
    func = cpu_blas.zgbmv if ref_AB.dtype.is_complex else cpu_blas.dgbmv

    yout = func(
        m,
        n,
        kl,
        ku,
        alpha,
        ref_AB.numpy().T,
        ref_x.numpy(),
        incx=incx,
        beta=beta,
        y=ref_y.numpy(),
        incy=incy,
        trans=trans,
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def cpu_gbmv_band_reference(
    trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy
):
    if m == 0 or n == 0:
        return to_cpu_blas_tensor(y)

    ref_AB = to_cpu_blas_tensor(AB)
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    input_len, output_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    logical_x = ref_x[::incx][:input_len]
    logical_y = ref_y[::incy][:output_len]

    if beta == 0:
        logical_y.zero_()
    else:
        logical_y.mul_(beta)

    for d in range(-ku, kl + 1):
        j_min = max(0, -d)
        j_max = min(n, m - d)
        if j_min >= j_max:
            continue
        j_idx = torch.arange(j_min, j_max)
        i_idx = j_idx + d
        values = ref_AB[i_idx, kl - d]
        if trans == CUBLAS_OP_N:
            logical_y[i_idx] += alpha * values * logical_x[j_idx]
        else:
            if trans == CUBLAS_OP_C:
                values = values.conj()
            logical_y[j_idx] += alpha * values * logical_x[i_idx]

    return ref_y


def gbmv_reference(trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy):
    if TO_CPU:
        if m < kl + ku + 1:
            return cpu_gbmv_band_reference(
                trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy
            )
        return cpu_gbmv_reference(
            trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy
        )

    ref_y = y.clone()
    if IS_HYGON:
        hipblas_gbmv_reference(
            trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, ref_y, incy
        )
    else:
        cublas_gbmv_reference(
            trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, ref_y, incy
        )
    return ref_y


GBMV_SHAPES = [
    (255, 255),
    (256, 256),
    (1023, 1023),
    (1024, 1024),
    (4095, 4095),
    (4096, 4096),
    (16384, 16384),
    (127, 255),
    (4096, 16384),
    (16384, 4096),
    (10000, 10000),
]

GBMV_STRIDE_SHAPES = [(64, 128), (128, 64), (256, 256)]

GBMV_BANDS = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 7),
    (32, 32),
    (128, 128),
    (256, 256),
]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def randn_tensor(shape, dtype, device):
    if IS_ASCEND and dtype == torch.complex64:
        if isinstance(shape, int):
            shape = (shape,)
        real = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(real)
    return torch.randn(shape, dtype=dtype, device=device)


def row_to_column_band(AB, m, n, kl, ku, lda):
    column_AB = torch.zeros((n, lda), dtype=AB.dtype, device=AB.device)
    for d in range(-ku, kl + 1):
        j_min = max(0, -d)
        j_max = min(n, m - d)
        if j_min < j_max:
            j_idx = torch.arange(j_min, j_max, device=AB.device)
            i_idx = j_idx + d
            column_AB[j_idx, ku + d] = AB[i_idx, kl - d]
    return column_AB


def create_banded_data(m, n, kl, ku, lda, dtype, device):
    if dtype == torch.complex64 and IS_ASCEND:
        AB_real = torch.zeros((m, lda, 2), dtype=torch.float32, device=device)
        for d in range(-ku, kl + 1):
            j_min = max(0, -d)
            j_max = min(n, m - d)
            if j_min < j_max:
                j_idx = torch.arange(j_min, j_max, device=device)
                i_idx = j_idx + d
                AB_real[i_idx, kl - d] = torch.randn(
                    (j_max - j_min, 2), dtype=torch.float32, device=device
                )
        return torch.view_as_complex(AB_real).contiguous()

    AB = torch.zeros((m, lda), dtype=dtype, device=device)
    for d in range(-ku, kl + 1):
        j_min = max(0, -d)
        j_max = min(n, m - d)
        if j_min < j_max:
            j_idx = torch.arange(j_min, j_max, device=device)
            i_idx = j_idx + d
            AB[i_idx, kl - d] = randn_tensor(j_max - j_min, dtype, device)
    return AB.contiguous()


def get_effective_bands(m, n, kl, ku):
    actual_kl = min(kl, max(0, m - 1))
    actual_ku = min(ku, max(0, n - 1))
    is_truncated = (actual_kl != kl) or (actual_ku != ku)
    return actual_kl, actual_ku, is_truncated


def gbmv_reduce_dim(trans, m, n, kl, ku):
    band_width = kl + ku + 1
    input_len = n if trans == CUBLAS_OP_N else m
    return max(1, min(input_len, band_width))


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


@pytest.mark.parametrize(
    "trans,dtype,alpha,beta",
    [
        (CUBLAS_OP_N, torch.float32, 1.5, -0.25),
        (CUBLAS_OP_N, torch.float32, 1.25, 0.0),
        (CUBLAS_OP_T, torch.float32, -0.75, 0.5),
        (CUBLAS_OP_N, torch.complex64, 1.5 - 0.25j, -0.5 + 0.75j),
        (CUBLAS_OP_T, torch.complex64, -0.75 + 0.5j, 0.25 - 0.5j),
        (CUBLAS_OP_C, torch.complex64, 0.5 + 1.25j, -0.25 + 0.5j),
    ],
)
def test_cpu_gbmv_band_reference(trans, dtype, alpha, beta):
    m, n, kl, ku, lda = 2, 5, 1, 3, 6
    AB = torch.zeros((m, lda), dtype=dtype)
    dense_dtype = torch.complex128 if dtype.is_complex else torch.float64
    dense = torch.zeros((m, n), dtype=dense_dtype)
    value = 1
    for d in range(-ku, kl + 1):
        j_min = max(0, -d)
        j_max = min(n, m - d)
        for j in range(j_min, j_max):
            i = j + d
            element = complex(value, -value / 2) if dtype.is_complex else value
            AB[i, kl - d] = element
            dense[i, j] = element
            value += 1

    input_len, output_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.zeros(input_len * 2, dtype=dtype)
    y = torch.full((output_len * 2,), 99, dtype=dtype)
    logical_x = torch.arange(1, input_len + 1, dtype=torch.float64)
    if dtype.is_complex:
        logical_x = torch.complex(logical_x, -logical_x / 4)
    x[::2][:input_len] = logical_x.to(dtype)
    logical_y = torch.arange(1, output_len + 1, dtype=torch.float64)
    if dtype.is_complex:
        logical_y = torch.complex(logical_y, logical_y / 3)
    y[::2][:output_len] = logical_y.to(dtype)
    if beta == 0:
        y[::2][:output_len] = float("nan")
    initial_y = y.to(dense_dtype)

    actual = cpu_gbmv_band_reference(
        trans, m, n, kl, ku, alpha, AB, lda, x, 2, beta, y, 2
    )
    matrix = dense
    if trans == CUBLAS_OP_T:
        matrix = dense.T
    elif trans == CUBLAS_OP_C:
        matrix = dense.conj().T
    expected = initial_y.clone()
    expected_update = alpha * (matrix @ x[::2][:input_len].to(dense_dtype))
    if beta != 0:
        expected_update += beta * initial_y[::2][:output_len]
    expected[::2][:output_len] = expected_update

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


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
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )
    flag_blas.sgbmv(trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1)

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


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
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )
    flag_blas.sgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


@pytest.mark.sgbmv
def test_sgbmv_alpha_zero():
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.float32
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(m, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = gbmv_reference(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y, 1)
    flag_blas.sgbmv(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y, 1)
    blas_assert_close(
        y, y_ref, dtype, reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku)
    )
    blas_assert_close(y, to_reference(y_orig * 2.0), dtype)


@pytest.mark.sgbmv
def test_sgbmv_beta_zero():
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.float32
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_nan, 1
    )
    flag_blas.sgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.sgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_zero, 1)
    blas_assert_close(
        y_nan,
        ref_y_nan,
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )
    blas_assert_close(
        y_nan,
        to_reference(y_zero),
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )


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
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )
    flag_blas.dgbmv(trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1)

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


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
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )
    flag_blas.dgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


@pytest.mark.dgbmv
def test_dgbmv_alpha_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.float64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(m, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = gbmv_reference(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y, 1)
    flag_blas.dgbmv(CUBLAS_OP_N, m, n, kl, ku, 0.0, AB, lda, x, 1, 2.0, y, 1)
    blas_assert_close(
        y, y_ref, dtype, reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku)
    )
    blas_assert_close(y, to_reference(y_orig * 2.0), dtype)


@pytest.mark.dgbmv
def test_dgbmv_beta_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.float64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_nan, 1
    )
    flag_blas.dgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.dgbmv(CUBLAS_OP_N, m, n, kl, ku, 1.0, AB, lda, x, 1, 0.0, y_zero, 1)
    blas_assert_close(
        y_nan,
        ref_y_nan,
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )
    blas_assert_close(
        y_nan,
        to_reference(y_zero),
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )


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
    x = randn_tensor(x_len, dtype, flag_blas.device)
    y = randn_tensor(y_len, dtype, flag_blas.device)
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )
    flag_blas.cgbmv(trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1)

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


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
    x = randn_tensor(x_len * incx, dtype, flag_blas.device)
    y = randn_tensor(y_len * incy, dtype, flag_blas.device)
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )
    flag_blas.cgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


@pytest.mark.cgbmv
def test_cgbmv_alpha_zero():
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.complex64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = randn_tensor(n, dtype, flag_blas.device)
    y = randn_tensor(m, dtype, flag_blas.device)
    y_orig = y.clone()
    y_ref = gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.cgbmv(CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(
        y, y_ref, dtype, reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku)
    )
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.cgbmv
def test_cgbmv_beta_zero():
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.complex64
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = randn_tensor(n, dtype, flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.cgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.cgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(
        y_nan,
        ref_y_nan,
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )
    blas_assert_close(
        y_nan,
        to_reference(y_zero),
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )


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
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1
    )
    flag_blas.zgbmv(trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, 1, beta, y, 1)

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


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
    ref_y = gbmv_reference(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )
    flag_blas.zgbmv(
        trans, m, n, actual_kl, actual_ku, alpha, AB, lda, x, incx, beta, y, incy
    )

    blas_assert_close(
        y, ref_y, dtype, reduce_dim=gbmv_reduce_dim(trans, m, n, actual_kl, actual_ku)
    )


@pytest.mark.zgbmv
def test_zgbmv_alpha_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 4, 4, 9
    dtype = torch.complex128
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(m, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.zgbmv(CUBLAS_OP_N, m, n, kl, ku, 0.0j, AB, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(
        y, y_ref, dtype, reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku)
    )
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.zgbmv
def test_zgbmv_beta_zero():
    check_fp64_support()
    m, n, kl, ku, lda = 128, 256, 2, 2, 5
    dtype = torch.complex128
    AB = create_banded_data(m, n, kl, ku, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((m,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(m, dtype=dtype, device=flag_blas.device)
    ref_y_nan = gbmv_reference(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.zgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.zgbmv(
        CUBLAS_OP_N, m, n, kl, ku, 1.0 + 0.5j, AB, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(
        y_nan,
        ref_y_nan,
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )
    blas_assert_close(
        y_nan,
        to_reference(y_zero),
        dtype,
        reduce_dim=gbmv_reduce_dim(CUBLAS_OP_N, m, n, kl, ku),
    )

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

import numpy as np
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import (
    AXPY_SHAPES,
    L1_NONUNIT_PAIR_STRIDES,
    L1_STRIDE_SHAPES,
    SCALARS,
    blas_assert_close,
    l1_randn,
    to_cpu_blas_tensor,
    to_reference,
)
from .conftest import TO_CPU

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    from .vendor_blas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp, cublas
elif flag_blas.vendor_name != "ascend":
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


# SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,)]

COMPLEX_SCALARS = [1.0 + 2.0j, -0.5 + 1.5j]


def hipblas_axpy_reference(n, alpha, x, incx, y, incy):
    if n == 0:
        return

    symbol, scalar_type = {
        torch.float32: ("hipblasSaxpy", ctypes.c_float),
        torch.float64: ("hipblasDaxpy", ctypes.c_double),
        torch.complex64: ("hipblasCaxpy_v2", HipComplex),
        torch.complex128: ("hipblasZaxpy_v2", HipDoubleComplex),
    }[x.dtype]
    if x.is_complex():
        alpha = complex(alpha)
        alpha_value = scalar_type(alpha.real, alpha.imag)
    else:
        alpha_value = scalar_type(alpha)

    library, handle = get_hipblas_context(x)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(scalar_type),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    check_hipblas_status(
        function(
            handle,
            n,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        symbol,
    )


def cublas_axpy_reference(n, alpha, x, incx, y, incy):
    assert x.dtype == y.dtype, "x and y must have the same dtype"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"

    if n == 0:
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func, np_dtype = cublas.saxpy, np.float32
    elif dtype == torch.float64:
        func, np_dtype = cublas.daxpy, np.float64
    elif dtype == torch.complex64:
        func, np_dtype = cublas.caxpy, np.complex64
    elif dtype == torch.complex128:
        func, np_dtype = cublas.zaxpy, np.complex128
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np_dtype)
    alpha_ptr = alpha_np.ctypes.data

    func(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)


def cpu_axpy_reference(n, alpha, x, incx, y, incy):
    if n == 0:
        return to_cpu_blas_tensor(y)

    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    dtype = ref_x.dtype
    if dtype == torch.float64:
        func = cpu_blas.daxpy
    elif dtype == torch.complex128:
        func = cpu_blas.zaxpy
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS: {dtype}")

    func(ref_x.numpy(), ref_y.numpy(), n=n, a=alpha, incx=incx, incy=incy)
    return ref_y


def axpy_reference(n, alpha, x, incx, y, incy):
    if TO_CPU:
        return cpu_axpy_reference(n, alpha, x, incx, y, incy)

    ref_x = to_reference(x)
    ref_y = to_reference(y).clone()

    if IS_HYGON:
        hipblas_axpy_reference(n, alpha, ref_x, incx, ref_y, incy)
    else:
        cublas_axpy_reference(n, alpha, ref_x, incx, ref_y, incy)
    return ref_y


@pytest.mark.axpy
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.saxpy),
        pytest.param(torch.float64, marks=pytest.mark.daxpy),
    ],
)
@pytest.mark.parametrize("shape", AXPY_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
def test_accuracy_axpy_real(dtype, shape, alpha):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    incy = 1
    x = l1_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = l1_randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_y = axpy_reference(n, alpha, x, incx, y, incy)

    if dtype == torch.float32:
        flag_blas.saxpy(n, alpha, x, incx, y, incy)
    else:
        flag_blas.daxpy(n, alpha, x, incx, y, incy)

    blas_assert_close(y, ref_y, dtype)


@pytest.mark.axpy
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.caxpy),
        pytest.param(torch.complex128, marks=pytest.mark.zaxpy),
    ],
)
@pytest.mark.parametrize("shape", AXPY_SHAPES)
@pytest.mark.parametrize("alpha", COMPLEX_SCALARS + SCALARS)
def test_accuracy_axpy_complex(dtype, shape, alpha):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    incy = 1
    x = l1_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = l1_randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_y = axpy_reference(n, alpha, x, incx, y, incy)

    if dtype == torch.complex64:
        flag_blas.caxpy(n, alpha, x, incx, y, incy)
    else:
        flag_blas.zaxpy(n, alpha, x, incx, y, incy)

    blas_assert_close(y, ref_y, dtype)


@pytest.mark.axpy
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.saxpy),
        pytest.param(torch.float64, marks=pytest.mark.daxpy),
    ],
)
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_NONUNIT_PAIR_STRIDES)
def test_accuracy_axpy_different_n_with_stride_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    alpha = -111.999
    x = l1_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = l1_randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_y = axpy_reference(n, alpha, x, incx, y, incy)

    if dtype == torch.float32:
        flag_blas.saxpy(n, alpha, x, incx, y, incy)
    else:
        flag_blas.daxpy(n, alpha, x, incx, y, incy)

    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


@pytest.mark.axpy
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.caxpy),
        pytest.param(torch.complex128, marks=pytest.mark.zaxpy),
    ],
)
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_NONUNIT_PAIR_STRIDES)
def test_accuracy_axpy_different_n_with_stride_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    alpha = -0.5 + 1.5j
    x = l1_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = l1_randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_y = axpy_reference(n, alpha, x, incx, y, incy)

    if dtype == torch.complex64:
        flag_blas.caxpy(n, alpha, x, incx, y, incy)
    else:
        flag_blas.zaxpy(n, alpha, x, incx, y, incy)

    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


@pytest.mark.axpy
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.saxpy),
        pytest.param(torch.float64, marks=pytest.mark.daxpy),
        pytest.param(torch.complex64, marks=pytest.mark.caxpy),
        pytest.param(torch.complex128, marks=pytest.mark.zaxpy),
    ],
)
def test_accuracy_axpy_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = l1_randn(0, dtype=dtype, device=flag_blas.device)
    y = l1_randn(0, dtype=dtype, device=flag_blas.device)

    alpha = 2.0 + 1.0j if dtype in [torch.complex64, torch.complex128] else 2.0
    n = 0

    if dtype == torch.float32:
        flag_blas.saxpy(n, alpha, x, 1, y, 1)
    elif dtype == torch.float64:
        flag_blas.daxpy(n, alpha, x, 1, y, 1)
    elif dtype == torch.complex64:
        flag_blas.caxpy(n, alpha, x, 1, y, 1)
    else:
        flag_blas.zaxpy(n, alpha, x, 1, y, 1)

    assert y.shape == (0,)
    assert y.dtype == dtype
    assert y.device == x.device


@pytest.mark.axpy
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.saxpy),
        pytest.param(torch.float64, marks=pytest.mark.daxpy),
        pytest.param(torch.complex64, marks=pytest.mark.caxpy),
        pytest.param(torch.complex128, marks=pytest.mark.zaxpy),
    ],
)
@pytest.mark.parametrize(
    "shape,incx,incy",
    [(shape, 1, 1) for shape in AXPY_SHAPES]
    + [
        (shape, ix, iy)
        for shape in L1_STRIDE_SHAPES
        for ix, iy in L1_NONUNIT_PAIR_STRIDES
    ],
)
def test_accuracy_axpy_perf_inputs(dtype, shape, incx, incy):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")
    n = shape[0]
    alpha = 0.01 + 0.01j if dtype.is_complex else 1e-5
    x = l1_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = l1_randn(n * incy, dtype=dtype, device=flag_blas.device)
    expected = axpy_reference(n, alpha, x, incx, y, incy)
    op = {
        torch.float32: flag_blas.saxpy,
        torch.float64: flag_blas.daxpy,
        torch.complex64: flag_blas.caxpy,
        torch.complex128: flag_blas.zaxpy,
    }[dtype]
    op(n, alpha, x, incx, y, incy)
    blas_assert_close(y, expected, dtype)

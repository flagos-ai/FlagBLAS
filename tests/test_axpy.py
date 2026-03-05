import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas

import flag_blas

from .accuracy_utils import SCALARS, AXPY_SHAPES, gems_assert_close, to_reference


#SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,)]
STRIDES = [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

COMPLEX_SCALARS = [1.0 + 2.0j, -0.5 + 1.5j]

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


@pytest.mark.axpy
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", AXPY_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_axpy_real(dtype, shape, alpha, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    cublas_axpy_reference(n, alpha, ref_x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.saxpy(n, alpha, x, incx, y, incy)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.daxpy(n, alpha, x, incx, y, incy)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)


@pytest.mark.axpy
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", AXPY_SHAPES)
@pytest.mark.parametrize("alpha", COMPLEX_SCALARS + SCALARS)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_axpy_complex(dtype, shape, alpha, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    cublas_axpy_reference(n, alpha, ref_x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.caxpy(n, alpha, x, incx, y, incy)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zaxpy(n, alpha, x, incx, y, incy)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)


@pytest.mark.axpy
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_accuracy_axpy_empty_tensor(dtype):
    if dtype in [torch.float64, torch.complex128] and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)

    alpha = 2.0 + 1.0j if dtype in [torch.complex64, torch.complex128] else 2.0
    n = 0

    if dtype == torch.float32:
        flag_blas.ops.saxpy(n, alpha, x, 1, y, 1)
    elif dtype == torch.float64:
        flag_blas.ops.daxpy(n, alpha, x, 1, y, 1)
    elif dtype == torch.complex64:
        flag_blas.ops.caxpy(n, alpha, x, 1, y, 1)
    else:
        flag_blas.ops.zaxpy(n, alpha, x, 1, y, 1)

    assert y.shape == (0,)
    assert y.dtype == dtype
    assert y.device == x.device

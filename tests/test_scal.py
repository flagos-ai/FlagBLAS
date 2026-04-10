import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas

import flag_blas

from .accuracy_utils import SCAL_SHAPES, SCALARS, gems_assert_close, to_reference

STRIDES = [1, 2, 3, 5]

COMPLEX_SCALARS = [1.0 + 2.0j, -0.5 + 1.5j]


def cublas_scal_reference(n, alpha, x, incx):
    assert x.dim() == 1, "x must be 1-dimensional"

    if n <= 0:
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func, np_dtype = cublas.sscal, np.float32
    elif dtype == torch.float64:
        func, np_dtype = cublas.dscal, np.float64
    elif dtype == torch.complex64:
        func, np_dtype = cublas.cscal, np.complex64
    elif dtype == torch.complex128:
        func, np_dtype = cublas.zscal, np.complex128
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np_dtype)
    alpha_ptr = alpha_np.ctypes.data

    func(handle, n, alpha_ptr, x.data_ptr(), incx)



def cublas_csscal_reference(n, alpha, x, incx):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.dtype == torch.complex64, "x must be complex64 for csscal"

    if n <= 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np.float32)
    alpha_ptr = alpha_np.ctypes.data

    cublas.csscal(handle, n, alpha_ptr, x.data_ptr(), incx)
    
def cublas_zdscal_reference(n, alpha, x, incx):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.dtype == torch.complex128, "x must be complex128 for zdscal"

    if n <= 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np.float64)
    alpha_ptr = alpha_np.ctypes.data

    cublas.zdscal(handle, n, alpha_ptr, x.data_ptr(), incx)

@pytest.mark.scal
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("incx", STRIDES)
def test_accuracy_csscal(shape, alpha, incx):
    n = shape[0]
    x = torch.randn(n * incx, dtype=torch.complex64, device=flag_blas.device)

    ref_x = x.clone()

    cublas_csscal_reference(n, alpha, ref_x, incx)

    flag_blas.ops.csscal(n, alpha, x, incx)
    torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)

@pytest.mark.scal
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("incx", STRIDES)
def test_accuracy_zdscal(shape, alpha, incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    n = shape[0]
    x = torch.randn(n * incx, dtype=torch.complex128, device=flag_blas.device)

    ref_x = x.clone()

    cublas_zdscal_reference(n, alpha, ref_x, incx)

    flag_blas.ops.zdscal(n, alpha, x, incx)
    torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)

@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("incx", STRIDES)
def test_accuracy_scal_real(dtype, shape, alpha, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()

    cublas_scal_reference(n, alpha, ref_x, incx)

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", COMPLEX_SCALARS + SCALARS)
@pytest.mark.parametrize("incx", STRIDES)
def test_accuracy_scal_complex(dtype, shape, alpha, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()

    cublas_scal_reference(n, alpha, ref_x, incx)

    if dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_accuracy_scal_empty_tensor(dtype):
    if dtype in [torch.float64, torch.complex128] and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    alpha = 2.0 + 1.0j if dtype in [torch.complex64, torch.complex128] else 2.0
    n = 0

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, 1)
    elif dtype == torch.float64:
        flag_blas.ops.dscal(n, alpha, x, 1)
    elif dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, 1)
    else:
        flag_blas.ops.zscal(n, alpha, x, 1)

    torch.testing.assert_close(x, ref_x)

@pytest.mark.scal
def test_accuracy_csscal_empty_tensor():
    x = torch.randn(0, dtype=torch.complex64, device=flag_blas.device)
    ref_x = x.clone()

    alpha = 2.0
    n = 0

    flag_blas.ops.csscal(n, alpha, x, 1)

    torch.testing.assert_close(x, ref_x)

@pytest.mark.scal
def test_accuracy_zdscal_empty_tensor():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    x = torch.randn(0, dtype=torch.complex128, device=flag_blas.device)
    ref_x = x.clone()

    alpha = 2.0
    n = 0

    flag_blas.ops.zdscal(n, alpha, x, 1)

    torch.testing.assert_close(x, ref_x)

@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)])
def test_accuracy_scal_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    alpha = 2.5
    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()

    cublas_scal_reference(n, alpha, ref_x, 1)

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, 1)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dscal(n, alpha, x, 1)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)])
def test_accuracy_scal_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    alpha = 2.0 + 1.0j
    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()

    cublas_scal_reference(n, alpha, ref_x, 1)

    if dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, 1)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zscal(n, alpha, x, 1)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)

@pytest.mark.scal
@pytest.mark.parametrize("n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)])
def test_accuracy_csscal_different_n(n, vec_size):
    alpha = 2.5
    x = torch.randn(vec_size, dtype=torch.complex64, device=flag_blas.device)

    ref_x = x.clone()

    cublas_csscal_reference(n, alpha, ref_x, 1)

    flag_blas.ops.csscal(n, alpha, x, 1)
    torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)

@pytest.mark.scal
@pytest.mark.parametrize("n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)])
def test_accuracy_zdscal_different_n(n, vec_size):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    alpha = 2.5
    x = torch.randn(vec_size, dtype=torch.complex128, device=flag_blas.device)

    ref_x = x.clone()

    cublas_zdscal_reference(n, alpha, ref_x, 1)

    flag_blas.ops.zdscal(n, alpha, x, 1)
    torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)

@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n, vec_size, incx", [
    (5, 20, 2),
    (5, 30, 3),
    (10, 50, 2),
    (10, 100, 5),
])
def test_accuracy_scal_different_n_with_stride_real(dtype, n, vec_size, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    alpha = -0.75
    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()

    cublas_scal_reference(n, alpha, ref_x, incx)

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("n, vec_size, incx", [
    (5, 20, 2),
    (5, 30, 3),
    (10, 50, 2),
    (10, 100, 5),
])
def test_accuracy_scal_different_n_with_stride_complex(dtype, n, vec_size, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    alpha = -0.5 + 1.5j
    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()

    cublas_scal_reference(n, alpha, ref_x, incx)

    if dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zscal(n, alpha, x, incx)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)


@pytest.mark.scal
@pytest.mark.parametrize("n, vec_size, incx", [
    (5, 20, 2),
    (5, 30, 3),
    (10, 50, 2),
    (10, 100, 5),
])
def test_accuracy_csscal_different_n_with_stride(n, vec_size, incx):
    alpha = -0.75
    x = torch.randn(vec_size, dtype=torch.complex64, device=flag_blas.device)

    ref_x = x.clone()

    cublas_csscal_reference(n, alpha, ref_x, incx)

    flag_blas.ops.csscal(n, alpha, x, incx)
    torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)

@pytest.mark.scal
@pytest.mark.parametrize("n, vec_size, incx", [
    (5, 20, 2),
    (5, 30, 3),
    (10, 50, 2),
    (10, 100, 5),
])
def test_accuracy_zdscal_different_n_with_stride(n, vec_size, incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    alpha = -0.75
    x = torch.randn(vec_size, dtype=torch.complex128, device=flag_blas.device)

    ref_x = x.clone()

    cublas_zdscal_reference(n, alpha, ref_x, incx)

    flag_blas.ops.zdscal(n, alpha, x, incx)
    torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)
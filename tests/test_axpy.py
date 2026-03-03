import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

tests_path = Path(__file__).parent
sys.path.insert(0, str(tests_path))

import flag_blas

from accuracy_utils import SCALARS, gems_assert_close, to_reference


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,)]
STRIDES = [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

COMPLEX_SCALARS = [1.0 + 2.0j, -0.5 + 1.5j]


def cublas_axpy_reference(x, y, alpha, incx=1, incy=1):
    assert x.dtype == y.dtype, "x and y must have the same dtype"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"

    x = x.contiguous()
    y = y.contiguous()

    n = (x.numel() + incx - 1) // incx
    if n == 0:
        return y

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

    return y


@pytest.mark.axpy
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_axpy_real(dtype, shape, alpha, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_x = to_reference(x, True)
    ref_y = to_reference(y, True)

    ref_out = cublas_axpy_reference(ref_x, ref_y, alpha, incx, incy)

    if dtype == torch.float32:
        res_out = flag_blas.ops.saxpy(x, y, alpha=alpha, incx=incx, incy=incy)
    else:
        res_out = flag_blas.ops.daxpy(x, y, alpha=alpha, incx=incx, incy=incy)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.axpy
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", SHAPES)
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

    ref_out = cublas_axpy_reference(ref_x, ref_y, alpha, incx, incy)

    if dtype == torch.complex64:
        res_out = flag_blas.ops.caxpy(x, y, alpha=alpha, incx=incx, incy=incy)
        torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4)
    else:
        res_out = flag_blas.ops.zaxpy(x, y, alpha=alpha, incx=incx, incy=incy)
        torch.testing.assert_close(res_out, ref_out, rtol=1.3e-6, atol=1e-5)


@pytest.mark.axpy
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_accuracy_axpy_empty_tensor(dtype):
    if dtype in [torch.float64, torch.complex128] and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)

    alpha = 2.0 + 1.0j if dtype in [torch.complex64, torch.complex128] else 2.0

    if dtype == torch.float32:
        res_out = flag_blas.ops.saxpy(x, y, alpha=alpha)
    elif dtype == torch.float64:
        res_out = flag_blas.ops.daxpy(x, y, alpha=alpha)
    elif dtype == torch.complex64:
        res_out = flag_blas.ops.caxpy(x, y, alpha=alpha)
    else:
        res_out = flag_blas.ops.zaxpy(x, y, alpha=alpha)

    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
    assert res_out.device == x.device

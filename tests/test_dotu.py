import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import DOTU_SHAPES, L1_PAIR_STRIDES
from .conftest import TO_CPU


def _dotu_tolerances(dtype):
    if dtype == torch.complex64:
        return (1e-3, 1e-3) if TO_CPU else (1e-4, 1e-4)
    return (1e-10, 1e-10) if TO_CPU else (1e-12, 1e-12)


def cublas_dotu_reference(n, x, incx, y, incy, result):
    """Reference implementation using cuBLAS cdotu/zdotu."""
    assert x.dtype == y.dtype, "x and y must have the same dtype"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert result.numel() == 1, "result must have exactly one element"
    assert result.dtype == x.dtype, "result dtype must match input dtype"

    if n == 0:
        result.zero_()
        return

    dtype = x.dtype
    if dtype == torch.complex64:
        func = cublas.cdotu
    elif dtype == torch.complex128:
        func = cublas.zdotu
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS dotu: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr())


def cpu_dotu_reference(n, x, incx, y, incy, result):
    if n == 0:
        result.zero_()
        return

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    if x.dtype == torch.complex64:
        value = cpu_blas.cdotu(x_np, y_np, n=n, incx=incx, incy=incy)
    elif x.dtype == torch.complex128:
        value = cpu_blas.zdotu(x_np, y_np, n=n, incx=incx, incy=incy)
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS dotu: {x.dtype}")
    result.fill_(value)


def dotu_reference(n, x, incx, y, incy, result):
    if TO_CPU:
        cpu_dotu_reference(n, x, incx, y, incy, result)
    else:
        cublas_dotu_reference(n, x, incx, y, incy, result)


@pytest.mark.dotu
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", DOTU_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_PAIR_STRIDES)
def test_accuracy_dotu_complex(dtype, shape, incx, incy):
    """Test dotu with various shapes, strides, and complex dtypes."""
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    dotu_reference(n, x.clone(), incx, y.clone(), incy, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.cdotu(n, x, incx, y, incy, result)
    else:
        flag_blas.ops.zdotu(n, x, incx, y, incy, result)

    rtol, atol = _dotu_tolerances(dtype)
    torch.testing.assert_close(result, ref_result, rtol=rtol, atol=atol)


@pytest.mark.dotu
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_accuracy_dotu_empty_tensor(dtype):
    """n <= 0 must be a no-op and write zero to result."""
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result = torch.ones(1, dtype=dtype, device=flag_blas.device)
    n = 0

    if dtype == torch.complex64:
        flag_blas.ops.cdotu(n, x, 1, y, 1, result)
    else:
        flag_blas.ops.zdotu(n, x, 1, y, 1, result)

    assert result.item() == 0j
    assert result.dtype == dtype
    assert result.device == x.device


@pytest.mark.dotu
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_dotu_different_n(dtype, n, vec_size):
    """Test n smaller than allocated tensor length; only first n elements used."""
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    # Reference uses only the first n elements
    dotu_reference(n, x[:n].clone(), 1, y[:n].clone(), 1, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.cdotu(n, x, 1, y, 1, result)
    else:
        flag_blas.ops.zdotu(n, x, 1, y, 1, result)

    rtol, atol = _dotu_tolerances(dtype)
    torch.testing.assert_close(result, ref_result, rtol=rtol, atol=atol)

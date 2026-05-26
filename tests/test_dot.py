import pytest
import torch
import cupy as cp
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas
from .accuracy_utils import DEFAULT_SHAPES, L1_PAIR_STRIDES
from .conftest import TO_CPU


def _dot_tolerances(dtype, n):
    if dtype == torch.float32:
        if TO_CPU:
            return 1e-3, max(1e-3, 1e-5 * (max(n, 1) ** 0.5))
        return 1e-5, max(1e-5, 1e-5 * (max(n, 1) ** 0.5))
    if TO_CPU:
        return 1e-10, max(1e-10, 1e-12 * (max(n, 1) ** 0.5))
    return 1e-12, max(1e-12, 1e-12 * (max(n, 1) ** 0.5))


def cublas_dot_reference(n, x, incx, y, incy, result):
    """Compute dot product using cuBLAS, store in result (scalar tensor)."""
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"
    assert x.dtype == y.dtype, "x and y must have the same dtype"
    assert result.dtype == x.dtype, "result dtype must equal input dtype"

    if n == 0:
        result.zero_()
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func = cublas.sdot
    elif dtype == torch.float64:
        func = cublas.ddot
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS dot: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

    func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr())


def cpu_dot_reference(n, x, incx, y, incy, result):
    if n == 0:
        result.zero_()
        return

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    if x.dtype == torch.float32:
        value = cpu_blas.sdot(x_np, y_np, n=n, incx=incx, incy=incy)
    elif x.dtype == torch.float64:
        value = cpu_blas.ddot(x_np, y_np, n=n, incx=incx, incy=incy)
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS dot: {x.dtype}")
    result.fill_(value)


def dot_reference(n, x, incx, y, incy, result):
    if TO_CPU:
        cpu_dot_reference(n, x, incx, y, incy, result)
    else:
        cublas_dot_reference(n, x, incx, y, incy, result)


@pytest.mark.dot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", DEFAULT_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_PAIR_STRIDES)
def test_accuracy_dot_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()
    ref_y = y.clone()
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    dot_reference(n, ref_x, incx, ref_y, incy, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.sdot(n, x, incx, y, incy, result)
    else:
        flag_blas.ops.ddot(n, x, incx, y, incy, result)
    rtol, atol = _dot_tolerances(dtype, n)

    torch.testing.assert_close(result, ref_result, rtol=rtol, atol=atol)


@pytest.mark.dot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_dot_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    n = 0

    if dtype == torch.float32:
        flag_blas.ops.sdot(n, x, 1, y, 1, result)
    else:
        flag_blas.ops.ddot(n, x, 1, y, 1, result)

    assert result.item() == 0.0
    assert result.dtype == dtype
    assert result.device == x.device


@pytest.mark.dot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_dot_different_n_real(dtype, n, vec_size):
    """Test n smaller than allocated tensor length; ensure only n elements used."""
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()
    ref_y = y.clone()
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    dot_reference(n, ref_x, 1, ref_y, 1, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.sdot(n, x, 1, y, 1, result)
    else:
        flag_blas.ops.ddot(n, x, 1, y, 1, result)
    rtol, atol = _dot_tolerances(dtype, n)

    torch.testing.assert_close(result, ref_result, rtol=rtol, atol=atol)


@pytest.mark.dot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_dot_negative_n(dtype):
    """Negative n should be treated as no-op, result set to zero."""
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(10, dtype=dtype, device=flag_blas.device)
    y = torch.randn(10, dtype=dtype, device=flag_blas.device)
    result = torch.ones(1, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        flag_blas.ops.sdot(-1, x, 1, y, 1, result)
    else:
        flag_blas.ops.ddot(-1, x, 1, y, 1, result)

    assert result.item() == 0.0

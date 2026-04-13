import pytest
import torch

import flag_blas

from .accuracy_utils import ASUM_SHAPES

STRIDES = [1, 2, 3, 5]


def torch_abs_reference(n, x, incx, y, incy):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"

    if n <= 0:
        return

    x_view = x[::incx][:n]
    ref = torch.abs(x_view)
    y[::incy][:n].copy_(ref)


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
@pytest.mark.parametrize("incx", STRIDES)
@pytest.mark.parametrize("incy", STRIDES)
def test_accuracy_abs_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)
    y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.sabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
@pytest.mark.parametrize("incx", STRIDES)
@pytest.mark.parametrize("incy", STRIDES)
def test_accuracy_abs_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_x = x.clone()
    ref_y = torch.empty(n * incy, dtype=result_dtype, device=flag_blas.device)
    y = torch.empty(n * incy, dtype=result_dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.cabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_abs_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        y = torch.empty(0, dtype=torch.float32, device=flag_blas.device)
        flag_blas.ops.sabs(0, x, 1, y, 1)
    elif dtype == torch.float64:
        y = torch.empty(0, dtype=torch.float64, device=flag_blas.device)
        flag_blas.ops.dabs(0, x, 1, y, 1)
    elif dtype == torch.complex64:
        y = torch.empty(0, dtype=torch.float32, device=flag_blas.device)
        flag_blas.ops.cabs(0, x, 1, y, 1)
    else:
        y = torch.empty(0, dtype=torch.float64, device=flag_blas.device)
        flag_blas.ops.zabs(0, x, 1, y, 1)

    assert y.numel() == 0
    assert y.device == x.device


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size,incx,incy",
    [
        (1, 10, 1, 1),
        (5, 20, 2, 1),
        (5, 20, 3, 2),
        (10, 50, 2, 3),
        (10, 100, 5, 5),
    ],
)
def test_accuracy_abs_different_n_with_stride_real(dtype, n, vec_size, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    out_size = 1 + (n - 1) * incy

    ref_x = x.clone()
    ref_y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)
    y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.sabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size,incx,incy",
    [
        (1, 10, 1, 1),
        (5, 20, 2, 1),
        (5, 20, 3, 2),
        (10, 50, 2, 3),
        (10, 100, 5, 5),
    ],
)
def test_accuracy_abs_different_n_with_stride_complex(dtype, n, vec_size, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    out_size = 1 + (n - 1) * incy

    ref_x = x.clone()
    ref_y = torch.empty(out_size, dtype=result_dtype, device=flag_blas.device)
    y = torch.empty(out_size, dtype=result_dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.cabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zabs(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)
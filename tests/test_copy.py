import pytest
import torch

import flag_blas

from .accuracy_utils import ASUM_SHAPES

STRIDES = [1, 2, 3, 5]


def torch_copy_reference(n, x, incx, y, incy):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"

    if n <= 0:
        return

    y[::incy][:n].copy_(x[::incx][:n])


@pytest.mark.copy
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
@pytest.mark.parametrize("incx", STRIDES)
@pytest.mark.parametrize("incy", STRIDES)
def test_accuracy_copy_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)
    y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)

    torch_copy_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.scopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dcopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)


@pytest.mark.copy
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
@pytest.mark.parametrize("incx", STRIDES)
@pytest.mark.parametrize("incy", STRIDES)
def test_accuracy_copy_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)
    y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)

    torch_copy_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.ccopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zcopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)


@pytest.mark.copy
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_copy_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.empty(0, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        flag_blas.ops.scopy(0, x, 1, y, 1)
    elif dtype == torch.float64:
        flag_blas.ops.dcopy(0, x, 1, y, 1)
    elif dtype == torch.complex64:
        flag_blas.ops.ccopy(0, x, 1, y, 1)
    else:
        flag_blas.ops.zcopy(0, x, 1, y, 1)

    assert y.numel() == 0
    assert y.device == x.device


@pytest.mark.copy
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
def test_accuracy_copy_different_n_with_stride_real(dtype, n, vec_size, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    out_size = 1 + (n - 1) * incy

    ref_x = x.clone()
    ref_y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)
    y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)

    torch_copy_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.scopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dcopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)


@pytest.mark.copy
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
def test_accuracy_copy_different_n_with_stride_complex(dtype, n, vec_size, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    out_size = 1 + (n - 1) * incy

    ref_x = x.clone()
    ref_y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)
    y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)

    torch_copy_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.ccopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zcopy(n, x, incx, y, incy)
        torch.testing.assert_close(y[::incy][:n], ref_y[::incy][:n], rtol=1e-15, atol=1e-15)
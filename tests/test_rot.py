import pytest
import torch

import flag_blas

from .accuracy_utils import DEFAULT_SHAPES


STRIDES = [(1, 1), (2, 1), (1, 3), (2, 2), (3, 5)]


def get_c_s(dtype, device):
    if dtype == torch.float32:
        c = torch.tensor([0.8], dtype=torch.float32, device=device)
        s = torch.tensor([0.6], dtype=torch.float32, device=device)
    elif dtype == torch.float64:
        c = torch.tensor([0.8], dtype=torch.float64, device=device)
        s = torch.tensor([0.6], dtype=torch.float64, device=device)
    elif dtype == torch.complex64:
        c = torch.tensor([0.8], dtype=torch.float32, device=device)
        s = torch.tensor([0.36 + 0.48j], dtype=torch.complex64, device=device)
    elif dtype == torch.complex128:
        c = torch.tensor([0.8], dtype=torch.float64, device=device)
        s = torch.tensor([0.36 + 0.48j], dtype=torch.complex128, device=device)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return c, s


def torch_rot_reference(n, x, incx, y, incy, c, s):
    if n <= 0:
        return

    c_val = c.item()
    s_val = s.item()

    x_view = x[:n * incx:incx]
    y_view = y[:n * incy:incy]

    new_x = c_val * x_view + s_val * y_view
    if x.is_complex():
        new_y = c_val * y_view - s_val.conjugate() * x_view
    else:
        new_y = c_val * y_view - s_val * x_view

    x_view.copy_(new_x)
    y_view.copy_(new_y)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", DEFAULT_SHAPES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_rot_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    c, s = get_c_s(dtype, flag_blas.device)

    torch_rot_reference(n, ref_x, incx, ref_y, incy, c, s)

    if dtype == torch.float32:
        flag_blas.ops.srot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.drot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", DEFAULT_SHAPES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_rot_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    c, s = get_c_s(dtype, flag_blas.device)

    torch_rot_reference(n, ref_x, incx, ref_y, incy, c, s)

    if dtype == torch.complex64:
        flag_blas.ops.crot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zrot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_accuracy_rot_empty_tensor(dtype):
    if dtype in [torch.float64, torch.complex128] and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()
    ref_y = y.clone()

    c, s = get_c_s(dtype, flag_blas.device)

    n = 0

    if dtype == torch.float32:
        flag_blas.ops.srot(n, x, 1, y, 1, c, s)
    elif dtype == torch.float64:
        flag_blas.ops.drot(n, x, 1, y, 1, c, s)
    elif dtype == torch.complex64:
        flag_blas.ops.crot(n, x, 1, y, 1, c, s)
    else:
        flag_blas.ops.zrot(n, x, 1, y, 1, c, s)

    torch.testing.assert_close(x, ref_x)
    torch.testing.assert_close(y, ref_y)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)])
def test_accuracy_rot_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    c, s = get_c_s(dtype, flag_blas.device)

    torch_rot_reference(n, ref_x, 1, ref_y, 1, c, s)

    if dtype == torch.float32:
        flag_blas.ops.srot(n, x, 1, y, 1, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.drot(n, x, 1, y, 1, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)])
def test_accuracy_rot_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    c, s = get_c_s(dtype, flag_blas.device)

    torch_rot_reference(n, ref_x, 1, ref_y, 1, c, s)

    if dtype == torch.complex64:
        flag_blas.ops.crot(n, x, 1, y, 1, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zrot(n, x, 1, y, 1, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n,vec_size_x,vec_size_y,incx,incy", [
    (5, 20, 20, 2, 2),
    (5, 20, 30, 2, 3),
    (10, 50, 40, 2, 1),
    (10, 100, 100, 5, 2),
])
def test_accuracy_rot_different_n_with_stride_real(dtype, n, vec_size_x, vec_size_y, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size_x, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size_y, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    c, s = get_c_s(dtype, flag_blas.device)

    torch_rot_reference(n, ref_x, incx, ref_y, incy, c, s)

    if dtype == torch.float32:
        flag_blas.ops.srot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.drot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("n,vec_size_x,vec_size_y,incx,incy", [
    (5, 20, 20, 2, 2),
    (5, 20, 30, 2, 3),
    (10, 50, 40, 2, 1),
    (10, 100, 100, 5, 2),
])
def test_accuracy_rot_different_n_with_stride_complex(dtype, n, vec_size_x, vec_size_y, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size_x, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size_y, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()

    c, s = get_c_s(dtype, flag_blas.device)

    torch_rot_reference(n, ref_x, incx, ref_y, incy, c, s)

    if dtype == torch.complex64:
        flag_blas.ops.crot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.zrot(n, x, incx, y, incy, c, s)
        torch.testing.assert_close(x, ref_x, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(y, ref_y, rtol=1e-15, atol=1e-15)

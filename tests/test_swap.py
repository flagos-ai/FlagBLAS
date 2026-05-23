import cupy as cp
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
from .accuracy_utils import L1_PAIR_STRIDES, SWAP_SHAPES
from .conftest import TO_CPU


def cublas_swap_reference(n, x, incx, y, incy):
    """CuPy reference swap, in-place."""
    if n <= 0:
        return
    x_view = cp.from_dlpack(x.detach())[0:n * incx:incx]
    y_view = cp.from_dlpack(y.detach())[0:n * incy:incy]
    tmp = x_view.copy()
    x_view[...] = y_view
    y_view[...] = tmp


def cpu_swap_reference(n, x, incx, y, incy):
    if n <= 0:
        return

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    if x.dtype == torch.float32:
        ref_x, ref_y = cpu_blas.sswap(x_np, y_np, n=n, incx=incx, incy=incy)
    elif x.dtype == torch.float64:
        ref_x, ref_y = cpu_blas.dswap(x_np, y_np, n=n, incx=incx, incy=incy)
    elif x.dtype == torch.complex64:
        ref_x, ref_y = cpu_blas.cswap(x_np, y_np, n=n, incx=incx, incy=incy)
    elif x.dtype == torch.complex128:
        ref_x, ref_y = cpu_blas.zswap(x_np, y_np, n=n, incx=incx, incy=incy)
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS swap: {x.dtype}")
    x.copy_(torch.from_numpy(ref_x).to(device=x.device))
    y.copy_(torch.from_numpy(ref_y).to(device=y.device))


def swap_reference(n, x, incx, y, incy):
    if TO_CPU:
        cpu_swap_reference(n, x, incx, y, incy)
    else:
        cublas_swap_reference(n, x, incx, y, incy)


# ==============================
# Accuracy tests - real dtypes
# ==============================

@pytest.mark.swap
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", SWAP_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_PAIR_STRIDES)
def test_accuracy_swap_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()
    swap_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.sswap(n, x, incx, y, incy)
    else:
        flag_blas.ops.dswap(n, x, incx, y, incy)

    tol = (
        {"rtol": 1e-5, "atol": 1e-5}
        if dtype == torch.float32
        else {"rtol": 1e-15, "atol": 1e-15}
    )
    torch.testing.assert_close(x, ref_x, **tol)
    torch.testing.assert_close(y, ref_y, **tol)


# ==============================
# Accuracy tests - complex dtypes
# ==============================

@pytest.mark.swap
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", SWAP_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_PAIR_STRIDES)
def test_accuracy_swap_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()
    swap_reference(n, ref_x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.cswap(n, x, incx, y, incy)
    else:
        flag_blas.ops.zswap(n, x, incx, y, incy)

    tol = (
        {"rtol": 1e-5, "atol": 1e-5}
        if dtype == torch.complex64
        else {"rtol": 1e-15, "atol": 1e-15}
    )
    torch.testing.assert_close(x, ref_x, **tol)
    torch.testing.assert_close(y, ref_y, **tol)


# ==============================
# Edge case: n <= 0 (empty tensor)
# ==============================

@pytest.mark.swap
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_swap_empty_tensor(dtype):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    n = 0

    if dtype == torch.float32:
        flag_blas.ops.sswap(n, x, 1, y, 1)
    elif dtype == torch.float64:
        flag_blas.ops.dswap(n, x, 1, y, 1)
    elif dtype == torch.complex64:
        flag_blas.ops.cswap(n, x, 1, y, 1)
    else:
        flag_blas.ops.zswap(n, x, 1, y, 1)

    assert x.shape == (0,)
    assert y.shape == (0,)
    assert x.dtype == dtype
    assert y.dtype == dtype


# ==============================
# Different n: n smaller than allocated length (real)
# ==============================

@pytest.mark.swap
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_swap_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()
    swap_reference(n, ref_x, 1, ref_y, 1)

    if dtype == torch.float32:
        flag_blas.ops.sswap(n, x, 1, y, 1)
    else:
        flag_blas.ops.dswap(n, x, 1, y, 1)

    tol = (
        {"rtol": 1e-5, "atol": 1e-5}
        if dtype == torch.float32
        else {"rtol": 1e-15, "atol": 1e-15}
    )
    torch.testing.assert_close(x, ref_x, **tol)
    torch.testing.assert_close(y, ref_y, **tol)


# ==============================
# Different n: n smaller than allocated length (complex)
# ==============================

@pytest.mark.swap
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_swap_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()
    swap_reference(n, ref_x, 1, ref_y, 1)

    if dtype == torch.complex64:
        flag_blas.ops.cswap(n, x, 1, y, 1)
    else:
        flag_blas.ops.zswap(n, x, 1, y, 1)

    tol = (
        {"rtol": 1e-5, "atol": 1e-5}
        if dtype == torch.complex64
        else {"rtol": 1e-15, "atol": 1e-15}
    )
    torch.testing.assert_close(x, ref_x, **tol)
    torch.testing.assert_close(y, ref_y, **tol)

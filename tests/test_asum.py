import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas

import flag_blas

from .accuracy_utils import ASUM_SHAPES, SCALARS, gems_assert_close, to_reference

STRIDES = [1, 2, 3, 5]

COMPLEX_SCALARS = [1.0 + 2.0j, -0.5 + 1.5j]


def cublas_asum_reference(n, x, incx, result):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"

    if n == 0:
        result.zero_()
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func = cublas.sasum
        result_dtype = torch.float32
    elif dtype == torch.float64:
        func = cublas.dasum
        result_dtype = torch.float64
    elif dtype == torch.complex64:
        func = cublas.scasum
        result_dtype = torch.float32
    elif dtype == torch.complex128:
        func = cublas.dzasum
        result_dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

    assert result.dtype == result_dtype, f"result must be {result_dtype}"

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

    func(handle, n, x.data_ptr(), incx, result.data_ptr())


@pytest.mark.asum
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
@pytest.mark.parametrize("incx", STRIDES)
def test_accuracy_asum_real(dtype, shape, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    cublas_asum_reference(n, ref_x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.sasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-15, atol=1e-15)


@pytest.mark.asum
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
@pytest.mark.parametrize("incx", STRIDES)
def test_accuracy_asum_complex(dtype, shape, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)

    cublas_asum_reference(n, ref_x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.scasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dzasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-15, atol=1e-15)


@pytest.mark.asum
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_asum_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result_dtype = (
        dtype
        if dtype in [torch.float32, torch.float64]
        else (torch.float32 if dtype == torch.complex64 else torch.float64)
    )
    result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)

    n = 0

    if dtype == torch.float32:
        flag_blas.ops.sasum(n, x, 1, result)
    elif dtype == torch.float64:
        flag_blas.ops.dasum(n, x, 1, result)
    elif dtype == torch.complex64:
        flag_blas.ops.scasum(n, x, 1, result)
    else:
        flag_blas.ops.dzasum(n, x, 1, result)

    assert result.item() == 0.0
    assert result.dtype == result_dtype
    assert result.device == x.device


@pytest.mark.asum
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_asum_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    cublas_asum_reference(n, ref_x, 1, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.sasum(n, x, 1, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dasum(n, x, 1, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-15, atol=1e-15)


@pytest.mark.asum
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_asum_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)

    cublas_asum_reference(n, ref_x, 1, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.scasum(n, x, 1, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dzasum(n, x, 1, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-15, atol=1e-15)


@pytest.mark.asum
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size,incx",
    [
        (5, 20, 2),
        (5, 20, 3),
        (10, 50, 2),
        (10, 100, 5),
    ],
)
def test_accuracy_asum_different_n_with_stride_real(dtype, n, vec_size, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    cublas_asum_reference(n, ref_x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.sasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-15, atol=1e-15)


@pytest.mark.asum
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size,incx",
    [
        (5, 20, 2),
        (5, 20, 3),
        (10, 50, 2),
        (10, 100, 5),
    ],
)
def test_accuracy_asum_different_n_with_stride_complex(dtype, n, vec_size, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)

    cublas_asum_reference(n, ref_x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.scasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-5, atol=1e-5)
    else:
        flag_blas.ops.dzasum(n, x, incx, result)
        torch.testing.assert_close(result, ref_result, rtol=1e-15, atol=1e-15)

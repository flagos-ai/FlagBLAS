import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import (
    ASUM_SHAPES,
    blas_assert_close,
    to_cpu_blas_tensor,
    to_reference,
)
from .conftest import TO_CPU

STRIDES = [1, 2, 3, 5]
STRIDE_SHAPES = [
    (1024,),
    (5333,),
    (65536,),
    (100000,),
    (1048576,),
    (3000000,),
    (4194304,),
]


def cublas_nrm2_reference(n, x, incx, result):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"

    if n == 0:
        result.zero_()
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func = cublas.snrm2
        result_dtype = torch.float32
    elif dtype == torch.float64:
        func = cublas.dnrm2
        result_dtype = torch.float64
    elif dtype == torch.complex64:
        func = cublas.scnrm2
        result_dtype = torch.float32
    elif dtype == torch.complex128:
        func = cublas.dznrm2
        result_dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

    assert result.dtype == result_dtype, f"result must be {result_dtype}"

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

    func(handle, n, x.data_ptr(), incx, result.data_ptr())


def cpu_nrm2_reference(n, x, incx, result):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"

    if n == 0:
        result.zero_()
        return

    ref_x = to_cpu_blas_tensor(x)
    dtype = ref_x.dtype
    if dtype == torch.float64:
        func = cpu_blas.dnrm2
    elif dtype == torch.complex128:
        func = cpu_blas.dznrm2
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS: {dtype}")

    value = func(ref_x.numpy(), n=n, incx=incx)
    result.fill_(value)


def nrm2_reference(n, x, incx, result):
    if TO_CPU:
        ref_result = torch.zeros(result.shape, dtype=result.dtype, device="cpu")
        cpu_nrm2_reference(n, x, incx, ref_result)
        return ref_result

    ref_x = to_reference(x)
    ref_result = to_reference(result).clone()

    cublas_nrm2_reference(n, ref_x, incx, ref_result)
    return ref_result


@pytest.mark.nrm2
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
def test_accuracy_nrm2_real(dtype, shape):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_result = nrm2_reference(n, x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.snrm2(n, x, incx, result)
    else:
        flag_blas.dnrm2(n, x, incx, result)

    blas_assert_close(result, ref_result, dtype, reduce_dim=n)


@pytest.mark.nrm2
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", ASUM_SHAPES)
def test_accuracy_nrm2_complex(dtype, shape):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)

    ref_result = nrm2_reference(n, x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.scnrm2(n, x, incx, result)
    else:
        flag_blas.dznrm2(n, x, incx, result)

    blas_assert_close(result, ref_result, result_dtype, reduce_dim=n)


@pytest.mark.nrm2
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_nrm2_empty_tensor(dtype):
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
        flag_blas.snrm2(n, x, 1, result)
    elif dtype == torch.float64:
        flag_blas.dnrm2(n, x, 1, result)
    elif dtype == torch.complex64:
        flag_blas.scnrm2(n, x, 1, result)
    else:
        flag_blas.dznrm2(n, x, 1, result)

    assert result.item() == 0.0
    assert result.dtype == result_dtype
    assert result.device == x.device


@pytest.mark.nrm2
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_nrm2_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_result = nrm2_reference(n, x, 1, ref_result)

    if dtype == torch.float32:
        flag_blas.snrm2(n, x, 1, result)
    else:
        flag_blas.dnrm2(n, x, 1, result)

    blas_assert_close(result, ref_result, dtype, reduce_dim=n)


@pytest.mark.nrm2
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_nrm2_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)

    ref_result = nrm2_reference(n, x, 1, ref_result)

    if dtype == torch.complex64:
        flag_blas.scnrm2(n, x, 1, result)
    else:
        flag_blas.dznrm2(n, x, 1, result)

    blas_assert_close(result, ref_result, result_dtype, reduce_dim=n)


@pytest.mark.nrm2
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", STRIDE_SHAPES)
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_accuracy_nrm2_different_n_with_stride_real(dtype, shape, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_result = nrm2_reference(n, x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.snrm2(n, x, incx, result)
    else:
        flag_blas.dnrm2(n, x, incx, result)

    blas_assert_close(result, ref_result, dtype, reduce_dim=n)


@pytest.mark.nrm2
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", STRIDE_SHAPES)
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_accuracy_nrm2_different_n_with_stride_complex(dtype, shape, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=result_dtype, device=flag_blas.device)

    ref_result = nrm2_reference(n, x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.scnrm2(n, x, incx, result)
    else:
        flag_blas.dznrm2(n, x, incx, result)

    blas_assert_close(result, ref_result, result_dtype, reduce_dim=n)

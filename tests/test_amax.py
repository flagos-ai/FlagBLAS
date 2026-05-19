import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import AMAX_SHAPES, to_cpu_blas_tensor, to_reference
from .conftest import TO_CPU

STRIDES = [1, 2, 3, 5]
STRIDE_SHAPES = [
    (256,),
    (512,),
    (768,),
    (1280,),
    (1536,),
    (1792,),
    (2048,),
    (2304,),
    (2560,),
    (3072,),
    (3584,),
    (4096,),
    (1024,),
    (5333,),
    (65536,),
    (100000,),
    (1048576,),
    (3000000,),
    (4194304,),
]


def cublas_amax_reference(n, x, incx, result):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"

    if n == 0:
        result.zero_()
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func = cublas.isamax
    elif dtype == torch.float64:
        func = cublas.idamax
    elif dtype == torch.complex64:
        func = cublas.icamax
    elif dtype == torch.complex128:
        func = cublas.izamax
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

    func(handle, n, x.data_ptr(), incx, result.data_ptr())


def cpu_amax_reference(n, x, incx, result):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"

    if n == 0:
        result.zero_()
        return

    ref_x = to_cpu_blas_tensor(x)
    dtype = ref_x.dtype
    if dtype == torch.float64:
        func = cpu_blas.idamax
    elif dtype == torch.complex128:
        func = cpu_blas.izamax
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS: {dtype}")

    result.fill_(func(ref_x.numpy(), n=n, incx=incx) + 1)


def amax_reference(n, x, incx, result):
    if TO_CPU:
        ref_result = torch.zeros(result.shape, dtype=result.dtype, device="cpu")
        cpu_amax_reference(n, x, incx, ref_result)
        return ref_result

    ref_x = to_reference(x)
    ref_result = to_reference(result).clone()

    cublas_amax_reference(n, ref_x, incx, ref_result)
    return ref_result


@pytest.mark.amax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", AMAX_SHAPES)
def test_accuracy_amax_real(dtype, shape):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amax_reference(n, x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.samax(n, x, incx, result)
    else:
        flag_blas.ops.damax(n, x, incx, result)

    assert (
        result.item() == ref_result.item()
    ), f"expected {ref_result.item()}, got {result.item()}"


@pytest.mark.amax
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", AMAX_SHAPES)
def test_accuracy_amax_complex(dtype, shape):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amax_reference(n, x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.camax(n, x, incx, result)
    else:
        flag_blas.ops.zamax(n, x, incx, result)

    assert (
        result.item() == ref_result.item()
    ), f"expected {ref_result.item()}, got {result.item()}"


@pytest.mark.amax
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_amax_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    n = 0

    if dtype == torch.float32:
        flag_blas.ops.samax(n, x, 1, result)
    elif dtype == torch.float64:
        flag_blas.ops.damax(n, x, 1, result)
    elif dtype == torch.complex64:
        flag_blas.ops.camax(n, x, 1, result)
    else:
        flag_blas.ops.zamax(n, x, 1, result)

    assert result.item() == 0
    assert result.dtype == torch.int32
    assert result.device == x.device


@pytest.mark.amax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_amax_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amax_reference(n, x, 1, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.samax(n, x, 1, result)
    else:
        flag_blas.ops.damax(n, x, 1, result)

    assert (
        result.item() == ref_result.item()
    ), f"expected {ref_result.item()}, got {result.item()}"


@pytest.mark.amax
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_amax_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amax_reference(n, x, 1, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.camax(n, x, 1, result)
    else:
        flag_blas.ops.zamax(n, x, 1, result)

    assert (
        result.item() == ref_result.item()
    ), f"expected {ref_result.item()}, got {result.item()}"


@pytest.mark.amax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", STRIDE_SHAPES)
@pytest.mark.parametrize("incx", [2, 5])
def test_accuracy_amax_different_n_with_stride_real(dtype, shape, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amax_reference(n, x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.samax(n, x, incx, result)
    else:
        flag_blas.ops.damax(n, x, incx, result)

    assert (
        result.item() == ref_result.item()
    ), f"expected {ref_result.item()}, got {result.item()}"


@pytest.mark.amax
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", STRIDE_SHAPES)
@pytest.mark.parametrize("incx", [2, 5])
def test_accuracy_amax_different_n_with_stride_complex(dtype, shape, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amax_reference(n, x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.camax(n, x, incx, result)
    else:
        flag_blas.ops.zamax(n, x, incx, result)

    assert (
        result.item() == ref_result.item()
    ), f"expected {ref_result.item()}, got {result.item()}"

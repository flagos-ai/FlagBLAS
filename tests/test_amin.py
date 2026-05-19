import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas

from .accuracy_utils import AMIN_SHAPES, L1_STRIDES


def cublas_amin_reference(n, x, incx, result):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"
    assert result.dtype == torch.int32, "result must be torch.int32"

    if n == 0:
        result.zero_()
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func = cublas.isamin
    elif dtype == torch.float64:
        func = cublas.idamin
    elif dtype == torch.complex64:
        func = cublas.icamin
    elif dtype == torch.complex128:
        func = cublas.izamin
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

    func(handle, n, x.data_ptr(), incx, result.data_ptr())


@pytest.mark.amin
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", AMIN_SHAPES)
@pytest.mark.parametrize("incx", L1_STRIDES)
def test_accuracy_amin_real(dtype, shape, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    cublas_amin_reference(n, ref_x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.samin(n, x, incx, result)
    else:
        flag_blas.ops.damin(n, x, incx, result)

    torch.testing.assert_close(result, ref_result, rtol=0, atol=0)


@pytest.mark.amin
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", AMIN_SHAPES)
@pytest.mark.parametrize("incx", L1_STRIDES)
def test_accuracy_amin_complex(dtype, shape, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    cublas_amin_reference(n, ref_x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.camin(n, x, incx, result)
    else:
        flag_blas.ops.zamin(n, x, incx, result)

    torch.testing.assert_close(result, ref_result, rtol=0, atol=0)


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_amin_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    n = 0

    if dtype == torch.float32:
        flag_blas.ops.samin(n, x, 1, result)
    elif dtype == torch.float64:
        flag_blas.ops.damin(n, x, 1, result)
    elif dtype == torch.complex64:
        flag_blas.ops.camin(n, x, 1, result)
    else:
        flag_blas.ops.zamin(n, x, 1, result)

    assert result.item() == 0
    assert result.dtype == torch.int32
    assert result.device == x.device


@pytest.mark.amin
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_amin_different_n_real(dtype, n, vec_size):
    """Test n smaller than allocated tensor length; verify only n elements considered."""
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    cublas_amin_reference(n, ref_x, 1, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.samin(n, x, 1, result)
    else:
        flag_blas.ops.damin(n, x, 1, result)

    assert (
        result.item() == ref_result.item()
    ), f"Expected {ref_result.item()}, got {result.item()}"
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"


@pytest.mark.amin
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_amin_different_n_complex(dtype, n, vec_size):
    """Test n smaller than allocated tensor length for complex types."""
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    cublas_amin_reference(n, ref_x, 1, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.camin(n, x, 1, result)
    else:
        flag_blas.ops.zamin(n, x, 1, result)

    assert (
        result.item() == ref_result.item()
    ), f"Expected {ref_result.item()}, got {result.item()}"
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"


@pytest.mark.amin
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
def test_accuracy_amin_different_n_with_stride_real(dtype, n, vec_size, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    cublas_amin_reference(n, ref_x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.ops.samin(n, x, incx, result)
    else:
        flag_blas.ops.damin(n, x, incx, result)

    assert (
        result.item() == ref_result.item()
    ), f"Expected {ref_result.item()}, got {result.item()}"
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"


@pytest.mark.amin
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
def test_accuracy_amin_different_n_with_stride_complex(dtype, n, vec_size, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    cublas_amin_reference(n, ref_x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.camin(n, x, incx, result)
    else:
        flag_blas.ops.zamin(n, x, incx, result)

    assert (
        result.item() == ref_result.item()
    ), f"Expected {ref_result.item()}, got {result.item()}"
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes

import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import DEFAULT_SHAPES, L1_PAIR_STRIDES, blas_assert_close
from .conftest import TO_CPU

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    from tests.accuracy_utils import get_l1_hipblas_context

    from .hipblas_reference import check_hipblas_status
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp, cublas
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


def hipblas_dot_reference(n, x, incx, y, incy, result):
    if n <= 0:
        result.zero_()
        return
    symbol = {
        torch.float32: "hipblasSdot",
        torch.float64: "hipblasDdot",
    }[x.dtype]
    library, handle = get_l1_hipblas_context(x)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    check_hipblas_status(
        function(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr()),
        symbol,
    )


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
        ref_result = torch.zeros(result.shape, dtype=result.dtype, device="cpu")
        cpu_dot_reference(n, x, incx, y, incy, ref_result)
        return ref_result

    if IS_HYGON:
        hipblas_dot_reference(n, x, incx, y, incy, result)
    else:
        cublas_dot_reference(n, x, incx, y, incy, result)
    return result


@pytest.mark.dot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.sdot),
        pytest.param(torch.float64, marks=pytest.mark.ddot),
    ],
)
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

    ref_result = dot_reference(n, ref_x, incx, ref_y, incy, ref_result)

    if dtype == torch.float32:
        flag_blas.sdot(n, x, incx, y, incy, result)
    else:
        flag_blas.ddot(n, x, incx, y, incy, result)
    blas_assert_close(result, ref_result, dtype, reduce_dim=n)


@pytest.mark.dot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.sdot),
        pytest.param(torch.float64, marks=pytest.mark.ddot),
    ],
)
def test_accuracy_dot_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    n = 0

    if dtype == torch.float32:
        flag_blas.sdot(n, x, 1, y, 1, result)
    else:
        flag_blas.ddot(n, x, 1, y, 1, result)

    assert result.item() == 0.0
    assert result.dtype == dtype
    assert result.device == x.device


@pytest.mark.dot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.sdot),
        pytest.param(torch.float64, marks=pytest.mark.ddot),
    ],
)
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

    ref_result = dot_reference(n, ref_x, 1, ref_y, 1, ref_result)

    if dtype == torch.float32:
        flag_blas.sdot(n, x, 1, y, 1, result)
    else:
        flag_blas.ddot(n, x, 1, y, 1, result)
    blas_assert_close(result, ref_result, dtype, reduce_dim=n)


@pytest.mark.dot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.sdot),
        pytest.param(torch.float64, marks=pytest.mark.ddot),
    ],
)
def test_accuracy_dot_negative_n(dtype):
    """Negative n should be treated as no-op, result set to zero."""
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(10, dtype=dtype, device=flag_blas.device)
    y = torch.randn(10, dtype=dtype, device=flag_blas.device)
    result = torch.ones(1, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        flag_blas.sdot(-1, x, 1, y, 1, result)
    else:
        flag_blas.ddot(-1, x, 1, y, 1, result)

    assert result.item() == 0.0

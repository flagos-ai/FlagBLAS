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

import pytest
import torch
import cupy as cp
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas
from .accuracy_utils import DOTC_SHAPES, L1_PAIR_STRIDES
from .conftest import TO_CPU


def _dotc_tolerances(dtype):
    if dtype == torch.complex64:
        return (1e-3, 1e-3) if TO_CPU else (1e-5, 1e-5)
    return (1e-10, 1e-10) if TO_CPU else (1e-12, 1e-12)


def cublas_dotc_reference(n, x, incx, y, incy, result):
    """cuBLAS reference for conjugated complex dot product."""
    assert x.dim() == 1
    assert y.dim() == 1
    assert result.numel() == 1

    if n == 0:
        result.zero_()
        return

    if x.dtype == torch.complex64:
        func = cublas.cdotc
    elif x.dtype == torch.complex128:
        func = cublas.zdotc
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS dotc: {x.dtype}")
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr())


def cpu_dotc_reference(n, x, incx, y, incy, result):
    if n == 0:
        result.zero_()
        return

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    if x.dtype == torch.complex64:
        value = cpu_blas.cdotc(x_np, y_np, n=n, incx=incx, incy=incy)
    elif x.dtype == torch.complex128:
        value = cpu_blas.zdotc(x_np, y_np, n=n, incx=incx, incy=incy)
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS dotc: {x.dtype}")
    result.fill_(value)


def dotc_reference(n, x, incx, y, incy, result):
    if TO_CPU:
        cpu_dotc_reference(n, x, incx, y, incy, result)
    else:
        cublas_dotc_reference(n, x, incx, y, incy, result)


@pytest.mark.dotc
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", DOTC_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_PAIR_STRIDES)
def test_accuracy_dotc_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    dotc_reference(n, ref_x, incx, ref_y, incy, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.cdotc(n, x, incx, y, incy, result)
    else:
        flag_blas.ops.zdotc(n, x, incx, y, incy, result)
    rtol, atol = _dotc_tolerances(dtype)
    torch.testing.assert_close(result, ref_result, rtol=rtol, atol=atol)


@pytest.mark.dotc
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_accuracy_dotc_empty_tensor(dtype):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result = torch.empty(1, dtype=dtype, device=flag_blas.device)

    n = 0
    if dtype == torch.complex64:
        flag_blas.ops.cdotc(n, x, 1, y, 1, result)
    else:
        flag_blas.ops.zdotc(n, x, 1, y, 1, result)

    assert result.item() == 0.0j
    assert result.dtype == dtype
    assert result.device == x.device


@pytest.mark.dotc
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_dotc_different_n(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = x.clone()
    ref_y = y.clone()
    ref_result = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    dotc_reference(n, ref_x, 1, ref_y, 1, ref_result)

    if dtype == torch.complex64:
        flag_blas.ops.cdotc(n, x, 1, y, 1, result)
    else:
        flag_blas.ops.zdotc(n, x, 1, y, 1, result)
    rtol, atol = _dotc_tolerances(dtype)
    torch.testing.assert_close(result, ref_result, rtol=rtol, atol=atol)


@pytest.mark.dotc
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("incx,incy", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_dotc_incx_incy_validation(dtype, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = 5
    x = torch.randn(10, dtype=dtype, device=flag_blas.device)
    y = torch.randn(10, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError):
        if dtype == torch.complex64:
            flag_blas.ops.cdotc(n, x, incx, y, incy, result)
        else:
            flag_blas.ops.zdotc(n, x, incx, y, incy, result)

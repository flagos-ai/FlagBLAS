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
import ctypes.util

import cupy as cp
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import (
    COPY_SHAPES,
    L1_NONUNIT_PAIR_STRIDES,
    L1_STRIDE_SHAPES,
    blas_assert_close,
    to_reference,
)
from .conftest import TO_CPU


def load_cublas():
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)

    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Cannot find libcublas.so in the system")


_cublas = load_cublas()


def cublas_copy_reference(n, x, incx, y, incy):
    if n <= 0:
        return y

    handle = cp.cuda.device.get_cublas_handle()
    dtype = x.dtype

    if dtype == torch.float32:
        func = _cublas.cublasScopy_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDcopy_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCcopy_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZcopy_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(n),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )

    if status != 0:
        raise RuntimeError(f"cuBLAS copy execution failed, error code: {status}")

    return y


def cpu_copy_reference(n, x, incx, y, incy):
    if n <= 0:
        return y

    if x.dtype in [torch.float32, torch.float64]:
        ref_dtype = torch.float64
    elif x.dtype in [torch.complex64, torch.complex128]:
        ref_dtype = torch.complex128
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}")

    ref_x = x.detach().to(device="cpu", dtype=ref_dtype).contiguous()
    ref_y = torch.empty(y.shape, dtype=ref_dtype, device="cpu")
    dtype = ref_x.dtype
    if dtype == torch.float64:
        func = cpu_blas.dcopy
    elif dtype == torch.complex128:
        func = cpu_blas.zcopy
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    func(ref_x.numpy(), ref_y.numpy(), n=n, incx=incx, incy=incy)
    return ref_y


def copy_reference(n, x, incx, y, incy):
    if TO_CPU:
        return cpu_copy_reference(n, x, incx, y, incy)

    ref_x = to_reference(x)
    ref_y = to_reference(y).clone()
    cublas_copy_reference(n, ref_x, incx, ref_y, incy)
    return ref_y


@pytest.mark.copy
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", COPY_SHAPES)
def test_accuracy_copy_real(dtype, shape):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    incy = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)
    y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)

    ref_y = copy_reference(n, x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.scopy(n, x, incx, y, incy)
    else:
        flag_blas.ops.dcopy(n, x, incx, y, incy)

    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


@pytest.mark.copy
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", COPY_SHAPES)
def test_accuracy_copy_complex(dtype, shape):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    incy = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)
    y = torch.empty(n * incy, dtype=dtype, device=flag_blas.device)

    ref_y = copy_reference(n, x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.ccopy(n, x, incx, y, incy)
    else:
        flag_blas.ops.zcopy(n, x, incx, y, incy)

    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


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
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_NONUNIT_PAIR_STRIDES)
def test_accuracy_copy_different_n_with_stride_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    out_size = n * incy

    ref_y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)
    y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)

    ref_y = copy_reference(n, x, incx, ref_y, incy)

    if dtype == torch.float32:
        flag_blas.ops.scopy(n, x, incx, y, incy)
    else:
        flag_blas.ops.dcopy(n, x, incx, y, incy)

    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


@pytest.mark.copy
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_NONUNIT_PAIR_STRIDES)
def test_accuracy_copy_different_n_with_stride_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    out_size = n * incy

    ref_y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)
    y = torch.empty(out_size, dtype=dtype, device=flag_blas.device)

    ref_y = copy_reference(n, x, incx, ref_y, incy)

    if dtype == torch.complex64:
        flag_blas.ops.ccopy(n, x, incx, y, incy)
    else:
        flag_blas.ops.zcopy(n, x, incx, y, incy)

    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)

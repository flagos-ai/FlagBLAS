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
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import (
    L1_SCALAR_STRIDES,
    L1_STRIDE_SHAPES,
    SCAL_SHAPES,
    SCALARS,
    blas_assert_close,
    to_cpu_blas_tensor,
    to_reference,
)
from .conftest import TO_CPU

COMPLEX_SCALARS = [1.0 + 2.0j, -0.5 + 1.5j]


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


def cublas_scal_reference(n, alpha, x, incx):
    assert x.dim() == 1, "x must be 1-dimensional"

    if n <= 0:
        return

    dtype = x.dtype
    if dtype == torch.float32:
        func, np_dtype = cublas.sscal, np.float32
    elif dtype == torch.float64:
        func, np_dtype = cublas.dscal, np.float64
    elif dtype == torch.complex64:
        func, np_dtype = cublas.cscal, np.complex64
    elif dtype == torch.complex128:
        func, np_dtype = cublas.zscal, np.complex128
    else:
        raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np_dtype)
    alpha_ptr = alpha_np.ctypes.data

    func(handle, n, alpha_ptr, x.data_ptr(), incx)


def cublas_csscal_reference(n, alpha, x, incx):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.dtype == torch.complex64, "x must be complex64 for csscal"

    if n <= 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    alpha_c = ctypes.c_float(alpha)
    status = _cublas.cublasCsscal_v2(
        ctypes.c_void_p(handle),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS csscal execution failed, error code: {status}")


def cublas_zdscal_reference(n, alpha, x, incx):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.dtype == torch.complex128, "x must be complex128 for zdscal"

    if n <= 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    alpha_c = ctypes.c_double(alpha)
    status = _cublas.cublasZdscal_v2(
        ctypes.c_void_p(handle),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS zdscal execution failed, error code: {status}")


def cpu_scal_reference(n, alpha, x, incx):
    if n <= 0:
        return to_cpu_blas_tensor(x)

    ref_x = to_cpu_blas_tensor(x)
    dtype = ref_x.dtype
    if dtype == torch.float64:
        func = cpu_blas.dscal
    elif dtype == torch.complex128:
        func = cpu_blas.zscal
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS: {dtype}")

    func(alpha, ref_x.numpy(), n=n, incx=incx)
    return ref_x


def cpu_csscal_reference(n, alpha, x, incx):
    if n <= 0:
        return to_cpu_blas_tensor(x)

    ref_x = to_cpu_blas_tensor(x)
    scaled_x = cpu_blas.zdscal(alpha, ref_x.numpy(), n=n, incx=incx)
    ref_x.copy_(torch.from_numpy(scaled_x))
    return ref_x


def cpu_zdscal_reference(n, alpha, x, incx):
    if n <= 0:
        return to_cpu_blas_tensor(x)

    ref_x = to_cpu_blas_tensor(x)
    scaled_x = cpu_blas.zdscal(alpha, ref_x.numpy(), n=n, incx=incx)
    ref_x.copy_(torch.from_numpy(scaled_x))
    return ref_x


def scal_reference(n, alpha, x, incx):
    if TO_CPU:
        return cpu_scal_reference(n, alpha, x, incx)

    ref_x = to_reference(x)
    ref_x = ref_x.clone()

    cublas_scal_reference(n, alpha, ref_x, incx)
    return ref_x


def csscal_reference(n, alpha, x, incx):
    if TO_CPU:
        return cpu_csscal_reference(n, alpha, x, incx)

    ref_x = to_reference(x)
    ref_x = ref_x.clone()

    cublas_csscal_reference(n, alpha, ref_x, incx)
    return ref_x


def zdscal_reference(n, alpha, x, incx):
    if TO_CPU:
        return cpu_zdscal_reference(n, alpha, x, incx)

    ref_x = to_reference(x)
    ref_x = ref_x.clone()

    cublas_zdscal_reference(n, alpha, ref_x, incx)
    return ref_x


@pytest.mark.scal
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
def test_accuracy_csscal(shape, alpha):
    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=torch.complex64, device=flag_blas.device)

    ref_x = csscal_reference(n, alpha, x, incx)

    flag_blas.ops.csscal(n, alpha, x, incx)
    blas_assert_close(x, ref_x, torch.complex64)


@pytest.mark.scal
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
def test_accuracy_zdscal(shape, alpha):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=torch.complex128, device=flag_blas.device)

    ref_x = zdscal_reference(n, alpha, x, incx)

    flag_blas.ops.zdscal(n, alpha, x, incx)
    blas_assert_close(x, ref_x, torch.complex128)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
def test_accuracy_scal_real(dtype, shape, alpha):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = scal_reference(n, alpha, x, incx)

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, incx)
    else:
        flag_blas.ops.dscal(n, alpha, x, incx)

    blas_assert_close(x, ref_x, dtype)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", SCAL_SHAPES)
@pytest.mark.parametrize("alpha", COMPLEX_SCALARS + SCALARS)
def test_accuracy_scal_complex(dtype, shape, alpha):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    n = shape[0]
    incx = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = scal_reference(n, alpha, x, incx)

    if dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, incx)
    else:
        flag_blas.ops.zscal(n, alpha, x, incx)

    blas_assert_close(x, ref_x, dtype)


@pytest.mark.scal
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_scal_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64/complex128")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    ref_x = to_reference(x).clone()

    alpha = 2.0 + 1.0j if dtype in [torch.complex64, torch.complex128] else 2.0
    n = 0

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, 1)
    elif dtype == torch.float64:
        flag_blas.ops.dscal(n, alpha, x, 1)
    elif dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, 1)
    else:
        flag_blas.ops.zscal(n, alpha, x, 1)

    blas_assert_close(x, ref_x, dtype)


@pytest.mark.scal
def test_accuracy_csscal_empty_tensor():
    x = torch.randn(0, dtype=torch.complex64, device=flag_blas.device)
    ref_x = to_reference(x).clone()

    alpha = 2.0
    n = 0

    flag_blas.ops.csscal(n, alpha, x, 1)

    blas_assert_close(x, ref_x, torch.complex64)


@pytest.mark.scal
def test_accuracy_zdscal_empty_tensor():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    x = torch.randn(0, dtype=torch.complex128, device=flag_blas.device)
    ref_x = to_reference(x).clone()

    alpha = 2.0
    n = 0

    flag_blas.ops.zdscal(n, alpha, x, 1)

    blas_assert_close(x, ref_x, torch.complex128)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_scal_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    alpha = 2.5
    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = scal_reference(n, alpha, x, 1)

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, 1)
    else:
        flag_blas.ops.dscal(n, alpha, x, 1)

    blas_assert_close(x, ref_x, dtype)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_scal_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    alpha = 2.0 + 1.0j
    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = scal_reference(n, alpha, x, 1)

    if dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, 1)
    else:
        flag_blas.ops.zscal(n, alpha, x, 1)

    blas_assert_close(x, ref_x, dtype)


@pytest.mark.scal
@pytest.mark.parametrize(
    "n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_csscal_different_n(n, vec_size):
    alpha = 2.5
    x = torch.randn(vec_size, dtype=torch.complex64, device=flag_blas.device)

    ref_x = csscal_reference(n, alpha, x, 1)

    flag_blas.ops.csscal(n, alpha, x, 1)
    blas_assert_close(x, ref_x, torch.complex64)


@pytest.mark.scal
@pytest.mark.parametrize(
    "n, vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_zdscal_different_n(n, vec_size):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    alpha = 2.5
    x = torch.randn(vec_size, dtype=torch.complex128, device=flag_blas.device)

    ref_x = zdscal_reference(n, alpha, x, 1)

    flag_blas.ops.zdscal(n, alpha, x, 1)
    blas_assert_close(x, ref_x, torch.complex128)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx", L1_SCALAR_STRIDES)
def test_accuracy_scal_different_n_with_stride_real(dtype, shape, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    alpha = -0.75
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = scal_reference(n, alpha, x, incx)

    if dtype == torch.float32:
        flag_blas.ops.sscal(n, alpha, x, incx)
    else:
        flag_blas.ops.dscal(n, alpha, x, incx)

    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], dtype)


@pytest.mark.scal
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx", L1_SCALAR_STRIDES)
def test_accuracy_scal_different_n_with_stride_complex(dtype, shape, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    n = shape[0]
    alpha = -0.5 + 1.5j
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_x = scal_reference(n, alpha, x, incx)

    if dtype == torch.complex64:
        flag_blas.ops.cscal(n, alpha, x, incx)
    else:
        flag_blas.ops.zscal(n, alpha, x, incx)

    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], dtype)


@pytest.mark.scal
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx", L1_SCALAR_STRIDES)
def test_accuracy_csscal_different_n_with_stride(shape, incx):
    n = shape[0]
    alpha = -0.75
    x = torch.randn(n * incx, dtype=torch.complex64, device=flag_blas.device)

    ref_x = csscal_reference(n, alpha, x, incx)

    flag_blas.ops.csscal(n, alpha, x, incx)
    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], torch.complex64)


@pytest.mark.scal
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx", L1_SCALAR_STRIDES)
def test_accuracy_zdscal_different_n_with_stride(shape, incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")

    n = shape[0]
    alpha = -0.75
    x = torch.randn(n * incx, dtype=torch.complex128, device=flag_blas.device)

    ref_x = zdscal_reference(n, alpha, x, incx)

    flag_blas.ops.zdscal(n, alpha, x, incx)
    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], torch.complex128)

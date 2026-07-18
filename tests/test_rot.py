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
import glob
import os

import cupy as cp
import pytest
import scipy
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import (
    L1_NONUNIT_PAIR_STRIDES,
    L1_STRIDE_SHAPES,
    ROT_SHAPES,
    blas_assert_close,
    to_cpu_blas_tensor,
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


def load_cpu_blas():
    lib_paths = glob.glob(
        os.path.join(
            os.path.dirname(scipy.__file__),
            "..",
            "scipy.libs",
            "libscipy_openblas*.so",
        )
    )
    for path in lib_paths:
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError:
            continue
    raise RuntimeError("Cannot find SciPy OpenBLAS library")


_cpu_blas_lib = load_cpu_blas()


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def cublas_rot_reference(n, x, incx, y, incy, c, s):
    if n <= 0:
        return x, y

    handle = cp.cuda.device.get_cublas_handle()
    dtype = x.dtype

    c_val = c.item() if isinstance(c, torch.Tensor) else c
    s_val = s.item() if isinstance(s, torch.Tensor) else s

    if dtype == torch.float32:
        func = _cublas.cublasSrot_v2
        c_c = ctypes.c_float(c_val)
        s_c = ctypes.c_float(s_val)
    elif dtype == torch.float64:
        func = _cublas.cublasDrot_v2
        c_c = ctypes.c_double(c_val)
        s_c = ctypes.c_double(s_val)
    elif dtype == torch.complex64:
        func = _cublas.cublasCrot_v2
        c_c = ctypes.c_float(c_val)
        s_c = cuComplex(s_val.real, s_val.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZrot_v2
        c_c = ctypes.c_double(c_val)
        s_c = cuDoubleComplex(s_val.real, s_val.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(n),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.byref(c_c),
        ctypes.byref(s_c),
    )

    if status != 0:
        raise RuntimeError(f"cuBLAS rot execution failed, error code: {status}")

    return x, y


def cpu_rot_reference(n, x, incx, y, incy, c, s):
    if n <= 0:
        return to_cpu_blas_tensor(x), to_cpu_blas_tensor(y)

    c_val = c.item() if isinstance(c, torch.Tensor) else c
    s_val = s.item() if isinstance(s, torch.Tensor) else s
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    dtype = ref_x.dtype
    x_np = ref_x.numpy()
    y_np = ref_y.numpy()

    if dtype == torch.float64:
        cpu_blas.drot(
            x_np,
            y_np,
            c_val,
            s_val,
            n=n,
            incx=incx,
            incy=incy,
            overwrite_x=1,
            overwrite_y=1,
        )
    elif dtype == torch.complex128:
        n_c = ctypes.c_int(n)
        incx_c = ctypes.c_int(incx)
        incy_c = ctypes.c_int(incy)
        c_c = ctypes.c_double(c_val)
        s_c = cuDoubleComplex(s_val.real, s_val.imag)
        _cpu_blas_lib.scipy_zrot_(
            ctypes.byref(n_c),
            ctypes.c_void_p(x_np.ctypes.data),
            ctypes.byref(incx_c),
            ctypes.c_void_p(y_np.ctypes.data),
            ctypes.byref(incy_c),
            ctypes.byref(c_c),
            ctypes.byref(s_c),
        )
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    return ref_x, ref_y


def rot_reference(n, x, incx, y, incy, c, s):
    if TO_CPU:
        return cpu_rot_reference(n, x, incx, y, incy, c, s)

    ref_x = to_reference(x)
    ref_y = to_reference(y)
    ref_x = ref_x.clone()
    ref_y = ref_y.clone()

    cublas_rot_reference(n, ref_x, incx, ref_y, incy, c, s)
    return ref_x, ref_y


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


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", ROT_SHAPES)
def test_accuracy_rot_real(dtype, shape):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    incy = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    c, s = get_c_s(dtype, flag_blas.device)

    ref_x, ref_y = rot_reference(n, x, incx, y, incy, c, s)

    if dtype == torch.float32:
        flag_blas.ops.srot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.ops.drot(n, x, incx, y, incy, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", ROT_SHAPES)
def test_accuracy_rot_complex(dtype, shape):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    incx = 1
    incy = 1
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    c, s = get_c_s(dtype, flag_blas.device)

    ref_x, ref_y = rot_reference(n, x, incx, y, incy, c, s)

    if dtype == torch.complex64:
        flag_blas.ops.crot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.ops.zrot(n, x, incx, y, incy, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_rot_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    ref_x = to_reference(x).clone()
    ref_y = to_reference(y).clone()

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

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_rot_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    c, s = get_c_s(dtype, flag_blas.device)

    ref_x, ref_y = rot_reference(n, x, 1, y, 1, c, s)

    if dtype == torch.float32:
        flag_blas.ops.srot(n, x, 1, y, 1, c, s)
    else:
        flag_blas.ops.drot(n, x, 1, y, 1, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_rot_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    c, s = get_c_s(dtype, flag_blas.device)

    ref_x, ref_y = rot_reference(n, x, 1, y, 1, c, s)

    if dtype == torch.complex64:
        flag_blas.ops.crot(n, x, 1, y, 1, c, s)
    else:
        flag_blas.ops.zrot(n, x, 1, y, 1, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_NONUNIT_PAIR_STRIDES)
def test_accuracy_rot_different_n_with_stride_real(dtype, shape, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    c, s = get_c_s(dtype, flag_blas.device)

    ref_x, ref_y = rot_reference(n, x, incx, y, incy, c, s)

    if dtype == torch.float32:
        flag_blas.ops.srot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.ops.drot(n, x, incx, y, incy, c, s)

    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], dtype)
    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


@pytest.mark.rot
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES)
@pytest.mark.parametrize("incx,incy", L1_NONUNIT_PAIR_STRIDES)
def test_accuracy_rot_different_n_with_stride_complex(dtype, shape, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)

    c, s = get_c_s(dtype, flag_blas.device)

    ref_x, ref_y = rot_reference(n, x, incx, y, incy, c, s)

    if dtype == torch.complex64:
        flag_blas.ops.crot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.ops.zrot(n, x, incx, y, incy, c, s)

    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], dtype)
    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)

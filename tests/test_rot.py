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

import pytest
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

IS_HYGON = flag_blas.vendor_name == "hygon"
if IS_HYGON:
    from tests.hipblas_reference import (
        check_hipblas_status,
        get_hipblas_context,
        get_hipblas_library,
    )
else:
    import cupy as cp


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


_cublas = None if IS_HYGON else load_cublas()


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
        func = _cublas.cublasCsrot_v2
        c_c = ctypes.c_float(c_val)
        s_c = ctypes.c_float(s_val)
    elif dtype == torch.complex128:
        func = _cublas.cublasZdrot_v2
        c_c = ctypes.c_double(c_val)
        s_c = ctypes.c_double(s_val)
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

    function = {
        torch.float32: cpu_blas.srot,
        torch.float64: cpu_blas.drot,
        torch.complex64: cpu_blas.csrot,
        torch.complex128: cpu_blas.zdrot,
    }[dtype]
    function(
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

    return ref_x, ref_y


if IS_HYGON:

    def _resolve_hipblas_rot(dtype):
        symbol = {
            torch.float32: "hipblasSrot",
            torch.float64: "hipblasDrot",
            torch.complex64: "hipblasCsrot",
            torch.complex128: "hipblasZdrot",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function

    def _rot_host_scalars(dtype, c, s):
        c = c.item() if isinstance(c, torch.Tensor) else c
        s = s.item() if isinstance(s, torch.Tensor) else s
        if dtype == torch.float32:
            return ctypes.c_float(c), ctypes.c_float(s)
        if dtype == torch.float64:
            return ctypes.c_double(c), ctypes.c_double(s)
        if dtype == torch.complex64:
            return ctypes.c_float(c), ctypes.c_float(s)
        return ctypes.c_double(c), ctypes.c_double(s)

    def hipblas_rot_reference(n, x, incx, y, incy, c, s):
        if n <= 0:
            return x, y
        _, handle = get_hipblas_context(x)
        function = _resolve_hipblas_rot(x.dtype)
        c_host, s_host = _rot_host_scalars(x.dtype, c, s)
        check_hipblas_status(
            function(
                handle,
                n,
                x.data_ptr(),
                incx,
                y.data_ptr(),
                incy,
                ctypes.byref(c_host),
                ctypes.byref(s_host),
            ),
            "hipBLAS ROT",
        )
        return x, y


def rot_reference(n, x, incx, y, incy, c, s):
    if TO_CPU:
        return cpu_rot_reference(n, x, incx, y, incy, c, s)

    ref_x = to_reference(x)
    ref_y = to_reference(y)
    ref_x = ref_x.clone()
    ref_y = ref_y.clone()

    if IS_HYGON:
        hipblas_rot_reference(n, ref_x, incx, ref_y, incy, c, s)
    else:
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
        s = torch.tensor([0.6], dtype=torch.float32, device=device)
    elif dtype == torch.complex128:
        c = torch.tensor([0.8], dtype=torch.float64, device=device)
        s = torch.tensor([0.6], dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return c, s


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srot),
        pytest.param(torch.float64, marks=pytest.mark.drot),
    ],
)
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
        flag_blas.srot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.drot(n, x, incx, y, incy, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.csrot),
        pytest.param(torch.complex128, marks=pytest.mark.zdrot),
    ],
)
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
        flag_blas.csrot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.zdrot(n, x, incx, y, incy, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srot),
        pytest.param(torch.float64, marks=pytest.mark.drot),
        pytest.param(torch.complex64, marks=pytest.mark.csrot),
        pytest.param(torch.complex128, marks=pytest.mark.zdrot),
    ],
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
        flag_blas.srot(n, x, 1, y, 1, c, s)
    elif dtype == torch.float64:
        flag_blas.drot(n, x, 1, y, 1, c, s)
    elif dtype == torch.complex64:
        flag_blas.csrot(n, x, 1, y, 1, c, s)
    else:
        flag_blas.zdrot(n, x, 1, y, 1, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srot),
        pytest.param(torch.float64, marks=pytest.mark.drot),
    ],
)
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
        flag_blas.srot(n, x, 1, y, 1, c, s)
    else:
        flag_blas.drot(n, x, 1, y, 1, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.csrot),
        pytest.param(torch.complex128, marks=pytest.mark.zdrot),
    ],
)
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
        flag_blas.csrot(n, x, 1, y, 1, c, s)
    else:
        flag_blas.zdrot(n, x, 1, y, 1, c, s)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srot),
        pytest.param(torch.float64, marks=pytest.mark.drot),
    ],
)
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
        flag_blas.srot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.drot(n, x, incx, y, incy, c, s)

    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], dtype)
    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.csrot),
        pytest.param(torch.complex128, marks=pytest.mark.zdrot),
    ],
)
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
        flag_blas.csrot(n, x, incx, y, incy, c, s)
    else:
        flag_blas.zdrot(n, x, incx, y, incy, c, s)

    blas_assert_close(x[::incx][:n], ref_x[::incx][:n], dtype)
    blas_assert_close(y[::incy][:n], ref_y[::incy][:n], dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "op,dtype",
    [
        pytest.param("csrot", torch.complex64, marks=pytest.mark.csrot),
        pytest.param("zdrot", torch.complex128, marks=pytest.mark.zdrot),
    ],
)
@pytest.mark.parametrize("c,s", [(1.0, 0.0), (0.0, 1.0), (-0.8, 0.6), (0.8, -0.6)])
@pytest.mark.parametrize("incx,incy", [(1, 1), (2, 3)])
def test_accuracy_complex_rot_real_coefficients(op, dtype, c, s, incx, incy):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    n = 17
    x = torch.randn(n * incx + 1, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy + 1, dtype=dtype, device=flag_blas.device)
    ref_x, ref_y = rot_reference(n, x, incx, y, incy, c, s)
    getattr(flag_blas, op)(n, x, incx, y, incy, c, s)
    # Include untouched stride gaps and the tail in the comparison.
    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rot
@pytest.mark.parametrize(
    "op,dtype",
    [
        pytest.param("csrot", torch.complex64, marks=pytest.mark.csrot),
        pytest.param("zdrot", torch.complex128, marks=pytest.mark.zdrot),
    ],
)
def test_complex_rot_rejects_complex_s(op, dtype):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.ones(1, dtype=dtype, device=flag_blas.device)
    y = torch.ones_like(x)
    with pytest.raises(TypeError):
        getattr(flag_blas, op)(1, x, 1, y, 1, 0.8, 0.36 + 0.48j)

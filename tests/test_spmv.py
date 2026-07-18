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
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
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
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = load_cublas()


def cublas_spmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy):
    if n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    dtype = AP.dtype

    if dtype == torch.float32:
        func = _cublas.cublasSspmv_v2
        alpha_c = ctypes.c_float(alpha)
        beta_c = ctypes.c_float(beta)
    elif dtype == torch.float64:
        func = _cublas.cublasDspmv_v2
        alpha_c = ctypes.c_double(alpha)
        beta_c = ctypes.c_double(beta)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXspmv_v2 execution failed with error code: {status}")


def cpu_spmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy):
    if n == 0:
        return to_cpu_blas_tensor(y)

    ref_AP = to_cpu_blas_tensor(AP)
    ref_x = to_cpu_blas_tensor(x)
    if beta == 0 and incy == 1:
        ref_y = torch.empty(y.shape, dtype=torch.float64)
    else:
        ref_y = to_cpu_blas_tensor(y)

    yout = cpu_blas.dspmv(
        n,
        alpha,
        ref_AP.numpy(),
        ref_x.numpy(),
        incx=incx,
        beta=beta,
        y=ref_y.numpy(),
        incy=incy,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def spmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy):
    if TO_CPU:
        return cpu_spmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy)

    ref_y = y.clone()
    cublas_spmv_reference(uplo, n, alpha, AP, x, incx, beta, ref_y, incy)
    return ref_y


SPMV_SIZES = [
    1,
    2,
    15,
    32,
    63,
    64,
    65,
    127,
    128,
    255,
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
]
SPMV_STRIDE_SIZES = [64, 127, 256]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def make_symmetric_packed(n, uplo, dtype, device):
    return torch.randn(n * (n + 1) // 2, dtype=dtype, device=device)


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _spmv_reduce_dim(n):
    return max(1, n)


@pytest.mark.sspmv
@pytest.mark.parametrize("n", SPMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.5])
def test_accuracy_sspmv(n, uplo, beta):
    dtype, alpha = torch.float32, 1.5

    AP = make_symmetric_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = spmv_reference(uplo, n, alpha, AP, x, 1, beta, y, 1)
    flag_blas.ops.sspmv(uplo, n, alpha, AP, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_spmv_reduce_dim(n))


@pytest.mark.sspmv
@pytest.mark.parametrize("n", SPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_sspmv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.float32, 2.0, 0.5

    AP = make_symmetric_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_y = spmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy)
    flag_blas.ops.sspmv(uplo, n, alpha, AP, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_spmv_reduce_dim(n))


@pytest.mark.sspmv
def test_sspmv_alpha_zero():
    n = 256
    dtype = torch.float32
    AP = make_symmetric_packed(n, CUBLAS_FILL_MODE_UPPER, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()

    y_ref = spmv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0, AP, x, 1, 2.0, y, 1)
    flag_blas.ops.sspmv(CUBLAS_FILL_MODE_UPPER, n, 0.0, AP, x, 1, 2.0, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=_spmv_reduce_dim(n))
    blas_assert_close(y, to_reference(y_orig, upcast=TO_CPU) * 2.0, dtype)


@pytest.mark.sspmv
def test_sspmv_beta_zero():
    n = 256
    dtype = torch.float32
    AP = make_symmetric_packed(n, CUBLAS_FILL_MODE_LOWER, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = spmv_reference(CUBLAS_FILL_MODE_LOWER, n, 1.0, AP, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.sspmv(CUBLAS_FILL_MODE_LOWER, n, 1.0, AP, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.sspmv(CUBLAS_FILL_MODE_LOWER, n, 1.0, AP, x, 1, 0.0, y_zero, 1)
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=_spmv_reduce_dim(n))
    blas_assert_close(
        y_nan,
        to_reference(y_zero, upcast=TO_CPU),
        dtype,
        reduce_dim=_spmv_reduce_dim(n),
    )


@pytest.mark.dspmv
@pytest.mark.parametrize("n", SPMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.5])
def test_accuracy_dspmv(n, uplo, beta):
    check_fp64_support()
    dtype, alpha = torch.float64, 1.5

    AP = make_symmetric_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = spmv_reference(uplo, n, alpha, AP, x, 1, beta, y, 1)
    flag_blas.ops.dspmv(uplo, n, alpha, AP, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_spmv_reduce_dim(n))


@pytest.mark.dspmv
@pytest.mark.parametrize("n", SPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dspmv_stride(n, uplo, incx, incy):
    check_fp64_support()
    dtype, alpha, beta = torch.float64, 2.0, 0.5

    AP = make_symmetric_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_y = spmv_reference(uplo, n, alpha, AP, x, incx, beta, y, incy)
    flag_blas.ops.dspmv(uplo, n, alpha, AP, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=_spmv_reduce_dim(n))


@pytest.mark.dspmv
def test_dspmv_alpha_zero():
    check_fp64_support()
    n = 256
    dtype = torch.float64
    AP = make_symmetric_packed(n, CUBLAS_FILL_MODE_UPPER, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()

    y_ref = spmv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0, AP, x, 1, 2.0, y, 1)
    flag_blas.ops.dspmv(CUBLAS_FILL_MODE_UPPER, n, 0.0, AP, x, 1, 2.0, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=_spmv_reduce_dim(n))
    blas_assert_close(y, to_reference(y_orig, upcast=TO_CPU) * 2.0, dtype)


@pytest.mark.dspmv
def test_dspmv_beta_zero():
    check_fp64_support()
    n = 256
    dtype = torch.float64
    AP = make_symmetric_packed(n, CUBLAS_FILL_MODE_LOWER, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = spmv_reference(CUBLAS_FILL_MODE_LOWER, n, 1.0, AP, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.dspmv(CUBLAS_FILL_MODE_LOWER, n, 1.0, AP, x, 1, 0.0, y_nan, 1)
    flag_blas.ops.dspmv(CUBLAS_FILL_MODE_LOWER, n, 1.0, AP, x, 1, 0.0, y_zero, 1)
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=_spmv_reduce_dim(n))
    blas_assert_close(
        y_nan,
        to_reference(y_zero, upcast=TO_CPU),
        dtype,
        reduce_dim=_spmv_reduce_dim(n),
    )


@pytest.mark.parametrize(
    "dtype, op, alpha, beta",
    [
        (torch.float32, flag_blas.ops.sspmv, 1.5, 0.5),
        (torch.float64, flag_blas.ops.dspmv, 1.5, 0.5),
    ],
)
def test_spmv_n_zero(dtype, op, alpha, beta):
    if dtype == torch.float64:
        check_fp64_support()

    AP = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, AP, x, 1, beta, y, 1)
    assert y.numel() == 0

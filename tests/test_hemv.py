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
import math

import cupy as cp
import pytest
import torch
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


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def cublas_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype

    if dtype == torch.complex64:
        func = _cublas.cublasChemv_v2
        alpha_c = cuComplex(alpha.real, alpha.imag)
        beta_c = cuComplex(beta.real, beta.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZhemv_v2
        alpha_c = cuDoubleComplex(alpha.real, alpha.imag)
        beta_c = cuDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXhemv_v2 execution failed with error code: {status}")


def cpu_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return to_cpu_blas_tensor(y)

    ref_A = to_cpu_blas_tensor(A)
    ref_x = to_cpu_blas_tensor(x)
    if beta == 0 and incy == 1:
        ref_y = torch.empty(y.shape, dtype=torch.complex128)
    else:
        ref_y = to_cpu_blas_tensor(y)
    yout = cpu_blas.zhemv(
        alpha,
        ref_A[:n, :n].T.numpy(),
        ref_x.numpy(),
        beta=beta,
        y=ref_y.numpy(),
        incx=incx,
        incy=incy,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if TO_CPU:
        return cpu_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    ref_y = y.clone()
    cublas_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    return ref_y


HEMV_SIZES = [
    1,
    63,
    128,
    256,
    512,
    1023,
    1024,
    2048,
    4096,
    4097,
    8192,
    16384,
]

HEMV_STRIDE_SIZES = [128, 256, 1024]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def create_hemv_data(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    data = torch.randn(n, n, dtype=dtype, device=device)
    diag_real = data.diagonal().real.clone()
    data.diagonal().copy_(diag_real.to(dtype))
    A[:, :n] = data
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _hemv_tol(dtype, n):
    K = max(1, n)
    if dtype == torch.complex64:
        return min(max(2e-5, 2e-6 * math.sqrt(K)), 2e-3)
    if dtype == torch.complex128:
        return min(max(2e-13, 2e-14 * math.sqrt(K)), 2e-11)
    raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.chemv
@pytest.mark.parametrize("n", HEMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_chemv(n, uplo, beta):
    dtype, alpha = torch.complex64, 1.5 + 0.5j
    lda = n + 2

    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.ops.chemv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.chemv
@pytest.mark.parametrize("n", HEMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_chemv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j
    lda = n

    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.ops.chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.chemv
def test_chemv_alpha_zero():
    n, lda = 256, 258
    dtype = torch.complex64
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = hemv_reference(
        CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.ops.chemv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.chemv
def test_chemv_beta_zero():
    n, lda = 256, 256
    dtype = torch.complex64
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hemv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.ops.chemv(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.ops.chemv(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=n)
    blas_assert_close(y_nan, to_reference(y_zero), dtype, reduce_dim=n)


@pytest.mark.zhemv
@pytest.mark.parametrize("n", HEMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_zhemv(n, uplo, beta):
    check_fp64_support()
    dtype, alpha = torch.complex128, 1.5 + 0.5j
    lda = n + 2

    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.ops.zhemv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.zhemv
@pytest.mark.parametrize("n", HEMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zhemv_stride(n, uplo, incx, incy):
    check_fp64_support()
    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j
    lda = n

    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.ops.zhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.zhemv
def test_zhemv_alpha_zero():
    check_fp64_support()
    n, lda = 256, 258
    dtype = torch.complex128
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = hemv_reference(
        CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.ops.zhemv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.zhemv
def test_zhemv_beta_zero():
    check_fp64_support()
    n, lda = 256, 256
    dtype = torch.complex128
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hemv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.ops.zhemv(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.ops.zhemv(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=n)
    blas_assert_close(y_nan, to_reference(y_zero), dtype, reduce_dim=n)


@pytest.mark.parametrize(
    "dtype, op, alpha, beta",
    [
        (torch.complex64, flag_blas.ops.chemv, 1.5 + 0.5j, 0.5 + 0.25j),
        (torch.complex128, flag_blas.ops.zhemv, 1.5 + 0.5j, 0.5 + 0.25j),
    ],
)
def test_hemv_n_zero(dtype, op, alpha, beta):
    if dtype == torch.complex128:
        check_fp64_support()

    A = torch.empty((0, 2), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, A, 2, x, 1, beta, y, 1)
    assert y.numel() == 0


@pytest.mark.parametrize(
    "dtype, op, alpha, beta, uplo",
    [
        (
            torch.complex64,
            flag_blas.ops.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex64,
            flag_blas.ops.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
        (
            torch.complex128,
            flag_blas.ops.zhemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex128,
            flag_blas.ops.zhemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
    ],
)
def test_hemv_ignored_triangle(dtype, op, alpha, beta, uplo):
    if dtype == torch.complex128:
        check_fp64_support()

    n = 64
    lda = n + 3
    A_clean = create_hemv_data(n, lda, dtype, flag_blas.device)
    A_dirty = A_clean.clone()
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_clean = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_dirty = y_clean.clone()

    tri_upper = torch.triu_indices(n, n, offset=1, device=flag_blas.device)
    tri_lower = torch.tril_indices(n, n, offset=-1, device=flag_blas.device)
    dirty_vals = torch.full(
        (tri_upper.shape[1],),
        complex(float("nan"), float("nan")),
        dtype=dtype,
        device=flag_blas.device,
    )
    if uplo == CUBLAS_FILL_MODE_UPPER:
        A_dirty[tri_upper[0], tri_upper[1]] = dirty_vals
    else:
        A_dirty[tri_lower[0], tri_lower[1]] = dirty_vals[: tri_lower.shape[1]]

    op(uplo, n, alpha, A_clean, lda, x, 1, beta, y_clean, 1)
    op(uplo, n, alpha, A_dirty, lda, x, 1, beta, y_dirty, 1)

    torch.testing.assert_close(y_dirty, y_clean)


@pytest.mark.parametrize(
    "dtype, op, alpha, beta, uplo",
    [
        (
            torch.complex64,
            flag_blas.ops.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex64,
            flag_blas.ops.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
        (
            torch.complex128,
            flag_blas.ops.zhemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex128,
            flag_blas.ops.zhemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
    ],
)
def test_hemv_diagonal_imag_ignored(dtype, op, alpha, beta, uplo):
    if dtype == torch.complex128:
        check_fp64_support()

    n = 128
    lda = n + 2
    A_clean = create_hemv_data(n, lda, dtype, flag_blas.device)
    A_dirty = A_clean.clone()
    diag_imag_noise = torch.randn(n, dtype=dtype, device=flag_blas.device).imag
    diag = A_dirty.diagonal()
    real_part = diag.real.clone()
    diag.copy_((real_part + 1j * diag_imag_noise).to(dtype))

    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_clean = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y_dirty = y_clean.clone()

    op(uplo, n, alpha, A_clean, lda, x, 1, beta, y_clean, 1)
    op(uplo, n, alpha, A_dirty, lda, x, 1, beta, y_dirty, 1)

    tol = _hemv_tol(dtype, n)
    torch.testing.assert_close(y_dirty, y_clean, rtol=tol, atol=tol)

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

if flag_blas.vendor_name in {"hygon", "mthreads"}:
    from .vendor_blas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )
elif flag_blas.vendor_name not in {"ascend", "mthreads"}:
    import cupy as cp

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


_cublas = (
    None
    if flag_blas.vendor_name in {"ascend", "hygon", "mthreads"}
    else load_cublas()
)


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def row_to_column_full(A, n, lda):
    column_A = torch.zeros((n, lda), dtype=A.dtype, device=A.device)
    column_A[:, :n] = A[:n, :n].T
    return column_A


def hipblas_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return y

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else complex(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else complex(beta)

    if A.dtype == torch.complex64:
        symbol = "hipblasChemv_v2"
        alpha_value = HipComplex(alpha.real, alpha.imag)
        beta_value = HipComplex(beta.real, beta.imag)
    elif A.dtype == torch.complex128:
        symbol = "hipblasZhemv_v2"
        alpha_value = HipDoubleComplex(alpha.real, alpha.imag)
        beta_value = HipDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS HEMV: {A.dtype}")

    column_A = row_to_column_full(A, n, lda)
    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    library, handle = get_hipblas_context(column_A)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    check_hipblas_status(
        function(
            handle,
            hip_uplo,
            n,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(column_A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        symbol,
    )
    return y


def cublas_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return

    column_A = row_to_column_full(A, n, lda)
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
        ctypes.c_void_p(column_A.data_ptr()),
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
        ref_A[:n, :n].numpy(),
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
    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        hipblas_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    else:
        cublas_hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    return ref_y


HEMV_SIZES = [
    1,
    63,
    128,
    192,
    256,
    384,
    512,
    768,
    1023,
    1024,
    1536,
    2048,
    3072,
    4096,
    4097,
    6144,
    8192,
    12288,
    16384,
]

HEMV_STRIDE_SIZES = [128, 256, 1024]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def hemv_randn(*shape, dtype, device):
    if flag_blas.vendor_name in ("ascend", "mthreads") and dtype == torch.complex64:
        # Build complex inputs from real-valued random tensors on non-CUDA backends.
        normalized = (
            tuple(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (tuple, torch.Size))
            else shape
        )
        values = torch.randn((*normalized, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(*shape, dtype=dtype, device=device)


def create_hemv_data(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    data = hemv_randn(n, n, dtype=dtype, device=device)
    diag_real = data.diagonal().real.clone()
    data.diagonal().copy_(diag_real.to(dtype))
    A[:, :n] = data
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


@pytest.mark.chemv
@pytest.mark.parametrize("n", HEMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_chemv(n, uplo, beta):
    dtype, alpha = torch.complex64, 1.5 + 0.5j
    lda = n + 2

    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.chemv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.chemv
@pytest.mark.parametrize("n", HEMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_chemv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j
    lda = n

    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = hemv_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = hemv_randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.chemv
def test_chemv_alpha_zero():
    n, lda = 256, 258
    dtype = torch.complex64
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = hemv_reference(
        CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.chemv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.chemv
def test_chemv_beta_zero():
    n, lda = 256, 256
    dtype = torch.complex64
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hemv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.chemv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1)
    flag_blas.chemv(
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
    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.zhemv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

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
    x = hemv_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = hemv_randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = hemv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.zhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.zhemv
def test_zhemv_alpha_zero():
    check_fp64_support()
    n, lda = 256, 258
    dtype = torch.complex128
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = hemv_reference(
        CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.zhemv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.zhemv
def test_zhemv_beta_zero():
    check_fp64_support()
    n, lda = 256, 256
    dtype = torch.complex128
    A = create_hemv_data(n, lda, dtype, flag_blas.device)
    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = hemv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.zhemv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1)
    flag_blas.zhemv(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=n)
    blas_assert_close(y_nan, to_reference(y_zero), dtype, reduce_dim=n)


@pytest.mark.parametrize(
    "dtype, op, alpha, beta",
    [
        (torch.complex64, flag_blas.chemv, 1.5 + 0.5j, 0.5 + 0.25j),
        (torch.complex128, flag_blas.zhemv, 1.5 + 0.5j, 0.5 + 0.25j),
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
            flag_blas.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex64,
            flag_blas.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
        (
            torch.complex128,
            flag_blas.zhemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex128,
            flag_blas.zhemv,
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
    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y_clean = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y_dirty = y_clean.clone()

    tri_upper = torch.triu_indices(n, n, offset=1, device=flag_blas.device)
    tri_lower = torch.tril_indices(n, n, offset=-1, device=flag_blas.device)
    dirty_index = tri_lower if uplo == CUBLAS_FILL_MODE_UPPER else tri_upper
    if flag_blas.vendor_name in ("ascend", "mthreads") and dtype == torch.complex64:
        dirty_parts = torch.view_as_real(A_dirty)
        dirty_parts[dirty_index[0], dirty_index[1], 0] = float("nan")
        dirty_parts[dirty_index[0], dirty_index[1], 1] = float("nan")
    else:
        dirty_vals = torch.full(
            (dirty_index.shape[1],),
            complex(float("nan"), float("nan")),
            dtype=dtype,
            device=flag_blas.device,
        )
        A_dirty[dirty_index[0], dirty_index[1]] = dirty_vals

    op(uplo, n, alpha, A_clean, lda, x, 1, beta, y_clean, 1)
    op(uplo, n, alpha, A_dirty, lda, x, 1, beta, y_dirty, 1)

    blas_assert_close(y_dirty, to_reference(y_clean), dtype, reduce_dim=n)


@pytest.mark.parametrize(
    "dtype, op, alpha, beta, uplo",
    [
        (
            torch.complex64,
            flag_blas.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex64,
            flag_blas.chemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
        (
            torch.complex128,
            flag_blas.zhemv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex128,
            flag_blas.zhemv,
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
    diag_imag_noise = hemv_randn(n, dtype=dtype, device=flag_blas.device).imag
    if flag_blas.vendor_name == "mthreads" and dtype == torch.complex64:
        diag_idx = torch.arange(n, device=flag_blas.device)
        torch.view_as_real(A_dirty)[diag_idx, diag_idx, 1] = diag_imag_noise
    else:
        diag = A_dirty.diagonal()
        real_part = diag.real.clone()
        diag.copy_((real_part + 1j * diag_imag_noise).to(dtype))

    x = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y_clean = hemv_randn(n, dtype=dtype, device=flag_blas.device)
    y_dirty = y_clean.clone()

    op(uplo, n, alpha, A_clean, lda, x, 1, beta, y_clean, 1)
    op(uplo, n, alpha, A_dirty, lda, x, 1, beta, y_dirty, 1)

    blas_assert_close(y_dirty, to_reference(y_clean), dtype, reduce_dim=n)

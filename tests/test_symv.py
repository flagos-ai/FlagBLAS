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

# DEBUG: symv 测试当前处于调试（debug）阶段，尚未稳定收敛

import ctypes
import ctypes.util

import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

if flag_blas.vendor_name == "hygon":
    from .hipblas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )
elif flag_blas.vendor_name != "ascend":
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


_cublas = None if flag_blas.vendor_name in {"ascend", "hygon"} else load_cublas()


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def hipblas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return y

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta

    if A.dtype == torch.float32:
        symbol = "hipblasSsymv"
        alpha_value = ctypes.c_float(float(alpha))
        beta_value = ctypes.c_float(float(beta))
    elif A.dtype == torch.float64:
        symbol = "hipblasDsymv"
        alpha_value = ctypes.c_double(float(alpha))
        beta_value = ctypes.c_double(float(beta))
    elif A.dtype == torch.complex64:
        symbol = "hipblasCsymv_v2"
        alpha = complex(alpha)
        beta = complex(beta)
        alpha_value = HipComplex(alpha.real, alpha.imag)
        beta_value = HipComplex(beta.real, beta.imag)
    elif A.dtype == torch.complex128:
        symbol = "hipblasZsymv_v2"
        alpha = complex(alpha)
        beta = complex(beta)
        alpha_value = HipDoubleComplex(alpha.real, alpha.imag)
        beta_value = HipDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS SYMV: {A.dtype}")

    hip_uplo = 122 if uplo == CUBLAS_FILL_MODE_UPPER else 121
    library, handle = get_hipblas_context(A)
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
            ctypes.c_void_p(A.data_ptr()),
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


def cublas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype

    if dtype == torch.float32:
        func = _cublas.cublasSsymv_v2
        alpha_c = ctypes.c_float(alpha)
        beta_c = ctypes.c_float(beta)
    elif dtype == torch.float64:
        func = _cublas.cublasDsymv_v2
        alpha_c = ctypes.c_double(alpha)
        beta_c = ctypes.c_double(beta)
    elif dtype == torch.complex64:
        func = _cublas.cublasCsymv_v2
        alpha_c = cuComplex(alpha.real, alpha.imag)
        beta_c = cuComplex(beta.real, beta.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZsymv_v2
        alpha_c = cuDoubleComplex(alpha.real, alpha.imag)
        beta_c = cuDoubleComplex(beta.real, beta.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(
            CUBLAS_FILL_MODE_LOWER
            if uplo == CUBLAS_FILL_MODE_UPPER
            else CUBLAS_FILL_MODE_UPPER
        ),
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
        raise RuntimeError(f"cublasXsymv_v2 execution failed with error code: {status}")


def cpu_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n == 0:
        return to_cpu_blas_tensor(y)

    ref_A = to_cpu_blas_tensor(A)
    ref_x = to_cpu_blas_tensor(x)
    if beta == 0 and incy == 1:
        ref_dtype = torch.complex128 if y.is_complex() else torch.float64
        ref_y = torch.empty(y.shape, dtype=ref_dtype)
    else:
        ref_y = to_cpu_blas_tensor(y)
    logical_A = ref_A[:n, :n]

    if ref_A.dtype.is_complex:
        # SciPy BLAS does not expose csymv/zsymv. Build the missing symmetric
        # half without conjugation and use zgemv as the CPU complex128 reference.
        block = 512
        for row_start in range(0, n, block):
            row_end = min(row_start + block, n)
            for col_start in range(0, n, block):
                col_end = min(col_start + block, n)
                if uplo == CUBLAS_FILL_MODE_UPPER and row_start >= col_end:
                    logical_A[row_start:row_end, col_start:col_end].copy_(
                        logical_A[col_start:col_end, row_start:row_end].T
                    )
                elif uplo == CUBLAS_FILL_MODE_LOWER and col_start >= row_end:
                    logical_A[row_start:row_end, col_start:col_end].copy_(
                        logical_A[col_start:col_end, row_start:row_end].T
                    )
                elif row_start == col_start:
                    size = row_end - row_start
                    if uplo == CUBLAS_FILL_MODE_UPPER:
                        dst_r, dst_c = torch.tril_indices(size, size, -1)
                    else:
                        dst_r, dst_c = torch.triu_indices(size, size, 1)
                    tile = logical_A[row_start:row_end, row_start:row_end]
                    tile[dst_r, dst_c] = tile[dst_c, dst_r]

        yout = cpu_blas.zgemv(
            alpha,
            logical_A.numpy(),
            ref_x.numpy(),
            beta=beta,
            y=ref_y.numpy(),
            incx=incx,
            incy=incy,
            overwrite_y=1,
        )
        return torch.from_numpy(yout)

    yout = cpu_blas.dsymv(
        alpha,
        logical_A.numpy(),
        ref_x.numpy(),
        beta=beta,
        y=ref_y.numpy(),
        incx=incx,
        incy=incy,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if TO_CPU:
        return cpu_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    ref_y = y.clone()
    if flag_blas.vendor_name == "hygon":
        hipblas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    else:
        cublas_symv_reference(uplo, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    return ref_y


SYMV_SIZES = [
    1,
    2,
    15,
    32,
    63,
    64,
    65,
    127,
    128,
    192,
    255,
    256,
    384,
    512,
    768,
    1024,
    1023,
    1025,
    1536,
    2048,
    3072,
    4095,
    4096,
    6144,
    8192,
    9999,
    10000,
    12288,
    16384,
]

SYMV_STRIDE_SIZES = [64, 127, 256]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def symv_randn(*shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        normalized = (
            tuple(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (tuple, torch.Size))
            else shape
        )
        values = torch.randn((*normalized, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(*shape, dtype=dtype, device=device)


def create_symv_data(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    data = symv_randn(n, n, dtype=dtype, device=device)
    A[:, :n] = data
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


@pytest.mark.ssymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_ssymv(n, uplo, beta):
    dtype, alpha = torch.float32, 1.5
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.ssymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.ssymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_ssymv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.float32, 2.0, 0.5
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.ssymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.ssymv
def test_ssymv_alpha_zero():
    n, lda = 256, 258
    dtype = torch.float32
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = symv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y, 1)
    flag_blas.ssymv(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * 2.0), dtype)


@pytest.mark.ssymv
def test_ssymv_beta_zero():
    n, lda = 256, 256
    dtype = torch.float32
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = symv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_nan, 1
    )
    flag_blas.ssymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.ssymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_zero, 1)
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=n)
    blas_assert_close(y_nan, to_reference(y_zero), dtype, reduce_dim=n)


@pytest.mark.dsymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_dsymv(n, uplo, beta):
    check_fp64_support()
    dtype, alpha = torch.float64, 1.5
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.dsymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.dsymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dsymv_stride(n, uplo, incx, incy):
    check_fp64_support()
    dtype, alpha, beta = torch.float64, 2.0, 0.5
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.dsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.dsymv
def test_dsymv_alpha_zero():
    check_fp64_support()
    n, lda = 256, 258
    dtype = torch.float64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = symv_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y, 1)
    flag_blas.dsymv(CUBLAS_FILL_MODE_UPPER, n, 0.0, A, lda, x, 1, 2.0, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * 2.0), dtype)


@pytest.mark.dsymv
def test_dsymv_beta_zero():
    check_fp64_support()
    n, lda = 256, 256
    dtype = torch.float64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = symv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_nan, 1
    )
    flag_blas.dsymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_nan, 1)
    flag_blas.dsymv(CUBLAS_FILL_MODE_LOWER, n, 1.0, A, lda, x, 1, 0.0, y_zero, 1)
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=n)
    blas_assert_close(y_nan, to_reference(y_zero), dtype, reduce_dim=n)


@pytest.mark.csymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_csymv(n, uplo, beta):
    dtype, alpha = torch.complex64, 1.5 + 0.5j
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.csymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.csymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_csymv_stride(n, uplo, incx, incy):
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.csymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.csymv
def test_csymv_alpha_zero():
    n, lda = 256, 258
    dtype = torch.complex64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = symv_reference(
        CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.csymv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.csymv
def test_csymv_beta_zero():
    n, lda = 256, 256
    dtype = torch.complex64
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = symv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.csymv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1)
    flag_blas.csymv(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=n)
    blas_assert_close(y_nan, to_reference(y_zero), dtype, reduce_dim=n)


@pytest.mark.zsymv
@pytest.mark.parametrize("n", SYMV_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("beta", [0.0j, 0.5 + 0.25j])
def test_accuracy_zsymv(n, uplo, beta):
    check_fp64_support()
    dtype, alpha = torch.complex128, 1.5 + 0.5j
    lda = n + 2

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, 1, beta, y, 1)
    flag_blas.zsymv(uplo, n, alpha, A, lda, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.zsymv
@pytest.mark.parametrize("n", SYMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zsymv_stride(n, uplo, incx, incy):
    check_fp64_support()
    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j
    lda = n

    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_y = symv_reference(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    flag_blas.zsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=n)


@pytest.mark.zsymv
def test_zsymv_alpha_zero():
    check_fp64_support()
    n, lda = 256, 258
    dtype = torch.complex128
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y_orig = y.clone()
    y_ref = symv_reference(
        CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1
    )
    flag_blas.zsymv(CUBLAS_FILL_MODE_UPPER, n, 0.0j, A, lda, x, 1, 2.0 + 1.0j, y, 1)
    blas_assert_close(y, y_ref, dtype, reduce_dim=n)
    blas_assert_close(y, to_reference(y_orig * (2.0 + 1.0j)), dtype)


@pytest.mark.zsymv
def test_zsymv_beta_zero():
    check_fp64_support()
    n, lda = 256, 256
    dtype = torch.complex128
    A = create_symv_data(n, lda, dtype, flag_blas.device)
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)

    y_nan = torch.full((n,), float("nan"), dtype=dtype, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=dtype, device=flag_blas.device)
    ref_y_nan = symv_reference(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1
    )
    flag_blas.zsymv(CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_nan, 1)
    flag_blas.zsymv(
        CUBLAS_FILL_MODE_LOWER, n, 1.0 + 0.5j, A, lda, x, 1, 0.0j, y_zero, 1
    )
    blas_assert_close(y_nan, ref_y_nan, dtype, reduce_dim=n)
    blas_assert_close(y_nan, to_reference(y_zero), dtype, reduce_dim=n)


@pytest.mark.parametrize(
    "dtype, op, alpha, beta",
    [
        (torch.float32, flag_blas.ssymv, 1.5, 0.5),
        (torch.float64, flag_blas.dsymv, 1.5, 0.5),
        (torch.complex64, flag_blas.csymv, 1.5 + 0.5j, 0.5 + 0.25j),
        (torch.complex128, flag_blas.zsymv, 1.5 + 0.5j, 0.5 + 0.25j),
    ],
)
def test_symv_n_zero(dtype, op, alpha, beta):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()

    A = torch.empty((0, 2), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, A, 2, x, 1, beta, y, 1)
    assert y.numel() == 0


@pytest.mark.parametrize(
    "dtype, op, alpha, beta, uplo",
    [
        (torch.float32, flag_blas.ssymv, 1.25, 0.5, CUBLAS_FILL_MODE_UPPER),
        (torch.float32, flag_blas.ssymv, 1.25, 0.5, CUBLAS_FILL_MODE_LOWER),
        (torch.float64, flag_blas.dsymv, 1.25, 0.5, CUBLAS_FILL_MODE_UPPER),
        (torch.float64, flag_blas.dsymv, 1.25, 0.5, CUBLAS_FILL_MODE_LOWER),
        (
            torch.complex64,
            flag_blas.csymv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex64,
            flag_blas.csymv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
        (
            torch.complex128,
            flag_blas.zsymv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_UPPER,
        ),
        (
            torch.complex128,
            flag_blas.zsymv,
            1.25 + 0.5j,
            0.5 + 0.25j,
            CUBLAS_FILL_MODE_LOWER,
        ),
    ],
)
def test_symv_ignored_triangle(dtype, op, alpha, beta, uplo):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()

    n = 64
    lda = n + 3
    A_clean = create_symv_data(n, lda, dtype, flag_blas.device)
    A_dirty = A_clean.clone()
    x = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y_clean = symv_randn(n, dtype=dtype, device=flag_blas.device)
    y_dirty = y_clean.clone()

    tri_upper = torch.triu_indices(n, n, offset=1, device=flag_blas.device)
    tri_lower = torch.tril_indices(n, n, offset=-1, device=flag_blas.device)
    dirty_index = tri_lower if uplo == CUBLAS_FILL_MODE_UPPER else tri_upper
    if flag_blas.vendor_name == "ascend" and dtype.is_complex:
        dirty_parts = torch.view_as_real(A_dirty)
        dirty_parts[dirty_index[0], dirty_index[1], 0] = float("nan")
        dirty_parts[dirty_index[0], dirty_index[1], 1] = float("nan")
    else:
        dirty_vals = torch.full(
            (dirty_index.shape[1],),
            complex(float("nan"), float("nan")) if dtype.is_complex else float("nan"),
            dtype=dtype,
            device=flag_blas.device,
        )
        A_dirty[dirty_index[0], dirty_index[1]] = dirty_vals

    op(uplo, n, alpha, A_clean, lda, x, 1, beta, y_clean, 1)
    op(uplo, n, alpha, A_dirty, lda, x, 1, beta, y_dirty, 1)

    blas_assert_close(y_dirty, to_reference(y_clean), dtype, reduce_dim=n)

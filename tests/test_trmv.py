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
    from .vendor_blas_reference import check_hipblas_status, get_hipblas_context
elif flag_blas.vendor_name not in {"ascend", "mthreads"}:
    import cupy as cp
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor
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


def row_to_column_full(A, n, lda):
    column_A = torch.empty((n, lda), dtype=A.dtype, device=A.device)
    if n > 0:
        column_A[:, :n] = A[:n, :n].T
    return column_A


def hipblas_trmv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if n == 0:
        return x

    if A.dtype == torch.float32:
        symbol = "hipblasStrmv"
    elif A.dtype == torch.float64:
        symbol = "hipblasDtrmv"
    elif A.dtype == torch.complex64:
        symbol = "hipblasCtrmv_v2"
    elif A.dtype == torch.complex128:
        symbol = "hipblasZtrmv_v2"
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS TRMV: {A.dtype}")

    column_A = row_to_column_full(A, n, lda)
    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    hip_trans = 111 + trans
    hip_diag = 131 + diag
    library, handle = get_hipblas_context(column_A)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    check_hipblas_status(
        function(
            handle,
            hip_uplo,
            hip_trans,
            hip_diag,
            n,
            ctypes.c_void_p(column_A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
        ),
        symbol,
    )
    return x


def cublas_trmv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if n == 0:
        return
    column_A = row_to_column_full(A, n, lda)
    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStrmv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtrmv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtrmv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtrmv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_void_p(column_A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtrmv_v2 execution failed with error code: {status}")


def cpu_trmv_reference(uplo, trans, diag, n, A, lda, x, incx):
    ref_x = to_cpu_blas_tensor(x)
    if n == 0:
        return ref_x

    ref_A = to_cpu_blas_tensor(A)
    func = cpu_blas.ztrmv if ref_A.dtype.is_complex else cpu_blas.dtrmv
    xout = func(
        ref_A[:n, :n].numpy(),
        ref_x.numpy(),
        incx=incx,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        trans=trans,
        diag=diag,
        overwrite_x=1,
    )
    return torch.from_numpy(xout)


def trmv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if TO_CPU:
        return cpu_trmv_reference(uplo, trans, diag, n, A, lda, x, incx)

    ref_x = x.clone()
    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        hipblas_trmv_reference(uplo, trans, diag, n, A, lda, ref_x, incx)
    else:
        cublas_trmv_reference(uplo, trans, diag, n, A, lda, ref_x, incx)
    return ref_x


TRMV_SIZES = [
    0,
    1,
    8,
    31,
    64,
    127,
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
    6144,
    8192,
    10000,
    12288,
    16384,
]
TRMV_STRIDE_SIZES = [64, 127, 256]
INCS = [1, 2, 3]
LDA_EXTRAS = [0]
LDA_EXTRAS_STRIDE = [0, 1]


def make_triangular(n, lda, uplo, diag, dtype, device):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    A = trmv_randn((n, lda), dtype, device)
    if not dtype.is_complex:
        A = A * 0.1
    rows = torch.arange(n, device=device).view(n, 1)
    cols = torch.arange(lda, device=device).view(1, lda)
    unit = diag == CUBLAS_DIAG_UNIT

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = (cols >= rows) & (cols < n)
    else:
        valid = cols <= rows
    if unit:
        valid &= rows != cols

    if dtype.is_complex:
        torch.view_as_real(A).masked_fill_(~valid.unsqueeze(-1), float("nan"))
    else:
        A.masked_fill_(~valid, float("nan"))
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def trmv_randn(shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


@pytest.mark.strmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_strmv(n, uplo, trans, diag, lda_extra):
    dtype = torch.float32
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.strmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.strmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_strmv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.float32
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.strmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_dtrmv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.dtrmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_dtrmv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.dtrmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ctrmv(n, uplo, trans, diag, lda_extra):
    dtype = torch.complex64
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ctrmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ctrmv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.complex64
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ctrmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrmv
@pytest.mark.parametrize("n", TRMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ztrmv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ztrmv(uplo, trans, diag, n, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrmv
@pytest.mark.parametrize("n", TRMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ztrmv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trmv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ztrmv(uplo, trans, diag, n, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n)

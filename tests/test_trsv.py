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

IS_ASCEND = flag_blas.vendor_name == "ascend"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if not IS_ASCEND:
    import ctypes
    import ctypes.util


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


def hipblas_trsv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if n == 0:
        return x

    if A.dtype == torch.float32:
        symbol = "hipblasStrsv"
    elif A.dtype == torch.float64:
        symbol = "hipblasDtrsv"
    elif A.dtype == torch.complex64:
        symbol = "hipblasCtrsv_v2"
    elif A.dtype == torch.complex128:
        symbol = "hipblasZtrsv_v2"
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS TRSV: {A.dtype}")

    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    hip_trans = 111 + trans
    hip_diag = 131 + diag
    library, handle = get_hipblas_context(A)
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
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
        ),
        symbol,
    )
    return x


def cublas_trsv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if n == 0:
        return
    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStrsv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtrsv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtrsv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtrsv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtrsv_v2 execution failed with error code: {status}")
    cp.cuda.runtime.deviceSynchronize()


def cpu_trsv_reference(uplo, trans, diag, n, A, lda, x, incx):
    ref_x = to_cpu_blas_tensor(x)
    if n == 0:
        return ref_x

    ref_A = to_cpu_blas_tensor(A)
    func = cpu_blas.ztrsv if ref_A.dtype.is_complex else cpu_blas.dtrsv
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


def trsv_reference(uplo, trans, diag, n, A, lda, x, incx):
    ref_x = x.clone()

    if TO_CPU:
        return cpu_trsv_reference(uplo, trans, diag, n, A, lda, ref_x, incx)

    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        hipblas_trsv_reference(uplo, trans, diag, n, A, lda, ref_x, incx)
    else:
        cublas_trsv_reference(uplo, trans, diag, n, A, lda, ref_x, incx)
    return ref_x


@pytest.mark.parametrize(
    "uplo,trans,diag,dtype,incx",
    [
        (
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            torch.float32,
            1,
        ),
        (
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T,
            CUBLAS_DIAG_UNIT,
            torch.float32,
            2,
        ),
        (
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,
            CUBLAS_DIAG_UNIT,
            torch.float32,
            3,
        ),
        (
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            torch.complex64,
            2,
        ),
        (
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T,
            CUBLAS_DIAG_UNIT,
            torch.complex64,
            3,
        ),
        (
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_C,
            CUBLAS_DIAG_NON_UNIT,
            torch.complex64,
            2,
        ),
    ],
)
def test_cpu_trsv_reference(uplo, trans, diag, dtype, incx):
    n, lda = 3, 5
    dense_dtype = torch.complex128 if dtype.is_complex else torch.float64
    values = torch.tensor(
        [[3.0, 0.5, -0.25], [0.75, 4.0, 0.4], [-0.5, 0.2, 5.0]],
        dtype=dense_dtype,
    )
    if dtype.is_complex:
        values = values + 1j * torch.tensor(
            [[0.25, -0.5, 0.1], [0.3, -0.2, 0.6], [-0.4, 0.15, 0.5]],
            dtype=torch.float64,
        )
    matrix = (
        torch.triu(values) if uplo == CUBLAS_FILL_MODE_UPPER else torch.tril(values)
    )
    physical = torch.full((n, lda), float("nan"), dtype=dtype)
    physical[:, :n] = matrix.to(dtype)
    if diag == CUBLAS_DIAG_UNIT:
        physical.diagonal().fill_(float("nan"))

    x = torch.full((1 + (n - 1) * incx,), 99, dtype=dtype)
    logical_x = torch.tensor([1.0, -2.0, 0.75], dtype=dense_dtype)
    if dtype.is_complex:
        logical_x = logical_x + 1j * torch.tensor(
            [0.5, -0.25, 1.25], dtype=torch.float64
        )
    x[::incx] = logical_x.to(dtype)

    actual = cpu_trsv_reference(uplo, trans, diag, n, physical, lda, x, incx)
    effective = physical[:, :n].to(dense_dtype)
    if diag == CUBLAS_DIAG_UNIT:
        effective.diagonal().fill_(1)
    if trans == CUBLAS_OP_T:
        effective = effective.T
    elif trans == CUBLAS_OP_C:
        effective = effective.mH
    expected = x.to(dense_dtype)
    expected[::incx] = torch.linalg.solve(effective, x[::incx].to(dense_dtype))

    blas_assert_close(actual, expected, actual.dtype, reduce_dim=n)


TRSV_CASES = [
    (0, 0),
    (1, 0),
    (64, 0),
    (256, 0),
    (512, 0),
    (1024, 0),
    (2048, 0),
    (4096, 0),
    (8192, 0),
    (255, 7),
    (1024, 16),
]
TRSV_STRIDE_CASES = [(64, 0), (1023, 0)]
INCS = [1, 2, 3]
UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


def trsv_randn(shape, dtype, device):
    if (IS_ASCEND or IS_MTHREADS) and dtype == torch.complex64:
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


def column_major_copy(A, n, lda):
    storage = torch.empty((n, lda), dtype=A.dtype, device=A.device)
    A_col = storage.T[:n, :]
    A_col.copy_(A[:, :n])
    return A_col


def make_triangular(n, lda, uplo, diag, dtype, device):
    if n == 0:
        A = torch.zeros((n, lda), dtype=dtype, device=device)
        return column_major_copy(A, n, lda), A
    A = torch.empty((n, lda), dtype=dtype, device=device)
    if dtype.is_complex:
        torch.view_as_real(A).fill_(float("nan"))
    else:
        A.fill_(float("nan"))
    vals = trsv_randn((n, n), dtype, device) * 0.02
    row_idx = torch.arange(n, device=device).view(n, 1)
    col_idx = torch.arange(n, device=device).view(1, n)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = row_idx <= col_idx
    else:
        valid = row_idx >= col_idx
    if diag == CUBLAS_DIAG_UNIT:
        valid = valid & (row_idx != col_idx)
    if (IS_ASCEND or IS_MTHREADS) and dtype == torch.complex64:
        vals_real = torch.view_as_real(vals)
        if diag == CUBLAS_DIAG_NON_UNIT:
            diag_real = torch.diagonal(vals_real, dim1=0, dim2=1)
            diag_real[0].add_(2.0)
            diag_real[1].add_(0.25)
        torch.view_as_real(A)[:, :n].copy_(
            vals_real.masked_fill(~valid.unsqueeze(-1), float("nan"))
        )
        return column_major_copy(A, n, lda), A.contiguous()
    A[:, :n] = vals.masked_fill(~valid, float("nan"))
    if diag == CUBLAS_DIAG_NON_UNIT:
        diag_vals = torch.diagonal(vals).clone()
        if dtype.is_complex:
            diag_vals = diag_vals + (2.0 + 0.25j)
        else:
            diag_vals = diag_vals + 2.0
        idx = torch.arange(n, device=device)
        A[idx, idx] = diag_vals
    return column_major_copy(A, n, lda), A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


@pytest.mark.strsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_strsv(n, uplo, trans, diag, lda_extra):
    dtype = torch.float32
    lda = max(1, n + lda_extra)
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, 1)
    flag_blas.strsv(uplo, trans, diag, n, A_row, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.strsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_strsv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.float32
    lda = n + lda_extra
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, incx)
    flag_blas.strsv(uplo, trans, diag, n, A_row, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_dtrsv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = max(1, n + lda_extra)
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, 1)
    flag_blas.dtrsv(uplo, trans, diag, n, A_row, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_dtrsv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = n + lda_extra
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, incx)
    flag_blas.dtrsv(uplo, trans, diag, n, A_row, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ctrsv(n, uplo, trans, diag, lda_extra):
    dtype = torch.complex64
    lda = max(1, n + lda_extra)
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, 1)
    flag_blas.ctrsv(uplo, trans, diag, n, A_row, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_ctrsv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.complex64
    lda = n + lda_extra
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, incx)
    flag_blas.ctrsv(uplo, trans, diag, n, A_row, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ztrsv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = max(1, n + lda_extra)
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, 1)
    flag_blas.ztrsv(uplo, trans, diag, n, A_row, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_ztrsv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = n + lda_extra
    A_col, A_row = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = trsv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A_col, lda, x, incx)
    flag_blas.ztrsv(uplo, trans, diag, n, A_row, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)

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

from .vendor_blas_reference import check_hipblas_status, get_hipblas_context
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
    None if flag_blas.vendor_name in {"ascend", "hygon", "mthreads"} else load_cublas()
)


def row_to_column_band(A, n, k, lda, uplo):
    column_A = torch.zeros((n, lda), dtype=A.dtype, device=A.device)
    if n == 0:
        return column_A
    rows = torch.arange(n, device=A.device).view(n, 1)
    bands = torch.arange(k + 1, device=A.device).view(1, k + 1)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        columns = rows + bands
    else:
        columns = rows + bands - k
    column_bands = (k - bands).expand(n, k + 1)
    valid = (columns >= 0) & (columns < n)
    dst_rows = columns.expand(n, k + 1)[valid]
    dst_cols = column_bands[valid]
    if flag_blas.vendor_name == "mthreads" and A.dtype.is_complex:
        column_real = torch.view_as_real(column_A)
        source_real = torch.view_as_real(A[:, : k + 1])
        column_real[dst_rows, dst_cols] = source_real[valid]
    else:
        column_A[dst_rows, dst_cols] = A[:, : k + 1][valid]
    return column_A.contiguous()


def cublas_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0:
        return

    handle = ctypes.c_void_p()
    status = _cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with error code: {status}")
    column_A = row_to_column_band(A, n, k, lda, uplo)
    dtype = column_A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStbsv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtbsv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtbsv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtbsv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    try:
        status = func(
            handle,
            ctypes.c_int(uplo),
            ctypes.c_int(trans),
            ctypes.c_int(diag),
            ctypes.c_int(n),
            ctypes.c_int(k),
            ctypes.c_void_p(column_A.data_ptr()),
            ctypes.c_int(lda),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int(incx),
        )
        if status != 0:
            raise RuntimeError(
                f"cublasXtbsv_v2 execution failed with error code: {status}"
            )
    finally:
        _cublas.cublasDestroy_v2(handle)


def hipblas_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0:
        return x

    if A.dtype == torch.float32:
        symbol = "hipblasStbsv"
    elif A.dtype == torch.float64:
        symbol = "hipblasDtbsv"
    elif A.dtype == torch.complex64:
        symbol = "hipblasCtbsv_v2"
    elif A.dtype == torch.complex128:
        symbol = "hipblasZtbsv_v2"
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS TBSV: {A.dtype}")

    column_A = row_to_column_band(A, n, k, lda, uplo)
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
            k,
            ctypes.c_void_p(column_A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
        ),
        symbol,
    )
    return x


def cublas_stbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    cublas_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)


def cpu_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    ref_x = to_cpu_blas_tensor(x)
    if n == 0:
        return ref_x

    if flag_blas.vendor_name == "ascend":
        ref_A = row_to_column_band(to_cpu_blas_tensor(A), n, k, lda, uplo)
    else:
        ref_A = to_cpu_blas_tensor(row_to_column_band(A, n, k, lda, uplo))
    func = cpu_blas.ztbsv if ref_A.dtype.is_complex else cpu_blas.dtbsv

    xout = func(
        k,
        ref_A.numpy().T,
        ref_x.numpy(),
        incx=incx,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        trans=trans,
        diag=diag,
        overwrite_x=1,
    )
    return torch.from_numpy(xout)


def tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if TO_CPU:
        return cpu_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    ref_x = x.clone()
    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        hipblas_tbsv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    else:
        cublas_tbsv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    return ref_x


STBSV_SIZES = [1, 33]
STBSV_KS = [0, 1, 16, 64]
STBSV_STRIDE_SIZES = [17, 127]
COMPLEX_TBSV_SIZES = [0, 1, 33]
TBSV_PERF_SIZES = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
TBSV_PERF_KS = [1, 4, 16, 64, 256]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
TRANS_MODES = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS_MODES = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
DIAG_MODES = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]


def tbsv_randn(*shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


def make_triangular_banded(n, k, lda, uplo, dtype, device, unit_diag=False):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    build_device = (
        "cpu"
        if flag_blas.vendor_name == "ascend" and dtype == torch.complex64
        else device
    )
    A = tbsv_randn(n, lda, dtype=dtype, device=build_device) * 0.1
    diag_floor = 2.0 * (k + 1) + 1.0
    bands = torch.arange(lda, device=build_device).view(1, lda)
    rows = torch.arange(n, device=build_device).view(n, 1)

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = bands <= torch.clamp(n - 1 - rows, max=k)
        diag_col = 0
    else:
        valid = (bands >= torch.clamp(k - rows, min=0)) & (bands <= k)
        diag_col = k

    A = A.masked_fill(~valid, 0.0)
    if unit_diag:
        A[:, diag_col] = 1.0
    else:
        real_dtype = (
            torch.float64
            if dtype in (torch.float64, torch.complex128)
            else torch.float32
        )
        sign = torch.where(
            torch.rand(n, device=build_device) < 0.5,
            torch.full((n,), -1.0, dtype=real_dtype, device=build_device),
            torch.full((n,), 1.0, dtype=real_dtype, device=build_device),
        )
        A[:, diag_col] = (
            sign * (diag_floor + torch.rand(n, dtype=real_dtype, device=build_device))
        ).to(dtype)
    return A.to(device).contiguous()


def _effective_k(n, k):
    return min(k, max(0, n - 1))


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("fp64 is not supported on this device")


def _make_x(length, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        return tbsv_randn(length, dtype=dtype, device=device)
    if dtype.is_complex:
        return torch.randn(length, dtype=dtype, device=device) + 1j * torch.randn(
            length, dtype=dtype, device=device
        )
    return torch.randn(length, dtype=dtype, device=device)


TBSV_PERF_VARIANTS = [
    pytest.param(
        op,
        dtype,
        uplo,
        trans,
        marks=getattr(pytest.mark, op_name),
        id=f"{op_name}-{uplo}-{trans}",
    )
    for op_name, op, dtype, trans_modes in (
        ("stbsv", flag_blas.stbsv, torch.float32, TRANS_MODES),
        ("dtbsv", flag_blas.dtbsv, torch.float64, TRANS_MODES),
        ("ctbsv", flag_blas.ctbsv, torch.complex64, COMPLEX_TRANS_MODES),
        ("ztbsv", flag_blas.ztbsv, torch.complex128, COMPLEX_TRANS_MODES),
    )
    for uplo in FILL_MODES
    for trans in trans_modes
]


@pytest.mark.parametrize("op,dtype,uplo,trans", TBSV_PERF_VARIANTS)
@pytest.mark.parametrize("n", TBSV_PERF_SIZES)
@pytest.mark.parametrize("k_requested", TBSV_PERF_KS)
def test_tbsv_perf_shape_coverage(op, dtype, uplo, trans, n, k_requested):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()
    k = _effective_k(n, k_requested)
    lda = k + 1
    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(
        uplo,
        trans,
        CUBLAS_DIAG_NON_UNIT,
        n,
        k,
        A,
        lda,
        x,
        1,
    )

    op(uplo, trans, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.stbsv
@pytest.mark.parametrize("n", STBSV_SIZES)
@pytest.mark.parametrize("k", STBSV_KS)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_stbsv(n, k, uplo, trans, diag):
    k = _effective_k(n, k)
    dtype = torch.float32
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.stbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_stbsv_stride(n, k, uplo, trans, incx):
    k = _effective_k(n, k)
    dtype = torch.float32
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.stbsv
def test_stbsv_n_zero():
    A = torch.empty((0, 1), dtype=torch.float32, device=flag_blas.device)
    x = torch.empty((0,), dtype=torch.float32, device=flag_blas.device)
    flag_blas.stbsv(
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        0,
        0,
        A,
        1,
        x,
        1,
    )
    assert x.numel() == 0


@pytest.mark.stbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
def test_stbsv_k_zero(uplo, trans):
    n, k = 256, 0
    lda = 1
    dtype = torch.float32
    diag = CUBLAS_DIAG_NON_UNIT

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.dtbsv
@pytest.mark.parametrize("n", STBSV_SIZES)
@pytest.mark.parametrize("k", STBSV_KS)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_dtbsv(n, k, uplo, trans, diag):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.dtbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.dtbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_dtbsv_stride(n, k, uplo, trans, incx):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(1 + (n - 1) * incx, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.dtbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ctbsv
@pytest.mark.parametrize("n", COMPLEX_TBSV_SIZES)
@pytest.mark.parametrize("k", [0, 1, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_ctbsv(n, k, uplo, trans, diag):
    k = _effective_k(n, k)
    dtype = torch.complex64
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.ctbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ctbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_ctbsv_stride(n, k, uplo, trans, incx):
    k = _effective_k(n, k)
    dtype = torch.complex64
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(1 + (n - 1) * incx, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.ctbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ztbsv
@pytest.mark.parametrize("n", COMPLEX_TBSV_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_ztbsv(n, k, uplo, trans, diag):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.ztbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ztbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_ztbsv_stride(n, k, uplo, trans, incx):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(1 + (n - 1) * incx, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.ztbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.stbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
def test_stbsv_unit_diag_ignored(uplo, trans):
    n, k = 128, 8
    lda = k + 1 + 1
    dtype = torch.float32

    A_clean = make_triangular_banded(
        n, k, lda, uplo, dtype, flag_blas.device, unit_diag=True
    )
    A_dirty = A_clean.clone()
    diag_row = 0 if uplo == CUBLAS_FILL_MODE_UPPER else k
    A_dirty[:, diag_row] = float("nan")

    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    x_clean = x.clone()
    x_dirty = x.clone()

    flag_blas.stbsv(uplo, trans, CUBLAS_DIAG_UNIT, n, k, A_clean, lda, x_clean, 1)
    flag_blas.stbsv(uplo, trans, CUBLAS_DIAG_UNIT, n, k, A_dirty, lda, x_dirty, 1)

    ref_x_clean = x_clean.cpu() if TO_CPU else x_clean
    blas_assert_close(x_dirty, ref_x_clean, dtype, reduce_dim=k + 1)


@pytest.mark.stbsv
def test_stbsv_solve_then_multiply_roundtrip():
    n, k = 512, 16
    lda = k + 1
    dtype = torch.float32
    uplo = CUBLAS_FILL_MODE_LOWER

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x_orig = torch.randn(n, dtype=dtype, device=flag_blas.device)

    x_buf = x_orig.clone()
    flag_blas.stbsv(uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x_buf, 1)

    A_dense = torch.zeros(n, n, dtype=dtype, device=flag_blas.device)
    rows = torch.arange(n, device=flag_blas.device).view(n, 1)
    cols = torch.arange(n, device=flag_blas.device).view(1, n)
    mask = (rows >= cols) & ((rows - cols) <= k)
    A_dense[mask] = A[rows.expand(n, n)[mask], k - (rows - cols)[mask]]
    Ay = A_dense @ x_buf

    ref_x_orig = x_orig.cpu() if TO_CPU else x_orig
    blas_assert_close(Ay, ref_x_orig, dtype, reduce_dim=k + 1)


TBSV_VARIANTS = [
    pytest.param(flag_blas.stbsv, torch.float32, CUBLAS_OP_T, id="stbsv"),
    pytest.param(flag_blas.dtbsv, torch.float64, CUBLAS_OP_T, id="dtbsv"),
    pytest.param(flag_blas.ctbsv, torch.complex64, CUBLAS_OP_C, id="ctbsv"),
    pytest.param(flag_blas.ztbsv, torch.complex128, CUBLAS_OP_C, id="ztbsv"),
]

NARROW_TBSV_VARIANTS = [
    pytest.param(
        flag_blas.dtbsv,
        torch.float64,
        CUBLAS_OP_N,
        id="dtbsv-n",
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        flag_blas.dtbsv,
        torch.float64,
        CUBLAS_OP_T,
        id="dtbsv-t",
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        flag_blas.ztbsv,
        torch.complex128,
        CUBLAS_OP_N,
        id="ztbsv-n",
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        flag_blas.ztbsv,
        torch.complex128,
        CUBLAS_OP_T,
        id="ztbsv-t",
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        flag_blas.ztbsv,
        torch.complex128,
        CUBLAS_OP_C,
        id="ztbsv-c",
        marks=pytest.mark.ztbsv,
    ),
]


@pytest.mark.parametrize("op,dtype,trans", NARROW_TBSV_VARIANTS)
@pytest.mark.parametrize("n", [127, 128, 129, 255, 256, 257, 513])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_tbsv_k1_panel_boundaries(op, dtype, trans, n, uplo, diag):
    check_fp64_support()
    k, lda = 1, 4
    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    op(uplo, trans, diag, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=2)


@pytest.mark.dtbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_dtbsv_large_k4_no_trans(uplo):
    check_fp64_support()
    n, k, lda = 8192, 4, 7
    dtype = torch.float64
    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1)

    flag_blas.dtbsv(uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ztbsv
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_ztbsv_k1_complex_diagonal_phase(trans, uplo):
    check_fp64_support()
    n, k, lda = 129, 1, 4
    A = make_triangular_banded(n, k, lda, uplo, torch.complex128, flag_blas.device)
    diag_col = 0 if uplo == CUBLAS_FILL_MODE_UPPER else k
    diagonal_imag = torch.linspace(
        0.25, 0.75, n, dtype=torch.float64, device=flag_blas.device
    )
    A[:, diag_col] = torch.complex(A[:, diag_col].real, diagonal_imag)
    x = _make_x(n, torch.complex128, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1)

    flag_blas.ztbsv(uplo, trans, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, torch.complex128, reduce_dim=2)


@pytest.mark.parametrize("op,dtype,trans", TBSV_VARIANTS)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_tbsv_unit_diag_ignores_stored_diagonal(op, dtype, trans, uplo):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()
    n, k, lda = 33, 8, 11
    clean = make_triangular_banded(
        n, k, lda, uplo, dtype, flag_blas.device, unit_diag=True
    )
    dirty = clean.clone()
    diag_col = 0 if uplo == CUBLAS_FILL_MODE_UPPER else k
    if dtype.is_complex:
        dirty[:, diag_col] = complex(float("nan"), float("nan"))
    else:
        dirty[:, diag_col] = float("nan")
    x = _make_x(n, dtype, flag_blas.device)
    clean_x = x.clone()
    dirty_x = x.clone()

    op(uplo, trans, CUBLAS_DIAG_UNIT, n, k, clean, lda, clean_x, 1)
    op(uplo, trans, CUBLAS_DIAG_UNIT, n, k, dirty, lda, dirty_x, 1)

    ref_clean_x = clean_x.cpu() if TO_CPU else clean_x
    blas_assert_close(dirty_x, ref_clean_x, dtype, reduce_dim=k + 1)

import ctypes
import ctypes.util
import math

import cupy as cp
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
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


_cublas = load_cublas()


def cublas_tpmv_reference(uplo, trans, diag, n, AP, x, incx):
    if n == 0:
        return
    handle = cp.cuda.device.get_cublas_handle()
    dtype = AP.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStpmv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtpmv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtpmv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtpmv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtpmv_v2 execution failed with error code: {status}")


def cpu_tpmv_reference(uplo, trans, diag, n, AP, x, incx):
    ref_x = to_cpu_blas_tensor(x)
    if n == 0:
        return ref_x

    ref_AP = to_cpu_blas_tensor(AP)
    func = cpu_blas.ztpmv if ref_AP.dtype.is_complex else cpu_blas.dtpmv
    xout = func(
        n,
        ref_AP.numpy(),
        ref_x.numpy(),
        incx=incx,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        trans=trans,
        diag=diag,
        overwrite_x=1,
    )
    return torch.from_numpy(xout)


def tpmv_reference(uplo, trans, diag, n, AP, x, incx):
    if TO_CPU:
        return cpu_tpmv_reference(uplo, trans, diag, n, AP, x, incx)

    ref_x = x.clone()
    cublas_tpmv_reference(uplo, trans, diag, n, AP, ref_x, incx)
    return ref_x


TPMV_SIZES = [
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
TPMV_STRIDE_SIZES = [64, 127, 256]
INCS = [1, 2, 3]

UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


def make_packed(n, uplo, dtype, device):
    size = n * (n + 1) // 2
    return torch.randn(size, dtype=dtype, device=device).contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _tpmv_tol(dtype, n):
    K = max(1, n)
    if dtype in (torch.float32, torch.complex64):
        return min(max(1e-4, 5e-6 * math.sqrt(K)), 5e-3)
    if dtype in (torch.float64, torch.complex128):
        return min(max(1e-12, 2e-14 * math.sqrt(K)), 1e-10)
    raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.stpmv
@pytest.mark.parametrize("n", TPMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_stpmv(n, uplo, trans, diag):
    dtype = torch.float32
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, 1)
    flag_blas.ops.stpmv(uplo, trans, diag, n, AP, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))


@pytest.mark.stpmv
@pytest.mark.parametrize("n", TPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_stpmv_stride(n, uplo, trans, diag, incx):
    dtype = torch.float32
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, incx)
    flag_blas.ops.stpmv(uplo, trans, diag, n, AP, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))


@pytest.mark.dtpmv
@pytest.mark.parametrize("n", TPMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_dtpmv(n, uplo, trans, diag):
    check_fp64_support()
    dtype = torch.float64
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, 1)
    flag_blas.ops.dtpmv(uplo, trans, diag, n, AP, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))


@pytest.mark.dtpmv
@pytest.mark.parametrize("n", TPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_dtpmv_stride(n, uplo, trans, diag, incx):
    check_fp64_support()
    dtype = torch.float64
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, incx)
    flag_blas.ops.dtpmv(uplo, trans, diag, n, AP, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))


@pytest.mark.ctpmv
@pytest.mark.parametrize("n", TPMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ctpmv(n, uplo, trans, diag):
    dtype = torch.complex64
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, 1)
    flag_blas.ops.ctpmv(uplo, trans, diag, n, AP, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))


@pytest.mark.ctpmv
@pytest.mark.parametrize("n", TPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_ctpmv_stride(n, uplo, trans, diag, incx):
    dtype = torch.complex64
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, incx)
    flag_blas.ops.ctpmv(uplo, trans, diag, n, AP, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))


@pytest.mark.ztpmv
@pytest.mark.parametrize("n", TPMV_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ztpmv(n, uplo, trans, diag):
    check_fp64_support()
    dtype = torch.complex128
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, 1)
    flag_blas.ops.ztpmv(uplo, trans, diag, n, AP, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))


@pytest.mark.ztpmv
@pytest.mark.parametrize("n", TPMV_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_ztpmv_stride(n, uplo, trans, diag, incx):
    check_fp64_support()
    dtype = torch.complex128
    AP = make_packed(n, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = tpmv_reference(uplo, trans, diag, n, AP, x, incx)
    flag_blas.ops.ztpmv(uplo, trans, diag, n, AP, x, incx)

    blas_assert_close(x, ref_x, dtype, reduce_dim=n, atol=_tpmv_tol(dtype, n))

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
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

if flag_blas.vendor_name == "hygon":
    from .hipblas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
from .conftest import TO_CPU


def load_cublas():
    lib_names = ["libcublas.so.13"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.append(found_path)
    lib_names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])
    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = None
_cublas_handle = None
_CUBLAS_HPR2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasChpr2_v2", "cublasZhpr2_v2"):
        func = getattr(_cublas, name)
        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_HPR2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_HPR2_FUNCS = {
            torch.complex64: (_cublas.cublasChpr2_v2, cuComplex),
            torch.complex128: (_cublas.cublasZhpr2_v2, cuDoubleComplex),
        }
    return _cublas


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is not None:
        return _cublas_handle
    cublas = _ensure_cublas()
    _cublas_handle = ctypes.c_void_p()
    status = cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with error code: {status}")
    status = cublas.cublasSetPointerMode_v2(_cublas_handle, CUBLAS_POINTER_MODE_HOST)
    if status != 0:
        raise RuntimeError(f"cublasSetPointerMode_v2 failed with error code: {status}")
    return _cublas_handle


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _make_scalar(ctor, value):
    value = value.item() if isinstance(value, torch.Tensor) else value
    return ctor(value.real, value.imag)


def cublas_hpr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    if n == 0:
        return
    handle = _get_cublas_handle()
    _ensure_cublas()
    func, ctor = _CUBLAS_HPR2_FUNCS[AP.dtype]
    alpha_c = _make_scalar(ctor, alpha)
    status = func(
        handle,
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.c_void_p(AP.data_ptr()),
    )
    if status != 0:
        raise RuntimeError(f"cublasXhpr2_v2 execution failed with error code: {status}")


def hipblas_hpr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    if n == 0:
        return
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else complex(alpha)
    if AP.dtype == torch.complex64:
        symbol = "hipblasChpr2_v2"
        alpha_value = HipComplex(alpha.real, alpha.imag)
    elif AP.dtype == torch.complex128:
        symbol = "hipblasZhpr2_v2"
        alpha_value = HipDoubleComplex(alpha.real, alpha.imag)
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS HPR2: {AP.dtype}")
    library, handle = get_hipblas_context(AP)
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
    ]
    function.restype = ctypes.c_int
    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    check_hipblas_status(
        function(
            handle,
            hip_uplo,
            n,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.c_void_p(y.data_ptr()),
            incy,
            ctypes.c_void_p(AP.data_ptr()),
        ),
        symbol,
    )


def cpu_hpr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    ref_AP = to_cpu_blas_tensor(AP)
    if n == 0:
        return ref_AP
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    updated = cpu_blas.zhpr2(
        n,
        alpha,
        ref_x.numpy(),
        ref_y.numpy(),
        ref_AP.numpy(),
        incx=incx,
        incy=incy,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        overwrite_ap=1,
    )
    return torch.from_numpy(updated)


def _column_major_transpose_args(uplo, alpha, x, y):
    physical_uplo = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    physical_alpha = complex(alpha).conjugate()
    physical_x = torch.resolve_conj(x.conj()).contiguous()
    physical_y = torch.resolve_conj(y.conj()).contiguous()
    return physical_uplo, physical_alpha, physical_x, physical_y


def hpr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    uplo, alpha, x, y = _column_major_transpose_args(uplo, alpha, x, y)
    if TO_CPU:
        return cpu_hpr2_reference(uplo, n, alpha, x, incx, y, incy, AP)
    ref_AP = AP.clone()
    if flag_blas.vendor_name == "hygon":
        hipblas_hpr2_reference(uplo, n, alpha, x, incx, y, incy, ref_AP)
    else:
        cublas_hpr2_reference(uplo, n, alpha, x, incx, y, incy, ref_AP)
    return ref_AP


HPR2_EDGE_SIZES = [
    1,
    2,
    3,
    4,
    7,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    47,
    48,
    49,
]
HPR2_PERF_SIZES = [
    64,
    96,
    127,
    128,
    129,
    160,
    191,
    192,
    193,
    224,
    255,
    256,
    257,
    320,
    383,
    384,
    385,
    448,
    511,
    512,
    513,
    640,
    767,
    768,
    769,
    896,
    1023,
    1024,
    1025,
    1280,
    1535,
    1536,
    1537,
    1792,
    2047,
    2048,
    2049,
    2304,
    2559,
    2560,
    2561,
    2816,
    3071,
    3072,
    3073,
    3328,
    3583,
    3584,
    3585,
    3840,
    4095,
    4096,
    4607,
    4608,
    4609,
    5119,
    5120,
    5121,
    5632,
    6143,
    6144,
    6145,
    7167,
    7168,
    7169,
    8191,
    8192,
]
HPR2_SIZES = sorted(set(HPR2_EDGE_SIZES + HPR2_PERF_SIZES))
HPR2_STRIDE_SIZES = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]
FILL_MODES = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def hpr2_randn(*shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


def make_hermitian_packed(n, dtype, device, uplo):
    AP = hpr2_randn(n * (n + 1) // 2, dtype=dtype, device=device)
    if n > 0:
        idx = torch.arange(n, dtype=torch.long, device=device)
        if uplo == CUBLAS_FILL_MODE_UPPER:
            diag = idx * n - idx * (idx - 1) // 2
        else:
            diag = idx * (idx + 3) // 2
        torch.view_as_real(AP)[diag, 1] = 0.0
    return AP


def _run_hpr2_case(op, dtype, alpha, uplo, n, incx=1, incy=1):
    if dtype == torch.complex128:
        check_fp64_support()
    AP = make_hermitian_packed(n, dtype, flag_blas.device, uplo)
    x = hpr2_randn(1 + max(0, n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = hpr2_randn(1 + max(0, n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_AP = hpr2_reference(uplo, n, alpha, x, incx, y, incy, AP)
    op(uplo, n, alpha, x, incx, y, incy, AP)
    blas_assert_close(AP, ref_AP, dtype, reduce_dim=2)


def _run_hpr2_row_packed_case(op, dtype, uplo):
    if dtype == torch.complex128:
        check_fp64_support()
    n = 3
    alpha = 0.75 - 0.5j
    build_device = "cpu" if flag_blas.vendor_name == "ascend" else flag_blas.device
    AP = torch.tensor(
        [
            1.0 + 0.0j,
            2.0 - 0.5j,
            -1.0 + 0.25j,
            3.0 + 0.0j,
            0.5 + 1.5j,
            -2.0 + 0.0j,
        ],
        dtype=dtype,
        device=build_device,
    )
    x = torch.tensor(
        [1.0 + 0.5j, -2.0 + 1.0j, 0.25 - 0.75j],
        dtype=dtype,
        device=build_device,
    )
    y = torch.tensor(
        [-0.5 + 1.0j, 1.5 - 0.25j, 2.0 + 0.5j],
        dtype=dtype,
        device=build_device,
    )
    expected = AP.clone()
    update = alpha * x[:, None] * y.conj()[None, :]
    update += alpha.conjugate() * y[:, None] * x.conj()[None, :]
    if uplo == CUBLAS_FILL_MODE_UPPER:
        rows, cols = torch.triu_indices(n, n, device=build_device)
    else:
        rows, cols = torch.tril_indices(n, n, device=build_device)
    expected += update[rows, cols]
    torch.view_as_real(expected)[rows == cols, 1] = 0.0
    if flag_blas.vendor_name == "ascend":
        AP = AP.to(flag_blas.device)
        x = x.to(flag_blas.device)
        y = y.to(flag_blas.device)
    op(uplo, n, alpha, x, 1, y, 1, AP)
    blas_assert_close(AP, to_reference(expected), dtype, reduce_dim=2)


@pytest.mark.zhpr2
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_zhpr2_uses_row_packed_storage(uplo):
    _run_hpr2_row_packed_case(flag_blas.zhpr2, torch.complex128, uplo)


@pytest.mark.chpr2
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_chpr2_uses_row_packed_storage(uplo):
    _run_hpr2_row_packed_case(flag_blas.chpr2, torch.complex64, uplo)


@pytest.mark.chpr2
@pytest.mark.parametrize("n", HPR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_chpr2_sizes(n, uplo):
    _run_hpr2_case(flag_blas.chpr2, torch.complex64, 1.5 + 0.5j, uplo, n)


@pytest.mark.chpr2
@pytest.mark.parametrize("n", HPR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_chpr2_stride(n, uplo, incx, incy):
    _run_hpr2_case(flag_blas.chpr2, torch.complex64, -0.75 + 0.25j, uplo, n, incx, incy)


@pytest.mark.zhpr2
@pytest.mark.parametrize("n", HPR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_zhpr2_sizes(n, uplo):
    _run_hpr2_case(flag_blas.zhpr2, torch.complex128, 1.5 + 0.5j, uplo, n)


@pytest.mark.zhpr2
@pytest.mark.parametrize("n", HPR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zhpr2_stride(n, uplo, incx, incy):
    _run_hpr2_case(
        flag_blas.zhpr2, torch.complex128, -0.75 + 0.25j, uplo, n, incx, incy
    )


@pytest.mark.chpr2
def test_chpr2_alpha_zero():
    _run_hpr2_case(
        flag_blas.chpr2,
        torch.complex64,
        0.0j,
        CUBLAS_FILL_MODE_UPPER,
        128,
    )


@pytest.mark.parametrize(
    "dtype,op,alpha",
    [
        (torch.complex64, flag_blas.chpr2, 1.0 + 0.5j),
        (torch.complex128, flag_blas.zhpr2, 1.0 + 0.5j),
    ],
)
def test_hpr2_n_zero(dtype, op, alpha):
    if dtype == torch.complex128:
        check_fp64_support()
    AP = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, x, 1, y, 1, AP)
    assert AP.numel() == 0

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

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
from .conftest import TO_CPU

if flag_blas.vendor_name in {"hygon", "mthreads"}:
    from .vendor_blas_reference import check_hipblas_status, get_hipblas_context


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
_CUBLAS_SPR2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasSspr2_v2", "cublasDspr2_v2"):
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
    global _cublas, _CUBLAS_SPR2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_SPR2_FUNCS = {
            torch.float32: (_cublas.cublasSspr2_v2, ctypes.c_float),
            torch.float64: (_cublas.cublasDspr2_v2, ctypes.c_double),
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


def cublas_spr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    if n == 0:
        return
    handle = _get_cublas_handle()
    _ensure_cublas()
    func, ctor = _CUBLAS_SPR2_FUNCS[AP.dtype]
    alpha_c = ctor(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
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
        raise RuntimeError(f"cublasXspr2_v2 execution failed with error code: {status}")


def hipblas_spr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    if n == 0:
        return
    if AP.dtype == torch.float32:
        symbol = "hipblasSspr2"
        scalar_type = ctypes.c_float
    elif AP.dtype == torch.float64:
        symbol = "hipblasDspr2"
        scalar_type = ctypes.c_double
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS SPR2: {AP.dtype}")
    library, handle = get_hipblas_context(AP)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(scalar_type),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    alpha_c = scalar_type(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    check_hipblas_status(
        function(
            handle,
            hip_uplo,
            n,
            ctypes.byref(alpha_c),
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.c_void_p(y.data_ptr()),
            incy,
            ctypes.c_void_p(AP.data_ptr()),
        ),
        symbol,
    )


def cpu_spr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    ref_AP = to_cpu_blas_tensor(AP)
    if n == 0:
        return ref_AP
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    updated = cpu_blas.dspr2(
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


def spr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    if TO_CPU:
        reference_uplo = (
            CUBLAS_FILL_MODE_LOWER
            if uplo == CUBLAS_FILL_MODE_UPPER
            else CUBLAS_FILL_MODE_UPPER
        )
        return cpu_spr2_reference(reference_uplo, n, alpha, x, incx, y, incy, AP)
    ref_AP = AP.clone()
    reference_uplo = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        hipblas_spr2_reference(reference_uplo, n, alpha, x, incx, y, incy, ref_AP)
    else:
        cublas_spr2_reference(reference_uplo, n, alpha, x, incx, y, incy, ref_AP)
    return ref_AP


SPR2_EDGE_SIZES = [
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
SPR2_PERF_SIZES = [
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
SPR2_SIZES = sorted(set(SPR2_EDGE_SIZES + SPR2_PERF_SIZES))
SPR2_STRIDE_SIZES = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]
FILL_MODES = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def make_packed(n, dtype, device):
    return torch.randn(n * (n + 1) // 2, dtype=dtype, device=device)


def _run_spr2_case(op, dtype, alpha, uplo, n, incx=1, incy=1):
    if dtype is torch.float64:
        check_fp64_support()
    AP = make_packed(n, dtype, flag_blas.device)
    x = torch.randn(1 + max(0, n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + max(0, n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_AP = spr2_reference(uplo, n, alpha, x, incx, y, incy, AP)
    op(uplo, n, alpha, x, incx, y, incy, AP)
    blas_assert_close(AP, ref_AP, dtype, reduce_dim=2)


def _run_spr2_row_packed_case(op, dtype, uplo):
    if dtype is torch.float64:
        check_fp64_support()
    n = 3
    alpha = 0.75
    AP = torch.tensor(
        [1.0, 2.0, -1.0, 3.0, 0.5, -2.0],
        dtype=dtype,
        device=flag_blas.device,
    )
    x = torch.tensor([1.0, -2.0, 0.25], dtype=dtype, device=flag_blas.device)
    y = torch.tensor([-0.5, 1.5, 2.0], dtype=dtype, device=flag_blas.device)
    expected = AP.clone()
    update = alpha * (x[:, None] * y[None, :] + y[:, None] * x[None, :])
    if uplo == CUBLAS_FILL_MODE_UPPER:
        rows, cols = torch.triu_indices(n, n, device=flag_blas.device)
    else:
        rows, cols = torch.tril_indices(n, n, device=flag_blas.device)
    expected += update[rows, cols]
    op(uplo, n, alpha, x, 1, y, 1, AP)
    blas_assert_close(AP, to_reference(expected), dtype, reduce_dim=2)


@pytest.mark.sspr2
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_sspr2_uses_row_packed_storage(uplo):
    _run_spr2_row_packed_case(flag_blas.sspr2, torch.float32, uplo)


@pytest.mark.dspr2
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_dspr2_uses_row_packed_storage(uplo):
    _run_spr2_row_packed_case(flag_blas.dspr2, torch.float64, uplo)


@pytest.mark.sspr2
@pytest.mark.parametrize("n", SPR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_sspr2_sizes(n, uplo):
    _run_spr2_case(flag_blas.sspr2, torch.float32, 1.5, uplo, n)


@pytest.mark.sspr2
@pytest.mark.parametrize("n", SPR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_sspr2_stride(n, uplo, incx, incy):
    _run_spr2_case(flag_blas.sspr2, torch.float32, -0.75, uplo, n, incx, incy)


@pytest.mark.dspr2
@pytest.mark.parametrize("n", SPR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_dspr2_sizes(n, uplo):
    _run_spr2_case(flag_blas.dspr2, torch.float64, 1.5, uplo, n)


@pytest.mark.dspr2
@pytest.mark.parametrize("n", SPR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dspr2_stride(n, uplo, incx, incy):
    _run_spr2_case(flag_blas.dspr2, torch.float64, -0.75, uplo, n, incx, incy)


@pytest.mark.sspr2
def test_sspr2_alpha_zero():
    _run_spr2_case(flag_blas.sspr2, torch.float32, 0.0, CUBLAS_FILL_MODE_UPPER, 128)


@pytest.mark.parametrize(
    "dtype,op,alpha",
    [
        (torch.float32, flag_blas.sspr2, 1.0),
        (torch.float64, flag_blas.dspr2, 1.0),
    ],
)
def test_spr2_n_zero(dtype, op, alpha):
    if dtype is torch.float64:
        check_fp64_support()
    AP = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, x, 1, y, 1, AP)
    assert AP.numel() == 0

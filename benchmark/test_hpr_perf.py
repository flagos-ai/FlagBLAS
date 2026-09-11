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

import atexit
import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

HPR_SIZES = [
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


def load_cublas():
    if IS_MTHREADS:
        from benchmark.mublas_compat import load_mublas

        return load_mublas()
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
_CUBLAS_HPR_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0

_HIPBLAS_LIBRARY = None
_HIPBLAS_HANDLES = {}
_HIPBLAS_HPR_FUNCS = {
    torch.complex64: ("hipblasChpr_v2", ctypes.c_float),
    torch.complex128: ("hipblasZhpr_v2", ctypes.c_double),
}


def _check_hipblas_status(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed with hipBLAS status {status}")


def _load_hipblas():
    global _HIPBLAS_LIBRARY
    if _HIPBLAS_LIBRARY is None:
        library_name = ctypes.util.find_library("hipblas")
        if library_name is None:
            raise RuntimeError("Unable to find the hipBLAS shared library")
        library = ctypes.CDLL(library_name)
        library.hipblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.hipblasCreate.restype = ctypes.c_int
        library.hipblasDestroy.argtypes = [ctypes.c_void_p]
        library.hipblasDestroy.restype = ctypes.c_int
        library.hipblasSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.hipblasSetStream.restype = ctypes.c_int
        library.hipblasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.hipblasSetPointerMode.restype = ctypes.c_int
        _HIPBLAS_LIBRARY = library
    return _HIPBLAS_LIBRARY


def _prepare_hipblas(device):
    library = _load_hipblas()
    torch_device = torch.device(device)
    device_index = torch_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    handle = _HIPBLAS_HANDLES.get(device_index)
    if handle is None:
        with torch.cuda.device(device_index):
            handle = ctypes.c_void_p()
            _check_hipblas_status(
                library.hipblasCreate(ctypes.byref(handle)), "hipblasCreate"
            )
            _check_hipblas_status(
                library.hipblasSetPointerMode(handle, 0),
                "hipblasSetPointerMode",
            )
        _HIPBLAS_HANDLES[device_index] = handle
    stream = torch.cuda.current_stream(device).cuda_stream
    _check_hipblas_status(
        library.hipblasSetStream(handle, ctypes.c_void_p(stream)),
        "hipblasSetStream",
    )
    return library, handle


def _destroy_hipblas_handles():
    if _HIPBLAS_LIBRARY is None:
        return
    for handle in tuple(_HIPBLAS_HANDLES.values()):
        try:
            _HIPBLAS_LIBRARY.hipblasDestroy(handle)
        except Exception:
            pass
    _HIPBLAS_HANDLES.clear()


def _resolve_hipblas_hpr(library, dtype):
    try:
        symbol, scalar_type = _HIPBLAS_HPR_FUNCS[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported Hygon HPR benchmark dtype: {dtype}") from error
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(scalar_type),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function, scalar_type


atexit.register(_destroy_hipblas_handles)


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasChpr_v2", "cublasZhpr_v2"):
        func = getattr(_cublas, name)
        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_HPR_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        if not IS_MTHREADS:
            _configure_cublas_signatures()
        _CUBLAS_HPR_FUNCS = {
            torch.complex64: (_cublas.cublasChpr_v2, ctypes.c_float),
            torch.complex128: (_cublas.cublasZhpr_v2, ctypes.c_double),
        }
    return _cublas


def _get_cublas_handle():
    if IS_MTHREADS:
        _ensure_cublas()
        from benchmark.mublas_compat import get_mublas_handle
        return get_mublas_handle()
    global _cublas_handle
    if _cublas_handle is not None:
        return _cublas_handle
    _ensure_cublas()
    _cublas_handle = ctypes.c_void_p()
    status = _cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
    status = _cublas.cublasSetPointerMode_v2(_cublas_handle, CUBLAS_POINTER_MODE_HOST)
    if status != 0:
        raise RuntimeError(f"cublasSetPointerMode_v2 failed with status code: {status}")
    return _cublas_handle


def hipblas_hpr_baseline(
    AP,
    x,
    reference_x,
    n,
    incx,
    handle,
    c_func,
    alpha_ptr,
    hip_uplo,
    vendor_args,
    **kwargs,
):
    if n == 0:
        return AP
    status = c_func(*vendor_args)
    _check_hipblas_status(status, "hipBLAS HPR")
    return AP


def cublas_hpr_baseline(
    AP,
    x,
    reference_x,
    reference_uplo,
    uplo,
    n,
    alpha,
    incx,
    handle,
    c_func,
    alpha_c,
    vendor_args,
    **kwargs,
):
    if n == 0:
        return AP
    status = c_func(*vendor_args)
    if status != 0:
        raise RuntimeError(f"cublasXhpr_v2 failed with status code: {status}")
    return AP


def gems_chpr_wrapper(AP, x, uplo, n, alpha, incx, handle, **kwargs):
    flag_blas.chpr(uplo, n, alpha, x, incx, AP)
    return AP


def gems_zhpr_wrapper(AP, x, uplo, n, alpha, incx, handle, **kwargs):
    flag_blas.zhpr(uplo, n, alpha, x, incx, AP)
    return AP


class HprBenchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in HPR_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HPR_SIZES]
        return None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        max_n = max(HPR_SIZES)
        if any(len(shape) != 1 or shape[0] > max_n for shape in self.shapes):
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype) -> Generator:
        reference_uplo = (
            CUBLAS_FILL_MODE_LOWER
            if self.uplo == CUBLAS_FILL_MODE_UPPER
            else CUBLAS_FILL_MODE_UPPER
        )
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, ctor = _resolve_hipblas_hpr(library, cur_dtype)
            hip_uplo = 121 if reference_uplo == CUBLAS_FILL_MODE_UPPER else 122
        else:
            handle = _get_cublas_handle()
            c_func, ctor = _CUBLAS_HPR_FUNCS[cur_dtype]
        alpha_c = ctor(self.alpha)
        alpha_ptr = ctypes.byref(alpha_c)
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP = torch.randn(n * (n + 1) // 2, dtype=cur_dtype, device=self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            reference_x = torch.empty_like(x)
            reference_x.copy_(x.conj())
            vendor_args = (
                (
                    handle,
                    hip_uplo,
                    n,
                    alpha_ptr,
                    ctypes.c_void_p(reference_x.data_ptr()),
                    1,
                    ctypes.c_void_p(AP.data_ptr()),
                )
                if IS_HYGON
                else (
                    handle,
                    ctypes.c_int(reference_uplo),
                    ctypes.c_int(n),
                    alpha_ptr,
                    ctypes.c_void_p(reference_x.data_ptr()),
                    ctypes.c_int(1),
                    ctypes.c_void_p(AP.data_ptr()),
                )
            )
            call_kwargs = {
                "uplo": self.uplo,
                "reference_uplo": reference_uplo,
                "reference_x": reference_x,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
                "vendor_args": vendor_args,
            }
            if IS_HYGON:
                call_kwargs.update(alpha_ptr=alpha_ptr, hip_uplo=hip_uplo)
            else:
                call_kwargs.update(alpha_c=alpha_c)
            yield AP, x, call_kwargs

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return n * n

    def get_gbps(self, args, latency):
        AP, x = args[0], args[1]
        io_amount = 2 * shape_utils.size_in_bytes(AP) + shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return 1

    def clone_correctness_inputs(self, args, kwargs):
        AP, x = args
        ref_AP = AP.clone()
        ref_kwargs = kwargs.copy()
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[6] = ctypes.c_void_p(ref_AP.data_ptr())
        ref_kwargs["vendor_args"] = tuple(vendor_args)
        return (ref_AP, x), ref_kwargs, (AP.clone(), x), kwargs


@pytest.mark.chpr
def test_perf_chpr():
    bench = HprBenchmark(
        op_name="chpr",
        torch_op=hipblas_hpr_baseline if IS_HYGON else cublas_hpr_baseline,
        gems_op=gems_chpr_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.chpr
def test_perf_chpr_upper():
    bench = HprBenchmark(
        op_name="chpr",
        torch_op=hipblas_hpr_baseline if IS_HYGON else cublas_hpr_baseline,
        gems_op=gems_chpr_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpr
def test_perf_zhpr():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support complex128")
    bench = HprBenchmark(
        op_name="zhpr",
        torch_op=hipblas_hpr_baseline if IS_HYGON else cublas_hpr_baseline,
        gems_op=gems_zhpr_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpr
def test_perf_zhpr_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support complex128")
    bench = HprBenchmark(
        op_name="zhpr",
        torch_op=hipblas_hpr_baseline if IS_HYGON else cublas_hpr_baseline,
        gems_op=gems_zhpr_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)

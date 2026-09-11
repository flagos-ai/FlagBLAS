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
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_HYGON:
    import atexit
elif IS_MTHREADS:
    from benchmark.mublas_compat import cp, cublas
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

HPMV_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
]


def load_cublas():
    if IS_MTHREADS:
        from benchmark.mublas_compat import load_mublas

        return load_mublas()
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


_cublas = None if IS_HYGON else load_cublas()


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


_CUBLAS_HPMV_FUNCS = (
    {}
    if IS_HYGON
    else {
        torch.complex64: (_cublas.cublasChpmv_v2, cuComplex),
        torch.complex128: (_cublas.cublasZhpmv_v2, cuDoubleComplex),
    }
)


if IS_HYGON:
    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    _HIPBLAS_HPMV_FUNCS = {
        torch.complex64: ("hipblasChpmv_v2", cuComplex),
        torch.complex128: ("hipblasZhpmv_v2", cuDoubleComplex),
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

    def _resolve_hipblas_hpmv(library, dtype):
        try:
            symbol, scalar_type = _HIPBLAS_HPMV_FUNCS[dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Hygon HPMV benchmark dtype: {dtype}"
            ) from error
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function, scalar_type

    atexit.register(_destroy_hipblas_handles)


def _make_scalar(ctor, value):
    return ctor(value.real, value.imag)


if IS_HYGON:

    def cublas_hpmv_baseline(
        AP,
        x,
        y,
        uplo,
        n,
        alpha,
        incx,
        beta,
        incy,
        c_func,
        vendor_args,
        **kwargs,
    ):
        status = c_func(*vendor_args)
        _check_hipblas_status(status, "hipBLAS HPMV")
        return y

else:

    def cublas_hpmv_baseline(
        AP,
        x,
        y,
        uplo,
        n,
        alpha,
        incx,
        beta,
        incy,
        c_func,
        vendor_args,
        **kwargs,
    ):
        if n == 0:
            return y
        status = c_func(*vendor_args)
        if status != 0:
            raise RuntimeError(f"cublasXhpmv_v2 failed with status code: {status}")
        return y


def _gems_wrapper(op):
    def _impl(AP, x, y, uplo, n, alpha, incx, beta, incy, handle=None, **kwargs):
        op(uplo, n, alpha, AP, x, incx, beta, y, incy)
        return y

    return _impl


gems_chpmv_wrapper = _gems_wrapper(flag_blas.chpmv)
gems_zhpmv_wrapper = _gems_wrapper(flag_blas.zhpmv)


def _generate_packed_her(n, dtype, device):
    AP = torch.randn(n * (n + 1) // 2, dtype=dtype, device=device)
    column_AP = torch.empty_like(AP)
    column_AP.copy_(AP.conj())
    return AP, column_AP


class HpmvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.beta = beta
        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HPMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, ctor = _resolve_hipblas_hpmv(library, cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            if cur_dtype not in _CUBLAS_HPMV_FUNCS:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")
            c_func, ctor = _CUBLAS_HPMV_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, self.alpha)
        beta_c = _make_scalar(ctor, self.beta)
        alpha_ptr = ctypes.byref(alpha_c)
        beta_ptr = ctypes.byref(beta_c)

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP, column_AP = _generate_packed_her(n, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)
            reference_uplo = (
                CUBLAS_FILL_MODE_LOWER
                if self.uplo == CUBLAS_FILL_MODE_UPPER
                else CUBLAS_FILL_MODE_UPPER
            )
            vendor_uplo = 121 if reference_uplo == CUBLAS_FILL_MODE_UPPER else 122
            if IS_HYGON:
                vendor_args = (
                    handle,
                    ctypes.c_int(vendor_uplo),
                    ctypes.c_int(n),
                    alpha_ptr,
                    ctypes.c_void_p(column_AP.data_ptr()),
                    ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_int(1),
                    beta_ptr,
                    ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_int(1),
                )
            else:
                vendor_args = (
                    ctypes.c_void_p(handle),
                    ctypes.c_int(reference_uplo),
                    ctypes.c_int(n),
                    alpha_ptr,
                    ctypes.c_void_p(column_AP.data_ptr()),
                    ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_int(1),
                    beta_ptr,
                    ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_int(1),
                )

            call_kwargs = {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "c_func": c_func,
                "vendor_args": vendor_args,
                "column_AP": column_AP,
                "alpha_c": alpha_c,
                "beta_c": beta_c,
            }
            yield AP, x, y, call_kwargs

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 8 * n * n

    def get_gbps(self, args, latency):
        AP, x, y = args[0], args[1], args[2]
        a_bytes = AP.numel() * AP.element_size()
        io_amount = (
            a_bytes + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs.get("n", 0))

    def clone_correctness_inputs(self, args, kwargs):
        AP, x, y = args
        ref_y = y.clone()
        ref_args = (AP, x, ref_y)
        blas_args = (AP, x, y.clone())
        ref_kwargs = kwargs.copy()
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[8] = ctypes.c_void_p(ref_y.data_ptr())
        ref_kwargs["vendor_args"] = tuple(vendor_args)
        return ref_args, ref_kwargs, blas_args, kwargs


@pytest.mark.chpmv
def test_perf_chpmv():
    bench = HpmvBenchmark(
        op_name="chpmv",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_chpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.chpmv
def test_perf_chpmv_upper():
    bench = HpmvBenchmark(
        op_name="chpmv",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_chpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpmv
def test_perf_zhpmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HpmvBenchmark(
        op_name="zhpmv",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_zhpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpmv
def test_perf_zhpmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HpmvBenchmark(
        op_name="zhpmv",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_zhpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)

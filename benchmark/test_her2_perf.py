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

HER2_SIZES = [
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
_CUBLAS_HER2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0

_HIPBLAS_LIBRARY = None
_HIPBLAS_HANDLES = {}
_HIPBLAS_HER2_SYMBOLS = {
    torch.complex64: "hipblasCher2_v2",
    torch.complex128: "hipblasZher2_v2",
}


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasCher2_v2", "cublasZher2_v2"):
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
            ctypes.c_int,
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_HER2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        if not IS_MTHREADS:
            _configure_cublas_signatures()
        _CUBLAS_HER2_FUNCS = {
            torch.complex64: (_cublas.cublasCher2_v2, cuComplex),
            torch.complex128: (_cublas.cublasZher2_v2, cuDoubleComplex),
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
                library.hipblasSetPointerMode(handle, 0), "hipblasSetPointerMode"
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


def _resolve_hipblas_her2(library, dtype):
    try:
        symbol = _HIPBLAS_HER2_SYMBOLS[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported Hygon HER2 benchmark dtype: {dtype}") from error
    ctor = cuComplex if dtype == torch.complex64 else cuDoubleComplex
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
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    return function, ctor


atexit.register(_destroy_hipblas_handles)


def _make_scalar(ctor, value):
    return ctor(value.real, value.imag)


def vendor_her2_baseline(
    A,
    x,
    y,
    uplo,
    n,
    alpha,
    incx,
    incy,
    lda,
    handle,
    c_func,
    alpha_ptr,
    hip_uplo=None,
    vendor_args=None,
    reference_result=None,
    **kwargs,
):
    if n == 0:
        return A
    if IS_HYGON:
        status = c_func(*vendor_args)
        _check_hipblas_status(status, "hipBLAS HER2")
        return reference_result
    status = c_func(*vendor_args)
    if status != 0:
        raise RuntimeError(f"cublasXher2_v2 failed with status code: {status}")
    return reference_result


def gems_cher2_wrapper(A, x, y, uplo, n, alpha, incx, incy, lda, handle=None, **kwargs):
    flag_blas.cher2(uplo, n, alpha, x, incx, y, incy, A, lda)
    return A


def gems_zher2_wrapper(A, x, y, uplo, n, alpha, incx, incy, lda, handle=None, **kwargs):
    flag_blas.zher2(uplo, n, alpha, x, incx, y, incy, A, lda)
    return A


def _generate_A(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    A[:, :n] = torch.randn(n, n, dtype=dtype, device=device)
    diag_real = A[:, :n].diagonal().real.clone()
    A[:, :n].diagonal().copy_(diag_real.to(dtype))
    return A.contiguous()


def _row_to_column_full(A, n, lda):
    column_A = torch.zeros((n, lda), dtype=A.dtype, device=A.device)
    column_A[:, :n] = A[:n, :n].T
    return column_A


class Her2Benchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in HER2_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=1.5 + 0.5j, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HER2_SIZES]
        return None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        max_n = max(HER2_SIZES)
        if any(len(shape) != 1 or shape[0] > max_n for shape in self.shapes):
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, ctor = _resolve_hipblas_her2(library, cur_dtype)
            hip_uplo = 121 if self.uplo == CUBLAS_FILL_MODE_UPPER else 122
        else:
            _ensure_cublas()
            handle = _get_cublas_handle()
            c_func, ctor = _CUBLAS_HER2_FUNCS[cur_dtype]
            hip_uplo = None
        alpha_c = _make_scalar(ctor, self.alpha)
        alpha_ptr = ctypes.byref(alpha_c)
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = n
            A = _generate_A(n, lda, cur_dtype, self.device)
            column_A = _row_to_column_full(A, n, lda)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)
            vendor_args = (
                handle,
                hip_uplo if IS_HYGON else ctypes.c_int(self.uplo),
                n if IS_HYGON else ctypes.c_int(n),
                alpha_ptr,
                ctypes.c_void_p(x.data_ptr()),
                1 if IS_HYGON else ctypes.c_int(1),
                ctypes.c_void_p(y.data_ptr()),
                1 if IS_HYGON else ctypes.c_int(1),
                ctypes.c_void_p(column_A.data_ptr()),
                lda if IS_HYGON else ctypes.c_int(lda),
            )
            yield A, x, y, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "incy": 1,
                "lda": lda,
                "handle": handle,
                "c_func": c_func,
                "alpha_ptr": alpha_ptr,
                "hip_uplo": hip_uplo,
                "vendor_args": vendor_args,
                "column_A": column_A,
                "reference_result": column_A[:, :n].T,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 4 * n * n

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        n = x.numel()
        a_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = (
            2 * a_bytes + shape_utils.size_in_bytes(x) + shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return 2

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        ref_A = A.clone()
        ref_x = x.clone()
        ref_y = y.clone()
        ref_column_A = _row_to_column_full(ref_A, kwargs["n"], kwargs["lda"])
        ref_kwargs = kwargs.copy()
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[4] = ctypes.c_void_p(ref_x.data_ptr())
        vendor_args[6] = ctypes.c_void_p(ref_y.data_ptr())
        vendor_args[8] = ctypes.c_void_p(ref_column_A.data_ptr())
        ref_kwargs.update(
            vendor_args=tuple(vendor_args),
            column_A=ref_column_A,
            reference_result=ref_column_A[:, : kwargs["n"]].T,
        )
        return (
            (ref_A, ref_x, ref_y),
            ref_kwargs,
            (A.clone(), x.clone(), y.clone()),
            kwargs,
        )


@pytest.mark.cher2
def test_perf_cher2():
    bench = Her2Benchmark(
        op_name="cher2",
        torch_op=vendor_her2_baseline,
        gems_op=gems_cher2_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cher2
def test_perf_cher2_upper():
    bench = Her2Benchmark(
        op_name="cher2",
        torch_op=vendor_her2_baseline,
        gems_op=gems_cher2_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zher2
def test_perf_zher2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support complex128")
    bench = Her2Benchmark(
        op_name="zher2",
        torch_op=vendor_her2_baseline,
        gems_op=gems_zher2_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zher2
def test_perf_zher2_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support complex128")
    bench = Her2Benchmark(
        op_name="zher2",
        torch_op=vendor_her2_baseline,
        gems_op=gems_zher2_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)

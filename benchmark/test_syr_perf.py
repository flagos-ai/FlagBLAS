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

SYR_SIZES = [
    64,
    96,
    127,
    128,
    129,
    192,
    255,
    256,
    257,
    384,
    511,
    512,
    513,
    768,
    1023,
    1024,
    1025,
    1536,
    2048,
    3072,
    4096,
]
IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"
CUBLAS_POINTER_MODE_HOST = 0
HIPBLAS_POINTER_MODE_HOST = 0
_HIPBLAS_LIBRARY = None
_HIPBLAS_HANDLES = {}
_HIPBLAS_SYR_FUNCS = {
    torch.float32: ("hipblasSsyr", ctypes.c_float),
    torch.float64: ("hipblasDsyr", ctypes.c_double),
    torch.complex64: ("hipblasCsyr_v2", None),
    torch.complex128: ("hipblasZsyr_v2", None),
}


class _ComplexFloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


def _load_cublas():
    if IS_MTHREADS:
        from benchmark.mublas_compat import load_mublas

        return load_mublas()
    names = ["libcublas.so.13"]
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so")


_cublas = None
_cublas_handle = None
_CUBLAS_SYR_FUNCS = None


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    _cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _cublas.cublasSetStream_v2.restype = ctypes.c_int
    for name in (
        "cublasSsyr_v2",
        "cublasDsyr_v2",
        "cublasCsyr_v2",
        "cublasZsyr_v2",
    ):
        function = getattr(_cublas, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_SYR_FUNCS
    if _cublas is None:
        _cublas = _load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_SYR_FUNCS = {
            torch.float32: (_cublas.cublasSsyr_v2, ctypes.c_float),
            torch.float64: (_cublas.cublasDsyr_v2, ctypes.c_double),
            torch.complex64: (_cublas.cublasCsyr_v2, _ComplexFloat),
            torch.complex128: (_cublas.cublasZsyr_v2, _ComplexDouble),
        }
    return _cublas


def _get_cublas_handle():
    if IS_MTHREADS:
        _ensure_cublas()
        from benchmark.mublas_compat import get_mublas_handle
        return get_mublas_handle()
    global _cublas_handle
    cublas = _ensure_cublas()
    if _cublas_handle is None:
        _cublas_handle = ctypes.c_void_p()
        status = cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        status = cublas.cublasSetPointerMode_v2(
            _cublas_handle, CUBLAS_POINTER_MODE_HOST
        )
        if status != 0:
            raise RuntimeError(
                f"cublasSetPointerMode_v2 failed with status code: {status}"
            )
    status = cublas.cublasSetStream_v2(
        _cublas_handle,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"cublasSetStream_v2 failed with status code: {status}")
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
                library.hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST),
                "hipblasSetPointerMode",
            )
        _HIPBLAS_HANDLES[device_index] = handle
    stream = torch.cuda.current_stream(device).cuda_stream
    _check_hipblas_status(
        library.hipblasSetStream(handle, ctypes.c_void_p(stream)),
        "hipblasSetStream",
    )
    return library, handle


def _resolve_hipblas_syr(library, dtype):
    symbol, scalar_type = _HIPBLAS_SYR_FUNCS[dtype]
    if dtype == torch.complex64:
        scalar_type = _ComplexFloat
    elif dtype == torch.complex128:
        scalar_type = _ComplexDouble
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
    ]
    function.restype = ctypes.c_int
    return function, scalar_type


def _destroy_hipblas_handles():
    if _HIPBLAS_LIBRARY is None:
        return
    for handle in tuple(_HIPBLAS_HANDLES.values()):
        try:
            _HIPBLAS_LIBRARY.hipblasDestroy(handle)
        except Exception:
            pass
    _HIPBLAS_HANDLES.clear()


atexit.register(_destroy_hipblas_handles)


def _make_scalar(ctor, value):
    if ctor in (_ComplexFloat, _ComplexDouble):
        value = complex(value)
        return ctor(value.real, value.imag)
    return ctor(value)


def vendor_syr_baseline(
    A,
    x,
    n,
    handle,
    c_func,
    uplo_arg,
    n_arg,
    alpha_ptr,
    x_ptr,
    incx_arg,
    A_ptr,
    lda_arg,
    vendor_name,
    reference_result,
    **kwargs,
):
    if n == 0:
        return A
    status = c_func(
        handle,
        uplo_arg,
        n_arg,
        alpha_ptr,
        x_ptr,
        incx_arg,
        A_ptr,
        lda_arg,
    )
    if status != 0:
        raise RuntimeError(f"{vendor_name} Xsyr failed with status code: {status}")
    return reference_result


def _gems_wrapper(op):
    def _impl(A, x, uplo, n, alpha, incx, lda, **kwargs):
        op(uplo, n, alpha, x, incx, A, lda)
        return A

    return _impl


GEMS_SYR_WRAPPERS = {
    "ssyr": _gems_wrapper(flag_blas.ssyr),
    "dsyr": _gems_wrapper(flag_blas.dsyr),
    "csyr": _gems_wrapper(flag_blas.csyr),
    "zsyr": _gems_wrapper(flag_blas.zsyr),
}


def _row_to_column_full(A, n, lda):
    column_A = torch.zeros((n, lda), dtype=A.dtype, device=A.device)
    column_A[:, :n] = A[:n, :n].T
    return column_A


class SyrBenchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in SYR_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SYR_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, ctor = _resolve_hipblas_syr(library, cur_dtype)
            uplo_value = 121 if self.uplo == CUBLAS_FILL_MODE_UPPER else 122
            vendor_name = "hipBLAS"
        else:
            handle = _get_cublas_handle()
            c_func, ctor = _CUBLAS_SYR_FUNCS[cur_dtype]
            uplo_value = self.uplo
            vendor_name = "muBLAS" if IS_MTHREADS else "cuBLAS"
        alpha_c = _make_scalar(ctor, self.alpha)
        alpha_ptr = ctypes.byref(alpha_c)
        for shape in self.shapes:
            n = shape[0]
            lda = n
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            A = torch.randn((n, lda), dtype=cur_dtype, device=self.device)
            column_A = _row_to_column_full(A, n, lda)
            yield A, x, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "lda": lda,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
                "alpha_ptr": alpha_ptr,
                "x_ptr": ctypes.c_void_p(x.data_ptr()),
                "A_ptr": ctypes.c_void_p(column_A.data_ptr()),
                "column_A": column_A,
                "reference_result": column_A[:, :n].T,
                "uplo_arg": ctypes.c_int(uplo_value),
                "n_arg": ctypes.c_int(n),
                "incx_arg": ctypes.c_int(1),
                "lda_arg": ctypes.c_int(lda),
                "vendor_name": vendor_name,
            }

    def get_correctness_reduce_dim(self, args, kwargs):
        return 1

    def clone_correctness_inputs(self, args, kwargs):
        A, x = args
        ref_A = A.clone()
        ref_x = x.clone()
        ref_column_A = _row_to_column_full(ref_A, kwargs["n"], kwargs["lda"])
        gems_A = A.clone()
        gems_x = x.clone()
        ref_kwargs = kwargs.copy()
        ref_kwargs.update(
            x_ptr=ctypes.c_void_p(ref_x.data_ptr()),
            A_ptr=ctypes.c_void_p(ref_column_A.data_ptr()),
            column_A=ref_column_A,
            reference_result=ref_column_A[:, : kwargs["n"]].T,
        )
        return (ref_A, ref_x), ref_kwargs, (gems_A, gems_x), kwargs

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return n * (n + 1)

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        n = x.numel()
        triangle_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = 2 * triangle_bytes + shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


def _run_syr(op_name, dtype, uplo, alpha):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    bench = SyrBenchmark(
        op_name=op_name,
        torch_op=vendor_syr_baseline,
        gems_op=GEMS_SYR_WRAPPERS[op_name],
        dtypes=[dtype],
        uplo=uplo,
        alpha=alpha,
    )
    run_correctness_then_benchmark(bench)


SYR_PERF_CASES = [
    pytest.param(
        "ssyr", torch.float32, 0.75, CUBLAS_FILL_MODE_LOWER, marks=pytest.mark.ssyr
    ),
    pytest.param(
        "ssyr", torch.float32, 0.75, CUBLAS_FILL_MODE_UPPER, marks=pytest.mark.ssyr
    ),
    pytest.param(
        "dsyr",
        torch.float64,
        0.12345678901234568,
        CUBLAS_FILL_MODE_LOWER,
        marks=pytest.mark.dsyr,
    ),
    pytest.param(
        "dsyr",
        torch.float64,
        0.12345678901234568,
        CUBLAS_FILL_MODE_UPPER,
        marks=pytest.mark.dsyr,
    ),
    pytest.param(
        "csyr",
        torch.complex64,
        complex(0.5, -0.25),
        CUBLAS_FILL_MODE_LOWER,
        marks=pytest.mark.csyr,
    ),
    pytest.param(
        "csyr",
        torch.complex64,
        complex(0.5, -0.25),
        CUBLAS_FILL_MODE_UPPER,
        marks=pytest.mark.csyr,
    ),
    pytest.param(
        "zsyr",
        torch.complex128,
        complex(0.12345678901234568, -0.2345678912345679),
        CUBLAS_FILL_MODE_LOWER,
        marks=pytest.mark.zsyr,
    ),
    pytest.param(
        "zsyr",
        torch.complex128,
        complex(0.12345678901234568, -0.2345678912345679),
        CUBLAS_FILL_MODE_UPPER,
        marks=pytest.mark.zsyr,
    ),
]


@pytest.mark.parametrize("op_name,dtype,alpha,uplo", SYR_PERF_CASES)
def test_perf_syr(op_name, dtype, alpha, uplo):
    _run_syr(op_name, dtype, uplo, alpha)

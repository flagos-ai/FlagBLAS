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

SYR2_SIZES = [
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
_CUBLAS_SYR2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


_HIPBLAS_LIBRARY = None
_HIPBLAS_HANDLES = {}
_HIPBLAS_SYR2_FUNCS = {
    torch.float32: ("hipblasSsyr2", ctypes.c_float),
    torch.float64: ("hipblasDsyr2", ctypes.c_double),
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


def _resolve_hipblas_syr2(library, dtype):
    try:
        symbol, scalar_type = _HIPBLAS_SYR2_FUNCS[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported Hygon SYR2 benchmark dtype: {dtype}") from error
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
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    return function, scalar_type


atexit.register(_destroy_hipblas_handles)


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasSsyr2_v2", "cublasDsyr2_v2"):
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
    global _cublas, _CUBLAS_SYR2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        if not IS_MTHREADS:
            _configure_cublas_signatures()
        _CUBLAS_SYR2_FUNCS = {
            torch.float32: (_cublas.cublasSsyr2_v2, ctypes.c_float, False),
            torch.float64: (_cublas.cublasDsyr2_v2, ctypes.c_double, False),
        }
    return _cublas


def _get_cublas_handle():
    """Return the active muBLAS handle for the MUSA reference path."""
    if IS_MTHREADS:
        from benchmark.mublas_compat import get_mublas_handle

        return get_mublas_handle()
    raise RuntimeError("_get_cublas_handle is only used for the MUSA path")


def _make_scalar(ctor, is_complex, value):
    if is_complex:
        return ctor(value.real, value.imag)
    return ctor(value)


def hipblas_syr2_baseline(
    A,
    x,
    y,
    uplo,
    n,
    incx,
    incy,
    lda,
    handle,
    c_func,
    alpha_ptr,
    hip_uplo,
    vendor_args,
    reference_result,
    **kwargs,
):
    if n == 0:
        return A
    status = c_func(*vendor_args)
    _check_hipblas_status(status, "hipBLAS SYR2")
    return reference_result


def cublas_syr2_baseline(
    A,
    x,
    y,
    uplo,
    n,
    incx,
    incy,
    lda,
    handle,
    c_func,
    alpha_c,
    vendor_args,
    reference_result,
    **kwargs,
):
    if n == 0:
        return A

    status = c_func(*vendor_args)
    if status != 0:
        raise RuntimeError(f"cublasXsyr2_v2 failed with status code: {status}")
    return reference_result


def gems_ssyr2_wrapper(A, x, y, uplo, n, alpha, incx, incy, lda, handle=None, **kwargs):
    flag_blas.ssyr2(uplo, n, alpha, x, incx, y, incy, A, lda)
    return A


def gems_dsyr2_wrapper(A, x, y, uplo, n, alpha, incx, incy, lda, handle=None, **kwargs):
    flag_blas.dsyr2(uplo, n, alpha, x, incx, y, incy, A, lda)
    return A


def _generate_syr2_A(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    if dtype.is_complex:
        A[:, :n] = torch.randn(n, n, dtype=dtype, device=device)
    else:
        A[:, :n] = torch.randn(n, n, dtype=dtype, device=device) * 0.1
    return A.contiguous()


def _row_to_column_full(A, n, lda):
    column_A = torch.zeros((n, lda), dtype=A.dtype, device=A.device)
    column_A[:, :n] = A[:n, :n].T
    return column_A


class Syr2Benchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in SYR2_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SYR2_SIZES]
        return None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        max_n = max(SYR2_SIZES)
        if any(
            len(shape) not in (1, 2)
            or shape[0] > max_n
            or (len(shape) == 2 and shape[1] > max_n)
            for shape in self.shapes
        ):
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, ctor = _resolve_hipblas_syr2(library, cur_dtype)
            hip_uplo = 121 if self.uplo == CUBLAS_FILL_MODE_UPPER else 122
            is_complex = False
        elif IS_MTHREADS:
            _ensure_cublas()
            handle = _get_cublas_handle()
            if cur_dtype not in _CUBLAS_SYR2_FUNCS:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")
            c_func, ctor, is_complex = _CUBLAS_SYR2_FUNCS[cur_dtype]
        else:
            cublas = _ensure_cublas()
            handle = ctypes.c_void_p()
            status = cublas.cublasCreate_v2(ctypes.byref(handle))
            if status != 0:
                raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
            status = cublas.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST)
            if status != 0:
                raise RuntimeError(
                    f"cublasSetPointerMode_v2 failed with status code: {status}"
                )
            if cur_dtype not in _CUBLAS_SYR2_FUNCS:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")
            c_func, ctor, is_complex = _CUBLAS_SYR2_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, is_complex, self.alpha)
        alpha_ptr = ctypes.byref(alpha_c)

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = shape[1] if isinstance(shape, (tuple, list)) and len(shape) > 1 else n
            A = _generate_syr2_A(n, lda, cur_dtype, self.device)
            column_A = _row_to_column_full(A, n, lda)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)
            vendor_args = (
                handle,
                (121 if self.uplo == CUBLAS_FILL_MODE_UPPER else 122)
                if IS_HYGON
                else ctypes.c_int(self.uplo),
                n if IS_HYGON else ctypes.c_int(n),
                alpha_ptr,
                ctypes.c_void_p(x.data_ptr()),
                1 if IS_HYGON else ctypes.c_int(1),
                ctypes.c_void_p(y.data_ptr()),
                1 if IS_HYGON else ctypes.c_int(1),
                ctypes.c_void_p(column_A.data_ptr()),
                lda if IS_HYGON else ctypes.c_int(lda),
            )

            call_kwargs = {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "incy": 1,
                "lda": lda,
                "handle": handle,
                "c_func": c_func,
                "vendor_args": vendor_args,
                "column_A": column_A,
                "reference_result": column_A[:, :n].T,
                "alpha_c": alpha_c,
            }
            if IS_HYGON:
                call_kwargs.update(alpha_ptr=alpha_ptr, hip_uplo=hip_uplo)
            yield A, x, y, call_kwargs

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 2 * n * n

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        n = x.numel()
        a_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = (
            a_bytes * 2 + shape_utils.size_in_bytes(x) + shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return 2

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        ref_A, ref_x, ref_y = A.clone(), x.clone(), y.clone()
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


@pytest.mark.ssyr2
def test_perf_ssyr2():
    bench = Syr2Benchmark(
        op_name="ssyr2",
        torch_op=hipblas_syr2_baseline if IS_HYGON else cublas_syr2_baseline,
        gems_op=gems_ssyr2_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ssyr2
def test_perf_ssyr2_upper():
    bench = Syr2Benchmark(
        op_name="ssyr2",
        torch_op=hipblas_syr2_baseline if IS_HYGON else cublas_syr2_baseline,
        gems_op=gems_ssyr2_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dsyr2
def test_perf_dsyr2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Syr2Benchmark(
        op_name="dsyr2",
        torch_op=hipblas_syr2_baseline if IS_HYGON else cublas_syr2_baseline,
        gems_op=gems_dsyr2_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dsyr2
def test_perf_dsyr2_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Syr2Benchmark(
        op_name="dsyr2",
        torch_op=hipblas_syr2_baseline if IS_HYGON else cublas_syr2_baseline,
        gems_op=gems_dsyr2_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


# csyr2/zsyr2 benchmarks are intentionally not registered: strict CPU
# correctness has no direct SciPy/CPU BLAS reference for those variants.

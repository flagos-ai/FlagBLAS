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

SBMV_SIZES = [
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

SBMV_KS = [0, 1, 4, 16, 64, 128, 256]


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


if IS_HYGON:
    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    _HIPBLAS_SBMV_FUNCS = {
        torch.float32: ("hipblasSsbmv", ctypes.c_float),
        torch.float64: ("hipblasDsbmv", ctypes.c_double),
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

    def _resolve_hipblas_sbmv(library, dtype):
        try:
            symbol, scalar_type = _HIPBLAS_SBMV_FUNCS[dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Hygon SBMV benchmark dtype: {dtype}"
            ) from error
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(scalar_type),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(scalar_type),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function, scalar_type

    atexit.register(_destroy_hipblas_handles)


_cublas = None if IS_HYGON else load_cublas()

_CUBLAS_SBMV_FUNCS = (
    {}
    if IS_HYGON
    else {
        torch.float32: (_cublas.cublasSsbmv_v2, ctypes.c_float),
        torch.float64: (_cublas.cublasDsbmv_v2, ctypes.c_double),
    }
)


if IS_HYGON:

    def cublas_sbmv_baseline(
        A,
        x,
        y,
        uplo,
        n,
        k,
        alpha,
        lda,
        incx,
        beta,
        incy,
        c_func,
        vendor_args,
        **kwargs,
    ):
        status = c_func(*vendor_args)
        _check_hipblas_status(status, "hipBLAS SBMV")
        return y

else:

    def cublas_sbmv_baseline(
        A,
        x,
        y,
        uplo,
        n,
        k,
        alpha,
        lda,
        incx,
        beta,
        incy,
        c_func,
        reference_uplo,
        vendor_args,
        **kwargs,
    ):
        if n == 0:
            return y
        status = c_func(*vendor_args)
        if status != 0:
            raise RuntimeError(f"cublasXsbmv_v2 failed with status code: {status}")
        return y


def _gems_wrapper(op):
    def _impl(A, x, y, uplo, n, k, alpha, lda, incx, beta, incy, handle=None, **kwargs):
        op(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
        return y

    return _impl


gems_ssbmv_wrapper = _gems_wrapper(flag_blas.ssbmv)
gems_dsbmv_wrapper = _gems_wrapper(flag_blas.dsbmv)


def _generate_symmetric_banded(n, k, lda, uplo, dtype, device):
    return torch.randn((n, lda), dtype=dtype, device=device)


def _band_nnz(n, k):
    if n <= 0:
        return 0
    k = min(k, n - 1)
    return n * (2 * k + 1) - k * (k + 1)


def _stored_band_nnz(n, k):
    if n <= 0:
        return 0
    if k >= n - 1:
        return n * (n + 1) // 2
    return (k + 1) * (k + 2) // 2 + (n - k - 1) * (k + 1)


class SbmvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.beta = beta
        self.ks = SBMV_KS

        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SBMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        reference_uplo = (
            CUBLAS_FILL_MODE_LOWER
            if self.uplo == CUBLAS_FILL_MODE_UPPER
            else CUBLAS_FILL_MODE_UPPER
        )
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, ctor = _resolve_hipblas_sbmv(library, cur_dtype)
            hip_uplo = 121 if reference_uplo == CUBLAS_FILL_MODE_UPPER else 122
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            if cur_dtype not in _CUBLAS_SBMV_FUNCS:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")
            c_func, ctor = _CUBLAS_SBMV_FUNCS[cur_dtype]
        alpha_c = ctor(self.alpha)
        beta_c = ctor(self.beta)
        alpha_ptr = ctypes.byref(alpha_c)
        beta_ptr = ctypes.byref(beta_c)

        seen = set()
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            for k_req in self.ks:
                k = min(k_req, max(0, n - 1))
                key = (n, k)
                if key in seen:
                    continue
                seen.add(key)
                lda = k + 1
                A = _generate_symmetric_banded(
                    n, k, lda, self.uplo, cur_dtype, self.device
                )
                x = torch.randn(n, dtype=cur_dtype, device=self.device)
                y = torch.randn(n, dtype=cur_dtype, device=self.device)
                vendor_args = (
                    (
                        handle,
                        hip_uplo,
                        n,
                        k,
                        alpha_ptr,
                        ctypes.c_void_p(A.data_ptr()),
                        lda,
                        ctypes.c_void_p(x.data_ptr()),
                        1,
                        beta_ptr,
                        ctypes.c_void_p(y.data_ptr()),
                        1,
                    )
                    if IS_HYGON
                    else (
                        ctypes.c_void_p(handle),
                        ctypes.c_int(reference_uplo),
                        ctypes.c_int(n),
                        ctypes.c_int(k),
                        alpha_ptr,
                        ctypes.c_void_p(A.data_ptr()),
                        ctypes.c_int(lda),
                        ctypes.c_void_p(x.data_ptr()),
                        ctypes.c_int(1),
                        beta_ptr,
                        ctypes.c_void_p(y.data_ptr()),
                        ctypes.c_int(1),
                    )
                )

                call_kwargs = {
                    "uplo": self.uplo,
                    "reference_uplo": reference_uplo,
                    "n": n,
                    "k": k,
                    "alpha": self.alpha,
                    "lda": lda,
                    "incx": 1,
                    "beta": self.beta,
                    "incy": 1,
                    "c_func": c_func,
                    "vendor_args": vendor_args,
                }
                if IS_HYGON:
                    call_kwargs.update(
                        alpha_ptr=alpha_ptr,
                        beta_ptr=beta_ptr,
                        hip_uplo=hip_uplo,
                    )
                else:
                    call_kwargs.update(alpha_c=alpha_c, beta_c=beta_c)
                yield A, x, y, call_kwargs

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
        nnz = _band_nnz(n, k)
        return 2 * nnz

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        n = y.numel()
        k = A.shape[-1] - 1
        stored = _stored_band_nnz(n, k)
        a_bytes = stored * A.element_size()
        io_amount = (
            a_bytes + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
        return max(1, min(2 * k + 1, n))

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        ref_y = y.clone()
        ref_kwargs = kwargs.copy()
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[10] = ctypes.c_void_p(ref_y.data_ptr())
        ref_kwargs["vendor_args"] = tuple(vendor_args)
        ref_args = (A, x, ref_y)
        blas_args = (A, x, y.clone())
        return ref_args, ref_kwargs, blas_args, kwargs


@pytest.mark.ssbmv
def test_perf_ssbmv():
    bench = SbmvBenchmark(
        op_name="ssbmv",
        torch_op=cublas_sbmv_baseline,
        gems_op=gems_ssbmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ssbmv
def test_perf_ssbmv_upper():
    bench = SbmvBenchmark(
        op_name="ssbmv",
        torch_op=cublas_sbmv_baseline,
        gems_op=gems_ssbmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dsbmv
def test_perf_dsbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SbmvBenchmark(
        op_name="dsbmv",
        torch_op=cublas_sbmv_baseline,
        gems_op=gems_dsbmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dsbmv
def test_perf_dsbmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SbmvBenchmark(
        op_name="dsbmv",
        torch_op=cublas_sbmv_baseline,
        gems_op=gems_dsbmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)

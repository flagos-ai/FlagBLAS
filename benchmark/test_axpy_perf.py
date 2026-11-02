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

from typing import Generator

import numpy as np
import pytest
import torch

import flag_blas
from benchmark.attri_util import L1_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    import atexit
    import ctypes
    import ctypes.util

    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    _HIPBLAS_AXPY_SYMBOLS = {
        torch.float32: "hipblasSaxpy",
        torch.float64: "hipblasDaxpy",
        torch.complex64: "hipblasCaxpy_v2",
        torch.complex128: "hipblasZaxpy_v2",
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
        device_index = torch.device(device).index
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

    def _resolve_hipblas_axpy(library, dtype):
        function = getattr(library, _HIPBLAS_AXPY_SYMBOLS[dtype])
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function

    def _destroy_hipblas_handles():
        if _HIPBLAS_LIBRARY is not None:
            for handle in tuple(_HIPBLAS_HANDLES.values()):
                try:
                    _HIPBLAS_LIBRARY.hipblasDestroy(handle)
                except Exception:
                    pass
        _HIPBLAS_HANDLES.clear()

    atexit.register(_destroy_hipblas_handles)
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp, cublas
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


if IS_HYGON:

    def hipblas_axpy_baseline(
        x,
        y,
        alpha,
        incx=1,
        incy=1,
        n=None,
        handle=None,
        alpha_ptr=None,
        c_func=None,
        vendor_args=None,
    ):
        _check_hipblas_status(c_func(*vendor_args), "hipBLAS AXPY")
        return y

    cublas_saxpy = hipblas_axpy_baseline
    cublas_daxpy = hipblas_axpy_baseline
    cublas_caxpy = hipblas_axpy_baseline
    cublas_zaxpy = hipblas_axpy_baseline

else:

    def cublas_saxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
        cublas.saxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
        return y

    def cublas_daxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
        cublas.daxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
        return y

    def cublas_caxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
        cublas.caxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
        return y

    def cublas_zaxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
        cublas.zaxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
        return y


if IS_HYGON:

    def _hygon_axpy_wrapper(op):
        def wrapper(
            x,
            y,
            alpha,
            incx=1,
            incy=1,
            n=None,
            handle=None,
            alpha_ptr=None,
            c_func=None,
            vendor_args=None,
        ):
            op(n, alpha, x, incx, y, incy)
            return y

        return wrapper

    blas_saxpy_wrapper = _hygon_axpy_wrapper(flag_blas.saxpy)
    blas_daxpy_wrapper = _hygon_axpy_wrapper(flag_blas.daxpy)
    blas_caxpy_wrapper = _hygon_axpy_wrapper(flag_blas.caxpy)
    blas_zaxpy_wrapper = _hygon_axpy_wrapper(flag_blas.zaxpy)

else:

    def blas_saxpy_wrapper(
        x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
    ):
        flag_blas.saxpy(n, alpha, x, incx, y, incy)
        return y

    def blas_daxpy_wrapper(
        x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
    ):
        flag_blas.daxpy(n, alpha, x, incx, y, incy)
        return y

    def blas_caxpy_wrapper(
        x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
    ):
        flag_blas.caxpy(n, alpha, x, incx, y, incy)
        return y

    def blas_zaxpy_wrapper(
        x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
    ):
        flag_blas.zaxpy(n, alpha, x, incx, y, incy)
        return y


class AxpyBenchmark(Benchmark):
    def __init__(self, *args, alpha=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func = _resolve_hipblas_axpy(library, cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            alpha_np = np.array(self.alpha, dtype=np.float32)
        elif cur_dtype == torch.float64:
            alpha_np = np.array(self.alpha, dtype=np.float64)
        elif cur_dtype == torch.complex64:
            alpha_np = np.array(self.alpha, dtype=np.complex64)
        elif cur_dtype == torch.complex128:
            alpha_np = np.array(self.alpha, dtype=np.complex128)

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp1.numel()
            if n > 0:
                kwargs = {
                    "alpha": self.alpha,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }
                if IS_HYGON:
                    kwargs.update(
                        c_func=c_func,
                        vendor_args=(
                            handle,
                            n,
                            ctypes.c_void_p(alpha_ptr),
                            ctypes.c_void_p(inp1.data_ptr()),
                            1,
                            ctypes.c_void_p(inp2.data_ptr()),
                            1,
                        ),
                    )
                yield inp1, inp2, kwargs

    def get_gbps(self, args, latency):
        inp1, inp2 = args[0], args[1]
        io_amount = shape_utils.size_in_bytes(inp1) + 2 * shape_utils.size_in_bytes(
            inp2
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # A complex multiply and add cost six plus two real FLOPs.
        return (8 if x.dtype.is_complex else 2) * n


class AxpyStrideBenchmark(AxpyBenchmark):
    def __init__(self, *args, incx=1, incy=1, alpha=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func = _resolve_hipblas_axpy(library, cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            alpha_np = np.array(self.alpha, dtype=np.float32)
        elif cur_dtype == torch.float64:
            alpha_np = np.array(self.alpha, dtype=np.float64)
        elif cur_dtype == torch.complex64:
            alpha_np = np.array(self.alpha, dtype=np.complex64)
        elif cur_dtype == torch.complex128:
            alpha_np = np.array(self.alpha, dtype=np.complex128)

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            n = shape[0]
            inp1 = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            if n > 0:
                kwargs = {
                    "alpha": self.alpha,
                    "incx": self.incx,
                    "incy": self.incy,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }
                if IS_HYGON:
                    kwargs.update(
                        c_func=c_func,
                        vendor_args=(
                            handle,
                            n,
                            ctypes.c_void_p(alpha_ptr),
                            ctypes.c_void_p(inp1.data_ptr()),
                            self.incx,
                            ctypes.c_void_p(inp2.data_ptr()),
                            self.incy,
                        ),
                    )
                yield inp1, inp2, kwargs

    def get_gbps(self, args, latency):
        inp1 = args[0]
        n = (inp1.numel() + self.incx - 1) // self.incx
        element_size = inp1.element_size()
        io_amount = n * element_size + 2 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # A complex multiply and add cost six plus two real FLOPs.
        return (8 if x.dtype.is_complex else 2) * n


if IS_HYGON:

    def _clone_hygon_axpy_inputs(self, args, kwargs):
        (
            ref_args,
            ref_kwargs,
            blas_args,
            blas_kwargs,
        ) = Benchmark.clone_correctness_inputs(self, args, kwargs)
        ref_kwargs["vendor_args"] = (
            ref_kwargs["handle"],
            ref_kwargs["n"],
            ctypes.c_void_p(ref_kwargs["alpha_ptr"]),
            ctypes.c_void_p(ref_args[0].data_ptr()),
            ref_kwargs.get("incx", 1),
            ctypes.c_void_p(ref_args[1].data_ptr()),
            ref_kwargs.get("incy", 1),
        )
        return ref_args, ref_kwargs, blas_args, blas_kwargs

    AxpyBenchmark.clone_correctness_inputs = _clone_hygon_axpy_inputs
    AxpyBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.saxpy
def test_perf_saxpy():
    bench = AxpyBenchmark(
        op_name="saxpy",
        torch_op=cublas_saxpy,
        blas_op=blas_saxpy_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.daxpy
def test_perf_daxpy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyBenchmark(
        op_name="daxpy",
        torch_op=cublas_daxpy,
        blas_op=blas_daxpy_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.caxpy
def test_perf_caxpy():
    bench = AxpyBenchmark(
        op_name="caxpy",
        torch_op=cublas_caxpy,
        blas_op=blas_caxpy_wrapper,
        dtypes=[torch.complex64],
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zaxpy
def test_perf_zaxpy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyBenchmark(
        op_name="zaxpy",
        torch_op=cublas_zaxpy,
        blas_op=blas_zaxpy_wrapper,
        dtypes=[torch.complex128],
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.saxpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_saxpy_stride(incx, incy):
    bench = AxpyStrideBenchmark(
        op_name=f"saxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_saxpy,
        blas_op=blas_saxpy_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.daxpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_daxpy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyStrideBenchmark(
        op_name=f"daxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_daxpy,
        blas_op=blas_daxpy_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.caxpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_caxpy_stride(incx, incy):
    bench = AxpyStrideBenchmark(
        op_name=f"caxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_caxpy,
        blas_op=blas_caxpy_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zaxpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_zaxpy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyStrideBenchmark(
        op_name=f"zaxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_zaxpy,
        blas_op=blas_zaxpy_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)

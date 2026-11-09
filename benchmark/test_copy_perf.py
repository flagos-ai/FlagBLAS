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
from benchmark.attri_util import L1_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    from tests.hipblas_reference import (
        check_hipblas_status,
        get_hipblas_context,
        get_hipblas_library,
    )
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp
elif flag_blas.vendor_name != "ascend":
    import cupy as cp


def load_cublas():
    if flag_blas.vendor_name == "mthreads":
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
    raise RuntimeError("Cannot find libcublas.so in the system")


_cublas = None if IS_HYGON or flag_blas.vendor_name == "ascend" else load_cublas()


def _resolve_cublas_copy(dtype):
    symbol = {
        torch.float32: "cublasScopy_v2",
        torch.float64: "cublasDcopy_v2",
        torch.complex64: "cublasCcopy_v2",
        torch.complex128: "cublasZcopy_v2",
    }[dtype]
    function = getattr(_cublas, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    return function


def blas_scopy_wrapper(x, y, incx=1, incy=1, n=None, handle=None, c_func=None):
    flag_blas.scopy(n, x, incx, y, incy)
    return x, y


def blas_dcopy_wrapper(x, y, incx=1, incy=1, n=None, handle=None, c_func=None):
    flag_blas.dcopy(n, x, incx, y, incy)
    return x, y


def blas_ccopy_wrapper(x, y, incx=1, incy=1, n=None, handle=None, c_func=None):
    flag_blas.ccopy(n, x, incx, y, incy)
    return x, y


def blas_zcopy_wrapper(x, y, incx=1, incy=1, n=None, handle=None, c_func=None):
    flag_blas.zcopy(n, x, incx, y, incy)
    return x, y


if IS_HYGON:

    def _resolve_hipblas_copy(dtype):
        symbol = {
            torch.float32: "hipblasScopy",
            torch.float64: "hipblasDcopy",
            torch.complex64: "hipblasCcopy_v2",
            torch.complex128: "hipblasZcopy_v2",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_copy_baseline(x, y, n=None, incx=1, incy=1, handle=None, c_func=None):
        if n is None:
            n = min(x.numel() // incx, y.numel() // incy)
        if n <= 0:
            return x, y
        check_hipblas_status(
            c_func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy), "hipBLAS COPY"
        )
        return x, y

    def _hygon_copy_wrapper(op):
        def wrapper(x, y, n=None, incx=1, incy=1, handle=None, c_func=None):
            return op(x, y, n=n, incx=incx, incy=incy)

        return wrapper

    cublas_copy_reference = hipblas_copy_baseline
    blas_scopy_wrapper = _hygon_copy_wrapper(blas_scopy_wrapper)
    blas_dcopy_wrapper = _hygon_copy_wrapper(blas_dcopy_wrapper)
    blas_ccopy_wrapper = _hygon_copy_wrapper(blas_ccopy_wrapper)
    blas_zcopy_wrapper = _hygon_copy_wrapper(blas_zcopy_wrapper)

else:

    def cublas_copy_reference(x, y, incx=1, incy=1, n=None, handle=None, c_func=None):
        if n <= 0:
            return x, y
        status = c_func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy)
        if status != 0:
            raise RuntimeError(f"cuBLAS copy execution failed, error code: {status}")
        return x, y


class CopyBenchmark(Benchmark):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_copy(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            c_func = _resolve_cublas_copy(cur_dtype)
        for shape in self.shapes:
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp2 = torch.zeros(shape, dtype=cur_dtype, device=self.device)
            n = inp1.numel()
            if n > 0:
                if IS_HYGON:
                    _, handle = get_hipblas_context(inp1)
                yield inp1, inp2, {"n": n, "handle": handle, "c_func": c_func}

    def get_gbps(self, args, latency):
        inp1, inp2 = args[0], args[1]
        io_amount = shape_utils.size_in_bytes(inp1) + shape_utils.size_in_bytes(inp2)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        # Data movement has no floating-point arithmetic.
        return 0


class CopyStrideBenchmark(Benchmark):
    def __init__(self, *args, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return L1_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_copy(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            c_func = _resolve_cublas_copy(cur_dtype)
        for shape in self.shapes:
            n = shape[0]
            inp1 = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            inp2 = torch.zeros(n * self.incy, dtype=cur_dtype, device=self.device)
            if n > 0:
                kwargs = {"incx": self.incx, "incy": self.incy, "n": n}
                if IS_HYGON:
                    _, handle = get_hipblas_context(inp1)
                kwargs.update(handle=handle, c_func=c_func)
                yield inp1, inp2, kwargs

    def get_gbps(self, args, latency):
        inp1 = args[0]
        n = (inp1.numel() + self.incx - 1) // self.incx
        element_size = inp1.element_size()
        io_amount = 2 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        # Data movement has no floating-point arithmetic.
        return 0


if IS_HYGON:
    CopyBenchmark.correctness_reference = "hipBLAS"
    CopyStrideBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.scopy
def test_perf_scopy():
    bench = CopyBenchmark(
        op_name="scopy",
        torch_op=cublas_copy_reference,
        blas_op=blas_scopy_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dcopy
def test_perf_dcopy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyBenchmark(
        op_name="dcopy",
        torch_op=cublas_copy_reference,
        blas_op=blas_dcopy_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ccopy
def test_perf_ccopy():
    bench = CopyBenchmark(
        op_name="ccopy",
        torch_op=cublas_copy_reference,
        blas_op=blas_ccopy_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zcopy
def test_perf_zcopy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyBenchmark(
        op_name="zcopy",
        torch_op=cublas_copy_reference,
        blas_op=blas_zcopy_wrapper,
        dtypes=[torch.complex128],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.scopy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_scopy_stride(incx, incy):
    bench = CopyStrideBenchmark(
        op_name=f"scopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        blas_op=blas_scopy_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dcopy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_dcopy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyStrideBenchmark(
        op_name=f"dcopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        blas_op=blas_dcopy_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ccopy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_ccopy_stride(incx, incy):
    bench = CopyStrideBenchmark(
        op_name=f"ccopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        blas_op=blas_ccopy_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zcopy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_zcopy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyStrideBenchmark(
        op_name=f"zcopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        blas_op=blas_zcopy_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)

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

import pytest
import torch

import flag_blas
from benchmark.attri_util import L1_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    import ctypes

    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp, cublas
elif flag_blas.vendor_name != "ascend":
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


def blas_sasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.sasum(n, x, incx, result)
    return result


def blas_dasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.dasum(n, x, incx, result)
    return result


def blas_scasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.scasum(n, x, incx, result)
    return result


def blas_dzasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.dzasum(n, x, incx, result)
    return result


if IS_HYGON:

    def _resolve_hipblas_asum(dtype):
        symbol = {
            torch.float32: "hipblasSasum",
            torch.float64: "hipblasDasum",
            torch.complex64: "hipblasScasum",
            torch.complex128: "hipblasDzasum",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_asum_baseline(x, incx=1, n=None, handle=None, result=None, c_func=None):
        check_hipblas_status(
            c_func(handle, n, x.data_ptr(), incx, result.data_ptr()), "hipBLAS ASUM"
        )
        return result

    def _hygon_asum_wrapper(op):
        def wrapper(x, incx=1, n=None, handle=None, result=None, c_func=None):
            return op(x, incx=incx, n=n, handle=handle, result=result)

        return wrapper

    cublas_sasum = hipblas_asum_baseline
    blas_sasum_wrapper = _hygon_asum_wrapper(blas_sasum_wrapper)
    cublas_dasum = hipblas_asum_baseline
    blas_dasum_wrapper = _hygon_asum_wrapper(blas_dasum_wrapper)
    cublas_scasum = hipblas_asum_baseline
    blas_scasum_wrapper = _hygon_asum_wrapper(blas_scasum_wrapper)
    cublas_dzasum = hipblas_asum_baseline
    blas_dzasum_wrapper = _hygon_asum_wrapper(blas_dzasum_wrapper)

else:

    def cublas_sasum(x, incx=1, n=None, handle=None, result=None):
        cublas.sasum(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_dasum(x, incx=1, n=None, handle=None, result=None):
        cublas.dasum(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_scasum(x, incx=1, n=None, handle=None, result=None):
        cublas.scasum(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_dzasum(x, incx=1, n=None, handle=None, result=None):
        cublas.dzasum(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result


class AsumBenchmark(Benchmark):
    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_asum(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp.numel()

            if cur_dtype in [torch.complex64, torch.complex128]:
                result_dtype = (
                    torch.float32 if cur_dtype == torch.complex64 else torch.float64
                )
            else:
                result_dtype = cur_dtype

            result = torch.zeros(1, dtype=result_dtype, device=self.device)

            if n > 0:
                if IS_HYGON:
                    _, handle = get_l1_hipblas_context(inp)
                kwargs = {"n": n, "handle": handle, "result": result}
                if IS_HYGON:
                    kwargs["c_func"] = c_func
                yield inp, kwargs

    def get_gbps(self, args, latency):
        inp = args[0]
        result_bytes = inp.element_size() // (2 if inp.dtype.is_complex else 1)
        io_amount = shape_utils.size_in_bytes(inp) + result_bytes
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # Leading-order reduction count; abs is not floating-point arithmetic.
        return (2 if x.dtype.is_complex else 1) * n

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]


class AsumStrideBenchmark(Benchmark):
    def __init__(self, *args, incx=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_asum(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            n = shape[0]
            inp = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)

            if cur_dtype in [torch.complex64, torch.complex128]:
                result_dtype = (
                    torch.float32 if cur_dtype == torch.complex64 else torch.float64
                )
            else:
                result_dtype = cur_dtype

            result = torch.zeros(1, dtype=result_dtype, device=self.device)

            if n > 0:
                if IS_HYGON:
                    _, handle = get_l1_hipblas_context(inp)
                kwargs = {
                    "incx": self.incx,
                    "n": n,
                    "handle": handle,
                    "result": result,
                }
                if IS_HYGON:
                    kwargs["c_func"] = c_func
                yield inp, kwargs

    def get_gbps(self, args, latency):
        inp = args[0]
        n = (inp.numel() + self.incx - 1) // self.incx
        element_size = inp.element_size()
        result_bytes = inp.element_size() // (2 if inp.dtype.is_complex else 1)
        io_amount = n * element_size + result_bytes
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # Leading-order reduction count; abs is not floating-point arithmetic.
        return (2 if x.dtype.is_complex else 1) * n

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]


if IS_HYGON:
    AsumBenchmark.correctness_reference = "hipBLAS"
    AsumStrideBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.sasum
def test_perf_sasum():
    bench = AsumBenchmark(
        op_name="sasum",
        torch_op=cublas_sasum,
        blas_op=blas_sasum_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dasum
def test_perf_dasum():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumBenchmark(
        op_name="dasum",
        torch_op=cublas_dasum,
        blas_op=blas_dasum_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.scasum
def test_perf_scasum():
    bench = AsumBenchmark(
        op_name="scasum",
        torch_op=cublas_scasum,
        blas_op=blas_scasum_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dzasum
def test_perf_dzasum():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumBenchmark(
        op_name="dzasum",
        torch_op=cublas_dzasum,
        blas_op=blas_dzasum_wrapper,
        dtypes=[torch.complex128],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sasum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_sasum_stride(incx):
    bench = AsumStrideBenchmark(
        op_name=f"sasum_stride_incx{incx}",
        torch_op=cublas_sasum,
        blas_op=blas_sasum_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dasum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dasum_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumStrideBenchmark(
        op_name=f"dasum_stride_incx{incx}",
        torch_op=cublas_dasum,
        blas_op=blas_dasum_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.scasum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_scasum_stride(incx):
    bench = AsumStrideBenchmark(
        op_name=f"scasum_stride_incx{incx}",
        torch_op=cublas_scasum,
        blas_op=blas_scasum_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dzasum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dzasum_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumStrideBenchmark(
        op_name=f"dzasum_stride_incx{incx}",
        torch_op=cublas_dzasum,
        blas_op=blas_dzasum_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)

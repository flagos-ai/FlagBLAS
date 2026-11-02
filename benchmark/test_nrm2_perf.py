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

    def _resolve_hipblas_nrm2(dtype):
        symbol = {
            torch.float32: "hipblasSnrm2",
            torch.float64: "hipblasDnrm2",
            torch.complex64: "hipblasScnrm2",
            torch.complex128: "hipblasDznrm2",
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

elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp, cublas
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


if IS_HYGON:

    def hipblas_nrm2_baseline(x, incx=1, n=None, handle=None, result=None, c_func=None):
        check_hipblas_status(
            c_func(handle, n, x.data_ptr(), incx, result.data_ptr()), "hipBLAS NRM2"
        )
        return result

    def _hygon_nrm2_wrapper(op):
        def wrapper(x, incx=1, n=None, handle=None, result=None, c_func=None):
            op(n, x, incx, result)
            return result

        return wrapper

    cublas_snrm2 = hipblas_nrm2_baseline
    cublas_dnrm2 = hipblas_nrm2_baseline
    cublas_scnrm2 = hipblas_nrm2_baseline
    cublas_dznrm2 = hipblas_nrm2_baseline
    blas_snrm2_wrapper = _hygon_nrm2_wrapper(flag_blas.snrm2)
    blas_dnrm2_wrapper = _hygon_nrm2_wrapper(flag_blas.dnrm2)
    blas_scnrm2_wrapper = _hygon_nrm2_wrapper(flag_blas.scnrm2)
    blas_dznrm2_wrapper = _hygon_nrm2_wrapper(flag_blas.dznrm2)

else:

    def cublas_snrm2(x, incx=1, n=None, handle=None, result=None):
        cublas.snrm2(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_dnrm2(x, incx=1, n=None, handle=None, result=None):
        cublas.dnrm2(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_scnrm2(x, incx=1, n=None, handle=None, result=None):
        cublas.scnrm2(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_dznrm2(x, incx=1, n=None, handle=None, result=None):
        cublas.dznrm2(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def blas_snrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
        flag_blas.snrm2(n, x, incx, result)
        return result

    def blas_dnrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
        flag_blas.dnrm2(n, x, incx, result)
        return result

    def blas_scnrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
        flag_blas.scnrm2(n, x, incx, result)
        return result

    def blas_dznrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
        flag_blas.dznrm2(n, x, incx, result)
        return result


class Nrm2Benchmark(Benchmark):
    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_nrm2(cur_dtype)
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
        # Sum squares of n real or 2*n real components, then take the sqrt.
        return (4 if x.dtype.is_complex else 2) * n

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]


class Nrm2StrideBenchmark(Benchmark):
    def __init__(self, *args, incx=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_nrm2(cur_dtype)
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
        # Sum squares of n real or 2*n real components, then take the sqrt.
        return (4 if x.dtype.is_complex else 2) * n

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]


if IS_HYGON:
    Nrm2Benchmark.correctness_reference = "hipBLAS"
    Nrm2StrideBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.snrm2
def test_perf_snrm2():
    bench = Nrm2Benchmark(
        op_name="snrm2",
        torch_op=cublas_snrm2,
        blas_op=blas_snrm2_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dnrm2
def test_perf_dnrm2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2Benchmark(
        op_name="dnrm2",
        torch_op=cublas_dnrm2,
        blas_op=blas_dnrm2_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.scnrm2
def test_perf_scnrm2():
    bench = Nrm2Benchmark(
        op_name="scnrm2",
        torch_op=cublas_scnrm2,
        blas_op=blas_scnrm2_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dznrm2
def test_perf_dznrm2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2Benchmark(
        op_name="dznrm2",
        torch_op=cublas_dznrm2,
        blas_op=blas_dznrm2_wrapper,
        dtypes=[torch.complex128],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.snrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_snrm2_stride(incx):
    bench = Nrm2StrideBenchmark(
        op_name=f"snrm2_stride_incx{incx}",
        torch_op=cublas_snrm2,
        blas_op=blas_snrm2_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dnrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dnrm2_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2StrideBenchmark(
        op_name=f"dnrm2_stride_incx{incx}",
        torch_op=cublas_dnrm2,
        blas_op=blas_dnrm2_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.scnrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_scnrm2_stride(incx):
    bench = Nrm2StrideBenchmark(
        op_name=f"scnrm2_stride_incx{incx}",
        torch_op=cublas_scnrm2,
        blas_op=blas_scnrm2_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dznrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dznrm2_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2StrideBenchmark(
        op_name=f"dznrm2_stride_incx{incx}",
        torch_op=cublas_dznrm2,
        blas_op=blas_dznrm2_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)

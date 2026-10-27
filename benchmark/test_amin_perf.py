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
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from tests.accuracy_utils import AMIN_SHAPES

IS_HYGON = flag_blas.vendor_name == "hygon"
if IS_HYGON:
    import ctypes

    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


STRIDES = [2, 3, 5]


def gems_isamin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.isamin(n, x, incx, result)
    return result


def gems_idamin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.idamin(n, x, incx, result)
    return result


def gems_icamin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.icamin(n, x, incx, result)
    return result


def gems_izamin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.izamin(n, x, incx, result)
    return result


if IS_HYGON:

    def _resolve_hipblas_amin(dtype):
        symbol = {
            torch.float32: "hipblasIsamin",
            torch.float64: "hipblasIdamin",
            torch.complex64: "hipblasIcamin_v2",
            torch.complex128: "hipblasIzamin_v2",
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

    def hipblas_amin_baseline(x, incx=1, n=None, handle=None, result=None, c_func=None):
        check_hipblas_status(
            c_func(handle, n, x.data_ptr(), incx, result.data_ptr()), "hipBLAS AMIN"
        )
        return result

    def _hygon_amin_wrapper(op):
        def wrapper(x, incx=1, n=None, handle=None, result=None, c_func=None):
            return op(x, incx=incx, n=n, handle=handle, result=result)

        return wrapper

    cublas_isamin = hipblas_amin_baseline
    cublas_idamin = hipblas_amin_baseline
    cublas_icamin = hipblas_amin_baseline
    cublas_izamin = hipblas_amin_baseline
    gems_isamin = _hygon_amin_wrapper(gems_isamin)
    gems_idamin = _hygon_amin_wrapper(gems_idamin)
    gems_icamin = _hygon_amin_wrapper(gems_icamin)
    gems_izamin = _hygon_amin_wrapper(gems_izamin)

else:
    if flag_blas.vendor_name == "thead":

        def _thead_amin_baseline(func):
            def call(x, incx=1, n=None, handle=None, result=None, host_result=None):
                func(handle, n, x.data_ptr(), incx, host_result.data_ptr())
                return host_result

            return call

        def _thead_amin_wrapper(op):
            def call(x, incx=1, n=None, handle=None, result=None, host_result=None):
                return op(x, incx=incx, n=n, handle=handle, result=result)

            return call

        cublas_isamin = _thead_amin_baseline(cublas.isamin)
        cublas_idamin = _thead_amin_baseline(cublas.idamin)
        gems_isamin = _thead_amin_wrapper(gems_isamin)
        gems_idamin = _thead_amin_wrapper(gems_idamin)
    else:

        def cublas_isamin(x, incx=1, n=None, handle=None, result=None):
            cublas.isamin(handle, n, x.data_ptr(), incx, result.data_ptr())
            return result

        def cublas_idamin(x, incx=1, n=None, handle=None, result=None):
            cublas.idamin(handle, n, x.data_ptr(), incx, result.data_ptr())
            return result

    def cublas_icamin(x, incx=1, n=None, handle=None, result=None):
        cublas.icamin(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_izamin(x, incx=1, n=None, handle=None, result=None):
        cublas.izamin(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result


class AminBenchmark(Benchmark):
    def __init__(self, *args, incx=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return AMIN_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_amin(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            if flag_blas.vendor_name == "thead" and not cur_dtype.is_complex:
                cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            else:
                cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            result = torch.zeros(1, dtype=torch.int32, device=self.device)
            if IS_HYGON:
                _, handle = get_l1_hipblas_context(x)
            kwargs = {
                "n": n,
                "incx": self.incx,
                "handle": handle,
                "result": result,
            }
            if IS_HYGON:
                kwargs["c_func"] = c_func
            if flag_blas.vendor_name == "thead" and not cur_dtype.is_complex:
                # Allocate outside the timer; FlagBLAS retains its device output.
                kwargs["host_result"] = torch.empty(1, dtype=torch.int32, device="cpu")
            yield x, kwargs

    def validate_results(self, reference_result, blas_result, dtype, reduce_dim=1):
        if flag_blas.vendor_name == "thead" and not dtype.is_complex:
            # The vendor API returns a host index; FlagBLAS returns a device index.
            reference_result = reference_result.cpu()
            blas_result = blas_result.cpu()
        return super().validate_results(
            reference_result, blas_result, dtype, reduce_dim=reduce_dim
        )

    def get_gbps(self, args, latency):
        x = args[0]
        n = (x.numel() + self.incx - 1) // self.incx
        io_amount = n * x.element_size() + 4
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # BLAS complex magnitude is abs(real) + abs(imag); comparisons are not FLOPs.
        return n if x.dtype.is_complex else 0


if IS_HYGON:
    AminBenchmark.correctness_reference = "hipBLAS"
elif flag_blas.vendor_name == "thead":
    AminBenchmark.correctness_reference = "PPU BLAS (host output)"


@pytest.mark.isamin
def test_perf_isamin():
    bench = AminBenchmark(
        op_name="isamin",
        torch_op=cublas_isamin,
        gems_op=gems_isamin,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.idamin
def test_perf_idamin():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name="idamin",
        torch_op=cublas_idamin,
        gems_op=gems_idamin,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.icamin
def test_perf_icamin():
    bench = AminBenchmark(
        op_name="icamin",
        torch_op=cublas_icamin,
        gems_op=gems_icamin,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.izamin
def test_perf_izamin():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name="izamin",
        torch_op=cublas_izamin,
        gems_op=gems_izamin,
        dtypes=[torch.complex128],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.isamin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_isamin_stride(incx):
    bench = AminBenchmark(
        op_name=f"isamin_stride_incx{incx}",
        torch_op=cublas_isamin,
        gems_op=gems_isamin,
        dtypes=[torch.float32],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.idamin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_idamin_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name=f"idamin_stride_incx{incx}",
        torch_op=cublas_idamin,
        gems_op=gems_idamin,
        dtypes=[torch.float64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.icamin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_icamin_stride(incx):
    bench = AminBenchmark(
        op_name=f"icamin_stride_incx{incx}",
        torch_op=cublas_icamin,
        gems_op=gems_icamin,
        dtypes=[torch.complex64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.izamin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_izamin_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name=f"izamin_stride_incx{incx}",
        torch_op=cublas_izamin,
        gems_op=gems_izamin,
        dtypes=[torch.complex128],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)

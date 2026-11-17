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
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from tests.accuracy_utils import DEFAULT_SHAPES

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp, cublas
elif flag_blas.vendor_name != "ascend":
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]


def gems_sdot_wrapper(x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None):
    flag_blas.sdot(n, x, incx, y, incy, result)
    return result


def gems_ddot_wrapper(x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None):
    flag_blas.ddot(n, x, incx, y, incy, result)
    return result


if IS_HYGON:

    def _resolve_hipblas_dot(dtype):
        symbol = {
            torch.float32: "hipblasSdot",
            torch.float64: "hipblasDdot",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_dot_baseline(
        x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None
    ):
        if n is None:
            n = min(x.numel() // incx, y.numel() // incy)
        if n <= 0:
            result.zero_()
            return result
        check_hipblas_status(
            c_func(
                handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr()
            ),
            "hipBLAS DOT",
        )
        return result

    def _hygon_dot_wrapper(op):
        def wrapper(x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None):
            return op(x, y, result, n=n, incx=incx, incy=incy, handle=handle)

        return wrapper

    cublas_dot = hipblas_dot_baseline
    gems_sdot_wrapper = _hygon_dot_wrapper(gems_sdot_wrapper)
    gems_ddot_wrapper = _hygon_dot_wrapper(gems_ddot_wrapper)

else:

    def cublas_dot(x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None):
        if n is None:
            n = min(x.numel() // incx, y.numel() // incy)
        if n <= 0:
            result.zero_()
            return result
        c_func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr())
        return result


class DotBenchmark(Benchmark):
    def __init__(self, *args, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_dot(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
            c_func = {torch.float32: cublas.sdot, torch.float64: cublas.ddot}[cur_dtype]
        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            result = torch.zeros(1, dtype=cur_dtype, device=self.device)
            if IS_HYGON:
                _, handle = get_l1_hipblas_context(x)
            kwargs = {
                "n": n,
                "incx": self.incx,
                "incy": self.incy,
                "handle": handle,
            }
            kwargs["c_func"] = c_func
            yield x, y, result, kwargs

    def get_gbps(self, args, latency):
        x, y, result = args[0], args[1], args[2]
        n = min(x.numel() // self.incx, y.numel() // self.incy)
        io_amount = n * (x.element_size() + y.element_size())
        io_amount += result.numel() * result.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # Leading-order BLAS count; conjugation does not add FLOPs.
        return (8 if x.dtype.is_complex else 2) * n

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]


if IS_HYGON:
    DotBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.sdot
def test_perf_sdot():
    bench = DotBenchmark(
        op_name="sdot",
        torch_op=cublas_dot,
        gems_op=gems_sdot_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ddot
def test_perf_ddot():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotBenchmark(
        op_name="ddot",
        torch_op=cublas_dot,
        gems_op=gems_ddot_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sdot
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_sdot_stride(incx, incy):
    bench = DotBenchmark(
        op_name=f"sdot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dot,
        gems_op=gems_sdot_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ddot
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_ddot_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotBenchmark(
        op_name=f"ddot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dot,
        gems_op=gems_ddot_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)

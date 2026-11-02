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
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]


def gems_dotc_wrapper(x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None):
    if x.dtype == torch.complex64:
        flag_blas.cdotc(n, x, incx, y, incy, result)
    elif x.dtype == torch.complex128:
        flag_blas.zdotc(n, x, incx, y, incy, result)
    else:
        raise TypeError(f"Unsupported dtype for dotc: {x.dtype}")
    return result


if IS_HYGON:

    def _resolve_hipblas_dotc(dtype):
        symbol = {
            torch.complex64: "hipblasCdotc_v2",
            torch.complex128: "hipblasZdotc_v2",
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

    def hipblas_dotc_baseline(
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
            "hipBLAS DOTC",
        )
        return result

    def _hygon_dotc_wrapper(op):
        def wrapper(x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None):
            return op(x, y, result, n=n, incx=incx, incy=incy, handle=handle)

        return wrapper

    cublas_dotc = hipblas_dotc_baseline
    gems_dotc_wrapper = _hygon_dotc_wrapper(gems_dotc_wrapper)

else:

    def cublas_dotc(x, y, result, n=None, incx=1, incy=1, handle=None, c_func=None):
        if n is None:
            n = min(x.numel() // incx, y.numel() // incy)
        if n <= 0:
            result.zero_()
            return result
        c_func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr())
        return result


class DotcBenchmark(Benchmark):
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
            c_func = _resolve_hipblas_dotc(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
            c_func = {torch.complex64: cublas.cdotc, torch.complex128: cublas.zdotc}[
                cur_dtype
            ]
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
    DotcBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.cdotc
def test_perf_cdotc():
    bench = DotcBenchmark(
        op_name="cdotc",
        torch_op=cublas_dotc,
        gems_op=gems_dotc_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zdotc
def test_perf_zdotc():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotcBenchmark(
        op_name="zdotc",
        torch_op=cublas_dotc,
        gems_op=gems_dotc_wrapper,
        dtypes=[torch.complex128],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cdotc
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_cdotc_stride(incx, incy):
    bench = DotcBenchmark(
        op_name=f"cdotc_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dotc,
        gems_op=gems_dotc_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zdotc
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_zdotc_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotcBenchmark(
        op_name=f"zdotc_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dotc,
        gems_op=gems_dotc_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)

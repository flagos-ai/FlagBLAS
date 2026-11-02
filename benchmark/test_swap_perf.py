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
from tests.accuracy_utils import DEFAULT_SHAPES

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    from tests.hipblas_reference import (
        check_hipblas_status,
        get_hipblas_context,
        get_hipblas_library,
    )
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp
else:
    import cupy as cp


PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]


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


_cublas = None if IS_HYGON else load_cublas()


def _get_cublas_func(*names):
    for name in names:
        if hasattr(_cublas, name):
            return getattr(_cublas, name)
    raise RuntimeError(f"Cannot find any cuBLAS symbol from: {', '.join(names)}")


def gems_swap_wrapper(x, y, n=None, incx=1, incy=1, handle=None, tmp=None, c_func=None):
    if x.dtype == torch.float32:
        flag_blas.sswap(n, x, incx, y, incy)
    elif x.dtype == torch.float64:
        flag_blas.dswap(n, x, incx, y, incy)
    elif x.dtype == torch.complex64:
        flag_blas.cswap(n, x, incx, y, incy)
    elif x.dtype == torch.complex128:
        flag_blas.zswap(n, x, incx, y, incy)
    else:
        raise TypeError(f"Unsupported dtype for swap: {x.dtype}")
    return x, y


if IS_HYGON:

    def _resolve_hipblas_swap(dtype):
        symbol = {
            torch.float32: "hipblasSswap",
            torch.float64: "hipblasDswap",
            torch.complex64: "hipblasCswap_v2",
            torch.complex128: "hipblasZswap_v2",
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

    def hipblas_swap_baseline(
        x, y, n=None, incx=1, incy=1, handle=None, tmp=None, c_func=None
    ):
        if n is None:
            n = min(x.numel() // incx, y.numel() // incy)
        if n <= 0:
            return x, y
        check_hipblas_status(
            c_func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy), "hipBLAS SWAP"
        )
        return x, y

    def _hygon_swap_wrapper(op):
        def wrapper(x, y, n=None, incx=1, incy=1, handle=None, tmp=None, c_func=None):
            return op(x, y, n=n, incx=incx, incy=incy)

        return wrapper

    cublas_swap = hipblas_swap_baseline
    gems_swap_wrapper = _hygon_swap_wrapper(gems_swap_wrapper)

else:

    def cublas_swap(x, y, n=None, incx=1, incy=1, handle=None, tmp=None, c_func=None):
        if n <= 0:
            return x, y
        status = c_func(handle, n, x.data_ptr(), incx, y.data_ptr(), incy)
        if status != 0:
            raise RuntimeError(f"cuBLAS swap failed with status code: {status}")
        return x, y


class SwapBenchmark(Benchmark):
    def __init__(self, *args, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_swap(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            prefix = {
                torch.float32: "S",
                torch.float64: "D",
                torch.complex64: "C",
                torch.complex128: "Z",
            }[cur_dtype]
            c_func = _get_cublas_func(f"cublas{prefix}swap_v2", f"cublas{prefix}swap")
            c_func.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            c_func.restype = ctypes.c_int
        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            if IS_HYGON:
                _, handle = get_hipblas_context(x)
            kwargs = {
                "n": n,
                "incx": self.incx,
                "incy": self.incy,
                "handle": handle,
            }
            kwargs["c_func"] = c_func
            yield x, y, kwargs

    def get_gbps(self, args, latency):
        x = args[0]
        n = x.numel() // self.incx
        io_amount = 4 * n * x.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        # Data movement has no floating-point arithmetic.
        return 0


if IS_HYGON:
    SwapBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.sswap
def test_perf_sswap():
    bench = SwapBenchmark(
        op_name="sswap",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dswap
def test_perf_dswap():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SwapBenchmark(
        op_name="dswap",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cswap
def test_perf_cswap():
    bench = SwapBenchmark(
        op_name="cswap",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zswap
def test_perf_zswap():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SwapBenchmark(
        op_name="zswap",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.complex128],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sswap
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_sswap_stride(incx, incy):
    bench = SwapBenchmark(
        op_name=f"sswap_stride_incx{incx}_incy{incy}",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dswap
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_dswap_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SwapBenchmark(
        op_name=f"dswap_stride_incx{incx}_incy{incy}",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cswap
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_cswap_stride(incx, incy):
    bench = SwapBenchmark(
        op_name=f"cswap_stride_incx{incx}_incy{incy}",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zswap
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_zswap_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SwapBenchmark(
        op_name=f"zswap_stride_incx{incx}_incy{incy}",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)

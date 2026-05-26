import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from tests.accuracy_utils import DEFAULT_SHAPES

PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]


def load_cublas():
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


_cublas = load_cublas()


def _get_cublas_func(*names):
    for name in names:
        if hasattr(_cublas, name):
            return getattr(_cublas, name)
    raise RuntimeError(f"Cannot find any cuBLAS symbol from: {', '.join(names)}")


def cublas_swap(x, y, n=None, incx=1, incy=1, tmp=None):
    x = x.clone()
    y = y.clone()
    if n is None:
        n = min(x.numel() // incx, y.numel() // incy)
    if n <= 0:
        return x, y

    if x.dtype == torch.float32:
        func = _get_cublas_func("cublasSswap_v2", "cublasSswap")
    elif x.dtype == torch.float64:
        func = _get_cublas_func("cublasDswap_v2", "cublasDswap")
    elif x.dtype == torch.complex64:
        func = _get_cublas_func("cublasCswap_v2", "cublasCswap")
    elif x.dtype == torch.complex128:
        func = _get_cublas_func("cublasZswap_v2", "cublasZswap")
    else:
        raise TypeError(f"Unsupported dtype for swap: {x.dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(n),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS swap failed with status code: {status}")
    return x, y


def gems_swap_wrapper(x, y, n=None, incx=1, incy=1, tmp=None):
    x = x.clone()
    y = y.clone()
    if x.dtype == torch.float32:
        flag_blas.ops.sswap(n, x, incx, y, incy)
    elif x.dtype == torch.float64:
        flag_blas.ops.dswap(n, x, incx, y, incy)
    elif x.dtype == torch.complex64:
        flag_blas.ops.cswap(n, x, incx, y, incy)
    elif x.dtype == torch.complex128:
        flag_blas.ops.zswap(n, x, incx, y, incy)
    else:
        raise TypeError(f"Unsupported dtype for swap: {x.dtype}")
    return x, y


class SwapBenchmark(Benchmark):
    def __init__(self, *args, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = DEFAULT_SHAPES
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            yield x, y, {
                "n": n,
                "incx": self.incx,
                "incy": self.incy,
            }

    def get_gbps(self, args, latency):
        x = args[0]
        n = x.numel() // self.incx
        io_amount = 4 * n * x.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.swap
def test_perf_sswap():
    bench = SwapBenchmark(
        op_name="sswap",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.swap
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


@pytest.mark.swap
def test_perf_cswap():
    bench = SwapBenchmark(
        op_name="cswap",
        torch_op=cublas_swap,
        gems_op=gems_swap_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.swap
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


@pytest.mark.swap
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


@pytest.mark.swap
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


@pytest.mark.swap
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


@pytest.mark.swap
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

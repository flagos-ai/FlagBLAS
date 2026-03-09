from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


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


def gems_sasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.sasum(n, x, incx, result)
    return result


def gems_dasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.dasum(n, x, incx, result)
    return result


def gems_scasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.scasum(n, x, incx, result)
    return result


def gems_dzasum_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.dzasum(n, x, incx, result)
    return result


class AsumBenchmark(Benchmark):

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        shapes = [
            (1024,),
            (5333,),
            (65536,),
            (100000,),
            (1048576,),
            (3000000,),
            (4194304,),
            (10000000,),
            (16777216,),
            (33554432,),
            (50000000,),
            (67108864,),
            (134217728,),
        ]
        self.shapes = shapes
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp.numel()

            if cur_dtype in [torch.complex64, torch.complex128]:
                result_dtype = torch.float32 if cur_dtype == torch.complex64 else torch.float64
            else:
                result_dtype = cur_dtype

            result = torch.zeros(1, dtype=result_dtype, device=self.device)

            if n > 0:
                yield inp, {"n": n, "handle": handle, "result": result}

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp)
        return io_amount * 1e-9 / (latency * 1e-3)


class AsumStrideBenchmark(Benchmark):

    def __init__(self, *args, incx=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        shapes_1d = [
            (1024,),
            (5333,),
            (65536,),
            (100000,),
            (1048576,),
            (3000000,),
            (4194304,),
        ]
        self.shapes = shapes_1d
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            n = shape[0]
            inp = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)

            if cur_dtype in [torch.complex64, torch.complex128]:
                result_dtype = torch.float32 if cur_dtype == torch.complex64 else torch.float64
            else:
                result_dtype = cur_dtype

            result = torch.zeros(1, dtype=result_dtype, device=self.device)

            if n > 0:
                yield inp, {"incx": self.incx, "n": n, "handle": handle, "result": result}

    def get_gbps(self, args, latency):
        inp = args[0]
        n = (inp.numel() + self.incx - 1) // self.incx
        element_size = inp.element_size()
        io_amount = n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.asum
def test_perf_sasum():
    bench = AsumBenchmark(
        op_name="sasum",
        torch_op=cublas_sasum,
        gems_op=gems_sasum_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.asum
def test_perf_dasum():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumBenchmark(
        op_name="dasum",
        torch_op=cublas_dasum,
        gems_op=gems_dasum_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.asum
def test_perf_scasum():
    bench = AsumBenchmark(
        op_name="scasum",
        torch_op=cublas_scasum,
        gems_op=gems_scasum_wrapper,
        dtypes=[torch.complex64],
    )
    bench.run()


@pytest.mark.asum
def test_perf_dzasum():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumBenchmark(
        op_name="dzasum",
        torch_op=cublas_dzasum,
        gems_op=gems_dzasum_wrapper,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.asum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_sasum_stride(incx):
    bench = AsumStrideBenchmark(
        op_name=f"sasum_stride_incx{incx}",
        torch_op=cublas_sasum,
        gems_op=gems_sasum_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    bench.run()


@pytest.mark.asum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dasum_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumStrideBenchmark(
        op_name=f"dasum_stride_incx{incx}",
        torch_op=cublas_dasum,
        gems_op=gems_dasum_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    bench.run()


@pytest.mark.asum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_scasum_stride(incx):
    bench = AsumStrideBenchmark(
        op_name=f"scasum_stride_incx{incx}",
        torch_op=cublas_scasum,
        gems_op=gems_scasum_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
    )
    bench.run()


@pytest.mark.asum
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dzasum_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AsumStrideBenchmark(
        op_name=f"dzasum_stride_incx{incx}",
        torch_op=cublas_dzasum,
        gems_op=gems_dzasum_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
    )
    bench.run()

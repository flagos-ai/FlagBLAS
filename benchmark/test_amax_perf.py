from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


def cublas_isamax(x, incx=1, n=None, handle=None, result=None):
    cublas.isamax(handle, n, x.data_ptr(), incx, result.data_ptr())
    return result


def cublas_idamax(x, incx=1, n=None, handle=None, result=None):
    cublas.idamax(handle, n, x.data_ptr(), incx, result.data_ptr())
    return result


def cublas_icamax(x, incx=1, n=None, handle=None, result=None):
    cublas.icamax(handle, n, x.data_ptr(), incx, result.data_ptr())
    return result


def cublas_izamax(x, incx=1, n=None, handle=None, result=None):
    cublas.izamax(handle, n, x.data_ptr(), incx, result.data_ptr())
    return result


def gems_samax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.samax(n, x, incx, result)
    return result


def gems_damax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.damax(n, x, incx, result)
    return result


def gems_camax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.camax(n, x, incx, result)
    return result


def gems_zamax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.zamax(n, x, incx, result)
    return result


class AmaxBenchmark(Benchmark):

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

            result = torch.zeros(1, dtype=torch.int32, device=self.device)

            if n > 0:
                yield inp, {"n": n, "handle": handle, "result": result}

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp)
        return io_amount * 1e-9 / (latency * 1e-3)


class AmaxStrideBenchmark(Benchmark):

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

            result = torch.zeros(1, dtype=torch.int32, device=self.device)

            if n > 0:
                yield inp, {
                    "incx": self.incx,
                    "n": n,
                    "handle": handle,
                    "result": result,
                }

    def get_gbps(self, args, latency):
        inp = args[0]
        n = (inp.numel() + self.incx - 1) // self.incx
        element_size = inp.element_size()
        io_amount = n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.amax
def test_perf_samax():
    bench = AmaxBenchmark(
        op_name="samax",
        torch_op=cublas_isamax,
        gems_op=gems_samax_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.amax
def test_perf_damax():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxBenchmark(
        op_name="damax",
        torch_op=cublas_idamax,
        gems_op=gems_damax_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.amax
def test_perf_camax():
    bench = AmaxBenchmark(
        op_name="camax",
        torch_op=cublas_icamax,
        gems_op=gems_camax_wrapper,
        dtypes=[torch.complex64],
    )
    bench.run()


@pytest.mark.amax
def test_perf_zamax():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxBenchmark(
        op_name="zamax",
        torch_op=cublas_izamax,
        gems_op=gems_zamax_wrapper,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.amax
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_samax_stride(incx):
    bench = AmaxStrideBenchmark(
        op_name=f"samax_stride_incx{incx}",
        torch_op=cublas_isamax,
        gems_op=gems_samax_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    bench.run()


@pytest.mark.amax
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_damax_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxStrideBenchmark(
        op_name=f"damax_stride_incx{incx}",
        torch_op=cublas_idamax,
        gems_op=gems_damax_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    bench.run()


@pytest.mark.amax
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_camax_stride(incx):
    bench = AmaxStrideBenchmark(
        op_name=f"camax_stride_incx{incx}",
        torch_op=cublas_icamax,
        gems_op=gems_camax_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
    )
    bench.run()


@pytest.mark.amax
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_zamax_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxStrideBenchmark(
        op_name=f"zamax_stride_incx{incx}",
        torch_op=cublas_izamax,
        gems_op=gems_zamax_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
    )
    bench.run()

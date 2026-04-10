from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


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


def gems_snrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.snrm2(n, x, incx, result)
    return result


def gems_dnrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.dnrm2(n, x, incx, result)
    return result


def gems_scnrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.scnrm2(n, x, incx, result)
    return result


def gems_dznrm2_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.dznrm2(n, x, incx, result)
    return result


class Nrm2Benchmark(Benchmark):

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
                result_dtype = (
                    torch.float32 if cur_dtype == torch.complex64 else torch.float64
                )
            else:
                result_dtype = cur_dtype

            result = torch.zeros(1, dtype=result_dtype, device=self.device)

            if n > 0:
                yield inp, {"n": n, "handle": handle, "result": result}

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp)
        return io_amount * 1e-9 / (latency * 1e-3)


class Nrm2StrideBenchmark(Benchmark):

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
                result_dtype = (
                    torch.float32 if cur_dtype == torch.complex64 else torch.float64
                )
            else:
                result_dtype = cur_dtype

            result = torch.zeros(1, dtype=result_dtype, device=self.device)

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


@pytest.mark.nrm2
def test_perf_snrm2():
    bench = Nrm2Benchmark(
        op_name="snrm2",
        torch_op=cublas_snrm2,
        gems_op=gems_snrm2_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.nrm2
def test_perf_dnrm2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2Benchmark(
        op_name="dnrm2",
        torch_op=cublas_dnrm2,
        gems_op=gems_dnrm2_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.nrm2
def test_perf_scnrm2():
    bench = Nrm2Benchmark(
        op_name="scnrm2",
        torch_op=cublas_scnrm2,
        gems_op=gems_scnrm2_wrapper,
        dtypes=[torch.complex64],
    )
    bench.run()


@pytest.mark.nrm2
def test_perf_dznrm2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2Benchmark(
        op_name="dznrm2",
        torch_op=cublas_dznrm2,
        gems_op=gems_dznrm2_wrapper,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.nrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_snrm2_stride(incx):
    bench = Nrm2StrideBenchmark(
        op_name=f"snrm2_stride_incx{incx}",
        torch_op=cublas_snrm2,
        gems_op=gems_snrm2_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    bench.run()


@pytest.mark.nrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dnrm2_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2StrideBenchmark(
        op_name=f"dnrm2_stride_incx{incx}",
        torch_op=cublas_dnrm2,
        gems_op=gems_dnrm2_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    bench.run()


@pytest.mark.nrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_scnrm2_stride(incx):
    bench = Nrm2StrideBenchmark(
        op_name=f"scnrm2_stride_incx{incx}",
        torch_op=cublas_scnrm2,
        gems_op=gems_scnrm2_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
    )
    bench.run()


@pytest.mark.nrm2
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dznrm2_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Nrm2StrideBenchmark(
        op_name=f"dznrm2_stride_incx{incx}",
        torch_op=cublas_dznrm2,
        gems_op=gems_dznrm2_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
    )
    bench.run()
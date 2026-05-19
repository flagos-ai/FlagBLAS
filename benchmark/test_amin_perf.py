from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark
from tests.accuracy_utils import DEFAULT_SHAPES

STRIDES = [2, 3, 5]


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


def gems_samin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.samin(n, x, incx, result)
    return result


def gems_damin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.damin(n, x, incx, result)
    return result


def gems_camin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.camin(n, x, incx, result)
    return result


def gems_zamin(x, incx=1, n=None, handle=None, result=None):
    flag_blas.ops.zamin(n, x, incx, result)
    return result


class AminBenchmark(Benchmark):
    def __init__(self, *args, incx=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = DEFAULT_SHAPES
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            result = torch.zeros(1, dtype=torch.int32, device=self.device)
            yield x, {
                "n": n,
                "incx": self.incx,
                "handle": handle,
                "result": result,
            }

    def get_gbps(self, args, latency):
        x = args[0]
        n = (x.numel() + self.incx - 1) // self.incx
        return n * x.element_size() * 1e-9 / (latency * 1e-3)


@pytest.mark.amin
def test_perf_samin():
    bench = AminBenchmark(
        op_name="samin",
        torch_op=cublas_isamin,
        gems_op=gems_samin,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.amin
def test_perf_damin():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name="damin",
        torch_op=cublas_idamin,
        gems_op=gems_damin,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.amin
def test_perf_camin():
    bench = AminBenchmark(
        op_name="camin",
        torch_op=cublas_icamin,
        gems_op=gems_camin,
        dtypes=[torch.complex64],
    )
    bench.run()


@pytest.mark.amin
def test_perf_zamin():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name="zamin",
        torch_op=cublas_izamin,
        gems_op=gems_zamin,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.amin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_samin_stride(incx):
    bench = AminBenchmark(
        op_name=f"samin_stride_incx{incx}",
        torch_op=cublas_isamin,
        gems_op=gems_samin,
        dtypes=[torch.float32],
        incx=incx,
    )
    bench.run()


@pytest.mark.amin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_damin_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name=f"damin_stride_incx{incx}",
        torch_op=cublas_idamin,
        gems_op=gems_damin,
        dtypes=[torch.float64],
        incx=incx,
    )
    bench.run()


@pytest.mark.amin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_camin_stride(incx):
    bench = AminBenchmark(
        op_name=f"camin_stride_incx{incx}",
        torch_op=cublas_icamin,
        gems_op=gems_camin,
        dtypes=[torch.complex64],
        incx=incx,
    )
    bench.run()


@pytest.mark.amin
@pytest.mark.parametrize("incx", STRIDES)
def test_perf_zamin_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AminBenchmark(
        op_name=f"zamin_stride_incx{incx}",
        torch_op=cublas_izamin,
        gems_op=gems_zamin,
        dtypes=[torch.complex128],
        incx=incx,
    )
    bench.run()

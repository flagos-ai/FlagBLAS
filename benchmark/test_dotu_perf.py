from typing import Generator

import pytest
import torch
import cupy as cp
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark
from tests.accuracy_utils import DEFAULT_SHAPES

PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]


def cublas_dotu(x, y, result, n=None, incx=1, incy=1, handle=None):
    if n is None:
        n = min(x.numel() // incx, y.numel() // incy)
    if n <= 0:
        result.zero_()
        return result
    if x.dtype == torch.complex64:
        cublas.cdotu(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr())
    elif x.dtype == torch.complex128:
        cublas.zdotu(handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr())
    else:
        raise TypeError(f"Unsupported dtype for dotu: {x.dtype}")
    return result


def gems_dotu_wrapper(x, y, result, n=None, incx=1, incy=1, handle=None):
    if x.dtype == torch.complex64:
        flag_blas.ops.cdotu(n, x, incx, y, incy, result)
    elif x.dtype == torch.complex128:
        flag_blas.ops.zdotu(n, x, incx, y, incy, result)
    else:
        raise TypeError(f"Unsupported dtype for dotu: {x.dtype}")
    return result


class DotuBenchmark(Benchmark):
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
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            result = torch.zeros(1, dtype=cur_dtype, device=self.device)
            yield x, y, result, {
                "n": n,
                "incx": self.incx,
                "incy": self.incy,
                "handle": handle,
            }

    def get_gbps(self, args, latency):
        x, y = args[0], args[1]
        n = min(x.numel() // self.incx, y.numel() // self.incy)
        io_amount = 2 * n * x.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.dotu
def test_perf_cdotu():
    bench = DotuBenchmark(
        op_name="cdotu",
        torch_op=cublas_dotu,
        gems_op=gems_dotu_wrapper,
        dtypes=[torch.complex64],
    )
    bench.run()


@pytest.mark.dotu
def test_perf_zdotu():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotuBenchmark(
        op_name="zdotu",
        torch_op=cublas_dotu,
        gems_op=gems_dotu_wrapper,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.dotu
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_cdotu_stride(incx, incy):
    bench = DotuBenchmark(
        op_name=f"cdotu_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dotu,
        gems_op=gems_dotu_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.dotu
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_zdotu_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotuBenchmark(
        op_name=f"zdotu_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dotu,
        gems_op=gems_dotu_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
    )
    bench.run()

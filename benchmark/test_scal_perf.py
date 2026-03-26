from typing import Generator

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


def cublas_sscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    cublas.sscal(handle, n, alpha_ptr, x.data_ptr(), incx)
    return x


def cublas_dscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    cublas.dscal(handle, n, alpha_ptr, x.data_ptr(), incx)
    return x


def cublas_cscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    cublas.cscal(handle, n, alpha_ptr, x.data_ptr(), incx)
    return x


def gems_sscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.ops.sscal(n, alpha, x, incx)
    return x


def gems_dscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.ops.dscal(n, alpha, x, incx)
    return x


def gems_cscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.ops.cscal(n, alpha, x, incx)
    return x


class ScalBenchmark(Benchmark):

    def __init__(self, *args, alpha=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

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
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            alpha_np = np.array(self.alpha, dtype=np.float32)
        elif cur_dtype == torch.float64:
            alpha_np = np.array(self.alpha, dtype=np.float64)
        elif cur_dtype == torch.complex64:
            alpha_np = np.array(self.alpha, dtype=np.complex64)

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp.numel()
            if n > 0:
                yield inp, {
                    "alpha": self.alpha,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = 2 * shape_utils.size_in_bytes(inp)
        return io_amount * 1e-9 / (latency * 1e-3)


class ScalStrideBenchmark(Benchmark):

    def __init__(self, *args, incx=1, alpha=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.alpha = alpha

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
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            alpha_np = np.array(self.alpha, dtype=np.float32)
        elif cur_dtype == torch.float64:
            alpha_np = np.array(self.alpha, dtype=np.float64)
        elif cur_dtype == torch.complex64:
            alpha_np = np.array(self.alpha, dtype=np.complex64)

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            n = shape[0]
            inp = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            if n > 0:
                yield inp, {
                    "alpha": self.alpha,
                    "incx": self.incx,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }

    def get_gbps(self, args, latency):
        inp = args[0]
        n = (inp.numel() + self.incx - 1) // self.incx
        element_size = inp.element_size()
        io_amount = 2 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.scal
def test_perf_sscal():
    bench = ScalBenchmark(
        op_name="sscal",
        torch_op=cublas_sscal,
        gems_op=gems_sscal_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.scal
def test_perf_dscal():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = ScalBenchmark(
        op_name="dscal",
        torch_op=cublas_dscal,
        gems_op=gems_dscal_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.scal
def test_perf_cscal():
    bench = ScalBenchmark(
        op_name="cscal",
        torch_op=cublas_cscal,
        gems_op=gems_cscal_wrapper,
        dtypes=[torch.complex64],
        alpha=0.01 + 0.01j,
    )
    bench.run()


@pytest.mark.scal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_sscal_stride(incx):
    bench = ScalStrideBenchmark(
        op_name=f"sscal_stride_incx{incx}",
        torch_op=cublas_sscal,
        gems_op=gems_sscal_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    bench.run()


@pytest.mark.scal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dscal_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = ScalStrideBenchmark(
        op_name=f"dscal_stride_incx{incx}",
        torch_op=cublas_dscal,
        gems_op=gems_dscal_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    bench.run()


@pytest.mark.scal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_cscal_stride(incx):
    bench = ScalStrideBenchmark(
        op_name=f"cscal_stride_incx{incx}",
        torch_op=cublas_cscal,
        gems_op=gems_cscal_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        alpha=0.01 + 0.01j,
    )
    bench.run()

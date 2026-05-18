from typing import Generator

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas

from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


def cublas_saxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
    cublas.saxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
    return y


def cublas_daxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
    cublas.daxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
    return y


def cublas_caxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
    cublas.caxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
    return y


def cublas_zaxpy(x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None):
    cublas.zaxpy(handle, n, alpha_ptr, x.data_ptr(), incx, y.data_ptr(), incy)
    return y


def blas_saxpy_wrapper(
    x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
):
    flag_blas.ops.saxpy(n, alpha, x, incx, y, incy)
    return y


def blas_daxpy_wrapper(
    x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
):
    flag_blas.ops.daxpy(n, alpha, x, incx, y, incy)
    return y


def blas_caxpy_wrapper(
    x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
):
    flag_blas.ops.caxpy(n, alpha, x, incx, y, incy)
    return y


def blas_zaxpy_wrapper(
    x, y, alpha, incx=1, incy=1, n=None, handle=None, alpha_ptr=None
):
    flag_blas.ops.zaxpy(n, alpha, x, incx, y, incy)
    return y


class AxpyBenchmark(Benchmark):

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
        elif cur_dtype == torch.complex128:
            alpha_np = np.array(self.alpha, dtype=np.complex128)

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp1.numel()
            if n > 0:
                yield inp1, inp2, {
                    "alpha": self.alpha,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }

    def get_gbps(self, args, latency):
        inp1, inp2 = args[0], args[1]
        io_amount = shape_utils.size_in_bytes(inp1) + 2 * shape_utils.size_in_bytes(
            inp2
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def validate_results(self, torch_result, blas_result, dtype):
        tolerance = flag_blas.testing.RESOLUTION[dtype]
        try:
            flag_blas.testing.assert_close(
                blas_result,
                torch_result,
                dtype,
                equal_nan=False,
                reduce_dim=1,
                atol=tolerance,
            )
        except AssertionError as e:
            torch_cpu = torch_result.cpu()
            blas_cpu = blas_result.cpu()
            max_abs_diff = torch.max(torch.abs(torch_cpu - blas_cpu))
            max_rel_diff = torch.max(
                torch.abs((torch_cpu - blas_cpu) / (torch.abs(torch_cpu) + 1e-9))
            )
            raise AssertionError(
                f"Results differ beyond tolerance {tolerance} for dtype {dtype}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {torch_cpu.shape}"
            ) from e

    def run_correctness_check(self):
        if self.blas_op is None:
            raise ValueError(f"Missing FlagBLAS op for {self.op_name}")

        self.init_user_config()
        total_cases = 0
        print(
            f"[correctness] {self.op_name}: comparing FlagBLAS against cuBLAS "
            "before benchmark...",
            flush=True,
        )
        for cur_dtype in self.to_bench_dtypes:
            dtype_cases = 0
            tolerance = flag_blas.testing.RESOLUTION[cur_dtype]
            for x, y, kwargs in self.get_input_iter(cur_dtype):
                y_blas = y.clone()
                torch_result = self.torch_op(x, y, **kwargs)
                blas_result = self.blas_op(x, y_blas, **kwargs)
                self.validate_results(torch_result, blas_result, cur_dtype)
                dtype_cases += 1
            total_cases += dtype_cases
            print(
                f"[correctness] {self.op_name}: PASSED dtype={cur_dtype} "
                f"total_cases={dtype_cases} tolerance={tolerance}",
                flush=True,
            )
        print(
            f"[correctness] {self.op_name}: all {total_cases} cuBLAS comparison "
            "cases passed; starting performance benchmark.",
            flush=True,
        )


class AxpyStrideBenchmark(AxpyBenchmark):

    def __init__(self, *args, incx=1, incy=1, alpha=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy
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
        elif cur_dtype == torch.complex128:
            alpha_np = np.array(self.alpha, dtype=np.complex128)

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            n = shape[0]
            inp1 = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            if n > 0:
                yield inp1, inp2, {
                    "alpha": self.alpha,
                    "incx": self.incx,
                    "incy": self.incy,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }

    def get_gbps(self, args, latency):
        inp1 = args[0]
        n = (inp1.numel() + self.incx - 1) // self.incx
        element_size = inp1.element_size()
        io_amount = n * element_size + 2 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


def run_correctness_then_benchmark(bench):
    if not Config.query and not Config.skip_correctness:
        bench.run_correctness_check()
    elif Config.skip_correctness:
        print(
            f"[correctness] {bench.op_name}: skipped by --skip_correctness; "
            "starting performance benchmark.",
            flush=True,
        )
    bench.run()


@pytest.mark.axpy
def test_perf_saxpy():
    bench = AxpyBenchmark(
        op_name="saxpy",
        torch_op=cublas_saxpy,
        blas_op=blas_saxpy_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.axpy
def test_perf_daxpy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyBenchmark(
        op_name="daxpy",
        torch_op=cublas_daxpy,
        blas_op=blas_daxpy_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.axpy
def test_perf_caxpy():
    bench = AxpyBenchmark(
        op_name="caxpy",
        torch_op=cublas_caxpy,
        blas_op=blas_caxpy_wrapper,
        dtypes=[torch.complex64],
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.axpy
def test_perf_zaxpy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyBenchmark(
        op_name="zaxpy",
        torch_op=cublas_zaxpy,
        blas_op=blas_zaxpy_wrapper,
        dtypes=[torch.complex128],
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.axpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_saxpy_stride(incx, incy):
    bench = AxpyStrideBenchmark(
        op_name=f"saxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_saxpy,
        blas_op=blas_saxpy_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.axpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_daxpy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyStrideBenchmark(
        op_name=f"daxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_daxpy,
        blas_op=blas_daxpy_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.axpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_caxpy_stride(incx, incy):
    bench = AxpyStrideBenchmark(
        op_name=f"caxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_caxpy,
        blas_op=blas_caxpy_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.axpy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_zaxpy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AxpyStrideBenchmark(
        op_name=f"zaxpy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_zaxpy,
        blas_op=blas_zaxpy_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)

from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from tests.accuracy_utils import DEFAULT_SHAPES

PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]


def cublas_dotc(x, y, result, n=None, incx=1, incy=1, handle=None):
    if n is None:
        n = min(x.numel() // incx, y.numel() // incy)
    if n <= 0:
        result.zero_()
        return result
    if x.dtype == torch.complex64:
        cublas.cdotc(
            handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr()
        )
    elif x.dtype == torch.complex128:
        cublas.zdotc(
            handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr()
        )
    else:
        raise TypeError(f"Unsupported dtype for dotc: {x.dtype}")
    return result


def gems_dotc_wrapper(x, y, result, n=None, incx=1, incy=1, handle=None):
    if x.dtype == torch.complex64:
        flag_blas.ops.cdotc(n, x, incx, y, incy, result)
    elif x.dtype == torch.complex128:
        flag_blas.ops.zdotc(n, x, incx, y, incy, result)
    else:
        raise TypeError(f"Unsupported dtype for dotc: {x.dtype}")
    return result


class DotcBenchmark(Benchmark):
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

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]

    def validate_results(self, reference_result, blas_result, dtype, reduce_dim=1):
        if dtype == torch.complex64:
            rtol = 1e-5
            atol = max(1e-5, 1e-5 * (max(reduce_dim, 1) ** 0.5))
            tolerance_desc = "rtol=1e-5,atol=max(1e-5,sqrt(n)*1e-5)"
        else:
            rtol = 1e-12
            atol = max(1e-12, 1e-12 * (max(reduce_dim, 1) ** 0.5))
            tolerance_desc = "rtol=1e-12,atol=max(1e-12,sqrt(n)*1e-12)"

        try:
            torch.testing.assert_close(
                blas_result, reference_result, rtol=rtol, atol=atol
            )
        except AssertionError as e:
            ref_cpu = reference_result.cpu()
            res_cpu = blas_result.cpu()
            max_abs_diff = torch.max(torch.abs(ref_cpu - res_cpu))
            max_rel_diff = torch.max(
                torch.abs((ref_cpu - res_cpu) / (torch.abs(ref_cpu) + 1e-9))
            )
            raise AssertionError(
                f"Results differ beyond rtol={rtol}, atol={atol} "
                f"for dtype {dtype} reduce_dim={reduce_dim}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {ref_cpu.shape}"
            ) from e
        return {(str(dtype), tolerance_desc)}


@pytest.mark.dotc
def test_perf_cdotc():
    bench = DotcBenchmark(
        op_name="cdotc",
        torch_op=cublas_dotc,
        gems_op=gems_dotc_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dotc
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


@pytest.mark.dotc
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


@pytest.mark.dotc
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

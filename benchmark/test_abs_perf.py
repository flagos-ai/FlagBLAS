from typing import Generator

import pytest
import torch

import flag_blas

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


def torch_sabs(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(torch.abs(x[::incx][:n]))
    return out


def torch_dabs(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(torch.abs(x[::incx][:n]))
    return out


def torch_cabs(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(torch.abs(x[::incx][:n]))
    return out


def torch_zabs(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(torch.abs(x[::incx][:n]))
    return out


def gems_sabs_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.sabs(n, x, incx, out, incy)
    return out


def gems_dabs_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.dabs(n, x, incx, out, incy)
    return out


def gems_cabs_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.cabs(n, x, incx, out, incy)
    return out


def gems_zabs_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.zabs(n, x, incx, out, incy)
    return out


class AbsBenchmark(Benchmark):

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
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp.numel()

            if cur_dtype in [torch.complex64, torch.complex128]:
                out_dtype = (
                    torch.float32 if cur_dtype == torch.complex64 else torch.float64
                )
            else:
                out_dtype = cur_dtype

            out = torch.empty(n, dtype=out_dtype, device=self.device)

            if n > 0:
                yield inp, {"n": n, "incx": 1, "incy": 1, "out": out}

    def get_gbps(self, args, latency):
        inp = args[0]

        if inp.dtype == torch.complex64:
            out_element_size = 4
        elif inp.dtype == torch.complex128:
            out_element_size = 8
        else:
            out_element_size = inp.element_size()

        io_amount = inp.numel() * (inp.element_size() + out_element_size)
        return io_amount * 1e-9 / (latency * 1e-3)


class AbsStrideBenchmark(Benchmark):

    def __init__(self, *args, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy

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
        for shape in self.shapes:
            n = shape[0]
            inp = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)

            if cur_dtype in [torch.complex64, torch.complex128]:
                out_dtype = (
                    torch.float32 if cur_dtype == torch.complex64 else torch.float64
                )
            else:
                out_dtype = cur_dtype

            out = torch.empty(n * self.incy, dtype=out_dtype, device=self.device)

            if n > 0:
                yield inp, {
                    "incx": self.incx,
                    "incy": self.incy,
                    "n": n,
                    "out": out,
                }

    def get_gbps(self, args, latency):
        inp = args[0]
        n = (inp.numel() + self.incx - 1) // self.incx

        if inp.dtype == torch.complex64:
            out_element_size = 4
        elif inp.dtype == torch.complex128:
            out_element_size = 8
        else:
            out_element_size = inp.element_size()

        io_amount = n * (inp.element_size() + out_element_size)
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.abs
def test_perf_sabs():
    bench = AbsBenchmark(
        op_name="sabs",
        torch_op=torch_sabs,
        gems_op=gems_sabs_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.abs
def test_perf_dabs():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AbsBenchmark(
        op_name="dabs",
        torch_op=torch_dabs,
        gems_op=gems_dabs_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.abs
def test_perf_cabs():
    bench = AbsBenchmark(
        op_name="cabs",
        torch_op=torch_cabs,
        gems_op=gems_cabs_wrapper,
        dtypes=[torch.complex64],
    )
    bench.run()


@pytest.mark.abs
def test_perf_zabs():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AbsBenchmark(
        op_name="zabs",
        torch_op=torch_zabs,
        gems_op=gems_zabs_wrapper,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.abs
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_sabs_stride(incx, incy):
    bench = AbsStrideBenchmark(
        op_name=f"sabs_stride_incx{incx}_incy{incy}",
        torch_op=torch_sabs,
        gems_op=gems_sabs_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.abs
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_dabs_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AbsStrideBenchmark(
        op_name=f"dabs_stride_incx{incx}_incy{incy}",
        torch_op=torch_dabs,
        gems_op=gems_dabs_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.abs
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_cabs_stride(incx, incy):
    bench = AbsStrideBenchmark(
        op_name=f"cabs_stride_incx{incx}_incy{incy}",
        torch_op=torch_cabs,
        gems_op=gems_cabs_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.abs
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_zabs_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AbsStrideBenchmark(
        op_name=f"zabs_stride_incx{incx}_incy{incy}",
        torch_op=torch_zabs,
        gems_op=gems_zabs_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
    )
    bench.run()
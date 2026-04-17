from typing import Generator

import pytest
import torch

import flag_blas

from benchmark.performance_utils import Benchmark


def torch_scopy(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(x[::incx][:n])
    return out


def torch_dcopy(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(x[::incx][:n])
    return out


def torch_ccopy(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(x[::incx][:n])
    return out


def torch_zcopy(x, incx=1, incy=1, n=None, out=None):
    out[::incy][:n].copy_(x[::incx][:n])
    return out


def gems_scopy_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.scopy(n, x, incx, out, incy)
    return out


def gems_dcopy_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.dcopy(n, x, incx, out, incy)
    return out


def gems_ccopy_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.ccopy(n, x, incx, out, incy)
    return out


def gems_zcopy_wrapper(x, incx=1, incy=1, n=None, out=None):
    flag_blas.ops.zcopy(n, x, incx, out, incy)
    return out


class CopyBenchmark(Benchmark):

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
            out = torch.empty_like(inp)

            if n > 0:
                yield inp, {"n": n, "incx": 1, "incy": 1, "out": out}

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = inp.numel() * inp.element_size() * 2
        return io_amount * 1e-9 / (latency * 1e-3)


class CopyStrideBenchmark(Benchmark):

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
            out = torch.empty(n * self.incy, dtype=cur_dtype, device=self.device)

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
        io_amount = n * inp.element_size() * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.copy
def test_perf_scopy():
    bench = CopyBenchmark(
        op_name="scopy",
        torch_op=torch_scopy,
        gems_op=gems_scopy_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.copy
def test_perf_dcopy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyBenchmark(
        op_name="dcopy",
        torch_op=torch_dcopy,
        gems_op=gems_dcopy_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.copy
def test_perf_ccopy():
    bench = CopyBenchmark(
        op_name="ccopy",
        torch_op=torch_ccopy,
        gems_op=gems_ccopy_wrapper,
        dtypes=[torch.complex64],
    )
    bench.run()


@pytest.mark.copy
def test_perf_zcopy():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyBenchmark(
        op_name="zcopy",
        torch_op=torch_zcopy,
        gems_op=gems_zcopy_wrapper,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_scopy_stride(incx, incy):
    bench = CopyStrideBenchmark(
        op_name=f"scopy_stride_incx{incx}_incy{incy}",
        torch_op=torch_scopy,
        gems_op=gems_scopy_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_dcopy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyStrideBenchmark(
        op_name=f"dcopy_stride_incx{incx}_incy{incy}",
        torch_op=torch_dcopy,
        gems_op=gems_dcopy_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_ccopy_stride(incx, incy):
    bench = CopyStrideBenchmark(
        op_name=f"ccopy_stride_incx{incx}_incy{incy}",
        torch_op=torch_ccopy,
        gems_op=gems_ccopy_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx", [2, 3, 5])
@pytest.mark.parametrize("incy", [1, 2, 3])
def test_perf_zcopy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyStrideBenchmark(
        op_name=f"zcopy_stride_incx{incx}_incy{incy}",
        torch_op=torch_zcopy,
        gems_op=gems_zcopy_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
    )
    bench.run()
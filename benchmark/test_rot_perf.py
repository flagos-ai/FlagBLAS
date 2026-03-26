from typing import Generator

import pytest
import torch

import flag_blas

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


def torch_rot_real(x, y, c, s, incx=1, incy=1, n=None):
    """Optimized PyTorch implementation of srot/drot as baseline."""
    x_view = x[: n * incx : incx]
    y_view = y[: n * incy : incy]
    new_x = c * x_view + s * y_view
    new_y = c * y_view - s * x_view
    x_view.copy_(new_x)
    y_view.copy_(new_y)
    return x, y


def torch_rot_complex(x, y, c, s, incx=1, incy=1, n=None):
    """Optimized PyTorch implementation of crot/zrot as baseline."""
    x_view = x[: n * incx : incx]
    y_view = y[: n * incy : incy]
    s_conj = s.conjugate() if isinstance(s, complex) else s
    new_x = c * x_view + s * y_view
    new_y = c * y_view - s_conj * x_view
    x_view.copy_(new_x)
    y_view.copy_(new_y)
    return x, y


def gems_srot_wrapper(x, y, c, s, incx=1, incy=1, n=None):
    flag_blas.ops.srot(n, x, incx, y, incy, c, s)
    return x, y


def gems_drot_wrapper(x, y, c, s, incx=1, incy=1, n=None):
    flag_blas.ops.drot(n, x, incx, y, incy, c, s)
    return x, y


def gems_crot_wrapper(x, y, c, s, incx=1, incy=1, n=None):
    flag_blas.ops.crot(n, x, incx, y, incy, c, s)
    return x, y


def gems_zrot_wrapper(x, y, c, s, incx=1, incy=1, n=None):
    flag_blas.ops.zrot(n, x, incx, y, incy, c, s)
    return x, y


class RotBenchmark(Benchmark):

    def __init__(self, *args, c=0.8, s=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.s = s

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
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp1.numel()
            if n > 0:
                yield inp1, inp2, {"c": self.c, "s": self.s, "n": n}

    def get_gbps(self, args, latency):
        inp1, inp2 = args[0], args[1]
        # rot reads x, y and writes x, y
        io_amount = 2 * shape_utils.size_in_bytes(inp1) + 2 * shape_utils.size_in_bytes(
            inp2
        )
        return io_amount * 1e-9 / (latency * 1e-3)


class RotStrideBenchmark(Benchmark):

    def __init__(self, *args, incx=1, incy=1, c=0.8, s=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy
        self.c = c
        self.s = s

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
            inp1 = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            if n > 0:
                yield inp1, inp2, {
                    "c": self.c,
                    "s": self.s,
                    "incx": self.incx,
                    "incy": self.incy,
                    "n": n,
                }

    def get_gbps(self, args, latency):
        inp1, inp2 = args[0], args[1]
        n = (inp1.numel() + self.incx - 1) // self.incx
        element_size = inp1.element_size()
        # rot reads n elements from x, y and writes n elements to x, y
        io_amount = 4 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.rot
def test_perf_srot():
    bench = RotBenchmark(
        op_name="srot",
        torch_op=torch_rot_real,
        gems_op=gems_srot_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.rot
def test_perf_drot():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotBenchmark(
        op_name="drot",
        torch_op=torch_rot_real,
        gems_op=gems_drot_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.rot
def test_perf_crot():
    bench = RotBenchmark(
        op_name="crot",
        torch_op=torch_rot_complex,
        gems_op=gems_crot_wrapper,
        dtypes=[torch.complex64],
        c=0.8,
        s=0.36 + 0.48j,
    )
    bench.run()


@pytest.mark.rot
def test_perf_zrot():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotBenchmark(
        op_name="zrot",
        torch_op=torch_rot_complex,
        gems_op=gems_zrot_wrapper,
        dtypes=[torch.complex128],
        c=0.8,
        s=0.36 + 0.48j,
    )
    bench.run()


@pytest.mark.rot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_srot_stride(incx, incy):
    bench = RotStrideBenchmark(
        op_name=f"srot_stride_incx{incx}_incy{incy}",
        torch_op=torch_rot_real,
        gems_op=gems_srot_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.rot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_drot_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotStrideBenchmark(
        op_name=f"drot_stride_incx{incx}_incy{incy}",
        torch_op=torch_rot_real,
        gems_op=gems_drot_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.rot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_crot_stride(incx, incy):
    bench = RotStrideBenchmark(
        op_name=f"crot_stride_incx{incx}_incy{incy}",
        torch_op=torch_rot_complex,
        gems_op=gems_crot_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
        c=0.8,
        s=0.36 + 0.48j,
    )
    bench.run()


@pytest.mark.rot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_zrot_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotStrideBenchmark(
        op_name=f"zrot_stride_incx{incx}_incy{incy}",
        torch_op=torch_rot_complex,
        gems_op=gems_zrot_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
        c=0.8,
        s=0.36 + 0.48j,
    )
    bench.run()

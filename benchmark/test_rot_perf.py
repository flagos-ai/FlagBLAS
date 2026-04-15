import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch
import cupy as cp

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


def load_cublas():
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)

    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Cannot find libcublas.so in the system")


_cublas = load_cublas()


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def cublas_rot_reference(x, y, c, s, incx=1, incy=1, n=None):
    if n is None:
        n = x.numel() // incx
    if n == 0:
        return x, y

    handle = cp.cuda.device.get_cublas_handle()
    dtype = x.dtype

    if dtype == torch.float32:
        func = _cublas.cublasSrot_v2
        c_val = ctypes.c_float(c)
        s_val = ctypes.c_float(s)
    elif dtype == torch.float64:
        func = _cublas.cublasDrot_v2
        c_val = ctypes.c_double(c)
        s_val = ctypes.c_double(s)
    elif dtype == torch.complex64:
        func = _cublas.cublasCrot_v2
        c_val = ctypes.c_float(c)
        s_val = cuComplex(s.real, s.imag)
    elif dtype == torch.complex128:
        func = _cublas.cublasZrot_v2
        c_val = ctypes.c_double(c)
        s_val = cuDoubleComplex(s.real, s.imag)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(n),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.byref(c_val),
        ctypes.byref(s_val),
    )

    if status != 0:
        raise RuntimeError(f"cuBLAS rot execution failed, error code: {status}")

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
        inp1 = args[0]
        n = (inp1.numel() + self.incx - 1) // self.incx
        element_size = inp1.element_size()
        io_amount = 4 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.rot
def test_perf_srot():
    bench = RotBenchmark(
        op_name="srot",
        torch_op=cublas_rot_reference,
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
        torch_op=cublas_rot_reference,
        gems_op=gems_drot_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.rot
def test_perf_crot():
    bench = RotBenchmark(
        op_name="crot",
        torch_op=cublas_rot_reference,
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
        torch_op=cublas_rot_reference,
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
        torch_op=cublas_rot_reference,
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
        torch_op=cublas_rot_reference,
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
        torch_op=cublas_rot_reference,
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
        torch_op=cublas_rot_reference,
        gems_op=gems_zrot_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
        c=0.8,
        s=0.36 + 0.48j,
    )
    bench.run()

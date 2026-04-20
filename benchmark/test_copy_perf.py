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


def cublas_copy_reference(x, y, incx=1, incy=1, n=None):
    if n is None:
        n = x.numel() // incx
    if n == 0:
        return x, y

    handle = cp.cuda.device.get_cublas_handle()
    dtype = x.dtype

    if dtype == torch.float32:
        func = _cublas.cublasScopy_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDcopy_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCcopy_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZcopy_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(n),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )

    if status != 0:
        raise RuntimeError(f"cuBLAS copy execution failed, error code: {status}")

    return x, y


def gems_scopy_wrapper(x, y, incx=1, incy=1, n=None):
    flag_blas.ops.scopy(n, x, incx, y, incy)
    return x, y


def gems_dcopy_wrapper(x, y, incx=1, incy=1, n=None):
    flag_blas.ops.dcopy(n, x, incx, y, incy)
    return x, y


def gems_ccopy_wrapper(x, y, incx=1, incy=1, n=None):
    flag_blas.ops.ccopy(n, x, incx, y, incy)
    return x, y


def gems_zcopy_wrapper(x, y, incx=1, incy=1, n=None):
    flag_blas.ops.zcopy(n, x, incx, y, incy)
    return x, y


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
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp2 = torch.empty(shape, dtype=cur_dtype, device=self.device)
            n = inp1.numel()
            if n > 0:
                yield inp1, inp2, {"n": n}

    def get_gbps(self, args, latency):
        inp1, inp2 = args[0], args[1]
        io_amount = shape_utils.size_in_bytes(inp1) + shape_utils.size_in_bytes(inp2)
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
            inp1 = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            inp2 = torch.empty(n * self.incy, dtype=cur_dtype, device=self.device)
            if n > 0:
                yield inp1, inp2, {
                    "incx": self.incx,
                    "incy": self.incy,
                    "n": n,
                }

    def get_gbps(self, args, latency):
        inp1 = args[0]
        n = (inp1.numel() + self.incx - 1) // self.incx
        element_size = inp1.element_size()
        io_amount = 2 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.copy
def test_perf_scopy():
    bench = CopyBenchmark(
        op_name="scopy",
        torch_op=cublas_copy_reference,
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
        torch_op=cublas_copy_reference,
        gems_op=gems_dcopy_wrapper,
        dtypes=[torch.float64],
    )
    bench.run()


@pytest.mark.copy
def test_perf_ccopy():
    bench = CopyBenchmark(
        op_name="ccopy",
        torch_op=cublas_copy_reference,
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
        torch_op=cublas_copy_reference,
        gems_op=gems_zcopy_wrapper,
        dtypes=[torch.complex128],
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_scopy_stride(incx, incy):
    bench = CopyStrideBenchmark(
        op_name=f"scopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        gems_op=gems_scopy_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_dcopy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyStrideBenchmark(
        op_name=f"dcopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        gems_op=gems_dcopy_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_ccopy_stride(incx, incy):
    bench = CopyStrideBenchmark(
        op_name=f"ccopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        gems_op=gems_ccopy_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
    )
    bench.run()


@pytest.mark.copy
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_zcopy_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CopyStrideBenchmark(
        op_name=f"zcopy_stride_incx{incx}_incy{incy}",
        torch_op=cublas_copy_reference,
        gems_op=gems_zcopy_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
    )
    bench.run()

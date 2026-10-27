import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from tests.test_rotg import (
    ROTG_COMPLEX_CASES_BY_DTYPE,
    ROTG_REAL_CASES,
    _scipy_rotg_reference,
)

IS_HYGON = flag_blas.vendor_name == "hygon"
if not IS_HYGON:
    import cupy as cp
if IS_HYGON:
    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library

CUBLAS_POINTER_MODE_DEVICE = 1


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


if not IS_HYGON:
    _cublas = load_cublas()
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int

    for _rotg_name in (
        "cublasSrotg_v2",
        "cublasDrotg_v2",
        "cublasCrotg_v2",
        "cublasZrotg_v2",
    ):
        _rotg_func = getattr(_cublas, _rotg_name)
        _rotg_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _rotg_func.restype = ctypes.c_int


def create_cublas_handle():
    handle = ctypes.c_void_p(cp.cuda.device.get_cublas_handle())
    status = _cublas.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE)
    if status != 0:
        raise RuntimeError(f"cuBLAS pointer mode setup failed, error code: {status}")
    return handle


def as_cublas_handle(handle):
    if isinstance(handle, ctypes.c_void_p):
        return handle
    return ctypes.c_void_p(handle)


def gems_rotg(a, b, c, s, handle=None, c_func=None):
    if a.dtype == torch.float32:
        flag_blas.srotg(a, b, c, s)
    elif a.dtype == torch.float64:
        flag_blas.drotg(a, b, c, s)
    elif a.dtype == torch.complex64:
        flag_blas.crotg(a, b, c, s)
    elif a.dtype == torch.complex128:
        flag_blas.zrotg(a, b, c, s)
    else:
        raise TypeError(f"Unsupported dtype for rotg: {a.dtype}")
    return a, b, c, s


if IS_HYGON:

    def _resolve_hipblas_rotg(dtype):
        symbol = {
            torch.float32: "hipblasSrotg",
            torch.float64: "hipblasDrotg",
            torch.complex64: "hipblasCrotg_v2",
            torch.complex128: "hipblasZrotg_v2",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [ctypes.c_void_p] * 5
        function.restype = ctypes.c_int
        return function

    def hipblas_rotg_baseline(a, b, c, s, handle=None, c_func=None):
        check_hipblas_status(
            c_func(handle, a.data_ptr(), b.data_ptr(), c.data_ptr(), s.data_ptr()),
            "hipBLAS ROTG",
        )
        return a, b, c, s

    cublas_rotg = hipblas_rotg_baseline

else:

    def cublas_rotg(a, b, c, s, handle=None, c_func=None):
        status = c_func(
            as_cublas_handle(handle),
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_void_p(s.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(f"cuBLAS rotg execution failed, error code: {status}")
        return a, b, c, s


def _cpu_real_rotg_reference(a, b, c, s, handle=None, c_func=None):
    r, z, c_value, s_value = _scipy_rotg_reference(a.dtype, a.item(), b.item())
    if c_value == 0.0 and r != 0.0:
        z = 1.0
    for tensor, value in ((a, r), (b, z), (c, c_value), (s, s_value)):
        tensor.fill_(value)
    return a, b, c, s


class RotgBenchmark(Benchmark):
    def run_correctness_check(self):
        if flag_blas.vendor_name != "thead" or self.op_name not in {"srotg", "drotg"}:
            return super().run_correctness_check()
        # PPU's vendor ROTG returns a different z when a == 0 and b != 0.
        # Check BLAS semantics on CPU, then restore the vendor function for timing.
        reference = self.torch_op
        reference_name = getattr(self, "correctness_reference", "cuBLAS")
        self.torch_op = _cpu_real_rotg_reference
        self.correctness_reference = "CPU BLAS"
        try:
            return super().run_correctness_check()
        finally:
            self.torch_op = reference
            self.correctness_reference = reference_name

    def set_more_shapes(self):
        dtype = self.dtypes[0]
        if dtype.is_complex:
            return [
                (a.real, a.imag, b.real, b.imag)
                for a, b in ROTG_COMPLEX_CASES_BY_DTYPE[dtype]
            ]
        return ROTG_REAL_CASES

    def record_shapes(self, *args, **kwargs):
        result = super().record_shapes(*args, **kwargs)
        # Scalar values describe the initial input, before in-place timing.
        result[1]["scalar_case"] = self.scalar_case
        return result

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_rotg(cur_dtype)
        else:
            handle = create_cublas_handle()
            prefix = {
                torch.float32: "S",
                torch.float64: "D",
                torch.complex64: "C",
                torch.complex128: "Z",
            }[cur_dtype]
            c_func = getattr(_cublas, f"cublas{prefix}rotg_v2")
        for shape in self.shapes:
            if cur_dtype.is_complex:
                if len(shape) != 4:
                    raise ValueError(
                        "Complex ROTG shapes must contain a.real, a.imag, b.real, b.imag"
                    )
                a_val = complex(shape[0], shape[1])
                b_val = complex(shape[2], shape[3])
            else:
                if len(shape) != 2:
                    raise ValueError("Real ROTG shapes must contain a, b")
                a_val, b_val = shape
            a = torch.tensor([a_val], dtype=cur_dtype, device=self.device)
            b = torch.tensor([b_val], dtype=cur_dtype, device=self.device)
            real_dtype = (
                torch.float32
                if cur_dtype in (torch.float32, torch.complex64)
                else torch.float64
            )
            c = torch.zeros(1, dtype=real_dtype, device=self.device)
            s = torch.zeros(1, dtype=cur_dtype, device=self.device)
            if IS_HYGON:
                _, handle = get_l1_hipblas_context(a)
            kwargs = {"handle": handle, "c_func": c_func}
            self.scalar_case = {"a": str(a_val), "b": str(b_val)}
            yield a, b, c, s, kwargs


if IS_HYGON:
    RotgBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.srotg
def test_perf_srotg():
    run_correctness_then_benchmark(
        RotgBenchmark(
            "srotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.float32],
        )
    )


@pytest.mark.drotg
def test_perf_drotg():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotgBenchmark(
            "drotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.float64],
        )
    )


@pytest.mark.crotg
def test_perf_crotg():
    run_correctness_then_benchmark(
        RotgBenchmark(
            "crotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.complex64],
        )
    )


@pytest.mark.zrotg
def test_perf_zrotg():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotgBenchmark(
            "zrotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.complex128],
        )
    )

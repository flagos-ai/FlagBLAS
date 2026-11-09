import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from tests.test_rotmg import ROTMG_CASES

IS_HYGON = flag_blas.vendor_name == "hygon"
if flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp
elif not IS_HYGON and flag_blas.vendor_name != "ascend":
    import cupy as cp
if IS_HYGON:
    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library


CUBLAS_POINTER_MODE_DEVICE = 1


def load_cublas():
    if flag_blas.vendor_name == "mthreads":
        from benchmark.mublas_compat import load_mublas

        return load_mublas()
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


if not IS_HYGON and flag_blas.vendor_name != "ascend":
    _cublas = load_cublas()
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int


def normalize_rotmg_param(param):
    flag = param[0].item()
    if flag == -2.0:
        param[1:].zero_()
    elif flag == 0.0:
        param[1].zero_()
        param[4].zero_()
    elif flag == 1.0:
        param[2].zero_()
        param[3].zero_()


if not IS_HYGON and flag_blas.vendor_name != "ascend":
    for _rotmg_name in ("cublasSrotmg_v2", "cublasDrotmg_v2"):
        _rotmg_func = getattr(_cublas, _rotmg_name)
        _rotmg_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _rotmg_func.restype = ctypes.c_int


def create_cublas_handle():
    handle = ctypes.c_void_p(cp.cuda.device.get_cublas_handle())
    status = _cublas.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE)
    if status != 0:
        raise RuntimeError(f"cuBLAS pointer mode setup failed, error code: {status}")
    return handle


def gems_rotmg_wrapper(d1, d2, x1, y1, param, handle=None, c_func=None):
    if d1.dtype == torch.float32:
        flag_blas.srotmg(d1, d2, x1, y1, param)
    elif d1.dtype == torch.float64:
        flag_blas.drotmg(d1, d2, x1, y1, param)
    else:
        raise TypeError(f"Unsupported dtype for rotmg: {d1.dtype}")
    return d1, d2, x1, y1, param


if IS_HYGON:

    def _resolve_hipblas_rotmg(dtype):
        symbol = {
            torch.float32: "hipblasSrotmg",
            torch.float64: "hipblasDrotmg",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [ctypes.c_void_p] * 6
        function.restype = ctypes.c_int
        return function

    def hipblas_rotmg_baseline(d1, d2, x1, y1, param, handle=None, c_func=None):
        check_hipblas_status(
            c_func(
                handle,
                d1.data_ptr(),
                d2.data_ptr(),
                x1.data_ptr(),
                y1.data_ptr(),
                param.data_ptr(),
            ),
            "hipBLAS ROTMG",
        )
        return d1, d2, x1, y1, param

    cublas_rotmg = hipblas_rotmg_baseline

else:

    def cublas_rotmg(d1, d2, x1, y1, param, handle=None, c_func=None):
        status = c_func(
            handle,
            ctypes.c_void_p(d1.data_ptr()),
            ctypes.c_void_p(d2.data_ptr()),
            ctypes.c_void_p(x1.data_ptr()),
            ctypes.c_void_p(y1.data_ptr()),
            ctypes.c_void_p(param.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(f"cuBLAS rotmg execution failed, error code: {status}")
        return d1, d2, x1, y1, param


class RotmgBenchmark(Benchmark):
    def set_more_shapes(self):
        return ROTMG_CASES

    def validate_results(self, reference_result, blas_result, dtype, reduce_dim=1):
        # Inactive BLAS parameter entries are undefined; normalize outside timing.
        normalize_rotmg_param(reference_result[-1])
        normalize_rotmg_param(blas_result[-1])
        return super().validate_results(
            reference_result, blas_result, dtype, reduce_dim=reduce_dim
        )

    def record_shapes(self, *args, **kwargs):
        result = super().record_shapes(*args, **kwargs)
        # Scalar values describe the initial input, before in-place timing.
        result[1]["scalar_case"] = self.scalar_case
        return result

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_rotmg(cur_dtype)
        else:
            handle = create_cublas_handle()
            prefix = {
                torch.float32: "S",
                torch.float64: "D",
                torch.complex64: "C",
                torch.complex128: "Z",
            }[cur_dtype]
            c_func = getattr(_cublas, f"cublas{prefix}rotmg_v2")
        for shape in self.shapes:
            if len(shape) != 4:
                raise ValueError("ROTMG shapes must contain d1, d2, x1, y1")
            d1_val, d2_val, x1_val, y1_val = shape
            d1 = torch.tensor([d1_val], dtype=cur_dtype, device=self.device)
            d2 = torch.tensor([d2_val], dtype=cur_dtype, device=self.device)
            x1 = torch.tensor([x1_val], dtype=cur_dtype, device=self.device)
            y1 = torch.tensor([y1_val], dtype=cur_dtype, device=self.device)
            param = torch.zeros(5, dtype=cur_dtype, device=self.device)
            if IS_HYGON:
                _, handle = get_l1_hipblas_context(d1)
            kwargs = {"handle": handle, "c_func": c_func}
            self.scalar_case = {"d1": d1_val, "d2": d2_val, "x1": x1_val, "y1": y1_val}
            yield d1, d2, x1, y1, param, kwargs


if IS_HYGON:
    RotmgBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.srotmg
def test_perf_srotmg():
    run_correctness_then_benchmark(
        RotmgBenchmark(
            op_name="srotmg",
            torch_op=cublas_rotmg,
            blas_op=gems_rotmg_wrapper,
            dtypes=[torch.float32],
        )
    )


@pytest.mark.drotmg
def test_perf_drotmg():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotmgBenchmark(
            op_name="drotmg",
            torch_op=cublas_rotmg,
            blas_op=gems_rotmg_wrapper,
            dtypes=[torch.float64],
        )
    )

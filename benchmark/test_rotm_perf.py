import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.attri_util import L1_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark

IS_HYGON = flag_blas.vendor_name == "hygon"
if flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp
elif not IS_HYGON:
    import cupy as cp
if IS_HYGON:
    from tests.hipblas_reference import (
        check_hipblas_status,
        get_hipblas_context,
        get_hipblas_library,
    )

PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]
ROTM_FLAGS = [-2.0, -1.0, 0.0, 1.0]


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


if not IS_HYGON:
    _cublas = load_cublas()
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int

    for _rotm_name in ("cublasSrotm_v2", "cublasDrotm_v2"):
        _rotm_func = getattr(_cublas, _rotm_name)
        _rotm_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _rotm_func.restype = ctypes.c_int


def create_cublas_handle():
    handle = ctypes.c_void_p(cp.cuda.device.get_cublas_handle())
    status = _cublas.cublasSetPointerMode_v2(handle, 0)
    if status != 0:
        raise RuntimeError(f"cuBLAS pointer mode setup failed, error code: {status}")
    return handle


def host_param(dtype, values):
    if dtype == torch.float32:
        return (ctypes.c_float * 5)(*values)
    return (ctypes.c_double * 5)(*values)


def gems_rotm_wrapper(
    x, y, param, n=None, incx=1, incy=1, handle=None, param_host=None, c_func=None
):
    if x.dtype == torch.float32:
        flag_blas.srotm(n, x, incx, y, incy, param, param_host=param_host)
    elif x.dtype == torch.float64:
        flag_blas.drotm(n, x, incx, y, incy, param, param_host=param_host)
    else:
        raise TypeError(f"Unsupported dtype for rotm: {x.dtype}")
    return x, y


if IS_HYGON:

    def _resolve_hipblas_rotm(dtype):
        symbol = {
            torch.float32: "hipblasSrotm",
            torch.float64: "hipblasDrotm",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_rotm_baseline(
        x, y, param, n=None, incx=1, incy=1, handle=None, param_host=None, c_func=None
    ):
        if n <= 0:
            return x, y
        check_hipblas_status(
            c_func(
                handle,
                n,
                x.data_ptr(),
                incx,
                y.data_ptr(),
                incy,
                ctypes.cast(param_host, ctypes.c_void_p),
            ),
            "hipBLAS ROTM",
        )
        return x, y

    _common_gems_rotm_wrapper = gems_rotm_wrapper

    def _hygon_gems_rotm_wrapper(
        x, y, param, n=None, incx=1, incy=1, handle=None, param_host=None, c_func=None
    ):
        return _common_gems_rotm_wrapper(
            x, y, param, n=n, incx=incx, incy=incy, handle=handle, param_host=param_host
        )

    cublas_rotm = hipblas_rotm_baseline
    gems_rotm_wrapper = _hygon_gems_rotm_wrapper

else:

    def cublas_rotm(
        x, y, param, n=None, incx=1, incy=1, handle=None, param_host=None, c_func=None
    ):
        if n is None:
            n = min(x.numel() // incx, y.numel() // incy)
        if n <= 0:
            return x, y

        status = c_func(
            handle,
            ctypes.c_int(n),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int(incx),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_int(incy),
            ctypes.cast(param_host, ctypes.c_void_p),
        )
        if status != 0:
            raise RuntimeError(f"cuBLAS rotm execution failed, error code: {status}")
        return x, y


class RotmBenchmark(Benchmark):
    def __init__(self, *args, incx=1, incy=1, flag=-1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy
        self.flag = flag

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_rotm(cur_dtype)
        else:
            handle = create_cublas_handle()
            prefix = {
                torch.float32: "S",
                torch.float64: "D",
                torch.complex64: "C",
                torch.complex128: "Z",
            }[cur_dtype]
            c_func = getattr(_cublas, f"cublas{prefix}rotm_v2")
        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            param_values = [self.flag, 0.8, 0.2, -0.3, 0.7]
            param = torch.tensor(
                param_values,
                dtype=cur_dtype,
                device=self.device,
            )
            if IS_HYGON:
                _, handle = get_hipblas_context(x)
            kwargs = {
                "n": n,
                "incx": self.incx,
                "incy": self.incy,
                "handle": handle,
                "param_host": host_param(cur_dtype, param_values),
            }
            kwargs["c_func"] = c_func
            yield x, y, param, kwargs

    def get_gbps(self, args, latency):
        # Count vector traffic; scalar coefficients are parameters, as in L2.
        if self.flag == -2.0:
            return 0.0
        x = args[0]
        n = x.numel() // self.incx
        element_size = x.element_size()
        io_amount = 4 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        if self.flag == -2.0:
            return 0
        # A full 2x2 transform costs six FLOPs; implicit unit entries cost four.
        return (6 if self.flag < 0.0 else 4) * n


class RotmStrideBenchmark(RotmBenchmark):
    def set_more_shapes(self):
        return L1_STRIDE_SHAPES


if IS_HYGON:
    RotmBenchmark.correctness_reference = "hipBLAS"
    RotmStrideBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.srotm
@pytest.mark.parametrize("flag", ROTM_FLAGS)
def test_perf_srotm(flag):
    run_correctness_then_benchmark(
        RotmBenchmark(
            op_name=f"srotm_flag{flag}",
            torch_op=cublas_rotm,
            blas_op=gems_rotm_wrapper,
            dtypes=[torch.float32],
            flag=flag,
        )
    )


@pytest.mark.drotm
@pytest.mark.parametrize("flag", ROTM_FLAGS)
def test_perf_drotm(flag):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotmBenchmark(
            op_name=f"drotm_flag{flag}",
            torch_op=cublas_rotm,
            blas_op=gems_rotm_wrapper,
            dtypes=[torch.float64],
            flag=flag,
        )
    )


@pytest.mark.srotm
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_srotm_stride(incx, incy):
    run_correctness_then_benchmark(
        RotmStrideBenchmark(
            op_name=f"srotm_stride_incx{incx}_incy{incy}",
            torch_op=cublas_rotm,
            blas_op=gems_rotm_wrapper,
            dtypes=[torch.float32],
            incx=incx,
            incy=incy,
        )
    )


@pytest.mark.drotm
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_drotm_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotmStrideBenchmark(
            op_name=f"drotm_stride_incx{incx}_incy{incy}",
            torch_op=cublas_rotm,
            blas_op=gems_rotm_wrapper,
            dtypes=[torch.float64],
            incx=incx,
            incy=incy,
        )
    )

import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils

HEMV_SIZES = [
    128,
    192,
    256,
    384,
    512,
    768,
    1023,
    1024,
    1536,
    2048,
    3072,
    4096,
    4097,
    6144,
    12288,
    16384,
]


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
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = load_cublas()


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


_CUBLAS_HEMV_FUNCS = {
    torch.complex64: (_cublas.cublasChemv_v2, cuComplex),
    torch.complex128: (_cublas.cublasZhemv_v2, cuDoubleComplex),
}


def _make_scalar(ctor, value):
    return ctor(value.real, value.imag)


def cublas_hemv_baseline(
    A,
    x,
    y,
    uplo,
    n,
    alpha,
    lda,
    incx,
    beta,
    incy,
    handle,
    c_func,
    alpha_c,
    beta_c,
    **kwargs,
):
    if n == 0:
        return y

    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXhemv_v2 execution failed with error code: {status}")
    return y


def _gems_wrapper(op):
    def _impl(A, x, y, uplo, n, alpha, lda, incx, beta, incy, handle, **kwargs):
        op(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
        return y

    return _impl


gems_chemv_wrapper = _gems_wrapper(flag_blas.ops.chemv)
gems_zhemv_wrapper = _gems_wrapper(flag_blas.ops.zhemv)


def _generate_her_A(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    data = torch.randn(n, n, dtype=dtype, device=device)
    diag_real = data.diagonal().real.clone()
    data.diagonal().copy_(diag_real.to(dtype))
    A[:, :n] = data
    return A.contiguous()


class HemvBenchmark(Benchmark):

    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HEMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_HEMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func, ctor = _CUBLAS_HEMV_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, self.alpha)
        beta_c = _make_scalar(ctor, self.beta)

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = n
            A = _generate_her_A(n, lda, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)

            yield A, x, y.clone(), {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "lda": lda,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
                "beta_c": beta_c,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 8 * n * n

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        n = y.numel()
        a_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = (
            a_bytes + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.chemv
def test_perf_chemv():
    bench = HemvBenchmark(
        op_name="chemv",
        torch_op=cublas_hemv_baseline,
        gems_op=gems_chemv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.chemv
def test_perf_chemv_upper():
    bench = HemvBenchmark(
        op_name="chemv_upper",
        torch_op=cublas_hemv_baseline,
        gems_op=gems_chemv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.zhemv
def test_perf_zhemv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HemvBenchmark(
        op_name="zhemv",
        torch_op=cublas_hemv_baseline,
        gems_op=gems_zhemv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.zhemv
def test_perf_zhemv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HemvBenchmark(
        op_name="zhemv_upper",
        torch_op=cublas_hemv_baseline,
        gems_op=gems_zhemv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()

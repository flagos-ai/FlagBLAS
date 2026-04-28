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

SYMV_SIZES = [
    128,
    192,
    255,
    256,
    384,
    512,
    768,
    1023,
    1024,
    1536,
    2048,
    3072,
    4095,
    4096,
    6144,
    8192,
    9999,
    10000,
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


_CUBLAS_SYMV_FUNCS = {
    torch.float32: (_cublas.cublasSsymv_v2, ctypes.c_float, False),
    torch.float64: (_cublas.cublasDsymv_v2, ctypes.c_double, False),
    torch.complex64: (_cublas.cublasCsymv_v2, cuComplex, True),
    torch.complex128: (_cublas.cublasZsymv_v2, cuDoubleComplex, True),
}


def _make_scalar(ctor, is_complex, value):
    if is_complex:
        return ctor(value.real, value.imag)
    return ctor(value)


def cublas_symv_baseline(
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
        raise RuntimeError(f"cublasXsymv_v2 execution failed with error code: {status}")
    return y


def _gems_wrapper(op):
    def _impl(A, x, y, uplo, n, alpha, lda, incx, beta, incy, handle, **kwargs):
        op(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
        return y

    return _impl


gems_ssymv_wrapper = _gems_wrapper(flag_blas.ops.ssymv)
gems_dsymv_wrapper = _gems_wrapper(flag_blas.ops.dsymv)
gems_csymv_wrapper = _gems_wrapper(flag_blas.ops.csymv)
gems_zsymv_wrapper = _gems_wrapper(flag_blas.ops.zsymv)


def _generate_sym_A(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    if dtype.is_complex:
        A[:, :n] = torch.randn(n, n, dtype=dtype, device=device)
    else:
        A[:, :n] = torch.randn(n, n, dtype=dtype, device=device) * 0.1
    return A.contiguous()


class SymvBenchmark(Benchmark):

    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SYMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_SYMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func, ctor, is_complex = _CUBLAS_SYMV_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, is_complex, self.alpha)
        beta_c = _make_scalar(ctor, is_complex, self.beta)

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = n
            A = _generate_sym_A(n, lda, cur_dtype, self.device)
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
        nnz = n * n
        A = args[0]
        if A.dtype in [torch.complex64, torch.complex128]:
            return 8 * nnz
        return 2 * nnz

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        n = y.numel()
        a_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = (
            a_bytes + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.ssymv
def test_perf_ssymv():
    bench = SymvBenchmark(
        op_name="ssymv",
        torch_op=cublas_symv_baseline,
        gems_op=gems_ssymv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    bench.run()


@pytest.mark.ssymv
def test_perf_ssymv_upper():
    bench = SymvBenchmark(
        op_name="ssymv_upper",
        torch_op=cublas_symv_baseline,
        gems_op=gems_ssymv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    bench.run()


@pytest.mark.dsymv
def test_perf_dsymv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SymvBenchmark(
        op_name="dsymv",
        torch_op=cublas_symv_baseline,
        gems_op=gems_dsymv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    bench.run()


@pytest.mark.dsymv
def test_perf_dsymv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SymvBenchmark(
        op_name="dsymv_upper",
        torch_op=cublas_symv_baseline,
        gems_op=gems_dsymv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    bench.run()


@pytest.mark.csymv
def test_perf_csymv():
    bench = SymvBenchmark(
        op_name="csymv",
        torch_op=cublas_symv_baseline,
        gems_op=gems_csymv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.csymv
def test_perf_csymv_upper():
    bench = SymvBenchmark(
        op_name="csymv_upper",
        torch_op=cublas_symv_baseline,
        gems_op=gems_csymv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.zsymv
def test_perf_zsymv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SymvBenchmark(
        op_name="zsymv",
        torch_op=cublas_symv_baseline,
        gems_op=gems_zsymv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.zsymv
def test_perf_zsymv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SymvBenchmark(
        op_name="zsymv_upper",
        torch_op=cublas_symv_baseline,
        gems_op=gems_zsymv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()

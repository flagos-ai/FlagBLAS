import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

HBMV_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
]

HBMV_KS = [0, 1, 4, 16, 64, 128, 256]


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


_CUBLAS_HBMV_FUNCS = {
    torch.complex64: (_cublas.cublasChbmv_v2, cuComplex),
    torch.complex128: (_cublas.cublasZhbmv_v2, cuDoubleComplex),
}


def _make_scalar(ctor, value):
    return ctor(value.real, value.imag)


def cublas_hbmv_baseline(
    A,
    x,
    y,
    uplo,
    n,
    k,
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
        ctypes.c_int(k),
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
        raise RuntimeError(f"cublasXhbmv_v2 failed with status code: {status}")
    return y


def _gems_wrapper(op):
    def _impl(A, x, y, uplo, n, k, alpha, lda, incx, beta, incy, handle, **kwargs):
        op(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
        return y

    return _impl


gems_chbmv_wrapper = _gems_wrapper(flag_blas.ops.chbmv)
gems_zhbmv_wrapper = _gems_wrapper(flag_blas.ops.zhbmv)


def _generate_hermitian_banded(n, k, lda, uplo, dtype, device):
    # cuBLAS hbmv only reads in-band elements; out-of-band cells can be arbitrary.
    A = torch.randn((n, lda), dtype=dtype, device=device)
    diag_col = k if uplo == CUBLAS_FILL_MODE_UPPER else 0
    torch.view_as_real(A)[:, diag_col, 1].zero_()
    return A


def _band_nnz(n, k):
    if n <= 0:
        return 0
    k = min(k, n - 1)
    return n * (2 * k + 1) - k * (k + 1)


def _stored_band_nnz(n, k):
    if n <= 0:
        return 0
    if k >= n - 1:
        return n * (n + 1) // 2
    return (k + 1) * (k + 2) // 2 + (n - k - 1) * (k + 1)


class HbmvBenchmark(Benchmark):
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
        self.ks = HBMV_KS

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HBMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_HBMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func, ctor = _CUBLAS_HBMV_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, self.alpha)
        beta_c = _make_scalar(ctor, self.beta)

        seen = set()
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            for k_req in self.ks:
                k = min(k_req, max(0, n - 1))
                key = (n, k)
                if key in seen:
                    continue
                seen.add(key)
                lda = k + 1
                A = _generate_hermitian_banded(
                    n, k, lda, self.uplo, cur_dtype, self.device
                )
                x = torch.randn(n, dtype=cur_dtype, device=self.device)
                y = torch.randn(n, dtype=cur_dtype, device=self.device)

                yield A, x, y, {
                    "uplo": self.uplo,
                    "n": n,
                    "k": k,
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
        k = kwargs.get("k", 0)
        nnz = _band_nnz(n, k)
        return 8 * nnz

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        n = y.numel()
        k = A.shape[-1] - 1
        stored = _stored_band_nnz(n, k)
        a_bytes = stored * A.element_size()
        io_amount = (
            a_bytes + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
        return max(1, min(2 * k + 1, n))

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        ref_args = (A, x, y.clone())
        blas_args = (A, x, y.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.chbmv
def test_perf_chbmv():
    bench = HbmvBenchmark(
        op_name="chbmv",
        torch_op=cublas_hbmv_baseline,
        gems_op=gems_chbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.chbmv
def test_perf_chbmv_upper():
    bench = HbmvBenchmark(
        op_name="chbmv_upper",
        torch_op=cublas_hbmv_baseline,
        gems_op=gems_chbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhbmv
def test_perf_zhbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HbmvBenchmark(
        op_name="zhbmv",
        torch_op=cublas_hbmv_baseline,
        gems_op=gems_zhbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhbmv
def test_perf_zhbmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HbmvBenchmark(
        op_name="zhbmv_upper",
        torch_op=cublas_hbmv_baseline,
        gems_op=gems_zhbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)

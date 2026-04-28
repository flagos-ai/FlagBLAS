import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils

GBMV_BANDS = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 7),
    (32, 32),
    (128, 128),
    (256, 256),
]

GBMV_SHAPES = [
    (255, 255),
    (256, 256),
    (1023, 1023),
    (1024, 1024),
    (4095, 4095),
    (4096, 4096),
    (16384, 16384),
    (127, 255),
    (4096, 16384),
    (16384, 4096),
    (10000, 10000),
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
    raise RuntimeError("Unable to find libcublas.so on the system.")

_cublas = load_cublas()

class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

_CUBLAS_GBMV_FUNCS = {
    torch.float32: (_cublas.cublasSgbmv_v2, ctypes.c_float, None),
    torch.float64: (_cublas.cublasDgbmv_v2, ctypes.c_double, None),
    torch.complex64: (_cublas.cublasCgbmv_v2, cuComplex, True),
    torch.complex128: (_cublas.cublasZgbmv_v2, cuDoubleComplex, True),
}


def _make_scalar(ctor, is_complex, value):
    if is_complex:
        return ctor(value.real, value.imag)
    return ctor(value)


def cublas_gbmv_baseline(
    AB, x, y, trans, m, n, kl, ku, alpha, lda, incx, beta, incy,
    handle, alpha_ptr, beta_ptr,
    c_func, alpha_c, beta_c, **kwargs,
):
    if m == 0 or n == 0:
        return y

    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(kl),
        ctypes.c_int(ku),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(AB.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXgbmv_v2 failed with status code: {status}")
    return y

def _gems_wrapper(op):
    def _impl(
        AB, x, y, trans, m, n, kl, ku, alpha, lda, incx, beta, incy, handle, alpha_ptr, beta_ptr, **kwargs
    ):
        op(trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy)
        return y
    return _impl

gems_sgbmv_wrapper = _gems_wrapper(flag_blas.ops.sgbmv)
gems_dgbmv_wrapper = _gems_wrapper(flag_blas.ops.dgbmv)
gems_cgbmv_wrapper = _gems_wrapper(flag_blas.ops.cgbmv)
gems_zgbmv_wrapper = _gems_wrapper(flag_blas.ops.zgbmv)

def _generate_banded_AB(m, n, kl, ku, lda, dtype, device):
    """Generate an AB tensor directly in band-storage format."""
    AB = torch.zeros((n, lda), dtype=dtype, device=device)
    for d in range(-ku, kl + 1):
        j_min = max(0, -d)
        j_max = min(n, m - d)
        if j_min < j_max:
            j_idx = torch.arange(j_min, j_max, device=device)
            if dtype.is_complex:
                vals = torch.randn(len(j_idx), dtype=dtype, device=device)
            else:
                vals = torch.randn(len(j_idx), dtype=dtype, device=device) * 0.1
            AB[j_idx, ku + d] = vals
    return AB.contiguous()


class GbmvBenchmark(Benchmark):

    def __init__(
        self,
        *args,
        trans=CUBLAS_OP_N,
        alpha=1.5,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trans = trans
        self.alpha = alpha
        self.beta = beta
        self.bands = GBMV_BANDS

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = GBMV_SHAPES
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            np_dtype = np.float32
        elif cur_dtype == torch.float64:
            np_dtype = np.float64
        elif cur_dtype == torch.complex64:
            np_dtype = np.complex64
        elif cur_dtype == torch.complex128:
            np_dtype = np.complex128
        else:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")

        alpha_np = np.array(self.alpha, dtype=np_dtype)
        beta_np = np.array(self.beta, dtype=np_dtype)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        if cur_dtype not in _CUBLAS_GBMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func, ctor, is_complex = _CUBLAS_GBMV_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, is_complex, self.alpha)
        beta_c = _make_scalar(ctor, is_complex, self.beta)

        seen_configs = set()

        for m, n in self.shapes:
            for kl, ku in self.bands:
                actual_kl = min(kl, max(0, m - 1))
                actual_ku = min(ku, max(0, n - 1))

                config_key = (m, n, actual_kl, actual_ku)
                if config_key in seen_configs:
                    continue
                seen_configs.add(config_key)

                lda = actual_kl + actual_ku + 1
                AB = _generate_banded_AB(m, n, actual_kl, actual_ku, lda, cur_dtype, self.device)

                x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)
                x = torch.randn(x_len, dtype=cur_dtype, device=self.device)
                y = torch.randn(y_len, dtype=cur_dtype, device=self.device)

                yield AB, x, y.clone(), {
                    "trans": self.trans,
                    "m": m,
                    "n": n,
                    "kl": actual_kl,
                    "ku": actual_ku,
                    "alpha": self.alpha,
                    "lda": lda,
                    "incx": 1,
                    "beta": self.beta,
                    "incy": 1,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                    "beta_ptr": beta_ptr,
                    "c_func": c_func,
                    "alpha_c": alpha_c,
                    "beta_c": beta_c,
                }

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        kl = kwargs.get("kl", 0)
        ku = kwargs.get("ku", 0)
        nnz = 0
        for d in range(-ku, kl + 1):
            j_min = max(0, -d)
            j_max = min(n, m - d)
            nnz += max(0, j_max - j_min)
        AB = args[0]
        if AB.dtype in [torch.complex64, torch.complex128]:
            return 8 * nnz
        return 2 * nnz

    def get_gbps(self, args, latency):
        AB, x, y = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(AB)
            + shape_utils.size_in_bytes(x)
            + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

@pytest.mark.sgbmv
def test_perf_sgbmv():
    bench = GbmvBenchmark(
        op_name="sgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_sgbmv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.sgbmv
def test_perf_sgbmv_trans():
    bench = GbmvBenchmark(
        op_name="sgbmv_trans",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_sgbmv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_T,
    )
    bench.run()


@pytest.mark.dgbmv
def test_perf_dgbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="dgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_dgbmv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.dgbmv
def test_perf_dgbmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="dgbmv_trans",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_dgbmv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_T,
    )
    bench.run()


@pytest.mark.cgbmv
def test_perf_cgbmv():
    bench = GbmvBenchmark(
        op_name="cgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_cgbmv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.cgbmv
def test_perf_cgbmv_trans():
    bench = GbmvBenchmark(
        op_name="cgbmv_trans",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_cgbmv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.cgbmv
def test_perf_cgbmv_conj():
    bench = GbmvBenchmark(
        op_name="cgbmv_conj",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_cgbmv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.zgbmv
def test_perf_zgbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="zgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_zgbmv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.zgbmv
def test_perf_zgbmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="zgbmv_trans",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_zgbmv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.zgbmv
def test_perf_zgbmv_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="zgbmv_conj",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_zgbmv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()

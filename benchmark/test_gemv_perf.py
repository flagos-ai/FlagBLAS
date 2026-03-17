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


def cublas_sgemv(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    cublas.sgemv(handle, trans, m, n, alpha_ptr, A.data_ptr(), lda_col, x.data_ptr(), incx, beta_ptr, y.data_ptr(), incy)
    return y


def cublas_dgemv(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    cublas.dgemv(handle, trans, m, n, alpha_ptr, A.data_ptr(), lda_col, x.data_ptr(), incx, beta_ptr, y.data_ptr(), incy)
    return y


def cublas_cgemv(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    cublas.cgemv(handle, trans, m, n, alpha_ptr, A.data_ptr(), lda_col, x.data_ptr(), incx, beta_ptr, y.data_ptr(), incy)
    return y


def cublas_zgemv(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    cublas.zgemv(handle, trans, m, n, alpha_ptr, A.data_ptr(), lda_col, x.data_ptr(), incx, beta_ptr, y.data_ptr(), incy)
    return y


def gems_sgemv_wrapper(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    flag_blas.ops.sgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_dgemv_wrapper(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    flag_blas.ops.dgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_cgemv_wrapper(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    flag_blas.ops.cgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_zgemv_wrapper(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    flag_blas.ops.zgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def torch_hgemv(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    """Reference implementation using torch.addmv for hgemv (float16)."""
    A_mat = A_row[:m * lda_row].view(m, lda_row)[:, :n]
    if trans == CUBLAS_OP_N:
        torch.addmv(y[:m], A_mat, x[:n], alpha=alpha, beta=beta, out=y[:m])
    else:
        torch.addmv(y[:n], A_mat.t(), x[:m], alpha=alpha, beta=beta, out=y[:n])
    return y


def gems_hgemv_wrapper(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    flag_blas.ops.hgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def torch_bfgemv(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    """Reference implementation using torch.addmv for bfgemv (bfloat16)."""
    A_mat = A_row[:m * lda_row].view(m, lda_row)[:, :n]
    if trans == CUBLAS_OP_N:
        torch.addmv(y[:m], A_mat, x[:n], alpha=alpha, beta=beta, out=y[:m])
    else:
        torch.addmv(y[:n], A_mat.t(), x[:m], alpha=alpha, beta=beta, out=y[:n])
    return y


def gems_bfgemv_wrapper(A, x, y, trans, m, n, alpha, A_row, lda_col, lda_row, incx, beta, incy, handle, alpha_ptr, beta_ptr):
    flag_blas.ops.bfgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


class GemvBenchmark(Benchmark):

    def __init__(self, *args, trans=CUBLAS_OP_N, alpha=1.5, beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.trans = trans
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        shapes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (3584, 3584),
            (4096, 4096),
            (7168, 7168),
            (8192, 8192),
            (16384, 16384),
            (18432, 18432),
            (1024, 4096),
            (3584, 18944),
            (4096, 14336),
            (6144, 16384),
            (7168, 18432),
            (8192, 28672),
            (16384, 53248),
            (4096, 1024),
            (18944, 3584),
            (14336, 4096),
            (16384, 6144),
            (18432, 7168),
            (28672, 8192),
            (53248, 16384),
            (63, 63),
            (127, 127),
            (255, 255),
            (511, 511),
            (1023, 1023),
            (3583, 3583),
            (4095, 4095),
            (7167, 7167),
            (8191, 8191),
            (1023, 4095),
            (4095, 14335),
            (4095, 1023),
            (14335, 4095),
        ]
        self.shapes = shapes
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

        alpha_np = np.array(self.alpha, dtype=np_dtype)
        beta_np = np.array(self.beta, dtype=np_dtype)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n = shape
            
            A_col = torch.randn(n, m, dtype=cur_dtype, device=self.device).t()
            A_row = A_col.contiguous()

            x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)
            x = torch.randn(x_len, dtype=cur_dtype, device=self.device)
            y = torch.randn(y_len, dtype=cur_dtype, device=self.device)

            yield A_col, x, y.clone(), {
                "trans": self.trans,
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_row": A_row,
                "lda_col": m,
                "lda_row": n,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        A = args[0]
        if A.dtype in [torch.complex64, torch.complex128]:
            return 8 * m * n
        return 2 * m * n

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        io_amount = shape_utils.size_in_bytes(A) + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        return io_amount * 1e-9 / (latency * 1e-3)

@pytest.mark.gemv
def test_perf_sgemv():
    bench = GemvBenchmark(
        op_name="sgemv",
        torch_op=cublas_sgemv,
        gems_op=gems_sgemv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_sgemv_trans():
    bench = GemvBenchmark(
        op_name="sgemv_trans",
        torch_op=cublas_sgemv,
        gems_op=gems_sgemv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_T,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_dgemv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="dgemv",
        torch_op=cublas_dgemv,
        gems_op=gems_dgemv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_dgemv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="dgemv_trans",
        torch_op=cublas_dgemv,
        gems_op=gems_dgemv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_T,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_cgemv():
    bench = GemvBenchmark(
        op_name="cgemv",
        torch_op=cublas_cgemv,
        gems_op=gems_cgemv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_cgemv_trans():
    bench = GemvBenchmark(
        op_name="cgemv_trans",
        torch_op=cublas_cgemv,
        gems_op=gems_cgemv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_cgemv_conj():
    bench = GemvBenchmark(
        op_name="cgemv_conj",
        torch_op=cublas_cgemv,
        gems_op=gems_cgemv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_zgemv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="zgemv",
        torch_op=cublas_zgemv,
        gems_op=gems_zgemv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_zgemv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="zgemv_trans",
        torch_op=cublas_zgemv,
        gems_op=gems_zgemv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_zgemv_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="zgemv_conj",
        torch_op=cublas_zgemv,
        gems_op=gems_zgemv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    bench.run()


class HgemvBenchmark(GemvBenchmark):
    """Benchmark for hgemv (float16) - compares with torch.addmv implementation."""

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n = shape

            A_row = torch.randn(m, n, dtype=cur_dtype, device=self.device).contiguous()

            x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)
            x = torch.randn(x_len, dtype=cur_dtype, device=self.device)
            y = torch.randn(y_len, dtype=cur_dtype, device=self.device)

            yield A_row, x, y.clone(), {
                "trans": self.trans,
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_row": A_row,
                "lda_col": m,
                "lda_row": n,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }


@pytest.mark.gemv
def test_perf_hgemv():
    bench = HgemvBenchmark(
        op_name="hgemv",
        torch_op=torch_hgemv,
        gems_op=gems_hgemv_wrapper,
        dtypes=[torch.float16],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_hgemv_trans():
    bench = HgemvBenchmark(
        op_name="hgemv_trans",
        torch_op=torch_hgemv,
        gems_op=gems_hgemv_wrapper,
        dtypes=[torch.float16],
        trans=CUBLAS_OP_T,
    )
    bench.run()




class BfgemvBenchmark(GemvBenchmark):
    """Benchmark for bfgemv (bfloat16) - compares with torch.addmv implementation."""

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n = shape

            A_row = torch.randn(m, n, dtype=cur_dtype, device=self.device).contiguous()

            x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)
            x = torch.randn(x_len, dtype=cur_dtype, device=self.device)
            y = torch.randn(y_len, dtype=cur_dtype, device=self.device)

            yield A_row, x, y.clone(), {
                "trans": self.trans,
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_row": A_row,
                "lda_col": m,
                "lda_row": n,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }


@pytest.mark.gemv
def test_perf_bfgemv():
    bench = BfgemvBenchmark(
        op_name="bfgemv",
        torch_op=torch_bfgemv,
        gems_op=gems_bfgemv_wrapper,
        dtypes=[torch.bfloat16],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.gemv
def test_perf_bfgemv_trans():
    bench = BfgemvBenchmark(
        op_name="bfgemv_trans",
        torch_op=torch_bfgemv,
        gems_op=gems_bfgemv_wrapper,
        dtypes=[torch.bfloat16],
        trans=CUBLAS_OP_T,
    )
    bench.run()
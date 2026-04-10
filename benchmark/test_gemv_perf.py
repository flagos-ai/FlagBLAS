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


def cublas_sgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.sgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def cublas_dgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.dgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def cublas_cgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.cgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def cublas_zgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.zgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def cublas_half_gemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    cuda_type,
):
    CUDA_R_32F = 0
    if trans == CUBLAS_OP_N:
        transA = cublas.CUBLAS_OP_T
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = m, 1, n
        lda_c, ldb_c, ldc_c = lda_row, n, m
    else:
        transA = cublas.CUBLAS_OP_N
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = n, 1, m
        lda_c, ldb_c, ldc_c = lda_row, m, n

    cublas.gemmEx(
        handle,
        transA,
        transB,
        m_c,
        n_c,
        k_c,
        alpha_ptr,
        A_row.data_ptr(),
        cuda_type,
        lda_c,
        x.data_ptr(),
        cuda_type,
        ldb_c,
        beta_ptr,
        y.data_ptr(),
        cuda_type,
        ldc_c,
        CUDA_R_32F,
        0,
    )
    return y


def gems_sgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.ops.sgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_dgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.ops.dgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_cgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.ops.cgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_zgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.ops.zgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_hgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    cuda_type=None,
):
    flag_blas.ops.hgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_bfgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    cuda_type=None,
):
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
            # Extreme shapes
            (1, 65536),
            (2, 65536),
            (3, 131071),
            (4, 131072),
            (64, 65536),
            (65536, 1),
            (65536, 2),
            (131071, 3),
            (131072, 4),
            (65536, 64),
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
        io_amount = (
            shape_utils.size_in_bytes(A)
            + shape_utils.size_in_bytes(x)
            + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.sgemv
def test_perf_sgemv():
    bench = GemvBenchmark(
        op_name="sgemv",
        torch_op=cublas_sgemv,
        gems_op=gems_sgemv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.sgemv
def test_perf_sgemv_trans():
    bench = GemvBenchmark(
        op_name="sgemv_trans",
        torch_op=cublas_sgemv,
        gems_op=gems_sgemv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_T,
    )
    bench.run()


@pytest.mark.dgemv
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


@pytest.mark.dgemv
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


@pytest.mark.cgemv
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


@pytest.mark.cgemv
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


@pytest.mark.cgemv
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


@pytest.mark.zgemv
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


@pytest.mark.zgemv
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


@pytest.mark.zgemv
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


class HalfGemvBenchmark(GemvBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        CUDA_R_16F = 2
        CUDA_R_16BF = 14
        cuda_type = CUDA_R_16F if cur_dtype == torch.float16 else CUDA_R_16BF

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
                "cuda_type": cuda_type,
            }


@pytest.mark.hgemv
def test_perf_hgemv():
    bench = HalfGemvBenchmark(
        op_name="hgemv",
        torch_op=cublas_half_gemv,
        gems_op=gems_hgemv_wrapper,
        dtypes=[torch.float16],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.hgemv
def test_perf_hgemv_trans():
    bench = HalfGemvBenchmark(
        op_name="hgemv_trans",
        torch_op=cublas_half_gemv,
        gems_op=gems_hgemv_wrapper,
        dtypes=[torch.float16],
        trans=CUBLAS_OP_T,
    )
    bench.run()


@pytest.mark.bfgemv
def test_perf_bfgemv():
    bench = HalfGemvBenchmark(
        op_name="bfgemv",
        torch_op=cublas_half_gemv,
        gems_op=gems_bfgemv_wrapper,
        dtypes=[torch.bfloat16],
        trans=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.bfgemv
def test_perf_bfgemv_trans():
    bench = HalfGemvBenchmark(
        op_name="bfgemv_trans",
        torch_op=cublas_half_gemv,
        gems_op=gems_bfgemv_wrapper,
        dtypes=[torch.bfloat16],
        trans=CUBLAS_OP_T,
    )
    bench.run()


def cublas_sgemv_fp8_baseline(
    A_fp8,
    x_fp8,
    y,
    trans,
    m,
    n,
    alpha,
    A_col_f32,
    x_f32_ref,
    lda,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.sgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A_col_f32.data_ptr(),
        lda,
        x_f32_ref.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def gems_fp8_gemv_wrapper(
    A_fp8,
    x_fp8,
    y,
    trans,
    m,
    n,
    alpha,
    A_col_f32,
    x_f32_ref,
    lda,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    **kwargs,
):
    flag_blas.ops.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, incx, beta, y, incy)
    return y


class Fp8GemvBenchmark(Benchmark):

    def __init__(
        self,
        *args,
        trans=CUBLAS_OP_N,
        alpha=1.5,
        beta=0.5,
        fp8_dtype=torch.float8_e4m3fn,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trans = trans
        self.alpha = alpha
        self.beta = beta
        self.fp8_dtype = fp8_dtype

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
            (64, 65536),
            (65536, 64),
        ]
        self.shapes = [(m, n) for m, n in shapes if m % 16 == 0 and n % 16 == 0]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        self.alpha_np = np.array(self.alpha, dtype=np.float32)
        self.beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = self.alpha_np.ctypes.data
        beta_ptr = self.beta_np.ctypes.data

        for shape in self.shapes:
            m, n = shape
            x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)

            A_f32 = torch.randn(m, n, dtype=torch.float32, device=self.device) * 0.1
            A_fp8 = A_f32.to(self.fp8_dtype)
            A_col_f32 = A_fp8.float().t().contiguous().t()

            x_f32 = torch.randn(x_len, dtype=torch.float32, device=self.device) * 0.1
            x_fp8 = x_f32.to(self.fp8_dtype)
            x_f32_ref = x_fp8.float()

            y = torch.randn(y_len, dtype=torch.float32, device=self.device)

            yield A_fp8, x_fp8, y.clone(), {
                "trans": self.trans,
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_col_f32": A_col_f32,
                "x_f32_ref": x_f32_ref,
                "lda": m,
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
        return 2 * m * n

    def get_gbps(self, args, latency):
        A_fp8, x_fp8, y = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(A_fp8)
            + shape_utils.size_in_bytes(x_fp8)
            + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.fp8gemv
def test_perf_fp8_gemv_e4m3_vs_sgemv_trans():
    bench = Fp8GemvBenchmark(
        op_name="fp8_gemv_e4m3_vs_sgemv_trans",
        torch_op=cublas_sgemv_fp8_baseline,
        gems_op=gems_fp8_gemv_wrapper,
        dtypes=[torch.float8_e4m3fn],
        trans=CUBLAS_OP_T,
        fp8_dtype=torch.float8_e4m3fn,
    )
    bench.run()


@pytest.mark.fp8gemv
def test_perf_fp8_gemv_e5m2_vs_sgemv_trans():
    bench = Fp8GemvBenchmark(
        op_name="fp8_gemv_e5m2_vs_sgemv_trans",
        torch_op=cublas_sgemv_fp8_baseline,
        gems_op=gems_fp8_gemv_wrapper,
        dtypes=[torch.float8_e5m2],
        trans=CUBLAS_OP_T,
        fp8_dtype=torch.float8_e5m2,
    )
    bench.run()

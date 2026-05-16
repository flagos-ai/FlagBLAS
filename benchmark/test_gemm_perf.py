import itertools
from typing import Generator, List, Tuple

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils

CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14

GEMM_SHAPES = [
    (1024, 1024, 1024),
    (511, 511, 511),
    (1023, 1023, 1023),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (2048, 12288, 4096),
    (2048, 11008, 4096),
    (2048, 4096, 11008),
    (4096, 24576, 8192),
    (4096, 8192, 28672),
    (8192, 28672, 8192),
    (16384, 2048, 2048),
    (2048, 16384, 2048),
    (2048, 2048, 16384),
    (32768, 1024, 1024),
    (4095, 4095, 4095),
    (8191, 8191, 8191),
    (4097, 8191, 4095),
]


def model_shapes() -> List[Tuple[int, int, int]]:
    """
    Generate shapes extracted from real-world LLMs (llama3-8b, qwen2.5-7b).
    These shapes represent common attention and FFN weight matrix dimensions.
    """
    # attn: wqkv, wo; ffn: w13, w2
    NK = [
        # extract from llama3-8b
        (1024, 4096),
        (128256, 4096),
        (14336, 4096),
        (4096, 14336),
        (4096, 4096),
        (6144, 4096),
        (28672, 4096),
        # extract from qwen2.5-7b
        (3584, 3584),
        (18944, 3584),
        (3584, 18944),
        (152064, 3584),
        (37888, 3584),
        (512, 3584),
        (4608, 3584),
    ]

    return [(bs, n, k) for bs, (n, k) in itertools.product([1, 2, 4, 8], NK)]


def cublas_sgemm(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.sgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        lda_cublas,
        B_col.data_ptr(),
        ldb_cublas,
        beta_ptr,
        C_col.data_ptr(),
        ldc_cublas,
    )
    return C_col


def gems_sgemm_wrapper(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.sgemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        lda_flag,
        B_row,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )
    return C_row


def cublas_hgemm(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.gemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        CUDA_R_16F,
        lda_cublas,
        B_col.data_ptr(),
        CUDA_R_16F,
        ldb_cublas,
        beta_ptr,
        C_col.data_ptr(),
        CUDA_R_16F,
        ldc_cublas,
        CUDA_R_32F,
        0,
    )
    return C_col


def gems_hgemm_wrapper(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.hgemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        lda_flag,
        B_row,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )
    return C_row


def cublas_bfgemm(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.gemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        CUDA_R_16BF,
        lda_cublas,
        B_col.data_ptr(),
        CUDA_R_16BF,
        ldb_cublas,
        beta_ptr,
        C_col.data_ptr(),
        CUDA_R_16BF,
        ldc_cublas,
        CUDA_R_32F,
        0,
    )
    return C_col


def gems_bfgemm_wrapper(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.bfgemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        lda_flag,
        B_row,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )
    return C_row


def cublas_fp8gemm_baseline(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.sgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        lda_cublas,
        B_col.data_ptr(),
        ldb_cublas,
        beta_ptr,
        C_col.data_ptr(),
        ldc_cublas,
    )
    return C_col


def gems_fp8gemm_wrapper(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.fp8gemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        lda_flag,
        B_row,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )
    return C_row


FP8_GEMM_SHAPES = [s for s in GEMM_SHAPES if all(d % 16 == 0 for d in s)]


class GemmBenchmark(Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def __init__(
        self,
        *args,
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
        alpha=1.0,
        beta=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transa = transa
        self.transb = transb
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        """
        Return additional shapes for COMPREHENSIVE benchmark level.
        These include shapes from real-world LLMs and special large-k cases.
        """
        return GEMM_SHAPES + model_shapes()

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        cublas.setMathMode(handle, 0)
        torch.backends.cuda.matmul.allow_tf32 = False

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n, k = shape
            scale = k**-0.5
            if self.transa == CUBLAS_OP_N:
                A_col = torch.randn(k, m, dtype=cur_dtype, device=self.device).t()
                lda_cublas, lda_flag = m, k
            else:
                A_col = torch.randn(m, k, dtype=cur_dtype, device=self.device).t()
                lda_cublas, lda_flag = k, m
            A_row = A_col.contiguous()

            if self.transb == CUBLAS_OP_N:
                B_col = torch.randn(n, k, dtype=cur_dtype, device=self.device).t()
                ldb_cublas, ldb_flag = k, n
            else:
                B_col = torch.randn(k, n, dtype=cur_dtype, device=self.device).t()
                ldb_cublas, ldb_flag = n, k
            B_row = B_col.contiguous()

            C_col = torch.randn(n, m, dtype=cur_dtype, device=self.device).t()
            C_row = C_col.contiguous()
            ldc_cublas, ldc_flag = m, n

            yield A_col, B_col, C_col.clone(), {
                "transa": self.transa,
                "transb": self.transb,
                "m": m,
                "n": n,
                "k": k,
                "alpha": self.alpha,
                "A_row": A_row,
                "B_row": B_row,
                "C_row": C_row,
                "lda_cublas": lda_cublas,
                "ldb_cublas": ldb_cublas,
                "ldc_cublas": ldc_cublas,
                "lda_flag": lda_flag,
                "ldb_flag": ldb_flag,
                "ldc_flag": ldc_flag,
                "beta": self.beta,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
        return 2 * m * n * k

    def get_gbps(self, args, latency):
        A, B, C = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(A)
            + shape_utils.size_in_bytes(B)
            + 2 * shape_utils.size_in_bytes(C)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def validate_results(self, torch_result, gems_result, reduce_dim, tolerance=1e-5):
        """
        Compare whether the two result tensors are equal within the specified tolerance.
        If the error exceeds the specified tolerance, throw an AssertionError.
        """
        torch_cpu = torch_result.cpu()
        gems_cpu = gems_result.cpu()

        try:
            flag_blas.testing.assert_close(
                gems_cpu, torch_cpu, torch_cpu.dtype, equal_nan=False, reduce_dim=reduce_dim, atol=tolerance
            )
        except AssertionError as e:
            max_abs_diff = torch.max(torch.abs(torch_cpu - gems_cpu))
            max_rel_diff = torch.max(torch.abs((torch_cpu - gems_cpu) / (torch.abs(torch_cpu) + 1e-9)))
            raise AssertionError(
                f"Results differ beyond tolerance {tolerance}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {torch_cpu.shape}"
            )

@pytest.mark.sgemm
def test_perf_sgemm_nn():
    bench = GemmBenchmark(
        op_name="sgemm",
        torch_op=cublas_sgemm,
        gems_op=gems_sgemm_wrapper,
        dtypes=[torch.float32],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1.3e-6)
    bench.run()


@pytest.mark.sgemm
def test_perf_sgemm_tn():
    bench = GemmBenchmark(
        op_name="sgemm_tn",
        torch_op=cublas_sgemm,
        gems_op=gems_sgemm_wrapper,
        dtypes=[torch.float32],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_N,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1.3e-6)
    bench.run()


@pytest.mark.sgemm
def test_perf_sgemm_nt():
    bench = GemmBenchmark(
        op_name="sgemm_nt",
        torch_op=cublas_sgemm,
        gems_op=gems_sgemm_wrapper,
        dtypes=[torch.float32],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_T,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1.3e-6)
    bench.run()


@pytest.mark.sgemm
def test_perf_sgemm_tt():
    bench = GemmBenchmark(
        op_name="sgemm_tt",
        torch_op=cublas_sgemm,
        gems_op=gems_sgemm_wrapper,
        dtypes=[torch.float32],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_T,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1.3e-6)
    bench.run()


@pytest.mark.hgemm
def test_perf_hgemm_nn():
    bench = GemmBenchmark(
        op_name="hgemm",
        torch_op=cublas_hgemm,
        gems_op=gems_hgemm_wrapper,
        dtypes=[torch.float16],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_hgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_hgemm_wrapper(A, B, C.clone(), **kwargs)

            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.hgemm
def test_perf_hgemm_tn():
    bench = GemmBenchmark(
        op_name="hgemm_tn",
        torch_op=cublas_hgemm,
        gems_op=gems_hgemm_wrapper,
        dtypes=[torch.float16],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_N,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_hgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_hgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.hgemm
def test_perf_hgemm_nt():
    bench = GemmBenchmark(
        op_name="hgemm_nt",
        torch_op=cublas_hgemm,
        gems_op=gems_hgemm_wrapper,
        dtypes=[torch.float16],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_T,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_hgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_hgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.hgemm
def test_perf_hgemm_tt():
    bench = GemmBenchmark(
        op_name="hgemm_tt",
        torch_op=cublas_hgemm,
        gems_op=gems_hgemm_wrapper,
        dtypes=[torch.float16],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_T,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_hgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_hgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.bfgemm
def test_perf_bfgemm_nn():
    bench = GemmBenchmark(
        op_name="bfgemm",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.bfgemm
def test_perf_bfgemm_tn():
    bench = GemmBenchmark(
        op_name="bfgemm_tn",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_N,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.bfgemm
def test_perf_bfgemm_nt():
    bench = GemmBenchmark(
        op_name="bfgemm_nt",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_T,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.bfgemm
def test_perf_bfgemm_tt():
    bench = GemmBenchmark(
        op_name="bfgemm_tt",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_T,
    )
    bench.set_more_shapes()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


class Fp8GemmBenchmark(GemmBenchmark):

    def __init__(
        self, *args, fp8_dtype=torch.float8_e4m3fn, out_dtype=torch.float16, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fp8_dtype = fp8_dtype
        self.out_dtype = out_dtype

    def set_more_shapes(self):
        # FP8 requires all dimensions divisible by 16
        filtered_model_shapes = [
            (bs, n, k) for bs, n, k in model_shapes() 
            if all(d % 16 == 0 for d in (bs, n, k))
        ]
        return FP8_GEMM_SHAPES + filtered_model_shapes

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        cublas.setMathMode(handle, 0)
        torch.backends.cuda.matmul.allow_tf32 = False

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n, k = shape

            if self.transa == CUBLAS_OP_N:
                A_f32_col = torch.randn(
                    k, m, dtype=torch.float32, device=self.device
                ).t()
                lda_cublas, lda_flag = m, k
            else:
                A_f32_col = torch.randn(
                    m, k, dtype=torch.float32, device=self.device
                ).t()
                lda_cublas, lda_flag = k, m
            A_fp8_row = A_f32_col.contiguous().to(self.fp8_dtype)

            if self.transb == CUBLAS_OP_N:
                B_f32_col = torch.randn(
                    n, k, dtype=torch.float32, device=self.device
                ).t()
                ldb_cublas, ldb_flag = k, n
            else:
                B_f32_col = torch.randn(
                    k, n, dtype=torch.float32, device=self.device
                ).t()
                ldb_cublas, ldb_flag = n, k
            B_fp8_row = B_f32_col.contiguous().to(self.fp8_dtype)

            C_f32_col = torch.randn(n, m, dtype=torch.float32, device=self.device).t()
            C_out_row = C_f32_col.contiguous().to(self.out_dtype)
            ldc_cublas, ldc_flag = m, n

            yield A_f32_col, B_f32_col, C_f32_col.clone(), {
                "transa": self.transa,
                "transb": self.transb,
                "m": m,
                "n": n,
                "k": k,
                "alpha": self.alpha,
                "A_row": A_fp8_row,
                "B_row": B_fp8_row,
                "C_row": C_out_row,
                "lda_cublas": lda_cublas,
                "ldb_cublas": ldb_cublas,
                "ldc_cublas": ldc_cublas,
                "lda_flag": lda_flag,
                "ldb_flag": ldb_flag,
                "ldc_flag": ldc_flag,
                "beta": self.beta,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }


@pytest.mark.fp8gemm
def test_perf_fp8gemm_nn():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.fp8gemm
def test_perf_fp8gemm_tn():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm_tn",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_N,
    )
    bench.run()


@pytest.mark.fp8gemm
def test_perf_fp8gemm_nt():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm_nt",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_T,
    )
    bench.run()


@pytest.mark.fp8gemm
def test_perf_fp8gemm_tt():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm_tt",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_T,
    )
    bench.run()

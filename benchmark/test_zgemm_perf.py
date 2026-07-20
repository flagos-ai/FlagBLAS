import os

import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.conftest import Config
from benchmark.test_gemm_perf import GemmBenchmark
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T


def cublas_zgemm(
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
    cublas.zgemm(
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


def gems_zgemm_wrapper(
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
    flag_blas.zgemm(
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


class ZgemmBenchmark(GemmBenchmark):
    DEFAULT_SHAPE_FILES = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_more_shapes(self):
        return None


def _run_zgemm_benchmark(op_name, transa, transb):
    bench = ZgemmBenchmark(
        op_name=op_name,
        torch_op=cublas_zgemm,
        gems_op=gems_zgemm_wrapper,
        dtypes=[torch.complex128],
        transa=transa,
        transb=transb,
        alpha=1.0 + 0.0j,
        beta=0.0 + 0.0j,
        alpha_dtype=np.complex128,
    )
    if Config.query:
        bench.run()
        return

    bench.init_user_config()
    if not Config.skip_correctness:
        for cur_dtype in bench.to_bench_dtypes:
            for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
                torch_result = cublas_zgemm(A, B, C.clone(), **kwargs)
                gems_result = gems_zgemm_wrapper(A, B, C.clone(), **kwargs)
                k = kwargs.get("k", 0)
                bench.validate_results(torch_result, gems_result, k, tolerance=1e-8)
    bench.run()


@pytest.mark.zgemm
def test_perf_zgemm_nn():
    _run_zgemm_benchmark("zgemm", CUBLAS_OP_N, CUBLAS_OP_N)


@pytest.mark.zgemm
def test_perf_zgemm_tn():
    _run_zgemm_benchmark("zgemm_tn", CUBLAS_OP_T, CUBLAS_OP_N)


@pytest.mark.zgemm
def test_perf_zgemm_nt():
    _run_zgemm_benchmark("zgemm_nt", CUBLAS_OP_N, CUBLAS_OP_T)


@pytest.mark.zgemm
def test_perf_zgemm_tt():
    _run_zgemm_benchmark("zgemm_tt", CUBLAS_OP_T, CUBLAS_OP_T)

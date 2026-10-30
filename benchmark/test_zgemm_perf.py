# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import pytest
import torch

import flag_blas
from benchmark.conftest import Config
from benchmark.gemm_perf_common import (
    IS_HYGON,
    GemmBenchmark,
    _HIPBLAS_OP_MAP,
    native_blas_available,
    native_gemm,
    torch_gemm_reference,
)
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T

pytestmark = pytest.mark.skipif(
    flag_blas.vendor_name == "iluvatar",
    reason="zgemm is not supported on Iluvatar",
)

if IS_HYGON:
    from benchmark.gemm_perf_common import _load_hipblas

IS_THEAD = flag_blas.vendor_name == "thead"


def zgemm_reference(
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
    """fp64-complex GEMM reference: official BLAS of the running vendor when
    available (cuBLAS / muBLAS), otherwise a torch.matmul-based fallback."""
    if IS_HYGON:
        # No cupy on the DTK side; use ctypes hipblasZgemm (alpha/beta are
        # hipblasDoubleComplex host scalars, packed into alpha_ptr/beta_ptr by
        # gemm_perf_common.get_input_iter).
        library = _load_hipblas()
        library.hipblasZgemm(
            handle,
            _HIPBLAS_OP_MAP[transa],
            _HIPBLAS_OP_MAP[transb],
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

    if native_blas_available():
        native_gemm(
            handle,
            torch.complex128,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A_col,
            lda_cublas,
            B_col,
            ldb_cublas,
            beta_ptr,
            C_col,
            ldc_cublas,
        )
        return C_col
    return torch_gemm_reference(A_row, B_row, C_row, transa, transb, alpha, beta)


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
        torch_op=zgemm_reference,
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

    if IS_THEAD:
        # MMAO has no cublasZgemm: time FlagBLAS only and compare it with the
        # saved H100 cuBLAS timings (same method as the T-Head Level 2 paths).
        # Accuracy is covered by tests/test_zgemm.py.
        from benchmark.thead_l2_reference import run_thead_gemm

        run_thead_gemm(bench)
        return

    bench.init_user_config()
    if not Config.skip_correctness:
        for cur_dtype in bench.to_bench_dtypes:
            for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
                torch_result = zgemm_reference(A, B, C.clone(), **kwargs)
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

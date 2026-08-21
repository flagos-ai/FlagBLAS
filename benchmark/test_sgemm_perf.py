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

import pytest
import torch

from benchmark.gemm_perf_common import GemmBenchmark, cublas_sgemm, gems_sgemm_wrapper
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T


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
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-4)
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
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-4)
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
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-4)
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
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-4)
    bench.run()

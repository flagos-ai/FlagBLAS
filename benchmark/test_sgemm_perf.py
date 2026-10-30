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

import flag_blas
from benchmark.gemm_perf_common import GemmBenchmark, cublas_sgemm, gems_sgemm_wrapper
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T

# T-Head (Zhenwu) fp32 GEMM accumulates in a different order than cuBLAS
# and routinely differs by ~1e-3 relative; other vendors keep the tighter
# default tolerance.
SGEMM_TOLERANCE = 1e-4 if flag_blas.vendor_name == "thead" else 1.3e-6

IS_ASCEND = flag_blas.vendor_name == "ascend"

if IS_ASCEND:
    from benchmark.ascend_l2_reference import AscendL2Benchmark

    class SgemmBenchmark(AscendL2Benchmark, GemmBenchmark):
        """Ascend GEMM benchmark against the saved H100 cuBLAS baseline.

        Ascend has no callable official BLAS, so only the FlagBLAS call is
        timed. ``latency_base``/``gbps_base`` come from
        benchmark/baselines/h100_l3.json (core shapes only) and the speedup is
        normalized by the H100/Ascend conversion factor; see
        benchmark/ascend_l2_reference.py. The default assumption is the memory
        bound factor, override it with FLAGBLAS_L2_BOTTLENECK=vector when
        scoring the compute-bound cubic shapes.
        """

        metric_family = "gemm"

        def set_more_shapes(self):
            # Only the core GEMM shapes have saved H100 timings, so an Ascend
            # run never appends the comprehensive-level extra shapes.
            return None

else:

    class SgemmBenchmark(GemmBenchmark):
        pass


def _run_sgemm_benchmark(op_name, transa, transb):
    bench = SgemmBenchmark(
        op_name=op_name,
        torch_op=cublas_sgemm,
        gems_op=gems_sgemm_wrapper,
        dtypes=[torch.float32],
        transa=transa,
        transb=transb,
    )
    if IS_ASCEND:
        # Accuracy is covered by tests/test_sgemm.py; the performance case only
        # times FlagBLAS and compares it with the saved H100 cuBLAS timings.
        bench.run()
        return

    bench.init_user_config()
    for cur_dtype in bench.to_bench_dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_sgemm(A, B, C.clone(), **kwargs)
            gems_result = gems_sgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(
                torch_result, gems_result, k, tolerance=SGEMM_TOLERANCE
            )
    bench.run()


@pytest.mark.sgemm
def test_perf_sgemm_nn():
    _run_sgemm_benchmark("sgemm", CUBLAS_OP_N, CUBLAS_OP_N)


@pytest.mark.sgemm
def test_perf_sgemm_tn():
    _run_sgemm_benchmark("sgemm_tn", CUBLAS_OP_T, CUBLAS_OP_N)


@pytest.mark.sgemm
def test_perf_sgemm_nt():
    _run_sgemm_benchmark("sgemm_nt", CUBLAS_OP_N, CUBLAS_OP_T)


@pytest.mark.sgemm
def test_perf_sgemm_tt():
    _run_sgemm_benchmark("sgemm_tt", CUBLAS_OP_T, CUBLAS_OP_T)

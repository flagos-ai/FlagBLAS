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

import itertools
from typing import Generator, List, Tuple

import cupy as cp
import numpy as np
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.ops import CUBLAS_OP_N
from flag_blas.utils import shape_utils

CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14

GEMM_SHAPES = [
    (511, 511, 511),
    (1023, 1023, 1023),
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
    Generate shapes with m ranging from 1 to 32 (step 1), then 64, 128, 256, 512, 1024, 2048, 4096.
    """
    m_values = list(range(1, 33)) + [64, 128, 256, 512, 1024, 2048, 4096]

    NK = [
        [6144, 4096],
        [4096, 4096],
        [24576, 4096],
        [4096, 12288],
        [5120, 5120],
        [5120, 4096],
        [25600, 5120],
        [5120, 12800],
        [2560, 5120],
        [5120, 2048],
        [12800, 5210],
        [5120, 6400],
        [5120, 2048],
        [2048, 4096],
        [2560, 2048],
        [2048, 1024],
        [1152, 4096],
        [4096, 1024],
        [4096, 7168],
        [7168, 2048],
        [2304, 2048],
        [1152, 2048],
        [2048, 1024],
        [2048, 512],
        [3072, 2048],
        [1536, 2048],
    ]

    return [(m, n, k) for m, (n, k) in itertools.product(m_values, NK)]


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
        cublas.CUBLAS_COMPUTE_32F,
        cublas.CUBLAS_GEMM_DEFAULT,
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
        cublas.CUBLAS_COMPUTE_32F,
        cublas.CUBLAS_GEMM_DEFAULT,
    )
    return C_col


def cublas_bfgemm_reference(
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
    C_fp32 = torch.empty_strided(
        C_col.shape, C_col.stride(), dtype=torch.float32, device=C_col.device
    )
    C_fp32.copy_(C_col)

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
        C_fp32.data_ptr(),
        CUDA_R_32F,
        ldc_cublas,
        cublas.CUBLAS_COMPUTE_32F,
        cublas.CUBLAS_GEMM_DEFAULT,
    )
    C_col.copy_(C_fp32.to(torch.bfloat16))
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


class GemmBenchmark(Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def __init__(
        self,
        *args,
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
        alpha=1.0,
        beta=0.0,
        alpha_dtype=np.float32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transa = transa
        self.transb = transb
        self.alpha = alpha
        self.beta = beta
        self.alpha_dtype = alpha_dtype

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

        alpha_np = np.array(self.alpha, dtype=self.alpha_dtype)
        beta_np = np.array(self.beta, dtype=self.alpha_dtype)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n, k = shape
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
                gems_cpu,
                torch_cpu,
                torch_cpu.dtype,
                equal_nan=False,
                reduce_dim=reduce_dim,
                atol=tolerance,
            )
        except AssertionError as e:
            max_abs_diff = torch.max(torch.abs(torch_cpu - gems_cpu))
            max_rel_diff = torch.max(
                torch.abs((torch_cpu - gems_cpu) / (torch.abs(torch_cpu) + 1e-9))
            )
            raise AssertionError(
                f"{e} Results differ beyond tolerance {tolerance}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {torch_cpu.shape}"
            )

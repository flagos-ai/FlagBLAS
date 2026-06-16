import cupy as cp
import numpy as np
import pytest
import scipy
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas

import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T

from . import accuracy_utils as utils
from .conftest import TO_CPU


def cublas_sgemm_reference(
    transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
):
    if m == 0 or n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)

    cublas.sgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_np.ctypes.data,
        A.data_ptr(),
        lda,
        B.data_ptr(),
        ldb,
        beta_np.ctypes.data,
        C.data_ptr(),
        ldc,
    )


CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14


@pytest.mark.sgemm
@pytest.mark.parametrize("m,n,k", utils.GEMM_SHAPES)
@pytest.mark.parametrize(
    "transa,transb",
    [
        (CUBLAS_OP_N, CUBLAS_OP_N),
        (CUBLAS_OP_N, CUBLAS_OP_T),
        (CUBLAS_OP_T, CUBLAS_OP_N),
        (CUBLAS_OP_T, CUBLAS_OP_T),
    ],
)
def test_accuracy_sgemm(m, n, k, transa, transb):
    dtype, alpha, beta = torch.float32, 2.5, 0.5

    if transa == CUBLAS_OP_N:
        A_col = (torch.randn(k, m, dtype=dtype, device=flag_blas.device)).t()
        lda_cublas, lda_flag = m, k
    else:
        A_col = (torch.randn(m, k, dtype=dtype, device=flag_blas.device)).t()
        lda_cublas, lda_flag = k, m
    A_row = A_col.contiguous()

    if transb == CUBLAS_OP_N:
        B_col = (torch.randn(n, k, dtype=dtype, device=flag_blas.device)).t()
        ldb_cublas, ldb_flag = k, n
    else:
        B_col = (torch.randn(k, n, dtype=dtype, device=flag_blas.device)).t()
        ldb_cublas, ldb_flag = n, k
    B_row = B_col.contiguous()

    C_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    C_row = C_col.contiguous()
    ldc_cublas, ldc_flag = m, n

    if TO_CPU:
        A_ref = A_row.to("cpu").to(torch.float64)
        B_ref = B_row.to("cpu").to(torch.float64)
        C_ref = C_row.to("cpu").to(torch.float64)
        C_ref = blas.dgemm(
            alpha,
            A_ref.numpy(),
            B_ref.numpy(),
            beta,
            c=C_ref.numpy(),
            trans_b=transb,
            trans_a=transa,
        )
    else:
        cublas_sgemm_reference(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            A_col,
            lda_cublas,
            B_col,
            ldb_cublas,
            beta,
            C_col,
            ldc_cublas,
        )
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
    if TO_CPU:
        utils.blas_assert_close(C_row, torch.tensor(C_ref), dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), dtype, reduce_dim=k)


@pytest.mark.sgemm
def test_sgemm_alpha_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    C = torch.randn(m, n, dtype=torch.float32, device=device)
    C_orig = C.clone()
    A = torch.randn(m, k, dtype=torch.float32, device=device)
    B = torch.randn(k, n, dtype=torch.float32, device=device)

    flag_blas.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 0.0, A, k, B, n, 2.0, C, n)

    if TO_CPU:
        utils.blas_assert_close(C, C_orig.to("cpu") * 2.0, torch.float32, reduce_dim=k)
    else:
        utils.blas_assert_close(C, C_orig * 2.0, torch.float32, reduce_dim=k)


@pytest.mark.sgemm
def test_sgemm_beta_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    A = torch.randn(m, k, dtype=torch.float32, device=device)
    B = torch.randn(k, n, dtype=torch.float32, device=device)
    C_nan = torch.full((m, n), float("nan"), dtype=torch.float32, device=device)
    C_zero = torch.zeros(m, n, dtype=torch.float32, device=device)

    flag_blas.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, A, k, B, n, 0.0, C_nan, n)
    flag_blas.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, A, k, B, n, 0.0, C_zero, n)

    if TO_CPU:
        utils.blas_assert_close(C_nan, C_zero.to("cpu"), torch.float32, reduce_dim=k)
    else:
        utils.blas_assert_close(C_nan, C_zero, torch.float32, reduce_dim=k)


@pytest.mark.sgemm
@pytest.mark.parametrize("m,n,k", [(0, 64, 64), (64, 0, 64), (64, 64, 0)])
def test_sgemm_empty(m, n, k):
    device = flag_blas.device
    rows_a, cols_a = (m, k) if k > 0 else (m, 1)
    rows_b, cols_b = (k, n) if k > 0 else (1, n)
    rows_c, cols_c = max(m, 1), max(n, 1)

    A = torch.randn(rows_a, cols_a, dtype=torch.float32, device=device)
    B = torch.randn(rows_b, cols_b, dtype=torch.float32, device=device)
    C = torch.randn(rows_c, cols_c, dtype=torch.float32, device=device)
    C_orig = C.clone()

    flag_blas.sgemm(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        1.0,
        A,
        max(cols_a, 1),
        B,
        max(cols_b, 1),
        0.5,
        C,
        max(cols_c, 1),
    )
    if TO_CPU:
        utils.blas_assert_close(C, C_orig.to("cpu") * 0.5, torch.float32, reduce_dim=k)
    else:
        utils.blas_assert_close(C, C_orig * 0.5, torch.float32, reduce_dim=k)


@pytest.mark.sgemm
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 2.5)]
)
def test_sgemm_alpha_beta(alpha, beta):
    m, n, k = 256, 256, 256
    dtype = torch.float32
    device = flag_blas.device

    A_col = (torch.randn(k, m, dtype=dtype, device=device)).t()
    A_row = A_col.contiguous()
    B_col = (torch.randn(n, k, dtype=dtype, device=device)).t()
    B_row = B_col.contiguous()
    C_col = (torch.randn(n, m, dtype=dtype, device=device)).t()
    C_row = C_col.contiguous()

    cublas_sgemm_reference(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_col, m, B_col, k, beta, C_col, m
    )
    flag_blas.sgemm(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_row, k, B_row, n, beta, C_row, n
    )
    if TO_CPU:
        utils.blas_assert_close(
            C_row, C_col.contiguous().to("cpu"), torch.float32, reduce_dim=k
        )

    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), torch.float32, reduce_dim=k)

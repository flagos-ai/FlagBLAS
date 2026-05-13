import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas
import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T

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


CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14



@pytest.mark.sgemm
@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize(
    "transa,transb",
    [
        (CUBLAS_OP_N, CUBLAS_OP_N),
        (CUBLAS_OP_N, CUBLAS_OP_T),
        (CUBLAS_OP_T, CUBLAS_OP_N),
        (CUBLAS_OP_T, CUBLAS_OP_T),
    ],
)


def cublas_bfgemm_reference(
    transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
):
    if m == 0 or n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.array([alpha], dtype=np.float32)
    beta_np = np.array([beta], dtype=np.float32)

    cublas.gemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_np.ctypes.data,
        A.data_ptr(),
        CUDA_R_16BF,
        lda,
        B.data_ptr(),
        CUDA_R_16BF,
        ldb,
        beta_np.ctypes.data,
        C.data_ptr(),
        CUDA_R_16BF,
        ldc,
        CUDA_R_32F,
        0,
    )


@pytest.mark.bfgemm
@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize(
    "transa,transb",
    [
        (CUBLAS_OP_N, CUBLAS_OP_N),
        (CUBLAS_OP_N, CUBLAS_OP_T),
        (CUBLAS_OP_T, CUBLAS_OP_N),
        (CUBLAS_OP_T, CUBLAS_OP_T),
    ],
)
def test_accuracy_bfgemm(m, n, k, transa, transb):
    dtype, alpha, beta = torch.bfloat16, 1.5, 0.5
    scale = k**-0.5

    if transa == CUBLAS_OP_N:
        A_col = (torch.randn(k, m, dtype=dtype, device=flag_blas.device) * scale).t()
        lda_cublas, lda_flag = m, k
    else:
        A_col = (torch.randn(m, k, dtype=dtype, device=flag_blas.device) * scale).t()
        lda_cublas, lda_flag = k, m
    A_row = A_col.contiguous()

    if transb == CUBLAS_OP_N:
        B_col = (torch.randn(n, k, dtype=dtype, device=flag_blas.device) * scale).t()
        ldb_cublas, ldb_flag = k, n
    else:
        B_col = (torch.randn(k, n, dtype=dtype, device=flag_blas.device) * scale).t()
        ldb_cublas, ldb_flag = n, k
    B_row = B_col.contiguous()

    C_col = (torch.randn(n, m, dtype=dtype, device=flag_blas.device) * scale).t()
    C_row = C_col.contiguous()
    ldc_cublas, ldc_flag = m, n

    cublas_bfgemm_reference(
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

    rtol = 1e-3
    atol = 1e-3
    torch.testing.assert_close(C_row, C_col.contiguous(), rtol=rtol, atol=atol)


@pytest.mark.bfgemm
def test_bfgemm_alpha_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    C = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    C_orig = C.clone()
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    B = torch.randn(k, n, dtype=torch.bfloat16, device=device)

    flag_blas.bfgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 0.0, A, k, B, n, 2.0, C, n)

    torch.testing.assert_close(C, C_orig * 2.0)


@pytest.mark.bfgemm
def test_bfgemm_beta_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    B = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    C_nan = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    C_zero = torch.zeros(m, n, dtype=torch.bfloat16, device=device)

    flag_blas.bfgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, A, k, B, n, 0.0, C_nan, n)
    flag_blas.bfgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, A, k, B, n, 0.0, C_zero, n)

    torch.testing.assert_close(C_nan, C_zero)


@pytest.mark.bfgemm
@pytest.mark.parametrize("m,n,k", [(0, 64, 64), (64, 0, 64), (64, 64, 0)])
def test_bfgemm_empty(m, n, k):
    device = flag_blas.device
    rows_a, cols_a = (m, k) if k > 0 else (m, 1)
    rows_b, cols_b = (k, n) if k > 0 else (1, n)
    rows_c, cols_c = max(m, 1), max(n, 1)

    A = torch.randn(rows_a, cols_a, dtype=torch.bfloat16, device=device)
    B = torch.randn(rows_b, cols_b, dtype=torch.bfloat16, device=device)
    C = torch.randn(rows_c, cols_c, dtype=torch.bfloat16, device=device)
    C_orig = C.clone()

    flag_blas.bfgemm(
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

    torch.testing.assert_close(C, C_orig * 0.5)


@pytest.mark.bfgemm
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 1.5)]
)
def test_bfgemm_alpha_beta(alpha, beta):
    m, n, k = 256, 256, 256
    dtype = torch.bfloat16
    device = flag_blas.device
    scale = k**-0.5

    A_col = (torch.randn(k, m, dtype=dtype, device=device) * scale).t()
    A_row = A_col.contiguous()
    B_col = (torch.randn(n, k, dtype=dtype, device=device) * scale).t()
    B_row = B_col.contiguous()
    C_col = (torch.randn(n, m, dtype=dtype, device=device) * scale).t()
    C_row = C_col.contiguous()

    cublas_bfgemm_reference(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_col, m, B_col, k, beta, C_col, m
    )
    flag_blas.bfgemm(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_row, k, B_row, n, beta, C_row, n
    )

    torch.testing.assert_close(C_row, C_col.contiguous(), rtol=1e-3, atol=1e-3)


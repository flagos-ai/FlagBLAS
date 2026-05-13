import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas
import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T


FP8_GEMM_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (2048, 12288, 4096),
    (2048, 4096, 11008),
    (4096, 24576, 8192),
    (4096, 8192, 28672),
    (8192, 28672, 8192),
    (16384, 2048, 2048),
    (2048, 16384, 2048),
    (2048, 2048, 16384),
    (32768, 1024, 1024),
    (4096, 8192, 4096),
]

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

@pytest.mark.fp8gemm
@pytest.mark.parametrize("m,n,k", FP8_GEMM_SHAPES)
@pytest.mark.parametrize(
    "transa,transb",
    [
        (CUBLAS_OP_N, CUBLAS_OP_N),
        (CUBLAS_OP_N, CUBLAS_OP_T),
        (CUBLAS_OP_T, CUBLAS_OP_N),
        (CUBLAS_OP_T, CUBLAS_OP_T),
    ],
)
def test_accuracy_fp8gemm_e4m3(m, n, k, transa, transb):
    fp8_dtype = torch.float8_e4m3fn
    out_dtype = torch.float16
    device = flag_blas.device
    alpha, beta = 1.5, 0.5
    scale = k**-0.5

    if transa == CUBLAS_OP_N:
        A_fp8 = (torch.randn(m, k, device=device) * scale).to(fp8_dtype)
        A_f32_row = A_fp8.float()
        A_f32_col = A_f32_row.T.contiguous().T
        lda_cublas, lda_flag = m, k
    else:
        A_fp8 = (torch.randn(k, m, device=device) * scale).to(fp8_dtype)
        A_f32_row = A_fp8.float()
        A_f32_col = A_f32_row.T.contiguous().T
        lda_cublas, lda_flag = k, m

    if transb == CUBLAS_OP_N:
        B_fp8 = (torch.randn(k, n, device=device) * scale).to(fp8_dtype)
        B_f32_row = B_fp8.float()
        B_f32_col = B_f32_row.T.contiguous().T
        ldb_cublas, ldb_flag = k, n
    else:
        B_fp8 = (torch.randn(n, k, device=device) * scale).to(fp8_dtype)
        B_f32_row = B_fp8.float()
        B_f32_col = B_f32_row.T.contiguous().T
        ldb_cublas, ldb_flag = n, k

    C_col = (torch.randn(n, m, device=device) * scale).t().to(torch.float32)
    C_row = C_col.contiguous().to(out_dtype)
    ldc_cublas, ldc_flag = m, n

    cublas_sgemm_reference(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_f32_col,
        lda_cublas,
        B_f32_col,
        ldb_cublas,
        beta,
        C_col,
        ldc_cublas,
    )
    flag_blas.fp8gemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_fp8,
        lda_flag,
        B_fp8,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )

    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(C_row.float(), C_col.contiguous(), rtol=rtol, atol=atol)


@pytest.mark.fp8gemm
@pytest.mark.parametrize("m,n,k", FP8_GEMM_SHAPES)
@pytest.mark.parametrize(
    "transa,transb",
    [
        (CUBLAS_OP_N, CUBLAS_OP_N),
        (CUBLAS_OP_N, CUBLAS_OP_T),
        (CUBLAS_OP_T, CUBLAS_OP_N),
        (CUBLAS_OP_T, CUBLAS_OP_T),
    ],
)
def test_accuracy_fp8gemm_e5m2(m, n, k, transa, transb):
    fp8_dtype = torch.float8_e5m2
    out_dtype = torch.float16
    device = flag_blas.device
    alpha, beta = 1.5, 0.5
    scale = k**-0.5

    if transa == CUBLAS_OP_N:
        A_fp8 = (torch.randn(m, k, device=device) * scale).to(fp8_dtype)
        A_f32_row = A_fp8.float()
        A_f32_col = A_f32_row.T.contiguous().T
        lda_cublas, lda_flag = m, k
    else:
        A_fp8 = (torch.randn(k, m, device=device) * scale).to(fp8_dtype)
        A_f32_row = A_fp8.float()
        A_f32_col = A_f32_row.T.contiguous().T
        lda_cublas, lda_flag = k, m

    if transb == CUBLAS_OP_N:
        B_fp8 = (torch.randn(k, n, device=device) * scale).to(fp8_dtype)
        B_f32_row = B_fp8.float()
        B_f32_col = B_f32_row.T.contiguous().T
        ldb_cublas, ldb_flag = k, n
    else:
        B_fp8 = (torch.randn(n, k, device=device) * scale).to(fp8_dtype)
        B_f32_row = B_fp8.float()
        B_f32_col = B_f32_row.T.contiguous().T
        ldb_cublas, ldb_flag = n, k

    C_col = (torch.randn(n, m, device=device) * scale).t().to(torch.float32)
    C_row = C_col.contiguous().to(out_dtype)
    ldc_cublas, ldc_flag = m, n

    cublas_sgemm_reference(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_f32_col,
        lda_cublas,
        B_f32_col,
        ldb_cublas,
        beta,
        C_col,
        ldc_cublas,
    )
    flag_blas.fp8gemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_fp8,
        lda_flag,
        B_fp8,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )

    rtol = 1.6e-2
    atol = 1.6e-2
    torch.testing.assert_close(C_row.float(), C_col.contiguous(), rtol=rtol, atol=atol)


@pytest.mark.fp8gemm
def test_fp8gemm_alpha_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    C = torch.randn(m, n, dtype=torch.float16, device=device)
    C_orig = C.clone()
    A = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
    B = torch.randn(k, n, device=device).to(torch.float8_e4m3fn)

    flag_blas.fp8gemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 0.0, A, k, B, n, 2.0, C, n)

    torch.testing.assert_close(C, C_orig * 2.0)


@pytest.mark.fp8gemm
def test_fp8gemm_beta_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    A = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
    B = torch.randn(k, n, device=device).to(torch.float8_e4m3fn)
    C_nan = torch.full((m, n), float("nan"), dtype=torch.float16, device=device)
    C_zero = torch.zeros(m, n, dtype=torch.float16, device=device)

    flag_blas.fp8gemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, A, k, B, n, 0.0, C_nan, n)
    flag_blas.fp8gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, A, k, B, n, 0.0, C_zero, n
    )

    torch.testing.assert_close(C_nan, C_zero)


@pytest.mark.fp8gemm
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 1.5)]
)
def test_fp8gemm_alpha_beta(alpha, beta):
    m, n, k = 256, 256, 256
    fp8_dtype = torch.float8_e4m3fn
    out_dtype = torch.float16
    device = flag_blas.device
    scale = k**-0.5

    A_fp8 = (torch.randn(m, k, device=device) * scale).to(fp8_dtype)
    B_fp8 = (torch.randn(k, n, device=device) * scale).to(fp8_dtype)
    A_f32_col = A_fp8.float().T.contiguous().T
    B_f32_col = B_fp8.float().T.contiguous().T

    C_col = (torch.randn(n, m, device=device) * scale).t().to(torch.float32)
    C_row = C_col.contiguous().to(out_dtype)

    cublas_sgemm_reference(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        alpha,
        A_f32_col,
        m,
        B_f32_col,
        k,
        beta,
        C_col,
        m,
    )
    flag_blas.fp8gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_fp8, k, B_fp8, n, beta, C_row, n
    )

    torch.testing.assert_close(C_row.float(), C_col.contiguous(), rtol=1e-2, atol=1e-2)


@pytest.mark.fp8gemm
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp8gemm_output_dtypes(out_dtype):
    m, n, k = 256, 256, 256
    device = flag_blas.device
    scale = k**-0.5

    A_fp8 = (torch.randn(m, k, device=device) * scale).to(torch.float8_e4m3fn)
    B_fp8 = (torch.randn(k, n, device=device) * scale).to(torch.float8_e4m3fn)
    A_f32_col = A_fp8.float().T.contiguous().T
    B_f32_col = B_fp8.float().T.contiguous().T

    C_col = torch.zeros(n, m, device=device, dtype=torch.float32).t()
    C_row = torch.zeros(m, n, device=device, dtype=out_dtype)

    cublas_sgemm_reference(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        1.0,
        A_f32_col,
        m,
        B_f32_col,
        k,
        0.0,
        C_col,
        m,
    )
    flag_blas.fp8gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, A_fp8, k, B_fp8, n, 0.0, C_row, n
    )

    assert C_row.dtype == out_dtype
    torch.testing.assert_close(C_row.float(), C_col.contiguous(), rtol=1e-2, atol=1e-2)

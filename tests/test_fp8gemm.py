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


@pytest.mark.fp8gemm
@pytest.mark.parametrize("m,n,k", utils.FP8_GEMM_SHAPES)
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

    C_col = torch.randn(n, m, device=device).t().to(torch.float32)
    C_row = C_col.contiguous().to(out_dtype)
    ldc_cublas, ldc_flag = m, n

    if TO_CPU:
        A_ref = A_f32_row.to("cpu").to(torch.float64)
        B_ref = B_f32_row.to("cpu").to(torch.float64)
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
    if TO_CPU:
        utils.blas_assert_close(C_row, torch.tensor(C_ref), out_dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), out_dtype, reduce_dim=k)


@pytest.mark.fp8gemm
@pytest.mark.parametrize("m,n,k", utils.FP8_GEMM_SHAPES)
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

    C_col = torch.randn(n, m, device=device).t().to(torch.float32)
    C_row = C_col.contiguous().to(out_dtype)
    ldc_cublas, ldc_flag = m, n

    if TO_CPU:
        A_ref = A_f32_row.to("cpu").to(torch.float64)
        B_ref = B_f32_row.to("cpu").to(torch.float64)
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
    if TO_CPU:
        utils.blas_assert_close(C_row, torch.tensor(C_ref), out_dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), out_dtype, reduce_dim=k)


@pytest.mark.fp8gemm
def test_fp8gemm_alpha_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    C = torch.randn(m, n, dtype=torch.float16, device=device)
    C_orig = C.clone()
    A = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
    B = torch.randn(k, n, device=device).to(torch.float8_e4m3fn)

    flag_blas.fp8gemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 0.0, A, k, B, n, 2.0, C, n)

    if TO_CPU:
        utils.blas_assert_close(C, C_orig.to("cpu") * 2.0, torch.float16, reduce_dim=k)
    else:
        utils.blas_assert_close(C, C_orig * 2.0, torch.float16, reduce_dim=k)


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

    if TO_CPU:
        utils.blas_assert_close(C_nan, C_zero.to("cpu"), torch.float16, reduce_dim=k)
    else:
        utils.blas_assert_close(C_nan, C_zero, torch.float16, reduce_dim=k)


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
    A_f32_row = A_fp8.float()
    A_f32_col = A_f32_row.T.contiguous().T
    B_f32_row = B_fp8.float()
    B_f32_col = B_f32_row.T.contiguous().T

    C_col = torch.randn(n, m, device=device).t().to(torch.float32)
    C_row = C_col.contiguous().to(out_dtype)

    if TO_CPU:
        A_ref = A_f32_row.to("cpu").to(torch.float64)
        B_ref = B_f32_row.to("cpu").to(torch.float64)
        C_ref = C_row.to("cpu").to(torch.float64)
        C_ref = blas.dgemm(
            alpha,
            A_ref.numpy(),
            B_ref.numpy(),
            beta,
            c=C_ref.numpy(),
            trans_b=CUBLAS_OP_N,
            trans_a=CUBLAS_OP_N,
        )
    else:
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

    if TO_CPU:
        utils.blas_assert_close(C_row, torch.tensor(C_ref), out_dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), out_dtype, reduce_dim=k)


@pytest.mark.fp8gemm
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp8gemm_output_dtypes(out_dtype):
    m, n, k = 256, 256, 256
    device = flag_blas.device
    scale = k**-0.5

    A_fp8 = (torch.randn(m, k, device=device) * scale).to(torch.float8_e4m3fn)
    B_fp8 = (torch.randn(k, n, device=device) * scale).to(torch.float8_e4m3fn)
    A_f32_row = A_fp8.float()
    A_f32_col = A_f32_row.T.contiguous().T
    B_f32_row = B_fp8.float()
    B_f32_col = B_f32_row.T.contiguous().T

    C_col = torch.zeros(n, m, device=device, dtype=torch.float32).t()
    C_row = torch.zeros(m, n, device=device, dtype=out_dtype)

    if TO_CPU:
        A_ref = A_f32_row.to("cpu").to(torch.float64)
        B_ref = B_f32_row.to("cpu").to(torch.float64)
        C_ref = C_row.to("cpu").to(torch.float64)
        C_ref = blas.dgemm(
            1.0,
            A_ref.numpy(),
            B_ref.numpy(),
            0.0,
            c=C_ref.numpy(),
            trans_b=CUBLAS_OP_N,
            trans_a=CUBLAS_OP_N,
        )
    else:
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
    if TO_CPU:
        utils.blas_assert_close(C_row, torch.tensor(C_ref), out_dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), out_dtype, reduce_dim=k)

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas

import flag_blas
from flag_blas.ops import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T

from . import accuracy_utils as utils
from .conftest import TO_CPU

ZGEMM_SHAPES = [(32, 32, 32), (64, 64, 64), (127, 65, 33)]


def cublas_zgemm_reference(
    transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
):
    if m == 0 or n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np.complex128)
    beta_np = np.asarray(beta, dtype=np.complex128)

    cublas.zgemm(
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


@pytest.mark.zgemm
@pytest.mark.parametrize("m,n,k", ZGEMM_SHAPES)
@pytest.mark.parametrize(
    "transa,transb",
    [
        (CUBLAS_OP_N, CUBLAS_OP_N),
        (CUBLAS_OP_N, CUBLAS_OP_T),
        (CUBLAS_OP_N, CUBLAS_OP_C),
        (CUBLAS_OP_T, CUBLAS_OP_N),
        (CUBLAS_OP_T, CUBLAS_OP_T),
        (CUBLAS_OP_T, CUBLAS_OP_C),
        (CUBLAS_OP_C, CUBLAS_OP_N),
        (CUBLAS_OP_C, CUBLAS_OP_T),
        (CUBLAS_OP_C, CUBLAS_OP_C),
    ],
)
def test_accuracy_zgemm(m, n, k, transa, transb):
    dtype, alpha, beta = torch.complex128, 2.5 + 0.5j, 0.5 - 0.25j

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
        C_ref = blas.zgemm(
            alpha,
            A_row.cpu().numpy(),
            B_row.cpu().numpy(),
            beta,
            c=C_row.cpu().numpy(),
            trans_a=transa,
            trans_b=transb,
        )
    else:
        cublas_zgemm_reference(
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

    if TO_CPU:
        utils.blas_assert_close(C_row, torch.tensor(C_ref), dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), dtype, reduce_dim=k)


@pytest.mark.zgemm
@pytest.mark.parametrize(
    "alpha,beta",
    [
        (1.0 + 0.0j, 0.0 + 0.0j),
        (2.0 + 0.5j, 0.0 + 0.0j),
        (2.0 - 0.5j, 0.5 + 0.25j),
        (0.0 + 0.0j, 1.0 + 0.0j),
        (0.5 + 0.25j, 2.5 - 0.5j),
    ],
)
def test_zgemm_alpha_beta(alpha, beta):
    m, n, k = 128, 128, 128
    device = flag_blas.device
    A = torch.randn(m, k, dtype=torch.complex128, device=device)
    B = torch.randn(k, n, dtype=torch.complex128, device=device)
    C = torch.randn(m, n, dtype=torch.complex128, device=device)
    expected = alpha * (A @ B) + beta * C

    flag_blas.zgemm(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, k, B, n, beta, C, n
    )

    if TO_CPU:
        utils.blas_assert_close(C, expected.cpu(), torch.complex128, reduce_dim=k)
    else:
        utils.blas_assert_close(C, expected, torch.complex128, reduce_dim=k)


@pytest.mark.zgemm
@pytest.mark.parametrize("m,n,k", [(0, 64, 64), (64, 0, 64), (64, 64, 0)])
def test_zgemm_empty(m, n, k):
    device = flag_blas.device
    rows_a, cols_a = (m, k) if k > 0 else (m, 1)
    rows_b, cols_b = (k, n) if k > 0 else (1, n)
    rows_c, cols_c = max(m, 1), max(n, 1)

    A = torch.randn(rows_a, cols_a, dtype=torch.complex128, device=device)
    B = torch.randn(rows_b, cols_b, dtype=torch.complex128, device=device)
    C = torch.randn(rows_c, cols_c, dtype=torch.complex128, device=device)
    C_orig = C.clone()

    flag_blas.zgemm(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        1.0 + 0.0j,
        A,
        max(cols_a, 1),
        B,
        max(cols_b, 1),
        0.5 - 0.25j,
        C,
        max(cols_c, 1),
    )

    expected = C_orig * (0.5 - 0.25j)
    if TO_CPU:
        expected = expected.cpu()
    utils.blas_assert_close(C, expected, torch.complex128, reduce_dim=k)

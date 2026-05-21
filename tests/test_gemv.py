import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas
from flag_blas.ops import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
from .conftest import TO_CPU

GEMV_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (3584, 3584),
    (4096, 4096),
    (7168, 7168),
    (8192, 8192),
    (16384, 16384),
    (18432, 18432),
    (1024, 4096),
    (3584, 18944),
    (4096, 14336),
    (6144, 16384),
    (7168, 18432),
    (8192, 28672),
    (16384, 53248),
    (4096, 1024),
    (18944, 3584),
    (14336, 4096),
    (16384, 6144),
    (18432, 7168),
    (28672, 8192),
    (53248, 16384),
    (63, 63),
    (127, 127),
    (255, 255),
    (511, 511),
    (1023, 1023),
    (3583, 3583),
    (4095, 4095),
    (7167, 7167),
    (8191, 8191),
    (1023, 4095),
    (4095, 14335),
    (4095, 1023),
    (14335, 4095),
    # Extreme shapes
    (1, 65536),
    (2, 65536),
    (3, 131071),
    (4, 131072),
    (64, 65536),
    (65536, 1),
    (65536, 2),
    (131071, 3),
    (131072, 4),
    (65536, 64),
]

FP8_GEMV_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (4096, 4096),
    (8192, 8192),
    (1024, 4096),
    (4096, 14336),
    (8192, 28672),
    (4096, 1024),
    (14336, 4096),
    (28672, 8192),
    (64, 65536),
    (65536, 64),
]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def prepare_fp8_gemv_data(m, n, incx, incy, fp8_dtype, y_dtype, device):
    A_f32 = torch.randn(m, n, dtype=torch.float32, device=device)
    A_fp8 = A_f32.to(fp8_dtype)

    x_f32 = torch.randn(m * incx, dtype=torch.float32, device=device)
    x_fp8 = x_f32.to(fp8_dtype)

    y = torch.randn(n * incy, dtype=y_dtype, device=device)
    if TO_CPU:
        return A_fp8, None, x_fp8, None, y, None

    A_col_f32 = A_fp8.float().t().contiguous().t()
    x_f32_ref = x_fp8.float()

    return A_fp8, A_col_f32, x_fp8, x_f32_ref, y, None


def cublas_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return

    dtype = A.dtype
    if dtype == torch.float32:
        func, np_dtype = cublas.sgemv, np.float32
    elif dtype == torch.float64:
        func, np_dtype = cublas.dgemv, np.float64
    elif dtype == torch.complex64:
        func, np_dtype = cublas.cgemv, np.complex64
    elif dtype == torch.complex128:
        func, np_dtype = cublas.zgemv, np.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np_dtype)
    beta_np = np.asarray(beta, dtype=np_dtype)

    func(
        handle,
        trans,
        m,
        n,
        alpha_np.ctypes.data,
        A.data_ptr(),
        lda,
        x.data_ptr(),
        incx,
        beta_np.ctypes.data,
        y.data_ptr(),
        incy,
    )


def cpu_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return to_cpu_blas_tensor(y)

    ref_A = to_cpu_blas_tensor(A)
    ref_x = to_cpu_blas_tensor(x)
    if beta == 0 and incy == 1:
        ref_dtype = torch.complex128 if y.is_complex() else torch.float64
        ref_y = torch.empty(y.shape, dtype=ref_dtype)
    else:
        ref_y = to_cpu_blas_tensor(y)
    func = cpu_blas.zgemv if ref_A.dtype.is_complex else cpu_blas.dgemv

    yout = func(
        alpha,
        ref_A.numpy(),
        ref_x.numpy(),
        beta=beta,
        y=ref_y.numpy(),
        incx=incx,
        incy=incy,
        trans=trans,
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if TO_CPU:
        return cpu_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    ref_y = y.clone()
    if A.dtype in (torch.float16, torch.bfloat16):
        cupy_half_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    else:
        cublas_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    return ref_y


def fp8_gemv_reference(trans, m, n, alpha, A, A_col_ref, x, x_ref, incx, beta, y, incy):
    if TO_CPU:
        return cpu_gemv_reference(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    ref_y = y.float().clone()
    cublas_gemv_reference(
        trans, m, n, alpha, A_col_ref, m, x_ref, incx, beta, ref_y, incy
    )
    return ref_y


def cupy_half_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return

    x_len = n if trans == CUBLAS_OP_N else m
    y_len = m if trans == CUBLAS_OP_N else n

    x_contig = x[::incx][:x_len].contiguous()
    y_contig = y[::incy][:y_len].contiguous()

    CUDA_R_32F = 0
    CUDA_R_16F = 2
    CUDA_R_16BF = 14

    dtype = A.dtype
    if dtype == torch.float16:
        cuda_type = CUDA_R_16F
    elif dtype == torch.bfloat16:
        cuda_type = CUDA_R_16BF
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    alpha_np = np.array([alpha], dtype=np.float32)
    beta_np = np.array([beta], dtype=np.float32)

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    if trans == CUBLAS_OP_N:
        transA = cublas.CUBLAS_OP_T
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = m, 1, n
        lda_c, ldb_c, ldc_c = lda, n, m
    else:
        transA = cublas.CUBLAS_OP_N
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = n, 1, m
        lda_c, ldb_c, ldc_c = lda, m, n

    cublas.gemmEx(
        handle,
        transA,
        transB,
        m_c,
        n_c,
        k_c,
        alpha_np.ctypes.data,
        A.data_ptr(),
        cuda_type,
        lda_c,
        x_contig.data_ptr(),
        cuda_type,
        ldb_c,
        beta_np.ctypes.data,
        y_contig.data_ptr(),
        cuda_type,
        ldc_c,
        CUDA_R_32F,
        0,
    )

    y[::incy][:y_len].copy_(y_contig)


@pytest.mark.sgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_sgemv(m, n, trans, beta):
    dtype, alpha = torch.float32, 1.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.ops.sgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.sgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_sgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.float32, 2.0, 0.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.ops.sgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.dgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dgemv_stride(m, n, trans, incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha, beta = torch.float64, 2.0, 0.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.ops.dgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-14 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.dgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_dgemv(m, n, trans, beta):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha = torch.float64, 1.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.ops.dgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-13 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.cgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_cgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.ops.cgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.cgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_accuracy_cgemv(m, n, trans, beta):
    dtype, alpha = torch.complex64, 1.5 + 0.5j

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.ops.cgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.zgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zgemv_stride(m, n, trans, incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.ops.zgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-14 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.zgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_accuracy_zgemv(m, n, trans, beta):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha = torch.complex128, 1.5 + 0.5j

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.ops.zgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-14 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.hgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_hgemv(m, n, trans, beta):
    dtype, alpha = torch.float16, 1.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, 1, beta, y, 1)
    flag_blas.ops.hgemv(trans, m, n, alpha, A, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        rtol = 1e-3
        atol = 3e-3 * (x_len**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.hgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_hgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.float16, 2.0, 0.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, incx, beta, y, incy)
    flag_blas.ops.hgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        rtol = 1e-3
        atol = 3e-3 * (x_len**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.bfgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_bfgemv(m, n, trans, beta):
    dtype, alpha = torch.bfloat16, 1.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, 1, beta, y, 1)
    flag_blas.ops.bfgemv(trans, m, n, alpha, A, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        rtol = 1.6e-2
        atol = 3e-2 * (x_len**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.bfgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_bfgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.bfloat16, 2.0, 0.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, incx, beta, y, incy)
    flag_blas.ops.bfgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        rtol = 1.6e-2
        atol = 3e-2 * (x_len**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.parametrize("m,n", FP8_GEMV_SHAPES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_fp8_gemv_e4m3(m, n, beta):
    fp8_dtype = torch.float8_e4m3fn
    alpha = 1.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, 1, 1, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, 1, beta, y, 1
    )
    flag_blas.ops.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_fp8_gemv_e4m3_stride(m, n, incx, incy):
    fp8_dtype = torch.float8_e4m3fn
    alpha, beta = 2.0, 0.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, incx, incy, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, incx, beta, y, incy
    )
    flag_blas.ops.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.parametrize("m,n", FP8_GEMV_SHAPES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_fp8_gemv_e5m2(m, n, beta):
    fp8_dtype = torch.float8_e5m2
    alpha = 1.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, 1, 1, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, 1, beta, y, 1
    )
    flag_blas.ops.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_fp8_gemv_e5m2_stride(m, n, incx, incy):
    fp8_dtype = torch.float8_e5m2
    alpha, beta = 2.0, 0.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, incx, incy, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, incx, beta, y, incy
    )
    flag_blas.ops.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.parametrize("m,n", [(256, 256), (1024, 1024), (4096, 4096)])
@pytest.mark.parametrize("y_dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fp8_gemv_output_dtype(m, n, y_dtype):
    fp8_dtype = torch.float8_e4m3fn
    alpha, beta = 1.0, 0.0
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, 1, 1, fp8_dtype, y_dtype, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, 1, beta, y, 1
    )
    flag_blas.ops.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, y_dtype, reduce_dim=m)
    else:
        rtol = 1e-3 if y_dtype == torch.float16 else 1.6e-2
        atol = (3e-3 if y_dtype == torch.float16 else 3e-2) * (m**0.5)
        torch.testing.assert_close(y.float(), ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
def test_fp8_gemv_alpha_zero():
    m, n = 128, 256
    fp8_dtype = torch.float8_e4m3fn

    A_f32 = torch.randn(m, n, dtype=torch.float32, device=flag_blas.device)
    A_fp8 = A_f32.to(fp8_dtype)

    x_f32 = torch.randn(m, dtype=torch.float32, device=flag_blas.device)
    x_fp8 = x_f32.to(fp8_dtype)

    y = torch.randn(n, dtype=torch.float32, device=flag_blas.device)
    y_orig = y.clone()

    flag_blas.ops.fp8_gemv(CUBLAS_OP_T, m, n, 0.0, A_fp8, n, x_fp8, 1, 2.0, y, 1)

    if TO_CPU:
        blas_assert_close(y, to_reference(y_orig * 2.0, upcast=True), torch.float32)
    else:
        torch.testing.assert_close(y, y_orig * 2.0)


@pytest.mark.fp8gemv
def test_fp8_gemv_beta_zero():
    m, n = 128, 256
    fp8_dtype = torch.float8_e4m3fn

    A_f32 = torch.randn(m, n, dtype=torch.float32, device=flag_blas.device)
    A_fp8 = A_f32.to(fp8_dtype)

    x_f32 = torch.randn(m, dtype=torch.float32, device=flag_blas.device)
    x_fp8 = x_f32.to(fp8_dtype)

    y_nan = torch.full((n,), float("nan"), dtype=torch.float32, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=torch.float32, device=flag_blas.device)
    if TO_CPU:
        ref_y_nan = fp8_gemv_reference(
            CUBLAS_OP_T, m, n, 1.0, A_fp8, None, x_fp8, None, 1, 0.0, y_nan, 1
        )

    flag_blas.ops.fp8_gemv(CUBLAS_OP_T, m, n, 1.0, A_fp8, n, x_fp8, 1, 0.0, y_nan, 1)
    flag_blas.ops.fp8_gemv(CUBLAS_OP_T, m, n, 1.0, A_fp8, n, x_fp8, 1, 0.0, y_zero, 1)

    if TO_CPU:
        blas_assert_close(y_nan, ref_y_nan, torch.float32, reduce_dim=m)
        blas_assert_close(
            y_nan, to_reference(y_zero, upcast=True), torch.float32, reduce_dim=m
        )
    else:
        torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.fp8gemv
def test_fp8_gemv_empty():
    fp8_dtype = torch.float8_e4m3fn
    device = flag_blas.device

    m, n = 0, 64
    A_fp8 = torch.zeros(m, n, dtype=fp8_dtype, device=device)
    x_fp8 = torch.zeros(m, dtype=fp8_dtype, device=device)

    y = torch.randn(n, dtype=torch.float32, device=device)
    y_orig = y.clone()

    flag_blas.ops.fp8_gemv(CUBLAS_OP_T, m, n, 1.0, A_fp8, n, x_fp8, 1, 0.0, y, 1)
    if TO_CPU:
        blas_assert_close(y, to_reference(y_orig, upcast=True), torch.float32)
    else:
        torch.testing.assert_close(y, y_orig)


@pytest.mark.fp8gemv
@pytest.mark.parametrize("m,n", [(256, 256), (1024, 1024)])
def test_accuracy_fp8_gemv_mixed_dtype(m, n):
    alpha, beta = 1.0, 0.0
    device = flag_blas.device
    trans = CUBLAS_OP_T

    A_f32 = torch.randn(m, n, dtype=torch.float32, device=device)
    A_fp8_e4m3 = A_f32.to(torch.float8_e4m3fn)

    A_col_f32 = None if TO_CPU else A_fp8_e4m3.float().t().contiguous().t()

    x_f32 = torch.randn(m, dtype=torch.float32, device=device)
    x_fp8_e4m3 = x_f32.to(torch.float8_e4m3fn)
    x_f32_ref = None if TO_CPU else x_fp8_e4m3.float()

    y = torch.zeros(n, dtype=torch.float32, device=device)

    ref_y = fp8_gemv_reference(
        trans,
        m,
        n,
        alpha,
        A_fp8_e4m3,
        A_col_f32,
        x_fp8_e4m3,
        x_f32_ref,
        1,
        beta,
        y,
        1,
    )
    flag_blas.ops.fp8_gemv(trans, m, n, alpha, A_fp8_e4m3, n, x_fp8_e4m3, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)

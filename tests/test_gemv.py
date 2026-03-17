import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas
import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C

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
]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]

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
    
    func(handle, trans, m, n, alpha_np.ctypes.data, A.data_ptr(), lda, 
         x.data_ptr(), incx, beta_np.ctypes.data, y.data_ptr(), incy)

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
        
    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, ref_y, 1)
    flag_blas.ops.sgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)
    
    tol = min(1e-5 * (x_len ** 0.5), 1e-3)
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
    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, ref_y, incy)
    flag_blas.ops.sgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    tol = min(1e-5 * (x_len ** 0.5), 1e-3)
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
    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, ref_y, incy)
    flag_blas.ops.dgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    tol = min(1e-14 * (x_len ** 0.5), 1e-11)
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
        
    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, ref_y, 1)
    flag_blas.ops.dgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)
    
    tol = min(1e-13 * (x_len ** 0.5), 1e-11)
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
    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, ref_y, incy)
    flag_blas.ops.cgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    tol = min(1e-5 * (x_len ** 0.5), 1e-3)
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
        
    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, ref_y, 1)
    flag_blas.ops.cgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)
    
    tol = min(1e-5 * (x_len ** 0.5), 1e-3)
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
    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, ref_y, incy)
    flag_blas.ops.zgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    tol = min(1e-14 * (x_len ** 0.5), 1e-11)
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

    ref_y = y.clone()

    cublas_gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, ref_y, 1)
    flag_blas.ops.zgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    tol = min(1e-14 * (x_len ** 0.5), 1e-11)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


def torch_hgemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Reference implementation using torch.addmv for hgemv (float16)."""
    if m == 0 or n == 0:
        return

    x_len = n if trans == CUBLAS_OP_N else m
    y_len = m if trans == CUBLAS_OP_N else n
    x_vec = x[::incx][:x_len]
    y_slice = y[::incy][:y_len]

    A_mat = A[:m * lda].view(m, lda)[:, :n]

    if trans == CUBLAS_OP_N:
        torch.addmv(y_slice, A_mat, x_vec, alpha=alpha, beta=beta, out=y_slice)
    else:
        torch.addmv(y_slice, A_mat.t(), x_vec, alpha=alpha, beta=beta, out=y_slice)


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

    ref_y = y.clone()

    torch_hgemv_reference(trans, m, n, alpha, A, n, x, 1, beta, ref_y, 1)
    flag_blas.ops.hgemv(trans, m, n, alpha, A, n, x, 1, beta, y, 1)

    tol = min(1e-3 * (x_len ** 0.5), 5e-2)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


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
    ref_y = y.clone()

    torch_hgemv_reference(trans, m, n, alpha, A, n, x, incx, beta, ref_y, incy)
    flag_blas.ops.hgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    tol = min(1e-3 * (x_len ** 0.5), 5e-2)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


def torch_bfgemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Reference implementation using torch.addmv for bfgemv (bfloat16)."""
    if m == 0 or n == 0:
        return

    x_len = n if trans == CUBLAS_OP_N else m
    y_len = m if trans == CUBLAS_OP_N else n
    x_vec = x[::incx][:x_len]
    y_slice = y[::incy][:y_len]

    A_mat = A[:m * lda].view(m, lda)[:, :n]

    if trans == CUBLAS_OP_N:
        torch.addmv(y_slice, A_mat, x_vec, alpha=alpha, beta=beta, out=y_slice)
    else:
        torch.addmv(y_slice, A_mat.t(), x_vec, alpha=alpha, beta=beta, out=y_slice)


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

    ref_y = y.clone()

    torch_bfgemv_reference(trans, m, n, alpha, A, n, x, 1, beta, ref_y, 1)
    flag_blas.ops.bfgemv(trans, m, n, alpha, A, n, x, 1, beta, y, 1)

    tol = min(1e-3 * (x_len ** 0.5), 5e-2)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


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
    ref_y = y.clone()

    torch_bfgemv_reference(trans, m, n, alpha, A, n, x, incx, beta, ref_y, incy)
    flag_blas.ops.bfgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    tol = min(1e-3 * (x_len ** 0.5), 5e-2)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)
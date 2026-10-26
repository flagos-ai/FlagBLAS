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

import ctypes
import os

import numpy as np
import pytest
import torch
from scipy.linalg import blas

import flag_blas
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T

from . import accuracy_utils as utils
from .conftest import TO_CPU

# cuPy/cuBLAS are only available on CUDA-capable vendors; on other backends the
# reference uses the vendor's native BLAS through ctypes (muBLAS/mublasLt on
# MThreads) or falls back to a torch.matmul-based implementation.  Hygon uses a
# device-side fp32 matmul oracle (see cublas_bfgemm_reference).
IS_HYGON = flag_blas.vendor_name == "hygon"

try:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    HAS_CUBLAS = True
except ImportError:
    cp = None
    cublas = None
    HAS_CUBLAS = False
CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14

# T-Head (Zhenwu) cuBLAS rejects a legacy cudaDataType passed where
# cublasGemmEx expects a cublasComputeType_t (CUBLAS_STATUS_NOT_SUPPORTED).
# Use the proper cublasComputeType_t / cublasGemmAlgo_t enums there; other
# vendors keep the original arguments.
if flag_blas.vendor_name == "thead":
    GEMM_COMPUTE_TYPE = cublas.CUBLAS_COMPUTE_32F
    GEMM_ALGO = cublas.CUBLAS_GEMM_DEFAULT
else:
    GEMM_COMPUTE_TYPE = CUDA_R_32F
    GEMM_ALGO = 0


def _load_lib(names, search_dir=None):
    """Load the first shared library that can be resolved."""
    if search_dir is not None:
        for name in names:
            path = os.path.join(search_dir, name)
            if os.path.exists(path):
                try:
                    return ctypes.cdll.LoadLibrary(path)
                except OSError:
                    continue
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    return None


# muBLAS itself has no working bf16 GEMM on the MThreads MUSA driver
# (mublasGemmEx returns MUBLAS_STATUS_ARCH_MISMATCH), but mublasLt handles bf16
# with fp32 accumulation, so it is used as the device-side bf16 reference.
_mublaslt = None
if flag_blas.vendor_name == "mthreads":
    _mublaslt = _load_lib(
        ["libmublasLt.so", "libmublasLt.so.1"],
        os.path.join(os.environ.get("MUSA_HOME", "/usr/local/musa"), "lib"),
    )

_mublaslt_handle = None


def _get_mublaslt_handle():
    """Lazily create the mublasLt handle used by ctypes references."""
    global _mublaslt_handle
    if _mublaslt_handle is None:
        handle = ctypes.c_void_p()
        _mublaslt.mublasLtCreate(ctypes.byref(handle))
        _mublaslt_handle = handle.value
    return _mublaslt_handle


def _lt_matrix_layout(type_code, rows, cols, ld):
    """Create a col-major mublasLt matrix layout descriptor."""
    layout = ctypes.c_void_p()
    _mublaslt.mublasLtMatrixLayoutCreate(
        ctypes.byref(layout),
        ctypes.c_int(type_code),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_int64(ld),
        ctypes.c_int(0),  # MUBLASLT_ORDER_COL
    )
    return layout


_musart_last_error_fn = None
_musart_loaded = False


def _clear_musa_sticky_error():
    """Consume the sticky error muBLAS leaves on the MUSA context."""
    global _musart_last_error_fn, _musart_loaded
    if not _musart_loaded:
        # MUSA 5.x 提供 libmusart.so.5，4.3.x 提供 libmusart.so.4。
        _musart_last_error_fn = _load_lib(
            ("libmusart.so.5", "libmusart.so.4", "libmusart.so"),
            os.path.join(os.environ.get("MUSA_HOME", "/usr/local/musa"), "lib"),
        )
        if _musart_last_error_fn is not None:
            _musart_last_error_fn.musaGetLastError.restype = ctypes.c_int
        _musart_loaded = True
    if _musart_last_error_fn is not None:
        _musart_last_error_fn.musaGetLastError()


def _mublaslt_bfgemm_reference(
    transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
):
    """bf16 reference computed by mublasLtMatmul (fp32 accumulation).

    Transposed operands are re-expressed as physical transposes with col-major
    layouts (col-major (m, k) layout == row-major (k, m) memory), so the call is
    always (N, N) and no matmulDesc attributes are needed.
    """
    if transa == CUBLAS_OP_N:
        A_mat, a_ld = A, m
    else:
        A_mat, a_ld = A.contiguous(), m  # row-major (k, m) memory
    if transb == CUBLAS_OP_N:
        B_mat, b_ld = B, k
    else:
        B_mat, b_ld = B.contiguous(), k  # row-major (n, k) memory

    desc = ctypes.c_void_p()
    # compute type fp32, scalar type fp32 (musaDataType_t code, not compute type)
    _mublaslt.mublasLtMatmulDescCreate(
        ctypes.byref(desc), ctypes.c_int(68), ctypes.c_int(0)
    )
    Ad = _lt_matrix_layout(CUDA_R_16BF, m, k, a_ld)
    Bd = _lt_matrix_layout(CUDA_R_16BF, k, n, b_ld)
    Cd = _lt_matrix_layout(CUDA_R_16BF, m, n, ldc)
    alpha_np = np.asarray(alpha, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)
    matmul = _mublaslt.mublasLtMatmul
    matmul.restype = ctypes.c_int
    matmul.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_uint64] + [ctypes.c_void_p]
    # In-place: D == C, as the test expects the reference to overwrite C.
    matmul(
        ctypes.c_void_p(_get_mublaslt_handle()),
        desc,
        ctypes.c_void_p(alpha_np.ctypes.data),
        ctypes.c_void_p(A_mat.data_ptr()),
        Ad,
        ctypes.c_void_p(B_mat.data_ptr()),
        Bd,
        ctypes.c_void_p(beta_np.ctypes.data),
        ctypes.c_void_p(C.data_ptr()),
        Cd,
        ctypes.c_void_p(C.data_ptr()),
        Cd,
        None,
        None,
        ctypes.c_uint64(0),
        None,
    )
    # Synchronize before reading MUSA errors: mublasLt kernels run asynchronously
    # and a failed launch would otherwise surface as a sticky error on a later
    # torch_musa call.
    if flag_blas.vendor_name == "mthreads":
        torch.musa.synchronize()
    _mublaslt.mublasLtMatmulDescDestroy(desc)
    for layout in (Ad, Bd, Cd):
        _mublaslt.mublasLtMatrixLayoutDestroy(layout)
    if flag_blas.vendor_name == "mthreads":
        _clear_musa_sticky_error()


def cublas_bfgemm_reference(
    transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
):
    if m == 0 or n == 0:
        return

    if IS_HYGON:
        # No usable bf16 gemm entry point in DTK hipBLAS (all combinations of
        # hipblasGemmEx/_v2 return NOT_SUPPORTED / INVALID_ENUM, and no legacy
        # hipblasBgemm); torch bf16 matmul uses rocBLAS bf16 Tensile (fp32
        # accumulation), same as the benchmark baseline.
        Aop = A.t() if transa == CUBLAS_OP_T else A
        Bop = B.t() if transb == CUBLAS_OP_T else B
        acc = Aop.to(torch.float32) @ Bop.to(torch.float32)
        out = alpha * acc
        if beta != 0.0:
            out = out + beta * C.to(torch.float32)
        C.copy_(out.to(torch.bfloat16))
        return

    if HAS_CUBLAS:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        alpha_np = np.array([alpha], dtype=np.float32)
        beta_np = np.array([beta], dtype=np.float32)

        C_fp32 = torch.empty_strided(
            C.shape, C.stride(), dtype=torch.float32, device=C.device
        )
        C_fp32.copy_(C)

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
            C_fp32.data_ptr(),
            CUDA_R_32F,
            ldc,
            GEMM_COMPUTE_TYPE,
            GEMM_ALGO,
        )
        C.copy_(C_fp32.to(torch.bfloat16))
        return

    if _mublaslt is not None:
        _mublaslt_bfgemm_reference(
            transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
        )
        return

    # torch.matmul fallback: rebuild row-major views from the column-major
    # (cuBLAS-layout) inputs, computing in fp32 with bf16 output.
    if transa == CUBLAS_OP_N:
        A_row = A.contiguous()
    else:
        A_row = A.t().contiguous()
    if transb == CUBLAS_OP_N:
        B_row = B.contiguous()
    else:
        B_row = B.t().contiguous()
    C_row = C.contiguous()
    C_row.copy_(
        (
            alpha * torch.matmul(A_row.float(), B_row.float())
            + beta * C_row.float()
        ).to(torch.bfloat16)
    )
    # 就地写回列主序 C：测试逻辑依赖 reference 修改 C_col（C 即 C_col）。
    C.copy_(C_row)


@pytest.mark.bfgemm
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
def test_accuracy_bfgemm(m, n, k, transa, transb):
    dtype, alpha, beta = torch.bfloat16, 1.5, 0.5
    if transa == CUBLAS_OP_N:
        A_col = torch.randn(k, m, dtype=dtype, device=flag_blas.device).t()
        lda_cublas, lda_flag = m, k
    else:
        A_col = torch.randn(m, k, dtype=dtype, device=flag_blas.device).t()
        lda_cublas, lda_flag = k, m
    A_row = A_col.contiguous()

    if transb == CUBLAS_OP_N:
        B_col = torch.randn(n, k, dtype=dtype, device=flag_blas.device).t()
        ldb_cublas, ldb_flag = k, n
    else:
        B_col = torch.randn(k, n, dtype=dtype, device=flag_blas.device).t()
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
    if TO_CPU:
        utils.blas_assert_close(C_row, torch.tensor(C_ref), dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), dtype, reduce_dim=k)


@pytest.mark.bfgemm
def test_bfgemm_alpha_zero():
    m, n, k = 128, 256, 512
    device = flag_blas.device
    C = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    C_orig = C.clone()
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    B = torch.randn(k, n, dtype=torch.bfloat16, device=device)

    flag_blas.bfgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 0.0, A, k, B, n, 2.0, C, n)

    if TO_CPU:
        utils.blas_assert_close(C, C_orig.to("cpu") * 2.0, torch.bfloat16, reduce_dim=k)
    else:
        utils.blas_assert_close(C, C_orig * 2.0, torch.bfloat16, reduce_dim=k)


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

    if TO_CPU:
        utils.blas_assert_close(C_nan, C_zero.to("cpu"), torch.bfloat16, reduce_dim=k)
    else:
        utils.blas_assert_close(C_nan, C_zero, torch.bfloat16, reduce_dim=k)


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
    if TO_CPU:
        utils.blas_assert_close(C, C_orig.to("cpu") * 0.5, torch.bfloat16, reduce_dim=k)
    else:
        utils.blas_assert_close(C, C_orig * 0.5, torch.bfloat16, reduce_dim=k)


@pytest.mark.bfgemm
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 1.5)]
)
def test_bfgemm_alpha_beta(alpha, beta):
    m, n, k = 256, 256, 256
    dtype = torch.bfloat16
    device = flag_blas.device

    A_col = torch.randn(k, m, dtype=dtype, device=device).t()
    A_row = A_col.contiguous()
    B_col = torch.randn(n, k, dtype=dtype, device=device).t()
    B_row = B_col.contiguous()
    C_col = torch.randn(n, m, dtype=dtype, device=device).t()
    C_row = C_col.contiguous()

    cublas_bfgemm_reference(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_col, m, B_col, k, beta, C_col, m
    )
    flag_blas.bfgemm(
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_row, k, B_row, n, beta, C_row, n
    )

    if TO_CPU:
        utils.blas_assert_close(
            C_row,
            torch.tensor(C_col.contiguous().to("cpu")),
            torch.bfloat16,
            reduce_dim=k,
        )
    else:
        utils.blas_assert_close(C_row, C_col.contiguous(), torch.bfloat16, reduce_dim=k)

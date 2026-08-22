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
# reference falls back to muBLAS (mthreads), a ctypes-loaded cuBLAS-compatible
# library (iluvatar), or a torch.matmul-based implementation.
try:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    HAS_CUBLAS = True
except ImportError:
    cp = None
    cublas = None
    HAS_CUBLAS = False

_CUBLAS_VENDORS = ("nvidia", "iluvatar")


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


# Native BLAS library of the current vendor: (lib, function-prefix).
_ctypes_blas = None
if flag_blas.vendor_name in _CUBLAS_VENDORS and not HAS_CUBLAS:
    _cuda_home = os.environ.get("CUDA_HOME")
    _lib = _load_lib(
        ["libcublas.so", "libcublas.so.12", "libcublas.so.11"],
        os.path.join(_cuda_home, "lib64") if _cuda_home else None,
    )
    if _lib is not None:
        _ctypes_blas = (_lib, "cublas")
elif flag_blas.vendor_name == "mthreads":
    _lib = _load_lib(
        ["libmublas.so", "libmublas.so.1"],
        os.path.join(os.environ.get("MUSA_HOME", "/usr/local/musa"), "lib"),
    )
    if _lib is not None:
        _ctypes_blas = (_lib, "mublas")

_ctypes_sgemm = None
_blas_handle = None
if _ctypes_blas is not None:
    _lib, _prefix = _ctypes_blas
    _ctypes_sgemm = getattr(_lib, f"{_prefix}Sgemm", None)
    if _ctypes_sgemm is not None:
        _ctypes_sgemm.restype = ctypes.c_int
        _ctypes_sgemm.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_int,  # transa
            ctypes.c_int,  # transb
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.c_int,  # k
            ctypes.c_void_p,  # alpha
            ctypes.c_void_p,  # A
            ctypes.c_int,  # lda
            ctypes.c_void_p,  # B
            ctypes.c_int,  # ldb
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # C
            ctypes.c_int,  # ldc
        ]


def _get_blas_handle():
    """Lazily create the native BLAS handle used by ctypes references."""
    global _blas_handle
    if _blas_handle is None:
        lib, prefix = _ctypes_blas
        create = getattr(lib, f"{prefix}Create_v2", None) or getattr(
            lib, f"{prefix}Create"
        )
        create.restype = ctypes.c_int
        create.argtypes = [ctypes.c_void_p]
        handle = ctypes.c_void_p()
        create(ctypes.byref(handle))
        _blas_handle = handle.value
    return _blas_handle


def _native_op(code):
    """Map a cuBLAS op code (N=0/T=1/C=2) to the native BLAS library's codes.

    cuBLAS-compatible libraries (iluvatar) reuse 0/1/2, but muBLAS uses
    MUBLAS_OP_N=111/MUBLAS_OP_T=112/MUBLAS_OP_C=113.
    """
    if _ctypes_blas is not None and _ctypes_blas[1] == "mublas":
        return {CUBLAS_OP_N: 111, CUBLAS_OP_T: 112, 2: 113}[code]
    return code


def cublas_sgemm_reference(
    transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
):
    if m == 0 or n == 0:
        return

    alpha_np = np.asarray(alpha, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)

    if HAS_CUBLAS:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
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
        return

    if _ctypes_sgemm is not None:
        _ctypes_sgemm(
            ctypes.c_void_p(_get_blas_handle()),
            ctypes.c_int(_native_op(transa)),
            ctypes.c_int(_native_op(transb)),
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
            ctypes.c_void_p(alpha_np.ctypes.data),
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_int(lda),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_int(ldb),
            ctypes.c_void_p(beta_np.ctypes.data),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_int(ldc),
        )
        if _ctypes_blas is not None and _ctypes_blas[1] == "mublas":
            _clear_musa_sticky_error()
        return

    # torch.matmul fallback: rebuild row-major views from the column-major
    # (cuBLAS-layout) inputs.
    if transa == CUBLAS_OP_N:
        A_row = A.t()
    else:
        A_row = A
    if transb == CUBLAS_OP_N:
        B_row = B.t()
    else:
        B_row = B
    C_row = C.t()
    C_row.copy_(alpha * torch.matmul(A_row, B_row) + beta * C_row)


CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14

_musart_last_error_fn = None


def _clear_musa_sticky_error():
    """Consume the sticky error muBLAS leaves on the MUSA context.

    When called through ctypes, muBLAS may hit driver APIs the installed MUSA
    driver does not implement (e.g. on large non-transposed shapes) and leave
    a sticky musaErrorNotSupported on the context. The error is only reported
    to the *next* torch_musa API call (e.g. a clone), which would otherwise
    fail spuriously. Reading musaGetLastError clears it.
    """
    global _musart_last_error_fn
    if _musart_last_error_fn is None:
        _musart_last_error_fn = ctypes.cdll.LoadLibrary("libmusart.so.5")
        _musart_last_error_fn.musaGetLastError.restype = ctypes.c_int
    _musart_last_error_fn.musaGetLastError()



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

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.cgemm import (
    ScalarType,
    _cgemm_dot_kernel,
    _complex_scalar_parts,
    _validate_cgemm_args,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._nvidia.hopper.ops.gemm import sgemm as _sgemm_hopper

try:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas
except ImportError:
    cp = None
    cublas = None

logger = logging.getLogger(__name__)

_CGEMM_WORKSPACE = {"key": None, "buffers": None}
_CGEMM_STREAMS = {"key": None, "streams": None}


def _try_cgemm_cublas(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha_r: float,
    alpha_i: float,
    beta_r: float,
    beta_i: float,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    if cp is None or cublas is None:
        return False
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        return False
    if ldc != n or C.shape[0] < m or C.shape[1] != n:
        return False

    if transa == 0:
        if A.shape[0] < m or A.shape[1] < k or lda != A.shape[1]:
            return False
    elif A.shape[0] < k or A.shape[1] < m or lda != A.shape[1]:
        return False

    if transb == 0:
        if B.shape[0] < k or B.shape[1] < n or ldb != B.shape[1]:
            return False
    elif B.shape[0] < n or B.shape[1] < k or ldb != B.shape[1]:
        return False

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    cublas.setMathMode(handle, 0)
    alpha = np.asarray(complex(alpha_r, alpha_i), dtype=np.complex64)
    beta = np.asarray(complex(beta_r, beta_i), dtype=np.complex64)

    # Row-major C = op(A) @ op(B) is column-major C.T = op(B).T @ op(A).T.
    cublas.cgemm(
        handle,
        transb,
        transa,
        n,
        m,
        k,
        alpha.ctypes.data,
        B.data_ptr(),
        ldb,
        A.data_ptr(),
        lda,
        beta.ctypes.data,
        C.data_ptr(),
        ldc,
    )
    return True


def _try_cgemm_torch_mm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        return False
    if ldc != n or C.shape[0] < m or C.shape[1] != n:
        return False

    if transa == 0:
        if A.shape[0] < m or A.shape[1] < k or lda != A.shape[1]:
            return False
        a = A[:m, :k]
    else:
        if A.shape[0] < k or A.shape[1] < m or lda != A.shape[1]:
            return False
        a = A[:k, :m].t()

    if transb == 0:
        if B.shape[0] < k or B.shape[1] < n or ldb != B.shape[1]:
            return False
        b = B[:k, :n]
    else:
        if B.shape[0] < n or B.shape[1] < k or ldb != B.shape[1]:
            return False
        b = B[:n, :k].t()

    torch.mm(a, b, out=C[:m, :n])
    return True

@triton.jit
def _cgemm_split_sum_op_kernel(
    src,
    dst_r,
    dst_i,
    dst_sum,
    total,
    cols: tl.constexpr,
    ld: tl.constexpr,
    TRANS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    row = offsets // cols
    col = offsets - row * cols
    if TRANS == 0:
        src_offsets = row * ld + col
    else:
        src_offsets = col * ld + row
    real = tl.load(src + 2 * src_offsets, mask=mask, other=0.0)
    imag = tl.load(src + 2 * src_offsets + 1, mask=mask, other=0.0)
    tl.store(dst_r + offsets, real, mask=mask)
    tl.store(dst_i + offsets, imag, mask=mask)
    tl.store(dst_sum + offsets, real + imag, mask=mask)


@triton.jit
def _cgemm_merge_3m_kernel(dst, prod_r, prod_i, prod_sum, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    real_prod = tl.load(prod_r + offsets, mask=mask, other=0.0)
    imag_prod = tl.load(prod_i + offsets, mask=mask, other=0.0)
    sum_prod = tl.load(prod_sum + offsets, mask=mask, other=0.0)
    tl.store(dst + 2 * offsets, real_prod - imag_prod, mask=mask)
    tl.store(dst + 2 * offsets + 1, sum_prod - real_prod - imag_prod, mask=mask)


def _get_cgemm_workspace(A: torch.Tensor, m: int, n: int, k: int):
    key = (A.device, m, n, k)
    if _CGEMM_WORKSPACE["key"] != key:
        _CGEMM_WORKSPACE["key"] = key
        _CGEMM_WORKSPACE["buffers"] = (
            torch.empty((m, k), device=A.device, dtype=torch.float32),
            torch.empty((m, k), device=A.device, dtype=torch.float32),
            torch.empty((m, k), device=A.device, dtype=torch.float32),
            torch.empty((k, n), device=A.device, dtype=torch.float32),
            torch.empty((k, n), device=A.device, dtype=torch.float32),
            torch.empty((k, n), device=A.device, dtype=torch.float32),
            torch.empty((m, n), device=A.device, dtype=torch.float32),
            torch.empty((m, n), device=A.device, dtype=torch.float32),
            torch.empty((m, n), device=A.device, dtype=torch.float32),
        )
    return _CGEMM_WORKSPACE["buffers"]


def _get_cgemm_streams(device):
    if _CGEMM_STREAMS["key"] != device:
        _CGEMM_STREAMS["key"] = device
        _CGEMM_STREAMS["streams"] = tuple(
            torch.cuda.Stream(device=device) for _ in range(3)
        )
    return _CGEMM_STREAMS["streams"]


def _launch_cgemm_pack_sgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
) -> None:
    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_cgemm_workspace(A, m, n, k)
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    split_grid_a = (triton.cdiv(m * k, 1024),)
    split_grid_b = (triton.cdiv(k * n, 1024),)
    merge_grid = (triton.cdiv(m * n, 1024),)

    _cgemm_split_sum_op_kernel[split_grid_a](
        A_real, Ar, Ai, As, m * k, k, lda, transa, BLOCK=1024
    )
    _cgemm_split_sum_op_kernel[split_grid_b](
        B_real, Br, Bi, Bs, k * n, n, ldb, transb, BLOCK=1024
    )

    current_stream = torch.cuda.current_stream(device=A.device)
    stream_r, stream_i, stream_sum = _get_cgemm_streams(A.device)
    stream_r.wait_stream(current_stream)
    stream_i.wait_stream(current_stream)
    stream_sum.wait_stream(current_stream)

    with torch.cuda.stream(stream_r):
        _sgemm_hopper(0, 0, m, n, k, 1.0, Ar, k, Br, n, 0.0, prod_r, n)
    with torch.cuda.stream(stream_i):
        _sgemm_hopper(0, 0, m, n, k, 1.0, Ai, k, Bi, n, 0.0, prod_i, n)
    with torch.cuda.stream(stream_sum):
        _sgemm_hopper(0, 0, m, n, k, 1.0, As, k, Bs, n, 0.0, prod_sum, n)

    current_stream.wait_stream(stream_r)
    current_stream.wait_stream(stream_i)
    current_stream.wait_stream(stream_sum)
    _cgemm_merge_3m_kernel[merge_grid](
        C_real, prod_r, prod_i, prod_sum, m * n, BLOCK=1024
    )


def _try_cgemm_pack_sgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if ldc != n or max(m, n, k) < 256:
        return False
    _launch_cgemm_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C)
    return True


def _select_cgemm_hopper_config(transa: int, transb: int, m: int, n: int, k: int):
    max_dim = max(m, n, k)
    min_dim = min(m, n, k)

    if max_dim <= 32:
        return 16, 16, 16, 4, 1, None
    if max_dim <= 128:
        return 16, 32, 32, 4, 4, None
    if max_dim <= 512:
        if transa == 0 and transb == 0:
            return 32, 64, 32, 4, 4, None
        return 64, 32, 32, 4, 4, None
    if max_dim <= 1024:
        if transa == 0 and transb == 0:
            return 32, 64, 32, 4, 8, None
        if transa == 0:
            return 32, 64, 32, 4, 8, None
        return 64, 32, 32, 4, 8, None

    if transa == 0 and transb == 0:
        if min_dim >= 4096:
            return 64, 64, 32, 4, 4, None
        return 32, 64, 32, 4, 8, None

    if transa == 0:
        if min_dim >= 4096:
            return 64, 64, 32, 4, 4, None
        return 32, 64, 32, 4, 8, None

    if transb == 0:
        if min_dim >= 4096:
            return 64, 64, 32, 4, 4, None
        return 64, 32, 32, 4, 8, None

    if min_dim >= 4096:
        return 64, 64, 32, 4, 4, None
    return 64, 32, 32, 4, 8, None


def cgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    _validate_cgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha_r, alpha_i = _complex_scalar_parts(alpha)
    beta_r, beta_i = _complex_scalar_parts(beta)

    if m == 0 or n == 0 or k == 0 or (alpha_r == 0.0 and alpha_i == 0.0):
        if beta_r == 0.0 and beta_i == 0.0:
            C.zero_()
        elif not (beta_r == 1.0 and beta_i == 0.0):
            C.mul_(complex(beta_r, beta_i))
        return

    beta_is_zero = beta_r == 0.0 and beta_i == 0.0
    alpha_is_one = alpha_r == 1.0 and alpha_i == 0.0

    if _try_cgemm_cublas(
        transa,
        transb,
        m,
        n,
        k,
        alpha_r,
        alpha_i,
        beta_r,
        beta_i,
        A,
        lda,
        B,
        ldb,
        C,
        ldc,
    ):
        return

    if (
        alpha_is_one
        and beta_is_zero
        and _try_cgemm_torch_mm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)
    ):
        return

    if (
        alpha_is_one
        and beta_is_zero
        and _try_cgemm_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)
    ):
        return

    block_m, block_n, block_k, num_warps, group_m, maxnreg = (
        _select_cgemm_hopper_config(transa, transb, m, n, k)
    )
    launch_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    with torch_device_fn.device(A.device):
        _cgemm_dot_kernel[grid](
            A_real,
            B_real,
            C_real,
            alpha_r,
            alpha_i,
            beta_r,
            beta_i,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            transa,
            transb,
            beta_is_zero,
            alpha_is_one,
            **launch_kwargs,
        )

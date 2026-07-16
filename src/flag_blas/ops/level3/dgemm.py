import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

_DGEMM_KEY = ["m", "n", "k", "TRANS_A", "TRANS_B", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dgemm"),
    key=_DGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _dgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    beta: tl.float64,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    for k_start in range(0, k, BLOCK_K):
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k_start + kk
            mask_k = k_idx < k

            if TRANS_A == 0:
                a_ptrs = a_ptr + offs_m * lda + k_idx
            else:
                a_ptrs = a_ptr + k_idx * lda + offs_m

            if TRANS_B == 0:
                b_ptrs = b_ptr + k_idx * ldb + offs_n
            else:
                b_ptrs = b_ptr + offs_n * ldb + k_idx

            a_vals = tl.load(a_ptrs, mask=mask_m & mask_k, other=0.0).to(tl.float64)
            b_vals = tl.load(b_ptrs, mask=mask_n & mask_k, other=0.0).to(tl.float64)
            acc += a_vals[:, None] * b_vals[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]

    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float64)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


def _validate_dgemm_args(
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
) -> None:
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.float64
    assert B.dtype == torch.float64
    assert C.dtype == torch.float64
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

    def required_numel(rows: int, cols: int, ld: int) -> int:
        if rows == 0 or cols == 0:
            return 0
        return (rows - 1) * ld + cols

    if transa == CUBLAS_OP_N:
        if m > 0 and k > 0:
            assert lda >= k
        assert A.numel() >= required_numel(m, k, lda)
    else:
        if m > 0 and k > 0:
            assert lda >= m
        assert A.numel() >= required_numel(k, m, lda)

    if transb == CUBLAS_OP_N:
        if n > 0 and k > 0:
            assert ldb >= n
        assert B.numel() >= required_numel(k, n, ldb)
    else:
        if n > 0 and k > 0:
            assert ldb >= k
        assert B.numel() >= required_numel(n, k, ldb)

    if m > 0 and n > 0:
        assert ldc >= n
    assert C.numel() >= required_numel(m, n, ldc)


def dgemm(
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
    _validate_dgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    beta_is_zero = beta == 0.0
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(A.device):
        _dgemm_kernel[grid](
            A,
            B,
            C,
            alpha,
            beta,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            transa,
            transb,
            beta_is_zero,
        )

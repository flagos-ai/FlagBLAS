import logging

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level3.dgemm import (
    ScalarType,
    _validate_dgemm_args,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_DGEMM_HOPPER_KEY = ["m", "n", "k", "TRANS_A", "TRANS_B", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dgemm_hopper"),
    key=_DGEMM_HOPPER_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _dgemm_hopper_kernel(
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



@triton.jit
def _dgemm_nn_dot_kernel(
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
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k_start in range(0, k, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < k
        a = tl.load(
            a_ptr + offs_m[:, None] * lda + cur_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + cur_k[:, None] * ldb + offs_n[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, acc, mask=c_mask)
        else:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float64)
        if ALPHA_IS_ONE:
            tl.store(c_ptrs, acc + beta * c_vals, mask=c_mask)
        else:
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)



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
    alpha_is_one = alpha == 1.0

    with torch_device_fn.device(A.device):
        if transa == 0 and transb == 0:
            max_dim = max(m, n, k)
            min_dim = min(m, n, k)
            is_aligned = (m % 64 == 0) and (n % 64 == 0) and (k % 64 == 0)

            maxnreg = None
            if max_dim <= 32:
                block_m, block_n, block_k, num_warps, group_m = 32, 16, 16, 4, 1
            elif max_dim <= 128:
                block_m, block_n, block_k, num_warps, group_m = 16, 32, 32, 4, 1
            elif max_dim <= 256:
                block_m, block_n, block_k, num_warps, group_m = 16, 32, 64, 4, 1
            elif max_dim <= 512:
                if is_aligned:
                    block_m, block_n, block_k, num_warps, group_m = 32, 32, 32, 4, 1
                else:
                    block_m, block_n, block_k, num_warps, group_m = 64, 32, 32, 4, 1
            elif max_dim <= 1024:
                if is_aligned:
                    block_m, block_n, block_k, num_warps, group_m = 64, 128, 64, 8, 1
                    maxnreg = 224
                else:
                    block_m, block_n, block_k, num_warps, group_m = 64, 128, 16, 4, 1
            elif max_dim <= 1536:
                block_m, block_n, block_k, num_warps, group_m = 64, 32, 32, 4, 1
                maxnreg = 128
            elif max_dim <= 2048:
                block_m, block_n, block_k, num_warps, group_m = 64, 128, 32, 4, 8
            elif max_dim == 4095:
                block_m, block_n, block_k, num_warps, group_m = 64, 128, 16, 4, 16
            elif max_dim < 4096 and min_dim >= 2048:
                block_m, block_n, block_k, num_warps, group_m = 64, 128, 32, 4, 2
            elif max_dim <= 4096:
                block_m, block_n, block_k, num_warps, group_m = 64, 128, 32, 4, 8
            elif min_dim >= 8192:
                block_m, block_n, block_k, num_warps, group_m = 64, 128, 32, 4, 8
            elif min_dim >= 6144:
                block_m, block_n, block_k, num_warps, group_m = 128, 64, 32, 4, 8
            else:
                block_m, block_n, block_k, num_warps, group_m = 64, 128, 32, 4, 8

            launch_kwargs = {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
                "num_warps": num_warps,
            }
            if maxnreg is not None:
                launch_kwargs["maxnreg"] = maxnreg

            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            _dgemm_nn_dot_kernel[grid](
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
                beta_is_zero,
                alpha_is_one,
                **launch_kwargs,
            )
            return

        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )
        _dgemm_hopper_kernel[grid](
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

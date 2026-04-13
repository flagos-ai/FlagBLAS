import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] + offs_k[None, :] * lda)

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, offset_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k_curr = k_idx * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < m) & (offs_k_curr[None, :] < k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last")
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * lda
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        tl.store(c_block_ptr, alpha * acc, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = acc * alpha + beta * c_vals
        tl.store(c_block_ptr, result.to(tl.float32), boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            a_t = tl.trans(a)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            a_t = tl.trans(a)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[:, None]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            a_t = tl.trans(a)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_n[:, None] & mask_k[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            a_t = tl.trans(a)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)

    acc = tl.trans(acc_t)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float32))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float32))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float32), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float32), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_k[:, None] * lda + offs_am[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a_t = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[None, :]
        b_mask_base = mask_n[:, None]

        for i in range(k_full_iters):
            a_t = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b_t = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_k[:, None] & mask_m[None, :]
            b_mask_tail = mask_n[:, None] & mask_k[None, :]
            a_t = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b_t = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)

    acc = tl.trans(acc_t)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float32))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float32))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float32), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float32), mask=c_mask)


def sgemm(
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
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

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
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _sgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _sgemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _sgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _sgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )


_HGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_tn"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nt"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_tt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)
    b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


def hgemm(
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
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert C.dtype == torch.float16
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

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
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _hgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _hgemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _hgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _hgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )


_BFGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_nn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_tn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nt"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_tt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)
    b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n

    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_am < m
        mask_n = offs_bn < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for i in range(k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16))
    else:
        mask_m = offs_cm < m
        mask_n = offs_cn < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.bfloat16), mask=c_mask)


def bfgemm(
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
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

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
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _bfgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _bfgemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _bfgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _bfgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )


FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_FP8GEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8gemm"),
    key=_FP8GEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _fp8gemm_nn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, offset_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last")
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8gemm"),
    key=_FP8GEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _fp8gemm_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] + offs_k[None, :] * lda)

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, offset_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k_curr = k_idx * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < m) & (offs_k_curr[None, :] < k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last")
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * lda
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8gemm"),
    key=_FP8GEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _fp8gemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    b_ptrs = b_ptr + (offs_k[:, None] + offs_n[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last")
        offs_k_curr = k_idx * BLOCK_K + offs_k
        b_mask = (offs_k_curr[:, None] < k) & (offs_n[None, :] < n)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy="evict_last")
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_ptrs += BLOCK_K

    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8gemm"),
    key=_FP8GEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _fp8gemm_tt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)

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

    a_ptrs = a_ptr + (offs_m[:, None] + offs_k[None, :] * lda)
    b_ptrs = b_ptr + (offs_k[:, None] + offs_n[None, :] * ldb)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k_curr = k_idx * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < m) & (offs_k_curr[None, :] < k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")
        b_mask = (offs_k_curr[:, None] < k) & (offs_n[None, :] < n)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy="evict_last")
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(c_ptr.dtype.element_ty)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


def fp8gemm(
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
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype in FP8_DTYPES
    assert B.dtype in FP8_DTYPES
    assert C.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert m % 16 == 0
    assert n % 16 == 0
    assert k % 16 == 0
    assert A.data_ptr() % 16 == 0
    assert B.data_ptr() % 16 == 0
    assert C.data_ptr() % 16 == 0

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    if alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    if transa == CUBLAS_OP_N:
        assert lda >= k
        assert A.numel() >= m * lda
    else:
        assert lda >= m
        assert A.numel() >= k * lda

    if transb == CUBLAS_OP_N:
        assert ldb >= n
        assert B.numel() >= k * ldb
    else:
        assert ldb >= k
        assert B.numel() >= n * ldb

    assert ldc >= n
    assert C.numel() >= m * ldc

    beta_is_zero = beta == 0.0

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _fp8gemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _fp8gemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _fp8gemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _fp8gemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )

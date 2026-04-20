import logging

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.ops.level3.gemm import (
    _sgemm_nn_kernel,
    _sgemm_tn_kernel,
    _sgemm_tt_kernel,
    _hgemm_nn_kernel,
    _hgemm_tn_kernel,
    _hgemm_nt_kernel,
    _hgemm_tt_kernel,
    _bfgemm_nn_kernel,
    _bfgemm_tn_kernel,
    _bfgemm_nt_kernel,
    _bfgemm_tt_kernel,
    _fp8gemm_nn_kernel,
    _fp8gemm_tn_kernel,
    _fp8gemm_nt_kernel,
    _fp8gemm_tt_kernel,
    _HGEMM_KEY,
    _BFGEMM_KEY,
    FP8_DTYPES,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
)

logger = logging.getLogger(__name__)

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nn_kernel2(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(a_t, b_t, acc, out_dtype=tl.float32, allow_tf32=False)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tn_kernel2(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[k, m], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([i * BLOCK_K, pid_m * BLOCK_M])
        b_t = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(tl.trans(a_t), b_t, acc, out_dtype=tl.float32, allow_tf32=False)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nt_kernel2(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[n, k], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([pid_n * BLOCK_N, i * BLOCK_K])

        acc = tl.dot(a_t, tl.trans(b_t), acc, out_dtype=tl.float32, allow_tf32=False)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tt_kernel2(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[k, m], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[n, k], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([i * BLOCK_K, pid_m * BLOCK_M])
        b_t = b_desc.load([pid_n * BLOCK_N, i * BLOCK_K])

        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)

    acc = tl.trans(acc_t)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


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

    strides_aligned = (lda % 8 == 0) and (ldb % 8 == 0) and (ldc % 8 == 0)
    ptrs_aligned = (
        (A.data_ptr() % 16 == 0)
        and (B.data_ptr() % 16 == 0)
        and (C.data_ptr() % 16 == 0)
    )
    aligned = strides_aligned and ptrs_aligned

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            if aligned:
                _sgemm_nn_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                if m > 1024 and n > 1024 and k > 1024:
                    pad_m = (16 - (m % 16)) % 16
                    pad_n = (16 - (n % 16)) % 16
                    pad_k = (16 - (k % 16)) % 16

                    A_2d = A.view(-1, lda)
                    B_2d = B.view(-1, ldb)
                    C_2d = C.view(-1, ldc)

                    A_padded = torch.nn.functional.pad(
                        A_2d[:m, :k], (0, pad_k, 0, pad_m)
                    )
                    B_padded = torch.nn.functional.pad(
                        B_2d[:k, :n], (0, pad_n, 0, pad_k)
                    )
                    C_padded = torch.nn.functional.pad(
                        C_2d[:m, :n], (0, pad_n, 0, pad_m)
                    )

                    m_pad = m + pad_m
                    n_pad = n + pad_n
                    k_pad = k + pad_k
                    lda_pad = k_pad
                    ldb_pad = n_pad
                    ldc_pad = n_pad

                    grid_pad = lambda meta: (
                        triton.cdiv(m_pad, meta["BLOCK_M"])
                        * triton.cdiv(n_pad, meta["BLOCK_N"]),
                    )

                    _sgemm_nn_kernel2[grid_pad](
                        A_padded,
                        B_padded,
                        C_padded,
                        alpha,
                        beta,
                        m_pad,
                        n_pad,
                        k_pad,
                        lda_pad,
                        ldb_pad,
                        ldc_pad,
                        beta_is_zero,
                    )

                    C_2d[:m, :n] = C_padded[:m, :n]
                else:
                    _sgemm_nn_kernel[grid](
                        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                    )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if aligned:
                _sgemm_tn_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                if m > 1024 and n > 1024 and k > 1024:
                    pad_m = (16 - (m % 16)) % 16
                    pad_n = (16 - (n % 16)) % 16
                    pad_k = (16 - (k % 16)) % 16

                    A_2d = A.view(-1, lda)
                    B_2d = B.view(-1, ldb)
                    C_2d = C.view(-1, ldc)

                    A_padded = torch.nn.functional.pad(
                        A_2d[:k, :m], (0, pad_m, 0, pad_k)
                    )
                    B_padded = torch.nn.functional.pad(
                        B_2d[:k, :n], (0, pad_n, 0, pad_k)
                    )
                    C_padded = torch.nn.functional.pad(
                        C_2d[:m, :n], (0, pad_n, 0, pad_m)
                    )

                    m_pad = m + pad_m
                    n_pad = n + pad_n
                    k_pad = k + pad_k
                    lda_pad = m_pad
                    ldb_pad = n_pad
                    ldc_pad = n_pad

                    grid_pad = lambda meta: (
                        triton.cdiv(m_pad, meta["BLOCK_M"])
                        * triton.cdiv(n_pad, meta["BLOCK_N"]),
                    )

                    _sgemm_tn_kernel2[grid_pad](
                        A_padded,
                        B_padded,
                        C_padded,
                        alpha,
                        beta,
                        m_pad,
                        n_pad,
                        k_pad,
                        lda_pad,
                        ldb_pad,
                        ldc_pad,
                        beta_is_zero,
                    )

                    C_2d[:m, :n] = C_padded[:m, :n]
                else:
                    _sgemm_tn_kernel[grid](
                        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                    )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            B_transposed = B.t().contiguous()
            ldb_new = n

            strides_aligned_new = (
                (lda % 8 == 0) and (ldb_new % 8 == 0) and (ldc % 8 == 0)
            )
            ptrs_aligned_new = (
                (A.data_ptr() % 16 == 0)
                and (B_transposed.data_ptr() % 16 == 0)
                and (C.data_ptr() % 16 == 0)
            )
            aligned_new = strides_aligned_new and ptrs_aligned_new

            if aligned_new:
                _sgemm_nn_kernel2[grid](
                    A,
                    B_transposed,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb_new,
                    ldc,
                    beta_is_zero,
                )
            else:
                if m > 1024 and n > 1024 and k > 1024:
                    pad_m = (16 - (m % 16)) % 16
                    pad_n = (16 - (n % 16)) % 16
                    pad_k = (16 - (k % 16)) % 16

                    A_2d = A.view(-1, lda)
                    B_2d = B_transposed.view(-1, ldb_new)
                    C_2d = C.view(-1, ldc)

                    A_padded = torch.nn.functional.pad(
                        A_2d[:m, :k], (0, pad_k, 0, pad_m)
                    )
                    B_padded = torch.nn.functional.pad(
                        B_2d[:k, :n], (0, pad_n, 0, pad_k)
                    )
                    C_padded = torch.nn.functional.pad(
                        C_2d[:m, :n], (0, pad_n, 0, pad_m)
                    )

                    m_pad = m + pad_m
                    n_pad = n + pad_n
                    k_pad = k + pad_k
                    lda_pad = k_pad
                    ldb_pad = n_pad
                    ldc_pad = n_pad

                    grid_pad = lambda meta: (
                        triton.cdiv(m_pad, meta["BLOCK_M"])
                        * triton.cdiv(n_pad, meta["BLOCK_N"]),
                    )

                    _sgemm_nn_kernel2[grid_pad](
                        A_padded,
                        B_padded,
                        C_padded,
                        alpha,
                        beta,
                        m_pad,
                        n_pad,
                        k_pad,
                        lda_pad,
                        ldb_pad,
                        ldc_pad,
                        beta_is_zero,
                    )

                    C_2d[:m, :n] = C_padded[:m, :n]
                else:
                    _sgemm_nn_kernel[grid](
                        A,
                        B_transposed,
                        C,
                        alpha,
                        beta,
                        m,
                        n,
                        k,
                        lda,
                        ldb_new,
                        ldc,
                        beta_is_zero,
                    )
        else:
            if aligned:
                _sgemm_tt_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                if m > 1024 and n > 1024 and k > 1024:
                    pad_m = (16 - (m % 16)) % 16
                    pad_n = (16 - (n % 16)) % 16
                    pad_k = (16 - (k % 16)) % 16

                    A_2d = A.view(-1, lda)
                    B_2d = B.view(-1, ldb)
                    C_2d = C.view(-1, ldc)

                    A_padded = torch.nn.functional.pad(
                        A_2d[:k, :m], (0, pad_m, 0, pad_k)
                    )
                    B_padded = torch.nn.functional.pad(
                        B_2d[:n, :k], (0, pad_k, 0, pad_n)
                    )
                    C_padded = torch.nn.functional.pad(
                        C_2d[:m, :n], (0, pad_n, 0, pad_m)
                    )

                    m_pad = m + pad_m
                    n_pad = n + pad_n
                    k_pad = k + pad_k
                    lda_pad = m_pad
                    ldb_pad = k_pad
                    ldc_pad = n_pad

                    grid_pad = lambda meta: (
                        triton.cdiv(m_pad, meta["BLOCK_M"])
                        * triton.cdiv(n_pad, meta["BLOCK_N"]),
                    )

                    _sgemm_tt_kernel2[grid_pad](
                        A_padded,
                        B_padded,
                        C_padded,
                        alpha,
                        beta,
                        m_pad,
                        n_pad,
                        k_pad,
                        lda_pad,
                        ldb_pad,
                        ldc_pad,
                        beta_is_zero,
                    )

                    C_2d[:m, :n] = C_padded[:m, :n]
                else:
                    _sgemm_tt_kernel[grid](
                        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nn_kernel2(
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

    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        result = acc * alpha
        tl.store(c_block_ptr, result.to(tl.float16), boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = acc * alpha + beta * c_vals
        tl.store(c_block_ptr, result.to(tl.float16), boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn2"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nn_kernel3(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@triton.jit
def _hgemm_nn_kernel4(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = desc_a.load([offs_m, offs_k])
        b = desc_b.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)


@libentry()
@triton.jit
def _hgemm_tn_kernel3(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K

        a_t = desc_a.load([offs_k, offs_m])
        a = tl.trans(a_t)

        b = desc_b.load([offs_k, offs_n])

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_tn2"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_tn_kernel2(
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

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(k, m),
        strides=(lda, 1),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)

        a = tl.trans(a_t)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        tl.store(c_block_ptr, result)
    else:
        c_vals = tl.load(c_block_ptr).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        tl.store(c_block_ptr, result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nt"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nt_kernel2(
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

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, offset_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_tt_kernel2(
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

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(1, lda),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        tl.store(c_block_ptr, result)
    else:
        c_vals = tl.load(c_block_ptr).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        tl.store(c_block_ptr, result)


@libentry()
@triton.jit
def _hgemm_tt_kernel3(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = desc_a.load([offs_k, offs_m])
        a = tl.trans(a_t)
        b_t = desc_b.load([offs_n, offs_k])
        b = tl.trans(b_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)


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

    strides_aligned = (lda % 8 == 0) and (ldb % 8 == 0) and (ldc % 8 == 0)
    ptrs_aligned = (
        (A.data_ptr() % 16 == 0)
        and (B.data_ptr() % 16 == 0)
        and (C.data_ptr() % 16 == 0)
    )
    aligned = strides_aligned and ptrs_aligned
    use_nn_kernel3 = aligned and (m * n > 2048 * 2048) and min(m, n) >= 64
    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            is_skinny = (m >= 16384 and max(n, k) <= 2048) or (
                n >= 16384 and max(m, k) <= 2048
            )
            if use_nn_kernel3 and is_skinny:
                BLOCK_M = 128
                BLOCK_N = 256
                BLOCK_K = 64
                GROUP_M = 8
                NUM_STAGES = 4
                NUM_WARPS = 8
                NUM_CTAS = 1
                desc_a = TensorDescriptor(
                    base=A,
                    shape=[m, k],
                    strides=[lda, 1],
                    block_shape=[BLOCK_M, BLOCK_K],
                )
                desc_b = TensorDescriptor(
                    base=B,
                    shape=[k, n],
                    strides=[ldb, 1],
                    block_shape=[BLOCK_K, BLOCK_N],
                )
                desc_c = TensorDescriptor(
                    base=C,
                    shape=[m, n],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )
                grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
                _hgemm_nn_kernel4[grid](
                    desc_a,
                    desc_b,
                    desc_c,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    beta_is_zero,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_M=GROUP_M,
                    num_stages=NUM_STAGES,
                    num_warps=NUM_WARPS,
                    num_ctas=NUM_CTAS,
                )
            elif use_nn_kernel3:
                _hgemm_nn_kernel3[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            elif aligned and max(m, n) <= 1024:
                _hgemm_nn_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _hgemm_nn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            is_skinny = (m >= 16384 and max(n, k) <= 2048) or (
                n >= 16384 and max(m, k) <= 2048
            )
            if aligned and is_skinny:
                BLOCK_M = 128
                BLOCK_N = 256
                BLOCK_K = 64
                GROUP_M = 8
                NUM_STAGES = 4
                NUM_WARPS = 8
                NUM_CTAS = 1
                desc_a = TensorDescriptor(
                    base=A,
                    shape=[k, m],
                    strides=[lda, 1],
                    block_shape=[BLOCK_K, BLOCK_M],
                )
                desc_b = TensorDescriptor(
                    base=B,
                    shape=[k, n],
                    strides=[ldb, 1],
                    block_shape=[BLOCK_K, BLOCK_N],
                )
                desc_c = TensorDescriptor(
                    base=C,
                    shape=[m, n],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )
                grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
                _hgemm_tn_kernel3[grid](
                    desc_a,
                    desc_b,
                    desc_c,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    beta_is_zero,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_M=GROUP_M,
                    num_stages=NUM_STAGES,
                    num_warps=NUM_WARPS,
                    num_ctas=NUM_CTAS,
                )
            elif aligned:
                _hgemm_tn_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _hgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            if aligned:
                _hgemm_nt_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _hgemm_nt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        else:
            is_skinny = m >= 16384 and max(n, k) <= 2048
            if is_skinny and aligned:
                BLOCK_M = 128
                BLOCK_N = 256
                BLOCK_K = 64
                GROUP_M = 8
                NUM_STAGES = 4
                NUM_WARPS = 8
                NUM_CTAS = 1
                desc_a = TensorDescriptor(
                    base=A,
                    shape=[k, m],
                    strides=[lda, 1],
                    block_shape=[BLOCK_K, BLOCK_M],
                )
                desc_b = TensorDescriptor(
                    base=B,
                    shape=[n, k],
                    strides=[ldb, 1],
                    block_shape=[BLOCK_N, BLOCK_K],
                )
                desc_c = TensorDescriptor(
                    base=C,
                    shape=[m, n],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )
                grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
                _hgemm_tt_kernel3[grid](
                    desc_a,
                    desc_b,
                    desc_c,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    beta == 0.0,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_M=GROUP_M,
                    num_stages=NUM_STAGES,
                    num_warps=NUM_WARPS,
                    num_ctas=NUM_CTAS,
                )
            elif aligned:
                _hgemm_tt_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _hgemm_tt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_nn_kernel2(
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

    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        result = acc * alpha
        tl.store(c_block_ptr, result.to(tl.bfloat16))
    else:
        c_vals = tl.load(c_block_ptr).to(tl.float32)
        result = acc * alpha + beta * c_vals
        tl.store(c_block_ptr, result.to(tl.bfloat16))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nn2"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_nn_kernel3(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.bfloat16)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.bfloat16)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@triton.jit
def _bfgemm_nn_kernel4(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = desc_a.load([offs_m, offs_k])
        b = desc_b.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.bfloat16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.bfloat16)
        desc_c.store([offs_m, offs_n], result)


@libentry()
@triton.jit
def _bfgemm_tn_kernel3(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K

        a_t = desc_a.load([offs_k, offs_m])
        a = tl.trans(a_t)

        b = desc_b.load([offs_k, offs_n])

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.bfloat16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.bfloat16)
        desc_c.store([offs_m, offs_n], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_tn2"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_tn_kernel2(
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

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(k, m),
        strides=(lda, 1),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)

        a = tl.trans(a_t)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.bfloat16)
        tl.store(c_block_ptr, result)
    else:
        c_vals = tl.load(c_block_ptr).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.bfloat16)
        tl.store(c_block_ptr, result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nt"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_nt_kernel2(
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

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, offset_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.bfloat16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.bfloat16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemm_nn"),
    key=_BFGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _bfgemm_tt_kernel2(
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

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(1, lda),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.bfloat16)
        tl.store(c_block_ptr, result)
    else:
        c_vals = tl.load(c_block_ptr).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.bfloat16)
        tl.store(c_block_ptr, result)


@libentry()
@triton.jit
def _bfgemm_tt_kernel3(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = desc_a.load([offs_k, offs_m])
        a = tl.trans(a_t)
        b_t = desc_b.load([offs_n, offs_k])
        b = tl.trans(b_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.bfloat16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.bfloat16)
        desc_c.store([offs_m, offs_n], result)


def bfgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta,
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

    strides_aligned = (lda % 8 == 0) and (ldb % 8 == 0) and (ldc % 8 == 0)
    ptrs_aligned = (
        (A.data_ptr() % 16 == 0)
        and (B.data_ptr() % 16 == 0)
        and (C.data_ptr() % 16 == 0)
    )
    aligned = strides_aligned and ptrs_aligned
    use_nn_kernel3 = aligned and (m * n > 2048 * 2048) and min(m, n) >= 64
    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            is_skinny = (m >= 16384 and max(n, k) <= 2048) or (
                n >= 16384 and max(m, k) <= 2048
            )
            if use_nn_kernel3 and is_skinny:
                BLOCK_M = 128
                BLOCK_N = 256
                BLOCK_K = 64
                GROUP_M = 8
                NUM_STAGES = 4
                NUM_WARPS = 8
                NUM_CTAS = 1
                desc_a = TensorDescriptor(
                    base=A,
                    shape=[m, k],
                    strides=[lda, 1],
                    block_shape=[BLOCK_M, BLOCK_K],
                )
                desc_b = TensorDescriptor(
                    base=B,
                    shape=[k, n],
                    strides=[ldb, 1],
                    block_shape=[BLOCK_K, BLOCK_N],
                )
                desc_c = TensorDescriptor(
                    base=C,
                    shape=[m, n],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )
                grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
                _bfgemm_nn_kernel4[grid](
                    desc_a,
                    desc_b,
                    desc_c,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    beta_is_zero,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_M=GROUP_M,
                    num_stages=NUM_STAGES,
                    num_warps=NUM_WARPS,
                    num_ctas=NUM_CTAS,
                )
            elif use_nn_kernel3:
                _bfgemm_nn_kernel3[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            elif aligned and max(m, n) <= 1024:
                _bfgemm_nn_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _bfgemm_nn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            is_skinny = (m >= 16384 and max(n, k) <= 2048) or (
                n >= 16384 and max(m, k) <= 2048
            )
            if aligned and is_skinny:
                BLOCK_M = 128
                BLOCK_N = 256
                BLOCK_K = 64
                GROUP_M = 8
                NUM_STAGES = 4
                NUM_WARPS = 8
                NUM_CTAS = 1
                desc_a = TensorDescriptor(
                    base=A,
                    shape=[k, m],
                    strides=[lda, 1],
                    block_shape=[BLOCK_K, BLOCK_M],
                )
                desc_b = TensorDescriptor(
                    base=B,
                    shape=[k, n],
                    strides=[ldb, 1],
                    block_shape=[BLOCK_K, BLOCK_N],
                )
                desc_c = TensorDescriptor(
                    base=C,
                    shape=[m, n],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )
                grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
                _bfgemm_tn_kernel3[grid](
                    desc_a,
                    desc_b,
                    desc_c,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    beta_is_zero,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_M=GROUP_M,
                    num_stages=NUM_STAGES,
                    num_warps=NUM_WARPS,
                    num_ctas=NUM_CTAS,
                )
            elif aligned:
                _bfgemm_tn_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _bfgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            if aligned:
                _bfgemm_nt_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _bfgemm_nt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        else:
            is_skinny = m >= 16384 and max(n, k) <= 2048
            if is_skinny and aligned:
                BLOCK_M = 128
                BLOCK_N = 256
                BLOCK_K = 64
                GROUP_M = 8
                NUM_STAGES = 4
                NUM_WARPS = 8
                NUM_CTAS = 1
                desc_a = TensorDescriptor(
                    base=A,
                    shape=[k, m],
                    strides=[lda, 1],
                    block_shape=[BLOCK_K, BLOCK_M],
                )
                desc_b = TensorDescriptor(
                    base=B,
                    shape=[n, k],
                    strides=[ldb, 1],
                    block_shape=[BLOCK_N, BLOCK_K],
                )
                desc_c = TensorDescriptor(
                    base=C,
                    shape=[m, n],
                    strides=[ldc, 1],
                    block_shape=[BLOCK_M, BLOCK_N],
                )
                grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
                _bfgemm_tt_kernel3[grid](
                    desc_a,
                    desc_b,
                    desc_c,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    beta == 0.0,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_M=GROUP_M,
                    num_stages=NUM_STAGES,
                    num_warps=NUM_WARPS,
                    num_ctas=NUM_CTAS,
                )
            elif aligned:
                _bfgemm_tt_kernel2[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
            else:
                _bfgemm_tt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )


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

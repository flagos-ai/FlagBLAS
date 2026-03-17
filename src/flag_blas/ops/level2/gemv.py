import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def sgemv_n_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha: tl.float32, beta: tl.float32,
    m, n, STRIDE_AM, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :]
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first")
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def sgemv_t_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha: tl.float32, beta: tl.float32,
    m, n, STRIDE_AK, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + row_offsets[:, None] + k_offsets_init[None, :] * STRIDE_AK
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K * STRIDE_AK
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first")
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def dgemv_n_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_int: tl.int64, beta_int: tl.int64,
    m, n, STRIDE_AM, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :]
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float64)

    step_a = BLOCK_SIZE_K
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first")
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def dgemv_t_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_int: tl.int64, beta_int: tl.int64,
    m, n, STRIDE_AK, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + row_offsets[:, None] + k_offsets_init[None, :] * STRIDE_AK
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float64)

    step_a = BLOCK_SIZE_K * STRIDE_AK
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first")
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def cgemv_n_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_real: tl.float32, alpha_imag: tl.float32,
    beta_real: tl.float32, beta_imag: tl.float32,
    m, n, STRIDE_AM, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_base = row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :]
    x_base = k_offsets_init * INCX

    a_ptr_int64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_int64 = x_ptr.to(tl.pointer_type(tl.int64))

    a_ptrs_int64 = a_ptr_int64 + a_base
    x_ptrs_int64 = x_ptr_int64 + x_base

    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_val = tl.load(a_ptrs_int64, mask=a_mask, other=0, eviction_policy="evict_first")
        x_val = tl.load(x_ptrs_int64, mask=k_mask, other=0, eviction_policy="evict_last")

        a_real = a_val.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        x_real = x_val.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        xr_block = x_real[None, :]
        xi_block = x_imag[None, :]

        acc_real_2d += a_real * xr_block - a_imag * xi_block
        acc_imag_2d += a_real * xi_block + a_imag * xr_block

        a_ptrs_int64 += step_a
        x_ptrs_int64 += step_x

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)

    y_base = row_offsets * INCY * 2
    if BETA_IS_ZERO:
        result_real = alpha_real * acc_real - alpha_imag * acc_imag
        result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    else:
        y_real = tl.load(y_ptr + y_base, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_base + 1, mask=row_mask, other=0.0)
        result_real = (alpha_real * acc_real - alpha_imag * acc_imag) + (beta_real * y_real - beta_imag * y_imag)
        result_imag = (alpha_real * acc_imag + alpha_imag * acc_real) + (beta_real * y_imag + beta_imag * y_real)

    tl.store(y_ptr + y_base, result_real, mask=row_mask)
    tl.store(y_ptr + y_base + 1, result_imag, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "STRIDE_AN", "INCX", "INCY", "CONJ", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def cgemv_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_real: tl.float32, alpha_imag: tl.float32,
    beta_real: tl.float32, beta_imag: tl.float32,
    m, n, STRIDE_AM, STRIDE_AN, INCX, INCY, CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_base = row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :] * STRIDE_AN
    x_base = k_offsets_init * INCX

    a_ptr_int64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_int64 = x_ptr.to(tl.pointer_type(tl.int64))

    a_ptrs_int64 = a_ptr_int64 + a_base
    x_ptrs_int64 = x_ptr_int64 + x_base

    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K * STRIDE_AN
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_val = tl.load(a_ptrs_int64, mask=a_mask, other=0, eviction_policy="evict_first")
        x_val = tl.load(x_ptrs_int64, mask=k_mask, other=0, eviction_policy="evict_last")

        a_real = a_val.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        x_real = x_val.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        xr_block = x_real[None, :]
        xi_block = x_imag[None, :]

        if CONJ == 1:
            acc_real_2d += a_real * xr_block + a_imag * xi_block
            acc_imag_2d += a_real * xi_block - a_imag * xr_block
        else:
            acc_real_2d += a_real * xr_block - a_imag * xi_block
            acc_imag_2d += a_real * xi_block + a_imag * xr_block

        a_ptrs_int64 += step_a
        x_ptrs_int64 += step_x

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)

    y_base = row_offsets * INCY * 2
    if BETA_IS_ZERO:
        result_real = alpha_real * acc_real - alpha_imag * acc_imag
        result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    else:
        y_real = tl.load(y_ptr + y_base, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_base + 1, mask=row_mask, other=0.0)
        result_real = (alpha_real * acc_real - alpha_imag * acc_imag) + (beta_real * y_real - beta_imag * y_imag)
        result_imag = (alpha_real * acc_imag + alpha_imag * acc_real) + (beta_real * y_imag + beta_imag * y_real)

    tl.store(y_ptr + y_base, result_real, mask=row_mask)
    tl.store(y_ptr + y_base + 1, result_imag, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def zgemv_n_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_real_int: tl.int64, alpha_imag_int: tl.int64,
    beta_real_int: tl.int64, beta_imag_int: tl.int64,
    m, n, STRIDE_AM, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    alpha_real = alpha_real_int.to(tl.float64, bitcast=True)
    alpha_imag = alpha_imag_int.to(tl.float64, bitcast=True)
    beta_real = beta_real_int.to(tl.float64, bitcast=True)
    beta_imag = beta_imag_int.to(tl.float64, bitcast=True)

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_base = row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :]
    x_base = k_offsets_init * INCX

    a_ptrs_real = a_ptr + a_base * 2
    a_ptrs_imag = a_ptr + a_base * 2 + 1
    x_ptrs_real = x_ptr + x_base * 2
    x_ptrs_imag = x_ptr + x_base * 2 + 1

    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float64)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float64)

    step_a = BLOCK_SIZE_K * 2
    step_x = BLOCK_SIZE_K * INCX * 2

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_real = tl.load(a_ptrs_real, mask=a_mask, other=0.0, eviction_policy="evict_first")
        a_imag = tl.load(a_ptrs_imag, mask=a_mask, other=0.0, eviction_policy="evict_first")

        x_real = tl.load(x_ptrs_real, mask=k_mask, other=0.0, eviction_policy="evict_last")
        x_imag = tl.load(x_ptrs_imag, mask=k_mask, other=0.0, eviction_policy="evict_last")

        xr_block = x_real[None, :]
        xi_block = x_imag[None, :]

        acc_real_2d += a_real * xr_block - a_imag * xi_block
        acc_imag_2d += a_real * xi_block + a_imag * xr_block

        a_ptrs_real += step_a
        a_ptrs_imag += step_a
        x_ptrs_real += step_x
        x_ptrs_imag += step_x

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)

    y_base = row_offsets * INCY * 2

    if BETA_IS_ZERO:
        result_real = alpha_real * acc_real - alpha_imag * acc_imag
        result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    else:
        y_real = tl.load(y_ptr + y_base, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_base + 1, mask=row_mask, other=0.0)
        result_real = (alpha_real * acc_real - alpha_imag * acc_imag) + (beta_real * y_real - beta_imag * y_imag)
        result_imag = (alpha_real * acc_imag + alpha_imag * acc_real) + (beta_real * y_imag + beta_imag * y_real)

    tl.store(y_ptr + y_base, result_real, mask=row_mask)
    tl.store(y_ptr + y_base + 1, result_imag, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "STRIDE_AN", "INCX", "INCY", "CONJ", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def zgemv_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_real_int: tl.int64, alpha_imag_int: tl.int64,
    beta_real_int: tl.int64, beta_imag_int: tl.int64,
    m, n, STRIDE_AM, STRIDE_AN, INCX, INCY, CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    alpha_real = alpha_real_int.to(tl.float64, bitcast=True)
    alpha_imag = alpha_imag_int.to(tl.float64, bitcast=True)
    beta_real = beta_real_int.to(tl.float64, bitcast=True)
    beta_imag = beta_imag_int.to(tl.float64, bitcast=True)

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_base = row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :] * STRIDE_AN
    x_base = k_offsets_init * INCX

    a_ptrs_real = a_ptr + a_base * 2
    a_ptrs_imag = a_ptr + a_base * 2 + 1
    x_ptrs_real = x_ptr + x_base * 2
    x_ptrs_imag = x_ptr + x_base * 2 + 1

    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float64)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float64)

    step_a = BLOCK_SIZE_K * STRIDE_AN * 2
    step_x = BLOCK_SIZE_K * INCX * 2

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_real = tl.load(a_ptrs_real, mask=a_mask, other=0.0, eviction_policy="evict_first")
        a_imag = tl.load(a_ptrs_imag, mask=a_mask, other=0.0, eviction_policy="evict_first")

        x_real = tl.load(x_ptrs_real, mask=k_mask, other=0.0, eviction_policy="evict_last")
        x_imag = tl.load(x_ptrs_imag, mask=k_mask, other=0.0, eviction_policy="evict_last")

        xr_block = x_real[None, :]
        xi_block = x_imag[None, :]

        if CONJ == 1:
            acc_real_2d += a_real * xr_block + a_imag * xi_block
            acc_imag_2d += a_real * xi_block - a_imag * xr_block
        else:
            acc_real_2d += a_real * xr_block - a_imag * xi_block
            acc_imag_2d += a_real * xi_block + a_imag * xr_block

        a_ptrs_real += step_a
        a_ptrs_imag += step_a
        x_ptrs_real += step_x
        x_ptrs_imag += step_x

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)

    y_base = row_offsets * INCY * 2

    if BETA_IS_ZERO:
        result_real = alpha_real * acc_real - alpha_imag * acc_imag
        result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    else:
        y_real = tl.load(y_ptr + y_base, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_base + 1, mask=row_mask, other=0.0)
        result_real = (alpha_real * acc_real - alpha_imag * acc_imag) + (beta_real * y_real - beta_imag * y_imag)
        result_imag = (alpha_real * acc_imag + alpha_imag * acc_real) + (beta_real * y_imag + beta_imag * y_real)

    tl.store(y_ptr + y_base, result_real, mask=row_mask)
    tl.store(y_ptr + y_base + 1, result_imag, mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def hgemv_n_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha: tl.float32, beta: tl.float32,
    m, n, STRIDE_AM, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :]
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result.to(tl.float16), mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def hgemv_t_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha: tl.float32, beta: tl.float32,
    m, n, STRIDE_AK, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + row_offsets[:, None] + k_offsets_init[None, :] * STRIDE_AK
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K * STRIDE_AK
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result.to(tl.float16), mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def bfgemv_n_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha: tl.float32, beta: tl.float32,
    m, n, STRIDE_AM, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :]
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result.to(tl.bfloat16), mask=row_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=4),
    ],
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def bfgemv_t_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha: tl.float32, beta: tl.float32,
    m, n, STRIDE_AK, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + row_offsets[:, None] + k_offsets_init[None, :] * STRIDE_AK
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K * STRIDE_AK
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x_block = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

        acc_2d += a_block * x_block[None, :]

        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)

    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result.to(tl.bfloat16), mask=row_mask)


def sgemv(trans: int, m: int, n: int, alpha: ScalarType, A: torch.Tensor, lda: int, x: torch.Tensor, incx: int, beta: ScalarType, y: torch.Tensor, incy: int) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.float32
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta == 0.0)
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            sgemv_n_kernel[grid](A, x, y, alpha, beta, m, n, lda, incx, incy, beta_is_zero)
    else:
        len_x, len_y = m, n
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta == 0.0)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            sgemv_t_kernel[grid](A, x, y, alpha, beta, n, m, lda, incx, incy, beta_is_zero)


def dgemv(trans: int, m: int, n: int, alpha: ScalarType, A: torch.Tensor, lda: int, x: torch.Tensor, incx: int, beta: ScalarType, y: torch.Tensor, incy: int) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.float64
    assert x.dtype == torch.float64
    assert y.dtype == torch.float64
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_val = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    
    if alpha_val == 0.0:
        if beta_val == 0.0:
            y.zero_()
        elif beta_val != 1.0:
            y.mul_(beta_val)
        return

    alpha_int = torch.tensor(alpha_val, dtype=torch.float64).view(torch.int64).item()
    beta_int = torch.tensor(beta_val, dtype=torch.float64).view(torch.int64).item()

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta_val == 0.0)
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            dgemv_n_kernel[grid](A, x, y, alpha_int, beta_int, m, n, lda, incx, incy, beta_is_zero)
    else:
        len_x, len_y = m, n
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta_val == 0.0)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            dgemv_t_kernel[grid](A, x, y, alpha_int, beta_int, n, m, lda, incx, incy, beta_is_zero)


def cgemv(trans: int, m: int, n: int, alpha: ScalarType, A: torch.Tensor, lda: int, x: torch.Tensor, incx: int, beta: ScalarType, y: torch.Tensor, incy: int) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.complex64
    assert x.dtype == torch.complex64
    assert y.dtype == torch.complex64
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    beta_real = float(beta.real) if isinstance(beta, complex) else float(beta)
    beta_imag = float(beta.imag) if isinstance(beta, complex) else 0.0

    if alpha_real == 0.0 and alpha_imag == 0.0:
        if beta_real == 0.0 and beta_imag == 0.0:
            y.zero_()
        elif beta_real != 1.0 or beta_imag != 0.0:
            y.mul_(beta)
        return

    conj = 0
    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        eff_m, eff_n = m, n
        stride_am, stride_an = lda, 1
    else:
        len_x, len_y = m, n
        eff_m, eff_n = n, m
        stride_am, stride_an = 1, lda
        if trans == CUBLAS_OP_C:
            conj = 1

    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy

    beta_is_zero = (beta_real == 0.0 and beta_imag == 0.0)
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    grid = lambda meta: (triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),)
    with torch_device_fn.device(A.device):
        if trans == CUBLAS_OP_N:
            cgemv_n_kernel[grid](A_real, x_real, y_real, alpha_real, alpha_imag, beta_real, beta_imag, eff_m, eff_n, stride_am, incx, incy, beta_is_zero)
        else:
            cgemv_kernel[grid](A_real, x_real, y_real, alpha_real, alpha_imag, beta_real, beta_imag, eff_m, eff_n, stride_am, stride_an, incx, incy, conj, beta_is_zero)


def zgemv(trans: int, m: int, n: int, alpha: ScalarType, A: torch.Tensor, lda: int, x: torch.Tensor, incx: int, beta: ScalarType, y: torch.Tensor, incy: int) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.complex128
    assert x.dtype == torch.complex128
    assert y.dtype == torch.complex128
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    beta_real = float(beta.real) if isinstance(beta, complex) else float(beta)
    beta_imag = float(beta.imag) if isinstance(beta, complex) else 0.0

    if alpha_real == 0.0 and alpha_imag == 0.0:
        if beta_real == 0.0 and beta_imag == 0.0:
            y.zero_()
        elif beta_real != 1.0 or beta_imag != 0.0:
            y.mul_(beta)
        return

    alpha_real_int = torch.tensor(alpha_real, dtype=torch.float64).view(torch.int64).item()
    alpha_imag_int = torch.tensor(alpha_imag, dtype=torch.float64).view(torch.int64).item()
    beta_real_int = torch.tensor(beta_real, dtype=torch.float64).view(torch.int64).item()
    beta_imag_int = torch.tensor(beta_imag, dtype=torch.float64).view(torch.int64).item()

    conj = 0
    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        eff_m, eff_n = m, n
        stride_am, stride_an = lda, 1
    else:
        len_x, len_y = m, n
        eff_m, eff_n = n, m
        stride_am, stride_an = 1, lda
        if trans == CUBLAS_OP_C:
            conj = 1

    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy

    beta_is_zero = (beta_real == 0.0 and beta_imag == 0.0)
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    grid = lambda meta: (triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),)
    with torch_device_fn.device(A.device):
        if trans == CUBLAS_OP_N:
            zgemv_n_kernel[grid](A_real, x_real, y_real, alpha_real_int, alpha_imag_int, beta_real_int, beta_imag_int, eff_m, eff_n, stride_am, incx, incy, beta_is_zero)
        else:
            zgemv_kernel[grid](A_real, x_real, y_real, alpha_real_int, alpha_imag_int, beta_real_int, beta_imag_int, eff_m, eff_n, stride_am, stride_an, incx, incy, conj, beta_is_zero)
            

def hgemv(trans: int, m: int, n: int, alpha: ScalarType, A: torch.Tensor, lda: int, x: torch.Tensor, incx: int, beta: ScalarType, y: torch.Tensor, incy: int) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.float16
    assert x.dtype == torch.float16
    assert y.dtype == torch.float16
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta == 0.0)
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            hgemv_n_kernel[grid](A, x, y, alpha, beta, m, n, lda, incx, incy, beta_is_zero)
    else:
        len_x, len_y = m, n
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta == 0.0)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            hgemv_t_kernel[grid](A, x, y, alpha, beta, n, m, lda, incx, incy, beta_is_zero)


def bfgemv(trans: int, m: int, n: int, alpha: ScalarType, A: torch.Tensor, lda: int, x: torch.Tensor, incx: int, beta: ScalarType, y: torch.Tensor, incy: int) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert y.dtype == torch.bfloat16
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta == 0.0)
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            bfgemv_n_kernel[grid](A, x, y, alpha, beta, m, n, lda, incx, incy, beta_is_zero)
    else:
        len_x, len_y = m, n
        assert x.numel() >= 1 + (len_x - 1) * incx
        assert y.numel() >= 1 + (len_y - 1) * incy
        beta_is_zero = (beta == 0.0)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            bfgemv_t_kernel[grid](A, x, y, alpha, beta, n, m, lda, incx, incy, beta_is_zero)
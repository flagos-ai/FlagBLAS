import logging
import struct
from typing import Union

import torch
import triton
import triton.language as tl
from flag_blas.ops.level2._constants import (
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]


_SGBMV_N_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 64}, num_warps=4, num_stages=2),
]

_SGBMV_T_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 64}, num_warps=4, num_stages=2),
]

_SGBMV_SPLIT_BAND_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 32}, num_warps=4, num_stages=2),
]


_DGBMV_N_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128}, num_warps=8, num_stages=2),
]

_DGBMV_T_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128}, num_warps=8, num_stages=2),
]

_DGBMV_SPLIT_BAND_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64}, num_warps=4, num_stages=3),
]

_CGBMV_N_CONFIGS = _DGBMV_N_CONFIGS
_CGBMV_T_CONFIGS = _DGBMV_T_CONFIGS
_CGBMV_SPLIT_BAND_CONFIGS = _DGBMV_SPLIT_BAND_CONFIGS

_ZGBMV_N_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_ZGBMV_T_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_ZGBMV_SPLIT_BAND_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=4, num_stages=2),
]

_REAL_KEY = ["out_len", "band_bucket"]
_COMPLEX_KEY = ["out_len", "band_bucket", "CONJ"]
_RESTORE = ["y_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


def _band_bucket(band: int) -> int:
    if band <= 1:
        return 1
    b = 1
    while b < band and b < 1024:
        b <<= 1
    return b


@triton.autotune(configs=_SGBMV_N_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE)
@triton.jit
def sgbmv_n_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    t_off = tl.arange(0, BAND_TILE)
    for d_base in tl.range(0, BAND, BAND_TILE):
        d_idx = d_base + t_off
        d = d_idx - KL
        band_row = KU - d
        band_mask = d_idx < BAND
        j = rows[:, None] + d[None, :]
        mask = row_mask[:, None] & band_mask[None, :] & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)
        a_off = band_row[None, :] + safe_j * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_j * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        yv = tl.load(y_ptrs, mask=row_mask, other=0.0)
        out = alpha * acc + beta * yv
    tl.store(y_ptrs, out, mask=row_mask)


@triton.autotune(configs=_SGBMV_T_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE)
@triton.jit
def sgbmv_t_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    t_off = tl.arange(0, BAND_TILE)
    for e_base in tl.range(0, BAND, BAND_TILE):
        e_idx = e_base + t_off
        e = e_idx - KU
        band_row = e_idx
        band_mask = e_idx < BAND
        i = cols[:, None] + e[None, :]
        mask = col_mask[:, None] & band_mask[None, :] & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)
        a_off = band_row[None, :] + cols[:, None] * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_i * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    y_ptrs = y_ptr + cols * INCY
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        yv = tl.load(y_ptrs, mask=col_mask, other=0.0)
        out = alpha * acc + beta * yv
    tl.store(y_ptrs, out, mask=col_mask)


@triton.autotune(
    configs=_SGBMV_SPLIT_BAND_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE
)
@triton.jit
def sgbmv_n_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    t_off = tl.arange(0, BAND_TILE)
    for d_base in tl.range(band_begin, band_end, BAND_TILE):
        d_idx = d_base + t_off
        d = d_idx - KL
        band_row = KU - d
        band_mask = d_idx < band_end
        j = rows[:, None] + d[None, :]
        mask = row_mask[:, None] & band_mask[None, :] & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)
        a_off = band_row[None, :] + safe_j * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_j * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    y_ptrs = y_ptr + rows * INCY
    tl.atomic_add(y_ptrs, alpha * acc, mask=row_mask, sem="relaxed")


@triton.autotune(
    configs=_SGBMV_SPLIT_BAND_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE
)
@triton.jit
def sgbmv_t_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    cols = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    t_off = tl.arange(0, BAND_TILE)
    for e_base in tl.range(band_begin, band_end, BAND_TILE):
        e_idx = e_base + t_off
        e = e_idx - KU
        band_row = e_idx
        band_mask = e_idx < band_end
        i = cols[:, None] + e[None, :]
        mask = col_mask[:, None] & band_mask[None, :] & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)
        a_off = band_row[None, :] + cols[:, None] * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_i * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    y_ptrs = y_ptr + cols * INCY
    tl.atomic_add(y_ptrs, alpha * acc, mask=col_mask, sem="relaxed")


@triton.autotune(configs=_DGBMV_N_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE)
@triton.jit
def dgbmv_n_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    for d_idx in tl.range(0, BAND):
        d = d_idx - KL
        j = rows + d
        band_row = KU - d
        mask = row_mask & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)
        a_off = band_row + safe_j * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_j * INCX, mask=mask, other=0.0)
        acc += a_vals * x_vals

    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        yv = tl.load(y_ptrs, mask=row_mask, other=0.0)
        out = alpha * acc + beta * yv
    tl.store(y_ptrs, out, mask=row_mask)


@triton.autotune(configs=_DGBMV_T_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE)
@triton.jit
def dgbmv_t_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    for e_idx in tl.range(0, BAND):
        e = e_idx - KU
        i = cols + e
        band_row = KU + e
        mask = col_mask & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)
        a_off = band_row + cols * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_i * INCX, mask=mask, other=0.0)
        acc += a_vals * x_vals

    y_ptrs = y_ptr + cols * INCY
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        yv = tl.load(y_ptrs, mask=col_mask, other=0.0)
        out = alpha * acc + beta * yv
    tl.store(y_ptrs, out, mask=col_mask)


@triton.autotune(
    configs=_DGBMV_SPLIT_BAND_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE
)
@triton.jit
def dgbmv_n_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    alpha = alpha_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    for d_idx in tl.range(band_begin, band_end):
        d = d_idx - KL
        j = rows + d
        band_row = KU - d
        mask = row_mask & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)
        a_off = band_row + safe_j * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_j * INCX, mask=mask, other=0.0)
        acc += a_vals * x_vals

    y_ptrs = y_ptr + rows * INCY
    tl.atomic_add(y_ptrs, alpha * acc, mask=row_mask, sem="relaxed")


@triton.autotune(
    configs=_DGBMV_SPLIT_BAND_CONFIGS, key=_REAL_KEY, restore_value=_RESTORE
)
@triton.jit
def dgbmv_t_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    cols = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    alpha = alpha_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    for e_idx in tl.range(band_begin, band_end):
        e = e_idx - KU
        i = cols + e
        band_row = KU + e
        mask = col_mask & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)
        a_off = band_row + cols * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_i * INCX, mask=mask, other=0.0)
        acc += a_vals * x_vals

    y_ptrs = y_ptr + cols * INCY
    tl.atomic_add(y_ptrs, alpha * acc, mask=col_mask, sem="relaxed")


@triton.autotune(configs=_CGBMV_N_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE)
@triton.jit
def cgbmv_n_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for d_idx in tl.range(0, BAND):
        d = d_idx - KL
        j = rows + d
        band_row = KU - d
        mask = row_mask & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)
        a_off = (band_row + safe_j * LDA) * 2
        x_off = safe_j * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

    y_off = rows * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        res_r += beta_r * yr - beta_i * yi
        res_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, res_r, mask=row_mask)
    tl.store(y_ptr + y_off + 1, res_i, mask=row_mask)


@triton.autotune(configs=_CGBMV_T_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE)
@triton.jit
def cgbmv_t_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for e_idx in tl.range(0, BAND):
        e = e_idx - KU
        i = cols + e
        band_row = KU + e
        mask = col_mask & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)
        a_off = (band_row + cols * LDA) * 2
        x_off = safe_i * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

    y_off = cols * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_off, mask=col_mask, other=0.0)
        yi = tl.load(y_ptr + y_off + 1, mask=col_mask, other=0.0)
        res_r += beta_r * yr - beta_i * yi
        res_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, res_r, mask=col_mask)
    tl.store(y_ptr + y_off + 1, res_i, mask=col_mask)


@triton.autotune(
    configs=_CGBMV_SPLIT_BAND_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE
)
@triton.jit
def cgbmv_n_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    for d_idx in tl.range(band_begin, band_end):
        d = d_idx - KL
        j = rows + d
        band_row = KU - d
        mask = row_mask & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)
        a_off = (band_row + safe_j * LDA) * 2
        x_off = safe_j * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

    y_off = rows * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    tl.atomic_add(y_ptr + y_off, res_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, res_i, mask=row_mask, sem="relaxed")


@triton.autotune(
    configs=_CGBMV_SPLIT_BAND_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE
)
@triton.jit
def cgbmv_t_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    cols = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    for e_idx in tl.range(band_begin, band_end):
        e = e_idx - KU
        i = cols + e
        band_row = KU + e
        mask = col_mask & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)
        a_off = (band_row + cols * LDA) * 2
        x_off = safe_i * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

    y_off = cols * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    tl.atomic_add(y_ptr + y_off, res_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, res_i, mask=col_mask, sem="relaxed")


@triton.autotune(configs=_ZGBMV_N_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE)
@triton.jit
def zgbmv_n_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    beta_r = beta_r_int.to(tl.float64, bitcast=True)
    beta_i = beta_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    t_off = tl.arange(0, BAND_TILE)
    for d_base in tl.range(0, BAND, BAND_TILE):
        d_idx = d_base + t_off
        d = d_idx - KL
        band_row = KU - d
        band_mask = d_idx < BAND
        j = rows[:, None] + d[None, :]
        mask = row_mask[:, None] & band_mask[None, :] & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)

        a_off = (band_row[None, :] + safe_j * LDA) * 2
        x_off = safe_j * INCX * 2

        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)

        if CONJ:
            ai = -ai

        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    y_off = rows * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        res_r += beta_r * yr - beta_i * yi
        res_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, res_r, mask=row_mask)
    tl.store(y_ptr + y_off + 1, res_i, mask=row_mask)


@triton.autotune(configs=_ZGBMV_T_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE)
@triton.jit
def zgbmv_t_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    beta_r = beta_r_int.to(tl.float64, bitcast=True)
    beta_i = beta_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    t_off = tl.arange(0, BAND_TILE)
    for e_base in tl.range(0, BAND, BAND_TILE):
        e_idx = e_base + t_off
        e = e_idx - KU
        band_row = e_idx
        band_mask = e_idx < BAND
        i = cols[:, None] + e[None, :]
        mask = col_mask[:, None] & band_mask[None, :] & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)

        a_off = (band_row[None, :] + cols[:, None] * LDA) * 2
        x_off = safe_i * INCX * 2

        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)

        if CONJ:
            ai = -ai

        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    y_off = cols * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_off, mask=col_mask, other=0.0)
        yi = tl.load(y_ptr + y_off + 1, mask=col_mask, other=0.0)
        res_r += beta_r * yr - beta_i * yi
        res_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, res_r, mask=col_mask)
    tl.store(y_ptr + y_off + 1, res_i, mask=col_mask)


@triton.autotune(
    configs=_ZGBMV_SPLIT_BAND_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE
)
@triton.jit
def zgbmv_n_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    t_off = tl.arange(0, BAND_TILE)
    for d_base in tl.range(band_begin, band_end, BAND_TILE):
        d_idx = d_base + t_off
        d = d_idx - KL
        band_row = KU - d
        band_mask = d_idx < band_end
        j = rows[:, None] + d[None, :]
        mask = row_mask[:, None] & band_mask[None, :] & (j >= 0) & (j < n)
        safe_j = tl.where(mask, j, 0)

        a_off = (band_row[None, :] + safe_j * LDA) * 2
        x_off = safe_j * INCX * 2

        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)

        if CONJ:
            ai = -ai

        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    y_off = rows * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    tl.atomic_add(y_ptr + y_off, res_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, res_i, mask=row_mask, sem="relaxed")


@triton.autotune(
    configs=_ZGBMV_SPLIT_BAND_CONFIGS, key=_COMPLEX_KEY, restore_value=_RESTORE
)
@triton.jit
def zgbmv_t_split_band_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    num_band_splits,
    out_len,
    band_bucket,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_band = tl.program_id(1)
    cols = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    chunk = tl.cdiv(BAND, num_band_splits)
    band_begin = pid_band * chunk
    band_end = tl.minimum(band_begin + chunk, BAND)

    t_off = tl.arange(0, BAND_TILE)
    for e_base in tl.range(band_begin, band_end, BAND_TILE):
        e_idx = e_base + t_off
        e = e_idx - KU
        band_row = e_idx
        band_mask = e_idx < band_end
        i = cols[:, None] + e[None, :]
        mask = col_mask[:, None] & band_mask[None, :] & (i >= 0) & (i < m)
        safe_i = tl.where(mask, i, 0)

        a_off = (band_row[None, :] + cols[:, None] * LDA) * 2
        x_off = safe_i * INCX * 2

        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)

        if CONJ:
            ai = -ai

        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    y_off = cols * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    tl.atomic_add(y_ptr + y_off, res_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, res_i, mask=col_mask, sem="relaxed")


def _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok):
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device
    allowed = (
        [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
        if complex_ok
        else [CUBLAS_OP_N, CUBLAS_OP_T]
    )
    assert trans in allowed
    assert incx > 0 and incy > 0
    assert kl >= 0 and ku >= 0
    assert lda >= kl + ku + 1
    assert m >= 0 and n >= 0
    len_x = n if trans == CUBLAS_OP_N else m
    len_y = m if trans == CUBLAS_OP_N else n
    assert x.numel() >= 1 + (len_x - 1) * incx if len_x > 0 else x.numel() >= 0
    assert y.numel() >= 1 + (len_y - 1) * incy if len_y > 0 else y.numel() >= 0
    assert A.numel() >= n * lda


def _pick_split_band(
    out_len: int, band: int, ref_block: int = 64, target_progs: int = 256
) -> int:
    if band < 32:
        return 1
    row_progs = (out_len + ref_block - 1) // ref_block
    if row_progs >= target_progs // 2:
        return 1
    want = max(1, target_progs // max(1, row_progs))
    cap = max(1, band // 16)
    split = min(want, cap, 64)
    return split if split >= 2 else 1


def sgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=False)
    if m == 0 or n == 0:
        return

    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)

    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    band = kl + ku + 1
    out_len = m if trans == CUBLAS_OP_N else n
    bucket = _band_bucket(band)
    split_band = _pick_split_band(out_len, band)
    if out_len <= 256 and band >= 256 and band <= out_len:
        split_band = 1

    with torch_device_fn.device(A.device):
        if split_band > 1:
            if beta == 0.0:
                y.zero_()
            elif beta != 1.0:
                y.mul_(beta)

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]), split_band)

            kernel = (
                sgbmv_n_split_band_kernel
                if trans == CUBLAS_OP_N
                else sgbmv_t_split_band_kernel
            )
            kernel[grid](
                A,
                x,
                y,
                alpha,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                split_band,
                out_len,
                bucket,
            )
        else:
            beta_is_zero = beta == 0.0

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)

            kernel = sgbmv_n_kernel if trans == CUBLAS_OP_N else sgbmv_t_kernel
            kernel[grid](
                A,
                x,
                y,
                alpha,
                beta,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                out_len,
                bucket,
                BETA_IS_ZERO=beta_is_zero,
            )


def dgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.float64 == x.dtype == y.dtype
    _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=False)
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
    band = kl + ku + 1
    out_len = m if trans == CUBLAS_OP_N else n
    inner_len = n if trans == CUBLAS_OP_N else m
    bucket = _band_bucket(band)
    split_band = _pick_split_band(out_len, band)

    if split_band == 1 and band >= 512 and out_len >= 4 * inner_len:
        split_band = 2

    with torch_device_fn.device(A.device):
        if split_band > 1:
            if beta_val == 0.0:
                y.zero_()
            elif beta_val != 1.0:
                y.mul_(beta_val)

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]), split_band)

            kernel = (
                dgbmv_n_split_band_kernel
                if trans == CUBLAS_OP_N
                else dgbmv_t_split_band_kernel
            )
            kernel[grid](
                A,
                x,
                y,
                alpha_int,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                split_band,
                out_len,
                bucket,
            )
        else:
            beta_is_zero = beta_val == 0.0

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)

            kernel = dgbmv_n_kernel if trans == CUBLAS_OP_N else dgbmv_t_kernel
            kernel[grid](
                A,
                x,
                y,
                alpha_int,
                beta_int,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                out_len,
                bucket,
                BETA_IS_ZERO=beta_is_zero,
            )


def _complex_scalars(alpha, beta):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    br = float(beta.real) if isinstance(beta, complex) else float(beta)
    bi = float(beta.imag) if isinstance(beta, complex) else 0.0
    return ar, ai, br, bi


def cgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=True)
    if m == 0 or n == 0:
        return

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y.zero_()
        elif br != 1.0 or bi != 0.0:
            y.mul_(complex(br, bi))
        return

    band = kl + ku + 1
    conj = 1 if trans == CUBLAS_OP_C else 0
    out_len = m if trans == CUBLAS_OP_N else n
    inner_len = n if trans == CUBLAS_OP_N else m
    bucket = _band_bucket(band)
    split_band = _pick_split_band(out_len, band)
    if split_band == 1 and band >= 512 and out_len >= 4 * inner_len:
        split_band = 2
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(A.device):
        if split_band > 1:
            if br == 0.0 and bi == 0.0:
                y.zero_()
            elif br != 1.0 or bi != 0.0:
                y.mul_(complex(br, bi))

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]), split_band)

            kernel = (
                cgbmv_n_split_band_kernel
                if trans == CUBLAS_OP_N
                else cgbmv_t_split_band_kernel
            )
            kernel[grid](
                A_real,
                x_real,
                y_real,
                ar,
                ai,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                split_band,
                out_len,
                bucket,
                CONJ=conj,
            )
        else:
            beta_is_zero = br == 0.0 and bi == 0.0

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)

            kernel = cgbmv_n_kernel if trans == CUBLAS_OP_N else cgbmv_t_kernel
            kernel[grid](
                A_real,
                x_real,
                y_real,
                ar,
                ai,
                br,
                bi,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                out_len,
                bucket,
                CONJ=conj,
                BETA_IS_ZERO=beta_is_zero,
            )


def zgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.complex128 == x.dtype == y.dtype
    _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=True)
    if m == 0 or n == 0:
        return

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y.zero_()
        elif br != 1.0 or bi != 0.0:
            y.mul_(complex(br, bi))
        return

    ar_i = _f64_to_i64(ar)
    ai_i = _f64_to_i64(ai)
    br_i = _f64_to_i64(br)
    bi_i = _f64_to_i64(bi)
    band = kl + ku + 1
    conj = 1 if trans == CUBLAS_OP_C else 0
    out_len = m if trans == CUBLAS_OP_N else n
    bucket = _band_bucket(band)
    split_band = _pick_split_band(out_len, band)
    if out_len <= 256 and band >= 256:
        split_band = 1
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(A.device):
        if split_band > 1:
            if br == 0.0 and bi == 0.0:
                y.zero_()
            elif br != 1.0 or bi != 0.0:
                y.mul_(complex(br, bi))

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]), split_band)

            kernel = (
                zgbmv_n_split_band_kernel
                if trans == CUBLAS_OP_N
                else zgbmv_t_split_band_kernel
            )
            kernel[grid](
                A_real,
                x_real,
                y_real,
                ar_i,
                ai_i,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                split_band,
                out_len,
                bucket,
                CONJ=conj,
            )
        else:
            beta_is_zero = br == 0.0 and bi == 0.0

            def grid(meta):
                return (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)

            kernel = zgbmv_n_kernel if trans == CUBLAS_OP_N else zgbmv_t_kernel
            kernel[grid](
                A_real,
                x_real,
                y_real,
                ar_i,
                ai_i,
                br_i,
                bi_i,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                out_len,
                bucket,
                CONJ=conj,
                BETA_IS_ZERO=beta_is_zero,
            )

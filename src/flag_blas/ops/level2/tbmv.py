import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2._constants import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]


_STBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 8}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_DTBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 8}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_CTBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 8}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 16, "BAND_TILE": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_CTBMV_T_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 8}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 8}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 16, "BAND_TILE": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_ZTBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 8}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 16, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_TBMV_KEY = ["n", "k_bucket", "mode_key"]
_TBMV_RESTORE = ["x_ptr"]
_TBMV_SMALL_N = 256
_TBMV_SMALL_K = 8
_STBMV_SMALL_K = 48


def _tbmv_grid(n):
    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)

    return grid


def _band_bucket(k: int) -> int:
    if k <= 1:
        return 1
    if k <= 8:
        return k
    b = 1
    while b < k and b < 1024:
        b <<= 1
    return b


@triton.jit
def stbmv_small_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < n
    xin = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            if TRANS == 0:
                j = rows[:, None] + (k - r[None, :])
                col = j
            else:
                j = rows[:, None] - (k - r[None, :])
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            if TRANS == 0:
                j = rows[:, None] - r[None, :]
                col = j
            else:
                j = rows[:, None] + r[None, :]
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = r[None, :] + safe_col * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_j * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    if UNIT:
        acc += xin

    tl.debug_barrier()
    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@triton.jit
def dtbmv_small_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < n
    xin = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float64)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            if TRANS == 0:
                j = rows[:, None] + (k - r[None, :])
                col = j
            else:
                j = rows[:, None] - (k - r[None, :])
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            if TRANS == 0:
                j = rows[:, None] - r[None, :]
                col = j
            else:
                j = rows[:, None] + r[None, :]
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = r[None, :] + safe_col * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + safe_j * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    if UNIT:
        acc += xin

    tl.debug_barrier()
    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@triton.jit
def ctbmv_small_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < n
    x_off = rows * INCX * 2
    xin_r = tl.load(x_ptr + x_off, mask=row_mask, other=0.0)
    xin_i = tl.load(x_ptr + x_off + 1, mask=row_mask, other=0.0)
    acc_r = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            if TRANS == 0:
                j = rows[:, None] + (k - r[None, :])
                col = j
            else:
                j = rows[:, None] - (k - r[None, :])
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            if TRANS == 0:
                j = rows[:, None] - r[None, :]
                col = j
            else:
                j = rows[:, None] + r[None, :]
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = (r[None, :] + safe_col * LDA) * 2
        x_idx = safe_j * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_idx + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    if UNIT:
        acc_r += xin_r
        acc_i += xin_i

    tl.debug_barrier()
    tl.store(x_ptr + x_off, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off + 1, acc_i, mask=row_mask)


@triton.jit
def ztbmv_small_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < n
    x_off = rows * INCX * 2
    xin_r = tl.load(x_ptr + x_off, mask=row_mask, other=0.0)
    xin_i = tl.load(x_ptr + x_off + 1, mask=row_mask, other=0.0)
    acc_r = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float64)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            if TRANS == 0:
                j = rows[:, None] + (k - r[None, :])
                col = j
            else:
                j = rows[:, None] - (k - r[None, :])
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            if TRANS == 0:
                j = rows[:, None] - r[None, :]
                col = j
            else:
                j = rows[:, None] + r[None, :]
                col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = (r[None, :] + safe_col * LDA) * 2
        x_idx = safe_j * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_idx + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    if UNIT:
        acc_r += xin_r
        acc_i += xin_i

    tl.debug_barrier()
    tl.store(x_ptr + x_off, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off + 1, acc_i, mask=row_mask)


@triton.autotune(configs=_STBMV_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def stbmv_n_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] + (k - r[None, :])
            col = j
            diag = r[None, :] == k
        else:
            j = rows[:, None] - r[None, :]
            col = j
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = r[None, :] + safe_col * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(xin_ptr + safe_j, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@triton.autotune(configs=_STBMV_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def stbmv_t_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] - (k - r[None, :])
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            j = rows[:, None] + r[None, :]
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = r[None, :] + safe_col * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(xin_ptr + safe_j, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@triton.autotune(configs=_DTBMV_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def dtbmv_n_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] + (k - r[None, :])
            col = j
            diag = r[None, :] == k
        else:
            j = rows[:, None] - r[None, :]
            col = j
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = r[None, :] + safe_col * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(xin_ptr + safe_j, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@triton.autotune(configs=_DTBMV_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def dtbmv_t_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] - (k - r[None, :])
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            j = rows[:, None] + r[None, :]
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = r[None, :] + safe_col * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(xin_ptr + safe_j, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@triton.autotune(configs=_CTBMV_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def ctbmv_n_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] + (k - r[None, :])
            col = j
            diag = r[None, :] == k
        else:
            j = rows[:, None] - r[None, :]
            col = j
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = (r[None, :] + safe_col * LDA) * 2
        x_off = safe_j * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(xin_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(xin_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


@triton.autotune(configs=_CTBMV_T_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def ctbmv_t_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] - (k - r[None, :])
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            j = rows[:, None] + r[None, :]
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = (r[None, :] + safe_col * LDA) * 2
        x_off = safe_j * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(xin_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(xin_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


@triton.autotune(configs=_ZTBMV_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def ztbmv_n_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] + (k - r[None, :])
            col = j
            diag = r[None, :] == k
        else:
            j = rows[:, None] - r[None, :]
            col = j
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = (r[None, :] + safe_col * LDA) * 2
        x_off = safe_j * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(xin_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(xin_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


@triton.autotune(configs=_ZTBMV_CONFIGS, key=_TBMV_KEY, restore_value=_TBMV_RESTORE)
@triton.jit
def ztbmv_t_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    K1 = k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, K1, BAND_TILE):
        r = r_base + r_off
        r_mask = r < K1
        if UPLO == 1:
            j = rows[:, None] - (k - r[None, :])
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == k
        else:
            j = rows[:, None] + r[None, :]
            col = rows[:, None] + tl.zeros((1, BAND_TILE), dtype=tl.int32)
            diag = r[None, :] == 0

        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        if UNIT:
            mask = mask & (~diag)
        safe_j = tl.where(mask, j, 0)
        safe_col = tl.where(mask, col, 0)
        a_off = (r[None, :] + safe_col * LDA) * 2
        x_off = safe_j * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(xin_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(xin_ptr + x_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


def _check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok):
    assert A.is_contiguous() and x.is_contiguous()
    assert A.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
    allowed = (
        [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
        if complex_ok
        else [CUBLAS_OP_N, CUBLAS_OP_T]
    )
    assert trans in allowed
    assert diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    assert incx > 0
    assert n >= 0
    assert k >= 0
    assert lda >= k + 1
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= n * lda


def _mode_key(uplo, trans, unit):
    return (uplo << 4) | (trans << 2) | unit


def _small_band_tile(k1: int) -> int:
    if k1 <= 2:
        return 2
    if k1 <= 4:
        return 4
    if k1 <= 8:
        return 8
    if k1 <= 16:
        return 16
    return 32


def _stbmv_small_num_warps(band_tile: int) -> int:
    if band_tile >= 32:
        return 8
    return 4


def stbmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype
    _check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        if n <= _TBMV_SMALL_N and k <= _STBMV_SMALL_K:
            band_tile = _small_band_tile(k + 1)
            stbmv_small_kernel[(1,)](
                A,
                x,
                n,
                k,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                BLOCK_SIZE_N=_TBMV_SMALL_N,
                BAND_TILE=band_tile,
                num_warps=_stbmv_small_num_warps(band_tile),
            )
            return
        xin = x.as_strided((n,), (incx,)).clone()
        grid = _tbmv_grid(n)
        kernel = stbmv_n_kernel if trans == CUBLAS_OP_N else stbmv_t_kernel
        kernel[grid](
            A,
            xin,
            x,
            n,
            k,
            lda,
            incx,
            _band_bucket(k + 1),
            _mode_key(uplo, trans_flag, unit),
            UPLO=uplo,
            UNIT=unit,
        )


def dtbmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float64 == x.dtype
    _check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        if n <= _TBMV_SMALL_N and k <= _TBMV_SMALL_K:
            dtbmv_small_kernel[(1,)](
                A,
                x,
                n,
                k,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                BLOCK_SIZE_N=_TBMV_SMALL_N,
                BAND_TILE=_small_band_tile(k + 1),
            )
            return
        xin = x.as_strided((n,), (incx,)).clone()
        grid = _tbmv_grid(n)
        kernel = dtbmv_n_kernel if trans == CUBLAS_OP_N else dtbmv_t_kernel
        kernel[grid](
            A,
            xin,
            x,
            n,
            k,
            lda,
            incx,
            _band_bucket(k + 1),
            _mode_key(uplo, trans_flag, unit),
            UPLO=uplo,
            UNIT=unit,
        )


def ctbmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype
    _check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0

    with torch_device_fn.device(A.device):
        if n <= _TBMV_SMALL_N and k <= _TBMV_SMALL_K:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            ctbmv_small_kernel[(1,)](
                A_real,
                x_real,
                n,
                k,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
                BLOCK_SIZE_N=_TBMV_SMALL_N,
                BAND_TILE=_small_band_tile(k + 1),
            )
            return
        xin = x.as_strided((n,), (incx,)).clone()
        A_real = torch.view_as_real(A)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)
        grid = _tbmv_grid(n)
        kernel = ctbmv_n_kernel if trans == CUBLAS_OP_N else ctbmv_t_kernel
        kernel[grid](
            A_real,
            xin_real,
            x_real,
            n,
            k,
            lda,
            incx,
            _band_bucket(k + 1),
            _mode_key(uplo, trans_flag, unit) | (conj << 8),
            UPLO=uplo,
            UNIT=unit,
            CONJ=conj,
        )


def ztbmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex128 == x.dtype
    _check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0

    with torch_device_fn.device(A.device):
        if n <= _TBMV_SMALL_N and k <= _TBMV_SMALL_K:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            ztbmv_small_kernel[(1,)](
                A_real,
                x_real,
                n,
                k,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
                BLOCK_SIZE_N=_TBMV_SMALL_N,
                BAND_TILE=_small_band_tile(k + 1),
            )
            return
        xin = x.as_strided((n,), (incx,)).clone()
        A_real = torch.view_as_real(A)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)
        grid = _tbmv_grid(n)
        kernel = ztbmv_n_kernel if trans == CUBLAS_OP_N else ztbmv_t_kernel
        kernel[grid](
            A_real,
            xin_real,
            x_real,
            n,
            k,
            lda,
            incx,
            _band_bucket(k + 1),
            _mode_key(uplo, trans_flag, unit) | (conj << 8),
            UPLO=uplo,
            UNIT=unit,
            CONJ=conj,
        )

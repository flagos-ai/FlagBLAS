import logging
import struct
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1


_SSPMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]

_DSPMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]

_SPMV_KEY = ["n", "uplo_key"]
_RESTORE = ["y_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@triton.autotune(configs=_SSPMV_CONFIGS, key=_SPMV_KEY, restore_value=_RESTORE)
@triton.jit
def sspmv_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    n,
    INCX,
    INCY,
    uplo_key,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    for kb in tl.range(0, n, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        j64 = j.to(tl.int64)

        i_ge_j = rows[:, None] >= j[None, :]
        a_lo64 = tl.where(i_ge_j, j64[None, :], rows64[:, None])
        b_hi64 = tl.where(i_ge_j, rows64[:, None], j64[None, :])

        if UPLO == 1:
            off = b_hi64 * (b_hi64 + 1) // 2 + a_lo64
        else:
            off = b_hi64 + a_lo64 * (2 * n64 - a_lo64 - 1) // 2

        mask = row_mask[:, None] & j_mask[None, :]
        safe_off = tl.where(mask, off, 0)
        a_vals = tl.load(ap_ptr + safe_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + j * INCX, mask=j_mask, other=0.0)
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        yv = tl.load(y_ptrs, mask=row_mask, other=0.0)
        out = alpha * acc + beta * yv
    tl.store(y_ptrs, out, mask=row_mask)


@triton.autotune(configs=_DSPMV_CONFIGS, key=_SPMV_KEY, restore_value=_RESTORE)
@triton.jit
def dspmv_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    n,
    INCX,
    INCY,
    uplo_key,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    for kb in tl.range(0, n, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        j64 = j.to(tl.int64)

        i_ge_j = rows[:, None] >= j[None, :]
        a_lo64 = tl.where(i_ge_j, j64[None, :], rows64[:, None])
        b_hi64 = tl.where(i_ge_j, rows64[:, None], j64[None, :])

        if UPLO == 1:
            off = b_hi64 * (b_hi64 + 1) // 2 + a_lo64
        else:
            off = b_hi64 + a_lo64 * (2 * n64 - a_lo64 - 1) // 2

        mask = row_mask[:, None] & j_mask[None, :]
        safe_off = tl.where(mask, off, 0)
        a_vals = tl.load(ap_ptr + safe_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + j * INCX, mask=j_mask, other=0.0)
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        yv = tl.load(y_ptrs, mask=row_mask, other=0.0)
        out = alpha * acc + beta * yv
    tl.store(y_ptrs, out, mask=row_mask)


def _check_common(AP, x, y, uplo, n, incx, incy):
    assert AP.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert AP.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert incx > 0 and incy > 0
    assert n >= 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert AP.numel() >= n * (n + 1) // 2


def _strided_y(y: torch.Tensor, n: int, incy: int) -> torch.Tensor:
    return y[: (n - 1) * incy + 1 : incy]


def sspmv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert AP.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return

    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)

    y_view = _strided_y(y, n, incy)

    if alpha == 0.0:
        if beta == 0.0:
            y_view.zero_()
        elif beta != 1.0:
            y_view.mul_(beta)
        return

    beta_is_zero = beta == 0.0
    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        sspmv_kernel[grid](
            AP,
            x,
            y,
            alpha,
            beta,
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_is_zero,
        )


def dspmv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert AP.dtype == torch.float64 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return

    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_val = float(beta.item() if isinstance(beta, torch.Tensor) else beta)

    y_view = _strided_y(y, n, incy)

    if alpha_val == 0.0:
        if beta_val == 0.0:
            y_view.zero_()
        elif beta_val != 1.0:
            y_view.mul_(beta_val)
        return

    alpha_int = _f64_to_i64(alpha_val)
    beta_int = _f64_to_i64(beta_val)
    beta_is_zero = beta_val == 0.0

    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        dspmv_kernel[grid](
            AP,
            x,
            y,
            alpha_int,
            beta_int,
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_is_zero,
        )

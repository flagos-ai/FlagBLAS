import logging
import struct
from typing import Union

import torch
import triton
import triton.language as tl
from flag_blas.ops.level2._constants import (
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
)
from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]


_SSYMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
]

_DSYMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
]

_CSYMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
]

_ZSYMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 8}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
]

_SYMV_KEY = ["n"]
_RESTORE = ["y_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@triton.autotune(configs=_SSYMV_CONFIGS, key=_SYMV_KEY, restore_value=_RESTORE)
@triton.jit
def ssymv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if UPLO == 0:
        if pid_m < pid_n:
            return
    else:
        if pid_m > pid_n:
            return

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]

    x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        off = tl.where(use_direct, i + j * LDA, j + i * LDA)
        a_vals = tl.load(a_ptr + off, mask=mask2d, other=0.0)
        acc = tl.sum(a_vals * x_cols[None, :], axis=1)
        tl.atomic_add(y_ptr + rows * INCY, alpha * acc, mask=row_mask, sem="relaxed")
        return

    off = rows[:, None] + cols[None, :] * LDA
    a_vals = tl.load(a_ptr + off, mask=mask2d, other=0.0)
    acc_rows = tl.sum(a_vals * x_cols[None, :], axis=1)
    acc_cols = tl.sum(a_vals * x_rows[:, None], axis=0)

    tl.atomic_add(y_ptr + rows * INCY, alpha * acc_rows, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + cols * INCY, alpha * acc_cols, mask=col_mask, sem="relaxed")


@triton.autotune(configs=_DSYMV_CONFIGS, key=_SYMV_KEY, restore_value=_RESTORE)
@triton.jit
def dsymv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    alpha = alpha_int.to(tl.float64, bitcast=True)

    if UPLO == 0:
        if pid_m < pid_n:
            return
    else:
        if pid_m > pid_n:
            return

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]

    x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        off = tl.where(use_direct, i + j * LDA, j + i * LDA)
        a_vals = tl.load(a_ptr + off, mask=mask2d, other=0.0)
        acc = tl.sum(a_vals * x_cols[None, :], axis=1)
        tl.atomic_add(y_ptr + rows * INCY, alpha * acc, mask=row_mask, sem="relaxed")
        return

    off = rows[:, None] + cols[None, :] * LDA
    a_vals = tl.load(a_ptr + off, mask=mask2d, other=0.0)
    acc_rows = tl.sum(a_vals * x_cols[None, :], axis=1)
    acc_cols = tl.sum(a_vals * x_rows[:, None], axis=0)

    tl.atomic_add(y_ptr + rows * INCY, alpha * acc_rows, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + cols * INCY, alpha * acc_cols, mask=col_mask, sem="relaxed")


@triton.autotune(configs=_CSYMV_CONFIGS, key=_SYMV_KEY, restore_value=_RESTORE)
@triton.jit
def csymv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if UPLO == 0:
        if pid_m < pid_n:
            return
    else:
        if pid_m > pid_n:
            return

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]
    y_rows_off = rows * INCY * 2
    y_cols_off = cols * INCY * 2

    x_rows_off = rows * INCX * 2
    x_cols_off = cols * INCX * 2
    xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i + j * LDA, j + i * LDA)
        a_off = elem_off * 2
        ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
        acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
        acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
        res_r = alpha_r * acc_r - alpha_i * acc_i
        res_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, res_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, res_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[:, None] + cols[None, :] * LDA
    a_off = elem_off * 2
    ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)

    acc_rows_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
    acc_rows_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
    acc_cols_r = tl.sum(ar * xrr[:, None] - ai * xri[:, None], axis=0)
    acc_cols_i = tl.sum(ar * xri[:, None] + ai * xrr[:, None], axis=0)

    row_res_r = alpha_r * acc_rows_r - alpha_i * acc_rows_i
    row_res_i = alpha_r * acc_rows_i + alpha_i * acc_rows_r
    col_res_r = alpha_r * acc_cols_r - alpha_i * acc_cols_i
    col_res_i = alpha_r * acc_cols_i + alpha_i * acc_cols_r

    tl.atomic_add(y_ptr + y_rows_off, row_res_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, row_res_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, col_res_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, col_res_i, mask=col_mask, sem="relaxed")


@triton.autotune(configs=_ZSYMV_CONFIGS, key=_SYMV_KEY, restore_value=_RESTORE)
@triton.jit
def zsymv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)

    if UPLO == 0:
        if pid_m < pid_n:
            return
    else:
        if pid_m > pid_n:
            return

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]
    y_rows_off = rows * INCY * 2
    y_cols_off = cols * INCY * 2

    x_rows_off = rows * INCX * 2
    x_cols_off = cols * INCX * 2
    xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i + j * LDA, j + i * LDA)
        a_off = elem_off * 2
        ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
        acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
        acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
        res_r = alpha_r * acc_r - alpha_i * acc_i
        res_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, res_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, res_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[:, None] + cols[None, :] * LDA
    a_off = elem_off * 2
    ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)

    acc_rows_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
    acc_rows_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
    acc_cols_r = tl.sum(ar * xrr[:, None] - ai * xri[:, None], axis=0)
    acc_cols_i = tl.sum(ar * xri[:, None] + ai * xrr[:, None], axis=0)

    row_res_r = alpha_r * acc_rows_r - alpha_i * acc_rows_i
    row_res_i = alpha_r * acc_rows_i + alpha_i * acc_rows_r
    col_res_r = alpha_r * acc_cols_r - alpha_i * acc_cols_i
    col_res_i = alpha_r * acc_cols_i + alpha_i * acc_cols_r

    tl.atomic_add(y_ptr + y_rows_off, row_res_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, row_res_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, col_res_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, col_res_i, mask=col_mask, sem="relaxed")


def _check_common(A, x, y, uplo, n, lda, incx, incy):
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert incx > 0 and incy > 0
    assert n >= 0
    assert lda >= max(1, n)
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert A.numel() >= n * lda


def _strided_y(y: torch.Tensor, n: int, incy: int) -> torch.Tensor:
    return y[: (n - 1) * incy + 1 : incy]


def _complex_scalars(alpha, beta):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    br = float(beta.real) if isinstance(beta, complex) else float(beta)
    bi = float(beta.imag) if isinstance(beta, complex) else 0.0
    return ar, ai, br, bi


def ssymv(
    uplo: int,
    n: int,
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
    _check_common(A, x, y, uplo, n, lda, incx, incy)
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

    with torch_device_fn.device(A.device):
        if beta == 0.0:
            y_view.zero_()
        elif beta != 1.0:
            y_view.mul_(beta)

        def grid(meta):
            return (
                triton.cdiv(n, meta["BLOCK_SIZE"]),
                triton.cdiv(n, meta["BLOCK_SIZE"]),
            )

        ssymv_kernel[grid](
            A,
            x,
            y,
            alpha,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def dsymv(
    uplo: int,
    n: int,
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
    _check_common(A, x, y, uplo, n, lda, incx, incy)
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

    with torch_device_fn.device(A.device):
        if beta_val == 0.0:
            y_view.zero_()
        elif beta_val != 1.0:
            y_view.mul_(beta_val)

        def grid(meta):
            return (
                triton.cdiv(n, meta["BLOCK_SIZE"]),
                triton.cdiv(n, meta["BLOCK_SIZE"]),
            )

        dsymv_kernel[grid](
            A,
            x,
            y,
            alpha_int,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def csymv(
    uplo: int,
    n: int,
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
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(A.device):
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))

        def grid(meta):
            return (
                triton.cdiv(n, meta["BLOCK_SIZE"]),
                triton.cdiv(n, meta["BLOCK_SIZE"]),
            )

        csymv_kernel[grid](
            A_real,
            x_real,
            y_real,
            ar,
            ai,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def zsymv(
    uplo: int,
    n: int,
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
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    ar_i = _f64_to_i64(ar)
    ai_i = _f64_to_i64(ai)
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(A.device):
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))

        def grid(meta):
            return (
                triton.cdiv(n, meta["BLOCK_SIZE"]),
                triton.cdiv(n, meta["BLOCK_SIZE"]),
            )

        zsymv_kernel[grid](
            A_real,
            x_real,
            y_real,
            ar_i,
            ai_i,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )

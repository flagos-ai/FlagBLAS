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


_CHBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 16}, num_warps=8, num_stages=2),
]

_ZHBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=8, num_stages=2),
]

_HBMV_KEY = ["n", "k_bucket", "uplo_key"]
_RESTORE = ["y_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


def _band_bucket(k: int) -> int:
    if k <= 1:
        return 1
    b = 1
    while b < k and b < 1024:
        b <<= 1
    return b


@triton.autotune(configs=_CHBMV_CONFIGS, key=_HBMV_KEY, restore_value=_RESTORE)
@triton.jit
def chbmv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    n,
    k,
    LDA,
    INCX,
    INCY,
    k_bucket,
    uplo_key,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    BAND = 2 * k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, BAND, BAND_TILE):
        r = r_base + r_off
        r_mask = r < BAND
        d = r - k
        abs_d = tl.where(d >= 0, d, -d)
        j = rows[:, None] + d[None, :]
        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        safe_j = tl.where(mask, j, 0)
        if UPLO == 1:
            packed_row = k - abs_d
            packed_col = tl.where(d[None, :] >= 0, safe_j, rows[:, None])
            use_conj = d[None, :] < 0
        else:
            packed_row = abs_d
            packed_col = tl.where(d[None, :] >= 0, rows[:, None], safe_j)
            use_conj = d[None, :] > 0
        a_off = (packed_row[None, :] + packed_col * LDA) * 2
        x_off = safe_j * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)
        ai = tl.where(use_conj, -ai, ai)
        ai = tl.where(d[None, :] == 0, 0.0, ai)
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


@triton.autotune(configs=_ZHBMV_CONFIGS, key=_HBMV_KEY, restore_value=_RESTORE)
@triton.jit
def zhbmv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    n,
    k,
    LDA,
    INCX,
    INCY,
    k_bucket,
    uplo_key,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    beta_r = beta_r_int.to(tl.float64, bitcast=True)
    beta_i = beta_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    BAND = 2 * k + 1
    r_off = tl.arange(0, BAND_TILE)
    for r_base in tl.range(0, BAND, BAND_TILE):
        r = r_base + r_off
        r_mask = r < BAND
        d = r - k
        abs_d = tl.where(d >= 0, d, -d)
        j = rows[:, None] + d[None, :]
        jmask = (j >= 0) & (j < n)
        mask = row_mask[:, None] & r_mask[None, :] & jmask
        safe_j = tl.where(mask, j, 0)
        if UPLO == 1:
            packed_row = k - abs_d
            packed_col = tl.where(d[None, :] >= 0, safe_j, rows[:, None])
            use_conj = d[None, :] < 0
        else:
            packed_row = abs_d
            packed_col = tl.where(d[None, :] >= 0, rows[:, None], safe_j)
            use_conj = d[None, :] > 0
        a_off = (packed_row[None, :] + packed_col * LDA) * 2
        x_off = safe_j * INCX * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)
        ai = tl.where(use_conj, -ai, ai)
        ai = tl.where(d[None, :] == 0, 0.0, ai)
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


def _check_common(A, x, y, uplo, n, k, lda, incx, incy):
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert incx > 0 and incy > 0
    assert n >= 0 and k >= 0
    assert lda >= k + 1
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert A.numel() >= n * lda


def _complex_scalars(alpha, beta):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    br = float(beta.real) if isinstance(beta, complex) else float(beta)
    bi = float(beta.imag) if isinstance(beta, complex) else 0.0
    return ar, ai, br, bi


def _strided_y(y: torch.Tensor, n: int, incy: int) -> torch.Tensor:
    return y[: (n - 1) * incy + 1 : incy]


def chbmv(
    uplo: int,
    n: int,
    k: int,
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
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
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

    beta_is_zero = br == 0.0 and bi == 0.0
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        chbmv_kernel[grid](
            A_real,
            x_real,
            y_real,
            ar,
            ai,
            br,
            bi,
            n,
            k,
            lda,
            incx,
            incy,
            _band_bucket(k + 1),
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_is_zero,
        )


def zhbmv(
    uplo: int,
    n: int,
    k: int,
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
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
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
    br_i = _f64_to_i64(br)
    bi_i = _f64_to_i64(bi)
    beta_is_zero = br == 0.0 and bi == 0.0
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        zhbmv_kernel[grid](
            A_real,
            x_real,
            y_real,
            ar_i,
            ai_i,
            br_i,
            bi_i,
            n,
            k,
            lda,
            incx,
            incy,
            _band_bucket(k + 1),
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_is_zero,
        )

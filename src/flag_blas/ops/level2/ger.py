import logging
import struct
from typing import Tuple, Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]


_GER_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128}, num_warps=8, num_stages=3),
]

_CGER_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256}, num_warps=8, num_stages=3),
]

_ZGER_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3),
]


def _f64_to_i64(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", value))[0]


def _scalar_to_float(alpha: ScalarType) -> float:
    return float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)


def _scalar_to_complex_parts(alpha: ScalarType) -> Tuple[float, float]:
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    return alpha_real, alpha_imag


def _check_ger_common(
    m: int,
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
    dtype: torch.dtype,
) -> bool:
    assert x.device == y.device == A.device, "x, y, and A must be on the same device"
    assert x.dtype == dtype, f"x must be {dtype}"
    assert y.dtype == dtype, f"y must be {dtype}"
    assert A.dtype == dtype, f"A must be {dtype}"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert A.dim() == 2, "A must be 2-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    assert A.is_contiguous(), "A must be contiguous"
    assert lda >= n, "lda must be at least n"
    assert incx > 0 and incy > 0, "incx and incy must be positive"

    if m <= 0 or n <= 0:
        return False

    assert (
        n == 1 or A.stride(1) == 1
    ), "A must be row-major contiguous in the inner dimension"
    assert m == 1 or A.stride(0) == lda, "lda must match the row stride of A"

    req_size_x = 1 + (m - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    req_size_A = (m - 1) * lda + n
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for m={m}, incx={incx}"
    assert (
        y.numel() >= req_size_y
    ), f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"
    assert (
        A.numel() >= req_size_A
    ), f"A is too small: need storage for m={m}, n={n}, lda={lda}"
    return True


@libentry()
@triton.autotune(
    configs=_GER_CONFIGS,
    key=["m", "n", "LDA", "INCX", "INCY"],
    restore_value=["A_ptr"],
)
@triton.jit
def sger_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha: tl.float32,
    m,
    n,
    INCX,
    INCY,
    LDA,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < m
    col_mask = cols < n
    safe_rows = tl.where(row_mask, rows, 0)
    safe_cols = tl.where(col_mask, cols, 0)

    x_vals = tl.load(x_ptr + safe_rows * INCX, mask=row_mask, other=0.0)
    y_vals = tl.load(y_ptr + safe_cols * INCY, mask=col_mask, other=0.0)

    a_offsets = safe_rows[:, None] * LDA + safe_cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
    out = alpha * x_vals[:, None] * y_vals[None, :] + a_vals
    tl.store(A_ptr + a_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=_GER_CONFIGS,
    key=["m", "n", "LDA", "INCX", "INCY"],
    restore_value=["A_ptr"],
)
@triton.jit
def dger_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha_int: tl.int64,
    m,
    n,
    INCX,
    INCY,
    LDA,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    alpha = alpha_int.to(tl.float64, bitcast=True)

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < m
    col_mask = cols < n
    safe_rows = tl.where(row_mask, rows, 0)
    safe_cols = tl.where(col_mask, cols, 0)

    x_vals = tl.load(x_ptr + safe_rows * INCX, mask=row_mask, other=0.0)
    y_vals = tl.load(y_ptr + safe_cols * INCY, mask=col_mask, other=0.0)

    a_offsets = safe_rows[:, None] * LDA + safe_cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
    out = alpha * x_vals[:, None] * y_vals[None, :] + a_vals
    tl.store(A_ptr + a_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=_CGER_CONFIGS,
    key=["m", "n", "LDA", "INCX", "INCY", "CONJ_Y"],
    restore_value=["A_ptr"],
)
@triton.jit
def cger_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    m,
    n,
    INCX,
    INCY,
    LDA,
    CONJ_Y: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    x_ptr_i64 = x_ptr.to(tl.pointer_type(tl.int64))
    y_ptr_i64 = y_ptr.to(tl.pointer_type(tl.int64))
    A_ptr_i64 = A_ptr.to(tl.pointer_type(tl.int64))

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < m
    col_mask = cols < n
    safe_rows = tl.where(row_mask, rows, 0)
    safe_cols = tl.where(col_mask, cols, 0)

    x_val = tl.load(x_ptr_i64 + safe_rows * INCX, mask=row_mask, other=0)
    y_val = tl.load(y_ptr_i64 + safe_cols * INCY, mask=col_mask, other=0)
    x_real = x_val.to(tl.int32).to(tl.float32, bitcast=True)
    x_imag = (x_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    y_real = y_val.to(tl.int32).to(tl.float32, bitcast=True)
    y_imag = (y_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    if CONJ_Y:
        y_imag = -y_imag

    ax_real = alpha_real * x_real - alpha_imag * x_imag
    ax_imag = alpha_real * x_imag + alpha_imag * x_real
    update_real = (
        ax_real[:, None] * y_real[None, :] - ax_imag[:, None] * y_imag[None, :]
    )
    update_imag = (
        ax_real[:, None] * y_imag[None, :] + ax_imag[:, None] * y_real[None, :]
    )

    a_offsets = safe_rows[:, None] * LDA + safe_cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    a_val = tl.load(A_ptr_i64 + a_offsets, mask=mask, other=0)
    a_real = a_val.to(tl.int32).to(tl.float32, bitcast=True)
    a_imag = (a_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    out_real = update_real + a_real
    out_imag = update_imag + a_imag
    out_real_i64 = out_real.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    out_imag_i64 = out_imag.to(tl.int32, bitcast=True).to(tl.int64) << 32
    out_val = out_real_i64 | out_imag_i64

    tl.store(A_ptr_i64 + a_offsets, out_val, mask=mask)


@libentry()
@triton.autotune(
    configs=_ZGER_CONFIGS,
    key=["m", "n", "LDA", "INCX", "INCY", "CONJ_Y"],
    restore_value=["A_ptr"],
)
@triton.jit
def zger_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha_real_int: tl.int64,
    alpha_imag_int: tl.int64,
    m,
    n,
    INCX,
    INCY,
    LDA,
    CONJ_Y: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    alpha_real = alpha_real_int.to(tl.float64, bitcast=True)
    alpha_imag = alpha_imag_int.to(tl.float64, bitcast=True)

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    lanes = tl.arange(0, BLOCK_SIZE_N * 2)
    col_lanes = lanes // 2
    cols = pid_n * BLOCK_SIZE_N + col_lanes
    lane_is_imag = (lanes & 1) == 1
    row_mask = rows < m
    col_mask = cols < n
    safe_rows = tl.where(row_mask, rows, 0)
    safe_cols = tl.where(col_mask, cols, 0)

    x_offsets = safe_rows * INCX * 2
    y_offsets = safe_cols * INCY * 2
    x_real = tl.load(x_ptr + x_offsets, mask=row_mask, other=0.0)
    x_imag = tl.load(x_ptr + x_offsets + 1, mask=row_mask, other=0.0)
    y_real = tl.load(y_ptr + y_offsets, mask=col_mask, other=0.0)
    y_imag = tl.load(y_ptr + y_offsets + 1, mask=col_mask, other=0.0)
    if CONJ_Y:
        y_imag = -y_imag

    ax_real = alpha_real * x_real - alpha_imag * x_imag
    ax_imag = alpha_real * x_imag + alpha_imag * x_real
    y_lhs = tl.where(lane_is_imag, y_imag, y_real)
    y_rhs = tl.where(lane_is_imag, y_real, y_imag)
    update_lhs = ax_real[:, None] * y_lhs[None, :]
    update_rhs = ax_imag[:, None] * y_rhs[None, :]
    update = tl.where(
        lane_is_imag[None, :], update_lhs + update_rhs, update_lhs - update_rhs
    )

    a_offsets = safe_rows[:, None] * LDA * 2 + safe_cols[None, :] * 2 + (lanes & 1)[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)

    tl.store(A_ptr + a_offsets, update + a_vals, mask=mask)


def _grid(m: int, n: int):
    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]),
            triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    return grid


def sger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    logger.debug("FLAG_BLAS SGER")

    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.float32):
        return

    alpha = _scalar_to_float(alpha)
    if alpha == 0.0:
        return

    with torch_device_fn.device(A.device):
        sger_kernel[_grid(m, n)](x, y, A, alpha, m, n, incx, incy, lda)


def dger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    logger.debug("FLAG_BLAS DGER")

    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.float64):
        return

    alpha = _scalar_to_float(alpha)
    if alpha == 0.0:
        return

    alpha_int = _f64_to_i64(alpha)
    with torch_device_fn.device(A.device):
        dger_kernel[_grid(m, n)](x, y, A, alpha_int, m, n, incx, incy, lda)


def _cger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
    conj_y: bool,
) -> None:
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.complex64):
        return

    alpha_real, alpha_imag = _scalar_to_complex_parts(alpha)
    if alpha_real == 0.0 and alpha_imag == 0.0:
        return

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    A_real = torch.view_as_real(A)

    with torch_device_fn.device(A.device):
        cger_kernel[_grid(m, n)](
            x_real,
            y_real,
            A_real,
            alpha_real,
            alpha_imag,
            m,
            n,
            incx,
            incy,
            lda,
            conj_y,
        )


def cgeru(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    logger.debug("FLAG_BLAS CGERU")
    _cger(m, n, alpha, x, incx, y, incy, A, lda, False)


def cgerc(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    logger.debug("FLAG_BLAS CGERC")
    _cger(m, n, alpha, x, incx, y, incy, A, lda, True)


def _zger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
    conj_y: bool,
) -> None:
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.complex128):
        return

    alpha_real, alpha_imag = _scalar_to_complex_parts(alpha)
    if alpha_real == 0.0 and alpha_imag == 0.0:
        return

    alpha_real_int = _f64_to_i64(alpha_real)
    alpha_imag_int = _f64_to_i64(alpha_imag)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    A_real = torch.view_as_real(A)

    with torch_device_fn.device(A.device):
        zger_kernel[_grid(m, n)](
            x_real,
            y_real,
            A_real,
            alpha_real_int,
            alpha_imag_int,
            m,
            n,
            incx,
            incy,
            lda,
            conj_y,
        )


def zgeru(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    logger.debug("FLAG_BLAS ZGERU")
    _zger(m, n, alpha, x, incx, y, incy, A, lda, False)


def zgerc(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    logger.debug("FLAG_BLAS ZGERC")
    _zger(m, n, alpha, x, incx, y, incy, A, lda, True)

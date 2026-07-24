# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


_SSBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 256, "BAND_TILE": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 64}, num_warps=4, num_stages=2),
]

_DSBMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BAND_TILE": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BAND_TILE": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BAND_TILE": 32}, num_warps=8, num_stages=2),
]

_SBMV_KEY = ["n", "k_bucket", "uplo_key"]
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


@triton.autotune(configs=_SSBMV_CONFIGS, key=_SBMV_KEY, restore_value=_RESTORE)
@triton.jit
def ssbmv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
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
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

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
        else:
            packed_row = abs_d
            packed_col = tl.where(d[None, :] >= 0, rows[:, None], safe_j)
        a_off = packed_row[None, :] + packed_col * LDA
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


@triton.autotune(configs=_DSBMV_CONFIGS, key=_SBMV_KEY, restore_value=_RESTORE)
@triton.jit
def dsbmv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
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
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

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
        else:
            packed_row = abs_d
            packed_col = tl.where(d[None, :] >= 0, rows[:, None], safe_j)
        a_off = packed_row[None, :] + packed_col * LDA
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


def _strided_y(y: torch.Tensor, n: int, incy: int) -> torch.Tensor:
    return y[: (n - 1) * incy + 1 : incy]


def ssbmv(
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
    assert A.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
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
    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        ssbmv_kernel[grid](
            A,
            x,
            y,
            alpha,
            beta,
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


def dsbmv(
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
    assert A.dtype == torch.float64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
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

    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        dsbmv_kernel[grid](
            A,
            x,
            y,
            alpha_int,
            beta_int,
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

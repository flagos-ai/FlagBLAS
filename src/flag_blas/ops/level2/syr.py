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

import struct
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import (
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

ScalarType = Union[float, int, complex, torch.Tensor]


def _f64_to_i64(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", value))[0]


def _row_major_uplo(uplo: int) -> int:
    return (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )


@libentry()
@triton.jit
def _syr_kernel(
    A,
    x,
    alpha,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    uplo: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    major = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    minor = pid - major * (major + 1) // 2
    if uplo == 0:
        pid_m = major
        pid_n = minor
    else:
        pid_m = minor
        pid_n = major
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_bounds = (rows[:, None] < n) & (cols[None, :] < n)
    if uplo == 0:
        mask_tri = rows[:, None] >= cols[None, :]
    else:
        mask_tri = rows[:, None] <= cols[None, :]
    mask = mask_bounds & mask_tri
    xv_r = tl.load(x + rows * incx, mask=rows < n, other=0.0)
    xv_c = tl.load(x + cols * incx, mask=cols < n, other=0.0)
    offs = rows[:, None] + cols[None, :] * lda
    old = tl.load(A + offs, mask=mask, other=0.0)
    if IS_DOUBLE:
        alpha_value = alpha.to(tl.float64, bitcast=True)
    else:
        alpha_value = alpha
    val = old + alpha_value * xv_r[:, None] * xv_c[None, :]
    tl.store(A + offs, val, mask=mask)


@libentry()
@triton.jit
def _syr_complex_kernel(
    A,
    x,
    alpha_r,
    alpha_i,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    uplo: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    TRIANGULAR_GRID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if TRIANGULAR_GRID:
        pid = tl.program_id(0)
        major = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
        minor = pid - major * (major + 1) // 2
        if uplo == 0:
            tile_count: tl.constexpr = (n + BLOCK_M - 1) // BLOCK_M
            pid_m = tile_count - 1 - minor
            pid_n = tile_count - 1 - major
        else:
            pid_m = minor
            pid_n = major
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_bounds = (rows[:, None] < n) & (cols[None, :] < n)
    if uplo == 0:
        mask_tri = rows[:, None] >= cols[None, :]
    else:
        mask_tri = rows[:, None] <= cols[None, :]
    mask = mask_bounds & mask_tri

    xr = tl.load(x + (rows * incx) * 2, mask=rows < n, other=0.0)
    xi = tl.load(x + (rows * incx) * 2 + 1, mask=rows < n, other=0.0)
    yr = tl.load(x + (cols * incx) * 2, mask=cols < n, other=0.0)
    yi = tl.load(x + (cols * incx) * 2 + 1, mask=cols < n, other=0.0)

    prod_r = xr[:, None] * yr[None, :] - xi[:, None] * yi[None, :]
    prod_i = xr[:, None] * yi[None, :] + xi[:, None] * yr[None, :]
    if IS_DOUBLE:
        alpha_real = alpha_r.to(tl.float64, bitcast=True)
        alpha_imag = alpha_i.to(tl.float64, bitcast=True)
    else:
        alpha_real = alpha_r
        alpha_imag = alpha_i
    upd_r = alpha_real * prod_r - alpha_imag * prod_i
    upd_i = alpha_real * prod_i + alpha_imag * prod_r

    elem = rows[:, None] + cols[None, :] * lda
    old_r = tl.load(A + elem * 2, mask=mask, other=0.0)
    old_i = tl.load(A + elem * 2 + 1, mask=mask, other=0.0)
    tl.store(A + elem * 2, old_r + upd_r, mask=mask)
    tl.store(A + elem * 2 + 1, old_i + upd_i, mask=mask)


@libentry()
@triton.jit
def _zsyr_vector_kernel(
    A,
    x,
    alpha_r,
    alpha_i,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    uplo: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    major = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    minor = pid - major * (major + 1) // 2
    if uplo == 0:
        tile_count: tl.constexpr = (n + BLOCK_M - 1) // BLOCK_M
        pid_m = tile_count - 1 - minor
        pid_n = tile_count - 1 - major
    else:
        pid_m = minor
        pid_n = major

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_bounds = (rows[:, None] < n) & (cols[None, :] < n)
    if uplo == 0:
        mask_tri = rows[:, None] >= cols[None, :]
    else:
        mask_tri = rows[:, None] <= cols[None, :]
    mask = mask_bounds & mask_tri

    xr = tl.load(x + rows * incx * 2, mask=rows < n, other=0.0)
    xi = tl.load(x + rows * incx * 2 + 1, mask=rows < n, other=0.0)
    yr = tl.load(x + cols * incx * 2, mask=cols < n, other=0.0)
    yi = tl.load(x + cols * incx * 2 + 1, mask=cols < n, other=0.0)
    prod_r = xr[:, None] * yr[None, :] - xi[:, None] * yi[None, :]
    prod_i = xr[:, None] * yi[None, :] + xi[:, None] * yr[None, :]
    alpha_real = alpha_r.to(tl.float64, bitcast=True)
    alpha_imag = alpha_i.to(tl.float64, bitcast=True)
    upd_r = alpha_real * prod_r - alpha_imag * prod_i
    upd_i = alpha_real * prod_i + alpha_imag * prod_r

    elem = (rows[:, None] + cols[None, :] * lda) * 2
    components = tl.arange(0, 2)
    complex_off = elem[:, :, None] + components[None, None, :]
    complex_mask = mask[:, :, None]
    old = tl.load(A + complex_off, mask=complex_mask, other=0.0)
    update = tl.where(
        components[None, None, :] == 0, upd_r[:, :, None], upd_i[:, :, None]
    )
    tl.store(A + complex_off, old + update, mask=complex_mask)


def _check_syr_args(uplo, n, x, incx, A, lda, dtype):
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0
    assert lda >= max(1, n)
    assert A.dtype == dtype == x.dtype
    assert A.is_contiguous() and x.is_contiguous()
    assert A.device == x.device and (
        A.is_cuda
        or A.device.type == "npu"
        or (
            runtime.device.vendor_name == "mthreads" and A.device.type == "musa"
        )
    )
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= lda * n


def _syr_real(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
    dtype: torch.dtype,
):
    _check_syr_args(uplo, n, x, incx, A, lda, dtype)
    if n == 0:
        return A
    uplo = _row_major_uplo(uplo)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    is_double = dtype == torch.float64
    kernel_alpha = _f64_to_i64(alpha_value) if is_double else alpha_value
    block_size = 32 if n >= 512 else 16
    tile_count = triton.cdiv(n, block_size)
    grid = (tile_count * (tile_count + 1) // 2,)
    with torch_device_fn.device(A.device):
        _syr_kernel[grid](
            A,
            x,
            kernel_alpha,
            n,
            lda,
            incx,
            uplo,
            IS_DOUBLE=is_double,
            BLOCK_M=block_size,
            BLOCK_N=block_size,
        )
    return A


def _syr_complex(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
    dtype: torch.dtype,
):
    _check_syr_args(uplo, n, x, incx, A, lda, dtype)
    if n == 0:
        return A
    uplo = _row_major_uplo(uplo)
    alpha_c = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    is_double = dtype == torch.complex128
    alpha_r = _f64_to_i64(alpha_c.real) if is_double else alpha_c.real
    alpha_i = _f64_to_i64(alpha_c.imag) if is_double else alpha_c.imag
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    use_wide_tile = (is_double and n >= 512) or (
        not is_double and n >= 1023 and n % 32 != 0
    )
    block_size = 32 if use_wide_tile else 16
    triangular_grid = is_double or use_wide_tile
    tile_count = triton.cdiv(n, block_size)
    if triangular_grid:
        grid = (tile_count * (tile_count + 1) // 2,)
    else:
        grid = (tile_count, tile_count)
    with torch_device_fn.device(A.device):
        if is_double:
            _zsyr_vector_kernel[grid](
                A_real,
                x_real,
                alpha_r,
                alpha_i,
                n,
                lda,
                incx,
                uplo,
                BLOCK_M=block_size,
                BLOCK_N=block_size,
                num_warps=8 if block_size == 32 else 4,
            )
        else:
            _syr_complex_kernel[grid](
                A_real,
                x_real,
                alpha_r,
                alpha_i,
                n,
                lda,
                incx,
                uplo,
                IS_DOUBLE=False,
                TRIANGULAR_GRID=triangular_grid,
                BLOCK_M=block_size,
                BLOCK_N=block_size,
                num_warps=4,
            )
    return A


def ssyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_real(uplo, n, alpha, x, incx, A, lda, torch.float32)


def dsyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_real(uplo, n, alpha, x, incx, A, lda, torch.float64)


def csyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_complex(uplo, n, alpha, x, incx, A, lda, torch.complex64)


def zsyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_complex(uplo, n, alpha, x, incx, A, lda, torch.complex128)

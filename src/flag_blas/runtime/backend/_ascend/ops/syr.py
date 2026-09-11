from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.syr import _check_syr_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from ._packed import triangular_grid, triangular_tile_ids

ScalarType = Union[float, int, complex, torch.Tensor]


@libentry()
@triton.jit
def csyr_scalar_kernel(
    a_ptr,
    x_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
):
    ar = tl.load(a_ptr)
    ai = tl.load(a_ptr + 1)
    xr = tl.load(x_ptr)
    xi = tl.load(x_ptr + 1)
    prod_r = xr * xr - xi * xi
    prod_i = 2.0 * xr * xi
    tl.store(a_ptr, ar + alpha_r * prod_r - alpha_i * prod_i)
    tl.store(a_ptr + 1, ai + alpha_r * prod_i + alpha_i * prod_r)


@libentry()
@triton.jit
def ssyr_kernel(
    a_ptr,
    x_ptr,
    alpha: tl.float32,
    n,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)

    while tile_id < tile_count:
        pid_m, pid_n = triangular_tile_ids(tile_id, UPLO)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        if UPLO == 0:
            tri_mask = rows[:, None] >= cols[None, :]
        else:
            tri_mask = rows[:, None] <= cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :] & tri_mask
        xr = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
        xc = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
        a_off = rows[:, None] * LDA + cols[None, :]
        old = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        tl.store(a_ptr + a_off, old + alpha * xr[:, None] * xc[None, :], mask=mask)
        tile_id += program_count


@libentry()
@triton.jit
def csyr_kernel(
    a_ptr,
    old_ptr,
    x_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    n,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)

    while tile_id < tile_count:
        pid_m, pid_n = triangular_tile_ids(tile_id, UPLO)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        if UPLO == 0:
            tri_mask = rows[None, :] >= cols[:, None]
        else:
            tri_mask = rows[None, :] <= cols[:, None]
        bounds = col_mask[:, None] & row_mask[None, :]

        xrr = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
        xri = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
        xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
        xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
        prod_r = xrr[None, :] * xcr[:, None] - xri[None, :] * xci[:, None]
        prod_i = xrr[None, :] * xci[:, None] + xri[None, :] * xcr[:, None]
        update_r = alpha_r * prod_r - alpha_i * prod_i
        update_i = alpha_r * prod_i + alpha_i * prod_r

        a_off = (rows[None, :] * LDA + cols[:, None]) * 2
        ar = tl.load(old_ptr + a_off, mask=bounds, other=0.0)
        ai = tl.load(old_ptr + a_off + 1, mask=bounds, other=0.0)
        tl.store(a_ptr + a_off, tl.where(tri_mask, ar + update_r, ar), mask=bounds)
        tl.store(a_ptr + a_off + 1, tl.where(tri_mask, ai + update_i, ai), mask=bounds)
        tile_id += program_count


def ssyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    _check_syr_args(uplo, n, x, incx, A, lda, torch.float32)
    if n == 0:
        return A
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return A
    with torch_device_fn.device(A.device):
        ssyr_kernel[triangular_grid(n)](
            A,
            x,
            alpha_value,
            n,
            lda,
            incx,
            UPLO=uplo,
            BLOCK_SIZE=16,
            num_warps=1,
        )
    return A


def csyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    _check_syr_args(uplo, n, x, incx, A, lda, torch.complex64)
    if n == 0:
        return A
    alpha_value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return A
    with torch_device_fn.device(A.device):
        if n == 1:
            csyr_scalar_kernel[(1,)](
                torch.view_as_real(A),
                torch.view_as_real(x),
                alpha_value.real,
                alpha_value.imag,
            )
            return A
        old_A = A.clone()
        csyr_kernel[triangular_grid(n)](
            torch.view_as_real(A),
            torch.view_as_real(old_A),
            torch.view_as_real(x),
            alpha_value.real,
            alpha_value.imag,
            n,
            lda,
            incx,
            UPLO=uplo,
            BLOCK_SIZE=16,
            num_warps=1,
        )
    return A

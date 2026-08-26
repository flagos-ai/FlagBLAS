from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.her2 import _check_her2_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from ._packed import triangular_grid, triangular_tile_ids

ScalarType = Union[float, int, complex, torch.Tensor]


@libentry()
@triton.jit
def cher2_scalar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
):
    ar = tl.load(a_ptr)
    xr = tl.load(x_ptr)
    xi = tl.load(x_ptr + 1)
    yr = tl.load(y_ptr)
    yi = tl.load(y_ptr + 1)
    prod_r = xr * yr + xi * yi
    prod_i = xi * yr - xr * yi
    update_r = 2.0 * (alpha_r * prod_r - alpha_i * prod_i)
    tl.store(a_ptr, ar + update_r)
    tl.store(a_ptr + 1, 0.0)


@libentry()
@triton.jit
def cher2_kernel(
    a_ptr,
    old_ptr,
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
        yrr = tl.load(y_ptr + rows * INCY * 2, mask=row_mask, other=0.0)
        yri = tl.load(y_ptr + rows * INCY * 2 + 1, mask=row_mask, other=0.0)
        xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
        xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
        ycr = tl.load(y_ptr + cols * INCY * 2, mask=col_mask, other=0.0)
        yci = tl.load(y_ptr + cols * INCY * 2 + 1, mask=col_mask, other=0.0)

        p1r = xrr[None, :] * ycr[:, None] + xri[None, :] * yci[:, None]
        p1i = xri[None, :] * ycr[:, None] - xrr[None, :] * yci[:, None]
        p2r = yrr[None, :] * xcr[:, None] + yri[None, :] * xci[:, None]
        p2i = yri[None, :] * xcr[:, None] - yrr[None, :] * xci[:, None]
        update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
        update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r

        a_off = (rows[None, :] * LDA + cols[:, None]) * 2
        ar = tl.load(old_ptr + a_off, mask=bounds, other=0.0)
        ai = tl.load(old_ptr + a_off + 1, mask=bounds, other=0.0)
        diag = rows[None, :] == cols[:, None]
        out_r = tl.where(tri_mask, ar + update_r, ar)
        out_i = tl.where(tri_mask, tl.where(diag, 0.0, ai + update_i), ai)
        tl.store(a_ptr + a_off, out_r, mask=bounds)
        tl.store(a_ptr + a_off + 1, out_i, mask=bounds)
        tile_id += program_count


def cher2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
):
    _check_her2_args(torch.complex64, uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return A
    alpha_value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return A
    with torch_device_fn.device(A.device):
        if n == 1:
            cher2_scalar_kernel[(1,)](
                torch.view_as_real(A),
                torch.view_as_real(x),
                torch.view_as_real(y),
                alpha_value.real,
                alpha_value.imag,
            )
            return A
        old_A = A.clone()
        cher2_kernel[triangular_grid(n)](
            torch.view_as_real(A),
            torch.view_as_real(old_A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            alpha_value.real,
            alpha_value.imag,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
            BLOCK_SIZE=16,
            num_warps=1,
        )
    return A


__all__ = ["cher2"]

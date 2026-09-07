from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.her import _check_her_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from ._packed import triangular_grid, triangular_tile_ids

ScalarType = Union[float, int, torch.Tensor]


@libentry()
@triton.jit
def cher_scalar_kernel(a_ptr, x_ptr, alpha: tl.float32):
    ar = tl.load(a_ptr)
    xr = tl.load(x_ptr)
    xi = tl.load(x_ptr + 1)
    tl.store(a_ptr, ar + alpha * (xr * xr + xi * xi))
    tl.store(a_ptr + 1, 0.0)


@libentry()
@triton.jit
def cher_kernel(
    a_ptr,
    old_ptr,
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
            tri_mask = rows[None, :] >= cols[:, None]
        else:
            tri_mask = rows[None, :] <= cols[:, None]
        bounds = col_mask[:, None] & row_mask[None, :]

        xrr = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
        xri = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
        xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
        xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
        update_r = alpha * (xrr[None, :] * xcr[:, None] + xri[None, :] * xci[:, None])
        update_i = alpha * (xri[None, :] * xcr[:, None] - xrr[None, :] * xci[:, None])

        a_off = (rows[None, :] * LDA + cols[:, None]) * 2
        ar = tl.load(old_ptr + a_off, mask=bounds, other=0.0)
        ai = tl.load(old_ptr + a_off + 1, mask=bounds, other=0.0)
        diag = rows[None, :] == cols[:, None]
        out_r = tl.where(tri_mask, ar + update_r, ar)
        out_i = tl.where(tri_mask, tl.where(diag, 0.0, ai + update_i), ai)
        tl.store(a_ptr + a_off, out_r, mask=bounds)
        tl.store(a_ptr + a_off + 1, out_i, mask=bounds)
        tile_id += program_count


def cher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    _check_her_args(
        "cher",
        uplo,
        n,
        alpha,
        x,
        incx,
        A,
        lda,
        torch.complex64,
        torch.float32,
    )
    if n == 0:
        return A
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return A
    with torch_device_fn.device(A.device):
        if n == 1:
            cher_scalar_kernel[(1,)](
                torch.view_as_real(A),
                torch.view_as_real(x),
                alpha_value,
            )
            return A
        old_A = A.clone()
        cher_kernel[triangular_grid(n)](
            torch.view_as_real(A),
            torch.view_as_real(old_A),
            torch.view_as_real(x),
            alpha_value,
            n,
            lda,
            incx,
            UPLO=uplo,
            BLOCK_SIZE=16,
            num_warps=1,
        )
    return A


__all__ = ["cher"]

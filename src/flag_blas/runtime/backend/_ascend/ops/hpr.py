from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.hpr import _check_hpr_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from ._packed import triangular_grid, triangular_tile_ids

ScalarType = Union[float, int, torch.Tensor]


@libentry()
@triton.jit
def chpr_scalar_kernel(ap_ptr, x_ptr, alpha: tl.float32):
    ar = tl.load(ap_ptr)
    xr = tl.load(x_ptr)
    xi = tl.load(x_ptr + 1)
    tl.store(ap_ptr, ar + alpha * (xr * xr + xi * xi))
    tl.store(ap_ptr + 1, 0.0)


@libentry()
@triton.jit
def chpr_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    n,
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
        rows64 = rows.to(tl.int64)
        cols64 = cols.to(tl.int64)
        n64 = tl.full((), n, tl.int64)
        if UPLO == 0:
            tri_mask = rows[None, :] >= cols[:, None]
            off = rows64[None, :] * (rows64[None, :] + 1) // 2 + cols64[:, None]
        else:
            tri_mask = rows[None, :] <= cols[:, None]
            off = (
                rows64[None, :] * n64
                - rows64[None, :] * (rows64[None, :] + 1) // 2
                + cols64[:, None]
            )
        mask = row_mask[None, :] & col_mask[:, None] & tri_mask
        safe_off = tl.where(mask, off, 0)

        xrr = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
        xri = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
        xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
        xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
        update_r = alpha * (xrr[None, :] * xcr[:, None] + xri[None, :] * xci[:, None])
        update_i = alpha * (xri[None, :] * xcr[:, None] - xrr[None, :] * xci[:, None])

        ap_off = safe_off * 2
        ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
        ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
        diag = rows[None, :] == cols[:, None]
        tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
        tl.store(
            ap_ptr + ap_off + 1,
            tl.where(diag, 0.0, ai + update_i),
            mask=mask,
        )
        tile_id += program_count


def chpr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
):
    _check_hpr_args(torch.complex64, uplo, n, x, incx, AP)
    if n == 0:
        return AP
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return AP
    with torch_device_fn.device(AP.device):
        if n == 1:
            chpr_scalar_kernel[(1,)](
                torch.view_as_real(AP),
                torch.view_as_real(x),
                alpha_value,
            )
            return AP
        chpr_kernel[triangular_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            alpha_value,
            n,
            incx,
            UPLO=uplo,
            BLOCK_SIZE=16,
            num_warps=1,
        )
    return AP


__all__ = ["chpr"]

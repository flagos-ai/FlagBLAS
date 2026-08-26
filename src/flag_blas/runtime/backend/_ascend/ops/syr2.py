from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.syr2 import _check_ssyr2_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from ._packed import triangular_grid, triangular_tile_ids

ScalarType = Union[float, int, torch.Tensor]


@libentry()
@triton.jit
def ssyr2_kernel(
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
        yr = tl.load(y_ptr + rows * INCY, mask=row_mask, other=0.0)
        xc = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
        yc = tl.load(y_ptr + cols * INCY, mask=col_mask, other=0.0)
        a_off = rows[:, None] * LDA + cols[None, :]
        old = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        update = alpha * (xr[:, None] * yc[None, :] + yr[:, None] * xc[None, :])
        tl.store(a_ptr + a_off, old + update, mask=mask)
        tile_id += program_count


def ssyr2(
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
    _check_ssyr2_args(uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return A
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return A
    with torch_device_fn.device(A.device):
        ssyr2_kernel[triangular_grid(n)](
            A,
            x,
            y,
            alpha_value,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
            BLOCK_SIZE=16,
            num_warps=1,
        )
    return A


__all__ = ["ssyr2"]

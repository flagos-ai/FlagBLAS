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

"""Ascend-specific complex64 HEMV kernel."""

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.hemv import _check_common, _complex_scalars, _strided_y
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ScalarType = Union[float, int, complex, torch.Tensor]

_MAX_CORE_DIM = 65535
_HEMV_KEY = ["n"]
_RESTORE = ["y_ptr"]


@triton.jit
def _triangular_tile_ids(tile_id, UPLO: tl.constexpr):
    high = ((tl.sqrt(8.0 * tile_id + 1.0) - 1.0) * 0.5).to(tl.int32)
    low = tile_id - high * (high + 1) // 2
    if UPLO == 0:
        return high, low
    return low, high


def _triangular_grid(n):
    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (min(tiles * (tiles + 1) // 2, _MAX_CORE_DIM),)

    return grid


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("chemv"),
    key=_HEMV_KEY,
    restore_value=_RESTORE,
)
@triton.jit
def chemv_kernel(
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
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)

    while tile_id < tile_count:
        pid_m, pid_n = _triangular_tile_ids(tile_id, UPLO)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        mask2d = col_mask[:, None] & row_mask[None, :]
        y_rows_off = rows * INCY * 2
        y_cols_off = cols * INCY * 2

        x_rows_off = rows * INCX * 2
        x_cols_off = cols * INCX * 2
        xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
        xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
        xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
        xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)

        if pid_m == pid_n:
            i = rows[None, :]
            j = cols[:, None]
            if UPLO == 0:
                use_direct = j <= i
            else:
                use_direct = j >= i
            elem_off = tl.where(use_direct, i * LDA + j, j * LDA + i)
            a_off = elem_off * 2
            ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
            ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
            ai = tl.where(use_direct, ai, -ai)
            ai = tl.where(i == j, 0.0, ai)
            acc_r = tl.sum(ar * xcr[:, None] - ai * xci[:, None], axis=0)
            acc_i = tl.sum(ar * xci[:, None] + ai * xcr[:, None], axis=0)
            res_r = alpha_r * acc_r - alpha_i * acc_i
            res_i = alpha_r * acc_i + alpha_i * acc_r
            tl.atomic_add(y_ptr + y_rows_off, res_r, mask=row_mask, sem="relaxed")
            tl.atomic_add(y_ptr + y_rows_off + 1, res_i, mask=row_mask, sem="relaxed")
        else:
            elem_off = rows[None, :] * LDA + cols[:, None]
            a_off = elem_off * 2
            ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
            ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)

            acc_rows_r = tl.sum(ar * xcr[:, None] - ai * xci[:, None], axis=0)
            acc_rows_i = tl.sum(ar * xci[:, None] + ai * xcr[:, None], axis=0)
            acc_cols_r = tl.sum(ar * xrr[None, :] + ai * xri[None, :], axis=1)
            acc_cols_i = tl.sum(ar * xri[None, :] - ai * xrr[None, :], axis=1)

            row_res_r = alpha_r * acc_rows_r - alpha_i * acc_rows_i
            row_res_i = alpha_r * acc_rows_i + alpha_i * acc_rows_r
            col_res_r = alpha_r * acc_cols_r - alpha_i * acc_cols_i
            col_res_i = alpha_r * acc_cols_i + alpha_i * acc_cols_r

            tl.atomic_add(y_ptr + y_rows_off, row_res_r, mask=row_mask, sem="relaxed")
            tl.atomic_add(
                y_ptr + y_rows_off + 1, row_res_i, mask=row_mask, sem="relaxed"
            )
            tl.atomic_add(y_ptr + y_cols_off, col_res_r, mask=col_mask, sem="relaxed")
            tl.atomic_add(
                y_ptr + y_cols_off + 1, col_res_i, mask=col_mask, sem="relaxed"
            )
        tile_id += program_count


def chemv(
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
        chemv_kernel[_triangular_grid(n)](
            A_real, x_real, y_real, ar, ai, n, lda, incx, incy, UPLO=uplo
        )

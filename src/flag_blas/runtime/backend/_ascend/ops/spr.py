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

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.spr import _check_spr_args, _row_major_uplo
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

from ._packed import triangular_grid, triangular_tile_ids

ScalarType = Union[float, int, complex, torch.Tensor]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sspr"),
    key=["n", "INCX", "UPLO"],
    restore_value=["ap_ptr"],
)
@triton.jit
def sspr_kernel(
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
            tri_mask = rows[:, None] >= cols[None, :]
            off = (
                rows64[:, None] + cols64[None, :] * (2 * n64 - cols64[None, :] - 1) // 2
            )
        else:
            tri_mask = rows[:, None] <= cols[None, :]
            off = cols64[None, :] * (cols64[None, :] + 1) // 2 + rows64[:, None]

        mask = row_mask[:, None] & col_mask[None, :] & tri_mask
        safe_off = tl.where(mask, off, 0)
        xr = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
        xc = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
        ap = tl.load(ap_ptr + safe_off, mask=mask, other=0.0)
        tl.store(
            ap_ptr + safe_off,
            ap + alpha * xr[:, None] * xc[None, :],
            mask=mask,
        )
        tile_id += program_count


def sspr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    _check_spr_args(torch.float32, uplo, n, alpha, x, incx, AP)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return
    with torch_device_fn.device(AP.device):
        sspr_kernel[triangular_grid(n)](AP, x, alpha, n, incx, UPLO=uplo)

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

import importlib

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


_common = importlib.import_module("flag_blas.ops.level2.syr")


@libentry()
@triton.jit
def _csyr_stripe_kernel(
    x,
    A,
    alpha_r,
    alpha_i,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    cols = pid * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_mask = rows < n
    col_mask = cols < n

    if UPLO == 0:
        triangle_mask = rows[:, None] >= cols[None, :]
    else:
        triangle_mask = rows[:, None] <= cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :] & triangle_mask

    xr = tl.load(x + rows * incx * 2, mask=row_mask, other=0.0)
    xi = tl.load(x + rows * incx * 2 + 1, mask=row_mask, other=0.0)
    yr = tl.load(x + cols * incx * 2, mask=col_mask, other=0.0)
    yi = tl.load(x + cols * incx * 2 + 1, mask=col_mask, other=0.0)

    prod_r = xr[:, None] * yr[None, :] - xi[:, None] * yi[None, :]
    prod_i = xr[:, None] * yi[None, :] + xi[:, None] * yr[None, :]
    upd_r = alpha_r * prod_r - alpha_i * prod_i
    upd_i = alpha_r * prod_i + alpha_i * prod_r

    elem = (rows[:, None] + cols[None, :] * lda) * 2
    old_r = tl.load(A + elem, mask=mask, other=0.0)
    old_i = tl.load(A + elem + 1, mask=mask, other=0.0)
    tl.store(A + elem, old_r + upd_r, mask=mask)
    tl.store(A + elem + 1, old_i + upd_i, mask=mask)


def _csyr_stripe(uplo, n, alpha, x, incx, A, lda):
    _common._check_syr_args(uplo, n, x, incx, A, lda, torch.complex64)
    physical_uplo = _common._row_major_uplo(uplo)
    alpha_value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    block_rows = triton.next_power_of_2(n)
    with torch_device_fn.device(A.device):
        _csyr_stripe_kernel[(n,)](
            torch.view_as_real(x).reshape(-1),
            torch.view_as_real(A).reshape(-1),
            alpha_value.real,
            alpha_value.imag,
            n,
            lda,
            incx,
            UPLO=physical_uplo,
            BLOCK_ROWS=block_rows,
            BLOCK_COLS=1,
            num_warps=4,
        )
    return A


def csyr(uplo, n, alpha, x, incx, A, lda):
    if 64 <= n <= 4096:
        return _csyr_stripe(uplo, n, alpha, x, incx, A, lda)
    return _common.csyr(uplo, n, alpha, x, incx, A, lda)

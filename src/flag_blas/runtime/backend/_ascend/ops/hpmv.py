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

"""Ascend-specific complex64 HPMV kernel."""

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.hpmv import (
    _check_common,
    _complex_scalars,
    _row_major_uplo,
    _strided_y,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ScalarType = Union[float, int, complex, torch.Tensor]

_HPMV_KEY = ["n", "uplo_key"]
_RESTORE = ["y_ptr"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("chpmv"),
    key=_HPMV_KEY,
    restore_value=_RESTORE,
)
@triton.jit
def chpmv_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    n,
    INCX,
    INCY,
    uplo_key,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    for kb in tl.range(0, n, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        j64 = j.to(tl.int64)

        i_ge_j = rows[:, None] >= j[None, :]
        a_lo64 = tl.where(i_ge_j, j64[None, :], rows64[:, None])
        b_hi64 = tl.where(i_ge_j, rows64[:, None], j64[None, :])
        diag = rows[:, None] == j[None, :]
        if UPLO == 1:
            off = b_hi64 * (b_hi64 + 1) // 2 + a_lo64
            use_conj = rows[:, None] > j[None, :]
        else:
            off = b_hi64 + a_lo64 * (2 * n64 - a_lo64 - 1) // 2
            use_conj = rows[:, None] < j[None, :]

        mask = row_mask[:, None] & j_mask[None, :]
        safe_off = tl.where(mask, off, 0)
        a_off = safe_off * 2
        x_off = j * INCX * 2
        ar = tl.load(ap_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(ap_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=j_mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=j_mask, other=0.0)
        ai = tl.where(use_conj, -ai, ai)
        ai = -ai
        ai = tl.where(diag, 0.0, ai)
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    y_off = rows * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        res_r += beta_r * yr - beta_i * yi
        res_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, res_r, mask=row_mask)
    tl.store(y_ptr + y_off + 1, res_i, mask=row_mask)


def chpmv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert AP.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    AP_real = torch.view_as_real(AP)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        chpmv_kernel[grid](
            AP_real,
            x_real,
            y_real,
            ar,
            ai,
            br,
            bi,
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=br == 0.0 and bi == 0.0,
        )

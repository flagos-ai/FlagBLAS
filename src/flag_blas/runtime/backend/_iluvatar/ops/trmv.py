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

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT, CUBLAS_OP_N
from flag_blas.runtime import torch_device_fn


_common = importlib.import_module("flag_blas.ops.level2.trmv")


@triton.jit
def _strmv_n_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols_lane = tl.arange(0, BLOCK_K)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    if UPLO == 1:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_M

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + cols_lane
        col_mask = cols < n
        if UPLO == 1:
            tri = cols[None, :] >= rows[:, None]
        else:
            tri = cols[None, :] <= rows[:, None]
        if UNIT:
            tri = tri & (cols[None, :] != rows[:, None])
        mask = row_mask[:, None] & col_mask[None, :] & tri
        a = tl.load(
            a_ptr + rows[:, None] * LDA + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        xin = tl.load(
            xin_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += a * xin[None, :]

    out = tl.sum(acc, axis=1)
    if UNIT:
        out += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows, out, mask=row_mask)


@triton.jit
def _ctrmv_n_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < n
    offs_k = tl.arange(0, BLOCK_K)
    diag_start = (row_start // BLOCK_K) * BLOCK_K
    acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    if UPLO == 0:
        a_offsets = (rows[:, None] * LDA + offs_k[None, :]) * 2
        x_offsets = offs_k * 2
        for kb in tl.range(0, diag_start, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            ar = tl.load(
                a_ptr + a_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            ai = tl.load(
                a_ptr + a_offsets + 1,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            xr = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            xi = tl.load(
                xin_ptr + x_offsets + 1,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * 2
            x_offsets += BLOCK_K * 2

    cols = diag_start + offs_k
    col_mask = cols < n
    if UPLO == 1:
        tri = cols[None, :] >= rows[:, None]
    else:
        tri = cols[None, :] <= rows[:, None]
    if UNIT:
        tri = tri & (cols[None, :] != rows[:, None])
    mask = row_mask[:, None] & col_mask[None, :] & tri
    a_offsets = (rows[:, None] * LDA + cols[None, :]) * 2
    ar = tl.load(
        a_ptr + a_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    ai = tl.load(
        a_ptr + a_offsets + 1,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    xr = tl.load(
        xin_ptr + cols * 2,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    xi = tl.load(
        xin_ptr + cols * 2 + 1,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    acc_r += ar * xr[None, :] - ai * xi[None, :]
    acc_i += ar * xi[None, :] + ai * xr[None, :]

    if UPLO == 1:
        post_start = diag_start + BLOCK_K
        a_offsets = (rows[:, None] * LDA + post_start + offs_k[None, :]) * 2
        x_offsets = (post_start + offs_k) * 2
        for kb in tl.range(post_start, n, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            ar = tl.load(
                a_ptr + a_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            ai = tl.load(
                a_ptr + a_offsets + 1,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            xr = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            xi = tl.load(
                xin_ptr + x_offsets + 1,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * 2
            x_offsets += BLOCK_K * 2

    out_r = tl.sum(acc_r, axis=1)
    out_i = tl.sum(acc_i, axis=1)
    if UNIT:
        out_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        out_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * 2, out_r, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, out_i, mask=row_mask)


strmv_n_kernel = triton.autotune(
    configs=runtime.get_tuned_config("strmv_n_iluvatar"),
    key=["n", "mode_key"],
    restore_value=["x_ptr"],
)(_strmv_n_kernel)

ctrmv_n_kernel = triton.autotune(
    configs=runtime.get_tuned_config("ctrmv_n_iluvatar"),
    key=["n", "mode_key"],
    restore_value=["x_ptr"],
)(_ctrmv_n_kernel)


def strmv(uplo, trans, diag, n, A, lda, x, incx):
    if trans != CUBLAS_OP_N or incx != 1 or n < 1024:
        _common.strmv(uplo, trans, diag, n, A, lda, x, incx)
        return

    assert A.dtype == torch.float32 == x.dtype
    _common._check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    with torch_device_fn.device(A.device):
        xin = x.clone()

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        strmv_n_kernel[grid](
            A,
            xin,
            x,
            n,
            lda,
            _common._mode_key(uplo, 0, unit),
            UPLO=uplo,
            UNIT=unit,
        )


def ctrmv(uplo, trans, diag, n, A, lda, x, incx):
    if trans != CUBLAS_OP_N or incx != 1 or n < 512:
        _common.ctrmv(uplo, trans, diag, n, A, lda, x, incx)
        return

    assert A.dtype == torch.complex64 == x.dtype
    _common._check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    with torch_device_fn.device(A.device):
        xin = x.clone()
        A_real = torch.view_as_real(A)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        ctrmv_n_kernel[grid](
            A_real,
            xin_real,
            x_real,
            n,
            lda,
            _common._mode_key(uplo, 0, unit),
            UPLO=uplo,
            UNIT=unit,
        )

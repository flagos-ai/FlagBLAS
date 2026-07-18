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

import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

_TPMV_KEY = ["n", "mode_key"]
_TPMV_RESTORE = ["x_ptr"]


def _prune_tpmv_configs(configs, named_args, **kwargs):
    n = named_args["n"]
    configs = [
        config
        for config in configs
        if config.kwargs["BLOCK_K"] <= config.kwargs["BLOCK_SIZE_M"]
    ]
    if n >= 512:
        return configs
    return [
        config
        for config in configs
        if config.num_stages <= 2
        and config.kwargs["BLOCK_SIZE_M"] <= 256
        and config.kwargs["BLOCK_K"] <= 64
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stpmv"),
    key=_TPMV_KEY,
    restore_value=_TPMV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def stpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(
                ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            x_vals = tl.load(
                xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_vals = tl.load(
            ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        x_vals = tl.load(
            xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(
                ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            x_vals = tl.load(
                xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stpmv"),
    key=_TPMV_KEY,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def stpmv_splitk_kernel(
    ap_ptr,
    xin_ptr,
    partial_ptr,
    n,
    mode_key,
    SPLIT_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO != TRANS:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_SIZE_M
    active_len = active_hi - active_lo
    total_tiles = (active_len + BLOCK_K - 1) // BLOCK_K
    tiles_per_chunk = (total_tiles + SPLIT_K - 1) // SPLIT_K
    my_tile_lo = pid_k * tiles_per_chunk
    my_tile_hi = tl.minimum((pid_k + 1) * tiles_per_chunk, total_tiles)
    my_lo = active_lo + my_tile_lo * BLOCK_K
    my_hi = active_lo + my_tile_hi * BLOCK_K

    for kb in tl.range(my_lo, my_hi, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_vals = tl.load(
            ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        x_vals = tl.load(
            xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    tl.store(partial_ptr + pid_k * n + rows, acc, mask=row_mask)


@libentry()
@triton.jit
def stpmv_init_kernel(
    xin_ptr,
    x_ptr,
    n,
    INCX,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    vals = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if UNIT:
        vals = tl.load(xin_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs * INCX, vals, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stpmv"),
    key=_TPMV_KEY,
    restore_value=_TPMV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def stpmv_notrans_atomic_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_K), BLOCK_K)
    row_mask = rows < n
    col_mask = cols < n
    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)

    if UPLO == 1:
        tri = cols[None, :] >= rows[:, None]
        col_base = cols64 * (cols64 + 1) // 2
        off = col_base[None, :] + rows64[:, None]
    else:
        tri = cols[None, :] <= rows[:, None]
        col_base = cols64 * n64 - cols64 * (cols64 + 1) // 2
        off = col_base[None, :] + rows64[:, None]
    if UNIT:
        tri = tri & (cols[None, :] != rows[:, None])
    mask = row_mask[:, None] & col_mask[None, :] & tri
    a_vals = tl.load(ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first")
    x_vals = tl.load(
        xin_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"
    )
    acc = tl.sum(a_vals * x_vals[None, :], axis=1)
    tl.atomic_add(x_ptr + rows * INCX, acc, mask=row_mask, sem="relaxed")


@libentry()
@triton.jit
def stpmv_reduce_kernel(
    partial_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    SPLIT_K: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(0, SPLIT_K):
        v = tl.load(partial_ptr + k * n + offs, mask=mask, other=0.0)
        acc += v
    if UNIT:
        acc += tl.load(xin_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs * INCX, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dtpmv"),
    key=_TPMV_KEY,
    restore_value=_TPMV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def dtpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    for kb in tl.range(0, n, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                off = j64[None, :] * (j64[None, :] + 1) // 2 + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                off = (
                    j64[None, :] * n64
                    - j64[None, :] * (j64[None, :] - 1) // 2
                    + (rows64[:, None] - j64[None, :])
                )
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
                off = rows64[:, None] * (rows64[:, None] + 1) // 2 + j64[None, :]
            else:
                tri = j[None, :] >= rows[:, None]
                off = (
                    rows64[:, None] * n64
                    - rows64[:, None] * (rows64[:, None] - 1) // 2
                    + (j64[None, :] - rows64[:, None])
                )
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        safe_off = tl.where(mask, off, 0)
        a_vals = tl.load(ap_ptr + safe_off, mask=mask, other=0.0)
        x_vals = tl.load(xin_ptr + j, mask=j_mask, other=0.0)
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows * INCX, acc, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctpmv"),
    key=_TPMV_KEY,
    restore_value=_TPMV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def ctpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off = off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_off = off * 2
        x_off = j * 2
        ar = tl.load(
            ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        ai = tl.load(
            ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        xr = tl.load(
            xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        xi = tl.load(
            xin_ptr + x_off + 1, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off = off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctpmv"),
    key=_TPMV_KEY,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def ctpmv_splitk_kernel(
    ap_ptr,
    xin_ptr,
    partial_ptr,
    n,
    mode_key,
    SPLIT_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO != TRANS:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_SIZE_M
    active_len = active_hi - active_lo
    total_tiles = (active_len + BLOCK_K - 1) // BLOCK_K
    tiles_per_chunk = (total_tiles + SPLIT_K - 1) // SPLIT_K
    my_tile_lo = pid_k * tiles_per_chunk
    my_tile_hi = tl.minimum((pid_k + 1) * tiles_per_chunk, total_tiles)
    my_lo = active_lo + my_tile_lo * BLOCK_K
    my_hi = active_lo + my_tile_hi * BLOCK_K

    diag_lo_v = row_start
    diag_hi_v = row_start + BLOCK_SIZE_M

    if UPLO == TRANS:
        pre_hi = tl.minimum(my_hi, diag_lo_v)
        for kb in tl.range(my_lo, pre_hi, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off = off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    diag_kb_lo = tl.maximum(my_lo, diag_lo_v)
    diag_kb_hi = tl.minimum(my_hi, diag_hi_v)
    for kb in tl.range(diag_kb_lo, diag_kb_hi, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_off = off * 2
        x_off = j * 2
        ar = tl.load(
            ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        ai = tl.load(
            ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        xr = tl.load(
            xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        xi = tl.load(
            xin_ptr + x_off + 1, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UPLO != TRANS:
        post_lo = tl.maximum(my_lo, diag_hi_v)
        for kb in tl.range(post_lo, my_hi, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off = off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    out_off = pid_k * n * 2 + rows * 2
    tl.store(partial_ptr + out_off, acc_r, mask=row_mask)
    tl.store(partial_ptr + out_off + 1, acc_i, mask=row_mask)


@libentry()
@triton.jit
def ctpmv_reduce_kernel(
    partial_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    SPLIT_K: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    acc_r = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(0, SPLIT_K):
        v_r = tl.load(partial_ptr + k * n * 2 + offs * 2, mask=mask, other=0.0)
        v_i = tl.load(partial_ptr + k * n * 2 + offs * 2 + 1, mask=mask, other=0.0)
        acc_r += v_r
        acc_i += v_i
    if UNIT:
        acc_r += tl.load(xin_ptr + offs * 2, mask=mask, other=0.0)
        acc_i += tl.load(xin_ptr + offs * 2 + 1, mask=mask, other=0.0)
    out_off = offs * INCX * 2
    tl.store(x_ptr + out_off, acc_r, mask=mask)
    tl.store(x_ptr + out_off + 1, acc_i, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ztpmv"),
    key=_TPMV_KEY,
    restore_value=_TPMV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def ztpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    for kb in tl.range(0, n, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                off = j64[None, :] * (j64[None, :] + 1) // 2 + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                off = (
                    j64[None, :] * n64
                    - j64[None, :] * (j64[None, :] - 1) // 2
                    + (rows64[:, None] - j64[None, :])
                )
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
                off = rows64[:, None] * (rows64[:, None] + 1) // 2 + j64[None, :]
            else:
                tri = j[None, :] >= rows[:, None]
                off = (
                    rows64[:, None] * n64
                    - rows64[:, None] * (rows64[:, None] - 1) // 2
                    + (j64[None, :] - rows64[:, None])
                )
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        safe_off = tl.where(mask, off, 0)
        a_off = safe_off * 2
        x_off = j * 2
        ar = tl.load(ap_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(ap_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(xin_ptr + x_off, mask=j_mask, other=0.0)
        xi = tl.load(xin_ptr + x_off + 1, mask=j_mask, other=0.0)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


def _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok):
    assert AP.is_contiguous() and x.is_contiguous()
    assert AP.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
    allowed = (
        [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
        if complex_ok
        else [CUBLAS_OP_N, CUBLAS_OP_T]
    )
    assert trans in allowed
    assert diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    assert incx > 0
    assert n >= 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert AP.numel() >= n * (n + 1) // 2


def _mode_key(uplo, trans, unit):
    return (uplo << 4) | (trans << 2) | unit


def _stpmv_split_k(n: int) -> int:
    if n < 3500 or n > 11000:
        return 1
    if n <= 7500:
        return 4
    return 3


def stpmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert AP.dtype == torch.float32 == x.dtype
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    split_k = _stpmv_split_k(n)

    with torch_device_fn.device(AP.device):
        xin = x.as_strided((n,), (incx,)).clone()
        if trans_flag == 0 and incx == 1 and 7500 < n <= 11000:
            BLOCK_N = 1024
            grid_init = (triton.cdiv(n, BLOCK_N),)
            stpmv_init_kernel[grid_init](
                xin,
                x,
                n,
                incx,
                UNIT=unit,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
            grid_atomic = lambda meta: (
                triton.cdiv(n, meta["BLOCK_SIZE_M"]),
                triton.cdiv(n, meta["BLOCK_K"]),
            )
            stpmv_notrans_atomic_kernel[grid_atomic](
                AP,
                xin,
                x,
                n,
                incx,
                _mode_key(uplo, trans_flag, unit),
                UPLO=uplo,
                UNIT=unit,
            )
        elif split_k > 1:
            partial = torch.empty((split_k, n), dtype=torch.float32, device=AP.device)
            grid_main = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]), split_k)
            stpmv_splitk_kernel[grid_main](
                AP,
                xin,
                partial,
                n,
                _mode_key(uplo, trans_flag, unit),
                SPLIT_K=split_k,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
            )
            BLOCK_N = 1024
            grid_red = (triton.cdiv(n, BLOCK_N),)
            stpmv_reduce_kernel[grid_red](
                partial,
                xin,
                x,
                n,
                incx,
                SPLIT_K=split_k,
                UNIT=unit,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
        else:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
            stpmv_kernel[grid](
                AP,
                xin,
                x,
                n,
                incx,
                _mode_key(uplo, trans_flag, unit),
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
            )


def dtpmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert AP.dtype == torch.float64 == x.dtype
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(AP.device):
        xin = x.as_strided((n,), (incx,)).clone()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        dtpmv_kernel[grid](
            AP,
            xin,
            x,
            n,
            incx,
            _mode_key(uplo, trans_flag, unit),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
        )


_CTPMV_SPLIT_K_NO_TRANS = (
    (511, 1),
    (2500, 2),
    (7500, 4),
    (13000, 3),
)
_CTPMV_SPLIT_K_TRANS = (
    (511, 1),
    (1500, 4),
    (3500, 3),
    (7500, 4),
    (11000, 3),
    (13000, 2),
)


def _ctpmv_split_k(n: int, trans: int) -> int:
    table = _CTPMV_SPLIT_K_NO_TRANS if trans == 0 else _CTPMV_SPLIT_K_TRANS
    for n_max, split_k in table:
        if n <= n_max:
            return split_k
    return 1


def ctpmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert AP.dtype == torch.complex64 == x.dtype
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0
    split_k = _ctpmv_split_k(n, trans_flag)

    with torch_device_fn.device(AP.device):
        xin = x.as_strided((n,), (incx,)).clone()
        AP_real = torch.view_as_real(AP)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)
        if split_k > 1:
            partial = torch.empty(
                (split_k, n, 2), dtype=torch.float32, device=AP.device
            )
            grid_main = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]), split_k)
            ctpmv_splitk_kernel[grid_main](
                AP_real,
                xin_real,
                partial,
                n,
                _mode_key(uplo, trans_flag, unit) | (conj << 8),
                SPLIT_K=split_k,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
            )
            BLOCK_N = 1024
            grid_red = (triton.cdiv(n, BLOCK_N),)
            ctpmv_reduce_kernel[grid_red](
                partial,
                xin_real,
                x_real,
                n,
                incx,
                SPLIT_K=split_k,
                UNIT=unit,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
        else:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
            ctpmv_kernel[grid](
                AP_real,
                xin_real,
                x_real,
                n,
                incx,
                _mode_key(uplo, trans_flag, unit) | (conj << 8),
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
            )


def ztpmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert AP.dtype == torch.complex128 == x.dtype
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0

    with torch_device_fn.device(AP.device):
        xin = x.as_strided((n,), (incx,)).clone()
        AP_real = torch.view_as_real(AP)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        ztpmv_kernel[grid](
            AP_real,
            xin_real,
            x_real,
            n,
            incx,
            _mode_key(uplo, trans_flag, unit) | (conj << 8),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
        )

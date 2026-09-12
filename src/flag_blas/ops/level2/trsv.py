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
from flag_blas.runtime.device_utils import get_stream_id
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]


_TRSV_KEY = ["n", "mode_key"]
_TRSV_RESTORE = ["x_ptr", "flag_ptr"]

_STRSV_INV_MIN_N = 256
_STRSV_INV_BB64_MIN_N = 4096
_STRSV_FUSE_MAX_N = 512
_DTRSV_INV_BB64_MIN_N = 1 << 30
_DTRSV_FUSE_MAX_N = 8192
_DTRSV_ROWLOAD_MAX_N = 4096
_ZTRSV_ROWLOAD_MAX_N = 8192

_TRSV_FLAG_POOL = {}
_TRSV_FLAG_SLOTS = 256


def _trsv_flags(device):
    try:
        if torch_device_fn.is_current_stream_capturing():
            return torch.full((1,), -1, dtype=torch.int32, device=device)
    except AttributeError:
        pass
    key = (device, get_stream_id(torch_device_fn, device))
    ent = _TRSV_FLAG_POOL.get(key)
    if ent is None:
        pool = torch.empty(_TRSV_FLAG_SLOTS * 4, dtype=torch.int32, device=device)
        views = [pool[i * 4 : i * 4 + 1] for i in range(_TRSV_FLAG_SLOTS)]
        ent = [pool, views, _TRSV_FLAG_SLOTS]
        _TRSV_FLAG_POOL[key] = ent
    pool, views, idx = ent
    if idx >= _TRSV_FLAG_SLOTS:
        pool.fill_(-1)
        idx = 0
    ent[2] = idx + 1
    return views[idx]


_DTRSV_FUSED_FORCE = None


def _dtrsv_fused_cfg(forward, n, bb, unit):
    if _DTRSV_FUSED_FORCE is not None:
        return _DTRSV_FUSED_FORCE
    if forward and unit and n >= 8192:
        return (1, 2)
    return (1, 4)


_ZTRSV_FUSED_FORCE = None


def _ztrsv_fused_cfg(forward, n, bb, unit):
    if _ZTRSV_FUSED_FORCE is not None:
        return _ZTRSV_FUSED_FORCE
    if unit and n <= 64:
        return (2, 1)
    return (1, 4)


_CTRSV_FUSED_FORCE = None


def _ctrsv_fused_cfg(forward, n, bb, unit):
    if _CTRSV_FUSED_FORCE is not None:
        return _CTRSV_FUSED_FORCE
    if unit:
        if n <= 64:
            return (2, 1)
        if n > _DTRSV_ROWLOAD_MAX_N:
            return (1, 4)
        return (4, 4)
    if forward:
        if n <= 256:
            return (4, 4)
        return (1, 4)
    if n <= 64:
        return (1, 2)
    return (1, 4)


@triton.jit
def strsv_n64_kernel(
    a_ptr,
    x_ptr,
    n,
    LOWER: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    sx = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")

    if LOWER:
        for step in tl.static_range(0, BLOCK_N):
            active = step < n
            cols = offs
            col_mask = (offs < step) & mask
            a_vals = tl.load(
                a_ptr + step + cols * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == step, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + step + step * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == step, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            row = BLOCK_N - 1 - step
            active = row < n
            cols = offs
            col_mask = (offs > row) & mask
            a_vals = tl.load(
                a_ptr + row + cols * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == row, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + row + row * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == row, cur, sx)

    tl.store(x_ptr + offs, sx, mask=mask)


@triton.jit
def _diag_block_inv_unit_lower_rowload(a_ptr, s, offs, BLOCK_N: tl.constexpr):
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    for i in tl.static_range(0, BLOCK_N):
        mrow = tl.load(
            a_ptr + (s + i) + (s + offs) * 256,
            mask=offs < i,
            other=0.0,
            eviction_policy="evict_last",
        )
        contrib = tl.sum(mrow[:, None] * X, axis=0)
        ei = tl.where(offs == i, 1.0, 0.0)
        xi = ei - contrib
        X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@triton.jit
def _diag_block_inv_nonunit_upper_rowload(a_ptr, s, offs, BLOCK_N: tl.constexpr):
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    for ii in tl.static_range(0, BLOCK_N):
        i = BLOCK_N - 1 - ii
        mrow = tl.load(
            a_ptr + (s + i) + (s + offs) * 256,
            mask=offs > i,
            other=0.0,
            eviction_policy="evict_last",
        )
        contrib = tl.sum(mrow[:, None] * X, axis=0)
        ei = tl.where(offs == i, 1.0, 0.0)
        diag = tl.load(a_ptr + (s + i) + (s + i) * 256, eviction_policy="evict_last")
        xi = (ei - contrib) / diag
        X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@triton.jit
def dtrsv_n64_kernel(
    a_ptr,
    x_ptr,
    n,
    LOWER: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    sx = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")
    if LOWER:
        for step in tl.static_range(0, BLOCK_N):
            active = step < n
            col_mask = (offs < step) & mask
            a_vals = tl.load(
                a_ptr + step + offs * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == step, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + step + step * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == step, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            row = BLOCK_N - 1 - step
            active = row < n
            col_mask = (offs > row) & mask
            a_vals = tl.load(
                a_ptr + row + offs * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == row, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + row + row * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == row, cur, sx)
    tl.store(x_ptr + offs, sx, mask=mask)


@triton.jit
def strsv_lu256_rowinv_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    rows = row_start + offs
    dinv = _diag_block_inv_unit_lower_rowload(a_ptr, row_start, offs, BLOCK_N)
    sx = tl.load(x_ptr + rows, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * 256,
            mask=col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def strsv_lu_neu64_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    rows = row_start + offs
    row_mask = rows < n
    m_off = rows[:, None] + rows[None, :] * LDA
    strict = (offs[:, None] > offs[None, :]) & row_mask[:, None] & row_mask[None, :]
    M = tl.load(a_ptr + m_off, mask=strict, other=0.0, eviction_policy="evict_last")
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
    )
    X = X - M
    S = tl.dot(M, M, input_precision="tf32x3")
    for k in tl.static_range(0, 5):
        X = X + tl.dot(X, S, input_precision="tf32x3")
        if k < 4:
            S = tl.dot(S, S, input_precision="tf32x3")
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        kcols = pid_lo * BLOCK_N + koffs
        col_mask = (kcols < pid_hi * BLOCK_N) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * LDA,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(X * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def strsv_un_neu64_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    rows = row_start + offs
    row_mask = rows < n
    m_off = rows[:, None] + rows[None, :] * LDA
    strict = (offs[:, None] < offs[None, :]) & row_mask[:, None] & row_mask[None, :]
    M = tl.load(a_ptr + m_off, mask=strict, other=0.0, eviction_policy="evict_last")
    d = tl.load(a_ptr + rows + rows * LDA, mask=row_mask, other=1.0)
    rinv = 1.0 / d
    M = M * rinv[:, None]
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
    )
    X = X - M
    S = tl.dot(M, M, input_precision="tf32x3")
    for k in tl.static_range(0, 5):
        X = X + tl.dot(X, S, input_precision="tf32x3")
        if k < 4:
            S = tl.dot(S, S, input_precision="tf32x3")
    X = X * rinv[None, :]
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * LDA,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(X * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def dtrsv_lu_neu64_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    rows = row_start + offs
    row_mask = rows < n
    m_off = rows[:, None] + rows[None, :] * LDA
    strict = (offs[:, None] > offs[None, :]) & row_mask[:, None] & row_mask[None, :]
    M = tl.load(a_ptr + m_off, mask=strict, other=0.0, eviction_policy="evict_last")
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
    )
    X = X - M
    S = tl.dot(M, M, input_precision="ieee")
    for k in tl.static_range(0, 5):
        X = X + tl.dot(X, S, input_precision="ieee")
        if k < 4:
            S = tl.dot(S, S, input_precision="ieee")
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        kcols = pid_lo * BLOCK_N + koffs
        col_mask = (kcols < pid_hi * BLOCK_N) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * LDA,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(X * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def _ctrsv_scalar_offset(row, col, LDA, TRANS: tl.constexpr):
    if TRANS == 0:
        return row + col * LDA
    return col + row * LDA


@triton.jit
def ctrsv_lu_scalar_kernel(
    a_ptr, x_ptr, flag_ptr, LDA, TRANS: tl.constexpr, CHUNK: tl.constexpr
):
    BLOCK_N: tl.constexpr = 32
    G: tl.constexpr = 16
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    goffs = tl.arange(0, G)
    two = tl.arange(0, 2)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    rows = row_start + offs
    rows_t = row_start + goffs
    rows_b = row_start + 16 + goffs
    xr_0_0 = tl.where(goffs == 0, 1.0, 0.0)
    xi_0_0 = tl.zeros((G,), tl.float32)
    accr = tl.where(goffs == 1, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 1, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    xr_0_1 = accr
    xi_0_1 = acci
    accr = tl.where(goffs == 2, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 2, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 2, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    xr_0_2 = accr
    xi_0_2 = acci
    accr = tl.where(goffs == 3, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 3, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 3, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 3, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    xr_0_3 = accr
    xi_0_3 = acci
    accr = tl.where(goffs == 4, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 4, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 4, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 4, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 4, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    xr_0_4 = accr
    xi_0_4 = acci
    accr = tl.where(goffs == 5, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 5, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 5, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 5, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 5, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 5, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    xr_0_5 = accr
    xi_0_5 = acci
    accr = tl.where(goffs == 6, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 6, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 6, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 6, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 6, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 6, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 6, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    xr_0_6 = accr
    xi_0_6 = acci
    accr = tl.where(goffs == 7, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 7, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 7, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 7, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 7, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 7, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 7, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 7, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    xr_0_7 = accr
    xi_0_7 = acci
    accr = tl.where(goffs == 8, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 8, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    xr_0_8 = accr
    xi_0_8 = acci
    accr = tl.where(goffs == 9, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 9, row_start + 8, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_8 - li * xi_0_8
    acci -= lr * xi_0_8 + li * xr_0_8
    xr_0_9 = accr
    xi_0_9 = acci
    accr = tl.where(goffs == 10, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 8, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_8 - li * xi_0_8
    acci -= lr * xi_0_8 + li * xr_0_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 10, row_start + 9, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_9 - li * xi_0_9
    acci -= lr * xi_0_9 + li * xr_0_9
    xr_0_10 = accr
    xi_0_10 = acci
    accr = tl.where(goffs == 11, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 8, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_8 - li * xi_0_8
    acci -= lr * xi_0_8 + li * xr_0_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 9, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_9 - li * xi_0_9
    acci -= lr * xi_0_9 + li * xr_0_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 11, row_start + 10, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_10 - li * xi_0_10
    acci -= lr * xi_0_10 + li * xr_0_10
    xr_0_11 = accr
    xi_0_11 = acci
    accr = tl.where(goffs == 12, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 8, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_8 - li * xi_0_8
    acci -= lr * xi_0_8 + li * xr_0_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 9, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_9 - li * xi_0_9
    acci -= lr * xi_0_9 + li * xr_0_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 10, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_10 - li * xi_0_10
    acci -= lr * xi_0_10 + li * xr_0_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 12, row_start + 11, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_11 - li * xi_0_11
    acci -= lr * xi_0_11 + li * xr_0_11
    xr_0_12 = accr
    xi_0_12 = acci
    accr = tl.where(goffs == 13, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 8, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_8 - li * xi_0_8
    acci -= lr * xi_0_8 + li * xr_0_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 9, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_9 - li * xi_0_9
    acci -= lr * xi_0_9 + li * xr_0_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 10, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_10 - li * xi_0_10
    acci -= lr * xi_0_10 + li * xr_0_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 11, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_11 - li * xi_0_11
    acci -= lr * xi_0_11 + li * xr_0_11
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 13, row_start + 12, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_12 - li * xi_0_12
    acci -= lr * xi_0_12 + li * xr_0_12
    xr_0_13 = accr
    xi_0_13 = acci
    accr = tl.where(goffs == 14, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 8, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_8 - li * xi_0_8
    acci -= lr * xi_0_8 + li * xr_0_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 9, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_9 - li * xi_0_9
    acci -= lr * xi_0_9 + li * xr_0_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 10, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_10 - li * xi_0_10
    acci -= lr * xi_0_10 + li * xr_0_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 11, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_11 - li * xi_0_11
    acci -= lr * xi_0_11 + li * xr_0_11
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 12, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_12 - li * xi_0_12
    acci -= lr * xi_0_12 + li * xr_0_12
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 14, row_start + 13, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_13 - li * xi_0_13
    acci -= lr * xi_0_13 + li * xr_0_13
    xr_0_14 = accr
    xi_0_14 = acci
    accr = tl.where(goffs == 15, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 0, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_0 - li * xi_0_0
    acci -= lr * xi_0_0 + li * xr_0_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 1, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_1 - li * xi_0_1
    acci -= lr * xi_0_1 + li * xr_0_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 2, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_2 - li * xi_0_2
    acci -= lr * xi_0_2 + li * xr_0_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 3, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_3 - li * xi_0_3
    acci -= lr * xi_0_3 + li * xr_0_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 4, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_4 - li * xi_0_4
    acci -= lr * xi_0_4 + li * xr_0_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 5, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_5 - li * xi_0_5
    acci -= lr * xi_0_5 + li * xr_0_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 6, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_6 - li * xi_0_6
    acci -= lr * xi_0_6 + li * xr_0_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 7, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_7 - li * xi_0_7
    acci -= lr * xi_0_7 + li * xr_0_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 8, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_8 - li * xi_0_8
    acci -= lr * xi_0_8 + li * xr_0_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 9, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_9 - li * xi_0_9
    acci -= lr * xi_0_9 + li * xr_0_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 10, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_10 - li * xi_0_10
    acci -= lr * xi_0_10 + li * xr_0_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 11, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_11 - li * xi_0_11
    acci -= lr * xi_0_11 + li * xr_0_11
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 12, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_12 - li * xi_0_12
    acci -= lr * xi_0_12 + li * xr_0_12
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 13, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_13 - li * xi_0_13
    acci -= lr * xi_0_13 + li * xr_0_13
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 15, row_start + 14, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_0_14 - li * xi_0_14
    acci -= lr * xi_0_14 + li * xr_0_14
    xr_0_15 = accr
    xi_0_15 = acci
    xr_1_0 = tl.where(goffs == 0, 1.0, 0.0)
    xi_1_0 = tl.zeros((G,), tl.float32)
    accr = tl.where(goffs == 1, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 17, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    xr_1_1 = accr
    xi_1_1 = acci
    accr = tl.where(goffs == 2, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 18, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 18, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    xr_1_2 = accr
    xi_1_2 = acci
    accr = tl.where(goffs == 3, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 19, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 19, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 19, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    xr_1_3 = accr
    xi_1_3 = acci
    accr = tl.where(goffs == 4, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 20, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 20, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 20, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 20, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    xr_1_4 = accr
    xi_1_4 = acci
    accr = tl.where(goffs == 5, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 21, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 21, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 21, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 21, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 21, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    xr_1_5 = accr
    xi_1_5 = acci
    accr = tl.where(goffs == 6, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 22, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 22, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 22, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 22, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 22, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 22, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    xr_1_6 = accr
    xi_1_6 = acci
    accr = tl.where(goffs == 7, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 23, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 23, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 23, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 23, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 23, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 23, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 23, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    xr_1_7 = accr
    xi_1_7 = acci
    accr = tl.where(goffs == 8, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 24, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    xr_1_8 = accr
    xi_1_8 = acci
    accr = tl.where(goffs == 9, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 25, row_start + 24, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_8 - li * xi_1_8
    acci -= lr * xi_1_8 + li * xr_1_8
    xr_1_9 = accr
    xi_1_9 = acci
    accr = tl.where(goffs == 10, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 24, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_8 - li * xi_1_8
    acci -= lr * xi_1_8 + li * xr_1_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 26, row_start + 25, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_9 - li * xi_1_9
    acci -= lr * xi_1_9 + li * xr_1_9
    xr_1_10 = accr
    xi_1_10 = acci
    accr = tl.where(goffs == 11, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 24, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_8 - li * xi_1_8
    acci -= lr * xi_1_8 + li * xr_1_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 25, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_9 - li * xi_1_9
    acci -= lr * xi_1_9 + li * xr_1_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 27, row_start + 26, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_10 - li * xi_1_10
    acci -= lr * xi_1_10 + li * xr_1_10
    xr_1_11 = accr
    xi_1_11 = acci
    accr = tl.where(goffs == 12, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 24, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_8 - li * xi_1_8
    acci -= lr * xi_1_8 + li * xr_1_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 25, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_9 - li * xi_1_9
    acci -= lr * xi_1_9 + li * xr_1_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 26, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_10 - li * xi_1_10
    acci -= lr * xi_1_10 + li * xr_1_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 28, row_start + 27, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_11 - li * xi_1_11
    acci -= lr * xi_1_11 + li * xr_1_11
    xr_1_12 = accr
    xi_1_12 = acci
    accr = tl.where(goffs == 13, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 24, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_8 - li * xi_1_8
    acci -= lr * xi_1_8 + li * xr_1_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 25, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_9 - li * xi_1_9
    acci -= lr * xi_1_9 + li * xr_1_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 26, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_10 - li * xi_1_10
    acci -= lr * xi_1_10 + li * xr_1_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 27, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_11 - li * xi_1_11
    acci -= lr * xi_1_11 + li * xr_1_11
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 29, row_start + 28, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_12 - li * xi_1_12
    acci -= lr * xi_1_12 + li * xr_1_12
    xr_1_13 = accr
    xi_1_13 = acci
    accr = tl.where(goffs == 14, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 24, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_8 - li * xi_1_8
    acci -= lr * xi_1_8 + li * xr_1_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 25, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_9 - li * xi_1_9
    acci -= lr * xi_1_9 + li * xr_1_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 26, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_10 - li * xi_1_10
    acci -= lr * xi_1_10 + li * xr_1_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 27, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_11 - li * xi_1_11
    acci -= lr * xi_1_11 + li * xr_1_11
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 28, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_12 - li * xi_1_12
    acci -= lr * xi_1_12 + li * xr_1_12
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 30, row_start + 29, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_13 - li * xi_1_13
    acci -= lr * xi_1_13 + li * xr_1_13
    xr_1_14 = accr
    xi_1_14 = acci
    accr = tl.where(goffs == 15, 1.0, 0.0)
    acci = tl.zeros((G,), tl.float32)
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 16, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_0 - li * xi_1_0
    acci -= lr * xi_1_0 + li * xr_1_0
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 17, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_1 - li * xi_1_1
    acci -= lr * xi_1_1 + li * xr_1_1
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 18, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_2 - li * xi_1_2
    acci -= lr * xi_1_2 + li * xr_1_2
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 19, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_3 - li * xi_1_3
    acci -= lr * xi_1_3 + li * xr_1_3
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 20, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_4 - li * xi_1_4
    acci -= lr * xi_1_4 + li * xr_1_4
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 21, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_5 - li * xi_1_5
    acci -= lr * xi_1_5 + li * xr_1_5
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 22, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_6 - li * xi_1_6
    acci -= lr * xi_1_6 + li * xr_1_6
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 23, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_7 - li * xi_1_7
    acci -= lr * xi_1_7 + li * xr_1_7
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 24, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_8 - li * xi_1_8
    acci -= lr * xi_1_8 + li * xr_1_8
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 25, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_9 - li * xi_1_9
    acci -= lr * xi_1_9 + li * xr_1_9
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 26, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_10 - li * xi_1_10
    acci -= lr * xi_1_10 + li * xr_1_10
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 27, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_11 - li * xi_1_11
    acci -= lr * xi_1_11 + li * xr_1_11
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 28, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_12 - li * xi_1_12
    acci -= lr * xi_1_12 + li * xr_1_12
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 29, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_13 - li * xi_1_13
    acci -= lr * xi_1_13 + li * xr_1_13
    m = tl.load(
        a_ptr
        + _ctrsv_scalar_offset(row_start + 31, row_start + 30, LDA, TRANS) * 2
        + two,
        eviction_policy="evict_last",
    )
    lr, li = tl.split(m)
    accr -= lr * xr_1_14 - li * xi_1_14
    acci -= lr * xi_1_14 + li * xr_1_14
    xr_1_15 = accr
    xi_1_15 = acci
    XAr = tl.zeros((G, G), tl.float32)
    XAi = tl.zeros((G, G), tl.float32)
    XCr = tl.zeros((G, G), tl.float32)
    XCi = tl.zeros((G, G), tl.float32)
    XAr = tl.where(goffs[:, None] == 0, xr_0_0[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 0, xi_0_0[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 0, xr_1_0[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 0, xi_1_0[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 1, xr_0_1[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 1, xi_0_1[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 1, xr_1_1[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 1, xi_1_1[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 2, xr_0_2[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 2, xi_0_2[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 2, xr_1_2[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 2, xi_1_2[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 3, xr_0_3[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 3, xi_0_3[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 3, xr_1_3[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 3, xi_1_3[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 4, xr_0_4[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 4, xi_0_4[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 4, xr_1_4[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 4, xi_1_4[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 5, xr_0_5[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 5, xi_0_5[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 5, xr_1_5[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 5, xi_1_5[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 6, xr_0_6[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 6, xi_0_6[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 6, xr_1_6[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 6, xi_1_6[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 7, xr_0_7[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 7, xi_0_7[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 7, xr_1_7[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 7, xi_1_7[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 8, xr_0_8[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 8, xi_0_8[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 8, xr_1_8[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 8, xi_1_8[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 9, xr_0_9[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 9, xi_0_9[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 9, xr_1_9[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 9, xi_1_9[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 10, xr_0_10[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 10, xi_0_10[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 10, xr_1_10[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 10, xi_1_10[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 11, xr_0_11[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 11, xi_0_11[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 11, xr_1_11[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 11, xi_1_11[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 12, xr_0_12[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 12, xi_0_12[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 12, xr_1_12[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 12, xi_1_12[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 13, xr_0_13[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 13, xi_0_13[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 13, xr_1_13[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 13, xi_1_13[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 14, xr_0_14[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 14, xi_0_14[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 14, xr_1_14[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 14, xi_1_14[None, :], XCi)
    XAr = tl.where(goffs[:, None] == 15, xr_0_15[None, :], XAr)
    XAi = tl.where(goffs[:, None] == 15, xi_0_15[None, :], XAi)
    XCr = tl.where(goffs[:, None] == 15, xr_1_15[None, :], XCr)
    XCi = tl.where(goffs[:, None] == 15, xi_1_15[None, :], XCi)
    b_off = _ctrsv_scalar_offset(rows_b[:, None], rows_t[None, :], LDA, TRANS) * 2
    Br = tl.load(a_ptr + b_off, eviction_policy="evict_last")
    Bi = tl.load(a_ptr + b_off + 1, eviction_policy="evict_last")
    T1r = tl.dot(Br, XAr, input_precision="tf32x3") - tl.dot(
        Bi, XAi, input_precision="tf32x3"
    )
    T1i = tl.dot(Br, XAi, input_precision="tf32x3") + tl.dot(
        Bi, XAr, input_precision="tf32x3"
    )
    T2r = tl.dot(XCr, T1r, input_precision="tf32x3") - tl.dot(
        XCi, T1i, input_precision="tf32x3"
    )
    T2i = tl.dot(XCr, T1i, input_precision="tf32x3") + tl.dot(
        XCi, T1r, input_precision="tf32x3"
    )
    lo = offs[None, :] < 16
    Xtopr = tl.where(
        lo,
        tl.reshape(tl.broadcast_to(tl.reshape(XAr, (G, 1, G)), (G, 2, G)), (G, 32)),
        0.0,
    )
    Xtopi = tl.where(
        lo,
        tl.reshape(tl.broadcast_to(tl.reshape(XAi, (G, 1, G)), (G, 2, G)), (G, 32)),
        0.0,
    )
    Xbotr = tl.where(
        lo,
        -tl.reshape(tl.broadcast_to(tl.reshape(T2r, (G, 1, G)), (G, 2, G)), (G, 32)),
        tl.reshape(tl.broadcast_to(tl.reshape(XCr, (G, 1, G)), (G, 2, G)), (G, 32)),
    )
    Xboti = tl.where(
        lo,
        -tl.reshape(tl.broadcast_to(tl.reshape(T2i, (G, 1, G)), (G, 2, G)), (G, 32)),
        tl.reshape(tl.broadcast_to(tl.reshape(XCi, (G, 1, G)), (G, 2, G)), (G, 32)),
    )
    sx2 = tl.load(
        x_ptr + (rows * 2)[:, None] + two[None, :], eviction_policy="evict_last"
    )
    sxr, sxi = tl.split(sx2)
    num_chunks = tl.cdiv(pid, CHUNK)
    for gg in tl.range(0, num_chunks):
        pid_lo = gg * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        kcols = pid_lo * BLOCK_N + koffs
        col_mask = kcols < pid_hi * BLOCK_N
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + two[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        a_off = _ctrsv_scalar_offset(rows[:, None], kcols[None, :], LDA, TRANS) * 2
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + two[None, None, :],
            mask=col_mask[None, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)
    otr = tl.sum(Xtopr * sxr[None, :] - Xtopi * sxi[None, :], axis=1)
    oti = tl.sum(Xtopr * sxi[None, :] + Xtopi * sxr[None, :], axis=1)
    obr = tl.sum(Xbotr * sxr[None, :] - Xboti * sxi[None, :], axis=1)
    obi = tl.sum(Xbotr * sxi[None, :] + Xboti * sxr[None, :], axis=1)
    tl.store(x_ptr + (rows_t * 2)[:, None] + two[None, :], tl.join(otr, oti))
    tl.store(x_ptr + (rows_b * 2)[:, None] + two[None, :], tl.join(obr, obi))
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def _cz_neumann16(Nr, Ni, offs16, DT: tl.constexpr, IP: tl.constexpr):
    I16 = tl.where(
        offs16[:, None] == offs16[None, :],
        tl.full((16, 16), 1.0, DT),
        tl.full((16, 16), 0.0, DT),
    )
    Xr = I16 - Nr
    Xi = -Ni
    Sr = tl.dot(Nr, Nr, input_precision=IP) - tl.dot(Ni, Ni, input_precision=IP)
    Si = tl.dot(Nr, Ni, input_precision=IP) + tl.dot(Ni, Nr, input_precision=IP)
    for k in tl.static_range(0, 3):
        dr = tl.dot(Xr, Sr, input_precision=IP) - tl.dot(Xi, Si, input_precision=IP)
        di = tl.dot(Xr, Si, input_precision=IP) + tl.dot(Xi, Sr, input_precision=IP)
        Xr = Xr + dr
        Xi = Xi + di
        if k < 2:
            dr = tl.dot(Sr, Sr, input_precision=IP) - tl.dot(Si, Si, input_precision=IP)
            di = tl.dot(Sr, Si, input_precision=IP) + tl.dot(Si, Sr, input_precision=IP)
            Sr = dr
            Si = di
    return Xr, Xi


@triton.jit
def ztrsv_lu256_neu16_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs16 = tl.arange(0, 16)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    rows_t = row_start + offs16
    rows_b = row_start + 16 + offs16
    strict16 = offs16[:, None] > offs16[None, :]
    a_off = (rows_t[:, None] + rows_t[None, :] * 256) * 2
    c_off = (rows_b[:, None] + rows_b[None, :] * 256) * 2
    b_off = (rows_b[:, None] + rows_t[None, :] * 256) * 2
    NAr = tl.load(a_ptr + a_off, mask=strict16, other=0.0, eviction_policy="evict_last")
    NAi = tl.load(
        a_ptr + a_off + 1, mask=strict16, other=0.0, eviction_policy="evict_last"
    )
    NCr = tl.load(a_ptr + c_off, mask=strict16, other=0.0, eviction_policy="evict_last")
    NCi = tl.load(
        a_ptr + c_off + 1, mask=strict16, other=0.0, eviction_policy="evict_last"
    )
    Br = tl.load(a_ptr + b_off, eviction_policy="evict_last")
    Bi = tl.load(a_ptr + b_off + 1, eviction_policy="evict_last")
    XAr, XAi = _cz_neumann16(NAr, NAi, offs16, tl.float64, "ieee")
    XCr, XCi = _cz_neumann16(NCr, NCi, offs16, tl.float64, "ieee")
    st2 = tl.load(
        x_ptr + (rows_t * 2)[:, None] + tl.arange(0, 2)[None, :],
        eviction_policy="evict_last",
    )
    sxtr, sxti = tl.split(st2)
    sb2 = tl.load(
        x_ptr + (rows_b * 2)[:, None] + tl.arange(0, 2)[None, :],
        eviction_policy="evict_last",
    )
    sxbr, sxbi = tl.split(sb2)

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        kcols = pid_lo * BLOCK_N + koffs
        col_mask = kcols < pid_hi * BLOCK_N
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        at_off = (rows_t[:, None] + kcols[None, :] * 256) * 2
        a2 = tl.load(
            a_ptr + at_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=col_mask[None, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        sxtr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxti -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)
        ab_off = (rows_b[:, None] + kcols[None, :] * 256) * 2
        a2 = tl.load(
            a_ptr + ab_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=col_mask[None, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        sxbr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxbi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    otr = tl.sum(XAr * sxtr[None, :] - XAi * sxti[None, :], axis=1)
    oti = tl.sum(XAr * sxti[None, :] + XAi * sxtr[None, :], axis=1)
    tr = sxbr - tl.sum(Br * otr[None, :] - Bi * oti[None, :], axis=1)
    ti = sxbi - tl.sum(Br * oti[None, :] + Bi * otr[None, :], axis=1)
    obr = tl.sum(XCr * tr[None, :] - XCi * ti[None, :], axis=1)
    obi = tl.sum(XCr * ti[None, :] + XCi * tr[None, :], axis=1)
    tl.store(
        x_ptr + (rows_t * 2)[:, None] + tl.arange(0, 2)[None, :],
        tl.join(otr, oti),
    )
    tl.store(
        x_ptr + (rows_b * 2)[:, None] + tl.arange(0, 2)[None, :],
        tl.join(obr, obi),
    )
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def strsv_un256_rowinv_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels: tl.constexpr = 8
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    rows = row_start + offs
    dinv = _diag_block_inv_nonunit_upper_rowload(a_ptr, row_start, offs, BLOCK_N)
    sx = tl.load(x_ptr + rows, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * 256,
            mask=col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_fwd"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_fwd_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)

    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size

    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    if TRANS == 1:
        rr = row_start + offs
        blk_off = rr[None, :] + rr[:, None] * LDA
        blk_mask = row_mask[:, None] & row_mask[None, :]
        diag_blk = tl.load(
            a_ptr + blk_off, mask=blk_mask, other=0.0, eviction_policy="evict_last"
        )
        for step in tl.static_range(0, BLOCK_N):
            active = step < size
            row_sel = offs == step
            arow = tl.sum(tl.where(row_sel[:, None], diag_blk, 0.0), axis=0)
            dval = tl.sum(tl.where(row_sel, arow, 0.0), axis=0)
            col_mask = (offs < step) & row_mask
            arow = tl.where(col_mask, arow, 0.0)
            inner_sum = tl.sum(arow * sx, axis=0)
            cur = tl.sum(tl.where(row_sel, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                cur = cur / tl.where(active, dval, 1.0)
            sx = tl.where(row_sel, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            active = step < size
            row = row_start + step
            cols = row_start + offs
            col_mask = (offs < step) & row_mask
            a_off = row + cols * LDA
            a_vals = tl.load(a_ptr + a_off, mask=col_mask & active, other=0.0)
            inner_sum = tl.sum(a_vals * sx, axis=0)
            cur = tl.sum(tl.where(offs == step, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                diag = tl.load(a_ptr + row + row * LDA, mask=active, other=1.0)
                cur = cur / diag
            sx = tl.where(offs == step, cur, sx)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_bwd"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_bwd_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)

    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size

    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    if TRANS == 1:
        rr = row_start + offs
        blk_off = rr[None, :] + rr[:, None] * LDA
        blk_mask = row_mask[:, None] & row_mask[None, :]
        diag_blk = tl.load(
            a_ptr + blk_off, mask=blk_mask, other=0.0, eviction_policy="evict_last"
        )
        for step in tl.static_range(0, BLOCK_N):
            local = BLOCK_N - 1 - step
            active = local < size
            row_sel = offs == local
            arow = tl.sum(tl.where(row_sel[:, None], diag_blk, 0.0), axis=0)
            dval = tl.sum(tl.where(row_sel, arow, 0.0), axis=0)
            col_mask = (offs > local) & (offs < size)
            arow = tl.where(col_mask, arow, 0.0)
            inner_sum = tl.sum(arow * sx, axis=0)
            cur = tl.sum(tl.where(row_sel, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                cur = cur / tl.where(active, dval, 1.0)
            sx = tl.where(row_sel, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            local = BLOCK_N - 1 - step
            active = local < size
            row = row_start + local
            cols = row_start + offs
            col_mask = (offs > local) & (offs < size)
            a_off = row + cols * LDA
            a_vals = tl.load(a_ptr + a_off, mask=col_mask & active, other=0.0)
            inner_sum = tl.sum(a_vals * sx, axis=0)
            cur = tl.sum(tl.where(offs == local, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                diag = tl.load(a_ptr + row + row * LDA, mask=active, other=1.0)
                cur = cur / diag
            sx = tl.where(offs == local, cur, sx)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def strsv_diag_inv_kernel(
    a_ptr,
    dinv_ptr,
    n,
    LDA,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BB: tl.constexpr,
):
    blk = tl.program_id(0)
    offs = tl.arange(0, BB)
    s = blk * BB
    grow = s + offs
    rmask = grow < n
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = rmask[:, None] & rmask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(rmask, diag_each, 1.0)
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    if LOWER_EFF:
        for i in tl.range(0, BB):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BB):
            i = BB - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    d_off = blk * BB * BB + offs[:, None] * BB + offs[None, :]
    tl.store(dinv_ptr + d_off, X, mask=mmask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_fwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_fwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = pid * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_bwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_bwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)

    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = blk * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def _diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = row_mask[:, None] & row_mask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(row_mask, diag_each, 1.0)
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_fwd_fused"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N
    )
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_bwd_fused"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)

    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N
    )
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def dtrsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    size = end - start
    for step in tl.static_range(0, BLOCK_N):
        if FORWARD:
            i = start + step
            active = step < size
            j = start + offs
            mask = (offs < step) & (offs < size)
        else:
            i = end - 1 - step
            active = step < size
            j = end - 1 - offs
            mask = (offs < step) & (offs < size)
        mask = mask & active
        if TRANS == 0:
            a_off = i + j * LDA
        else:
            a_off = j + i * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + j * INCX, mask=mask, other=0.0)
        value = tl.load(x_ptr + i * INCX, mask=active, other=0.0) - tl.sum(
            a_vals * x_vals, axis=0
        )
        if not UNIT:
            diag = tl.load(a_ptr + i + i * LDA, mask=active, other=1.0)
            value = value / diag
        tl.store(x_ptr + i * INCX, value, mask=active)


@libentry()
@triton.jit
def dtrsv_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    LDA,
    INCX,
    TRANS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, BLOCK_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_off = rows[:, None] + cols[None, :] * LDA
    else:
        a_off = cols[None, :] + rows[:, None] * LDA
    mask = row_mask[:, None] & col_mask[None, :]
    a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
    acc = tl.sum(a_vals * x_vals[None, :], axis=1)
    rhs = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * INCX, rhs - acc, mask=row_mask)


@triton.jit
def dtrsv_diag_inv_kernel(
    a_ptr,
    dinv_ptr,
    n,
    LDA,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BB: tl.constexpr,
):
    blk = tl.program_id(0)
    offs = tl.arange(0, BB)
    s = blk * BB
    grow = s + offs
    rmask = grow < n
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = rmask[:, None] & rmask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    one = tl.full((BB,), 1.0, tl.float64)
    zero = tl.full((BB,), 0.0, tl.float64)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(rmask, diag_each, one)
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BB, BB), 1.0, tl.float64),
        tl.full((BB, BB), 0.0, tl.float64),
    )
    if LOWER_EFF:
        for i in tl.range(0, BB):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BB):
            i = BB - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    d_off = blk * BB * BB + offs[:, None] * BB + offs[None, :]
    tl.store(dinv_ptr + d_off, X, mask=mmask)


@triton.jit
def _dtrsv_diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    if UNIT and LOWER_EFF and TRANS == 0 and ROWLOAD:
        if BLOCK_N == 16:
            LEVELS: tl.constexpr = 3
        elif BLOCK_N == 32:
            LEVELS: tl.constexpr = 4
        else:
            LEVELS: tl.constexpr = 5
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
        strict = (offs[:, None] > offs[None, :]) & row_mask[:, None] & row_mask[None, :]
        M = tl.load(a_ptr + m_off, mask=strict, other=0.0, eviction_policy="evict_last")
        X = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
        )
        X = X - M
        S = tl.dot(M, M, input_precision="ieee")
        for k in tl.static_range(0, LEVELS):
            X = X + tl.dot(X, S, input_precision="ieee")
            if k < LEVELS - 1:
                S = tl.dot(S, S, input_precision="ieee")
        return X
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = row_mask[:, None] & row_mask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    one = tl.full((BLOCK_N,), 1.0, tl.float64)
    zero = tl.full((BLOCK_N,), 0.0, tl.float64)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(row_mask, diag_each, one)
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
    )
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@triton.jit
def _dtrsv_diag_block_inv_32_nomask(
    a_ptr, s, offs, LOWER_EFF: tl.constexpr, BLOCK_N: tl.constexpr
):
    m_off = (s + offs)[:, None] + (s + offs)[None, :] * 256
    M = tl.load(a_ptr + m_off)
    diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
    rinv = 1.0 / diag_each
    if LOWER_EFF:
        strict = offs[:, None] > offs[None, :]
    else:
        strict = offs[:, None] < offs[None, :]
    M = tl.where(strict, M, tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64))
    M = M * rinv[:, None]
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
    )
    X = X - M
    S = tl.dot(M, M, input_precision="ieee")
    for k in tl.static_range(0, 4):
        X = X + tl.dot(X, S, input_precision="ieee")
        if k < 3:
            S = tl.dot(S, S, input_precision="ieee")
    X = X * rinv[None, :]
    return X


@triton.jit
def dtrsv_n256_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels: tl.constexpr = 8
    if LOWER_EFF:
        blk = pid
    else:
        blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    rows = row_start + offs
    dinv = _dtrsv_diag_block_inv_32_nomask(a_ptr, row_start, offs, LOWER_EFF, BLOCK_N)
    sx = tl.load(x_ptr + rows, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        if LOWER_EFF:
            c0 = pid_lo * BLOCK_N
            c1 = pid_hi * BLOCK_N
        else:
            c0 = (num_panels - pid_hi) * BLOCK_N
            c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * 256,
            mask=col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dtrsv_fwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def dtrsv_fwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = pid * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dtrsv_bwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def dtrsv_bwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = blk * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def dtrsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _dtrsv_diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N, ROWLOAD
    )
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def dtrsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _dtrsv_diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N, ROWLOAD
    )
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def ctrsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    size = end - start
    incx2 = INCX * 2
    for step in tl.static_range(0, BLOCK_N):
        if FORWARD:
            i = start + step
            active = step < size
            j = start + offs
            mask = (offs < step) & (offs < size)
        else:
            i = end - 1 - step
            active = step < size
            j = end - 1 - offs
            mask = (offs < step) & (offs < size)
        mask = mask & active
        if TRANS == 0:
            a_base = (i + j * LDA) * 2
        else:
            a_base = (j + i * LDA) * 2
        ar = tl.load(
            a_ptr + a_base, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        ai = tl.load(
            a_ptr + a_base + 1, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        xr = tl.load(
            x_ptr + j * incx2, mask=mask, other=0.0, eviction_policy="evict_last"
        )
        xi = tl.load(
            x_ptr + j * incx2 + 1, mask=mask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            ai = -ai
        value_r = tl.load(x_ptr + i * incx2, mask=active, other=0.0) - tl.sum(
            ar * xr - ai * xi, axis=0
        )
        value_i = tl.load(x_ptr + i * incx2 + 1, mask=active, other=0.0) - tl.sum(
            ar * xi + ai * xr, axis=0
        )
        if not UNIT:
            diag_r = tl.load(a_ptr + (i + i * LDA) * 2, mask=active, other=1.0)
            diag_i = tl.load(a_ptr + (i + i * LDA) * 2 + 1, mask=active, other=0.0)
            if CONJ:
                diag_i = -diag_i
            denom = diag_r * diag_r + diag_i * diag_i
            out_r = (value_r * diag_r + value_i * diag_i) / denom
            out_i = (value_i * diag_r - value_r * diag_i) / denom
            value_r = out_r
            value_i = out_i
        tl.store(x_ptr + i * incx2, value_r, mask=active)
        tl.store(x_ptr + i * incx2 + 1, value_i, mask=active)


@libentry()
@triton.jit
def ztrsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    size = end - start
    incx2 = INCX * 2
    for step in tl.static_range(0, BLOCK_N):
        if FORWARD:
            i = start + step
            active = step < size
            j = start + offs
            mask = (offs < step) & (offs < size)
        else:
            i = end - 1 - step
            active = step < size
            j = end - 1 - offs
            mask = (offs < step) & (offs < size)
        mask = mask & active
        if TRANS == 0:
            a_base = (i + j * LDA) * 2
        else:
            a_base = (j + i * LDA) * 2
        ar = tl.load(a_ptr + a_base, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_base + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + j * incx2, mask=mask, other=0.0)
        xi = tl.load(x_ptr + j * incx2 + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        value_r = tl.load(x_ptr + i * incx2, mask=active, other=0.0) - tl.sum(
            ar * xr - ai * xi, axis=0
        )
        value_i = tl.load(x_ptr + i * incx2 + 1, mask=active, other=0.0) - tl.sum(
            ar * xi + ai * xr, axis=0
        )
        if not UNIT:
            diag_r = tl.load(a_ptr + (i + i * LDA) * 2, mask=active, other=1.0)
            diag_i = tl.load(a_ptr + (i + i * LDA) * 2 + 1, mask=active, other=0.0)
            if CONJ:
                diag_i = -diag_i
            denom = diag_r * diag_r + diag_i * diag_i
            out_r = (value_r * diag_r + value_i * diag_i) / denom
            out_i = (value_i * diag_r - value_r * diag_i) / denom
            value_r = out_r
            value_i = out_i
        tl.store(x_ptr + i * incx2, value_r, mask=active)
        tl.store(x_ptr + i * incx2 + 1, value_i, mask=active)


@libentry()
@triton.jit
def ctrsv_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    LDA,
    INCX,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, BLOCK_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_base = (rows[:, None] + cols[None, :] * LDA) * 2
    else:
        a_base = (cols[None, :] + rows[:, None] * LDA) * 2
    mask = row_mask[:, None] & col_mask[None, :]
    ar = tl.load(a_ptr + a_base, mask=mask, other=0.0, eviction_policy="evict_first")
    ai = tl.load(
        a_ptr + a_base + 1, mask=mask, other=0.0, eviction_policy="evict_first"
    )
    xr = tl.load(
        x_ptr + cols * INCX * 2, mask=col_mask, other=0.0, eviction_policy="evict_last"
    )
    xi = tl.load(
        x_ptr + cols * INCX * 2 + 1,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    if CONJ:
        ai = -ai
    acc_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
    acc_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
    rhs_r = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    rhs_i = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * INCX * 2, rhs_r - acc_r, mask=row_mask)
    tl.store(x_ptr + rows * INCX * 2 + 1, rhs_i - acc_i, mask=row_mask)


@libentry()
@triton.jit
def ztrsv_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    LDA,
    INCX,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, BLOCK_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_base = (rows[:, None] + cols[None, :] * LDA) * 2
    else:
        a_base = (cols[None, :] + rows[:, None] * LDA) * 2
    mask = row_mask[:, None] & col_mask[None, :]
    ar = tl.load(a_ptr + a_base, mask=mask, other=0.0)
    ai = tl.load(a_ptr + a_base + 1, mask=mask, other=0.0)
    xr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
    xi = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
    if CONJ:
        ai = -ai
    acc_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
    acc_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
    rhs_r = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    rhs_i = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * INCX * 2, rhs_r - acc_r, mask=row_mask)
    tl.store(x_ptr + rows * INCX * 2 + 1, rhs_i - acc_i, mask=row_mask)


def _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok):
    assert A.is_contiguous() and x.is_contiguous()
    assert A.device == x.device
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
    assert lda >= max(1, n)
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= n * lda


def _row_major_dispatch(uplo, trans):
    internal_uplo = (
        CUBLAS_FILL_MODE_UPPER
        if uplo == CUBLAS_FILL_MODE_LOWER
        else CUBLAS_FILL_MODE_LOWER
    )
    trans_flag = 1 if trans == CUBLAS_OP_N else 0
    conj = 1 if trans == CUBLAS_OP_C else 0
    return internal_uplo, trans_flag, conj


def _mode_key(uplo, trans, unit):
    return (uplo << 4) | (trans << 2) | unit


def _forward(uplo, trans):
    if trans == CUBLAS_OP_N:
        return 1 if uplo == CUBLAS_FILL_MODE_LOWER else 0
    return 1 if uplo == CUBLAS_FILL_MODE_UPPER else 0


def _panel_ranges(n, block_n, forward):
    if forward:
        start = 0
        while start < n:
            end = min(start + block_n, n)
            yield start, end
            start = end
    else:
        end = n
        while end > 0:
            start = max(0, end - block_n)
            yield start, end
            end = start


def strsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, _ = _row_major_dispatch(uplo, trans)
    forward = _forward(uplo, trans_flag)
    mode_key = _mode_key(uplo, trans_flag, unit)

    with torch_device_fn.device(A.device):
        if trans_flag == 0 and incx == 1 and lda == n and n <= 64:
            strsv_n64_kernel[(1,)](
                A,
                x,
                n,
                LOWER=1 if uplo == CUBLAS_FILL_MODE_LOWER else 0,
                UNIT=unit,
                BLOCK_N=64,
                num_warps=1,
            )
            return
        if (
            trans_flag == 0
            and incx == 1
            and lda == n
            and n >= 256
            and uplo == CUBLAS_FILL_MODE_LOWER
            and unit == 1
        ):
            flags = _trsv_flags(A.device)
            strsv_lu_neu64_kernel[(triton.cdiv(n, 64),)](
                A,
                x,
                flags,
                n,
                lda,
                BLOCK_N=64,
                CHUNK=2 if n <= 256 else 1,
                num_warps=4,
            )
            return
        if (
            trans_flag == 0
            and incx == 1
            and lda == n
            and n == 256
            and uplo == CUBLAS_FILL_MODE_LOWER
            and unit == 1
        ):
            flags = _trsv_flags(A.device)
            strsv_lu256_rowinv_kernel[(8,)](
                A,
                x,
                flags,
                BLOCK_N=32,
                CHUNK=2,
                num_warps=4,
            )
            return

        if (
            trans_flag == 0
            and incx == 1
            and lda == n
            and n >= 256
            and uplo == CUBLAS_FILL_MODE_UPPER
            and unit == 0
        ):
            flags = _trsv_flags(A.device)
            strsv_un_neu64_kernel[(triton.cdiv(n, 64),)](
                A,
                x,
                flags,
                n,
                lda,
                BLOCK_N=64,
                CHUNK=2 if n <= 256 else 1,
                num_warps=4,
            )
            return
        if (
            trans_flag == 0
            and incx == 1
            and lda == n
            and n == 256
            and uplo == CUBLAS_FILL_MODE_UPPER
            and unit == 0
        ):
            flags = _trsv_flags(A.device)
            strsv_un256_rowinv_kernel[(8,)](
                A,
                x,
                flags,
                BLOCK_N=32,
                CHUNK=2,
                num_warps=4,
            )
            return

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_N"]),)

        flags = _trsv_flags(A.device)
        if n >= _STRSV_INV_MIN_N:
            bb = 64 if n >= _STRSV_INV_BB64_MIN_N else 32
            npanel = triton.cdiv(n, bb)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            inv_grid = (npanel,)
            if bb == 32 and n <= _STRSV_FUSE_MAX_N and trans_flag == 0:
                if forward:
                    strsv_fwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                    )
                else:
                    strsv_bwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                    )
            else:
                dinv = torch.empty(
                    (npanel, bb, bb), dtype=torch.float32, device=A.device
                )
                strsv_diag_inv_kernel[(npanel,)](
                    A,
                    dinv,
                    n,
                    lda,
                    TRANS=trans_flag,
                    UNIT=unit,
                    LOWER_EFF=lower_eff,
                    BB=bb,
                    num_warps=1 if bb == 32 else 2,
                )
                if forward:
                    strsv_fwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
                else:
                    strsv_bwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
        elif forward:
            strsv_fwd_kernel[grid](
                A,
                x,
                flags,
                n,
                lda,
                incx,
                mode_key,
                TRANS=trans_flag,
                UNIT=unit,
            )
        else:
            strsv_bwd_kernel[grid](
                A,
                x,
                flags,
                n,
                lda,
                incx,
                mode_key,
                TRANS=trans_flag,
                UNIT=unit,
            )


def dtrsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float64 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, _ = _row_major_dispatch(uplo, trans)
    forward = _forward(uplo, trans_flag)
    mode_key = _mode_key(uplo, trans_flag, unit)

    with torch_device_fn.device(A.device):
        if incx == 1 and lda == n:
            if (
                trans_flag == 0
                and uplo == CUBLAS_FILL_MODE_LOWER
                and unit == 1
                and n >= 1024
            ):
                flags = _trsv_flags(A.device)
                dtrsv_lu_neu64_kernel[(triton.cdiv(n, 64),)](
                    A,
                    x,
                    flags,
                    n,
                    lda,
                    BLOCK_N=64,
                    CHUNK=1,
                    num_warps=4,
                )
                return
            if (
                trans_flag == 0
                and uplo == CUBLAS_FILL_MODE_LOWER
                and unit == 1
                and n <= 64
            ):
                dtrsv_n64_kernel[(1,)](
                    A,
                    x,
                    n,
                    LOWER=1,
                    UNIT=1,
                    BLOCK_N=64,
                    num_warps=1,
                )
                return
            if trans_flag == 0 and n == 256 and unit == 0:
                flags = _trsv_flags(A.device)
                dtrsv_n256_fused_kernel[(8,)](
                    A,
                    x,
                    flags,
                    LOWER_EFF=1 if uplo == CUBLAS_FILL_MODE_LOWER else 0,
                    BLOCK_N=32,
                    CHUNK=4,
                    num_warps=4,
                )
                return
            if n <= 64:
                bb = 16
            else:
                bb = 64 if n >= _DTRSV_INV_BB64_MIN_N else 32
            npanel = triton.cdiv(n, bb)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            inv_grid = (npanel,)
            rowload = 1 if n <= _DTRSV_ROWLOAD_MAX_N else 0
            if (bb == 16 or bb == 32) and n <= _DTRSV_FUSE_MAX_N:
                chunk, nw = _dtrsv_fused_cfg(forward, n, bb, unit)
                flags = _trsv_flags(A.device)
                if forward:
                    dtrsv_fwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                        CHUNK=chunk,
                        ROWLOAD=rowload,
                        num_warps=nw,
                    )
                else:
                    dtrsv_bwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                        CHUNK=chunk,
                        ROWLOAD=rowload,
                        num_warps=nw,
                    )
            else:
                flags = _trsv_flags(A.device)
                dinv = torch.empty(
                    (npanel, bb, bb), dtype=torch.float64, device=A.device
                )
                dtrsv_diag_inv_kernel[(npanel,)](
                    A,
                    dinv,
                    n,
                    lda,
                    TRANS=trans_flag,
                    UNIT=unit,
                    LOWER_EFF=lower_eff,
                    BB=bb,
                    num_warps=1 if bb == 32 else 2,
                )
                if forward:
                    dtrsv_fwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
                else:
                    dtrsv_bwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
            return

        block_n = 16
        block_m = 128
        for start, end in _panel_ranges(n, block_n, forward):
            dtrsv_panel_kernel[(1,)](
                A,
                x,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                grid = (triton.cdiv(row_count, block_m),)
                dtrsv_update_kernel[grid](
                    A,
                    x,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )


@triton.jit
def _ctrsv_diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWLOAD: tl.constexpr,
    INV_DOT: tl.constexpr,
):
    if (
        BLOCK_N == 16
        and INV_DOT
        and not (UNIT and LOWER_EFF and TRANS == 0 and ROWLOAD)
    ):
        if TRANS == 0:
            m_off = ((s + offs)[:, None] + (s + offs)[None, :] * LDA) * 2
        else:
            m_off = ((s + offs)[None, :] + (s + offs)[:, None] * LDA) * 2
        if LOWER_EFF:
            strict = offs[:, None] > offs[None, :]
        else:
            strict = offs[:, None] < offs[None, :]
        smask = strict & row_mask[:, None] & row_mask[None, :]
        Mr = tl.load(a_ptr + m_off, mask=smask, other=0.0, eviction_policy="evict_last")
        Mi = tl.load(
            a_ptr + m_off + 1, mask=smask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            Mi = -Mi
        if not UNIT:
            d_off = ((s + offs) + (s + offs) * LDA) * 2
            nb_dr = tl.load(a_ptr + d_off, mask=row_mask, other=1.0)
            nb_di = tl.load(a_ptr + d_off + 1, mask=row_mask, other=0.0)
            if CONJ:
                nb_di = -nb_di
            nb_den = nb_dr * nb_dr + nb_di * nb_di
            nb_rr = nb_dr / nb_den
            nb_ri = -nb_di / nb_den
            nb_tmr = Mr * nb_rr[:, None] - Mi * nb_ri[:, None]
            Mi = Mr * nb_ri[:, None] + Mi * nb_rr[:, None]
            Mr = nb_tmr
        Xr = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
        )
        Xr = Xr - Mr
        Xi = -Mi
        Sr = tl.dot(Mr, Mr, input_precision="ieee") - tl.dot(
            Mi, Mi, input_precision="ieee"
        )
        Si = tl.dot(Mr, Mi, input_precision="ieee") + tl.dot(
            Mi, Mr, input_precision="ieee"
        )
        for k in tl.static_range(0, 3):
            nb_tr = tl.dot(Xr, Sr, input_precision="ieee") - tl.dot(
                Xi, Si, input_precision="ieee"
            )
            nb_ti = tl.dot(Xr, Si, input_precision="ieee") + tl.dot(
                Xi, Sr, input_precision="ieee"
            )
            Xr = Xr + nb_tr
            Xi = Xi + nb_ti
            if k < 2:
                nb_tr = tl.dot(Sr, Sr, input_precision="ieee") - tl.dot(
                    Si, Si, input_precision="ieee"
                )
                nb_ti = tl.dot(Sr, Si, input_precision="ieee") + tl.dot(
                    Si, Sr, input_precision="ieee"
                )
                Sr = nb_tr
                Si = nb_ti
        if not UNIT:
            nb_tr = Xr * nb_rr[None, :] - Xi * nb_ri[None, :]
            Xi = Xr * nb_ri[None, :] + Xi * nb_rr[None, :]
            Xr = nb_tr
        return Xr, Xi
    if UNIT and LOWER_EFF and TRANS == 0 and ROWLOAD:
        one = tl.full((BLOCK_N,), 1.0, tl.float32)
        zero = tl.full((BLOCK_N,), 0.0, tl.float32)
        Xr = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
        )
        Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
        for i in tl.static_range(0, BLOCK_N):
            roff = ((s + i) + (s + offs) * LDA) * 2
            mrr = tl.load(
                a_ptr + roff,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            mri = tl.load(
                a_ptr + roff + 1,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                mri = -mri
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            Xr = tl.where(offs[:, None] == i, (ei - cr)[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, (-ci)[None, :], Xi)
        return Xr, Xi
    if TRANS == 0:
        m_off = ((s + offs)[:, None] + (s + offs)[None, :] * LDA) * 2
    else:
        m_off = ((s + offs)[None, :] + (s + offs)[:, None] * LDA) * 2
    mmask = row_mask[:, None] & row_mask[None, :]
    Mr = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    Mi = tl.load(a_ptr + m_off + 1, mask=mmask, other=0.0)
    if CONJ:
        Mi = -Mi
    one = tl.full((BLOCK_N,), 1.0, tl.float32)
    zero = tl.full((BLOCK_N,), 0.0, tl.float32)
    if not UNIT:
        dr = tl.sum(tl.where(offs[:, None] == offs[None, :], Mr, 0.0), axis=1)
        di = tl.sum(tl.where(offs[:, None] == offs[None, :], Mi, 0.0), axis=1)
        dr = tl.where(row_mask, dr, one)
        di = tl.where(row_mask, di, zero)
        denom = dr * dr + di * di
    Xr = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
    )
    Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs < i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs > i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    return Xr, Xi


@libentry()
@triton.jit
def ctrsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
    INV_DOT: tl.constexpr,
    VEC64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    a_ptr_int64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_int64 = x_ptr.to(tl.pointer_type(tl.int64))
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ctrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
        INV_DOT,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x_bits = tl.load(
            x_ptr_int64 + kcols,
            mask=col_mask,
            other=0,
            eviction_policy="evict_last",
        )
        xpr = x_bits.to(tl.int32).to(tl.float32, bitcast=True)
        xpi = (x_bits >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        m2 = row_mask[:, None] & col_mask[None, :]
        a_bits = tl.load(
            a_ptr_int64 + a_off,
            mask=m2,
            other=0,
            eviction_policy="evict_first",
        )
        ar = a_bits.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (a_bits >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def ctrsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
    INV_DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ctrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
        INV_DOT,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


def ctrsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, conj = _row_major_dispatch(uplo, trans)
    forward = _forward(uplo, trans_flag)
    mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)

    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        if incx == 1 and lda == n:
            scalar_lower = (
                conj == 0
                and unit == 1
                and (
                    (trans_flag == 0 and uplo == CUBLAS_FILL_MODE_LOWER)
                    or (trans_flag == 1 and uplo == CUBLAS_FILL_MODE_UPPER)
                )
            )
            scalar_size = n == 256 or (
                n == 512 and trans_flag == 1 and uplo == CUBLAS_FILL_MODE_UPPER
            )
            if scalar_lower and scalar_size:
                flags = _trsv_flags(A.device)
                ctrsv_lu_scalar_kernel[(triton.cdiv(n, 32),)](
                    A_real,
                    x_real,
                    flags,
                    lda,
                    TRANS=trans_flag,
                    CHUNK=4,
                    num_warps=4,
                )
                return
            bb = 16 if n <= 64 else 32
            npanel = triton.cdiv(n, bb)
            flags = _trsv_flags(A.device)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            rowload = 1 if n <= _DTRSV_ROWLOAD_MAX_N else 0
            inv_dot = 1 if n <= 64 else 0
            chunk, nw = _ctrsv_fused_cfg(forward, n, bb, unit)
            if forward:
                ctrsv_fwd_fused_kernel[(npanel,)](
                    A_real,
                    x_real,
                    flags,
                    n,
                    lda,
                    mode_key,
                    TRANS=trans_flag,
                    UNIT=unit,
                    CONJ=conj,
                    LOWER_EFF=lower_eff,
                    BLOCK_N=bb,
                    CHUNK=chunk,
                    ROWLOAD=rowload,
                    INV_DOT=inv_dot,
                    VEC64=1 if n >= 1024 else 0,
                    num_warps=nw,
                )
            else:
                ctrsv_bwd_fused_kernel[(npanel,)](
                    A_real,
                    x_real,
                    flags,
                    n,
                    lda,
                    mode_key,
                    TRANS=trans_flag,
                    UNIT=unit,
                    CONJ=conj,
                    LOWER_EFF=lower_eff,
                    BLOCK_N=bb,
                    CHUNK=chunk,
                    ROWLOAD=rowload,
                    INV_DOT=inv_dot,
                    num_warps=nw,
                )
            return

        block_n = 16
        block_m = 128
        for start, end in _panel_ranges(n, block_n, forward):
            ctrsv_panel_kernel[(1,)](
                A_real,
                x_real,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                CONJ=conj,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                grid = (triton.cdiv(row_count, block_m),)
                ctrsv_update_kernel[grid](
                    A_real,
                    x_real,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    CONJ=conj,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )


@triton.jit
def _ztrsv_diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWLOAD: tl.constexpr,
    INV_DOT: tl.constexpr,
):
    if BLOCK_N == 16 and INV_DOT and not (UNIT and LOWER_EFF and ROWLOAD):
        if TRANS == 0:
            m_off = ((s + offs)[:, None] + (s + offs)[None, :] * LDA) * 2
        else:
            m_off = ((s + offs)[None, :] + (s + offs)[:, None] * LDA) * 2
        if LOWER_EFF:
            strict = offs[:, None] > offs[None, :]
        else:
            strict = offs[:, None] < offs[None, :]
        smask = strict & row_mask[:, None] & row_mask[None, :]
        Mr = tl.load(a_ptr + m_off, mask=smask, other=0.0, eviction_policy="evict_last")
        Mi = tl.load(
            a_ptr + m_off + 1, mask=smask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            Mi = -Mi
        if not UNIT:
            d_off = ((s + offs) + (s + offs) * LDA) * 2
            nb_dr = tl.load(a_ptr + d_off, mask=row_mask, other=1.0)
            nb_di = tl.load(a_ptr + d_off + 1, mask=row_mask, other=0.0)
            if CONJ:
                nb_di = -nb_di
            nb_den = nb_dr * nb_dr + nb_di * nb_di
            nb_rr = nb_dr / nb_den
            nb_ri = -nb_di / nb_den
            nb_tmr = Mr * nb_rr[:, None] - Mi * nb_ri[:, None]
            Mi = Mr * nb_ri[:, None] + Mi * nb_rr[:, None]
            Mr = nb_tmr
        Xr = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
        )
        Xr = Xr - Mr
        Xi = -Mi
        Sr = tl.dot(Mr, Mr, input_precision="ieee") - tl.dot(
            Mi, Mi, input_precision="ieee"
        )
        Si = tl.dot(Mr, Mi, input_precision="ieee") + tl.dot(
            Mi, Mr, input_precision="ieee"
        )
        for k in tl.static_range(0, 3):
            nb_tr = tl.dot(Xr, Sr, input_precision="ieee") - tl.dot(
                Xi, Si, input_precision="ieee"
            )
            nb_ti = tl.dot(Xr, Si, input_precision="ieee") + tl.dot(
                Xi, Sr, input_precision="ieee"
            )
            Xr = Xr + nb_tr
            Xi = Xi + nb_ti
            if k < 2:
                nb_tr = tl.dot(Sr, Sr, input_precision="ieee") - tl.dot(
                    Si, Si, input_precision="ieee"
                )
                nb_ti = tl.dot(Sr, Si, input_precision="ieee") + tl.dot(
                    Si, Sr, input_precision="ieee"
                )
                Sr = nb_tr
                Si = nb_ti
        if not UNIT:
            nb_tr = Xr * nb_rr[None, :] - Xi * nb_ri[None, :]
            Xi = Xr * nb_ri[None, :] + Xi * nb_rr[None, :]
            Xr = nb_tr
        return Xr, Xi
    if UNIT and LOWER_EFF and ROWLOAD:
        one = tl.full((BLOCK_N,), 1.0, tl.float64)
        zero = tl.full((BLOCK_N,), 0.0, tl.float64)
        Xr = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
        )
        Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float64)
        for i in tl.static_range(0, BLOCK_N):
            if TRANS == 0:
                roff = ((s + i) + (s + offs) * LDA) * 2
            else:
                roff = ((s + offs) + (s + i) * LDA) * 2
            mrr = tl.load(
                a_ptr + roff,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            mri = tl.load(
                a_ptr + roff + 1,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                mri = -mri
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            Xr = tl.where(offs[:, None] == i, (ei - cr)[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, (-ci)[None, :], Xi)
        return Xr, Xi
    if TRANS == 0:
        m_off = ((s + offs)[:, None] + (s + offs)[None, :] * LDA) * 2
    else:
        m_off = ((s + offs)[None, :] + (s + offs)[:, None] * LDA) * 2
    mmask = row_mask[:, None] & row_mask[None, :]
    Mr = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    Mi = tl.load(a_ptr + m_off + 1, mask=mmask, other=0.0)
    if CONJ:
        Mi = -Mi
    one = tl.full((BLOCK_N,), 1.0, tl.float64)
    zero = tl.full((BLOCK_N,), 0.0, tl.float64)
    if not UNIT:
        dr = tl.sum(tl.where(offs[:, None] == offs[None, :], Mr, 0.0), axis=1)
        di = tl.sum(tl.where(offs[:, None] == offs[None, :], Mi, 0.0), axis=1)
        dr = tl.where(row_mask, dr, one)
        di = tl.where(row_mask, di, zero)
        denom = dr * dr + di * di
    Xr = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
    )
    Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float64)
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs < i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs > i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    return Xr, Xi


@libentry()
@triton.jit
def ztrsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
    INV_DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ztrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
        INV_DOT,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def ztrsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
    INV_DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ztrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
        INV_DOT,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


def ztrsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex128 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, conj = _row_major_dispatch(uplo, trans)
    forward = _forward(uplo, trans_flag)
    mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)

    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        if incx == 1 and lda == n:
            inv_precompute = (
                n == 8192
                and unit == 1
                and conj == 0
                and trans_flag == 1
                and uplo == CUBLAS_FILL_MODE_UPPER
            )
            if inv_precompute:
                bb = 32
                npanel = triton.cdiv(n, bb)
                flags = _trsv_flags(A.device)
                dinv = torch.empty((npanel, bb, bb), dtype=A.dtype, device=A.device)
                dinv_real = torch.view_as_real(dinv)
                ztrsv_diag_inv_kernel[(npanel,)](
                    A_real, dinv_real, n, lda, TRANS=trans_flag, BLOCK_N=bb, num_warps=4
                )
                ztrsv_fwd_inv_kernel[(npanel,)](
                    A_real,
                    x_real,
                    dinv_real,
                    flags,
                    n,
                    lda,
                    TRANS=trans_flag,
                    BLOCK_N=bb,
                    CHUNK=1,
                    num_warps=4,
                )
                return
            if (
                n == 256
                and trans_flag == 0
                and conj == 0
                and unit == 1
                and uplo == CUBLAS_FILL_MODE_LOWER
            ):
                flags = _trsv_flags(A.device)
                ztrsv_lu256_neu16_kernel[(8,)](
                    A_real,
                    x_real,
                    flags,
                    BLOCK_N=32,
                    CHUNK=1,
                    num_warps=4,
                )
                return
            if unit:
                bb = 16 if n <= 64 else 32
            else:
                bb = 16 if n <= 512 else 32
            npanel = triton.cdiv(n, bb)
            flags = _trsv_flags(A.device)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            rowload = 1 if n <= _ZTRSV_ROWLOAD_MAX_N else 0
            inv_dot = 1 if n <= 64 else 0
            chunk, nw = _ztrsv_fused_cfg(forward, n, bb, unit)
            if forward:
                ztrsv_fwd_fused_kernel[(npanel,)](
                    A_real,
                    x_real,
                    flags,
                    n,
                    lda,
                    mode_key,
                    TRANS=trans_flag,
                    UNIT=unit,
                    CONJ=conj,
                    LOWER_EFF=lower_eff,
                    BLOCK_N=bb,
                    CHUNK=chunk,
                    ROWLOAD=rowload,
                    INV_DOT=inv_dot,
                    num_warps=nw,
                )
            else:
                ztrsv_bwd_fused_kernel[(npanel,)](
                    A_real,
                    x_real,
                    flags,
                    n,
                    lda,
                    mode_key,
                    TRANS=trans_flag,
                    UNIT=unit,
                    CONJ=conj,
                    LOWER_EFF=lower_eff,
                    BLOCK_N=bb,
                    CHUNK=chunk,
                    ROWLOAD=rowload,
                    INV_DOT=inv_dot,
                    num_warps=nw,
                )
            return

        block_n = 8
        block_m = 64
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        for start, end in _panel_ranges(n, block_n, forward):
            ztrsv_panel_kernel[(1,)](
                A_real,
                x_real,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                CONJ=conj,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                grid = (triton.cdiv(row_count, block_m),)
                ztrsv_update_kernel[grid](
                    A_real,
                    x_real,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    CONJ=conj,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )


@triton.jit
def ztrsv_diag_inv_kernel(
    a_ptr,
    dinv_ptr,
    n,
    LDA,
    TRANS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    s = pid * BLOCK_N
    row_mask = s + offs < n
    Xr, Xi = _ztrsv_diag_block_inv(
        a_ptr,
        s,
        LDA,
        offs,
        row_mask,
        TRANS,
        1,
        0,
        1,
        BLOCK_N,
        1,
        0,
    )
    out = pid * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    mask = row_mask[:, None] & row_mask[None, :]
    tl.store(dinv_ptr + out * 2, Xr, mask=mask)
    tl.store(dinv_ptr + out * 2 + 1, Xi, mask=mask)


@triton.jit
def ztrsv_fwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    TRANS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    rows = row_start + offs
    row_mask = rows < n
    inv_off = pid * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    mask = row_mask[:, None] & row_mask[None, :]
    Xr = tl.load(dinv_ptr + inv_off * 2, mask=mask, other=0.0)
    Xi = tl.load(dinv_ptr + inv_off * 2 + 1, mask=mask, other=0.0)
    sxr = tl.load(x_ptr + rows * 2, mask=row_mask, other=0.0)
    sxi = tl.load(x_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        kcols = pid_lo * BLOCK_N + koffs
        col_mask = (kcols < pid_hi * BLOCK_N) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")

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
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.device_utils import get_stream_id
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_DIAG_UNIT = 1

_TBSV_KEY = ["n", "k_bucket", "mode_key"]
_TBSV_RESTORE = ["x_ptr"]

_TBSV_FLAG_POOL = {}
_TBSV_FLAG_SLOTS = 256


def _tbsv_flags(device):
    try:
        if torch_device_fn.is_current_stream_capturing():
            return torch.full((1,), -1, dtype=torch.int32, device=device)
    except AttributeError:
        pass
    key = (device, get_stream_id(torch_device_fn, device))
    entry = _TBSV_FLAG_POOL.get(key)
    if entry is None:
        pool = torch.full((_TBSV_FLAG_SLOTS * 4,), -1, dtype=torch.int32, device=device)
        views = [pool[index * 4 : index * 4 + 1] for index in range(_TBSV_FLAG_SLOTS)]
        entry = [pool, views, 0]
        _TBSV_FLAG_POOL[key] = entry
    pool, views, index = entry
    if index >= _TBSV_FLAG_SLOTS:
        pool.fill_(-1)
        index = 0
    entry[2] = index + 1
    return views[index]


def _band_bucket(k: int) -> int:
    if k <= 1:
        return 1
    b = 1
    while b < k and b < 1024:
        b <<= 1
    return b


def _mode_key(uplo: int, trans: int, unit: int) -> int:
    return (uplo << 4) | (trans << 2) | unit


@triton.jit
def _tbsv_band_offset(row, col, k, lda, UPLO: tl.constexpr):
    if UPLO == 1:
        return k + row - col + col * lda
    return row - col + col * lda


@triton.jit
def _real_affine_combine(a_left, b_left, a_right, b_right):
    return a_right * a_left, a_right * b_left + b_right


@triton.jit
def _complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _complex_affine_combine(
    ar_left,
    ai_left,
    br_left,
    bi_left,
    ar_right,
    ai_right,
    br_right,
    bi_right,
):
    ar, ai = _complex_mul(ar_right, ai_right, ar_left, ai_left)
    carry_r, carry_i = _complex_mul(ar_right, ai_right, br_left, bi_left)
    return ar, ai, carry_r + br_right, carry_i + bi_right


@triton.jit
def _real_tbsv_diag_block_inv(
    A,
    start,
    n,
    k,
    lda,
    offs,
    row_mask,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = start + offs
    if TRANS == 0:
        a_off = _tbsv_band_offset(rows[:, None], rows[None, :], k, lda, UPLO)
    else:
        a_off = _tbsv_band_offset(rows[None, :], rows[:, None], k, lda, UPLO)
    distance = tl.abs(rows[:, None] - rows[None, :])
    tile_mask = row_mask[:, None] & row_mask[None, :] & (distance <= k)
    matrix = tl.load(A + a_off, mask=tile_mask, other=0.0)
    if IS_DOUBLE:
        one = tl.full((BLOCK_N,), 1.0, tl.float64)
        zero = tl.full((BLOCK_N,), 0.0, tl.float64)
        inverse = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
        )
    else:
        one = tl.full((BLOCK_N,), 1.0, tl.float32)
        zero = tl.full((BLOCK_N,), 0.0, tl.float32)
        inverse = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
        )
    if not UNIT:
        diagonal = tl.sum(
            tl.where(offs[:, None] == offs[None, :], matrix, zero), axis=1
        )
        safe_diagonal = tl.where(row_mask, diagonal, one)
    if LOWER_EFF:
        for index in tl.range(0, BLOCK_N):
            matrix_row = tl.sum(tl.where(offs[:, None] == index, matrix, zero), axis=0)
            strict_row = tl.where(offs < index, matrix_row, zero)
            contribution = tl.sum(strict_row[:, None] * inverse, axis=0)
            identity_row = tl.where(offs == index, one, zero)
            if UNIT:
                inverse_row = identity_row - contribution
            else:
                diag_value = tl.sum(
                    tl.where(offs == index, safe_diagonal, zero), axis=0
                )
                inverse_row = (identity_row - contribution) / diag_value
            inverse = tl.where(offs[:, None] == index, inverse_row[None, :], inverse)
    else:
        for reverse_index in tl.range(0, BLOCK_N):
            index = BLOCK_N - 1 - reverse_index
            matrix_row = tl.sum(tl.where(offs[:, None] == index, matrix, zero), axis=0)
            strict_row = tl.where(offs > index, matrix_row, zero)
            contribution = tl.sum(strict_row[:, None] * inverse, axis=0)
            identity_row = tl.where(offs == index, one, zero)
            if UNIT:
                inverse_row = identity_row - contribution
            else:
                diag_value = tl.sum(
                    tl.where(offs == index, safe_diagonal, zero), axis=0
                )
                inverse_row = (identity_row - contribution) / diag_value
            inverse = tl.where(offs[:, None] == index, inverse_row[None, :], inverse)
    return inverse


@libentry()
@triton.jit
def _real_tbsv_blocked_kernel(
    A,
    x,
    flags,
    n,
    k,
    lda,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    FORWARD: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BAND_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    band_offs = tl.arange(0, BAND_K)
    panel_count = tl.cdiv(n, BLOCK_N)
    if FORWARD:
        block = pid
    else:
        block = panel_count - 1 - pid
    row_start = block * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    rows = row_start + offs
    row_mask = rows < row_end
    inverse = _real_tbsv_diag_block_inv(
        A,
        row_start,
        n,
        k,
        lda,
        offs,
        row_mask,
        UPLO,
        TRANS,
        UNIT,
        LOWER_EFF,
        IS_DOUBLE,
        BLOCK_N,
    )
    rhs = tl.load(x + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    while tl.atomic_add(flags, 0, sem="acquire") < pid - 1:
        pass
    if FORWARD:
        col_start = tl.maximum(0, row_start - k)
        col_end = row_start
    else:
        col_start = row_end
        col_end = tl.minimum(n, row_end + k)
    cols = col_start + band_offs
    col_mask = cols < col_end
    solved = tl.load(x + cols, mask=col_mask, other=0.0, eviction_policy="evict_last")
    if TRANS == 0:
        a_off = _tbsv_band_offset(rows[:, None], cols[None, :], k, lda, UPLO)
    else:
        a_off = _tbsv_band_offset(cols[None, :], rows[:, None], k, lda, UPLO)
    distance = tl.abs(rows[:, None] - cols[None, :])
    matrix_mask = row_mask[:, None] & col_mask[None, :] & (distance <= k)
    matrix = tl.load(
        A + a_off,
        mask=matrix_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    rhs -= tl.sum(matrix * solved[None, :], axis=1)

    result = tl.sum(inverse * rhs[None, :], axis=1)
    tl.store(x + rows, result, mask=row_mask)
    tl.atomic_xchg(flags, pid, sem="release")


@triton.jit
def _complex_tbsv_diag_block_inv(
    A,
    start,
    n,
    k,
    lda,
    offs,
    row_mask,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = start + offs
    if TRANS == 0:
        a_off = 2 * _tbsv_band_offset(rows[:, None], rows[None, :], k, lda, UPLO)
    else:
        a_off = 2 * _tbsv_band_offset(rows[None, :], rows[:, None], k, lda, UPLO)
    distance = tl.abs(rows[:, None] - rows[None, :])
    tile_mask = row_mask[:, None] & row_mask[None, :] & (distance <= k)
    matrix_r = tl.load(A + a_off, mask=tile_mask, other=0.0)
    matrix_i = tl.load(A + a_off + 1, mask=tile_mask, other=0.0)
    if CONJ:
        matrix_i = -matrix_i
    if IS_DOUBLE:
        one = tl.full((BLOCK_N,), 1.0, tl.float64)
        zero = tl.full((BLOCK_N,), 0.0, tl.float64)
        inverse_r = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
        )
        inverse_i = tl.zeros((BLOCK_N, BLOCK_N), tl.float64)
    else:
        one = tl.full((BLOCK_N,), 1.0, tl.float32)
        zero = tl.full((BLOCK_N,), 0.0, tl.float32)
        inverse_r = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
        )
        inverse_i = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    if not UNIT:
        diagonal_r = tl.sum(
            tl.where(offs[:, None] == offs[None, :], matrix_r, zero), axis=1
        )
        diagonal_i = tl.sum(
            tl.where(offs[:, None] == offs[None, :], matrix_i, zero), axis=1
        )
        diagonal_r = tl.where(row_mask, diagonal_r, one)
        diagonal_i = tl.where(row_mask, diagonal_i, zero)
        denominator = diagonal_r * diagonal_r + diagonal_i * diagonal_i
    if LOWER_EFF:
        for index in tl.range(0, BLOCK_N):
            matrix_row_r = tl.sum(
                tl.where(offs[:, None] == index, matrix_r, zero), axis=0
            )
            matrix_row_i = tl.sum(
                tl.where(offs[:, None] == index, matrix_i, zero), axis=0
            )
            strict = offs < index
            matrix_row_r = tl.where(strict, matrix_row_r, zero)
            matrix_row_i = tl.where(strict, matrix_row_i, zero)
            contribution_r = tl.sum(
                matrix_row_r[:, None] * inverse_r - matrix_row_i[:, None] * inverse_i,
                axis=0,
            )
            contribution_i = tl.sum(
                matrix_row_r[:, None] * inverse_i + matrix_row_i[:, None] * inverse_r,
                axis=0,
            )
            identity_row = tl.where(offs == index, one, zero)
            numerator_r = identity_row - contribution_r
            numerator_i = -contribution_i
            if UNIT:
                inverse_row_r = numerator_r
                inverse_row_i = numerator_i
            else:
                diag_r = tl.sum(tl.where(offs == index, diagonal_r, zero), axis=0)
                diag_i = tl.sum(tl.where(offs == index, diagonal_i, zero), axis=0)
                denom = tl.sum(tl.where(offs == index, denominator, zero), axis=0)
                inverse_row_r = (numerator_r * diag_r + numerator_i * diag_i) / denom
                inverse_row_i = (numerator_i * diag_r - numerator_r * diag_i) / denom
            inverse_r = tl.where(
                offs[:, None] == index, inverse_row_r[None, :], inverse_r
            )
            inverse_i = tl.where(
                offs[:, None] == index, inverse_row_i[None, :], inverse_i
            )
    else:
        for reverse_index in tl.range(0, BLOCK_N):
            index = BLOCK_N - 1 - reverse_index
            matrix_row_r = tl.sum(
                tl.where(offs[:, None] == index, matrix_r, zero), axis=0
            )
            matrix_row_i = tl.sum(
                tl.where(offs[:, None] == index, matrix_i, zero), axis=0
            )
            strict = offs > index
            matrix_row_r = tl.where(strict, matrix_row_r, zero)
            matrix_row_i = tl.where(strict, matrix_row_i, zero)
            contribution_r = tl.sum(
                matrix_row_r[:, None] * inverse_r - matrix_row_i[:, None] * inverse_i,
                axis=0,
            )
            contribution_i = tl.sum(
                matrix_row_r[:, None] * inverse_i + matrix_row_i[:, None] * inverse_r,
                axis=0,
            )
            identity_row = tl.where(offs == index, one, zero)
            numerator_r = identity_row - contribution_r
            numerator_i = -contribution_i
            if UNIT:
                inverse_row_r = numerator_r
                inverse_row_i = numerator_i
            else:
                diag_r = tl.sum(tl.where(offs == index, diagonal_r, zero), axis=0)
                diag_i = tl.sum(tl.where(offs == index, diagonal_i, zero), axis=0)
                denom = tl.sum(tl.where(offs == index, denominator, zero), axis=0)
                inverse_row_r = (numerator_r * diag_r + numerator_i * diag_i) / denom
                inverse_row_i = (numerator_i * diag_r - numerator_r * diag_i) / denom
            inverse_r = tl.where(
                offs[:, None] == index, inverse_row_r[None, :], inverse_r
            )
            inverse_i = tl.where(
                offs[:, None] == index, inverse_row_i[None, :], inverse_i
            )
    return inverse_r, inverse_i


@libentry()
@triton.jit
def _complex_tbsv_blocked_kernel(
    A,
    x,
    flags,
    n,
    k,
    lda,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    FORWARD: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BAND_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    band_offs = tl.arange(0, BAND_K)
    panel_count = tl.cdiv(n, BLOCK_N)
    if FORWARD:
        block = pid
    else:
        block = panel_count - 1 - pid
    row_start = block * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    rows = row_start + offs
    row_mask = rows < row_end
    inverse_r, inverse_i = _complex_tbsv_diag_block_inv(
        A,
        row_start,
        n,
        k,
        lda,
        offs,
        row_mask,
        UPLO,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        IS_DOUBLE,
        BLOCK_N,
    )
    rhs_r = tl.load(
        x + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    rhs_i = tl.load(
        x + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    while tl.atomic_add(flags, 0, sem="acquire") < pid - 1:
        pass
    if FORWARD:
        col_start = tl.maximum(0, row_start - k)
        col_end = row_start
    else:
        col_start = row_end
        col_end = tl.minimum(n, row_end + k)
    cols = col_start + band_offs
    col_mask = cols < col_end
    solved_r = tl.load(
        x + cols * 2,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    solved_i = tl.load(
        x + cols * 2 + 1,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    if TRANS == 0:
        a_off = 2 * _tbsv_band_offset(rows[:, None], cols[None, :], k, lda, UPLO)
    else:
        a_off = 2 * _tbsv_band_offset(cols[None, :], rows[:, None], k, lda, UPLO)
    distance = tl.abs(rows[:, None] - cols[None, :])
    matrix_mask = row_mask[:, None] & col_mask[None, :] & (distance <= k)
    matrix_r = tl.load(
        A + a_off,
        mask=matrix_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    matrix_i = tl.load(
        A + a_off + 1,
        mask=matrix_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    if CONJ:
        matrix_i = -matrix_i
    rhs_r -= tl.sum(matrix_r * solved_r[None, :] - matrix_i * solved_i[None, :], axis=1)
    rhs_i -= tl.sum(matrix_r * solved_i[None, :] + matrix_i * solved_r[None, :], axis=1)

    result_r = tl.sum(inverse_r * rhs_r[None, :] - inverse_i * rhs_i[None, :], axis=1)
    result_i = tl.sum(inverse_r * rhs_i[None, :] + inverse_i * rhs_r[None, :], axis=1)
    tl.store(x + rows * 2, result_r, mask=row_mask)
    tl.store(x + rows * 2 + 1, result_i, mask=row_mask)
    tl.atomic_xchg(flags, pid, sem="release")


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------
@libentry()
@triton.jit
def _dtbsv_k1_scan_kernel(
    A,
    x,
    flags,
    n,
    lda,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    solve_index = pid * BLOCK_N + offs
    valid = solve_index < n
    if FORWARD:
        rows = solve_index
        predecessor = rows - 1
    else:
        rows = n - 1 - solve_index
        predecessor = rows + 1
    has_predecessor = valid & (solve_index != 0)

    diagonal_offset = _tbsv_band_offset(rows, rows, 1, lda, UPLO)
    if TRANS == 0:
        band_offset = _tbsv_band_offset(rows, predecessor, 1, lda, UPLO)
    else:
        band_offset = _tbsv_band_offset(predecessor, rows, 1, lda, UPLO)

    rhs = tl.load(x + rows, mask=valid, other=0.0)
    if UNIT:
        alpha = -tl.load(A + band_offset, mask=has_predecessor, other=0.0)
        beta = rhs
    else:
        diagonal = tl.load(A + diagonal_offset, mask=valid, other=1.0)
        off_diagonal = tl.load(A + band_offset, mask=has_predecessor, other=0.0)
        alpha = -off_diagonal / diagonal
        beta = rhs / diagonal

    first_row = solve_index == 0
    alpha = tl.where(first_row, 0.0, alpha)
    alpha = tl.where(valid, alpha, 1.0)
    beta = tl.where(valid, beta, 0.0)
    prefix_a, prefix_b = tl.associative_scan(
        (alpha, beta), axis=0, combine_fn=_real_affine_combine
    )

    while tl.atomic_add(flags, 0, sem="acquire") < pid - 1:
        pass
    previous_index = pid * BLOCK_N - 1
    if FORWARD:
        previous_row = previous_index
    else:
        previous_row = n - 1 - previous_index
    carry = tl.load(x + previous_row, mask=pid > 0, other=0.0)

    result = prefix_a * carry + prefix_b
    tl.store(x + rows, result, mask=valid)
    tl.atomic_xchg(flags, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
)
@triton.jit
def stbsv_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)

    # Lower / NoTrans : forward substitution
    if (UPLO == 0) and (TRANS == 0):
        for j in tl.range(0, n):
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j + d
                m = (d <= k) & (i < n)
                a_off = d + j * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                xv = xv - av * xj
                tl.store(x_ptr + i * INCX, xv, mask=m)

    # Upper / NoTrans : back substitution
    elif (UPLO == 1) and (TRANS == 0):
        for jc in tl.range(0, n):
            j = n - 1 - jc
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j + d
                m = (d <= k) & (i < n)
                a_off = (k - d) + i * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                acc += av * xv
            s = tl.sum(acc, axis=0)
            xj = tl.load(x_ptr + j * INCX) - s
            if not UNIT:
                ajj = tl.load(a_ptr + k + j * LDA)
                xj = xj / ajj
            tl.store(x_ptr + j * INCX, xj)

    # Upper / Trans : forward substitution
    elif (UPLO == 1) and (TRANS == 1):
        for j in tl.range(0, n):
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j - d
                m = (d <= k) & (i >= 0)
                a_off = (k - d) + j * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                acc += av * xv
            s = tl.sum(acc, axis=0)
            xj = tl.load(x_ptr + j * INCX) - s
            if not UNIT:
                ajj = tl.load(a_ptr + k + j * LDA)
                xj = xj / ajj
            tl.store(x_ptr + j * INCX, xj)

    # Lower / Trans : back substitution
    else:
        for jc in tl.range(0, n):
            j = n - 1 - jc
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for kb in tl.range(0, k, BLOCK_K):
                d = kb + 1 + offs
                i = j + d
                m = (d <= k) & (i < n)
                a_off = d + j * LDA
                av = tl.load(a_ptr + a_off, mask=m, other=0.0)
                xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
                acc += av * xv
            s = tl.sum(acc, axis=0)
            xj = tl.load(x_ptr + j * INCX) - s
            if not UNIT:
                ajj = tl.load(a_ptr + j * LDA)
                xj = xj / ajj
            tl.store(x_ptr + j * INCX, xj)


@libentry()
@triton.jit
def _real_tbsv_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
):
    if (UPLO == 0) and (TRANS == 0):
        j = 0
        while j < n:
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                xj = xj / tl.load(a_ptr + j * LDA)
                tl.store(x_ptr + j * INCX, xj)
            d = 1
            while d <= k:
                if j + d < n:
                    i = j + d
                    xv = tl.load(x_ptr + i * INCX) - tl.load(a_ptr + d + j * LDA) * xj
                    tl.store(x_ptr + i * INCX, xv)
                d += 1
            j += 1
    elif (UPLO == 1) and (TRANS == 0):
        j = n - 1
        while j >= 0:
            acc = tl.load(x_ptr + j * INCX)
            d = 1
            while d <= k:
                if j + d < n:
                    i = j + d
                    acc -= tl.load(a_ptr + (k - d) + i * LDA) * tl.load(
                        x_ptr + i * INCX
                    )
                d += 1
            if not UNIT:
                acc = acc / tl.load(a_ptr + k + j * LDA)
            tl.store(x_ptr + j * INCX, acc)
            j -= 1
    elif (UPLO == 1) and (TRANS != 0):
        j = 0
        while j < n:
            acc = tl.load(x_ptr + j * INCX)
            d = 1
            while d <= k:
                if j - d >= 0:
                    i = j - d
                    acc -= tl.load(a_ptr + (k - d) + j * LDA) * tl.load(
                        x_ptr + i * INCX
                    )
                d += 1
            if not UNIT:
                acc = acc / tl.load(a_ptr + k + j * LDA)
            tl.store(x_ptr + j * INCX, acc)
            j += 1
    else:
        j = n - 1
        while j >= 0:
            acc = tl.load(x_ptr + j * INCX)
            d = 1
            while d <= k:
                if j + d < n:
                    i = j + d
                    acc -= tl.load(a_ptr + d + j * LDA) * tl.load(x_ptr + i * INCX)
                d += 1
            if not UNIT:
                acc = acc / tl.load(a_ptr + j * LDA)
            tl.store(x_ptr + j * INCX, acc)
            j -= 1


@libentry()
@triton.jit
def _ztbsv_k1_scan_kernel(
    A,
    x,
    flags,
    n,
    lda,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    FORWARD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    solve_index = pid * BLOCK_N + offs
    valid = solve_index < n
    if FORWARD:
        rows = solve_index
        predecessor = rows - 1
    else:
        rows = n - 1 - solve_index
        predecessor = rows + 1
    has_predecessor = valid & (solve_index != 0)

    diagonal_offset = 2 * _tbsv_band_offset(rows, rows, 1, lda, UPLO)
    if TRANS == 0:
        band_offset = 2 * _tbsv_band_offset(rows, predecessor, 1, lda, UPLO)
    else:
        band_offset = 2 * _tbsv_band_offset(predecessor, rows, 1, lda, UPLO)

    rhs_r = tl.load(x + 2 * rows, mask=valid, other=0.0)
    rhs_i = tl.load(x + 2 * rows + 1, mask=valid, other=0.0)
    off_r = tl.load(A + band_offset, mask=has_predecessor, other=0.0)
    off_i = tl.load(A + band_offset + 1, mask=has_predecessor, other=0.0)
    if CONJ:
        off_i = -off_i

    if UNIT:
        alpha_r = -off_r
        alpha_i = -off_i
        beta_r = rhs_r
        beta_i = rhs_i
    else:
        diag_r = tl.load(A + diagonal_offset, mask=valid, other=1.0)
        diag_i = tl.load(A + diagonal_offset + 1, mask=valid, other=0.0)
        if CONJ:
            diag_i = -diag_i
        denominator = diag_r * diag_r + diag_i * diag_i
        alpha_r = -(off_r * diag_r + off_i * diag_i) / denominator
        alpha_i = -(off_i * diag_r - off_r * diag_i) / denominator
        beta_r = (rhs_r * diag_r + rhs_i * diag_i) / denominator
        beta_i = (rhs_i * diag_r - rhs_r * diag_i) / denominator

    first_row = solve_index == 0
    alpha_r = tl.where(first_row, 0.0, alpha_r)
    alpha_i = tl.where(first_row, 0.0, alpha_i)
    alpha_r = tl.where(valid, alpha_r, 1.0)
    alpha_i = tl.where(valid, alpha_i, 0.0)
    beta_r = tl.where(valid, beta_r, 0.0)
    beta_i = tl.where(valid, beta_i, 0.0)
    prefix_ar, prefix_ai, prefix_br, prefix_bi = tl.associative_scan(
        (alpha_r, alpha_i, beta_r, beta_i),
        axis=0,
        combine_fn=_complex_affine_combine,
    )

    while tl.atomic_add(flags, 0, sem="acquire") < pid - 1:
        pass
    previous_index = pid * BLOCK_N - 1
    if FORWARD:
        previous_row = previous_index
    else:
        previous_row = n - 1 - previous_index
    carry_r = tl.load(x + 2 * previous_row, mask=pid > 0, other=0.0)
    carry_i = tl.load(x + 2 * previous_row + 1, mask=pid > 0, other=0.0)

    correction_r, correction_i = _complex_mul(prefix_ar, prefix_ai, carry_r, carry_i)
    tl.store(x + 2 * rows, correction_r + prefix_br, mask=valid)
    tl.store(x + 2 * rows + 1, correction_i + prefix_bi, mask=valid)
    tl.atomic_xchg(flags, pid, sem="release")


@libentry()
@triton.jit
def _complex_tbsv_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
):
    if (UPLO == 0) and (TRANS == 0):
        j = 0
        while j < n:
            xr = tl.load(x_ptr + 2 * j * INCX)
            xi = tl.load(x_ptr + 2 * j * INCX + 1)
            if not UNIT:
                a_off = 2 * j * LDA
                ar = tl.load(a_ptr + a_off)
                ai = tl.load(a_ptr + a_off + 1)
                if CONJ:
                    ai = -ai
                den = ar * ar + ai * ai
                nr = xr * ar + xi * ai
                ni = xi * ar - xr * ai
                xr = nr / den
                xi = ni / den
                tl.store(x_ptr + 2 * j * INCX, xr)
                tl.store(x_ptr + 2 * j * INCX + 1, xi)
            d = 1
            while d <= k:
                if j + d < n:
                    i = j + d
                    a_off = 2 * (d + j * LDA)
                    ar = tl.load(a_ptr + a_off)
                    ai = tl.load(a_ptr + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x_ptr + 2 * i * INCX)
                    bi = tl.load(x_ptr + 2 * i * INCX + 1)
                    br -= ar * xr - ai * xi
                    bi -= ar * xi + ai * xr
                    tl.store(x_ptr + 2 * i * INCX, br)
                    tl.store(x_ptr + 2 * i * INCX + 1, bi)
                d += 1
            j += 1
    elif (UPLO == 1) and (TRANS == 0):
        j = n - 1
        while j >= 0:
            xr = tl.load(x_ptr + 2 * j * INCX)
            xi = tl.load(x_ptr + 2 * j * INCX + 1)
            d = 1
            while d <= k:
                if j + d < n:
                    i = j + d
                    a_off = 2 * ((k - d) + i * LDA)
                    ar = tl.load(a_ptr + a_off)
                    ai = tl.load(a_ptr + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x_ptr + 2 * i * INCX)
                    bi = tl.load(x_ptr + 2 * i * INCX + 1)
                    xr -= ar * br - ai * bi
                    xi -= ar * bi + ai * br
                d += 1
            if not UNIT:
                a_off = 2 * (k + j * LDA)
                ar = tl.load(a_ptr + a_off)
                ai = tl.load(a_ptr + a_off + 1)
                if CONJ:
                    ai = -ai
                den = ar * ar + ai * ai
                nr = xr * ar + xi * ai
                ni = xi * ar - xr * ai
                xr = nr / den
                xi = ni / den
            tl.store(x_ptr + 2 * j * INCX, xr)
            tl.store(x_ptr + 2 * j * INCX + 1, xi)
            j -= 1
    elif (UPLO == 1) and (TRANS != 0):
        j = 0
        while j < n:
            xr = tl.load(x_ptr + 2 * j * INCX)
            xi = tl.load(x_ptr + 2 * j * INCX + 1)
            d = 1
            while d <= k:
                if j - d >= 0:
                    i = j - d
                    a_off = 2 * ((k - d) + j * LDA)
                    ar = tl.load(a_ptr + a_off)
                    ai = tl.load(a_ptr + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x_ptr + 2 * i * INCX)
                    bi = tl.load(x_ptr + 2 * i * INCX + 1)
                    xr -= ar * br - ai * bi
                    xi -= ar * bi + ai * br
                d += 1
            if not UNIT:
                a_off = 2 * (k + j * LDA)
                ar = tl.load(a_ptr + a_off)
                ai = tl.load(a_ptr + a_off + 1)
                if CONJ:
                    ai = -ai
                den = ar * ar + ai * ai
                nr = xr * ar + xi * ai
                ni = xi * ar - xr * ai
                xr = nr / den
                xi = ni / den
            tl.store(x_ptr + 2 * j * INCX, xr)
            tl.store(x_ptr + 2 * j * INCX + 1, xi)
            j += 1
    else:
        j = n - 1
        while j >= 0:
            xr = tl.load(x_ptr + 2 * j * INCX)
            xi = tl.load(x_ptr + 2 * j * INCX + 1)
            d = 1
            while d <= k:
                if j + d < n:
                    i = j + d
                    a_off = 2 * (d + j * LDA)
                    ar = tl.load(a_ptr + a_off)
                    ai = tl.load(a_ptr + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x_ptr + 2 * i * INCX)
                    bi = tl.load(x_ptr + 2 * i * INCX + 1)
                    xr -= ar * br - ai * bi
                    xi -= ar * bi + ai * br
                d += 1
            if not UNIT:
                a_off = 2 * j * LDA
                ar = tl.load(a_ptr + a_off)
                ai = tl.load(a_ptr + a_off + 1)
                if CONJ:
                    ai = -ai
                den = ar * ar + ai * ai
                nr = xr * ar + xi * ai
                ni = xi * ar - xr * ai
                xr = nr / den
                xi = ni / den
            tl.store(x_ptr + 2 * j * INCX, xr)
            tl.store(x_ptr + 2 * j * INCX + 1, xi)
            j -= 1


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------
def _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok):
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
    assert n >= 0 and k >= 0
    assert lda >= k + 1
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= n * lda


def _row_major_tbsv_args(uplo, trans):
    physical_uplo = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    physical_trans = CUBLAS_OP_T if trans == CUBLAS_OP_N else CUBLAS_OP_N
    return physical_uplo, physical_trans, int(trans == CUBLAS_OP_C)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def stbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a real single-precision triangular banded system in-place."""
    assert A.dtype == torch.float32 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    uplo, trans, _ = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        if incx == 1 and trans_flag == 1 and k >= 64:
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
            panel_count = triton.cdiv(n, 32)
            _real_tbsv_blocked_kernel[(panel_count,)](
                A,
                x,
                _tbsv_flags(A.device),
                n,
                k,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                LOWER_EFF=lower_eff,
                FORWARD=lower_eff,
                IS_DOUBLE=False,
                BLOCK_N=32,
                BAND_K=_band_bucket(k),
                num_warps=4,
            )
            return
        grid = (1,)
        stbsv_kernel[grid](
            A,
            x,
            n,
            k,
            lda,
            incx,
            _band_bucket(k + 1),
            _mode_key(uplo, trans_flag, unit),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
        )


def dtbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a real double-precision triangular banded system in-place."""
    assert A.dtype == torch.float64 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    uplo, trans, _ = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        if incx == 1 and k == 1:
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
            block_n = 256
            panel_count = triton.cdiv(n, block_n)
            _dtbsv_k1_scan_kernel[(panel_count,)](
                A,
                x,
                _tbsv_flags(A.device),
                n,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=lower_eff,
                BLOCK_N=block_n,
                num_warps=8,
            )
            return
        use_blocked = incx == 1 and (
            k >= 16
            or (k >= 4 and trans_flag == 1)
            or (k == 4 and trans_flag == 0 and n >= 8192)
        )
        if use_blocked:
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
            panel_count = triton.cdiv(n, 32)
            _real_tbsv_blocked_kernel[(panel_count,)](
                A,
                x,
                _tbsv_flags(A.device),
                n,
                k,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                LOWER_EFF=lower_eff,
                FORWARD=lower_eff,
                IS_DOUBLE=True,
                BLOCK_N=32,
                BAND_K=_band_bucket(k),
                num_warps=4,
            )
            return
        _real_tbsv_kernel[(1,)](
            A,
            x,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
        )


def _complex_tbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    dtype: torch.dtype,
) -> None:
    assert A.dtype == dtype == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return
    uplo, trans, conj = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)

    with torch_device_fn.device(A.device):
        if dtype == torch.complex128 and incx == 1 and k == 1:
            trans_flag = int(trans != CUBLAS_OP_N)
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
            block_n = 64
            panel_count = triton.cdiv(n, block_n)
            _ztbsv_k1_scan_kernel[(panel_count,)](
                A_real,
                x_real,
                _tbsv_flags(A.device),
                n,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
                FORWARD=lower_eff,
                BLOCK_N=block_n,
                num_warps=4,
            )
            return
        use_blocked = incx == 1 and (
            (dtype == torch.complex64 and k >= 16)
            or (dtype == torch.complex128 and k >= 4)
        )
        if use_blocked:
            trans_flag = int(trans != CUBLAS_OP_N)
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
            block_n = 32 if dtype == torch.complex64 else 16
            panel_count = triton.cdiv(n, block_n)
            _complex_tbsv_blocked_kernel[(panel_count,)](
                A_real,
                x_real,
                _tbsv_flags(A.device),
                n,
                k,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
                LOWER_EFF=lower_eff,
                FORWARD=lower_eff,
                IS_DOUBLE=dtype == torch.complex128,
                BLOCK_N=block_n,
                BAND_K=_band_bucket(k),
                num_warps=4,
            )
            return
        _complex_tbsv_kernel[(1,)](
            A_real,
            x_real,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans,
            UNIT=unit,
            CONJ=conj,
        )


def ctbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a complex single-precision triangular banded system in-place."""
    _complex_tbsv(uplo, trans, diag, n, k, A, lda, x, incx, torch.complex64)


def ztbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a complex double-precision triangular banded system in-place."""
    _complex_tbsv(uplo, trans, diag, n, k, A, lda, x, incx, torch.complex128)

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2
CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_DIAG_UNIT = 1

_TPSV_FLAG_POOL = {}
_TPSV_FLAG_SLOTS = 256


def _tpsv_flags(device):
    try:
        if torch_device_fn.is_current_stream_capturing():
            return torch.full((1,), -1, dtype=torch.int32, device=device)
    except AttributeError:
        pass
    key = (device, torch_device_fn.current_stream(device).cuda_stream)
    entry = _TPSV_FLAG_POOL.get(key)
    if entry is None:
        pool = torch.full((_TPSV_FLAG_SLOTS * 4,), -1, dtype=torch.int32, device=device)
        views = [pool[index * 4 : index * 4 + 1] for index in range(_TPSV_FLAG_SLOTS)]
        entry = [pool, views, 0]
        _TPSV_FLAG_POOL[key] = entry
    pool, views, index = entry
    if index >= _TPSV_FLAG_SLOTS:
        pool.fill_(-1)
        index = 0
    entry[2] = index + 1
    return views[index]


@triton.jit
def _tpsv_packed_offset(row, col, n, UPLO: tl.constexpr):
    if UPLO == 1:
        return col * (col + 1) // 2 + row
    return col * n - (col * (col - 1)) // 2 + (row - col)


@libentry()
@triton.jit
def _real_tpsv_n64_kernel(
    AP,
    x,
    n,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    sx = tl.load(x + offs, mask=mask, other=0.0, eviction_policy="evict_last")

    if LOWER_EFF:
        for step in tl.static_range(0, BLOCK_N):
            active = step < n
            col_mask = (offs < step) & mask
            if TRANS == 0:
                a_off = _tpsv_packed_offset(step, offs, n, UPLO)
            else:
                a_off = _tpsv_packed_offset(offs, step, n, UPLO)
            a_vals = tl.load(
                AP + a_off,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == step, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag_off = _tpsv_packed_offset(step, step, n, UPLO)
                diag_val = tl.load(
                    AP + diag_off,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag_val
            sx = tl.where(offs == step, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            row = BLOCK_N - 1 - step
            active = row < n
            col_mask = (offs > row) & mask
            if TRANS == 0:
                a_off = _tpsv_packed_offset(row, offs, n, UPLO)
            else:
                a_off = _tpsv_packed_offset(offs, row, n, UPLO)
            a_vals = tl.load(
                AP + a_off,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == row, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag_off = _tpsv_packed_offset(row, row, n, UPLO)
                diag_val = tl.load(
                    AP + diag_off,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag_val
            sx = tl.where(offs == row, cur, sx)

    tl.store(x + offs, sx, mask=mask)


@triton.jit
def _real_tpsv_diag_block_inv(
    AP,
    start,
    n,
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
        a_off = _tpsv_packed_offset(rows[:, None], rows[None, :], n, UPLO)
    else:
        a_off = _tpsv_packed_offset(rows[None, :], rows[:, None], n, UPLO)
    tile_mask = row_mask[:, None] & row_mask[None, :]
    matrix = tl.load(AP + a_off, mask=tile_mask, other=0.0)
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
def _real_tpsv_blocked_kernel(
    AP,
    x,
    flags,
    n,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    FORWARD: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    chunk_size: tl.constexpr = BLOCK_N * CHUNK
    chunk_offs = tl.arange(0, chunk_size)
    panel_count = tl.cdiv(n, BLOCK_N)
    if FORWARD:
        block = pid
    else:
        block = panel_count - 1 - pid
    row_start = block * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    rows = row_start + offs
    row_mask = rows < row_end
    inverse = _real_tpsv_diag_block_inv(
        AP,
        row_start,
        n,
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

    chunk_count = tl.cdiv(pid, CHUNK)
    for group in tl.range(0, chunk_count):
        pid_lo = group * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flags, 0, sem="acquire") < pid_hi - 1:
            pass
        if FORWARD:
            col_start = pid_lo * BLOCK_N
            col_end = pid_hi * BLOCK_N
        else:
            col_start = (panel_count - pid_hi) * BLOCK_N
            col_end = (panel_count - pid_lo) * BLOCK_N
        cols = col_start + chunk_offs
        col_mask = (cols < col_end) & (cols < n)
        solved = tl.load(
            x + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = _tpsv_packed_offset(rows[:, None], cols[None, :], n, UPLO)
        else:
            a_off = _tpsv_packed_offset(cols[None, :], rows[:, None], n, UPLO)
        matrix_mask = row_mask[:, None] & col_mask[None, :]
        matrix = tl.load(
            AP + a_off,
            mask=matrix_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        rhs -= tl.sum(matrix * solved[None, :], axis=1)

    result = tl.sum(inverse * rhs[None, :], axis=1)
    tl.store(x + rows, result, mask=row_mask)
    tl.atomic_xchg(flags, pid, sem="release")


@libentry()
@triton.jit
def _real_tpsv_panel_kernel(
    AP,
    x,
    n,
    start,
    end,
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
            row = start + step
            cols = start + offs
        else:
            row = end - 1 - step
            cols = end - 1 - offs
        active = step < size
        col_mask = (offs < step) & (offs < size) & active
        if TRANS == 0:
            a_off = _tpsv_packed_offset(row, cols, n, UPLO)
        else:
            a_off = _tpsv_packed_offset(cols, row, n, UPLO)
        matrix = tl.load(AP + a_off, mask=col_mask, other=0.0)
        solved = tl.load(x + cols, mask=col_mask, other=0.0)
        value = tl.load(x + row, mask=active, other=0.0) - tl.sum(
            matrix * solved, axis=0
        )
        if not UNIT:
            diag_off = _tpsv_packed_offset(row, row, n, UPLO)
            diagonal = tl.load(AP + diag_off, mask=active, other=1.0)
            value = value / diagonal
        tl.store(x + row, value, mask=active)


@libentry()
@triton.jit
def _real_tpsv_update_kernel(
    AP,
    x,
    n,
    row_base,
    row_count,
    start,
    end,
    UPLO: tl.constexpr,
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
        a_off = _tpsv_packed_offset(rows[:, None], cols[None, :], n, UPLO)
    else:
        a_off = _tpsv_packed_offset(cols[None, :], rows[:, None], n, UPLO)
    matrix_mask = row_mask[:, None] & col_mask[None, :]
    matrix = tl.load(AP + a_off, mask=matrix_mask, other=0.0)
    solved = tl.load(x + cols, mask=col_mask, other=0.0)
    update = tl.sum(matrix * solved[None, :], axis=1)
    rhs = tl.load(x + rows, mask=row_mask, other=0.0)
    tl.store(x + rows, rhs - update, mask=row_mask)


def _real_tpsv_panel_ranges(n, block_n, forward):
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


def _real_tpsv_host_panels(uplo, trans, diag, n, AP, x):
    block_n = 32
    block_m = 128
    forward = bool((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans != CUBLAS_OP_N))
    for start, end in _real_tpsv_panel_ranges(n, block_n, forward):
        _real_tpsv_panel_kernel[(1,)](
            AP,
            x,
            n,
            start,
            end,
            UPLO=uplo,
            TRANS=trans,
            UNIT=int(diag == CUBLAS_DIAG_UNIT),
            FORWARD=forward,
            BLOCK_N=block_n,
            num_warps=1,
        )
        if forward:
            row_base = end
            row_count = n - end
        else:
            row_base = 0
            row_count = start
        if row_count > 0:
            _real_tpsv_update_kernel[(triton.cdiv(row_count, block_m),)](
                AP,
                x,
                n,
                row_base,
                row_count,
                start,
                end,
                UPLO=uplo,
                TRANS=trans,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
            )


@libentry()
@triton.jit
def _real_tpsv_kernel(
    uplo: tl.constexpr,
    trans: tl.constexpr,
    diag: tl.constexpr,
    n,
    AP,
    x,
    incx: tl.constexpr,
):
    if trans == 0:
        if uplo == 1:
            jj = n - 1
            while jj >= 0:
                acc = tl.load(x + jj * incx)
                kk = jj + 1
                while kk < n:
                    a_off = _tpsv_packed_offset(jj, kk, n, uplo)
                    acc -= tl.load(AP + a_off) * tl.load(x + kk * incx)
                    kk += 1
                if diag == 0:
                    a_diag = tl.load(AP + _tpsv_packed_offset(jj, jj, n, uplo))
                    acc = acc / a_diag
                tl.store(x + jj * incx, acc)
                jj -= 1
        else:
            jj = 0
            while jj < n:
                acc = tl.load(x + jj * incx)
                kk = 0
                while kk < jj:
                    a_off = _tpsv_packed_offset(jj, kk, n, uplo)
                    acc -= tl.load(AP + a_off) * tl.load(x + kk * incx)
                    kk += 1
                if diag == 0:
                    a_diag = tl.load(AP + _tpsv_packed_offset(jj, jj, n, uplo))
                    acc = acc / a_diag
                tl.store(x + jj * incx, acc)
                jj += 1
    else:
        if uplo == 1:
            jj = 0
            while jj < n:
                acc = tl.load(x + jj * incx)
                kk = 0
                while kk < jj:
                    a_off = _tpsv_packed_offset(kk, jj, n, uplo)
                    acc -= tl.load(AP + a_off) * tl.load(x + kk * incx)
                    kk += 1
                if diag == 0:
                    a_diag = tl.load(AP + _tpsv_packed_offset(jj, jj, n, uplo))
                    acc = acc / a_diag
                tl.store(x + jj * incx, acc)
                jj += 1
        else:
            jj = n - 1
            while jj >= 0:
                acc = tl.load(x + jj * incx)
                kk = jj + 1
                while kk < n:
                    a_off = _tpsv_packed_offset(kk, jj, n, uplo)
                    acc -= tl.load(AP + a_off) * tl.load(x + kk * incx)
                    kk += 1
                if diag == 0:
                    a_diag = tl.load(AP + _tpsv_packed_offset(jj, jj, n, uplo))
                    acc = acc / a_diag
                tl.store(x + jj * incx, acc)
                jj -= 1


@triton.jit
def _complex_tpsv_diag_block_inv(
    AP,
    start,
    n,
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
        a_off = 2 * _tpsv_packed_offset(rows[:, None], rows[None, :], n, UPLO)
    else:
        a_off = 2 * _tpsv_packed_offset(rows[None, :], rows[:, None], n, UPLO)
    tile_mask = row_mask[:, None] & row_mask[None, :]
    matrix_r = tl.load(AP + a_off, mask=tile_mask, other=0.0)
    matrix_i = tl.load(AP + a_off + 1, mask=tile_mask, other=0.0)
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
def _complex_tpsv_blocked_kernel(
    AP,
    x,
    flags,
    n,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    FORWARD: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    chunk_size: tl.constexpr = BLOCK_N * CHUNK
    chunk_offs = tl.arange(0, chunk_size)
    panel_count = tl.cdiv(n, BLOCK_N)
    if FORWARD:
        block = pid
    else:
        block = panel_count - 1 - pid
    row_start = block * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    rows = row_start + offs
    row_mask = rows < row_end
    inverse_r, inverse_i = _complex_tpsv_diag_block_inv(
        AP,
        row_start,
        n,
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

    chunk_count = tl.cdiv(pid, CHUNK)
    for group in tl.range(0, chunk_count):
        pid_lo = group * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flags, 0, sem="acquire") < pid_hi - 1:
            pass
        if FORWARD:
            col_start = pid_lo * BLOCK_N
            col_end = pid_hi * BLOCK_N
        else:
            col_start = (panel_count - pid_hi) * BLOCK_N
            col_end = (panel_count - pid_lo) * BLOCK_N
        cols = col_start + chunk_offs
        col_mask = (cols < col_end) & (cols < n)
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
            a_off = 2 * _tpsv_packed_offset(rows[:, None], cols[None, :], n, UPLO)
        else:
            a_off = 2 * _tpsv_packed_offset(cols[None, :], rows[:, None], n, UPLO)
        matrix_mask = row_mask[:, None] & col_mask[None, :]
        matrix_r = tl.load(
            AP + a_off,
            mask=matrix_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        matrix_i = tl.load(
            AP + a_off + 1,
            mask=matrix_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        if CONJ:
            matrix_i = -matrix_i
        rhs_r -= tl.sum(
            matrix_r * solved_r[None, :] - matrix_i * solved_i[None, :], axis=1
        )
        rhs_i -= tl.sum(
            matrix_r * solved_i[None, :] + matrix_i * solved_r[None, :], axis=1
        )

    result_r = tl.sum(inverse_r * rhs_r[None, :] - inverse_i * rhs_i[None, :], axis=1)
    result_i = tl.sum(inverse_r * rhs_i[None, :] + inverse_i * rhs_r[None, :], axis=1)
    tl.store(x + rows * 2, result_r, mask=row_mask)
    tl.store(x + rows * 2 + 1, result_i, mask=row_mask)
    tl.atomic_xchg(flags, pid, sem="release")


@libentry()
@triton.jit
def _complex_tpsv_kernel(
    uplo: tl.constexpr,
    trans: tl.constexpr,
    diag: tl.constexpr,
    n,
    AP,
    x,
    incx: tl.constexpr,
    CONJ: tl.constexpr,
):
    if trans == 0:
        if uplo == 1:
            jj = n - 1
            while jj >= 0:
                xr = tl.load(x + 2 * jj * incx)
                xi = tl.load(x + 2 * jj * incx + 1)
                kk = jj + 1
                while kk < n:
                    a_off = 2 * _tpsv_packed_offset(jj, kk, n, uplo)
                    ar = tl.load(AP + a_off)
                    ai = tl.load(AP + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x + 2 * kk * incx)
                    bi = tl.load(x + 2 * kk * incx + 1)
                    xr -= ar * br - ai * bi
                    xi -= ar * bi + ai * br
                    kk += 1
                if diag == 0:
                    a_diag = 2 * _tpsv_packed_offset(jj, jj, n, uplo)
                    ar = tl.load(AP + a_diag)
                    ai = tl.load(AP + a_diag + 1)
                    if CONJ:
                        ai = -ai
                    den = ar * ar + ai * ai
                    nr = xr * ar + xi * ai
                    ni = xi * ar - xr * ai
                    xr = nr / den
                    xi = ni / den
                tl.store(x + 2 * jj * incx, xr)
                tl.store(x + 2 * jj * incx + 1, xi)
                jj -= 1
        else:
            jj = 0
            while jj < n:
                xr = tl.load(x + 2 * jj * incx)
                xi = tl.load(x + 2 * jj * incx + 1)
                kk = 0
                while kk < jj:
                    a_off = 2 * _tpsv_packed_offset(jj, kk, n, uplo)
                    ar = tl.load(AP + a_off)
                    ai = tl.load(AP + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x + 2 * kk * incx)
                    bi = tl.load(x + 2 * kk * incx + 1)
                    xr -= ar * br - ai * bi
                    xi -= ar * bi + ai * br
                    kk += 1
                if diag == 0:
                    a_diag = 2 * _tpsv_packed_offset(jj, jj, n, uplo)
                    ar = tl.load(AP + a_diag)
                    ai = tl.load(AP + a_diag + 1)
                    if CONJ:
                        ai = -ai
                    den = ar * ar + ai * ai
                    nr = xr * ar + xi * ai
                    ni = xi * ar - xr * ai
                    xr = nr / den
                    xi = ni / den
                tl.store(x + 2 * jj * incx, xr)
                tl.store(x + 2 * jj * incx + 1, xi)
                jj += 1
    else:
        if uplo == 1:
            jj = 0
            while jj < n:
                xr = tl.load(x + 2 * jj * incx)
                xi = tl.load(x + 2 * jj * incx + 1)
                kk = 0
                while kk < jj:
                    a_off = 2 * _tpsv_packed_offset(kk, jj, n, uplo)
                    ar = tl.load(AP + a_off)
                    ai = tl.load(AP + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x + 2 * kk * incx)
                    bi = tl.load(x + 2 * kk * incx + 1)
                    xr -= ar * br - ai * bi
                    xi -= ar * bi + ai * br
                    kk += 1
                if diag == 0:
                    a_diag = 2 * _tpsv_packed_offset(jj, jj, n, uplo)
                    ar = tl.load(AP + a_diag)
                    ai = tl.load(AP + a_diag + 1)
                    if CONJ:
                        ai = -ai
                    den = ar * ar + ai * ai
                    nr = xr * ar + xi * ai
                    ni = xi * ar - xr * ai
                    xr = nr / den
                    xi = ni / den
                tl.store(x + 2 * jj * incx, xr)
                tl.store(x + 2 * jj * incx + 1, xi)
                jj += 1
        else:
            jj = n - 1
            while jj >= 0:
                xr = tl.load(x + 2 * jj * incx)
                xi = tl.load(x + 2 * jj * incx + 1)
                kk = jj + 1
                while kk < n:
                    a_off = 2 * _tpsv_packed_offset(kk, jj, n, uplo)
                    ar = tl.load(AP + a_off)
                    ai = tl.load(AP + a_off + 1)
                    if CONJ:
                        ai = -ai
                    br = tl.load(x + 2 * kk * incx)
                    bi = tl.load(x + 2 * kk * incx + 1)
                    xr -= ar * br - ai * bi
                    xi -= ar * bi + ai * br
                    kk += 1
                if diag == 0:
                    a_diag = 2 * _tpsv_packed_offset(jj, jj, n, uplo)
                    ar = tl.load(AP + a_diag)
                    ai = tl.load(AP + a_diag + 1)
                    if CONJ:
                        ai = -ai
                    den = ar * ar + ai * ai
                    nr = xr * ar + xi * ai
                    ni = xi * ar - xr * ai
                    xr = nr / den
                    xi = ni / den
                tl.store(x + 2 * jj * incx, xr)
                tl.store(x + 2 * jj * incx + 1, xi)
                jj -= 1


def _check_common(uplo, trans, diag, n, AP, x, incx):
    assert AP.is_contiguous() and x.is_contiguous()
    assert AP.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C)
    assert diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    assert n >= 0
    assert incx > 0
    assert (AP.is_cuda and x.is_cuda) or (
        AP.device.type == "npu" and x.device.type == "npu"
    )
    assert AP.numel() >= n * (n + 1) // 2
    assert x.numel() >= 1 + (n - 1) * incx if n > 0 else True


def _row_major_tpsv_args(uplo, trans):
    physical_uplo = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    physical_trans = CUBLAS_OP_T if trans == CUBLAS_OP_N else CUBLAS_OP_N
    return physical_uplo, physical_trans, int(trans == CUBLAS_OP_C)


def _real_tpsv(uplo, trans, diag, n, AP, x, incx, dtype):
    _check_common(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype is dtype and x.dtype is dtype
    if n == 0:
        return x
    uplo, trans, _ = _row_major_tpsv_args(uplo, trans)
    with torch_device_fn.device(AP.device):
        if (
            incx == 1
            and dtype == torch.float64
            and n <= 64
            and not (uplo == CUBLAS_FILL_MODE_LOWER and trans == CUBLAS_OP_N)
        ):
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans != CUBLAS_OP_N))
            panel_count = triton.cdiv(n, 16)
            _real_tpsv_blocked_kernel[(panel_count,)](
                AP,
                x,
                _tpsv_flags(AP.device),
                n,
                UPLO=uplo,
                TRANS=trans,
                UNIT=int(diag == CUBLAS_DIAG_UNIT),
                LOWER_EFF=lower_eff,
                FORWARD=lower_eff,
                IS_DOUBLE=True,
                BLOCK_N=16,
                CHUNK=1,
                num_warps=4,
            )
        elif incx == 1 and n <= 64:
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans != CUBLAS_OP_N))
            _real_tpsv_n64_kernel[(1,)](
                AP,
                x,
                n,
                UPLO=uplo,
                TRANS=trans,
                UNIT=int(diag == CUBLAS_DIAG_UNIT),
                LOWER_EFF=lower_eff,
                BLOCK_N=64,
                num_warps=1,
            )
        elif (
            incx == 1
            and dtype == torch.float64
            and n <= 96
            and uplo == CUBLAS_FILL_MODE_LOWER
            and trans == CUBLAS_OP_N
        ):
            _real_tpsv_host_panels(uplo, trans, diag, n, AP, x)
        elif incx == 1:
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans != CUBLAS_OP_N))
            panel_count = triton.cdiv(n, 32)
            _real_tpsv_blocked_kernel[(panel_count,)](
                AP,
                x,
                _tpsv_flags(AP.device),
                n,
                UPLO=uplo,
                TRANS=trans,
                UNIT=int(diag == CUBLAS_DIAG_UNIT),
                LOWER_EFF=lower_eff,
                FORWARD=lower_eff,
                IS_DOUBLE=dtype == torch.float64,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
            )
        else:
            _real_tpsv_kernel[(1,)](uplo, trans, diag, n, AP, x, incx)
    return x


def _complex_tpsv(uplo, trans, diag, n, AP, x, incx, dtype):
    _check_common(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype is dtype and x.dtype is dtype
    if n == 0:
        return x
    uplo, trans, conj = _row_major_tpsv_args(uplo, trans)
    AP_real = torch.view_as_real(AP)
    x_real = torch.view_as_real(x)
    with torch_device_fn.device(AP.device):
        if incx == 1:
            trans_flag = int(trans != CUBLAS_OP_N)
            lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans != CUBLAS_OP_N))
            if dtype == torch.complex128 and diag == CUBLAS_DIAG_NON_UNIT:
                block_n = 16 if n <= 512 else 32
            else:
                block_n = 16 if n <= 64 else 32
            panel_count = triton.cdiv(n, block_n)
            forward = bool(lower_eff)
            if dtype == torch.complex64 and forward and n <= 256:
                chunk = 4
            else:
                chunk = 1
            num_warps = 2 if dtype == torch.complex64 and not forward and n <= 64 else 4
            _complex_tpsv_blocked_kernel[(panel_count,)](
                AP_real,
                x_real,
                _tpsv_flags(AP.device),
                n,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=int(diag == CUBLAS_DIAG_UNIT),
                CONJ=conj,
                LOWER_EFF=lower_eff,
                FORWARD=forward,
                IS_DOUBLE=dtype == torch.complex128,
                BLOCK_N=block_n,
                CHUNK=chunk,
                num_warps=num_warps,
            )
        else:
            _complex_tpsv_kernel[(1,)](
                uplo,
                trans,
                diag,
                n,
                AP_real,
                x_real,
                incx,
                CONJ=conj,
            )
    return x


def stpsv(uplo, trans, diag, n, AP, x, incx):
    assert trans != CUBLAS_OP_C
    return _real_tpsv(uplo, trans, diag, n, AP, x, incx, torch.float32)


def dtpsv(uplo, trans, diag, n, AP, x, incx):
    assert trans != CUBLAS_OP_C
    return _real_tpsv(uplo, trans, diag, n, AP, x, incx, torch.float64)


def ctpsv(uplo, trans, diag, n, AP, x, incx):
    return _complex_tpsv(uplo, trans, diag, n, AP, x, incx, torch.complex64)


def ztpsv(uplo, trans, diag, n, AP, x, incx):
    return _complex_tpsv(uplo, trans, diag, n, AP, x, incx, torch.complex128)

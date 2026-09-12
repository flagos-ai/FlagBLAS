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


_common = importlib.import_module("flag_blas.ops.level2.trsv")

CUBLAS_DIAG_UNIT = 1
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_OP_N = 0


@libentry()
@triton.jit
def _strsv_streaming_kernel(
    a_ptr,
    x_ptr,
    n,
    lda,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    NATURAL_ROWS: tl.constexpr,
    PRELOAD_DIAG: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    positions = tl.arange(0, BLOCK_N)
    rows = positions if NATURAL_ROWS or FORWARD else n - 1 - positions
    valid = positions < n
    rhs = tl.load(x_ptr + rows, mask=valid, other=0.0)
    if PRELOAD_DIAG and not UNIT:
        diagonal = tl.load(
            a_ptr + rows * lda + rows, mask=valid, other=1.0
        )

    for step in tl.range(0, n):
        active = step < n
        row = step if FORWARD else n - 1 - step
        if NATURAL_ROWS:
            current = (rows == row) & valid & active
        else:
            current = (positions == step) & valid & active

        value = rhs
        if not UNIT:
            if PRELOAD_DIAG:
                value = rhs / diagonal
            else:
                diagonal = tl.load(
                    a_ptr + rows * lda + rows,
                    mask=current,
                    other=1.0,
                )
                value = rhs / diagonal
        rhs = tl.where(current, value, rhs)
        tl.store(x_ptr + rows, rhs, mask=current)
        tl.debug_barrier()
        solved = tl.load(x_ptr + row, mask=active, other=0.0)

        if NATURAL_ROWS:
            if FORWARD:
                future = (rows > row) & valid & active
            else:
                future = (rows < row) & valid & active
        else:
            future = (positions > step) & valid & active
        if TRANS == 0:
            matrix_offset = rows * lda + row
        else:
            matrix_offset = row * lda + rows
        matrix = tl.load(
            a_ptr + matrix_offset, mask=future, other=0.0
        )
        rhs = tl.where(future, rhs - matrix * solved, rhs)


@libentry()
@triton.jit
def _ctrsv_streaming_kernel(
    a_ptr,
    x_ptr,
    n,
    lda,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    FORWARD: tl.constexpr,
    NATURAL_ROWS: tl.constexpr,
    PRELOAD_DIAG: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    positions = tl.arange(0, BLOCK_N)
    rows = positions if NATURAL_ROWS or FORWARD else n - 1 - positions
    valid = positions < n
    rhs_r = tl.load(x_ptr + 2 * rows, mask=valid, other=0.0)
    rhs_i = tl.load(x_ptr + 2 * rows + 1, mask=valid, other=0.0)
    if PRELOAD_DIAG and not UNIT:
        diagonal_offset = 2 * (rows * lda + rows)
        preload_diagonal_r = tl.load(
            a_ptr + diagonal_offset, mask=valid, other=1.0
        )
        preload_diagonal_i = tl.load(
            a_ptr + diagonal_offset + 1, mask=valid, other=0.0
        )
        if CONJ:
            preload_diagonal_i = -preload_diagonal_i
        preload_denominator = (
            preload_diagonal_r * preload_diagonal_r
            + preload_diagonal_i * preload_diagonal_i
        )

    for step in tl.range(0, n):
        active = step < n
        row = step if FORWARD else n - 1 - step
        if NATURAL_ROWS:
            current = (rows == row) & valid & active
        else:
            current = (positions == step) & valid & active

        value_r = rhs_r
        value_i = rhs_i
        if not UNIT:
            if PRELOAD_DIAG:
                diagonal_r = preload_diagonal_r
                diagonal_i = preload_diagonal_i
                denominator = preload_denominator
            else:
                diagonal_offset = 2 * (rows * lda + rows)
                diagonal_r = tl.load(
                    a_ptr + diagonal_offset,
                    mask=current,
                    other=1.0,
                )
                diagonal_i = tl.load(
                    a_ptr + diagonal_offset + 1,
                    mask=current,
                    other=0.0,
                )
                if CONJ:
                    diagonal_i = -diagonal_i
                denominator = (
                    diagonal_r * diagonal_r + diagonal_i * diagonal_i
                )
            value_r = (
                rhs_r * diagonal_r + rhs_i * diagonal_i
            ) / denominator
            value_i = (
                rhs_i * diagonal_r - rhs_r * diagonal_i
            ) / denominator
        rhs_r = tl.where(current, value_r, rhs_r)
        rhs_i = tl.where(current, value_i, rhs_i)
        tl.store(x_ptr + 2 * rows, rhs_r, mask=current)
        tl.store(x_ptr + 2 * rows + 1, rhs_i, mask=current)
        tl.debug_barrier()
        solved_r = tl.load(x_ptr + 2 * row, mask=active, other=0.0)
        solved_i = tl.load(
            x_ptr + 2 * row + 1, mask=active, other=0.0
        )

        if NATURAL_ROWS:
            if FORWARD:
                future = (rows > row) & valid & active
            else:
                future = (rows < row) & valid & active
        else:
            future = (positions > step) & valid & active
        if TRANS == 0:
            matrix_offset = 2 * (rows * lda + row)
        else:
            matrix_offset = 2 * (row * lda + rows)
        matrix_r = tl.load(
            a_ptr + matrix_offset, mask=future, other=0.0
        )
        matrix_i = tl.load(
            a_ptr + matrix_offset + 1, mask=future, other=0.0
        )
        if CONJ:
            matrix_i = -matrix_i
        update_r = matrix_r * solved_r - matrix_i * solved_i
        update_i = matrix_r * solved_i + matrix_i * solved_r
        rhs_r = tl.where(future, rhs_r - update_r, rhs_r)
        rhs_i = tl.where(future, rhs_i - update_i, rhs_i)


@libentry()
@triton.jit
def _strsv_ordered_panel_kernel(
    a_ptr,
    x_ptr,
    n,
    lda,
    incx,
    panel_id,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    PANEL_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Solve one host-ordered panel without inter-program spin synchronization."""
    offs = tl.arange(0, BLOCK_K)
    if FORWARD:
        panel_start = panel_id * PANEL_N
        panel_end = tl.minimum(panel_start + PANEL_N, n)
    else:
        panel_end = n - panel_id * PANEL_N
        panel_start = tl.maximum(0, panel_end - PANEL_N)
    panel_count = panel_end - panel_start

    for local in tl.range(0, panel_count):
        if FORWARD:
            row = panel_start + local
        else:
            row = panel_end - 1 - local

        acc = 0.0
        for k0 in tl.range(0, n, BLOCK_K):
            cols = k0 + offs
            if FORWARD:
                dep_mask = cols < row
            else:
                dep_mask = cols > row
            dep_mask = dep_mask & (cols < n)
            if TRANS == 0:
                a_off = row * lda + cols
            else:
                a_off = cols * lda + row
            a_vals = tl.load(a_ptr + a_off, mask=dep_mask, other=0.0)
            x_vals = tl.load(x_ptr + cols * incx, mask=dep_mask, other=0.0)
            acc += tl.sum(a_vals * x_vals, axis=0)

        value = tl.load(x_ptr + row * incx) - acc
        if not UNIT:
            value /= tl.load(a_ptr + row * lda + row)
        tl.store(x_ptr + row * incx, value)
        tl.debug_barrier()


@libentry()
@triton.jit
def _ctrsv_ordered_panel_kernel(
    a_ptr,
    x_ptr,
    n,
    lda,
    incx,
    panel_id,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    FORWARD: tl.constexpr,
    PANEL_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Complex counterpart of the ordered official-library-style panel solve."""
    offs = tl.arange(0, BLOCK_K)
    if FORWARD:
        panel_start = panel_id * PANEL_N
        panel_end = tl.minimum(panel_start + PANEL_N, n)
    else:
        panel_end = n - panel_id * PANEL_N
        panel_start = tl.maximum(0, panel_end - PANEL_N)
    panel_count = panel_end - panel_start
    incx2 = incx * 2

    for local in tl.range(0, panel_count):
        if FORWARD:
            row = panel_start + local
        else:
            row = panel_end - 1 - local

        acc_r = 0.0
        acc_i = 0.0
        for k0 in tl.range(0, n, BLOCK_K):
            cols = k0 + offs
            if FORWARD:
                dep_mask = cols < row
            else:
                dep_mask = cols > row
            dep_mask = dep_mask & (cols < n)
            if TRANS == 0:
                a_off = (row * lda + cols) * 2
            else:
                a_off = (cols * lda + row) * 2
            ar = tl.load(a_ptr + a_off, mask=dep_mask, other=0.0)
            ai = tl.load(a_ptr + a_off + 1, mask=dep_mask, other=0.0)
            if CONJ:
                ai = -ai
            xr = tl.load(x_ptr + cols * incx2, mask=dep_mask, other=0.0)
            xi = tl.load(x_ptr + cols * incx2 + 1, mask=dep_mask, other=0.0)
            acc_r += tl.sum(ar * xr - ai * xi, axis=0)
            acc_i += tl.sum(ar * xi + ai * xr, axis=0)

        value_r = tl.load(x_ptr + row * incx2) - acc_r
        value_i = tl.load(x_ptr + row * incx2 + 1) - acc_i
        if not UNIT:
            diag_off = (row * lda + row) * 2
            diag_r = tl.load(a_ptr + diag_off)
            diag_i = tl.load(a_ptr + diag_off + 1)
            if CONJ:
                diag_i = -diag_i
            denom = diag_r * diag_r + diag_i * diag_i
            out_r = (value_r * diag_r + value_i * diag_i) / denom
            out_i = (value_i * diag_r - value_r * diag_i) / denom
            value_r = out_r
            value_i = out_i
        tl.store(x_ptr + row * incx2, value_r)
        tl.store(x_ptr + row * incx2 + 1, value_i)
        tl.debug_barrier()


@libentry()
@triton.jit
def _strsv_block_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    lda,
    incx,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    PANEL_N: tl.constexpr,
):
    """Solve one diagonal panel; static_range avoids the common-path warning."""
    offs = tl.arange(0, PANEL_N)
    size = end - start
    for step in tl.static_range(0, PANEL_N):
        if FORWARD:
            row = start + step
            cols = start + offs
        else:
            row = end - 1 - step
            cols = end - 1 - offs
        active = step < size
        mask = (offs < step) & (offs < size) & active
        if TRANS == 0:
            a_off = row * lda + cols
        else:
            a_off = cols * lda + row
        av = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        xv = tl.load(x_ptr + cols * incx, mask=mask, other=0.0)
        value = tl.load(x_ptr + row * incx, mask=active, other=0.0)
        value -= tl.sum(av * xv, axis=0)
        if not UNIT:
            value /= tl.load(a_ptr + row * lda + row, mask=active, other=1.0)
        tl.store(x_ptr + row * incx, value, mask=active)


@libentry()
@triton.jit
def _strsv_block_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    lda,
    incx,
    TRANS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PANEL_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, PANEL_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_off = rows[:, None] * lda + cols[None, :]
    else:
        a_off = cols[None, :] * lda + rows[:, None]
    mask = row_mask[:, None] & col_mask[None, :]
    av = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    xv = tl.load(x_ptr + cols * incx, mask=col_mask, other=0.0)
    update = tl.sum(av * xv[None, :], axis=1)
    rhs = tl.load(x_ptr + rows * incx, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * incx, rhs - update, mask=row_mask)


@libentry()
@triton.jit
def _ctrsv_block_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    lda,
    incx,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    FORWARD: tl.constexpr,
    PANEL_N: tl.constexpr,
):
    offs = tl.arange(0, PANEL_N)
    size = end - start
    incx2 = incx * 2
    for step in tl.static_range(0, PANEL_N):
        if FORWARD:
            row = start + step
            cols = start + offs
        else:
            row = end - 1 - step
            cols = end - 1 - offs
        active = step < size
        mask = (offs < step) & (offs < size) & active
        if TRANS == 0:
            a_off = (row * lda + cols) * 2
        else:
            a_off = (cols * lda + row) * 2
        ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        xr = tl.load(x_ptr + cols * incx2, mask=mask, other=0.0)
        xi = tl.load(x_ptr + cols * incx2 + 1, mask=mask, other=0.0)
        vr = tl.load(x_ptr + row * incx2, mask=active, other=0.0)
        vi = tl.load(x_ptr + row * incx2 + 1, mask=active, other=0.0)
        vr -= tl.sum(ar * xr - ai * xi, axis=0)
        vi -= tl.sum(ar * xi + ai * xr, axis=0)
        if not UNIT:
            diag_off = (row * lda + row) * 2
            dr = tl.load(a_ptr + diag_off, mask=active, other=1.0)
            di = tl.load(a_ptr + diag_off + 1, mask=active, other=0.0)
            if CONJ:
                di = -di
            denom = dr * dr + di * di
            out_r = (vr * dr + vi * di) / denom
            out_i = (vi * dr - vr * di) / denom
            vr, vi = out_r, out_i
        tl.store(x_ptr + row * incx2, vr, mask=active)
        tl.store(x_ptr + row * incx2 + 1, vi, mask=active)


@libentry()
@triton.jit
def _ctrsv_block_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    lda,
    incx,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PANEL_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, PANEL_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_off = (rows[:, None] * lda + cols[None, :]) * 2
    else:
        a_off = (cols[None, :] * lda + rows[:, None]) * 2
    mask = row_mask[:, None] & col_mask[None, :]
    ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
    if CONJ:
        ai = -ai
    incx2 = incx * 2
    xr = tl.load(x_ptr + cols * incx2, mask=col_mask, other=0.0)
    xi = tl.load(x_ptr + cols * incx2 + 1, mask=col_mask, other=0.0)
    update_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
    update_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
    rhs_r = tl.load(x_ptr + rows * incx2, mask=row_mask, other=0.0)
    rhs_i = tl.load(x_ptr + rows * incx2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * incx2, rhs_r - update_r, mask=row_mask)
    tl.store(x_ptr + rows * incx2 + 1, rhs_i - update_i, mask=row_mask)


def _official_panel_size(n: int) -> int:
    return min(triton.next_power_of_2(n), 2048)


def _block_k(n: int) -> int:
    return min(triton.next_power_of_2(n), 2048)


def _num_warps(n: int) -> int:
    return 8 if n >= 512 else 4


def _block_panel_ranges(n: int, panel_n: int, forward: bool):
    if forward:
        start = 0
        while start < n:
            end = min(start + panel_n, n)
            yield start, end
            start = end
    else:
        end = n
        while end > 0:
            start = max(0, end - panel_n)
            yield start, end
            end = start


def _use_blocked_path(n: int) -> bool:
    return n >= 64


def _streaming_config(n: int, trans: int, complex_: bool):
    block_n = max(64, triton.next_power_of_2(n))
    num_warps = min(64, max(1, block_n // 64))
    if trans == CUBLAS_OP_N:
        limit = 512
        if 64 <= n <= limit:
            return block_n, num_warps, False, False
        return None
    if complex_:
        if 64 <= n <= 1024:
            return block_n, num_warps, True, True
        if n <= 4096:
            return block_n, num_warps, False, False
        return None
    if 64 <= n <= 2048:
        return block_n, num_warps, True, True
    if n <= 4096:
        return block_n, num_warps, False, False
    return None


def _blocked_config(n: int, trans: int, complex_: bool):
    if 64 < n < 512 and not complex_ and trans != CUBLAS_OP_N:
        return 64, 32, 2, 1
    if complex_:
        return 64, 128, 2, 8
    return 64, 64, 8, 4


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
    _common._check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    panel_n = _official_panel_size(n)
    forward = (trans == CUBLAS_OP_N and uplo == CUBLAS_FILL_MODE_LOWER) or (
        trans != CUBLAS_OP_N and uplo != CUBLAS_FILL_MODE_LOWER
    )
    with torch_device_fn.device(A.device):
        streaming = (
            _streaming_config(n, trans, False)
            if incx == 1 and lda == n
            else None
        )
        if streaming is not None:
            block_n, num_warps, natural_rows, preload_diag = streaming
            _strsv_streaming_kernel[(1,)](
                A,
                x,
                n,
                lda,
                TRANS=trans,
                UNIT=diag == CUBLAS_DIAG_UNIT,
                FORWARD=forward,
                NATURAL_ROWS=natural_rows,
                PRELOAD_DIAG=preload_diag,
                BLOCK_N=block_n,
                num_warps=num_warps,
                num_stages=1,
            )
            return
        if _use_blocked_path(n):
            panel_n, block_m, panel_warps, update_warps = _blocked_config(
                n, trans, False
            )
            for start, end in _block_panel_ranges(n, panel_n, forward):
                _strsv_block_panel_kernel[(1,)](
                    A,
                    x,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans,
                    UNIT=diag == CUBLAS_DIAG_UNIT,
                    FORWARD=forward,
                    PANEL_N=panel_n,
                    num_warps=panel_warps,
                )
                if forward:
                    row_base, row_count = end, n - end
                else:
                    row_base, row_count = 0, start
                if row_count > 0:
                    _strsv_block_update_kernel[
                        (triton.cdiv(row_count, block_m),)
                    ](
                        A,
                        x,
                        row_base,
                        row_count,
                        start,
                        end,
                        lda,
                        incx,
                        TRANS=trans,
                        BLOCK_M=block_m,
                        PANEL_N=panel_n,
                        num_warps=update_warps,
                    )
            return
        for panel_id in range(triton.cdiv(n, panel_n)):
            _strsv_ordered_panel_kernel[(1,)](
                A,
                x,
                n,
                lda,
                incx,
                panel_id,
                UPLO=uplo,
                TRANS=trans,
                UNIT=diag == CUBLAS_DIAG_UNIT,
                FORWARD=forward,
                PANEL_N=panel_n,
                BLOCK_K=_block_k(n),
                num_warps=_num_warps(n),
            )


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
    _common._check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    panel_n = _official_panel_size(n)
    forward = (trans == CUBLAS_OP_N and uplo == CUBLAS_FILL_MODE_LOWER) or (
        trans != CUBLAS_OP_N and uplo != CUBLAS_FILL_MODE_LOWER
    )
    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        streaming = (
            _streaming_config(n, trans, True)
            if incx == 1 and lda == n
            else None
        )
        if streaming is not None:
            block_n, num_warps, natural_rows, preload_diag = streaming
            _ctrsv_streaming_kernel[(1,)](
                A_real,
                x_real,
                n,
                lda,
                TRANS=trans,
                UNIT=diag == CUBLAS_DIAG_UNIT,
                CONJ=trans == 2,
                FORWARD=forward,
                NATURAL_ROWS=natural_rows,
                PRELOAD_DIAG=preload_diag,
                BLOCK_N=block_n,
                num_warps=num_warps,
                num_stages=1,
            )
            return
        if _use_blocked_path(n):
            panel_n, block_m, panel_warps, update_warps = _blocked_config(
                n, trans, True
            )
            for start, end in _block_panel_ranges(n, panel_n, forward):
                _ctrsv_block_panel_kernel[(1,)](
                    A_real,
                    x_real,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans,
                    UNIT=diag == CUBLAS_DIAG_UNIT,
                    CONJ=trans == 2,
                    FORWARD=forward,
                    PANEL_N=panel_n,
                    num_warps=panel_warps,
                )
                if forward:
                    row_base, row_count = end, n - end
                else:
                    row_base, row_count = 0, start
                if row_count > 0:
                    _ctrsv_block_update_kernel[
                        (triton.cdiv(row_count, block_m),)
                    ](
                        A_real,
                        x_real,
                        row_base,
                        row_count,
                        start,
                        end,
                        lda,
                        incx,
                        TRANS=trans,
                        CONJ=trans == 2,
                        BLOCK_M=block_m,
                        PANEL_N=panel_n,
                        num_warps=update_warps,
                    )
            return
        for panel_id in range(triton.cdiv(n, panel_n)):
            _ctrsv_ordered_panel_kernel[(1,)](
                A_real,
                x_real,
                n,
                lda,
                incx,
                panel_id,
                UPLO=uplo,
                TRANS=trans,
                UNIT=diag == CUBLAS_DIAG_UNIT,
                CONJ=trans == 2,
                FORWARD=forward,
                PANEL_N=panel_n,
                BLOCK_K=_block_k(n),
                num_warps=_num_warps(n),
            )

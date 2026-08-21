# DEBUG: hygon 平台 symv 算子实现，当前处于调试（debug）阶段，尚未稳定收敛
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

import copy

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.symv import (
    ScalarType,
    _check_common,
    _complex_scalars,
    _f64_to_i64,
    _row_major_uplo,
    _strided_y,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils.libentry import LibTuner


def _prune_zsymv_row_configs(configs, named_args, **kwargs):
    configs = copy.deepcopy(configs)
    n = named_args["n"]
    valid = [config for config in configs if config.kwargs["BLOCK_N"] >= n]
    block_n = min(config.kwargs["BLOCK_N"] for config in valid)
    return [config for config in valid if config.kwargs["BLOCK_N"] == block_n]


@LibTuner.register_policy("hygon_symv_tail_stable")
def _hygon_symv_tail_stable_policy(bench_fn, configs, args, kwargs):
    timings = {config: bench_fn(config) for config in configs}
    best_config = min(timings, key=lambda config: timings[config][-1])
    return best_config, timings


@triton.autotune(
    configs=runtime.get_tuned_config("csymv_hygon"),
    key=["n", "UPLO"],
    restore_value=["y_ptr"],
)
@triton.jit
def csymv_hygon_kernel(
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if UPLO == 0:
        if pid_m < pid_n:
            return
    else:
        if pid_m > pid_n:
            return

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]
    y_rows_off = rows * INCY * 2
    y_cols_off = cols * INCY * 2

    x_rows_bits = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0)
    x_cols_bits = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0)
    xrr = (x_rows_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xri = ((x_rows_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xcr = (x_cols_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xci = ((x_cols_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i + j * LDA, j + i * LDA)
        a_bits = tl.load(a_ptr + elem_off, mask=mask2d, other=0)
        ar = (a_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        ai = ((a_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
        acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, out_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, out_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[:, None] + cols[None, :] * LDA
    a_bits = tl.load(a_ptr + elem_off, mask=mask2d, other=0)
    ar = (a_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    ai = ((a_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    row_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
    row_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
    col_r = tl.sum(ar * xrr[:, None] - ai * xri[:, None], axis=0)
    col_i = tl.sum(ar * xri[:, None] + ai * xrr[:, None], axis=0)
    out_row_r = alpha_r * row_r - alpha_i * row_i
    out_row_i = alpha_r * row_i + alpha_i * row_r
    out_col_r = alpha_r * col_r - alpha_i * col_i
    out_col_i = alpha_r * col_i + alpha_i * col_r
    tl.atomic_add(y_ptr + y_rows_off, out_row_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, out_row_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, out_col_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, out_col_i, mask=col_mask, sem="relaxed")


@triton.jit
def scale_zsymv_y_kernel(
    y_ptr,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    n,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    y_offsets = offsets * INCY * 2
    if BETA_IS_ZERO:
        out_r = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        out_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    else:
        beta_r = beta_r_int.to(tl.float64, bitcast=True)
        beta_i = beta_i_int.to(tl.float64, bitcast=True)
        yr = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)
        yi = tl.load(y_ptr + y_offsets + 1, mask=mask, other=0.0)
        out_r = beta_r * yr - beta_i * yi
        out_i = beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, out_r, mask=mask)
    tl.store(y_ptr + y_offsets + 1, out_i, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("zsymv_hygon_row_fused"),
    key=["n", "UPLO"],
    restore_value=["y_ptr"],
    prune_configs_by={"early_config_prune": _prune_zsymv_row_configs},
)
@triton.jit
def zsymv_hygon_row_fused_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n
    if UPLO == 0:
        use_direct = cols <= row
    else:
        use_direct = cols >= row
    elem_off = tl.where(use_direct, row + cols * LDA, cols + row * LDA)
    a_off = elem_off * 2
    ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
    x_off = cols * INCX * 2
    xr = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    xi = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0)
    acc_r = tl.sum(ar * xr - ai * xi, axis=0)
    acc_i = tl.sum(ar * xi + ai * xr, axis=0)

    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    beta_r = beta_r_int.to(tl.float64, bitcast=True)
    beta_i = beta_i_int.to(tl.float64, bitcast=True)
    y_off = row * INCY * 2
    if BETA_IS_ZERO:
        yr = 0.0
        yi = 0.0
    else:
        yr = tl.load(y_ptr + y_off)
        yi = tl.load(y_ptr + y_off + 1)
    out_r = alpha_r * acc_r - alpha_i * acc_i + beta_r * yr - beta_i * yi
    out_i = alpha_r * acc_i + alpha_i * acc_r + beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, out_r)
    tl.store(y_ptr + y_off + 1, out_i)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("zsymv_hygon_compact"),
    key=["n", "UPLO"],
    restore_value=["y_ptr"],
    policy="hygon_symv_tail_stable",
)
@triton.jit
def zsymv_hygon_compact_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tri_row = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    tri_col = pid - tri_row * (tri_row + 1) // 2
    if UPLO == 0:
        pid_m = tiles - 1 - tri_col
        pid_n = tiles - 1 - tri_row
    else:
        pid_m = tri_col
        pid_n = tri_row

    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]
    y_rows_off = rows * INCY * 2
    y_cols_off = cols * INCY * 2
    x_rows_off = rows * INCX * 2
    x_cols_off = cols * INCX * 2
    xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i + j * LDA, j + i * LDA)
        a_off = elem_off * 2
        ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
        acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
        acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, out_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, out_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[:, None] + cols[None, :] * LDA
    a_off = elem_off * 2
    ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
    row_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
    row_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
    col_r = tl.sum(ar * xrr[:, None] - ai * xri[:, None], axis=0)
    col_i = tl.sum(ar * xri[:, None] + ai * xrr[:, None], axis=0)
    out_row_r = alpha_r * row_r - alpha_i * row_i
    out_row_i = alpha_r * row_i + alpha_i * row_r
    out_col_r = alpha_r * col_r - alpha_i * col_i
    out_col_i = alpha_r * col_i + alpha_i * col_r
    tl.atomic_add(y_ptr + y_rows_off, out_row_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, out_row_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, out_col_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, out_col_i, mask=col_mask, sem="relaxed")


def csymv(
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

    A_packed = A.view(torch.int64)
    x_packed = x.view(torch.int64)
    y_real = torch.view_as_real(y)
    with torch_device_fn.device(A.device):
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))

        def grid(meta):
            tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
            return tiles, tiles

        csymv_hygon_kernel[grid](
            A_packed,
            x_packed,
            y_real,
            ar,
            ai,
            n,
            lda,
            incx,
            incy,
            UPLO=_row_major_uplo(uplo),
        )


def zsymv(
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
    assert A.dtype == torch.complex128 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        y_view = _strided_y(y, n, incy)
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    ar_i = _f64_to_i64(ar)
    ai_i = _f64_to_i64(ai)
    br_i = _f64_to_i64(br)
    bi_i = _f64_to_i64(bi)
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    with torch_device_fn.device(A.device):
        if n <= 1024:
            zsymv_hygon_row_fused_kernel[(n,)](
                A_real,
                x_real,
                y_real,
                ar_i,
                ai_i,
                br_i,
                bi_i,
                n,
                lda,
                incx,
                incy,
                UPLO=_row_major_uplo(uplo),
                BETA_IS_ZERO=br == 0.0 and bi == 0.0,
            )
            return

        if br != 1.0 or bi != 0.0:
            scale_zsymv_y_kernel[(triton.cdiv(n, 256),)](
                y_real,
                br_i,
                bi_i,
                n,
                incy,
                BETA_IS_ZERO=br == 0.0 and bi == 0.0,
                BLOCK_SIZE=256,
                num_warps=4,
            )

        def grid(meta):
            tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
            return (tiles * (tiles + 1) // 2,)

        zsymv_hygon_compact_kernel[grid](
            A_real,
            x_real,
            y_real,
            ar_i,
            ai_i,
            n,
            lda,
            incx,
            incy,
            UPLO=_row_major_uplo(uplo),
        )

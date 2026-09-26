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
import importlib

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn


_common = importlib.import_module("flag_blas.ops.level2.hemv")


@triton.jit
def _load_complex_tile(A, offsets, mask):
    # Fetch both components together; scalar strided loads waste load bandwidth
    # on the PPU. FP32 uses one 64-bit load, FP64 an adjacent pair of doubles.
    if A.dtype.element_ty == tl.float32:
        packed = tl.load(A.to(tl.pointer_type(tl.int64)) + offsets, mask, 0)
        ar = packed.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (packed >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    else:
        values = tl.load(
            A + 2 * offsets[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask[:, :, None],
            0,
        )
        ar, ai = tl.split(values)
    return ar, ai


@triton.jit
def hemv_atomic_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r,
    alpha_i,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP64: tl.constexpr,
    FOLDED: tl.constexpr = False,
):
    if FP64:
        alpha_r = alpha_r.to(tl.int64).to(tl.float64, bitcast=True)
        alpha_i = alpha_i.to(tl.int64).to(tl.float64, bitcast=True)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if FOLDED:
        # Fold an approximately half-sized rectangular grid onto the stored
        # triangle. For odd tile counts, exclude the duplicated middle row.
        tiles = tl.cdiv(n, BLOCK_SIZE)
        if pid_m * 2 == tiles - 1 and pid_n > pid_m:
            return
        reflect = pid_n > pid_m
        pid_m = tl.where(reflect, tiles - 1 - pid_m, pid_m)
        pid_n = tl.where(reflect, tiles - pid_n, pid_n)
        if UPLO == 1:
            pid_m, pid_n = pid_n, pid_m
    else:
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
    mask2d = col_mask[:, None] & row_mask[None, :]
    y_rows_off = rows * INCY * 2
    y_cols_off = cols * INCY * 2

    x_rows_off = rows * INCX * 2
    x_cols_off = cols * INCX * 2
    xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)

    if pid_m == pid_n:
        i = rows[None, :]
        j = cols[:, None]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i * LDA + j, j * LDA + i)
        ar, ai = _load_complex_tile(a_ptr, elem_off, mask2d)
        ai = tl.where(use_direct, ai, -ai)
        ai = tl.where(i == j, 0.0, ai)
        acc_r = tl.sum(ar * xcr[:, None] - ai * xci[:, None], axis=0)
        acc_i = tl.sum(ar * xci[:, None] + ai * xcr[:, None], axis=0)
        res_r = alpha_r * acc_r - alpha_i * acc_i
        res_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, res_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, res_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[None, :] * LDA + cols[:, None]
    ar, ai = _load_complex_tile(a_ptr, elem_off, mask2d)

    acc_rows_r = tl.sum(ar * xcr[:, None] - ai * xci[:, None], axis=0)
    acc_rows_i = tl.sum(ar * xci[:, None] + ai * xcr[:, None], axis=0)
    acc_cols_r = tl.sum(ar * xrr[None, :] + ai * xri[None, :], axis=1)
    acc_cols_i = tl.sum(ar * xri[None, :] - ai * xrr[None, :], axis=1)

    row_res_r = alpha_r * acc_rows_r - alpha_i * acc_rows_i
    row_res_i = alpha_r * acc_rows_i + alpha_i * acc_rows_r
    col_res_r = alpha_r * acc_cols_r - alpha_i * acc_cols_i
    col_res_i = alpha_r * acc_cols_i + alpha_i * acc_cols_r

    tl.atomic_add(y_ptr + y_rows_off, row_res_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, row_res_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, col_res_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, col_res_i, mask=col_mask, sem="relaxed")


# Keep independent autotune caches and include the low-warp configurations
# measured on the PPU. Larger FP64 tiles were slower in the configuration sweep.
chemv_kernel = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": block, "FOLDED": folded}, num_warps=warps, num_stages=1
        )
        for block, warps, folded in (
            (16, 1, True), (32, 1, True), (32, 2, True),
            (32, 4, True), (32, 1, False), (64, 4, True),
        )
    ],
    key=["n", "LDA", "INCX", "INCY", "UPLO"],
    restore_value=["y_ptr"],
)(hemv_atomic_kernel)

zhemv_kernel = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": block, "FOLDED": folded}, num_warps=warps, num_stages=1
        )
        for block, warps, folded in (
            (16, 1, True), (16, 2, True), (32, 4, True),
            (16, 1, False), (32, 4, False),
        )
    ],
    key=["n", "LDA", "INCX", "INCY", "UPLO"],
    restore_value=["y_ptr"],
)(hemv_atomic_kernel)


def _hemv(dtype, kernel, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    assert A.dtype == dtype == x.dtype == y.dtype
    _common._check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return
    ar, ai, br, bi = _common._complex_scalars(alpha, beta)
    fp64 = dtype == torch.complex128
    with torch_device_fn.device(A.device):
        y_view = _common._strided_y(y, n, incy)
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        if ar == 0.0 and ai == 0.0:
            return
        if fp64:
            ar, ai = _common._f64_to_i64(ar), _common._f64_to_i64(ai)

        def grid(meta):
            tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
            if meta["FOLDED"]:
                return (triton.cdiv(tiles, 2), tiles + 1)
            return (tiles, tiles)

        kernel[grid](
            torch.view_as_real(A), torch.view_as_real(x), torch.view_as_real(y),
            ar, ai, n, lda, incx, incy, UPLO=uplo, FP64=fp64,
        )


# Derive BLOCK_N inside autotune, after AABS has adjusted the tunable config.
# Supplying it at the call site lets AABS also add it to Config.kwargs, which
# makes the final launch receive BLOCK_N twice for non-power-of-two sizes.
chemv_small_kernel = triton.autotune(
    configs=copy.deepcopy(_common._CHEMV_SMALL_CONFIGS),
    key=list(_common._HEMV_KEY),
    restore_value=["y_ptr"],
)(
    triton.heuristics(
        {"BLOCK_N": lambda args: triton.next_power_of_2(args["n"])}
    )(_common.chemv_small_kernel.fn)
)


def chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n >= 256:
        return _hemv(
            torch.complex64, chemv_kernel, uplo, n, alpha, A, lda, x, incx, beta, y, incy
        )
    if not (
        uplo == _common.CUBLAS_FILL_MODE_LOWER
        and 0 < n <= 192
        and incx == 1
        and incy == 1
    ):
        return _common.chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _common._check_common(A, x, y, uplo, n, lda, incx, incy)
    ar, ai, br, bi = _common._complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        return _common.chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    with torch_device_fn.device(A.device):
        chemv_small_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_M"]),)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            br,
            bi,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
            BETA_IS_ZERO=br == 0.0 and bi == 0.0,
        )


def zhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if n < 256:
        return _common.zhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    return _hemv(
        torch.complex128, zhemv_kernel, uplo, n, alpha, A, lda, x, incx, beta, y, incy
    )

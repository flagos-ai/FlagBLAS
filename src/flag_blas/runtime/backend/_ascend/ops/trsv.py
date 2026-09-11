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
from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_common = importlib.import_module("flag_blas.ops.level2.trsv")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_panel"),
    key=["mode_key"],
    restore_value=["x_ptr"],
)
@triton.jit
def strsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    size = end - start
    for step in tl.static_range(0, BLOCK_N):
        if FORWARD:
            raw_i = start + step
            raw_j = start + offs
        else:
            raw_i = end - 1 - step
            raw_j = end - 1 - offs
        active = step < size
        mask = (offs < step) & (offs < size) & active
        i = tl.maximum(raw_i, 0)
        j = tl.maximum(raw_j, 0)
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
            diagonal = tl.load(a_ptr + i + i * LDA, mask=active, other=1.0)
            value = value / diagonal
        tl.store(x_ptr + i * INCX, value, mask=active)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctrsv_panel"),
    key=["mode_key"],
    restore_value=["x_ptr"],
)
@triton.jit
def ctrsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    mode_key,
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
            raw_i = start + step
            raw_j = start + offs
        else:
            raw_i = end - 1 - step
            raw_j = end - 1 - offs
        active = step < size
        mask = (offs < step) & (offs < size) & active
        i = tl.maximum(raw_i, 0)
        j = tl.maximum(raw_j, 0)
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
            diag_base = (i + i * LDA) * 2
            diag_r = tl.load(a_ptr + diag_base, mask=active, other=1.0)
            diag_i = tl.load(a_ptr + diag_base + 1, mask=active, other=0.0)
            if CONJ:
                diag_i = -diag_i
            denom = diag_r * diag_r + diag_i * diag_i
            out_r = (value_r * diag_r + value_i * diag_i) / denom
            out_i = (value_i * diag_r - value_r * diag_i) / denom
            value_r = out_r
            value_i = out_i
        tl.store(x_ptr + i * incx2, value_r, mask=active)
        tl.store(x_ptr + i * incx2 + 1, value_i, mask=active)


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


def strsv(uplo, trans, diag, n, A, lda, x, incx):
    assert A.dtype == torch.float32 == x.dtype
    _common._check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, _ = _common._row_major_dispatch(uplo, trans)
    forward = _common._forward(uplo, trans_flag)
    block_n = 16
    block_m = 128

    with torch_device_fn.device(A.device):
        for start, end in _panel_ranges(n, block_n, forward):
            strsv_panel_kernel[(1,)](
                A,
                x,
                start,
                end,
                lda,
                incx,
                _common._mode_key(uplo, trans_flag, unit),
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                _common.dtrsv_update_kernel[(triton.cdiv(row_count, block_m),)](
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


def ctrsv(uplo, trans, diag, n, A, lda, x, incx):
    assert A.dtype == torch.complex64 == x.dtype
    _common._check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, conj = _common._row_major_dispatch(uplo, trans)
    forward = _common._forward(uplo, trans_flag)
    block_n = 8
    block_m = 64

    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        logical_x_real = None
        kernel_incx = incx
        if incx != 1:
            logical_x_real = x_real.as_strided((n, 2), (incx * 2, 1))
            x_real = logical_x_real.clone()
            kernel_incx = 1
        for start, end in _panel_ranges(n, block_n, forward):
            ctrsv_panel_kernel[(1,)](
                A_real,
                x_real,
                start,
                end,
                lda,
                kernel_incx,
                _common._mode_key(uplo, trans_flag, unit) | (conj << 8),
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                CONJ=conj,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                _common.ctrsv_update_kernel[(triton.cdiv(row_count, block_m),)](
                    A_real,
                    x_real,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    kernel_incx,
                    TRANS=trans_flag,
                    CONJ=conj,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )
        if logical_x_real is not None:
            logical_x_real.copy_(x_real)

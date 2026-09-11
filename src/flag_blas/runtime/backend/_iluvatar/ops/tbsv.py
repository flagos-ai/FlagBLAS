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
import logging
from typing import Union

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, torch.Tensor]

_common = importlib.import_module("flag_blas.ops.level2.tbsv")

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_DIAG_UNIT = 1


_TBSV_KEY = ["n", "k_bucket", "mode_key"]
_TBSV_RESTORE = ["x_ptr"]


def _band_bucket(k: int) -> int:
    if k <= 1:
        return 1
    b = 1
    while b < k and b < 1024:
        b <<= 1
    return b


def _mode_key(uplo: int, trans: int, unit: int) -> int:
    return (uplo << 4) | (trans << 2) | unit


def _prune_stbsv_direct_configs(configs, named_args, **kwargs):
    k = named_args["k"]
    block_k = _band_bucket(k)
    return [c for c in configs if c.kwargs["BLOCK_K"] == block_k]


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------
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
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_direct_kernel(
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
    d = offs + 1

    if (UPLO == 0) and (TRANS == 0):
        full_n = n - k
        for j in tl.range(0, full_n):
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            i = j + d
            av = tl.load(a_ptr + d + j * LDA)
            xv = tl.load(x_ptr + i * INCX)
            xv = xv - av * xj
            tl.store(x_ptr + i * INCX, xv)

        for j in tl.range(full_n, n):
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            i = j + d
            m = i < n
            av = tl.load(a_ptr + d + j * LDA, mask=m, other=0.0)
            xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
            xv = xv - av * xj
            tl.store(x_ptr + i * INCX, xv, mask=m)

    elif (UPLO == 1) and (TRANS == 0):
        rd = BLOCK_K - offs
        full_n = n - k
        for jc in tl.range(0, full_n):
            j = n - 1 - jc
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + k + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            i = j - rd
            av = tl.load(a_ptr + (k - rd) + j * LDA)
            xv = tl.load(x_ptr + i * INCX)
            xv = xv - av * xj
            tl.store(x_ptr + i * INCX, xv)

        for jc in tl.range(full_n, n):
            j = n - 1 - jc
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + k + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            i = j - rd
            m = (rd <= k) & (i >= 0)
            av = tl.load(a_ptr + (k - rd) + j * LDA, mask=m, other=0.0)
            xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
            xv = xv - av * xj
            tl.store(x_ptr + i * INCX, xv, mask=m)

    elif (UPLO == 1) and (TRANS == 1):
        for j in tl.range(0, n):
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + k + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            i = j + d
            m = (d <= k) & (i < n)
            av = tl.load(a_ptr + (k - d) + i * LDA, mask=m, other=0.0)
            xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
            xv = xv - av * xj
            tl.store(x_ptr + i * INCX, xv, mask=m)

    else:
        for jc in tl.range(0, n):
            j = n - 1 - jc
            xj = tl.load(x_ptr + j * INCX)
            if not UNIT:
                ajj = tl.load(a_ptr + j * LDA)
                xj = xj / ajj
                tl.store(x_ptr + j * INCX, xj)
            i = j - d
            m = (d <= k) & (i >= 0)
            av = tl.load(a_ptr + d + i * LDA, mask=m, other=0.0)
            xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
            xv = xv - av * xj
            tl.store(x_ptr + i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_upper_full_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    rd = BLOCK_K - offs

    for jc in tl.range(0, n):
        j = n - 1 - jc
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + k + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j - rd
        m = (rd <= k) & (i >= 0)
        safe_i = tl.where(m, i, 0)
        safe_a = tl.where(m, k - rd, 0)
        av = tl.load(a_ptr + safe_a + j * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + safe_i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + safe_i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_lower_trans_full_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1

    for jc in tl.range(0, n):
        j = n - 1 - jc
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j - d
        m = i >= 0
        safe_i = tl.where(m, i, 0)
        av = tl.load(a_ptr + d + safe_i * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + safe_i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + safe_i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_upper_trans_full_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1

    for j in tl.range(0, n):
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + k + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j + d
        m = i < n
        safe_i = tl.where(m, i, 0)
        safe_a = tl.where(m, k - d, 0)
        av = tl.load(a_ptr + safe_a + safe_i * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + safe_i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + safe_i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_lower_trans16_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1
    full_n = n - 16

    for jc in tl.range(0, full_n):
        j = n - 1 - jc
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j - d
        av = tl.load(a_ptr + d + i * LDA)
        xv = tl.load(x_ptr + i * INCX)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv)

    for jc in tl.range(full_n, n):
        j = n - 1 - jc
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j - d
        m = i >= 0
        av = tl.load(a_ptr + d + i * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_upper_trans_small_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1
    full_n = n - k

    for j in tl.range(0, full_n):
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + k + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j + d
        av = tl.load(a_ptr + (k - d) + i * LDA)
        xv = tl.load(x_ptr + i * INCX)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv)

    for j in tl.range(full_n, n):
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + k + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j + d
        m = i < n
        av = tl.load(a_ptr + (k - d) + i * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_lower64_small_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1
    full_n = n - 64

    for j in tl.range(0, full_n):
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + j * LDA)
        xj = xj / ajj

        i = j + d
        av = tl.load(a_ptr + d + j * LDA)
        xv = tl.load(x_ptr + i * INCX)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv)
        tl.store(x_ptr + j * INCX, xj)

    for j in tl.range(full_n, n):
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + j * LDA)
        xj = xj / ajj

        i = j + d
        m = i < n
        av = tl.load(a_ptr + d + j * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv, mask=m)
        tl.store(x_ptr + j * INCX, xj)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_lower64_1024_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1

    for j in tl.range(0, 960):
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + j * LDA)
        xj = xj / ajj

        i = j + d
        av = tl.load(a_ptr + d + j * LDA)
        xv = tl.load(x_ptr + i * INCX)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv)
        tl.store(x_ptr + j * INCX, xj)

    for j in tl.range(960, 1024):
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + j * LDA)
        xj = xj / ajj

        i = j + d
        m = i < 1024
        av = tl.load(a_ptr + d + j * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv, mask=m)
        tl.store(x_ptr + j * INCX, xj)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_upper16_pair_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    rd = BLOCK_K - offs
    pair_count = (n - 16) // 2

    for pc in tl.range(0, pair_count):
        j0 = n - 1 - pc * 2
        x0 = tl.load(x_ptr + j0 * INCX)
        a00 = tl.load(a_ptr + k + j0 * LDA)
        x0 = x0 / a00
        tl.store(x_ptr + j0 * INCX, x0)

        i0 = j0 - rd
        av0 = tl.load(a_ptr + (k - rd) + j0 * LDA)
        xv0 = tl.load(x_ptr + i0 * INCX)
        xv0 = xv0 - av0 * x0
        tl.store(x_ptr + i0 * INCX, xv0)

        j1 = j0 - 1
        x1 = tl.load(x_ptr + j1 * INCX)
        a11 = tl.load(a_ptr + k + j1 * LDA)
        x1 = x1 / a11
        tl.store(x_ptr + j1 * INCX, x1)

        i1 = j1 - rd
        av1 = tl.load(a_ptr + (k - rd) + j1 * LDA)
        xv1 = tl.load(x_ptr + i1 * INCX)
        xv1 = xv1 - av1 * x1
        tl.store(x_ptr + i1 * INCX, xv1)

    for jc in tl.range(0, 16):
        j = 15 - jc
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + k + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j - rd
        m = i >= 0
        av = tl.load(a_ptr + (k - rd) + j * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_upper256_pair_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    rd = BLOCK_K - offs
    pair_count = (n - 256) // 2

    for pc in tl.range(0, pair_count):
        j0 = n - 1 - pc * 2
        x0 = tl.load(x_ptr + j0 * INCX)
        a00 = tl.load(a_ptr + k + j0 * LDA)
        x0 = x0 / a00
        tl.store(x_ptr + j0 * INCX, x0)

        i0 = j0 - rd
        av0 = tl.load(a_ptr + (k - rd) + j0 * LDA)
        xv0 = tl.load(x_ptr + i0 * INCX)
        xv0 = xv0 - av0 * x0
        tl.store(x_ptr + i0 * INCX, xv0)

        j1 = j0 - 1
        x1 = tl.load(x_ptr + j1 * INCX)
        a11 = tl.load(a_ptr + k + j1 * LDA)
        x1 = x1 / a11
        tl.store(x_ptr + j1 * INCX, x1)

        i1 = j1 - rd
        av1 = tl.load(a_ptr + (k - rd) + j1 * LDA)
        xv1 = tl.load(x_ptr + i1 * INCX)
        xv1 = xv1 - av1 * x1
        tl.store(x_ptr + i1 * INCX, xv1)

    for jc in tl.range(0, 256):
        j = 255 - jc
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + k + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j - rd
        m = i >= 0
        av = tl.load(a_ptr + (k - rd) + j * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv, mask=m)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv_direct"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_stbsv_direct_configs},
)
@triton.jit
def stbsv_upper64_hex_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    BLOCK_K: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_K)
    rd = BLOCK_K - offs
    hex_count = (n - 64) // 16
    tail_start = n - hex_count * 16

    for hc in tl.range(0, hex_count):
        j_base = n - 1 - hc * 16
        for u in tl.static_range(0, 16):
            j = j_base - u
            xj = tl.load(x_ptr + j * INCX)
            ajj = tl.load(a_ptr + k + j * LDA)
            xj = xj / ajj
            tl.store(x_ptr + j * INCX, xj)

            i = j - rd
            av = tl.load(a_ptr + (k - rd) + j * LDA)
            xv = tl.load(x_ptr + i * INCX)
            xv = xv - av * xj
            tl.store(x_ptr + i * INCX, xv)

    for jc in tl.range(0, tail_start):
        j = tail_start - 1 - jc
        xj = tl.load(x_ptr + j * INCX)
        ajj = tl.load(a_ptr + k + j * LDA)
        xj = xj / ajj
        tl.store(x_ptr + j * INCX, xj)

        i = j - rd
        m = i >= 0
        av = tl.load(a_ptr + (k - rd) + j * LDA, mask=m, other=0.0)
        xv = tl.load(x_ptr + i * INCX, mask=m, other=0.0)
        xv = xv - av * xj
        tl.store(x_ptr + i * INCX, xv, mask=m)


@libentry()
@triton.jit
def stbsv_n_update_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    UPLO: tl.constexpr,
    BLOCK_K: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Vector-update path for row-major no-trans solves."""
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1

    if UPLO == 1:
        for j in tl.range(0, n, loop_unroll_factor=UNROLL):
            xj = tl.load(x_ptr + j) / tl.load(a_ptr + k + j * LDA)
            tl.store(x_ptr + j, xj)
            i = j + d
            mask = (d <= k) & (i < n)
            av = tl.load(
                a_ptr + (k - d) + i * LDA, mask=mask, other=0.0
            )
            xv = tl.load(x_ptr + i, mask=mask, other=0.0)
            tl.store(x_ptr + i, xv - av * xj, mask=mask)
    else:
        for jc in tl.range(0, n, loop_unroll_factor=UNROLL):
            j = n - 1 - jc
            xj = tl.load(x_ptr + j) / tl.load(a_ptr + j * LDA)
            tl.store(x_ptr + j, xj)
            i = j - d
            mask = (d <= k) & (i >= 0)
            av = tl.load(a_ptr + d + i * LDA, mask=mask, other=0.0)
            xv = tl.load(x_ptr + i, mask=mask, other=0.0)
            tl.store(x_ptr + i, xv - av * xj, mask=mask)


@libentry()
@triton.jit
def stbsv_pair_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FAST_DIV: tl.constexpr,
):
    """Explicit two-row panel for all real physical TBSV modes."""
    offs = tl.arange(0, BLOCK_K)
    panel_count = n // 2
    forward = ((UPLO == 0) and (TRANS == 0)) or (
        (UPLO == 1) and (TRANS == 1)
    )

    for panel in tl.range(0, panel_count):
        if forward:
            j0 = panel * 2
            j1 = j0 + 1
        else:
            j0 = n - 1 - panel * 2
            j1 = j0 - 1

        diag_band = 0 if UPLO == 0 else k
        diagonal0 = tl.load(a_ptr + diag_band + j0 * LDA)
        if FAST_DIV:
            x0 = tl.load(x_ptr + j0) * libdevice.fast_dividef(1.0, diagonal0)
        else:
            x0 = tl.load(x_ptr + j0) / diagonal0
        inner_band = 1 if UPLO == 0 else k - 1
        inner_row = j0 if TRANS == 0 else j1
        x1 = tl.load(x_ptr + j1)
        x1 -= tl.load(a_ptr + inner_band + inner_row * LDA) * x0
        diagonal1 = tl.load(a_ptr + diag_band + j1 * LDA)
        if FAST_DIV:
            x1 *= libdevice.fast_dividef(1.0, diagonal1)
        else:
            x1 /= diagonal1
        tl.store(x_ptr + j0, x0)
        tl.store(x_ptr + j1, x1)

        if forward:
            targets = j1 + 1 + offs
            valid = targets < n
            d0, d1 = targets - j0, targets - j1
        else:
            targets = j1 - 1 - offs
            valid = targets >= 0
            d0, d1 = j0 - targets, j1 - targets
        band0 = d0 if UPLO == 0 else k - d0
        band1 = d1 if UPLO == 0 else k - d1
        row0 = j0 if TRANS == 0 else targets
        row1 = j1 if TRANS == 0 else targets
        a0 = tl.load(
            a_ptr + band0 + row0 * LDA,
            mask=valid & (d0 <= k),
            other=0.0,
        )
        a1 = tl.load(
            a_ptr + band1 + row1 * LDA,
            mask=valid & (d1 <= k),
            other=0.0,
        )
        current = tl.load(x_ptr + targets, mask=valid, other=0.0)
        tl.store(x_ptr + targets, current - a0 * x0 - a1 * x1, mask=valid)
        tl.debug_barrier()


@libentry()
@triton.jit
def stbsv_quad_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    UPLO: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Four-row panel for the physical-transpose k=256 solve."""
    offs = tl.arange(0, BLOCK_K)
    panel_count = n // 4

    if UPLO == 1:
        for panel in tl.range(0, panel_count):
            j0 = panel * 4
            j1, j2, j3 = j0 + 1, j0 + 2, j0 + 3
            x0 = tl.load(x_ptr + j0) * libdevice.fast_dividef(
                1.0, tl.load(a_ptr + k + j0 * LDA)
            )
            x1 = tl.load(x_ptr + j1)
            x1 -= tl.load(a_ptr + k - 1 + j1 * LDA) * x0
            x1 *= libdevice.fast_dividef(
                1.0, tl.load(a_ptr + k + j1 * LDA)
            )
            x2 = tl.load(x_ptr + j2)
            x2 -= tl.load(a_ptr + k - 2 + j2 * LDA) * x0
            x2 -= tl.load(a_ptr + k - 1 + j2 * LDA) * x1
            x2 *= libdevice.fast_dividef(
                1.0, tl.load(a_ptr + k + j2 * LDA)
            )
            x3 = tl.load(x_ptr + j3)
            x3 -= tl.load(a_ptr + k - 3 + j3 * LDA) * x0
            x3 -= tl.load(a_ptr + k - 2 + j3 * LDA) * x1
            x3 -= tl.load(a_ptr + k - 1 + j3 * LDA) * x2
            x3 *= libdevice.fast_dividef(
                1.0, tl.load(a_ptr + k + j3 * LDA)
            )
            tl.store(x_ptr + j0, x0)
            tl.store(x_ptr + j1, x1)
            tl.store(x_ptr + j2, x2)
            tl.store(x_ptr + j3, x3)

            targets = j0 + 4 + offs
            valid = targets < n
            d0, d1 = targets - j0, targets - j1
            d2, d3 = targets - j2, targets - j3
            a0 = tl.load(
                a_ptr + (k - d0) + targets * LDA,
                mask=valid & (d0 <= k), other=0.0,
            )
            a1 = tl.load(
                a_ptr + (k - d1) + targets * LDA,
                mask=valid & (d1 <= k), other=0.0,
            )
            a2 = tl.load(
                a_ptr + (k - d2) + targets * LDA,
                mask=valid & (d2 <= k), other=0.0,
            )
            a3 = tl.load(
                a_ptr + (k - d3) + targets * LDA,
                mask=valid & (d3 <= k), other=0.0,
            )
            current = tl.load(x_ptr + targets, mask=valid, other=0.0)
            current -= a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
            tl.store(x_ptr + targets, current, mask=valid)
            tl.debug_barrier()
    else:
        for panel in tl.range(0, panel_count):
            j0 = n - 1 - panel * 4
            j1, j2, j3 = j0 - 1, j0 - 2, j0 - 3
            x0 = tl.load(x_ptr + j0) * libdevice.fast_dividef(
                1.0, tl.load(a_ptr + j0 * LDA)
            )
            x1 = tl.load(x_ptr + j1)
            x1 -= tl.load(a_ptr + 1 + j1 * LDA) * x0
            x1 *= libdevice.fast_dividef(1.0, tl.load(a_ptr + j1 * LDA))
            x2 = tl.load(x_ptr + j2)
            x2 -= tl.load(a_ptr + 2 + j2 * LDA) * x0
            x2 -= tl.load(a_ptr + 1 + j2 * LDA) * x1
            x2 *= libdevice.fast_dividef(1.0, tl.load(a_ptr + j2 * LDA))
            x3 = tl.load(x_ptr + j3)
            x3 -= tl.load(a_ptr + 3 + j3 * LDA) * x0
            x3 -= tl.load(a_ptr + 2 + j3 * LDA) * x1
            x3 -= tl.load(a_ptr + 1 + j3 * LDA) * x2
            x3 *= libdevice.fast_dividef(1.0, tl.load(a_ptr + j3 * LDA))
            tl.store(x_ptr + j0, x0)
            tl.store(x_ptr + j1, x1)
            tl.store(x_ptr + j2, x2)
            tl.store(x_ptr + j3, x3)

            targets = j3 - 1 - offs
            valid = targets >= 0
            d0, d1 = j0 - targets, j1 - targets
            d2, d3 = j2 - targets, j3 - targets
            a0 = tl.load(
                a_ptr + d0 + targets * LDA,
                mask=valid & (d0 <= k), other=0.0,
            )
            a1 = tl.load(
                a_ptr + d1 + targets * LDA,
                mask=valid & (d1 <= k), other=0.0,
            )
            a2 = tl.load(
                a_ptr + d2 + targets * LDA,
                mask=valid & (d2 <= k), other=0.0,
            )
            a3 = tl.load(
                a_ptr + d3 + targets * LDA,
                mask=valid & (d3 <= k), other=0.0,
            )
            current = tl.load(x_ptr + targets, mask=valid, other=0.0)
            current -= a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
            tl.store(x_ptr + targets, current, mask=valid)
            tl.debug_barrier()


@libentry()
@triton.jit
def stbsv_upper_n256_tiled_kernel(A, X, n):
    """Coalesced band reads and an eight-row register solve, without scratch."""
    rr = tl.arange(0, 8)
    kk = tl.arange(0, 256)
    for block in tl.range(0, n // 8, num_stages=1, disable_licm=True):
        tl.debug_barrier()
        start = block * 8
        rows = n - 1 - start - rr
        previous = n - 1 - (start - 256 + kk)
        valid = (previous >= 0) & (previous < n)
        distance = 256 + rr[:, None] - kk[None, :]
        coefficients = tl.load(
            A + rows[:, None] * 257 + distance,
            mask=valid[None, :] & (distance <= 256),
            other=0.0,
        )
        # Match the complex tiled path: acquire reads prevent stale in-place
        # solution values on CoreX. Integer zero-add preserves the float bits.
        bits = tl.atomic_add(
            X.to(tl.pointer_type(tl.int32)) + previous,
            0,
            mask=valid,
            sem="acquire",
        )
        solved = tl.where(valid, bits, 0).to(tl.float32, bitcast=True)
        rhs = tl.load(X + rows) - tl.sum(coefficients * solved[None, :], 1)
        inner_distance = rr[:, None] - rr[None, :]
        inner = tl.load(
            A + rows[:, None] * 257 + inner_distance,
            mask=inner_distance >= 0,
            other=0.0,
        )
        diagonal = tl.load(A + rows * 257)
        inverse = libdevice.fast_dividef(1.0, diagonal)
        rhs *= inverse
        inner *= inverse[:, None]
        for step in tl.static_range(8):
            pivot = tl.full((8,), step, tl.int32)
            column = tl.full((8, 1), step, tl.int32)
            value = tl.gather(rhs, pivot, 0)
            coeff = tl.gather(inner, column, 1).reshape(8)
            rhs = tl.where(rr > step, rhs - coeff * value, rhs)
        tl.store(X + rows, rhs)
        tl.debug_barrier()


@libentry()
@triton.jit
def stbsv_n_panel_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    UPLO: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Solve a small row panel, then merge its trailing vector update."""
    row_offs = tl.arange(0, BLOCK_ROWS)
    update_offs = tl.arange(0, BLOCK_K)
    panel_count = n // BLOCK_ROWS

    if UPLO == 1:
        for panel in tl.range(0, panel_count):
            first = panel * BLOCK_ROWS
            rows = first + row_offs
            values = tl.load(x_ptr + rows)

            for p in tl.static_range(0, BLOCK_ROWS):
                row = first + p
                xp = tl.sum(tl.where(row_offs == p, values, 0.0), axis=0)
                xp = xp / tl.load(a_ptr + k + row * LDA)
                values = tl.where(row_offs == p, xp, values)
                distance = row_offs - p
                mask = distance > 0
                coeff = tl.load(
                    a_ptr + (k - distance) + rows * LDA,
                    mask=mask,
                    other=0.0,
                )
                values -= tl.where(mask, coeff * xp, 0.0)

            tl.store(x_ptr + rows, values)
            targets = first + BLOCK_ROWS + update_offs
            valid_target = targets < n
            contribution = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for p in tl.static_range(0, BLOCK_ROWS):
                source = first + p
                xp = tl.sum(tl.where(row_offs == p, values, 0.0), axis=0)
                distance = targets - source
                mask = valid_target & (distance <= k)
                coeff = tl.load(
                    a_ptr + (k - distance) + targets * LDA,
                    mask=mask,
                    other=0.0,
                )
                contribution += coeff * xp
            current = tl.load(x_ptr + targets, mask=valid_target, other=0.0)
            tl.store(
                x_ptr + targets, current - contribution, mask=valid_target
            )
            tl.debug_barrier()
    else:
        for panel in tl.range(0, panel_count):
            last = n - 1 - panel * BLOCK_ROWS
            rows = last - row_offs
            values = tl.load(x_ptr + rows)

            for p in tl.static_range(0, BLOCK_ROWS):
                row = last - p
                xp = tl.sum(tl.where(row_offs == p, values, 0.0), axis=0)
                xp = xp / tl.load(a_ptr + row * LDA)
                values = tl.where(row_offs == p, xp, values)
                distance = row_offs - p
                mask = distance > 0
                coeff = tl.load(
                    a_ptr + distance + rows * LDA,
                    mask=mask,
                    other=0.0,
                )
                values -= tl.where(mask, coeff * xp, 0.0)

            tl.store(x_ptr + rows, values)
            targets = last - BLOCK_ROWS - update_offs
            valid_target = targets >= 0
            contribution = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for p in tl.static_range(0, BLOCK_ROWS):
                source = last - p
                xp = tl.sum(tl.where(row_offs == p, values, 0.0), axis=0)
                distance = source - targets
                mask = valid_target & (distance <= k)
                coeff = tl.load(
                    a_ptr + distance + targets * LDA,
                    mask=mask,
                    other=0.0,
                )
                contribution += coeff * xp
            current = tl.load(x_ptr + targets, mask=valid_target, other=0.0)
            tl.store(
                x_ptr + targets, current - contribution, mask=valid_target
            )
            tl.debug_barrier()


@triton.jit
def _unpack_complex64(value):
    real = value.to(tl.int32).to(tl.float32, bitcast=True)
    imag = (value >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    return real, imag


@triton.jit
def _complex64_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _complex64_div(xr, xi, ar, ai, FAST_DIV: tl.constexpr):
    denominator = ar * ar + ai * ai
    if FAST_DIV:
        inv_den = libdevice.fast_dividef(1.0, denominator)
    else:
        inv_den = 1.0 / denominator
    return (
        (xr * ar + xi * ai) * inv_den,
        (xi * ar - xr * ai) * inv_den,
    )


@triton.jit
def _pack_complex64(real, imag):
    real_bits = real.to(tl.int32, bitcast=True).to(tl.uint32).to(tl.uint64)
    imag_bits = imag.to(tl.int32, bitcast=True).to(tl.uint32).to(tl.uint64)
    return real_bits | (imag_bits << 32)


@libentry()
@triton.jit
def ctbsv_n256_prefetch_kernel(
    a_ptr,
    x_packed_ptr,
    n,
    LDA,
    UPLO: tl.constexpr,
):
    """Two-row logical no-transpose solve with the next update prefetched."""
    offs = tl.arange(0, 256)
    panel_count = n // 2

    for panel in tl.range(0, panel_count):
        if UPLO == 1:
            j0 = panel * 2
            j1 = j0 + 1
            targets = j0 + 2 + offs
            valid = targets < n
            d0, d1 = targets - j0, targets - j1
            band0, band1 = 256 - d0, 256 - d1
            diagonal_band, inner_band = 256, 255
        else:
            j0 = n - 1 - panel * 2
            j1 = j0 - 1
            targets = j1 - 1 - offs
            valid = targets >= 0
            d0, d1 = j0 - targets, j1 - targets
            band0, band1 = d0, d1
            diagonal_band, inner_band = 0, 1

        # Start the long-latency vector reads before the scalar pair solve.
        a0r, a0i = _unpack_complex64(
            tl.load(
                a_ptr + band0 + targets * LDA,
                mask=valid & (d0 <= 256),
                other=0,
            )
        )
        a1r, a1i = _unpack_complex64(
            tl.load(
                a_ptr + band1 + targets * LDA,
                mask=valid & (d1 <= 256),
                other=0,
            )
        )
        current_r, current_i = _unpack_complex64(
            tl.load(x_packed_ptr + targets, mask=valid, other=0)
        )

        x0r, x0i = _unpack_complex64(tl.load(x_packed_ptr + j0))
        diagonal0r, diagonal0i = _unpack_complex64(
            tl.load(a_ptr + diagonal_band + j0 * LDA)
        )
        x0r, x0i = _complex64_div(
            x0r, x0i, diagonal0r, diagonal0i, True
        )

        x1r, x1i = _unpack_complex64(tl.load(x_packed_ptr + j1))
        inner_r, inner_i = _unpack_complex64(
            tl.load(a_ptr + inner_band + j1 * LDA)
        )
        p10r, p10i = _complex64_mul(inner_r, inner_i, x0r, x0i)
        diagonal1r, diagonal1i = _unpack_complex64(
            tl.load(a_ptr + diagonal_band + j1 * LDA)
        )
        x1r, x1i = _complex64_div(
            x1r - p10r,
            x1i - p10i,
            diagonal1r,
            diagonal1i,
            True,
        )
        tl.store(x_packed_ptr + j0, _pack_complex64(x0r, x0i))
        tl.store(x_packed_ptr + j1, _pack_complex64(x1r, x1i))

        p0r, p0i = _complex64_mul(a0r, a0i, x0r, x0i)
        p1r, p1i = _complex64_mul(a1r, a1i, x1r, x1i)
        tl.store(
            x_packed_ptr + targets,
            _pack_complex64(
                current_r - p0r - p1r,
                current_i - p0i - p1i,
            ),
            mask=valid,
        )
        tl.debug_barrier()


@libentry()
@triton.jit
def ctbsv_n256_tiled_kernel(A, X, n, LOWER: tl.constexpr):
    """Coalesced external update, followed by an eight-row register solve."""
    rows_in_block = tl.arange(0, 8)
    band_offsets = tl.arange(0, 256)
    for block in tl.range(0, n // 8, num_stages=1, disable_licm=True):
        tl.debug_barrier()
        start = block * 8
        if LOWER:
            rows = start + rows_in_block
            previous = start - 256 + band_offsets
        else:
            rows = n - 1 - start - rows_in_block
            previous = n - 1 - (start - 256 + band_offsets)
        previous_valid = (previous >= 0) & (previous < n)
        distance = 256 + rows_in_block[:, None] - band_offsets[None, :]
        bands = 256 - distance if LOWER else distance
        ar, ai = _unpack_complex64(tl.load(
            A + rows[:, None] * 257 + bands,
            mask=previous_valid[None, :] & (distance <= 256), other=0,
        ))
        # CoreX ordinary loads can reuse stale in-place RHS values across
        # blocks. Acquire atomic reads preserve visibility without a flag
        # allocation or an additional output/workspace buffer.
        previous_bits = tl.atomic_add(
            X + previous, 0, mask=previous_valid, sem="acquire",
        )
        xr, xi = _unpack_complex64(tl.where(previous_valid, previous_bits, 0))
        br, bi = _unpack_complex64(tl.load(X + rows))
        br -= tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
        bi -= tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)

        inner_distance = rows_in_block[:, None] - rows_in_block[None, :]
        inner_band = 256 - inner_distance if LOWER else inner_distance
        cr, ci = _unpack_complex64(tl.load(
            A + rows[:, None] * 257 + inner_band,
            mask=inner_distance >= 0, other=0,
        ))
        dr, di = _unpack_complex64(tl.load(A + rows * 257 + (256 if LOWER else 0)))
        inverse = libdevice.fast_dividef(1.0, dr * dr + di * di)
        br, bi = (br * dr + bi * di) * inverse, (bi * dr - br * di) * inverse
        cr, ci = (
            (cr * dr[:, None] + ci * di[:, None]) * inverse[:, None],
            (ci * dr[:, None] - cr * di[:, None]) * inverse[:, None],
        )
        for step in tl.static_range(8):
            # These are indexed selections, not arithmetic reductions. Gather
            # avoids the masked cross-lane sums in every triangular-solve step.
            pivot = tl.full((8,), step, tl.int32)
            column = tl.full((8, 1), step, tl.int32)
            sr = tl.gather(br, pivot, 0)
            si = tl.gather(bi, pivot, 0)
            vr = tl.gather(cr, column, 1).reshape(8)
            vi = tl.gather(ci, column, 1).reshape(8)
            br = tl.where(rows_in_block > step, br - (vr * sr - vi * si), br)
            bi = tl.where(rows_in_block > step, bi - (vr * si + vi * sr), bi)
        tl.store(X + rows, _pack_complex64(br, bi))
        tl.debug_barrier()


@libentry()
@triton.jit
def ctbsv_pair_kernel(
    a_ptr,
    x_packed_ptr,
    x_out_ptr,
    n,
    k,
    LDA,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FAST_DIV: tl.constexpr,
):
    """Explicit packed two-row panel for all complex64 physical modes."""
    offs = tl.arange(0, BLOCK_K)
    panel_count = n // 2
    forward = ((UPLO == 0) and (TRANS == 0)) or (
        (UPLO == 1) and (TRANS == 1)
    )

    for panel in tl.range(0, panel_count):
        if forward:
            j0 = panel * 2
            j1 = j0 + 1
        else:
            j0 = n - 1 - panel * 2
            j1 = j0 - 1

        diag_band = 0 if UPLO == 0 else k
        x0r, x0i = _unpack_complex64(tl.load(x_packed_ptr + j0))
        a0r, a0i = _unpack_complex64(
            tl.load(a_ptr + diag_band + j0 * LDA)
        )
        if CONJ:
            a0i = -a0i
        x0r, x0i = _complex64_div(x0r, x0i, a0r, a0i, FAST_DIV)

        x1r, x1i = _unpack_complex64(tl.load(x_packed_ptr + j1))
        inner_band = 1 if UPLO == 0 else k - 1
        inner_row = j0 if TRANS == 0 else j1
        a10r, a10i = _unpack_complex64(
            tl.load(a_ptr + inner_band + inner_row * LDA)
        )
        if CONJ:
            a10i = -a10i
        p10r, p10i = _complex64_mul(a10r, a10i, x0r, x0i)
        a1r, a1i = _unpack_complex64(
            tl.load(a_ptr + diag_band + j1 * LDA)
        )
        if CONJ:
            a1i = -a1i
        x1r, x1i = _complex64_div(
            x1r - p10r, x1i - p10i, a1r, a1i, FAST_DIV
        )
        tl.store(x_packed_ptr + j0, _pack_complex64(x0r, x0i))
        tl.store(x_packed_ptr + j1, _pack_complex64(x1r, x1i))

        if forward:
            targets = j1 + 1 + offs
            valid = targets < n
            d0, d1 = targets - j0, targets - j1
        else:
            targets = j1 - 1 - offs
            valid = targets >= 0
            d0, d1 = j0 - targets, j1 - targets
        band0 = d0 if UPLO == 0 else k - d0
        band1 = d1 if UPLO == 0 else k - d1
        row0 = j0 if TRANS == 0 else targets
        row1 = j1 if TRANS == 0 else targets
        a0 = tl.load(
            a_ptr + band0 + row0 * LDA,
            mask=valid & (d0 <= k),
            other=0,
        )
        a1 = tl.load(
            a_ptr + band1 + row1 * LDA,
            mask=valid & (d1 <= k),
            other=0,
        )
        a0r, a0i = _unpack_complex64(a0)
        a1r, a1i = _unpack_complex64(a1)
        if CONJ:
            a0i, a1i = -a0i, -a1i
        p0r, p0i = _complex64_mul(a0r, a0i, x0r, x0i)
        p1r, p1i = _complex64_mul(a1r, a1i, x1r, x1i)
        current = tl.load(
            x_packed_ptr + targets, mask=valid, other=0
        )
        current_r, current_i = _unpack_complex64(current)
        tl.store(
            x_packed_ptr + targets,
            _pack_complex64(current_r - p0r - p1r, current_i - p0i - p1i),
            mask=valid,
        )
        tl.debug_barrier()



@libentry()
@triton.jit
def ctbsv_n_panel_kernel(
    a_ptr,
    x_packed_ptr,
    x_out_ptr,
    n,
    k,
    LDA,
    UPLO: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Packed complex64 small-panel solve for row-major no-trans."""
    row_offs = tl.arange(0, BLOCK_ROWS)
    update_offs = tl.arange(0, BLOCK_K)
    panel_count = n // BLOCK_ROWS

    if UPLO == 1:
        for panel in tl.range(0, panel_count):
            first = panel * BLOCK_ROWS
            rows = first + row_offs
            values_r, values_i = _unpack_complex64(
                tl.load(x_packed_ptr + rows)
            )
            for p in tl.static_range(0, BLOCK_ROWS):
                row = first + p
                xr = tl.sum(tl.where(row_offs == p, values_r, 0.0), axis=0)
                xi = tl.sum(tl.where(row_offs == p, values_i, 0.0), axis=0)
                ar, ai = _unpack_complex64(tl.load(a_ptr + k + row * LDA))
                inv_den = 1.0 / (ar * ar + ai * ai)
                xr, xi = (
                    (xr * ar + xi * ai) * inv_den,
                    (xi * ar - xr * ai) * inv_den,
                )
                values_r = tl.where(row_offs == p, xr, values_r)
                values_i = tl.where(row_offs == p, xi, values_i)
                distance = row_offs - p
                mask = distance > 0
                ar, ai = _unpack_complex64(
                    tl.load(
                        a_ptr + (k - distance) + rows * LDA,
                        mask=mask,
                        other=0,
                    )
                )
                values_r -= tl.where(mask, ar * xr - ai * xi, 0.0)
                values_i -= tl.where(mask, ar * xi + ai * xr, 0.0)

            tl.store(x_out_ptr + 2 * rows, values_r)
            tl.store(x_out_ptr + 2 * rows + 1, values_i)
            targets = first + BLOCK_ROWS + update_offs
            valid_target = targets < n
            contribution_r = tl.zeros((BLOCK_K,), dtype=tl.float32)
            contribution_i = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for p in tl.static_range(0, BLOCK_ROWS):
                source = first + p
                xr = tl.sum(tl.where(row_offs == p, values_r, 0.0), axis=0)
                xi = tl.sum(tl.where(row_offs == p, values_i, 0.0), axis=0)
                distance = targets - source
                mask = valid_target & (distance <= k)
                ar, ai = _unpack_complex64(
                    tl.load(
                        a_ptr + (k - distance) + targets * LDA,
                        mask=mask,
                        other=0,
                    )
                )
                contribution_r += ar * xr - ai * xi
                contribution_i += ar * xi + ai * xr
            current_r, current_i = _unpack_complex64(
                tl.load(x_packed_ptr + targets, mask=valid_target, other=0)
            )
            tl.store(
                x_out_ptr + 2 * targets,
                current_r - contribution_r,
                mask=valid_target,
            )
            tl.store(
                x_out_ptr + 2 * targets + 1,
                current_i - contribution_i,
                mask=valid_target,
            )
            tl.debug_barrier()
    else:
        for panel in tl.range(0, panel_count):
            last = n - 1 - panel * BLOCK_ROWS
            rows = last - row_offs
            values_r, values_i = _unpack_complex64(
                tl.load(x_packed_ptr + rows)
            )
            for p in tl.static_range(0, BLOCK_ROWS):
                row = last - p
                xr = tl.sum(tl.where(row_offs == p, values_r, 0.0), axis=0)
                xi = tl.sum(tl.where(row_offs == p, values_i, 0.0), axis=0)
                ar, ai = _unpack_complex64(tl.load(a_ptr + row * LDA))
                inv_den = 1.0 / (ar * ar + ai * ai)
                xr, xi = (
                    (xr * ar + xi * ai) * inv_den,
                    (xi * ar - xr * ai) * inv_den,
                )
                values_r = tl.where(row_offs == p, xr, values_r)
                values_i = tl.where(row_offs == p, xi, values_i)
                distance = row_offs - p
                mask = distance > 0
                ar, ai = _unpack_complex64(
                    tl.load(
                        a_ptr + distance + rows * LDA,
                        mask=mask,
                        other=0,
                    )
                )
                values_r -= tl.where(mask, ar * xr - ai * xi, 0.0)
                values_i -= tl.where(mask, ar * xi + ai * xr, 0.0)

            tl.store(x_out_ptr + 2 * rows, values_r)
            tl.store(x_out_ptr + 2 * rows + 1, values_i)
            targets = last - BLOCK_ROWS - update_offs
            valid_target = targets >= 0
            contribution_r = tl.zeros((BLOCK_K,), dtype=tl.float32)
            contribution_i = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for p in tl.static_range(0, BLOCK_ROWS):
                source = last - p
                xr = tl.sum(tl.where(row_offs == p, values_r, 0.0), axis=0)
                xi = tl.sum(tl.where(row_offs == p, values_i, 0.0), axis=0)
                distance = source - targets
                mask = valid_target & (distance <= k)
                ar, ai = _unpack_complex64(
                    tl.load(
                        a_ptr + distance + targets * LDA,
                        mask=mask,
                        other=0,
                    )
                )
                contribution_r += ar * xr - ai * xi
                contribution_i += ar * xi + ai * xr
            current_r, current_i = _unpack_complex64(
                tl.load(x_packed_ptr + targets, mask=valid_target, other=0)
            )
            tl.store(
                x_out_ptr + 2 * targets,
                current_r - contribution_r,
                mask=valid_target,
            )
            tl.store(
                x_out_ptr + 2 * targets + 1,
                current_i - contribution_i,
                mask=valid_target,
            )
            tl.debug_barrier()


@libentry()
@triton.jit
def ctbsv_packed_kernel(
    a_ptr,
    x_packed_ptr,
    x_out_ptr,
    n,
    k,
    LDA,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_BARRIER: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Single-program complex solve with packed complex64 memory accesses."""
    offs = tl.arange(0, BLOCK_K)
    d = offs + 1

    if (UPLO == 0) and (TRANS == 0):
        for j in tl.range(0, n, loop_unroll_factor=UNROLL):
            xr, xi = _unpack_complex64(tl.load(x_packed_ptr + j))
            ar, ai = _unpack_complex64(tl.load(a_ptr + j * LDA))
            if CONJ:
                ai = -ai
            inv_den = 1.0 / (ar * ar + ai * ai)
            xr, xi = (xr * ar + xi * ai) * inv_den, (xi * ar - xr * ai) * inv_den
            tl.store(x_out_ptr + 2 * j, xr)
            tl.store(x_out_ptr + 2 * j + 1, xi)

            i = j + d
            mask = (d <= k) & (i < n)
            ar, ai = _unpack_complex64(
                tl.load(a_ptr + d + j * LDA, mask=mask, other=0)
            )
            br, bi = _unpack_complex64(
                tl.load(x_packed_ptr + i, mask=mask, other=0)
            )
            if CONJ:
                ai = -ai
            tl.store(x_out_ptr + 2 * i, br - (ar * xr - ai * xi), mask=mask)
            tl.store(x_out_ptr + 2 * i + 1, bi - (ar * xi + ai * xr), mask=mask)
            if USE_BARRIER:
                tl.debug_barrier()

    elif (UPLO == 1) and (TRANS == 0):
        rd = BLOCK_K - offs
        for jc in tl.range(0, n, loop_unroll_factor=UNROLL):
            j = n - 1 - jc
            xr, xi = _unpack_complex64(tl.load(x_packed_ptr + j))
            ar, ai = _unpack_complex64(tl.load(a_ptr + k + j * LDA))
            if CONJ:
                ai = -ai
            inv_den = 1.0 / (ar * ar + ai * ai)
            xr, xi = (xr * ar + xi * ai) * inv_den, (xi * ar - xr * ai) * inv_den
            tl.store(x_out_ptr + 2 * j, xr)
            tl.store(x_out_ptr + 2 * j + 1, xi)

            i = j - rd
            mask = (rd <= k) & (i >= 0)
            ar, ai = _unpack_complex64(
                tl.load(a_ptr + (k - rd) + j * LDA, mask=mask, other=0)
            )
            br, bi = _unpack_complex64(
                tl.load(x_packed_ptr + i, mask=mask, other=0)
            )
            if CONJ:
                ai = -ai
            tl.store(x_out_ptr + 2 * i, br - (ar * xr - ai * xi), mask=mask)
            tl.store(x_out_ptr + 2 * i + 1, bi - (ar * xi + ai * xr), mask=mask)
            if USE_BARRIER:
                tl.debug_barrier()

    elif (UPLO == 1) and (TRANS == 1):
        for j in tl.range(0, n, loop_unroll_factor=UNROLL):
            xr, xi = _unpack_complex64(tl.load(x_packed_ptr + j))
            ar, ai = _unpack_complex64(tl.load(a_ptr + k + j * LDA))
            if CONJ:
                ai = -ai
            inv_den = 1.0 / (ar * ar + ai * ai)
            xr, xi = (xr * ar + xi * ai) * inv_den, (xi * ar - xr * ai) * inv_den
            tl.store(x_out_ptr + 2 * j, xr)
            tl.store(x_out_ptr + 2 * j + 1, xi)

            i = j + d
            mask = (d <= k) & (i < n)
            ar, ai = _unpack_complex64(
                tl.load(a_ptr + (k - d) + i * LDA, mask=mask, other=0)
            )
            br, bi = _unpack_complex64(
                tl.load(x_packed_ptr + i, mask=mask, other=0)
            )
            if CONJ:
                ai = -ai
            tl.store(x_out_ptr + 2 * i, br - (ar * xr - ai * xi), mask=mask)
            tl.store(x_out_ptr + 2 * i + 1, bi - (ar * xi + ai * xr), mask=mask)
            if USE_BARRIER:
                tl.debug_barrier()

    else:
        for jc in tl.range(0, n, loop_unroll_factor=UNROLL):
            j = n - 1 - jc
            xr, xi = _unpack_complex64(tl.load(x_packed_ptr + j))
            ar, ai = _unpack_complex64(tl.load(a_ptr + j * LDA))
            if CONJ:
                ai = -ai
            inv_den = 1.0 / (ar * ar + ai * ai)
            xr, xi = (xr * ar + xi * ai) * inv_den, (xi * ar - xr * ai) * inv_den
            tl.store(x_out_ptr + 2 * j, xr)
            tl.store(x_out_ptr + 2 * j + 1, xi)

            i = j - d
            mask = (d <= k) & (i >= 0)
            ar, ai = _unpack_complex64(
                tl.load(a_ptr + d + i * LDA, mask=mask, other=0)
            )
            br, bi = _unpack_complex64(
                tl.load(x_packed_ptr + i, mask=mask, other=0)
            )
            if CONJ:
                ai = -ai
            tl.store(x_out_ptr + 2 * i, br - (ar * xr - ai * xi), mask=mask)
            tl.store(x_out_ptr + 2 * i + 1, bi - (ar * xi + ai * xr), mask=mask)
            if USE_BARRIER:
                tl.debug_barrier()


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
    return physical_uplo, physical_trans


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
    uplo, trans = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        grid = (1,)
        band_key = _band_bucket(k + 1)
        mode = _mode_key(uplo, trans_flag, unit)
        # The k=256 direct variants are not synchronization-safe on Iluvatar:
        # failures move between the final updates at different n/mode values.
        # Keep this width on the single-program generic path.
        direct_ready = incx == 1 and unit == 0 and lda == k + 1 and k < 256
        upper_n = uplo == CUBLAS_FILL_MODE_UPPER and trans_flag == 0
        lower_n = uplo == CUBLAS_FILL_MODE_LOWER and trans_flag == 0
        upper_t = uplo == CUBLAS_FILL_MODE_UPPER and trans_flag == 1
        lower_t = uplo == CUBLAS_FILL_MODE_LOWER and trans_flag == 1
        full_band_256 = n == 256 and k == 255
        row_major_n_ready = (
            incx == 1
            and unit == 0
            and lda == k + 1
            and trans_flag == 1
        )

        pair_ready = (
            incx == 1
            and unit == 0
            and lda == k + 1
            and k in (255, 256)
            and n % 2 == 0
        )
        if pair_ready and k == 256 and trans_flag == 1 and n % 4 == 0:
            if lower_t and n >= 12288 and n % 8 == 0:
                stbsv_upper_n256_tiled_kernel[grid](
                    A, x, n, num_warps=4, num_stages=1
                )
                return
            quad_warps = (
                2
                if uplo == CUBLAS_FILL_MODE_LOWER and n >= 12288
                else 4
            )
            stbsv_quad_kernel[grid](
                A,
                x,
                n,
                k,
                lda,
                UPLO=uplo,
                BLOCK_K=_band_bucket(k),
                num_warps=quad_warps,
                num_stages=2,
            )
            return
        if pair_ready:
            stbsv_pair_kernel[grid](
                A,
                x,
                n,
                k,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                BLOCK_K=_band_bucket(k),
                FAST_DIV=True,
                num_warps=4,
                num_stages=2,
            )
            return

        if row_major_n_ready and k == 16 and n % 2 == 0:
            stbsv_pair_kernel[grid](
                A,
                x,
                n,
                k,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                BLOCK_K=_band_bucket(k),
                FAST_DIV=True,
                num_warps=1,
                num_stages=2 if n <= 1024 else 1,
            )
            return

        if row_major_n_ready and k in (16, 64):
            if k == 16:
                unroll, num_warps, num_stages = 2, 4, 1
            elif uplo == CUBLAS_FILL_MODE_UPPER:
                unroll, num_warps, num_stages = 16, 2, 2
            else:
                unroll, num_warps, num_stages = 2, 2, 2
            stbsv_n_update_kernel[grid](
                A,
                x,
                n,
                k,
                lda,
                UPLO=uplo,
                BLOCK_K=_band_bucket(k),
                UNROLL=unroll,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return

        if direct_ready:
            if upper_n:
                if k == 16 and n % 2 == 0:
                    stbsv_upper16_pair_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
                if k == 64 and n % 16 == 0:
                    stbsv_upper64_hex_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
                if k == 256 and n % 2 == 0:
                    stbsv_upper256_pair_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
                if full_band_256:
                    stbsv_upper_full_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
            elif lower_n:
                if n == 1024 and k == 64:
                    stbsv_lower64_1024_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
                if n <= 2048 and k == 64:
                    stbsv_lower64_small_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
            elif lower_t:
                if full_band_256:
                    stbsv_lower_trans_full_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
                if k == 16:
                    stbsv_lower_trans16_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
            elif upper_t:
                if full_band_256:
                    stbsv_upper_trans_full_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
                if n == 256 and (k == 16 or k == 64):
                    stbsv_upper_trans_small_kernel[grid](
                        A,
                        x,
                        n,
                        k,
                        lda,
                        incx,
                        band_key,
                        mode,
                    )
                    return
        kernel = (
            stbsv_direct_kernel
            if (
                (k == 1 or k == 4 or k == 16 or k == 64 or k == 256)
                and (upper_n or lower_n or upper_t or lower_t)
                and direct_ready
            )
            else stbsv_kernel
        )
        kernel[grid](
            A,
            x,
            n,
            k,
            lda,
            incx,
            band_key,
            mode,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
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
    """Solve complex64 TBSV with an Iluvatar packed-access fast path."""
    assert A.dtype == torch.complex64 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return
    physical_uplo, physical_trans, conj = _common._row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    with torch_device_fn.device(A.device):
        pair_ready = (
            incx == 1
            and unit == 0
            and lda == k + 1
            and k in (255, 256)
            and n % 2 == 0
        )
        if (
            pair_ready
            and trans == CUBLAS_OP_N
            and k == 256
            and n >= 512
            and n % 8 == 0
        ):
            ctbsv_n256_tiled_kernel[(1,)](
                A.view(torch.int64),
                x.view(torch.int64),
                n,
                LOWER=uplo == CUBLAS_FILL_MODE_LOWER,
                num_warps=4,
                num_stages=1,
            )
            return
        if pair_ready and trans == CUBLAS_OP_N and k == 256:
            ctbsv_n256_prefetch_kernel[(1,)](
                A.view(torch.int64),
                x.view(torch.int64),
                n,
                lda,
                UPLO=physical_uplo,
                num_warps=4,
                num_stages=1,
            )
            return
        if pair_ready:
            physical_trans_flag = int(physical_trans != CUBLAS_OP_N)
            if physical_trans_flag:
                num_warps, num_stages = 4, 2
            else:
                num_warps = 8
                num_stages = (
                    1 if physical_uplo == CUBLAS_FILL_MODE_UPPER else 2
                )
            ctbsv_pair_kernel[(1,)](
                A.view(torch.int64),
                x.view(torch.int64),
                torch.view_as_real(x),
                n,
                k,
                lda,
                UPLO=physical_uplo,
                TRANS=physical_trans_flag,
                CONJ=conj,
                BLOCK_K=_band_bucket(k),
                FAST_DIV=True,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return
        panel_ready = (
            incx == 1
            and unit == 0
            and lda == k + 1
            and trans == CUBLAS_OP_N
            and k == 16
            and n % 2 == 0
        )
        if panel_ready:
            num_warps = 1 if k == 16 else 4
            num_stages = (
                1
                if k == 16 or physical_uplo == CUBLAS_FILL_MODE_UPPER
                else 2
            )
            ctbsv_n_panel_kernel[(1,)](
                A.view(torch.int64),
                x.view(torch.int64),
                torch.view_as_real(x),
                n,
                k,
                lda,
                UPLO=physical_uplo,
                BLOCK_K=_band_bucket(k),
                BLOCK_ROWS=2,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return
        packed_ready = (
            incx == 1
            and unit == 0
            and lda == k + 1
            and k in (1, 4, 16, 64, 255, 256)
        )
        if packed_ready:
            use_barrier = k >= 255
            if k <= 64:
                unroll, num_warps, num_stages = 8, 1, 2 if k == 4 else 1
            elif k == 255:
                unroll, num_warps, num_stages = 4, 4, 2
            elif (
                physical_trans != CUBLAS_OP_N
                and physical_uplo == CUBLAS_FILL_MODE_UPPER
            ):
                unroll, num_warps, num_stages = 8, 4, 2
            elif physical_trans != CUBLAS_OP_N:
                unroll, num_warps, num_stages = 2, 2, 2
            else:
                unroll, num_warps, num_stages = 4, 4, 1
            ctbsv_packed_kernel[(1,)](
                A.view(torch.int64),
                x.view(torch.int64),
                torch.view_as_real(x),
                n,
                k,
                lda,
                UPLO=physical_uplo,
                TRANS=int(physical_trans != CUBLAS_OP_N),
                CONJ=conj,
                BLOCK_K=_band_bucket(k),
                USE_BARRIER=use_barrier,
                UNROLL=unroll,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return
        _common._complex_tbsv_kernel[(1,)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            n,
            k,
            lda,
            incx,
            UPLO=physical_uplo,
            TRANS=physical_trans,
            UNIT=unit,
            CONJ=conj,
        )

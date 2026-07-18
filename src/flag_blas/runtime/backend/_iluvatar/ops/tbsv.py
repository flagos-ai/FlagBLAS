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
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    with torch_device_fn.device(A.device):
        grid = (1,)
        band_key = _band_bucket(k + 1)
        mode = _mode_key(uplo, trans_flag, unit)
        direct_ready = incx == 1 and unit == 0 and lda == k + 1
        upper_n = uplo == CUBLAS_FILL_MODE_UPPER and trans_flag == 0
        lower_n = uplo == CUBLAS_FILL_MODE_LOWER and trans_flag == 0
        upper_t = uplo == CUBLAS_FILL_MODE_UPPER and trans_flag == 1
        lower_t = uplo == CUBLAS_FILL_MODE_LOWER and trans_flag == 1
        full_band_256 = n == 256 and k == 255

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

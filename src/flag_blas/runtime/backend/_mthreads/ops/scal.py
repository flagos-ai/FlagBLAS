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

from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = import_module("flag_blas.ops.level1.scal")


@libentry()
@triton.jit
def _complex_update(
    X,
    Y,
    N,
    AR: tl.float32,
    AI: tl.float32,
    AXPY: tl.constexpr,
    REAL: tl.constexpr,
    BLOCK: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK * 2
    lane = tl.arange(0, BLOCK * 2)
    step = tl.num_programs(0) * BLOCK * 2
    if PERSISTENT:
        while start < N * 2:
            index = start + lane
            value = tl.load(X + index, index < N * 2, other=0.0)
            if REAL:
                result = value * AR
            else:
                xr, xi = tl.split(tl.reshape(value, (BLOCK, 2)))
                rr = AR * xr - AI * xi
                ri = AR * xi + AI * xr
                result = tl.reshape(tl.join(rr, ri), (BLOCK * 2,))
            if AXPY:
                result += tl.load(Y + index, index < N * 2, other=0.0)
            tl.store(Y + index, result, index < N * 2)
            start += step
    else:
        index = start + lane
        value = tl.load(X + index, index < N * 2, other=0.0)
        if REAL:
            result = value * AR
        else:
            xr, xi = tl.split(tl.reshape(value, (BLOCK, 2)))
            rr = AR * xr - AI * xi
            ri = AR * xi + AI * xr
            result = tl.reshape(tl.join(rr, ri), (BLOCK * 2,))
        if AXPY:
            result += tl.load(Y + index, index < N * 2, other=0.0)
        tl.store(Y + index, result, index < N * 2)


def _run_scale(n, alpha, x, incx, real):
    original = _common.csscal if real else _common.cscal
    if (
        n <= 8192
        or incx != 1
        or x.dtype != torch.complex64
        or x.dim() != 1
        or not x.is_contiguous()
        or x.numel() < n
    ):
        return original(n, alpha, x, incx)
    value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    if real and isinstance(value, complex):
        return original(n, alpha, x, incx)
    a = complex(value)
    raw = torch.view_as_real(x)
    with torch_device_fn.device(x.device):
        _complex_update[(triton.cdiv(n, 512),)](
            raw,
            raw,
            n,
            a.real,
            a.imag,
            False,
            real,
            512,
            False,
            num_warps=4,
            num_stages=1,
        )


def cscal(n, alpha, x, incx):
    return _run_scale(n, alpha, x, incx, False)


def csscal(n, alpha, x, incx):
    return _run_scale(n, alpha, x, incx, True)

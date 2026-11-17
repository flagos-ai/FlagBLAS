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

from .rotg import _abs_key

_rotmg = import_module("flag_blas.ops.level1.rotmg")


@triton.jit
def _negative_double(x):
    bits = x.to(tl.uint64, bitcast=True)
    magnitude = bits & 0x7FFFFFFFFFFFFFFF
    return (
        ((bits & 0x8000000000000000) != 0)
        & (magnitude != 0)
        & (magnitude <= 0x7FF0000000000000)
    )


@libentry()
@triton.jit
def _drotmg_kernel(D1, D2, X1, Y1, P):
    d1 = tl.load(D1)
    d2 = tl.load(D2)
    x1 = tl.load(X1)
    y1 = tl.load(Y1)
    zero = tl.full((), 0.0, tl.float64)
    one = tl.full((), 1.0, tl.float64)
    gam = tl.full((), 4096.0, tl.float64)
    gamsq = tl.full((), 16777216.0, tl.float64)
    flag = tl.full((), -2, tl.int32)
    h11 = zero
    h12 = zero
    h21 = zero
    h22 = zero
    if _negative_double(d1):
        flag = -1
        d1 = zero
        d2 = zero
        x1 = zero
    else:
        p2 = d2 * y1
        if _abs_key(p2) != 0:
            p1 = d1 * x1
            q2 = p2 * y1
            q1 = p1 * x1
            if _abs_key(q1) > _abs_key(q2):
                h21 = -y1 / x1
                h12 = p2 / p1
                u = one - h12 * h21
                if (
                    (~_negative_double(u))
                    & (_abs_key(u) > 0)
                    & (_abs_key(u) <= 0x7FF0000000000000)
                ):
                    flag = 0
                    inverse_u = one / u
                    d1 = d1 * inverse_u
                    d2 = d2 * inverse_u
                    x1 = x1 * u
                else:
                    flag = -1
                    d1 = zero
                    d2 = zero
                    x1 = zero
            else:
                if _negative_double(q2):
                    flag = -1
                    d1 = zero
                    d2 = zero
                    x1 = zero
                else:
                    flag = 1
                    h11 = p1 / p2
                    h22 = x1 / y1
                    u = one + h11 * h22
                    inverse_u = one / u
                    temp = d2 * inverse_u
                    d2 = d1 * inverse_u
                    d1 = temp
                    x1 = y1 * u
            if flag >= 0:
                if _abs_key(d1) != 0:
                    while (_abs_key(d1) <= 0x3E70000000000000) | (
                        _abs_key(d1) >= 0x4170000000000000
                    ):
                        if flag == 0:
                            h11 = one
                            h22 = one
                        elif flag == 1:
                            h21 = -one
                            h12 = one
                        flag = -1
                        if _abs_key(d1) <= 0x3E70000000000000:
                            d1 = d1 * gamsq
                            x1 = x1 / gam
                            h11 = h11 / gam
                            h12 = h12 / gam
                        else:
                            d1 = d1 / gamsq
                            x1 = x1 * gam
                            h11 = h11 * gam
                            h12 = h12 * gam
                if _abs_key(d2) != 0:
                    while (_abs_key(d2) <= 0x3E70000000000000) | (
                        _abs_key(d2) >= 0x4170000000000000
                    ):
                        if flag == 0:
                            h11 = one
                            h22 = one
                        elif flag == 1:
                            h21 = -one
                            h12 = one
                        flag = -1
                        if _abs_key(d2) <= 0x3E70000000000000:
                            d2 = d2 * gamsq
                            h21 = h21 / gam
                            h22 = h22 / gam
                        else:
                            d2 = d2 / gamsq
                            h21 = h21 * gam
                            h22 = h22 * gam
    flag_value = tl.where(
        flag == -2, -2 * one, tl.where(flag == -1, -one, tl.where(flag == 0, zero, one))
    )
    if flag != -2:
        tl.store(D1, d1)
        tl.store(D2, d2)
        tl.store(X1, x1)
    tl.store(P, flag_value)
    if flag < 0:
        tl.store(P + 1, h11)
        tl.store(P + 2, h21)
        tl.store(P + 3, h12)
        tl.store(P + 4, h22)
    elif flag == 0:
        tl.store(P + 1, zero)
        tl.store(P + 2, h21)
        tl.store(P + 3, h12)
        tl.store(P + 4, zero)
    else:
        tl.store(P + 1, h11)
        tl.store(P + 2, zero)
        tl.store(P + 3, zero)
        tl.store(P + 4, h22)


def drotmg(d1, d2, x1, y1, param):
    assert d1.dtype == torch.float64, "d1 must be float64"
    _rotmg._validate_rotmg_inputs(d1, d2, x1, y1, param)
    with torch_device_fn.device(d1.device):
        _drotmg_kernel[(1,)](d1, d2, x1, y1, param, num_warps=1, num_stages=1)

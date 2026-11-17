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

_rotg = import_module("flag_blas.ops.level1.rotg")


@triton.jit
def _abs_key(x):
    # Positive IEEE encodings preserve magnitude order without FP64 comparisons.
    if x.dtype == tl.float64:
        return x.to(tl.uint64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
    else:
        return x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF


@libentry()
@triton.jit
def _real_rotg_kernel(A, B, C, S):
    b = tl.load(B)
    kb = _abs_key(b)
    zero = tl.full((), 0.0, b.dtype)
    one = tl.full((), 1.0, b.dtype)
    if kb == 0:
        tl.store(B, zero)
        tl.store(C, one)
        tl.store(S, zero)
    else:
        a = tl.load(A)
        ka = _abs_key(a)
        if ka > kb:
            ratio = b / a
            h = tl.sqrt(one + ratio * ratio)
            c = one / h
            value = ratio * c
            tl.store(A, a * h)
            tl.store(B, value)
            tl.store(C, c)
            tl.store(S, value)
        else:
            ratio = a / b
            h = tl.sqrt(one + ratio * ratio)
            value = one / h
            c = ratio * value
            z = one
            if _abs_key(c) != 0:
                z = one / c
            tl.store(A, b * h)
            tl.store(B, z)
            tl.store(C, c)
            tl.store(S, value)


def _launch_rotg(a, b, c, s, dtype):
    for name, tensor in (("a", a), ("b", b), ("c", c), ("s", s)):
        assert tensor.dtype == dtype, f"{name} must be {str(dtype).split('.')[-1]}"
    _rotg._validate_rotg_inputs(a, b, c, s)
    with torch_device_fn.device(a.device):
        _real_rotg_kernel[(1,)](a, b, c, s, num_warps=1, num_stages=1)


def srotg(a, b, c, s):
    return _launch_rotg(a, b, c, s, torch.float32)


def drotg(a, b, c, s):
    return _launch_rotg(a, b, c, s, torch.float64)

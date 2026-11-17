# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
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


@libentry()
@triton.jit
def _real_rotg_kernel(A, B, C, S):
    # Only one thread owns the four scalar BLAS outputs.
    tid = tl.inline_asm_elementwise(
        "ppu.mov.u32 $0, %tid.x;", "=r", [], dtype=tl.int32, is_pure=True, pack=1
    )
    if tid == 0:
        b = tl.load(B)
        if b == 0:
            tl.store(C, 1.0, cache_modifier=".cg")
            tl.store(S, 0.0, cache_modifier=".cg")
            return
        a = tl.load(A)
        if a == 0:
            tl.store(A, b, cache_modifier=".cg")
            tl.store(B, 1.0, cache_modifier=".cg")
            tl.store(C, 0.0, cache_modifier=".cg")
            tl.store(S, 1.0, cache_modifier=".cg")
            return
        abs_a, abs_b = tl.abs(a), tl.abs(b)
        # Scaling keeps the squared norm in range for both precisions.
        scale = tl.maximum(abs_a, abs_b)
        x, y = a / scale, b / scale
        norm = tl.sqrt(x * x + y * y)
        roe = tl.where(abs_a > abs_b, a, b)
        sign = tl.where(roe < 0, -1.0, 1.0)
        inv_norm = sign / norm
        c, s = x * inv_norm, y * inv_norm
        r = sign * scale * norm
        z = tl.full((), 1.0, A.dtype.element_ty)
        if abs_a > abs_b:
            z = s
        elif c != 0:
            z = 1.0 / c
        tl.store(A, r, cache_modifier=".cg")
        tl.store(B, z, cache_modifier=".cg")
        tl.store(C, c, cache_modifier=".cg")
        tl.store(S, s, cache_modifier=".cg")


# Keep the nonzero calculation out of line so the scalar fast path stays compact.
@triton.jit(noinline=True)
def _srotg_nonzero(A, B, C, S, b):
    a = tl.load(A)
    if a == 0:
        tl.store(A, b, cache_modifier=".cg")
        tl.store(B, 1.0, cache_modifier=".cg")
        tl.store(C, 0.0, cache_modifier=".cg")
        tl.store(S, 1.0, cache_modifier=".cg")
        return
    abs_a, abs_b = tl.abs(a), tl.abs(b)
    scale = tl.maximum(abs_a, abs_b)
    x, y = a / scale, b / scale
    norm = tl.sqrt(x * x + y * y)
    roe = tl.where(abs_a > abs_b, a, b)
    sign = tl.where(roe < 0, -1.0, 1.0)
    inv_norm = sign / norm
    c, s = x * inv_norm, y * inv_norm
    r = sign * scale * norm
    z = tl.full((), 1.0, A.dtype.element_ty)
    if abs_a > abs_b:
        z = s
    elif c != 0:
        z = 1.0 / c
    tl.store(A, r, cache_modifier=".cg")
    tl.store(B, z, cache_modifier=".cg")
    tl.store(C, c, cache_modifier=".cg")
    tl.store(S, s, cache_modifier=".cg")


@libentry()
@triton.jit
def _srotg_kernel(A, B, C, S):
    tid = tl.inline_asm_elementwise(
        "ppu.mov.u32 $0, %tid.x;", "=r", [], dtype=tl.int32, is_pure=True, pack=1
    )
    if tid == 0:
        b = tl.load(B)
        if b == 0:
            tl.store(C, 1.0, cache_modifier=".cg")
            tl.store(S, 0.0, cache_modifier=".cg")
            return
        _srotg_nonzero(A, B, C, S, b)


@libentry()
@triton.jit
def _zrotg_kernel(A, B, C, S):
    ar, ai = tl.load(A), tl.load(A + 1)
    br, bi = tl.load(B), tl.load(B + 1)
    az = tl.maximum(tl.abs(ar), tl.abs(ai))
    bz = tl.maximum(tl.abs(br), tl.abs(bi))
    if az == 0:
        tl.store(C, 0.0)
        tl.store(S, 1.0)
        tl.store(S + 1, 0.0)
        tl.store(A, br)
        tl.store(A + 1, bi)
        return
    if bz == 0:
        tl.store(C, 1.0)
        tl.store(S, 0.0)
        tl.store(S + 1, 0.0)
        return
    # Normalize a independently so its phase survives an extreme |a|/|b|.
    anr, ani = ar / az, ai / az
    anorm = tl.sqrt(anr * anr + ani * ani)
    phase_r, phase_i = anr / anorm, ani / anorm
    scale = tl.maximum(az, bz)
    a_scaled = (az / scale) * anorm
    bnr, bni = br / scale, bi / scale
    norm = tl.sqrt(a_scaled * a_scaled + bnr * bnr + bni * bni)
    inv_norm = 1.0 / norm
    tl.store(C, a_scaled * inv_norm)
    tl.store(S, (phase_r * bnr + phase_i * bni) * inv_norm)
    tl.store(S + 1, (phase_i * bnr - phase_r * bni) * inv_norm)
    magnitude = scale * norm
    tl.store(A, phase_r * magnitude)
    tl.store(A + 1, phase_i * magnitude)


_common_rotg = import_module("flag_blas.ops.level1.rotg")


def srotg(a, b, c, s):
    assert a.dtype == torch.float32, "a must be float32"
    assert b.dtype == torch.float32, "b must be float32"
    assert c.dtype == torch.float32, "c must be float32"
    assert s.dtype == torch.float32, "s must be float32"
    _common_rotg._validate_rotg_inputs(a, b, c, s)
    with torch_device_fn.device(a.device):
        _srotg_kernel[(1,)](
            a, b, c, s, num_warps=1, num_stages=1, enable_fp_fusion=False
        )


def drotg(a, b, c, s):
    assert a.dtype == torch.float64, "a must be float64"
    assert b.dtype == torch.float64, "b must be float64"
    assert c.dtype == torch.float64, "c must be float64"
    assert s.dtype == torch.float64, "s must be float64"
    _common_rotg._validate_rotg_inputs(a, b, c, s)
    with torch_device_fn.device(a.device):
        _real_rotg_kernel[(1,)](
            a, b, c, s, num_warps=1, num_stages=1, enable_fp_fusion=False
        )


def zrotg(a, b, c, s):
    assert a.dtype == torch.complex128, "a must be complex128"
    assert b.dtype == torch.complex128, "b must be complex128"
    assert c.dtype == torch.float64, "c must be float64"
    assert s.dtype == torch.complex128, "s must be complex128"
    _common_rotg._validate_rotg_inputs(a, b, c, s)
    with torch_device_fn.device(a.device):
        _zrotg_kernel[(1,)](
            torch.view_as_real(a),
            torch.view_as_real(b),
            c,
            torch.view_as_real(s),
            num_warps=2,
            num_stages=1,
            enable_fp_fusion=False,
        )

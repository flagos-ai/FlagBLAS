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

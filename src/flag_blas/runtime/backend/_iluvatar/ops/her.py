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

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


_common = importlib.import_module("flag_blas.ops.level2.her")


@libentry()
@triton.jit
def _cher_kernel(
    x,
    A,
    n: tl.constexpr,
    alpha,
    incx: tl.constexpr,
    lda: tl.constexpr,
    uplo: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # The device sqrt estimate may round across an exact triangular-number
    # boundary. Correct the estimate with integer bounds before deriving the
    # minor tile coordinate, otherwise one 16x16 tile can be skipped.
    major = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    major_start = major * (major + 1) // 2
    major = tl.where(pid < major_start, major - 1, major)
    next_major_start = (major + 1) * (major + 2) // 2
    major = tl.where(pid >= next_major_start, major + 1, major)
    minor = pid - major * (major + 1) // 2

    if uplo == 0:
        pid_m = major
        pid_n = minor
    else:
        pid_m = minor
        pid_n = major

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < n) & (offs_n[None, :] < n)
    if uplo == 1:
        mask = mask & (offs_m[:, None] <= offs_n[None, :])
    else:
        mask = mask & (offs_m[:, None] >= offs_n[None, :])

    x_m = tl.load(x + offs_m * incx * 2, mask=offs_m < n, other=0.0)
    y_m = tl.load(x + offs_m * incx * 2 + 1, mask=offs_m < n, other=0.0)
    x_n = tl.load(x + offs_n * incx * 2, mask=offs_n < n, other=0.0)
    y_n = tl.load(x + offs_n * incx * 2 + 1, mask=offs_n < n, other=0.0)

    prod_r = x_m[:, None] * x_n[None, :] + y_m[:, None] * y_n[None, :]
    prod_i = x_m[:, None] * y_n[None, :] - y_m[:, None] * x_n[None, :]

    a_off = (offs_m[:, None] + offs_n[None, :] * lda) * 2
    old_r = tl.load(A + a_off, mask=mask, other=0.0)
    old_i = tl.load(A + a_off + 1, mask=mask, other=0.0)
    out_r = old_r + alpha * prod_r
    out_i = old_i + alpha * prod_i
    diag = offs_m[:, None] == offs_n[None, :]
    out_i = tl.where(diag, 0.0, out_i)

    tl.store(A + a_off, out_r, mask=mask)
    tl.store(A + a_off + 1, out_i, mask=mask)


def cher(uplo, n, alpha, x, incx, A, lda):
    _common._check_her_args(
        "cher", uplo, n, alpha, x, incx, A, lda, torch.complex64, torch.float32
    )
    if n == 0:
        return A

    physical_uplo = _common._row_major_uplo(uplo)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    x_real = torch.view_as_real(x).reshape(-1)
    A_real = torch.view_as_real(A).reshape(-1)
    block_m = 16
    tile_count = triton.cdiv(n, block_m)
    grid = (tile_count * (tile_count + 1) // 2,)
    with torch_device_fn.device(A.device):
        _cher_kernel[grid](
            x_real,
            A_real,
            n,
            alpha_value,
            incx,
            lda,
            physical_uplo,
            BLOCK_M=block_m,
            BLOCK_N=block_m,
        )
    return A

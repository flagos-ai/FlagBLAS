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

_common = importlib.import_module("flag_blas.ops.level2.tpmv")


@libentry()
@triton.jit
def _ctpmv_upper_conj_partial_kernel(
    AP,
    X,
    P,
    n,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    kk = tl.arange(0, BLOCK_K)
    split = tl.program_id(1)
    width = tl.cdiv(n, SPLIT_K * BLOCK_K) * BLOCK_K
    start = split * width
    end = tl.minimum(start + width, tl.minimum(n, (tl.program_id(0) + 1) * BLOCK_M))
    acc_r = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for begin in tl.range(start, end, BLOCK_K):
        cols = begin + kk
        c = cols.to(tl.int64)
        # Row-major upper packed equals column-major lower packed. Taking
        # its conjugate gives the requested conjugate-transpose product.
        offset = c[None, :] * n - c[None, :] * (c[None, :] + 1) // 2 + rows[:, None]
        mask = (
            (rows[:, None] < n) & (cols[None, :] < n) & (cols[None, :] <= rows[:, None])
        )
        value = tl.load(AP + offset, mask=mask, other=0)
        ar = value.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (value >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        xr = tl.load(X + 2 * cols, mask=cols < n, other=0.0)
        xi = tl.load(X + 2 * cols + 1, mask=cols < n, other=0.0)
        # Accumulate independently along K lanes; reduce only after the loop.
        acc_r += ar * xr[None, :] + ai * xi[None, :]
        acc_i += ar * xi[None, :] - ai * xr[None, :]
    tl.store(P + 2 * (split * n + rows), tl.sum(acc_r, 1), mask=rows < n)
    tl.store(P + 2 * (split * n + rows) + 1, tl.sum(acc_i, 1), mask=rows < n)


def ctpmv(uplo, trans, diag, n, AP, x, incx):
    # Only specialize the large upper/conjugate-transpose, nonunit path.
    # Keep exactly the common split count and temporary buffer sizes.
    if uplo != 1 or trans != 2 or diag != 0 or incx != 1 or n < 4096:
        return _common.ctpmv(uplo, trans, diag, n, AP, x, incx)
    split_k = _common._ctpmv_split_k(n, 0)
    if split_k <= 1:
        return _common.ctpmv(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype == torch.complex64 == x.dtype
    _common._check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=True)
    with torch_device_fn.device(AP.device):
        xin = x.as_strided((n,), (incx,)).clone()
        partial = torch.empty((split_k, n, 2), dtype=torch.float32, device=AP.device)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)
        _ctpmv_upper_conj_partial_kernel[(triton.cdiv(n, 64), split_k)](
            AP.view(torch.int64),
            xin_real,
            partial,
            n,
            SPLIT_K=split_k,
            BLOCK_M=64,
            BLOCK_K=64,
            num_warps=8,
            num_stages=1,
        )
        _common.ctpmv_reduce_kernel[(triton.cdiv(n, 1024),)](
            partial,
            xin_real,
            x_real,
            n,
            incx,
            SPLIT_K=split_k,
            UNIT=0,
            BLOCK_N=1024,
            num_warps=4,
        )

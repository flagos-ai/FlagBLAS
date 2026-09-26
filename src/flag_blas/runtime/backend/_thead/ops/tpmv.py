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
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_common = importlib.import_module("flag_blas.ops.level2.tpmv")


def _prune_dtpmv_configs(configs, named_args, **kwargs):
    trans = named_args["TRANS"]
    return [
        config
        for config in configs
        if (config.kwargs["BLOCK_SIZE_M"] < 16) == (trans == 1)
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_dtpmv"),
    key=["n", "INCX", "UPLO", "TRANS", "UNIT"],
    restore_value=["x_ptr"],
    prune_configs_by={"early_config_prune": _prune_dtpmv_configs},
)
@triton.jit
def dtpmv_thead_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_K)
    index_type: tl.constexpr = tl.int32 if n <= 32768 else tl.int64
    row_offsets = rows.to(index_type)
    # Visit only reduction tiles that intersect this triangular row block.
    if UPLO == TRANS:
        lo = 0
        hi = tl.minimum(n, row_start + BLOCK_SIZE_M)
    else:
        lo = row_start // BLOCK_K * BLOCK_K
        hi = n
    if TRANS == 1:
        if UPLO == 1:
            row_base = row_offsets * (row_offsets + 1) // 2
        else:
            row_base = row_offsets * n - row_offsets * (row_offsets + 1) // 2
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_K), tl.float64)
    else:
        # Packed columns are contiguous along the output rows.
        acc = tl.zeros((BLOCK_K, BLOCK_SIZE_M), tl.float64)
    for kb in tl.range(lo, hi, BLOCK_K):
        cols = kb + offs_k
        col_offsets = cols.to(index_type)
        if TRANS == 0:
            if UPLO == 1:
                col_base = col_offsets * (col_offsets + 1) // 2
                tri = cols[:, None] >= rows[None, :]
            else:
                col_base = col_offsets * n - col_offsets * (col_offsets + 1) // 2
                tri = cols[:, None] <= rows[None, :]
            offsets = col_base[:, None] + row_offsets[None, :]
            if UNIT:
                tri = tri & (cols[:, None] != rows[None, :])
            mask = (rows[None, :] < n) & (cols[:, None] < n) & tri
            matrix = tl.load(ap_ptr + offsets, mask=mask, other=0.0)
            x_vals = tl.load(xin_ptr + cols, mask=cols < n, other=0.0)
            acc = tl.fma(matrix, x_vals[:, None], acc)
        else:
            offsets = row_base[:, None] + col_offsets[None, :]
            if UPLO == 1:
                tri = cols[None, :] <= rows[:, None]
            else:
                tri = cols[None, :] >= rows[:, None]
            if UNIT:
                tri = tri & (cols[None, :] != rows[:, None])
            mask = (rows[:, None] < n) & (cols[None, :] < n) & tri
            matrix = tl.load(ap_ptr + offsets, mask=mask, other=0.0)
            x_vals = tl.load(xin_ptr + cols, mask=cols < n, other=0.0)
            acc = tl.fma(matrix, x_vals[None, :], acc)
    if TRANS == 0:
        output = tl.sum(acc, axis=0)
    else:
        output = tl.sum(acc, axis=1)
    if UNIT:
        output += tl.load(xin_ptr + rows, mask=rows < n, other=0.0)
    tl.store(x_ptr + rows * INCX, output, mask=rows < n)


def dtpmv(uplo, trans, diag, n, AP, x, incx):
    assert AP.dtype == torch.float64 == x.dtype
    _common._check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=False)
    if n == 0:
        return
    if n <= _common._TPMV_SMALL_N:
        return _common.dtpmv(uplo, trans, diag, n, AP, x, incx)
    uplo, trans, _ = _common._row_major_tpmv_args(uplo, trans)
    with torch_device_fn.device(AP.device):
        # Preserve the original vector for the in-place triangular product.
        xin = x.as_strided((n,), (incx,)).clone()
        dtpmv_thead_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)](
            AP,
            xin,
            x,
            n,
            incx,
            uplo,
            trans,
            diag,
        )

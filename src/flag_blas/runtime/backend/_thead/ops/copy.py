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

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


@libentry()
@triton.jit
def _zcopy_bits(
    X,
    Y,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + (i // 4) * IX * 4 + i % 4, i < N * 4, other=0)
    tl.store(Y + (i // 4) * IY * 4 + i % 4, value, i < N * 4)


_common_copy = import_module("flag_blas.ops.level1.copy")
_ZCOPY_PIPELINE_CONFIG = runtime.get_tuned_config("thead_l1_zcopy_pipeline")[0]


def zcopy(n, x, incx, y, incy):
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    common = _common_copy
    if n < 1048576 or (n >= 8388608 and (incx != 1 or incy != 1)):
        return common._copy_impl(n, x, incx, y, incy, True)
    common._validate_copy_inputs(n, x, incx, y, incy)
    contiguous = incx == 1 and incy == 1
    if 2097152 <= n < 8388608 and not contiguous:
        config = _ZCOPY_PIPELINE_CONFIG
        block = config.kwargs["BLOCK"]
        grid = (min(triton.cdiv(n * 4, block), config.kwargs["CAP"]),)
        with torch_device_fn.device(x.device):
            _ccopy_bits[grid](
                torch.view_as_real(x).view(torch.int32),
                torch.view_as_real(y).view(torch.int32),
                n,
                incx,
                incy,
                4,
                block,
                config.kwargs["CHUNK"],
                PIPE_STAGES=config.num_stages,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        return
    block, warps = (4096, 4) if contiguous and n < 8388608 else (16384, 8)
    with torch_device_fn.device(x.device):
        _zcopy_bits[(triton.cdiv(n * 4, block),)](
            # Reinterpret storage without converting or rounding FP64 values.
            torch.view_as_real(x).view(torch.int32),
            torch.view_as_real(y).view(torch.int32),
            n,
            incx,
            incy,
            block,
            num_warps=warps,
            num_stages=1,
        )


@libentry()
@triton.jit
def _ccopy_bits(
    X,
    Y,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    tiles = tl.cdiv(N * PARTS, BLOCK)
    if CHUNK:
        chunk = tl.cdiv(tiles, grid) | 1
        start, end, step = pid * chunk, tl.minimum((pid + 1) * chunk, tiles), 1
    else:
        start, end, step = pid, tiles, grid
    for tile in tl.range(
        start, end, step, num_stages=PIPE_STAGES, loop_unroll_factor=1
    ):
        i = tile * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(X + (i // PARTS) * IX * PARTS + i % PARTS, i < N * PARTS, other=0)
        tl.store(Y + (i // PARTS) * IY * PARTS + i % PARTS, v, i < N * PARTS)


_ccopy_contiguous = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("thead_l1_ccopy_contiguous"),
        key=["N"],
        warmup=20,
        rep=50,
    )(_ccopy_bits.jit_function)
)


@libentry()
@triton.jit
def _ccopy_batch(
    X,
    Y,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    BLOCK: tl.constexpr,
    BATCH: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    groups = tl.arange(0, BATCH)
    lane = tl.arange(0, BLOCK)
    pid, grid = tl.program_id(0), tl.num_programs(0)
    tiles = tl.cdiv(N * 2, BLOCK * BATCH)
    chunk = tl.cdiv(tiles, grid) | 1
    begin, end = pid * chunk, tl.minimum((pid + 1) * chunk, tiles)
    for tile in tl.range(begin, end, num_stages=PIPE_STAGES, loop_unroll_factor=1):
        j = tile * BLOCK * BATCH + groups[:, None] * BLOCK + lane[None, :]
        value = tl.load(X + (j // 2) * IX * 2 + j % 2, j < N * 2, other=0)
        tl.store(Y + (j // 2) * IY * 2 + j % 2, value, j < N * 2)


_CCOPY_BATCH_CONFIG = runtime.get_tuned_config("thead_l1_ccopy_batch")[0]
_CCOPY_PIPELINE_CONFIG = runtime.get_tuned_config("thead_l1_ccopy_pipeline")[0]


def ccopy(n, x, incx, y, incy):
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    common = _common_copy
    if n < 1048576 or n >= 8388608:
        return common._copy_impl(n, x, incx, y, incy, True)
    common._validate_copy_inputs(n, x, incx, y, incy)
    contiguous = incx == incy == 1
    if n >= 2097152 and incx > 1 and incy == 2:
        config = _CCOPY_PIPELINE_CONFIG
        block, cap = config.kwargs["BLOCK"], config.kwargs["CAP"]
        with torch_device_fn.device(x.device):
            _ccopy_bits[(min(triton.cdiv(n * 2, block), cap),)](
                torch.view_as_real(x).view(torch.int32),
                torch.view_as_real(y).view(torch.int32),
                n,
                incx,
                incy,
                2,
                block,
                True,
                PIPE_STAGES=config.num_stages,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        return
    if n >= 2097152 and incx > 1 and incy > 1:
        config = _CCOPY_BATCH_CONFIG
        block, batch = config.kwargs["BLOCK"], config.kwargs["BATCH"]
        grid = (min(triton.cdiv(n * 2, block * batch), config.kwargs["CAP"]),)
        with torch_device_fn.device(x.device):
            _ccopy_batch[grid](
                torch.view_as_real(x).view(torch.int32),
                torch.view_as_real(y).view(torch.int32),
                n,
                incx,
                incy,
                block,
                batch,
                PIPE_STAGES=config.num_stages,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        return
    block = 2048 if contiguous else 512
    grid = triton.cdiv(n, block) if contiguous else min(triton.cdiv(n, block), 128)
    with torch_device_fn.device(x.device):
        if contiguous:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
            _ccopy_contiguous[grid](
                torch.view_as_real(x).view(torch.int64),
                torch.view_as_real(y).view(torch.int64),
                n,
                incx,
                incy,
                1,
                CHUNK=False,
            )
            return
        _ccopy_bits[(grid,)](
            torch.view_as_real(x).view(torch.int64),
            torch.view_as_real(y).view(torch.int64),
            n,
            incx,
            incy,
            1,
            block,
            not contiguous,
            num_warps=4,
            num_stages=1,
        )

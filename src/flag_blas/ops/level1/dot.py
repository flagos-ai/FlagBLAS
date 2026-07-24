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

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

DDOT_FAST_MAX_N = 300000
DDOT_ATOMIC_MAX_N = 65536
DDOT_STAGE1_BLOCK_SIZE = 2048


@libentry()
@triton.jit
def dot_kernel_stage1(
    x_ptr,
    y_ptr,
    partial_ptr,
    n,
    incx,
    incy,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * incx
    y_offset = idx * incy

    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0, cache_modifier=".cg")
    y = tl.load(y_ptr + y_offset, mask=mask, other=0.0, cache_modifier=".cg")

    dot = (x * y).to(DTYPE)
    block_sum = tl.sum(dot, axis=0)
    tl.store(partial_ptr + pid, block_sum)


@libentry()
@triton.jit
def dot_kernel_reduce(
    partial_ptr,
    out_ptr,
    partial_size,
    BLOCK_PARTIAL: tl.constexpr,
    DTYPE: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_PARTIAL)
    mask = offset < partial_size
    val = tl.load(partial_ptr + offset, mask=mask, other=0.0).to(DTYPE)

    final_sum = tl.sum(val, axis=0).to(DTYPE)
    if tl.program_id(0) == 0:
        tl.store(out_ptr, final_sum)


@libentry()
@triton.jit
def dot_kernel_reduce_stage(
    partial_ptr,
    next_ptr,
    partial_size,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < partial_size
    val = tl.load(partial_ptr + offset, mask=mask, other=0.0).to(DTYPE)

    block_sum = tl.sum(val, axis=0).to(DTYPE)
    tl.store(next_ptr + pid, block_sum)


@libentry()
@triton.jit
def dot_kernel_atomic(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    incx,
    incy,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * incx
    y_offset = idx * incy

    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0, cache_modifier=".cg")
    y = tl.load(y_ptr + y_offset, mask=mask, other=0.0, cache_modifier=".cg")

    dot = x * y
    block_sum = tl.sum(dot, axis=0).to(DTYPE)
    tl.atomic_add(out_ptr, block_sum)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 65536}, num_warps=8),
    ],
    key=["n", "incx", "incy"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def dot_kernel_atomic_tuned(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    incx,
    incy,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * incx
    y_offset = idx * incy

    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0, cache_modifier=".cg")
    y = tl.load(y_ptr + y_offset, mask=mask, other=0.0, cache_modifier=".cg")

    block_sum = tl.sum((x * y).to(DTYPE), axis=0)
    tl.atomic_add(out_ptr, block_sum)


def _dot_stage1_num_warps(block_size: int) -> int:
    if block_size <= 2048:
        return 4
    return 8


def _run_dot_two_stage(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
    block_size: int,
    triton_dtype,
) -> None:
    num_blocks = triton.cdiv(n, block_size)
    partial = torch.empty((num_blocks,), dtype=x.dtype, device=x.device)
    dot_kernel_stage1[(num_blocks, 1, 1)](
        x,
        y,
        partial,
        n,
        incx,
        incy,
        BLOCK_SIZE=block_size,
        DTYPE=triton_dtype,
        num_warps=_dot_stage1_num_warps(block_size),
    )

    block_partial = triton.next_power_of_2(num_blocks)
    dot_kernel_reduce[(1, 1, 1)](
        partial,
        result,
        num_blocks,
        BLOCK_PARTIAL=block_partial,
        DTYPE=triton_dtype,
        num_warps=1 if block_partial <= 64 else 4,
    )


def _dot_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    if n <= 0:
        result.zero_()
        return

    # Validate strides and dimensions
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.device == result.device, "result must be on the same device"
    assert result.dtype == x.dtype, "result dtype must match input dtype"
    assert result.numel() == 1, "result must be a scalar tensor"

    required_x = 1 + (n - 1) * incx
    required_y = 1 + (n - 1) * incy
    assert x.numel() >= required_x, f"x too short: need {required_x}, got {x.numel()}"
    assert y.numel() >= required_y, f"y too short: need {required_y}, got {y.numel()}"

    if n <= 4096:
        block_size = triton.next_power_of_2(n)
    elif n <= 32768:
        block_size = 32768
    elif n <= 300000:
        block_size = 65536
    else:
        block_size = 4096
    num_blocks = triton.cdiv(n, block_size)
    MAX_ATOMIC_BLOCKS = 128

    dtype_map = {
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }
    triton_dtype = dtype_map[x.dtype]
    reduce_dtype = triton_dtype

    with torch_device_fn.device(x.device):
        if x.dtype == torch.float32 and n <= 300000:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), 1, 1)
            dot_kernel_atomic_tuned[grid](
                x,
                y,
                result,
                n,
                incx,
                incy,
                DTYPE=triton_dtype,
            )
        elif x.dtype == torch.float64 and n <= DDOT_FAST_MAX_N:
            if n <= DDOT_ATOMIC_MAX_N:
                grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), 1, 1)
                dot_kernel_atomic_tuned[grid](
                    x,
                    y,
                    result,
                    n,
                    incx,
                    incy,
                    DTYPE=triton_dtype,
                )
            else:
                _run_dot_two_stage(
                    n,
                    x,
                    incx,
                    y,
                    incy,
                    result,
                    DDOT_STAGE1_BLOCK_SIZE,
                    triton_dtype,
                )
        elif num_blocks == 1:
            dot_kernel_stage1[(1, 1, 1)](
                x, y, result, n, incx, incy, BLOCK_SIZE=block_size, DTYPE=triton_dtype
            )
        elif num_blocks <= MAX_ATOMIC_BLOCKS:
            result.zero_()
            grid = (num_blocks, 1, 1)
            if block_size >= 32768:
                dot_kernel_atomic[grid](
                    x,
                    y,
                    result,
                    n,
                    incx,
                    incy,
                    BLOCK_SIZE=block_size,
                    DTYPE=triton_dtype,
                    num_warps=8,
                )
            else:
                dot_kernel_atomic[grid](
                    x,
                    y,
                    result,
                    n,
                    incx,
                    incy,
                    BLOCK_SIZE=block_size,
                    DTYPE=triton_dtype,
                )
        else:
            partial_dtype = x.dtype
            partial = torch.empty((num_blocks,), dtype=partial_dtype, device=x.device)
            grid_stage1 = (num_blocks, 1, 1)
            dot_kernel_stage1[grid_stage1](
                x, y, partial, n, incx, incy, BLOCK_SIZE=block_size, DTYPE=triton_dtype
            )
            partial_size = num_blocks
            reduce_block_size = 1024
            reduce_tensor_dtype = x.dtype
            while partial_size > reduce_block_size:
                next_size = triton.cdiv(partial_size, reduce_block_size)
                next_partial = torch.empty(
                    (next_size,), dtype=reduce_tensor_dtype, device=x.device
                )
                dot_kernel_reduce_stage[(next_size, 1, 1)](
                    partial,
                    next_partial,
                    partial_size,
                    BLOCK_SIZE=reduce_block_size,
                    DTYPE=reduce_dtype,
                )
                partial = next_partial
                partial_size = next_size

            block_partial = triton.next_power_of_2(partial_size)
            dot_kernel_reduce[(1, 1, 1)](
                partial,
                result,
                partial_size,
                BLOCK_PARTIAL=block_partial,
                DTYPE=reduce_dtype,
            )


def sdot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS SDOT")
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert result.dtype == torch.float32, "result must be float32"
    _dot_impl(n, x, incx, y, incy, result)


def ddot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS DDOT")
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    assert result.dtype == torch.float64, "result must be float64"
    _dot_impl(n, x, incx, y, incy, result)

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

_SMALL_N = 16384
_C64_CONTIG_PERSISTENT_BLOCK_SIZE = 1024
_C64_CONTIG_CTAS_PER_SM = 8
_ZDOT_ATOMIC_MAX_N = 131072


def _small_block_size(n: int) -> int:
    return triton.next_power_of_2(n)


def _small_num_warps(block_size: int) -> int:
    return 4 if block_size <= 16384 else 8


def _persistent_grid_size(n: int, block_size: int, device: torch.device) -> int:
    blocks = triton.cdiv(n, block_size)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return min(blocks, sm_count * _C64_CONTIG_CTAS_PER_SM)


@libentry()
@triton.jit
def _dotu_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    incx_real,
    incy_real,
    BLOCK_SIZE: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x_base = offs * incx_real
    y_base = offs * incy_real
    xr = tl.load(x_ptr + x_base, mask=mask, other=0.0)
    xi = tl.load(x_ptr + x_base + 1, mask=mask, other=0.0)
    yr = tl.load(y_ptr + y_base, mask=mask, other=0.0)
    yi = tl.load(y_ptr + y_base + 1, mask=mask, other=0.0)

    sum_real = tl.sum(xr * yr - xi * yi, axis=0)
    sum_imag = tl.sum(xr * yi + xi * yr, axis=0)
    if USE_ATOMIC:
        tl.atomic_add(out_ptr, sum_real, sem="relaxed")
        tl.atomic_add(out_ptr + 1, sum_imag, sem="relaxed")
    else:
        tl.store(out_ptr, sum_real)
        tl.store(out_ptr + 1, sum_imag)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8),
    ],
    key=["n", "incx_real", "incy_real"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _dotu_atomic_tuned_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    incx_real,
    incy_real,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x_base = offs * incx_real
    y_base = offs * incy_real
    xr = tl.load(x_ptr + x_base, mask=mask, other=0.0)
    xi = tl.load(x_ptr + x_base + 1, mask=mask, other=0.0)
    yr = tl.load(y_ptr + y_base, mask=mask, other=0.0)
    yi = tl.load(y_ptr + y_base + 1, mask=mask, other=0.0)

    sum_real = tl.sum(xr * yr - xi * yi, axis=0)
    sum_imag = tl.sum(xr * yi + xi * yr, axis=0)
    tl.atomic_add(out_ptr, sum_real, sem="relaxed")
    tl.atomic_add(out_ptr + 1, sum_imag, sem="relaxed")


@libentry()
@triton.jit
def _cdotu_contig_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    base = offs * 2
    xr = tl.load(x_ptr + base, mask=mask, other=0.0)
    xi = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
    yr = tl.load(y_ptr + base, mask=mask, other=0.0)
    yi = tl.load(y_ptr + base + 1, mask=mask, other=0.0)

    sum_real = tl.sum(xr * yr - xi * yi, axis=0)
    sum_imag = tl.sum(xr * yi + xi * yr, axis=0)
    if USE_ATOMIC:
        tl.atomic_add(out_ptr, sum_real, sem="relaxed")
        tl.atomic_add(out_ptr + 1, sum_imag, sem="relaxed")
    else:
        tl.store(out_ptr, sum_real)
        tl.store(out_ptr + 1, sum_imag)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=8),
    ],
    key=["n"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _cdotu_contig_atomic_tuned_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    elem_offsets = tl.arange(0, BLOCK_SIZE * 2)
    block_start = pid * BLOCK_SIZE
    mask = elem_offsets < (n - block_start) * 2

    vals_x = tl.load(x_ptr + block_start * 2 + elem_offsets, mask=mask, other=0.0)
    vals_y = tl.load(y_ptr + block_start * 2 + elem_offsets, mask=mask, other=0.0)
    x_pairs = tl.reshape(vals_x, (BLOCK_SIZE, 2))
    y_pairs = tl.reshape(vals_y, (BLOCK_SIZE, 2))
    xr, xi = tl.split(x_pairs)
    yr, yi = tl.split(y_pairs)

    sum_real = tl.sum(xr * yr - xi * yi, axis=0)
    sum_imag = tl.sum(xr * yi + xi * yr, axis=0)
    tl.atomic_add(out_ptr, sum_real, sem="relaxed")
    tl.atomic_add(out_ptr + 1, sum_imag, sem="relaxed")


@libentry()
@triton.jit
def _cdotu_contig_stage1_kernel(
    x_ptr,
    y_ptr,
    partial_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    base = offs * 2
    xr = tl.load(x_ptr + base, mask=mask, other=0.0)
    xi = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
    yr = tl.load(y_ptr + base, mask=mask, other=0.0)
    yi = tl.load(y_ptr + base + 1, mask=mask, other=0.0)

    sum_real = tl.sum(xr * yr - xi * yi, axis=0)
    sum_imag = tl.sum(xr * yi + xi * yr, axis=0)

    out_base = pid * 2
    tl.store(partial_ptr + out_base, sum_real)
    tl.store(partial_ptr + out_base + 1, sum_imag)


@libentry()
@triton.jit
def _cdotu_contig_persistent_stage1_kernel(
    x_ptr,
    y_ptr,
    partial_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_ctas = tl.num_programs(0)
    elem_offsets = tl.arange(0, BLOCK_SIZE * 2)

    acc_real = tl.full((), 0.0, tl.float32)
    acc_imag = tl.full((), 0.0, tl.float32)
    block_start = pid * BLOCK_SIZE

    while block_start + BLOCK_SIZE <= n:
        vals_x = tl.load(x_ptr + block_start * 2 + elem_offsets)
        vals_y = tl.load(y_ptr + block_start * 2 + elem_offsets)
        x_pairs = tl.reshape(vals_x, (BLOCK_SIZE, 2))
        y_pairs = tl.reshape(vals_y, (BLOCK_SIZE, 2))
        xr, xi = tl.split(x_pairs)
        yr, yi = tl.split(y_pairs)

        acc_real += tl.sum(xr * yr - xi * yi, axis=0)
        acc_imag += tl.sum(xr * yi + xi * yr, axis=0)
        block_start += num_ctas * BLOCK_SIZE

    if block_start < n:
        valid_elems = (n - block_start) * 2
        mask = elem_offsets < valid_elems
        vals_x = tl.load(x_ptr + block_start * 2 + elem_offsets, mask=mask, other=0.0)
        vals_y = tl.load(y_ptr + block_start * 2 + elem_offsets, mask=mask, other=0.0)
        x_pairs = tl.reshape(vals_x, (BLOCK_SIZE, 2))
        y_pairs = tl.reshape(vals_y, (BLOCK_SIZE, 2))
        xr, xi = tl.split(x_pairs)
        yr, yi = tl.split(y_pairs)

        acc_real += tl.sum(xr * yr - xi * yi, axis=0)
        acc_imag += tl.sum(xr * yi + xi * yr, axis=0)

    out_base = pid * 2
    tl.store(partial_ptr + out_base, acc_real)
    tl.store(partial_ptr + out_base + 1, acc_imag)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _cdotu_contig_stage1_tuned_kernel(
    x_ptr,
    y_ptr,
    partial_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    base = offs * 2
    xr = tl.load(x_ptr + base, mask=mask, other=0.0)
    xi = tl.load(x_ptr + base + 1, mask=mask, other=0.0)
    yr = tl.load(y_ptr + base, mask=mask, other=0.0)
    yi = tl.load(y_ptr + base + 1, mask=mask, other=0.0)

    sum_real = tl.sum(xr * yr - xi * yi, axis=0)
    sum_imag = tl.sum(xr * yi + xi * yr, axis=0)

    out_base = pid * 2
    tl.store(partial_ptr + out_base, sum_real)
    tl.store(partial_ptr + out_base + 1, sum_imag)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=8),
    ],
    key=["n", "incx_real", "incy_real"],
)
@triton.jit
def _dotu_stage1_kernel(
    x_ptr,
    y_ptr,
    partial_ptr,
    n,
    incx_real,
    incy_real,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x_base = offs * incx_real
    y_base = offs * incy_real
    xr = tl.load(x_ptr + x_base, mask=mask, other=0.0)
    xi = tl.load(x_ptr + x_base + 1, mask=mask, other=0.0)
    yr = tl.load(y_ptr + y_base, mask=mask, other=0.0)
    yi = tl.load(y_ptr + y_base + 1, mask=mask, other=0.0)

    sum_real = tl.sum(xr * yr - xi * yi, axis=0)
    sum_imag = tl.sum(xr * yi + xi * yr, axis=0)

    out_base = pid * 2
    tl.store(partial_ptr + out_base, sum_real)
    tl.store(partial_ptr + out_base + 1, sum_imag)


@libentry()
@triton.jit
def _complex_reduce_kernel(
    in_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    in_base = offs * 2
    real = tl.load(in_ptr + in_base, mask=mask, other=0.0)
    imag = tl.load(in_ptr + in_base + 1, mask=mask, other=0.0)

    sum_real = tl.sum(real, axis=0)
    sum_imag = tl.sum(imag, axis=0)

    out_base = pid * 2
    tl.store(out_ptr + out_base, sum_real)
    tl.store(out_ptr + out_base + 1, sum_imag)


def _validate_dotu_inputs(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    if n <= 0:
        return

    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert result.numel() == 1, "result must have exactly one element"
    assert x.dtype in (torch.complex64, torch.complex128), "x must be complex"
    assert y.dtype == x.dtype, "y must have same dtype as x"
    assert result.dtype == x.dtype, "result must have same dtype as inputs"
    assert (
        x.device == y.device == result.device
    ), "all tensors must be on the same device"

    required_x = 1 + (n - 1) * incx
    required_y = 1 + (n - 1) * incy
    assert x.numel() >= required_x, (
        f"x too short: need at least {required_x} elements for n={n}, incx={incx}"
    )
    assert y.numel() >= required_y, (
        f"y too short: need at least {required_y} elements for n={n}, incy={incy}"
    )


def _dotu_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    _validate_dotu_inputs(n, x, incx, y, incy, result)
    if n <= 0:
        result.zero_()
        return

    x_real = torch.view_as_real(x).reshape(-1)
    y_real = torch.view_as_real(y).reshape(-1)
    result_real = torch.view_as_real(result).reshape(-1)

    use_contig = incx == 1 and incy == 1
    use_contig_c64 = x.dtype == torch.complex64 and use_contig
    max_atomic_blocks = 512 if x.dtype == torch.complex64 else 1024

    with torch_device_fn.device(x.device):
        if x.dtype == torch.complex64 and n <= _SMALL_N:
            block_size = _small_block_size(n)
            if use_contig_c64:
                _cdotu_contig_kernel[(1, 1, 1)](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    BLOCK_SIZE=block_size,
                    USE_ATOMIC=False,
                    num_warps=_small_num_warps(block_size),
                )
            else:
                _dotu_kernel[(1, 1, 1)](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    incx * 2,
                    incy * 2,
                    BLOCK_SIZE=block_size,
                    USE_ATOMIC=False,
                    num_warps=_small_num_warps(block_size),
                )
            return

        if x.dtype == torch.complex128 and n <= 2048:
            block_size = _small_block_size(n)
            if use_contig:
                _cdotu_contig_kernel[(1, 1, 1)](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    BLOCK_SIZE=block_size,
                    USE_ATOMIC=False,
                    num_warps=4,
                )
            else:
                _dotu_kernel[(1, 1, 1)](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    incx * 2,
                    incy * 2,
                    BLOCK_SIZE=block_size,
                    USE_ATOMIC=False,
                    num_warps=4,
                )
            return

        block_size = 4096
        grid_size = triton.cdiv(n, block_size)

        if grid_size == 1:
            _dotu_kernel[(1, 1, 1)](
                x_real,
                y_real,
                result_real,
                n,
                incx * 2,
                incy * 2,
                BLOCK_SIZE=block_size,
                USE_ATOMIC=False,
                num_warps=4,
            )
            return

        if x.dtype == torch.complex128 and n <= _ZDOT_ATOMIC_MAX_N:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), 1, 1)
            if use_contig:
                _cdotu_contig_atomic_tuned_kernel[grid](
                    x_real,
                    y_real,
                    result_real,
                    n,
                )
            else:
                _dotu_atomic_tuned_kernel[grid](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    incx * 2,
                    incy * 2,
                )
            return

        if grid_size <= max_atomic_blocks:
            if use_contig_c64 and grid_size > 4:
                grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), 1, 1)
                _cdotu_contig_atomic_tuned_kernel[grid](
                    x_real,
                    y_real,
                    result_real,
                    n,
                )
            elif use_contig_c64:
                result.zero_()
                _cdotu_contig_kernel[(grid_size, 1, 1)](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    BLOCK_SIZE=block_size,
                    USE_ATOMIC=True,
                    num_warps=4,
                )
            elif x.dtype == torch.complex64 and grid_size > 4:
                grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), 1, 1)
                _dotu_atomic_tuned_kernel[grid](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    incx * 2,
                    incy * 2,
                )
            else:
                result.zero_()
                _dotu_kernel[(grid_size, 1, 1)](
                    x_real,
                    y_real,
                    result_real,
                    n,
                    incx * 2,
                    incy * 2,
                    BLOCK_SIZE=block_size,
                    USE_ATOMIC=True,
                    num_warps=4 if grid_size <= 4 else 8,
                )
            return

        if use_contig_c64:
            stage1_block_size = _C64_CONTIG_PERSISTENT_BLOCK_SIZE
            partial_size = _persistent_grid_size(n, stage1_block_size, x.device)
            partial = torch.empty((partial_size,), dtype=x.dtype, device=x.device)
            partial_real = torch.view_as_real(partial).reshape(-1)
            _cdotu_contig_persistent_stage1_kernel[(partial_size, 1, 1)](
                x_real,
                y_real,
                partial_real,
                n,
                BLOCK_SIZE=stage1_block_size,
                num_warps=8,
            )
        else:
            stage1_block_size = block_size
            partial_size = triton.cdiv(n, stage1_block_size)
            partial = torch.empty((partial_size,), dtype=x.dtype, device=x.device)
            partial_real = torch.view_as_real(partial).reshape(-1)
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), 1, 1)
            _, meta = _dotu_stage1_kernel[grid](
                x_real,
                y_real,
                partial_real,
                n,
                incx * 2,
                incy * 2,
            )
            partial_size = triton.cdiv(n, meta["BLOCK_SIZE"])

        while partial_size > 1:
            if x.dtype == torch.complex64:
                reduce_block_size = min(32768, triton.next_power_of_2(partial_size))
                reduce_num_warps = 4 if reduce_block_size <= 2048 else 8
            else:
                reduce_block_size = 1024
                reduce_num_warps = 8
            next_size = triton.cdiv(partial_size, reduce_block_size)
            if next_size == 1:
                next_real = result_real
            else:
                next_partial = torch.empty((next_size,), dtype=x.dtype, device=x.device)
                next_real = torch.view_as_real(next_partial).reshape(-1)

            _complex_reduce_kernel[(next_size, 1, 1)](
                partial_real,
                next_real,
                partial_size,
                BLOCK_SIZE=reduce_block_size,
                num_warps=reduce_num_warps,
            )
            partial_real = next_real
            partial_size = next_size


def cdotu(
    n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, result: torch.Tensor
) -> None:
    logger.debug("FLAG_BLAS CDOTU")
    assert x.dtype == torch.complex64, "x must be complex64"
    _dotu_impl(n, x, incx, y, incy, result)


def zdotu(
    n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, result: torch.Tensor
) -> None:
    logger.debug("FLAG_BLAS ZDOTU")
    assert x.dtype == torch.complex128, "x must be complex128"
    _dotu_impl(n, x, incx, y, incy, result)

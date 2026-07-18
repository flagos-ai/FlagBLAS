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
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

_NRM2_KEY = ["n", "INCX"]
_NRM2_STAGE1_SMALL_N = 1_000_000
_NRM2_STAGE1_LARGE_N = 8_000_000
_NRM2_STAGE1_HUGE_N = 134_217_728


def _nrm2_stage1_min_block(n: int, x: torch.Tensor, incx: int) -> int:
    if x.dtype == torch.complex128 and incx == 1 and n >= _NRM2_STAGE1_HUGE_N:
        return 8192
    if n >= _NRM2_STAGE1_LARGE_N:
        return 4096
    if n >= _NRM2_STAGE1_SMALL_N:
        return 1024
    return 512


def _prune_nrm2_stage1_configs(configs, named_args, **kwargs):
    n = named_args["n"]
    incx = named_args["INCX"]
    x = named_args["x_ptr"]
    min_block = _nrm2_stage1_min_block(n, x, incx)
    return [config for config in configs if config.kwargs["BLOCK_SIZE"] >= min_block]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("nrm2_iluvatar_stage1_real"),
    key=_NRM2_KEY,
    prune_configs_by={"early_config_prune": _prune_nrm2_stage1_configs},
)
@triton.jit
def nrm2_kernel1_real(
    x_ptr,
    mid_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * INCX
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

    sq_x = x * x
    block_sum = tl.sum(sq_x, axis=0)

    tl.store(mid_ptr + pid, block_sum)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("nrm2_iluvatar_stage1_complex"),
    key=_NRM2_KEY,
    prune_configs_by={"early_config_prune": _prune_nrm2_stage1_configs},
)
@triton.jit
def nrm2_kernel1_complex(
    x_ptr,
    mid_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_base_offset = idx * INCX * 2
    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)

    sq_val = x_real * x_real + x_imag * x_imag
    block_sum = tl.sum(sq_val, axis=0)

    tl.store(mid_ptr + pid, block_sum)


@libentry()
@triton.jit
def nrm2_kernel2(mid_ptr, out_ptr, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)

    final_sum = tl.sum(mid_val, axis=0)
    tl.store(out_ptr, final_sum)


@libentry()
@triton.jit
def nrm2_kernel_atomic_real(
    x_ptr,
    out_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * INCX
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

    sq_x = x * x
    block_sum = tl.sum(sq_x, axis=0)

    tl.atomic_add(out_ptr, block_sum)


@libentry()
@triton.jit
def nrm2_kernel_atomic_complex(
    x_ptr,
    out_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_base_offset = idx * INCX * 2
    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)

    sq_val = x_real * x_real + x_imag * x_imag
    block_sum = tl.sum(sq_val, axis=0)

    tl.atomic_add(out_ptr, block_sum)


@libentry()
@triton.jit
def nrm2_kernel_direct_real(
    x_ptr,
    out_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * INCX
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

    sq_x = x * x
    block_sum = tl.sum(sq_x, axis=0)

    tl.store(out_ptr, tl.sqrt(block_sum))


@libentry()
@triton.jit
def nrm2_kernel_direct_complex(
    x_ptr,
    out_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_base_offset = idx * INCX * 2
    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)

    sq_val = x_real * x_real + x_imag * x_imag
    block_sum = tl.sum(sq_val, axis=0)

    tl.store(out_ptr, tl.sqrt(block_sum))


def _nrm2_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    result: torch.Tensor,
    is_complex: bool,
) -> None:
    if n <= 0 or incx <= 0:
        result.zero_()
        return

    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, (
        f"x is too short: need at least {required_size} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )

    assert x.is_contiguous(), "x must be contiguous"
    assert (
        result.dim() == 1 and result.numel() == 1
    ), "result must be a 1-element tensor"
    assert x.device == result.device, "x and result must be on the same device"

    if x.dtype in (torch.float32, torch.complex64) and n == 1024:
        block_size = triton.next_power_of_2(n)
        num_warps = 4
        if block_size >= 4096:
            num_warps = 8
        with torch_device_fn.device(x.device):
            if is_complex:
                x_real = torch.view_as_real(x)
                nrm2_kernel_direct_complex[(1, 1, 1)](
                    x_real, result, n, incx, block_size, num_warps=num_warps
                )
            else:
                nrm2_kernel_direct_real[(1, 1, 1)](
                    x, result, n, incx, block_size, num_warps=num_warps
                )
        return

    result.zero_()

    use_atomic_for_real_1048576 = (
        x.dtype == torch.float32 and incx == 1 and n == 1048576
    )

    block_size = 1024
    if use_atomic_for_real_1048576:
        block_size = 8192
    elif incx >= 3 and n == 65536:
        block_size = 512
    grid_size = triton.cdiv(n, block_size)

    max_atomic_blocks = 128

    if grid_size <= max_atomic_blocks:
        with torch_device_fn.device(x.device):
            if is_complex:
                x_real = torch.view_as_real(x)
                nrm2_kernel_atomic_complex[(grid_size, 1, 1)](
                    x_real, result, n, incx, block_size
                )
            else:
                nrm2_kernel_atomic_real[(grid_size, 1, 1)](
                    x, result, n, incx, block_size
                )
        result.sqrt_()
    else:
        min_block_size = _nrm2_stage1_min_block(n, x, incx)
        mid_size = triton.cdiv(n, min_block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid = torch.empty((mid_size,), dtype=result.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            if is_complex:
                x_real = torch.view_as_real(x)
                nrm2_kernel1_complex[(mid_size, 1, 1)](x_real, mid, n, incx)
            else:
                nrm2_kernel1_real[(mid_size, 1, 1)](x, mid, n, incx)

            nrm2_kernel2[(1, 1, 1)](mid, result, mid_size, block_mid)

        result.sqrt_()


def snrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ILUVATAR SNRM2")
    assert x.dtype == torch.float32, "x must be float32"
    assert result.dtype == torch.float32, "result must be float32"
    _nrm2_impl(n, x, incx, result, is_complex=False)


def dnrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ILUVATAR DNRM2")
    assert x.dtype == torch.float64, "x must be float64"
    assert result.dtype == torch.float64, "result must be float64"
    _nrm2_impl(n, x, incx, result, is_complex=False)


def scnrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ILUVATAR SCNRM2")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert result.dtype == torch.float32, "result for scnrm2 must be float32"
    _nrm2_impl(n, x, incx, result, is_complex=True)


def dznrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ILUVATAR DZNRM2")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert result.dtype == torch.float64, "result for dznrm2 must be float64"
    _nrm2_impl(n, x, incx, result, is_complex=True)

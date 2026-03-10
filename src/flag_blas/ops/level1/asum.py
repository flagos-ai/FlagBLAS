import logging
import math
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

@libentry()
@triton.jit
def asum_kernel1_real(
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

    abs_x = tl.abs(x)
    block_sum = tl.sum(abs_x)

    tl.store(mid_ptr + pid, block_sum)

@libentry()
@triton.jit
def asum_kernel1_complex(
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

    abs_sum_val = tl.abs(x_real) + tl.abs(x_imag)
    block_sum = tl.sum(abs_sum_val)

    tl.store(mid_ptr + pid, block_sum)

@libentry()
@triton.jit
def asum_kernel2(
    mid_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)

    final_sum = tl.sum(mid_val)
    tl.store(out_ptr, final_sum)

@libentry()
@triton.jit
def asum_kernel_atomic_real(
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

    abs_x = tl.abs(x)
    block_sum = tl.sum(abs_x, axis=0)

    tl.atomic_add(out_ptr, block_sum)

@libentry()
@triton.jit
def asum_kernel_atomic_complex(
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

    abs_sum_val = tl.abs(x_real) + tl.abs(x_imag)
    block_sum = tl.sum(abs_sum_val, axis=0)

    tl.atomic_add(out_ptr, block_sum)

def _asum_impl(n: int, x: torch.Tensor, incx: int, result: torch.Tensor, is_complex: bool) -> None:
    if n <= 0 or incx <= 0:
        result.zero_()
        return

    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, \
        f"x is too short: need at least {required_size} elements for n={n}, incx={incx}, got {x.numel()}"

    assert x.is_contiguous(), "x must be contiguous"
    result.zero_()

    block_size = 1024
    grid_size = triton.cdiv(n, block_size)

    MAX_ATOMIC_BLOCKS = 128

    if grid_size <= MAX_ATOMIC_BLOCKS:
        with torch_device_fn.device(x.device):
            if is_complex:
                x_real = torch.view_as_real(x)
                asum_kernel_atomic_complex[(grid_size, 1, 1)](x_real, result, n, incx, block_size)
            else:
                asum_kernel_atomic_real[(grid_size, 1, 1)](x, result, n, incx, block_size)
    else:
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n)))
        block_size = min(block_size, 16384)

        mid_size = triton.cdiv(n, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid = torch.empty((mid_size,), dtype=result.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            if is_complex:
                x_real = torch.view_as_real(x)
                asum_kernel1_complex[(mid_size, 1, 1)](x_real, mid, n, incx, block_size)
            else:
                asum_kernel1_real[(mid_size, 1, 1)](x, mid, n, incx, block_size)

            asum_kernel2[(1, 1, 1)](mid, result, mid_size, block_mid)

def sasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SASUM")
    assert x.dtype == torch.float32, "x must be float32"
    assert result.dtype == torch.float32, "result must be float32"
    _asum_impl(n, x, incx, result, is_complex=False)

def dasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DASUM")
    assert x.dtype == torch.float64, "x must be float64"
    assert result.dtype == torch.float64, "result must be float64"
    _asum_impl(n, x, incx, result, is_complex=False)

def scasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SCASUM")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert result.dtype == torch.float32, "result for scasum must be float32"
    _asum_impl(n, x, incx, result, is_complex=True)

def dzasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DZASUM")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert result.dtype == torch.float64, "result for dzasum must be float64"
    _asum_impl(n, x, incx, result, is_complex=True)
import logging

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def abs_kernel_real(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y = tl.abs(x)
    tl.store(y_ptr + idx, y, mask=mask)


@libentry()
@triton.jit
def abs_kernel_complex(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_base_offset = idx * 2
    xr = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    xi = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)

    y = tl.sqrt(xr * xr + xi * xi)
    tl.store(y_ptr + idx, y, mask=mask)


def _abs_impl(
    n: int,
    x: torch.Tensor,
    y: torch.Tensor,
    is_complex: bool,
) -> None:
    if n <= 0:
        return

    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    assert x.numel() >= n, f"x is too short: need at least {n} elements, got {x.numel()}"
    assert y.numel() >= n, f"y is too short: need at least {n} elements, got {y.numel()}"

    block_size = 1024
    grid = (triton.cdiv(n, block_size), 1, 1)

    with torch_device_fn.device(x.device):
        if is_complex:
            x_real = torch.view_as_real(x)
            abs_kernel_complex[grid](x_real, y, n, BLOCK_SIZE=block_size)
        else:
            abs_kernel_real[grid](x, y, n, BLOCK_SIZE=block_size)


def sabs(n: int, x: torch.Tensor, y: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SABS")
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    _abs_impl(n, x, y, is_complex=False)


def dabs(n: int, x: torch.Tensor, y: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DABS")
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    _abs_impl(n, x, y, is_complex=False)


def cabs(n: int, x: torch.Tensor, y: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS CABS")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.float32, "y must be float32"
    _abs_impl(n, x, y, is_complex=True)


def zabs(n: int, x: torch.Tensor, y: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ZABS")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.float64, "y must be float64"
    _abs_impl(n, x, y, is_complex=True)
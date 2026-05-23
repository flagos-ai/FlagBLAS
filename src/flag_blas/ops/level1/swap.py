import logging

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n", "incx", "incy"],
    restore_value=["x_ptr", "y_ptr"],
)
@triton.jit
def swap_kernel(
    x_ptr,
    y_ptr,
    n,
    incx,
    incy,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offs = idx * incx
    y_offs = idx * incy

    x = tl.load(x_ptr + x_offs, mask=mask)
    y = tl.load(y_ptr + y_offs, mask=mask)

    tl.store(x_ptr + x_offs, y, mask=mask)
    tl.store(y_ptr + y_offs, x, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n", "incx", "incy"],
    restore_value=["x_ptr", "y_ptr"],
)
@triton.jit
def swap_complex_kernel(
    x_ptr,
    y_ptr,
    n,
    incx,
    incy,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_base = idx * incx * 2
    y_base = idx * incy * 2

    xr = tl.load(x_ptr + x_base, mask=mask)
    xi = tl.load(x_ptr + x_base + 1, mask=mask)
    yr = tl.load(y_ptr + y_base, mask=mask)
    yi = tl.load(y_ptr + y_base + 1, mask=mask)

    tl.store(x_ptr + x_base, yr, mask=mask)
    tl.store(x_ptr + x_base + 1, yi, mask=mask)
    tl.store(y_ptr + y_base, xr, mask=mask)
    tl.store(y_ptr + y_base + 1, xi, mask=mask)


def _validate_swap_inputs(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
) -> None:
    if n <= 0:
        return

    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == y.dtype, "x and y must have the same dtype"

    required_x = 1 + (n - 1) * incx
    required_y = 1 + (n - 1) * incy

    assert x.numel() >= required_x, (
        f"x is too short: need at least {required_x} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )

    assert y.numel() >= required_y, (
        f"y is too short: need at least {required_y} elements "
        f"for n={n}, incy={incy}, got {y.numel()}"
    )


def _swap_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
) -> None:
    if n <= 0:
        return

    _validate_swap_inputs(n, x, incx, y, incy)

    if x.dtype.is_complex:
        x_real = torch.view_as_real(x).reshape(-1)
        y_real = torch.view_as_real(y).reshape(-1)

        with torch_device_fn.device(x.device):
            grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
            swap_complex_kernel[grid](x_real, y_real, n, incx, incy)
        return

    with torch_device_fn.device(x.device):
        grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
        swap_kernel[grid](x, y, n, incx, incy)


def sswap(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS SSWAP")
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    _swap_impl(n, x, incx, y, incy)


def dswap(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS DSWAP")
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    _swap_impl(n, x, incx, y, incy)


def cswap(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS CSWAP")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    _swap_impl(n, x, incx, y, incy)


def zswap(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS ZSWAP")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    _swap_impl(n, x, incx, y, incy)

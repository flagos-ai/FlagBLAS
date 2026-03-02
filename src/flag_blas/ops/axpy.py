import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX", "INCY"], 
    restore_value=["y_ptr"],
)
@triton.jit
def axpy_real_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * INCX
    y_offset = idx * INCY

    x = tl.load(x_ptr + x_offset, mask=mask)
    y = tl.load(y_ptr + y_offset, mask=mask)

    tl.store(y_ptr + y_offset, alpha * x + y, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX", "INCY"],
    restore_value=["y_ptr"],
)
@triton.jit
def axpy_complex_kernel(
    x_ptr,
    y_ptr,
    alpha_real,
    alpha_imag,
    n,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_base_offset = idx * INCX * 2
    y_base_offset = idx * INCY * 2

    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)
    y_real = tl.load(y_ptr + y_base_offset, mask=mask, other=0.0)
    y_imag = tl.load(y_ptr + y_base_offset + 1, mask=mask, other=0.0)

    out_real = alpha_real * x_real - alpha_imag * x_imag + y_real
    out_imag = alpha_real * x_imag + alpha_imag * x_real + y_imag

    tl.store(y_ptr + y_base_offset, out_real, mask=mask)
    tl.store(y_ptr + y_base_offset + 1, out_imag, mask=mask)


def saxpy(x: torch.Tensor, y: torch.Tensor, alpha: ScalarType = 1.0, incx: int = 1, incy: int = 1) -> torch.Tensor:
    logger.debug("FLAG_BLAS SAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    n = (x.numel() + incx - 1) // incx
    n_y = (y.numel() + incy - 1) // incy
    assert n <= n_y, f"y is too short: need at least {n * incy} elements, got {y.numel()}"

    x = x.contiguous()
    y = y.contiguous()

    if n == 0:
        return y

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        axpy_real_kernel[grid](x, y, float(alpha), n, incx, incy)

    return y


def daxpy(x: torch.Tensor, y: torch.Tensor, alpha: ScalarType = 1.0, incx: int = 1, incy: int = 1) -> torch.Tensor:
    logger.debug("FLAG_BLAS DAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    n = (x.numel() + incx - 1) // incx
    n_y = (y.numel() + incy - 1) // incy
    assert n <= n_y, f"y is too short: need at least {n * incy} elements, got {y.numel()}"

    x = x.contiguous()
    y = y.contiguous()

    if n == 0:
        return y

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        axpy_real_kernel[grid](x, y, float(alpha), n, incx, incy)

    return y


def caxpy(x: torch.Tensor, y: torch.Tensor, alpha: ScalarType = 1.0, incx: int = 1, incy: int = 1) -> torch.Tensor:
    logger.debug("FLAG_BLAS CAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    n = (x.numel() + incx - 1) // incx
    n_y = (y.numel() + incy - 1) // incy
    assert n <= n_y, f"y is too short: need at least {n * incy} elements, got {y.numel()}"

    x_real = torch.view_as_real(x.contiguous())
    y_real = torch.view_as_real(y.contiguous())

    if n == 0:
        return y

    if isinstance(alpha, complex):
        alpha_real = float(alpha.real)
        alpha_imag = float(alpha.imag)
    else:
        alpha_real = float(alpha)
        alpha_imag = 0.0

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        axpy_complex_kernel[grid](
            x_real,
            y_real,
            alpha_real,
            alpha_imag,
            n,
            incx,
            incy,
        )

    return y


def zaxpy(x: torch.Tensor, y: torch.Tensor, alpha: ScalarType = 1.0, incx: int = 1, incy: int = 1) -> torch.Tensor:
    logger.debug("FLAG_BLAS ZAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    n = (x.numel() + incx - 1) // incx
    n_y = (y.numel() + incy - 1) // incy
    assert n <= n_y, f"y is too short: need at least {n * incy} elements, got {y.numel()}"

    x_real = torch.view_as_real(x.contiguous())
    y_real = torch.view_as_real(y.contiguous())

    if n == 0:
        return y

    if isinstance(alpha, complex):
        alpha_real = float(alpha.real)
        alpha_imag = float(alpha.imag)
    else:
        alpha_real = float(alpha)
        alpha_imag = 0.0

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        axpy_complex_kernel[grid](
            x_real,
            y_real,
            alpha_real,
            alpha_imag,
            n,
            incx,
            incy,
        )

    return y
import logging
from typing import Union

import torch
import triton
import triton.language as tl

from  flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry,libtuner
from flag_blas.utils import triton_lang_extension as tle

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
    alpha: tl.constexpr,
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
    alpha_real: tl.constexpr,
    alpha_imag: tl.constexpr,
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


def saxpy(n: int, alpha: ScalarType, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS SAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if n <= 0:
        return

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        axpy_real_kernel[grid](x, y, float(alpha), n, incx, incy)


def daxpy(n: int, alpha: ScalarType, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS DAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if n <= 0:
        return

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        axpy_real_kernel[grid](x, y, float(alpha), n, incx, incy)


def caxpy(n: int, alpha: ScalarType, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS CAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if n <= 0:
        return

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    if isinstance(alpha, complex):
        alpha_real = float(alpha.real)
        alpha_imag = float(alpha.imag)
    else:
        alpha_real = float(alpha)
        alpha_imag = 0.0

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

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


def zaxpy(n: int, alpha: ScalarType, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS ZAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"

    if n <= 0:
        return

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    if isinstance(alpha, complex):
        alpha_real = float(alpha.real)
        alpha_imag = float(alpha.imag)
    else:
        alpha_real = float(alpha)
        alpha_imag = 0.0

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

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
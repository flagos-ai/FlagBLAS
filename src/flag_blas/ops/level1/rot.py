import logging
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
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX", "INCY"],
    restore_value=["x_ptr", "y_ptr"],
)
@triton.jit
def srot_kernel(
    x_ptr,
    y_ptr,
    c: tl.float32,
    s: tl.float32,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * INCX
    y_offset = idx * INCY

    x = tl.load(x_ptr + x_offset, mask=mask)
    y = tl.load(y_ptr + y_offset, mask=mask)

    new_x = c * x + s * y
    new_y = -s * x + c * y

    tl.store(x_ptr + x_offset, new_x, mask=mask)
    tl.store(y_ptr + y_offset, new_y, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX", "INCY"],
    restore_value=["x_ptr", "y_ptr"],
)
@triton.jit
def drot_kernel(
    x_ptr,
    y_ptr,
    c_int: tl.int64,
    s_int: tl.int64,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    c = c_int.to(tl.float64, bitcast=True)
    s = s_int.to(tl.float64, bitcast=True)

    x_offset = idx * INCX
    y_offset = idx * INCY

    x = tl.load(x_ptr + x_offset, mask=mask)
    y = tl.load(y_ptr + y_offset, mask=mask)

    new_x = c * x + s * y
    new_y = -s * x + c * y

    tl.store(x_ptr + x_offset, new_x, mask=mask)
    tl.store(y_ptr + y_offset, new_y, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX", "INCY"],
    restore_value=["x_ptr", "y_ptr"],
)
@triton.jit
def crot_kernel(
    x_ptr,
    y_ptr,
    c: tl.float32,
    s_real: tl.float32,
    s_imag: tl.float32,
    n,
    INCX,
    INCY,
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

    new_x_real = c * x_real + s_real * y_real - s_imag * y_imag
    new_x_imag = c * x_imag + s_real * y_imag + s_imag * y_real

    new_y_real = -s_real * x_real - s_imag * x_imag + c * y_real
    new_y_imag = -s_real * x_imag + s_imag * x_real + c * y_imag

    tl.store(x_ptr + x_base_offset, new_x_real, mask=mask)
    tl.store(x_ptr + x_base_offset + 1, new_x_imag, mask=mask)
    tl.store(y_ptr + y_base_offset, new_y_real, mask=mask)
    tl.store(y_ptr + y_base_offset + 1, new_y_imag, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX", "INCY"],
    restore_value=["x_ptr", "y_ptr"],
)
@triton.jit
def zrot_kernel(
    x_ptr,
    y_ptr,
    c_int: tl.int64,
    s_real_int: tl.int64,
    s_imag_int: tl.int64,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    c = c_int.to(tl.float64, bitcast=True)
    s_real = s_real_int.to(tl.float64, bitcast=True)
    s_imag = s_imag_int.to(tl.float64, bitcast=True)

    x_base_offset = idx * INCX * 2
    y_base_offset = idx * INCY * 2

    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)
    y_real = tl.load(y_ptr + y_base_offset, mask=mask, other=0.0)
    y_imag = tl.load(y_ptr + y_base_offset + 1, mask=mask, other=0.0)

    new_x_real = c * x_real + s_real * y_real - s_imag * y_imag
    new_x_imag = c * x_imag + s_real * y_imag + s_imag * y_real

    new_y_real = -s_real * x_real - s_imag * x_imag + c * y_real
    new_y_imag = -s_real * x_imag + s_imag * x_real + c * y_imag

    tl.store(x_ptr + x_base_offset, new_x_real, mask=mask)
    tl.store(x_ptr + x_base_offset + 1, new_x_imag, mask=mask)
    tl.store(y_ptr + y_base_offset, new_y_real, mask=mask)
    tl.store(y_ptr + y_base_offset + 1, new_y_imag, mask=mask)


def srot(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, c: ScalarType, s: ScalarType) -> None:
    logger.debug("FLAG_BLAS SROT")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    assert incx > 0 and incy > 0, "incx and incy must be positive"

    if n <= 0:
        return

    c = c.item() if isinstance(c, torch.Tensor) else float(c)
    s = s.item() if isinstance(s, torch.Tensor) else float(s)

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        srot_kernel[grid](x, y, c, s, n, incx, incy)


def drot(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, c: ScalarType, s: ScalarType) -> None:
    logger.debug("FLAG_BLAS DROT")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    assert incx > 0 and incy > 0, "incx and incy must be positive"

    if n <= 0:
        return

    c_val = float(c.item() if isinstance(c, torch.Tensor) else c)
    s_val = float(s.item() if isinstance(s, torch.Tensor) else s)
    c_int = torch.tensor(c_val, dtype=torch.float64).view(torch.int64).item()
    s_int = torch.tensor(s_val, dtype=torch.float64).view(torch.int64).item()

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        drot_kernel[grid](x, y, c_int, s_int, n, incx, incy)


def crot(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, c: ScalarType, s: ScalarType) -> None:
    logger.debug("FLAG_BLAS CROT")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    assert incx > 0 and incy > 0, "incx and incy must be positive"

    if n <= 0:
        return

    c = c.item() if isinstance(c, torch.Tensor) else float(c)
    s = s.item() if isinstance(s, torch.Tensor) else s
    s_real = float(s.real) if isinstance(s, complex) else float(s)
    s_imag = float(s.imag) if isinstance(s, complex) else 0.0

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        crot_kernel[grid](x_real, y_real, c, s_real, s_imag, n, incx, incy)


def zrot(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, c: ScalarType, s: ScalarType) -> None:
    logger.debug("FLAG_BLAS ZROT")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    assert incx > 0 and incy > 0, "incx and incy must be positive"

    if n <= 0:
        return

    c_val = float(c.item() if isinstance(c, torch.Tensor) else c)
    s = s.item() if isinstance(s, torch.Tensor) else s
    s_real = float(s.real) if isinstance(s, complex) else float(s)
    s_imag = float(s.imag) if isinstance(s, complex) else 0.0

    c_int = torch.tensor(c_val, dtype=torch.float64).view(torch.int64).item()
    s_real_int = torch.tensor(s_real, dtype=torch.float64).view(torch.int64).item()
    s_imag_int = torch.tensor(s_imag, dtype=torch.float64).view(torch.int64).item()

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert y.numel() >= req_size_y, f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        zrot_kernel[grid](x_real, y_real, c_int, s_real_int, s_imag_int, n, incx, incy)

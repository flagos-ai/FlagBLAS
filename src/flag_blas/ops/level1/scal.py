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
    key=["n", "INCX"],
    restore_value=["x_ptr"],
)
@triton.jit
def sscal_kernel(
    x_ptr,
    alpha: tl.float32,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offset = idx * INCX
    x = tl.load(x_ptr + x_offset, mask=mask)

    tl.store(x_ptr + x_offset, alpha * x, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX"],
    restore_value=["x_ptr"],
)
@triton.jit
def dscal_kernel(
    x_ptr,
    alpha_int: tl.int64,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    alpha = alpha_int.to(tl.float64, bitcast=True)

    x_offset = idx * INCX
    x = tl.load(x_ptr + x_offset, mask=mask)

    tl.store(x_ptr + x_offset, alpha * x, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX"],
    restore_value=["x_ptr"],
)
@triton.jit
def cscal_kernel(
    x_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
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

    out_real = alpha_real * x_real - alpha_imag * x_imag
    out_imag = alpha_real * x_imag + alpha_imag * x_real

    tl.store(x_ptr + x_base_offset, out_real, mask=mask)
    tl.store(x_ptr + x_base_offset + 1, out_imag, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX"],
    restore_value=["x_ptr"],
)
@triton.jit
def zscal_kernel(
    x_ptr,
    alpha_real_bits: tl.int64,
    alpha_imag_bits: tl.int64,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    alpha_real = alpha_real_bits.to(tl.float64, bitcast=True)
    alpha_imag = alpha_imag_bits.to(tl.float64, bitcast=True)

    x_base_offset = idx * INCX * 2
    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)

    out_real = alpha_real * x_real - alpha_imag * x_imag
    out_imag = alpha_real * x_imag + alpha_imag * x_real

    tl.store(x_ptr + x_base_offset, out_real, mask=mask)
    tl.store(x_ptr + x_base_offset + 1, out_imag, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX"],
    restore_value=["x_ptr"],
)
@triton.jit
def csscal_kernel(
    x_ptr,
    alpha: tl.float32,
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

    tl.store(x_ptr + x_base_offset, alpha * x_real, mask=mask)
    tl.store(x_ptr + x_base_offset + 1, alpha * x_imag, mask=mask)

@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ],
    key=["n", "INCX"],
    restore_value=["x_ptr"],
)
@triton.jit
def zdscal_kernel(
    x_ptr,
    alpha_bits: tl.int64,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    alpha = alpha_bits.to(tl.float64, bitcast=True)

    x_base_offset = idx * INCX * 2
    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)

    tl.store(x_ptr + x_base_offset, alpha * x_real, mask=mask)
    tl.store(x_ptr + x_base_offset + 1, alpha * x_imag, mask=mask)

def sscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS SSCAL")

    assert x.dtype == torch.float32, "x must be float32"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert incx > 0, "incx must be positive"

    if n <= 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)

    req_size_x = 1 + (n - 1) * incx
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        sscal_kernel[grid](x, alpha, n, incx)


def dscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS DSCAL")

    assert x.dtype == torch.float64, "x must be float64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert incx > 0, "incx must be positive"

    if n <= 0:
        return

    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    alpha_int = torch.tensor(alpha_val, dtype=torch.float64).view(torch.int64).item()

    req_size_x = 1 + (n - 1) * incx
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        dscal_kernel[grid](x, alpha_int, n, incx)


def cscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS CSCAL")

    assert x.dtype == torch.complex64, "x must be complex64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert incx > 0, "incx must be positive"

    if n <= 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0

    req_size_x = 1 + (n - 1) * incx
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"

    x_real = torch.view_as_real(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        cscal_kernel[grid](x_real, alpha_real, alpha_imag, n, incx)

def zscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS ZSCAL")

    assert x.dtype == torch.complex128, "x must be complex128"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert incx > 0, "incx must be positive"

    if n <= 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0

    alpha_real_bits = (
        torch.tensor(alpha_real, dtype=torch.float64)
        .view(torch.int64)
        .item()
    )
    alpha_imag_bits = (
        torch.tensor(alpha_imag, dtype=torch.float64)
        .view(torch.int64)
        .item()
    )

    req_size_x = 1 + (n - 1) * incx
    assert x.numel() >= req_size_x, (
        f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    )

    x_real = torch.view_as_real(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        zscal_kernel[grid](x_real, alpha_real_bits, alpha_imag_bits, n, incx)


def csscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS CSSCAL")

    assert x.dtype == torch.complex64, "x must be complex64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert incx > 0, "incx must be positive"

    if n <= 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    assert not isinstance(alpha, complex), "alpha must be real for csscal"
    alpha = float(alpha)

    req_size_x = 1 + (n - 1) * incx
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"

    x_real = torch.view_as_real(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        csscal_kernel[grid](x_real, alpha, n, incx)


def zdscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS ZDSCAL")

    assert x.dtype == torch.complex128, "x must be complex128"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert incx > 0, "incx must be positive"

    if n <= 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    assert not isinstance(alpha, complex), "alpha must be real for zdscal"
    alpha_val = float(alpha)

    alpha_bits = (
        torch.tensor(alpha_val, dtype=torch.float64)
        .view(torch.int64)
        .item()
    )

    req_size_x = 1 + (n - 1) * incx
    assert x.numel() >= req_size_x, (
        f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    )

    x_real = torch.view_as_real(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        zdscal_kernel[grid](x_real, alpha_bits, n, incx)
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
def scal_real_kernel(
    x_ptr,
    alpha: tl.constexpr,
    n,
    INCX: tl.constexpr,
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
def scal_complex_kernel(
    x_ptr,
    alpha_real: tl.constexpr,
    alpha_imag: tl.constexpr,
    n,
    INCX: tl.constexpr,
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


def sscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS SSCAL")

    assert x.dtype == torch.float32, "x must be float32"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert incx > 0, "incx must be positive"

    # Type check for alpha (const float *)
    if isinstance(alpha, torch.Tensor):
        assert alpha.dtype == torch.float32, "alpha must be a float32 tensor"
        alpha_val = float(alpha.item())
    else:
        assert isinstance(alpha, (float, int)), "alpha must be a real scalar"
        alpha_val = float(alpha)

    if n <= 0:
        return

    req_size_x = 1 + (n - 1) * incx
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"

    assert x.is_contiguous(), "x must be contiguous"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        scal_real_kernel[grid](x, alpha_val, n, incx)


def dscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS DSCAL")

    assert x.dtype == torch.float64, "x must be float64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert incx > 0, "incx must be positive"

    # Type check for alpha (const double *)
    if isinstance(alpha, torch.Tensor):
        assert alpha.dtype == torch.float64, "alpha must be a float64 tensor"
        alpha_val = float(alpha.item())
    else:
        assert isinstance(alpha, (float, int)), "alpha must be a real scalar"
        alpha_val = float(alpha)

    if n <= 0:
        return

    req_size_x = 1 + (n - 1) * incx
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"

    assert x.is_contiguous(), "x must be contiguous"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        scal_real_kernel[grid](x, alpha_val, n, incx)


def cscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int) -> None:
    logger.debug("FLAG_BLAS CSCAL")

    assert x.dtype == torch.complex64, "x must be complex64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert incx > 0, "incx must be positive"

    # Type check for alpha (const cuComplex *)
    if isinstance(alpha, torch.Tensor):
        assert alpha.dtype == torch.complex64, "alpha must be a complex64 tensor"
        alpha_item = alpha.item()
        alpha_real, alpha_imag = float(alpha_item.real), float(alpha_item.imag)
    else:
        assert isinstance(alpha, (complex, float, int)), "alpha must be a complex scalar"
        alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
        alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0

    if n <= 0:
        return

    req_size_x = 1 + (n - 1) * incx
    assert x.numel() >= req_size_x, f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"

    assert x.is_contiguous(), "x must be contiguous"

    x_real = torch.view_as_real(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        scal_complex_kernel[grid](x_real, alpha_real, alpha_imag, n, incx)

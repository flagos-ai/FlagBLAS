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
@libtuner(
    configs=runtime.get_tuned_config("scal"),
                # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["n"],
    strategy=["align32"],
    warmup=5,
    rep=10,
    restore_value=["y_ptr"],
)

@triton.jit
def scal_real_kernel(
    x_ptr,
    y_ptr,
    alpha: tl.constexpr,
    n,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    if KERNEL_ID == 0: 
        pid = tle.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < n

        x_offset = idx * INCX
        y_offset = idx * INCY

        x = tl.load(x_ptr + x_offset, mask=mask)
        y = tl.load(y_ptr + y_offset, mask=mask)

        tl.store(y_ptr + y_offset, alpha * x + y, mask=mask)
    else:
        pid = tle.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < n

        x_offset = idx * INCX
        y_offset = idx * INCY

        x = tl.load(x_ptr + x_offset, mask=mask)
        y = tl.load(y_ptr + y_offset, mask=mask)
        
        a = (x - y) * alpha
        b = y * alpha + y
        result = a + b

        tl.store(y_ptr + y_offset, result, mask=mask)





def sscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
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
        scal_real_kernel[grid](x, y, float(alpha), n, incx, incy)


def dscal(n: int, alpha: ScalarType, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
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
        scal_real_kernel[grid](x, y, float(alpha), n, incx, incy)


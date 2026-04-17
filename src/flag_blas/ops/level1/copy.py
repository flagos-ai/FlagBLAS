import logging

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# -----------------------------
# Triton kernels
# -----------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n"],
)
@libentry()
@triton.jit
def copy_contig_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n", "INCX", "INCY"],
)
@libentry()
@triton.jit
def copy_strided_real_kernel(
    x_ptr,
    y_ptr,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offs = idx * INCX
    y_offs = idx * INCY

    x = tl.load(x_ptr + x_offs, mask=mask, other=0)
    tl.store(y_ptr + y_offs, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n", "INCX", "INCY"],
)
@libentry()
@triton.jit
def copy_strided_complex_kernel(
    x_ptr,
    y_ptr,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generic complex kernel for ccopy.
    x_ptr / y_ptr point to flattened real storage:
      complex64  -> float32 flat array of length 2 * numel
      complex128 -> float64 flat array of length 2 * numel
    """
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    base_x = idx * INCX * 2
    base_y = idx * INCY * 2

    pair = tl.arange(0, 2)
    x_offs = base_x[:, None] + pair[None, :]
    y_offs = base_y[:, None] + pair[None, :]

    vals = tl.load(x_ptr + x_offs, mask=mask[:, None], other=0)
    tl.store(y_ptr + y_offs, vals, mask=mask[:, None])


# -----------------------------
# zcopy specialized kernels
# -----------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_scalar"],
)
@libentry()
@triton.jit
def zcopy_contig_kernel(
    x_ptr,   # flattened float64 storage
    y_ptr,
    n_scalar,   # 2 * n
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_scalar

    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
    ],
    key=["n", "INCY"],
)
@libentry()
@triton.jit
def zcopy_src_contig_kernel(
    x_ptr,   # flattened float64 storage
    y_ptr,
    n,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    """
    complex128:
      source contiguous  (incx == 1)
      destination strided
    """
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    base_x = idx * 2
    base_y = idx * INCY * 2

    pair = tl.arange(0, 2)
    x_offs = base_x[:, None] + pair[None, :]
    y_offs = base_y[:, None] + pair[None, :]

    vals = tl.load(x_ptr + x_offs, mask=mask[:, None], other=0.0)
    tl.store(y_ptr + y_offs, vals, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
    ],
    key=["n", "INCX"],
)
@libentry()
@triton.jit
def zcopy_dst_contig_kernel(
    x_ptr,   # flattened float64 storage
    y_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    """
    complex128:
      source strided
      destination contiguous (incy == 1)
    """
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    base_x = idx * INCX * 2
    base_y = idx * 2

    pair = tl.arange(0, 2)
    x_offs = base_x[:, None] + pair[None, :]
    y_offs = base_y[:, None] + pair[None, :]

    vals = tl.load(x_ptr + x_offs, mask=mask[:, None], other=0.0)
    tl.store(y_ptr + y_offs, vals, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
    ],
    key=["n", "INCX", "INCY"],
)
@libentry()
@triton.jit
def zcopy_both_strided_small_stride_kernel(
    x_ptr,   # flattened float64 storage
    y_ptr,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    """
    complex128:
      both source and destination are strided
      specialized for relatively small strides (e.g. <= 4)
    """
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    base_x = idx * INCX * 2
    base_y = idx * INCY * 2

    pair = tl.arange(0, 2)
    x_offs = base_x[:, None] + pair[None, :]
    y_offs = base_y[:, None] + pair[None, :]

    vals = tl.load(x_ptr + x_offs, mask=mask[:, None], other=0.0)
    tl.store(y_ptr + y_offs, vals, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=1),
    ],
    key=["n", "INCX", "INCY"],
)
@libentry()
@triton.jit
def zcopy_both_strided_generic_kernel(
    x_ptr,   # flattened float64 storage
    y_ptr,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    """
    complex128:
      both source and destination are strided
      generic path for larger strides
    """
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    base_x = idx * INCX * 2
    base_y = idx * INCY * 2

    pair = tl.arange(0, 2)
    x_offs = base_x[:, None] + pair[None, :]
    y_offs = base_y[:, None] + pair[None, :]

    vals = tl.load(x_ptr + x_offs, mask=mask[:, None], other=0.0)
    tl.store(y_ptr + y_offs, vals, mask=mask[:, None])


# -----------------------------
# Helper functions
# -----------------------------

def _validate_copy_inputs(
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
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"

    required_x = 1 + (n - 1) * incx
    required_y = 1 + (n - 1) * incy

    assert (
        x.numel() >= required_x
    ), f"x is too short: need at least {required_x} elements for n={n}, incx={incx}, got {x.numel()}"

    assert (
        y.numel() >= required_y
    ), f"y is too short: need at least {required_y} elements for n={n}, incy={incy}, got {y.numel()}"



def _copy_impl_zcopy_kernel_only(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
) -> None:
    """
    Benchmark-agnostic kernel-only dispatch for zcopy (complex128).
    No torch fallback.
    No hard-coded hot stride pairs.
    """
    if n <= 0:
        return

    _validate_copy_inputs(n, x, incx, y, incy)

    with torch_device_fn.device(x.device):
        x_flat = torch.view_as_real(x).reshape(-1)
        y_flat = torch.view_as_real(y).reshape(-1)

        # 1) contiguous
        if incx == 1 and incy == 1:
            n_scalar = 2 * n
            grid = lambda META: (triton.cdiv(n_scalar, META["BLOCK_SIZE"]),)
            zcopy_contig_kernel[grid](x_flat, y_flat, n_scalar)
            return

        # 2) one-sided stride
        if incx == 1:
            grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
            zcopy_src_contig_kernel[grid](x_flat, y_flat, n, incy)
            return

        if incy == 1:
            grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
            zcopy_dst_contig_kernel[grid](x_flat, y_flat, n, incx)
            return

        # 3) both-sided stride: bucket by stride size
        grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)

        if incx <= 4 and incy <= 4:
            zcopy_both_strided_small_stride_kernel[grid](x_flat, y_flat, n, incx, incy)
            return

        zcopy_both_strided_generic_kernel[grid](x_flat, y_flat, n, incx, incy)


def _copy_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    is_complex: bool,
) -> None:
    if n <= 0:
        return

    _validate_copy_inputs(n, x, incx, y, incy)

    # zcopy: kernel-only specialized path
    if is_complex and x.dtype == torch.complex128:
        _copy_impl_zcopy_kernel_only(n, x, incx, y, incy)
        return

    with torch_device_fn.device(x.device):
        # real contiguous
        if not is_complex and incx == 1 and incy == 1:
            grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
            copy_contig_kernel[grid](x, y, n)
            return

        # ccopy contiguous
        if is_complex and incx == 1 and incy == 1:
            x_flat = torch.view_as_real(x).reshape(-1)
            y_flat = torch.view_as_real(y).reshape(-1)
            n_scalar = 2 * n
            grid = lambda META: (triton.cdiv(n_scalar, META["BLOCK_SIZE"]),)
            copy_contig_kernel[grid](x_flat, y_flat, n_scalar)
            return

        # real generic strided
        if not is_complex:
            grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
            copy_strided_real_kernel[grid](x, y, n, incx, incy)
            return

        # ccopy generic strided
        x_flat = torch.view_as_real(x).reshape(-1)
        y_flat = torch.view_as_real(y).reshape(-1)
        grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
        copy_strided_complex_kernel[grid](x_flat, y_flat, n, incx, incy)
        return


# -----------------------------
# Public APIs
# -----------------------------

def scopy(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS SCOPY OPT")
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    _copy_impl(n, x, incx, y, incy, is_complex=False)


def dcopy(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS DCOPY OPT")
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    _copy_impl(n, x, incx, y, incy, is_complex=False)


def ccopy(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS CCOPY OPT")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    _copy_impl(n, x, incx, y, incy, is_complex=True)


def zcopy(n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int) -> None:
    logger.debug("FLAG_BLAS ZCOPY OPT KERNEL ONLY")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    _copy_impl(n, x, incx, y, incy, is_complex=True)
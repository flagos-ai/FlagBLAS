import logging

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

MAX_NUM_BLOCKS = 1024


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
    ],
    key=["n", "INCX"],
)
@triton.jit
def amax_kernel1_real(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_ctas = tl.num_programs(0)

    dtype = x_ptr.dtype.element_ty
    local_max_val = tl.max(tl.zeros([1], dtype=dtype), axis=0) - 1.0
    local_max_idx = tl.max(tl.zeros([1], dtype=tl.int32), axis=0) + 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start < n:
        idx = block_start + tl.arange(0, BLOCK_SIZE)
        idx = idx.to(tl.int32)
        mask = idx < n
        x_offset = idx * INCX
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        abs_x = tl.abs(x)

        block_max = tl.max(abs_x, axis=0)
        is_max_in_chunk = (abs_x == block_max) & mask
        candidate_idx = tl.where(is_max_in_chunk, idx, 2147483647)
        chunk_best_idx = tl.min(candidate_idx, axis=0).to(tl.int32)

        new_max = tl.maximum(local_max_val, block_max)
        use_new = (block_max > local_max_val) | (
            (block_max == local_max_val) & (chunk_best_idx < local_max_idx)
        )
        local_max_idx = tl.where(use_new, chunk_best_idx, local_max_idx)
        local_max_val = new_max

        block_start += num_ctas * BLOCK_SIZE

    tl.store(mid_val_ptr + pid, local_max_val)
    tl.store(mid_idx_ptr + pid, local_max_idx)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
    ],
    key=["n", "INCX"],
)
@triton.jit
def amax_kernel1_complex(
    x_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    n,
    INCX,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_ctas = tl.num_programs(0)

    dtype = x_ptr.dtype.element_ty
    local_max_val = tl.max(tl.zeros([1], dtype=dtype), axis=0) - 1.0
    local_max_idx = tl.max(tl.zeros([1], dtype=tl.int32), axis=0) + 2147483647

    block_start = pid * BLOCK_SIZE
    while block_start < n:
        idx = block_start + tl.arange(0, BLOCK_SIZE)
        idx = idx.to(tl.int32)
        mask = idx < n
        x_base_offset = idx * INCX * 2
        x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
        x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)
        abs_x = tl.abs(x_real) + tl.abs(x_imag)

        block_max = tl.max(abs_x, axis=0)
        is_max_in_chunk = (abs_x == block_max) & mask
        candidate_idx = tl.where(is_max_in_chunk, idx, 2147483647)
        chunk_best_idx = tl.min(candidate_idx, axis=0).to(tl.int32)

        new_max = tl.maximum(local_max_val, block_max)
        use_new = (block_max > local_max_val) | (
            (block_max == local_max_val) & (chunk_best_idx < local_max_idx)
        )
        local_max_idx = tl.where(use_new, chunk_best_idx, local_max_idx)
        local_max_val = new_max

        block_start += num_ctas * BLOCK_SIZE

    tl.store(mid_val_ptr + pid, local_max_val)
    tl.store(mid_idx_ptr + pid, local_max_idx)


@libentry()
@triton.jit
def amax_kernel2(
    mid_val_ptr,
    mid_idx_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid_val_ptr + offset, mask=mask, other=-1.0)
    mid_idx = tl.load(mid_idx_ptr + offset, mask=mask, other=2147483647)

    max_val = tl.max(mid_val, axis=0)

    is_max = (mid_val == max_val) & mask
    candidate_idx = tl.where(is_max, mid_idx, 2147483647)
    final_idx = tl.min(candidate_idx, axis=0)

    tl.store(out_ptr, final_idx + 1)


def _amax_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    result: torch.Tensor,
    is_complex: bool,
    val_dtype: torch.dtype,
) -> None:
    if n <= 0 or incx <= 0:
        result.zero_()
        return

    required_size = 1 + (n - 1) * incx
    assert (
        x.numel() >= required_size
    ), f"x is too short: need at least {required_size} elements for n={n}, incx={incx}, got {x.numel()}"

    assert x.is_contiguous(), "x must be contiguous"
    result.zero_()

    grid_size = min(triton.cdiv(n, 256), MAX_NUM_BLOCKS)
    block_mid = triton.next_power_of_2(grid_size)

    mid_val = torch.empty((grid_size,), dtype=val_dtype, device=x.device)
    mid_idx = torch.empty((grid_size,), dtype=torch.int32, device=x.device)

    with torch_device_fn.device(x.device):
        if is_complex:
            x_real = torch.view_as_real(x)
            amax_kernel1_complex[(grid_size, 1, 1)](x_real, mid_val, mid_idx, n, incx)
        else:
            amax_kernel1_real[(grid_size, 1, 1)](x, mid_val, mid_idx, n, incx)

        amax_kernel2[(1, 1, 1)](mid_val, mid_idx, result, grid_size, block_mid)


def samax(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SAMAX")
    assert x.dtype == torch.float32, "x must be float32"
    assert result.dtype == torch.int32, "result must be int32"
    _amax_impl(n, x, incx, result, is_complex=False, val_dtype=torch.float32)


def damax(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DAMAX")
    assert x.dtype == torch.float64, "x must be float64"
    assert result.dtype == torch.int32, "result must be int32"
    _amax_impl(n, x, incx, result, is_complex=False, val_dtype=torch.float64)


def camax(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS CAMAX")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert result.dtype == torch.int32, "result must be int32"
    _amax_impl(n, x, incx, result, is_complex=True, val_dtype=torch.float32)


def zamax(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ZAMAX")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert result.dtype == torch.int32, "result must be int32"
    _amax_impl(n, x, incx, result, is_complex=True, val_dtype=torch.float64)

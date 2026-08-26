import importlib

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .trsv import ctrsv, strsv

_common = importlib.import_module("flag_blas.ops.level2.tpsv")

_MAX_CORE_DIM = 65535


@libentry()
@triton.jit
def unpack_tpsv_kernel(
    ap_ptr,
    a_ptr,
    n,
    UPLO: tl.constexpr,
    COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total = n * n
    step = tl.num_programs(0) * BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    while block_start < total:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        in_bounds = offsets < total
        safe_offsets = tl.where(in_bounds, offsets, 0)
        row = safe_offsets // n
        col = safe_offsets % n
        if UPLO == 1:
            valid = (row <= col) & in_bounds
            packed = row * n - row * (row + 1) // 2 + col
        else:
            valid = (row >= col) & in_bounds
            packed = row * (row + 1) // 2 + col
        if COMPLEX:
            ar = tl.load(ap_ptr + packed * 2, mask=valid, other=0.0)
            ai = tl.load(ap_ptr + packed * 2 + 1, mask=valid, other=0.0)
            tl.store(a_ptr + offsets * 2, ar, mask=in_bounds)
            tl.store(a_ptr + offsets * 2 + 1, ai, mask=in_bounds)
        else:
            value = tl.load(ap_ptr + packed, mask=valid, other=0.0)
            tl.store(a_ptr + offsets, value, mask=in_bounds)
        block_start += step


def _unpack_grid(n, block_size):
    return (min(triton.cdiv(n * n, block_size), _MAX_CORE_DIM),)


def _dense_stpsv(uplo, trans, diag, n, AP, x, incx):
    A = torch.empty((n, n), dtype=AP.dtype, device=AP.device)
    block_size = 256
    unpack_tpsv_kernel[_unpack_grid(n, block_size)](
        AP, A, n, UPLO=uplo, COMPLEX=False, BLOCK_SIZE=block_size
    )
    strsv(uplo, trans, diag, n, A, n, x, incx)


def _dense_ctpsv(uplo, trans, diag, n, AP, x, incx):
    A = torch.empty((n, n), dtype=AP.dtype, device=AP.device)
    block_size = 256
    unpack_tpsv_kernel[_unpack_grid(n, block_size)](
        torch.view_as_real(AP),
        torch.view_as_real(A),
        n,
        UPLO=uplo,
        COMPLEX=True,
        BLOCK_SIZE=block_size,
    )
    ctrsv(uplo, trans, diag, n, A, n, x, incx)


def stpsv(uplo, trans, diag, n, AP, x, incx):
    assert AP.dtype == torch.float32 == x.dtype
    _common._check_common(uplo, trans, diag, n, AP, x, incx)
    if n == 0:
        return x
    if n >= 1024:
        _dense_stpsv(uplo, trans, diag, n, AP, x, incx)
        return x
    uplo, trans, _ = _common._row_major_tpsv_args(uplo, trans)
    with torch_device_fn.device(AP.device):
        _common._real_tpsv_kernel[(1,)](uplo, trans, diag, n, AP, x, incx)
    return x


def ctpsv(uplo, trans, diag, n, AP, x, incx):
    assert AP.dtype == torch.complex64 == x.dtype
    _common._check_common(uplo, trans, diag, n, AP, x, incx)
    if n == 0:
        return x
    if n >= 1024:
        _dense_ctpsv(uplo, trans, diag, n, AP, x, incx)
        return x
    uplo, trans, conj = _common._row_major_tpsv_args(uplo, trans)
    with torch_device_fn.device(AP.device):
        _common._complex_tpsv_kernel[(1,)](
            uplo,
            trans,
            diag,
            n,
            torch.view_as_real(AP),
            torch.view_as_real(x),
            incx,
            CONJ=conj,
        )
    return x


__all__ = ["stpsv", "ctpsv"]

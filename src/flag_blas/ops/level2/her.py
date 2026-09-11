# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import struct
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, torch.Tensor]

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1


def _f64_to_i64(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", value))[0]


def _row_major_uplo(uplo: int) -> int:
    return (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )


@libentry()
@triton.jit
def _her_kernel(
    x,
    A,
    n: tl.constexpr,
    alpha,
    incx: tl.constexpr,
    lda: tl.constexpr,
    uplo: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    major = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    minor = pid - major * (major + 1) // 2
    if uplo == 0:
        pid_m = major
        pid_n = minor
    else:
        pid_m = minor
        pid_n = major

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < n) & (offs_n[None, :] < n)

    if uplo == 1:
        mask = mask & (offs_m[:, None] <= offs_n[None, :])
    else:
        mask = mask & (offs_m[:, None] >= offs_n[None, :])

    x_m = tl.load(x + offs_m * incx * 2, mask=offs_m < n, other=0.0)
    y_m = tl.load(x + offs_m * incx * 2 + 1, mask=offs_m < n, other=0.0)
    x_n = tl.load(x + offs_n * incx * 2, mask=offs_n < n, other=0.0)
    y_n = tl.load(x + offs_n * incx * 2 + 1, mask=offs_n < n, other=0.0)

    prod_r = x_m[:, None] * x_n[None, :] + y_m[:, None] * y_n[None, :]
    prod_i = x_m[:, None] * y_n[None, :] - y_m[:, None] * x_n[None, :]

    a_off = (offs_m[:, None] + offs_n[None, :] * lda) * 2
    old_r = tl.load(A + a_off, mask=mask, other=0.0)
    old_i = tl.load(A + a_off + 1, mask=mask, other=0.0)

    if IS_DOUBLE:
        alpha_value = alpha.to(tl.float64, bitcast=True)
    else:
        alpha_value = alpha
    out_r = old_r + alpha_value * prod_r
    out_i = old_i + alpha_value * prod_i
    diag = offs_m[:, None] == offs_n[None, :]
    out_i = tl.where(diag, 0.0, out_i)

    tl.store(A + a_off, out_r, mask=mask)
    tl.store(A + a_off + 1, out_i, mask=mask)


@libentry()
@triton.jit
def _zher_vector_kernel(
    x,
    A,
    n: tl.constexpr,
    alpha,
    incx: tl.constexpr,
    lda: tl.constexpr,
    uplo: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    major = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    minor = pid - major * (major + 1) // 2
    if uplo == 0:
        pid_m = major
        pid_n = minor
    else:
        pid_m = minor
        pid_n = major

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < n) & (offs_n[None, :] < n)
    if uplo == 1:
        mask = mask & (offs_m[:, None] <= offs_n[None, :])
    else:
        mask = mask & (offs_m[:, None] >= offs_n[None, :])

    x_mr = tl.load(x + offs_m * incx * 2, mask=offs_m < n, other=0.0)
    x_mi = tl.load(x + offs_m * incx * 2 + 1, mask=offs_m < n, other=0.0)
    x_nr = tl.load(x + offs_n * incx * 2, mask=offs_n < n, other=0.0)
    x_ni = tl.load(x + offs_n * incx * 2 + 1, mask=offs_n < n, other=0.0)

    prod_r = x_mr[:, None] * x_nr[None, :] + x_mi[:, None] * x_ni[None, :]
    prod_i = x_mr[:, None] * x_ni[None, :] - x_mi[:, None] * x_nr[None, :]
    alpha_value = alpha.to(tl.float64, bitcast=True)

    a_off = (offs_m[:, None] + offs_n[None, :] * lda) * 2
    components = tl.arange(0, 2)
    complex_off = a_off[:, :, None] + components[None, None, :]
    complex_mask = mask[:, :, None]
    old = tl.load(A + complex_off, mask=complex_mask, other=0.0)
    is_real = components[None, None, :] == 0
    prod = tl.where(is_real, prod_r[:, :, None], prod_i[:, :, None])
    out = old + alpha_value * prod
    diag = offs_m[:, None] == offs_n[None, :]
    out = tl.where(diag[:, :, None] & ~is_real, 0.0, out)
    tl.store(A + complex_off, out, mask=complex_mask)


def _check_her_args(name, uplo, n, alpha, x, incx, A, lda, dtype, alpha_dtype):
    assert uplo in (
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_FILL_MODE_UPPER,
    ), "uplo must be CUBLAS_FILL_MODE_LOWER or CUBLAS_FILL_MODE_UPPER"
    assert isinstance(n, int) and n >= 0, "n must be a non-negative integer"
    assert A.dtype == dtype, f"A must be {dtype} for {name}"
    assert x.dtype == dtype, f"x must be {dtype} for {name}"
    assert (
        (A.is_cuda and x.is_cuda)
        or (A.device.type == "npu" and x.device.type == "npu")
        or (
            runtime.device.vendor_name == "mthreads"
            and A.device.type == "musa"
            and x.device.type == "musa"
        )
    ), "A and x must be CUDA, NPU, or active MUSA tensors"
    assert A.is_contiguous() and x.is_contiguous(), "A and x must be contiguous"
    assert A.device == x.device, "A and x must be on the same device"
    assert A.ndim == 2, "A must be a 2-D tensor"
    assert x.ndim == 1, "x must be a 1-D tensor"
    assert isinstance(incx, int) and incx > 0, "incx must be a positive integer"
    assert isinstance(lda, int) and lda >= max(1, n), "lda must be at least max(1, n)"
    assert A.shape[0] >= n, "A has too few rows"
    assert A.shape[1] >= lda, "A leading dimension is too small"
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx, "x is too small for n and incx"
        assert A.numel() >= n * lda, "A is too small for n and lda"
    if isinstance(alpha, torch.Tensor):
        assert alpha.numel() == 1, "alpha tensor must contain one value"
        assert alpha.dtype == alpha_dtype, f"alpha tensor must be {alpha_dtype}"
    else:
        assert isinstance(alpha, (float, int)), "alpha must be real"


def _her_impl(name, uplo, n, alpha, x, incx, A, lda, dtype, alpha_dtype):
    _check_her_args(name, uplo, n, alpha, x, incx, A, lda, dtype, alpha_dtype)
    if n == 0:
        return A
    uplo = _row_major_uplo(uplo)

    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    is_double = dtype == torch.complex128
    kernel_alpha = _f64_to_i64(alpha_value) if is_double else alpha_value
    x_real = torch.view_as_real(x).reshape(-1)
    A_real = torch.view_as_real(A).reshape(-1)
    block_m = 32 if is_double and n >= 512 else 16
    block_n = block_m
    tile_count = triton.cdiv(n, block_m)
    grid = (tile_count * (tile_count + 1) // 2,)
    with torch_device_fn.device(A.device):
        if is_double:
            _zher_vector_kernel[grid](
                x_real,
                A_real,
                n,
                kernel_alpha,
                incx,
                lda,
                uplo,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        else:
            _her_kernel[grid](
                x_real,
                A_real,
                n,
                kernel_alpha,
                incx,
                lda,
                uplo,
                IS_DOUBLE=False,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
    return A


def cher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _her_impl(
        "cher", uplo, n, alpha, x, incx, A, lda, torch.complex64, torch.float32
    )


def zher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _her_impl(
        "zher", uplo, n, alpha, x, incx, A, lda, torch.complex128, torch.float64
    )

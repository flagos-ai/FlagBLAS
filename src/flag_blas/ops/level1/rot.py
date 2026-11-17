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

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, torch.Tensor]


def _f64_to_i64(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", value))[0]


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
def csrot_kernel(
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

    x_base_offset = idx * INCX * 2
    y_base_offset = idx * INCY * 2

    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)
    y_real = tl.load(y_ptr + y_base_offset, mask=mask, other=0.0)
    y_imag = tl.load(y_ptr + y_base_offset + 1, mask=mask, other=0.0)

    new_x_real = c * x_real + s * y_real
    new_x_imag = c * x_imag + s * y_imag

    new_y_real = -s * x_real + c * y_real
    new_y_imag = -s * x_imag + c * y_imag

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
def zdrot_kernel(
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

    x_base_offset = idx * INCX * 2
    y_base_offset = idx * INCY * 2

    x_real = tl.load(x_ptr + x_base_offset, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + x_base_offset + 1, mask=mask, other=0.0)
    y_real = tl.load(y_ptr + y_base_offset, mask=mask, other=0.0)
    y_imag = tl.load(y_ptr + y_base_offset + 1, mask=mask, other=0.0)

    new_x_real = c * x_real + s * y_real
    new_x_imag = c * x_imag + s * y_imag

    new_y_real = -s * x_real + c * y_real
    new_y_imag = -s * x_imag + c * y_imag

    tl.store(x_ptr + x_base_offset, new_x_real, mask=mask)
    tl.store(x_ptr + x_base_offset + 1, new_x_imag, mask=mask)
    tl.store(y_ptr + y_base_offset, new_y_real, mask=mask)
    tl.store(y_ptr + y_base_offset + 1, new_y_imag, mask=mask)


def srot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    c: ScalarType,
    s: ScalarType,
) -> None:
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
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert (
        y.numel() >= req_size_y
    ), f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        srot_kernel[grid](x, y, c, s, n, incx, incy)


def drot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    c: ScalarType,
    s: ScalarType,
) -> None:
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
    c_int = _f64_to_i64(c_val)
    s_int = _f64_to_i64(s_val)

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert (
        y.numel() >= req_size_y
    ), f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        drot_kernel[grid](x, y, c_int, s_int, n, incx, incy)


def csrot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    c: ScalarType,
    s: ScalarType,
) -> None:
    logger.debug("FLAG_BLAS CSROT")

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

    c = float(c.item() if isinstance(c, torch.Tensor) else c)
    s = float(s.item() if isinstance(s, torch.Tensor) else s)

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert (
        y.numel() >= req_size_y
    ), f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        csrot_kernel[grid](x_real, y_real, c, s, n, incx, incy)


def zdrot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    c: ScalarType,
    s: ScalarType,
) -> None:
    logger.debug("FLAG_BLAS ZDROT")

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
    s = float(s.item() if isinstance(s, torch.Tensor) else s)

    c_int = _f64_to_i64(c_val)
    s_int = _f64_to_i64(s)

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert (
        x.numel() >= req_size_x
    ), f"x is too short: need {req_size_x} elements for n={n}, incx={incx}"
    assert (
        y.numel() >= req_size_y
    ), f"y is too short: need {req_size_y} elements for n={n}, incy={incy}"

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        zdrot_kernel[grid](x_real, y_real, c_int, s_int, n, incx, incy)

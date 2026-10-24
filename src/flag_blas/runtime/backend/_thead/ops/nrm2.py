# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .dot import _zero_result


@libentry()
@triton.jit
def _nrm2_single(
    X,
    OUT,
    N: tl.constexpr,
    INC: tl.constexpr,
    COMPLEX: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    if COMPLEX and FLAT:
        x = tl.load(X + (i // 2) * (2 * INC) + i % 2, i < 2 * N, other=0)
        total = tl.sum(x * x, 0)
    elif COMPLEX:
        r = tl.load(X + i * INC * 2, i < N, other=0)
        v = tl.load(X + i * INC * 2 + 1, i < N, other=0)
        total = tl.sum(r * r + v * v, 0)
    else:
        x = tl.load(X + i * INC, i < N, other=0)
        total = tl.sum(x * x, 0)
    tl.store(OUT, tl.sqrt(total))


@libentry()
@triton.jit
def _sum_atomic(
    X,
    OUT,
    N: tl.constexpr,
    INC: tl.constexpr,
    SQUARE: tl.constexpr,
    COMPLEX: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    count = N * 2 if COMPLEX and FLAT else N
    tiles = tl.cdiv(count, BLOCK)
    if CHUNK:
        chunk = tl.cdiv(tiles, grid) | 1
        begin, end, step = pid * chunk, tl.minimum((pid + 1) * chunk, tiles), 1
    else:
        begin, end, step = pid, tiles, grid
    lane = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, OUT.dtype.element_ty)
    for tile in tl.range(
        begin, end, step, num_stages=PIPE_STAGES, loop_unroll_factor=1
    ):
        i = tile * BLOCK + lane
        if COMPLEX and FLAT:
            v = tl.load(X + (i // 2) * INC * 2 + i % 2, i < count, other=0)
            value = v * v if SQUARE else tl.abs(v)
        elif COMPLEX:
            re = tl.load(X + i * INC * 2, i < N, other=0)
            im = tl.load(X + i * INC * 2 + 1, i < N, other=0)
            value = re * re + im * im if SQUARE else tl.abs(re) + tl.abs(im)
        else:
            v = tl.load(X + i * INC, i < N, other=0)
            value = v * v if SQUARE else tl.abs(v)
        acc += value
    total = tl.sum(acc, 0)
    tl.atomic_add(OUT, total, sem="relaxed")


@libentry()
@triton.jit
def _sqrt_output(OUT):
    tl.store(OUT, tl.sqrt(tl.load(OUT)))


_common_nrm2 = import_module("flag_blas.ops.level1.nrm2")
_common_asum = import_module("flag_blas.ops.level1.asum")


def _run_reduction(n, x, incx, result, square):
    complex_input = x.is_complex()
    # Keep already competitive large real reductions and small ASUM paths.
    if (square and not complex_input and n > 1048576) or (not square and n <= 131072):
        common = _common_nrm2 if square else _common_asum
        if square:
            return common._nrm2_impl(n, x, incx, result, complex_input)
        return common._asum_impl(n, x, incx, result, complex_input)
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, (
        f"x is too short: need at least {required_size} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert x.is_contiguous(), "x must be contiguous"
    if square:
        assert (
            result.dim() == 1 and result.numel() == 1
        ), "result must be a 1-element tensor"
    assert x.device == result.device, "x and result must be on the same device"

    raw = torch.view_as_real(x) if complex_input else x
    options = dict(num_stages=1, enable_fp_fusion=False)
    with torch_device_fn.device(x.device):
        if n <= 8192:
            flat = x.dtype == torch.complex64
            block = max(32, triton.next_power_of_2(n * (2 if flat else 1)))
            warps = 1 if n <= 128 else 4
            _nrm2_single[(1,)](
                raw,
                result,
                n,
                incx,
                complex_input,
                flat,
                block,
                num_warps=warps,
                **options,
            )
            return
        double = x.dtype == torch.complex128
        block = (
            (256 if incx > 1 and (n < 2097152 or incx >= 4) else 1024)
            if double
            else (2048 if incx < 4 else 1024)
        )
        flat = complex_input and (incx == 1 or not double)
        chunk = n < 16777216
        cap = 64 if double and 2097152 <= n < 8388608 and incx == 3 else 128
        if n <= 131072:
            block, cap, chunk = 1024, (32 if incx > 1 else 128), False
            flat = complex_input and not double
        elif double and incx == 1 and n >= 8388608:
            block, flat, chunk = 2048, False, True
        elif n >= 8388608:
            block = 4096 if chunk else (8192 if incx == 1 else 2048)
            flat = complex_input
        use_pipeline = (
            complex_input
            and incx > 1
            and (
                (not double and n >= 2097152 and (chunk or incx >= 3))
                or (double and n >= 1048576 and incx >= 4)
            )
        )
        if use_pipeline:
            options["num_stages"] = 2
        count = n * (2 if flat else 1)
        grid = min(triton.cdiv(count, block), cap)
        _zero_result[(1,)](result, 1, num_warps=1)
        _sum_atomic[(grid,)](
            raw,
            result,
            n,
            incx,
            square,
            complex_input,
            flat,
            block,
            chunk,
            options["num_stages"],
            num_warps=4,
            **options,
        )
        if square:
            _sqrt_output[(1,)](result, num_warps=1)


def snrm2(n, x, incx, result):
    assert x.dtype == torch.float32, "x must be float32"
    assert result.dtype == torch.float32, "result must be float32"
    _run_reduction(n, x, incx, result, True)


def dnrm2(n, x, incx, result):
    assert x.dtype == torch.float64, "x must be float64"
    assert result.dtype == torch.float64, "result must be float64"
    _run_reduction(n, x, incx, result, True)


def scnrm2(n, x, incx, result):
    assert x.dtype == torch.complex64, "x must be complex64"
    assert result.dtype == torch.float32, "result for scnrm2 must be float32"
    _run_reduction(n, x, incx, result, True)


def dznrm2(n, x, incx, result):
    assert x.dtype == torch.complex128, "x must be complex128"
    assert result.dtype == torch.float64, "result for dznrm2 must be float64"
    _run_reduction(n, x, incx, result, True)

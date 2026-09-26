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

import importlib

import torch
import triton

from flag_blas.runtime import torch_device_fn

_common = importlib.import_module("flag_blas.ops.level2.spmv")


def _spmv_configs():
    # Small row tiles expose more parallelism on the PPU. Larger reduction
    # tiles reduce the serial loop count compared with the generic configs.
    return [
        triton.Config({"BLOCK_SIZE_M": m, "BLOCK_K": k}, num_warps=4, num_stages=1)
        for m in (4, 8)
        for k in (128, 256, 512)
    ]


# Reuse the generic kernels unchanged, with independent PPU autotune caches.
sspmv_kernel = triton.autotune(
    configs=_spmv_configs(),
    key=["n", "uplo_key", "INCX", "INCY"],
    restore_value=["y_ptr"],
)(_common.sspmv_kernel.fn)

dspmv_kernel = triton.autotune(
    configs=_spmv_configs(),
    key=["n", "uplo_key", "INCX", "INCY"],
    restore_value=["y_ptr"],
)(_common.dspmv_kernel.fn)


def _spmv(dtype, kernel, uplo, n, alpha, AP, x, incx, beta, y, incy):
    assert AP.dtype == dtype == x.dtype == y.dtype
    _common._check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return

    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha == 0.0:
        y_view = _common._strided_y(y, n, incy)
        if beta == 0.0:
            y_view.zero_()
        elif beta != 1.0:
            y_view.mul_(beta)
        return

    uplo = _common._row_major_uplo(uplo)
    beta_is_zero = beta == 0.0
    if dtype == torch.float64:
        alpha = _common._f64_to_i64(alpha)
        beta = _common._f64_to_i64(beta)
    with torch_device_fn.device(AP.device):
        kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)](
            AP,
            x,
            y,
            alpha,
            beta,
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_is_zero,
        )


def sspmv(uplo, n, alpha, AP, x, incx, beta, y, incy):
    return _spmv(
        torch.float32, sspmv_kernel, uplo, n, alpha, AP, x, incx, beta, y, incy
    )


def dspmv(uplo, n, alpha, AP, x, incx, beta, y, incy):
    return _spmv(
        torch.float64, dspmv_kernel, uplo, n, alpha, AP, x, incx, beta, y, incy
    )

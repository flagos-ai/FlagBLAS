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

"""MUSA implementations for GEMV variants that need vendor lowering.

TorchMUSA routes ``torch.matmul`` to muBLAS.  This is both faster for SGEMV
and more reliable for FP16/BF16 than the corresponding kernels emitted by the
currently supported Triton-MUSA compiler.
"""

from typing import Union

import torch

from flag_blas.ops.level2._constants import CUBLAS_OP_N, CUBLAS_OP_T

ScalarType = Union[float, int, torch.Tensor]


def _logical_matrix(A: torch.Tensor, m: int, n: int, lda: int) -> torch.Tensor:
    required = 0 if m == 0 or n == 0 else (m - 1) * lda + n
    assert A.numel() >= required
    return A.reshape(-1).as_strided((m, n), (lda, 1))


def _gemv(
    trans: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
    dtype: torch.dtype,
    fp32_accumulation: bool,
) -> None:
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == x.dtype == y.dtype == dtype
    assert A.device == x.device == y.device
    assert A.device.type == "musa"
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T)
    assert isinstance(m, int) and isinstance(n, int) and m >= 0 and n >= 0
    assert isinstance(lda, int) and lda >= max(1, n)
    assert isinstance(incx, int) and isinstance(incy, int) and incx > 0 and incy > 0

    if m == 0 or n == 0:
        return

    x_len = n if trans == CUBLAS_OP_N else m
    y_len = m if trans == CUBLAS_OP_N else n
    assert x.numel() >= 1 + (x_len - 1) * incx
    assert y.numel() >= 1 + (y_len - 1) * incy

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    y_view = y[: 1 + (y_len - 1) * incy : incy]
    if alpha == 0.0:
        if beta == 0.0:
            y_view.zero_()
        elif beta != 1.0:
            y_view.mul_(beta)
        return

    matrix = _logical_matrix(A, m, n, lda)
    if trans == CUBLAS_OP_T:
        matrix = matrix.transpose(0, 1)
    x_view = x[: 1 + (x_len - 1) * incx : incx]

    if (
        not fp32_accumulation
        and alpha == 1.0
        and beta == 0.0
        and incx == 1
        and incy == 1
    ):
        torch.matmul(matrix, x_view, out=y_view)
        return

    if fp32_accumulation:
        values = torch.matmul(matrix.float(), x_view.float())
    else:
        values = torch.matmul(matrix, x_view)
    values.mul_(alpha)
    if beta != 0.0:
        values.add_(y_view if not fp32_accumulation else y_view.float(), alpha=beta)
    y_view.copy_(values.to(dtype=dtype))


def sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) -> None:
    _gemv(
        trans,
        m,
        n,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
        torch.float32,
        False,
    )


def hgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) -> None:
    _gemv(
        trans,
        m,
        n,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
        torch.float16,
        True,
    )


def bfgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) -> None:
    _gemv(
        trans,
        m,
        n,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
        torch.bfloat16,
        True,
    )

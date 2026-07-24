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
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


def _complex_scalar_parts(value: ScalarType):
    value = value.item() if isinstance(value, torch.Tensor) else value
    value = complex(value)
    return float(value.real), float(value.imag)


def _validate_cgemm_args(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.complex64
    assert B.dtype == torch.complex64
    assert C.dtype == torch.complex64
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]

    def required_numel(rows: int, cols: int, ld: int) -> int:
        if rows == 0 or cols == 0:
            return 0
        return (rows - 1) * ld + cols

    if transa == CUBLAS_OP_N:
        if m > 0 and k > 0:
            assert lda >= k
        assert A.numel() >= required_numel(m, k, lda)
    else:
        if m > 0 and k > 0:
            assert lda >= m
        assert A.numel() >= required_numel(k, m, lda)

    if transb == CUBLAS_OP_N:
        if n > 0 and k > 0:
            assert ldb >= n
        assert B.numel() >= required_numel(k, n, ldb)
    else:
        if n > 0 and k > 0:
            assert ldb >= k
        assert B.numel() >= required_numel(n, k, ldb)

    if m > 0 and n > 0:
        assert ldc >= n
    assert C.numel() >= required_numel(m, n, ldc)


@triton.jit
def _cgemm_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < k
        if TRANS_A == 0:
            a_elem = offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_elem = cur_k[None, :] * lda + offs_m[:, None]

        if TRANS_B == 0:
            b_elem = cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_elem = offs_n[None, :] * ldb + cur_k[:, None]

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        ar = tl.load(a_ptr + 2 * a_elem, mask=a_mask, other=0.0)
        ai = tl.load(a_ptr + 2 * a_elem + 1, mask=a_mask, other=0.0)
        br = tl.load(b_ptr + 2 * b_elem, mask=b_mask, other=0.0)
        bi = tl.load(b_ptr + 2 * b_elem + 1, mask=b_mask, other=0.0)
        if TRANS_A == 2:
            ai = -ai
        if TRANS_B == 2:
            bi = -bi

        acc_r += tl.dot(
            ar, br, out_dtype=tl.float32, input_precision="tf32x3"
        ) - tl.dot(ai, bi, out_dtype=tl.float32, input_precision="tf32x3")
        acc_i += tl.dot(
            ar, bi, out_dtype=tl.float32, input_precision="tf32x3"
        ) + tl.dot(ai, br, out_dtype=tl.float32, input_precision="tf32x3")

    if ALPHA_IS_ONE:
        out_r = acc_r
        out_i = acc_i
    else:
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r

    c_elem = offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if not BETA_IS_ZERO:
        cr = tl.load(c_ptr + 2 * c_elem, mask=c_mask, other=0.0)
        ci = tl.load(c_ptr + 2 * c_elem + 1, mask=c_mask, other=0.0)
        out_r += beta_r * cr - beta_i * ci
        out_i += beta_r * ci + beta_i * cr

    tl.store(c_ptr + 2 * c_elem, out_r, mask=c_mask)
    tl.store(c_ptr + 2 * c_elem + 1, out_i, mask=c_mask)


def _select_cgemm_config(transa: int, transb: int, m: int, n: int, k: int):
    max_dim = max(m, n, k)
    if max_dim <= 32:
        return 16, 16, 16, 4, 1
    if max_dim <= 128:
        return 16, 16, 32, 4, 1
    if max_dim <= 512:
        return 32, 32, 32, 4, 1
    return 32, 64, 32, 4, 4


def cgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    _validate_cgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha_r, alpha_i = _complex_scalar_parts(alpha)
    beta_r, beta_i = _complex_scalar_parts(beta)

    if m == 0 or n == 0 or k == 0 or (alpha_r == 0.0 and alpha_i == 0.0):
        if beta_r == 0.0 and beta_i == 0.0:
            C.zero_()
        elif not (beta_r == 1.0 and beta_i == 0.0):
            C.mul_(complex(beta_r, beta_i))
        return

    beta_is_zero = beta_r == 0.0 and beta_i == 0.0
    alpha_is_one = alpha_r == 1.0 and alpha_i == 0.0
    block_m, block_n, block_k, num_warps, group_m = _select_cgemm_config(
        transa, transb, m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    with torch_device_fn.device(A.device):
        _cgemm_dot_kernel[grid](
            A_real,
            B_real,
            C_real,
            alpha_r,
            alpha_i,
            beta_r,
            beta_i,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            transa,
            transb,
            beta_is_zero,
            alpha_is_one,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
        )

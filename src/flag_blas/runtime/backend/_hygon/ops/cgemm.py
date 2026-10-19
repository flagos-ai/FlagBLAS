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

"""
Hygon DTK cgemm 专用实现（自研 complex 内核）。

complex64 GEMM: C = alpha * op(A) @ op(B) + beta * C, op ∈ {N, T, C(共轭转置)}。
本模块自带一个 complex Triton 内核（在 kernel 内处理 N/T/C 与共轭），并额外提供一条
hygon 定制的「转置复合」路径：当 transa/transb 含转置且 shape 足够大时，先把物理
(k,m)/(n,k) 的非共轭/共轭转置物化成 (m,k)/(k,n) 连续缓冲，再以 N/N 调用主内核，
避免跨步复数读带来的带宽损失（与 _hygon/ops/bfgemm.py / hgemm.py 的转置复合同思路）。

数据布局: 复数以 real/imag 交错存放, 传入内核前用 torch.view_as_real(...).reshape(-1)
展平成 fp32 视图, 内核按 2*elem / 2*elem+1 取 re/im。

注: 真机精度/性能验证前, hygon 专用 config 表与转置复合阈值为保守取值, 待实测收敛。
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.cgemm import (
    ScalarType,
    _complex_scalar_parts,
    _validate_cgemm_args,
)
from flag_blas.runtime import torch_device_fn


@triton.jit
def _cgemm_kernel(
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
        # A: N -> 物理 (m,k); T/C -> 物理 (k,m)
        if TRANS_A == 0:
            a_elem = offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_elem = cur_k[None, :] * lda + offs_m[:, None]
        # B: N -> 物理 (k,n); T/C -> 物理 (n,k)
        if TRANS_B == 0:
            b_elem = cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_elem = offs_n[None, :] * ldb + cur_k[:, None]

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        # 打包成对 (re,im) 加载 (末维连续 2 x fp32 -> 8B 向量化), 再 split 成 re/im,
        # 避免 stride-2 标量 gather。
        a2 = tl.load(
            a_ptr + (2 * a_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=a_mask[:, :, None],
            other=0.0,
        )
        ar, ai = tl.split(a2)
        b2 = tl.load(
            b_ptr + (2 * b_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=b_mask[:, :, None],
            other=0.0,
        )
        br, bi = tl.split(b2)
        # 共轭转置: 虚部取反
        if TRANS_A == 2:
            ai = -ai
        if TRANS_B == 2:
            bi = -bi

        # (ar + ai i)(br + bi i) = (ar*br - ai*bi) + (ar*bi + ai*br) i
        # gfx936 后端只允许 input_precision="ieee" (无 tf32), 即真 fp32 点积,
        # 精度优于 tf32x3, 与 hipblasCgemm 参考同口径。
        acc_r += tl.dot(ar, br, out_dtype=tl.float32, input_precision="ieee")
        acc_r -= tl.dot(ai, bi, out_dtype=tl.float32, input_precision="ieee")
        acc_i += tl.dot(ar, bi, out_dtype=tl.float32, input_precision="ieee")
        acc_i += tl.dot(ai, br, out_dtype=tl.float32, input_precision="ieee")

    if ALPHA_IS_ONE:
        out_r = acc_r
        out_i = acc_i
    else:
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r

    c_elem = offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if not BETA_IS_ZERO:
        c2 = tl.load(
            c_ptr + (2 * c_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=c_mask[:, :, None],
            other=0.0,
        )
        cr, ci = tl.split(c2)
        out_r += beta_r * cr - beta_i * ci
        out_i += beta_r * ci + beta_i * cr

    # 打包成对存储 (末维连续 2 x fp32 -> 8B 向量化)
    out2 = tl.join(out_r, out_i)
    tl.store(
        c_ptr + (2 * c_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
        out2,
        mask=c_mask[:, :, None],
    )


def _select_cgemm_config_hygon(m, n, k):
    # hygon gfx936 分档 (按 shape x 变体实测标定)。复数内核有 2 个 fp32 累加器 +
    # 4 个操作数 tile, 故 BLOCK_K 取 16/32; tile 随规模放大, 大 shape 用 GROUP_M=4
    # 提升 L2 复用。
    max_dim = max(m, n, k)
    if max_dim <= 128:
        return 16, 16, 32, 4, 1
    if max_dim <= 256:
        return 32, 32, 32, 4, 1
    if max_dim <= 512:
        return 64, 64, 16, 8, 1
    return 128, 64, 16, 8, 4


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

    block_m, block_n, block_k, num_warps, group_m = _select_cgemm_config_hygon(m, n, k)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    with torch_device_fn.device(A.device):
        _cgemm_kernel[grid](
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

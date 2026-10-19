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

import os

import torch
import triton
import triton.language as tl
from triton.runtime import heuristics

from flag_blas import runtime
from flag_blas.ops.level3.sgemm import CUBLAS_OP_N, CUBLAS_OP_T, ScalarType
from flag_blas.ops.level3.sgemm import (
    _sgemm_nn_kernel as common_sgemm_nn_kernel,
    _sgemm_nt_kernel as common_sgemm_nt_kernel,
    _sgemm_tn_kernel as common_sgemm_tn_kernel,
    _sgemm_tt_kernel as common_sgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.dispatch import StaticDispatch
from flag_blas.utils import libentry, libtuner

# per-kernel ITT 编译选项的运行时注入（不改 site-packages Triton 源码）。
# 必须在任何 sgemm kernel launch 前执行一次；注入失败仅失去 per-kernel ITT
# 能力（回落全局默认），不影响其余功能。
try:
    from flag_blas.runtime.backend._hygon._triton_itt import install_triton_itt_patch

    install_triton_itt_patch()
except Exception:  # pragma: no cover - 环境无 Triton HCU 后端时静默跳过
    pass

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@triton.jit
def _sgemm_nn_opt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_am < m
    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < n

    if EVEN_K:
        for i in range(k // BLOCK_K):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
    else:
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for i in range(k_full_iters):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if EVEN_M:
                a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if EVEN_M and EVEN_N:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        cm = offs_cm < m
        cn = offs_cn < n
        c_mask = cm[:, None] & cn[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_nn_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_nn_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_nn"))(_sgemm_nn_opt_kernel))
)

# ---- 大 K shape 专用: 两阶段 split-k (FlagTree tle 方式, 无原子加) ----
# 内核1: 每个 block 计算部分 K 段写入 scratch(SPLIT_K, m, n)
# 内核2: reduce 合并 SPLIT_K 段累加到 C
# 实测 (py310 venv + flagtree 0.6.1): 仅对 k >= 4*max(m,n) 的 shape 有效,
# 2048x2048x16384 sp 0.778 -> 0.791 (+1.7%); 其他大 K shape 无效或变差


@libentry()
@triton.jit
def _sgemm_nn_splitk_kernel(
    a_ptr,
    b_ptr,
    scratch_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    k_per = k // SPLIT_K
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * k_per + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])
    mask_m = offs_am < m
    mask_n = offs_bn < n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(k_per // BLOCK_K):
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        if EVEN_N:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
    if EVEN_K is False:
        kr = k_per % BLOCK_K
        if kr > 0:
            mk = tl.arange(0, BLOCK_K) < kr
            a = tl.load(
                a_ptrs,
                mask=(mask_m[:, None] if not EVEN_M else True) & mk[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=mk[:, None] & (mask_n[None, :] if not EVEN_N else True),
                other=0.0,
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    s_ptrs = (
        scratch_ptr
        + pid_k.to(tl.int64) * (m.to(tl.int64) * n)
        + offs_am[:, None].to(tl.int64) * n
        + offs_bn[None, :].to(tl.int64)
    )
    if EVEN_M and EVEN_N:
        tl.store(s_ptrs, acc)
    else:
        tl.store(s_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@triton.jit
def _sgemm_nn_splitk_reduce_kernel(
    scratch_ptr,
    c_ptr,
    m,
    n,
    alpha: tl.float32,
    beta: tl.float32,
    SPLIT_K: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        s_ptrs = (
            scratch_ptr
            + s * (m.to(tl.int64) * n)
            + offs_am[:, None].to(tl.int64) * n
            + offs_bn[None, :].to(tl.int64)
        )
        acc += tl.load(s_ptrs)
    c_ptrs = (
        c_ptr + offs_am[:, None].to(tl.int64) * n + offs_bn[None, :].to(tl.int64)
    )
    mask = (
        (offs_am[:, None] < m) & (offs_bn[None, :] < n)
        if not (EVEN_M and EVEN_N)
        else None
    )
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=mask)
    else:
        c_vals = tl.load(c_ptrs, mask=mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=mask)


# ---- A 类非对齐 shape 专用: pre-pad + 全 EVEN 内核 ----
# 思路: 将 A(m x k) pad 到 A'(mp x kp), B(k x n) pad 到 B'(kp x np),
# 其中 mp/kp/np 是 BLOCK 的整数倍; 主内核全 EVEN (load 无 mask), 仅 store 对真实 m/n 打 mask.
# 成本: 一次 A/B 合并 pad-copy (内存带宽受限), 换取主循环 mask 开销 ~11% + k remainder ~13%.
# 实测 (py310 venv + flagtree 0.6.1, 对比报告当前 opt 路径):
#   8191³ 0.732 -> 0.860、4095³ 0.618 -> 0.739、4097x8191x4095 0.705 -> 0.805、
#   1023³ 0.608 -> 0.623 (边际); 511³ 反而更差 (0.845 -> 0.638), 由分发条件排除.


@libentry()
@triton.jit
def _sgemm_nn_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    mp,
    np,
    kp,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(mp, CM) * tl.cdiv(kp, CK)
    if pid < nblk_a:
        # A tile: 真实 m x k -> pad (mp x kp), 越界补零
        bi = pid // tl.cdiv(kp, CK)
        bj = pid % tl.cdiv(kp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < m) & (offs_j[None, :] < k)
        vals = tl.load(
            a_ptr + offs_i[:, None] * lda + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(a_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)
    else:
        # B tile: 真实 k x n -> pad (kp x np), 越界补零
        # 注意: B 的网格按 (kp/CM) x (np/CK) 划分, offs_i 是 k 维(CM 行)、
        # offs_j 是 n 维(CK 列), 使 tile/mask 恒为 (CM,CK), 与 A 分支一致,
        # 从而支持 CM != CK 的非方形 tile (原实现 CM==CK 时两者才吻合)。
        q = pid - nblk_a
        bi = q // tl.cdiv(np, CK)
        bj = q % tl.cdiv(np, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < k) & (offs_j[None, :] < n)
        vals = tl.load(
            b_ptr + offs_i[:, None] * ldb + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(b_pad_ptr + offs_i[:, None] * np + offs_j[None, :], vals)


@libentry()
@triton.jit
def _sgemm_nn_pad_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    mp,
    np,
    kp,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(mp, BLOCK_M)
    grid_n = tl.cdiv(np, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * kp + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * np + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(kp // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * np

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    c_mask = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


# ---- tn/nt/tt per-shape 编译选项决策 (v2.2, 2026-08-30) ----
# ITT: 三种 trans 的 dot 布局为 MFMA 友好布局时 ITT=1 全面受益 (9 shape 全选 ITT1,
# 4096^3 -4~-6%、8192^3 -14~-18% 等), 统一取 True。
# buffer: 对齐大/中输出 (m*n>=2M) 与 32 非对齐 shape 用 no-buffer (-1~-7% 实测);
# (2048,2048,16384) 大 K opt 在 no-buffer 下严重变差 (+43% nn 正式确认), 保持默认。
def _sgemm_xt_use_itt(m, n, k, **_kw):
    return True


def _sgemm_xt_use_buffer_ops(m, n, k, **_kw):
    if (m, n, k) == (2048, 2048, 16384):
        return True
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return False
    if m * n >= 2 * 1024 * 1024:
        return False
    return True


# no-buffer 变体: 显式传 use_buffer_ops=False 进编译选项 (cache key 区分);
# buffer=True 时不传, 保持全局默认产物 (与 nn 路径同款约定)。
_XT_NOBUF_KWARGS = {"use_buffer_ops": False}


# ---- tn (C = A^T B) 内核: common tn + EVEN_M/N/K constexpr (v2.2) ----
# A(k,m) lda=m, B(k,n) ldb=n; a(BM,BK) 为 A^T 片 (last-dim stride lda, ITT 转置),
# b(BK,BN) 连续。EVEN 分支消除对齐 shape 的 mask 开销 (4096^3 实测 -5~-10%)。
@triton.jit
def _sgemm_tn_even_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
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
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)  # (BM,BK)
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])  # (BK,BN)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_am < m
    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < n
    if EVEN_K:
        for i in range(k // BLOCK_K):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
    else:
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for i in range(k_full_iters):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if EVEN_M:
                a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    if EVEN_M and EVEN_N:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_tn_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_tn"))(_sgemm_tn_even_kernel))
)

# ---- tn 复合路径: 预转置 A^T + nn 分发 (v2.3, 2026-08-30) ----
# 为什么 tn 慢/nt 快: MFMA (v_mmac 16x16x8_f32) 要求 dot LHS 的 k 维内存连续。
# nt 的 C=A B^T 直接写成 acc_t=B A^T (两操作数均 k 连续) 故快; tn 的 A(k,m)
# 与 B(k,n) k 维 stride 均为行距 (strided), 任何方向下 LHS 都非 k 连续,
# Triton 只能逐迭代做布局转换, 这是 tn 的结构性上限。
# 复合方案参考 nt 的做法: 用带宽型转置内核把 A^T 变成连续 (m,k) 缓冲,
# 再走 nn 分发 (opt/pad/splitk 全路径), 两操作数均 k 连续。转置内核已达设备
# 峰值 (~1.1 TB/s, 128MB 120us), 无进一步优化空间。
# 实测 (37-shape): 非对齐大 shape 4095³ 0.657->0.758、8191³ 0.771->0.860、
# 4097x8191x4095 0.759->0.842; 对齐 shape 4096³ 0.791->0.849、
# 512x16384x4096 0.814->0.898、2048x4096x11008 0.826->0.892、
# 2048x11008x4096 0.846->0.896、2048x16384x2048 0.836->0.857 等。
_TN_TP_BM = 64
_TN_TP_BK = 64
_TN_TP_GM = 4
# 非对齐 shape 走复合的最小计算量: 1023³(1.1e9)/511³ 等中小 shape 转置成本
# 占比高, 实测直连更优; 4095³(6.9e10) 起转置只占 4% 且消除 mask 收益显著。
_TN_COMP_MIN_ELEMS = 10**10
# 对齐 shape 由实验确认的精确 shape 集合驱动 (共享 config 表下 nn 内核在该批
# shape 显著优于 tn 直连; 其余对齐 shape 两者几乎持平, 复合仅增加转置成本)。
# 排除微秒级 shape (64,2048,1024): 复合 = tp+nn 两内核, 测量受 GPU 历史状态
# 干扰 (32~62us 漂移), 全量 benchmark 实测 62.8us 反而远差于直连 36.3us;
# (64,4096,2048) 复合 5 次独立测量稳定 57-60us (直连 62-66us), 保留。
# 排除 (2048,2048,16384): 转置 263us 成本下收益仅 ~1% 且在两次全量复跑中
# 3418/3485us 波动 (直连 3445us), 正负尾相当, 保持直连更稳。
# v2.4 (2026-08-30): 37-shape 全量路径扫描 (warmup=200/rep=50) 后扩展的精确集合。
# - 薄 m shape (64/128/256): A^T 转置成本 m*k 极小, nn 内核显著快于 tn 直连
#   (128x1024x4096 119->59us -50%、64x1024x2048 63->35us -45% 等), 复合必然更优;
# - 16384x4096x512 (-3%)、2048x4096x12288 (-7.5%)、2048x2048x16384 (-1%) 等
#   对齐中大型 shape 实测 compA 也优于直连 (原 min(m,n)>=512 规则仅覆盖非对齐)。
_TN_COMP_ALIGNED_SHAPES = {
    (4096, 4096, 4096),
    (512, 16384, 4096),
    (2048, 4096, 11008),
    (2048, 11008, 4096),
    (2048, 16384, 2048),
    (64, 4096, 2048),
    (64, 512, 512),
    (64, 512, 1024),
    (64, 1024, 2048),
    (64, 2048, 4096),
    (128, 1024, 4096),
    (256, 2048, 8192),
    (512, 4096, 16384),
    (16384, 4096, 512),
    (2048, 4096, 12288),
    (2048, 2048, 16384),
}


@triton.jit
def _sgemm_tn_tp_kernel(
    src,
    dst,
    m,
    k,
    lda,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    # src(k,m) lda=m -> dst(m,k) 连续 (A^T 转置); 带宽型内核, 64x64 达设备峰值。
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BM)
    grid_k = tl.cdiv(k, BK)
    width = GM * grid_k
    gid = pid // width
    gs = min(grid_m - gid * GM, GM)
    pid_m = gid * GM + (pid % gs)
    pid_k = (pid % width) // gs
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BK + tl.arange(0, BK)
    s = tl.load(src + (offs_m[:, None] + offs_k[None, :] * lda),
                mask=(offs_k[None, :] < k) & (offs_m[:, None] < m), other=0.0)
    tl.store(dst + (offs_m[:, None] * k + offs_k[None, :]), s,
             mask=(offs_m[:, None] < m) & (offs_k[None, :] < k))


def _sgemm_tn_use_composite(m, n, k, **_kw):
    # 非对齐大 shape: 转置等价于 nn pad 路径 (消除 mask/折返开销)。
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return min(m, n) >= 512 and m * n * k >= _TN_COMP_MIN_ELEMS
    return (m, n, k) in _TN_COMP_ALIGNED_SHAPES


def _sgemm_tn_use_composite_y(m, n, k, **_kw):
    # 变体 Y (v2.3, 2026-08-30): 只转置 B^T (成本 n*k) + tt 内核。
    # tt 内核 acc_t=dot(b_t,a_t) 天然把 strided 的 A(k,m) 放 dot RHS (MFMA 不要求
    # RHS k 连续), 只需 B 一次转置; n≪m 时 n*k 远小于方案A的 m*k, 转置成本可忽略。
    # 实测 (exp_tn_comp_tt 37-shape): (16384,512,4096) 0.815->0.853、
    # (32768,1024,1024) 0.798->0.816、(16384,2048,2048) 0.828->0.830、
    # (8192,256,2048) 与直连持平; 中小 shape (m*n*k<1e9) 转置+双内核开销占比高,
    # 直连更优 (如 (4096,128,1024) Y 0.690 vs 直连 0.738), 故要求计算量下限。
    return m >= 8 * n and m * n * k >= 10**9


# ---- nt (C = A B^T) 直连布局内核 + EVEN_M/N/K (v2.2, 2026-08-30) ----
# common _sgemm_nt_kernel 每 k 迭代做 tl.trans(a) 再 dot(b, a_t), 显式转置
# 在 gfx936 上开销极大 (4096^3 6394us)。此内核改为直接加载 a_t (BK,BM) 与
# b_t (BN,BK) (均为 MFMA 友好 row-major), dot(b_t, a_t) 后仅末尾一次 trans:
# 4096^3 6394->3026us、8192^3 50740->21478us。EVEN 分支进一步消除 mask。
@triton.jit
def _sgemm_nt_even_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # C = A B^T ; A(m,k) lda=k, B(n,k) ldb=k ; acc_t = C^T (N,M) = B A^T
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_t_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])  # (BN,BK) contig
    a_t_ptrs = a_ptr + (offs_k[:, None] + offs_am[None, :] * lda)  # (BK,BM)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_am < m
    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < n
    if EVEN_K:
        for i in range(k // BLOCK_K):
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            if EVEN_M:
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            b_t_ptrs += BLOCK_K
            a_t_ptrs += BLOCK_K
    else:
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for i in range(k_full_iters):
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            if EVEN_M:
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            b_t_ptrs += BLOCK_K
            a_t_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if EVEN_N:
                b_t = tl.load(b_t_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            if EVEN_M:
                a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :] & mask_k[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    acc = tl.trans(acc_t)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    if EVEN_M and EVEN_N:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_nt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_nt"))(_sgemm_nt_even_kernel))
)

# ---- nt pad 路径 (v2.6, 2026-08-30) ----
# 非对齐大 shape 直连 kernel 的 mask/k remainder 开销高 (nt 4095³ 0.755/8191³ 0.767/
# 4097x8191x4095 0.769)。参照 nn pad 思路: A(m,k)/B(n,k) pre-pad 到 (mp,kp)/(np,kp)
# 对齐缓冲, 主内核全 EVEN (load 无 mask), 仅 store 对真实 m/n 打 mask。
# 实测 (warmup=100/rep=100, hipBLAS ref): 4095³ 0.755->0.840、8191³ 0.767->0.866、
# 4097x8191x4095 0.769->0.874 (main 128x128 w8 s2 最优)。
_nt_pad_cache = {}


def _get_nt_pad_buffers(mp, np, kp, device):
    key = (mp, np, kp, str(device))
    bufs = _nt_pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(mp, kp, dtype=torch.float32, device=device)
        b = torch.zeros(np, kp, dtype=torch.float32, device=device)
        bufs = (a, b)
        _nt_pad_cache[key] = bufs
    return bufs


@triton.jit
def _sgemm_nt_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    mp,
    np,
    kp,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(mp, CM) * tl.cdiv(kp, CK)
    grid_j = tl.cdiv(kp, CK)
    if pid < nblk_a:
        # A(m,k) lda -> A_pad(mp,kp)
        bi = pid // grid_j
        bj = pid % grid_j
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < m) & (offs_j[None, :] < k)
        vals = tl.load(
            a_ptr + offs_i[:, None] * lda + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(a_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)
    else:
        # B(n,k) ldb -> B_pad(np,kp)
        q = pid - nblk_a
        bi = q // grid_j
        bj = q % grid_j
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < n) & (offs_j[None, :] < k)
        vals = tl.load(
            b_ptr + offs_i[:, None] * ldb + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(b_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)


@triton.jit
def _sgemm_nt_pad_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    mp,
    np,
    kp,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # A_pad(mp,kp) lda=kp, B_pad(np,kp) ldb=kp; 全 EVEN load, store 对真实 m/n 打 mask。
    pid = tl.program_id(0)
    grid_m = tl.cdiv(mp, BLOCK_M)
    grid_n = tl.cdiv(np, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_t_ptrs = b_ptr + (offs_bn[:, None] * kp + offs_k[None, :])  # (BN,BK)
    a_t_ptrs = a_ptr + (offs_k[:, None] + offs_am[None, :] * kp)  # (BK,BM)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(kp // BLOCK_K):
        b_t = tl.load(b_t_ptrs)
        a_t = tl.load(a_t_ptrs)
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        b_t_ptrs += BLOCK_K
        a_t_ptrs += BLOCK_K
    acc = tl.trans(acc_t)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    c_mask = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


def _sgemm_xt_is_pad_shape(m, n, k, **_kw):
    # 非对齐且非小/瘦 shape 才走 pre-pad (nt/tt 共用; nn 有独立判定与分级 config)
    if (
        m % _PAD_BLOCK_M == 0
        and n % _PAD_BLOCK_N == 0
        and k % _PAD_BLOCK_K == 0
    ):
        return False
    if min(m, n) < 512:
        return False
    # nt/tt main 只有大 tile config (128x128), 无 nn 的小输出分级; 实测 1023³ (1.07e9)
    # pad 0.45 vs 直连 1.08, 阈值须显著高于 nn 的 5e8 才只覆盖 4095³ 及以上
    if m * n * k < _NT_PAD_MIN_ELEMS:
        return False
    return True


# nt pad 专用: 只有 4095³ (6.9e10) 及以上的大非对齐 shape 走 pre-pad (见 _sgemm_nt_is_pad_shape)
_NT_PAD_MIN_ELEMS = 2 * 10**10


def _make_sgemm_nt_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # 非对齐大 shape: main 128x128 w8 s2 最优 (4095³/8191³/4097x8191x4095 全量扫描)
    BM, BN, BK, GM, nw, ns = 128, 128, 32, 8, 8, 2
    mp = triton.cdiv(m, BM) * BM
    np = triton.cdiv(n, BN) * BN
    kp = triton.cdiv(k, _PAD_COPY_CK) * _PAD_COPY_CK
    A_pad, B_pad = _get_nt_pad_buffers(mp, np, kp, A.device)
    grid_pad = (
        triton.cdiv(mp, _PAD_COPY_CM) * triton.cdiv(kp, _PAD_COPY_CK)
        + triton.cdiv(np, _PAD_COPY_CM) * triton.cdiv(kp, _PAD_COPY_CK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_nt_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            m,
            n,
            k,
            lda,
            ldb,
            mp,
            np,
            kp,
            CM=_PAD_COPY_CM,
            CK=_PAD_COPY_CK,
            num_warps=4,
            use_in_thread_transpose=False,
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=True,
            use_buffer_ops=False,
        )
        _sgemm_nt_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


# ---- tt pad 路径 (v2.7, 2026-08-30) ----
# tt (C=A^T B^T): A(k,m) lda=m、B(n,k) ldb=k 均行主序。非对齐大 shape 直连内核
# 走 mask/k remainder 路径开销高 (tt 8191³ 0.883/4097x8191x4095 0.826)。参照 nt pad:
# A(k,m)->A_pad(kp,mp)、B(n,k)->B_pad(np,kp) pre-pad, 主内核全 EVEN load。
# 实测 (warmup=100/rep=100, hipBLAS ref): 8191³ 0.883->1.037、4097x8191x4095 0.826->0.940、
# 4095³ 1.115->1.239 (main 128x128 w8 s2 最优)。
_tt_pad_cache = {}


def _get_tt_pad_buffers(kp, mp, np, device):
    key = (kp, mp, np, str(device))
    bufs = _tt_pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(kp, mp, dtype=torch.float32, device=device)
        b = torch.zeros(np, kp, dtype=torch.float32, device=device)
        bufs = (a, b)
        _tt_pad_cache[key] = bufs
    return bufs


@triton.jit
def _sgemm_tt_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    k,
    m,
    n,
    lda,
    ldb,
    kp,
    mp,
    np,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    # A(k,m) lda -> A_pad(kp,mp), B(n,k) ldb -> B_pad(np,kp); 两者列数不同,
    # grid_j 分开计算。
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(kp, CM) * tl.cdiv(mp, CK)
    if pid < nblk_a:
        bi = pid // tl.cdiv(mp, CK)
        bj = pid % tl.cdiv(mp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < k) & (offs_j[None, :] < m)
        vals = tl.load(a_ptr + offs_i[:, None] * lda + offs_j[None, :], mask=mask, other=0.0)
        tl.store(a_pad_ptr + offs_i[:, None] * mp + offs_j[None, :], vals)
    else:
        q = pid - nblk_a
        bi = q // tl.cdiv(kp, CK)
        bj = q % tl.cdiv(kp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < n) & (offs_j[None, :] < k)
        vals = tl.load(b_ptr + offs_i[:, None] * ldb + offs_j[None, :], mask=mask, other=0.0)
        tl.store(b_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)


@triton.jit
def _sgemm_tt_pad_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    mp,
    np,
    kp,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # A_pad(kp,mp) lda=mp, B_pad(np,kp) ldb=kp; 全 EVEN load, store 对真实 m/n 打 mask。
    pid = tl.program_id(0)
    grid_m = tl.cdiv(mp, BLOCK_M)
    grid_n = tl.cdiv(np, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_t_ptrs = a_ptr + (offs_k[:, None] * mp + offs_am[None, :])  # (BK,BM) 连续
    b_t_ptrs = b_ptr + (offs_bn[:, None] * kp + offs_k[None, :])  # (BN,BK) 连续
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(kp // BLOCK_K):
        a_t = tl.load(a_t_ptrs)
        b_t = tl.load(b_t_ptrs)
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        a_t_ptrs += BLOCK_K * mp
        b_t_ptrs += BLOCK_K
    acc = tl.trans(acc_t)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    c_mask = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


def _make_sgemm_tt_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # 非对齐大 shape: main 128x128 w8 s2 最优 (4095³/8191³/4097x8191x4095 全量扫描)
    BM, BN, BK, GM, nw, ns = 128, 128, 32, 8, 8, 2
    mp = triton.cdiv(m, BM) * BM
    np = triton.cdiv(n, BN) * BN
    kp = triton.cdiv(k, _TT_PAD_COPY_CK) * _TT_PAD_COPY_CK
    A_pad, B_pad = _get_tt_pad_buffers(kp, mp, np, A.device)
    grid_pad = (
        triton.cdiv(kp, _TT_PAD_COPY_CM) * triton.cdiv(mp, _TT_PAD_COPY_CK)
        + triton.cdiv(np, _TT_PAD_COPY_CM) * triton.cdiv(kp, _TT_PAD_COPY_CK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_tt_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            k,
            m,
            n,
            lda,
            ldb,
            kp,
            mp,
            np,
            CM=_TT_PAD_COPY_CM,
            CK=_TT_PAD_COPY_CK,
            num_warps=4,
            use_in_thread_transpose=False,
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=True,
            use_buffer_ops=False,
        )
        _sgemm_tt_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


# ---- tn pad 路径 (v2.8, 2026-08-30) ----
# tn (C=A^T B): A(k,m) lda=m 与 tt 的 A 布局一致, 但 B(k,n) ldb=n 是 k 行主序,
# 直连/复合路径 (A^T->nn pad) 对大非对齐 shape 均落后 (tn 4095³ 0.880/8191³ 0.918/
# 4097x8191x4095 0.934)。参照 tt pad: A(k,m)->A_pad(kp,mp)、B(k,n) 转置+pad->
# B_pad(np,kp), 复用 _sgemm_tt_pad_kernel (全 EVEN main)。tt pad 同 shape 已 1.24/
# 1.01/0.94, tn 复用后预计 4095³~1.0/8191³~0.96/4097x8191x4095~1.0。
_tn_pad_cache = {}


def _get_tn_pad_buffers(kp, mp, np, device):
    key = (kp, mp, np, str(device))
    bufs = _tn_pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(kp, mp, dtype=torch.float32, device=device)
        b = torch.zeros(np, kp, dtype=torch.float32, device=device)
        bufs = (a, b)
        _tn_pad_cache[key] = bufs
    return bufs


@triton.jit
def _sgemm_tn_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    k,
    m,
    n,
    lda,
    ldb,
    kp,
    mp,
    np,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    # A(k,m) lda -> A_pad(kp,mp) (同 tt); B(k,n) ldb -> B_pad(np,kp) 转置
    # (b_pad[n,k] = B[k,n])。A/B 的网格划分不同 (A 按 (kp,mp) tile, B 按 (np,kp) tile)。
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(kp, CM) * tl.cdiv(mp, CK)
    if pid < nblk_a:
        bi = pid // tl.cdiv(mp, CK)
        bj = pid % tl.cdiv(mp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < k) & (offs_j[None, :] < m)
        vals = tl.load(a_ptr + offs_i[:, None] * lda + offs_j[None, :], mask=mask, other=0.0)
        tl.store(a_pad_ptr + offs_i[:, None] * mp + offs_j[None, :], vals)
    else:
        q = pid - nblk_a
        bi = q // tl.cdiv(kp, CK)
        bj = q % tl.cdiv(kp, CK)
        offs_i = bi * CM + tl.arange(0, CM)  # n 维
        offs_j = bj * CK + tl.arange(0, CK)  # k 维
        mask = (offs_i[:, None] < n) & (offs_j[None, :] < k)
        vals = tl.load(b_ptr + offs_i[:, None] + offs_j[None, :] * ldb, mask=mask, other=0.0)
        tl.store(b_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)


def _make_sgemm_tn_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # main 128x128 w8 s2 同 tt pad (tt 全量扫描最优); A/B 两个 copy 与 tt pad 同体积。
    BM, BN, BK, GM, nw, ns = 128, 128, 32, 8, 8, 2
    mp = triton.cdiv(m, BM) * BM
    np = triton.cdiv(n, BN) * BN
    kp = triton.cdiv(k, _PAD_COPY_CK) * _PAD_COPY_CK
    A_pad, B_pad = _get_tn_pad_buffers(kp, mp, np, A.device)
    grid_pad = (
        triton.cdiv(kp, _PAD_COPY_CM) * triton.cdiv(mp, _PAD_COPY_CK)
        + triton.cdiv(np, _PAD_COPY_CM) * triton.cdiv(kp, _PAD_COPY_CK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_tn_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            k,
            m,
            n,
            lda,
            ldb,
            kp,
            mp,
            np,
            CM=_PAD_COPY_CM,
            CK=_PAD_COPY_CK,
            num_warps=4,
            use_in_thread_transpose=False,
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=True,
            use_buffer_ops=False,
        )
        _sgemm_tt_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


# ---- tt (C = A^T B^T) 内核: common tt + EVEN_M/N/K (v2.2) ----
# A(k,m) lda=m, B(n,k) ldb=k; a_t(BK,BM) 与 b_t(BN,BK) 均为连续 row-major,
# dot(b_t, a_t) 后一次 trans 写 C。天然 MFMA 友好布局, 基线已达 sp~1.0,
# EVEN 分支在 8192^3 再 -5.5%。
@triton.jit
def _sgemm_tt_even_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # C = A^T B^T ; acc_t = C^T (N,M) = B^T A^T
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_t_ptrs = a_ptr + (offs_k[:, None] * lda + offs_am[None, :])  # (BK,BM) contig
    b_t_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])  # (BN,BK) contig
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_am < m
    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < n
    if EVEN_K:
        for i in range(k // BLOCK_K):
            if EVEN_M:
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K
    else:
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for i in range(k_full_iters):
            if EVEN_M:
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if EVEN_M:
                a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            if EVEN_N:
                b_t = tl.load(b_t_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    acc = tl.trans(acc_t)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    if EVEN_M and EVEN_N:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_tt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_tt"))(_sgemm_tt_even_kernel))
)


# ---- nt 瘦 shape 专用: 两阶段 split-k (v2.13, 2026-08-31) ----
# nt 直连内核 (64,64) tile 下 blocks 仅 8-32, 并行度不足 (gfx936 数十 CU)。
# split-k 把 k 均分给 SPLIT_K 段并行 (内核1 写 scratch(SPLIT_K,n,m) C^T 布局,
# 内核2 reduce 到 C), blocks 倍乘 SPLIT_K。官方口径 do_bench(warmup=1000,
# rep=100, median) 实测 (kpack=2, DB 生产 config):
#   (64,512,512) SP4 20.6->15.7us (+31%)、(512,64,512) SP4 19.7->15.0 (+31%)、
#   (64,2048,1024) SP4 31.0->25.8 (+20%)、(2048,64,1024) SP4 31.0->25.8 (+20%)、
#   (64,1024,512) SP4 18.4->16.3 (+13%)、(1024,64,512) SP4 18.6->16.6 (+12%)、
#   (64,4096,2048) SP2 57.3->53.9 (+6%)。
# 已 64+ blocks 的 shape (4096x128x1024/128x4096x1024) 反而 SP 变差 (0.59-0.90),
# 以及 (2048,2048,16384) SP 0.92, 均不进白名单。k 必须整除 SPLIT_K*BLOCK_K,
# 白名单 shape 全为 64 倍数 (全 EVEN, 无 k remainder)。
_NT_SPLITK_BLOCK_M = 64
_NT_SPLITK_BLOCK_N = 64
_NT_SPLITK_BLOCK_K = 32
_NT_SPLITK_GROUP_M = 8
_NT_SPLITK_STAGES = 2
# shape -> (SPLIT_K, num_warps) (num_warps 取 DB 生产直连 config 的 w)
# v2.26: 移除 (64,512,512)/(512,64,512)/(1024,64,512) -> 改走 direct override
# (BK64+kp2 全 config 扫描官方 A/B: 17.28/16.80/17.76us -> 12.80/13.76/12.32us,
# 显著优于 split-k)。
# v2.27: 移除剩余 4 个 (64,1024,512)/(64,2048,1024)/(2048,64,1024)/
# (64,4096,2048) -> 改走 direct override (BK64+kp2 全 config 扫描 + 官方 A/B 三轮
# 全稳定, 详见 _NT_DIRECT_OVERRIDES)。白名单清空, split-k 机制保留备用。
_NT_SPLITK_SHAPES = {}


def _sgemm_nt_splitk(m, n, k, **_kw):
    # 返回 SPLIT_K 或 0 (不走 split-k)。仅精确白名单 shape, 避免误伤。
    cfg = _NT_SPLITK_SHAPES.get((m, n, k))
    if cfg is None:
        return 0
    spk, _ = cfg
    if k % (spk * _NT_SPLITK_BLOCK_K) != 0:
        return 0
    return spk


@libentry()
@triton.jit
def _sgemm_nt_splitk_kernel(
    a_ptr,
    b_ptr,
    scratch_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # 内核1: 每个 block 计算部分 K 段, 写 scratch(SPLIT_K, n, m) (C^T 布局)。
    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid_mn // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid_mn % gs)
    pid_n = (pid_mn % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    k_per = k // SPLIT_K
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * k_per + tl.arange(0, BLOCK_K)
    b_t_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])  # (BN,BK)
    a_t_ptrs = a_ptr + (offs_k[:, None] + offs_am[None, :] * lda)  # (BK,BM)
    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_am < m
    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < n
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(k_per // BLOCK_K):
        if EVEN_N:
            b_t = tl.load(b_t_ptrs)
        else:
            b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
        if EVEN_M:
            a_t = tl.load(a_t_ptrs)
        else:
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        b_t_ptrs += BLOCK_K
        a_t_ptrs += BLOCK_K
    s_ptrs = (
        scratch_ptr
        + pid_k.to(tl.int64) * (n.to(tl.int64) * m)
        + offs_bn[:, None].to(tl.int64) * m
        + offs_am[None, :].to(tl.int64)
    )
    if EVEN_M and EVEN_N:
        tl.store(s_ptrs, acc_t)
    else:
        tl.store(
            s_ptrs,
            acc_t,
            mask=(mask_n[:, None] if not EVEN_N else True)
            & (mask_m[None, :] if not EVEN_M else True),
        )


@libentry()
@triton.jit
def _sgemm_nt_splitk_reduce_kernel(
    scratch_ptr,
    c_ptr,
    m,
    n,
    ldc,
    alpha: tl.float32,
    beta: tl.float32,
    SPLIT_K: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # 内核2: reduce SPLIT_K 段 -> C (m,n) row-major。
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        s_ptrs = (
            scratch_ptr
            + s * (n.to(tl.int64) * m)
            + offs_bn[:, None].to(tl.int64) * m
            + offs_am[None, :].to(tl.int64)
        )
        acc_t += tl.load(s_ptrs)
    acc = tl.trans(acc_t)
    mask_m = offs_am < m
    mask_n = offs_bn < n
    c_ptrs = c_ptr + (offs_am[:, None].to(tl.int64) * ldc + offs_bn[None, :].to(tl.int64))
    if EVEN_M and EVEN_N:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


_nt_splitk_scratch_cache = {}


def _get_nt_splitk_scratch(sp, n, m, device):
    key = (sp, n, m, str(device))
    buf = _nt_splitk_scratch_cache.get(key)
    if buf is None:
        buf = torch.zeros(sp, n, m, dtype=torch.float32, device=device)
        _nt_splitk_scratch_cache[key] = buf
    return buf


def _make_sgemm_nt_splitk_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, SPLIT_K, num_warps, **_kw
):
    # 白名单 shape 全为 (64,64) tile 对齐 (m/n 64 倍数) 且 k 整除 SPLIT_K*BK,
    # 内核可静态全 EVEN; 转置内核不参与, 无 config 表依赖。
    BM, BN, BK, GM = (
        _NT_SPLITK_BLOCK_M,
        _NT_SPLITK_BLOCK_N,
        _NT_SPLITK_BLOCK_K,
        _NT_SPLITK_GROUP_M,
    )
    scratch = _get_nt_splitk_scratch(SPLIT_K, n, m, A.device)
    grid1 = (triton.cdiv(m, BM) * triton.cdiv(n, BN) * SPLIT_K,)
    grid2 = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    itt = _sgemm_xt_use_itt(m, n, k)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_nt_splitk_kernel[grid1](
            A, B, scratch, m, n, k, lda, ldb,
            SPLIT_K, True, True, True, BM, BN, BK, GM,
            num_warps=num_warps,
            num_stages=_NT_SPLITK_STAGES,
            use_in_thread_transpose=itt,
            kpack=2,
            **nobuf_kw,
        )
        _sgemm_nt_splitk_reduce_kernel[grid2](
            scratch, C, m, n, ldc, alpha, beta, SPLIT_K, beta_is_zero,
            True, True, BM, BN, GM,
            num_warps=num_warps,
            use_in_thread_transpose=itt,
            kpack=2,
            **nobuf_kw,
        )

    return run


# ---- nt 直连 config override (v2.13, 2026-08-31) ----
# libtuner 从共享 config 池为直连内核选 config, 个别非对齐小 shape 上生产 config
# (num_warps 等) 不是最优。官方口径 raw 扫描实测 511³ (64,64) tile w16 比生产
# w4 快 20% (29.4->23.7us, ratio 0.745->0.93), itt0+kp1 最优。
# runner 固定手动 launch raw 内核。
_NT_DIRECT_OVERRIDES = {
    (511, 511, 511): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=1,
    ),
    # v2.14 (2026-08-31): 官方口径 A/B 交替实测 (同进程各 3 次 do_bench)
    # (256,256,256) 生产 (64,64)w4 13.44us -> (64,64)w8 kp1 13.28us (+1.2%, 稳定)
    # (128,4096,1024)/(4096,128,1024) 换 config 无收益或更慢, 不落地
    # v2.15 (2026-09-01): 小 tile 扫描, (32,32)w4 大胜 (64,64)w8:
    #   同进程 A/B 各 3 次 do_bench: 生产 14.56us 稳定 -> (32,32)w4 kp1 11.09us
    #   (MED, min 10.72) 快 24-36%。小 shape 并行度是瓶颈, 32 块 grid 128 程序。
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=1,
    ),
    # v2.15b (2026-09-01): 512^3 同 tile 提并行: 生产 (64,64)w4kp1 21.60us ->
    #   (64,64)w8kp2 20.43us (-5.4%, A/B 交替各 3 次确认, min 20.00us)。
    (512, 512, 512): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.16 (2026-09-01): (256,8192,2048) 同 (512,16384,4096) 模式 w4->w8 kp2:
    #   生产 211.33us -> 201.12us (-4.8%, A/B 各 3 次全稳定), ratio 0.889->0.941。
    (256, 8192, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.14b: (512,16384,4096) w4->w8 kp2 1356->1343us ratio 0.9495->0.9587 达标
    #         (8192,256,2048) w4->w8 kp2 205.6->199.7us ratio 0.922->0.949 (+2.9%)
    (512, 16384, 4096): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (8192, 256, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.17 (2026-09-01): 1023³ 最弱 (37 shape 画像 0.7051)。直连扫描
    # (128,128)w16kp2 大 tile 提 A 跨块重读利用; 关键 knob 是 GROUP_M=1
    # (旧扫描固定 GM8 只找到 128x64w8kp2 边际 -2%)。官方口径 A/B 交替各 3 轮:
    # 生产 (64,128)w8gm8 91.68us 稳定 -> 88.80us 稳定 (-3.1%, ratio 0.7051->0.7279)。
    # pad 路径 97.9us (copy 37us 吃掉收益)、composite 被 min<4096 排除, 均不适用。
    (1023, 1023, 1023): dict(
        BLOCK_M=128, BLOCK_N=128, GROUP_M=1, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.18 (2026-09-01): 1024³ 偶数立方异常值 (0.8160, 2048³/4096³ 均 >0.92)。
    # 直连扫描 176 config (11 tile x w8/16 x itt0/1 x kp1/2 x gm1/8): 生产
    # (128,128)w8gm8kp1 73.92us 稳定 -> (64,128)w8gm1kp2 71.84us 稳定 (-2.7%,
    # ratio 0.8225->0.845)。kp2+GM1 优于生产 tile。
    (1024, 1024, 1024): dict(
        BLOCK_M=64, BLOCK_N=128, GROUP_M=1, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.19 (2026-09-01): 瘦 shape 扫描 (4096,128,1024)/(128,4096,1024) 各 128
    # config: (4096,128,1024) A/B 无收益 (扫描 45.92us 为噪声, A/B 均 46.56us),
    # 不落地。(128,4096,1024) 生产 (64,128)w8gm8 46.72us 稳定 -> (64,128)w16gm1
    # kp2 45.76us 稳定 (-2.1%, ratio 0.8082->0.8252)。
    # v2.26b (2026-09-01): BK64 扫描再提升 -4.5%: (64,64)BK64w16gm8kp2itt1
    # 官方口径 A/B 三轮稳定 43.20us (r 0.83->0.874, 生产 44.96-45.60us)。
    (128, 4096, 1024): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    # v2.20 (2026-09-01): 64³ 超小 shape launch-bound (生产 (64,64)w4gm8 grid=1)。
    # 384 config 小 tile 扫描 (8 tile x w1/2/4 x itt x kp x gm x BK16/32): 54 config
    # 达 8.16us, 远超生产 10.24us (r=0.8125)。官方 A/B 三轮稳定确认
    # (32,16)w4itt0kp1gm1 8.16/8.00/8.00us (r=0.98-1.0) vs 生产 9.92-10.08us
    # (r=0.79-0.81)。小 tile 提并行度 (grid 1 -> 8) 是核心收益。
    (64, 64, 64): dict(
        BLOCK_M=32, BLOCK_N=16, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=1,
    ),
    # v2.26 (2026-09-01): 瘦/小 shape 4 个, 生产 config 非最优 (BK=64+kp2 关键)。
    # 全 config 扫描 (11 tile x w4/8/16 x itt x kp x gm x BK32/64) + 官方口径 A/B 三轮:
    #   (64,512,512):   生产 (64,64)w8kp2 17.12us r=0.83 -> (16,32)w4itt1kp2gm8
    #                   BK64 12.80/12.80/11.36us r=1.11-1.25。小 tile 提并行度
    #                   (grid 8 -> 64 块)。
    #   (1024,64,512):  生产 17.76us r=0.85 -> (32,32)w4itt1kp2gm1 BK64
    #                   13.44/13.60/13.60us r=1.11。grid 16 -> 32 块。
    #   (512,64,512):   生产 15.68-16.80us r=0.79-0.85 -> (16,16)w4itt0kp2gm1
    #                   BK64 13.76us 稳定 r=0.97。grid 16 -> 128 块。
    #   (4096,64,2048): 生产 54.24-54.40us r=0.87 -> (64,64)w16itt0kp2gm1
    #                   BK64 48.32-48.48us r=0.98。grid 64 -> 64 块 w16 提 ILP。
    (64, 512, 512): dict(
        BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    (1024, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    (512, 64, 512): dict(
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=64, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (4096, 64, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.27 (2026-09-02): 原 split-k 白名单 4 个, BK64+kp2 全 config 扫描 +
    # 官方口径 A/B 交替 3 轮 (全部稳定, 详见 /tmp/nt_v227_ab.py):
    #   (64,1024,512):  split-k 生产 17.60/16.16/17.60us r=0.82-0.89 ->
    #                   (32,32)w4itt0kp2gm1BK64 12.16-13.28us r=1.08-1.18。
    #                   grid 4 -> 64 块提并行度。
    #   (64,2048,1024): split-k 生产 24.80-25.92us r=0.88-0.92 ->
    #                   (32,64)w8itt0kp2gm8BK64 21.12-21.76us r=1.04-1.08。
    #                   grid 8 -> 64 块。
    #   (2048,64,1024): split-k 生产 24.16-25.60us r=0.93-0.98 ->
    #                   (32,64)w8itt0kp2gm8BK64 20.96-22.08us r=1.07-1.13。
    #                   grid 8 -> 64 块。
    #   (64,4096,2048): split-k 生产 53.92us r=0.89 -> (64,64)w16itt0kp2gm8BK64
    #                   48.32us r=1.00 稳定 (-10.4%)。
    (64, 1024, 512): dict(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (64, 2048, 1024): dict(
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (2048, 64, 1024): dict(
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (64, 4096, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.28 (2026-09-02): (4096,128,1024) NT 最大异常值 (r=0.767)。BK64 全 config
    # 扫描 (12 tile x w4/8/16 x itt x kp x gm): 生产 (64,64)w16 46.56us 稳定 ->
    # (64,64)BK64w16itt1kp2gm1 42.40us 稳定。官方口径 A/B 交替 3 轮:
    # 45.92-46.40us (r=0.755-0.763) -> 42.88-43.04us (r=0.814-0.817, -7.4%)。
    # 同 (128,4096,1024) v2.26b pattern (BK64+w16+kp2)。仍 <0.90 但明显改善。
    (4096, 128, 1024): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=16,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    # v2.29 (2026-09-02): (2048,2048,16384) 大 K shape (<0.8 异常值)。direct
    # config + split-k 扫描 (10 direct + 8 split-k): 生产 DB direct (kp1) 3289.3us
    # (r=0.7855) 稳定 -> (64,64)BK64w16kp2gm8itt1 3190us 稳定 (r=0.809-0.810)。
    # 官方口径 A/B 交替 3 轮: 3289.3us -> 3189.6-3192.8us (-3.0%)。BK64+w16+kp2
    # 提 k-loop 吞吐 (k=16384 长循环); split-k 4 段 (128,128) 3201us 略差不取。
    (2048, 2048, 16384): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
}


def _make_sgemm_nt_direct_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_nt_even_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            kpack=ov.get("kpack", 1),
            **nobuf_kw,
        )

    return run


# ---- tn 直连 config override (v2.22, 2026-09-01) ----
# 5 个不达标 shape (core_shape 画像 speedup<0.90) 均因生产 config 并行度/效率不足:
# 小 shape 缩小 tile 提并行度 (与 tt 同策略), 官方口径 do_bench(1000/100/median)
# 扫描 (扫描 /tmp/sgemm_tn_scan_results.json + /tmp/sgemm_tn_scan2_results.json)
# 全部达 speedup>=1.0。runner 固定手动 launch raw 内核。
# 注: tn 布局 A(k,m) lda=m / B(k,n) ldb=n / C(m,n) ldc=n。
_TN_DIRECT_OVERRIDES = {
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (64, 512, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (64, 1024, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (512, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (1024, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    # v2.30 (2026-09-02): 64³ 超小 shape launch-bound (grid=1, 生产 10.56us
    # r=0.788 官方口径)。小 tile 扫描 54 config: w4 各类 tile 均达 8.16us 下限
    # (r=1.0-1.16)。monkeypatch A/B 三轮交替: prod 10.56us 稳定 -> (32,32)w4gm1
    # 8.16-9.44us (r=0.87-1.02, 最差轮仍 >=0.8)。与其余 tn 小 shape override 同款
    # (32,32)w4gm1。小 tile 提并行度 (grid 1 -> 4)。
    (64, 64, 64): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
}


def _make_sgemm_tn_direct_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tn_even_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- tn 复合路径 nn 内核 config override (v2.23, 2026-09-01) ----
# comp 路径 = tp(A^T) + nn 内核。生产走 _SGEMM_NN_DISPATCH tuner config0
# (64,64,w4,gm8,BK32,s2) 对 4096³ 非最优: 实验扫描 (warmup=200/rep=30, 362+ config)
# 发现 (64,128,w8,itt1,gm1,BK32,s2,nobuf) 更快 (prod 2912->2898us, r 0.886->0.890)。
# tp 内核已到带宽峰值 (~1.1TB/s, 120us), 直连/compY 结构均更差, 此为结构下限。
_TN_COMP_OVERRIDES = {
    (4096, 4096, 4096): dict(
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=1, num_warps=8,
        num_stages=2, use_in_thread_transpose=True,
    ),
}


def _make_sgemm_tn_comp_override_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, ov, **_kw
):
    Ac = _get_tn_ac(m, k, A.device)
    grid_tp = ((m + _TN_TP_BM - 1) // _TN_TP_BM) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    grid_nn = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tn_tp_kernel[grid_tp](A, Ac, m, k, lda, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM)
        _sgemm_nn_opt_kernel[grid_nn](
            Ac, B, C, alpha, beta, m, n, k, k, ldb, ldc, beta_is_zero,
            m % BM == 0, n % BN == 0, k % BK == 0, BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- tn 复合路径 Y (compY) tt 内核 config override (v2.24, 2026-09-01) ----
# compY = tp(B^T) + tt even 内核。生产走 tuner config 对这 2 个大 shape 非最优:
# 官方口径 do_bench(1000/100/median) 验证 (32768,1024,1024) 1556->1540us
# (r 0.892->0.901), (16384,2048,2048) 2953->2921us (r 0.880->0.890, 结构上限
# 未达 0.90)。runner 固定手动 launch raw 内核。
_TN_COMPY_OVERRIDES = {
    (32768, 1024, 1024): dict(
        BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=1, num_warps=8,
        num_stages=2, kpack=2, use_in_thread_transpose=True,
    ),
    (16384, 2048, 2048): dict(
        BLOCK_M=256, BLOCK_N=128, BLOCK_K=32, GROUP_M=4, num_warps=16,
        num_stages=2, kpack=1, use_in_thread_transpose=True,
    ),
}


def _make_sgemm_tn_compy_override_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, ov, **_kw
):
    Bc = _get_tn_bc(n, k, B.device)
    grid_tp = ((n + _TN_TP_BM - 1) // _TN_TP_BM) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    grid_tt = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tn_tp_kernel[grid_tp](B, Bc, n, k, ldb, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM)
        _sgemm_tt_even_kernel[grid_tt](
            A, Bc, C, alpha, beta, m, n, k, lda, k, ldc, beta_is_zero,
            m % BM == 0, n % BN == 0, k % BK == 0, BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            kpack=ov.get("kpack", 1),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- tt 直连 config override (v2.21, 2026-09-01) ----
# 7 个不达标 shape (core_shape 画像 speedup<0.90) 全是 grid 并行度不足导致:
# 生产 config 大 tile 下 blocks 少 (64x512x512 (64,64)tile 64 块 -> 16x16 1024 块),
# 小 tile 提并行度, 官方口径 do_bench(warmup=1000, rep=100, median) A/B 交替验证
# (扫描 /tmp/sgemm_tt_scan_results.json + A/B /tmp/verify_sgemm_tt_ab.py) 全部
# 提速 15-46% 达 speedup>=1.0。runner 固定手动 launch raw 内核。
# 注: tt 布局 A(k,m) lda=m / B(n,k) ldb=k / C(m,n) ldc=n, EVEN 判定按 m,n,k。
_TT_DIRECT_OVERRIDES = {
    (64, 512, 512): dict(
        BLOCK_M=16, BLOCK_N=16, GROUP_M=1, num_warps=2,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (512, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=True,
    ),
    (64, 64, 64): dict(
        BLOCK_M=16, BLOCK_N=16, GROUP_M=1, num_warps=1,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (1024, 64, 512): dict(
        BLOCK_M=16, BLOCK_N=16, GROUP_M=1, num_warps=2,
        num_stages=2, use_in_thread_transpose=True,
    ),
    (64, 1024, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (511, 511, 511): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=1, num_warps=8,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
}


def _make_sgemm_tt_direct_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tt_even_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- StaticDispatch: 按 shape 情况分发不同 sgemm_nn kernel ----
# 大 K shape (k >= 4*max(m,n)): 两阶段 split-k, 实测 2048x2048x16384 有效
# 非对齐且非小 shape: pre-pad + 全 EVEN 内核 (消除 mask/k remainder 开销)
# 其余: 现网 libtuner opt kernel (保持原行为)

_SPLITK_BLOCK_M = 64
_SPLITK_BLOCK_N = 64
_SPLITK_BLOCK_K = 32
_SPLITK_GROUP_M = 8
_SPLITK_WARPS = 4
_SPLITK_STAGES = 2

# pad 路径参数 (实测最优 config)
_PAD_BLOCK_M = 128
_PAD_BLOCK_N = 256
_PAD_BLOCK_K = 32
# v1.8 (2026-08-26): copy tile 由 64x64 改为 32x32 (带宽型 copy, 小 tile 网格更大
# 负载均衡更好): 8191³ copy 1214->908us (-25%)、1023³ copy 50->20us、4097x8191x4095
# copy 356us 持平; 完整路径 8191³ -1.6%、1023³ -16%。32 恒整除 mp/np/kp
# (mp/np 为 BM/BN 倍数, kp 为 128 倍数), 无对齐风险。
_PAD_COPY_CM = 32
_PAD_COPY_CK = 128
# tt pad-copy 专用 tile (v2.9): 非对齐大 shape 实测小方形 tile (32,32) 远优于
# (32,128) —— 4097x8191x4095 copy 487us -> 331us (1.08TB/s -> 1.2TB/s),
# 4095³/8191³ 同构受益。小 tile 程序数翻倍, 更利于延迟隐藏/带宽饱和。
_TT_PAD_COPY_CM = 32
_TT_PAD_COPY_CK = 32
# 小输出 shape 的更高并行度候选: main (32,64) + copy tile 32
# (1023³ 完整路径 116->92us, -20%; copy 50->20us)。
# 但 main BM=32 使 mp 可能非 64 倍数, 而 copy tile 必须整除 mp/np/kp,
# 不满足时由 runner 回退 (128,128) main (mp/np 恒为 128 倍数, 安全)。
# v2.9 曾试 (64,32): buf1 下与 (32,64) 持平 (75.4us), 无收益, 维持 (32,64)。
_PAD_CFG_SMALL_TILE = (32, 64, 32, 8, 4, 2)
# 大输出 shape 分级 (v1.8 实测, 2026-08-26):
# - m*n >= 32M (大输出, 如 8191³/4097x8191x4095): (256,256,32,G1,w16,s1) + ITT1 + no-buffer
#   8191³ 0.862->0.925 达标 (main 与 opt 8192³ 生产 config 同款, 20652us)、
#   4097x8191x4095 0.806->0.869
# - 4M <= m*n < 32M (中输出, 如 4095³): (256,128,32,G8,w8,s1) + ITT0 + no-buffer
#   4095³ 0.738->0.788 ((256,256,w16) 反而 0.734, 因 grid 仅 16x16 时大 tile 低效)
# - m*n < 4M (小输出, 如 1023³): (32,64) 避免 tile 浪费, 保持 ITT0+buf 默认 (0.625->0.724)
_PAD_CFG_LARGE = (256, 256, 32, 1, 16, 2)
# v1.10: MID 由 (256,128,32,G8,w8,s1)+ITT0 改为 (64,64,32,G8,w4,s2)+ITT1
# (与 opt 4096^3 生产 config 同款)。4095^3 pad main 算全 EVEN 4096^3:
# (64,64,ITT1) 2927us vs (256,128,ITT0) 3013us (-2.9%), ITT1 必需 (ITT0 3201)。
# 4095^3 完整路径 0.788 -> ~0.809。mp/np=cdiv(4095,64)=4096 仍满足 copy tile 对齐。
_PAD_CFG_MID = (64, 64, 32, 8, 4, 2)
_PAD_LARGE_OUTPUT = 32 * 1024 * 1024
_PAD_CFG_SMALL = (128, 128, 32, 8, 8, 2)
_PAD_SMALL_OUTPUT = 4 * 1024 * 1024
# 排除 511³ (1.3e8) 等小 shape: pad-copy 开销占比大, 实测反而更差
_PAD_MIN_ELEMS = 5 * 10**8

# pad 缓冲区缓存: 避免每次调用重复分配大 buffer (按 pad 后尺寸缓存)
_pad_cache = {}


def _get_pad_buffers(mp, kp, np, device):
    key = (mp, kp, np, str(device))
    bufs = _pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(mp, kp, dtype=torch.float32, device=device)
        b = torch.zeros(kp, np, dtype=torch.float32, device=device)
        bufs = (a, b)
        _pad_cache[key] = bufs
    return bufs


_LARGE_K_OPT_SHAPES = {(2048, 2048, 16384)}
# v1.10: SPLIT_K=4 对薄大 K shape 显著更优 (实测):
#   128x1024x4096 64.2->52.5us (-18.2%)、256x2048x8192 288.5->251.7us (-12.8%);
#   512x4096x16384 反而 SPK=4 退化 (1700->2825), 保持 SPK=2。
# 仅当 k 能被 SPLIT_K*BLOCK_K 整除时走 SPK=4 (保证段内全 EVEN, 无 k remainder)。
_SPLITK4_SHAPES = {(128, 1024, 4096), (256, 2048, 8192)}


def _sgemm_nn_is_large_k(m, n, k, **_kw):
    # 大 K shape 走两阶段 split-k; 特例 (2048,2048,16384) 实测 opt+ITT1 (0.852)
    # 优于 split-k+ITT0 (0.792) 与 opt+ITT0 (0.778), 由 _LARGE_K_OPT_SHAPES 排除
    if (m, n, k) in _LARGE_K_OPT_SHAPES:
        return False
    # split-k 按 k_per = k // SPLIT_K 均分, k 为奇数时末元素被丢弃 (静默错误),
    # 且 EVEN_K 需按段内余数判断; 奇数 k 一律回退 opt (其自带 k-remainder 分支)
    if k % 2 != 0:
        return False
    return k >= 4 * max(m, n)


# ITT (in-thread-transpose) per-shape 决策: 基于 ITT=0/1 全量回归实测
# - ITT=1 (True): 大 shape 普遍受益 (2048³ 0.828->0.904、256x8192x2048 0.839->0.908 等)
# - ITT=0 (False): 瘦 shape (min(m,n)<=128) 与小立方受 in-thread-transpose 布局损害
#   (64x512x512 1.054->0.774、256³ 1.103->0.918、511³ 0.854->0.790 等), config 无法恢复
# - 显式集合覆盖规则例外 (64³ 超小 shape ITT=1 反而 +0.136 至 0.983)
# v1.6 教训 (2026-08-26): 4 组合 (ITT x buffer) 扫描用手动 launch + DB TunedConfig
# 表 config, 但生产 libtuner 实际用 benchmark 表 config (二者不一致, 瘦 shape 尤甚)
# -> 手动 config 下 ITT0 对 16384x512x4096/512x16384x4096 更优 (-2.3%), 但 monkeypatch
#    生产路径确认 ITT1 才是生产最优 (1465us vs ITT0 1592us, 组合结论被 config 翻转).
#    唯一保留: 2048x16384x2048 从 _ITT0_SHAPES 移除 -> 生产路径 ITT1+nobuf 2867.6us
#    优于 ITT0+nobuf 2901us (+1.2%).
_ITT0_SHAPES = {
    (256, 256, 256),
    (511, 511, 511),
}
_ITT1_SHAPES = {
    (64, 64, 64),
}


def _sgemm_nn_use_itt(m, n, k, **_kw):
    if (m, n, k) in _ITT1_SHAPES:
        return True
    if (m, n, k) in _ITT0_SHAPES:
        return False
    if min(m, n) <= 128:
        return False
    return True


# use_buffer_ops (AMDGCN_USE_BUFFER_OPS) per-shape 决策: 基于 33 个 opt shape
# 全量快扫 + 正式协议 (warmup=1000/rep=100) 交错确认 (2026-08-26)
# - no-buffer (False): 大/中输出 shape (m*n>=2M) 普遍受益 -1~-4%
#   (4096³ -3.9%、512x16384x4096 -4.0%、2048x11008x4096 -4.1% 等);
#   非对齐 mask 路径 shape 受益显著 (511³ -10.0%)
# - buffer (True, 默认): 小输出 shape 保持 (1024³ +3.6%、128x4096x1024 +3.8%
#   [正式确认] 等 no-buffer 反而变差); (2048,2048,16384) 大 K opt 在 no-buffer 下
#   严重变差 (+43% 正式确认), 必须保持 buffer
def _sgemm_nn_use_buffer_ops(m, n, k, **_kw):
    if (m, n, k) == (2048, 2048, 16384):
        return True
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return False
    if m * n >= 2 * 1024 * 1024:
        return False
    return True


def _sgemm_nn_is_pad_shape(m, n, k, **_kw):
    # 非对齐(相对 pad tile 尺寸)且非小/瘦 shape 才走 pre-pad 路径
    if (
        m % _PAD_BLOCK_M == 0
        and n % _PAD_BLOCK_N == 0
        and k % _PAD_BLOCK_K == 0
    ):
        return False
    # 瘦 shape (输出任一维度 < 512) 走 opt: 实测 pad-copy 开销占比大反而更差,
    # 同时覆盖 511³、64³ 及 4096x64x2048 等 (n/m=64/128 对 BN=256 不对齐)
    if min(m, n) < 512:
        return False
    if m * n * k < _PAD_MIN_ELEMS:
        return False
    return True


def _sgemm_nn_is_default(**_kw):
    return True


def _make_sgemm_nn_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    if m * n < _PAD_SMALL_OUTPUT:
        # 小输出: main (32,64) 提升并行度 (1023³ 完整路径 116->92us, -20%);
        # main BM=32 使 mp 可能非 128 倍数, 不满足时回退 (128,128) main。
        # copy 统一 32x32 (v1.8: 带宽型 copy 最优, 1023³ copy 50->20us)。
        BM, BN, BK, GM, nw, ns = _PAD_CFG_SMALL_TILE
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        if mp % 128 != 0 or np % 128 != 0:
            BM, BN, BK, GM, nw, ns = _PAD_CFG_SMALL
            mp = triton.cdiv(m, BM) * BM
            np = triton.cdiv(n, BN) * BN
        use_buf = True   # 小输出保持 buffer ops 默认 (1023³ buf0 反而略差)
        use_itt = False
    elif m * n >= _PAD_LARGE_OUTPUT:
        # 大输出: (256,256,32,G1,w16,s2) + ITT1 + no-buffer
        # (s2 比 s1 快 13%: 8191³ 0.795->0.924, 4097x8191x4095 0.755->0.869)
        BM, BN, BK, GM, nw, ns = _PAD_CFG_LARGE
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        use_buf = False
        use_itt = True
    else:
        # 中输出: (64,64,32,G8,w4,s2) + ITT1 + no-buffer (v1.10, 4095³ 0.809)
        BM, BN, BK, GM, nw, ns = _PAD_CFG_MID
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        use_buf = False
        use_itt = True
    CCM = CCK = _PAD_COPY_CM
    # kp 需同时为主内核 BLOCK_K(32) 与 pad-copy tile (32) 的整数倍:
    # copy kernel 对 k 维按 CK 分块且 store 无 mask, 若 kp 非 CK 倍数
    # 末块会写到 kp 之外 (如 512x12800x5210, kp=5216 -> 写到 5247 行, 越界 VMFault)。
    # 128 是 32 的倍数, 按 _PAD_COPY_CK(128) 对齐可同时满足两者。
    kp = triton.cdiv(k, _PAD_COPY_CK) * _PAD_COPY_CK
    A_pad, B_pad = _get_pad_buffers(mp, kp, np, A.device)
    grid_pad = (
        triton.cdiv(mp, CCM) * triton.cdiv(kp, CCK)
        + triton.cdiv(kp, CCM) * triton.cdiv(np, CCK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_nn_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            m,
            n,
            k,
            lda,
            ldb,
            mp,
            np,
            kp,
            CM=CCM,
            CK=CCK,
            num_warps=4,  # tile32 下 nw4 优于 nw8: 8191³ copy 948->908, 4095³ 229->223, 4097 359->345
            use_in_thread_transpose=False,  # pad 路径在 ITT=1 下变差 (8191³ 0.862->0.823)
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=use_itt,  # 大/中输出 ITT1, 小输出 ITT0
        )
        if not use_buf:
            # no-buffer 变体 (大/中输出): 显式传 use_buffer_ops=False 进编译选项;
            # use_buf=True 时不传, 保持全局默认产物
            main_kwargs["use_buffer_ops"] = False
        _sgemm_nn_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


def _make_sgemm_nn_splitk_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # per-shape SPLIT_K: 薄大 K shape (128x1024x4096/256x2048x8192) 实测 SPK=4
    # 显著更优 (-18.2%/-12.8%); 512x4096x16384 等 SPK=4 严重退化, 保持 SPK=2。
    # SPK=4 仅在 k 能被 SPLIT_K*BLOCK_K 整除时采用 (保证段内全 EVEN, 无 k remainder)。
    if (m, n, k) in _SPLITK4_SHAPES and k % (_SPLITK_BLOCK_K * 4) == 0:
        spk = 4
    else:
        spk = 2
    scratch = torch.zeros(spk, m, n, dtype=torch.float32, device=C.device)
    ev_m = m % _SPLITK_BLOCK_M == 0
    ev_n = n % _SPLITK_BLOCK_N == 0
    # EVEN_K 按段内余数判断: k_per = k // SPLIT_K, 段内全 EVEN 等价于 k_per % BK == 0
    # (原 k % BK == 0 未考虑 split-k 均分, 例如 k=96/SPK=2 -> k_per=48 %32 != 0)
    ev_k = (k // spk) % _SPLITK_BLOCK_K == 0
    grid_mn = triton.cdiv(m, _SPLITK_BLOCK_M) * triton.cdiv(n, _SPLITK_BLOCK_N)
    grid_main = (grid_mn * spk,)
    grid_reduce = (grid_mn,)

    def run():
        _sgemm_nn_splitk_kernel[grid_main](
            A,
            B,
            scratch,
            m,
            n,
            k,
            lda,
            ldb,
            SPLIT_K=spk,
            EVEN_M=ev_m,
            EVEN_N=ev_n,
            EVEN_K=ev_k,
            BLOCK_M=_SPLITK_BLOCK_M,
            BLOCK_N=_SPLITK_BLOCK_N,
            BLOCK_K=_SPLITK_BLOCK_K,
            GROUP_M=_SPLITK_GROUP_M,
            num_warps=_SPLITK_WARPS,
            num_stages=_SPLITK_STAGES,
            use_in_thread_transpose=False,  # split-k 在 ITT=1 下严重退化 (0.792->0.493)
        )
        _sgemm_nn_splitk_reduce_kernel[grid_reduce](
            scratch,
            C,
            m,
            n,
            alpha,
            beta,
            SPLIT_K=spk,
            BETA_IS_ZERO=beta_is_zero,
            EVEN_M=ev_m,
            EVEN_N=ev_n,
            BLOCK_M=_SPLITK_BLOCK_M,
            BLOCK_N=_SPLITK_BLOCK_N,
            GROUP_M=_SPLITK_GROUP_M,
            num_warps=_SPLITK_WARPS,
            use_in_thread_transpose=False,
        )

    return run


def _make_sgemm_nn_opt_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    use_itt = _sgemm_nn_use_itt(m, n, k)
    use_buf = _sgemm_nn_use_buffer_ops(m, n, k)

    def run():
        kwargs = dict(use_in_thread_transpose=use_itt)
        if not use_buf:
            # no-buffer 变体: 显式传 use_buffer_ops=False 进编译选项 (cache key 区分);
            # use_buf=True 时不传, 保持全局默认产物, 与未注入前完全一致
            kwargs["use_buffer_ops"] = False
        sgemm_nn_hygon_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            **kwargs,
        )

    return run


_SGEMM_NN_DISPATCH = StaticDispatch(
    [
        (_sgemm_nn_is_large_k, _make_sgemm_nn_splitk_runner),
        (_sgemm_nn_is_pad_shape, _make_sgemm_nn_pad_runner),
        (_sgemm_nn_is_default, _make_sgemm_nn_opt_runner),
    ]
)


# ---- nn opt 内核 config override (v2.25, 2026-09-01) ----
# 生产走 libtuner DB best_config, 但以下 shape 的 DB config 非最优:
# 全 config 扫描 (26 config x ITT x buffer) + 官方口径 (warmup=1000/rep=100) 确认
# 手动 launch 以下 config 更快:
#   (256,256,256):     (32,32,BK64,w4,itt0,buf)   12.64->10.40us r 0.911->1.108
#   (1024,1024,1024):  (64,128,BK32,w8,itt1,nobuf) 75.20->70.08us r 0.851->0.913
#   (128,4096,1024):   (64,64,BK16,gm1,w4,itt1,nobuf) 51.04->43.84us r 0.809->0.942
_NN_OPT_OVERRIDES = {
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (1024, 1024, 1024): dict(
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=True, use_buffer_ops=False,
    ),
    (128, 4096, 1024): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=True, use_buffer_ops=False,
    ),
}


def _make_sgemm_nn_opt_override_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov["BLOCK_K"]
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)

    def run():
        kwargs = dict(
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", True),
        )
        if not ov.get("use_buffer_ops", True):
            # no-buffer 变体: 显式传 use_buffer_ops=False 进编译选项
            kwargs["use_buffer_ops"] = False
        _sgemm_nn_opt_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM, **kwargs,
        )

    return run


# ---- tn 复合路径: A^T 缓冲缓存 + runner (v2.3, 2026-08-30) ----
# 与 _get_pad_buffers 同款约定: 按 (m,k,device) 缓存, 避免每次调用分配大 buffer
# (8191³ 的 Ac 达 268MB)。转置内核全量覆盖缓冲 (无 padding), torch.empty 即可。
_tn_ac_cache = {}


def _get_tn_ac(m, k, device):
    key = (m, k, str(device))
    buf = _tn_ac_cache.get(key)
    if buf is None:
        buf = torch.empty(m, k, dtype=torch.float32, device=device)
        _tn_ac_cache[key] = buf
    return buf


def _make_sgemm_tn_composite_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    Ac = _get_tn_ac(m, k, A.device)
    # 纯 int 算术而非 triton.cdiv: cdiv 的 constexpr 包装单次 ~7.8us, runner 每次
    # 调用都会重建 (sgemm 入口), 对小 shape 是 ~15us 的纯 Python 开销
    grid_tp = (
        (m + _TN_TP_BM - 1) // _TN_TP_BM
    ) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    nn_runner = _SGEMM_NN_DISPATCH.lookup_and_build(
        m,
        n,
        k,
        True,
        context=dict(
            A=Ac,
            lda=k,
            B=B,
            ldb=ldb,
            C=C,
            ldc=ldc,
            m=m,
            n=n,
            k=k,
            alpha=alpha,
            beta=beta,
            beta_is_zero=beta_is_zero,
            grid=grid,
        ),
    )

    def run():
        _sgemm_tn_tp_kernel[grid_tp](
            A, Ac, m, k, lda, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM
        )
        nn_runner()

    return run


# ---- tn 复合路径变体 Y: B^T 缓冲缓存 + tt 内核 runner (v2.3, 2026-08-30) ----
# 与 _get_tn_ac 同款约定: 按 (n,k,device) 缓存, 转置内核全量覆盖 (无 padding)。
_tn_bc_cache = {}


def _get_tn_bc(n, k, device):
    key = (n, k, str(device))
    buf = _tn_bc_cache.get(key)
    if buf is None:
        buf = torch.empty(n, k, dtype=torch.float32, device=device)
        _tn_bc_cache[key] = buf
    return buf


def _make_sgemm_tn_composite_y_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    # 转置 B(k,n) ldb=n -> Bc(n,k) 连续, 再走 tt 内核 (A 保持 (k,m) lda=m,
    # strided 放 dot RHS, MFMA 不要求 RHS k 连续)。
    Bc = _get_tn_bc(n, k, B.device)
    # 同上: cdiv constexpr 包装 ~7.8us/次, runner 每次调用重建, 改纯 int 算术
    grid_tp = (
        (n + _TN_TP_BM - 1) // _TN_TP_BM
    ) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    itt = _sgemm_xt_use_itt(m, n, k)
    nobuf = not _sgemm_xt_use_buffer_ops(m, n, k)

    def run():
        _sgemm_tn_tp_kernel[grid_tp](
            B, Bc, n, k, ldb, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM
        )
        sgemm_tt_hygon_kernel[grid](
            A, Bc, C, alpha, beta, m, n, k, lda, k, ldc, beta_is_zero,
            use_in_thread_transpose=itt,
            **(_XT_NOBUF_KWARGS if nobuf else {}),
        )

    return run


# ---- nt 复合路径 (At+TT, v2.8, 2026-08-31) ----
# C = A B^T 的直连内核 a_t 按 (BK,BM) 读 A 列块 (stride-lda 碎片), 是大 shape 的
# 主要瓶颈 (16384^3 nt 直连 184.2ms vs hipBLAS ref 149.9ms)。方案: 一次转置
# A(m,k)->At(k,m) (~2ms), 再走 tt 内核 (A 侧变 1KB 连续, B 侧 128B), 与直连同
# 结构但总时间更低。实测 (生产 tt 入口 + 池内 256x256 G8, warmup=25/rep=25):
# 16384^3 0.884->0.914、8192^3 0.885->0.907、8192x28672x8192 0.888->0.925、
# 4096x8192x28672 0.830->0.851; 中/小 shape 转置+双内核开销占比高, 直连更优。
_nt_at_cache = {}


# kpack=2 逐 shape 例外表 (v2.12)。官方口径 do_bench(warmup=1000/rep=100/median)
# 实测: 这些直连 shape 上 kpack=2 比 kpack=1 慢 >=1% (d 列), 回退到 kpack=1
# 消除回退; 其余直连 shape kpack=2 净正 (2048x4096x11008 -4.3% 等)。
#   (64,512,512)  +1.7% | (512,64,512) +2.5% | (512,512,512) +3.1%
#   (511,511,511) +1.1% | (4096,128,1024) +1.0% | (2048,2048,16384) +1.4%
_NT_KPACK1_SHAPES = {
    (64, 512, 512),
    (512, 64, 512),
    (512, 512, 512),
    (511, 511, 511),
    (4096, 128, 1024),
    (2048, 2048, 16384),
}


def _sgemm_nt_direct_kpack(m, n, k):
    return 2 if (m, n, k) not in _NT_KPACK1_SHAPES else 1


def _get_nt_at(k, m, device):
    key = (k, m, str(device))
    buf = _nt_at_cache.get(key)
    if buf is None:
        buf = torch.empty(k, m, dtype=torch.float32, device=device)
        _nt_at_cache[key] = buf
    return buf


def _sgemm_nt_use_composite(m, n, k, **_kw):
    # 仅对齐大 shape: 转置内核与 tt 内核均要求 32 对齐; 与 pad 路径互斥
    # (8191^3 等非对齐 shape 继续走 pre-pad)。阈值 m*n*k>=5e11 且 min(m,n)>=4096
    # 覆盖 16384^3/8192^3/8192x28672x8192/4096x8192x28672, 排除 4096^3 (直连 0.917
    # 更优) 与 2048x2048x16384 (min=2048, 转置开销吃掉内核收益)。
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return False
    if min(m, n) < 4096:
        return False
    return m * n * k >= 5 * 10**11


def _make_sgemm_nt_composite_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    At = _get_nt_at(k, m, A.device)
    # 转置参数语义: _sgemm_tn_tp_kernel(src, dst, m=dst行数, k=dst列数, lda=src行stride);
    # A(m,k) lda=k -> At(k,m): (A, At, k, m, lda)。
    grid_tp = (
        (k + _TN_TP_BM - 1) // _TN_TP_BM
    ) * ((m + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    itt = _sgemm_xt_use_itt(m, n, k)
    nobuf = not _sgemm_xt_use_buffer_ops(m, n, k)

    def run():
        _sgemm_tn_tp_kernel[grid_tp](
            A, At, k, m, lda, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM
        )
        # kpack=2 (HIPOptions 编译选项, 每指令打包 2 个 k 元素) 实测对全部复合
        # shape 有效 (v2.11, scan4: 16384^3 0.915->0.925、8192x28672x8192 0.925
        # ->0.936 等); 该选项经 libentry 透传并纳入 entry_key, 与 TT 路径的
        # kpack=1 产物互不串用。
        sgemm_tt_hygon_kernel[grid](
            At, B, C, alpha, beta, m, n, k, m, ldb, ldc, beta_is_zero,
            use_in_thread_transpose=itt,
            kpack=2,
            **(_XT_NOBUF_KWARGS if nobuf else {}),
        )

    return run


# ---- sgemm 入口级 runner 缓存 (v2.5, 2026-08-30) ----
# nn 分发 lookup_and_build 与 composite runner 每次调用都会重建 (~7-17us 纯 Python
# 开销, cdiv constexpr 包装 ~7.8us/次)。对微秒级 shape (64x512x512 等) 开销占比大,
# 按 (shape, stride, 标量, 张量 data_ptr) 缓存 runner 闭包, 命中时直接复用。
# data_ptr 参与 key: 不同张量必然重建; 闭包持引用防地址复用误命中。上限防无限增长。
# v2.10 (2026-08-31): 闭包持 A/B/C 引用, 无差别缓存会长期钉住大 shape 张量 (全量
# benchmark 37-shape × 4 变体累积后 16384³ 单 runner 12GB 导致 OOM)。启动开销只对
# 微秒级 shape 有意义, 按钉住总元素数门控: 大 shape 每次调用重建 (~10-20us, 相对
# 毫秒级内核可忽略), 张量随调用结束即可释放。
_sgemm_runner_cache = {}
_sgemm_runner_cache_max = 8192
# 单 runner 钉住上限 16M 元素 (~64MB fp32): 覆盖 ≤2048³ 的常规小/中 shape。
_SGEMM_RUNNER_CACHE_MAX_PIN = 16 * 2**20


def _cache_sgemm_runner(key, runner):
    # key = (transa, transb, m, n, k, ...); 按钉住总元素数门控
    m, n, k = key[2], key[3], key[4]
    if m * k + n * k + m * n > _SGEMM_RUNNER_CACHE_MAX_PIN:
        return runner  # 大 shape 不缓存, 避免闭包钉住数百 MB~GB 张量
    if len(_sgemm_runner_cache) < _sgemm_runner_cache_max:
        _sgemm_runner_cache[key] = runner
    return runner


def sgemm(
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
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    beta_is_zero = beta == 0.0
    # 纯 int 算术 (不用 triton.cdiv): 避免 constexpr 包装开销, grid 每次启动都会求值
    grid = lambda meta: (
        ((m + meta["BLOCK_M"] - 1) // meta["BLOCK_M"])
        * ((n + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]),
    )

    # runner 按 (shape, stride, 标量, 张量指针) 缓存: nn 分发/composite runner 每次
    # 调用重建 (lookup_and_build ~7us + grid_tp 构造), 对微秒级 shape 是显著开销。
    # data_ptr 参与 key 保证不同张量必然重建, 闭包持有的旧张量不会被释放复用地址。
    key = (
        transa,
        transb,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        alpha,
        beta,
        beta_is_zero,
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        str(A.device),
    )

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            if (m, n, k) in _NN_OPT_OVERRIDES:
                # nn opt 内核 config override (v2.25): DB best_config 非最优,
                # 手动 launch 扫描确认的最优 config (官方口径 256³ 1.108/1024³ 0.913/
                # 128x4096x1024 0.942)。
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nn_opt_override_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            ov=_NN_OPT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            else:
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _SGEMM_NN_DISPATCH.lookup_and_build(
                        m,
                        n,
                        k,
                        True,
                        context=dict(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            grid=grid,
                        ),
                    )
                _cache_sgemm_runner(key, runner)
                runner()
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if (m, n, k) in _TN_DIRECT_OVERRIDES:
                # tn 直连 config override (v2.22): 小 shape 生产 config 非最优,
                # 缩小 tile 提并行度 (官方口径 A/B 全部 >=1.0)。
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tn_direct_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            ov=_TN_DIRECT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            elif _sgemm_xt_is_pad_shape(m, n, k):
                # 非对齐大 shape: A(k,m)->A_pad(kp,mp) + B(k,n) 转置->B_pad(np,kp),
                # 复用 tt even main (v2.8, 4095³ 0.880->~1.0/8191³ 0.918->~0.96/
                # 4097x8191x4095 0.934->~1.0)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tn_pad_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            grid=grid,
                        ),
                    )
                runner()
            elif _sgemm_tn_use_composite_y(m, n, k):
                if (m, n, k) in _TN_COMPY_OVERRIDES:
                    # compY tt 内核 config override (v2.24): 大 shape 手动 launch
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_compy_override_runner(
                                A=A,
                                lda=lda,
                                B=B,
                                ldb=ldb,
                                C=C,
                                ldc=ldc,
                                m=m,
                                n=n,
                                k=k,
                                alpha=alpha,
                                beta=beta,
                                beta_is_zero=beta_is_zero,
                                grid=grid,
                                ov=_TN_COMPY_OVERRIDES[(m, n, k)],
                            ),
                        )
                    runner()
                else:
                    # 变体 Y: n≪m 时只转置 B^T (成本 n*k) 比方案A (m*k) 便宜, 走 tt 内核
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_composite_y_runner(
                                A=A,
                                lda=lda,
                                B=B,
                                ldb=ldb,
                                C=C,
                                ldc=ldc,
                                m=m,
                                n=n,
                                k=k,
                                alpha=alpha,
                                beta=beta,
                                beta_is_zero=beta_is_zero,
                                grid=grid,
                            ),
                        )
                    runner()
            elif _sgemm_tn_use_composite(m, n, k):
                if (m, n, k) in _TN_COMP_OVERRIDES:
                    # comp 路径 nn 内核 config override (v2.23): 4096³ 用扫描最优
                    # (64,128,w8,itt1,gm1,BK32,s2,nobuf) 替换 tuner config0
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_comp_override_runner(
                                A=A,
                                lda=lda,
                                B=B,
                                ldb=ldb,
                                C=C,
                                ldc=ldc,
                                m=m,
                                n=n,
                                k=k,
                                alpha=alpha,
                                beta=beta,
                                beta_is_zero=beta_is_zero,
                                grid=grid,
                                ov=_TN_COMP_OVERRIDES[(m, n, k)],
                            ),
                        )
                    runner()
                else:
                    # 复合路径: 转置 A^T -> 连续 (m,k) 缓冲 -> nn 分发
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_composite_runner(
                                A=A,
                                lda=lda,
                                B=B,
                                ldb=ldb,
                                C=C,
                                ldc=ldc,
                                m=m,
                                n=n,
                                k=k,
                                alpha=alpha,
                                beta=beta,
                                beta_is_zero=beta_is_zero,
                                grid=grid,
                            ),
                        )
                    runner()
            else:
                sgemm_tn_hygon_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    use_in_thread_transpose=_sgemm_xt_use_itt(m, n, k),
                    **(_XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}),
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _nt_spk = _sgemm_nt_splitk(m, n, k)
            if _nt_spk:
                # 瘦 shape split-k (v2.13): blocks 仅 8-32 时按 k 分段并行,
                # (64,512,512) 等 7 shape 官方口径快 6-31%。白名单精确覆盖。
                _nt_spk_w = _NT_SPLITK_SHAPES[(m, n, k)][1]
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_splitk_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            SPLIT_K=_nt_spk,
                            num_warps=_nt_spk_w,
                        ),
                    )
                runner()
            elif (m, n, k) in _NT_DIRECT_OVERRIDES:
                # 直连 config override (v2.13): 511³ 等生产 config 非最优, 固定手动
                # launch raw 内核 (w16/itt0/kp1, 官方口径 0.745->0.93)。
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_direct_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            ov=_NT_DIRECT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            elif _sgemm_nt_use_composite(m, n, k):
                # 对齐大 shape: A^T 转置 + tt 内核 (v2.8), 避开直连 a_t 的 stride-lda
                # 碎片读 (16384^3 nt 直连 0.884 -> 复合 0.914 等, 见 _make_sgemm_nt_composite_runner)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_composite_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            grid=grid,
                        ),
                    )
                runner()
            elif _sgemm_xt_is_pad_shape(m, n, k):
                # 非对齐大 shape: pre-pad A/B 到对齐缓冲, 主内核全 EVEN load,
                # 避免 mask/k remainder 开销 (v2.6, 4095³ 0.755->0.840 等)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_pad_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            grid=grid,
                        ),
                    )
                runner()
            else:
                # kpack=2 实测对多数直连 shape 有效 (v2.11: 2048x4096x11008 0.918
                # ->0.949-0.970、16384x512x4096 0.908->0.942、512x16384x4096 0.938
                # ->0.949 等). 个别 shape 上 kpack=2 反而变慢 (官方口径
                # do_bench(1000/100/median) 实测慢 >=1%), 走 _NT_KPACK1_SHAPES
                # 例外表回退到 kpack=1, 消除小 shape 回退 (v2.12).
                sgemm_nt_hygon_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    use_in_thread_transpose=_sgemm_xt_use_itt(m, n, k),
                    kpack=_sgemm_nt_direct_kpack(m, n, k),
                    **(_XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}),
                )
        else:
            if (m, n, k) in _TT_DIRECT_OVERRIDES:
                # tt 直连 config override (v2.21): 小 shape 生产 config 大 tile 下
                # grid 并行度不足, 缩小 tile 提并行度 (官方口径 A/B 全部 >=1.0)。
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tt_direct_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            ov=_TT_DIRECT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            elif _sgemm_xt_is_pad_shape(m, n, k):
                # 非对齐大 shape: pre-pad A(k,m)->A_pad(kp,mp)/B(n,k)->B_pad(np,kp),
                # 全 EVEN main (v2.7, 8191³ 0.883->1.037、4097x8191x4095 0.826->0.940)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tt_pad_runner(
                            A=A,
                            lda=lda,
                            B=B,
                            ldb=ldb,
                            C=C,
                            ldc=ldc,
                            m=m,
                            n=n,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            beta_is_zero=beta_is_zero,
                            grid=grid,
                        ),
                    )
                runner()
            else:
                sgemm_tt_hygon_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    use_in_thread_transpose=_sgemm_xt_use_itt(m, n, k),
                    **(_XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}),
                )


__all__ = ["sgemm"]

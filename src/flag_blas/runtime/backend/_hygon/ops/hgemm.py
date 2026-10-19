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
Hygon DTK hgemm 定制路径。

hygon 侧此前无 `hgemm` 专用实现, 直接复用通用 level3 内核 + NVIDIA 缺省 tune 池。
本模块对齐 `_hygon/ops/bfgemm.py` 的既有模式, 为 fp16 gemm 增加两条 hygon 定制路径,
其余布局仍复用通用 level3 内核:

1. TN (transa=T, transb=N): A 物理 (k, m) 行主序, 直接主循环沿 k 步进时 A tile 的
   末维 (k) 跨度为 lda, 无法向量化/连续读, 带宽效率差。这里先用专用转置拷贝内核把
   A(k,m) 落成 A^T(m,k) 连续缓冲区, 再走 NN 主内核, 使 A 的加载与 NN 变体一致。
   转置内核读写均按连续片段进行, 开销远小于直接 TN 的跨步读损失。

2. odd / 非对齐布局 (任一 ld % 8 != 0): 行起始地址 = row * ld * 2B, ld 非 8 倍数时
   连 16B 对齐都无法保证, Triton 对 2D tile 行向量化 load/store 全面降窄。对大 shape
   把输入 pad 到对齐缓冲 (NN 顺带把 B 转置成 Bt 走 NT 主内核), 再裁回真实 C。

NN / NT / TT 与通用 level3 内核一致 (实测 median >= 0.95, 不额外处理), 仅补 hygon
专用 tune 池 (tune_configs.yaml 的 hgemm_* 段), 避免回落 NVIDIA 缺省池。

注: bfgemm 侧按 shape 白名单叠加的 sched_latency / kpack / NN->NT 路由等编译器级调优
均为 bf16 实测结论, 未在本模块搬运; hgemm 仅保留与 dtype 无关的结构路径。
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.hgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _hgemm_nn_kernel,
    _hgemm_nt_kernel,
    _hgemm_tn_kernel,
    _hgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn

# TN 复合转置内核 tile; 末维 (load 沿 m, store 沿 k) 均连续可向量化。
# 与 _hygon/ops/bfgemm.py 同值 (同为 2B dtype, 转置带宽特性一致)。
_TN_TRANS_BLOCK_M = 128
_TN_TRANS_BLOCK_K = 128


@triton.jit
def _hgemm_tn_trans_kernel(
    src,
    dst,
    K,
    M,
    sr,
    dr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: A 物理 (k, m) 行主序 (k 行 stride=sr, m 列 stride=1)
    # dst: A^T 物理 (m, k) 行主序 (m 行 stride=dr, k 列 stride=1) -> dst = src^T
    pid = tl.program_id(0)
    ngrid_c = tl.cdiv(M, BC)
    pr = pid // ngrid_c
    pc = pid % ngrid_c
    offs_r = pr * BR + tl.arange(0, BR)  # k
    offs_c = pc * BC + tl.arange(0, BC)  # m
    msk = (offs_r[:, None] < K) & (offs_c[None, :] < M)
    v = tl.load(src + offs_r[:, None] * sr + offs_c[None, :], mask=msk)
    tl.store(
        dst + offs_c[:, None] * dr + offs_r[None, :],
        tl.trans(v),
        mask=tl.trans(msk),
    )


def _tn_use_composite(m, n, k):
    # A^T composite (transpose A + NN main) 仅在大输出非瘦 shape 上净正收益;
    # 瘦/小 shape (m 或 n <= 64) 直接 TN 更快, 故设下界。判据与 bfgemm 一致。
    return min(m, n) >= 128 and m * n * k >= (4 << 27)


# ---- odd / 非对齐布局的 pad 路径 ----
# 判据 / 阈值与 _hygon/ops/bfgemm.py 一致: fp16 与 bf16 同为 2B dtype, 行对齐条件
# (ld % 8 == 0 对应 16B 行起始对齐) 与 pad tile 完全相同。
_PAD_ALIGN_MN = 256  # 池内最大 BLOCK_M/BLOCK_N
_PAD_ALIGN_K = 128  # 池内最大 BLOCK_K
_PAD_BLK = 64  # 转置拷贝 tile 宽 BC (src n 向)
_PAD_TRANS_BLK = 128  # 转置拷贝 tile 高 BR (src k 向; k 行主序跨大 stride 读)
_PAD_MIN_FLOPS = 5.0e9  # 启用 pad 的 2*m*n*k 下界 (小 odd shape 启动开销占优, 不 pad)


def _padded_size(m, n, k, lda, ldb, ldc):
    # 行起始地址 16B 对齐需 ld*2B % 16 == 0 -> ld % 8 == 0。任一不满足即
    # misalign -> 需要 pad (覆盖全部奇数 shape 与 lda/ldb/ldc 非 8 倍数情形)。
    if lda % 8 == 0 and ldb % 8 == 0 and ldc % 8 == 0:
        return None
    return (
        triton.cdiv(m, _PAD_ALIGN_MN) * _PAD_ALIGN_MN,
        triton.cdiv(n, _PAD_ALIGN_MN) * _PAD_ALIGN_MN,
        triton.cdiv(k, _PAD_ALIGN_K) * _PAD_ALIGN_K,
    )


def _row_view(t, ld):
    # 物理缓冲 -> (行, ld) 行主序视图 (无拷贝), 供切片拷贝取逻辑 (m,k) 区;
    # 兼容 2D (行,ld) 与 1D 平铺缓冲。
    if t.dim() == 2 and t.shape[1] == ld:
        return t
    return t.view(-1, ld)


@triton.jit
def _hgemm_pad_trans_kernel(
    src,
    dst,
    K,
    N,
    sr,
    dr,
    np,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: B(k,n) 行主序 (k 行 stride=sr, n 列连续)
    # dst: Bt(n_pad,k_pad) 行主序 (n 行 stride=dr, k 列连续); 一次完成 pad+转置,
    # grid = cdiv(k_pad,BR) x cdiv(n_pad,BC) 全覆盖 -> store 无掩码; 源越界
    # (k>=K 或 n>=N) 读 0 -> pad 区为 0。
    pid = tl.program_id(0)
    ngrid_c = tl.cdiv(np, BC)
    pr = pid // ngrid_c  # k tiles (k_pad 维)
    pc = pid % ngrid_c  # n tiles
    offs_k = pr * BR + tl.arange(0, BR)
    offs_n = pc * BC + tl.arange(0, BC)
    msk = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    v = tl.load(src + offs_k[:, None] * sr + offs_n[None, :], mask=msk, other=0.0)
    tl.store(dst + offs_n[:, None] * dr + offs_k[None, :], tl.trans(v))


def _launch_pad_nt(
    A,
    Ap,
    Bt,
    Cp,
    alpha,
    beta,
    beta_is_zero,
    mp,
    np,
    kp,
    lda_p,
):
    # NT 主内核 (输入 Ap(m,kp) lda=lda_p, Bt(np,kp) ldb=kp) -> Cp(mp,np);
    # grid lambda 交给 libtuner 展开。
    grid = lambda meta: (
        triton.cdiv(mp, meta["BLOCK_M"]) * triton.cdiv(np, meta["BLOCK_N"]),
    )
    _hgemm_nt_kernel[grid](
        Ap,
        Bt,
        Cp,
        alpha,
        beta,
        mp,
        np,
        kp,
        lda_p,
        kp,
        np,
        beta_is_zero,
    )


def hgemm(
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
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert C.dtype == torch.float16
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
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(A.device):
        # odd/misalign 布局 (任一 ld%8!=0) + 大 shape: pad 到对齐缓冲统一走 NT 主
        # 内核 (B 沿 k 列连续读)。有效输出 (i<m,j<n) 的 k 方向点积只跨 A/B 的 k-pad
        # 列, 故仅需把两缓冲的 [:, k:kp] 条带清零; m-pad 行 / n-pad 行 / Cp 的 pad 区
        # 只影响被裁掉的输出, 无需清零。alpha/beta 语义保持。
        ps = _padded_size(m, n, k, lda, ldb, ldc)
        if ps is not None and 2.0 * m * n * k >= _PAD_MIN_FLOPS:
            mp, np, kp = ps
            Ap = torch.empty(mp, kp, dtype=A.dtype, device=A.device)
            Bt = torch.empty(np, kp, dtype=A.dtype, device=A.device)
            Cp = torch.empty(mp, np, dtype=A.dtype, device=A.device)
            if transa == CUBLAS_OP_N:  # A 物理 (m,k): 同布局 pad
                Ap[:m, :k].copy_(_row_view(A, lda)[:m, :k])
            else:  # A 物理 (k,m): pad+转置 -> Ap(m,k)
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(mp, _PAD_BLK),
                )
                _hgemm_pad_trans_kernel[g](
                    A,
                    Ap,
                    k,
                    m,
                    lda,
                    kp,
                    mp,
                    BR=_PAD_TRANS_BLK,
                    BC=_PAD_BLK,
                    num_warps=8,
                )
            if transb == CUBLAS_OP_N:  # B 物理 (k,n): pad+转置 -> Bt(n,k)
                b_pad_trans = True
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(np, _PAD_BLK),
                )
                _hgemm_pad_trans_kernel[g](
                    B,
                    Bt,
                    k,
                    n,
                    ldb,
                    kp,
                    np,
                    BR=_PAD_TRANS_BLK,
                    BC=_PAD_BLK,
                    num_warps=8,
                )
            else:  # B 物理 (n,k): 同布局 pad
                b_pad_trans = False
                Bt[:n, :k].copy_(_row_view(B, ldb)[:n, :k])
            if kp > k:
                # k-pad 条带归零: 仅 same-layout copy_ 的缓冲需要 (copy_ 只写
                # [:m,:k]/[:n,:k], k-pad 是 torch.empty 残留); 转置 kernel 的
                # store 覆盖整块缓冲且源越界处读 0 写入, k-pad 已为 0, 跳过。
                if transa == CUBLAS_OP_N:  # A 同布局 copy, k-pad 需归零
                    Ap[:, k:].zero_()
                if not b_pad_trans:
                    Bt[:, k:].zero_()
            if not beta_is_zero:  # 预置 beta*C 语义: 拷贝 C 到 pad 缓冲左上
                Cp[:m, :n].copy_(_row_view(C, ldc)[:m, :n])
            _launch_pad_nt(A, Ap, Bt, Cp, alpha, beta, beta_is_zero, mp, np, kp, kp)
            _row_view(C, ldc)[:m, :n].copy_(Cp[:m, :n])
            return

        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _hgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if _tn_use_composite(m, n, k):
                # TN 复合: A(k,m) 行主序 (行长=lda) -> A^T(m,k) 行主序 (行长=k),
                # 再走 NN 主内核, 使 A 的加载与 NN 变体一致 (末维连续)。
                At = torch.empty((m, k), dtype=A.dtype, device=A.device)
                g = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(m, _TN_TRANS_BLOCK_M),
                )
                _hgemm_tn_trans_kernel[g](
                    A,
                    At,
                    k,
                    m,
                    lda,
                    k,
                    BR=_TN_TRANS_BLOCK_K,
                    BC=_TN_TRANS_BLOCK_M,
                    num_warps=8,
                )
                _hgemm_nn_kernel[grid](
                    At, B, C, alpha, beta, m, n, k, k, ldb, ldc, beta_is_zero
                )
            else:
                _hgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _hgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _hgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )

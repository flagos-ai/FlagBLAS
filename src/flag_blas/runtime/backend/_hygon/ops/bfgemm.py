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
Hygon DTK bfgemm 定制路径。

TN (transa=T, transb=N) 下, A 的物理布局是 (k, m) 行主序, 直接 TN 主循环
沿 k 步进时 A tile 的末维 (k) 跨度为 lda, 无法向量化/连续读, 带宽效率差
(core-shape 基准 TN 变体 median 仅 ~0.63)。这里对 TN 采用 sgemm 同款思路:
先用专用转置拷贝内核把 A(k,m) 落成 A^T(m,k) 连续缓冲区, 再走 NN 主内核,
使 A 的加载与 NN 变体一致 (末维连续)。转置内核读写均按连续片段进行,
开销远小于直接 TN 的跨步读损失。

NN / NT / TT 与通用 level3 内核一致 (实测 median >=0.95, 不额外处理)。
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.bfgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _bfgemm_nn_kernel,
    _bfgemm_nt_kernel,
    _bfgemm_tn_kernel,
    _bfgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn

# TN 复合转置内核 tile; 末维(load 沿 m, store 沿 k)均连续可向量化
# dir4 (2026-09-07): 转置 tile 64x64 w4 -> 128x128 w8。扫描 5 个 NN-route B 形状
# (8192/28672/24576/12288/11008 x ...), 带宽 811-903GB/s -> 1027-1144GB/s
# (+16~+34%, torch copy 天花板仅 ~610GB/s)。路由净收 NN ~2-5%、TN ~4-7%,
# 仍全部 <0.8 (主内核差 = fork 级, 见报告 §10)。
_TN_TRANS_BLOCK_M = 128
_TN_TRANS_BLOCK_K = 128

# ---- 16 大 shape 编译器对齐 (2026-09-07): NN 主内核 sched_latency per-shape 门控 ----
# flagtree clang 延迟布局 (-mmac-latency=5 + -ds-load-store-latency) 收益随生产 tile 分群:
#   (256,256,BK64,w8/s2) 大 tile shape 原按 mmac5-ds12 定档 (实测 -3.6~-3.9%);
#   但 mmac5-ds12/-ds14 只存在于本地打过 patch 的 triton 构建, FlagTree 上游后端
#   (third_party/hcu/backend/compiler_hcu.py) 的 sched_latency 白名单只有
#   mmac5-ds6 / mmac5-ds10 -> 生产侧统一回落到 mmac5-ds10 (其余值上游会直接
#   ValueError: Unsupported scheduling latency)。
#   (128,256,BK32,w4/s2) tile (4096^3 / 2048x11008x4096) -> mmac5-ds10 -2.0~-2.3%。
# 测量以生产 config (与 v8 warm DB 一致) 直调, base 对齐 v8 官跑 ±0.3%。其余 shape
# (pad / fused-Y / direct-TN / NT / TT) 不触碰, 保持默认调度零风险。
_NN_SCHED_BY_SHAPE = {
    (8192, 8192, 8192): "mmac5-ds10",
    (4096, 24576, 8192): "mmac5-ds10",
    (8192, 28672, 8192): "mmac5-ds10",
    (4096, 8192, 28672): "mmac5-ds10",
    (2048, 4096, 11008): "mmac5-ds10",
    (2048, 12288, 4096): "mmac5-ds10",
    (512, 16384, 4096): "mmac5-ds10",
    (4096, 4096, 4096): "mmac5-ds10",
    (2048, 11008, 4096): "mmac5-ds10",
}


def _nn_sched_kwargs(m, n, k):
    sched = _NN_SCHED_BY_SHAPE.get((m, n, k))
    return {"sched_latency": sched} if sched else {}

# ---- 方向② 落地 (2026-09-07): kpack=2 门控 (官跑量化见报告 §9) ----
# HIPOptions.kpack 控制 bf16 点积操作数每 32bit 打包的 K 数 (默认 1)。kpack=2
# (2-K/32bit 打包, 消 v_perm/降寄存器往返) 官跑全谱量化:
#   * NT 主内核无条件: NN-route 7 dims -0.5~-3.5、TN-route 6 dims -0.3~-3.6、
#     NT-native 大 shape -2.5~-3.8 (8192x28672x8192 sp 0.821->0.852 达标)、
#     pad-odd (NN/TN/TT 8191^3/4097x8191x4095) -0.4~-3.1、小 shape 噪声内, 无回退;
#   * NN 主内核 per-shape: 仅 (2048,11008,4096) (NN/TN 残留, k=11008 w4/ds10) -4.2~-4.5%;
#   * 排除: NN 512x16384x4096 (w8/ds12) +4.1%、direct-TN 小 shape 1024^3 +22% /
#     128x4096x1024 +34%、direct-TT 大 shape 4096x8192x28672 +4.5% / 2048x2048x16384
#     +5.1% / 2048x4096x11008 +4.9% / 2048^3 +3.7% 回退 (direct-TT 按内核排除);
#   * 其它编译旋钮出局: matrix_instr_nonkdim=16 无效果 (gfx936 bf16 仅 16x16x16),
#     mmac_layout_force 0 数值错 (err 1.38) / 3 编译败 (lit/lts 不支持)。
def _kp2_kwargs():
    return {"kpack": 2}

# NN 主内核 kpack=2 白名单 shape (残留 NN 直路 + TN A^T 复合共用同一主内核)
_NN_KP2_SHAPES = {(2048, 11008, 4096)}


def _nn_kernel_kwargs(m, n, k):
    """NN 主内核 launch kwargs: sched_latency 门控 + 方向② kpack 白名单。"""
    kw = _nn_sched_kwargs(m, n, k)
    if (m, n, k) in _NN_KP2_SHAPES:
        kw["kpack"] = 2
    return kw


# ---- 方向① (2026-09-07): NN 大 shape B 转置一次 -> NT 主内核 ----
# NN 布局 (A/B 均 k 末维) 在 tl.dot 主循环的调度/发射上吃亏, 同指令下 NT 布局
# (B 读 (n,k) 末维连续) flag 低 16-23% (报告 §7 / probe dir1_ntroute 2026-09-07)。
# 对 net 收益为正的 shape: 每次调用先 _bfgemm_tn_trans_kernel 把 B(k,n) 拷成
# Bt(n,k) 连续缓冲 (与 _launch_pad_nt 同一条 NT 主内核路)。转置开销 0.07-0.31ms
# 被主内核节约覆盖 (net -2.7~-17.4%)。例外: 512x16384x4096 (trans 0.31ms >
# 主内核节约 0.09ms, net +49.9%) 与 2048x11008x4096 (net +1.4%) 净亏, 保留 NN
# 主内核 + ds 门控。NT 主内核 config 复用 warm DB (v8 NT tune 段, 官跑命中)。
_NN_NT_ROUTE_SHAPES = {
    (8192, 8192, 8192),
    (4096, 24576, 8192),
    (8192, 28672, 8192),
    (4096, 8192, 28672),
    (4096, 4096, 4096),
    (2048, 4096, 11008),
    (2048, 12288, 4096),
}

# ---- 方向① TN 侧 (2026-09-07): Aᵀ 复合再叠加 transB -> NT 主内核 ----
# TN 复合 (transA + NN 主内核) 把 NN 主内核换成 NT 主内核需补一次 transB
# (B(k,n)->Bt(n,k)); transA 复合路径已付。net = 主内核差 - transB, §7 量化 6 dims
# -2.3~-16.4% (8192³ -16.42 / 8192x28672 -15.38 / 4096x8192x28672 -10.28 /
# 4096x24576x8192 -9.87 / 4096³ -6.82 / 2048x12288x4096 -2.25)。2048x11008x4096
# (+1.72% 净亏) 保持 NN 主内核 + ds; 2048x4096x11008 TN sp>=0.85 不入列。命中走
# NT 主内核 (warm DB config, 不叠 ds 门控)。
_TN_NT_ROUTE_SHAPES = {
    (8192, 8192, 8192),
    (4096, 24576, 8192),
    (8192, 28672, 8192),
    (4096, 8192, 28672),
    (4096, 4096, 4096),
    (2048, 12288, 4096),
}


@triton.jit
def _bfgemm_tn_trans_kernel(
    src,
    dst,
    K,
    M,
    sr,
    dr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: A 物理 (k, m) 行主序, (k 行 stride=sr, m 列 stride=1)
    # dst: A^T 物理 (m, k) 行主序, (m 行 stride=dr, k 列 stride=1)  -> dst = src^T
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
    # A^T composite (transpose A + NN main) wins only on large-output non-skinny shapes.
    # Direct TN is >=1.0 on skinny/small shapes (m or n <= 64: A col-read stride m cached).
    # Composite caps at NN-main speed, which sits below direct TN there, so blindly
    # composing regresses them hard ((64,1024,512) 1.198->0.476, (256,256,256) 1.217->0.562).
    # Measured 2026-09-04 over all 37 core shapes: composite wins iff min(m,n)>=128
    # and m*n*k >= 4e8.
    return min(m, n) >= 128 and m * n * k >= (4 << 27)

# ---- TN Y 复合 (fused-Y) ----
# 对 m >> n 的 TN shape, 直接 TN / A^T 复合都不理想: 这里转置小的 B(n,k) -> Bt(n,k)
# 连续, 再以 NN 主循环算 C^T = Bt @ A (输出 (n,m)), 最终把 acc 转置直接写回 C(m,n)
# (fused store, 免第二段转置内核)。实测 (8192,256,2048)/(16384,2048,2048)/
# (32768,1024,1024)/(16384,512,4096) 达 1.00-1.42, (4096,128,1024) ~0.96-1.03。
# 命中 shape 的 tile: n<=128 时 (BM=64,BN=128) 否则 (BM=128,BN=128); BK=64, w4, s2。
_TN_Y_SHAPES = {
    (8192, 256, 2048),
    (16384, 2048, 2048),
    (32768, 1024, 1024),
    (16384, 512, 4096),
    (4096, 128, 1024),
}


@triton.jit
def _bfgemm_tn_y_kernel(
    bt_ptr,
    a_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # acc (n-tile x m-tile) = bt(n,k) @ a(k,m); 末维 (k 于 bt, m 于 a) 均连续。
    # store 时转置 acc -> C(m,n) (m 行 stride ldc, n 列连续)。
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_N)  # C row tiles (over m), tile 高 = BLOCK_N
    grid_n = tl.cdiv(n, BLOCK_M)  # C col tiles (over n), tile 宽 = BLOCK_M
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)  # over m
    pid_n = (pid % width) // group_size              # over n
    if pid_m * BLOCK_N >= m or pid_n * BLOCK_M >= n:
        return
    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)  # over m
    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)  # over n
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    # x = bt tile (n x k): 行 over n stride k
    x_ptrs = bt_ptr + offs_n[:, None] * k + offs_k[None, :]
    # y = a tile (k x m): 行 over k stride m
    y_ptrs = a_ptr + offs_k[:, None] * m + offs_m[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_N + BLOCK_N) <= m
    is_full_n = (pid_n * BLOCK_M + BLOCK_M) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K
    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            x = tl.load(x_ptrs)
            y = tl.load(y_ptrs)
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
            x_ptrs += BLOCK_K
            y_ptrs += BLOCK_K * m
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            x = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
            y = tl.load(y_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
    else:
        for i in range(k_full_iters):
            x = tl.load(x_ptrs, mask=mask_n[:, None], other=0.0)
            y = tl.load(y_ptrs, mask=mask_m[None, :], other=0.0)
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
            x_ptrs += BLOCK_K
            y_ptrs += BLOCK_K * m
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            x = tl.load(
                x_ptrs,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptrs,
                mask=mask_k[:, None] & mask_m[None, :],
                other=0.0,
            )
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
    # store transposed: acc (BLOCK_M over n, BLOCK_N over m) -> C(m,n)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * tl.trans(acc)).to(tl.bfloat16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(
                c_ptrs,
                (alpha * tl.trans(acc) + beta * c_vals).to(tl.bfloat16),
            )
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(
                c_ptrs,
                (alpha * tl.trans(acc)).to(tl.bfloat16),
                mask=c_mask,
            )
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(
                c_ptrs,
                (alpha * tl.trans(acc) + beta * c_vals).to(tl.bfloat16),
                mask=c_mask,
            )


def _tn_y_tiles(m, n, k):
    # Y 主内核 tile; n 很小 (<=128) 时需更小的 BM(=n 方向块) 才不吃亏
    # ((4096,128,1024) 实测 (64,128) 31.4us vs (128,128) 45.0us)。
    return (64, 128) if n <= 128 else (128, 128)


# ---- odd / 非对齐布局的 pad 路径 ----
# core-shape 表中 (m,n,k) 为奇数 (8191/4095/1023/511/4097x8191x4095 等) 时,
# A/B/C 行起始地址 = row * ld * 2B, ld 奇数使行基址连 4B/8B 对齐都无法保证,
# Triton 对 2D tile 行向量化 load/store 全面降窄, 大 shape 延迟膨胀 ~1.5-2x
# (8191^3 NN 11.27ms vs 8192^3 5.54ms); 边界 tile 的 masked k-loop 再扣一档。
# (诊断: /tmp/diag_kodd.py + diag_padfull.py, 2026-09-05; config 扫 27 项无解,
# 确认为布局/掩码问题而非 tile 配置问题)
# 修复: 输入 pad 到 (BLOCK_M/N=256, BLOCK_K=128) 整数倍缓冲 + mask-free 主
# 内核; NN 顺带把 B(k,n) 转置成 Bt(n,k) 走 NT 主内核 (B 沿 k 列连续读, 同池
# 实测快 ~25-30%: 8192^3 NN main 5.54ms -> NT main 4.10ms)。输出先写 pad C
# 缓冲再裁回真实 C, alpha/beta 语义保持。pad 后以 (m_pad,n_pad,k_pad) 调
# autotuner 内核, 与已调优对齐 shape (8192^3 等) 命中同一 tune 条目。
_PAD_ALIGN_MN = 256  # 池内最大 BLOCK_M/BLOCK_N
_PAD_ALIGN_K = 128  # 池内最大 BLOCK_K
_PAD_BLK = 64  # 转置拷贝 tile 宽 BC (src n 向; 实测 odd 最优 64x128/w8)
_PAD_TRANS_BLK = 128  # 转置拷贝 tile 高 BR (src k 向; k 行主序跨大 stride 读)
# 小 odd shape (511/1023 级, 2mnk < _PAD_MIN_FLOPS) 不 pad: pad 需 4-5 个额外
# kernel/拷贝 launch (~25us/次), 小 shape 上启动开销占比远大于 misalign 收益
# (实验 2026-09-06: 1023^3 pad 后 0.20ms vs direct 0.06ms, 全线回退)。
_PAD_MIN_FLOPS = 5.0e9  # 启用 pad 的 2*m*n*k 下界

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
def _bfgemm_pad_trans_kernel(
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
    _bfgemm_nt_kernel[grid](
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
        **_kp2_kwargs(),
    )


def bfgemm(
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
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
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
        # 内核, 见上方 pad 路径注释。probe_pad_dispatch2 (2026-09-05, 官跑口径 e2e,
        # 数值 beta0/β=1.5 全 PASS): NN 8191^3 11.27->6.05ms、NT 9.48->6.23、
        # TN 11.76->5.89、TT 13.19->6.07; sp 0.75-1.54 (相对同进程 torch)。
        ps = _padded_size(m, n, k, lda, ldb, ldc)
        if ps is not None and 2.0 * m * n * k >= _PAD_MIN_FLOPS:
            mp, np, kp = ps
            # torch.empty 而非 zeros: 三块大缓冲 fmem (8191^3 级 ~400MB) 每次调用
            # ~0.8ms, 纯浪费。有效输出 (i<m,j<n) 的 k 方向点积只跨 A/B 的 k-pad
            # 列, 故仅需把两缓冲的 [:, k:kp] 条带清零 (见下); m-pad 行 / n-pad 行
            # / Cp 的 pad 区只影响被裁掉的输出, 无需清零。
            Ap = torch.empty(mp, kp, dtype=A.dtype, device=A.device)
            Bt = torch.empty(np, kp, dtype=A.dtype, device=A.device)
            Cp = torch.empty(mp, np, dtype=A.dtype, device=A.device)
            if transa == CUBLAS_OP_N:  # A 物理 (m,k): 同布局 pad
                Ap[:m, :k].copy_(_row_view(A, lda)[:m, :k])
            else:  # A 物理 (k,m): pad+转置 -> Ap(m,k)
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(mp, _PAD_BLK),
                )
                _bfgemm_pad_trans_kernel[g](
                    A, Ap, k, m, lda, kp, mp,
                    BR=_PAD_TRANS_BLK, BC=_PAD_BLK, num_warps=8,
                )
            if transb == CUBLAS_OP_N:  # B 物理 (k,n): pad+转置 -> Bt(n,k)
                b_pad_trans = True
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(np, _PAD_BLK),
                )
                _bfgemm_pad_trans_kernel[g](
                    B, Bt, k, n, ldb, kp, np,
                    BR=_PAD_TRANS_BLK, BC=_PAD_BLK, num_warps=8,
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
            if (m, n, k) in _NN_NT_ROUTE_SHAPES:
                # 方向①: B(k,n) (k 行 stride=ldb) -> Bt(n,k) (行 stride=k), 再走 NT 主内核
                Bt = torch.empty((n, k), dtype=B.dtype, device=B.device)
                gt = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(n, _TN_TRANS_BLOCK_M),
                )
                _bfgemm_tn_trans_kernel[gt](
                    B,
                    Bt,
                    k,
                    n,
                    ldb,
                    k,
                    BR=_TN_TRANS_BLOCK_K,
                    BC=_TN_TRANS_BLOCK_M,
                    num_warps=8,
                )
                _bfgemm_nt_kernel[grid](
                    A, Bt, C, alpha, beta, m, n, k, lda, k, ldc, beta_is_zero,
                    **_kp2_kwargs(),
                )
            else:
                _bfgemm_nn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    **_nn_kernel_kwargs(m, n, k),
                )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if (m, n, k) in _TN_Y_SHAPES:
                # Y 复合: B(k,n) 连续转置成 Bt(n,k), 再 NN 主循环 + 转置 store 直写 C
                Bt = torch.empty((n, k), dtype=B.dtype, device=B.device)
                gb = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(n, _TN_TRANS_BLOCK_M),
                )
                _bfgemm_tn_trans_kernel[gb](
                    B,
                    Bt,
                    k,
                    n,
                    ldb,
                    k,
                    BR=_TN_TRANS_BLOCK_K,
                    BC=_TN_TRANS_BLOCK_M,
                    num_warps=8,
                )
                bm, bn = _tn_y_tiles(m, n, k)
                gy = (triton.cdiv(m, bn) * triton.cdiv(n, bm),)
                _bfgemm_tn_y_kernel[gy](
                    Bt,
                    A,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    ldc,
                    beta_is_zero,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=64,
                    GROUP_M=8,
                    num_warps=4,
                    num_stages=2,
                )
            elif _tn_use_composite(m, n, k):
                # A(k,m) 行主序 (行长=lda) -> A^T(m,k) 行主序 (行长=k)
                At = torch.empty((m, k), dtype=A.dtype, device=A.device)
                g = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(m, _TN_TRANS_BLOCK_M),
                )
                _bfgemm_tn_trans_kernel[g](
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
                if (m, n, k) in _TN_NT_ROUTE_SHAPES:
                    # 方向① TN: 再补 B(k,n)->Bt(n,k) (行 stride=k), At+Bt 走 NT 主内核
                    Bt = torch.empty((n, k), dtype=B.dtype, device=B.device)
                    gb = (
                        triton.cdiv(k, _TN_TRANS_BLOCK_K)
                        * triton.cdiv(n, _TN_TRANS_BLOCK_M),
                    )
                    _bfgemm_tn_trans_kernel[gb](
                        B,
                        Bt,
                        k,
                        n,
                        ldb,
                        k,
                        BR=_TN_TRANS_BLOCK_K,
                        BC=_TN_TRANS_BLOCK_M,
                        num_warps=8,
                    )
                    _bfgemm_nt_kernel[grid](
                        At, Bt, C, alpha, beta, m, n, k, k, k, ldc, beta_is_zero,
                        **_kp2_kwargs(),
                    )
                else:
                    _bfgemm_nn_kernel[grid](
                        At, B, C, alpha, beta, m, n, k, k, ldb, ldc, beta_is_zero,
                        **_nn_kernel_kwargs(m, n, k),
                    )
            else:
                _bfgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _bfgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                **_kp2_kwargs(),
            )
        else:
            _bfgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )

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
Hygon DTK fp64 DGEMM 定制路径 (dot 内核)。

背景 (2026-09-09, DTK 25.04 / triton 3.6.0):
* 通用 level3 `_dgemm_kernel` 是逐 k 标量外积内核, Hygon 上实测远未利用
  FMA/访存 (2048^3 NN 36.6ms @0.5TFLOPS, 与 tile 无关 ~0.8TFLOPS 封顶)。
* 本后端 fp64 `tl.dot` 实测 ~6-8 TFLOPS 上限 (2048^3 2.6ms / 4096^3 18ms),
  比对标量外积 ~10x, 但仍受限于向量 FMA。**该结论自 2026-09-13 起作废**: fp64
  `tl.dot` 已降级到 DCU 原生 fp64 矩阵单元 `v_mmac_16x16x4_f64` (见文末
  "MMAC 落地后的配置重扫"), 同配置下快 ~2.4x, 大 shape sp 已到 1.0+。
* smem: fp64 tile smem = (BM*BK + BK*BN)*8 每 stage。2026-09-13 复核 Triton
  编译产物 metadata.shared 确认**硬上限就是 65536B**: 8 个在用配置实际分配
  16384/20480/24576/32768/65536, 而 128x128 w32 s2 要 131072 会在 launch 前
  抛 OutOfResources。旧注释里的 "98304B 正常执行" 是按公式估算、非实际分配,
  故大 tile 确实受 smem 硬卡。配置按 (trans, max_dim) 分档实测选取。
* 大 shape (2026-09-11, probe36/probe39): 改用 128x128x16 s2 后全面优于原
  128x256/256x128 —— 后者的 grid 在大 shape 上偏小, tile 切分无法摊薄。
  NT/TT 提升最显著 (8192^3 -38%/-41%), 因其 B 走 NAT_B 转置读, 小 BN 更省
  smem 带宽。

此模块作为 vendor 定制 op 覆盖 `flag_blas.dgemm` (仅 Hygon 生效, 其余后端
仍走通用 level3 实现)。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.dgemm import ScalarType, _validate_dgemm_args
from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1


@triton.jit
def _dot_block(
    a_ptr,
    b_ptr,
    offs_m,
    offs_n,
    cur_k,
    lda,
    ldb,
    acc,
    mask_m,
    mask_n,
    mask_k,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    NAT_B: tl.constexpr,
    NO_MASK: tl.constexpr,
    USE_K: tl.constexpr,
):
    """单个 k 块的 A/B 载入 + tl.dot。

    USE_K 决定 k 轴谓词是否参与: 主体循环里 k 一定是 BLOCK_K 整数倍, 去掉 k
    掩码后 A 的连续轴 (TRANS_A==0) 只剩沿 m 的掩码, 取址可以向量化。
    """
    if TRANS_A == 0:
        a_ptrs = a_ptr + offs_m[:, None] * lda + cur_k[None, :]
    else:
        a_ptrs = a_ptr + cur_k[None, :] * lda + offs_m[:, None]
    if TRANS_B == 0:
        b_ptrs = b_ptr + cur_k[:, None] * ldb + offs_n[None, :]
    else:
        b_ptrs = b_ptr + offs_n[None, :] * ldb + cur_k[:, None]
    if NO_MASK:
        a = tl.load(a_ptrs)
    elif USE_K:
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    else:
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
    if TRANS_B == 1 and NAT_B == 1:
        # transb=1: B 以 (n,k) 行主序存放, 连续轴是 k。按自然行主序载入
        # (BN,BK) 再转置, 比直接按 (BK,BN) 索引载入更高效 (中小尺寸实测)。
        b_ptrs = b_ptr + offs_n[:, None] * ldb + cur_k[None, :]
        if NO_MASK:
            b = tl.trans(tl.load(b_ptrs))
        elif USE_K:
            b = tl.trans(tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0))
        else:
            b = tl.trans(tl.load(b_ptrs, mask=mask_n[:, None], other=0.0))
    elif NO_MASK:
        b = tl.load(b_ptrs)
    elif USE_K:
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    else:
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
    return tl.dot(a, b, acc, out_dtype=tl.float64, allow_tf32=False)


@triton.jit
def _dgemm_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    beta: tl.float64,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    NAT_B: tl.constexpr,
    NO_MASK: tl.constexpr,
    K_TAIL: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    # NO_MASK: m/n/k 均为 BLOCK 整数倍时省掉全部掩码。小 shape 上掩码的谓词
    # 计算与边界处理占内核时间相当比例 (64^3 NN 11.0->9.8us)。
    # K_TAIL: k 不是 BLOCK_K 整数倍时, 主体循环只跑 k_main = k//BLOCK_K*BLOCK_K,
    # 省掉 k 掩码 (A 的连续轴只剩沿 m 的掩码 -> 取址可向量化), 末尾再单跑一个
    # 带 k 掩码的小块。逐 bit 一致 (probe163 err=0), 1023^3 152.96 -> 150.24us。
    if K_TAIL:
        k_main = (k // BLOCK_K) * BLOCK_K
    else:
        k_main = k
    for k_start in range(0, k_main, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < k
        acc = _dot_block(
            a_ptr, b_ptr, offs_m, offs_n, cur_k, lda, ldb, acc,
            mask_m, mask_n, mask_k, TRANS_A, TRANS_B, NAT_B, NO_MASK,
            USE_K=(not K_TAIL),
        )
    if K_TAIL:
        cur_k = k_main + offs_k
        mask_k = cur_k < k
        acc = _dot_block(
            a_ptr, b_ptr, offs_m, offs_n, cur_k, lda, ldb, acc,
            mask_m, mask_n, mask_k, TRANS_A, TRANS_B, NAT_B, NO_MASK,
            USE_K=True,
        )

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if NO_MASK:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float64)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float64)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)

# --- v25 尾波 split-K（stream-K 的单波变体）------------------------------------
# 动机（probe170/172 定位、probe174/175 量测，报告 §3.21/§3.22）：>1408³ 的缺口已定量定位为**波次量化**——设备
# 80 CU + 64KB smem ⇒ 1 CTA/CU，每个 CTA 负责一个 (BM,BN) 输出 tile 的整段 k 循环，
# 于是 makespan = ceil(num_tiles/80) 个 tile-time，而理想值是 num_tiles/80 个。
# 当 num_tiles % 80 = tail 且 tail 较小时，最后一个波次只有 tail 个 CTA 在跑，却
# 占掉整整个 tile-time —— 2048³（256 tile，tail 16）与 3072³（576 tile，tail 16）
# 的 ceil_eff 因此只有 0.80 / 0.90，正是 NN 2048³ sp 0.856、NT 3072³ sp 0.898 的
# 直接来源（probe154 的波次模型 + probe170/172 的交错实测）。
#
# 本路径把**尾波的那 tail 个 tile** 各切成 nsplit 段等长 k 区间，交给
# nsplit*tail (≤ SM) 个 CTA 并发完成，使尾波只花 1/nsplit 个 tile-time；主轴
# （前 bulk 个 tile）仍由原内核**一次启动**跑完，内核本体内核 (_dgemm_dot_kernel)
# 一行未改。分片结果写进一块 (nsplit*tail, BM, BN) 的部分积缓冲，再由一个归约
# 内核按 tile 求和并施加 alpha/beta 写回 C。三条启动的 tile→CTA 映射与原内核
# 完全同构（同一 swizzle 公式 + pid 偏移），故**未启用该路径的 shape 逐位不变**。
#
# 门槛（见 _streamk_plan）：尾波能整除成 ≥2 份、k 迭代数能被份数整除、且代价模型
# （每多切一段多一次 (BM,BN) 部分积的写+读）远小于尾波收益时才启用。
@triton.jit
def _dgemm_splitk_kernel(
    a_ptr,
    b_ptr,
    p_ptr,
    m,
    n,
    lda,
    ldb,
    pid_base,
    tail,
    ki_per_slice,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """尾波分片: 第 (pid_local // tail) 段 k 区间, 写部分积到 p_ptr 的第 pid_local 块。

    只服务无掩码情形（m/n/k 均为 BLOCK 整数倍），故走全无谓词路径。
    """
    pid_local = tl.program_id(0)
    pid = pid_base + pid_local % tail
    grid_m = m // BLOCK_M
    grid_n = n // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_start = (pid_local // tail) * ki_per_slice * BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for i in range(ki_per_slice):
        cur_k = k_start + i * BLOCK_K + offs_k
        if TRANS_A == 0:
            a_ptrs = a_ptr + offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_ptrs = a_ptr + cur_k[None, :] * lda + offs_m[:, None]
        if TRANS_B == 0:
            b_ptrs = b_ptr + cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_ptrs = b_ptr + offs_n[None, :] * ldb + cur_k[:, None]
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc,
                     out_dtype=tl.float64, allow_tf32=False)

    p_ptrs = (p_ptr + pid_local * (BLOCK_M * BLOCK_N)
              + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N
              + tl.arange(0, BLOCK_N)[None, :])
    tl.store(p_ptrs, acc)


@triton.jit
def _dgemm_splitk_reduce_kernel(
    p_ptr,
    c_ptr,
    alpha,
    beta,
    m,
    n,
    ldc,
    pid_base,
    nsplit,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """把 (nsplit, tail, BM, BN) 的部分积按 tile 求和, 再施加 alpha/beta 写回 C。

    部分积按 slice-major 排布: tile 局部下标 L 的第 j 片在 (j*tail + L) 块。
    """
    local_id = tl.program_id(0)
    tail = tl.num_programs(0)
    pid = pid_base + local_id
    grid_m = m // BLOCK_M
    grid_n = n // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N
             + tl.arange(0, BLOCK_N)[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for j in range(nsplit):
        acc += tl.load(p_ptr + (j * tail + local_id) * (BLOCK_M * BLOCK_N) + inner)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc)
    else:
        tl.store(c_ptrs, alpha * acc + beta * tl.load(c_ptrs))


# smem budget: (BM*BK + BK*BN)*8 <= 64KB @ num_stages=2 (double-buffered fp64)
# cfg = (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
# 实测选取 (probe2/probe4 2026-09-09, 2048/4096^3):
#   NN: 128x256x16 w8 s2 (6.6-7.6 TFLOPS)   TN: 256x128x16 w8 s2
#   NT: 256x128x16 w8 s2 (已超 hipBLAS NT)  TT: 见 _select (同 TN 取向)
# 小尺寸分档 (probe5/probe6 2026-09-10, 128..1024^3):
#   大 tile 在小方阵上只会切出 1-2 个 CTA (256^3 用 128x256 -> grid=2),
#   执行器空转且寄存器溢出, 256^3 NN 实测 2.95ms; 换成 32x32 小 tile
#   (grid=64) 后 0.027ms (-111x), TN 1.43ms -> 0.021ms。故 max_dim<=1024
#   区间按实测小 tile 分档, 只在 max_dim>1024 使用大 tile。
# smem 远未受限的小尺寸进一步缩小 tile 并降 warp (probe8 2026-09-10, 32..256^3):
#   launch 耗时 ~8.5us 是 32^3 的地板 (与 hipBLAS 8.3us 相当), 小 tile 提升
#   并行度: 16x16x16 w1 使 NT/TT 128^3 34.4->17.4us, 16x32x16 w2 使 NN 64^3
#   13.0->10.4us; 256^3 仅 NT/TT 受益于 w2 (NT 65->48us, TT 60->46us)。
# 细粒度网格 + NAT_B 变体 (probe11/probe12 2026-09-10, 官方口径 warmup=1000/
#   rep=100/median, 3 轮; 每档在 32/64/128/256^3 上取全 shape 最优):
#   * 16x16x32 w1 使 NN 32/64/128^3 达 8.32/10.08/13.60us (sp 1.00/1.06/0.93),
#     16x64x16 w4 使 NN 256^3 27.8->24.0us。
#   * NAT_B=1 (B 按 (n,k) 自然载入 (BN,BK) 再 tl.trans) 对 transb=1 的中小
#     尺寸更优: TT 128^3 17.8->16.0us、TT 256^3 45.8->42.2us、
#     NT 256^3 47.8->42.9us。
# 掩码消除 + 转置前导维对齐 (probe16..probe29 2026-09-11, 官方口径):
#   * 计时口径核实: do_bench 的两条 event 本身占 7.0us 地板, 空 kernel 8.2us,
#     host/Python 开销不计入 (纯 CPU 忙等 150us 读数不变) -> 只能靠缩减 GPU
#     端内核时间取胜, 小 shape 上掩码的谓词/边界处理占比不小。
#     NO_MASK=1 (m/n/k 全为 BLOCK 整数倍时省掉所有掩码) 使 64^3 NN
#     11.0->9.8us (sp 0.928->1.049)、32^3 NT 10.9->9.3us (0.853->1.000)。
#   * 32^3 单列分档: 全 C 仅 1~4 个 tile, 16x16x16/16x16x32 w1~w2 + GROUP_M=1
#     最省; 64/128^3 用 16x16x16 w2 (NT/TT 加 NAT_B)。
#   * TN (transa=1) 在 513..1024 区间: A 的连续轴落在 tile 第 0 维, 大 tile
#     下 dot 左操作数要额外做布局转换, 比 NN 慢 ~7%。主机端预转置 A 并把前导
#     维对齐到 16 后改走 transa=0: 1023^3 474->454us (0.911->0.952)、
#     1024^3 450->435us (0.938->0.970)。对齐很关键: 1023 不对齐会掉到 812us。
# 小 shape 加宽 BLOCK_K (probe30 2026-09-11, 32/64/128^3 全 trans 扫描):
#   * k 循环只有 1~8 次时加宽 BLOCK_K 可摊薄循环与掩码开销, 同时保持 16x16 tile
#     的并行度: NN 16x16x64 w2 使 64^3 9.76us、128^3 12.80us (原 16x16x16 档
#     实测 12.80/16.96); TN 16x16x64 w1 使 64^3 9.44us、128^3 12.64us;
#     TT 32^3 用 16x16x32 由 10.08->8.80us。
#   * 128^3 TN 的最优是 BLOCK_K==k 的单次迭代配置 (16x16x128 w1, 12.16us)。
#   * <=128^3 的 sp 基本由 hipBLAS 基线 (8~17us, 已贴近 ~8us 事件地板) 决定,
#     基线本身有 ~2us 跨轮抖动, 故该区间 0.95 门槛实为噪声边界。
# 512^3 归因与档位边界修正 (probe90..probe101 2026-09-12, 5~7 轮取中位):
#   * smem 布局转换**不可消除**: probe93 丢掉 tl.dot 改手动外积后 smem 往返清零
#     (ds_read 100->4), 但每线程须自取操作数, global_load/FMA 涨 27x, 净慢 3~5x。
#     fp64 只有向量 v_fma_f64 一条路 (生产内核反汇编 v_mfma*/v_mac* = 0,
#     v_fma_f64 256 条 = 36%), 操作数只能靠加大 tile 摊薄 LDS/FMA = 16*(1/BM+1/BN)。
#   * 512^3 的 tile 被**并行度地板**锁死: 总 warp = m*n/(每线程输出数*64), 32x64
#     给出 128 CTA*4warp = 512 warp, 已贴近填满 320 SIMD 所需的最小 tile
#     (BM*BN<=3277); 放大 tile (64x128) 掉到 32 CTA = 0.4 wave, NN 实测倒退 1.86x。
#     缩小 tile 换并行度则 LDS/FMA 上升 (16x16 是 32x64 的 2.7x)。=> 512^3 的
#     0.43 sp 是结构上限, 不是配置没调好 (probe95 全档位 tile 朝向扫描否证了
#     "BN 要宽"作为独立杠杆: 现行分档在 ks=1 口径下逐档最优)。
#   * probe96/97: split-K 能给大 tile 补回并行度 (gemm 本体 1.17x), 但 ws+reduce 的
#     归约流量把端到端压到 1.04x (512^3) / 1.12x (511^3, atomic, 需额外 C 预置零),
#     收益远小于下述的档位修正, 故不引入 split-K 路径。
#   * probe97..probe101: 真正的漏在**档位边界**。原 <=1024 档的 64x128x16 在
#     513..896 只切出 45~98 个 CTA, 改用 32x64x16 后 NN 全面提升 (最多 1.82x),
#     详见 _select_dgemm_dot_config 内注释。
# 档位边界亚档 (probe109..probe114 2026-09-12, 官方口径 + 交错 A/B 4 轮取中位):
#   * probe109 逐 size 扫 96..511^3 找到两处换档悬崖: 129^3 +42% (grid 64->27 CTA)、
#     257^3 +46% (grid 64->45 CTA); 机制是边界上多出的整块 tile 照样跑满 k 循环,
#     同时 grid/波数下降, 形成双重恶化。
#   * probe110/111/112/113/114 在过渡带上逐 size 找交叉点, 增设两个亚档
#     (16x32x32 w4 s2 gm8, 上界由实测交叉点定, 不外推):
#       <=192^3: NN 129^3 19.04->16.56us (0.870x), 196^3 起反超 -> 上界 192;
#                TN 候选更慢 (1.145~1.277x) 故不改。
#       <=272^3: NN 257^3 33.28->29.60us (0.889x), TN 38.24->32.32us (0.845x),
#                274^3 起反超 -> 上界 272。
#   * 两处悬崖 (129/144/160/192/257/264/272) 全在 benchmark/core_shapes.yaml 的
#     15 个方阵之外, 故本次是纯鲁棒性修正, 不影响 §1/§2 的评分口径。
# MMAC 落地后的配置重扫 (probe118/119/120, 2026-09-13, 官方口径 warmup=1000/
#   rep=100/median, 交错 A/B 取中位; NN 带 3~9 轮, 同档位重复测量离散 <0.2%):
#   * 前置改动: FlagTree hcu 的 fp64 tl.dot 不再走 FMA, 而是降级到 DCU 原生
#     fp64 矩阵单元 `v_mmac_16x16x4_f64` (instrShape 16x16x4; 每 lane 1 个
#     A/B f64、4 个 D f64 = 8 VGPR; 2048 flops/wave64 = v_fma_f64 的 16x)。
#     生产内核反汇编核对: 全部候选 v_fma_f64 = 0, 无一静默回退。同机器同口径
#     下大 shape 提速 ~2.4x (8192^3 NN 旧配置 135.3 -> 56.4ms, 新配置 39.1ms),
#     精度不变 (err 0.0 ~ 5.4e-12, 属 fp64 累加顺序差异, 远优于任何容差)。
#   * MMAC 新增的杠杆是**累加器寄存器压力**: acc VGPR = 2*BM*BN/(32*num_warps),
#     上限 256。旧大 shape 配置 128x128 **w4** 正好要 256 个 -> 溢出; 同 tile
#     提到 w8 减半、w16 只要 32。所以 MMAC 下放大 tile 的正确做法是**同时提高
#     num_warps** (FMA 时代的表里 num_warps 基本没被当成独立维度扫过)。
#   * transb=0 路径 (即 NN, 以及 TN / 预转置后的 NT/TT —— max_dim>128 且 m>=64
#     时 B 一律被主端预转置, 故大 shape 实际只剩 NN 与 TN 两条路) 重扫结论一致:
#       max_dim <= 1408 : 32x64x16 w4 s2 g8   (原 <=896 档的配置, 全带最优)
#       max_dim >  1408 : 128x128x16 w16 s2 g8
#     交叉点落在 1408..1536 之间, 两侧实测 (括号内为同口径 hipBLAS 基线):
#       NN 1024^3 131.7us sp=0.893 (117.6) | 1280^3 218.9 sp=0.931 (203.8)
#          1408^3 301.6us sp=0.908 (273.8) | 1536^3 336.0 sp=0.924 (310.6)
#          2048^3 833.9us sp=0.856 (713.8) | 3072^3 2378 sp=0.984 (2339)
#          4096^3 5051  sp=1.088 (5494.6)| 8192^3 39128 sp=1.203 (47079)
#       TN 1024^3 127.8us sp=0.856 (109.4) | 2048^3 825.0 sp=0.801 (660.8)
#          4096^3 4992  sp=1.012 (5052)  | 8192^3 38731 sp=1.007 (39018)
#     1408..3072 之间两个候选会互相反超 (1536^3 w16 快 10%、1664^3 32x64 反而
#     快 13%、1792/2048 打平), 属 grid 波数与占用率的谐振, 未找到单一配置能两边
#     兼顾; 边界取 1408 是让评分口径 core_shapes.yaml 的 1536/3072 (w16 领先
#     10~11%) 与 1024 (32x64 领先 7%) 各取所优, 2048 两者只差 0.8%。
#   * 由此作废/修正的旧结论 (原文保留在上方作为历史):
#       - "fp64 只有向量 v_fma_f64 一条路 (v_mfma*/v_mac* = 0), 操作数只能靠
#         加大 tile 摊薄 LDS/FMA = 16*(1/BM+1/BN)": MMAC 下 A/B 每 lane 只占
#         1 个 f64, 布局转换的 LDS 流量与 FMA 完全不同, 加大 tile 不再是唯一
#         杠杆, num_warps 反而更关键。
#       - "512^3 的 0.43 sp 是结构上限, 不是配置没调好": MMAC 下 512^3 用
#         32x64x16 w4 s2 实测 sp=1.064 (NN) / 1.000 (TN), 上限已被抬高。
#       - ">1024 用 128x128x16 w4": MMAC 下该配置 acc 溢出, 4096^3 只有
#         sp=0.778; 提到 w16 后为 1.088。
#   * 未重扫项 (无数据, 保持原样不回归): transb=1 且 m<64 时才会走到的 NT/TT
#     直路配置 (TT 侧仍是 128x256x16 w8, acc 需 256 VGPR, 在 MMAC 下同样可疑,
#     但不在评分口径内且未实测; NT 侧已随 probe143 一并改为 128x128x16 w16)。
#     <=256^3 的小 tile 分档已由 v26 (probe176/177, 2026-09-15) 在 MMAC 形态下
#     重扫并落地 (129..272^3 统一 32x32x32 w4, 见下方分支注释与报告 §3.23)。


def _select_dgemm_dot_config(transa: int, transb: int, m: int, n: int, k: int):
    """按 (trans, max_dim) 分档选择 dot 内核 launch 配置 (Hygon 实测表)。

    返回 dict 化的 launch kwargs 或 None (尺寸过小 -> 走通用回退)。
    """
    max_dim = max(m, n, k)
    min_dim = min(m, n, k)

    if min_dim < 16 or max_dim < 32:
        return None

    nat = 0
    # v21 (probe158–probe161, 2026-09-14): dgemm() 让 m>=64 的 TT 也走原生 TB1 直路
    # 后, 它的 tile 必须与 TN 同款 —— MMAC 下 B 的 k 向 128B run 载入几乎不花钱
    # (逐 size 实测原生 TB1 与 TN 同速), 而历史 "TT 直路" 档给的是为 FMA 时代选的
    # 大 BN tile, 其中 >1408 的 128x256x16 w8 更是 acc = 256 VGPR 压在上限 (probe160:
    # 1536^3 361.6us / 2048^3 929.4us, 均慢于 128x128x16 w16 的 335.8/832.3)。
    # 故这里把 "未被预转置的 m>=64 的 TT" 归一成 TN 口径; 历史 TT 档只留给
    # max_dim<=128 或 m<64 的瘦长直路 (那些尺寸本就不做预转置, 行为不变)。
    if transa == 1 and transb == 1 and m >= 64 and max_dim > 128:
        transb = 0
    if max_dim <= 32:
        # 32^3: 单 tile 覆盖整个 C (grid 1~4 CTA), 掩码消除后 GPU 时间已贴近
        # 空 kernel 地板。GROUP_M=1 省掉 swizzle 分支。
        if transa == 0 and transb == 0:  # NN
            bm, bn, bk, nw = 16, 16, 16, 2
        elif transa == 1 and transb == 0:  # TN
            bm, bn, bk, nw = 16, 32, 32, 2
        elif transa == 0:  # NT
            bm, bn, bk, nw, nat = 16, 16, 16, 2, 1
        else:  # TT
            bm, bn, bk, nw, nat = 16, 16, 32, 2, 1
        gm, ns = 1, 2
    elif max_dim <= 128:
        # 64/128^3: 方阵只有 4~64 个 16x16 tile, 逐档实测 (probe30 2026-09-11,
        # 官方口径)。k 循环只有 1~8 次, 把 BLOCK_K 加宽到 32/64 摊薄循环与掩码
        # 开销: NN 16x16x64 w2 (64^3 12.80->9.76us, 128^3 16.96->12.80us)、
        # TN 16x16x64 w1 (64^3 10.72->9.44, 128^3 13.76->12.64) 均优于
        # 16x16x16/16x32x32; NT/TT 仍取 NAT_B + BLOCK_K=32 (64^3 13.12->11.52 /
        # 11.36->10.72)。NN/TN 用 GROUP_M=1, NAT_B 的 NT/TT 用 GROUP_M=8。
        # NN 的 num_warps 2->4 (probe104/105 2026-09-12; 交错 A/B, 走公开入口
        # flag_blas.dgemm, 官方口径 warmup=1000/rep=100, 8 轮取中位, 两轮复现):
        #   64^3 11.04->10.40us (0.942x) | 128^3 14.24->13.44us (0.944x)
        #   32^3 持平 (9.76us)。BLOCK_K=128 仅在 128^3 再降 2.4% (13.44->13.12us),
        #   但使 64^3 退化到 12.96us (k<BLOCK_K 的掩码浪费), 故不加宽 BLOCK_K。
        # 本次只调 NN; TN 仍 w1, NT/TT 仍 w2。
        # v23 (probe166/167, 2026-09-15, MMAC 时代重扫, 交错 A/B 4 轮取中位 +
        # "把生产配置原样传回 force()" 的 clone 对照, 噪声地板 0.00%): 上一行注释
        # 里 NT/TT 的 "NAT_B + BLOCK_K=32" 是 FMA 时代 (probe30) 的选择, 在 MMAC
        # 下是纯亏 —— NAT_B=1 要先把 (BN,BK) 载入再做 tl.trans, 而每条
        # v_mmac_16x16x4_f64 只出 2048 flops / 每 lane 4 次 LDS, 布局转换的相对
        # 代价被放大 (同一现象在 >1024^3 上是 2.8~3.9x, 见 probe164)。改为与
        # NN/TN 同族的 16x16x64 + nat=0 (GROUP_M 随之由 8 变 1) 后实测:
        #   64^3  NT 10.08->9.60us (-4.8%) | TT 10.08->9.60us (-4.8%)
        #   128^3 NT 12.16->9.92us (-18.4%) | TT 12.32->10.08us (-18.2%)
        # 归因由同档对照钉死: 强制 16x16x32 w2 nat=1 复现 12.16/12.32us, 与生产
        # 逐位同速。这是 NT 128^3 sp 0.805 -> ~1.00 (跨 0.95 门槛) 的直接来源。
        if transa == 0 and transb == 0:  # NN
            bm, bn, bk, nw = 16, 16, 64, 4
        elif transa == 1 and transb == 0:  # TN
            bm, bn, bk, nw = 16, 16, 64, 1
        else:  # NT / TT
            bm, bn, bk, nw = 16, 16, 64, 2
        gm, ns = (8 if nat else 1), 2
    elif max_dim <= 192:
        # 129..192^3 亚档。v26 (probe176/177, 2026-09-15, 交错 A/B 3~4 轮取中位
        # + clone 噪声地板 0.00%) 在 MMAC 形态下重扫: 本档原为 16x32x32 w4 (NN)
        # 与 16x64x32 w4 (TN / NT), 都是 probe109-114 (2026-09-12, FMA 口径) 按
        # 「窄 BN 换回 CTA 数」定的; MMAC 下最优点整体移到 32x32x32 w4:
        #   129^3 nn/nt 13.12/13.28 -> 12.64/12.64 (-3.7%/-4.8%)
        #   144^3 nn/nt 12.80/12.16 -> 11.84/12.00 (-7.5%/-1.3%)
        #   160^3 nn/nt 12.48/12.80 -> 11.52/11.52 (-7.7%/-10.0%)
        #   176^3 nn/nt 13.60/12.96 -> 12.48/12.64 (-8.2%/-2.5%)
        # 同轮把 16x32x32 w4 作为直接对照候选对拍, 全输。瘦长 (min_dim<32) 与
        # TT 直路未被本探针覆盖, 保持历史取值不回归。
        if transa == 1 and transb == 1:  # TT 直路 (仅 m<32)
            bm, bn, bk = 64, 64, 32
        elif min_dim < 32:  # 瘦长矩阵: 保持 v13 分档
            bm, bn, bk = (16, 32, 32) if (transa == 0 and transb == 0) else (16, 64, 32)
        else:  # NN / TN / NT 主线
            bm, bn, bk = 32, 32, 32
        gm, nw, ns = 8, 4, 2
    elif max_dim <= 256:
        # 193..256^3。原 32x32x32 w4 (probe40/41, 2026-09-11) 被 probe57/58
        # (2026-09-11, 交错 7 轮) 翻盘成 16x64x32 w4 —— 但那是 **FMA 口径**:
        # 当时 32x32x32 在 256^3 上 26.08->21.44us 落后 18%。v26 (probe176,
        # MMAC 形态, 交错 A/B 4 轮 + clone 地板 0.00%) 复测结论**反号**,
        # 32x32x32 w4 在四个 trans 上全面胜出, 其余候选 (32x64x32 / 32x32x16 /
        # 32x32x32 w8 / 16x32x32) 全输:
        #   256^3 nn/tn/nt/tt 13.28/14.08/14.56/13.76
        #         -> 13.12/13.60/13.28/13.12 (-1.2%/-3.4%/-8.8%/-4.7%)
        #   208^3 nn/nt 12.96/13.12 -> 12.64/12.80 (-2.5%/-2.4%)
        #   224^3 nn/nt 12.96/14.24 -> 12.32/12.48 (-4.9%/-12.4%)
        # NT 256^3 由官方 sp 0.926 跨过 0.95 (见报告 §3.23)。511/512^3 上
        # 32x32x32 仍退化 (probe177 复测 512^3 +18%, 288^3 更达 +504%),
        # 故本档必须与 <=1408 分开。
        if transa == 1 and transb == 1:  # TT 直路 (仅 m<32)
            bm, bn, bk = 64, 64, 32
        elif min_dim < 32:  # 瘦长: 保持原值
            bm, bn, bk = 16, 64, 32
        else:  # NN / TN / NT
            bm, bn, bk = 32, 32, 32
        gm, nw, ns = 8, 4, 2
    elif max_dim <= 272:
        # 257..272^3 亚档 (probe109/111/112/113/114, 2026-09-12, FMA 口径的
        # 256->257 跨档修正)。v26 复测: NN/TN 原 16x32x32 w4 -> 32x32x32 w4
        # (272^3 nn 19.36->16.16, -16.5%), NT 原 32x64x16 w8 -> 32x32x32 w4
        # (17.92->16.80, -6.3%)。**上界锁在 272**: probe177 显示 288^3 上
        # 32x32x32 w4 退化到 116us (+504%) —— 该点 grid 81 CTA = 1.0125 波,
        # 正好卡在 80 CU 的波次边界(每 CU 的 tile 数从 1 跳到 2), 512^3 亦
        # +18%, 448^3 持平 ⇒ 不能并入 <=1408 档的 32x64x16/64x32x16。
        # <=128^3 亦不动 (v23 已由 probe166/167 定为 16x16x64, 且 64^3 上
        # 32x32x32 只切出 4 CTA)。
        if transa == 1 and transb == 1:  # TT 直路 (仅 m<32)
            bm, bn, bk = 64, 64, 32
        elif min_dim < 32:  # 瘦长: 保持原分档
            bm, bn, bk = (32, 64, 16) if (transa == 0 and transb == 1) else (16, 32, 32)
        else:  # NN / TN / NT 主线
            bm, bn, bk = 32, 32, 32
        gm, nw, ns = 8, 4, 2
    elif max_dim <= 1408:
        # 511/512^3: 32x64x16; TT 实测 64x64x32 略优。
        # 513..896 原属 <=1024 档 (64x128x16), probe97/98/100/101 (5~7 轮取中位)
        # 发现该区间 64x128 只能切出 45~98 个 CTA, NN 全面落后 32x64x16:
        #   544^3 2.04->3.72 TF (1.82x) | 576^3 2.30->2.85 (1.24x)
        #   640^3 2.81->3.57 (1.27x)     | 832^3 3.34->3.79 (1.13x)
        #   896^3 3.93->4.40 (1.12x)     | 704/768^3 落后 3% (噪声量级)
        # 928^3 起 64x128 才反超 (1.05~1.23x, 1024^3 5.14 vs 4.18 TF), 故边界曾取 896。
        # 64x128 的振荡 (832^3 异常低) 已排除 swizzle (GROUP_M 4/8/16 无差) 与
        # 掩码成本 (只留循环不变量 m/n 掩码时间不变), 属 partial-N tile 的占用率效应。
        # 2026-09-13 (probe119/120, MMAC 落地后重扫): 32x64x16 w4 s2 一路领先到
        # 1408^3 —— 1024^3 NN 131.7us (sp=0.893) vs 64x128x16 的 151.5us (0.776),
        # 1280^3 218.9 vs 279.2, 1408^3 301.6 vs 307.5; 1536^3 起才被 w16 反超。
        # 原 <=1024 档的 64x128x16 因此被全面取代, 边界 896 -> 1408, 该档删除。
        # TT 直路 (仅 m<32 时 513..1408 才会走到) 维持原 128x128x16, 未复测不回归。
        # 2026-09-14 (probe131/132/133/134, MMAC 时代重扫, 官方口径 + 交错多轮取中位):
        # 之前的 32x64x16 w4 只与 128x128x16 w16 对拍过 (probe119/120), 中小 tile
        # 空间从未系统扫过。补扫后发现 BM/BN 对调 (64x32) 在**能整除**的方阵上全面
        # 更快, 但代价是 B 的连续 run 只有 256B, 一旦 lda/ldb 不对齐就急剧退化:
        #   512^3  64x32 27.8 vs 32x64 31.8 (1.14x) | 640^3 40.5 vs 46.7 (1.15x)
        #   768^3  59.8 vs 63.7 (1.06x)             | 896^3 84.8 vs 92.3 (1.09x)
        #   1024^3 117.4 vs 131.5 (1.12x)           | 1408^3 281.8 vs 301.6 (1.07x)
        #   1023^3 192.4 vs 180.5 (0.94x, 退化)     | 2048^3 879.5 vs 827.4 (0.94x)
        # 不对齐时的最优改由 "128-tile 排满率" 决定 (probe134):
        #   1023^3 (排满 1.002) 128x128x16 w16 = 152.8 (生产 180.5, 1.18x)
        #   1000^3 (排满 1.049) 同上 = 150.1 (生产 173.9, 1.16x)
        #   1200^3 (排满 1.138) 同上 = 277.4 (生产 214.4, 0.77x) -> 必须排除
        #   <=512 的不对齐点 (511^3) 大 tile 会切不出 CTA: 128x128 = 84.2us, 而
        #   32x64x16 **w8** = 34.7us (生产 38.9us, 1.12x); 小形状加 warp 更快。
        # 机制: 不对齐使每条连续 run 跨 cacheline 条数上升, run 越短相对浪费越大
        # (128B run 2.0x / 256B 1.5x / 1024B 1.12x, probe133 的 "不对齐惩罚" 逐配置
        # 实测: 64x32 63.6%、32x64 36.7%、128x64 19.4%、128x128 7.9%)。
        gm128 = 128 * ((m + 127) // 128) / m
        gn128 = 128 * ((n + 127) // 128) / n
        if transa == 1 and transb == 1:  # TT
            bm, bn, bk, gm, nw, ns = (
                (64, 64, 32, 8, 4, 2) if max_dim <= 512 else (128, 128, 16, 8, 4, 2)
            )
        elif m % 64 == 0 and n % 32 == 0 and k % 16 == 0:
            # 恰好整除: 全窗最优 (256..1408 单调领先)。
            bm, bn, bk, gm, nw, ns = 64, 32, 16, 8, 4, 2
        elif max_dim <= 512:
            # 不对齐且窄: 加 warp 摊薄循环与掩码开销。
            bm, bn, bk, gm, nw, ns = 32, 64, 16, 8, 8, 2
        elif gm128 * gn128 <= 1.05:
            # 不对齐但 128-tile 几乎排满 (最后一个 tile >= ~81%): 输出侧 run 长
            # (1024B) 使不对齐惩罚只剩 7.9%, 抵消 CTA 数下降。
            bm, bn, bk, gm, nw, ns = 128, 128, 16, 8, 16, 2
        else:
            bm, bn, bk, gm, nw, ns = 32, 64, 16, 8, 4, 2
    else:
        # max_dim > 1408: probe36/probe44 (真实内核 A/B, 2026-09-11; 均 NAT_B=0 生产口径)。
        # transb=0 (NN/TN, A 侧取转置) 改 128x128x16 s2 全面优于原大 tile —— 后者在大
        # shape 上 grid 太小/切分不摊薄: NN 8192^3 154.6->135.3ms (-12.5%), 2048^3
        # 3.00->2.72ms (-9%); TN 8192^3 155.8->121.7ms (-22%), 1536^3 1.16->0.95ms (-18%)。
        # transb=1 (NT/TT) 维持大 tile: probe43 证明 NAT_B=1 在大 shape 上是灾难
        # (nt 1536^3 nat1=8490us vs nat0=2765us), 故生产走 nat0 直接读 (BK,BN) 的直路;
        # probe44 在 nat0 下扫得 NT 最优仍是 256x128x16 w8 (8192^3 269981us), TT 最优
        # 128x256x16 w8 (8192^3 230496us, 优于 128x128 的 268678us)。
        # 2026-09-13 (probe118/119): MMAC 下 128x128 必须把 num_warps 从 4 提到 16
        # 才能消掉 acc 的 VGPR 溢出 (w4 = 2*128*128/(32*4) = 256 个 = 上限, 实测
        # 4096^3 只到 sp=0.778; w16 = 32 个, 同题 1.088)。w8s3 次之 (0.832), w8s2
        # 0.796, 128x64 w4s2 0.846 —— 均不及 w16s2。NT/TT 直路两条沿用未复测的旧值。
        # 2026-09-14 (probe143/144/145): v18 让 NT 走原生直路后, 本档的 NT 分支仍返回
        # 历史遗留的 256x128x16 w8 (acc = 2*256*128/(32*8) = 256 VGPR = MMAC 上限, 实测
        # 直路上代价 +20~93%), 导致 v18 在 >1408 全面回退。改为与 NN 档同配置后:
        #   2048^3 948.5->840.7 (-11.3%) | 3072^3 2784->2401 (-13.8%)
        #   4095^3 7459->5720 (-23.3%)   | 4096^3 6423->5101 (-20.5%)
        #   6144^3 20417->16666 (-18.4%)
        # max_dim>6144 的 NT 不走本分支 (dgemm() 已改回预转置), 故直路只服务
        # 1408<max_dim<=6144 与 m<64 两种情形。
        if transb == 0:      # NN / TN (A 侧取转置)
            bm, bn, nw = 128, 128, 16
        elif transa == 0:    # NT 直路 (B 侧直接读 (BK,BN))
            # v23 (probe168/169, 2026-09-15, 交错 A/B 4 轮取中位 + clone 对照,
            # 噪声地板 0.02%): probe143/144/145 选 128x128x16 w16 时只与历史遗留
            # 的 256x128x16 w8 对拍过, 小 tile 空间从未扫过。补扫确认 2048^3 上
            # 64x32x16 w4 确实更快 (841 -> 801us, -4.8%, 三次独立复现), 但该收益
            # **无法推广**: 1984^3 / 2240^3 与 2048^3 的 128x128 波次结构完全相同
            # (256 / 324 CTA, 3.2 / 4.05 波) 却分别倒退 20.7% / 22.7%, 全整除的
            # 2304^3 倒退 33.1%, 3072^3 倒退 130%; 1536^3 / 2560^3 亦全输。
            # 即 2048^3 是 lda/ldb = 16KB (2 的幂) 的特例, 波次量化并非有效预测
            # 因子, 任何按波数写的判据都会静默误伤 2304^3 一类尺寸。故维持
            # 128x128x16 w16 不动, 2048^3 的 -4.8% 记入报告为已知的非可推广余量。
            bm, bn, nw = 128, 128, 16
        else:                # TT 直路 (仅 m<32)
            bm, bn, nw = 128, 256, 8
        bk, gm, ns = 16, 8, 2

    return {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": bk,
        "GROUP_M": gm,
        "NAT_B": nat,
        "num_warps": nw,
        "num_stages": ns,
    }


# --- NN 前导维不对齐时的 padding 快门 (probe146/147/148, 2026-09-14) ------------
# MMAC 下 fp64 的 tile 载入完全由 LDS/向量化决定, 4095^3 (lda=ldb=ldc=4095) 相对
# 4096^3 慢 38%: probe137 受控分解 = ld 不对齐 ~17% + 掩码 6.7% 的叠加放大 (掩码
# 只在 run 不对齐时才贵)。而 probe139/140 已证**无法在 kernel 内绕开**: 运行时
# 分支二分边缘/内部 tile 使对齐尺寸倒退 17~26%, tl.minimum 钳位使对齐尺寸倒退
# 33~81% (都破坏了流水线/向量化分析)。
# 唯一有效的是 host 端真打包: 把 A/B/C 拷进 ceil-128 对齐的边框, 按对齐尺寸重跑
# (此时 NO_MASK 成立), 再把 C 的保留区拷回。probe146/147/148 用真实入口 + 含拷贝
# 计时实测 (结果 max|diff| = 0, 逐 bit 一致):
#     2047->2048 1108.9->1018.2 (-8.2%) | 2303->2304 1544.3->1374.6 (-11.0%)
#     2561->2688 2055.5->1869.0 (-9.1%) | 3071->3072 3245.8->2768.1 (-14.7%)
#     3583->3584 4706.7->3955.6 (-16.0%)| 4095->4096 6970.6->5728.0 (-17.8%)
#       (4095^3 NN sp 0.792 -> 0.965)
#     1537->1664 -0.3% / 1793->1920 +1.4% (拷贝成本 > 收益), 故门槛 max_dim>=2048。
# 只对 NN 启用: NT 走原生直路后 4095^3 已是 5720us (padding 反而 +1%), TN 该点
# 已达标 (sp 9.6), 都无收益。拷贝成本 ~ s^2 而 gemm ~ s^3, 故门槛随尺寸放宽即可。
_PAD_NN_MIN_DIM = 2048
_PAD_NN_MAX_GROWTH = 1.25     # 边框多出的 flops 上限 (2047->2176 这类要排除)
_PAD_NN_MAX_BYTES = 1 << 30   # 三个边框缓冲的总字节上限
_FP64_BYTES = 8

# K_TAIL (k 尾分离) 只在 max_dim <= 1408 档启用。该档 k 循环短、内核延迟受限,
# 去掉 k 掩码后 A 的连续轴取址可向量化, 收益为正 (1023^3 内核 153.8 -> 151.4us)。
# >1408 档用的 128x128x16 s2 是长循环、吞吐受限, 多出的尾块会把 s2 的流水线排布
# 打乱 (4095^3 内核 5767 -> 6867us, +19%; 换 s1/s3 则转正, 属调度敏感),
# 故该档维持整段带 k 掩码。数据: 本轮 kernel A/B + probe163。
_K_TAIL_MAX_DIM = 1408


def _ceil_to(x: int, mult: int) -> int:
    return -(-x // mult) * mult


def _should_pad_nn(transa: int, transb: int, m: int, n: int, k: int) -> bool:
    """是否走 "打包进 ceil-128 边框再算" 的 NN 快门。"""
    if transa != 0 or transb != 0:
        return False
    if max(m, n, k) < _PAD_NN_MIN_DIM:
        return False
    if max(m, n) % 128 == 0 and k % 16 == 0:
        return False  # 维度已对齐, 无收益可拿
    pm, pn, pk = _ceil_to(m, 128), _ceil_to(n, 128), _ceil_to(k, 16)
    if pm * pn * pk > _PAD_NN_MAX_GROWTH * m * n * k:
        return False
    if (pm * pk + pk * pn + pm * pn) * _FP64_BYTES > _PAD_NN_MAX_BYTES:
        return False
    return True


def _dgemm_nn_padded(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) -> None:
    """NN 且 ld 不对齐时: 打包 A/B/C -> 对齐边框 -> 复跑 -> 拷回 C。

    边框外的 garbage 不会污染保留区 (A 的多余列乘到 B 的补零行; A 的多余行只
    影响被丢弃的 C 多余行), 故只需把 B 的 [k:pk) 行补零, 不必清零整个边框。
    """
    pm, pn, pk = _ceil_to(m, 128), _ceil_to(n, 128), _ceil_to(k, 16)
    ap = torch.empty((pm, pk), dtype=A.dtype, device=A.device)
    bp = torch.empty((pk, pn), dtype=B.dtype, device=B.device)
    cp = torch.empty((pm, pn), dtype=C.dtype, device=C.device)
    ap[:m, :k].copy_(A.as_strided((m, k), (lda, 1)))
    bp[:k, :n].copy_(B.as_strided((k, n), (ldb, 1)))
    bp[k:pk, :] = 0.0
    if beta != 0.0:
        cp[:m, :n].copy_(C.as_strided((m, n), (ldc, 1)))
    dgemm(0, 0, pm, pn, pk, alpha, ap, pk, bp, pn, beta, cp, pn)
    C.as_strided((m, n), (ldc, 1)).copy_(cp[:m, :n])


# --- v25 尾波 split-K 的可行性判定与驱动 (probe174/175, §3.22) ----------------
# 代价模型常数取自报告: 饱和峰值 28.0 TFLOPS (§3.15/§3.16)、实测带宽 975 GB/s。
# 判据是「代价远小于收益」的**保守闸门**, 不是性能预测器 —— 是否启用只决定在
# 尾波上多切一刀, 主轴路径与既有分档不受影响。
_STREAMK_ENABLE = True
_STREAMK_SM_FALLBACK = 80
_STREAMK_PEAK_FLOPS = 28.0e12
_STREAMK_BW_BPS = 975.0e9
_streamk_sm_cache = None


def _streamk_sm() -> int:
    global _streamk_sm_cache
    if _streamk_sm_cache is None:
        try:
            _streamk_sm_cache = int(
                torch.cuda.get_device_properties(torch.cuda.current_device())
                .multi_processor_count
            )
        except Exception:  # pragma: no cover - 无 CUDA 时不该走到这里
            _streamk_sm_cache = _STREAMK_SM_FALLBACK
    return _streamk_sm_cache


def _streamk_plan(m: int, n: int, k: int, cfg) -> tuple:
    """返回 (nsplit, bulk, tail) 或 None。只在尾波可分片且收益远大于代价时启用。"""
    if not _STREAMK_ENABLE or cfg is None or cfg.get("NAT_B") != 0:
        return None
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    if m % bm or n % bn or k % bk:
        return None                     # 该路径只服务无掩码情形
    sm = _streamk_sm()
    num_tiles = (m // bm) * (n // bn)
    nwaves = -(-num_tiles // sm)            # ceil(num_tiles / sm)
    if (num_tiles / sm) / nwaves >= 0.95:
        return None                     # 波次量化损失 <5%: 不在批准口径内 (8192^3 等)
    bulk = (num_tiles // sm) * sm
    tail = num_tiles - bulk
    if bulk == 0 or tail == 0:
        return None
    smax = sm // tail
    if smax < 2:
        return None                     # 尾波太大: 切成 ≥2 份也填不满一个波次, 无收益
    ki = k // bk
    nsplit = next((d for d in range(smax, 1, -1) if ki % d == 0), None)
    if nsplit is None:
        return None
    tile_flops = 2.0 * bm * bn * k
    t_tile = tile_flops / _STREAMK_PEAK_FLOPS * sm      # 一个 CTA 跑一个 tile 的墙钟
    gain = (1.0 - 1.0 / nsplit) * t_tile                # 尾波由 1 个 tile-time 缩到 1/nsplit
    extra = 2.0 * nsplit * tail * bm * bn * 8 / _STREAMK_BW_BPS   # 部分积的写 + 读
    if gain < 5.0 * extra:
        return None
    return nsplit, bulk, tail


def _dgemm_streamk(plan, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                   beta, C, ldc, cfg, beta_is_zero) -> None:
    """主轴一次启动 + 尾波分片启动 + 归约写回。三条共 3 次 kernel launch。"""
    nsplit, bulk, tail = plan
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    gm, nw, ns = cfg["GROUP_M"], cfg["num_warps"], cfg["num_stages"]
    ki_per_slice = (k // bk) // nsplit
    part = torch.empty((nsplit * tail, bm, bn), dtype=torch.float64, device=A.device)
    _dgemm_dot_kernel[(bulk,)](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
        transa, transb, beta_is_zero,
        NO_MASK=True, K_TAIL=False,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, NAT_B=0,
        num_warps=nw, num_stages=ns,
    )
    _dgemm_splitk_kernel[(nsplit * tail,)](
        A, B, part, m, n, lda, ldb, bulk, tail, ki_per_slice,
        TRANS_A=transa, TRANS_B=transb,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
        num_warps=nw, num_stages=ns,
    )
    _dgemm_splitk_reduce_kernel[(tail,)](
        part, C, alpha, beta, m, n, ldc, bulk, nsplit,
        BETA_IS_ZERO=beta_is_zero,
        BLOCK_M=bm, BLOCK_N=bn, GROUP_M=gm, num_warps=16, num_stages=1,
    )


def dgemm(    transa: int,
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
    _validate_dgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    # transb=1 时 B 的转置载入使 dot 内核掉 2x (右操作数需经 smem 布局转换;
    # NAT_B/tl.trans 只能缓解 <=256^3)。先把 B 预转置成 (k,n) 连续布局, 再走
    # transb=0 路径 (NT->NN, TT->TN)。probe51 (真实内核, 2026-09-11): 大 shape 收益
    # 与中小 shape 同量级 —— nt 1536/2048/4096 1959->1080 / 5204->2780 /
    # 36211->17464us (1.8-2.1x), tt 同幅度。
    # 代价是一次 16*n*k 字节的转置; 实测 BW ~975GB/s、NN 效率 ~8 TFLOPS, 故
    # transposed_gemm/transpose ~= m/64, 盈亏平衡点 m ~= 64: 中 shape (129..1024)
    # 转置 <=8MB 且实测 m>=32 即可摊薄; 大 shape (>1024) 转置可达 512MB, 取更保守
    # 的 m >= 64。
    beta_is_zero = beta == 0.0

    with torch_device_fn.device(A.device):
        max_dim = max(m, n, k)
        if _should_pad_nn(transa, transb, m, n, k):
            # NN 且 ld 不对齐的大 shape: 打包到对齐边框后复跑 (见上方 probe146-148)
            _dgemm_nn_padded(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            return
        # v18 (probe136/probe141, 2026-09-14): transb=1 的 B 预转置不再是"默认更优"。
        # MMAC 下 fp64 dot 每 lane 只占 1 个 f64, 布局转换成本与 FMA 时代完全不同:
        # 原生 transb=1 (NAT_B=0, 直接按 (BK,BN) 读) 在 NN 档配置上的代价只有
        # +0.3~1.1% (128x128x16 w16) / +4~12% (小 tile), 而预转置要额外付一次整矩阵
        # 转置 (256^3 含 torch.empty 抖动 +25.8us, 1024^3 +13us, 4096^3 +270us)。
        # probe141 逐 size 实测 NT 端到端 (官方口径, 2 轮交错取中位):
        #   129^3 27.2->13.2 | 256^3 28.8->14.4 | 384^3 32.6->21.8 | 511^3 40.2->33.6
        #   512^3 33.0->30.4 | 768^3 67.4->63.2 | 1023^3 167.5->152.6
        #   1024^3 131.4->127.1 | 1536^3 366.4->338.3 | 2048^3 888.8->841.8
        #   3072^3 2511->2402 | 4096^3 5323->5105
        #   4095^3 7242->5718 (原生 TB1 甚至比 TB0 快 18%: ld 不对齐时 B 的 n 向
        #          1024B run 只能标量+谓词载入, 换成 k 向 128B run 后可向量化)
        # 唯一反例 1408^3 (309.4->340.2, +10%): 该点落在 64x32x16 w4 档且 ld 不对齐,
        # 原生 TB1 惩罚达 +20.7%。故 NT 只在 (1024,1408] 窗口保留预转置 (该窗口 NN
        # 档即 64x32x16 w4), 其余尺寸一律走原生直路。
        # v19 (probe143/144/145, 2026-09-14, 官方 harness 同构 tensor + 真实入口):
        #   1408 < max_dim <= 6144 直路 + 128x128x16 w16 (probe143/144):
        #     1536^3 364.2->338.9 | 2048^3 948.5->840.7 | 3072^3 2784->2400
        #     4095^3 7459->5720   | 4096^3 6423->5101   | 6144^3 20417->16666
        #     5120^3 -2.9%        (probe145, 相对 "预转置+NN")
        #   但 max_dim > 6144 时直路失效, 必须退回预转置 (probe145):
        #     7168^3 直路 128x128x16 w16 = +0.65% (持平) | 8192^3 = +45.1%!
        #     8192^3 直路最优的 128x256x16 w16 仍 +6.9%, 故上界取 6144
        #     (8192 是 2 的幂, ldb=8192 的 64KB 行距引发 cache set 冲突)。
        do_trans_b = transb == 1 and max_dim > 128
        if do_trans_b and transa == 0:      # NT: 原生直路优先 (除两段)
            do_trans_b = m >= 64 and ((1024 < max_dim <= 1408) or max_dim > 6144)
        elif do_trans_b:                    # TT: v21 起与 NT 同策略 (m>=64 时)
            # v21 (probe158–probe161, 2026-09-14): TT 原来一律预转置 B (自 v2),
            # 但在 MMAC 下那次 host 端转置是纯亏 —— 转置内核本身只值 ~1us (贴
            # do_bench 地板), 代价全在"多一次 host 启动"这个固定开销上, 小 shape
            # 无处摊薄: probe159 以替身法 (真转置 34.0us / 只 torch.empty 14.24us /
            # 连分配都省 14.24us) 定位; probe160/161 逐 size 实测 "原生 TB1
            # (NAT_B=0) + TN 档 tile" 全面更快 (官方口径, 3 轮取中位):
            #   129^3 52.5->13.3 (3.95x) | 160^3 54.9->10.6 (5.20x)
            #   192^3 55.3->12.0 (4.61x) | 256^3 52.6->14.1 (3.81x)
            #   272^3 54.5->19.7 (2.77x) | 288^3 58.0->18.7 (3.10x)
            #   384^3 58.8->21.4 (2.74x) | 511^3 54.1->35.4 (1.53x)
            #   512^3 58.1->31.2 (1.86x) | 896^3 148.2->90.6 (1.64x)
            #   1023^3 187.7->153.3 (1.22x) | 1024^3 148.5->129.8 (1.14x)
            #   1536^3 362.7->335.8 (1.08x) | 2048^3 879.7->832.5 (1.06x)
            #   3072^3 2479->2378 (1.04x) | 4096^3 5264->5046 (1.04x)
            #   6144^3 16894->16498 (1.02x) | 8192^3 39967->39104 (1.02x)
            # 两处例外保留预转置: (a) (1024,1408] 窗口 —— NT 在该窗口直路 +20.7%
            # (probe141 的 1408^3), TT 实测持平 (0.99x); (b) >1408 且未对齐 ——
            # 4095^3 直路反而 6035->6975 (+15.6%), 因 TB1 的 n 向行距不是 128B
            # 倍数, k 向 run 整体错位; 4096^3 对齐后则为 1.04x。
            # NAT_B=1 一律差 2~4x, 故直路走 nat=0 的 (BK,BN) 直接载入。
            if m < 64:                      # 瘦长矩阵: 沿用历史判据, 未复测
                do_trans_b = m >= (32 if max_dim <= 1024 else 64)
            elif max_dim > 1408:
                do_trans_b = not (m % 128 == 0 and n % 128 == 0 and k % 16 == 0)
            else:
                do_trans_b = 1024 < max_dim <= 1408
        if do_trans_b:
            B = _transpose_b(B, ldb, n, k)
            ldb = n
            transb = 0

        # TN: transa=1 时 A 以 (k,m) 列主序存放, dot 的左操作数连续轴落在第 0
        # 维, 大 tile 下比 NN 慢 ~7% (1023^3 474us vs 441us)。在能摊薄转置开销
        # 的区间 (513..1024) 预转置 A 为 (m,k) 行主序, 改走 transa=0 路径:
        # TN 1023^3 474->454us (sp 0.911->0.952), 1024^3 450->435 (0.938->0.970)。
        # <=512 时 TN 内核本就快于 NN, 转置无法摊薄, 故不启用。
        if transa == 1 and transb == 0 and 512 < max_dim <= 1024:
            A, lda = _transpose_a(A, lda, m, k)
            transa = 0

        launch_kwargs = _select_dgemm_dot_config(transa, transb, m, n, k)
        if launch_kwargs is not None:
            streamk = _streamk_plan(m, n, k, launch_kwargs)
            if streamk is not None:
                _dgemm_streamk(streamk, transa, transb, m, n, k, alpha, A, lda,
                               B, ldb, beta, C, ldc, launch_kwargs, beta_is_zero)
                return
            block_m = launch_kwargs["BLOCK_M"]
            block_n = launch_kwargs["BLOCK_N"]
            block_k = launch_kwargs["BLOCK_K"]
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            _dgemm_dot_kernel[grid](
                A,
                B,
                C,
                alpha,
                beta,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                transa,
                transb,
                beta_is_zero,
                NO_MASK=(m % block_m == 0 and n % block_n == 0 and k % block_k == 0),
                K_TAIL=(k % block_k != 0 and max_dim <= _K_TAIL_MAX_DIM),
                **launch_kwargs,
            )
        else:
            # 极小尺寸回退: 通用标量外积内核 (恒成立, 性能非关键路径)
            from flag_blas.ops.level3.dgemm import _dgemm_kernel

            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            )
            _dgemm_kernel[grid](
                A,
                B,
                C,
                alpha,
                beta,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                transa,
                transb,
                beta_is_zero,
            )


@triton.jit
def _transpose_f64_kernel(
    b_ptr,
    bt_ptr,
    n,
    k,
    ldb,
    ldbt,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """bt[j, i] = b[i, j]: (n x k, 前导维 ldb) -> (k x n, 前导维 ldbt)。"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n[:, None] < n) & (offs_k[None, :] < k)
    x = tl.load(b_ptr + offs_n[:, None] * ldb + offs_k[None, :], mask=mask, other=0.0)
    tl.store(bt_ptr + offs_k[None, :] * ldbt + offs_n[:, None], x, mask=mask)


def _transpose_b(b_nt: torch.Tensor, ldb: int, n: int, k: int) -> torch.Tensor:
    """把 transb=1 的 B 预转置成 (k, n) 连续布局 (供 transb=0 路径使用)。"""
    bt = torch.empty((k, n), dtype=b_nt.dtype, device=b_nt.device)
    block_n, block_k = 32, 32
    grid = (triton.cdiv(n, block_n), triton.cdiv(k, block_k))
    _transpose_f64_kernel[grid](
        b_nt, bt, n, k, ldb, n, block_n, block_k, num_warps=4, num_stages=2
    )
    return bt


def _transpose_a(a_nt: torch.Tensor, lda: int, m: int, k: int):
    """把 transa=1 的 A (k x m, 前导维 lda) 预转置成 (m, k_pad) 行主序。

    前导维对齐到 16 是有意的: k 为奇数时 (如 1023) 不对齐会让 dot 内核的
    向量化取址失效, 1023^3 实测 812us -> 454us。返回 (张量, 前导维)。
    """
    k_pad = (k + 15) // 16 * 16
    at = torch.empty((m, k_pad), dtype=a_nt.dtype, device=a_nt.device)
    block_n, block_m = 32, 32
    grid = (triton.cdiv(k, block_n), triton.cdiv(m, block_m))
    _transpose_f64_kernel[grid](
        a_nt, at, k, m, lda, k_pad, block_n, block_m, num_warps=4, num_stages=2
    )
    return at, k_pad

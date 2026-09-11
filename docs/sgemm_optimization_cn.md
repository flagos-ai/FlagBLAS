<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# SGEMM 转置变体性能优化

## 概述

本文档记录 MThreads (MUSA) 后端 `sgemm` 算子四个转置变体
`sgemm_nn` / `sgemm_nt` / `sgemm_tn` / `sgemm_tt` 的性能优化设计与验证结果。

优化参考已完成的 `sgemm_nn` 快速路径（tf32x3 + fp16 hi/lo 拆分），将全部四种
转置变体统一路由到同一套快速实现（`src/flag_blas/runtime/backend/_mthreads/ops/sgemm.py`），
在不降低精度容忍度的前提下显著提升吞吐。

## 背景与验收标准

- 目标平台：沐曦 MUSA GPU，muBLAS 参考库（`libmublas.so`），flagtree Triton 兼容编译器
- kernel 必须为 Triton 实现
- `benchmark/core_shapes` 中全部 shape 加速比 ≥ 0.95
- 精度容忍度不劣于 `sgemm_nn`（perf tolerance = 1.3e-6）

优化前转置变体走通用 fp32 参考 kernel，存在两个性能瓶颈：

1. **fp32 IEEE dot 吞吐极低**（约 0.9 TFLOPS），大 shape 与 muBLAS 差距明显
2. **小 shape 转置**需经 pad + 拷贝缓冲，启动开销占比过高（64³ 转置加速比仅 0.47–0.62）

## 总体设计：双路径

按问题规模选择两条计算路径（与 `sgemm_nn` 一致）：

| 路径 | 适用规模 | 说明 |
|---|---|---|
| **tf32x3** | `m*n*k ≤ 512³` | 单 kernel，`tl.dot(input_precision="tf32x3")`，pad 到 64 倍数 |
| **fp16 hi/lo 拆分** | 大 shape | 拆分为 fp16 hi/lo（低位 ×4096），3 次 fp16 MMA（`hi@hi`+`hi@lo`+`lo@hi`），pad 到 256 倍数 |

`fp16 hi/lo 拆分` 用 fp16 张量核心替代 fp32 FMA，在保持 fp32 精度的同时获得接近
fp16 的峰值吞吐。两种路径下 MMA 始终运行在**规范行主序** `(m,k)×(k,n)` 上。

## 转置规范化

四种变体的 MMA 全部运行在规范布局上，转置差异在计算前被折叠，避免在 MMA 循环内
引入转置开销。不同路径采用不同策略：

### 1. fp16x3 路径：split kernel 内融合转置

`_split_a_kernel_t` / `_split_b_kernel_t` 使用 2D tile 加载（`BLOCK_J × BLOCK_I`）读取
转置源，经 `tl.trans` 寄存器转置后连续写出规范布局。

> 动机：1D 分解读取转置源是步长 `lda` 的跳跃访问，tall shape 下 split 耗时 1.373ms；
> 改为 2D tile + `tl.trans` 后降至 0.231ms（实测 (2048,16384) TT 的 A split）。
> `tl.trans` 在 MThreads 后端可用且位精确。

### 2. tf32x3 路径：小 shape 直接读转置源

无 padding（`m/n/k % 64 == 0`）且 `m*n*k ≤ 256³`（或非转置）时，直接对原始操作数
运行 mask-free kernel，kernel 通过 `TRANS_A` / `TRANS_B` 常量参数控制转置读取。
该尺寸下转置读全部命中缓存，避免了占主导的缓冲拷贝开销（64³ 转置由此从 0.47–0.62
提升到 1.51–1.77）。

### 3. tf32x3 路径：其余 shape 经缓冲拷贝规范化

`_fill_padded` 通过 `src.t()` 转置视图将操作数拷贝进 pad 后的缓存缓冲，缓冲为规范
行主序布局，kernel 不再需要转置标志。

## 关键实现细节

### 缓存缓冲步长

`_get_buf` 缓存的输入缓冲可能大于当前 padded shape（由先前更大 shape 填充）。
split kernel 与 MMA kernel 必须使用 `a.stride(0)` / `b.stride(0)` 等**实际行步长**，
而非当前 `pm/pn/pk`，否则大 shape 缓存后小 shape 会读错位。

### tf32x3 direct path 的 TRANS 传递

直接路径（不拷贝缓冲）向 kernel 传递真实 `trans_a/trans_b`；缓冲路径的操作数已是
规范布局，必须传 `False, False`，否则会发生二次转置读取垃圾数据（曾导致 9 个 shape
精度失败，`max_abs` 高达 1e7）。

### 其他

- 缓存输入缓冲避免重复分配与 pad 拷贝
- tile store 时编译器用目标 tensor 行步长替代显式 `ldc`，输出缓冲必须精确尺寸
- `+ beta`（运行时 0）trick 避免纯标量 store 误编译
- 输出写回通过 `copyout[:m,:n].copy_(...)` 在 host 侧完成，不额外启动 kernel

## 性能数据

`benchmark/test_sgemm_perf.py`（`--level core`，37 shape × 3 转置变体）：

| 变体 | shape 数 | 最小加速比 | 大 shape 加速比 |
|---|---|---|---|
| `sgemm_tn` | 37 | 1.300 | 约 1.30–1.69 |
| `sgemm_nt` | 37 | 1.151 | 约 1.15–1.68 |
| `sgemm_tt` | 37 | 1.232 | 约 1.23–1.66 |

64³ 小 shape 转置（修复 direct path 前 → 后）：

| 变体 | 修复前 | 修复后 |
|---|---|---|
| `sgemm_tn` | 0.616 | 1.558 |
| `sgemm_nt` | 0.470 | 1.771 |
| `sgemm_tt` | 0.514 | 1.511 |

## 精度验证

| 验证项 | 结果 |
|---|---|
| 16 shape × 4 变体 muBLAS 对照（beta=0） | 64/64 PASS |
| alpha=2.5 / beta=0.5 | 12/12 PASS |
| `tests/test_sgemm.py` 全量 | 106 PASS |

## 经验教训

- MThreads 后端 fp32 IEEE dot 仅约 0.9 TFLOPS，大 shape 必须走 fp16/tf32 张量核心
- `acc2 * INV_S` 出现在循环内会触发慢路径，需提前缩放
- mthreads JIT 要求 kernel 定义在 `.py` 文件中（不能 `python -c` 内联）
- 转置读取在寄存器和缓存层面折叠，比在 MMA 层处理转置高效得多

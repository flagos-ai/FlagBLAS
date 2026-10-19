# patches / flagtree

FlagBLAS 依赖的 Triton 定制版（flagtree 0.6.1+hcu3.6）以 pip 包形式安装在
site-packages，不属于 FlagBLAS git 仓库。对 Triton 源码的定制改动以 patch 形式
固化在此目录，部署/重建环境时重新应用。

## 现状（2026-08-22 起）

**per-kernel ITT 编译选项已改为 FlagBLAS 侧运行时注入实现，不再需要修改
site-packages**。注入代码见
`src/flag_blas/runtime/backend/_hygon/_triton_itt.py`（sgemm.py 导入时自动安装）：
- 动态给 `HIPOptions`（frozen dataclass）增加 `use_in_thread_transpose` 字段，
  使 `parse_options` 从 launch kwargs 自动吸收、`compute_cache_key` 的
  `str(options)` 自动区分 ITT 变体编译产物；
- 包装 `HIPBackend.make_ttgir`，编译时按 per-kernel 值临时设置
  `TRITON_HIP_USE_IN_THREAD_TRANSPOSE` env（knob 动态读 env），编译后恢复。

已实测验证（2048³ 冷缓存 A/B）：ITT=1 比 ITT=0 快 ~7.5%，cache key 区分、顺序无关；
10 个代表 shape 的 per-shape ITT 决策与 site-packages patch 版一致。

site-packages 中 `triton/backends/hcu/compiler_hcu.py` 已恢复为原始版本。
若未来仍需要手工 patch（例如注入机制与某 Triton 版本不兼容），可应用下方 patch。

## flagtree-0.6.1-compiler-hcu-per-kernel-itt.patch

**用途（历史/兜底）**：为 Triton HIPOptions 新增 `use_in_thread_transpose` 编译选项，
使 sgemm_nn 可按 shape 控制 in-thread-transpose（ITT）pass 开关（v1.2 per-kernel
ITT 优化，见 benchmark/sgemm_nn_perf_regression_report.md §3.4）。

**目标文件**：`triton/backends/hcu/compiler_hcu.py`（flagtree 0.6.1+hcu3.6）

**应用**（在 flagtree 安装目录的 site-packages 下）：

```bash
cd /data/zhangbin/py310/lib/python3.10/site-packages
patch -p1 < /data/zhangbin/cicd/FlagBLAS/patches/flagtree-0.6.1-compiler-hcu-per-kernel-itt.patch
```

**验证**：`grep use_in_thread_transpose backends/hcu/compiler_hcu.py` 应命中
HIPOptions 字段与 compile 中 `itt_enabled = options.use_in_thread_transpose`。

**注意**：该改动会改变 Triton `cache_key`（HIPOptions 字段增加），导致
`/root/.flagblas/config_cache/` 下 config DB 表名漂移；首次运行会重建新表
（运行时自动 benchmark 填充，37 个核心 shape 的 config 值与原表一致）。

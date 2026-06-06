# FlagBLAS Dispatch 模块

## 概述

`dispatch` 模块 (`src/flag_blas/runtime/dispatch.py`) 提供了**基于问题尺寸的自动调优 kernel 变体选择**框架。它解决了这样一个核心问题：对于同一个数学运算（例如 GEMM），存在多个不同实现的 kernel 变体，每种变体在不同的矩阵形状、对齐条件下性能各异。`SizeAutoDispatch` 在首次遇到新形状时通过实际 GPU benchmark 选出最优变体，并将结果缓存到进程内内存和持久化数据库中，后续调用零开销命中。

## 适用场景

- 同一个算子有**多种 kernel 实现变体**（如对齐优化版、padding 版、Split-K 版、通用 fallback 版）
- 不同变体在**不同形状和对齐条件**下各有优势
- 希望**首次运行自动 benchmark，后续零开销**直接选择最优实现
- 需要**跨进程重启持久化**自动调优结果

## 架构设计

### 模块组成

```
dispatch.py
├── _autotune_result_cache    # 模块级进程内内存缓存
├── _autotune_result_lock     # 线程安全锁
├── _SizeFilter               # 尺寸过滤函数类型别名
├── SizeAutoDispatch          # 自动调优 dispatch 核心类
└── KernelRunner              # 预绑定参数的 kernel 调用包装器
```

### 缓存架构

`SizeAutoDispatch.lookup_and_build()` 使用**四级查询**的方式查找最优 kernel 变体：

```
                         ┌─────────────────────┐
                         │  lookup_and_build()  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  len(entries) == 1  │
                         │  直接返回唯一候选     │  ← O(1), 零开销
                         └──────────┬──────────┘
                                    │ 候选数 ≥ 2
                         ┌──────────▼──────────┐
                         │  内存缓存查询         │  ← dict lookup, O(1)
                         │  (进程内, 跨实例)     │
                         └──────────┬──────────┘
                                    │ miss
                         ┌──────────▼──────────┐
                         │  DB 缓存查询          │  ← SQLite I/O
                         │  (跨进程重启持久化)    │
                         └──────────┬──────────┘
                                    │ miss
                         ┌──────────▼──────────┐
                         │  自动调优 (benchmark) │  ← GPU benchmark
                         │  写入 内存 + DB 缓存   │
                         └─────────────────────┘
```

#### 缓存层级详解

| 层级 | 存储介质 | 生命周期 | 查询开销 | 用途 |
|------|---------|---------|---------|------|
| `len(entries)==1` | 无 | 每次调用 | O(1) | 唯一候选直接返回 |
| 内存缓存 | Python dict | 进程存活期间 | O(1) | 解决"每次 new instance"问题 |
| DB 缓存 | SQLite 文件 | 永久 | ~0.1ms | 跨进程重启持久化 |
| Autotune | GPU | 首次遇到 | seconds | 实际 benchmark 选出最优 |

**内存缓存** (第 29-30 行):

```python
_autotune_result_cache: Dict[Tuple[str, tuple], str] = {}
_autotune_result_lock = threading.Lock()
```

- Key: `(table_name, cache_key)` — table_name 区分不同 dispatch（如 `"sgemm_nn_variant"` vs `"hgemm_nn_variant"`），cache_key 编码问题形状
- Value: 最优候选变体的名称字符串
- 模块级全局变量，所有 `SizeAutoDispatch` 实例共享
- 无持久化开销，进程重启自动清空

---

## 核心类: SizeAutoDispatch

### 构造参数

```python
SizeAutoDispatch(
    table_name: str,
    build_key: Callable[..., tuple],
    model: Optional[PersistantModel] = None,
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `table_name` | `str` | DB 表名，用于持久化 `(key → best_candidate)` 映射。不同 dispatch 使用不同表名避免冲突 |
| `build_key` | `(m, n, k, aligned, **extra) -> tuple` | 从问题维度构建唯一缓存 key 的函数 |
| `model` | `PersistantModel \| None` | 数据库访问实例。`None` 则仅使用内存缓存，不持久化 |

### add() — 注册候选变体

```python
def add(
    self,
    factory: Callable[[], Callable[[], None]],
    *,
    aligned: Optional[bool] = None,
    name: Optional[str] = None,
    filter: Optional[_SizeFilter] = None,
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `factory` | `() -> Callable[[], None]` | **零参工厂函数**，返回一个可直接调用执行的 runner。采用惰性构造：只有被选中的候选才会调用 factory |
| `aligned` | `bool \| None` | 候选匹配的对齐条件。`True` 仅匹配对齐场景，`False` 仅匹配未对齐场景，`None` 匹配所有 |
| `name` | `str \| None` | 候选名称，用于 DB 存储和日志。不指定则自动生成 `"variant_0"`, `"variant_1"` ... |
| `filter` | `_SizeFilter \| None` | 尺寸过滤函数，签名 `(m, n, k, aligned, **extra) -> bool`。返回 `True` 时该候选加入候选池 |

**关键设计: Lazy Runner Construction**

`add()` 接受的是**工厂函数**而非 runner 本身。这是因为：

1. 创建 runner 可能包含昂贵的内存分配（如 padding 新 tensor）
2. autotune 阶段每个候选都需要 runner，但**最终只有胜出的候选被实际使用**
3. 缓存命中时其他候选的 factory 根本不会被调用

**filter 的设计约束:**

- 同一 `aligned` 取值下的候选 filter 应**互斥**，否则多个候选同时进入池触发不必要的 autotune
- filter 的评判应**仅基于形状维度**，不应涉及任何状态
- 不应出现某个形状**所有候选都被 filter 排除**的真空地带

### lookup_and_build() — 查询并构建最优 runner

```python
def lookup_and_build(
    self,
    m: int,
    n: int,
    k: int,
    aligned: bool,
    *,
    snapshot_tensor=None,
    **extra,
) -> Callable[[], None]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `m` | `int` | 矩阵 M 维度 |
| `n` | `int` | 矩阵 N 维度 |
| `k` | `int` | 矩阵 K 维度 |
| `aligned` | `bool` | 当前输入是否满足对齐条件 |
| `snapshot_tensor` | `Tensor \| None` | 用于 autotune 中恢复状态的快照 tensor。如果提供，每个候选 benchmark 前后都会恢复其原始值，确保公平比较 |
| `**extra` | — | 透传给 `build_key` 和 `filter` 的额外参数 |

**返回值**: 一个零参 callable runner，调用即执行选中的 kernel

**执行流程**:

1. 调用 `build_key(m, n, k, aligned, **extra)` 构建缓存 key
2. 调用 `_get_entries(m, n, k, aligned, **extra)` 过滤出匹配的候选
3. 若仅 1 个候选 → 直接返回其 factory()
4. 查内存缓存 → 命中则返回
5. 查 DB 缓存 → 命中则返回，同时提升到内存缓存
6. 触发 `_autotune()` → benchmark 选出最优 → 写入内存 + DB

### _autotune() — 自动调优

```python
def _autotune(
    self,
    cache_key: tuple,
    entries: List[_Entry],
    snapshot_tensor=None,
) -> Optional[Tuple[_Entry, Callable[[], None]]]
```

对每个候选执行 **1 次 warmup + 5 次计时**:

1. 备份 `snapshot_tensor`（如果提供）
2. 调用 `entry.factory()` 构造 runner（惰性构造，仅此一次）
3. 执行 1 次 warmup + `torch.cuda.synchronize()`
4. 恢复 snapshot
5. 执行 5 次计时，使用 CUDA Event 精确测量 GPU 耗时
6. 每次计时前后恢复 snapshot，消除候选间的状态干扰
7. 记录每个候选的最小耗时
8. 返回耗时最小的 `(entry, runner)` 对

**容错**: 单个候选崩溃不影响其他候选，会记录 warning 日志并跳过。如果所有候选都失败，返回 `None`，caller 回退到第一个候选且**不持久化**，下次调用会重试 autotune。

### 可配置参数

| 类属性 | 默认值 | 说明 |
|--------|--------|------|
| `_WARMUP_ITERS` | `1` | 每个候选的 warmup 次数 |
| `_TIMING_ITERS` | `5` | 每个候选的计时次数 |
| `_CACHE_FLUSH_BYTES` | `16 * 1024 * 1024` | 候选间 flush L2 cache 的 dummy buffer 大小 (16 MiB) |

---

## 辅助类: KernelRunner

```python
class KernelRunner:
    def __init__(self, kernel: Callable, *args, **kwargs):
        ...

    def __call__(self):
        return self._kernel(*self._args, **self._kwargs)
```

一个可调用包装器，将 kernel 函数与其参数绑定在一起。当不需要 autotune、只需固定实现的场景下使用。例如：

```python
runner = KernelRunner(my_kernel_fn, A, B, C, alpha=1.0)
runner()  # 等价于 my_kernel_fn(A, B, C, alpha=1.0)
```

---

## 使用方法

### 基本用法

```python
from flag_blas.runtime.dispatch import SizeAutoDispatch
from flag_blas.utils.libentry import libcache

# Step 1: 定义尺寸过滤函数
def is_large(m, n, k, **extra):
    return m > 1024 and n > 1024 and k > 1024

def is_thin(m, n, k, **extra):
    return min(m, n) <= 64 and k >= 256

# Step 2: 定义 runner 工厂函数（每个返回一个零参 callable）
def make_aligned_runner(A, B, C, m, n, k, ...):
    def run():
        aligned_kernel(A, B, C, ...)
    return run

def make_fallback_runner(A, B, C, m, n, k, ...):
    def run():
        fallback_kernel(A, B, C, ...)
    return run

# Step 3: 构造 dispatch table
def build_dispatch(A, B, C, m, n, k, ...):
    dispatch = SizeAutoDispatch(
        table_name="my_gemm_variant",                 # 唯一表名
        build_key=lambda m, n, k, aligned, **kw: (m, n, k, int(aligned)),
        model=libcache.model,                         # 复用 FlagBLAS 全局 DB
    )

    # Step 4: 注册候选变体
    dispatch.add(
        lambda: make_aligned_runner(A, B, C, m, n, k, ...),
        aligned=True,
        name="aligned",
    )
    dispatch.add(
        lambda: make_fallback_runner(A, B, C, m, n, k, ...),
        aligned=False,
        name="fallback",
    )

    return dispatch

# Step 5: 在算子入口使用
def my_gemm(A, B, C, m, n, k, ...):
    aligned = check_alignment(A, B, C)
    dispatch = build_dispatch(A, B, C, m, n, k, ...)
    runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
    runner()
```

### 完整示例: Hopper sgemm NN 分支

FlagBLAS 中 Hopper 架构下 `sgemm` 的 NN 转置分支是典型应用场景（[hopper/ops/gemm.py:538-570](src/flag_blas/runtime/backend/_nvidia/hopper/ops/gemm.py)）：

```python
def _build_sgemm_nn_dispatch_table(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="sgemm_nn_variant",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned)),
        model=model,
    )

    # 对齐路径: TensorDescriptor + TF32 加速
    dispatch.add(
        lambda: _make_sgemm_nn_aligned_runner(A, lda, B, ldb, C, ldc,
                                               m, n, k, alpha, beta,
                                               beta_is_zero, grid),
        aligned=True,
        name="aligned_k2",
    )

    # 不对齐 + 大矩阵: padding 到 16 对齐
    dispatch.add(
        lambda: _make_sgemm_nn_padded_runner(A, lda, B, ldb, C, ldc,
                                              m, n, k, alpha, beta,
                                              beta_is_zero),
        aligned=False,
        name="padded_k2",
        filter=_is_sgemm_large,   # m>1024 and n>1024 and k>1024
    )

    # 不对齐 + 瘦矩阵: Split-K 策略
    dispatch.add(
        lambda: _make_sgemm_nn_thin_runner(A, lda, B, ldb, C, ldc,
                                            m, n, k, alpha, beta,
                                            beta_is_zero),
        aligned=False,
        name="thin",
        filter=_is_sgemm_thin,   # min(m,n)<=64 and k>=256
    )

    # 不对齐 + 中等尺寸: 通用 fallback
    dispatch.add(
        lambda: _make_sgemm_nn_fallback_runner(A, lda, B, ldb, C, ldc,
                                                m, n, k, alpha, beta,
                                                beta_is_zero, grid),
        aligned=False,
        name="fallback",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k),
    )

    return dispatch


# 在 sgemm() 中的调用
def sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    ...
    aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            dispatch = _build_sgemm_nn_dispatch_table(
                A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid,
            )
            runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
            runner()
        elif ...
```

### 候选变体间的 filter 互斥关系

对于 `aligned=False`（不对齐场景），四个候选的 filter 覆盖关系：

| 候选 | filter | 目标场景 |
|------|--------|---------|
| `padded_k2` | `m>1024 and n>1024 and k>1024` | 大矩阵 → pad + TensorDescriptor |
| `thin` | `min(m,n)≤64 and k≥256 and grid<32` | 瘦矩阵 → Split-K + atomic_add |
| `fallback` | `not (m>1024 and n>1024 and k>1024)` | 中等尺寸 → 通用指针 kernel |
| `aligned_k2` | `aligned=True` (不同 aligned 取值) | 对齐矩阵 → TensorDescriptor + TF32 |

`thin` 和 `padded_k2` / `fallback` 的 filter 互斥（一个要求 min(m,n)≤64，另一个要求 m,n>1024），确保任意给定形状 `aligned=False` 时最多只有 2 个候选（且多数时候仅 1 个），避免不必要的 autotune。

---

## 运行时行为

### 首次调用新形状

```
lookup_and_build(m=4095, n=4095, k=4095, aligned=False)
  → entries = [padded_k2]  (仅 padded_k2 满足 filter)
  → len(entries)==1 → 直接返回 padded_k2 runner
  → 无 autotune, 无 DB I/O
```

```
lookup_and_build(m=1023, n=1023, k=1023, aligned=False)
  → entries = [fallback]  (仅 fallback 满足 filter)
  → len(entries)==1 → 直接返回 fallback runner
```

### 再次调用 (缓存命中)

```
lookup_and_build(m=4095, n=4095, k=4095, aligned=False)
  → 新的 SizeAutoDispatch 实例
  → entries = [padded_k2]
  → len(entries)==1 → 直接返回  ← 每次都是 O(1)
```

### 多个候选时的缓存命中

假设 filter 设计导致 `len(entries) >= 2`:

```
# 第一次
lookup_and_build(m=X, n=X, k=X, aligned=False)
  → entries = [candidate_a, candidate_b]
  → 内存缓存 miss → DB miss → autotune → candidate_a 胜出
  → 写入内存缓存 + DB

# 后续调用（同一进程）
lookup_and_build(m=X, n=X, k=X, aligned=False)
  → entries = [candidate_a, candidate_b]
  → 内存缓存命中 "candidate_a" → 直接返回  ← O(1) dict lookup
```

---

## 测试与验证

在 benchmark 环境中验证 dispatch 行为:

```bash
# 查看 autotune 日志
TRITON_PRINT_AUTOTUNING=1 pytest test_gemm_perf.py::test_perf_sgemm_nn -v -s

# 跳过 correctness check，仅测量 kernel 性能
pytest test_gemm_perf.py::test_perf_sgemm_nn -v -s --level core --skip_correctness
```

### 性能验证要点

1. **首个 shape 的 autotune 开销属于一次性成本**，不应计入 benchmark 计时
2. Benchmark 框架应在计时前通过 correctness check 或 warmup 调用预填充所有缓存
3. 若 benchmark 中仍观察到 autotune 日志输出，说明缓存未生效（检查 filter 互斥性、DB 文件权限）

---

## 常见问题

### Q: 为什么每次调用都要 new 一个新实例？

`SizeAutoDispatch` 的设计假设实例是轻量的（仅持有 filter/规则列表和 DB 连接引用），而 runner factory 闭包捕获了具体的 tensor 引用（A, B, C, lda, ldb ...），这些是每次调用都不同的。因此 dispatch 实例随 `sgemm()` 调用创建是合理的。性能保障由**模块级内存缓存**提供——新实例直接走缓存，无需 DB 查询或 autotune。

### Q: filter 函数可以捕获外部状态吗？

不可以。filter 函数的输入参数已经足够判断（m, n, k, aligned）。不要在 filter 中引入额外状态，否则会导致无法稳定复现的问题。

### Q: 如何清除 autotune 缓存？

```bash
# 清除进程内内存缓存（重启进程即可）
# 清除 DB 持久化缓存
rm ~/.cache/flag_blas/TunedConfig_*.db
```

### Q: 修改 filter 逻辑后是否需要清除缓存？

需要。旧的 autotune 结果可能与新逻辑不兼容。`lookup_and_build` 会检测到缓存中的候选名不再匹配任何 entry 并自动清除内存中的 stale entry，但 DB 中的旧记录需要手动清理或忽略（下次 autotune 会覆盖）。

---

## API 速查表

| 类/函数 | 用途 |
|---------|------|
| `SizeAutoDispatch(table_name, build_key, model)` | 构造 dispatch 实例 |
| `dispatch.add(factory, *, aligned, name, filter)` | 注册候选 kernel 变体 |
| `dispatch.lookup_and_build(m, n, k, aligned, **extra)` | 查询并构建最优 runner |
| `KernelRunner(kernel, *args, **kwargs)` | 预绑定参数的 kernel 调用包装器 |
| `_SizeFilter` | 类型别名 `Callable[..., bool]` |

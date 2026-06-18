# StaticDispatch 静态调度表

[源码: `src/flag_blas/runtime/dispatch.py`](../src/flag_blas/runtime/dispatch.py)

`StaticDispatch` 是开发者维护的静态 kernel 调度表，将 shape 条件直接映射到预定的 kernel 工厂函数。**不进行 autotune、不 benchmark、不缓存** —— 条件按顺序求值，首次命中即返回。

---

## 适用场景

当**最优的 shape → kernel 映射已经明确**时（例如通过离线 benchmark、架构分析或领域启发式规则得出），使用 `StaticDispatch`。它提供：

- **可预测的性能** — 无运行时 autotune 开销
- **确定性行为** — 相同 shape 总是映射到相同 kernel
- **极简资源占用** — 无缓存文件、无 DB 读写

对比 `SizeAutoDispatch`，后者会在运行时对所有候选做 benchmark 并持久化最优选择。

---

## 核心概念

### Condition（条件函数）

签名：`(m: int, n: int, k: int, aligned: bool, **extra: Any) -> bool`

判断当前 shape 是否匹配该表条目的 callable。条件**按表中顺序**求值，第一个返回 `True` 的条目胜出。

### Factory（工厂函数，双层 Lambda）

签名：`() -> Callable[[], None]`

零参数 callable，返回一个 **runner**。runner 本身也是零参数 callable，调用时执行 kernel。

**关键要点**：当搭配 `@libentry()` 装饰的 Triton kernel 使用时，factory 必须使用**双层 lambda** 包装：

```python
# 正确：双层 lambda
lambda: lambda: kernel_fn[grid](arg1, arg2, ...)
#         ^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^
#         factory         runner（延迟到 runner() 时执行）

# 错误：单层 lambda —— kernel 会在 factory() 时立即执行
lambda: kernel_fn[grid](arg1, arg2, ...)
```

**原因**：`kernel_fn[grid](args...)` 调用的是 `LibEntry.run()`，它会立即启动 kernel 并返回 `(kernel, constexprs)` 元组，而非可调用的 runner。内层 `lambda:` 将执行延迟到 `runner()` 调用时。

---

## 架构

与 `SizeAutoDispatch` 的多级缓存架构不同，`StaticDispatch` 设计极简：

```
lookup_and_build(m, n, k, aligned, **extra)
  │
  ├─ 条目 1: condition(m,n,k,aligned)? ─── True → factory() → runner
  ├─ 条目 2: condition(m,n,k,aligned)? ─── True → factory() → runner
  ├─ ...
  └─ 无匹配 → 抛出 ValueError
```

- **无过滤逻辑** — 条件内联编码所有匹配规则（无需单独的 `aligned`/`filter` 参数）
- **无缓存** — 每次调用重新求值条件（极其廉价，仅布尔逻辑运算）
- **无匹配即抛错** — 如果没有任何条目匹配，抛出 `ValueError`（最后一条应为兜底条目）

---

## API

### 构造函数

```python
dispatch = StaticDispatch(table)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `table` | `List[Tuple[Condition, Factory]]` | `(condition, factory)` 对的有序列表。**最后一条必须是兜底条目**（condition 始终返回 `True`）。 |

### lookup_and_build()

```python
runner = dispatch.lookup_and_build(m, n, k, aligned, **extra)
runner()
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `m` | `int` | M 维度 |
| `n` | `int` | N 维度 |
| `k` | `int` | K 维度 |
| `aligned` | `bool` | 输入是否内存对齐 |
| `**extra` | — | 传递给每个 condition 的额外关键字参数 |

**返回值**：`Callable[[], None]` — 零参数 runner，调用即执行选中的 kernel。

**异常**：如果没有任何条目匹配，抛出 `ValueError`（有兜底条目时不应发生）。

---

## 实战示例：hgemm NN

来自 [hopper/ops/gemm.py](../src/flag_blas/runtime/backend/_nvidia/hopper/ops/gemm.py) 中 `hgemm` 函数的 NN 分支：

```python
from flag_blas.runtime.dispatch import StaticDispatch
from triton.tools.tensor_descriptor import TensorDescriptor

dispatch = StaticDispatch([
    # ── 优先级 1（最高）──────────────────────────────────────────
    # Skinny 矩阵 + 对齐 + 大尺寸。
    # 使用 kernel4，搭配 TensorDescriptor 和硬编码最优 config
    # （无需 autotune —— 此 config 已证明是该 shape 类型的最优解）。
    (
        lambda m, n, k, aligned, **_kw:
            aligned and (m * n > 2048 * 2048) and min(m, n) >= 64
            and ((m >= 16384 and max(n, k) <= 2048)
                 or (n >= 16384 and max(m, k) <= 2048)),
        lambda: lambda: _hgemm_nn_kernel4[(
            triton.cdiv(m, 128) * triton.cdiv(n, 256),
        )](
            TensorDescriptor(
                base=A, shape=[m, k], strides=[lda, 1],
                block_shape=[128, 64],
            ),
            TensorDescriptor(
                base=B, shape=[k, n], strides=[ldb, 1],
                block_shape=[64, 256],
            ),
            TensorDescriptor(
                base=C, shape=[m, n], strides=[ldc, 1],
                block_shape=[128, 256],
            ),
            alpha, beta, m, n, k, beta_is_zero,
            BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_M=8,
            num_stages=4, num_warps=8, num_ctas=1,
        ),
    ),

    # ── 优先级 2 ─────────────────────────────────────────────────
    # 对齐 + 大尺寸（m×n > 4M，min 维度 ≥ 64）。
    # 使用 kernel3，搭配 TensorDescriptor 和 autotuned configs
    # （其 @libtuner 装饰器在运行时选择最优 BLOCK_M/N/K 等参数）。
    (
        lambda m, n, k, aligned, **_kw:
            aligned and (m * n > 2048 * 2048) and min(m, n) >= 64,
        lambda: lambda: _hgemm_nn_kernel3[grid](
            A, B, C, alpha, beta, m, n, k,
            lda, ldb, ldc, beta_is_zero,
        ),
    ),

    # ── 优先级 3 ─────────────────────────────────────────────────
    # 对齐 + 小/中等尺寸（max ≤ 1024）。
    # 使用 kernel2，搭配 block_ptr 和 autotuned configs。
    (
        lambda m, n, k, aligned, **_kw:
            aligned and max(m, n) <= 1024,
        lambda: lambda: _hgemm_nn_kernel2[grid](
            A, B, C, alpha, beta, m, n, k,
            lda, ldb, ldc, beta_is_zero,
        ),
    ),

    # ── 优先级 4（兜底）──────────────────────────────────────────
    # 其余所有情况：未对齐，或中等/大尺寸但未被上述条目覆盖。
    # 使用原始 level3 kernel，基于指针访问。
    (
        lambda **_kw: True,
        lambda: lambda: _hgemm_nn_kernel[grid](
            A, B, C, alpha, beta, m, n, k,
            lda, ldb, ldc, beta_is_zero,
        ),
    ),
])

runner = dispatch.lookup_and_build(m, n, k, aligned)
runner()
```

### 调度逻辑总结

| 优先级 | 条件 | Kernel |
|--------|------|--------|
| 1 | 对齐 + 大尺寸 + skinny（单维 ≥ 16384 其他 ≤ 2048） | `kernel4` — TensorDescriptor，硬编码 config |
| 2 | 对齐 + 大尺寸（m×n > 2048²，min ≥ 64） | `kernel3` — TensorDescriptor，autotuned |
| 3 | 对齐 + 小尺寸（max ≤ 1024） | `kernel2` — block_ptr，autotuned |
| 4 | 其余所有 | `kernel` — 原始指针实现，autotuned |

### Kernel 变体特征

| Kernel | 来源 | 数据访问 | Config 策略 | 最佳场景 |
|--------|------|---------|-------------|----------|
| `_hgemm_nn_kernel4` | Hopper（本地） | `TensorDescriptor.load/store` + `int32` 偏移 | 硬编码 `(128,256,64,8,4,8,1)` | Skinny 大矩阵 |
| `_hgemm_nn_kernel3` | Hopper（本地） | `TensorDescriptor.load/store` | `@libtuner("hgemm_nn2")` | 常规大矩阵 |
| `_hgemm_nn_kernel2` | Hopper（本地） | `tl.make_block_ptr` + `tl.advance` | `@libtuner("hgemm_nn")` | 小/中等对齐 |
| `_hgemm_nn_kernel` | Level3（导入） | 原始指针 + `offs_{am,bn,k}` + mask 逻辑 | `@libtuner("hgemm_nn")` | 其余情况 |

---

## 设计原则

1. **无 autotune**：表由人工维护；无运行时 benchmark。
2. **顺序匹配**：条件自上而下求值；首次 `True` 即胜出。
3. **必须兜底**：最后一条必须匹配所有 shape（防止 `ValueError`）。
4. **条件互斥**：条目间不应重叠以保证行为可预测。高优先级条目应有更具体的条件。
5. **双层 lambda 工厂**：搭配 `@libentry()` 装饰的 Triton kernel 时必须使用。内层 `lambda:` 将 `LibEntry.run()` 延迟到 `runner()` 时。

---

## 与 SizeAutoDispatch 对比

| | `SizeAutoDispatch` | `StaticDispatch` |
|---|---|---|
| **选择方式** | Autotune（benchmark 全部 → 选最快） | 静态条件（首次命中即返回） |
| **缓存** | 内存 + SQLite DB | 无 |
| **持久化** | 跨进程，SQLite | 无 |
| **首次调用** | benchmark 开销（秒级） | 瞬时（仅条件求值） |
| **后续调用** | ~缓存查找 | 瞬时（同首次调用） |
| **表构建** | 每个变体 `add()` | 构造函数一次性传入列表 |
| **失败处理** | 降级到第一个候选 | `ValueError`（兜底条目规避） |
| **最佳用途** | 最优映射未知 | 最优映射已知 |

---

## 构建 StaticDispatch 表

### 步骤

1. **列出 kernel 变体**。列出所有能处理该运算的 kernel，注明各自优势。

2. **编写条件函数**。为每个变体定义 lambda，在其表现最优的 shape 上返回 `True`。

3. **按优先级排序**。最具体的条件（最窄范围）放在最前面，最宽泛的放在最后。

4. **添加兜底条目**。最后一条必须匹配所有情况（`lambda **_kw: True`）。

5. **测试边界**。确保处于条件边界的 shape 路由到预期的 kernel。开发时可用日志或调试打印来验证。

### 常见模式

**按维度分优先级**：
```python
[
    (lambda m, *_kw, **__: m > 8192,  factory_a),   # 超大
    (lambda m, *_kw, **__: m > 1024,  factory_b),   # 大
    (lambda m, *_kw, **__: m > 256,   factory_c),   # 中等
    (lambda **_kw: True,              factory_d),   # 小
]
```

**按对齐分优先级**：
```python
[
    (lambda aligned, **_kw: aligned and is_large(**kw), aligned_large_factory),
    (lambda aligned, **_kw: aligned,                    aligned_small_factory),
    (lambda **_kw: True,                                unaligned_factory),
]
```

**维度 + 对齐组合**（如 hgemm_nn）：
```python
[
    (lambda aligned, m, n, k, **_kw:
         aligned and meets_criteria_A(m, n, k),
     factory_a),
    (lambda aligned, m, n, k, **_kw:
         aligned and meets_criteria_B(m, n, k),
     factory_b),
    (lambda **_kw: True,
     fallback_factory),
]
```

---

## 辅助类：KernelRunner

```python
class KernelRunner:
    def __init__(self, kernel: Callable, *args, **kwargs): ...
    def __call__(self):
        return self._kernel(*self._args, **self._kwargs)
```

将 kernel 函数与其参数绑定的简单可调用包装器。适用于不需要 autotune、只需固定实现的场景：

```python
runner = KernelRunner(my_kernel_fn, A, B, C, alpha=1.0)
runner()  # 等价于 my_kernel_fn(A, B, C, alpha=1.0)
```

---

## 常见问题

### Q: 什么时候应该用 StaticDispatch 而不是 SizeAutoDispatch？

当每个 shape 对应的最优 kernel **已知且稳定**时用 `StaticDispatch`。当最优选择依赖运行时因素（微架构特性、驱动版本等）且你希望系统自动发现时，用 `SizeAutoDispatch`。

### Q: 如果条件 lambda 抛出异常会怎样？

异常会向上传播不被捕获。保持条件 lambda 尽量简单（仅布尔算术，无 IO 或 tensor 操作）。

### Q: 能在同一个算子里混用 StaticDispatch 和 SizeAutoDispatch 吗？

可以。例如，用 `StaticDispatch` 处理已知最优的 shape 类型，用 `SizeAutoDispatch` 处理其余情况。只需确保从正确的 dispatch 返回 runner 即可。

### Q: 为什么 @libentry kernel 需要双层 lambda？

`@libentry()` 将 Triton `JITFunction` 包装为 `LibEntry` 对象。调用 `entry[grid](args...)` 会触发 `LibEntry.run()`，该方法编译（如果需要）、启动 kernel 并返回 `(kernel_obj, constexprs)`。内层 `lambda:` 将这个过程包装为可调用的 runner 而不立即执行 kernel。

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

# StaticDispatch 静态调度表

[源码: `src/flag_blas/runtime/dispatch.py`](../src/flag_blas/runtime/dispatch.py)

`StaticDispatch` 是开发者维护的静态 kernel 调度表，将 shape 条件直接映射到预定的 kernel 工厂函数。**不进行 autotune、不 benchmark、不缓存** —— 条件按顺序求值，首次命中即返回。

调度表设计为模块级别创建一次，跨调用复用。每次调用的变化数据（tensor、标量等）通过 `context` 字典传入 `lookup_and_build()`，工厂函数无需通过闭包捕获变量。

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

### Factory（工厂函数）

签名：`(context_key1=..., context_key2=..., ...) -> Callable[[], None]`

一个命名函数（推荐使用命名函数而非 lambda），通过关键字参数接收每次调用的变化数据，返回一个 **runner** —— 执行 kernel 的零参数可调用对象。参数通过 `lookup_and_build()` 中的 `context` 字典传入。

当调度表在模块级别创建时，工厂函数只是对命名函数的纯引用 —— 没有闭包，没有每次调用重新构建的 lambda。每次调用的数据通过 `context` 流入。

**关键要点**：当搭配 `@libentry()` 装饰的 Triton kernel 使用时，工厂必须返回一个被 **lambda** 包装的 kernel 调用：

```python
# 模块级别 —— 定义一次：
def build_my_kernel(A, B, C, alpha, beta, m, n, k, lda, ldb, ldc):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _my_kernel[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
    )

# 每次调用 —— context 携带变化数据：
runner = dispatch.lookup_and_build(
    m, n, k, aligned,
    context=dict(A=A, B=B, C=C, m=m, n=n, k=k,
                 lda=lda, ldb=ldb, ldc=ldc,
                 alpha=alpha, beta=beta),
)
runner()
```

**原因**：`kernel_fn[grid](args...)` 调用的是 `LibEntry.run()`，它会立即启动 kernel 并返回 `(kernel_obj, constexprs)` 元组，而非可调用的 runner。`lambda:` 将执行延迟到 `runner()` 调用时。

---

## 架构

与 `SizeAutoDispatch` 的多级缓存架构不同，`StaticDispatch` 设计极简：

```
lookup_and_build(m, n, k, aligned, *, context, **extra)
  │
  ├─ 条目 1: condition(m,n,k,aligned)? ─── True → factory(**context) → runner
  ├─ 条目 2: condition(m,n,k,aligned)? ─── True → factory(**context) → runner
  ├─ ...
  └─ 无匹配 → 抛出 ValueError
```

- **无过滤逻辑** — 条件内联编码所有匹配规则（无需单独的 `aligned`/`filter` 参数）
- **无缓存** — 每次调用重新求值条件（极其廉价，仅布尔逻辑运算）
- **`context` 字典** — 将每次调用的变化数据（tensor、标量等）传入工厂函数，使调度表本身可以驻留在模块级别
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
runner = dispatch.lookup_and_build(m, n, k, aligned, *, context=None, **extra)
runner()
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `m` | `int` | M 维度 |
| `n` | `int` | N 维度 |
| `k` | `int` | K 维度 |
| `aligned` | `bool` | 输入是否内存对齐 |
| `context` | `dict` 或 `None` | 每次调用的变化数据（tensor、标量等），以关键字参数形式传入匹配的工厂函数。为 `None` 时工厂函数无参调用。 |
| `**extra` | — | 传递给每个 condition 的额外关键字参数 |

**返回值**：`Callable[[], None]` — 零参数 runner，调用即执行选中的 kernel。

**异常**：如果没有任何条目匹配，抛出 `ValueError`（有兜底条目时不应发生）。

---

## 实战示例：hgemm NN

来自 [hopper/ops/gemm.py](../src/flag_blas/runtime/backend/_nvidia/hopper/ops/gemm.py) 中 `hgemm` 函数的 NN 分支。

### 模块级别 —— 定义一次

```python
from flag_blas.runtime.dispatch import StaticDispatch
from triton.tools.tensor_descriptor import TensorDescriptor

# ── 条件谓词（命名函数，非 lambda）───────────────────────────────

def _hgemm_nn_is_skinny_aligned_large(m, n, k, aligned, **_kw):
    return (
        aligned
        and (m * n > 2048 * 2048)
        and min(m, n) >= 64
        and (
            (m >= 16384 and max(n, k) <= 2048)
            or (n >= 16384 and max(m, k) <= 2048)
        )
    )

def _hgemm_nn_is_aligned_large(m, n, k, aligned, **_kw):
    return aligned and (m * n > 2048 * 2048) and min(m, n) >= 64

def _hgemm_nn_is_aligned_small(m, n, k, aligned, **_kw):
    return aligned and max(m, n) <= 1024

def _hgemm_nn_is_default(**_kw):
    return True

# ── 工厂函数（接收 context 字典的键作为关键字参数）────────────────

def _hgemm_nn_build_kernel4(A, B, C, m, n, k, lda, ldb, ldc,
                            alpha, beta, beta_is_zero):
    return lambda: _hgemm_nn_kernel4[(
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
    )

def _hgemm_nn_build_kernel3(A, B, C, m, n, k, lda, ldb, ldc,
                            alpha, beta, beta_is_zero):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nn_kernel3[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
    )

def _hgemm_nn_build_kernel2(A, B, C, m, n, k, lda, ldb, ldc,
                            alpha, beta, beta_is_zero):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nn_kernel2[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
    )

def _hgemm_nn_build_kernel(A, B, C, m, n, k, lda, ldb, ldc,
                           alpha, beta, beta_is_zero):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nn_kernel[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
    )

_HGEMM_NN_DISPATCH = StaticDispatch([
    # skinny + 对齐大尺寸 → kernel4（TensorDescriptor，硬编码 config）
    (_hgemm_nn_is_skinny_aligned_large, _hgemm_nn_build_kernel4),
    # 对齐大尺寸 → kernel3（TensorDescriptor，autotuned config）
    (_hgemm_nn_is_aligned_large, _hgemm_nn_build_kernel3),
    # 对齐小尺寸 → kernel2（block_ptr）
    (_hgemm_nn_is_aligned_small, _hgemm_nn_build_kernel2),
    # 默认 → kernel（原始实现）
    (_hgemm_nn_is_default, _hgemm_nn_build_kernel),
])
```

### 每次调用 —— 在 `hgemm()` 内部

```python
def hgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    # ... 参数校验 ...
    beta_is_zero = beta == 0.0
    aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            runner = _HGEMM_NN_DISPATCH.lookup_and_build(
                m, n, k, aligned,
                context=dict(
                    A=A, B=B, C=C, m=m, n=n, k=k,
                    lda=lda, ldb=ldb, ldc=ldc,
                    alpha=alpha, beta=beta, beta_is_zero=beta_is_zero,
                ),
            )
            runner()
        # ... 其他 transa/transb 分支 ...
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
5. **使用命名函数而非 lambda**：条件和工厂应为模块级别的命名函数，在 `StaticDispatch` 表中按名称引用。每次调用的变化数据（tensor、标量）通过 `context` 字典传入 `lookup_and_build()`，避免每次调用重新创建闭包。

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

**按维度分优先级**（模块级命名函数）：
```python
def is_very_large(m, **_kw):
    return m > 8192

def is_large(m, **_kw):
    return m > 1024

def is_medium(m, **_kw):
    return m > 256

def is_default(**_kw):
    return True

_DISPATCH = StaticDispatch([
    (is_very_large, build_kernel_a),
    (is_large,      build_kernel_b),
    (is_medium,     build_kernel_c),
    (is_default,    build_kernel_d),
])
```

**按对齐分优先级**：
```python
def is_aligned_large(aligned, m, n, k, **_kw):
    return aligned and (m * n > 2048 * 2048)

def is_aligned_only(aligned, **_kw):
    return aligned

def is_default(**_kw):
    return True

_DISPATCH = StaticDispatch([
    (is_aligned_large, build_aligned_large),
    (is_aligned_only,  build_aligned),
    (is_default,       build_fallback),
])
```

**维度 + 对齐组合**（如 hgemm_nn）：
```python
def is_skinny_aligned_large(m, n, k, aligned, **_kw):
    return (aligned and (m * n > 2048 * 2048) and min(m, n) >= 64
            and ((m >= 16384 and max(n, k) <= 2048)
                 or (n >= 16384 and max(m, k) <= 2048)))

def is_aligned_large(m, n, k, aligned, **_kw):
    return aligned and (m * n > 2048 * 2048) and min(m, n) >= 64

def is_aligned_small(m, n, k, aligned, **_kw):
    return aligned and max(m, n) <= 1024

def is_default(**_kw):
    return True

_DISPATCH = StaticDispatch([
    (is_skinny_aligned_large, build_kernel4),
    (is_aligned_large,        build_kernel3),
    (is_aligned_small,        build_kernel2),
    (is_default,              build_kernel),
])
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

### Q: 为什么需要双层 lambda？

`@libentry()` 将 Triton `JITFunction` 包装为 `LibEntry` 对象。调用 `entry[grid](args...)` 会触发 `LibEntry.run()`，该方法编译（如果需要）、启动 kernel 并返回 `(kernel_obj, constexprs)`。返回的 `lambda:` 将这个过程包装为可调用的 runner 而不立即执行 kernel。

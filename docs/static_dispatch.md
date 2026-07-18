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

# StaticDispatch

[Source: `src/flag_blas/runtime/dispatch.py`](../src/flag_blas/runtime/dispatch.py)

`StaticDispatch` is a developer-maintained static dispatch table that maps shape conditions directly to pre-determined kernel factories. **No autotune, no benchmarking, no caching** — conditions are evaluated in order, and the first match wins immediately.

The dispatch table is designed to be created once at module level and reused across calls. Per-call varying data (tensors, scalars) is passed through a `context` dict, so factories don't need to capture variables via closures.

---

## When to Use

Use `StaticDispatch` when the **optimal shape → kernel mapping is already known** (e.g., through offline benchmarking, architectural analysis, or domain-specific heuristics). It provides:

- **Predictable performance** — no runtime autotune overhead
- **Deterministic behavior** — same shape always maps to the same kernel
- **Minimal footprint** — no cache files, no DB I/O

Contrast with `SizeAutoDispatch`, which benchmarks all candidates at runtime and persists the winner.

---

## Core Concepts

### Condition

Signature: `(m: int, n: int, k: int, aligned: bool, **extra: Any) -> bool`

A callable that determines whether the current shape matches this table entry. Conditions are evaluated **in table order**; the first that returns `True` wins.

### Factory

Signature: `(context_key1=..., context_key2=..., ...) -> Callable[[], None]`

A named function (recommended over lambdas) that accepts per-call varying arguments via keyword arguments and returns a **runner** — a zero-arg callable that executes the kernel. The arguments are passed from the `context` dict in `lookup_and_build()`.

When the dispatch table is created at module level, factories are pure references to named functions — no closures, no lambdas rebuilt on every call. The per-call data flows in through `context`.

**Critical**: When used with `@libentry()`-decorated Triton kernels, the factory must return the kernel call wrapped in a **lambda**:

```python
# Module level — defined once:
def build_my_kernel(A, B, C, alpha, beta, m, n, k, lda, ldb, ldc):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _my_kernel[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
    )

# Per-call — context carries the varying data:
runner = dispatch.lookup_and_build(
    m, n, k, aligned,
    context=dict(A=A, B=B, C=C, m=m, n=n, k=k,
                 lda=lda, ldb=ldb, ldc=ldc,
                 alpha=alpha, beta=beta),
)
runner()
```

**Why lambda**: `kernel_fn[grid](args...)` calls `LibEntry.run()`, which launches the kernel immediately and returns a `(kernel, constexprs)` tuple — not a callable runner. The `lambda:` defers execution to `runner()` time.

---

## Architecture

Unlike `SizeAutoDispatch` with its multi-tier cache, `StaticDispatch` has a minimal design:

```
lookup_and_build(m, n, k, aligned, *, context, **extra)
  │
  ├─ Entry 1: condition(m,n,k,aligned)? ─── True → factory(**context) → runner
  ├─ Entry 2: condition(m,n,k,aligned)? ─── True → factory(**context) → runner
  ├─ ...
  └─ No match → raise ValueError
```

- **No filtering logic** — conditions encode all matching criteria inline (no separate `aligned`/`filter` params)
- **No cache** — every call re-evaluates conditions (cheap, just boolean logic)
- **`context` dict** — passes per-call varying data (tensors, scalars) to factories, so the dispatch table itself can live at module level
- **Throw on miss** — if no entry matches, raises `ValueError` (the last entry should always be a catch-all)

---

## API

### Constructor

```python
dispatch = StaticDispatch(table)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `table` | `List[Tuple[Condition, Factory]]` | Ordered list of `(condition, factory)` pairs. The **last entry must be a catch-all** (condition always returns `True`). |

### lookup_and_build()

```python
runner = dispatch.lookup_and_build(m, n, k, aligned, *, context=None, **extra)
runner()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | `int` | M dimension |
| `n` | `int` | N dimension |
| `k` | `int` | K dimension |
| `aligned` | `bool` | Whether inputs are memory-aligned |
| `context` | `dict` or `None` | Per-call varying data (tensors, scalars, etc.) passed as keyword arguments to the matched factory. When `None`, factories are called with no arguments. |
| `**extra` | — | Additional keyword arguments passed to each condition |

**Returns**: `Callable[[], None]` — zero-arg runner; calling it executes the selected kernel.

**Raises**: `ValueError` if no entry matches (should never happen with a proper catch-all).

---

## Real-World Example: hgemm NN

This example comes from [hopper/ops/gemm.py](../src/flag_blas/runtime/backend/_nvidia/hopper/ops/gemm.py), the `hgemm` function's NN branch.

### Module level — defined once

```python
from flag_blas.runtime.dispatch import StaticDispatch
from triton.tools.tensor_descriptor import TensorDescriptor

# ── Condition predicates (named functions, not lambdas) ──────────

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

# ── Factory functions (accept context dict keys as kwargs) ───────

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
    # skinny + aligned large → kernel4 (TensorDescriptor, hardcoded config)
    (_hgemm_nn_is_skinny_aligned_large, _hgemm_nn_build_kernel4),
    # aligned large → kernel3 (TensorDescriptor, autotuned config)
    (_hgemm_nn_is_aligned_large, _hgemm_nn_build_kernel3),
    # aligned small → kernel2 (block_ptr)
    (_hgemm_nn_is_aligned_small, _hgemm_nn_build_kernel2),
    # default → kernel (original)
    (_hgemm_nn_is_default, _hgemm_nn_build_kernel),
])
```

### Per-call — inside `hgemm()`

```python
def hgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    # ... validation ...
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
        # ... other transa/transb branches ...
```

### Dispatch Logic Summary

| Priority | Condition | Kernel |
|----------|-----------|--------|
| 1 | aligned + large + skinny (one dim ≥ 16384, others ≤ 2048) | `kernel4` — TensorDescriptor, hardcoded config |
| 2 | aligned + large (m×n > 2048², min ≥ 64) | `kernel3` — TensorDescriptor, autotuned |
| 3 | aligned + small (max ≤ 1024) | `kernel2` — block_ptr, autotuned |
| 4 | everything else | `kernel` — original pointer-based, autotuned |

### Kernel Variant Characteristics

| Kernel | Source | Data Access | Config Strategy | Best For |
|--------|--------|-------------|-----------------|----------|
| `_hgemm_nn_kernel4` | Hopper (local) | `TensorDescriptor.load/store` + `int32` offsets | Hardcoded `(128,256,64,8,4,8,1)` | Skinny large matrices |
| `_hgemm_nn_kernel3` | Hopper (local) | `TensorDescriptor.load/store` | `@libtuner("hgemm_nn2")` | General large matrices |
| `_hgemm_nn_kernel2` | Hopper (local) | `tl.make_block_ptr` + `tl.advance` | `@libtuner("hgemm_nn")` | Small/moderate aligned |
| `_hgemm_nn_kernel` | Level3 (imported) | Raw pointers + `offs_{am,bn,k}` + mask logic | `@libtuner("hgemm_nn")` | Everything else |

---

## Design Principles

1. **No autotune**: The table is human-curated; no runtime benchmarking.
2. **Ordered matching**: Conditions evaluated top-to-bottom; first `True` wins.
3. **Catch-all required**: The last entry must match any shape (prevent `ValueError`).
4. **Mutually exclusive conditions**: Entries should not overlap to make behavior predictable. Higher-priority entries should have more specific conditions.
5. **Named functions, not lambdas in the table**: Conditions and factories should be module-level named functions referenced by name in the `StaticDispatch` table. Per-call varying data (tensors, scalars) is passed via the `context` dict to `lookup_and_build()`. This avoids recreating closures on every call.

---

## Comparison with SizeAutoDispatch

| | `SizeAutoDispatch` | `StaticDispatch` |
|---|---|---|
| **Selection** | Autotune (benchmark all → pick fastest) | Static conditions (first match wins) |
| **Cache** | In-memory + SQLite DB | None |
| **Persistence** | Cross-process via SQLite | None |
| **First call** | Benchmark cost (seconds) | Instant (condition evaluation only) |
| **Subsequent calls** | ~cache lookup | Instant (same as first call) |
| **Table building** | `add()` per variant | Single list in constructor |
| **Failure mode** | Fallback to first candidate | `ValueError` (mitigated by catch-all) |
| **Best for** | Unknown optimal mapping | Known optimal mapping |

---

## Building a StaticDispatch Table

### Step-by-step

1. **Identify kernel variants**. List every kernel that could handle this operation, along with its strengths.

2. **Write condition functions**. For each variant, define a lambda that returns `True` for the shapes where it shines.

3. **Order by priority**. Put the most specific (narrowest condition) first, broadest last.

4. **Add a catch-all**. The final entry must match everything (`lambda **_kw: True`).

5. **Test edge cases**. Ensure shapes at condition boundaries route to the intended kernel. Use logging or debug prints during development.

### Common Patterns

**Priority by dimension** (module-level named functions):
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

**Priority by alignment**:
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

**Combined dimensions + alignment** (as in hgemm_nn):
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

## Helper: KernelRunner

```python
class KernelRunner:
    def __init__(self, kernel: Callable, *args, **kwargs): ...
    def __call__(self):
        return self._kernel(*self._args, **self._kwargs)
```

A simple callable wrapper that binds a kernel function with its arguments. Useful when you don't need autotune, just a fixed implementation:

```python
runner = KernelRunner(my_kernel_fn, A, B, C, alpha=1.0)
runner()  # equivalent to my_kernel_fn(A, B, C, alpha=1.0)
```

---

## FAQ

### Q: When should I use StaticDispatch vs SizeAutoDispatch?

Use `StaticDispatch` when the optimal kernel for each shape is **already known and stable**. Use `SizeAutoDispatch` when the best choice depends on runtime factors (micro-architecture quirks, driver versions, etc.) and you want the system to figure it out automatically.

### Q: What happens if a condition lambda raises an exception?

The exception propagates uncaught. Keep condition lambdas simple (boolean arithmetic only, no I/O or tensor operations).

### Q: Can I mix StaticDispatch and SizeAutoDispatch in the same operator?

Yes. For example, use `StaticDispatch` for well-understood shape classes and `SizeAutoDispatch` for the remainder. Just ensure you return the runner from the appropriate dispatch.

### Q: Why return a lambda from the factory?

`@libentry()` wraps a Triton `JITFunction` in a `LibEntry` object. Calling `entry[grid](args...)` triggers `LibEntry.run()`, which compiles (if needed), launches the kernel, and returns `(kernel_obj, constexprs)`. The returned `lambda:` wraps this into a callable runner without executing the kernel immediately.

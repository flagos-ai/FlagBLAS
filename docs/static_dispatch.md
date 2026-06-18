# StaticDispatch

[Source: `src/flag_blas/runtime/dispatch.py`](../src/flag_blas/runtime/dispatch.py)

`StaticDispatch` is a developer-maintained static dispatch table that maps shape conditions directly to pre-determined kernel factories. **No autotune, no benchmarking, no caching** — conditions are evaluated in order, and the first match wins immediately.

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

### Factory (Double-Lambda)

Signature: `() -> Callable[[], None]`

A zero-arg callable that returns a **runner**. The runner is itself a zero-arg callable that executes the kernel.

**Critical**: When used with `@libentry()`-decorated Triton kernels, the factory must use a **double-lambda** wrapper:

```python
# Correct: double lambda
lambda: lambda: kernel_fn[grid](arg1, arg2, ...)
#         ^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^
#         factory         runner (deferred to runner())

# Wrong: single lambda — kernel executes immediately in factory()
lambda: kernel_fn[grid](arg1, arg2, ...)
```

**Why**: `kernel_fn[grid](args...)` calls `LibEntry.run()`, which launches the kernel immediately and returns a `(kernel, constexprs)` tuple — not a callable runner. The extra `lambda:` defers execution to `runner()` time.

---

## Architecture

Unlike `SizeAutoDispatch` with its multi-tier cache, `StaticDispatch` has a minimal design:

```
lookup_and_build(m, n, k, aligned, **extra)
  │
  ├─ Entry 1: condition(m,n,k,aligned)? ─── True → factory() → runner
  ├─ Entry 2: condition(m,n,k,aligned)? ─── True → factory() → runner
  ├─ ...
  └─ No match → raise ValueError
```

- **No filtering logic** — conditions encode all matching criteria inline (no separate `aligned`/`filter` params)
- **No cache** — every call re-evaluates conditions (cheap, just boolean logic)
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
runner = dispatch.lookup_and_build(m, n, k, aligned, **extra)
runner()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | `int` | M dimension |
| `n` | `int` | N dimension |
| `k` | `int` | K dimension |
| `aligned` | `bool` | Whether inputs are memory-aligned |
| `**extra` | — | Additional keyword arguments passed to each condition |

**Returns**: `Callable[[], None]` — zero-arg runner; calling it executes the selected kernel.

**Raises**: `ValueError` if no entry matches (should never happen with a proper catch-all).

---

## Real-World Example: hgemm NN

This example comes from [hopper/ops/gemm.py](../src/flag_blas/runtime/backend/_nvidia/hopper/ops/gemm.py), the `hgemm` function's NN branch:

```python
from flag_blas.runtime.dispatch import StaticDispatch
from triton.tools.tensor_descriptor import TensorDescriptor

dispatch = StaticDispatch([
    # ── Priority 1 (highest) ─────────────────────────────────────
    # Skinny matrix with aligned large dimensions.
    # Uses kernel4 with TensorDescriptor and hardcoded optimal config
    # (no autotune needed — config proven optimal for this shape class).
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

    # ── Priority 2 ───────────────────────────────────────────────
    # Aligned + large dimensions (m×n > 4M, min dim ≥ 64).
    # Uses kernel3 with TensorDescriptor and autotuned configs
    # (its @libtuner picks the best BLOCK_M/N/K etc. at runtime).
    (
        lambda m, n, k, aligned, **_kw:
            aligned and (m * n > 2048 * 2048) and min(m, n) >= 64,
        lambda: lambda: _hgemm_nn_kernel3[grid](
            A, B, C, alpha, beta, m, n, k,
            lda, ldb, ldc, beta_is_zero,
        ),
    ),

    # ── Priority 3 ───────────────────────────────────────────────
    # Aligned + small/moderate dimensions (max ≤ 1024).
    # Uses kernel2 with block_ptr and autotuned configs.
    (
        lambda m, n, k, aligned, **_kw:
            aligned and max(m, n) <= 1024,
        lambda: lambda: _hgemm_nn_kernel2[grid](
            A, B, C, alpha, beta, m, n, k,
            lda, ldb, ldc, beta_is_zero,
        ),
    ),

    # ── Priority 4 (catch-all) ───────────────────────────────────
    # Everything else: unaligned or moderate/large but not covered above.
    # Uses the original level3 kernel with pointer-based access.
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
5. **Double-lambda factories**: Required when using `@libentry()`-decorated Triton kernels. The inner `lambda:` defers `LibEntry.run()` to `runner()` time.

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

**Priority by dimension**:
```python
[
    (lambda m, *_kw, **__: m > 8192,  factory_a),   # very large
    (lambda m, *_kw, **__: m > 1024,  factory_b),   # large
    (lambda m, *_kw, **__: m > 256,   factory_c),   # medium
    (lambda **_kw: True,              factory_d),   # small
]
```

**Priority by alignment**:
```python
[
    (lambda aligned, **_kw: aligned and is_large(**kw), aligned_large_factory),
    (lambda aligned, **_kw: aligned,                    aligned_small_factory),
    (lambda **_kw: True,                                unaligned_factory),
]
```

**Combined dimensions + alignment** (as in hgemm_nn):
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

### Q: Why double-lambda for @libentry kernels?

`@libentry()` wraps a Triton `JITFunction` in a `LibEntry` object. Calling `entry[grid](args...)` triggers `LibEntry.run()`, which compiles (if needed), launches the kernel, and returns `(kernel_obj, constexprs)`. The inner `lambda:` wraps this into a callable runner without executing the kernel.

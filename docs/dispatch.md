# FlagBLAS Dispatch Module

## Overview

The `dispatch` module (`src/flag_blas/runtime/dispatch.py`) provides a **size-based auto-tuning kernel variant selection** framework. It addresses a core problem: for a single mathematical operation (e.g., GEMM), multiple kernel implementation variants exist, each performing optimally under different matrix shapes and alignment conditions. `SizeAutoDispatch` benchmarks all applicable variants on actual GPU hardware when a new shape is first encountered, selects the fastest one, and caches the result in both an in-process memory cache and a persistent database, enabling zero-overhead lookups on subsequent calls.

## Use Cases

- An operator has **multiple kernel implementation variants** (e.g., alignment-optimized, padded, Split-K, generic fallback)
- Different variants perform best under **different shape and alignment conditions**
- The goal is to **auto-benchmark on first run, then select the optimal implementation with zero overhead** on subsequent runs
- Auto-tuning results should **persist across process restarts**

## Architecture

### Module Layout

```
dispatch.py
├── _autotune_result_cache    # Module-level in-process memory cache
├── _autotune_result_lock     # Thread-safety lock
├── _SizeFilter               # Type alias for size filter functions
├── SizeAutoDispatch          # Core auto-tuning dispatch class
└── KernelRunner              # Pre-bound kernel invocation wrapper
```

### Cache Hierarchy

`SizeAutoDispatch.lookup_and_build()` uses a **four-tier look-up** mechanism to find the optimal kernel variant:

```
                         ┌─────────────────────┐
                         │  lookup_and_build()  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  len(entries) == 1  │
                         │  Return immediately  │  ← O(1), zero overhead
                         └──────────┬──────────┘
                                    │ ≥ 2 candidates
                         ┌──────────▼──────────┐
                         │  In-memory cache     │  ← dict lookup, O(1)
                         │  (process-scoped,    │
                         │   cross-instance)    │
                         └──────────┬──────────┘
                                    │ miss
                         ┌──────────▼──────────┐
                         │  DB cache            │  ← SQLite I/O
                         │  (persists across    │
                         │   process restarts)  │
                         └──────────┬──────────┘
                                    │ miss
                         ┌──────────▼──────────┐
                         │  Auto-tune (bench)   │  ← GPU benchmark
                         │  Write to memory + DB│
                         └─────────────────────┘
```

#### Cache Tier Details

| Tier | Storage | Lifetime | Lookup Cost | Purpose |
|------|---------|----------|-------------|---------|
| `len(entries)==1` | None | Per call | O(1) | Single candidate, return directly |
| In-memory cache | Python dict | Process lifetime | O(1) | Solves "new instance on every call" problem |
| DB cache | SQLite file | Permanent | ~0.1 ms | Persistence across process restarts |
| Auto-tune | GPU | First encounter | Seconds | Actual GPU benchmarking to pick the winner |

**In-memory cache** (lines 29–30):

```python
_autotune_result_cache: Dict[Tuple[str, tuple], str] = {}
_autotune_result_lock = threading.Lock()
```

- Key: `(table_name, cache_key)` — `table_name` distinguishes different dispatches (e.g., `"sgemm_nn_variant"` vs `"hgemm_nn_variant"`), while `cache_key` encodes the problem shape
- Value: Name string of the winning candidate variant
- Module-level global variable, shared across all `SizeAutoDispatch` instances
- No persistence overhead; automatically cleared on process restart

---

## Core Class: SizeAutoDispatch

### Constructor Parameters

```python
SizeAutoDispatch(
    table_name: str,
    build_key: Callable[..., tuple],
    model: Optional[PersistantModel] = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_name` | `str` | DB table name for persisting the `(key → best_candidate)` mapping. Use unique names for different dispatches to avoid collisions |
| `build_key` | `(m, n, k, aligned, **extra) -> tuple` | Function that builds a unique cache key from problem dimensions |
| `model` | `PersistantModel \| None` | Database access instance. If `None`, only in-memory caching is used (no persistence) |

### add() — Register a Candidate Variant

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `factory` | `() -> Callable[[], None]` | **Zero-arg factory function** that returns a callable runner. Lazy construction: only the selected candidate's factory is invoked |
| `aligned` | `bool \| None` | Alignment condition. `True` = matches aligned inputs only, `False` = matches unaligned only, `None` = matches all |
| `name` | `str \| None` | Candidate name for DB storage and logging. Auto-generated as `"variant_0"`, `"variant_1"`, etc. if not specified |
| `filter` | `_SizeFilter \| None` | Size filter function, signature `(m, n, k, aligned, **extra) -> bool`. Returns `True` to include this candidate in the pool |

**Key Design: Lazy Runner Construction**

`add()` accepts a **factory function** rather than the runner itself because:

1. Creating a runner may involve expensive memory allocations (e.g., padding new tensors)
2. During auto-tuning, every candidate needs a runner, but **only the winner is actually used** past that point
3. On cache hits, factories for non-winning candidates are never called at all

**Filter Design Constraints:**

- Filters for the same `aligned` value should be **mutually exclusive**; overlapping filters cause multiple candidates to enter the pool simultaneously, triggering unnecessary auto-tuning
- Filter logic should depend **solely on shape dimensions**, not any external state
- There must be no "vacuum" where **all candidates are excluded** for a given shape

### lookup_and_build() — Look Up and Build the Optimal Runner

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | `int` | Matrix M dimension |
| `n` | `int` | Matrix N dimension |
| `k` | `int` | Matrix K dimension |
| `aligned` | `bool` | Whether the current inputs satisfy alignment constraints |
| `snapshot_tensor` | `Tensor \| None` | Snapshot tensor for state restoration during auto-tuning. When provided, its original value is restored before and after each candidate benchmark to ensure fair comparison |
| `**extra` | — | Additional parameters forwarded to `build_key` and `filter` |

**Return value**: A zero-arg callable runner; calling it executes the selected kernel.

**Execution Flow**:

1. Call `build_key(m, n, k, aligned, **extra)` to construct the cache key
2. Call `_get_entries(m, n, k, aligned, **extra)` to filter matching candidates
3. If only 1 candidate → return its `factory()` immediately
4. Check in-memory cache → return on hit
5. Check DB cache → return on hit, also promote to in-memory cache
6. Trigger `_autotune()` → benchmark to find the winner → write to both in-memory and DB caches

### _autotune() — Automatic Tuning

```python
def _autotune(
    self,
    cache_key: tuple,
    entries: List[_Entry],
    snapshot_tensor=None,
) -> Optional[Tuple[_Entry, Callable[[], None]]]
```

Performs **1 warm-up + 5 timed iterations** per candidate:

1. Back up `snapshot_tensor` (if provided)
2. Call `entry.factory()` to construct the runner (lazy, one-time)
3. Execute 1 warm-up + `torch.cuda.synchronize()`
4. Restore snapshot
5. Execute 5 timed iterations using CUDA Events for precise GPU time measurement
6. Restore snapshot before and after each timed iteration to eliminate cross-candidate state interference
7. Record the minimum elapsed time for each candidate
8. Return the `(entry, runner)` pair with the lowest timing

**Fault tolerance**: A single candidate crash does not affect other candidates; a warning is logged and the candidate is skipped. If all candidates fail, `None` is returned and the caller falls back to the first entry **without persisting**, so future calls will retry auto-tuning.

### Configurable Parameters

| Class Attribute | Default | Description |
|-----------------|---------|-------------|
| `_WARMUP_ITERS` | `1` | Number of warm-up launches per candidate |
| `_TIMING_ITERS` | `5` | Number of timed launches per candidate |
| `_CACHE_FLUSH_BYTES` | `16 * 1024 * 1024` | Dummy buffer size (bytes) for L2 cache eviction between candidates (16 MiB) |

---

## Helper Class: KernelRunner

```python
class KernelRunner:
    def __init__(self, kernel: Callable, *args, **kwargs):
        ...

    def __call__(self):
        return self._kernel(*self._args, **self._kwargs)
```

A callable wrapper that binds a kernel function with its arguments. Useful when no auto-tuning is needed and a fixed implementation suffices. Example:

```python
runner = KernelRunner(my_kernel_fn, A, B, C, alpha=1.0)
runner()  # Equivalent to my_kernel_fn(A, B, C, alpha=1.0)
```

---

## Usage

### Basic Pattern

```python
from flag_blas.runtime.dispatch import SizeAutoDispatch
from flag_blas.utils.libentry import libcache

# Step 1: Define size filter functions
def is_large(m, n, k, **extra):
    return m > 1024 and n > 1024 and k > 1024

def is_thin(m, n, k, **extra):
    return min(m, n) <= 64 and k >= 256

# Step 2: Define runner factory functions (each returns a zero-arg callable)
def make_aligned_runner(A, B, C, m, n, k, ...):
    def run():
        aligned_kernel(A, B, C, ...)
    return run

def make_fallback_runner(A, B, C, m, n, k, ...):
    def run():
        fallback_kernel(A, B, C, ...)
    return run

# Step 3: Build the dispatch table
def build_dispatch(A, B, C, m, n, k, ...):
    dispatch = SizeAutoDispatch(
        table_name="my_gemm_variant",                   # Unique table name
        build_key=lambda m, n, k, aligned, **kw: (m, n, k, int(aligned)),
        model=libcache.model,                           # Reuse FlagBLAS global DB
    )

    # Step 4: Register candidate variants
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

# Step 5: Use in the operator entry point
def my_gemm(A, B, C, m, n, k, ...):
    aligned = check_alignment(A, B, C)
    dispatch = build_dispatch(A, B, C, m, n, k, ...)
    runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
    runner()
```

### Full Example: Hopper sgemm NN Branch

The `sgemm` NN transpose branch on Hopper architecture is the canonical usage ([hopper/ops/gemm.py:538-570](src/flag_blas/runtime/backend/_nvidia/hopper/ops/gemm.py)):

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

    # Aligned path: TensorDescriptor + TF32 acceleration
    dispatch.add(
        lambda: _make_sgemm_nn_aligned_runner(A, lda, B, ldb, C, ldc,
                                               m, n, k, alpha, beta,
                                               beta_is_zero, grid),
        aligned=True,
        name="aligned_k2",
    )

    # Unaligned + large: pad to 16-alignment
    dispatch.add(
        lambda: _make_sgemm_nn_padded_runner(A, lda, B, ldb, C, ldc,
                                              m, n, k, alpha, beta,
                                              beta_is_zero),
        aligned=False,
        name="padded_k2",
        filter=_is_sgemm_large,   # m>1024 and n>1024 and k>1024
    )

    # Unaligned + thin: Split-K strategy
    dispatch.add(
        lambda: _make_sgemm_nn_thin_runner(A, lda, B, ldb, C, ldc,
                                            m, n, k, alpha, beta,
                                            beta_is_zero),
        aligned=False,
        name="thin",
        filter=_is_sgemm_thin,   # min(m,n)<=64 and k>=256
    )

    # Unaligned + medium: generic fallback
    dispatch.add(
        lambda: _make_sgemm_nn_fallback_runner(A, lda, B, ldb, C, ldc,
                                                m, n, k, alpha, beta,
                                                beta_is_zero, grid),
        aligned=False,
        name="fallback",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k),
    )

    return dispatch


# Usage inside sgemm()
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

### Filter Mutual Exclusivity

For `aligned=False` (unaligned scenarios), the coverage of the four candidates:

| Candidate | Filter | Target Scenario |
|-----------|--------|-----------------|
| `padded_k2` | `m>1024 and n>1024 and k>1024` | Large matrices → pad + TensorDescriptor |
| `thin` | `min(m,n)≤64 and k≥256 and grid<32` | Thin matrices → Split-K + atomic_add |
| `fallback` | `not (m>1024 and n>1024 and k>1024)` | Medium sized → generic pointer kernel |
| `aligned_k2` | `aligned=True` (different aligned value) | Aligned matrices → TensorDescriptor + TF32 |

The filters for `thin` and `padded_k2` / `fallback` are mutually exclusive (one requires `min(m,n)≤64`, the other requires `m,n>1024`), ensuring that for any given shape with `aligned=False`, at most 2 candidates enter the pool (and in most cases only 1), avoiding unnecessary auto-tuning.

---

## Runtime Behavior

### First Call with a New Shape

```
lookup_and_build(m=4095, n=4095, k=4095, aligned=False)
  → entries = [padded_k2]  (only padded_k2 matches filter)
  → len(entries)==1 → return padded_k2 runner immediately
  → No auto-tuning, no DB I/O
```

```
lookup_and_build(m=1023, n=1023, k=1023, aligned=False)
  → entries = [fallback]   (only fallback matches filter)
  → len(entries)==1 → return fallback runner immediately
```

### Subsequent Calls (Cache Hit)

```
lookup_and_build(m=4095, n=4095, k=4095, aligned=False)
  → New SizeAutoDispatch instance
  → entries = [padded_k2]
  → len(entries)==1 → return immediately   ← O(1) every time
```

### Cache Hit with Multiple Candidates

When filter design results in `len(entries) >= 2`:

```
# First call
lookup_and_build(m=X, n=X, k=X, aligned=False)
  → entries = [candidate_a, candidate_b]
  → In-memory cache miss → DB miss → auto-tune → candidate_a wins
  → Written to in-memory cache + DB

# Subsequent calls (same process)
lookup_and_build(m=X, n=X, k=X, aligned=False)
  → entries = [candidate_a, candidate_b]
  → In-memory cache hit "candidate_a" → return immediately  ← O(1) dict lookup
```

---

## Testing and Verification

Verify dispatch behavior in a benchmark environment:

```bash
# Enable auto-tuning logs
TRITON_PRINT_AUTOTUNING=1 pytest test_gemm_perf.py::test_perf_sgemm_nn -v -s

# Skip correctness check, measure kernel performance only
pytest test_gemm_perf.py::test_perf_sgemm_nn -v -s --level core --skip_correctness
```

### Key Verification Points

1. **The one-time auto-tune cost for the first shape should not be counted in benchmark timing**
2. Benchmark frameworks should pre-populate all caches via correctness checks or warm-up calls before timing
3. If auto-tune log output is still observed during benchmarking, the cache is not working (check filter mutual exclusivity, DB file permissions)

### Verifying _save_db Success

**Method 1 — Log observation** (recommended):

Enable INFO-level logging for the dispatch module:

```python
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(name)s %(levelname)s %(message)s')
```

A successful write produces:
```
SizeAutoDispatch DB save success: table=sgemm_nn_variant key=(1023, 1023, 1023, 0) -> fallback
```

A failure produces (with full traceback):
```
SizeAutoDispatch DB save error for table=sgemm_nn_variant key=(1023, 1023, 1023, 0) candidate=fallback
Traceback (most recent call last):
  ...
```

**Method 2 — Direct SQLite query**:

```bash
python3 -c "
import sqlite3

db = '~/.flagblas/config_cache/TunedConfig_*.db'  # adjust path
conn = sqlite3.connect(db)
tables = conn.execute(\"SELECT name FROM sqlite_master
    WHERE type='table' AND name LIKE '%sgemm_nn_variant%'\").fetchall()
for (tname,) in tables:
    rows = conn.execute(f'SELECT * FROM \"{tname}\"').fetchall()
    print(f'{tname}: {len(rows)} row(s)')
    for r in rows:
        print(f'  {r}')
conn.close()
"
```

**Method 3 — Programmatic verification via `_lookup_db`**:

```python
from flag_blas.runtime.dispatch import SizeAutoDispatch
from flag_blas.utils.libentry import libcache

dispatch = SizeAutoDispatch(
    table_name='sgemm_nn_variant',
    build_key=lambda m, n, k, aligned, **kw: (m, n, k, int(aligned)),
    model=libcache.model,
)

result = dispatch._lookup_db((2048, 11008, 4096, 1))
if result:
    print(f'DB write verified: best candidate for key is {result}')
else:
    print('DB lookup returned None — write may have failed or key mismatch')
```

---

## FAQ

### Q: Why create a new instance on every call?

`SizeAutoDispatch` is designed to be lightweight (it only holds filter/rule lists and a DB connection reference), while the runner factory closures capture concrete tensor references (A, B, C, lda, ldb, etc.) that differ per call. Creating a dispatch instance per `sgemm()` invocation is therefore expected. Performance is guaranteed by the **module-level in-memory cache** — new instances hit the cache directly, bypassing DB queries and auto-tuning.

### Q: Can filter functions capture external state?

No. Filter input parameters (m, n, k, aligned) are already sufficient for the decision. Do not introduce additional state into filters, as this leads to non-reproducible behavior.

### Q: How do I clear the auto-tune cache?

```bash
# Clear in-process in-memory cache — restart the process

# Clear persistent DB cache
rm ~/.cache/flag_blas/TunedConfig_*.db
```

### Q: Do I need to clear the cache after modifying filter logic?

Yes. Old auto-tune results may be incompatible with the new logic. `lookup_and_build` detects that a cached candidate name no longer matches any registered entry and automatically clears the stale in-memory entry, but **old DB records must be manually removed** or left to be overwritten by subsequent auto-tune runs.

### Q: Why does the DB table name have an MD5 suffix (e.g., `sgemm_nn_variant-9924b48a...`)?

The actual table name in SQLite is generated by `SQLPersistantModel.get_sql_model()` by appending the MD5 hash of the key column names to the `table_name` parameter. This allows the same logical table name to coexist with different key schemas. When querying the DB directly, use `LIKE '%sgemm_nn_variant%'` instead of an exact match.

---

## API Reference

| Class / Function | Purpose |
|------------------|---------|
| `SizeAutoDispatch(table_name, build_key, model)` | Construct a dispatch instance |
| `dispatch.add(factory, *, aligned, name, filter)` | Register a candidate kernel variant |
| `dispatch.lookup_and_build(m, n, k, aligned, **extra)` | Look up and build the optimal runner |
| `KernelRunner(kernel, *args, **kwargs)` | Pre-bound kernel invocation wrapper |
| `_SizeFilter` | Type alias `Callable[..., bool]` |

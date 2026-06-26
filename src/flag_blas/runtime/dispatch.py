from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple

import torch
import triton

from flag_blas.utils.models import PersistantModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level in-memory cache for autotune results.
#
# Keys are (table_name, cache_key) tuples; values are the winning candidate
# name string.  This cache survives across SizeAutoDispatch instances so that
# once autotune has selected a winner for a given problem size, every
# subsequent call — even from a freshly constructed dispatch — skips both the
# DB query and re-autotune entirely.
#
# The cache is process-local.  If the filter functions or candidate set
# change between runs the user must restart the process (or delete the
# corresponding SQLite DB row) — exactly the same invalidation contract as
# the persistent DB cache.
# ---------------------------------------------------------------------------
_autotune_result_cache: Dict[Tuple[str, tuple], str] = {}
_autotune_result_lock = threading.Lock()


_SizeFilter = Callable[..., bool]


class SizeAutoDispatch:
    """
    Auto-tuning kernel variant dispatch with database persistence.

    For each problem size (identified by a cache key), benchmarks all
    applicable kernel candidates, selects the fastest, and stores the
    choice in the database. On subsequent calls with the same size,
    returns the cached best candidate instantly.

    Uses lazy runner construction: ``add()`` accepts a **factory**
    (zero-arg callable that returns a runner), and only the selected
    candidate's factory is called — eliminating unnecessary closure
    creation overhead.

    Usage::

        import functools

        dispatch = SizeAutoDispatch(
            table_name="sgemm_nn_variant",
            build_key=lambda m, n, k, aligned, **kw: (m, n, k, int(aligned)),
        )

        dispatch.add(
            lambda: make_thin_runner(A, lda, ...),
            name="thin", filter=is_thin,
        )
        dispatch.add(
            lambda: make_k2_runner(A, lda, ...),
            aligned=True, name="aligned_k2", filter=not_thin,
        )

        runner = dispatch.lookup_and_build(m, n, k, aligned)
        runner()

    Parameters
    ----------
    table_name:
        Database table name for persisting the (key -> best_candidate) mapping.
    build_key:
        Function that builds a unique cache key from problem dimensions.
        Signature: (m, n, k, aligned, **extra) -> tuple
    model:
        A ``PersistantModel`` instance for database access.
        If None, no persistence is used and autotuning runs on every call.
    """

    _RunnerFactory = Callable[[], Callable[[], None]]

    # Number of warmup kernel launches before timing each candidate.
    _WARMUP_ITERS: int = 1
    # Number of timed kernel launches per candidate.
    _TIMING_ITERS: int = 5
    # Size (bytes) of a dummy buffer used to evict L2 cache between
    # candidates.  Set to 0 to disable.
    _CACHE_FLUSH_BYTES: int = 16 * 1024 * 1024  # 16 MiB

    def __init__(
        self,
        table_name: str,
        build_key: Callable[..., tuple],
        model: Optional[PersistantModel] = None,
    ):
        self._table_name = table_name
        self._build_key = build_key
        self._model = model
        self._entries: List[_Entry] = []
        self._candidate_index: int = 0
        self._flush_buf: Optional[torch.Tensor] = None

    class _Entry:
        __slots__ = ("name", "factory", "align", "size_filter")

        def __init__(
            self,
            name: str,
            factory: Callable[[], Callable[[], None]],
            align: Optional[bool] = None,
            size_filter: Optional[_SizeFilter] = None,
        ):
            self.name = name
            self.factory = factory
            self.align = align
            self.size_filter = size_filter

    def add(
        self,
        factory: Callable[[], Callable[[], None]],
        *,
        aligned: Optional[bool] = None,
        name: Optional[str] = None,
        filter: Optional[_SizeFilter] = None,
    ):
        candidate_name = name or f"variant_{self._candidate_index}"
        self._candidate_index += 1
        self._entries.append(
            self._Entry(
                name=candidate_name,
                factory=factory,
                align=aligned,
                size_filter=filter,
            )
        )

    def _get_entries(
        self, m: int, n: int, k: int, aligned: bool, **extra
    ) -> List[_Entry]:
        result: List[SizeAutoDispatch._Entry] = []
        for entry in self._entries:
            if entry.align is not None and entry.align != aligned:
                continue
            if entry.size_filter is not None and not entry.size_filter(
                m=m, n=n, k=k, aligned=aligned, **extra
            ):
                continue
            result.append(entry)
        return result

    def _lookup_db(self, cache_key: tuple) -> Optional[str]:
        if self._model is None:
            return None
        try:
            config = self._model.get_config(self._table_name, cache_key)
            if config is not None and isinstance(config, triton.Config):
                name = config.kwargs.get("_candidate_name", None)
                logger.debug(
                    "SizeAutoDispatch DB lookup hit: table=%s key=%s -> %s",
                    self._table_name, cache_key, name,
                )
                return name
            logger.debug(
                "SizeAutoDispatch DB lookup miss: table=%s key=%s",
                self._table_name, cache_key,
            )
            return None
        except Exception:
            logger.error(
                "SizeAutoDispatch DB lookup error for table=%s key=%s",
                self._table_name, cache_key, exc_info=True,
            )
            return None

    def _save_db(self, cache_key: tuple, candidate_name: str) -> None:
        if self._model is None:
            logger.debug(
                "SizeAutoDispatch DB save skipped (no model): table=%s key=%s candidate=%s",
                self._table_name, cache_key, candidate_name,
            )
            return
        try:
            c = triton.Config(kwargs={"_candidate_name": candidate_name})
            self._model.put_config(self._table_name, cache_key, c)
            logger.info(
                "SizeAutoDispatch DB save success: table=%s key=%s -> %s",
                self._table_name, cache_key, candidate_name,
            )
        except Exception:
            logger.error(
                "SizeAutoDispatch DB save error for table=%s key=%s candidate=%s",
                self._table_name, cache_key, candidate_name, exc_info=True,
            )

    def lookup_and_build(
        self,
        m: int,
        n: int,
        k: int,
        aligned: bool,
        *,
        snapshot_tensor=None,
        **extra,
    ) -> Callable[[], None]:
        cache_key = self._build_key(m, n, k, aligned, **extra)
        entries = self._get_entries(m, n, k, aligned, **extra)

        if not entries:
            raise ValueError(
                f"No kernel candidates for m={m}, n={n}, k={k}, aligned={aligned}"
            )

        if len(entries) == 1:
            return entries[0].factory()

        # --- fast path: in-memory cache (survives across dispatch instances) ---
        mem_key = (self._table_name, cache_key)
        with _autotune_result_lock:
            cached_name = _autotune_result_cache.get(mem_key)
        if cached_name is not None:
            for entry in entries:
                if entry.name == cached_name:
                    return entry.factory()
            # The cached name no longer matches any registered entry
            # (filter functions may have changed).  Clear the stale
            # entry and fall through to DB/autotune.
            with _autotune_result_lock:
                _autotune_result_cache.pop(mem_key, None)

        # --- DB-backed path (persists across process restarts) ---
        cached_name = self._lookup_db(cache_key)
        if cached_name is not None:
            for entry in entries:
                if entry.name == cached_name:
                    # Promote to memory cache so the next instance skips DB I/O.
                    with _autotune_result_lock:
                        _autotune_result_cache[mem_key] = cached_name
                    return entry.factory()

        # --- autotune (first time this problem size is ever seen) ---
        result = self._autotune(cache_key, entries, snapshot_tensor)
        if result is not None:
            best_entry, best_runner = result
            # Persist to both memory and DB.
            with _autotune_result_lock:
                _autotune_result_cache[mem_key] = best_entry.name
            self._save_db(cache_key, best_entry.name)
            return best_runner
        # All candidates failed during tuning; fall back to the first
        # entry WITHOUT persisting to DB or memory, so future calls will
        # retry autotune instead of blindly reusing a broken candidate.
        logger.warning(
            "SizeAutoDispatch: all candidates failed for key %s, "
            "falling back to %r without persisting",
            cache_key, entries[0].name,
        )
        return entries[0].factory()

    def _autotune(
        self,
        cache_key: tuple,
        entries: List[_Entry],
        snapshot_tensor=None,
    ) -> Optional[Tuple[_Entry, Callable[[], None]]]:
        timings: Dict[str, float] = {}
        runners: Dict[str, Tuple[_Entry, Callable[[], None]]] = {}
        snapshot = None
        if snapshot_tensor is not None:
            snapshot = snapshot_tensor.clone()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for entry in entries:
            try:
                if snapshot is not None:
                    snapshot_tensor.copy_(snapshot)
                runner = entry.factory()
                runners[entry.name] = (entry, runner)

                runner()
                torch.cuda.synchronize()

                if snapshot is not None:
                    snapshot_tensor.copy_(snapshot)

                best_elapsed = float("inf")
                for _ in range(5):
                    if snapshot is not None:
                        snapshot_tensor.copy_(snapshot)
                    start.record()
                    runner()
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end)
                    if elapsed < best_elapsed:
                        best_elapsed = elapsed
                timings[entry.name] = best_elapsed
            except Exception as e:
                logger.warning(
                    "SizeAutoDispatch: candidate %r failed for key %s: %s",
                    entry.name, cache_key, e,
                )

        if not timings:
            logger.warning(
                "SizeAutoDispatch: all candidates failed for key %s, using first",
                cache_key,
            )
            if snapshot is not None:
                snapshot_tensor.copy_(snapshot)
            return None

        best_name = min(timings, key=timings.get)
        logger.debug(
            "SizeAutoDispatch: key=%s best=%s timings=%s",
            cache_key, best_name, timings,
        )
        if snapshot is not None:
            snapshot_tensor.copy_(snapshot)
        return runners.get(best_name)


class StaticDispatch:
    """
    Developer-maintained static dispatch table.

    Maps shape conditions directly to kernel factories. No autotune,
    no caching — the first matching entry wins immediately.

    The dispatch table itself is designed to be created once at module
    level and reused across calls.  The per-call varying arguments (e.g.
    tensors A, B, C) are passed to ``lookup_and_build`` via the
    ``context`` dict.  Factories receive these as keyword arguments,
    avoiding the need to re-create closures on every call.

    Usage
    -----
    Module level (once)::

        def is_aligned(m, n, k, aligned, **_kw):
            return aligned

        def is_default(**_kw):
            return True

        def build_aligned_runner(A, B, C, m, n, k, lda, ldb, ldc,
                                 alpha, beta, beta_is_zero):
            return lambda: _kernel2[...](A, B, C, alpha, beta, ...)

        def build_fallback_runner(A, B, C, m, n, k, lda, ldb, ldc,
                                  alpha, beta, beta_is_zero):
            return lambda: _kernel[...](A, B, C, alpha, beta, ...)

        _MY_DISPATCH = StaticDispatch([
            (is_aligned, build_aligned_runner),
            (is_default, build_fallback_runner),
        ])

    Per-call (inside the function)::

        runner = _MY_DISPATCH.lookup_and_build(
            m, n, k, aligned,
            context=dict(A=A, B=B, C=C, m=m, n=n, k=k,
                         lda=lda, ldb=ldb, ldc=ldc,
                         alpha=alpha, beta=beta, beta_is_zero=beta_is_zero),
        )
        runner()

    Parameters
    ----------
    table:
        A list of ``(condition, factory)`` pairs evaluated **in order**.
        Each ``condition`` is a callable with signature
        ``(m, n, k, aligned, **extra) -> bool``.
        Each ``factory`` is a callable that accepts keyword arguments
        from ``context`` (or no arguments if ``context`` is None) and
        returns a ``Callable[[], None]`` runner.
        The last entry should be a catch-all (condition always True).
    """

    _Entry = Tuple[Callable[..., bool], Callable[..., Callable[[], None]]]

    def __init__(
        self,
        table: List[_Entry],
    ):
        self._table = table

    def lookup_and_build(
        self,
        m: int,
        n: int,
        k: int,
        aligned: bool,
        *,
        context: Optional[dict] = None,
        **extra,
    ) -> Callable[[], None]:
        for condition, factory in self._table:
            if condition(m=m, n=n, k=k, aligned=aligned, **extra):
                if context is not None:
                    return factory(**context)
                return factory()
        raise ValueError(
            f"StaticDispatch: no matching entry for "
            f"m={m}, n={n}, k={k}, aligned={aligned}"
        )


class KernelRunner:
    """
    A callable wrapper that executes a kernel with pre-bound arguments.

    Usage::

        runner = KernelRunner(kernel_fn, arg1, arg2, kw1=val1)
        runner()  # calls kernel_fn(arg1, arg2, kw1=val1)
    """

    def __init__(self, kernel: Callable, *args, **kwargs):
        self._kernel = kernel
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        return self._kernel(*self._args, **self._kwargs)

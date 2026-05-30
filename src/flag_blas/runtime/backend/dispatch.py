from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import triton

from flag_blas.utils.models import PersistantModel

logger = logging.getLogger(__name__)


_SizeFilter = Callable[..., bool]


class SizeDispatchTable:
    """
    A generic size-based kernel dispatch table.

    Maps (size_category, aligned) pairs to kernel implementations.
    Supports wildcard alignment (aligned=None) that matches any alignment.

    Usage::

        table = SizeDispatchTable()
        table.add("thin", thin_kernel)              # wildcard: any alignment
        table.add("large", k2, aligned=True)        # aligned only
        table.add("large", padded_k2, aligned=False) # misaligned only

        kernel = table.lookup("large", aligned=True)  # -> k2
        kernel = table.lookup("large", aligned=False) # -> padded_k2
        kernel = table.lookup("thin", aligned=False)  # -> thin_kernel (wildcard)
    """

    def __init__(self):
        self._exact: Dict[Tuple[str, bool], Callable[[], None]] = OrderedDict()
        self._wildcard: Dict[str, Callable[[], None]] = OrderedDict()

    def add(
        self,
        size_category: str,
        kernel: Callable[[], None],
        *,
        aligned: Optional[bool] = None,
    ):
        if aligned is None:
            self._wildcard[size_category] = kernel
        else:
            self._exact[(size_category, aligned)] = kernel

    def lookup(self, size_category: str, aligned: bool) -> Callable[[], None]:
        key = (size_category, aligned)
        if key in self._exact:
            return self._exact[key]
        if size_category in self._wildcard:
            return self._wildcard[size_category]
        raise ValueError(
            f"No kernel registered for size_category={size_category!r}, "
            f"aligned={aligned}"
        )

    @property
    def rules(self):
        return list(self._exact.keys()) + [
            (k, None) for k in self._wildcard.keys()
        ]


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
                return config.kwargs.get("_candidate_name", None)
            return None
        except Exception as e:
            logger.warning("SizeAutoDispatch DB lookup error: %s", e)
            return None

    def _save_db(self, cache_key: tuple, candidate_name: str) -> None:
        if self._model is None:
            return
        try:
            c = triton.Config({}, kwargs={"_candidate_name": candidate_name})
            self._model.put_config(self._table_name, cache_key, c)
        except Exception as e:
            logger.warning("SizeAutoDispatch DB save error: %s", e)

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

        cached_name = self._lookup_db(cache_key)
        if cached_name is not None:
            for entry in entries:
                if entry.name == cached_name:
                    return entry.factory()

        best_entry = self._autotune(cache_key, entries, snapshot_tensor)
        self._save_db(cache_key, best_entry.name)
        return best_entry.factory()

    def _autotune(
        self,
        cache_key: tuple,
        entries: List[_Entry],
        snapshot_tensor=None,
    ) -> _Entry:
        timings: Dict[str, float] = {}
        snapshot = None
        if snapshot_tensor is not None:
            snapshot = snapshot_tensor.clone()

        for entry in entries:
            try:
                if snapshot is not None:
                    snapshot_tensor.copy_(snapshot)
                runner = entry.factory()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                runner()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                timings[entry.name] = elapsed
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
            return entries[0]

        best_name = min(timings, key=timings.get)
        logger.debug(
            "SizeAutoDispatch: key=%s best=%s timings=%s",
            cache_key, best_name, timings,
        )
        if snapshot is not None:
            snapshot_tensor.copy_(snapshot)
        for entry in entries:
            if entry.name == best_name:
                return entry
        return entries[0]


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

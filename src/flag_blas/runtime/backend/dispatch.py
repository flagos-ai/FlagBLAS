from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import triton

from flag_blas.utils.models import PersistantModel

logger = logging.getLogger(__name__)


_CandidateKernel = Callable[[], None]


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
        self._exact: Dict[Tuple[str, bool], _CandidateKernel] = OrderedDict()
        self._wildcard: Dict[str, _CandidateKernel] = OrderedDict()

    def add(
        self,
        size_category: str,
        kernel: _CandidateKernel,
        *,
        aligned: Optional[bool] = None,
    ):
        if aligned is None:
            self._wildcard[size_category] = kernel
        else:
            self._exact[(size_category, aligned)] = kernel

    def lookup(self, size_category: str, aligned: bool) -> _CandidateKernel:
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
    Auto-tuning size dispatch table with database persistence.

    For each problem size (identified by a cache key), benchmarks all applicable
    kernel candidates, selects the fastest, and stores the choice in the database.
    On subsequent calls with the same size, uses the cached best candidate.

    Like ``libtuner``, but at the kernel-variant level:
    instead of tuning a single kernel's block/thread configuration,
    it tunes **which kernel variant** to use for a given problem size.

    Usage::

        dispatch = SizeAutoDispatch(
            table_name="sgemm_nn_variant",
            classify_size=my_classify_fn,
            build_key=my_key_fn,
        )

        dispatch.add("thin", thin_runner)
        dispatch.add("large", aligned_k2, aligned=True)
        dispatch.add("large", padded_k2, aligned=False)

        runner = dispatch.lookup(m, n, k, aligned)
        runner()

    Parameters
    ----------
    table_name:
        Database table name for persisting the (key -> best_candidate) mapping.
    classify_size:
        Function that maps problem dimensions to a size_category string.
        Signature: (m, n, k, **extra) -> str
    build_key:
        Function that builds a unique cache key from problem dimensions.
        Signature: (m, n, k, aligned, **extra) -> tuple
    model:
        A ``PersistantModel`` instance for database access.
        If None, no persistence is used and autotuning runs on every call.
    """

    def __init__(
        self,
        table_name: str,
        classify_size: Callable[..., str],
        build_key: Callable[..., tuple],
        model: Optional[PersistantModel] = None,
    ):
        self._table_name = table_name
        self._classify_size = classify_size
        self._build_key = build_key
        self._model = model
        self._exact: Dict[Tuple[str, bool], List[Tuple[str, _CandidateKernel]]] = OrderedDict()
        self._wildcard: Dict[str, List[Tuple[str, _CandidateKernel]]] = OrderedDict()
        self._candidate_index: int = 0

    def add(
        self,
        size_category: str,
        kernel: _CandidateKernel,
        *,
        aligned: Optional[bool] = None,
        name: Optional[str] = None,
    ):
        candidate_name = name or f"variant_{self._candidate_index}"
        self._candidate_index += 1
        entry = (candidate_name, kernel)
        if aligned is None:
            self._wildcard.setdefault(size_category, []).append(entry)
        else:
            self._exact.setdefault((size_category, aligned), []).append(entry)

    def _get_candidates(self, size_category: str, aligned: bool) -> List[Tuple[str, _CandidateKernel]]:
        key = (size_category, aligned)
        candidates = list(self._exact.get(key, []))
        candidates.extend(self._wildcard.get(size_category, []))
        return candidates

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

    def lookup(
        self,
        m: int,
        n: int,
        k: int,
        aligned: bool,
        **extra,
    ) -> _CandidateKernel:
        size_category = self._classify_size(m, n, k, **extra)
        cache_key = self._build_key(m, n, k, aligned, **extra)
        candidates = self._get_candidates(size_category, aligned)

        if not candidates:
            raise ValueError(
                f"No kernel candidates for size_category={size_category!r}, "
                f"aligned={aligned}"
            )

        if len(candidates) == 1:
            runner = candidates[0][1]
            self._save_db(cache_key, candidates[0][0])
            return runner

        cached_name = self._lookup_db(cache_key)
        if cached_name is not None:
            for name, runner in candidates:
                if name == cached_name:
                    return runner

        best_name, best_runner = self._autotune(cache_key, candidates)
        self._save_db(cache_key, best_name)
        return best_runner

    def _autotune(
        self,
        cache_key: tuple,
        candidates: List[Tuple[str, _CandidateKernel]],
    ) -> Tuple[str, _CandidateKernel]:
        timings: Dict[str, float] = {}

        for name, runner in candidates:
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                runner()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                timings[name] = elapsed
            except Exception as e:
                logger.warning(
                    "SizeAutoDispatch: candidate %r failed for key %s: %s",
                    name, cache_key, e,
                )

        if not timings:
            logger.warning(
                "SizeAutoDispatch: all candidates failed for key %s, using first",
                cache_key,
            )
            return candidates[0]

        best_name = min(timings, key=timings.get)
        logger.debug(
            "SizeAutoDispatch: key=%s best=%s timings=%s",
            cache_key, best_name, timings,
        )
        return best_name, dict(candidates)[best_name]


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

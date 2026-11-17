# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import ctypes
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

from ..utils import CORE_NUM
from . import hpmv as _launch_utils
from . import spr as _spr_utils


def _launch_l1_entry(compiled, run, grid, tensors, pointers, arguments, device):
    knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    hooks = (
        getattr(knobs, "launch_enter_hook", None),
        getattr(knobs, "launch_exit_hook", None),
    )
    has_hooks = any(
        (
            h is not None
            and (type(h) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h.calls))
            for h in hooks
        )
    )
    if (
        not _launch_utils._FAST_LAUNCH_SUPPORTED
        or type(run) is not _launch_utils._NPU_LAUNCHER
        or knobs is None
        or has_hooks
        or getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
        or getattr(getattr(run, "metadata", None), "debug_enabled", False)
        or any(
            (type(t) is not torch.Tensor or t.is_conj() or t.is_neg() for t in tensors)
        )
    ):
        compiled[grid](*_launch_utils._real_tensors(tensors), *arguments)
        return
    wrapped = []
    for tensor, address in zip(tensors, pointers):
        pointer = _launch_utils._DevicePointer(address)
        pointer.tensor = tensor
        wrapped.append(pointer)
    registered = run.launch(
        *grid,
        _launch_utils._current_raw_stream(device),
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *wrapped,
        *arguments,
    )
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_extrema"),
    key=["N", "INCX", "COMPLEX", "MINIMUM", "PARTS"],
)
@triton.jit
def _amax_fallback_partial(
    X,
    V,
    IDX,
    OUT,
    N: tl.constexpr,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    MINIMUM: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    best = -float("inf")
    best_index = 2147483647
    for start in range(pid * BLOCK, N, PARTS * BLOCK):
        index = start + lane
        mask = index < N
        offset = index.to(tl.int64) * INCX
        if COMPLEX:
            re = tl.load(X + 2 * offset, mask, other=0.0)
            im = tl.load(X + 2 * offset + 1, mask, other=0.0)
            magnitude = tl.abs(re) + tl.abs(im)
        else:
            magnitude = tl.abs(tl.load(X + offset, mask, other=0.0))
        values = tl.where(mask, magnitude, -float("inf"))
        value = tl.max(values, 0)
        better = value > best
        candidate = tl.min(tl.where(mask & (values == value), index, 2147483647), 0)
        take = better | (value == best) & (candidate < best_index)
        best = tl.where(take, value, best)
        best_index = tl.where(take, candidate, best_index)
    if PARTS == 1:
        tl.store(OUT, best_index + 1)
    else:
        tl.store(V + pid, best)
        tl.store(IDX + pid, best_index)


@libentry()
@triton.jit
def _amax_finish(
    V,
    IDX,
    OUT,
    PARTS: tl.constexpr,
    MINIMUM: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED: tl.constexpr = False,
):
    lane = tl.arange(0, BLOCK)
    mask = lane < PARTS
    if PACKED:
        bits = tl.load(V + lane, mask, other=0)
        values = tl.where(mask, bits.to(tl.float32, bitcast=True), -float("inf"))
        best = tl.max(values, 0)
    else:
        values = tl.load(V + lane, mask, other=-float("inf"))
        best = tl.max(values, 0)
    indices = tl.load(IDX + lane, mask, other=2147483647)
    index = tl.min(tl.where(mask & (values == best), indices, 2147483647), 0)
    tl.store(OUT, index + 1)


def _prune_amax_configs(configs, args, **kwargs):
    if args["WIDE"] or args["INCX"] > 8:
        limit = 256
    elif args["COMPLEX"]:
        limit = 4096 if args["INCX"] == 1 else 512
    else:
        limit = 2048 if args["INCX"] != 1 else 4096
    if args["N"] <= 512:
        limit = min(limit, max(256, triton.next_power_of_2(args["N"])))
    return [config for config in configs if config.kwargs["BLOCK"] <= limit]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_amax"),
    key=["N", "INCX", "COMPLEX", "PARTS", "WIDE", "GATHER"],
    prune_configs_by={"early_config_prune": _prune_amax_configs},
)
@triton.jit
def _amax_partial(
    X,
    VALUES,
    INDICES,
    OUT,
    N,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED: tl.constexpr = False,
):
    lane = tl.arange(0, BLOCK)
    # Accumulate per lane and reduce only after scanning the assigned segment.
    best = tl.full((BLOCK,), -float("inf"), tl.float32)
    best_index = tl.full((BLOCK,), 2147483647, tl.int32)
    tiles_per_part = tl.cdiv(N, PARTS * BLOCK)
    first = tl.program_id(0) * tiles_per_part * BLOCK
    end = tl.minimum(first + tiles_per_part * BLOCK, N)
    step = BLOCK
    for start in range(first, end, step):
        index = start + lane
        mask = index < N
        if GATHER:
            component = tl.arange(0, BLOCK * WIDTH)
            address_index = start.to(tl.int64) if WIDE else start
            size = N.to(tl.int64) if WIDE else N
            if COMPLEX:
                physical = address_index * (2 * INCX) + component
                values = tl.load(
                    X + physical, physical < ((size - 1) * INCX + 1) * 2, other=0.0
                )
                real = tl.gather(values, lane * (2 * INCX), axis=0)
                imag = tl.gather(values, lane * (2 * INCX) + 1, axis=0)
                magnitude = tl.abs(real) + tl.abs(imag)
            else:
                physical = address_index * INCX + component
                values = tl.load(
                    X + physical, physical < (size - 1) * INCX + 1, other=0.0
                )
                magnitude = tl.abs(tl.gather(values, lane * INCX, axis=0))
        else:
            address_index = index.to(tl.int64) if WIDE else index
            value = tl.load(X + address_index * INCX, mask, other=0.0)
            magnitude = tl.abs(value)
        take = mask & (magnitude > best)
        best = tl.where(take, magnitude, best)
        best_index = tl.where(take, index, best_index)
    maximum = tl.max(best, 0)
    index = tl.min(tl.where(best == maximum, best_index, 2147483647), 0)
    if PARTS == 1:
        tl.store(OUT, index + 1)
    else:
        if PACKED:
            tl.store(VALUES + tl.program_id(0), maximum.to(tl.int32, bitcast=True))
        else:
            tl.store(VALUES + tl.program_id(0), maximum)
        tl.store(INDICES + tl.program_id(0), index)


_AMAX_COMPUTE = _amax_partial.jit_function


_AMAX_SMALL_CONFIGS = runtime.get_tuned_config("ascend_l1_amax")


_LAUNCH_CACHE = {}


@libentry()
@triton.jit
def _amax_single(
    X,
    OUT,
    N: tl.constexpr,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _AMAX_COMPUTE(X, OUT, OUT, OUT, N, INCX, COMPLEX, 1, WIDE, GATHER, WIDTH, BLOCK)


def _amax_small(n, x, incx, result, complex_data):
    expected = torch.complex64 if complex_data else torch.float32
    assert x.dtype == expected, f"x must be {expected}"
    assert result.dtype == torch.int32, "result must be int32"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert result.numel() >= 1, "result must not be empty"
    device = x.device
    assert device == result.device, "result must be on the same device"
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _amax_small(n, x, incx, result, complex_data)
    (x_pointer, result_pointer) = (x.data_ptr(), result.data_ptr())
    key = (
        "small",
        device.index,
        n,
        incx,
        complex_data,
        (x_pointer % 16 == 0, x_pointer % 64 == 0),
        result_pointer % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        gather = complex_data or incx > 1
        configs = _prune_amax_configs(
            _AMAX_SMALL_CONFIGS,
            {
                "N": n,
                "INCX": incx,
                "COMPLEX": complex_data,
                "GATHER": gather,
                "WIDE": False,
            },
        )
        eligible = [
            c for c in configs if c.kwargs["BLOCK"] <= triton.next_power_of_2(n)
        ]
        config = (
            max(eligible, key=lambda c: c.kwargs["BLOCK"])
            if eligible
            else min(configs, key=lambda c: c.kwargs["BLOCK"])
        )
        constants = (
            n,
            incx,
            complex_data,
            False,
            gather,
            triton.next_power_of_2(incx * (2 if complex_data else 1)),
            config.kwargs["BLOCK"],
        )
        pointer = torch.view_as_real(x) if complex_data else x
        (compiled, _) = _amax_small_kernel(complex_data, incx, x_pointer)[1,](
            pointer,
            result,
            *constants,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        if len(_LAUNCH_CACHE) >= 1024:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = (compiled, compiled.run, constants)
        return
    (compiled, run, constants) = entry
    _launch_l1_entry(
        compiled,
        run,
        (1, 1, 1),
        (x, result),
        (x_pointer, result_pointer),
        constants,
        device.index,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_amax"),
    key=["N", "INCX", "COMPLEX", "PARTS", "WIDE", "GATHER"],
    prune_configs_by={"early_config_prune": _prune_amax_configs},
)
@triton.jit
def _amax_packed_partial(
    X,
    WORK,
    OUT,
    N,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _AMAX_COMPUTE(
        X,
        WORK,
        WORK + PARTS,
        OUT,
        N,
        INCX,
        COMPLEX,
        PARTS,
        WIDE,
        GATHER,
        WIDTH,
        BLOCK,
        True,
    )


_AMAX_FINISH_COMPUTE = _amax_finish.jit_function


@libentry()
@triton.jit
def _amax_packed_finish(WORK, OUT, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    _AMAX_FINISH_COMPUTE(WORK, WORK + PARTS, OUT, PARTS, False, BLOCK, True)


def _amax_large(n, x, incx, result, complex_data):
    expected = torch.complex64 if complex_data else torch.float32
    assert x.dtype == expected, f"x must be {expected}"
    assert result.dtype == torch.int32, "result must be int32"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert result.numel() >= 1, "result must not be empty"
    device = x.device
    assert device == result.device, "result must be on the same device"
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _amax_large(n, x, incx, result, complex_data)
    parts = min((n + 511) // 512, CORE_NUM)
    work = torch.empty(2 * parts, dtype=torch.int32, device=device)
    pointers = (x.data_ptr(), work.data_ptr(), result.data_ptr())
    if (
        n >= 1048576
        and (complex_data or incx == 1 or incx >= 5 or (n >= 2097152))
        and (not (complex_data and incx == 1 and (n >= 33554432)))
        and (0 < incx <= 5)
        and (
            (n + (CORE_NUM + 2) * 8192) * incx * (2 if complex_data else 1) < 2147483647
        )
        and (pointers[0] % 64 == pointers[1] % 16 == pointers[2] % 16 == 0)
        and all(
            (
                type(t) is torch.Tensor and (not t.is_conj()) and (not t.is_neg())
                for t in (x, result)
            )
        )
    ):
        offset = _amax_stream_offset(device.index)
        if offset:
            return _amax_stream_reduce(
                n, x, incx, result, complex_data, work, parts, pointers, offset
            )
    aligned = tuple((pointer % 16 == 0 for pointer in pointers))
    use_pair = (
        complex_data
        and incx == 1
        and (n >= CORE_NUM * 131072)
        and (2 * (n + CORE_NUM * 4096) < 2147483647)
        and (pointers[0] % 64 == 0)
    )
    offset = (
        _amax_stream_offset(device.index)
        if use_pair
        and n >= 33554432
        and (type(x) is torch.Tensor)
        and (not x.is_conj())
        and (not x.is_neg())
        else 0
    )
    key = ("large", device.index, n, incx, complex_data, aligned, use_pair, offset)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        gather = complex_data or incx > 1
        arguments = (
            n,
            incx,
            complex_data,
            parts,
            n * incx * (2 if complex_data else 1) > 2147483647,
            gather,
            triton.next_power_of_2(incx * (2 if complex_data else 1)),
        )
        pointer = torch.view_as_real(x) if complex_data else x
        kernel = _amax_complex_partial if use_pair else _amax_packed_partial
        if use_pair:
            (compiled, meta) = kernel[parts,](
                pointer, work, result, *arguments, OFFSET=offset
            )
        else:
            (compiled, meta) = kernel[parts,](pointer, work, result, *arguments)
        if len(_LAUNCH_CACHE) >= 1024:
            _LAUNCH_CACHE.clear()
        constants = (
            (*arguments, meta["BLOCK"], offset)
            if use_pair
            else (*arguments, meta["BLOCK"])
        )
        _LAUNCH_CACHE[key] = (compiled, compiled.run, constants)
    else:
        (compiled, run, arguments) = entry
        _launch_l1_entry(
            compiled,
            run,
            (parts, 1, 1),
            (x, work, result),
            pointers,
            arguments,
            device.index,
        )
    finish_key = ("finish", device.index, parts, aligned[1:])
    finish = _LAUNCH_CACHE.get(finish_key)
    arguments = (parts, 1 << (parts - 1).bit_length())
    if finish is None:
        (compiled, _) = _amax_packed_finish[1,](
            work, result, *arguments, num_warps=1, num_stages=1
        )
        if len(_LAUNCH_CACHE) >= 1024:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[finish_key] = (compiled, compiled.run)
    else:
        _launch_l1_entry(
            *finish, (1, 1, 1), (work, result), pointers[1:], arguments, device.index
        )


def _prune_amax_parallel_configs(configs, args, **kwargs):
    configs = [config for config in configs if config.kwargs["PARTS"] == CORE_NUM]
    if args["F"] == 1:
        lower = 1024 if args["INCX"] == 2 and 131072 < args["N"] <= 1048576 else 512
        return [config for config in configs if lower <= config.kwargs["BLOCK"] <= 2048]
    limit = 512 if args["INCX"] > 3 else 1024
    return [config for config in configs if config.kwargs["BLOCK"] <= limit]


@triton.jit
def _amax_parallel_scores(
    X, start, N, INCX: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    physical = tl.arange(0, F * BLOCK * triton.next_power_of_2(INCX))
    offset = F * start * INCX + physical
    limit = tl.minimum(F * ((N - 1) * INCX + 1), F * (start + BLOCK) * INCX)
    raw = tl.load(X + offset, offset < limit, other=0.0)
    if F == 2:
        real = tl.gather(raw, 2 * lane * INCX, 0)
        imag = tl.gather(raw, 2 * lane * INCX + 1, 0)
        return tl.abs(real) + tl.abs(imag)
    return tl.abs(tl.gather(raw, lane * INCX, 0))


@libentry()
@triton.jit
def _amax_parallel_init(OUT, N):
    # Preserve the existing sentinel if no non-NaN magnitude is found.
    tl.store(OUT, -2147483648)


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"PARTS": CORE_NUM, **config.kwargs},
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in runtime.get_tuned_config("ascend_l1_amax_parallel")
    ],
    key=["N", "INCX", "F"],
    prune_configs_by={"early_config_prune": _prune_amax_parallel_configs},
)
@triton.jit
def _amax_parallel_max(
    X,
    OUT,
    N: tl.constexpr,
    INCX: tl.constexpr,
    F: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    best = tl.full((BLOCK,), -float("inf"), tl.float32)
    for start in range(first, end, BLOCK):
        value = _amax_parallel_scores(X, start, N, INCX, F, BLOCK)
        best = tl.where((start + lane < N) & (value > best), value, best)
    maximum = tl.max(best, 0)
    # Nonnegative finite/Inf binary32 codes leave the NaN interval unused.
    # For N < 2**23 - 1 this shift fits int32 and lies strictly above 1..N.
    if maximum >= 0.0:
        encoded = maximum.to(tl.int32, bitcast=True) + N + 1
        tl.atomic_max(OUT, encoded, sem="relaxed")


@libentry()
@triton.jit
def _amax_parallel_index(
    X,
    OUT,
    N,
    INCX: tl.constexpr,
    F: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    state = tl.load(OUT, volatile=True)
    if (state > 0) & (state <= N):
        # A later block may see a published index. The immutable winning
        # element recovers the same maximum without another scratch tensor.
        winner = state - 1
        magnitude = tl.abs(tl.load(X + winner * INCX * F))
        if F == 2:
            magnitude += tl.abs(tl.load(X + winner * INCX * F + 1))
        wanted = magnitude.to(tl.int32, bitcast=True)
    elif state > N:
        wanted = state - N - 1
    else:
        wanted = 0
    lane = tl.arange(0, BLOCK)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    position = tl.full((BLOCK,), 2147483647, tl.int32)
    for start in range(first, end, BLOCK):
        value = _amax_parallel_scores(X, start, N, INCX, F, BLOCK)
        index = start + lane
        hit = (state > 0) & (index < N) & (value.to(tl.int32, bitcast=True) == wanted)
        position = tl.minimum(position, tl.where(hit, index + 1, 2147483647))
    index = tl.min(position, 0)
    if index != 2147483647:
        tl.atomic_min(OUT, index, sem="relaxed")


def _amax_parallel_launch(entry, x, result, pointers, device):
    (partial, partial_run, constants, init, init_run, index, index_run) = entry
    knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    safe = (
        _launch_utils._FAST_LAUNCH_SUPPORTED
        and knobs is not None
        and (type(partial_run) is _launch_utils._NPU_LAUNCHER)
        and (type(init_run) is _launch_utils._NPU_LAUNCHER)
        and (type(index_run) is _launch_utils._NPU_LAUNCHER)
    )
    if safe:
        (h0, h1) = (knobs.launch_enter_hook, knobs.launch_exit_hook)
        hooks = (
            h0 is not None
            and (type(h0) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h0.calls))
            or (
                h1 is not None
                and (type(h1) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h1.calls))
            )
        )
        safe = (
            not hooks
            and (not getattr(partial_run, "compile_only", False))
            and (not getattr(partial_run, "enable_msprof_register_tensor", False))
            and (not getattr(partial.metadata, "debug_enabled", False))
            and (
                not getattr(
                    getattr(partial_run, "metadata", None), "debug_enabled", False
                )
            )
            and (not getattr(init_run, "compile_only", False))
            and (not getattr(init_run, "enable_msprof_register_tensor", False))
            and (not getattr(init.metadata, "debug_enabled", False))
            and (
                not getattr(getattr(init_run, "metadata", None), "debug_enabled", False)
            )
            and (not getattr(index_run, "compile_only", False))
            and (not getattr(index_run, "enable_msprof_register_tensor", False))
            and (not getattr(index.metadata, "debug_enabled", False))
            and (
                not getattr(
                    getattr(index_run, "metadata", None), "debug_enabled", False
                )
            )
        )
        if safe:
            pointer_type = _launch_utils._DevicePointer
            (xp, out) = (pointer_type(pointers[0]), pointer_type(pointers[1]))
            (xp.tensor, out.tensor) = (x, result)
            stream = _launch_utils._current_raw_stream(device)
            parts = constants[3]
            init_args = (
                1,
                1,
                1,
                stream,
                init.function,
                init.packed_metadata,
                None,
                None,
                None,
                out,
                constants[0],
            )
            partial_args = (
                parts,
                1,
                1,
                stream,
                partial.function,
                partial.packed_metadata,
                None,
                None,
                None,
                xp,
                out,
                *constants,
            )
            index_args = (
                parts,
                1,
                1,
                stream,
                index.function,
                index.packed_metadata,
                None,
                None,
                None,
                xp,
                out,
                *constants,
            )
            registered = init_run.launch(*init_args)
            _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
            registered = partial_run.launch(*partial_args)
            _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
            registered = index_run.launch(*index_args)
            _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
            return
    parts = constants[3]
    kernels = (
        (init, init_run, (1, 1, 1), (result,), (pointers[1],), (constants[0],)),
        (partial, partial_run, (parts, 1, 1), (x, result), pointers, constants),
        (index, index_run, (parts, 1, 1), (x, result), pointers, constants),
    )
    for compiled, launcher, grid, tensors, addresses, arguments in kernels:
        _launch_l1_entry(
            compiled, launcher, grid, tensors, addresses, arguments, device
        )


def _amax_parallel(n, x, incx, result, complex_data):
    fallback = _amax_small if n <= 65536 else _amax_large
    if any(
        type(tensor) is not torch.Tensor or tensor.is_conj() or tensor.is_neg()
        for tensor in (x, result)
    ):
        return fallback(n, x, incx, result, complex_data)
    expected = torch.complex64 if complex_data else torch.float32
    assert x.dtype == expected, f"x must be {expected}"
    assert result.dtype == torch.int32, "result must be int32"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert result.numel() >= 1, "result must not be empty"
    device = x.device
    assert device == result.device, "result must be on the same device"
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _amax_parallel(n, x, incx, result, complex_data)
    pointers = (x.data_ptr(), result.data_ptr())
    xp, op = pointers
    if (
        xp % 64
        or op % 16
        or (xp < op + 4 and op < xp + (1 + (n - 1) * incx) * x.element_size())
    ):
        return fallback(n, x, incx, result, complex_data)
    key = ("parallel", device.index, n, incx, complex_data)
    entry = _LAUNCH_CACHE.get(key)
    if entry is not None:
        return _amax_parallel_launch(entry, x, result, pointers, device.index)
    raw = torch.view_as_real(x) if complex_data else x
    factor = 2 if complex_data else 1
    init, _ = _amax_parallel_init[(1,)](result, n, num_warps=1, num_stages=1)
    partial, meta = _amax_parallel_max[lambda config: (config["PARTS"],)](
        raw, result, n, incx, factor, multibuffer=False
    )
    constants = (n, incx, factor, meta["PARTS"], meta["BLOCK"])
    index, _ = _amax_parallel_index[(meta["PARTS"],)](
        raw, result, *constants, num_warps=1, num_stages=1, multibuffer=False
    )
    for kernel in (init, partial, index):
        assert not getattr(kernel.metadata, "workspace_size", 0)
        assert not getattr(kernel.metadata, "lock_num", 0)
    if len(_LAUNCH_CACHE) >= 1024:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = (
        partial,
        partial.run,
        constants,
        init,
        init.run,
        index,
        index.run,
    )


def _amax_dispatch(n, x, incx, result, complex_data):
    minimum = False
    if not minimum and type(n) is int and type(incx) is int and n > 0 and 0 < incx <= 8:
        if (
            65536 <= n <= 131072
            and incx in (2, 5)
            and (complex_data or n > 65536)
            or not complex_data
            and 131072 < n <= 1048576
            and incx in (1, 2)
        ):
            return _amax_parallel(n, x, incx, result, complex_data)
        single_limit = 131072 if incx == 1 else 65536
        implementation = _amax_small if n <= single_limit else _amax_large
        return implementation(n, x, incx, result, complex_data)
    expected = torch.complex64 if complex_data else torch.float32
    assert x.dtype == expected, f"x must be {expected}"
    assert result.dtype == torch.int32, "result must be int32"
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    with torch_device_fn.device(x.device):
        pointer = torch.view_as_real(x) if complex_data else x
        optimized = not minimum and n > 16384 and (not complex_data or incx <= 8)
        gather = optimized and (complex_data or 1 < incx <= 8)
        single_limit = 16384 if not optimized or (incx > 1 and not gather) else 65536
        parts = 1 if n <= single_limit else min(triton.cdiv(n, 512), CORE_NUM)
        if parts == 1:
            values, indices = result, result
        else:
            values = torch.empty(parts, dtype=torch.float32, device=x.device)
            indices = torch.empty(parts, dtype=torch.int32, device=x.device)
        if not optimized:
            _amax_fallback_partial[(parts,)](
                pointer,
                values,
                indices,
                result,
                n,
                incx,
                complex_data,
                minimum,
                parts,
            )
        else:
            _amax_partial[(parts,)](
                pointer,
                values,
                indices,
                result,
                n,
                incx,
                complex_data,
                parts,
                n * incx * (2 if complex_data else 1) > 2147483647,
                gather,
                triton.next_power_of_2(incx * (2 if complex_data else 1)),
            )
        if parts > 1:
            _amax_finish[(1,)](
                values,
                indices,
                result,
                parts,
                minimum,
                triton.next_power_of_2(parts),
                num_warps=1,
                num_stages=1,
            )


def isamax(n, x, incx, result):
    _amax_dispatch(n, x, incx, result, False)


def icamax(n, x, incx, result):
    _amax_dispatch(n, x, incx, result, True)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_amax"),
    key=["N", "INCX", "COMPLEX", "PARTS", "WIDE", "GATHER", "OFFSET"],
    prune_configs_by={"early_config_prune": _prune_amax_configs},
)
@triton.jit
def _amax_complex_partial(
    X,
    WORK,
    OUT,
    N,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    OFFSET: tl.constexpr = 0,
):
    if OFFSET:
        X = (X.to(tl.uint64) + OFFSET).to(tl.pointer_type(tl.float32))
    lane = tl.arange(0, BLOCK)
    component = tl.arange(0, 2)
    best = tl.full((BLOCK,), -float("inf"), tl.float32)
    best_index = tl.full((BLOCK,), 2147483647, tl.int32)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    for start in range(first, end, BLOCK):
        index = start + lane
        active = index < N
        value = tl.load(
            X + 2 * index[:, None] * INCX + component[None, :],
            active[:, None],
            other=0.0,
        )
        magnitude = tl.sum(tl.abs(value), 1)
        take = active & (magnitude > best)
        best = tl.where(take, magnitude, best)
        best_index = tl.where(take, index, best_index)
    maximum = tl.max(best, 0)
    index = tl.min(tl.where(best == maximum, best_index, 2147483647), 0)
    tl.store(WORK + tl.program_id(0), maximum.to(tl.int32, bitcast=True))
    tl.store(WORK + PARTS + tl.program_id(0), index)


def _amax_small_kernel(complex_data, incx, pointer):
    if complex_data and incx == 1 and pointer % 64 == 0:
        return _amax_complex_single
    return _amax_single


@triton.jit
def _amax_complex_compute(
    X,
    WORK,
    OUT,
    N,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    component = tl.arange(0, 2)
    best = tl.full((BLOCK,), -float("inf"), tl.float32)
    best_relative = tl.full((BLOCK,), float("inf"), tl.float32)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    for start in range(first, end, BLOCK):
        index = start + lane
        active = index < N
        values = tl.load(
            X + 2 * index[:, None] * INCX + component[None, :],
            active[:, None],
            other=0.0,
        )
        magnitude = tl.sum(tl.abs(values), 1)
        take = active & (magnitude > best)
        best = tl.where(take, magnitude, best)
        relative = (start - first).to(tl.float32) + lane.to(tl.float32)
        best_relative = tl.where(take, relative, best_relative)
    maximum = tl.max(best, 0)
    relative = tl.min(tl.where(best == maximum, best_relative, float("inf")), 0)
    index = tl.where(
        relative == float("inf"), 2147483647, first + relative.to(tl.int32)
    )
    if PARTS == 1:
        tl.store(OUT, index + 1)
    else:
        tl.store(WORK + tl.program_id(0), maximum.to(tl.int32, bitcast=True))
        tl.store(WORK + PARTS + tl.program_id(0), index)


@libentry()
@triton.jit
def _amax_complex_single(
    X,
    OUT,
    N: tl.constexpr,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _amax_complex_compute(X, OUT, OUT, N, INCX, COMPLEX, 1, WIDE, GATHER, WIDTH, BLOCK)


@lru_cache(maxsize=None)
def _amax_stream_offset(device):
    if not torch.npu.get_device_name(device).startswith("Ascend910B"):
        return 0
    try:
        query = ctypes.CDLL("libruntime.so").rtGetL2CacheOffset
        query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int32
        offset = ctypes.c_uint64()
        return offset.value if query(device, ctypes.byref(offset)) == 0 else 0
    except (AttributeError, OSError):
        return 0


_AMAX_STREAM_CONFIGS = tuple(
    triton.Config({"BLOCK": block}, num_warps=1, num_stages=1)
    for block in (512, 1024, 2048, 4096)
)


def _amax_stream_prune(configs, args, **kwargs):
    if args["IX"] == 1:
        blocks = (2048, 4096)
    elif args["F"] == 2 and args["IX"] > 3:
        blocks = (512, 1024)
    else:
        blocks = (1024, 2048)
    return [c for c in configs if c.kwargs["BLOCK"] in blocks]


@libentry()
@libtuner(
    configs=_AMAX_STREAM_CONFIGS,
    key=["N", "IX", "F", "OFFSET", "PARTS"],
    prune_configs_by={"early_config_prune": _amax_stream_prune},
)
@triton.jit
def _amax_stream_partial(
    X,
    WORK,
    OUT,
    N,
    IX: tl.constexpr,
    F: tl.constexpr,
    OFFSET: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if OFFSET:
        X = (X.to(tl.uint64) + OFFSET).to(tl.pointer_type(tl.float32))
    lane = tl.arange(0, BLOCK)
    raw_lane = tl.arange(0, F * BLOCK * triton.next_power_of_2(IX))
    best = tl.full((BLOCK,), -float("inf"), tl.float32)
    position = tl.full((BLOCK,), 2147483647, tl.int32)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    for start in range(first, end, BLOCK):
        index = start + lane
        physical = F * start * IX + raw_lane
        limit = tl.minimum(F * ((N - 1) * IX + 1), F * (start + BLOCK) * IX)
        values = tl.load(X + physical, physical < limit, other=0.0)
        if F == 2:
            if IX == 1:
                (real, imag) = tl.split(values.reshape((BLOCK, 2)))
            else:
                real = tl.gather(values, 2 * lane * IX, 0)
                imag = tl.gather(values, 2 * lane * IX + 1, 0)
            magnitude = tl.abs(real) + tl.abs(imag)
        elif IX == 1:
            magnitude = tl.abs(values)
        else:
            magnitude = tl.abs(tl.gather(values, lane * IX, 0))
        take = (index < N) & (magnitude > best)
        best = tl.where(take, magnitude, best)
        value = index
        position = tl.where(take, value, position)
    maximum = tl.max(best, 0)
    index = tl.min(tl.where(best == maximum, position, 2147483647), 0)
    tl.store(WORK + tl.program_id(0), maximum.to(tl.int32, bitcast=True))
    tl.store(WORK + PARTS + tl.program_id(0), index)


def _amax_stream_launch(entry, finish, tensors, pointers, device, parts):
    kernel, run, arguments = entry
    last, lrun = finish
    knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    hooks = (
        getattr(knobs, "launch_enter_hook", None),
        getattr(knobs, "launch_exit_hook", None),
    )
    has_hooks = any(
        h is not None
        and (type(h) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h.calls))
        for h in hooks
    )
    safe = _launch_utils._FAST_LAUNCH_SUPPORTED and knobs is not None and not has_hooks
    for compiled, launcher in ((kernel, run), (last, lrun)):
        safe = (
            safe
            and type(launcher) is _launch_utils._NPU_LAUNCHER
            and not getattr(launcher, "compile_only", False)
            and not getattr(launcher, "enable_msprof_register_tensor", False)
            and not getattr(compiled.metadata, "debug_enabled", False)
            and not getattr(getattr(launcher, "metadata", None), "debug_enabled", False)
        )
    finish_args = (parts, 1 << (parts - 1).bit_length())
    if not safe:
        _launch_l1_entry(
            kernel, run, (parts, 1, 1), tensors, pointers, arguments, device
        )
        _launch_l1_entry(
            last, lrun, (1, 1, 1), tensors[1:], pointers[1:], finish_args, device
        )
        return
    wrapped = []
    for tensor, address in zip(tensors, pointers):
        pointer = _launch_utils._DevicePointer(address)
        pointer.tensor = tensor
        wrapped.append(pointer)
    stream = _launch_utils._current_raw_stream(device)
    first_call = (
        parts,
        1,
        1,
        stream,
        kernel.function,
        kernel.packed_metadata,
        None,
        None,
        None,
        *wrapped,
        *arguments,
    )
    last_call = (
        1,
        1,
        1,
        stream,
        last.function,
        last.packed_metadata,
        None,
        None,
        None,
        *wrapped[1:],
        *finish_args,
    )
    registered = run.launch(*first_call)
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
    registered = lrun.launch(*last_call)
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


def _amax_stream_reduce(
    n, x, incx, result, complex_data, work, parts, pointers, offset
):
    device = x.device.index
    factor = 2 if complex_data else 1
    key = ("stream_amax", device, n, incx, factor)
    entry = _LAUNCH_CACHE.get(key)
    finish_key = ("stream_amax_finish", device, parts)
    finish = _LAUNCH_CACHE.get(finish_key)
    if entry is not None and finish is not None:
        return _amax_stream_launch(
            entry, finish, (x, work, result), pointers, device, parts
        )
    raw = torch.view_as_real(x) if complex_data else x
    arguments = (n, incx, factor, offset, parts)
    compiled, meta = _amax_stream_partial[(parts,)](
        raw, work, result, *arguments, multibuffer=False
    )
    last, _ = _amax_packed_finish[(1,)](
        work, result, parts, triton.next_power_of_2(parts), num_warps=1, num_stages=1
    )
    if len(_LAUNCH_CACHE) >= 1022:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = (compiled, compiled.run, (*arguments, meta["BLOCK"]))
    _LAUNCH_CACHE[finish_key] = (last, last.run)

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import ctypes
from functools import lru_cache
from importlib import import_module

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


_LAUNCH_CACHE = {}


def _remember_l1_entry(key, entry):
    if len(_LAUNCH_CACHE) >= 2048:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = entry


def _prune_asum_configs(configs, args, **kwargs):
    limit = 512 if args["IX"] > 1 else 8192
    if args["N"] <= 1024:
        limit = min(limit, max(256, triton.next_power_of_2(args["N"])))
    return [c for c in configs if c.kwargs["BLOCK"] <= limit]


@libentry()
@triton.jit
def _asum_zero_result(OUT):
    tl.store(OUT, 0.0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_sum"),
    key=["N", "IX", "IY", "COMPLEX", "KIND", "CONJ", "PARTS"],
    reset_to_zero=["OUT"],
)
@triton.jit
def _asum_fallback_partial(
    X,
    Y,
    OUT,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    COMPLEX: tl.constexpr,
    KIND: tl.constexpr,
    CONJ: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    real = tl.zeros((BLOCK,), tl.float32)
    for start in range(tl.program_id(0) * BLOCK, N, PARTS * BLOCK):
        index = start + lane
        mask = index < N
        xo = index.to(tl.int64) * IX
        if COMPLEX:
            xr = tl.load(X + 2 * xo, mask, other=0.0)
            xi = tl.load(X + 2 * xo + 1, mask, other=0.0)
        else:
            xr = tl.load(X + xo, mask, other=0.0)
        real += tl.abs(xr)
        if COMPLEX:
            real += tl.abs(xi)
    total = tl.sum(real, 0)
    if PARTS == 1:
        tl.store(OUT, total)
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _reduce_fallback(
    name, family, kind, n, x, ix, result, y=None, iy=1, conjugate=False
):
    common = getattr(import_module("flag_blas.ops.level1." + family), name)
    common(0, x, ix, result)
    if n <= 0:
        return
    if ix <= 0:
        return
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert (
        result.numel() == 1 and result.dim() == 1
    ), "result must be a 1-element tensor"
    assert x.device == result.device, "result must be on the same device"
    assert x.numel() >= 1 + (n - 1) * ix, "x is too short"
    y = x
    pointer_x = torch.view_as_real(x) if x.is_complex() else x
    pointer_y = torch.view_as_real(y) if y.is_complex() else y
    output = torch.view_as_real(result) if result.is_complex() else result
    parts = 1 if n <= 16384 else min(triton.cdiv(n, 512), CORE_NUM)
    with torch_device_fn.device(x.device):
        output.zero_()
        _asum_fallback_partial[parts,](
            pointer_x, pointer_y, output, n, ix, iy, x.is_complex(), 0, conjugate, parts
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_asum"),
    key=["N", "IX", "COMPLEX", "PARTS", "WIDE", "GATHER", "WIDTH"],
    prune_configs_by={"early_config_prune": _prune_asum_configs},
    reset_to_zero=["OUT"],
)
@triton.jit
def _asum_partial(
    X,
    OUT,
    N,
    IX: tl.constexpr,
    COMPLEX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    accumulated = tl.zeros((BLOCK,), tl.float32)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    for start in range(first, end, BLOCK):
        index = start + lane
        offset = index.to(tl.int64) if WIDE else index
        if GATHER:
            component = tl.arange(0, BLOCK * WIDTH)
            first_index = start.to(tl.int64) if WIDE else start
            size = N.to(tl.int64) if WIDE else N
            physical = first_index * IX + component
            if COMPLEX:
                limit = (size - 2) * IX + 2
                selection = (lane >> 1) * (2 * IX) + (lane & 1)
            else:
                limit = (size - 1) * IX + 1
                selection = lane * IX
            values = tl.load(X + physical, physical < limit, other=0.0)
            value = tl.gather(values, selection, axis=0)
            accumulated += tl.where(index < N, tl.abs(value), 0.0)
        else:
            value = tl.load(X + offset * IX, index < N, other=0.0)
            accumulated += tl.abs(value)
    total = tl.sum(accumulated, 0)
    if PARTS == 1:
        tl.store(OUT, total)
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


_ASUM_SINGLE_LIMIT = 8192


def _asum_reduce(n, x, incx, result, complex_data):
    expected = torch.complex64 if complex_data else torch.float32
    assert (
        x.dtype == expected
    ), f"x must be {('complex64' if complex_data else 'float32')}"
    assert result.dtype == torch.float32, (
        "result for scasum must be float32"
        if complex_data
        else "result must be float32"
    )
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert (
        result.numel() == 1 and result.dim() == 1
    ), "result must be a 1-element tensor"
    device = x.device
    assert device == result.device, "result must be on the same device"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _asum_reduce(n, x, incx, result, complex_data)
    (x_pointer, out_pointer) = (x.data_ptr(), result.data_ptr())
    key = (
        device.index,
        n,
        incx,
        complex_data,
        x_pointer % 16 == 0,
        out_pointer % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(("asum", False, *key))
    parts = (
        entry[3]
        if entry is not None
        else 1
        if n <= _ASUM_SINGLE_LIMIT
        else min(triton.cdiv(n, 512), CORE_NUM)
    )
    if parts > 1:
        zero_key = (device.index, out_pointer % 16 == 0)
        zero = _LAUNCH_CACHE.get(("asum_zero", *zero_key))
        if zero is None:
            (compiled, _) = _asum_zero_result[(1,)](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("asum_zero", *zero_key), ((compiled, compiled.run)))
        else:
            _launch_l1_entry(
                *zero, (1, 1, 1), (result,), (out_pointer,), (), device.index
            )
    if entry is None:
        count = n * (2 if complex_data else 1)
        pointer = torch.view_as_real(x) if complex_data else x
        wide = count * incx > 2147483647
        gather = incx > 1
        width = triton.next_power_of_2(incx)

        kernel = _asum_cold_kernel(count, incx, complex_data, parts, wide)
        (compiled, meta) = kernel[parts,](
            pointer, result, count, incx, complex_data, parts, wide, gather, width
        )

        _remember_l1_entry(
            ("asum", False, *key),
            (
                (
                    compiled,
                    compiled.run,
                    (
                        count,
                        incx,
                        complex_data,
                        parts,
                        wide,
                        gather,
                        width,
                        meta["BLOCK"],
                    ),
                    parts,
                )
            ),
        )
        return
    (compiled, run, arguments, parts) = entry
    _launch_l1_entry(
        compiled,
        run,
        (parts, 1, 1),
        (x, result),
        (x_pointer, out_pointer),
        arguments,
        device.index,
    )


def sasum(n, x, incx, result):
    if _asum_stream_public_ok(n, x, incx, result):
        return _asum_stream_reduce(n, x, incx, result, 1)
    return _asum_group_reduce(n, x, incx, result, False)


def scasum(n, x, incx, result):
    if _asum_stream_public_ok(n, x, incx, result):
        return _asum_stream_reduce(n, x, incx, result, 2)
    return _asum_group_reduce(n, x, incx, result, True)


_SCASUM_COARSE_SAFE_SPAN = 2147483647 - 32768


_ASUM_COMPUTE = _asum_partial.jit_function


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_scasum_strided"),
    key=["N", "IX", "COMPLEX", "PARTS", "WIDE", "GATHER", "WIDTH"],
    reset_to_zero=["OUT"],
)
@triton.jit
def _asum_coarse_partial(
    X,
    OUT,
    N,
    IX: tl.constexpr,
    COMPLEX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _ASUM_COMPUTE(X, OUT, N, IX, COMPLEX, PARTS, WIDE, GATHER, WIDTH, BLOCK)


def _asum_cold_kernel(count, incx, complex_data, parts, wide):
    if (
        complex_data
        and 1 < incx <= 8
        and parts > 1
        and not wide
        and count * incx < _SCASUM_COARSE_SAFE_SPAN
        and (count + parts - 1) // parts >= 2048
    ):
        return _asum_coarse_partial
    return _asum_partial


_ASUM_GROUP_CONFIGS = [
    config
    for config in runtime.get_tuned_config("ascend_l1_asum_group")
    if config.kwargs["PARTS"] <= CORE_NUM
]


def _prune_asum_group_configs(configs, args, **kw):
    n, ix, width = args["N"], args["IX"], args["WIDTH"]
    single = n <= 131072 and n * ix * 4 <= 2**21
    parts = (
        (1,)
        if single
        else tuple(
            p
            for p in (8, 16, 20, 40)
            if p <= CORE_NUM and 4096 <= (n + p - 1) // p <= 262144
        )
    )
    if not parts:
        parts = (max(p for p in (1, 8, 16, 20, 40) if p <= CORE_NUM),)
    upper = 4096 if ix == 1 else 2048 if width <= 4 else 1024
    blocks = (upper // 2, upper)
    return [
        c for c in configs if c.kwargs["PARTS"] in parts and c.kwargs["BLOCK"] in blocks
    ]


@triton.jit
def _asum_group_read(
    X,
    start,
    N: tl.constexpr,
    IX: tl.constexpr,
    COMPLEX: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    index = start + lane
    if IX > 1:
        component = tl.arange(0, BLOCK * WIDTH)
        physical = start * IX + component
        limit = (N - 2) * IX + 2 if COMPLEX else (N - 1) * IX + 1
        raw = tl.load(X + physical, physical < limit, other=0.0)
        select = (lane >> 1) * (2 * IX) + (lane & 1) if COMPLEX else lane * IX
        value = tl.gather(raw, select, axis=0)
        return tl.where(index < N, value, 0.0)
    return tl.load(X + index, index < N, other=0.0)


@libentry()
@libtuner(
    configs=_ASUM_GROUP_CONFIGS,
    key=["N", "IX", "COMPLEX", "KIND", "WIDTH"],
    prune_configs_by={"early_config_prune": _prune_asum_group_configs},
    reset_to_zero=["OUT"],
)
@triton.jit
def _asum_group_partial(
    X,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    COMPLEX: tl.constexpr,
    KIND: tl.constexpr,
    WIDTH: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    acc0 = tl.zeros((BLOCK,), tl.float32)
    acc1 = tl.zeros((BLOCK,), tl.float32)
    chunk = tl.cdiv(N, PARTS * 2 * BLOCK) * (2 * BLOCK)
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    for start in range(first, end, 2 * BLOCK):
        a = _asum_group_read(X, start, N, IX, COMPLEX, WIDTH, BLOCK)
        b = _asum_group_read(X, start + BLOCK, N, IX, COMPLEX, WIDTH, BLOCK)
        acc0 += tl.abs(a)
        acc1 += tl.abs(b)
    total = tl.sum(acc0 + acc1, 0)
    if PARTS == 1:
        tl.store(OUT, total)
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _asum_group_reduce(n, x, incx, result, complex_data):
    name = "scasum" if complex_data else "sasum"
    if not (
        CORE_NUM >= 8
        and type(n) is int
        and n > _ASUM_SINGLE_LIMIT
        and type(incx) is int
        and 0 < incx <= 8
    ):
        return _reduce(name, "asum", 0, n, x, incx, result)
    count = n * (2 if complex_data else 1)
    single = count <= 131072 and count * incx * 4 <= 2**21
    parallel = not complex_data and incx > 1 and 131072 < count <= CORE_NUM * 262144
    expected = torch.complex64 if complex_data else torch.float32
    if not (
        (single or parallel)
        and type(x) is torch.Tensor
        and type(result) is torch.Tensor
        and x.dtype == expected
        and result.dtype == torch.float32
        and x.device.type == "npu"
        and not x.is_conj()
        and not x.is_neg()
        and not result.is_conj()
        and not result.is_neg()
        and (count + CORE_NUM * 8192) * incx < 2147483647
    ):
        return _reduce(name, "asum", 0, n, x, incx, result)
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert (
        result.numel() == 1 and result.dim() == 1
    ), "result must be a 1-element tensor"
    assert x.device == result.device, "result must be on the same device"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    xp, op = x.data_ptr(), result.data_ptr()
    if (
        xp % 16 != 0
        or op % 16 != 0
        or (xp < op + 4 and op < xp + x.numel() * x.element_size())
    ):
        return _reduce(name, "asum", 0, n, x, incx, result)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _asum_group_reduce(n, x, incx, result, complex_data)
    key = (
        "asum",
        True,
        device.index,
        n,
        incx,
        complex_data,
        xp % 16 == 0,
        op % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        if not single:
            _asum_group_zero(result, device.index)
        pointer = torch.view_as_real(x) if complex_data else x
        width = triton.next_power_of_2(incx)
        compiled, meta = _asum_group_kernel(complex_data, single, width)[
            lambda cfg: (cfg["PARTS"],)
        ](pointer, result, count, incx, complex_data, 0, width)
        parts = meta["PARTS"]
        arguments = (count, incx, complex_data, 0, width, parts, meta["BLOCK"])
        _remember_l1_entry(key, (compiled, compiled.run, arguments, parts))
    else:
        compiled, run, arguments, parts = entry
        if parts > 1:
            _asum_group_zero(result, device.index)
        _launch_l1_entry(
            compiled, run, (parts, 1, 1), (x, result), (xp, op), arguments, device.index
        )


def _asum_group_zero(result, device):
    pointer = result.data_ptr()
    key = ("asum_zero", device, pointer % 16 == 0)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        compiled, _ = _asum_zero_result[(1,)](result, num_warps=1, num_stages=1)
        _remember_l1_entry(key, (compiled, compiled.run))
    else:
        _launch_l1_entry(*entry, (1, 1, 1), (result,), (pointer,), (), device)


def _prune_asum_tiled_configs(configs, args, **kwargs):
    n, ix, width = args["N"], args["IX"], args["WIDTH"]
    single = n <= 131072 and n * ix * 4 <= 2**21
    upper = 4096 if ix == 1 else 2048 if width <= 4 else 1024
    parts = (
        (1,)
        if single
        else tuple(
            p for p in (20, 40) if p <= CORE_NUM and 4096 <= (n + p - 1) // p <= 262144
        )
    )
    if not parts:
        parts = (40,)
    blocks = ((upper // 2, upper) if width <= 4 else (upper,)) if single else (upper,)
    if single and args["COMPLEX"] and 1 < width <= 4:
        blocks = (upper,)
    return [
        c for c in configs if c.kwargs["PARTS"] in parts and c.kwargs["BLOCK"] in blocks
    ]


_ASUM_TILED_CONFIGS = runtime.get_tuned_config("ascend_l1_asum_tiled")


_asum_tiled_partial = (
    libentry()(
        libtuner(
            configs=_ASUM_TILED_CONFIGS,
            key=["N", "IX", "COMPLEX", "KIND", "WIDTH"],
            prune_configs_by={"early_config_prune": _prune_asum_tiled_configs},
            reset_to_zero=["OUT"],
        )(_asum_group_partial.jit_function)
    )
    if CORE_NUM >= 40
    else _asum_group_partial
)


def _asum_group_kernel(complex_data, single, width):
    if complex_data:
        return _asum_tiled_partial if single and 1 < width <= 4 else _asum_group_partial
    return _asum_group_partial if single and width <= 4 else _asum_tiled_partial


def _reduce(name, family, kind, n, x, ix, result, y=None, iy=1, conjugate=False):
    if (
        kind == 0
        and type(n) is int
        and type(ix) is int
        and name in ("sasum", "scasum")
        and (
            ix == 1
            or (1 < ix <= 8 and n * ix * (2 if name == "scasum" else 1) < 2147483647)
        )
    ):
        return _asum_reduce(n, x, ix, result, name == "scasum")
    return _reduce_fallback(name, family, kind, n, x, ix, result, y, iy, conjugate)


@lru_cache(maxsize=None)
def _asum_stream_offset(device):
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


def _asum_stream_launch_pair(zero, entry, tensors, pointers, device):
    (z, zrun) = zero
    (compiled, run, arguments, parts) = entry
    knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    (h0, h1) = (
        getattr(knobs, "launch_enter_hook", None),
        getattr(knobs, "launch_exit_hook", None),
    )
    has_hooks = (
        h0 is not None
        and (type(h0) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h0.calls))
        or (
            h1 is not None
            and (type(h1) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h1.calls))
        )
    )
    safe = (
        _launch_utils._FAST_LAUNCH_SUPPORTED and knobs is not None and (not has_hooks)
    )
    safe = (
        safe
        and (
            type(zrun) is _launch_utils._NPU_LAUNCHER
            and (not getattr(zrun, "compile_only", False))
            and (not getattr(zrun, "enable_msprof_register_tensor", False))
            and (not getattr(z.metadata, "debug_enabled", False))
            and (not getattr(getattr(zrun, "metadata", None), "debug_enabled", False))
        )
        and (
            type(run) is _launch_utils._NPU_LAUNCHER
            and (not getattr(run, "compile_only", False))
            and (not getattr(run, "enable_msprof_register_tensor", False))
            and (not getattr(compiled.metadata, "debug_enabled", False))
            and (not getattr(getattr(run, "metadata", None), "debug_enabled", False))
        )
    )
    if not safe:
        _launch_l1_entry(z, zrun, (1, 1, 1), (tensors[1],), (pointers[1],), (), device)
        _launch_l1_entry(
            compiled, run, (parts, 1, 1), tensors, pointers, arguments, device
        )
        return
    stream = _launch_utils._current_raw_stream(device)
    pointer_type = _launch_utils._DevicePointer
    p0 = pointer_type(pointers[0])
    p0.tensor = tensors[0]
    p1 = pointer_type(pointers[1])
    p1.tensor = tensors[1]
    zero_args = (
        1,
        1,
        1,
        stream,
        z.function,
        z.packed_metadata,
        None,
        None,
        None,
        p1,
    )
    compute_args = (
        parts,
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        p0,
        p1,
        *arguments,
    )
    registered = zrun.launch(*zero_args)
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
    registered = run.launch(*compute_args)
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


_ASUM_STREAM_CONFIGS = tuple(
    triton.Config({"BLOCK": block}, num_warps=1, num_stages=1)
    for block in (512, 1024, 2048, 4096, 8192)
)


def _asum_stream_prune(configs, args, **kwargs):
    if args["IX"] == 1:
        blocks = (2048, 4096, 8192)
    elif args["F"] == 2 and args["IX"] > 3:
        blocks = (512, 1024)
    else:
        blocks = (1024, 2048)
    return [cfg for cfg in configs if cfg.kwargs["BLOCK"] in blocks]


@triton.jit
def _asum_stream_read(
    X, start, COUNT, IX: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    physical = start * IX + tl.arange(0, BLOCK * triton.next_power_of_2(IX))
    limit = tl.minimum((COUNT - F) * IX + F, (start + BLOCK) * IX)
    raw = tl.load(X + physical, physical < limit, other=0.0)
    if IX != 1:
        raw = tl.gather(raw, (lane // F) * (IX * F) + (lane % F), 0)
    return tl.where(start + lane < COUNT, raw, 0.0)


@libentry()
@libtuner(
    configs=_ASUM_STREAM_CONFIGS,
    key=["N", "IX", "F", "OFFSET", "PARTS"],
    prune_configs_by={"early_config_prune": _asum_stream_prune},
    reset_to_zero=["OUT"],
)
@triton.jit
def _asum_stream_kernel(
    X,
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
    count = N if F == 2 and IX > 1 else N * F
    acc = tl.zeros((BLOCK,), tl.float32)
    chunk = tl.cdiv(count, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, count)
    for start in range(first, end, BLOCK):
        if F == 2 and IX > 1:
            real, imag = _asum_stream_complex_read(X, start, N, IX, BLOCK)
            acc += tl.where(
                start + tl.arange(0, BLOCK) < N, tl.abs(real) + tl.abs(imag), 0.0
            )
        else:
            value = _asum_stream_read(X, start, count, IX, F, BLOCK)
            acc += tl.abs(value)
    total = tl.sum(acc, 0)
    if PARTS == 1:
        tl.store(OUT, total)
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _asum_stream_public_ok(n, x, incx, result):
    return (
        type(n) is int
        and (
            n >= 1048576
            or n >= 65536
            and type(x) is torch.Tensor
            and x.dtype == torch.complex64
            and type(incx) is int
            and 1 < incx <= 5
        )
        and type(incx) is int
        and 0 < incx <= 5
        and all(
            type(t) is torch.Tensor and not t.is_conj() and not t.is_neg()
            for t in (x, result)
        )
    )


def _asum_stream_reduce(n, x, incx, result, factor):
    expected = torch.float32 if factor == 1 else torch.complex64
    assert x.dtype == expected and result.dtype == torch.float32
    assert x.dim() == 1 and x.is_contiguous()
    assert result.dim() == 1 and result.numel() == 1
    device = x.device
    assert device == result.device
    assert x.numel() >= 1 + (n - 1) * incx
    xp, op = x.data_ptr(), result.data_ptr()
    size = factor * 4
    if (
        (n * factor + (CORE_NUM + 2) * 8192) * incx >= 2147483647
        or xp % 64
        or op % 16
        or (xp < op + 4 and op < xp + ((n - 1) * incx + 1) * size)
    ):
        return _asum_group_reduce(n, x, incx, result, factor == 2)
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _asum_stream_reduce(n, x, incx, result, factor)
    key = ("stream_asum", device.index, n, incx, factor)
    entry = _LAUNCH_CACHE.get(key)
    zero_key = ("stream_zero", device.index)
    zero = _LAUNCH_CACHE.get(zero_key)
    if entry is not None and zero is not None:
        return _asum_stream_launch_pair(
            zero, entry, (x, result), (xp, op), device.index
        )
    offset = _asum_stream_offset(device.index)
    if not offset:
        return _asum_group_reduce(n, x, incx, result, factor == 2)
    raw = torch.view_as_real(x) if factor == 2 else x
    z, _ = _asum_zero_result[(1,)](result, num_warps=1, num_stages=1)
    parts = min(CORE_NUM, (n + 511) // 512)
    arguments = (n, incx, factor, offset, parts)
    compiled, meta = _asum_stream_kernel[(parts,)](
        raw, result, *arguments, multibuffer=False
    )
    _remember_l1_entry(zero_key, (z, z.run))
    _remember_l1_entry(
        key, (compiled, compiled.run, (*arguments, meta["BLOCK"]), parts)
    )


@triton.jit
def _asum_stream_complex_read(X, start, N, IX: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    physical = 2 * start * IX + tl.arange(0, 2 * BLOCK * triton.next_power_of_2(IX))
    limit = tl.minimum(2 * ((N - 1) * IX + 1), 2 * (start + BLOCK) * IX)
    raw = tl.load(X + physical, physical < limit, other=0.0)
    real = tl.gather(raw, 2 * lane * IX, 0)
    imag = tl.gather(raw, 2 * lane * IX + 1, 0)
    return real, imag

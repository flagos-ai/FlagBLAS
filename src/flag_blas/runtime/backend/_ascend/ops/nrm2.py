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


def _prune_nrm2_sum_configs(configs, args, **kwargs):
    limit = 512 if args["IX"] > 1 else 8192
    if args["N"] <= 1024:
        limit = min(limit, max(256, triton.next_power_of_2(args["N"])))
    return [c for c in configs if c.kwargs["BLOCK"] <= limit]


@libentry()
@triton.jit
def _nrm2_zero_result(OUT):
    tl.store(OUT, 0.0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_sum"),
    key=["N", "IX", "IY", "COMPLEX", "KIND", "CONJ", "PARTS"],
    reset_to_zero=["OUT"],
)
@triton.jit
def _nrm2_fallback_partial(
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
        real += xr * xr
        if COMPLEX:
            real += xi * xi
    total = tl.sum(real, 0)
    if PARTS == 1:
        total = tl.sqrt(total)
        tl.store(OUT, total)
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


@libentry()
@triton.jit
def _finish_sqrt(OUT):
    tl.store(OUT, tl.sqrt(tl.load(OUT)))


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
        _nrm2_fallback_partial[parts,](
            pointer_x, pointer_y, output, n, ix, iy, x.is_complex(), 1, conjugate, parts
        )
        if parts > 1:
            _finish_sqrt[1,](output, num_warps=1, num_stages=1)


def _prune_nrm2_configs(configs, args, **kwargs):
    configs = _prune_nrm2_sum_configs(configs, args, **kwargs)
    if args["PARTS"] != 1:
        return configs
    upper = min(8192, max(256, triton.next_power_of_2(args["N"])))
    lower = min(4096, upper)
    return [c for c in configs if lower <= c.kwargs["BLOCK"] <= upper]


_NRM2_SINGLE_LIMIT = 131072


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_nrm2"),
    key=["N", "IX", "PARTS", "WIDE"],
    prune_configs_by={"early_config_prune": _prune_nrm2_configs},
    reset_to_zero=["OUT"],
)
@triton.jit
def _snrm2_partial(
    X,
    OUT,
    N,
    IX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
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
        value = tl.load(X + offset, index < N, other=0.0)
        accumulated += value * value
    total = tl.sum(accumulated, 0)
    if PARTS == 1:
        tl.store(OUT, tl.sqrt(total))
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _snrm2_contiguous(n, x, result):
    assert x.dtype == torch.float32
    assert result.dtype == torch.float32
    assert x.dim() == 1 and x.is_contiguous() and (x.numel() >= n)
    assert result.dim() == 1 and result.numel() == 1 and (result.device == x.device)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _snrm2_contiguous(n, x, result)
    (xp, op) = (x.data_ptr(), result.data_ptr())
    key = (device.index, n, False, xp % 16 == 0, op % 16 == 0)
    small_key = (device.index, op % 16 == 0)
    parts = 1 if n <= _NRM2_SINGLE_LIMIT else min(triton.cdiv(n, 512), CORE_NUM)
    if parts > 1:
        zero = _LAUNCH_CACHE.get(("nrm2_zero", *small_key))
        if zero is None:
            (compiled, _) = _nrm2_zero_result[1,](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("nrm2_zero", *small_key), ((compiled, compiled.run)))
        else:
            _launch_l1_entry(*zero, (1, 1, 1), (result,), (op,), (), device.index)
    entry = _LAUNCH_CACHE.get(("nrm2", *key))
    if entry is None:
        count = n * 1
        pointer = x
        arguments = (count, 1, parts, count > 2147483647)
        (compiled, meta) = _snrm2_partial[parts,](pointer, result, *arguments)

        _remember_l1_entry(
            ("nrm2", *key), ((compiled, compiled.run, (*arguments, meta["BLOCK"])))
        )
    else:
        _launch_l1_entry(
            entry[0],
            entry[1],
            (parts, 1, 1),
            (x, result),
            (xp, op),
            entry[2],
            device.index,
        )
    if parts > 1:
        finish = _LAUNCH_CACHE.get(("sqrt", *small_key))
        if finish is None:
            (compiled, _) = _finish_sqrt[1,](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("sqrt", *small_key), ((compiled, compiled.run)))
        else:
            _launch_l1_entry(*finish, (1, 1, 1), (result,), (op,), (), device.index)


_SNRM2_STRIDED_SINGLE_LIMIT = 131072


_SNRM2_STRIDED_SAFE_SPAN = 2147483647 - 32768


def _prune_snrm2_strided_configs(configs, args, **kwargs):
    per_part = triton.cdiv(args["N"], args["PARTS"])
    safe_limit = 4096 if args["WIDTH"] <= 4 else 2048
    upper = min(safe_limit, max(256, triton.next_power_of_2(per_part)))
    lower = min(2048, upper)
    return [c for c in configs if lower <= c.kwargs["BLOCK"] <= upper]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_nrm2_strided"),
    key=["N", "IX", "PARTS", "WIDTH"],
    reset_to_zero=["OUT"],
    prune_configs_by={"early_config_prune": _prune_snrm2_strided_configs},
)
@triton.jit
def _snrm2_strided_partial(
    X,
    OUT,
    N,
    IX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    component = tl.arange(0, BLOCK * WIDTH)
    accumulated = tl.zeros((BLOCK,), tl.float32)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    for start in range(first, end, BLOCK):
        physical = start * IX + component
        limit = (N - 1) * IX + 1
        selection = lane * IX
        values = tl.load(X + physical, physical < limit, other=0.0)
        chosen = tl.gather(values, selection, axis=0)
        accumulated += tl.where(start + lane < N, chosen * chosen, 0.0)
    total = tl.sum(accumulated, 0)
    if PARTS == 1:
        tl.store(OUT, tl.sqrt(total))
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _snrm2_strided(n, x, incx, result):
    device = x.device
    xp, op = x.data_ptr(), result.data_ptr()
    key = device.index, n, incx, xp % 16 == 0, op % 16 == 0
    entry = _LAUNCH_CACHE.get(("snrm2_strided", *key))
    parts = (
        1 if n <= _SNRM2_STRIDED_SINGLE_LIMIT else min(triton.cdiv(n, 512), CORE_NUM)
    )
    if parts > 1:
        zero_key = device.index, op % 16 == 0
        zero = _LAUNCH_CACHE.get(("nrm2_zero", *zero_key))
        if zero is None:
            compiled, _ = _nrm2_zero_result[(1,)](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("nrm2_zero", *zero_key), (compiled, compiled.run))
        else:
            _launch_l1_entry(*zero, (1, 1, 1), (result,), (op,), (), device.index)
    if entry is None:
        width = triton.next_power_of_2(incx)
        arguments = n, incx, parts, width
        compiled, meta = _snrm2_strided_partial[(parts,)](x, result, *arguments)

        _remember_l1_entry(
            ("snrm2_strided", *key),
            (
                (
                    compiled,
                    compiled.run,
                    (*arguments, meta["BLOCK"]),
                )
            ),
        )
    else:
        _launch_l1_entry(
            entry[0],
            entry[1],
            (parts, 1, 1),
            (x, result),
            (xp, op),
            entry[2],
            device.index,
        )
    if parts > 1:
        finish_key = device.index, op % 16 == 0
        finish = _LAUNCH_CACHE.get(("sqrt", *finish_key))
        if finish is None:
            compiled, _ = _finish_sqrt[(1,)](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("sqrt", *finish_key), (compiled, compiled.run))
        else:
            _launch_l1_entry(*finish, (1, 1, 1), (result,), (op,), (), device.index)


_SCNRM2_CONTIG_SINGLE_LIMIT = 131072


_SCNRM2_STRIDE_SINGLE_LIMIT = 8192


_SCNRM2_SAFE_SPAN = 2147483647 - CORE_NUM * 8192 - 65536


def _prune_scnrm2_configs(configs, args, **kwargs):
    per_part = triton.cdiv(args["N"], args["PARTS"])
    safe = 8192 if args["IX"] == 1 else 4096 if args["WIDTH"] <= 4 else 2048
    upper = min(safe, max(256, triton.next_power_of_2(per_part)))
    lower = min(4096 if args["IX"] == 1 else 2048, upper)
    if args["WIDE"]:
        upper = min(upper, 256)
        lower = min(lower, upper)
    return [c for c in configs if lower <= c.kwargs["BLOCK"] <= upper]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_scnrm2"),
    key=["N", "IX", "PARTS", "WIDE", "WIDTH"],
    prune_configs_by={"early_config_prune": _prune_scnrm2_configs},
    reset_to_zero=["OUT"],
)
@triton.jit
def _scnrm2_flat_partial(
    X,
    OUT,
    N,
    IX: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
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
        if IX > 1:
            component = tl.arange(0, BLOCK * WIDTH)
            first_index = start.to(tl.int64) if WIDE else start
            size = N.to(tl.int64) if WIDE else N
            physical = first_index * IX + component
            values = tl.load(X + physical, physical < (size - 2) * IX + 2, other=0.0)
            selection = (lane >> 1) * (2 * IX) + (lane & 1)
            value = tl.gather(values, selection, axis=0)
            accumulated += tl.where(index < N, value * value, 0.0)
        else:
            value = tl.load(X + offset, index < N, other=0.0)
            accumulated += value * value
    total = tl.sum(accumulated, 0)
    if PARTS == 1:
        tl.store(OUT, tl.sqrt(total))
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _scnrm2_reduce(n, x, incx, result):
    assert x.dtype == torch.complex64
    assert result.dtype == torch.float32
    assert x.dim() == 1 and x.is_contiguous()
    assert result.dim() == 1 and result.numel() == 1
    assert x.device == result.device
    assert x.numel() >= 1 + (n - 1) * incx
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _scnrm2_reduce(n, x, incx, result)
    xp, op = x.data_ptr(), result.data_ptr()
    key = (device.index, n, incx, xp % 16 == 0, op % 16 == 0)
    entry = _LAUNCH_CACHE.get(("scnrm2", *key))
    if entry is None:
        single_limit = (
            _SCNRM2_CONTIG_SINGLE_LIMIT if incx == 1 else _SCNRM2_STRIDE_SINGLE_LIMIT
        )
        parts = (
            1
            if n <= single_limit or (incx > 1 and n * incx * 8 <= 2**21)
            else min(((n + 511) // 512), CORE_NUM)
        )
        medium = (
            incx == 1
            and parts > 1
            and 2**17 < ((2 * n + parts - 1) // parts) <= 2**18
        )
        if medium:
            parts = max(1, parts // 2)
    else:
        parts = entry[2][2]
    small_key = (device.index, op % 16 == 0)
    if parts > 1:
        zero = _LAUNCH_CACHE.get(("nrm2_zero", *small_key))
        if zero is None:
            compiled, _ = _nrm2_zero_result[(1,)](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("nrm2_zero", *small_key), ((compiled, compiled.run)))
        else:
            _launch_l1_entry(*zero, (1, 1, 1), (result,), (op,), (), device.index)
    if entry is None:
        arguments = (2 * n, incx, parts, False, triton.next_power_of_2(incx))
        kernel = _scnrm2_medium_partial if medium else _scnrm2_flat_partial
        compiled, meta = kernel[(parts,)](torch.view_as_real(x), result, *arguments)

        _remember_l1_entry(
            ("scnrm2", *key), ((compiled, compiled.run, (*arguments, meta["BLOCK"])))
        )
    else:
        _launch_l1_entry(
            entry[0],
            entry[1],
            (parts, 1, 1),
            (x, result),
            (xp, op),
            entry[2],
            device.index,
        )
    if parts > 1:
        finish = _LAUNCH_CACHE.get(("sqrt", *small_key))
        if finish is None:
            compiled, _ = _finish_sqrt[(1,)](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("sqrt", *small_key), ((compiled, compiled.run)))
        else:
            _launch_l1_entry(*finish, (1, 1, 1), (result,), (op,), (), device.index)


def _snrm2_original(n, x, incx, result):
    if not (
        type(n) is int
        and type(incx) is int
        and n > 0
        and 1 < incx <= 8
        and n * incx <= _SNRM2_STRIDED_SAFE_SPAN
    ):
        return _reduce("snrm2", "nrm2", 1, n, x, incx, result)
    assert x.dtype == torch.float32
    assert result.dtype == torch.float32
    assert x.dim() == 1 and x.is_contiguous()
    assert result.dim() == 1 and result.numel() == 1
    assert x.device == result.device
    assert x.numel() >= (n - 1) * incx + 1
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _snrm2_strided(n, x, incx, result)
    return _snrm2_strided(n, x, incx, result)


def _scnrm2_original(n, x, incx, result):
    if (
        type(n) is int
        and type(incx) is int
        and n > 0
        and 0 < incx <= 8
        and 2 * n * incx <= _SCNRM2_SAFE_SPAN
    ):
        return _scnrm2_reduce(n, x, incx, result)
    return _reduce("scnrm2", "nrm2", 1, n, x, incx, result)


_scnrm2_medium_partial = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("ascend_l1_scnrm2_contiguous_medium"),
        key=["N", "IX", "PARTS", "WIDE", "WIDTH"],
        reset_to_zero=["OUT"],
    )(_scnrm2_flat_partial.jit_function)
)


def _reduce(name, family, kind, n, x, ix, result, y=None, iy=1, conjugate=False):
    if kind == 1 and name == "snrm2" and type(n) is int and n > 0 and ix == 1:
        return _snrm2_contiguous(n, x, result)
    return _reduce_fallback(name, family, kind, n, x, ix, result, y, iy, conjugate)


@lru_cache(maxsize=None)
def _nrm2_stream_offset(device):
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


_NRM2_STREAM_CONFIGS = tuple(
    triton.Config({"BLOCK": block}, num_warps=1, num_stages=1)
    for block in (512, 1024, 2048, 4096, 8192)
)


def _nrm2_stream_prune(configs, args, **kwargs):
    if args["IX"] == 1:
        blocks = (2048, 4096, 8192)
    elif args["F"] == 2 and args["IX"] > 3:
        blocks = (512, 1024)
    else:
        blocks = (1024, 2048)
    return [cfg for cfg in configs if cfg.kwargs["BLOCK"] in blocks]


@triton.jit
def _nrm2_stream_read(
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
    configs=_NRM2_STREAM_CONFIGS,
    key=["N", "IX", "F", "OFFSET", "PARTS"],
    prune_configs_by={"early_config_prune": _nrm2_stream_prune},
    reset_to_zero=["OUT"],
)
@triton.jit
def _nrm2_stream_kernel(
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
            real, imag = _nrm2_stream_complex_read(X, start, N, IX, BLOCK)
            acc += tl.where(
                start + tl.arange(0, BLOCK) < N, real * real + imag * imag, 0.0
            )
        else:
            value = _nrm2_stream_read(X, start, count, IX, F, BLOCK)
            acc += value * value
    total = tl.sum(acc, 0)
    if PARTS == 1:
        tl.store(OUT, total)
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _nrm2_stream_public_ok(n, x, incx, result):
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


def _nrm2_stream_reduce(n, x, incx, result, factor):
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
        return _nrm2_original(n, x, incx, result, factor)
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _nrm2_stream_reduce(n, x, incx, result, factor)
    key = ("stream_nrm2", device.index, n, incx, factor)
    entry = _LAUNCH_CACHE.get(key)
    zero_key = ("stream_zero", device.index)
    zero = _LAUNCH_CACHE.get(zero_key)
    finish_key = ("stream_sqrt", device.index)
    finish = _LAUNCH_CACHE.get(finish_key)
    if entry is not None and zero is not None and finish is not None:
        return _nrm2_stream_launch(
            zero, entry, finish, (x, result), (xp, op), device.index
        )
    offset = _nrm2_stream_offset(device.index)
    if not offset:
        return _nrm2_original(n, x, incx, result, factor)
    raw = torch.view_as_real(x) if factor == 2 else x
    z, _ = _nrm2_zero_result[(1,)](result, num_warps=1, num_stages=1)
    parts = min(CORE_NUM, (n + 511) // 512)
    arguments = (n, incx, factor, offset, parts)
    compiled, meta = _nrm2_stream_kernel[(parts,)](
        raw, result, *arguments, multibuffer=False
    )
    last, _ = _finish_sqrt[(1,)](result, num_warps=1, num_stages=1)
    _remember_l1_entry(finish_key, (last, last.run))
    _remember_l1_entry(zero_key, (z, z.run))
    _remember_l1_entry(
        key, (compiled, compiled.run, (*arguments, meta["BLOCK"]), parts)
    )


def _nrm2_stream_launch(zero, entry, finish, tensors, pointers, device):
    (z, zrun) = zero
    (compiled, run, arguments, parts) = entry
    (last, lrun) = finish
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
        and (
            type(lrun) is _launch_utils._NPU_LAUNCHER
            and (not getattr(lrun, "compile_only", False))
            and (not getattr(lrun, "enable_msprof_register_tensor", False))
            and (not getattr(last.metadata, "debug_enabled", False))
            and (not getattr(getattr(lrun, "metadata", None), "debug_enabled", False))
        )
    )
    if not safe:
        _launch_l1_entry(z, zrun, (1, 1, 1), (tensors[1],), (pointers[1],), (), device)
        _launch_l1_entry(
            compiled, run, (parts, 1, 1), tensors, pointers, arguments, device
        )
        _launch_l1_entry(
            last, lrun, (1, 1, 1), (tensors[1],), (pointers[1],), (), device
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
    finish_args = (
        1,
        1,
        1,
        stream,
        last.function,
        last.packed_metadata,
        None,
        None,
        None,
        p1,
    )
    registered = zrun.launch(*zero_args)
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
    registered = run.launch(*compute_args)
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
    registered = lrun.launch(*finish_args)
    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


def _nrm2_original(n, x, incx, result, factor):
    return (_snrm2_original if factor == 1 else _scnrm2_original)(n, x, incx, result)


def snrm2(n, x, incx, result):
    if _nrm2_stream_public_ok(n, x, incx, result):
        return _nrm2_stream_reduce(n, x, incx, result, 1)
    return _snrm2_original(n, x, incx, result)


def scnrm2(n, x, incx, result):
    if _nrm2_stream_public_ok(n, x, incx, result):
        return _nrm2_stream_reduce(n, x, incx, result, 2)
    return _scnrm2_original(n, x, incx, result)


@triton.jit
def _nrm2_stream_complex_read(X, start, N, IX: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    physical = 2 * start * IX + tl.arange(0, 2 * BLOCK * triton.next_power_of_2(IX))
    limit = tl.minimum(2 * ((N - 1) * IX + 1), 2 * (start + BLOCK) * IX)
    raw = tl.load(X + physical, physical < limit, other=0.0)
    real = tl.gather(raw, 2 * lane * IX, 0)
    imag = tl.gather(raw, 2 * lane * IX + 1, 0)
    return real, imag

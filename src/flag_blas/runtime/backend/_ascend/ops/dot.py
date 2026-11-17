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


@libentry()
@triton.jit
def _dot_zero_result(OUT):
    tl.store(OUT, 0.0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_sum"),
    key=["N", "IX", "IY", "COMPLEX", "KIND", "CONJ", "PARTS"],
    reset_to_zero=["OUT"],
)
@triton.jit
def _dot_fallback_partial(
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
    imag = tl.zeros((BLOCK,), tl.float32)
    for start in range(tl.program_id(0) * BLOCK, N, PARTS * BLOCK):
        index = start + lane
        mask = index < N
        xo = index.to(tl.int64) * IX
        if COMPLEX:
            xr = tl.load(X + 2 * xo, mask, other=0.0)
            xi = tl.load(X + 2 * xo + 1, mask, other=0.0)
        else:
            xr = tl.load(X + xo, mask, other=0.0)
        yo = index.to(tl.int64) * IY
        if COMPLEX:
            yr = tl.load(Y + 2 * yo, mask, other=0.0)
            yi = tl.load(Y + 2 * yo + 1, mask, other=0.0)
            if CONJ:
                real += xr * yr + xi * yi
                imag += xr * yi - xi * yr
            else:
                real += xr * yr - xi * yi
                imag += xr * yi + xi * yr
        else:
            yr = tl.load(Y + yo, mask, other=0.0)
            real += xr * yr
    total = tl.sum(real, 0)
    if PARTS == 1:
        tl.store(OUT, total)
        if COMPLEX:
            tl.store(OUT + 1, tl.sum(imag, 0))
    else:
        tl.atomic_add(OUT, total, sem="relaxed")
        if COMPLEX:
            tl.atomic_add(OUT + 1, tl.sum(imag, 0), sem="relaxed")


def _reduce_fallback(
    name, family, kind, n, x, ix, result, y=None, iy=1, conjugate=False
):
    common = getattr(import_module("flag_blas.ops.level1." + family), name)
    common(0, x, ix, y, iy, result)
    if n <= 0:
        return
    assert ix > 0 and iy > 0, "incx and incy must be positive"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert (
        result.numel() == 1 and result.dim() == 1
    ), "result must be a 1-element tensor"
    assert x.device == result.device, "result must be on the same device"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert y.is_contiguous(), "y must be contiguous"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.numel() >= 1 + (n - 1) * ix, "x is too short"
    assert y.numel() >= 1 + (n - 1) * iy, "y is too short"
    pointer_x = torch.view_as_real(x) if x.is_complex() else x
    pointer_y = torch.view_as_real(y) if y.is_complex() else y
    output = torch.view_as_real(result) if result.is_complex() else result
    parts = 1 if n <= 16384 else min(triton.cdiv(n, 512), CORE_NUM)
    with torch_device_fn.device(x.device):
        output.zero_()
        _dot_fallback_partial[parts,](
            pointer_x, pointer_y, output, n, ix, iy, x.is_complex(), 2, conjugate, parts
        )


_SDOT_SINGLE_LIMIT = 131072


def _prune_sdot_configs(configs, args, **kwargs):
    per_part = triton.cdiv(args["N"], args["PARTS"])
    upper = min(8192, max(256, triton.next_power_of_2(per_part)))
    if args["GATHER"]:
        # Two dense input spans plus index/product/accumulator vectors must fit UB.
        footprint = 4 * (args["WIDTH_X"] + args["WIDTH_Y"] + 3)
        allowed = [
            c.kwargs["BLOCK"]
            for c in configs
            if c.kwargs["BLOCK"] * footprint <= 131072
        ]
        upper = min(upper, max(allowed))
    if args["WIDE"]:
        upper = min(upper, 256)
    lower = min(4096, upper)
    return [c for c in configs if lower <= c.kwargs["BLOCK"] <= upper]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_dot"),
    key=["N", "IX", "IY", "PARTS", "WIDE", "GATHER", "WIDTH_X", "WIDTH_Y"],
    prune_configs_by={"early_config_prune": _prune_sdot_configs},
    reset_to_zero=["OUT"],
)
@triton.jit
def _sdot_partial(
    X,
    Y,
    OUT,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH_X: tl.constexpr,
    WIDTH_Y: tl.constexpr,
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
        if GATHER and IX > 1:
            physical_x = (start.to(tl.int64) if WIDE else start) * IX + tl.arange(
                0, BLOCK * WIDTH_X
            )
            size = N.to(tl.int64) if WIDE else N
            values_x = tl.load(
                X + physical_x, physical_x < (size - 1) * IX + 1, other=0.0
            )
            xv = tl.gather(values_x, lane * IX, axis=0)
        else:
            xv = tl.load(X + offset * IX, index < N, other=0.0)
        if GATHER and IY > 1:
            physical_y = (start.to(tl.int64) if WIDE else start) * IY + tl.arange(
                0, BLOCK * WIDTH_Y
            )
            size = N.to(tl.int64) if WIDE else N
            values_y = tl.load(
                Y + physical_y, physical_y < (size - 1) * IY + 1, other=0.0
            )
            yv = tl.gather(values_y, lane * IY, axis=0)
        else:
            yv = tl.load(Y + offset * IY, index < N, other=0.0)
        accumulated += tl.where(index < N, xv * yv, 0.0)
    total = tl.sum(accumulated, 0)
    if PARTS == 1:
        tl.store(OUT, total)
    else:
        tl.atomic_add(OUT, total, sem="relaxed")


def _sdot_reduce(n, x, incx, y, incy, result):
    if not (
        type(n) is int
        and type(incx) is int
        and type(incy) is int
        and n > 0
        and incx > 0
        and incy > 0
        and max(incx, incy) <= 8
    ):
        return _reduce_fallback("sdot", "dot", 2, n, x, incx, result, y, incy)
    assert x.dtype == y.dtype == result.dtype == torch.float32
    assert x.dim() == y.dim() == 1 and x.is_contiguous() and y.is_contiguous()
    assert result.dim() == 1 and result.numel() == 1
    assert x.device == y.device == result.device
    assert x.numel() >= 1 + (n - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _sdot_reduce(n, x, incx, y, incy, result)
    xp, yp, op = x.data_ptr(), y.data_ptr(), result.data_ptr()
    key = device.index, n, incx, incy, xp % 16 == 0, yp % 16 == 0, op % 16 == 0
    entry = _LAUNCH_CACHE.get(("sdot", *key))
    if entry is None:
        parts = 1 if n <= _SDOT_SINGLE_LIMIT else min(triton.cdiv(n, 512), CORE_NUM)
        if 65536 < n <= _SDOT_SINGLE_LIMIT and min(incx, incy) > 1:
            parts = min(8, CORE_NUM)
    else:
        parts = entry[2][3]
    if parts > 1:
        zero_key = device.index, op % 16 == 0
        zero = _LAUNCH_CACHE.get(("dot_zero", *zero_key))
        if (
            entry is not None
            and zero is not None
            and type(x) is torch.Tensor
            and type(y) is torch.Tensor
            and type(result) is torch.Tensor
            and not x.is_conj()
            and not y.is_conj()
            and not result.is_conj()
            and not x.is_neg()
            and not y.is_neg()
            and not result.is_neg()
        ):
            return _dot_stream_launch_pair(
                zero, (*entry, parts), (x, y, result), (xp, yp, op), device.index
            )
        if zero is None:
            compiled, _ = _dot_zero_result[(1,)](result, num_warps=1, num_stages=1)
            _remember_l1_entry(("dot_zero", *zero_key), (compiled, compiled.run))
        else:
            _launch_l1_entry(*zero, (1, 1, 1), (result,), (op,), (), device.index)
    if entry is None:
        arguments = (
            n,
            incx,
            incy,
            parts,
            n * max(incx, incy) > 2147483647,
            max(incx, incy) > 1,
            triton.next_power_of_2(incx),
            triton.next_power_of_2(incy),
        )
        kernel = _sdot_partial if arguments[4] or arguments[5] else _sdot_static_n
        compiled, meta = kernel[(parts,)](x, y, result, *arguments)

        _remember_l1_entry(
            ("sdot", *key), (compiled, compiled.run, (*arguments, meta["BLOCK"]))
        )
    else:
        _launch_l1_entry(
            entry[0],
            entry[1],
            (parts, 1, 1),
            (x, y, result),
            (xp, yp, op),
            entry[2],
            device.index,
        )


def sdot(n, x, incx, y, incy, result):
    if _dot_stream_public_ok(n, x, incx, y, incy, result):
        return _dot_stream_reduce(n, x, incx, y, incy, result, 1, False)
    return _sdot_reduce(n, x, incx, y, incy, result)


def cdotc(n, x, incx, y, incy, result):
    if _dot_stream_public_ok(n, x, incx, y, incy, result):
        return _dot_stream_reduce(n, x, incx, y, incy, result, 2, True)
    return _cdot_reduce(n, x, incx, y, incy, result, True)


def cdotu(n, x, incx, y, incy, result):
    if _dot_stream_public_ok(n, x, incx, y, incy, result):
        return _dot_stream_reduce(n, x, incx, y, incy, result, 2, False)
    return _cdot_reduce(n, x, incx, y, incy, result, False)


_SDOT_COMPUTE = _sdot_partial.jit_function


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_dot"),
    key=["N", "IX", "IY", "PARTS", "WIDE", "GATHER", "WIDTH_X", "WIDTH_Y"],
    prune_configs_by={"early_config_prune": _prune_sdot_configs},
    reset_to_zero=["OUT"],
)
@triton.jit
def _sdot_static_n(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    PARTS: tl.constexpr,
    WIDE: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH_X: tl.constexpr,
    WIDTH_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _SDOT_COMPUTE(X, Y, OUT, N, IX, IY, PARTS, WIDE, GATHER, WIDTH_X, WIDTH_Y, BLOCK)


_CDOT_CONFIGS = runtime.get_tuned_config("ascend_l1_cdot")


def _cdot_prune(configs, args, **kwargs):
    per_part = (args["N"] + args["PARTS"] - 1) // args["PARTS"]
    upper = min(2048, max(256, triton.next_power_of_2(per_part)))
    footprint = 4 * (2 * (args["WIDTH_X"] + args["WIDTH_Y"]) + 10)
    allowed = [
        config
        for config in configs
        if config.kwargs["BLOCK"] <= upper
        and config.kwargs["BLOCK"] * footprint <= 131072
    ]
    largest = max(config.kwargs["BLOCK"] for config in allowed)
    lower = max(256, largest // 2)
    return [config for config in allowed if config.kwargs["BLOCK"] >= lower]


@libentry()
@triton.jit
def _cdot_zero_result(OUT):
    tl.store(OUT + tl.arange(0, 2), 0.0)


@libentry()
@libtuner(
    configs=_CDOT_CONFIGS,
    key=["N", "IX", "IY", "PARTS", "CONJ_MODE", "WIDTH_X", "WIDTH_Y"],
    prune_configs_by={"early_config_prune": _cdot_prune},
    reset_to_zero=["OUT"],
)
@triton.jit
def _cdot_paired_partial(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    PARTS: tl.constexpr,
    CONJ_MODE: tl.constexpr,
    WIDTH_X: tl.constexpr,
    WIDTH_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    raw_lane_x = tl.arange(0, 2 * BLOCK * WIDTH_X)
    raw_lane_y = tl.arange(0, 2 * BLOCK * WIDTH_Y)
    select_x = 2 * lane * IX
    select_y = 2 * lane * IY
    real = tl.zeros((BLOCK,), tl.float32)
    imag = tl.zeros((BLOCK,), tl.float32)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    for start in range(first, end, BLOCK):
        logical = start + lane
        physical_x = 2 * start * IX + raw_lane_x
        physical_y = 2 * start * IY + raw_lane_y
        limit_x = tl.minimum(2 * ((N - 1) * IX + 1), 2 * (start + BLOCK) * IX)
        limit_y = tl.minimum(2 * ((N - 1) * IY + 1), 2 * (start + BLOCK) * IY)
        raw_x = tl.load(X + physical_x, physical_x < limit_x, other=0.0)
        raw_y = tl.load(Y + physical_y, physical_y < limit_y, other=0.0)
        xr = tl.gather(raw_x, select_x, axis=0)
        xi = tl.gather(raw_x, select_x + 1, axis=0)
        yr = tl.gather(raw_y, select_y, axis=0)
        yi = tl.gather(raw_y, select_y + 1, axis=0)
        if CONJ_MODE == 1:
            xi = -xi
        elif CONJ_MODE == 2:
            yi = -yi
        real += tl.where(logical < N, xr * yr - xi * yi, 0.0)
        imag += tl.where(logical < N, xr * yi + xi * yr, 0.0)
    sr = tl.sum(real, 0)
    si = tl.sum(imag, 0)
    if PARTS == 1:
        tl.store(OUT, sr)
        tl.store(OUT + 1, si)
    else:
        tl.atomic_add(OUT, sr, sem="relaxed")
        tl.atomic_add(OUT + 1, si, sem="relaxed")


def _cdot_contiguous_prune(configs, args, **kwargs):
    per_part = triton.cdiv(args["N"], args["PARTS"])
    lower = 64 if per_part <= 128 else 128 if per_part <= 1024 else 256
    upper = min(1024, max(256, triton.next_power_of_2(per_part)))
    return [
        c
        for c in configs
        if lower <= c.kwargs["BLOCK"] <= upper
        and (args["N"] < 16384 or c.kwargs["BLOCK"] >= 512)
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_cdot_contiguous"),
    key=["N", "IX", "IY", "PARTS", "CONJ", "WIDTH_X", "WIDTH_Y"],
    reset_to_zero=["OUT"],
    prune_configs_by={"early_config_prune": _cdot_contiguous_prune},
)
@triton.jit
def _cdot_contiguous_partial(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    PARTS: tl.constexpr,
    CONJ: tl.constexpr,
    WIDTH_X: tl.constexpr,
    WIDTH_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    component = tl.arange(0, 2)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    end = tl.minimum(first + chunk, N)
    if PARTS == 1 or first < N:
        real = tl.zeros((BLOCK,), tl.float32)
        imag = tl.zeros((BLOCK,), tl.float32)
        for start in range(first, end, BLOCK):
            logical = start + lane
            mask = logical < N
            vx = tl.load(
                X + 2 * logical[:, None] * IX + component[None, :],
                mask[:, None],
                other=0.0,
            )
            vy = tl.load(
                Y + 2 * logical[:, None] * IY + component[None, :],
                mask[:, None],
                other=0.0,
            )
            (xr, xi) = tl.split(vx)
            (yr, yi) = tl.split(vy)
            if CONJ:
                real += xr * yr + xi * yi
                imag += xr * yi - xi * yr
            else:
                real += xr * yr - xi * yi
                imag += xr * yi + xi * yr
        (sr, si) = (tl.sum(real, 0), tl.sum(imag, 0))
        if PARTS == 1:
            tl.store(OUT, sr)
            tl.store(OUT + 1, si)
        else:
            tl.atomic_add(OUT, sr, sem="relaxed")
            tl.atomic_add(OUT + 1, si, sem="relaxed")


def _cdot_reduce_general(n, x, incx, y, incy, result, conjugate):
    name, family = ("cdotc", "dotc") if conjugate else ("cdotu", "dotu")
    if not (
        type(n) is int
        and type(incx) is int
        and type(incy) is int
        and n > 0
        and 0 < incx <= 8
        and 0 < incy <= 8
    ):
        return _reduce_fallback(name, family, 2, n, x, incx, result, y, incy, conjugate)
    assert x.dtype == y.dtype == result.dtype == torch.complex64
    assert x.dim() == y.dim() == 1 and x.is_contiguous() and y.is_contiguous()
    assert result.dim() == 1 and result.numel() == 1
    assert x.device == y.device == result.device
    assert x.numel() >= 1 + (n - 1) * incx and y.numel() >= 1 + (n - 1) * incy
    if any(
        type(t) is not torch.Tensor or t.is_conj() or t.is_neg() for t in (x, y, result)
    ):
        return _reduce_fallback(name, family, 2, n, x, incx, result, y, incy, conjugate)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _cdot_reduce(n, x, incx, y, incy, result, conjugate)
    xp, yp, op = x.data_ptr(), y.data_ptr(), result.data_ptr()
    if xp % 32 or yp % 32 or op % 16:
        return _reduce_fallback(name, family, 2, n, x, incx, result, y, incy, conjugate)
    if (xp < op + 8 and op < xp + 8 * (1 + (n - 1) * incx)) or (
        yp < op + 8 and op < yp + 8 * (1 + (n - 1) * incy)
    ):
        return _reduce_fallback(name, family, 2, n, x, incx, result, y, incy, conjugate)
    if 2 * (1 + (n - 1) * max(incx, incy)) > 2147483647 - 2 * CORE_NUM * 2048 * 8:
        return _reduce_fallback(name, family, 2, n, x, incx, result, y, incy, conjugate)
    if n >= 33554432 and incx == 3 and incy == 2:
        # Read the lower-stride input first. CONJ_MODE=2 conjugates the second input.
        x, y = y, x
        incx, incy = incy, incx
        xp, yp = yp, xp
        if conjugate:
            conjugate = 2
    key = device.index, n, incx, incy, conjugate
    entry = _LAUNCH_CACHE.get(("cdot", *key))
    parts = (
        (1 if n <= 16384 else min((n + 511) // 512, CORE_NUM))
        if entry is None
        else entry[3]
    )
    if parts > 1:
        zero_key = device.index, op % 16 == 0
        zero = _LAUNCH_CACHE.get(("cdot_zero", *zero_key))
        if entry is not None and zero is not None:
            return _dot_stream_launch_pair(
                zero, entry, (x, y, result), (xp, yp, op), device.index
            )
        if zero is None:
            compiled, _ = _cdot_zero_result[(1,)](
                torch.view_as_real(result), num_warps=1, num_stages=1
            )
            _remember_l1_entry(("cdot_zero", *zero_key), (compiled, compiled.run))
        else:
            _launch_l1_entry(*zero, (1, 1, 1), (result,), (op,), (), device.index)
    if entry is None:
        unit = incx == incy == 1
        arguments = (
            n,
            incx,
            incy,
            parts,
            conjugate if unit else int(conjugate),
            triton.next_power_of_2(incx),
            triton.next_power_of_2(incy),
        )
        kernel = (
            _cdot_contiguous_small
            if unit and n <= 8192
            else _cdot_contiguous_partial
            if unit
            else _cdot_paired_partial
        )
        compiled, meta = kernel[(parts,)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            torch.view_as_real(result),
            *arguments,
            **({"multibuffer": False} if unit else {}),
        )

        _remember_l1_entry(
            ("cdot", *key), (compiled, compiled.run, (*arguments, meta["BLOCK"]), parts)
        )
    else:
        _launch_l1_entry(
            entry[0],
            entry[1],
            (parts, 1, 1),
            (x, y, result),
            (xp, yp, op),
            entry[2],
            device.index,
        )


# Large streaming DOT inputs use the SDK L2-bypass read alias on Ascend 910B.
@lru_cache(maxsize=None)
def _dot_stream_offset(device):
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


_DOT_STREAM_CONFIGS = tuple(
    triton.Config({"BLOCK": block}, num_warps=1, num_stages=1)
    for block in (512, 1024, 2048, 4096, 8192)
)


def _dot_stream_prune(configs, args, **kwargs):
    unit = args["IX"] == args["IY"] == 1
    if args["F"] == 1:
        blocks = (2048, 4096, 8192) if unit else (1024, 2048)
    else:
        blocks = (1024, 2048, 4096) if unit else (512, 1024, 2048)
    return [cfg for cfg in configs if cfg.kwargs["BLOCK"] in blocks]


@triton.jit
def _dot_stream_real_read(X, start, N, IX: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    physical = start * IX + tl.arange(0, BLOCK * triton.next_power_of_2(IX))
    limit = tl.minimum((N - 1) * IX + 1, (start + BLOCK) * IX)
    raw = tl.load(X + physical, physical < limit, other=0.0)
    if IX != 1:
        raw = tl.gather(raw, lane * IX, 0)
    return tl.where(start + lane < N, raw, 0.0)


@triton.jit
def _dot_stream_complex_read(X, start, N, IX: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    physical = 2 * start * IX + tl.arange(0, 2 * BLOCK * triton.next_power_of_2(IX))
    limit = tl.minimum(2 * ((N - 1) * IX + 1), 2 * (start + BLOCK) * IX)
    raw = tl.load(X + physical, physical < limit, other=0.0)
    if IX == 1:
        # Split contiguous pairs directly; a 2-D gather is slow for strided input.
        real, imag = tl.split(raw.reshape((BLOCK, 2)))
    else:
        real = tl.gather(raw, 2 * lane * IX, 0)
        imag = tl.gather(raw, 2 * lane * IX + 1, 0)
    return real, imag


@libentry()
@libtuner(
    configs=_DOT_STREAM_CONFIGS,
    key=["N", "IX", "IY", "F", "CONJ", "OFFSET", "PARTS"],
    prune_configs_by={"early_config_prune": _dot_stream_prune},
    reset_to_zero=["OUT"],
)
@triton.jit
def _dot_stream_kernel(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    CONJ: tl.constexpr,
    OFFSET: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    X = (X.to(tl.uint64) + OFFSET).to(tl.pointer_type(tl.float32))
    Y = (Y.to(tl.uint64) + OFFSET).to(tl.pointer_type(tl.float32))
    lane = tl.arange(0, BLOCK)
    real = tl.zeros((BLOCK,), tl.float32)
    if F == 2:
        imag = tl.zeros((BLOCK,), tl.float32)
    if N >= 33554432 and (IX > 1 or IY > 1):
        first = tl.program_id(0) * BLOCK
        end = N
        step = PARTS * BLOCK
    else:
        chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
        first = tl.program_id(0) * chunk
        end = tl.minimum(first + chunk, N)
        step = BLOCK
    for start in range(first, end, step):
        if F == 1:
            xv = _dot_stream_real_read(X, start, N, IX, BLOCK)
            yv = _dot_stream_real_read(Y, start, N, IY, BLOCK)
            real += xv * yv
        else:
            if N >= 33554432 and IX == 2 and IY == 3:
                xr, xi = _dot_stream_complex_read(Y, start, N, IY, BLOCK)
                yr, yi = _dot_stream_complex_read(X, start, N, IX, BLOCK)
                if CONJ:
                    yi = -yi
            else:
                xr, xi = _dot_stream_complex_read(X, start, N, IX, BLOCK)
                yr, yi = _dot_stream_complex_read(Y, start, N, IY, BLOCK)
                if CONJ:
                    xi = -xi
            real += tl.where(start + lane < N, xr * yr - xi * yi, 0.0)
            imag += tl.where(start + lane < N, xr * yi + xi * yr, 0.0)
    total = tl.sum(real, 0)
    if PARTS == 1:
        tl.store(OUT, total)
        if F == 2:
            tl.store(OUT + 1, tl.sum(imag, 0))
    else:
        tl.atomic_add(OUT, total, sem="relaxed")
        if F == 2:
            tl.atomic_add(OUT + 1, tl.sum(imag, 0), sem="relaxed")


def _dot_stream_public_ok(n, x, ix, y, iy, result):
    return (
        type(n) is int
        and n >= 1048576
        and type(ix) is int
        and 0 < ix <= 3
        and type(iy) is int
        and 0 < iy <= 3
        # Keep the original gather path for the remaining large strides.
        and (max(ix, iy) == 1 or n < 33554432 or n <= 67108864 and min(ix, iy) == 2)
        and all(
            type(t) is torch.Tensor and not t.is_conj() and not t.is_neg()
            for t in (x, y, result)
        )
    )


def _dot_stream_launch_pair(zero, entry, tensors, pointers, device):
    (z, zrun) = zero
    (compiled, run, arguments, parts) = entry
    knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    safe = (
        _launch_utils._FAST_LAUNCH_SUPPORTED
        and knobs is not None
        and (type(zrun) is _launch_utils._NPU_LAUNCHER)
        and (type(run) is _launch_utils._NPU_LAUNCHER)
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
            and (not getattr(zrun, "compile_only", False))
            and (not getattr(zrun, "enable_msprof_register_tensor", False))
            and (not getattr(z.metadata, "debug_enabled", False))
            and (not getattr(getattr(zrun, "metadata", None), "debug_enabled", False))
            and (not getattr(run, "compile_only", False))
            and (not getattr(run, "enable_msprof_register_tensor", False))
            and (not getattr(compiled.metadata, "debug_enabled", False))
            and (not getattr(getattr(run, "metadata", None), "debug_enabled", False))
        )
        if safe:
            pointer_type = _launch_utils._DevicePointer
            (xp, yp, out) = (
                pointer_type(pointers[0]),
                pointer_type(pointers[1]),
                pointer_type(pointers[2]),
            )
            (xp.tensor, yp.tensor, out.tensor) = tensors
            stream = _launch_utils._current_raw_stream(device)
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
                out,
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
                xp,
                yp,
                out,
                *arguments,
            )
            registered = zrun.launch(*zero_args)
            _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
            registered = run.launch(*compute_args)
            _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
            return
    _launch_l1_entry(z, zrun, (1, 1, 1), (tensors[2],), (pointers[2],), (), device)
    _launch_l1_entry(compiled, run, (parts, 1, 1), tensors, pointers, arguments, device)


def _dot_stream_reduce(n, x, ix, y, iy, result, factor, conjugate):
    expected = torch.float32 if factor == 1 else torch.complex64
    assert x.dtype == y.dtype == result.dtype == expected
    assert x.dim() == y.dim() == 1 and x.is_contiguous() and y.is_contiguous()
    assert result.dim() == 1 and result.numel() == 1
    device = x.device
    assert device == y.device == result.device
    assert x.numel() >= 1 + (n - 1) * ix and y.numel() >= 1 + (n - 1) * iy
    pointers = x.data_ptr(), y.data_ptr(), result.data_ptr()
    xp, yp, op = pointers
    size = factor * 4
    eligible = (
        (n + (CORE_NUM + 2) * 8192) * max(ix, iy) * factor < 2147483647
        and xp % 64 == yp % 64 == op % 16 == 0
        and not (xp < op + size and op < xp + ((n - 1) * ix + 1) * size)
        and not (yp < op + size and op < yp + ((n - 1) * iy + 1) * size)
    )
    if not eligible:
        if factor == 1:
            return _sdot_reduce(n, x, ix, y, iy, result)
        return _cdot_reduce(n, x, ix, y, iy, result, conjugate)
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _dot_stream_reduce(n, x, ix, y, iy, result, factor, conjugate)
    key = ("stream_dot", device.index, n, ix, iy, factor, conjugate)
    entry = _LAUNCH_CACHE.get(key)
    zero_key = ("stream_zero", device.index, factor)
    zero = _LAUNCH_CACHE.get(zero_key)
    if entry is not None and zero is not None:
        return _dot_stream_launch_pair(
            zero, entry, (x, y, result), pointers, device.index
        )
    offset = _dot_stream_offset(device.index)
    if not offset:
        if factor == 1:
            return _sdot_reduce(n, x, ix, y, iy, result)
        return _cdot_reduce(n, x, ix, y, iy, result, conjugate)
    raw = tuple(torch.view_as_real(t) if factor == 2 else t for t in (x, y, result))
    zero_kernel = _cdot_zero_result if factor == 2 else _dot_zero_result
    z, _ = zero_kernel[(1,)](raw[2], num_warps=1, num_stages=1)
    parts = min(CORE_NUM, (n + 511) // 512)
    arguments = (n, ix, iy, factor, conjugate, offset, parts)
    compiled, meta = _dot_stream_kernel[(parts,)](*raw, *arguments, multibuffer=False)
    _remember_l1_entry(zero_key, (z, z.run))
    _remember_l1_entry(
        key, (compiled, compiled.run, (*arguments, meta["BLOCK"]), parts)
    )


def _cdot_reduce(n, x, incx, y, incy, result, conjugate):
    if not (
        type(n) is int
        and 0 < n <= 1048576
        and (type(incx) is int)
        and (type(incy) is int)
        and (0 < incx <= 8)
        and (0 < incy <= 8)
        and (type(x) is torch.Tensor)
        and (type(y) is torch.Tensor)
        and (type(result) is torch.Tensor)
    ):
        return _cdot_reduce_general(n, x, incx, y, incy, result, conjugate)
    assert x.dtype == y.dtype == result.dtype == torch.complex64
    assert x.dim() == y.dim() == 1 and x.is_contiguous() and y.is_contiguous()
    assert result.dim() == 1 and result.numel() == 1
    device = x.device
    assert device == y.device == result.device
    assert x.numel() >= 1 + (n - 1) * incx and y.numel() >= 1 + (n - 1) * incy
    if (
        x.is_conj()
        or y.is_conj()
        or result.is_conj()
        or x.is_neg()
        or y.is_neg()
        or result.is_neg()
        or (device.index != _launch_utils._current_device())
    ):
        return _cdot_reduce_general(n, x, incx, y, incy, result, conjugate)
    (xp, yp, op) = (x.data_ptr(), y.data_ptr(), result.data_ptr())
    if (
        xp % 32
        or yp % 32
        or op % 16
        or (xp < op + 8 and op < xp + 8 * (1 + (n - 1) * incx))
        or (yp < op + 8 and op < yp + 8 * (1 + (n - 1) * incy))
    ):
        return _cdot_reduce_general(n, x, incx, y, incy, result, conjugate)
    entry = _LAUNCH_CACHE.get(("cdot", device.index, n, incx, incy, conjugate))
    if entry is None:
        return _cdot_reduce_general(n, x, incx, y, incy, result, conjugate)
    if entry[3] > 1:
        zero = _LAUNCH_CACHE.get(("cdot_zero", device.index, True))
        if zero is None:
            return _cdot_reduce_general(n, x, incx, y, incy, result, conjugate)
        return _dot_stream_launch_pair(
            zero, entry, (x, y, result), (xp, yp, op), device.index
        )
    (compiled, run, arguments, parts) = entry
    knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    if (
        _launch_utils._FAST_LAUNCH_SUPPORTED
        and type(run) is _launch_utils._NPU_LAUNCHER
        and (knobs is not None)
        and (not getattr(run, "compile_only", False))
        and (not getattr(run, "enable_msprof_register_tensor", False))
        and (not getattr(compiled.metadata, "debug_enabled", False))
        and (not getattr(getattr(run, "metadata", None), "debug_enabled", False))
    ):
        (h0, h1) = (knobs.launch_enter_hook, knobs.launch_exit_hook)
        hooks = (
            h0 is not None
            and (type(h0) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h0.calls))
            or (
                h1 is not None
                and (type(h1) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h1.calls))
            )
        )
        if not hooks:
            pointer_type = _launch_utils._DevicePointer
            (wx, wy, out) = (pointer_type(xp), pointer_type(yp), pointer_type(op))
            (wx.tensor, wy.tensor, out.tensor) = (x, y, result)
            registered = run.launch(
                parts,
                1,
                1,
                _launch_utils._current_raw_stream(device.index),
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                wx,
                wy,
                out,
                *arguments,
            )
            _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
            return
    return _cdot_reduce_general(n, x, incx, y, incy, result, conjugate)


def _cdot_small_prune(configs, args, **kw):
    if 1024 <= args["N"] <= 8192:
        return [c for c in configs if c.kwargs["BLOCK"] == 1024]
    return _cdot_contiguous_prune(configs, args, **kw)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_cdot_contiguous"),
    key=["N", "IX", "IY", "PARTS", "CONJ", "WIDTH_X", "WIDTH_Y"],
    reset_to_zero=["OUT"],
    prune_configs_by={"early_config_prune": _cdot_small_prune},
)
@triton.jit
def _cdot_contiguous_small(
    X,
    Y,
    OUT,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    PARTS: tl.constexpr,
    CONJ: tl.constexpr,
    WIDTH_X: tl.constexpr,
    WIDTH_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    component = tl.arange(0, 2)
    chunk = tl.cdiv(N, PARTS * BLOCK) * BLOCK
    first = tl.program_id(0) * chunk
    if PARTS == 1 or first < N:
        real = tl.zeros((BLOCK,), tl.float32)
        imag = tl.zeros((BLOCK,), tl.float32)
        for tile in tl.static_range(0, triton.cdiv(N, PARTS * BLOCK)):
            start = first + tile * BLOCK
            logical = start + lane
            mask = logical < N
            vx = tl.load(
                X + 2 * logical[:, None] * IX + component[None, :],
                mask[:, None],
                other=0.0,
            )
            vy = tl.load(
                Y + 2 * logical[:, None] * IY + component[None, :],
                mask[:, None],
                other=0.0,
            )
            (xr, xi) = tl.split(vx)
            (yr, yi) = tl.split(vy)
            if CONJ:
                real += xr * yr + xi * yi
                imag += xr * yi - xi * yr
            else:
                real += xr * yr - xi * yi
                imag += xr * yi + xi * yr
        (sr, si) = (tl.sum(real, 0), tl.sum(imag, 0))
        if PARTS == 1:
            tl.store(OUT, sr)
            tl.store(OUT + 1, si)
        else:
            tl.atomic_add(OUT, sr, sem="relaxed")
            tl.atomic_add(OUT + 1, si, sem="relaxed")

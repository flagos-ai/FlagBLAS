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
            if len(tensors) == 2:
                (t0, t1) = tensors
                if (
                    type(t0) is torch.Tensor
                    and (not t0.is_conj())
                    and (not t0.is_neg())
                    and (type(t1) is torch.Tensor)
                    and (not t1.is_conj())
                    and (not t1.is_neg())
                ):
                    pointer_type = _launch_utils._DevicePointer
                    w0 = pointer_type(pointers[0])
                    w0.tensor = t0
                    w1 = pointer_type(pointers[1])
                    w1.tensor = t1
                    registered = run.launch(
                        *grid,
                        _launch_utils._current_raw_stream(device),
                        compiled.function,
                        compiled.packed_metadata,
                        None,
                        None,
                        None,
                        w0,
                        w1,
                        *arguments,
                    )
                    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = (
                        registered == 1
                    )
                    return
    compiled[grid](*_launch_utils._real_tensors(tensors), *arguments)


_LAUNCH_CACHE = {}


_LAUNCH_CACHE_LIMIT = 2048


def _store_launch(key, entry):
    if len(_LAUNCH_CACHE) >= _LAUNCH_CACHE_LIMIT:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = entry


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_vector"),
    key=["N", "IX", "IY", "COMPLEX", "OP"],
    restore_value=["X", "Y"],
)
@triton.jit
def _vector_update(
    X,
    Y,
    N,
    AR: tl.float32,
    AI: tl.float32,
    C: tl.float32,
    S: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    COMPLEX: tl.constexpr,
    OP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        index = start + lane
        mask = index < N
        xo = index.to(tl.int64) * IX
        yo = index.to(tl.int64) * IY
        if COMPLEX:
            xr = tl.load(X + 2 * xo, mask, other=0.0)
            xi = tl.load(X + 2 * xo + 1, mask, other=0.0)
            (nr, ni) = (xr, xi)
            tl.store(Y + 2 * yo, nr, mask)
            tl.store(Y + 2 * yo + 1, ni, mask)
        else:
            x = tl.load(X + xo, mask, other=0.0)
            value = x
            tl.store(Y + yo, value, mask)


def _launch(n, x, incx, y, incy, op, alpha=1.0, c=0.0, s=0.0):
    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"
    assert x.dim() == 1 and y.dim() == 1, "x and y must be 1-dimensional"
    assert x.is_contiguous() and y.is_contiguous(), "x and y must be contiguous"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == y.dtype, "x and y must have the same dtype"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    value = complex(value)
    c = float(c.item() if isinstance(c, torch.Tensor) else c)
    s = float(s.item() if isinstance(s, torch.Tensor) else s)
    raw_x = torch.view_as_real(x) if x.is_complex() else x
    raw_y = torch.view_as_real(y) if y.is_complex() else y
    with torch_device_fn.device(x.device):
        _vector_update[lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), CORE_NUM),)](
            raw_x, raw_y, n, value.real, value.imag, c, s, incx, incy, x.is_complex(), 0
        )


def _copy_fallback(name, family, op, n, x, incx, y, incy):
    common = getattr(import_module("flag_blas.ops.level1." + family), name)
    if n <= 0:
        return common(n, x, incx, y, incy)
    common(0, x, incx, y, incy)
    return _launch(n, x, incx, y, incy, op)


_copy_l1_core = CORE_NUM


_copy_l1_safe = 2147483647 - CORE_NUM * 8192 - 65536


_copy_l1_configs = tuple(
    c
    for c in runtime.get_tuned_config("ascend_l1_memory")
    if c.num_warps == c.num_stages == 1
)


@libentry()
@triton.jit
def _copy_l1_flat(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        index = start + lane
        mask = index < N
        x = tl.load(X + index, mask, other=0.0)
        value = x
        tl.store(Y + index, value, mask)


def _copy_l1_call(name, n, x, incx, y, incy, mode, alpha=None):
    comp = name == "ccopy"
    expected = torch.complex64 if comp else torch.float32
    assert x.dtype == expected, f"x must be {expected}"
    assert y.dtype == expected, f"y must be {expected}"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    device = x.device
    assert device == y.device, "x and y must be on the same device"
    (ar, ai) = (1.0, 0.0)
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _copy_l1_call(name, n, x, incx, y, incy, 0, alpha)
    (px, py) = (x.data_ptr(), y.data_ptr())
    if 0 in (0, 1):
        if px == py and incx == incy:
            return
        size = x.element_size()
        endx = px + (1 + (n - 1) * incx) * size
        endy = py + (1 + (n - 1) * incy) * size
        if px < endy and py < endx:
            return _launch(n, x, incx, y, incy, 0)
    if comp and 1048576 <= n and 2 * n < 2147483647 - (CORE_NUM + 1) * 16384:
        offset = _copy_l2_offset(device.index)
        if offset:
            return _copy_stream_call(n, x, y, offset)
    components = n * (2 if comp else 1)
    block = (
        256
        if components <= 512
        else 2048
        if components <= _copy_l1_core * 8192
        else 8192
    )
    grid = (min((components + block - 1) // block, _copy_l1_core), 1, 1)
    key = (
        "memory",
        device.index,
        n,
        incx,
        incy,
        comp,
        0,
        0,
        px % 16 == 0,
        py % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    arguments = (components, ar, ai, 0, block)
    if entry is None:
        raw_x = torch.view_as_real(x) if comp else x
        raw_y = torch.view_as_real(y) if comp else y
        config = next((c for c in _copy_l1_configs if c.kwargs["BLOCK"] == block))
        (compiled, _) = _copy_l1_flat[grid](
            raw_x,
            raw_y,
            *arguments,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _store_launch(key, (compiled, compiled.run, grid))
        return
    (compiled, run, cached_grid) = entry
    assert grid == cached_grid
    _launch_l1_entry(compiled, run, grid, (x, y), (px, py), arguments, device.index)


def _copy_l1_eligible(n, x, incx, y, incy):
    return (
        type(n) is int
        and 0 < n <= _copy_l1_safe // 2
        and type(incx) is int
        and type(incy) is int
        and incx == 1
        and incy == 1
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and not x.is_conj()
        and not y.is_conj()
        and not x.is_neg()
        and not y.is_neg()
        and x.data_ptr() % 64 == 0
        and y.data_ptr() % 64 == 0
    )


def _copy_l1_copy_swap(name, family, op, n, x, incx, y, incy):
    if _copy_l1_eligible(n, x, incx, y, incy):
        return _copy_l1_call(name, n, x, incx, y, incy, op)
    return _copy_fallback(name, family, op, n, x, incx, y, incy)


@libentry()
@triton.jit
def _copy_l1_pair_strided(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    COMPLEX: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    XP = X
    YP = Y
    lane = tl.arange(0, BLOCK)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        index = start + lane
        mask = index < N
        x = tl.load(XP + index * IX, mask, other=0)
        tl.store(YP + index * IY, x, mask)


def _copy_l1_pair_eligible(n, x, incx, y, incy):
    return (
        type(n) is int
        and n > 0
        and type(incx) is int
        and type(incy) is int
        and incx > 0
        and incy > 0
        and max(incx, incy) > 1
        and 2 * (n + _copy_l1_core * 512) * max(incx, incy) <= _copy_l1_safe
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and x.dtype == y.dtype == torch.complex64
        and not x.is_conj()
        and not y.is_conj()
        and not x.is_neg()
        and not y.is_neg()
        and x.data_ptr() % 64 == 0
        and y.data_ptr() % 64 == 0
    )


def _copy_l1_pair_copy_swap(name, family, op, n, x, incx, y, incy):
    if type(incx) is int and type(incy) is int and incx == incy == 1:
        return _copy_l1_copy_swap(name, family, op, n, x, incx, y, incy)
    if not _copy_l1_pair_eligible(n, x, incx, y, incy):
        return _copy_l1_copy_swap(name, family, op, n, x, incx, y, incy)
    assert x.dim() == y.dim() == 1, "x and y must be 1-dimensional"
    assert x.is_contiguous() and y.is_contiguous(), "x and y must be contiguous"
    device = x.device
    assert device == y.device, "x and y must be on the same device"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _copy_l1_pair_copy_swap(name, family, op, n, x, incx, y, incy)
    px, py = x.data_ptr(), y.data_ptr()
    if px == py and incx == incy:
        return
    size = x.element_size()
    endx = px + (1 + (n - 1) * incx) * size
    endy = py + (1 + (n - 1) * incy) * size
    if px < endy and py < endx:
        return _copy_l1_copy_swap(name, family, op, n, x, incx, y, incy)
    key = (
        "memory",
        device.index,
        n,
        incx,
        incy,
        True,
        op,
        1,
        px % 16 == 0,
        py % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    block = 256 if n <= 1024 else 512
    arguments = (n, 1.0, 0.0, incx, incy, True, op, block)
    # Fresh dtype views share caller storage and give profiler/debug paths an int64 ABI.
    raw_x, raw_y = x.view(torch.int64), y.view(torch.int64)
    if entry is None:
        grid = (min((n + block - 1) // block, _copy_l1_core), 1, 1)
        config = next(c for c in _copy_l1_configs if c.kwargs["BLOCK"] == block)
        compiled, _ = _copy_l1_pair_strided[grid](
            raw_x,
            raw_y,
            *arguments,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _store_launch(key, (compiled, compiled.run, grid))
        return
    compiled, run, grid = entry
    _launch_l1_entry(
        compiled, run, grid, (raw_x, raw_y), (px, py), arguments, device.index
    )


def scopy(n, x, incx, y, incy):
    if _copy_span_eligible(n, x, incx, y, incy):
        return _copy_span_call("scopy", n, x, incx, y, incy)
    return _copy_l1_copy_swap("scopy", "copy", 0, n, x, incx, y, incy)


def ccopy(n, x, incx, y, incy):
    if _copy_span_eligible(n, x, incx, y, incy):
        return _copy_span_call("ccopy", n, x, incx, y, incy)
    return _copy_l1_pair_copy_swap("ccopy", "copy", 0, n, x, incx, y, incy)


@lru_cache(maxsize=None)
def _copy_l2_offset(device):
    if not torch.npu.get_device_name(device).startswith("Ascend910B"):
        return 0
    try:
        # Query the SDK alias for streaming reads; leave output stores cached.
        query = ctypes.CDLL("libruntime.so").rtGetL2CacheOffset
        query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int32
        offset = ctypes.c_uint64(0)
        return offset.value if query(device, ctypes.byref(offset)) == 0 else 0
    except (AttributeError, OSError):
        return 0


_COPY_SPAN_CONFIGS = tuple(
    triton.Config({"BLOCK": block, "PIPE": pipe}, num_warps=1, num_stages=1)
    for block in (128, 256, 512, 1024, 2048)
    for pipe in (False, True)
    if not (block == 2048 and pipe)
)
_COPY_STREAM_CONFIGS = tuple(
    triton.Config({"BLOCK": block, "PIPE": True}, num_warps=1, num_stages=1)
    for block in (4096, 8192, 16384)
)


def _copy_span_prune(configs, args, **kwargs):
    n, factor, iy = args["N"], args["F"], args["IY"]
    lower = 128 if n < 65536 else (256 if factor == 2 else 512)
    upper = (
        512
        if n < 65536
        else (512 if factor == 2 and iy == 3 else 1024 if factor == 2 else 2048)
    )
    return [
        cfg
        for cfg in configs
        if lower <= cfg.kwargs["BLOCK"] <= upper
        and (not cfg.kwargs["PIPE"] or n >= 1048576)
    ]


def _copy_span_eligible(n, x, incx, y, incy):
    return (
        type(n) is int
        and n > 0
        and type(incx) is int
        and type(incy) is int
        and 1 < incx <= 3
        and 1 < incy <= 3
        and 2 * (n + CORE_NUM * 2048) * max(incx, incy) < _copy_l1_safe
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and not x.is_conj()
        and not y.is_conj()
        and not x.is_neg()
        and not y.is_neg()
        and x.data_ptr() % 64 == y.data_ptr() % 64 == 0
    )


def _copy_stream_call(n, x, y, offset):
    device = x.device
    key = ("copy_stream", device.index, n)
    entry = _LAUNCH_CACHE.get(key)
    components = n * 2
    if entry is None:
        compiled, meta = _copy_stream_kernel[
            lambda cfg: (min(triton.cdiv(components, cfg["BLOCK"]), CORE_NUM),)
        ](
            torch.view_as_real(x),
            torch.view_as_real(y),
            components,
            OFFSET=offset,
            multibuffer=False,
        )
        constants = (meta["BLOCK"], offset, meta["PIPE"])
        grid = (min(triton.cdiv(components, meta["BLOCK"]), CORE_NUM), 1, 1)
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled,
        run,
        grid,
        (x, y),
        (x.data_ptr(), y.data_ptr()),
        (components, *constants),
        device.index,
    )


def _copy_span_call(name, n, x, incx, y, incy):
    expected = torch.complex64 if name == "ccopy" else torch.float32
    assert x.dtype == y.dtype == expected, f"x and y must be {expected}"
    assert x.dim() == y.dim() == 1, "x and y must be 1-dimensional"
    assert x.is_contiguous() and y.is_contiguous(), "x and y must be contiguous"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    px, py = x.data_ptr(), y.data_ptr()
    if px == py and incx == incy:
        return
    size = x.element_size()
    if px < py + (1 + (n - 1) * incy) * size and py < px + (1 + (n - 1) * incx) * size:
        return _copy_l1_pair_copy_swap(name, "copy", 0, n, x, incx, y, incy)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _copy_span_call(name, n, x, incx, y, incy)
    factor = 2 if name == "ccopy" else 1
    key = ("copy_span", device.index, n, incx, incy, factor)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        offset = _copy_l2_offset(device.index) if n >= 1048576 else 0
        raw_x = torch.view_as_real(x) if factor == 2 else x
        raw_y = torch.view_as_real(y) if factor == 2 else y
        compiled, meta = _copy_span_kernel[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), CORE_NUM),)
        ](raw_x, raw_y, n, incx, incy, factor, OFFSET=offset, multibuffer=False)
        constants = (incx, incy, factor, meta["BLOCK"], offset, meta["PIPE"])
        grid = (min(triton.cdiv(n, meta["BLOCK"]), CORE_NUM), 1, 1)
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled, run, grid, (x, y), (px, py), (n, *constants), device.index
    )


@libentry()
@libtuner(
    configs=_COPY_SPAN_CONFIGS,
    key=["N", "IX", "IY", "F", "OFFSET"],
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _copy_span_prune},
)
@triton.jit
def _copy_span_kernel(
    X,
    Y,
    N,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    OFFSET: tl.constexpr,
    PIPE: tl.constexpr,
):
    if OFFSET:
        X = (X.to(tl.uint64) + OFFSET).to(tl.pointer_type(tl.float32))
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    # Exact division by 3/6 within the bounded tile avoids vector integer division.
    if IY * F == 2:
        q = ly >> 1
    elif IY * F == 4:
        q = ly >> 2
    elif IY * F == 3:
        q = (ly * 21846) >> 16
    else:
        q = (ly * 10923) >> 16
    active = ly - q * (IY * F) < F
    mapping = tl.minimum(
        q * (IX * F) + (ly & (F - 1)), BLOCK * triton.next_power_of_2(IX) * F - 1
    )
    begin = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    if PIPE:
        px = begin * IX * F + lx
        py = begin * IY * F + ly
        xv = tl.load(
            X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F), other=0.0
        )
        yv = tl.load(
            Y + py, (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F), other=0.0
        )
    for start in range(begin, N, step):
        px = start * IX * F + lx
        py = start * IY * F + ly
        xm = (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F)
        ym = (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        if PIPE:
            npx = (start + step) * IX * F + lx
            npy = (start + step) * IY * F + ly
            nx = tl.load(
                X + npx,
                (lx < BLOCK * IX * F) & (npx < ((N - 1) * IX + 1) * F),
                other=0.0,
            )
            ny = tl.load(
                Y + npy,
                (ly < BLOCK * IY * F) & (npy < ((N - 1) * IY + 1) * F),
                other=0.0,
            )
        else:
            xv = tl.load(X + px, xm, other=0.0)
            yv = tl.load(Y + py, ym, other=0.0)
        if IX != IY:
            value = tl.gather(xv, mapping, axis=0)
        else:
            value = xv
        # Dense stores retain the original bits of every inter-element gap.
        tl.store(Y + py, tl.where(active, value, yv), ym)
        if PIPE:
            xv = nx
            yv = ny


@libentry()
@libtuner(configs=_COPY_STREAM_CONFIGS, key=["N", "OFFSET"], restore_value=["Y"])
@triton.jit
def _copy_stream_kernel(
    X, Y, N, BLOCK: tl.constexpr, OFFSET: tl.constexpr, PIPE: tl.constexpr
):
    if OFFSET:
        X = (X.to(tl.uint64) + OFFSET).to(tl.pointer_type(tl.float32))
    lane = tl.arange(0, BLOCK)
    begin = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    if PIPE:
        i = begin + lane
        value = tl.load(X + i, i < N, other=0.0)
    for start in range(begin, N, step):
        i = start + lane
        if PIPE:
            nxt = start + step + lane
            value_next = tl.load(X + nxt, nxt < N, other=0.0)
        else:
            value = tl.load(X + i, i < N, other=0.0)
        tl.store(Y + i, value, i < N)
        if PIPE:
            value = value_next

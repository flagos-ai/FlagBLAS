# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

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
            if len(tensors) == 1:
                (t0,) = tensors
                if (
                    type(t0) is torch.Tensor
                    and (not t0.is_conj())
                    and (not t0.is_neg())
                ):
                    pointer_type = _launch_utils._DevicePointer
                    w0 = pointer_type(pointers[0])
                    w0.tensor = t0
                    registered = run.launch(
                        *grid,
                        _launch_utils._current_raw_stream(device),
                        compiled.function,
                        compiled.packed_metadata,
                        None,
                        None,
                        None,
                        w0,
                        *arguments,
                    )
                    _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = (
                        registered == 1
                    )
                    return
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
            (nr, ni) = (AR * xr - AI * xi, AR * xi + AI * xr)
            tl.store(Y + 2 * yo, nr, mask)
            tl.store(Y + 2 * yo + 1, ni, mask)
        else:
            x = tl.load(X + xo, mask, other=0.0)
            value = AR * x
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
            raw_x, raw_y, n, value.real, value.imag, c, s, incx, incy, x.is_complex(), 2
        )


_scal_l1_core = CORE_NUM


_scal_l1_safe = 2147483647 - CORE_NUM * 8192 - 65536


_scal_l1_configs = tuple(
    c
    for c in runtime.get_tuned_config("ascend_l1_memory")
    if c.num_warps == c.num_stages == 1
)


@libentry()
@triton.jit
def _scal_l1_flat(
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
        if MODE == 2:
            value = AR * x
        else:
            peer = tl.gather(x, lane ^ 1, axis=0)
            coefficient = tl.where(lane & 1 == 0, -AI, AI)
            value = AR * x + coefficient * peer
        tl.store(Y + index, value, mask)


def _scal_l1_call(name, n, x, incx, y, incy, mode, alpha=None):
    comp = name in ("cscal", "csscal")
    expected = torch.complex64 if comp else torch.float32
    assert x.dtype == expected, f"x must be {expected}"
    assert y.dtype == expected, f"y must be {expected}"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    device = x.device
    assert device == y.device, "x and y must be on the same device"
    ar, ai = 1.0, 0.0
    if mode >= 2:
        value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        if mode == 3:
            value = complex(value)
            ar, ai = value.real, value.imag
        else:
            ar = float(value)
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    assert y.numel() >= 1 + (n - 1) * incy, "y is too short"
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _scal_l1_call(name, n, x, incx, y, incy, mode, alpha)
    px, py = x.data_ptr(), y.data_ptr()
    if mode in (0, 1):
        if px == py and incx == incy:
            return
        size = x.element_size()
        endx = px + (1 + (n - 1) * incx) * size
        endy = py + (1 + (n - 1) * incy) * size
        if px < endy and py < endx:
            return _launch(n, x, incx, y, incy, mode)
    components = n * (2 if comp else 1)
    block = (
        256
        if components <= 512
        else 2048
        if components <= _scal_l1_core * 8192
        else 8192
    )
    if mode == 3:
        block = min(block, 4096)
    grid = (min((components + block - 1) // block, _scal_l1_core), 1, 1)
    key = (
        "memory",
        device.index,
        n,
        incx,
        incy,
        comp,
        mode,
        0,
        px % 16 == 0,
        py % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    arguments = (components, ar, ai, mode, block)
    if entry is None:
        raw_x = torch.view_as_real(x) if comp else x
        raw_y = torch.view_as_real(y) if comp else y
        config = next(c for c in _scal_l1_configs if c.kwargs["BLOCK"] == block)
        compiled, _ = _scal_l1_flat[grid](
            raw_x,
            raw_y,
            *arguments,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _store_launch(key, (compiled, compiled.run, grid))
        return
    compiled, run, cached_grid = entry
    assert grid == cached_grid
    _launch_l1_entry(compiled, run, grid, (x, y), (px, py), arguments, device.index)


def _scal_l1_eligible(n, x, incx, y, incy):
    return (
        type(n) is int
        and 0 < n <= _scal_l1_safe // 2
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


def _scale(name, n, alpha, x, incx):
    common = getattr(import_module("flag_blas.ops.level1.scal"), name)
    if n <= 0:
        return common(n, alpha, x, incx)
    common(0, alpha, x, incx)
    if name in ("sscal", "csscal"):
        alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    return _launch(n, x, incx, x, incx, 2, alpha=alpha)


def sscal(n, alpha, x, incx):
    return _scal_l1_scale("sscal", n, alpha, x, incx)


def cscal(n, alpha, x, incx):
    if type(n) is int and n >= 1048576 and _scal_l1_eligible(n, x, incx, x, incx):
        return _cscal_contiguous(n, alpha, x)
    return _scal_l1_scale("cscal", n, alpha, x, incx)


def csscal(n, alpha, x, incx):
    return _scal_l1_pair_csscal(n, alpha, x, incx)


def _scal_l1_scale(name, n, alpha, x, incx):
    if _scal_span_eligible(n, x, incx):
        return _scal_span_call(name, n, alpha, x, incx)
    if _scal_l1_eligible(n, x, incx, x, incx):
        mode = 3 if name == "cscal" else 2
        return _scal_l1_call(name, n, x, incx, x, incx, mode, alpha)
    return _scale(name, n, alpha, x, incx)


@libentry()
@triton.jit
def _scal_l1_scal_pair(
    X,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    INC: tl.constexpr,
    REAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    component = tl.arange(0, 2)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        index = start + lane
        offsets = index[:, None] * (2 * INC) + component[None, :]
        values = tl.load(X + offsets, index[:, None] < N, other=0.0)
        if REAL:
            result = AR * values
        else:
            xr, xi = tl.split(values)
            nr = AR * xr - AI * xi
            ni = AR * xi + AI * xr
            result = tl.join(nr, ni)
        tl.store(X + offsets, result, index[:, None] < N)


def _scal_l1_pair_csscal(n, alpha, x, incx):
    if (
        type(n) is int
        and 0 < n <= 65536
        and type(incx) is int
        and incx == 1
        and type(x) is torch.Tensor
        and not x.is_conj()
        and not x.is_neg()
        and x.data_ptr() % 64 == 0
    ):
        return _csscal_contiguous(n, alpha, x)
    if _scal_span_eligible(n, x, incx):
        return _scal_span_call("csscal", n, alpha, x, incx)
    if not (
        type(n) is int
        and n > 0
        and type(incx) is int
        and incx > 1
        and 2 * (n + _scal_l1_core * 512) * incx <= _scal_l1_safe
        and type(x) is torch.Tensor
        and x.dtype == torch.complex64
        and not x.is_conj()
        and not x.is_neg()
        and x.data_ptr() % 64 == 0
    ):
        return _scal_l1_scale("csscal", n, alpha, x, incx)
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() >= 1 + (n - 1) * incx, "x is too short"
    value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    assert not isinstance(value, complex), "alpha must be real for csscal"
    value = float(value)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _scal_l1_pair_csscal(n, value, x, incx)
    pointer = x.data_ptr()
    key = ("memory", "csscal_pair", device.index, n, incx, True, pointer % 16 == 0)
    entry = _LAUNCH_CACHE.get(key)
    block = 256 if n <= 1024 else 512
    arguments = (n, value, 0.0, incx, True, block)
    raw = torch.view_as_real(x)
    if entry is None:
        grid = (min((n + block - 1) // block, _scal_l1_core), 1, 1)
        config = next(c for c in _scal_l1_configs if c.kwargs["BLOCK"] == block)
        compiled, _ = _scal_l1_scal_pair[grid](
            raw, *arguments, num_warps=config.num_warps, num_stages=config.num_stages
        )
        _store_launch(key, (compiled, compiled.run, grid))
        return
    compiled, run, grid = entry
    _launch_l1_entry(compiled, run, grid, (raw,), (pointer,), arguments, device.index)


_SCAL_SPAN_CONFIGS = tuple(
    triton.Config({"BLOCK": block, "PIPE": pipe}, num_warps=1, num_stages=1)
    for block in (128, 256, 512, 1024, 2048)
    for pipe in (False, True)
    if not (block == 2048 and pipe)
)


def _scal_span_prune(configs, args, **kwargs):
    n, factor, inc = args["N"], args["F"], args["INC"]
    upper = (
        512 if n < 65536 or factor == 2 and inc > 2 else 1024 if factor == 2 else 2048
    )
    return [
        c
        for c in configs
        if c.kwargs["BLOCK"] <= upper
        and (not c.kwargs["PIPE"] or n >= 1048576)
        and not (
            args["COMPLEX_ALPHA"]
            and c.kwargs["PIPE"]
            and c.kwargs["BLOCK"] * triton.next_power_of_2(inc) * factor > 4096
        )
    ]


def _scal_span_eligible(n, x, inc):
    return (
        type(n) is int
        and n > 0
        and type(inc) is int
        and 1 < inc <= 5
        and 2 * (n + CORE_NUM * 2048) * inc < _scal_l1_safe
        and type(x) is torch.Tensor
        and not x.is_conj()
        and not x.is_neg()
        and x.data_ptr() % 64 == 0
    )


_CSCAL_PHYSICAL_CORES = tl.constexpr(CORE_NUM)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_cscal_physical"),
    key=["N", "INC"],
    restore_value=["X"],
)
@triton.jit
def _cscal_physical_kernel(
    X,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    INC: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    first = tl.program_id(0) * BLOCK
    step: tl.constexpr = (
        _CSCAL_PHYSICAL_CORES
        if 2 * ((N - 1) * INC + 1) >= BLOCK * _CSCAL_PHYSICAL_CORES
        else (2 * ((N - 1) * INC + 1) + BLOCK - 1) // BLOCK
    ) * BLOCK
    end = 2 * ((N - 1) * INC + 1)
    component = lane // 2
    if INC == 3:
        quotient = (component * 21846) >> 16
    elif INC == 5:
        quotient = (component * 52429) >> 18
    else:
        quotient = component // INC
    residue = component - quotient * INC
    if PIPE:
        value = tl.load(X + first + lane, first + lane < end, other=0.0)
    phase = (first // 2) % INC
    for start in range(first, end, step):
        index = start + lane
        if PIPE:
            nxt = tl.load(X + index + step, index + step < end, other=0.0)
        else:
            value = tl.load(X + index, index < end, other=0.0)
        target = tl.where(phase == 0, 0, INC - phase)
        active = residue == target
        peer = tl.gather(value, lane ^ 1, 0)
        coefficient = tl.where(lane & 1 == 0, -AI, AI)
        updated = AR * value + coefficient * peer
        # Preserve gap bits while loading and storing contiguous physical tiles.
        bits = tl.where(
            active, updated.to(tl.int32, bitcast=True), value.to(tl.int32, bitcast=True)
        )
        tl.store(X + index, bits.to(tl.float32, bitcast=True), index < end)
        if PIPE:
            value = nxt
        phase += (step // 2) % INC
        phase = tl.where(phase >= INC, phase - INC, phase)


def _cscal_physical(n, x, inc, ar, ai, device):
    key = ("cscal_physical", device.index, n, inc)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        raw = torch.view_as_real(x)
        end = 2 * ((n - 1) * inc + 1)
        compiled, meta = _cscal_physical_kernel[
            lambda config: (min(triton.cdiv(end, config["BLOCK"]), CORE_NUM),)
        ](raw, n, ar, ai, inc, multibuffer=False)
        assert not getattr(compiled.metadata, "workspace_size", 0)
        assert not getattr(compiled.metadata, "lock_num", 0)
        constants = (n, inc, meta["BLOCK"], meta["PIPE"])
        grid = (min(triton.cdiv(end, meta["BLOCK"]), CORE_NUM), 1, 1)
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    compiled, run, constants, grid = entry
    n, inc, block, pipe = constants
    _launch_l1_entry(
        compiled,
        run,
        grid,
        (x,),
        (x.data_ptr(),),
        (n, ar, ai, inc, block, pipe),
        device.index,
    )


def _scal_span_dense_prune(configs, args, **kwargs):
    allowed = _scal_span_large_prune(configs, args, **kwargs)
    span = triton.next_power_of_2(args["INC"]) * args["F"]
    return [
        config
        for config in allowed
        if not (
            args["COMPLEX_ALPHA"]
            and config.kwargs["PIPE"]
            and config.kwargs["BLOCK"] * span > 4096
        )
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_scal_strided_large"),
    key=["N", "INC", "F", "COMPLEX_ALPHA"],
    restore_value=["X"],
    prune_configs_by={"early_config_prune": _scal_span_dense_prune},
)
@triton.jit
def _scal_span_dense_kernel(
    X,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    INC: tl.constexpr,
    F: tl.constexpr,
    COMPLEX_ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK * triton.next_power_of_2(INC) * F)
    CHUNK: tl.constexpr = (BLOCK * triton.next_power_of_2(INC) // INC // 16) * 16
    component_index = lane // F
    if INC == 3:
        q = component_index * 21846 >> 16
    elif INC == 5:
        q = component_index * 52429 >> 18
    else:
        q = component_index // INC
    active = component_index - q * INC == 0
    begin = tl.program_id(0) * CHUNK
    step = tl.num_programs(0) * CHUNK
    if PIPE:
        i = begin * INC * F + lane
        value = tl.load(
            X + i, (lane < CHUNK * INC * F) & (i < ((N - 1) * INC + 1) * F), other=0.0
        )
    for start in range(begin, N, step):
        i = start * INC * F + lane
        mask = (lane < CHUNK * INC * F) & (i < ((N - 1) * INC + 1) * F)
        if PIPE:
            ni = (start + step) * INC * F + lane
            nxt = tl.load(
                X + ni,
                (lane < CHUNK * INC * F) & (ni < ((N - 1) * INC + 1) * F),
                other=0.0,
            )
        else:
            value = tl.load(X + i, mask, other=0.0)
        if COMPLEX_ALPHA:
            peer = tl.gather(value, lane ^ 1, axis=0)
            coefficient = tl.where(lane & 1 == 0, -AI, AI)
            result = AR * value + coefficient * peer
        else:
            result = AR * value
        bits = tl.where(
            active, result.to(tl.int32, bitcast=True), value.to(tl.int32, bitcast=True)
        )
        tl.store(X + i, bits.to(tl.float32, bitcast=True), mask)
        if PIPE:
            value = nxt


def _scal_span_call(name, n, alpha, x, inc):
    factor = 1 if name == "sscal" else 2
    expected = torch.float32 if factor == 1 else torch.complex64
    assert x.dtype == expected, f"x must be {expected}"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() >= 1 + (n - 1) * inc, "x is too short"
    value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    mode = name == "cscal"
    if mode:
        value = complex(value)
        ar, ai = value.real, value.imag
    else:
        assert not isinstance(value, complex), "alpha must be real"
        ar, ai = float(value), 0.0
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _scal_span_call(name, n, alpha, x, inc)
    dense = n >= 1048576 and (
        (not mode and inc in (3, 5) and (factor == 1 or inc == 5 or n >= 2097152))
        or (mode and inc == 2 and n < 2097152)
    )
    if (
        mode
        and not dense
        and n >= 65536
        and (inc in (3, 5) or inc == 2 and n >= 2097152)
    ):
        return _cscal_physical(n, x, inc, ar, ai, device)
    key = ("scal_span", device.index, n, inc, factor, mode)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        raw = torch.view_as_real(x) if factor == 2 else x
        kernel = (
            _scal_span_dense_kernel
            if dense
            else (
                _scal_span_large_kernel
                if n >= 1048576 and not mode
                else _scal_span_kernel
            )
        )
        compiled, meta = kernel[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), CORE_NUM),)
        ](raw, n, ar, ai, inc, factor, mode, multibuffer=False)
        constants = (inc, factor, mode, meta["BLOCK"], meta["PIPE"])
        grid = (min(triton.cdiv(n, meta["BLOCK"]), CORE_NUM), 1, 1)
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled,
        run,
        grid,
        (x,),
        (x.data_ptr(),),
        (n, ar, ai, *constants),
        device.index,
    )


@libentry()
@libtuner(
    configs=_SCAL_SPAN_CONFIGS,
    key=["N", "INC", "F", "COMPLEX_ALPHA"],
    restore_value=["X"],
    prune_configs_by={"early_config_prune": _scal_span_prune},
)
@triton.jit
def _scal_span_kernel(
    X,
    N,
    AR: tl.float32,
    AI: tl.float32,
    INC: tl.constexpr,
    F: tl.constexpr,
    COMPLEX_ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK * triton.next_power_of_2(INC) * F)
    component_index = lane // F
    if INC == 3:
        q = (component_index * 21846) >> 16
    elif INC == 5:
        q = (component_index * 52429) >> 18
    else:
        q = component_index // INC
    active = component_index - q * INC == 0
    begin = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    if PIPE:
        i = begin * INC * F + lane
        value = tl.load(
            X + i, (lane < BLOCK * INC * F) & (i < ((N - 1) * INC + 1) * F), other=0.0
        )
    for start in range(begin, N, step):
        i = start * INC * F + lane
        mask = (lane < BLOCK * INC * F) & (i < ((N - 1) * INC + 1) * F)
        if PIPE:
            ni = (start + step) * INC * F + lane
            nxt = tl.load(
                X + ni,
                (lane < BLOCK * INC * F) & (ni < ((N - 1) * INC + 1) * F),
                other=0.0,
            )
        else:
            value = tl.load(X + i, mask, other=0.0)
        if COMPLEX_ALPHA:
            peer = tl.gather(value, lane ^ 1, axis=0)
            coefficient = tl.where((lane & 1) == 0, -AI, AI)
            result = AR * value + coefficient * peer
        else:
            result = AR * value
        # Store each physical tile once, preserving all inter-element gap bits.
        bits = tl.where(
            active, result.to(tl.int32, bitcast=True), value.to(tl.int32, bitcast=True)
        )
        tl.store(X + i, bits.to(tl.float32, bitcast=True), mask)
        if PIPE:
            value = nxt


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_cscal_contiguous"),
    key=["N"],
    restore_value=["X"],
)
@triton.jit
def _cscal_contiguous_kernel(
    X,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    (first, step) = (tl.program_id(0) * BLOCK, tl.num_programs(0) * BLOCK)
    if PIPE:
        value = tl.load(X + first + lane, first + lane < 2 * N, other=0.0)
    for start in range(first, 2 * N, step):
        index = start + lane
        (offsets, mask) = (index, index < 2 * N)
        if PIPE:
            nxt = tl.load(X + index + step, index + step < 2 * N, other=0.0)
        else:
            value = tl.load(X + offsets, mask, other=0.0)
        peer = tl.gather(value, lane ^ 1, 0)
        coefficient = tl.where(lane & 1 == 0, -AI, AI)
        result = AR * value + coefficient * peer
        tl.store(X + offsets, result, mask)
        if PIPE:
            value = nxt


def _cscal_contiguous(n, alpha, x):
    assert x.dtype == torch.complex64, "x must be complex64"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() >= n, "x is too short"
    value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    ar, ai = value.real, value.imag
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _cscal_contiguous(n, alpha, x)
    key = ("cscal_contiguous", device.index, n)
    pointer = x.data_ptr()
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        raw = torch.view_as_real(x)
        compiled, meta = _cscal_contiguous_kernel[
            lambda cfg: (min(triton.cdiv(2 * n, cfg["BLOCK"]), _scal_l1_core),)
        ](raw, n, ar, ai)
        block, pipe = meta["BLOCK"], meta["PIPE"]
        grid = (min(triton.cdiv(2 * n, block), _scal_l1_core), 1, 1)
        _store_launch(key, (compiled, compiled.run, grid, block, pipe))
        return
    compiled, run, grid, block, pipe = entry
    _launch_l1_entry(
        compiled, run, grid, (x,), (pointer,), (n, ar, ai, block, pipe), device.index
    )


def _scal_span_large_prune(configs, args, **kwargs):
    # Bound the physical tile size to fit the on-chip vector buffers.
    span = triton.next_power_of_2(args["INC"]) * args["F"]
    return [c for c in configs if c.kwargs["BLOCK"] * span <= 8192]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_scal_strided_large"),
    key=["N", "INC", "F", "COMPLEX_ALPHA"],
    restore_value=["X"],
    prune_configs_by={"early_config_prune": _scal_span_large_prune},
)
@triton.jit
def _scal_span_large_kernel(
    X,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    INC: tl.constexpr,
    F: tl.constexpr,
    COMPLEX_ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK * triton.next_power_of_2(INC) * F)
    component_index = lane // F
    if INC == 3:
        q = component_index * 21846 >> 16
    elif INC == 5:
        q = component_index * 52429 >> 18
    else:
        q = component_index // INC
    active = component_index - q * INC == 0
    begin = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    if PIPE:
        i = begin * INC * F + lane
        value = tl.load(
            X + i, (lane < BLOCK * INC * F) & (i < ((N - 1) * INC + 1) * F), other=0.0
        )
    for start in range(begin, N, step):
        i = start * INC * F + lane
        mask = (lane < BLOCK * INC * F) & (i < ((N - 1) * INC + 1) * F)
        if PIPE:
            ni = (start + step) * INC * F + lane
            nxt = tl.load(
                X + ni,
                (lane < BLOCK * INC * F) & (ni < ((N - 1) * INC + 1) * F),
                other=0.0,
            )
        else:
            value = tl.load(X + i, mask, other=0.0)
        if COMPLEX_ALPHA:
            peer = tl.gather(value, lane ^ 1, axis=0)
            coefficient = tl.where(lane & 1 == 0, -AI, AI)
            result = AR * value + coefficient * peer
        else:
            result = AR * value
        bits = tl.where(
            active, result.to(tl.int32, bitcast=True), value.to(tl.int32, bitcast=True)
        )
        tl.store(X + i, bits.to(tl.float32, bitcast=True), mask)
        if PIPE:
            value = nxt


def _csscal_contiguous_prune(configs, args, **kw):
    upper = min(8192, max(256, triton.next_power_of_2(args["N"])))
    return [
        c
        for c in configs
        if c.kwargs["BLOCK"] <= upper and c.kwargs["BLOCK"] >= min(256, upper)
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_memory"),
    key=["N"],
    restore_value=["X"],
    prune_configs_by={"early_config_prune": _csscal_contiguous_prune},
)
@triton.jit
def _csscal_contiguous_kernel(
    X, N: tl.constexpr, ALPHA: tl.float32, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        i = start + lane
        value = tl.load(X + i, i < N, other=0.0)
        tl.store(X + i, ALPHA * value, i < N)


def _csscal_contiguous(n, alpha, x):
    assert x.dtype == torch.complex64, "x must be complex64"
    assert x.dim() == 1 and x.is_contiguous(), "x must be contiguous and 1-dimensional"
    assert x.numel() >= n, "x is too short"
    value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    assert not isinstance(value, complex), "alpha must be real for csscal"
    value = float(value)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _csscal_contiguous(n, value, x)
    pointer = x.data_ptr()
    key = ("csscal_contiguous", device.index, n)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        compiled, meta = _csscal_contiguous_kernel[
            lambda c: (min(triton.cdiv(2 * n, c["BLOCK"]), CORE_NUM),)
        ](torch.view_as_real(x), 2 * n, value)
        block = meta["BLOCK"]
        grid = (min(triton.cdiv(2 * n, block), CORE_NUM), 1, 1)
        _store_launch(key, (compiled, compiled.run, grid, block))
        return
    compiled, run, grid, block = entry
    _launch_l1_entry(
        compiled, run, grid, (x,), (pointer,), (2 * n, value, block), device.index
    )

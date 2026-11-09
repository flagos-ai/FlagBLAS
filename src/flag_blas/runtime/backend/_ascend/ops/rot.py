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
            yr = tl.load(Y + 2 * yo, mask, other=0.0)
            yi = tl.load(Y + 2 * yo + 1, mask, other=0.0)
            tl.store(X + 2 * xo, C * xr + S * yr, mask)
            tl.store(X + 2 * xo + 1, C * xi + S * yi, mask)
            (nr, ni) = (C * yr - S * xr, C * yi - S * xi)
            tl.store(Y + 2 * yo, nr, mask)
            tl.store(Y + 2 * yo + 1, ni, mask)
        else:
            x = tl.load(X + xo, mask, other=0.0)
            y = tl.load(Y + yo, mask, other=0.0)
            tl.store(X + xo, C * x + S * y, mask)
            value = C * y - S * x
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
            raw_x, raw_y, n, value.real, value.imag, c, s, incx, incy, x.is_complex(), 4
        )


_rotation_l1_core_num = CORE_NUM


_rotation_l1_configs = tuple(runtime.get_tuned_config("ascend_l1_rotation"))


def _rotation_l1_prune(configs, args, **kwargs):
    n = args["N"]
    if n <= 4096:
        upper = min(1024, max(256, triton.next_power_of_2(n)))
        return [cfg for cfg in configs if cfg.kwargs["BLOCK"] <= upper]
    return [cfg for cfg in configs if cfg.kwargs["BLOCK"] >= 1024]


def _rotation_l1_safe(n, incx, incy, complex_data):
    if not (
        type(n) is int
        and type(incx) is int
        and type(incy) is int
        and n > 0
        and incx == incy == 1
    ):
        return False
    factor = 2 if complex_data else 1
    stride = max(incx, incy)
    return (
        factor * n * stride
        < 2147483647 - factor * (_rotation_l1_core_num + 1) * 8192 * stride
    )


def _rotation_l1_validate(n, x, incx, y, incy, dtype):
    assert x.dtype == dtype, f"x must be {str(dtype).replace('torch.', '')}"
    assert y.dtype == dtype, f"y must be {str(dtype).replace('torch.', '')}"
    assert x.device == y.device
    assert x.dim() == y.dim() == 1
    assert x.is_contiguous() and y.is_contiguous()
    assert x.numel() >= 1 + (n - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy


def _rotation_l1_launch(n, x, incx, y, incy, op, alpha, c, s, kernel):
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _rotation_l1_launch(n, x, incx, y, incy, op, alpha, c, s, kernel)
    pointers = x.data_ptr(), y.data_ptr()
    complex_data = x.is_complex()
    key = (
        "rotation",
        device.index,
        "update",
        n,
        incx,
        incy,
        complex_data,
        op,
        (pointers[0] % 16 == 0, pointers[1] % 16 == 0),
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        raw_x = torch.view_as_real(x) if complex_data else x
        raw_y = torch.view_as_real(y) if complex_data else y
        compiled, meta = kernel[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), _rotation_l1_core_num),)
        ](raw_x, raw_y, n, alpha.real, alpha.imag, c, s, incx, incy, complex_data, op)
        block = meta["BLOCK"]
        grid = min(triton.cdiv(n, block), _rotation_l1_core_num), 1, 1
        constants = incx, incy, complex_data, op, block
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled,
        run,
        grid,
        (x, y),
        pointers,
        (n, alpha.real, alpha.imag, c, s, *constants),
        device.index,
    )


_rotation_l1_pair_configs = tuple(
    runtime.get_tuned_config("ascend_l1_rotation_strided")
)


def _rotation_l1_pair_prune(configs, args, **kwargs):
    if args["N"] <= 1024:
        return [c for c in configs if c.kwargs["BLOCK"] == 256]
    return configs


@libentry()
@libtuner(
    configs=_rotation_l1_pair_configs,
    key=["N", "IX", "IY", "ROT"],
    prune_configs_by={"early_config_prune": _rotation_l1_pair_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _rotation_l1_pair_update(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    C: tl.float32,
    S: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    column = tl.arange(0, 2)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        logical = start + lane
        active = logical < N
        xp = logical[:, None] * (2 * IX) + column[None, :]
        yp = logical[:, None] * (2 * IY) + column[None, :]
        x = tl.load(X + xp, active[:, None], other=0.0)
        y = tl.load(Y + yp, active[:, None], other=0.0)
        nx = C * x + S * y
        ny = C * y - S * x
        tl.store(X + xp, nx, active[:, None])
        tl.store(Y + yp, ny, active[:, None])


def _rotation_l1_pair_eligible(n, x, ix, y, iy):
    return (
        type(n) is int
        and n > 0
        and type(ix) is int
        and type(iy) is int
        and 0 < ix <= 8
        and 0 < iy <= 8
        and max(ix, iy) > 1
        and 2 * (n + CORE_NUM * 1024) * max(ix, iy) < 2147483647 - 32768
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and not x.is_conj()
        and not y.is_conj()
        and not x.is_neg()
        and not y.is_neg()
        and x.data_ptr() % 64 == y.data_ptr() % 64 == 0
    )


def _rotation_l1_pair_call(
    name, n, x, ix, y, iy, alpha=1 + 0j, c=0.0, s=0.0, *, fallback
):
    if name == "csrot" and ix == iy == 1:
        return _rotation_l1_csrot(n, x, ix, y, iy, c, s)
    if not _rotation_l1_pair_eligible(n, x, ix, y, iy):
        return fallback(name, n, x, ix, y, iy, alpha, c, s)
    _rotation_l1_validate(n, x, ix, y, iy, torch.complex64)
    px, py = x.data_ptr(), y.data_ptr()
    if px < py + (1 + (n - 1) * iy) * 8 and py < px + (1 + (n - 1) * ix) * 8:
        return fallback(name, n, x, ix, y, iy, alpha, c, s)
    value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    c = float(c.item() if isinstance(c, torch.Tensor) else c)
    s = float(s.item() if isinstance(s, torch.Tensor) else s)
    if name == "caxpy" and value == 0:
        return
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _rotation_l1_pair_call(
                name, n, x, ix, y, iy, value, c, s, fallback=fallback
            )
    rot = name == "csrot"
    key = "pair", device.index, name, n, ix, iy
    entry = _LAUNCH_CACHE.get(key)
    # Fresh typed views: no tensor, address, coefficient or stream is retained.
    raw_x, raw_y = torch.view_as_real(x), torch.view_as_real(y)
    if entry is None:
        compiled, meta = _rotation_l1_pair_update[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), CORE_NUM),)
        ](raw_x, raw_y, n, value.real, value.imag, c, s, ix, iy, rot)
        block = meta["BLOCK"]
        grid = min(triton.cdiv(n, block), CORE_NUM), 1, 1
        _store_launch(key, (compiled, compiled.run, (ix, iy, rot, block), grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled,
        run,
        grid,
        (raw_x, raw_y),
        (px, py),
        (n, value.real, value.imag, c, s, *constants),
        device.index,
    )


def _rot(name, n, x, incx, y, incy, c, s):
    common = getattr(import_module("flag_blas.ops.level1.rot"), name)
    if n <= 0:
        return common(n, x, incx, y, incy, c, s)
    common(0, x, incx, y, incy, c, s)
    return _launch(n, x, incx, y, incy, 4, c=c, s=s)


def srot(n, x, incx, y, incy, c, s):
    if _pair_span_eligible(n, x, incx, y, incy):
        return _pair_span_call("srot", n, x, incx, y, incy, c, s)
    return _rotation_l1_srot(n, x, incx, y, incy, c, s)


def csrot(n, x, incx, y, incy, c, s):
    if _pair_span_eligible(n, x, incx, y, incy):
        return _pair_span_call("csrot", n, x, incx, y, incy, c, s)
    return _rotation_l1_pair_call(
        "csrot", n, x, incx, y, incy, c=c, s=s, fallback=_csrot_pair_fallback
    )


def _rotation_l1_srot_fallback(n, x, incx, y, incy, c, s):
    return _rot("srot", n, x, incx, y, incy, c, s)


def _rotation_l1_csrot_fallback(n, x, incx, y, incy, c, s):
    return _rot("csrot", n, x, incx, y, incy, c, s)


@libentry()
@libtuner(
    configs=_rotation_l1_configs,
    key=["N", "IX", "IY", "COMPLEX", "OP"],
    prune_configs_by={"early_config_prune": _rotation_l1_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _rotation_l1_update(
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
        if COMPLEX:
            component = tl.arange(0, 2)
            xo = index[:, None] * (2 * IX) + component[None, :]
            yo = index[:, None] * (2 * IY) + component[None, :]
            x = tl.load(X + xo, mask[:, None], other=0.0)
            y = tl.load(Y + yo, mask[:, None], other=0.0)
            tl.store(X + xo, C * x + S * y, mask[:, None])
            tl.store(Y + yo, C * y - S * x, mask[:, None])
        else:
            xo = index * IX
            yo = index * IY
            x = tl.load(X + xo, mask, other=0.0)
            y = tl.load(Y + yo, mask, other=0.0)
            tl.store(X + xo, C * x + S * y, mask)
            tl.store(Y + yo, C * y - S * x, mask)


def _rotation_l1_rot(n, x, incx, y, incy, c, s, complex_data):
    if not _rotation_l1_safe(n, incx, incy, complex_data):
        original = (
            _rotation_l1_csrot_fallback if complex_data else _rotation_l1_srot_fallback
        )
        return original(n, x, incx, y, incy, c, s)
    _rotation_l1_validate(
        n, x, incx, y, incy, torch.complex64 if complex_data else torch.float32
    )
    c = float(c.item() if isinstance(c, torch.Tensor) else c)
    s = float(s.item() if isinstance(s, torch.Tensor) else s)
    return _rotation_l1_launch(
        n, x, incx, y, incy, 4, 1.0 + 0.0j, c, s, _rotation_l1_update
    )


def _rotation_l1_srot(n, x, incx, y, incy, c, s):
    return _rotation_l1_rot(n, x, incx, y, incy, c, s, False)


def _rotation_l1_csrot(n, x, incx, y, incy, c, s):
    return _rotation_l1_rot(n, x, incx, y, incy, c, s, True)


def _csrot_pair_fallback(name, n, x, ix, y, iy, alpha, c, s):
    return _rotation_l1_csrot(n, x, ix, y, iy, c, s)


_PAIR_SPAN_CONFIGS = tuple(
    triton.Config({"BLOCK": block, "PIPE": pipe}, num_warps=1, num_stages=1)
    for block in (128, 256, 512, 1024, 2048)
    for pipe in (False, True)
    if not (block >= 2048 and pipe)
)


def _pair_span_prune(configs, args, **kwargs):
    n, factor, step = args["N"], args["F"], max(args["IX"], args["IY"])
    upper = (
        512 if n < 65536 or factor == 2 and step > 2 else 1024 if factor == 2 else 2048
    )
    return [
        cfg
        for cfg in configs
        if cfg.kwargs["BLOCK"] <= upper
        and (not cfg.kwargs["PIPE"] or n >= 1048576)
        and (
            not cfg.kwargs["PIPE"]
            or cfg.kwargs["BLOCK"] < 1024
            or factor == 1
            and step == 3
            and n >= 2097152
        )
    ]


def _pair_span_eligible(n, x, ix, y, iy):
    if not (
        type(n) is int
        and n > 0
        and type(ix) is int
        and type(iy) is int
        and 0 < ix <= 5
        and 0 < iy <= 5
        and max(ix, iy) > 1
        and 2 * (n + CORE_NUM * 2048) * max(ix, iy) < 2147483647 - 65536
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and not x.is_conj()
        and not y.is_conj()
        and not x.is_neg()
        and not y.is_neg()
    ):
        return False
    px, py = x.data_ptr(), y.data_ptr()
    size = x.element_size()
    return px % 64 == py % 64 == 0 and not (
        px < py + ((n - 1) * iy + 1) * size and py < px + ((n - 1) * ix + 1) * size
    )


def _pair_span_call(name, n, x, ix, y, iy, c=0.0, s=0.0):
    factor = 2 if name.startswith("c") else 1
    dtype = torch.complex64 if factor == 2 else torch.float32
    assert x.dtype == y.dtype == dtype, f"x and y must be {dtype}"
    assert x.dim() == y.dim() == 1, "x and y must be 1-dimensional"
    assert x.is_contiguous() and y.is_contiguous(), "x and y must be contiguous"
    assert x.device == y.device, "x and y must be on the same device"
    assert (
        x.numel() >= (n - 1) * ix + 1 and y.numel() >= (n - 1) * iy + 1
    ), "vector is too short"
    c = float(c.item() if isinstance(c, torch.Tensor) else c)
    s = float(s.item() if isinstance(s, torch.Tensor) else s)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _pair_span_call(name, n, x, ix, y, iy, c, s)
    key = ("span", device.index, n, ix, iy, factor)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        rx, ry = (
            (torch.view_as_real(x), torch.view_as_real(y)) if factor == 2 else (x, y)
        )
        kernel = (
            _srot_span_kernel
            if factor == 1 and n >= 1048576 and max(ix, iy) <= 3
            else _csrot_span_unrolled
            if factor == 2 and n >= 1048576 and ix == 3 and iy == 2
            else _pair_span_kernel
        )
        compiled, meta = kernel[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), CORE_NUM),)
        ](rx, ry, n, c, s, ix, iy, factor, multibuffer=False)
        constants = (ix, iy, factor, meta["BLOCK"], meta["PIPE"])
        grid = (min(triton.cdiv(n, meta["BLOCK"]), CORE_NUM), 1, 1)
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled,
        run,
        grid,
        (x, y),
        (x.data_ptr(), y.data_ptr()),
        (n, c, s, *constants),
        device.index,
    )


@triton.jit
def _pair_span_quotient(lane, factor: tl.constexpr, inc: tl.constexpr):
    i = lane // factor
    if inc == 3:
        q = (i * 21846) >> 16
    elif inc == 5:
        q = (i * 52429) >> 18
    else:
        q = i // inc
    return q


@libentry()
@libtuner(
    configs=_PAIR_SPAN_CONFIGS,
    key=["N", "IX", "IY", "F"],
    restore_value=["X", "Y"],
    prune_configs_by={"early_config_prune": _pair_span_prune},
)
@triton.jit
def _pair_span_kernel(
    X,
    Y,
    N,
    C: tl.float32,
    S: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    qx = _pair_span_quotient(lx, F, IX)
    qy = _pair_span_quotient(ly, F, IY)
    ax = lx // F - qx * IX == 0
    ay = ly // F - qy * IY == 0
    map_y = tl.minimum(
        qx * IY * F + (lx & (F - 1)), BLOCK * triton.next_power_of_2(IY) * F - 1
    )
    map_x = tl.minimum(
        qy * IX * F + (ly & (F - 1)), BLOCK * triton.next_power_of_2(IX) * F - 1
    )
    CHUNK: tl.constexpr = (BLOCK * 4 // 3 // 16) * 16 if IX == 3 and IY == 3 else BLOCK
    begin = tl.program_id(0) * CHUNK
    step = tl.num_programs(0) * CHUNK
    if PIPE:
        px = begin * IX * F + lx
        py = begin * IY * F + ly
        xv = tl.load(
            X + px, (lx < CHUNK * IX * F) & (px < ((N - 1) * IX + 1) * F), other=0.0
        )
        yv = tl.load(
            Y + py, (ly < CHUNK * IY * F) & (py < ((N - 1) * IY + 1) * F), other=0.0
        )
    for start in range(begin, N, step):
        px = start * IX * F + lx
        py = start * IY * F + ly
        mx = (lx < CHUNK * IX * F) & (px < ((N - 1) * IX + 1) * F)
        my = (ly < CHUNK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        if PIPE:
            nx = (start + step) * IX * F + lx
            ny = (start + step) * IY * F + ly
            xv_next = tl.load(
                X + nx, (lx < CHUNK * IX * F) & (nx < ((N - 1) * IX + 1) * F), other=0.0
            )
            yv_next = tl.load(
                Y + ny, (ly < CHUNK * IY * F) & (ny < ((N - 1) * IY + 1) * F), other=0.0
            )
        else:
            xv = tl.load(X + px, mx, other=0.0)
            yv = tl.load(Y + py, my, other=0.0)
        if IX == IY:
            xy, yx = yv, xv
        else:
            xy = tl.gather(yv, map_y, axis=0)
            yx = tl.gather(xv, map_x, axis=0)
        ox, oy = C * xv + S * xy, C * yv - S * yx
        # Physical tiles are disjoint; retain original bits in vector gaps.
        bx = tl.where(ax, ox.to(tl.int32, bitcast=True), xv.to(tl.int32, bitcast=True))
        by = tl.where(ay, oy.to(tl.int32, bitcast=True), yv.to(tl.int32, bitcast=True))
        tl.store(X + px, bx.to(tl.float32, bitcast=True), mx)
        tl.store(Y + py, by.to(tl.float32, bitcast=True), my)
        if PIPE:
            xv, yv = xv_next, yv_next


_SROT_SPAN_CORES = tl.constexpr(CORE_NUM)


def _srot_span_prune(configs, args, **kwargs):
    step = max(args["IX"], args["IY"])
    if step <= 2:
        return [
            c
            for c in configs
            if c.kwargs["BLOCK"] in (1024, 2048)
            and (c.kwargs["BLOCK"] < 2048 or not c.kwargs["PIPE"])
        ]
    blocks = (512, 1024) if args["N"] < 2097152 else (1024,)
    return [c for c in configs if c.kwargs["BLOCK"] in blocks and c.kwargs["PIPE"]]


@libentry()
@libtuner(
    configs=_PAIR_SPAN_CONFIGS,
    key=["N", "IX", "IY", "F"],
    restore_value=["X", "Y"],
    prune_configs_by={"early_config_prune": _srot_span_prune},
)
@triton.jit
def _srot_span_kernel(
    X,
    Y,
    N: tl.constexpr,
    C: tl.float32,
    S: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    qx = _pair_span_quotient(lx, F, IX)
    qy = _pair_span_quotient(ly, F, IY)
    ax = lx // F - qx * IX == 0
    ay = ly // F - qy * IY == 0
    map_y = tl.minimum(
        qx * IY * F + (lx & (F - 1)), BLOCK * triton.next_power_of_2(IY) * F - 1
    )
    map_x = tl.minimum(
        qy * IX * F + (ly & (F - 1)), BLOCK * triton.next_power_of_2(IX) * F - 1
    )
    CHUNK: tl.constexpr = (BLOCK * 4 // 3 // 16) * 16 if IX == 3 and IY == 3 else BLOCK
    begin = tl.program_id(0) * CHUNK
    step = _SROT_SPAN_CORES * CHUNK
    if PIPE:
        px = begin * IX * F + lx
        py = begin * IY * F + ly
        xv = tl.load(
            X + px, (lx < CHUNK * IX * F) & (px < ((N - 1) * IX + 1) * F), other=0.0
        )
        yv = tl.load(
            Y + py, (ly < CHUNK * IY * F) & (py < ((N - 1) * IY + 1) * F), other=0.0
        )
    for start in range(begin, N, step):
        px = start * IX * F + lx
        py = start * IY * F + ly
        mx = (lx < CHUNK * IX * F) & (px < ((N - 1) * IX + 1) * F)
        my = (ly < CHUNK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        if PIPE:
            nx = (start + step) * IX * F + lx
            ny = (start + step) * IY * F + ly
            xv_next = tl.load(
                X + nx, (lx < CHUNK * IX * F) & (nx < ((N - 1) * IX + 1) * F), other=0.0
            )
            yv_next = tl.load(
                Y + ny, (ly < CHUNK * IY * F) & (ny < ((N - 1) * IY + 1) * F), other=0.0
            )
        else:
            xv = tl.load(X + px, mx, other=0.0)
            yv = tl.load(Y + py, my, other=0.0)
        if IX == IY:
            xy, yx = yv, xv
        else:
            xy = tl.gather(yv, map_y, axis=0)
            yx = tl.gather(xv, map_x, axis=0)
        ox, oy = C * xv + S * xy, C * yv - S * yx
        # Physical tiles are disjoint; retain original bits in vector gaps.
        bx = tl.where(ax, ox.to(tl.int32, bitcast=True), xv.to(tl.int32, bitcast=True))
        by = tl.where(ay, oy.to(tl.int32, bitcast=True), yv.to(tl.int32, bitcast=True))
        tl.store(X + px, bx.to(tl.float32, bitcast=True), mx)
        tl.store(Y + py, by.to(tl.float32, bitcast=True), my)
        if PIPE:
            xv, yv = xv_next, yv_next


@libentry()
@libtuner(
    configs=_PAIR_SPAN_CONFIGS,
    key=["N", "IX", "IY", "F"],
    restore_value=["X", "Y"],
    prune_configs_by={"early_config_prune": _pair_span_prune},
)
@triton.jit
def _csrot_span_unrolled(
    X,
    Y,
    N,
    C: tl.float32,
    S: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
):
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    qx = _pair_span_quotient(lx, F, IX)
    qy = _pair_span_quotient(ly, F, IY)
    ax = lx // F - qx * IX == 0
    ay = ly // F - qy * IY == 0
    map_y = tl.minimum(
        qx * IY * F + (lx & (F - 1)), BLOCK * triton.next_power_of_2(IY) * F - 1
    )
    map_x = tl.minimum(
        qy * IX * F + (ly & (F - 1)), BLOCK * triton.next_power_of_2(IX) * F - 1
    )
    CHUNK: tl.constexpr = (BLOCK * 4 // 3 // 16) * 16 if IX == 3 and IY == 3 else BLOCK
    begin = tl.program_id(0) * CHUNK
    step = tl.num_programs(0) * CHUNK
    if PIPE:
        px = begin * IX * F + lx
        py = begin * IY * F + ly
        xv = tl.load(
            X + px, (lx < CHUNK * IX * F) & (px < ((N - 1) * IX + 1) * F), other=0.0
        )
        yv = tl.load(
            Y + py, (ly < CHUNK * IY * F) & (py < ((N - 1) * IY + 1) * F), other=0.0
        )
    for start in tl.range(begin, N, step, loop_unroll_factor=2):
        px = start * IX * F + lx
        py = start * IY * F + ly
        mx = (lx < CHUNK * IX * F) & (px < ((N - 1) * IX + 1) * F)
        my = (ly < CHUNK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        if PIPE:
            nx = (start + step) * IX * F + lx
            ny = (start + step) * IY * F + ly
            xv_next = tl.load(
                X + nx, (lx < CHUNK * IX * F) & (nx < ((N - 1) * IX + 1) * F), other=0.0
            )
            yv_next = tl.load(
                Y + ny, (ly < CHUNK * IY * F) & (ny < ((N - 1) * IY + 1) * F), other=0.0
            )
        else:
            xv = tl.load(X + px, mx, other=0.0)
            yv = tl.load(Y + py, my, other=0.0)
        if IX == IY:
            xy, yx = yv, xv
        else:
            xy = tl.gather(yv, map_y, axis=0)
            yx = tl.gather(xv, map_x, axis=0)
        ox, oy = C * xv + S * xy, C * yv - S * yx
        # Physical tiles are disjoint; retain original bits in vector gaps.
        bx = tl.where(ax, ox.to(tl.int32, bitcast=True), xv.to(tl.int32, bitcast=True))
        by = tl.where(ay, oy.to(tl.int32, bitcast=True), yv.to(tl.int32, bitcast=True))
        tl.store(X + px, bx.to(tl.float32, bitcast=True), mx)
        tl.store(Y + py, by.to(tl.float32, bitcast=True), my)
        if PIPE:
            xv, yv = xv_next, yv_next

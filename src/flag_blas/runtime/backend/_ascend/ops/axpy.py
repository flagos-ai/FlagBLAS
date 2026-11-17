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
            yr = tl.load(Y + 2 * yo, mask, other=0.0)
            yi = tl.load(Y + 2 * yo + 1, mask, other=0.0)
            (nr, ni) = (yr + AR * xr - AI * xi, yi + AR * xi + AI * xr)
            tl.store(Y + 2 * yo, nr, mask)
            tl.store(Y + 2 * yo + 1, ni, mask)
        else:
            x = tl.load(X + xo, mask, other=0.0)
            y = tl.load(Y + yo, mask, other=0.0)
            value = y + AR * x
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
    if value == 0:
        return
    raw_x = torch.view_as_real(x) if x.is_complex() else x
    raw_y = torch.view_as_real(y) if y.is_complex() else y
    with torch_device_fn.device(x.device):
        _vector_update[lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), CORE_NUM),)](
            raw_x, raw_y, n, value.real, value.imag, c, s, incx, incy, x.is_complex(), 3
        )


_axpy_l1_core_num = CORE_NUM


def _axpy_l1_safe(n, incx, incy, complex_data):
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
        < 2147483647 - factor * (_axpy_l1_core_num + 1) * 8192 * stride
    )


def _axpy_l1_validate(n, x, incx, y, incy, dtype):
    assert x.dtype == dtype, f"x must be {str(dtype).replace('torch.', '')}"
    assert y.dtype == dtype, f"y must be {str(dtype).replace('torch.', '')}"
    assert x.device == y.device
    assert x.dim() == y.dim() == 1
    assert x.is_contiguous() and y.is_contiguous()
    assert x.numel() >= 1 + (n - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy


@lru_cache(maxsize=None)
def _axpy_l2_offset(device):
    # Stream read-only X around L2; leave the read/write Y on its normal address.
    if not torch.npu.get_device_name(device).startswith("Ascend910B"):
        return 0
    try:
        query = ctypes.CDLL("libruntime.so").rtGetL2CacheOffset
        query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int32
        offset = ctypes.c_uint64(0)
        return offset.value if query(device, ctypes.byref(offset)) == 0 else 0
    except (AttributeError, OSError):
        return 0


def _axpy_l1_launch(n, x, incx, y, incy, op, alpha, c, s, kernel):
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _axpy_l1_launch(n, x, incx, y, incy, op, alpha, c, s, kernel)
    pointers = x.data_ptr(), y.data_ptr()
    complex_data = x.is_complex()
    # Only disjoint large vectors use streaming reads; retain the original
    # kernel and tuning path for small vectors and aliases.
    stream_x = (
        n >= 1048576
        and incx == incy == 1
        and (
            pointers[0] + n * x.element_size() <= pointers[1]
            or pointers[1] + n * y.element_size() <= pointers[0]
        )
    )
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
    if stream_x:
        key += ("stream_x",)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        raw_x = torch.view_as_real(x) if complex_data else x
        raw_y = torch.view_as_real(y) if complex_data else y
        offset = _axpy_l2_offset(device.index) if stream_x else 0
        options = {}
        if offset:
            kernel = _axpy_l1_flat_stream_update
            options = {"L2_OFFSET": offset, "multibuffer": False}
        compiled, meta = kernel[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), _axpy_l1_core_num),)
        ](
            raw_x,
            raw_y,
            n,
            alpha.real,
            alpha.imag,
            c,
            s,
            incx,
            incy,
            complex_data,
            op,
            **options,
        )
        block = meta["BLOCK"]
        grid = min(triton.cdiv(n, block), _axpy_l1_core_num), 1, 1
        constants = incx, incy, complex_data, op, block
        if offset:
            constants += (offset,)
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


_axpy_l1_pair_configs = tuple(runtime.get_tuned_config("ascend_l1_rotation_strided"))


def _axpy_l1_pair_prune(configs, args, **kwargs):
    if args["N"] <= 1024:
        return [c for c in configs if c.kwargs["BLOCK"] == 256]
    return configs


@libentry()
@libtuner(
    configs=_axpy_l1_pair_configs,
    key=["N", "IX", "IY", "ROT"],
    prune_configs_by={"early_config_prune": _axpy_l1_pair_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _axpy_l1_pair_update(
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
    peer_column = tl.broadcast_to((column ^ 1)[None, :], (BLOCK, 2))
    imag_coefficient = tl.where(column == 0, -AI, AI)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        logical = start + lane
        active = logical < N
        xp = logical[:, None] * (2 * IX) + column[None, :]
        yp = logical[:, None] * (2 * IY) + column[None, :]
        x = tl.load(X + xp, active[:, None], other=0.0)
        y = tl.load(Y + yp, active[:, None], other=0.0)
        peer = tl.gather(x, peer_column, axis=1)
        ny = y + AR * x + imag_coefficient[None, :] * peer
        tl.store(Y + yp, ny, active[:, None])


def _axpy_l1_pair_eligible(n, x, ix, y, iy):
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


def _axpy_l1_pair_call(name, n, x, ix, y, iy, alpha=1 + 0j, c=0.0, s=0.0, *, fallback):
    if name == "caxpy" and ix == iy == 1:
        return _axpy_l1_caxpy(n, alpha, x, ix, y, iy)
    if not _axpy_l1_pair_eligible(n, x, ix, y, iy):
        return fallback(name, n, x, ix, y, iy, alpha, c, s)
    _axpy_l1_validate(n, x, ix, y, iy, torch.complex64)
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
            return _axpy_l1_pair_call(
                name, n, x, ix, y, iy, value, c, s, fallback=fallback
            )
    rot = name == "csrot"
    key = "pair", device.index, name, n, ix, iy
    entry = _LAUNCH_CACHE.get(key)
    # Fresh typed views: no tensor, address, coefficient or stream is retained.
    raw_x, raw_y = torch.view_as_real(x), torch.view_as_real(y)
    if entry is None:
        compiled, meta = _axpy_l1_pair_update[
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


_SAXPY_CONTIGUOUS_CONFIGS = tuple(
    config
    for config in runtime.get_tuned_config("ascend_l1_saxpy")
    if 2048 <= config.kwargs["BLOCK"] <= 8192
    and config.kwargs["BLOCK"] & (config.kwargs["BLOCK"] - 1) == 0
    and config.num_warps == config.num_stages == 1
)


_SAXPY_SMALL_CONFIG = min(_SAXPY_CONTIGUOUS_CONFIGS, key=lambda c: c.kwargs["BLOCK"])


_SAXPY_LARGE_CONFIG = max(_SAXPY_CONTIGUOUS_CONFIGS, key=lambda c: c.kwargs["BLOCK"])


_SAXPY_SAFE_LIMIT = 2147483647 - CORE_NUM * _SAXPY_LARGE_CONFIG.kwargs["BLOCK"]


@libentry()
@triton.jit
def _saxpy_contiguous(X, Y, N: tl.constexpr, ALPHA: tl.float32, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        index = start + lane
        x = tl.load(X + index, index < N, other=0.0)
        y = tl.load(Y + index, index < N, other=0.0)
        tl.store(Y + index, y + ALPHA * x, index < N)


def _axpy(name, n, alpha, x, incx, y, incy):
    common = getattr(import_module("flag_blas.ops.level1.axpy"), name)
    if n <= 0:
        return common(n, alpha, x, incx, y, incy)
    common(0, alpha, x, incx, y, incy)
    if name == "saxpy":
        alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    return _launch(n, x, incx, y, incy, 3, alpha=alpha)


def _saxpy_before_span(n, alpha, x, incx, y, incy):
    if (
        type(n) is not int
        or type(incx) is not int
        or type(incy) is not int
        or n <= 0
        or incx != 1
        or incy != 1
    ):
        return _axpy("saxpy", n, alpha, x, incx, y, incy)
    assert (
        x.dtype == torch.float32 and y.dtype == torch.float32
    ), "x and y must be float32"
    device = x.device
    assert device == y.device, "x and y must be on the same device"
    assert x.dim() == y.dim() == 1, "x and y must be 1-dimensional"
    assert x.is_contiguous() and y.is_contiguous(), "x and y must be contiguous"
    assert x.numel() >= n, "x is too short"
    assert y.numel() >= n, "y is too short"
    value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if value == 0:
        return
    # Keep every padded tile index in signed int32.
    if n > _SAXPY_SAFE_LIMIT:
        return _axpy("saxpy", n, value, x, incx, y, incy)
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return saxpy(n, value, x, incx, y, incy)
    pointers = x.data_ptr(), y.data_ptr()
    if pointers[0] % 32 != 0 or pointers[1] % 64 != 0:
        return _axpy("saxpy", n, value, x, incx, y, incy)
    key = "saxpy", device.index, n
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        config = (
            _SAXPY_SMALL_CONFIG
            if n <= CORE_NUM * _SAXPY_LARGE_CONFIG.kwargs["BLOCK"]
            else _SAXPY_LARGE_CONFIG
        )
        block = config.kwargs["BLOCK"]
        grid = (min(triton.cdiv(n, block), CORE_NUM), 1, 1)
        compiled, _ = _saxpy_contiguous[grid](
            x,
            y,
            n,
            value,
            block,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _store_launch(key, (compiled, compiled.run, (block,), grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled, run, grid, (x, y), pointers, (n, value, *constants), device.index
    )


def _caxpy_before_span(n, alpha, x, incx, y, incy):
    return _axpy_l1_pair_call(
        "caxpy", n, x, incx, y, incy, alpha=alpha, fallback=_caxpy_pair_fallback
    )


_axpy_l1_flat_configs = tuple(runtime.get_tuned_config("ascend_l1_caxpy_contiguous"))


def _axpy_l1_caxpy_fallback(n, alpha, x, incx, y, incy):
    return _axpy("caxpy", n, alpha, x, incx, y, incy)


def _axpy_l1_flat_prune(configs, args, **kwargs):
    n = args["N"]
    if n <= 4096:
        upper = max(
            min(cfg.kwargs["BLOCK"] for cfg in configs),
            min(1024, max(256, triton.next_power_of_2(n))),
        )
        return [cfg for cfg in configs if cfg.kwargs["BLOCK"] <= upper]
    return [cfg for cfg in configs if cfg.kwargs["BLOCK"] >= 1024]


@libentry()
@libtuner(
    configs=_axpy_l1_flat_configs,
    key=["N", "IX", "IY", "COMPLEX", "OP"],
    prune_configs_by={"early_config_prune": _axpy_l1_flat_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _axpy_l1_flat_update(
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
    if IX == 1 and IY == 1:
        for start in range(tl.program_id(0) * BLOCK, 2 * N, tl.num_programs(0) * BLOCK):
            index = start + lane
            mask = index < 2 * N
            x = tl.load(X + index, mask, other=0.0)
            y = tl.load(Y + index, mask, other=0.0)
            partner = tl.gather(x, lane ^ 1, 0)
            factor = tl.where(lane & 1 == 0, -AI, AI)
            tl.store(Y + index, y + AR * x + factor * partner, mask)
    else:
        for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
            index = start + lane
            mask = index < N
            component = tl.arange(0, 2)
            xo = index[:, None] * (2 * IX) + component[None, :]
            yo = index[:, None] * (2 * IY) + component[None, :]
            x = tl.load(X + xo, mask[:, None], other=0.0)
            y = tl.load(Y + yo, mask[:, None], other=0.0)
            (xr, xi) = tl.split(x)
            (yr, yi) = tl.split(y)
            tl.store(
                Y + yo,
                tl.join(yr + AR * xr - AI * xi, yi + AR * xi + AI * xr),
                mask[:, None],
            )


@libentry()
@libtuner(
    configs=_axpy_l1_flat_configs,
    key=["N", "IX", "IY", "COMPLEX", "OP", "L2_OFFSET"],
    prune_configs_by={"early_config_prune": _axpy_l1_flat_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _axpy_l1_flat_stream_update(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    C: tl.float32,
    S: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    COMPLEX: tl.constexpr,
    OP: tl.constexpr,
    BLOCK: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
):
    if L2_OFFSET:
        X = (X.to(tl.uint64) + L2_OFFSET).to(tl.pointer_type(tl.float32))
    lane = tl.arange(0, BLOCK)
    if COMPLEX and IX == 1 and IY == 1:
        for start in range(tl.program_id(0) * BLOCK, 2 * N, tl.num_programs(0) * BLOCK):
            index = start + lane
            mask = index < 2 * N
            x = tl.load(X + index, mask, other=0.0)
            y = tl.load(Y + index, mask, other=0.0)
            if OP == 3:
                partner = tl.gather(x, lane ^ 1, 0)
                factor = tl.where((lane & 1) == 0, -AI, AI)
                tl.store(Y + index, y + AR * x + factor * partner, mask)
            else:
                tl.store(X + index, C * x + S * y, mask)
                tl.store(Y + index, C * y - S * x, mask)
    else:
        for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
            index = start + lane
            mask = index < N
            if COMPLEX:
                component = tl.arange(0, 2)
                xo = index[:, None] * (2 * IX) + component[None, :]
                yo = index[:, None] * (2 * IY) + component[None, :]
                x = tl.load(X + xo, mask[:, None], other=0.0)
                y = tl.load(Y + yo, mask[:, None], other=0.0)
                if OP == 3:
                    xr, xi = tl.split(x)
                    yr, yi = tl.split(y)
                    tl.store(
                        Y + yo,
                        tl.join(yr + AR * xr - AI * xi, yi + AR * xi + AI * xr),
                        mask[:, None],
                    )
                else:
                    tl.store(X + xo, C * x + S * y, mask[:, None])
                    tl.store(Y + yo, C * y - S * x, mask[:, None])
            else:
                xo = index * IX
                yo = index * IY
                x = tl.load(X + xo, mask, other=0.0)
                y = tl.load(Y + yo, mask, other=0.0)
                tl.store(X + xo, C * x + S * y, mask)
                tl.store(Y + yo, C * y - S * x, mask)


def _axpy_l1_caxpy(n, alpha, x, incx, y, incy):
    if incx != 1 or incy != 1 or not _axpy_l1_safe(n, incx, incy, True):
        return _axpy_l1_caxpy_fallback(n, alpha, x, incx, y, incy)
    _axpy_l1_validate(n, x, incx, y, incy, torch.complex64)
    value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if value == 0:
        return
    return _axpy_l1_launch(
        n, x, incx, y, incy, 3, value, 0.0, 0.0, _axpy_l1_flat_update
    )


_axpy_span_saxpy_previous = _saxpy_before_span


_axpy_span_caxpy_previous = _caxpy_before_span


_axpy_span_configs = tuple(runtime.get_tuned_config("ascend_l1_axpy_span"))


def _axpy_span_prune(configs, args, **kwargs):
    upper = 128 if args["N"] < 65536 else 512
    lower = 0
    if args["N"] >= 1048576:
        lower, upper = 512, 2048
    return [c for c in configs if lower <= c.kwargs["BLOCK"] <= upper]


@libentry()
@libtuner(
    configs=_axpy_span_configs,
    key=["N", "IX", "IY", "F"],
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _axpy_span_prune},
)
@triton.jit
def _axpy_span_kernel(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert((F == 1 or F == 2) and BLOCK <= 2048)
    # The bounded local lanes make division by three/six exact below.
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    if IY == 2:
        quotient = ly >> (1 if F == 1 else 2)
    else:
        quotient = (ly * (21846 if F == 1 else 10923)) >> 16
    component = ly & (F - 1)
    gather = quotient * IX * F + component
    gather = tl.minimum(gather, BLOCK * triton.next_power_of_2(IX) * F - 1)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        px = start * IX * F + lx
        py = start * IY * F + ly
        xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
        ym = (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        yv = tl.load(Y + py, ym)
        if IX == IY:
            x = xv
        else:
            x = tl.gather(xv, gather, 0)
        out = yv + AR * x
        if F == 2:
            if IX == IY:
                peer = tl.gather(xv, ly ^ 1, 0)
            else:
                peer = tl.gather(xv, gather ^ 1, 0)
            out = out + tl.where((ly & 1) == 0, -AI, AI) * peer
        out = tl.where(ly - quotient * (IY * F) < F, out, yv)
        tl.store(Y + py, out, ym)


@libentry()
@libtuner(
    configs=_axpy_span_configs,
    key=["N", "IX", "IY", "F", "L2_OFFSET"],
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _axpy_span_prune},
)
@triton.jit
def _axpy_span_stream_kernel(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
):
    if L2_OFFSET:
        X = (X.to(tl.uint64) + L2_OFFSET).to(tl.pointer_type(tl.float32))
    tl.static_assert((F == 1 or F == 2) and BLOCK <= 2048)
    # The bounded local lanes make division by three/six exact below.
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    if IY == 2:
        quotient = ly >> (1 if F == 1 else 2)
    else:
        quotient = (ly * (21846 if F == 1 else 10923)) >> 16
    component = ly & (F - 1)
    gather = quotient * IX * F + component
    gather = tl.minimum(gather, BLOCK * triton.next_power_of_2(IX) * F - 1)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        px = start * IX * F + lx
        py = start * IY * F + ly
        xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
        ym = (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        yv = tl.load(Y + py, ym)
        if IX == IY:
            x = xv
        else:
            x = tl.gather(xv, gather, 0)
        out = yv + AR * x
        if F == 2:
            if IX == IY:
                peer = tl.gather(xv, ly ^ 1, 0)
            else:
                peer = tl.gather(xv, gather ^ 1, 0)
            out = out + tl.where((ly & 1) == 0, -AI, AI) * peer
        out = tl.where(ly - quotient * (IY * F) < F, out, yv)
        tl.store(Y + py, out, ym)


def _axpy_span_call(name, n, alpha, x, incx, y, incy):
    previous = (
        _axpy_span_saxpy_previous if name == "saxpy" else _axpy_span_caxpy_previous
    )
    complex_data = name == "caxpy"
    eligible = (
        type(n) is int
        and n > 0
        and type(incx) is int
        and type(incy) is int
        and 1 < incx <= 3
        and 1 < incy <= 3
        and (not complex_data or n >= 1024)
        and 2 * (n + CORE_NUM * (1024 if complex_data else 512)) * max(incx, incy)
        < 2147483647
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and not x.is_conj()
        and not y.is_conj()
        and not x.is_neg()
        and not y.is_neg()
        and x.data_ptr() % 64 == y.data_ptr() % 64 == 0
    )
    if not eligible:
        return previous(n, alpha, x, incx, y, incy)
    dtype = torch.complex64 if complex_data else torch.float32
    _axpy_l1_validate(n, x, incx, y, incy, dtype)
    (px, py) = (x.data_ptr(), y.data_ptr())
    size = x.element_size()
    if px < py + (1 + (n - 1) * incy) * size and py < px + (1 + (n - 1) * incx) * size:
        return previous(n, alpha, x, incx, y, incy)
    value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if not complex_data:
        value = complex(
            float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
        )
    if value == 0:
        return
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _axpy_span_call(name, n, alpha, x, incx, y, incy)
    factor = 2 if complex_data else 1
    key = (
        "caxpy_span" if complex_data else "axpy_span",
        device.index,
        n,
        incx,
        incy,
        factor,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        raw_x = torch.view_as_real(x) if complex_data else x
        raw_y = torch.view_as_real(y) if complex_data else y
        kernel = _caxpy_span_kernel if complex_data else _axpy_span_kernel
        options = {"multibuffer": False} if complex_data else {}
        offset = _axpy_l2_offset(device.index) if n >= 1048576 else 0
        if offset:
            kernel = (
                _caxpy_span_stream_kernel if complex_data else _axpy_span_stream_kernel
            )
            options["L2_OFFSET"] = offset
            if complex_data and incx == 2 and incy == 3:
                kernel = _caxpy_span_stream_unrolled
            if complex_data and incx > incy:
                options["enable_ubuf_saving"] = True
        compiled, meta = kernel[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), CORE_NUM),)
        ](raw_x, raw_y, n, value.real, value.imag, incx, incy, factor, **options)
        block = meta["BLOCK"]
        grid = (min(triton.cdiv(n, block), CORE_NUM), 1, 1)
        constants = (block, offset) if offset else (block,)
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    (compiled, run, constants, grid) = entry
    args = (n, value.real, value.imag, incx, incy, factor, *constants)
    _launch_l1_entry(compiled, run, grid, (x, y), (px, py), args, device.index)


def saxpy(n, alpha, x, incx, y, incy):
    if incx == incy == 1:
        return _axpy_span_saxpy_previous(n, alpha, x, incx, y, incy)
    return _axpy_span_call("saxpy", n, alpha, x, incx, y, incy)


def caxpy(n, alpha, x, incx, y, incy):
    if incx == incy == 1:
        return _axpy_l1_pair_call(
            "caxpy", n, x, incx, y, incy, alpha=alpha, fallback=_caxpy_pair_fallback
        )
    return _axpy_span_call("caxpy", n, alpha, x, incx, y, incy)


def _caxpy_span_prune(configs, args, **kwargs):
    upper = 128 if args["N"] < 65536 else 512
    lower = 0
    if args["N"] >= 1048576:
        lower, upper = (256, 512) if args["IY"] == 3 else (512, 1024)
    if args.get("L2_OFFSET", 0) and args["N"] >= 1048576 and args["IX"] > args["IY"]:
        lower, upper = 256, 512
    return [c for c in configs if lower <= c.kwargs["BLOCK"] <= upper]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_caxpy_span"),
    key=["N", "IX", "IY", "F"],
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _caxpy_span_prune},
)
@triton.jit
def _caxpy_span_kernel(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert(F == 2 and BLOCK <= 1024)
    # Local lanes stay below 8192, where division by six is exact.
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    if IY == 2:
        quotient = ly >> 2
    else:
        quotient = (ly * 10923) >> 16
    component = ly & (F - 1)
    gather = quotient * IX * F + component
    gather = tl.minimum(gather, BLOCK * triton.next_power_of_2(IX) * F - 1)
    # Preload the next disjoint span while computing the current span.
    PIPE: tl.constexpr = IY == 3 and N >= 65536
    begin = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    if PIPE:
        px = begin * IX * F + lx
        py = begin * IY * F + ly
        xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
        yv = tl.load(Y + py, (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F))
    for start in range(begin, N, step):
        py = start * IY * F + ly
        ym = (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        if PIPE:
            npx = (start + step) * IX * F + lx
            npy = (start + step) * IY * F + ly
            nx = tl.load(
                X + npx, (lx < BLOCK * IX * F) & (npx < ((N - 1) * IX + 1) * F)
            )
            ny = tl.load(
                Y + npy, (ly < BLOCK * IY * F) & (npy < ((N - 1) * IY + 1) * F)
            )
        else:
            px = start * IX * F + lx
            xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
            yv = tl.load(Y + py, ym)
        if IX <= IY:
            peer = tl.gather(xv, lx ^ 1, 0)
            ax = AR * xv + tl.where((lx & 1) == 0, -AI, AI) * peer
            if IX != IY:
                ax = tl.gather(ax, gather, 0)
            out = yv + ax
        else:
            x = tl.gather(xv, gather, 0)
            peer = tl.gather(xv, gather ^ 1, 0)
            out = yv + AR * x + tl.where((ly & 1) == 0, -AI, AI) * peer
        out = tl.where(ly - quotient * (IY * F) < F, out, yv)
        tl.store(Y + py, out, ym)
        if PIPE:
            xv = nx
            yv = ny


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_caxpy_span"),
    key=["N", "IX", "IY", "F", "L2_OFFSET"],
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _caxpy_span_prune},
)
@triton.jit
def _caxpy_span_stream_kernel(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
):
    if L2_OFFSET:
        X = (X.to(tl.uint64) + L2_OFFSET).to(tl.pointer_type(tl.float32))
    tl.static_assert(F == 2 and BLOCK <= 1024)
    # Local lanes stay below 8192, where division by six is exact.
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    if IY == 2:
        quotient = ly >> 2
    else:
        quotient = (ly * 10923) >> 16
    component = ly & (F - 1)
    gather = quotient * IX * F + component
    gather = tl.minimum(gather, BLOCK * triton.next_power_of_2(IX) * F - 1)
    # Preload the next disjoint span while computing the current span.
    PIPE: tl.constexpr = (IY == 3 and N >= 65536) or (
        L2_OFFSET != 0 and N >= 1048576 and IX > IY
    )
    begin = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    if PIPE:
        px = begin * IX * F + lx
        py = begin * IY * F + ly
        xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
        yv = tl.load(Y + py, (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F))
    for start in range(begin, N, step):
        py = start * IY * F + ly
        ym = (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        if PIPE:
            npx = (start + step) * IX * F + lx
            npy = (start + step) * IY * F + ly
            nx = tl.load(
                X + npx, (lx < BLOCK * IX * F) & (npx < ((N - 1) * IX + 1) * F)
            )
            ny = tl.load(
                Y + npy, (ly < BLOCK * IY * F) & (npy < ((N - 1) * IY + 1) * F)
            )
        else:
            px = start * IX * F + lx
            xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
            yv = tl.load(Y + py, ym)
        if IX <= IY:
            peer = tl.gather(xv, lx ^ 1, 0)
            ax = AR * xv + tl.where((lx & 1) == 0, -AI, AI) * peer
            if IX != IY:
                ax = tl.gather(ax, gather, 0)
            out = yv + ax
        else:
            x = tl.gather(xv, gather, 0)
            peer = tl.gather(xv, gather ^ 1, 0)
            out = yv + AR * x + tl.where((ly & 1) == 0, -AI, AI) * peer
        out = tl.where(ly - quotient * (IY * F) < F, out, yv)
        tl.store(Y + py, out, ym)
        if PIPE:
            xv = nx
            yv = ny


def _caxpy_pair_fallback(name, n, x, ix, y, iy, alpha, c, s):
    return _axpy_l1_caxpy(n, alpha, x, ix, y, iy)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_caxpy_span"),
    key=["N", "IX", "IY", "F", "L2_OFFSET"],
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _caxpy_span_prune},
)
@triton.jit
def _caxpy_span_stream_unrolled(
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    IX: tl.constexpr,
    IY: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
):
    if L2_OFFSET:
        X = (X.to(tl.uint64) + L2_OFFSET).to(tl.pointer_type(tl.float32))
    tl.static_assert(F == 2 and BLOCK <= 1024)
    # Local lanes stay below 8192, where division by six is exact.
    lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX) * F)
    ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY) * F)
    if IY == 2:
        quotient = ly >> 2
    else:
        quotient = (ly * 10923) >> 16
    component = ly & (F - 1)
    gather = quotient * IX * F + component
    gather = tl.minimum(gather, BLOCK * triton.next_power_of_2(IX) * F - 1)
    # Preload the next disjoint span while computing the current span.
    PIPE: tl.constexpr = (IY == 3 and N >= 65536) or (
        L2_OFFSET != 0 and N >= 1048576 and IX > IY
    )
    begin = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    if PIPE:
        px = begin * IX * F + lx
        py = begin * IY * F + ly
        xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
        yv = tl.load(Y + py, (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F))
    for start in tl.range(begin, N, step, loop_unroll_factor=2):
        py = start * IY * F + ly
        ym = (ly < BLOCK * IY * F) & (py < ((N - 1) * IY + 1) * F)
        if PIPE:
            npx = (start + step) * IX * F + lx
            npy = (start + step) * IY * F + ly
            nx = tl.load(
                X + npx, (lx < BLOCK * IX * F) & (npx < ((N - 1) * IX + 1) * F)
            )
            ny = tl.load(
                Y + npy, (ly < BLOCK * IY * F) & (npy < ((N - 1) * IY + 1) * F)
            )
        else:
            px = start * IX * F + lx
            xv = tl.load(X + px, (lx < BLOCK * IX * F) & (px < ((N - 1) * IX + 1) * F))
            yv = tl.load(Y + py, ym)
        if IX <= IY:
            peer = tl.gather(xv, lx ^ 1, 0)
            ax = AR * xv + tl.where((lx & 1) == 0, -AI, AI) * peer
            if IX != IY:
                ax = tl.gather(ax, gather, 0)
            out = yv + ax
        else:
            x = tl.gather(xv, gather, 0)
            peer = tl.gather(xv, gather ^ 1, 0)
            out = yv + AR * x + tl.where((ly & 1) == 0, -AI, AI) * peer
        out = tl.where(ly - quotient * (IY * F) < F, out, yv)
        tl.store(Y + py, out, ym)
        if PIPE:
            xv = nx
            yv = ny

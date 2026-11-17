# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level1.rotm import _validate_rotm_inputs
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
            if len(tensors) == 3:
                (t0, t1, t2) = tensors
                if (
                    type(t0) is torch.Tensor
                    and (not t0.is_conj())
                    and (not t0.is_neg())
                    and (type(t1) is torch.Tensor)
                    and (not t1.is_conj())
                    and (not t1.is_neg())
                    and (type(t2) is torch.Tensor)
                    and (not t2.is_conj())
                    and (not t2.is_neg())
                ):
                    pointer_type = _launch_utils._DevicePointer
                    w0 = pointer_type(pointers[0])
                    w0.tensor = t0
                    w1 = pointer_type(pointers[1])
                    w1.tensor = t1
                    w2 = pointer_type(pointers[2])
                    w2.tensor = t2
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
                        w2,
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_vector"),
    key=["N", "IX", "IY"],
    restore_value=["X", "Y"],
)
@triton.jit
def _rotm_update(X, Y, P, N, IX: tl.constexpr, IY: tl.constexpr, BLOCK: tl.constexpr):
    flag = tl.load(P)
    if flag == -2.0:
        return
    h11, h21 = tl.load(P + 1), tl.load(P + 2)
    h12, h22 = tl.load(P + 3), tl.load(P + 4)
    lane = tl.arange(0, BLOCK)
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        index = start + lane
        mask = index < N
        xo, yo = index.to(tl.int64) * IX, index.to(tl.int64) * IY
        x, y = tl.load(X + xo, mask, other=0.0), tl.load(Y + yo, mask, other=0.0)
        if flag < 0.0:
            nx, ny = h11 * x + h12 * y, h21 * x + h22 * y
        elif flag == 0.0:
            nx, ny = x + h12 * y, h21 * x + y
        else:
            nx, ny = h11 * x + y, -x + h22 * y
        tl.store(X + xo, nx, mask)
        tl.store(Y + yo, ny, mask)


def srotm(n, x, incx, y, incy, param, param_host=None):
    return _rotation_l1_srotm(n, x, incx, y, incy, param, param_host=param_host)


def _rotation_l1_srotm_fallback(n, x, incx, y, incy, param, param_host=None):
    common = import_module("flag_blas.ops.level1.rotm")
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert param.dtype == torch.float32, "param must be float32"
    common._validate_rotm_inputs(n, x, incx, y, incy, param)
    if n <= 0:
        return
    with torch_device_fn.device(x.device):
        _rotm_update[lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), CORE_NUM),)](
            x,
            y,
            param,
            n,
            incx,
            incy,
        )


@libentry()
@libtuner(
    configs=_rotation_l1_configs,
    key=["N", "IX", "IY"],
    prune_configs_by={"early_config_prune": _rotation_l1_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _rotation_l1_rotm_direct(
    X, Y, P, N, IX: tl.constexpr, IY: tl.constexpr, BLOCK: tl.constexpr
):
    flag = tl.load(P)
    if flag != -2.0:
        h11, h21 = tl.load(P + 1), tl.load(P + 2)
        h12, h22 = tl.load(P + 3), tl.load(P + 4)
        lane = tl.arange(0, BLOCK)
        for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
            index = start + lane
            mask = index < N
            xo, yo = index * IX, index * IY
            x, y = tl.load(X + xo, mask, other=0.0), tl.load(Y + yo, mask, other=0.0)
            if flag < 0.0:
                nx, ny = h11 * x + h12 * y, h21 * x + h22 * y
            elif flag == 0.0:
                nx, ny = x + h12 * y, h21 * x + y
            else:
                nx, ny = h11 * x + y, -x + h22 * y
            tl.store(X + xo, nx, mask)
            tl.store(Y + yo, ny, mask)


def _rotation_l1_srotm(n, x, incx, y, incy, param, param_host=None):
    if type(incx) is int and type(incy) is int and (incx != 1 or incy != 1):
        return _rotation_l1_srotm_strided(
            n, x, incx, y, incy, param, param_host=param_host
        )
    if not (
        type(n) is int
        and type(incx) is int
        and (type(incy) is int)
        and (incx == incy == 1)
        and (0 < n < 2147483647 - 32768 - (_rotation_l1_core_num + 1) * 8192)
        and (type(x) is torch.Tensor)
        and (type(y) is torch.Tensor)
        and (type(param) is torch.Tensor)
    ):
        return _rotation_l1_srotm_legacy(
            n, x, incx, y, incy, param, param_host=param_host
        )
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert param.dtype == torch.float32, "param must be float32"
    assert x.is_contiguous() and y.is_contiguous()
    assert x.ndim == 1, "x must be 1-dimensional"
    assert y.ndim == 1, "y must be 1-dimensional"
    assert param.ndim == 1, "param must be 1-dimensional"
    assert param.numel() == 5, "param must have exactly five elements"
    device = x.device
    assert (
        device == y.device == param.device
    ), "x, y, and param must be on the same device"
    assert x.numel() >= n, "x is too short"
    assert y.numel() >= n, "y is too short"
    if (
        not param.is_contiguous()
        or x.is_conj()
        or y.is_conj()
        or param.is_conj()
        or x.is_neg()
        or y.is_neg()
        or param.is_neg()
    ):
        return _rotation_l1_srotm_legacy(
            n, x, incx, y, incy, param, param_host=param_host
        )
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _rotation_l1_srotm(n, x, incx, y, incy, param, param_host=param_host)
    (px, py, pp) = (x.data_ptr(), y.data_ptr(), param.data_ptr())
    if (px | py | pp) & 63:
        return _rotation_l1_srotm_legacy(
            n, x, incx, y, incy, param, param_host=param_host
        )
    (ex, ey, ep) = (px + n * 4, py + n * 4, pp + 20)
    if px < ey and py < ex or (px < ep and pp < ex) or (py < ep and pp < ey):
        return _rotation_l1_srotm_legacy(
            n, x, incx, y, incy, param, param_host=param_host
        )
    key = ("unit_batched", device.index, n)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        (compiled, meta) = _rotm_contiguous_batched[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), _rotation_l1_core_num),)
        ](x, y, param, n, incx, incy)
        block = meta["BLOCK"]
        grid = (min(triton.cdiv(n, block), _rotation_l1_core_num), 1, 1)
        _store_launch(key, (compiled, compiled.run, (incx, incy, block), grid))
        return
    (compiled, run, constants, grid) = entry
    _launch_l1_entry(
        compiled, run, grid, (x, y, param), (px, py, pp), (n, *constants), device.index
    )


def _rotation_l1_srotm_legacy(n, x, incx, y, incy, param, param_host=None):
    if not _rotation_l1_safe(n, incx, incy, False):
        return _rotation_l1_srotm_strided(
            n, x, incx, y, incy, param, param_host=param_host
        )
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert param.dtype == torch.float32, "param must be float32"
    assert x.is_contiguous() and y.is_contiguous()
    common = import_module("flag_blas.ops.level1.rotm")
    common._validate_rotm_inputs(n, x, incx, y, incy, param)
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _rotation_l1_srotm_legacy(
                n, x, incx, y, incy, param, param_host=param_host
            )
    tensors = x, y, param
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    use_static = (
        type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and type(param) is torch.Tensor
        and param.is_contiguous()
        and not x.is_conj()
        and not y.is_conj()
        and not param.is_conj()
        and not x.is_neg()
        and not y.is_neg()
        and not param.is_neg()
    )
    px, py, pp = pointers
    if (
        (px < py + n * 4 and py < px + n * 4)
        or (pp < px + n * 4 and px < pp + 20)
        or (pp < py + n * 4 and py < pp + 20)
    ):
        use_static = False
    key = (
        "rotation",
        device.index,
        "rotm",
        n,
        incx,
        incy,
        use_static,
        tuple(pointer % 16 == 0 for pointer in pointers),
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        if use_static:
            compiled, meta = _rotation_l1_rotm_static[
                lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), _rotation_l1_core_num),)
            ](x, y, param, n, 0)
            arguments = n, 0, meta["BLOCK"]
        else:
            compiled, meta = _rotation_l1_rotm_contiguous[
                lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), _rotation_l1_core_num),)
            ](x, y, param, n, incx, incy)
            arguments = n, incx, incy, meta["BLOCK"]
        grid = min(triton.cdiv(n, meta["BLOCK"]), _rotation_l1_core_num), 1, 1
        _store_launch(key, (compiled, compiled.run, arguments, grid))
        return
    compiled, run, arguments, grid = entry
    _launch_l1_entry(compiled, run, grid, tensors, pointers, arguments, device.index)


_rotation_l1_rotm_configs = tuple(runtime.get_tuned_config("ascend_l1_rotm"))


def _rotation_l1_rotm_prune(configs, args, **kwargs):
    n = args["N"]
    if n <= 4096:
        upper = min(4096, max(256, triton.next_power_of_2(n)))
        return [cfg for cfg in configs if cfg.kwargs["BLOCK"] <= upper]
    return [cfg for cfg in configs if cfg.kwargs["BLOCK"] == 8192]


@libentry()
@libtuner(
    configs=_rotation_l1_rotm_configs,
    key=["N", "IX", "IY"],
    prune_configs_by={"early_config_prune": _rotation_l1_rotm_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _rotation_l1_rotm_contiguous(
    X, Y, P, N, IX: tl.constexpr, IY: tl.constexpr, BLOCK: tl.constexpr
):
    flag = tl.load(P)
    h11, h21 = tl.load(P + 1), tl.load(P + 2)
    h12, h22 = tl.load(P + 3), tl.load(P + 4)
    h11 = tl.where(flag == 0.0, 1.0, h11)
    h22 = tl.where(flag == 0.0, 1.0, h22)
    h12 = tl.where(flag < 0.0, h12, tl.where(flag == 0.0, h12, 1.0))
    h21 = tl.where(flag < 0.0, h21, tl.where(flag == 0.0, h21, -1.0))
    lane = tl.arange(0, BLOCK)
    end = tl.where(flag == -2.0, 0, N)
    for start in range(tl.program_id(0) * BLOCK, end, tl.num_programs(0) * BLOCK):
        index = start + lane
        mask = index < N
        xo, yo = index * IX, index * IY
        x = tl.load(X + xo, mask, other=0.0)
        y = tl.load(Y + yo, mask, other=0.0)
        nx, ny = h11 * x + h12 * y, h21 * x + h22 * y
        tl.store(X + xo, nx, mask)
        tl.store(Y + yo, ny, mask)


@libentry()
@libtuner(
    configs=_rotation_l1_rotm_configs,
    key=["N", "MODE"],
    prune_configs_by={"early_config_prune": _rotation_l1_rotm_prune},
    restore_value=["X", "Y"],
)
@triton.jit
def _rotation_l1_rotm_static(
    X, Y, P, N: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    flag = tl.load(P)
    if flag != -2.0:
        h11, h21 = tl.load(P + 1), tl.load(P + 2)
        h12, h22 = tl.load(P + 3), tl.load(P + 4)
        h11 = tl.where(flag == 0.0, 1.0, h11)
        h22 = tl.where(flag == 0.0, 1.0, h22)
        h12 = tl.where(flag < 0.0, h12, tl.where(flag == 0.0, h12, 1.0))
        h21 = tl.where(flag < 0.0, h21, tl.where(flag == 0.0, h21, -1.0))
        lane = tl.arange(0, BLOCK)
        chunk = tl.cdiv(N, tl.num_programs(0) * BLOCK) * BLOCK
        first = tl.program_id(0) * chunk
        end = tl.minimum(first + chunk, N)
        for start in range(first, end, BLOCK):
            i = start + lane
            x = tl.load(X + i, i < N, other=0.0)
            y = tl.load(Y + i, i < N, other=0.0)
            nx, ny = h11 * x + h12 * y, h21 * x + h22 * y
            tl.store(X + i, nx, i < N)
            tl.store(Y + i, ny, i < N)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_rotation_strided"),
    key=["N", "IX", "IY"],
    restore_value=["X", "Y"],
)
@triton.jit
def _rotm_strided_span(
    X, Y, P, N: tl.constexpr, IX: tl.constexpr, IY: tl.constexpr, BLOCK: tl.constexpr
):
    if N <= 131072:
        flag = tl.load(P)
    else:
        coefficient_lane = tl.arange(0, 8)
        coefficient_bits = tl.load(
            P + coefficient_lane, coefficient_lane < 5, other=0.0
        ).to(tl.int32, bitcast=True)
        flag = tl.sum(tl.where(coefficient_lane == 0, coefficient_bits, 0), 0).to(
            tl.float32, bitcast=True
        )
    if flag != -2.0:
        if N <= 131072:
            h11 = tl.load(P + 1)
            h21 = tl.load(P + 2)
            h12 = tl.load(P + 3)
            h22 = tl.load(P + 4)
        else:
            h11 = tl.sum(tl.where(coefficient_lane == 1, coefficient_bits, 0), 0).to(
                tl.float32, bitcast=True
            )
            h21 = tl.sum(tl.where(coefficient_lane == 2, coefficient_bits, 0), 0).to(
                tl.float32, bitcast=True
            )
            h12 = tl.sum(tl.where(coefficient_lane == 3, coefficient_bits, 0), 0).to(
                tl.float32, bitcast=True
            )
            h22 = tl.sum(tl.where(coefficient_lane == 4, coefficient_bits, 0), 0).to(
                tl.float32, bitcast=True
            )
        h11 = tl.where(flag == 0.0, 1.0, h11)
        h22 = tl.where(flag == 0.0, 1.0, h22)
        h12 = tl.where(flag < 0.0, h12, tl.where(flag == 0.0, h12, 1.0))
        h21 = tl.where(flag < 0.0, h21, tl.where(flag == 0.0, h21, -1.0))
        lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX))
        ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY))
        if IX == 1:
            qx = lx
        elif IX == 2:
            qx = lx >> 1
        else:
            qx = lx * 21846 >> 16
        if IY == 1:
            qy = ly
        elif IY == 2:
            qy = ly >> 1
        else:
            qy = ly * 21846 >> 16
        map_y = tl.minimum(qx * IY, BLOCK * triton.next_power_of_2(IY) - 1)
        map_x = tl.minimum(qy * IX, BLOCK * triton.next_power_of_2(IX) - 1)
        for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
            (px, py) = (start * IX + lx, start * IY + ly)
            mx = (lx < BLOCK * IX) & (px < (N - 1) * IX + 1)
            my = (ly < BLOCK * IY) & (py < (N - 1) * IY + 1)
            (xv, yv) = (tl.load(X + px, mx, other=0.0), tl.load(Y + py, my, other=0.0))
            if IX == IY:
                (xy, yx) = (yv, xv)
            else:
                (xy, yx) = (tl.gather(yv, map_y, 0), tl.gather(xv, map_x, 0))
            (ox, oy) = (h11 * xv + h12 * xy, h21 * yx + h22 * yv)
            bx = tl.where(
                lx - qx * IX == 0,
                ox.to(tl.int32, bitcast=True),
                xv.to(tl.int32, bitcast=True),
            )
            by = tl.where(
                ly - qy * IY == 0,
                oy.to(tl.int32, bitcast=True),
                yv.to(tl.int32, bitcast=True),
            )
            tl.store(X + px, bx.to(tl.float32, bitcast=True), mx)
            tl.store(Y + py, by.to(tl.float32, bitcast=True), my)


def _rotm_strided_tuned_prune(configs, args, **kwargs):
    span = triton.next_power_of_2(args["IX"]) + triton.next_power_of_2(args["IY"])
    allowed = [
        c
        for c in configs
        if c.kwargs["BLOCK"] * span <= 6144
        and (args["N"] < 1048576 or c.kwargs["BLOCK"] >= 512)
    ]
    if args["N"] <= 2097152 and (args["IX"] == 2 or args["IY"] == 2):
        return [
            c
            for c in allowed
            if c.kwargs["BLOCK"] in (512, 1024)
            and c.kwargs["PIPE"]
            and not c.kwargs["LOGICAL"]
        ]
    return allowed


_ROTM_STRIDED_CORES = tl.constexpr(_rotation_l1_core_num)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_rotm_strided_large"),
    key=["N", "IX", "IY"],
    restore_value=["X", "Y"],
    prune_configs_by={"early_config_prune": _rotm_strided_tuned_prune},
)
@triton.jit
def _rotm_strided_tuned(
    X,
    Y,
    P,
    N: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    BLOCK: tl.constexpr,
    PIPE: tl.constexpr,
    LOGICAL: tl.constexpr,
):
    coefficient_lane = tl.arange(0, 8)
    coefficient_bits = tl.load(
        P + coefficient_lane, coefficient_lane < 5, other=0.0
    ).to(tl.int32, bitcast=True)
    flag = tl.sum(tl.where(coefficient_lane == 0, coefficient_bits, 0), 0).to(
        tl.float32, bitcast=True
    )
    if flag != -2.0:
        h11 = tl.sum(tl.where(coefficient_lane == 1, coefficient_bits, 0), 0).to(
            tl.float32, bitcast=True
        )
        h21 = tl.sum(tl.where(coefficient_lane == 2, coefficient_bits, 0), 0).to(
            tl.float32, bitcast=True
        )
        h12 = tl.sum(tl.where(coefficient_lane == 3, coefficient_bits, 0), 0).to(
            tl.float32, bitcast=True
        )
        h22 = tl.sum(tl.where(coefficient_lane == 4, coefficient_bits, 0), 0).to(
            tl.float32, bitcast=True
        )
        h11 = tl.where(flag == 0.0, 1.0, h11)
        h22 = tl.where(flag == 0.0, 1.0, h22)
        h12 = tl.where(flag < 0.0, h12, tl.where(flag == 0.0, h12, 1.0))
        h21 = tl.where(flag < 0.0, h21, tl.where(flag == 0.0, h21, -1.0))
        lx = tl.arange(0, BLOCK * triton.next_power_of_2(IX))
        ly = tl.arange(0, BLOCK * triton.next_power_of_2(IY))
        if IX == 1:
            qx = lx
        elif IX == 2:
            qx = lx >> 1
        else:
            qx = lx * 21846 >> 16
        if IY == 1:
            qy = ly
        elif IY == 2:
            qy = ly >> 1
        else:
            qy = ly * 21846 >> 16
        map_y = tl.minimum(qx * IY, BLOCK * triton.next_power_of_2(IY) - 1)
        map_x = tl.minimum(qy * IX, BLOCK * triton.next_power_of_2(IX) - 1)
        lane = tl.arange(0, BLOCK)
        CHUNK: tl.constexpr = (
            (BLOCK * 4 // 3 // 16) * 16 if IX == 3 and IY == 3 else BLOCK
        )
        if N <= 2097152 and (IX == 2 or IY == 2):
            chunk = tl.cdiv(N, _ROTM_STRIDED_CORES * BLOCK) * BLOCK
            begin = tl.program_id(0) * chunk
            end = tl.minimum(begin + chunk, N)
            step = BLOCK
        else:
            begin = tl.program_id(0) * CHUNK
            end = N
            step = _ROTM_STRIDED_CORES * CHUNK
        if PIPE:
            (px, py) = (begin * IX + lx, begin * IY + ly)
            xv = tl.load(X + px, (lx < CHUNK * IX) & (px < (N - 1) * IX + 1), other=0.0)
            yv = tl.load(Y + py, (ly < CHUNK * IY) & (py < (N - 1) * IY + 1), other=0.0)
        for start in range(begin, end, step):
            (px, py) = (start * IX + lx, start * IY + ly)
            mx = (lx < CHUNK * IX) & (px < (N - 1) * IX + 1)
            my = (ly < CHUNK * IY) & (py < (N - 1) * IY + 1)
            if PIPE:
                (nx, ny) = ((start + step) * IX + lx, (start + step) * IY + ly)
                next_x = tl.load(
                    X + nx,
                    (lx < CHUNK * IX) & (nx < (N - 1) * IX + 1) & (start + step < end),
                    other=0.0,
                )
                next_y = tl.load(
                    Y + ny,
                    (ly < CHUNK * IY) & (ny < (N - 1) * IY + 1) & (start + step < end),
                    other=0.0,
                )
            else:
                (xv, yv) = (
                    tl.load(X + px, mx, other=0.0),
                    tl.load(Y + py, my, other=0.0),
                )
            if LOGICAL and not (IX == 3 and IY == 3):
                x = tl.gather(xv, lane * IX, 0)
                y = tl.gather(yv, lane * IY, 0)
                (ox, oy) = (h11 * x + h12 * y, h21 * x + h22 * y)
                ox = tl.gather(ox, tl.minimum(qx, BLOCK - 1), 0)
                oy = tl.gather(oy, tl.minimum(qy, BLOCK - 1), 0)
            else:
                if IX == IY:
                    (xy, yx) = (yv, xv)
                else:
                    (xy, yx) = (tl.gather(yv, map_y, 0), tl.gather(xv, map_x, 0))
                (ox, oy) = (h11 * xv + h12 * xy, h21 * yx + h22 * yv)
            bx = tl.where(
                lx - qx * IX == 0,
                ox.to(tl.int32, bitcast=True),
                xv.to(tl.int32, bitcast=True),
            )
            by = tl.where(
                ly - qy * IY == 0,
                oy.to(tl.int32, bitcast=True),
                yv.to(tl.int32, bitcast=True),
            )
            tl.store(X + px, bx.to(tl.float32, bitcast=True), mx)
            tl.store(Y + py, by.to(tl.float32, bitcast=True), my)
            if PIPE:
                (xv, yv) = (next_x, next_y)


def _rotation_l1_srotm_strided(n, x, incx, y, incy, param, param_host=None):
    if not (
        type(n) is int
        and type(incx) is int
        and type(incy) is int
        and n > 0
        and 1 <= incx <= 3
        and 1 <= incy <= 3
        and (incx != 1 or incy != 1)
        and (n + (_rotation_l1_core_num + 1) * 1024) * max(incx, incy)
        < 2147483647 - 32768
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and type(param) is torch.Tensor
    ):
        return _rotation_l1_srotm_fallback(
            n, x, incx, y, incy, param, param_host=param_host
        )
    assert x.dtype == torch.float32, "x must be float32"
    assert x.is_contiguous() and y.is_contiguous()
    _validate_rotm_inputs(n, x, incx, y, incy, param)
    if (
        not param.is_contiguous()
        or x.is_conj()
        or y.is_conj()
        or param.is_conj()
        or x.is_neg()
        or y.is_neg()
        or param.is_neg()
    ):
        return _rotation_l1_srotm_fallback(
            n, x, incx, y, incy, param, param_host=param_host
        )
    device = x.device
    if device.index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _rotation_l1_srotm_strided(
                n, x, incx, y, incy, param, param_host=param_host
            )
    px, py, pp = x.data_ptr(), y.data_ptr(), param.data_ptr()
    if (px | py | pp) & 63:
        return _rotation_l1_srotm_fallback(
            n, x, incx, y, incy, param, param_host=param_host
        )
    ex, ey, ep = px + (1 + (n - 1) * incx) * 4, py + (1 + (n - 1) * incy) * 4, pp + 20
    if (px < ey and py < ex) or (px < ep and pp < ex) or (py < ep and pp < ey):
        return _rotation_l1_srotm_fallback(
            n, x, incx, y, incy, param, param_host=param_host
        )
    key = (device.index, n, incx, incy)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        kernel = _rotm_strided_tuned if n >= 1048576 else _rotm_strided_span
        compiled, meta = kernel[
            lambda cfg: (min(triton.cdiv(n, cfg["BLOCK"]), _rotation_l1_core_num),)
        ](x, y, param, n, incx, incy)
        block = meta["BLOCK"]
        grid = min(triton.cdiv(n, block), _rotation_l1_core_num), 1, 1
        constants = (incx, incy, block)
        if n >= 1048576:
            constants += (meta["PIPE"], meta["LOGICAL"])
        _store_launch(key, (compiled, compiled.run, constants, grid))
        return
    compiled, run, constants, grid = entry
    _launch_l1_entry(
        compiled, run, grid, (x, y, param), (px, py, pp), (n, *constants), device.index
    )


def _rotm_contiguous_prune(configs, args, **kwargs):
    n = args["N"]
    blocks = (
        (256, 512, 1024) if n <= 4096 else (512, 1024, 2048) if n < 1048576 else (8192,)
    )
    return [c for c in configs if c.kwargs["BLOCK"] in blocks]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ascend_l1_rotm_contiguous"),
    key=["N", "IX", "IY"],
    restore_value=["X", "Y"],
    prune_configs_by={"early_config_prune": _rotm_contiguous_prune},
)
@triton.jit
def _rotm_contiguous_batched(
    X, Y, P, N: tl.constexpr, IX: tl.constexpr, IY: tl.constexpr, BLOCK: tl.constexpr
):
    flag = tl.load(P)
    if flag != -2.0:
        h11 = tl.load(P + 1)
        h21 = tl.load(P + 2)
        h12 = tl.load(P + 3)
        h22 = tl.load(P + 4)
        h11 = tl.where(flag == 0.0, 1.0, h11)
        h22 = tl.where(flag == 0.0, 1.0, h22)
        h12 = tl.where(flag < 0.0, h12, tl.where(flag == 0.0, h12, 1.0))
        h21 = tl.where(flag < 0.0, h21, tl.where(flag == 0.0, h21, -1.0))
        lane = tl.arange(0, BLOCK)
        chunk = tl.cdiv(N, tl.num_programs(0) * BLOCK) * BLOCK
        first = tl.program_id(0) * chunk
        end = tl.minimum(first + chunk, N)
        for start in range(first, end, BLOCK):
            i = start + lane
            x = tl.load(X + i, i < N, other=0.0)
            y = tl.load(Y + i, i < N, other=0.0)
            (nx, ny) = (h11 * x + h12 * y, h21 * x + h22 * y)
            tl.store(X + i, nx, i < N)
            tl.store(Y + i, ny, i < N)

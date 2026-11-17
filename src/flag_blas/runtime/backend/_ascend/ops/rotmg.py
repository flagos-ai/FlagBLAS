# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

import torch
import triton

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn

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


_SCALAR_CONFIG = runtime.get_tuned_config("ascend_l1_rotation_scalar")[0]


def _store_scalar_launch(key, entry):
    if len(_LAUNCH_CACHE) >= 128:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = entry


def _scalar_launch(name, tensors, kernel):
    a, b, c, s, p = tensors
    device = a.device
    index = device.index
    if index != _launch_utils._current_device():
        with torch_device_fn.device(device):
            return _scalar_launch(name, tensors, kernel)
    pa, pb, pc, ps, pp = (
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        s.data_ptr(),
        p.data_ptr(),
    )
    pointers = pa, pb, pc, ps, pp
    key = (
        index,
        name,
        (pa % 16 == 0, pb % 16 == 0, pc % 16 == 0, ps % 16 == 0, pp % 16 == 0),
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        compiled, _ = kernel[(1,)](
            *tensors,
            num_warps=_SCALAR_CONFIG.num_warps,
            num_stages=_SCALAR_CONFIG.num_stages,
        )
        run = compiled.run
        static_fast = (
            type(run) is _launch_utils._NPU_LAUNCHER
            and not getattr(run, "compile_only", False)
            and not getattr(run, "enable_msprof_register_tensor", False)
            and not getattr(compiled.metadata, "debug_enabled", False)
            and not getattr(getattr(run, "metadata", None), "debug_enabled", False)
        )
        _store_scalar_launch(
            key,
            (
                compiled,
                run,
                static_fast,
                (compiled.function, compiled.packed_metadata, None, None, None),
            ),
        )
        return
    compiled, run, static_fast, tail = entry
    knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    if (
        static_fast
        and _launch_utils._FAST_LAUNCH_SUPPORTED
        and knobs is not None
        and not getattr(run, "compile_only", False)
        and not getattr(run, "enable_msprof_register_tensor", False)
        and not getattr(compiled.metadata, "debug_enabled", False)
        and not getattr(getattr(run, "metadata", None), "debug_enabled", False)
    ):
        h0, h1 = knobs.launch_enter_hook, knobs.launch_exit_hook
        hooks = (
            h0 is not None
            and (type(h0) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h0.calls))
        ) or (
            h1 is not None
            and (type(h1) is not _launch_utils._HOOK_CHAIN_TYPE or bool(h1.calls))
        )
        if not hooks and (
            type(a) is torch.Tensor
            and type(b) is torch.Tensor
            and type(c) is torch.Tensor
            and type(s) is torch.Tensor
            and type(p) is torch.Tensor
            and not a.is_conj()
            and not b.is_conj()
            and not c.is_conj()
            and not s.is_conj()
            and not p.is_conj()
            and not a.is_neg()
            and not b.is_neg()
            and not c.is_neg()
            and not s.is_neg()
            and not p.is_neg()
        ):
            wa, wb, wc, ws, wp = (
                _launch_utils._DevicePointer(pa),
                _launch_utils._DevicePointer(pb),
                _launch_utils._DevicePointer(pc),
                _launch_utils._DevicePointer(ps),
                _launch_utils._DevicePointer(pp),
            )
            wa.tensor, wb.tensor, wc.tensor, ws.tensor, wp.tensor = a, b, c, s, p
            registered = run.launch(
                1,
                1,
                1,
                _launch_utils._current_raw_stream(index),
                *tail,
                wa,
                wb,
                wc,
                ws,
                wp,
            )
            _spr_utils._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
            return
    return _launch_l1_entry(compiled, run, (1, 1, 1), tensors, pointers, (), index)


_rotation_l1_rotmg_common = import_module("flag_blas.ops.level1.rotmg")


def srotmg(d1, d2, x1, y1, param):
    assert d1.dtype == torch.float32, "d1 must be float32"
    assert d2.dtype == torch.float32, "d2 must be float32"
    assert x1.dtype == torch.float32, "x1 must be float32"
    assert y1.dtype == torch.float32, "y1 must be float32"
    assert param.dtype == torch.float32, "param must be float32"
    _rotation_l1_rotmg_common._validate_rotmg_inputs(d1, d2, x1, y1, param)
    return _scalar_launch(
        "srotmg", (d1, d2, x1, y1, param), _rotation_l1_rotmg_common._rotmg_kernel
    )

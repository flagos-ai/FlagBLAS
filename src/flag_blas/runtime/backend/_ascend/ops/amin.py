# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

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
def _amin_fallback_partial(
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
    best = float("inf")
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
        values = tl.where(mask, magnitude, float("inf"))
        value = tl.min(values, 0)
        better = value < best
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
def _amin_finish(
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
    values = tl.load(V + lane, mask, other=float("inf"))
    best = tl.min(values, 0)
    indices = tl.load(IDX + lane, mask, other=2147483647)
    index = tl.min(tl.where(mask & (values == best), indices, 2147483647), 0)
    tl.store(OUT, index + 1)


_AMIN_SMALL_CONFIGS = runtime.get_tuned_config("ascend_l1_amax")


_LAUNCH_CACHE = {}


def _prune_amin_configs(configs, args):
    if args["COMPLEX"]:
        limit = 4096 if args["INCX"] == 1 else 512
    else:
        limit = 2048 if args["INCX"] != 1 else 4096
    if args["N"] <= 512:
        limit = min(limit, max(256, triton.next_power_of_2(args["N"])))
    return [c for c in configs if c.kwargs["BLOCK"] <= limit]


@libentry()
@triton.jit
def _icamin_small_compute(
    X, OUT, N: tl.constexpr, INCX: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.arange(0, BLOCK)
    best = tl.full((BLOCK,), float("inf"), tl.float32)
    idx = tl.full((BLOCK,), 2147483647, tl.int32)
    if INCX == 1:
        pair = tl.arange(0, 2)
    else:
        physical = tl.arange(0, BLOCK * triton.next_power_of_2(INCX * 2))
        components = tl.arange(0, BLOCK * 2)
        gather = (components // 2) * (INCX * 2) + (components & 1)
    for tile in tl.static_range(0, triton.cdiv(N, BLOCK)):
        start = tile * BLOCK
        if INCX == 1:
            values = tl.load(
                X + (start + row[:, None]) * 2 + pair[None, :],
                (start + row[:, None]) < N,
                other=0.0,
            )
        else:
            addresses = start * INCX * 2 + physical
            span = tl.load(
                X + addresses, addresses < ((N - 1) * INCX + 1) * 2, other=0.0
            )
            values = tl.reshape(tl.gather(span, gather, 0), (BLOCK, 2))
        real, imag = tl.split(values)
        magnitude = tl.abs(real) + tl.abs(imag)
        take = (start + row < N) & (magnitude < best)
        best = tl.where(take, magnitude, best)
        idx = tl.where(take, start + row, idx)
    minimum = tl.min(best, 0)
    index = tl.min(tl.where(best == minimum, idx, 2147483647), 0)
    index = tl.where(minimum == float("inf"), 0, index)
    tl.store(OUT, index + 1)


@libentry()
@triton.jit
def _amin_small_compute(
    X,
    OUT,
    N: tl.constexpr,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    GATHER: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    best = tl.full((BLOCK,), float("inf"), tl.float32)
    best_index = tl.full((BLOCK,), 2147483647, tl.int32)
    for start in range(0, N, BLOCK):
        index = start + lane
        mask = index < N
        if GATHER:
            component = tl.arange(0, BLOCK * WIDTH)
            address_index = start
            size = N
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
            address_index = index
            value = tl.load(X + address_index * INCX, mask, other=0.0)
            magnitude = tl.abs(value)
        take = mask & ((magnitude < best) | (magnitude == best) & (index < best_index))
        best = tl.where(take, magnitude, best)
        best_index = tl.where(take, index, best_index)
    minimum = tl.min(best, 0)
    index = tl.min(tl.where(best == minimum, best_index, 2147483647), 0)
    index = tl.where(minimum == float("inf"), 0, index)
    tl.store(OUT, index + 1)


def _amin_small(n, x, incx, result, complex_data):
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
            return _amin_small(n, x, incx, result, complex_data)
    (x_pointer, result_pointer) = (x.data_ptr(), result.data_ptr())
    key = (
        device.index,
        n,
        incx,
        complex_data,
        x_pointer % 16 == 0,
        result_pointer % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        gather = complex_data or incx > 1
        configs = _prune_amin_configs(
            _AMIN_SMALL_CONFIGS,
            {"N": n, "INCX": incx, "COMPLEX": complex_data, "GATHER": gather},
        )
        eligible = [
            c for c in configs if c.kwargs["BLOCK"] <= triton.next_power_of_2(n)
        ]
        config = (
            max(eligible, key=lambda c: c.kwargs["BLOCK"])
            if eligible
            else min(configs, key=lambda c: c.kwargs["BLOCK"])
        )
        # Bound the physical read span for strided complex inputs.
        if complex_data and (incx == 1 or (n > 1024 and incx <= 5)):
            kernel = _icamin_small_compute
            constants = (n, incx, triton.next_power_of_2(n) if incx == 1 else 1024)
        else:
            kernel = _amin_small_compute
            constants = (
                n,
                incx,
                complex_data,
                gather,
                triton.next_power_of_2(incx * (2 if complex_data else 1)),
                config.kwargs["BLOCK"],
            )
        pointer = torch.view_as_real(x) if complex_data else x
        (compiled, _) = kernel[1,](
            pointer,
            result,
            *constants,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        if len(_LAUNCH_CACHE) >= 256:
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


def _amin_dispatch(n, x, incx, result, complex_data):
    if type(n) is int and type(incx) is int and 0 < n <= 8192 and 0 < incx <= 8:
        return _amin_small(n, x, incx, result, complex_data)
    return _amin_large(n, x, incx, result, complex_data)


def isamin(n, x, incx, result):
    return _amin_dispatch(n, x, incx, result, False)


def icamin(n, x, incx, result):
    return _amin_dispatch(n, x, incx, result, True)


def _amin_large(n, x, incx, result, complex_data):
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
        single_limit = 16384
        parts = 1 if n <= single_limit else min(triton.cdiv(n, 512), CORE_NUM)
        if parts == 1:
            (values, indices) = (result, result)
        else:
            values = torch.empty(parts, dtype=torch.float32, device=x.device)
            indices = torch.empty(parts, dtype=torch.int32, device=x.device)
        _amin_fallback_partial[parts,](
            pointer, values, indices, result, n, incx, complex_data, True, parts
        )
        if parts > 1:
            _amin_finish[1,](
                values,
                indices,
                result,
                parts,
                True,
                triton.next_power_of_2(parts),
                num_warps=1,
                num_stages=1,
            )

"""Ascend GEMV with dtype-specific dispatch and workspace-free Triton kernels."""

import ctypes
from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

try:
    from triton.backends.ascend import utils as driver_utils
    from triton.backends.ascend.driver import NPULauncher
except ImportError:
    NPULauncher = None
try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as current_stream
except ImportError:
    current_stream = triton.runtime.driver.active.get_current_stream
try:
    from torch_npu._C import _npu_getDevice as current_device
except ImportError:
    current_device = torch_device_fn.current_device

_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)


class _RealGemvPlan(NamedTuple):
    kernel: str = "block"
    pool: str = ""
    bm: int = 0
    bk: int = 0
    stages: int = 1
    warps: int = 1
    alias: bool = False
    fixed: bool = False
    head_tail: bool = False
    whole: bool = False
    dense: bool = False
    reduce_each: bool = False
    full_tiles: bool = False
    splits: int = 1
    filter_bm: int = 0
    limit: int = 16384


@lru_cache(maxsize=None)
def _gemv_l2_offset(device):
    if not torch.npu.get_device_name(device).startswith("Ascend910B"):
        return 0
    try:
        library = ctypes.CDLL("libruntime.so")
        query = library.rtGetL2CacheOffset
        query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int32
        result = ctypes.c_uint64(0)
        return result.value if query(device, ctypes.byref(result)) == 0 else 0
    except (AttributeError, OSError):
        return 0


class _RealGemvPointer(int):
    def size(self):
        return self.tensor.size()


class _RealGemvLaunch(NamedTuple):
    compiled: object
    run: object
    grid: tuple
    constants: tuple
    tensor_count: int
    special: bool

    def __call__(self, args, stream, pointers=None):
        enter = triton.knobs.runtime.launch_enter_hook
        leave = triton.knobs.runtime.launch_exit_hook
        hooked = (
            enter is not None
            and (type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls))
        ) or (
            leave is not None
            and (type(leave) is not _HOOK_CHAIN_TYPE or bool(leave.calls))
        )
        if self.special or hooked or type(self.run) is not NPULauncher:
            self.compiled[self.grid](*args, *self.constants, stream=stream)
            return
        tensors = args[: self.tensor_count]
        if pointers is None:
            pointers = tuple(t.data_ptr() for t in tensors)
        if self.tensor_count == 3:
            pa, px, py = (
                _RealGemvPointer(pointers[0]),
                _RealGemvPointer(pointers[1]),
                _RealGemvPointer(pointers[2]),
            )
            pa.tensor, px.tensor, py.tensor = tensors
            packed = (pa, px, py)
        else:
            py = _RealGemvPointer(pointers[0])
            py.tensor = tensors[0]
            packed = (py,)
        registered = self.run.launch(
            *self.grid,
            stream,
            self.compiled.function,
            self.compiled.packed_metadata,
            None,
            None,
            None,
            *packed,
            *args[self.tensor_count :],
            *self.constants,
        )
        driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


def _real_gemv_meta(raw, plan, trans, out, reduce, lda, offset, config):
    meta = dict(
        BM=16,
        BK=128,
        TRANSPOSE=bool(trans),
        L2_OFFSET=offset,
        OUT_FIXED=out if plan.fixed else 0,
        REDUCE_FIXED=reduce if plan.fixed else 0,
        LDA_FIXED=lda if plan.fixed else 0,
        HEAD_TAIL=plan.head_tail,
        FULL_TILES=plan.full_tiles,
        REDUCE_EACH_TILE=plan.reduce_each,
        WHOLE_REDUCTION=plan.whole,
        DENSE=plan.dense,
        SPLIT_K=plan.splits if plan.splits else 1,
        UNROLL=4,
    )
    for key, value in config.items():
        meta[
            {
                "BLOCK_SIZE_M": "BM",
                "BLOCK_SIZE_K": "BK",
                "L2_BYPASS_OFFSET": "L2_OFFSET",
            }.get(key, key)
        ] = value
    if plan.bm:
        meta["BM"] = plan.bm
    if plan.bk:
        meta["BK"] = plan.bk
    if plan.splits:
        meta["SPLIT_K"] = plan.splits
    if plan.full_tiles:
        meta["FULL_TILES"] = True
    meta["FULL_TILES"] = (
        meta["FULL_TILES"] and out % meta["BM"] == 0 and reduce % meta["BK"] == 0
    )
    meta["DENSE"] = meta["DENSE"] and lda == reduce == meta["BK"]
    return {key: value for key, value in meta.items() if key in raw.arg_names}


@lru_cache(maxsize=512)
def _real_gemv_entry(raw, plan, trans, out, reduce, lda, offset):
    if not plan.pool:
        return libentry()(raw)
    configs = []
    for cfg in runtime.get_tuned_config(plan.pool):
        meta = _real_gemv_meta(raw, plan, trans, out, reduce, lda, offset, cfg.kwargs)
        if plan.filter_bm and meta["BM"] != plan.filter_bm:
            continue
        if meta.get("BM", 1) * meta.get("BK", 1) > plan.limit:
            continue
        configs.append(
            triton.Config(meta, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
        )
    assert configs, (plan, out, reduce)
    keys = [
        key
        for key in ("OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_ZERO")
        if key in raw.arg_names
    ]
    return libentry()(libtuner(configs=configs, key=keys, restore_value=["y_ptr"])(raw))


def _gemv_launch_metadata(compiled, raw, args, resolved, grid):
    # The compiled source is authoritative: libentry may return stale defaults.
    for index, value in compiled.src.constants.items():
        if isinstance(index, tuple) and len(index) == 1:
            resolved[raw.arg_names[index[0]]] = value
        elif isinstance(index, str):
            resolved[index] = value
    constants = tuple(resolved[name] for name in raw.arg_names[len(args) :])
    dims = grid(resolved)
    count = 0
    while count < len(args) and isinstance(args[count], torch.Tensor):
        count += 1
    run = compiled.run
    special = (
        getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
    )
    assert int(getattr(compiled.metadata, "workspace_size", 0)) == 0
    return (*dims, *([1] * (3 - len(dims)))), constants, count, special


def _compile_real_gemv_launch(raw, plan, trans, args, device, grid):
    out, reduce, lda = args[5:8] if len(args) > 5 else (args[2], 0, 0)
    offset = _gemv_l2_offset(device) if plan.alias else 0
    launch_entry, grid = _gemv_launch_entry_grid(
        raw, plan, trans, out, reduce, lda, offset, grid
    )
    meta = (
        {}
        if plan.pool
        else _real_gemv_meta(raw, plan, trans, out, reduce, lda, offset, {})
    )
    if not plan.pool:
        meta.update(
            num_warps=plan.warps,
            num_stages=plan.stages,
            **_lowp_compile_options(raw, plan, trans),
        )
        compiled = raw.run(*args, grid=grid(meta), warmup=True, **meta)
        assert int(getattr(compiled.metadata, "workspace_size", 0)) == 0
    compiled, tuned = launch_entry[grid](*args, **meta)
    info = _gemv_launch_metadata(compiled, raw, args, {**tuned, **meta}, grid)
    return _RealGemvLaunch(compiled, compiled.run, *info)


# SGEMV kernels


@triton.jit
def _sgemv_tile_product(
    a_ptr,
    x_ptr,
    outputs,
    ks,
    out,
    end,
    lda,
    incx,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    MASK_OUT: tl.constexpr,
    MASK_K: tl.constexpr,
    DENSE: tl.constexpr = False,
):
    if MASK_K:
        xv = tl.load(x_ptr + ks * incx, ks < end, 0).to(tl.float32)
    else:
        xv = tl.load(x_ptr + ks * incx).to(tl.float32)
    if TRANSPOSE:
        offsets = ks[:, None] * lda + outputs[None, :]
        mask = tl.full((BK, BM), True, tl.int1)
        if MASK_OUT:
            mask = mask & (outputs[None, :] < out)
        if MASK_K:
            mask = mask & (ks[:, None] < end)
        if MASK_OUT or MASK_K:
            av = tl.load(a_ptr + offsets, mask, 0).to(tl.float32)
        else:
            av = tl.load(a_ptr + offsets).to(tl.float32)
        return av * xv[:, None]
    else:
        if DENSE:
            offsets = tl.program_id(0) * BM * BK + tl.arange(0, BM * BK)
            raw = tl.load(a_ptr + offsets, offsets < out * BK, 0)
            av = tl.reshape(raw, (BM, BK)).to(tl.float32)
        else:
            offsets = outputs[:, None] * lda + ks[None, :]
            mask = tl.full((BM, BK), True, tl.int1)
            if MASK_OUT:
                mask = mask & (outputs[:, None] < out)
            if MASK_K:
                mask = mask & (ks[None, :] < end)
            if MASK_OUT or MASK_K:
                av = tl.load(a_ptr + offsets, mask, 0).to(tl.float32)
            else:
                av = tl.load(a_ptr + offsets).to(tl.float32)
        return av * xv[None, :]


@triton.jit
def _sgemv_block_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT,
    REDUCE,
    LDA,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
    OUT_FIXED: tl.constexpr = 0,
    REDUCE_FIXED: tl.constexpr = 0,
    LDA_FIXED: tl.constexpr = 0,
    HEAD_TAIL: tl.constexpr = False,
    FULL_TILES: tl.constexpr = False,
    REDUCE_EACH_TILE: tl.constexpr = False,
    WHOLE_REDUCTION: tl.constexpr = False,
    DENSE: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
):
    """One tiled reduction with optional streaming, tail peeling and split K."""
    out = OUT_FIXED if OUT_FIXED else OUT
    reduce = REDUCE_FIXED if REDUCE_FIXED else REDUCE
    lda = LDA_FIXED if LDA_FIXED else LDA
    if L2_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_OFFSET).to(
            tl.pointer_type(a_ptr.dtype.element_ty)
        )
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    ki = tl.arange(0, BK)
    if SPLIT_K > 1:
        chunk = tl.cdiv(reduce, SPLIT_K * BK) * BK
        begin = tl.program_id(1) * chunk
        end = tl.minimum(begin + chunk, reduce)
    else:
        begin, end = 0, reduce
    if WHOLE_REDUCTION:
        product = _sgemv_tile_product(
            a_ptr,
            x_ptr,
            outputs,
            ki,
            out,
            reduce,
            lda,
            INCX,
            BM,
            BK,
            TRANSPOSE,
            True,
            True,
            DENSE,
        )
        total = tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
    else:
        if REDUCE_EACH_TILE:
            acc = tl.zeros((BM,), tl.float32)
        elif TRANSPOSE:
            acc = tl.zeros((BK, BM), tl.float32)
        else:
            acc = tl.zeros((BM, BK), tl.float32)
        loop_end = end // BK * BK if HEAD_TAIL else end
        full_out = (tl.program_id(0) + 1) * BM <= out
        for start in range(begin, loop_end, BK):
            ks = start + ki
            if HEAD_TAIL:
                if full_out:
                    product = _sgemv_tile_product(
                        a_ptr,
                        x_ptr,
                        outputs,
                        ks,
                        out,
                        end,
                        lda,
                        INCX,
                        BM,
                        BK,
                        TRANSPOSE,
                        False,
                        False,
                    )
                else:
                    product = _sgemv_tile_product(
                        a_ptr,
                        x_ptr,
                        outputs,
                        ks,
                        out,
                        end,
                        lda,
                        INCX,
                        BM,
                        BK,
                        TRANSPOSE,
                        True,
                        False,
                    )
            elif FULL_TILES:
                product = _sgemv_tile_product(
                    a_ptr,
                    x_ptr,
                    outputs,
                    ks,
                    out,
                    end,
                    lda,
                    INCX,
                    BM,
                    BK,
                    TRANSPOSE,
                    False,
                    False,
                )
            else:
                product = _sgemv_tile_product(
                    a_ptr,
                    x_ptr,
                    outputs,
                    ks,
                    out,
                    end,
                    lda,
                    INCX,
                    BM,
                    BK,
                    TRANSPOSE,
                    True,
                    True,
                )
            if REDUCE_EACH_TILE:
                acc += tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
            else:
                acc += product
        if HEAD_TAIL and end > begin and end % BK:
            ks = end // BK * BK + ki
            product = _sgemv_tile_product(
                a_ptr,
                x_ptr,
                outputs,
                ks,
                out,
                end,
                lda,
                INCX,
                BM,
                BK,
                TRANSPOSE,
                True,
                True,
            )
            if REDUCE_EACH_TILE:
                acc += tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
            else:
                acc += product
        if REDUCE_EACH_TILE:
            total = acc
        else:
            total = tl.sum(acc, 0) if TRANSPOSE else tl.sum(acc, 1)
    value = alpha * total
    if SPLIT_K > 1:
        tl.static_assert(y_ptr.dtype.element_ty == tl.float32)
        tl.atomic_add(y_ptr + outputs * INCY, value, outputs < out, sem="relaxed")
    else:
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < out, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < out)


@triton.jit
def _sgemv_narrow_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """Grouped narrow outputs accumulate in FP32 without intermediate storage."""
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    ki = tl.arange(0, BK)
    chunk = tl.cdiv(REDUCE, SPLIT_K * BK) * BK
    begin = tl.program_id(1) * chunk
    end = tl.minimum(begin + chunk, REDUCE)
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(begin, end, BK):
        ks = start + ki
        xv = tl.load(x_ptr + ks * INCX, ks < end, 0).to(tl.float32)
        if TRANSPOSE and LDA <= 128:
            fi = tl.arange(0, BK * triton.next_power_of_2(LDA))
            offsets = start * LDA + fi
            limit = (REDUCE - 1) * LDA + OUT
            raw = tl.load(
                a_ptr + offsets,
                (fi < BK * LDA) & (offsets < end * LDA) & (offsets < limit),
                0,
            )
            indices = ki[None, :] * LDA + outputs[:, None]
            gathered = tl.gather(raw, tl.reshape(indices, (BM * BK,)), 0)
            av = tl.reshape(gathered, (BM, BK)).to(tl.float32)
        elif TRANSPOSE:
            av = tl.load(
                a_ptr + ks[None, :] * LDA + outputs[:, None],
                (ks[None, :] < end) & (outputs[:, None] < OUT),
                0,
            ).to(tl.float32)
        else:
            av = tl.load(
                a_ptr + outputs[:, None] * LDA + ks[None, :],
                (outputs[:, None] < OUT) & (ks[None, :] < end),
                0,
            ).to(tl.float32)
        acc += av * xv[None, :]
    value = alpha * tl.sum(acc, 1)
    if SPLIT_K > 1:
        tl.static_assert(y_ptr.dtype.element_ty == tl.float32)
        tl.atomic_add(y_ptr + outputs * INCY, value, outputs < OUT, sem="relaxed")
    else:
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < OUT)


@triton.jit
def _sgemv_scale_kernel(
    y_ptr, beta: tl.float32, OUT, INCY, BETA_ZERO: tl.constexpr, BM: tl.constexpr
):
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    if BETA_ZERO:
        value = tl.zeros((BM,), tl.float32)
    else:
        value = beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(tl.float32)
    tl.store(y_ptr + outputs * INCY, value, outputs < OUT)


@triton.jit
def _sgemv_short_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    if not TRANSPOSE and REDUCE == 3 and LDA == REDUCE:
        outputs = tl.program_id(0) * BM + tl.arange(0, BM)
        fi = tl.arange(0, BM * 4)
        flat = tl.load(
            a_ptr + tl.program_id(0) * BM * 3 + fi,
            (fi < BM * 3) & (tl.program_id(0) * BM * 3 + fi < OUT * 3),
            0,
        ).to(tl.float32)
        total = tl.zeros((BM,), tl.float32)
        for k in tl.static_range(3):
            av = tl.gather(flat, tl.arange(0, BM) * 3 + k, 0)
            total += av * tl.load(x_ptr + k * INCX).to(tl.float32)
        value = alpha * total
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < OUT)
    else:
        oi = tl.arange(0, BM)
        outputs = tl.program_id(0) * BM + oi
        total = tl.zeros((BM // 8, 8), tl.float32)
        if not TRANSPOSE and LDA == REDUCE:
            lanes = tl.arange(0, BM * REDUCE)
            offsets = tl.program_id(0) * BM * REDUCE + lanes
            raw = tl.load(a_ptr + offsets, offsets < OUT * REDUCE, 0)
            if REDUCE == 1:
                v0 = tl.reshape(raw, (BM // 8, 8))
            elif REDUCE == 2:
                (v0, v1) = tl.split(tl.reshape(raw, (BM // 8, 8, 2)))
            else:
                (even, odd) = tl.split(tl.reshape(raw, (BM // 8, 8, 2, 2)))
                (v0, v2) = tl.split(even)
                (v1, v3) = tl.split(odd)
            total += v0 * tl.load(x_ptr)
            if REDUCE >= 2:
                total += v1 * tl.load(x_ptr + 1 * INCX)
            if REDUCE >= 3:
                total += v2 * tl.load(x_ptr + 2 * INCX)
            if REDUCE == 4:
                total += v3 * tl.load(x_ptr + 3 * INCX)
        else:
            for k in tl.static_range(REDUCE):
                offsets = k * LDA + outputs if TRANSPOSE else outputs * LDA + k
                av = tl.load(a_ptr + offsets, outputs < OUT, 0)
                total += tl.reshape(av, (BM // 8, 8)) * tl.load(x_ptr + k * INCX)
        result = alpha * total
        if not BETA_ZERO:
            old = tl.load(y_ptr + outputs * INCY, outputs < OUT, 0)
            result += beta * tl.reshape(old, (BM // 8, 8))
        tl.store(y_ptr + outputs * INCY, tl.reshape(result, (BM,)), outputs < OUT)


@triton.jit
def _sgemv_rows_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    SPLIT_K: tl.constexpr,
    UNROLL: tl.constexpr,
    L2_OFFSET: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    """Stream transposed odd rows; FP32 atomics replace the old scratch/finish pair."""
    if L2_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_OFFSET).to(tl.pointer_type(tl.float32))
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    chunk: tl.constexpr = triton.cdiv(REDUCE, SPLIT_K * UNROLL) * UNROLL
    begin = tl.program_id(1) * chunk
    end = tl.minimum(begin + chunk, REDUCE)
    acc = tl.zeros((BM,), tl.float32)
    for start in range(begin, end, UNROLL):
        for u in tl.static_range(UNROLL):
            k = start + u
            av = tl.load(a_ptr + k * LDA + outputs, (outputs < OUT) & (k < end), 0)
            xv = tl.load(x_ptr + k * INCX, k < end, 0)
            acc += av * xv
    tl.atomic_add(y_ptr + outputs * INCY, alpha * acc, outputs < OUT, sem="relaxed")


# SGEMV parameter selection. Specialized flags compile away in the same kernels.
def _select_sgemv_base_plan(trans, m, n, lda, incx=1, incy=1):
    if incx != 1 or incy != 1:
        return _RealGemvPlan(bm=8, bk=128)
    out, reduce = (m, n) if trans == 0 else (n, m)
    if max(m, n) <= 64:
        return _RealGemvPlan(
            bm=16 if trans else 8,
            bk=triton.next_power_of_2(reduce),
            fixed=True,
            whole=True,
        )
    if trans and 65536 <= reduce <= 262144 and out == 3 and lda <= 8:
        return _RealGemvPlan("narrow", bm=1, bk=1024, splits=16)
    if trans and 32 <= out <= 64 and reduce >= 32768 and lda == n:
        return _RealGemvPlan(bm=64, bk=64, splits=32, stages=2, fixed=True, alias=True)
    if trans and (m % 2 or n % 2) and 1023 <= min(m, n) and 3072 <= max(m, n) <= 4095:
        return _RealGemvPlan(bm=512, bk=8, splits=32, stages=2, fixed=True, alias=True)
    size = m * n * 4
    side = "t" if trans else "n"
    pool = lambda name: "sgemv_" + side + "_" + name + "_ascend"
    if trans == 0 and m == 64 and 32768 <= n <= 262144:
        return _RealGemvPlan(bm=1, bk=8192, fixed=True, head_tail=True)
    if trans and reduce in (3, 4) and out >= 4096:
        return _RealGemvPlan("short", bm=2048)
    if (
        trans
        and m % 2
        and n % 2
        and min(m, n) >= 3072
        and max(m, n) <= 16384
        and lda == n
        and 32 * 1024**2 <= size < 512 * 1024**2
    ):
        return _RealGemvPlan(
            "rows",
            bm=min(8192, triton.next_power_of_2(out)),
            splits=_gemv_row_splits(),
            alias=True,
        )
    if 32 <= out <= 64 and reduce >= 16 * out and size < 512 * 1024**2:
        return _RealGemvPlan(pool=pool("extreme_split"), fixed=True, splits=0)
    if min(m, n) <= 4 and max(m, n) >= 4096 and size < 512 * 1024**2:
        if reduce <= 4:
            return _RealGemvPlan("short", pool=pool("tiny_full"))
        if trans and out in (3, 4):
            return _RealGemvPlan("narrow", pool=pool("tiny_split"), bm=1, splits=0)
        return _RealGemvPlan("narrow", pool=pool("tiny_reduce"), bm=1)
    if 32 <= reduce <= 64 and out >= 16 * reduce and size < 512 * 1024**2:
        return _RealGemvPlan(
            pool=pool("extreme_full"),
            alias=not trans,
            bk=triton.next_power_of_2(reduce) if not trans else 0,
            whole=not trans,
        )
    odd_tail = (m % 2 or n % 2) and (
        (
            trans
            and ((511 <= m == n <= 1024) or (1023 <= n <= 1024 and 2 * n <= m <= 8192))
        )
        or (
            not trans
            and min(m, n) >= 1023
            and 8 * 1024**2 <= size < 512 * 1024**2
            and (m <= n or size >= 56 * 1024**2)
        )
    )
    if odd_tail:
        return _RealGemvPlan(
            pool=pool(
                "odd_large_tail" if not trans and size >= 56 * 1024**2 else "odd_tail"
            ),
            fixed=True,
            head_tail=True,
            alias=not trans,
            filter_bm=max(16, triton.next_power_of_2(out) // 32) if trans else 0,
        )
    if trans and min(m, n) >= 1023 and (m % 2 or n % 2):
        return _RealGemvPlan(
            pool=pool(
                "large"
                if min(m, n) >= 1024 and size >= 256 * 1024**2
                else "coalesced"
            )
        )
    if (
        not trans
        and min(m, n) >= 1024
        and max(m, n) <= 4096
        and m % 128 == 0
        and n % 256 == 0
        and 16 * 1024**2 <= size <= 32 * 1024**2
    ):
        return _RealGemvPlan(pool=pool("short_full"), alias=True, full_tiles=True)
    if (
        trans
        and min(m, n) >= 1024
        and m >= 4 * n
        and 16 * 1024**2 <= size <= 64 * 1024**2
    ):
        return _RealGemvPlan(pool=pool("medium_tiles"), alias=True, reduce_each=True)
    if (
        not trans
        and min(m, n) >= 1024
        and m <= n <= 4096
        and 16 * 1024**2 <= size <= 64 * 1024**2
    ) or (
        trans
        and min(m, n) >= 1024
        and m >= 4 * n
        and 64 * 1024**2 <= size < 512 * 1024**2
    ):
        return _RealGemvPlan(pool=pool("reduce_tiles"), alias=True, reduce_each=True)
    if (min(m, n) >= 1024 and size >= 64 * 1024**2) or (
        trans and min(m, n) >= 2048 and size >= 32 * 1024**2
    ):
        return _RealGemvPlan(pool=pool("stream"), alias=True)
    if trans and ((m == n and m % 2 and 63 <= m <= 1023) or 64 <= m == n <= 127):
        return _RealGemvPlan(
            pool=pool("small_odd_full"), bk=triton.next_power_of_2(reduce), whole=True
        )
    if trans:
        return _RealGemvPlan(pool=pool("coalesced"))
    if 63 <= m == n <= 1024:
        return _RealGemvPlan(pool=pool("small_odd" if m % 2 else "small_square"))
    if 4095 <= m <= 4096 and 1023 <= n <= 1024:
        return _RealGemvPlan(pool=pool("four_to_one"))
    return _RealGemvPlan(pool="sgemv_n")


_SGEMV_CACHE = {}
_SGEMV_KERNELS = dict(
    block=_sgemv_block_kernel,
    short=_sgemv_short_kernel,
    narrow=_sgemv_narrow_kernel,
    rows=_sgemv_rows_kernel,
)


def sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Ascend SGEMV; choose once and reuse a workspace-free launch."""
    assert trans in (0, 1) and m >= 0 and n >= 0 and lda >= n
    assert incx > 0 and incy > 0
    assert A.dtype == x.dtype == y.dtype == torch.float32
    assert A.device == x.device == y.device and A.device.type == "npu"
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    if not m or not n:
        return
    out, reduce = (m, n) if trans == 0 else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= 1 + (reduce - 1) * incx
    assert y.numel() >= 1 + (out - 1) * incy
    if type(alpha) is not float:
        alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
    if type(beta) is not float:
        beta = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    device = A.device.index
    if device != current_device():
        with torch_device_fn.device(A.device):
            return sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    if alpha == 0 and beta == 1:
        return
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    scalars = (alpha, beta, out, reduce, lda, incx, incy, beta == 0)
    key = (device, trans, scalars, tuple(p % 16 for p in pointers))
    saved = _SGEMV_CACHE.get(key)
    args = (A, x, y, *scalars)
    scale_args = (y, beta, out, incy, beta == 0)
    if saved is None:
        if len(_SGEMV_CACHE) >= 512:
            _SGEMV_CACHE.clear()
        if alpha == 0:
            launch = _compile_real_gemv_launch(
                _sgemv_scale_kernel,
                _RealGemvPlan(bm=256),
                trans,
                scale_args,
                device,
                lambda meta: (triton.cdiv(out, meta["BM"]),),
            )
            _SGEMV_CACHE[key] = (launch, None)
            return
        plan = _select_sgemv_plan(trans, m, n, lda, incx, incy)
        scale = None
        if plan.splits != 1:
            scale = _compile_real_gemv_launch(
                _sgemv_scale_kernel,
                _RealGemvPlan(bm=256),
                trans,
                scale_args,
                device,
                lambda meta: (triton.cdiv(out, meta["BM"]),),
            )
        main = _compile_real_gemv_launch(
            _SGEMV_KERNELS[plan.kernel],
            plan,
            trans,
            args,
            device,
            lambda meta: (triton.cdiv(out, meta["BM"]), meta.get("SPLIT_K", 1)),
        )
        _SGEMV_CACHE[key] = (scale, main)
        return
    scale, main = saved
    stream = current_stream(device)
    if scale is not None:
        scale(scale_args, stream, (pointers[2],))
    if main is not None:
        main(args, stream, pointers)


# HGEMV kernels


@triton.jit
def _hgemv_tile_product(
    a_ptr,
    x_ptr,
    outputs,
    ks,
    out,
    end,
    lda,
    incx,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    MASK_OUT: tl.constexpr,
    MASK_K: tl.constexpr,
    DENSE: tl.constexpr = False,
):
    if MASK_K:
        xv = tl.load(x_ptr + ks * incx, ks < end, 0).to(tl.float32)
    else:
        xv = tl.load(x_ptr + ks * incx).to(tl.float32)
    if TRANSPOSE:
        offsets = ks[:, None] * lda + outputs[None, :]
        mask = tl.full((BK, BM), True, tl.int1)
        if MASK_OUT:
            mask = mask & (outputs[None, :] < out)
        if MASK_K:
            mask = mask & (ks[:, None] < end)
        if MASK_OUT or MASK_K:
            av = tl.load(a_ptr + offsets, mask, 0).to(tl.float32)
        else:
            av = tl.load(a_ptr + offsets).to(tl.float32)
        return av * xv[:, None]
    else:
        if DENSE:
            offsets = tl.program_id(0) * BM * BK + tl.arange(0, BM * BK)
            raw = tl.load(a_ptr + offsets, offsets < out * BK, 0)
            av = tl.reshape(raw, (BM, BK)).to(tl.float32)
        else:
            offsets = outputs[:, None] * lda + ks[None, :]
            mask = tl.full((BM, BK), True, tl.int1)
            if MASK_OUT:
                mask = mask & (outputs[:, None] < out)
            if MASK_K:
                mask = mask & (ks[None, :] < end)
            if MASK_OUT or MASK_K:
                av = tl.load(a_ptr + offsets, mask, 0).to(tl.float32)
            else:
                av = tl.load(a_ptr + offsets).to(tl.float32)
        return av * xv[None, :]


@triton.jit
def _hgemv_block_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT,
    REDUCE,
    LDA,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
    OUT_FIXED: tl.constexpr = 0,
    REDUCE_FIXED: tl.constexpr = 0,
    LDA_FIXED: tl.constexpr = 0,
    HEAD_TAIL: tl.constexpr = False,
    FULL_TILES: tl.constexpr = False,
    REDUCE_EACH_TILE: tl.constexpr = False,
    WHOLE_REDUCTION: tl.constexpr = False,
    DENSE: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
):
    """One tiled reduction with optional streaming, tail peeling and split K."""
    out = OUT_FIXED if OUT_FIXED else OUT
    reduce = REDUCE_FIXED if REDUCE_FIXED else REDUCE
    lda = LDA_FIXED if LDA_FIXED else LDA
    if L2_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_OFFSET).to(
            tl.pointer_type(a_ptr.dtype.element_ty)
        )
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    ki = tl.arange(0, BK)
    if SPLIT_K > 1:
        chunk = tl.cdiv(reduce, SPLIT_K * BK) * BK
        begin = tl.program_id(1) * chunk
        end = tl.minimum(begin + chunk, reduce)
    else:
        begin, end = 0, reduce
    if WHOLE_REDUCTION:
        product = _hgemv_tile_product(
            a_ptr,
            x_ptr,
            outputs,
            ki,
            out,
            reduce,
            lda,
            INCX,
            BM,
            BK,
            TRANSPOSE,
            True,
            True,
            DENSE,
        )
        total = tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
    else:
        if REDUCE_EACH_TILE:
            acc = tl.zeros((BM,), tl.float32)
        elif TRANSPOSE:
            acc = tl.zeros((BK, BM), tl.float32)
        else:
            acc = tl.zeros((BM, BK), tl.float32)
        loop_end = end // BK * BK if HEAD_TAIL else end
        full_out = (tl.program_id(0) + 1) * BM <= out
        for start in range(begin, loop_end, BK):
            ks = start + ki
            if HEAD_TAIL:
                if full_out:
                    product = _hgemv_tile_product(
                        a_ptr,
                        x_ptr,
                        outputs,
                        ks,
                        out,
                        end,
                        lda,
                        INCX,
                        BM,
                        BK,
                        TRANSPOSE,
                        False,
                        False,
                    )
                else:
                    product = _hgemv_tile_product(
                        a_ptr,
                        x_ptr,
                        outputs,
                        ks,
                        out,
                        end,
                        lda,
                        INCX,
                        BM,
                        BK,
                        TRANSPOSE,
                        True,
                        False,
                    )
            elif FULL_TILES:
                product = _hgemv_tile_product(
                    a_ptr,
                    x_ptr,
                    outputs,
                    ks,
                    out,
                    end,
                    lda,
                    INCX,
                    BM,
                    BK,
                    TRANSPOSE,
                    False,
                    False,
                )
            else:
                product = _hgemv_tile_product(
                    a_ptr,
                    x_ptr,
                    outputs,
                    ks,
                    out,
                    end,
                    lda,
                    INCX,
                    BM,
                    BK,
                    TRANSPOSE,
                    True,
                    True,
                )
            if REDUCE_EACH_TILE:
                acc += tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
            else:
                acc += product
        if HEAD_TAIL and end > begin and end % BK:
            ks = end // BK * BK + ki
            product = _hgemv_tile_product(
                a_ptr,
                x_ptr,
                outputs,
                ks,
                out,
                end,
                lda,
                INCX,
                BM,
                BK,
                TRANSPOSE,
                True,
                True,
            )
            if REDUCE_EACH_TILE:
                acc += tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
            else:
                acc += product
        if REDUCE_EACH_TILE:
            total = acc
        else:
            total = tl.sum(acc, 0) if TRANSPOSE else tl.sum(acc, 1)
    value = alpha * total
    if SPLIT_K > 1:
        tl.static_assert(y_ptr.dtype.element_ty == tl.float32)
        tl.atomic_add(y_ptr + outputs * INCY, value, outputs < out, sem="relaxed")
    else:
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < out, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < out)


@triton.jit
def _hgemv_narrow_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """Grouped narrow outputs accumulate in FP32 without intermediate storage."""
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    ki = tl.arange(0, BK)
    chunk = tl.cdiv(REDUCE, SPLIT_K * BK) * BK
    begin = tl.program_id(1) * chunk
    end = tl.minimum(begin + chunk, REDUCE)
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(begin, end, BK):
        ks = start + ki
        xv = tl.load(x_ptr + ks * INCX, ks < end, 0).to(tl.float32)
        if TRANSPOSE and LDA <= 64:
            fi = tl.arange(0, BK * triton.next_power_of_2(LDA))
            offsets = start * LDA + fi
            limit = (REDUCE - 1) * LDA + OUT
            raw = tl.load(
                a_ptr + offsets,
                (fi < BK * LDA) & (offsets < end * LDA) & (offsets < limit),
                0,
            )
            indices = ki[None, :] * LDA + outputs[:, None]
            gathered = tl.gather(raw, tl.reshape(indices, (BM * BK,)), 0)
            av = tl.reshape(gathered, (BM, BK)).to(tl.float32)
        elif TRANSPOSE:
            av = tl.load(
                a_ptr + ks[None, :] * LDA + outputs[:, None],
                (ks[None, :] < end) & (outputs[:, None] < OUT),
                0,
            ).to(tl.float32)
        else:
            av = tl.load(
                a_ptr + outputs[:, None] * LDA + ks[None, :],
                (outputs[:, None] < OUT) & (ks[None, :] < end),
                0,
            ).to(tl.float32)
        acc += av * xv[None, :]
    value = alpha * tl.sum(acc, 1)
    if SPLIT_K > 1:
        tl.static_assert(y_ptr.dtype.element_ty == tl.float32)
        tl.atomic_add(y_ptr + outputs * INCY, value, outputs < OUT, sem="relaxed")
    else:
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < OUT)


@triton.jit
def _hgemv_scale_kernel(
    y_ptr, beta: tl.float32, OUT, INCY, BETA_ZERO: tl.constexpr, BM: tl.constexpr
):
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    if BETA_ZERO:
        value = tl.zeros((BM,), tl.float32)
    else:
        value = beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(tl.float32)
    tl.store(y_ptr + outputs * INCY, value, outputs < OUT)


@triton.jit
def _hgemv_short_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    if not TRANSPOSE and REDUCE == 3 and LDA == REDUCE:
        outputs = tl.program_id(0) * BM + tl.arange(0, BM)
        fi = tl.arange(0, BM * 4)
        flat = tl.load(
            a_ptr + tl.program_id(0) * BM * 3 + fi,
            (fi < BM * 3) & (tl.program_id(0) * BM * 3 + fi < OUT * 3),
            0,
        ).to(tl.float32)
        total = tl.zeros((BM,), tl.float32)
        for k in tl.static_range(3):
            av = tl.gather(flat, tl.arange(0, BM) * 3 + k, 0)
            total += av * tl.load(x_ptr + k * INCX).to(tl.float32)
        value = alpha * total
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < OUT)
    else:
        outputs = tl.program_id(0) * BM + tl.arange(0, BM)
        total = tl.zeros((BM // 16, 16), tl.float32)
        if not TRANSPOSE and LDA == REDUCE:
            offsets = tl.program_id(0) * BM * REDUCE + tl.arange(0, BM * REDUCE)
            raw = tl.load(a_ptr + offsets, offsets < OUT * REDUCE, 0)
            if REDUCE == 1:
                v0 = tl.reshape(raw, (BM // 16, 16))
            elif REDUCE == 2:
                (v0, v1) = tl.split(tl.reshape(raw, (BM // 16, 16, 2)))
            else:
                (even, odd) = tl.split(tl.reshape(raw, (BM // 16, 16, 2, 2)))
                (v0, v2) = tl.split(even)
                (v1, v3) = tl.split(odd)
            total += v0.to(tl.float32) * tl.load(x_ptr).to(tl.float32)
            if REDUCE >= 2:
                total += v1.to(tl.float32) * tl.load(x_ptr + 1 * INCX).to(tl.float32)
            if REDUCE >= 3:
                total += v2.to(tl.float32) * tl.load(x_ptr + 2 * INCX).to(tl.float32)
            if REDUCE == 4:
                total += v3.to(tl.float32) * tl.load(x_ptr + 3 * INCX).to(tl.float32)
        else:
            for k in tl.static_range(REDUCE):
                offsets = k * LDA + outputs if TRANSPOSE else outputs * LDA + k
                av = tl.load(a_ptr + offsets, outputs < OUT, 0)
                total += tl.reshape(av, (BM // 16, 16)).to(tl.float32) * tl.load(
                    x_ptr + k * INCX
                ).to(tl.float32)
        value = alpha * total
        if not BETA_ZERO:
            value += beta * tl.reshape(
                tl.load(y_ptr + outputs * INCY, outputs < OUT, 0), (BM // 16, 16)
            ).to(tl.float32)
        tl.store(y_ptr + outputs * INCY, tl.reshape(value, (BM,)), outputs < OUT)


# HGEMV keeps FP32 accumulation and rounds only on the final output store.
def _select_hgemv_base_plan(trans, m, n, lda, incx=1, incy=1):
    if incx != 1 or incy != 1:
        return _RealGemvPlan(bm=8, bk=128)
    out, reduce = (m, n) if not trans else (n, m)
    if trans and 32 <= reduce <= 64 and out >= 16384:
        return _RealGemvPlan(
            bm=256, bk=triton.next_power_of_2(reduce), fixed=True, whole=True
        )
    size = m * n * 2
    side = "t" if trans else "n"
    pool = lambda name: "hgemv_" + side + "_" + name + "_ascend"
    # Use one output tile per core for medium skinny and large odd T shapes.
    if trans and (
        (
            m % 2
            and n % 2
            and (4095 <= min(m, n) or 3072 <= m == n)
            and max(m, n) <= 16384
            and m <= 2 * n
        )
        or (
            m % 256 == n % 256 == 0
            and 1024 <= n
            and 4 * n <= m <= 8 * n
            and size <= 16 * 1024**2
        )
    ):
        bm = max(32, triton.next_power_of_2(out) // 32)
        return _RealGemvPlan(bm=bm, bk=8192 // bm, fixed=True, head_tail=True)
    if not trans:
        if 3072 <= m <= 4096 and 768 <= n <= 1024:
            return _RealGemvPlan(
                bm=32, bk=256, stages=2, fixed=True, head_tail=True, alias=True
            )
        if m == n and m in (1023, 1024):
            return _RealGemvPlan(bm=32, bk=128, fixed=True, head_tail=True, alias=True)
        if m == 1024 and n == 4096:
            return _RealGemvPlan(bm=16, bk=512, fixed=True, head_tail=True, alias=True)
        if (
            m % 2
            and n % 2
            and 2047 <= m <= n
            and 6143 <= n <= 16384
            and size < 512 * 1024**2
        ):
            return _RealGemvPlan(
                bm=8, bk=1024, stages=2, fixed=True, head_tail=True, alias=True
            )
        if 1 <= m <= 2 and 4096 <= n <= 262144:
            return _RealGemvPlan(bm=1, bk=8192, fixed=True, head_tail=True)
    if 63 <= m == n <= 512 and not (trans and m == 256):
        return _RealGemvPlan(
            bm=16 if (not trans and m == 255) or (trans and m == 512) else 32,
            bk=triton.next_power_of_2(reduce),
            fixed=True,
            head_tail=True,
        )
    if (
        not trans
        and m % 2
        and n % 2
        and min(m, n) >= 2047
        and max(m, n) <= 16384
        and size < 512 * 1024**2
    ):
        return _RealGemvPlan(
            bm=16, bk=512, stages=2, fixed=True, head_tail=True, alias=True
        )
    if not trans and reduce == 3 and out >= 4096:
        return _RealGemvPlan("short", bm=2048)
    if reduce <= 4 and out >= 4096:
        return _RealGemvPlan("short", bm=2048)
    if out <= 4 and reduce >= 4096:
        return _RealGemvPlan("narrow", bm=1, bk=4096 if lda <= 8 or not trans else 512)
    if not trans and out <= 64 and reduce >= 4096:
        return _RealGemvPlan(
            pool="hgemv_n_long_reduce_ascend", fixed=True, head_tail=True, limit=8192
        )
    if trans and out <= 64 and reduce >= 4096:
        long_narrow = 32 <= out and reduce >= 32768 and lda <= 64
        return _lowp_narrow_plan(
            (out, reduce, lda),
            bm=2 if long_narrow else min(4, triton.next_power_of_2(out)),
            bk=1024 if long_narrow else 512,
        )
    if reduce <= 64 and out >= 4096:
        if not trans and reduce == 64 and out >= 16384:
            return _RealGemvPlan(
                bm=256, bk=64, whole=True, dense=lda == reduce, fixed=True
            )
        return _RealGemvPlan(
            pool="hgemv_short_reduce_ascend" if trans else "hgemv_short_full_ascend",
            bk=triton.next_power_of_2(reduce) if not trans else 0,
            whole=not trans,
            dense=not trans and lda == reduce,
        )
    if (
        trans
        and m % 512 == n % 512 == 0
        and n >= 2 * m
        and 1024**3 <= size < 2 * 1024**3
    ):
        return _RealGemvPlan(
            pool="hgemv_t_huge_full_ascend", alias=True, full_tiles=True, limit=8192
        )
    extended = (
        (min(m, n) <= 4 and max(m, n) >= 4096)
        or ((m % 2 or n % 2) and 5 <= min(m, n) <= 1023 and max(m, n) <= 4095)
        or (not trans and 256 < m == n < 1024)
        or (
            5 <= min(m, n) <= 64
            and max(m, n) >= max(4096, 16 * min(m, n))
            and size < 512 * 1024**2
        )
    )
    if extended:
        if trans and reduce >= 512 and out > 2 * reduce:
            bm = max(16, triton.next_power_of_2(out) // 32)
            return _RealGemvPlan(
                bm=bm, bk=min(256, 8192 // bm), fixed=True, head_tail=True
            )
        return _RealGemvPlan(
            pool=pool("head_tail"),
            fixed=True,
            head_tail=True,
            limit=8192,
            bm=max(16, triton.next_power_of_2(out) // 32) if trans else 0,
        )
    if (
        m % 512 == n % 512 == 0
        and 128 * 1024**2 <= size < 1024**3
        and (not trans or (m <= 6144 and n >= 2 * m))
    ):
        return _RealGemvPlan(
            pool=pool("large_block"),
            alias=True,
            full_tiles=True,
            limit=8192 if trans else 16384,
        )
    if (
        m % 512 == n % 512 == 0
        and 2048 <= min(m, n) <= 4096
        and 16 * 1024**2 <= size < 128 * 1024**2
        and ((not trans and m >= n) or (trans and n >= m and size < 32 * 1024**2))
    ):
        return _RealGemvPlan(
            pool=pool("block"), alias=True, limit=8192 if trans else 16384
        )
    if (
        m % 2 == n % 2 == 0
        and min(m, n) >= 1024
        and max(m, n) < 16 * min(m, n)
        and 64 * 1024**2 <= size < 1024**3
    ):
        return _RealGemvPlan(
            pool=pool("stream"), alias=True, limit=8192 if trans else 16384
        )
    if trans:
        return _RealGemvPlan(
            pool=pool(
                "large"
                if min(m, n) >= 1024 and size >= 256 * 1024**2
                else "coalesced"
            ),
            limit=8192,
        )
    if reduce <= 1024:
        return _RealGemvPlan(
            pool="hgemv_n_small",
            bk=triton.next_power_of_2(reduce),
            whole=True,
            limit=8192,
        )
    return _RealGemvPlan(pool="hgemv_n")


_HGEMV_CACHE = {}
_HGEMV_KERNELS = dict(
    block=_hgemv_block_kernel, short=_hgemv_short_kernel, narrow=_hgemv_narrow_kernel
)


def hgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Ascend HGEMV; choose once and reuse a workspace-free launch."""
    assert trans in (0, 1) and m >= 0 and n >= 0 and lda >= n
    assert incx > 0 and incy > 0
    assert A.dtype == x.dtype == y.dtype == torch.float16
    assert A.device == x.device == y.device and A.device.type == "npu"
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    if not m or not n:
        return
    out, reduce = (m, n) if trans == 0 else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= 1 + (reduce - 1) * incx
    assert y.numel() >= 1 + (out - 1) * incy
    if type(alpha) is not float:
        alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
    if type(beta) is not float:
        beta = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    device = A.device.index
    if device != current_device():
        with torch_device_fn.device(A.device):
            return hgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    if alpha == 0 and beta == 1:
        return
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    scalars = (alpha, beta, out, reduce, lda, incx, incy, beta == 0)
    key = (device, trans, scalars, tuple(p % 16 for p in pointers))
    saved = _HGEMV_CACHE.get(key)
    args = (A, x, y, *scalars)
    scale_args = (y, beta, out, incy, beta == 0)
    if saved is None:
        if len(_HGEMV_CACHE) >= 512:
            _HGEMV_CACHE.clear()
        if alpha == 0:
            launch = _compile_real_gemv_launch(
                _hgemv_scale_kernel,
                _RealGemvPlan(bm=256),
                trans,
                scale_args,
                device,
                lambda meta: (triton.cdiv(out, meta["BM"]),),
            )
            _HGEMV_CACHE[key] = (launch, None)
            return
        plan = _select_hgemv_plan(trans, m, n, lda, incx, incy)
        scale = None
        if plan.splits != 1:
            scale = _compile_real_gemv_launch(
                _hgemv_scale_kernel,
                _RealGemvPlan(bm=256),
                trans,
                scale_args,
                device,
                lambda meta: (triton.cdiv(out, meta["BM"]),),
            )
        main = _compile_real_gemv_launch(
            _HGEMV_KERNELS[plan.kernel],
            plan,
            trans,
            args,
            device,
            lambda meta: (triton.cdiv(out, meta["BM"]), meta.get("SPLIT_K", 1)),
        )
        _HGEMV_CACHE[key] = (scale, main)
        return
    scale, main = saved
    stream = current_stream(device)
    if scale is not None:
        scale(scale_args, stream, (pointers[2],))
    if main is not None:
        main(args, stream, pointers)


# BFGEMV kernels


@triton.jit
def _bfgemv_tile_product(
    a_ptr,
    x_ptr,
    outputs,
    ks,
    out,
    end,
    lda,
    incx,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    MASK_OUT: tl.constexpr,
    MASK_K: tl.constexpr,
    DENSE: tl.constexpr = False,
):
    if MASK_K:
        xv = tl.load(x_ptr + ks * incx, ks < end, 0).to(tl.float32)
    else:
        xv = tl.load(x_ptr + ks * incx).to(tl.float32)
    if TRANSPOSE:
        offsets = ks[:, None] * lda + outputs[None, :]
        mask = tl.full((BK, BM), True, tl.int1)
        if MASK_OUT:
            mask = mask & (outputs[None, :] < out)
        if MASK_K:
            mask = mask & (ks[:, None] < end)
        if MASK_OUT or MASK_K:
            av = tl.load(a_ptr + offsets, mask, 0).to(tl.float32)
        else:
            av = tl.load(a_ptr + offsets).to(tl.float32)
        return av * xv[:, None]
    else:
        if DENSE:
            offsets = tl.program_id(0) * BM * BK + tl.arange(0, BM * BK)
            raw = tl.load(a_ptr + offsets, offsets < out * BK, 0)
            av = tl.reshape(raw, (BM, BK)).to(tl.float32)
        else:
            offsets = outputs[:, None] * lda + ks[None, :]
            mask = tl.full((BM, BK), True, tl.int1)
            if MASK_OUT:
                mask = mask & (outputs[:, None] < out)
            if MASK_K:
                mask = mask & (ks[None, :] < end)
            if MASK_OUT or MASK_K:
                av = tl.load(a_ptr + offsets, mask, 0).to(tl.float32)
            else:
                av = tl.load(a_ptr + offsets).to(tl.float32)
        return av * xv[None, :]


@triton.jit
def _bfgemv_block_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT,
    REDUCE,
    LDA,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
    OUT_FIXED: tl.constexpr = 0,
    REDUCE_FIXED: tl.constexpr = 0,
    LDA_FIXED: tl.constexpr = 0,
    HEAD_TAIL: tl.constexpr = False,
    FULL_TILES: tl.constexpr = False,
    REDUCE_EACH_TILE: tl.constexpr = False,
    WHOLE_REDUCTION: tl.constexpr = False,
    DENSE: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
):
    """One tiled reduction with optional streaming, tail peeling and split K."""
    out = OUT_FIXED if OUT_FIXED else OUT
    reduce = REDUCE_FIXED if REDUCE_FIXED else REDUCE
    lda = LDA_FIXED if LDA_FIXED else LDA
    if L2_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_OFFSET).to(
            tl.pointer_type(a_ptr.dtype.element_ty)
        )
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    ki = tl.arange(0, BK)
    if SPLIT_K > 1:
        chunk = tl.cdiv(reduce, SPLIT_K * BK) * BK
        begin = tl.program_id(1) * chunk
        end = tl.minimum(begin + chunk, reduce)
    else:
        begin, end = 0, reduce
    if WHOLE_REDUCTION:
        product = _bfgemv_tile_product(
            a_ptr,
            x_ptr,
            outputs,
            ki,
            out,
            reduce,
            lda,
            INCX,
            BM,
            BK,
            TRANSPOSE,
            True,
            True,
            DENSE,
        )
        total = tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
    else:
        if REDUCE_EACH_TILE:
            acc = tl.zeros((BM,), tl.float32)
        elif TRANSPOSE:
            acc = tl.zeros((BK, BM), tl.float32)
        else:
            acc = tl.zeros((BM, BK), tl.float32)
        loop_end = end // BK * BK if HEAD_TAIL else end
        full_out = (tl.program_id(0) + 1) * BM <= out
        for start in range(begin, loop_end, BK):
            ks = start + ki
            if HEAD_TAIL:
                if full_out:
                    product = _bfgemv_tile_product(
                        a_ptr,
                        x_ptr,
                        outputs,
                        ks,
                        out,
                        end,
                        lda,
                        INCX,
                        BM,
                        BK,
                        TRANSPOSE,
                        False,
                        False,
                    )
                else:
                    product = _bfgemv_tile_product(
                        a_ptr,
                        x_ptr,
                        outputs,
                        ks,
                        out,
                        end,
                        lda,
                        INCX,
                        BM,
                        BK,
                        TRANSPOSE,
                        True,
                        False,
                    )
            elif FULL_TILES:
                product = _bfgemv_tile_product(
                    a_ptr,
                    x_ptr,
                    outputs,
                    ks,
                    out,
                    end,
                    lda,
                    INCX,
                    BM,
                    BK,
                    TRANSPOSE,
                    False,
                    False,
                )
            else:
                product = _bfgemv_tile_product(
                    a_ptr,
                    x_ptr,
                    outputs,
                    ks,
                    out,
                    end,
                    lda,
                    INCX,
                    BM,
                    BK,
                    TRANSPOSE,
                    True,
                    True,
                )
            if REDUCE_EACH_TILE:
                acc += tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
            else:
                acc += product
        if HEAD_TAIL and end > begin and end % BK:
            ks = end // BK * BK + ki
            product = _bfgemv_tile_product(
                a_ptr,
                x_ptr,
                outputs,
                ks,
                out,
                end,
                lda,
                INCX,
                BM,
                BK,
                TRANSPOSE,
                True,
                True,
            )
            if REDUCE_EACH_TILE:
                acc += tl.sum(product, 0) if TRANSPOSE else tl.sum(product, 1)
            else:
                acc += product
        if REDUCE_EACH_TILE:
            total = acc
        else:
            total = tl.sum(acc, 0) if TRANSPOSE else tl.sum(acc, 1)
    value = alpha * total
    if SPLIT_K > 1:
        tl.static_assert(y_ptr.dtype.element_ty == tl.float32)
        tl.atomic_add(y_ptr + outputs * INCY, value, outputs < out, sem="relaxed")
    else:
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < out, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < out)


@triton.jit
def _bfgemv_narrow_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """Grouped narrow outputs accumulate in FP32 without intermediate storage."""
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    ki = tl.arange(0, BK)
    chunk = tl.cdiv(REDUCE, SPLIT_K * BK) * BK
    begin = tl.program_id(1) * chunk
    end = tl.minimum(begin + chunk, REDUCE)
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(begin, end, BK):
        ks = start + ki
        xv = tl.load(x_ptr + ks * INCX, ks < end, 0).to(tl.float32)
        if TRANSPOSE and LDA <= 64:
            fi = tl.arange(0, BK * triton.next_power_of_2(LDA))
            offsets = start * LDA + fi
            limit = (REDUCE - 1) * LDA + OUT
            raw = tl.load(
                a_ptr + offsets,
                (fi < BK * LDA) & (offsets < end * LDA) & (offsets < limit),
                0,
            )
            indices = ki[None, :] * LDA + outputs[:, None]
            # Preserve BF16 bits through the smaller FP16 gather lowering.
            gathered = tl.gather(
                raw.to(tl.float16, bitcast=True), tl.reshape(indices, (BM * BK,)), 0
            ).to(tl.bfloat16, bitcast=True)
            av = tl.reshape(gathered, (BM, BK)).to(tl.float32)
        elif TRANSPOSE:
            av = tl.load(
                a_ptr + ks[None, :] * LDA + outputs[:, None],
                (ks[None, :] < end) & (outputs[:, None] < OUT),
                0,
            ).to(tl.float32)
        else:
            av = tl.load(
                a_ptr + outputs[:, None] * LDA + ks[None, :],
                (outputs[:, None] < OUT) & (ks[None, :] < end),
                0,
            ).to(tl.float32)
        acc += av * xv[None, :]
    value = alpha * tl.sum(acc, 1)
    if SPLIT_K > 1:
        tl.static_assert(y_ptr.dtype.element_ty == tl.float32)
        tl.atomic_add(y_ptr + outputs * INCY, value, outputs < OUT, sem="relaxed")
    else:
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < OUT)


@triton.jit
def _bfgemv_scale_kernel(
    y_ptr, beta: tl.float32, OUT, INCY, BETA_ZERO: tl.constexpr, BM: tl.constexpr
):
    outputs = tl.program_id(0) * BM + tl.arange(0, BM)
    if BETA_ZERO:
        value = tl.zeros((BM,), tl.float32)
    else:
        value = beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(tl.float32)
    tl.store(y_ptr + outputs * INCY, value, outputs < OUT)


@triton.jit
def _bfgemv_short_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    if not TRANSPOSE and REDUCE == 3 and LDA == REDUCE:
        outputs = tl.program_id(0) * BM + tl.arange(0, BM)
        fi = tl.arange(0, BM * 4)
        flat = tl.load(
            a_ptr + tl.program_id(0) * BM * 3 + fi,
            (fi < BM * 3) & (tl.program_id(0) * BM * 3 + fi < OUT * 3),
            0,
        ).to(tl.float32)
        total = tl.zeros((BM,), tl.float32)
        for k in tl.static_range(3):
            av = tl.gather(flat, tl.arange(0, BM) * 3 + k, 0)
            total += av * tl.load(x_ptr + k * INCX).to(tl.float32)
        value = alpha * total
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < OUT, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < OUT)
    else:
        outputs = tl.program_id(0) * BM + tl.arange(0, BM)
        total = tl.zeros((BM // 16, 16), tl.float32)
        if not TRANSPOSE and LDA == REDUCE:
            offsets = tl.program_id(0) * BM * REDUCE + tl.arange(0, BM * REDUCE)
            raw = tl.load(a_ptr + offsets, offsets < OUT * REDUCE, 0)
            if REDUCE == 1:
                v0 = tl.reshape(raw, (BM // 16, 16))
            elif REDUCE == 2:
                (v0, v1) = tl.split(tl.reshape(raw, (BM // 16, 16, 2)))
            else:
                (even, odd) = tl.split(tl.reshape(raw, (BM // 16, 16, 2, 2)))
                (v0, v2) = tl.split(even)
                (v1, v3) = tl.split(odd)
            total += v0.to(tl.float32) * tl.load(x_ptr).to(tl.float32)
            if REDUCE >= 2:
                total += v1.to(tl.float32) * tl.load(x_ptr + 1 * INCX).to(tl.float32)
            if REDUCE >= 3:
                total += v2.to(tl.float32) * tl.load(x_ptr + 2 * INCX).to(tl.float32)
            if REDUCE == 4:
                total += v3.to(tl.float32) * tl.load(x_ptr + 3 * INCX).to(tl.float32)
        else:
            for k in tl.static_range(REDUCE):
                offsets = k * LDA + outputs if TRANSPOSE else outputs * LDA + k
                av = tl.load(a_ptr + offsets, outputs < OUT, 0)
                total += tl.reshape(av, (BM // 16, 16)).to(tl.float32) * tl.load(
                    x_ptr + k * INCX
                ).to(tl.float32)
        value = alpha * total
        if not BETA_ZERO:
            value += beta * tl.reshape(
                tl.load(y_ptr + outputs * INCY, outputs < OUT, 0), (BM // 16, 16)
            ).to(tl.float32)
        tl.store(y_ptr + outputs * INCY, tl.reshape(value, (BM,)), outputs < OUT)


# BFGEMV keeps its base policy and shares the low-precision whole-row path.
def _select_bfgemv_base_plan(trans, m, n, lda, incx=1, incy=1):
    if incx != 1 or incy != 1:
        return _RealGemvPlan(bm=8, bk=128)
    out, reduce = (m, n) if not trans else (n, m)
    if trans and 32 <= reduce <= 64 and out >= 16384:
        return _RealGemvPlan(
            bm=256, bk=triton.next_power_of_2(reduce), fixed=True, whole=True
        )
    size = m * n * 2
    side = "t" if trans else "n"
    pool = lambda name: "bfgemv_" + side + "_" + name + "_ascend"
    # Use one output tile per core for medium skinny and large odd T shapes.
    if trans and (
        (
            m % 2
            and n % 2
            and (4095 <= min(m, n) or 3072 <= m == n)
            and max(m, n) <= 16384
            and m <= 2 * n
        )
        or (
            m % 256 == n % 256 == 0
            and 1024 <= n
            and 4 * n <= m <= 8 * n
            and size <= 16 * 1024**2
        )
    ):
        bm = max(32, triton.next_power_of_2(out) // 32)
        return _RealGemvPlan(bm=bm, bk=8192 // bm, fixed=True, head_tail=True)
    if not trans:
        if 2048 <= m <= 4096 and 512 < n <= 1024:
            return _RealGemvPlan(
                bm=32, bk=256, stages=2, fixed=True, head_tail=True, alias=True
            )
        if m == 1024 and n == 4096:
            return _RealGemvPlan(bm=16, bk=512, fixed=True, head_tail=True, alias=True)
        if (
            m % 2
            and n % 2
            and 2047 <= m <= n
            and 6143 <= n <= 16384
            and size < 512 * 1024**2
        ):
            return _RealGemvPlan(
                bm=8, bk=1024, stages=2, fixed=True, head_tail=True, alias=True
            )
        if (
            m % 2
            and n % 2
            and min(m, n) >= 2047
            and max(m, n) <= 16384
            and size < 512 * 1024**2
        ):
            return _RealGemvPlan(
                bm=16, bk=512, stages=2, fixed=True, head_tail=True, alias=True
            )
    if trans and m == n and m in (64, 128):
        return _RealGemvPlan(bm=32, bk=m, fixed=True, whole=True)
    if trans and m == n and m % 2 and 513 <= m <= 1023:
        return _RealGemvPlan(bm=32, bk=128, fixed=True, head_tail=True)
    if not trans and 1024 < m == n <= 4095:
        return _RealGemvPlan(bm=32, bk=256, fixed=True, head_tail=True)
    if trans and reduce in (3, 4) and out >= 4096:
        return _RealGemvPlan(bm=2048, bk=4, fixed=True, whole=True)
    if 32 <= m == n <= 1024 and (not trans or m % 2):
        bm = (
            (16 if m % 2 == 0 and 256 <= m < 512 else 32 if m <= 512 else 16)
            if not trans
            else 16
        )
        return _RealGemvPlan(
            bm=bm,
            bk=triton.next_power_of_2(reduce),
            fixed=True,
            whole=True,
            dense=not trans and lda == reduce,
            limit=32768,
        )
    if not trans and reduce == 3 and out >= 4096:
        return _RealGemvPlan("short", bm=2048)
    if reduce <= 4 and out >= 4096:
        return _RealGemvPlan("short", pool="bfgemv_tiny_full_ascend")
    if out <= 4 and reduce >= 4096:
        return _RealGemvPlan("narrow", bm=1, bk=4096 if lda <= 8 or not trans else 512)
    if not trans and out <= 64 and reduce >= 4096:
        return _RealGemvPlan(
            pool="bfgemv_n_long_reduce_ascend", fixed=True, head_tail=True, limit=8192
        )
    if trans and out <= 64 and reduce >= 4096:
        long_narrow = 32 <= out and reduce >= 32768 and lda <= 64
        return _lowp_narrow_plan(
            (out, reduce, lda),
            bm=2 if long_narrow else min(4, triton.next_power_of_2(out)),
            bk=1024 if long_narrow else (256 if lda >= 32 else 512),
        )
    if reduce <= 64 and out >= 4096:
        return _RealGemvPlan(
            pool="bfgemv_n64_full_ascend"
            if not trans and reduce == 64
            else "bfgemv_short_full_ascend",
            bk=triton.next_power_of_2(reduce),
            whole=True,
            dense=not trans and lda == reduce,
        )
    extended = (
        (min(m, n) <= 4 and max(m, n) >= 4096)
        or ((m % 2 or n % 2) and 5 <= min(m, n) <= 1023 and max(m, n) <= 4095)
        or (
            5 <= min(m, n) <= 64
            and max(m, n) >= max(4096, 16 * min(m, n))
            and size < 512 * 1024**2
        )
    )
    if extended:
        return _RealGemvPlan(
            pool=pool("head_tail"),
            fixed=True,
            head_tail=True,
            limit=8192,
            bm=max(16, triton.next_power_of_2(out) // 32) if trans else 0,
        )
    if (
        m % 512 == n % 512 == 0
        and 2048 <= min(m, n) <= 4096
        and 16 * 1024**2 <= size < 128 * 1024**2
        and ((not trans and m >= n) or (trans and n >= m))
    ):
        return _RealGemvPlan(pool=pool("block"), alias=True)
    if (
        m % 2 == n % 2 == 0
        and min(m, n) >= 1024
        and max(m, n) < 16 * min(m, n)
        and 64 * 1024**2 <= size < 1024**3
    ):
        return _RealGemvPlan(pool=pool("stream"), alias=True)
    if trans:
        if (
            m % 512 == n % 512 == 0
            and 1024 <= n <= 4096
            and m >= 4 * n
            and 8 * 1024**2 <= size < 64 * 1024**2
        ):
            return _RealGemvPlan(pool=pool("medium_skinny"))
        return _RealGemvPlan(
            pool=pool(
                "large"
                if min(m, n) >= 1024 and size >= 256 * 1024**2
                else "coalesced"
            )
        )
    if reduce <= 1024:
        return _RealGemvPlan(
            pool="bfgemv_n_small",
            bk=triton.next_power_of_2(reduce),
            whole=True,
            limit=8192,
        )
    return _RealGemvPlan(pool="bfgemv_n")


_BFGEMV_CACHE = {}
_BFGEMV_KERNELS = dict(
    block=_bfgemv_block_kernel, short=_bfgemv_short_kernel, narrow=_bfgemv_narrow_kernel
)


def bfgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Ascend BFGEMV; choose once and reuse a workspace-free launch."""
    assert trans in (0, 1) and m >= 0 and n >= 0 and lda >= n
    assert incx > 0 and incy > 0
    assert A.dtype == x.dtype == y.dtype == torch.bfloat16
    assert A.device == x.device == y.device and A.device.type == "npu"
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    if not m or not n:
        return
    out, reduce = (m, n) if trans == 0 else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= 1 + (reduce - 1) * incx
    assert y.numel() >= 1 + (out - 1) * incy
    if type(alpha) is not float:
        alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
    if type(beta) is not float:
        beta = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    device = A.device.index
    if device != current_device():
        with torch_device_fn.device(A.device):
            return bfgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    if alpha == 0 and beta == 1:
        return
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    scalars = (alpha, beta, out, reduce, lda, incx, incy, beta == 0)
    key = (device, trans, scalars, tuple(p % 16 for p in pointers))
    saved = _BFGEMV_CACHE.get(key)
    args = (A, x, y, *scalars)
    scale_args = (y, beta, out, incy, beta == 0)
    if saved is None:
        if len(_BFGEMV_CACHE) >= 512:
            _BFGEMV_CACHE.clear()
        if alpha == 0:
            launch = _compile_real_gemv_launch(
                _bfgemv_scale_kernel,
                _RealGemvPlan(bm=256),
                trans,
                scale_args,
                device,
                lambda meta: (triton.cdiv(out, meta["BM"]),),
            )
            _BFGEMV_CACHE[key] = (launch, None)
            return
        plan = _select_bfgemv_plan(trans, m, n, lda, incx, incy)
        scale = None
        if plan.splits != 1:
            scale = _compile_real_gemv_launch(
                _bfgemv_scale_kernel,
                _RealGemvPlan(bm=256),
                trans,
                scale_args,
                device,
                lambda meta: (triton.cdiv(out, meta["BM"]),),
            )
        main = _compile_real_gemv_launch(
            _BFGEMV_KERNELS[plan.kernel],
            plan,
            trans,
            args,
            device,
            lambda meta: (triton.cdiv(out, meta["BM"]), meta.get("SPLIT_K", 1)),
        )
        _BFGEMV_CACHE[key] = (scale, main)
        return
    scale, main = saved
    stream = current_stream(device)
    if scale is not None:
        scale(scale_args, stream, (pointers[2],))
    if main is not None:
        main(args, stream, pointers)


# CGEMV kernels and policy


@triton.jit
def _cgemv_block_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT,
    REDUCE,
    LDA,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    L2_BYPASS_OFFSET: tl.constexpr = 0,
    SPLIT_COUNT: tl.constexpr = 1,
    OUT_FIXED: tl.constexpr = 0,
    REDUCE_FIXED: tl.constexpr = 0,
    LDA_FIXED: tl.constexpr = 0,
):
    """Packed loads for N/T/C; split reductions accumulate directly into y."""
    out = OUT_FIXED if OUT_FIXED else OUT
    reduce = REDUCE_FIXED if REDUCE_FIXED else REDUCE
    lda = LDA_FIXED if LDA_FIXED else LDA
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(tl.float32))
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    ki = tl.arange(0, BLOCK_SIZE_K)
    if TRANSPOSE:
        acc_r = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    if SPLIT_COUNT > 1:
        chunk = tl.cdiv(reduce, SPLIT_COUNT * BLOCK_SIZE_K) * BLOCK_SIZE_K
        begin = tl.program_id(1) * chunk
        end = tl.minimum(begin + chunk, reduce)
    else:
        begin, end = 0, reduce
    for start in range(begin, end, BLOCK_SIZE_K):
        ks = start + ki
        xp = tl.load(
            x_ptr + ks[:, None] * 2 + tl.arange(0, 2)[None, :], ks[:, None] < end, 0
        )
        xr, xi = tl.split(xp)
        if CONJ and SPLIT_COUNT > 1:
            xi = -xi
        if TRANSPOSE:
            packed = tl.load(
                a_ptr + ks[:, None] * lda * 2 + cp[None, :],
                (ks[:, None] < end) & (cp[None, :] < out * 2),
                0,
            )
            avr, avi = tl.split(tl.reshape(packed, (BLOCK_SIZE_K, BLOCK_SIZE_M, 2)))
            if CONJ and SPLIT_COUNT == 1:
                avi = -avi
            acc_r += avr * xr[:, None] - avi * xi[:, None]
            acc_i += avr * xi[:, None] + avi * xr[:, None]
        else:
            kp = start * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
            packed = tl.load(
                a_ptr + outputs[:, None] * lda * 2 + kp[None, :],
                (outputs[:, None] < out) & (kp[None, :] < end * 2),
                0,
            )
            avr, avi = tl.split(tl.reshape(packed, (BLOCK_SIZE_M, BLOCK_SIZE_K, 2)))
            acc_r += avr * xr[None, :] - avi * xi[None, :]
            acc_i += avr * xi[None, :] + avi * xr[None, :]
    if TRANSPOSE:
        rr, ri = tl.sum(acc_r, 0), tl.sum(acc_i, 0)
    else:
        rr, ri = tl.sum(acc_r, 1), tl.sum(acc_i, 1)
    if CONJ and SPLIT_COUNT > 1:
        ri = -ri
    vr, vi = ar * rr - ai * ri, ar * ri + ai * rr
    if SPLIT_COUNT == 1:
        if not BETA_IS_ZERO:
            yr, yi = tl.split(
                tl.load(
                    y_ptr + outputs[:, None] * 2 + tl.arange(0, 2)[None, :],
                    outputs[:, None] < out,
                    0,
                )
            )
            vr += br * yr - bi * yi
            vi += br * yi + bi * yr
        tl.store(
            y_ptr + cp, tl.reshape(tl.join(vr, vi), (BLOCK_SIZE_M * 2,)), cp < out * 2
        )
    else:
        tl.atomic_add(
            y_ptr + cp,
            tl.reshape(tl.join(vr, vi), (BLOCK_SIZE_M * 2,)),
            cp < out * 2,
            sem="relaxed",
        )


@triton.jit
def _cgemv_short_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Unrolled K <= 4; dense three-column rows use one flat load."""
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    if not TRANSPOSE and REDUCE == 3:
        lanes = tl.arange(0, BLOCK_SIZE_M * triton.next_power_of_2(LDA * 2))
        begin = tl.program_id(0) * BLOCK_SIZE_M * LDA * 2
        limit = (OUT - 1) * LDA * 2 + REDUCE * 2
        raw = tl.load(
            a_ptr + begin + lanes,
            (lanes < BLOCK_SIZE_M * LDA * 2) & (begin + lanes < limit),
            0,
        )
        sr = tl.zeros((BLOCK_SIZE_M,), tl.float32)
        si = tl.zeros((BLOCK_SIZE_M,), tl.float32)
        for k in tl.static_range(3):
            avr = tl.gather(raw, tl.arange(0, BLOCK_SIZE_M) * LDA * 2 + k * 2, 0)
            avi = tl.gather(raw, tl.arange(0, BLOCK_SIZE_M) * LDA * 2 + k * 2 + 1, 0)
            xr, xi = tl.load(x_ptr + k * 2), tl.load(x_ptr + k * 2 + 1)
            sr += avr * xr - avi * xi
            si += avr * xi + avi * xr
        rr, ri = ar * sr - ai * si, ar * si + ai * sr
        if not BETA_IS_ZERO:
            yr, yi = tl.split(
                tl.load(
                    y_ptr + outputs[:, None] * 2 + tl.arange(0, 2)[None, :],
                    outputs[:, None] < OUT,
                    0,
                )
            )
            rr += br * yr - bi * yi
            ri += br * yi + bi * yr
        tl.store(
            y_ptr + cp, tl.reshape(tl.join(rr, ri), (BLOCK_SIZE_M * 2,)), cp < OUT * 2
        )
    else:
        sr = tl.zeros((BLOCK_SIZE_M // 8, 8), tl.float32)
        si = tl.zeros((BLOCK_SIZE_M // 8, 8), tl.float32)
        if not TRANSPOSE:
            lane = tl.arange(0, BLOCK_SIZE_M * REDUCE * 2)
            begin = tl.program_id(0) * BLOCK_SIZE_M * LDA * 2
            raw = tl.load(a_ptr + begin + lane, begin + lane < OUT * LDA * 2, 0)
            if REDUCE == 1:
                r0, i0 = tl.split(tl.reshape(raw, (BLOCK_SIZE_M // 8, 8, 2)))
            elif REDUCE == 2:
                vr, vi = tl.split(tl.reshape(raw, (BLOCK_SIZE_M // 8, 8, 2, 2)))
                r0, r1 = tl.split(vr)
                i0, i1 = tl.split(vi)
            else:
                vr, vi = tl.split(tl.reshape(raw, (BLOCK_SIZE_M // 8, 8, 2, 2, 2)))
                r02, r13 = tl.split(vr)
                i02, i13 = tl.split(vi)
                r0, r2 = tl.split(r02)
                r1, r3 = tl.split(r13)
                i0, i2 = tl.split(i02)
                i1, i3 = tl.split(i13)
            xr, xi = tl.load(x_ptr), tl.load(x_ptr + 1)
            sr += r0 * xr - i0 * xi
            si += r0 * xi + i0 * xr
            if REDUCE >= 2:
                xr, xi = tl.load(x_ptr + 2), tl.load(x_ptr + 3)
                sr += r1 * xr - i1 * xi
                si += r1 * xi + i1 * xr
            if REDUCE == 4:
                xr, xi = tl.load(x_ptr + 4), tl.load(x_ptr + 5)
                sr += r2 * xr - i2 * xi
                si += r2 * xi + i2 * xr
                xr, xi = tl.load(x_ptr + 6), tl.load(x_ptr + 7)
                sr += r3 * xr - i3 * xi
                si += r3 * xi + i3 * xr
        else:
            for k in tl.static_range(REDUCE):
                packed = tl.load(a_ptr + k * LDA * 2 + cp, cp < OUT * 2, 0)
                avr, avi = tl.split(tl.reshape(packed, (BLOCK_SIZE_M // 8, 8, 2)))
                if CONJ:
                    avi = -avi
                xr, xi = tl.load(x_ptr + k * 2), tl.load(x_ptr + k * 2 + 1)
                sr += avr * xr - avi * xi
                si += avr * xi + avi * xr
        rr, ri = ar * sr - ai * si, ar * si + ai * sr
        if not BETA_IS_ZERO:
            yr, yi = tl.split(
                tl.reshape(
                    tl.load(y_ptr + cp, cp < OUT * 2, 0), (BLOCK_SIZE_M // 8, 8, 2)
                )
            )
            rr += br * yr - bi * yi
            ri += br * yi + bi * yr
        tl.store(
            y_ptr + cp, tl.reshape(tl.join(rr, ri), (BLOCK_SIZE_M * 2,)), cp < OUT * 2
        )


@triton.jit
def _cgemv_narrow_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT: tl.constexpr,
    REDUCE: tl.constexpr,
    LDA: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
):
    """Contiguous loads for narrow T/C; no global reduction workspace."""
    output = tl.program_id(0)
    chunk = tl.cdiv(REDUCE, SPLIT_COUNT * BLOCK_SIZE_K) * BLOCK_SIZE_K
    begin = tl.program_id(1) * chunk
    end = tl.minimum(begin + chunk, REDUCE)
    lane = tl.arange(0, BLOCK_SIZE_K * triton.next_power_of_2(LDA * 2))
    ki = tl.arange(0, BLOCK_SIZE_K)
    sr = tl.zeros((BLOCK_SIZE_K,), tl.float32)
    si = tl.zeros((BLOCK_SIZE_K,), tl.float32)
    limit = (REDUCE - 1) * LDA * 2 + OUT * 2
    for start in range(begin, end, BLOCK_SIZE_K):
        offsets = start * LDA * 2 + lane
        if LDA == OUT:
            raw = tl.load(
                a_ptr + offsets,
                (lane < BLOCK_SIZE_K * LDA * 2) & (offsets < end * LDA * 2),
                0,
            )
        else:
            raw = tl.load(
                a_ptr + offsets,
                (lane < BLOCK_SIZE_K * LDA * 2)
                & (offsets < end * LDA * 2)
                & (offsets < limit),
                0,
            )
        avr = tl.gather(raw, ki * LDA * 2 + output * 2, 0)
        avi = tl.gather(raw, ki * LDA * 2 + output * 2 + 1, 0)
        if CONJ:
            avi = -avi
        xp = tl.load(
            x_ptr + (start + ki[:, None]) * 2 + tl.arange(0, 2)[None, :],
            start + ki[:, None] < end,
            0,
        )
        xr, xi = tl.split(xp)
        sr += avr * xr - avi * xi
        si += avr * xi + avi * xr
    rr, ri = tl.sum(sr, 0), tl.sum(si, 0)
    tl.atomic_add(y_ptr + output * 2, ar * rr - ai * ri, sem="relaxed")
    tl.atomic_add(y_ptr + output * 2 + 1, ar * ri + ai * rr, sem="relaxed")


@triton.jit
def _cgemv_scale_kernel(
    y_ptr,
    OUT,
    br: tl.float32,
    bi: tl.float32,
    ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    offsets = cp
    if ZERO:
        result = tl.zeros((BLOCK_SIZE_M * 2,), tl.float32)
    else:
        yr, yi = tl.split(
            tl.reshape(tl.load(y_ptr + offsets, cp < OUT * 2, 0), (BLOCK_SIZE_M, 2))
        )
        result = tl.reshape(
            tl.join(br * yr - bi * yi, br * yi + bi * yr), (BLOCK_SIZE_M * 2,)
        )
    tl.store(y_ptr + offsets, result, cp < OUT * 2)


@triton.jit
def _cgemv_strided_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    INCX,
    INCY,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    k_init = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k_start in range(0, n, BLOCK_SIZE_K):
        ks = k_start + k_init
        k_mask = ks < n
        mask = row_mask[:, None] & k_mask[None, :]
        a_elem = rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AN
        a_off = a_elem * 2
        x_off = ks * INCX * 2
        a_real = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        a_imag = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        x_real = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + x_off + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]
    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)
    out_real = alpha_real * acc_real - alpha_imag * acc_imag
    out_imag = alpha_real * acc_imag + alpha_imag * acc_real
    y_off = rows * INCY * 2
    if not BETA_IS_ZERO:
        y_real = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        out_real += beta_real * y_real - beta_imag * y_imag
        out_imag += beta_real * y_imag + beta_imag * y_real
    tl.store(y_ptr + y_off, out_real, mask=row_mask)
    tl.store(y_ptr + y_off + 1, out_imag, mask=row_mask)


class _CgemvPlan(NamedTuple):
    kernel: str = "block"
    pool: str = ""
    bm: int = 0
    bk: int = 0
    splits: int = 1
    bypass_l2: bool = False
    static: bool = False
    stages: int = 1


def _select_cgemv_base_plan(trans, m, n, lda, incx, incy):
    """Priority: strides, short K, narrow output, streaming, regular tiles."""
    if incx != 1 or incy != 1 or min(m, n) < 1:
        return _CgemvPlan("strided", "cgemv_ascend")
    out, reduce = (m, n) if trans == 0 else (n, m)
    size = m * n * 8
    if reduce <= 4 and out >= 4096:
        if trans != 0 or lda == n:
            bm = (
                2048
                if trans == 0 and reduce == 3
                else (1024 if trans != 0 or reduce == 1 else 512)
            )
            return _CgemvPlan("short", bm=bm, bk=triton.next_power_of_2(reduce))
    if trans != 0 and out <= 4 and reduce >= 4096 and lda <= 8:
        return _CgemvPlan("narrow", bm=1, bk=1024, splits=16 if out in (1, 4) else 32)
    if out <= 64 and reduce >= 4096 and size < 256 * 1024**2:
        if trans == 0 and out <= 2 and reduce <= 65536:
            return _CgemvPlan(bm=1, bk=4096, static=True, bypass_l2=True)
        if trans == 0 and 32 <= out and 32768 <= reduce <= 65536:
            return _CgemvPlan(
                bm=16, bk=256, splits=8, stages=2, static=True, bypass_l2=True
            )
        if trans != 0 and out == 64:
            if reduce >= 32768:
                return _CgemvPlan(
                    bm=64, bk=64, splits=64, stages=2, static=True, bypass_l2=True
                )
            return _CgemvPlan(bm=32, bk=128, splits=8, static=True)
        return _CgemvPlan(
            bm=min(16, triton.next_power_of_2(out)),
            bk=256 if trans == 0 else 128,
            splits=32,
            static=True,
        )
    if min(m, n) >= 1024 and size >= 256 * 1024**2:
        return _CgemvPlan(
            pool="cgemv_n_large_ascend"
            if trans == 0
            else "cgemv_t_packed_split_ascend",
            splits=0 if trans else 1,
            bypass_l2=True,
        )
    if m == 1024 and n == 4096:
        return _CgemvPlan(
            bm=32 if trans == 0 else 128,
            bk=128 if trans == 0 else 32,
            bypass_l2=True,
            static=True,
        )
    if trans == 0:
        if 1023 <= m <= 4096 and 2048 <= n <= 4096:
            return _CgemvPlan(bm=8, bk=512, stages=2, bypass_l2=True, static=True)
        if 512 < m == n <= 1024:
            return _CgemvPlan(bm=32, bk=64, stages=2)
        if 256 < m == n <= 512:
            return _CgemvPlan(bm=16, bk=128)
        if 2048 <= m <= 4096 and 512 < n <= 1024:
            return _CgemvPlan(
                bm=16 if n % 2 else 32,
                bk=256 if n % 2 else 128,
                stages=2,
                bypass_l2=True,
                static=True,
            )
    else:
        if 32 <= m == n <= 128:
            return _CgemvPlan(bm=16, bk=64)
        if 128 < m == n <= 256:
            return _CgemvPlan(bm=16 if m % 2 else 32, bk=64)
        # Medium transpose tiles need enough output blocks to fill the device;
        # very wide tiles can win the tuning probe but serialize the hot path.
        # An unaligned eight-tile reduction benefits from more output columns
        # per block and split reduction; aligned rows keep the single-launch path.
        if 896 < m == n <= 1024 and m % 2 and lda % 2:
            return _CgemvPlan(
                bm=512, bk=8, splits=32, stages=2, bypass_l2=True, static=True
            )
        if 256 < m == n <= 1024:
            return _CgemvPlan(bm=32, bk=128)
        if (
            (m == n and 3072 <= m <= 4096)
            or (1023 <= n <= 1024 and 2 * n <= m <= 4096)
            or (m == 1023 and n == 4095)
        ):
            if max(m, n) < 3072:
                return _CgemvPlan(
                    bm=256, bk=16, splits=16, stages=2, bypass_l2=True, static=True
                )
            return _CgemvPlan(
                bm=512, bk=8, splits=32, stages=2, bypass_l2=True, static=True
            )
    if min(m, n) <= 64 and max(m, n) >= 4096 and size < 256 * 1024**2:
        if reduce <= 64:
            return _CgemvPlan(
                bm=128 if trans == 0 and reduce == 64 and out >= 16384 else 64,
                bk=triton.next_power_of_2(reduce),
                static=True,
                bypass_l2=trans == 0 and reduce == 64 and out >= 16384,
            )
        return _CgemvPlan(
            bm=16 if trans == 0 else 64, bk=128 if trans == 0 else 32, static=True
        )
    if min(m, n) < 32:
        return _CgemvPlan("strided", "cgemv_ascend")
    return _CgemvPlan(
        pool="cgemv_n_packed_ascend" if trans == 0 else "cgemv_t_packed_ascend"
    )


_CGEMV_CACHE = {}
_CGEMV_RAW = dict(
    block=_cgemv_block_kernel,
    short=_cgemv_short_kernel,
    narrow=_cgemv_narrow_kernel,
    strided=_cgemv_strided_kernel,
    scale=_cgemv_scale_kernel,
)
_CGEMV_DEFAULTS = dict(
    L2_BYPASS_OFFSET=0, SPLIT_COUNT=1, OUT_FIXED=0, REDUCE_FIXED=0, LDA_FIXED=0
)


class _CgemvPointer(int):
    def size(self):
        return (*self.tensor.size(), 2)


class _CgemvLaunch(NamedTuple):
    compiled: object
    run: object
    grid: tuple
    constants: tuple
    tensor_count: int
    special: bool
    strided: bool
    scalars: tuple = ()

    def __call__(self, tensors, stream, pointers=None):
        enter = triton.knobs.runtime.launch_enter_hook
        leave = triton.knobs.runtime.launch_exit_hook
        hooked = (
            enter is not None
            and (type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls))
        ) or (
            leave is not None
            and (type(leave) is not _HOOK_CHAIN_TYPE or bool(leave.calls))
        )
        if self.special or hooked or type(self.run) is not NPULauncher:
            actual = tuple(
                torch.view_as_real(t)
                if isinstance(t, torch.Tensor) and t.dtype == torch.complex64
                else t
                for t in tensors
            )
            self.compiled[self.grid](
                *actual, *self.scalars, *self.constants, stream=stream
            )
            return
        if pointers is None:
            pointers = tuple(t.data_ptr() for t in tensors)
        if self.tensor_count == 3:
            pa, px, py = (
                _CgemvPointer(pointers[0]),
                _CgemvPointer(pointers[1]),
                _CgemvPointer(pointers[2]),
            )
            pa.tensor, px.tensor, py.tensor = tensors
            packed = (pa, px, py)
        else:
            py = _CgemvPointer(pointers[0])
            py.tensor = tensors[0]
            packed = (py,)
        registered = self.run.launch(
            *self.grid,
            stream,
            self.compiled.function,
            self.compiled.packed_metadata,
            None,
            None,
            None,
            *packed,
            *self.scalars,
            *self.constants,
        )
        driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


@lru_cache(maxsize=None)
def _cgemv_entry(kernel, pool, offset):
    raw = _CGEMV_RAW[kernel]
    if not pool:
        return libentry()(raw)
    configs = []
    for cfg in runtime.get_tuned_config(pool):
        meta = dict(cfg.kwargs)
        if kernel == "block":
            meta = {**_CGEMV_DEFAULTS, **meta, "L2_BYPASS_OFFSET": offset}
        configs.append(
            triton.Config(
                meta,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                num_ctas=cfg.num_ctas,
            )
        )
    keys = (
        ["m", "n", "STRIDE_AM", "STRIDE_AN", "INCX", "INCY", "CONJ", "BETA_IS_ZERO"]
        if kernel == "strided"
        else ["OUT", "REDUCE", "LDA", "TRANSPOSE", "CONJ", "BETA_IS_ZERO"]
    )
    return libentry()(libtuner(configs=configs, key=keys, restore_value=["y_ptr"])(raw))


def _cgemv_compile(entry, args, grid, meta):
    real_args = tuple(
        torch.view_as_real(t)
        if isinstance(t, torch.Tensor) and t.dtype == torch.complex64
        else t
        for t in args
    )
    compiled, tuned = entry[grid](*real_args, **meta)
    raw = entry.jit_function
    info = _gemv_launch_metadata(
        compiled, raw, args, {**_CGEMV_DEFAULTS, **tuned, **meta}, grid
    )
    return _CgemvLaunch(
        compiled, compiled.run, *info, raw is _cgemv_strided_kernel, args[info[2] :]
    )


def _cgemv_scale(y, out, br, bi, key, stream):
    args = (y, out, br, bi, br == bi == 0)
    saved = _CGEMV_CACHE.get(key)
    if saved is None:
        saved = _cgemv_compile(
            _cgemv_entry("scale", "", 0),
            args,
            lambda meta: (triton.cdiv(out, meta["BLOCK_SIZE_M"]),),
            dict(BLOCK_SIZE_M=256, num_warps=1, num_stages=1),
        )
        _CGEMV_CACHE[key] = saved
    else:
        saved((y,), stream)


def _cgemv_strided(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Keep the existing gather/scatter fallback for non-unit vector strides."""
    out, reduce = (m, n) if trans == 0 else (n, m)
    logical_y = torch.view_as_real(y).as_strided((out, 2), (incy * 2, 1))
    if alpha == 0:
        if beta == 0:
            logical_y.zero_()
        elif beta != 1:
            old = logical_y.clone()
            logical_y[:, 0].copy_(beta.real * old[:, 0] - beta.imag * old[:, 1])
            logical_y[:, 1].copy_(beta.real * old[:, 1] + beta.imag * old[:, 0])
        return
    real_x = torch.view_as_real(x)
    if incx != 1:
        real_x = real_x.as_strided((reduce, 2), (incx * 2, 1)).clone()
    real_y = logical_y.clone() if incy != 1 else logical_y
    cgemv(
        trans,
        m,
        n,
        alpha,
        A,
        lda,
        torch.view_as_complex(real_x),
        1,
        beta,
        torch.view_as_complex(real_y),
        1,
    )
    if incy != 1:
        logical_y.copy_(real_y)


def cgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    assert trans in (0, 1, 2) and m >= 0 and n >= 0
    assert A.dtype == x.dtype == y.dtype == torch.complex64
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    tensor_device = A.device
    assert tensor_device == x.device == y.device and tensor_device.type == "npu"
    assert incx > 0 and incy > 0 and lda >= n
    if m == 0 or n == 0:
        return
    out, reduce = (m, n) if trans == 0 else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= (reduce - 1) * incx + 1 and y.numel() >= (out - 1) * incy + 1
    device = tensor_device.index
    if device != current_device():
        with torch_device_fn.device(tensor_device):
            return cgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    if type(alpha) not in (complex, float, int):
        alpha = alpha.item() if isinstance(alpha, torch.Tensor) else complex(alpha)
    if type(beta) not in (complex, float, int):
        beta = beta.item() if isinstance(beta, torch.Tensor) else complex(beta)
    if incx != 1 or incy != 1:
        return _cgemv_strided(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    alignment = (pointers[0] % 16, pointers[1] % 16, pointers[2] % 16)
    stream = current_stream(device)
    if alpha == 0:
        br, bi = float(beta.real), float(beta.imag)
        if br == 1 and bi == 0:
            return
        key = ("scale", device, out, br, bi, incy, alignment[2])
        _cgemv_scale(y, out, br, bi, key, stream)
        return
    key = (device, alpha, beta, out, reduce, lda, trans, alignment)
    saved = _CGEMV_CACHE.get(key)
    if saved is not None:
        scale, main = saved
        if scale is not None:
            scale((y,), stream, (pointers[2],))
        main((A, x, y), stream, pointers)
        return
    ar, ai, br, bi = (
        float(alpha.real),
        float(alpha.imag),
        float(beta.real),
        float(beta.imag),
    )
    scalars = (ar, ai, br, bi, out, reduce, lda, trans != 0, trans == 2, br == bi == 0)
    plan = _select_cgemv_plan(trans, m, n, lda, incx, incy)
    if plan.kernel == "strided":
        am, an = (lda, 1) if trans == 0 else (1, lda)
        args = (A, x, y, ar, ai, br, bi, out, reduce, am, an, incx, incy)
    else:
        args = (A, x, y, *scalars)
    offset = _gemv_l2_offset(device) if plan.bypass_l2 else 0
    entry = _cgemv_entry(plan.kernel, plan.pool, offset)
    meta = {}
    if not plan.pool:
        meta = dict(
            BLOCK_SIZE_M=plan.bm,
            BLOCK_SIZE_K=plan.bk,
            num_warps=1,
            num_stages=plan.stages,
        )
        if plan.kernel in ("block", "prefetch", "prefetch_four"):
            meta.update(
                _cgemv_kernel_defaults(plan.kernel),
                L2_BYPASS_OFFSET=offset,
                SPLIT_COUNT=plan.splits,
            )
            if plan.static:
                meta.update(OUT_FIXED=out, REDUCE_FIXED=reduce, LDA_FIXED=lda)
        elif plan.kernel == "narrow":
            meta["SPLIT_COUNT"] = plan.splits
    if plan.kernel == "strided":
        meta.update(CONJ=trans == 2, BETA_IS_ZERO=br == bi == 0)
    grid = lambda cfg: (
        out if plan.kernel == "narrow" else triton.cdiv(out, cfg["BLOCK_SIZE_M"]),
        cfg.get("SPLIT_COUNT", 1),
    )
    # Fixed specializations are compiled without launching first, so a compiler
    # workspace is rejected before beta scaling or an output write.
    if not plan.pool:
        real_args = tuple(torch.view_as_real(t) for t in (A, x, y)) + args[3:]
        compiled = _CGEMV_RAW[plan.kernel].run(
            *real_args, grid=grid(meta), warmup=True, **meta
        )
        assert int(getattr(compiled.metadata, "workspace_size", 0)) == 0
    scale = None
    if plan.splits != 1:
        scale = _cgemv_compile(
            _cgemv_entry("scale", "", 0),
            (y, out, br, bi, br == bi == 0),
            lambda cfg: (triton.cdiv(out, cfg["BLOCK_SIZE_M"]),),
            dict(BLOCK_SIZE_M=256, num_warps=1, num_stages=1),
        )
    main = _cgemv_compile(entry, args, grid, meta)
    if len(_CGEMV_CACHE) >= 512:
        _CGEMV_CACHE.clear()
    _CGEMV_CACHE[key] = (scale, main)


def _lowp_narrow_plan(shape, bm, bk):
    """Pipeline contiguous 128-byte rows while keeping FP32 accumulation."""
    out, reduce, lda = shape
    if lda == out == 64 and 65536 <= reduce <= 262144:
        return _RealGemvPlan("narrow", bm=4, bk=512, stages=2)
    return _RealGemvPlan("narrow", bm=bm, bk=bk)


@lru_cache(maxsize=None)
def _gemv_vector_cores(device):
    try:
        return max(1, int(triton.runtime.driver.active.utils.get_aivector_core_num()))
    except (AttributeError, RuntimeError):
        return 32


def _gemv_launch_entry_grid(raw, plan, trans, out, reduce, lda, offset, grid):
    if raw is _lowp_whole_rows_kernel:
        cores = _gemv_vector_cores(current_device())
        grid = lambda meta: (min(cores, triton.cdiv(out, meta["BM"])),)
    return _real_gemv_entry(raw, plan, trans, out, reduce, lda, offset), grid


@triton.jit
def _lowp_whole_rows_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    OUT,
    REDUCE,
    LDA,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    L2_OFFSET: tl.constexpr = 0,
    OUT_FIXED: tl.constexpr = 0,
    REDUCE_FIXED: tl.constexpr = 0,
    LDA_FIXED: tl.constexpr = 0,
):
    """Reuse x in a vector core while reducing complete rows in FP32."""
    out = OUT_FIXED if OUT_FIXED else OUT
    reduce = REDUCE_FIXED if REDUCE_FIXED else REDUCE
    lda = LDA_FIXED if LDA_FIXED else LDA
    if L2_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_OFFSET).to(
            tl.pointer_type(a_ptr.dtype.element_ty)
        )
    ks = tl.arange(0, BK)
    xv = tl.load(x_ptr + ks * INCX, ks < reduce, 0).to(tl.float32)
    for tile in range(tl.program_id(0), tl.cdiv(out, BM), tl.num_programs(0)):
        outputs = tile * BM + tl.arange(0, BM)
        av = tl.load(
            a_ptr + outputs[:, None] * lda + ks[None, :],
            (outputs[:, None] < out) & (ks[None, :] < reduce),
            0,
        ).to(tl.float32)
        value = alpha * tl.sum(av * xv[None, :], 1)
        if not BETA_ZERO:
            value += beta * tl.load(y_ptr + outputs * INCY, outputs < out, 0).to(
                tl.float32
            )
        tl.store(y_ptr + outputs * INCY, value, outputs < out)


_HGEMV_KERNELS["whole_rows"] = _lowp_whole_rows_kernel
_BFGEMV_KERNELS["whole_rows"] = _lowp_whole_rows_kernel


def _lowp_whole_rows_plan(trans, m, n, lda, incx, incy):
    if (
        not trans
        and incx == incy == 1
        and lda == n
        and m % 2
        and n % 2
        and 512 <= m <= 16384
        and 2048 < n <= 4096
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _RealGemvPlan(
            "whole_rows", bm=4, bk=triton.next_power_of_2(n), alias=True, fixed=True
        )
    return None


def _select_hgemv_regular_plan(trans, m, n, lda, incx=1, incy=1):
    if (
        not trans
        and incx == incy == 1
        and lda == n
        and m % 512 == n % 512 == 0
        and 2048 <= m <= 8192
        and 2 * m <= n <= 8 * m
        and n <= 32768
        and 64 * 1024**2 <= m * n * 2 <= 512 * 1024**2
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _RealGemvPlan(
            bm=16, bk=512, stages=2, fixed=True, alias=True, full_tiles=True
        )
    if (
        trans
        and incx == incy == 1
        and lda == n
        and m % 2
        and n % 2
        and 512 <= n <= 1023
        and 3 * n <= m <= 8 * n
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _RealGemvPlan(bm=32, bk=128, stages=1, fixed=True)
    return _lowp_whole_rows_plan(
        trans, m, n, lda, incx, incy
    ) or _select_hgemv_base_plan(trans, m, n, lda, incx, incy)


def _select_bfgemv_regular_plan(trans, m, n, lda, incx=1, incy=1):
    return _lowp_whole_rows_plan(
        trans, m, n, lda, incx, incy
    ) or _select_bfgemv_base_plan(trans, m, n, lda, incx, incy)


def _select_cgemv_plan(trans, m, n, lda, incx, incy):
    if (
        trans in (1, 2)
        and incx == incy == 1
        and lda == n
        and m % 2
        and n % 2
        and 768 <= n <= 1023
        and 2 * n <= m <= 4 * (n + 1)
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _CgemvPlan(
            kernel="prefetch",
            bm=512,
            bk=8,
            splits=20,
            stages=2,
            static=True,
            bypass_l2=True,
        )

    if (
        trans in (1, 2)
        and incx == incy == 1
        and lda == n
        and m % 2
        and n % 2
        and min(m, n) >= 4095
        and max(m, n) <= 16384
        and _gemv_vector_cores(current_device()) == 40
    ):
        plan = _cgemv_odd_transpose_plan(n, m)
        if triton.cdiv(m, plan.splits) >= 192:
            return plan._replace(kernel="prefetch_four")
        return plan
    return _select_cgemv_base_plan(trans, m, n, lda, incx, incy)


def _cgemv_odd_transpose_plan(out, reduce):
    """Use complete core waves and avoid long per-program reductions."""
    cores = _gemv_vector_cores(current_device())
    tiles = triton.cdiv(out, 1024)
    a, b = cores, tiles
    while b:
        a, b = b, a % b
    quantum = cores // a
    needed = max(triton.cdiv(reduce, 256), triton.cdiv(4 * cores, tiles))
    splits = min(cores, triton.cdiv(needed, quantum) * quantum)
    return _CgemvPlan(
        bm=1024, bk=4, splits=splits, stages=2, static=True, bypass_l2=True
    )


def _gemv_row_splits():
    return 40 if _gemv_vector_cores(current_device()) == 40 else 32


# Complete-row reduction also avoids repeated x loads in FP32 GEMV.
_SGEMV_KERNELS["whole_rows"] = _lowp_whole_rows_kernel


def _select_sgemv_plan(trans, m, n, lda, incx=1, incy=1):
    if (
        not trans
        and incx == incy == 1
        and lda == n
        and m % 2
        and n % 2
        and 512 <= m <= 16384
        and 2048 < n <= 4096
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _RealGemvPlan(
            "whole_rows", bm=4, bk=triton.next_power_of_2(n), fixed=True, alias=True
        )
    return _select_sgemv_base_plan(trans, m, n, lda, incx, incy)


def _cgemv_kernel_defaults(kernel):
    if kernel in ("prefetch", "prefetch_four"):
        return {**_CGEMV_DEFAULTS, "multibuffer": False}
    return _CGEMV_DEFAULTS


@triton.jit
def _cgemv_prefetch_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT,
    REDUCE,
    LDA,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    L2_BYPASS_OFFSET: tl.constexpr = 0,
    SPLIT_COUNT: tl.constexpr = 1,
    OUT_FIXED: tl.constexpr = 0,
    REDUCE_FIXED: tl.constexpr = 0,
    LDA_FIXED: tl.constexpr = 0,
):
    """Packed loads for N/T/C; split reductions accumulate directly into y."""
    out = OUT_FIXED if OUT_FIXED else OUT
    reduce = REDUCE_FIXED if REDUCE_FIXED else REDUCE
    lda = LDA_FIXED if LDA_FIXED else LDA
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(tl.float32))
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    ki = tl.arange(0, BLOCK_SIZE_K)
    if TRANSPOSE:
        acc_r = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    if SPLIT_COUNT > 1:
        chunk = tl.cdiv(reduce, SPLIT_COUNT * BLOCK_SIZE_K) * BLOCK_SIZE_K
        begin = tl.program_id(1) * chunk
        end = tl.minimum(begin + chunk, reduce)
    else:
        begin, end = 0, reduce
    ks = begin + ki
    xp = tl.load(
        x_ptr + ks[:, None] * 2 + tl.arange(0, 2)[None, :], ks[:, None] < end, 0
    )
    if TRANSPOSE:
        packed = tl.load(
            a_ptr + ks[:, None] * lda * 2 + cp[None, :],
            (ks[:, None] < end) & (cp[None, :] < out * 2),
            0,
        )
    else:
        kp = begin * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
        packed = tl.load(
            a_ptr + outputs[:, None] * lda * 2 + kp[None, :],
            (outputs[:, None] < out) & (kp[None, :] < end * 2),
            0,
        )
    for start in range(begin, end, BLOCK_SIZE_K):
        nks = start + BLOCK_SIZE_K + ki
        next_xp = tl.load(
            x_ptr + nks[:, None] * 2 + tl.arange(0, 2)[None, :], nks[:, None] < end, 0
        )
        if TRANSPOSE:
            next_packed = tl.load(
                a_ptr + nks[:, None] * lda * 2 + cp[None, :],
                (nks[:, None] < end) & (cp[None, :] < out * 2),
                0,
            )
        else:
            nkp = (start + BLOCK_SIZE_K) * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
            next_packed = tl.load(
                a_ptr + outputs[:, None] * lda * 2 + nkp[None, :],
                (outputs[:, None] < out) & (nkp[None, :] < end * 2),
                0,
            )
        xr, xi = tl.split(xp)
        if CONJ and SPLIT_COUNT > 1:
            xi = -xi
        if TRANSPOSE:
            avr, avi = tl.split(tl.reshape(packed, (BLOCK_SIZE_K, BLOCK_SIZE_M, 2)))
            if CONJ and SPLIT_COUNT == 1:
                avi = -avi
            acc_r += avr * xr[:, None] - avi * xi[:, None]
            acc_i += avr * xi[:, None] + avi * xr[:, None]
        else:
            avr, avi = tl.split(tl.reshape(packed, (BLOCK_SIZE_M, BLOCK_SIZE_K, 2)))
            acc_r += avr * xr[None, :] - avi * xi[None, :]
            acc_i += avr * xi[None, :] + avi * xr[None, :]
        xp = next_xp
        packed = next_packed
    if TRANSPOSE:
        rr, ri = tl.sum(acc_r, 0), tl.sum(acc_i, 0)
    else:
        rr, ri = tl.sum(acc_r, 1), tl.sum(acc_i, 1)
    if CONJ and SPLIT_COUNT > 1:
        ri = -ri
    vr, vi = ar * rr - ai * ri, ar * ri + ai * rr
    if SPLIT_COUNT == 1:
        if not BETA_IS_ZERO:
            yr, yi = tl.split(
                tl.load(
                    y_ptr + outputs[:, None] * 2 + tl.arange(0, 2)[None, :],
                    outputs[:, None] < out,
                    0,
                )
            )
            vr += br * yr - bi * yi
            vi += br * yi + bi * yr
        tl.store(
            y_ptr + cp, tl.reshape(tl.join(vr, vi), (BLOCK_SIZE_M * 2,)), cp < out * 2
        )
    else:
        tl.atomic_add(
            y_ptr + cp,
            tl.reshape(tl.join(vr, vi), (BLOCK_SIZE_M * 2,)),
            cp < out * 2,
            sem="relaxed",
        )


_CGEMV_RAW["prefetch"] = _cgemv_prefetch_kernel


def _lowp_compile_options(raw, plan, trans):
    if (
        trans
        and plan.reduce_each
        and raw in (_hgemv_block_kernel, _bfgemv_block_kernel)
    ):
        return {"multibuffer": False}
    return {}


def _lowp_aligned_transpose_plan(trans, m, n, lda, incx, incy):
    if (
        trans
        and incx == incy == 1
        and lda == n
        and m % 512 == n % 512 == 0
        and 16384 <= n <= 18432
        and n // 2 <= m <= 65536
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _RealGemvPlan(
            bm=512,
            bk=32,
            stages=2,
            fixed=True,
            alias=True,
            full_tiles=True,
            reduce_each=True,
        )
    return None


def _select_hgemv_plan(trans, m, n, lda, incx=1, incy=1):
    return _lowp_transpose_plan(
        trans, m, n, lda, incx, incy
    ) or _select_hgemv_regular_plan(trans, m, n, lda, incx, incy)


def _select_bfgemv_plan(trans, m, n, lda, incx=1, incy=1):
    return _bfgemv_transpose_plan(
        trans, m, n, lda, incx, incy
    ) or _select_bfgemv_regular_plan(trans, m, n, lda, incx, incy)


@triton.jit
def _cgemv_four_prefetch_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT,
    REDUCE,
    LDA,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    L2_BYPASS_OFFSET: tl.constexpr = 0,
    SPLIT_COUNT: tl.constexpr = 1,
    OUT_FIXED: tl.constexpr = 0,
    REDUCE_FIXED: tl.constexpr = 0,
    LDA_FIXED: tl.constexpr = 0,
):
    out = OUT_FIXED if OUT_FIXED else OUT
    reduce = REDUCE_FIXED if REDUCE_FIXED else REDUCE
    lda = LDA_FIXED if LDA_FIXED else LDA
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(tl.float32))
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    ki = tl.arange(0, BLOCK_SIZE_K)
    if TRANSPOSE:
        p = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        q = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        r = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        s = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        p = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        r = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        s = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    chunk = tl.cdiv(reduce, SPLIT_COUNT * BLOCK_SIZE_K) * BLOCK_SIZE_K
    begin = tl.program_id(1) * chunk
    end = tl.minimum(begin + chunk, reduce)
    ks = begin + ki
    xp = tl.load(
        x_ptr + ks[:, None] * 2 + tl.arange(0, 2)[None, :], ks[:, None] < end, 0
    )
    if TRANSPOSE:
        packed = tl.load(
            a_ptr + ks[:, None] * lda * 2 + cp[None, :],
            (ks[:, None] < end) & (cp[None, :] < out * 2),
            0,
        )
    else:
        kp = begin * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
        packed = tl.load(
            a_ptr + outputs[:, None] * lda * 2 + kp[None, :],
            (outputs[:, None] < out) & (kp[None, :] < end * 2),
            0,
        )
    for start in range(begin, end, BLOCK_SIZE_K):
        nks = start + BLOCK_SIZE_K + ki
        next_xp = tl.load(
            x_ptr + nks[:, None] * 2 + tl.arange(0, 2)[None, :], nks[:, None] < end, 0
        )
        if TRANSPOSE:
            next_packed = tl.load(
                a_ptr + nks[:, None] * lda * 2 + cp[None, :],
                (nks[:, None] < end) & (cp[None, :] < out * 2),
                0,
            )
        else:
            nkp = (start + BLOCK_SIZE_K) * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
            next_packed = tl.load(
                a_ptr + outputs[:, None] * lda * 2 + nkp[None, :],
                (outputs[:, None] < out) & (nkp[None, :] < end * 2),
                0,
            )
        xr, xi = tl.split(xp)
        if CONJ:
            xi = -xi
        if TRANSPOSE:
            avr, avi = tl.split(tl.reshape(packed, (BLOCK_SIZE_K, BLOCK_SIZE_M, 2)))
            p = tl.fma(avr, xr[:, None], p)
            q = tl.fma(avi, xi[:, None], q)
            r = tl.fma(avr, xi[:, None], r)
            s = tl.fma(avi, xr[:, None], s)
        else:
            avr, avi = tl.split(tl.reshape(packed, (BLOCK_SIZE_M, BLOCK_SIZE_K, 2)))
            p = tl.fma(avr, xr[None, :], p)
            q = tl.fma(avi, xi[None, :], q)
            r = tl.fma(avr, xi[None, :], r)
            s = tl.fma(avi, xr[None, :], s)
        xp = next_xp
        packed = next_packed
    if TRANSPOSE:
        pp, qq, rrr, sss = tl.sum(p, 0), tl.sum(q, 0), tl.sum(r, 0), tl.sum(s, 0)
    else:
        pp, qq, rrr, sss = tl.sum(p, 1), tl.sum(q, 1), tl.sum(r, 1), tl.sum(s, 1)
    rr, ri = pp - qq, rrr + sss
    if CONJ:
        ri = -ri
    vr, vi = ar * rr - ai * ri, ar * ri + ai * rr
    if SPLIT_COUNT == 1:
        if not BETA_IS_ZERO:
            yr, yi = tl.split(
                tl.load(
                    y_ptr + outputs[:, None] * 2 + tl.arange(0, 2)[None, :],
                    outputs[:, None] < out,
                    0,
                )
            )
            vr += br * yr - bi * yi
            vi += br * yi + bi * yr
        tl.store(
            y_ptr + cp, tl.reshape(tl.join(vr, vi), (BLOCK_SIZE_M * 2,)), cp < out * 2
        )
    else:
        tl.atomic_add(
            y_ptr + cp,
            tl.reshape(tl.join(vr, vi), (BLOCK_SIZE_M * 2,)),
            cp < out * 2,
            sem="relaxed",
        )


_CGEMV_RAW["prefetch_four"] = _cgemv_four_prefetch_kernel


def _lowp_transpose_plan(trans, m, n, lda, incx, incy):
    if (
        trans
        and incx == incy == 1
        and lda == n
        and m % 512 == n % 512 == 0
        and 1024 <= m <= 2048
        and 4096 <= n <= 8192
        and 2 * m < n <= 8 * m
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _RealGemvPlan(
            bm=128, bk=64, stages=2, fixed=True, alias=True, full_tiles=True
        )
    return _lowp_aligned_transpose_plan(trans, m, n, lda, incx, incy)


def _bfgemv_transpose_plan(trans, m, n, lda, incx, incy):
    tail_blocks = (n // 512) % 40
    if (
        trans
        and incx == incy == 1
        and lda == n
        and m % 512 == n % 512 == 0
        and 16384 <= m <= 32768
        and 49152 <= n <= 65536
        and 3 * m <= n <= 4 * m
        and (tail_blocks == 0 or tail_blocks >= 24)
        and _gemv_vector_cores(current_device()) == 40
    ):
        return _RealGemvPlan(
            bm=512,
            bk=32,
            stages=2,
            fixed=True,
            alias=True,
            full_tiles=True,
            reduce_each=True,
        )
    return _lowp_transpose_plan(trans, m, n, lda, incx, incy)

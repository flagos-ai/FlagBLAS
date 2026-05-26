import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT, CUBLAS_OP_C, CUBLAS_OP_N
from flag_blas.ops.level2.tpmv import (
    _check_tpmv,
    _ctpmv_split_k,
    _mode_key,
    ctpmv_reduce_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_TPMV_KEY = ["n", "mode_key"]
_TPMV_RESTORE = ["x_ptr"]


def _prune_tpmv_configs(configs, named_args, **kwargs):
    n = named_args["n"]
    configs = [
        config
        for config in configs
        if config.kwargs["BLOCK_K"] <= config.kwargs["BLOCK_SIZE_M"]
    ]
    if n >= 512:
        return configs
    return [
        config
        for config in configs
        if config.num_stages <= 2
        and config.kwargs["BLOCK_SIZE_M"] <= 256
        and config.kwargs["BLOCK_K"] <= 64
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctpmv"),
    key=_TPMV_KEY,
    restore_value=_TPMV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def ctpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    pair = tl.arange(0, 2)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off3 = off[:, :, None] * 2 + pair[None, None, :]
            x_off2 = j[:, None] * 2 + pair[None, :]
            a_pair = tl.load(
                ap_ptr + a_off3,
                mask=mask[:, :, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            x_pair = tl.load(
                xin_ptr + x_off2,
                mask=j_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            ar, ai = tl.split(a_pair)
            xr, xi = tl.split(x_pair)
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_off3 = off[:, :, None] * 2 + pair[None, None, :]
        x_off2 = j[:, None] * 2 + pair[None, :]
        a_pair = tl.load(
            ap_ptr + a_off3,
            mask=mask[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        x_pair = tl.load(
            xin_ptr + x_off2,
            mask=j_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        ar, ai = tl.split(a_pair)
        xr, xi = tl.split(x_pair)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off3 = off[:, :, None] * 2 + pair[None, None, :]
            x_off2 = j[:, None] * 2 + pair[None, :]
            a_pair = tl.load(
                ap_ptr + a_off3,
                mask=mask[:, :, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            x_pair = tl.load(
                xin_ptr + x_off2,
                mask=j_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            ar, ai = tl.split(a_pair)
            xr, xi = tl.split(x_pair)
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UNIT:
        xs_pair = tl.load(
            xin_ptr + rows[:, None] * 2 + pair[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        xs_r, xs_i = tl.split(xs_pair)
        acc_r += xs_r
        acc_i += xs_i

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctpmv"),
    key=_TPMV_KEY,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def ctpmv_splitk_kernel(
    ap_ptr,
    xin_ptr,
    partial_ptr,
    n,
    mode_key,
    SPLIT_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    pair = tl.arange(0, 2)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO != TRANS:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_SIZE_M
    active_len = active_hi - active_lo
    total_tiles = (active_len + BLOCK_K - 1) // BLOCK_K
    tiles_per_chunk = (total_tiles + SPLIT_K - 1) // SPLIT_K
    my_tile_lo = pid_k * tiles_per_chunk
    my_tile_hi = tl.minimum((pid_k + 1) * tiles_per_chunk, total_tiles)
    my_lo = active_lo + my_tile_lo * BLOCK_K
    my_hi = active_lo + my_tile_hi * BLOCK_K

    diag_lo_v = row_start
    diag_hi_v = row_start + BLOCK_SIZE_M

    if UPLO == TRANS:
        pre_hi = tl.minimum(my_hi, diag_lo_v)
        for kb in tl.range(my_lo, pre_hi, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off3 = off[:, :, None] * 2 + pair[None, None, :]
            x_off2 = j[:, None] * 2 + pair[None, :]
            a_pair = tl.load(
                ap_ptr + a_off3,
                mask=mask[:, :, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            x_pair = tl.load(
                xin_ptr + x_off2,
                mask=j_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            ar, ai = tl.split(a_pair)
            xr, xi = tl.split(x_pair)
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    diag_kb_lo = tl.maximum(my_lo, diag_lo_v)
    diag_kb_hi = tl.minimum(my_hi, diag_hi_v)
    for kb in tl.range(diag_kb_lo, diag_kb_hi, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        a_off3 = off[:, :, None] * 2 + pair[None, None, :]
        x_off2 = j[:, None] * 2 + pair[None, :]
        a_pair = tl.load(
            ap_ptr + a_off3,
            mask=mask[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        x_pair = tl.load(
            xin_ptr + x_off2,
            mask=j_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        ar, ai = tl.split(a_pair)
        xr, xi = tl.split(x_pair)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UPLO != TRANS:
        post_lo = tl.maximum(my_lo, diag_hi_v)
        for kb in tl.range(post_lo, my_hi, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off3 = off[:, :, None] * 2 + pair[None, None, :]
            x_off2 = j[:, None] * 2 + pair[None, :]
            a_pair = tl.load(
                ap_ptr + a_off3,
                mask=mask[:, :, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            x_pair = tl.load(
                xin_ptr + x_off2,
                mask=j_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            ar, ai = tl.split(a_pair)
            xr, xi = tl.split(x_pair)
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    out_off = pid_k * n * 2 + rows * 2
    tl.store(partial_ptr + out_off, acc_r, mask=row_mask)
    tl.store(partial_ptr + out_off + 1, acc_i, mask=row_mask)


def ctpmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert AP.dtype == torch.complex64 == x.dtype
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0
    split_k = _ctpmv_split_k(n, trans_flag)

    with torch_device_fn.device(AP.device):
        xin = x.as_strided((n,), (incx,)).clone()
        AP_real = torch.view_as_real(AP)
        xin_real = torch.view_as_real(xin)
        x_real = torch.view_as_real(x)
        if split_k > 1:
            partial = torch.empty(
                (split_k, n, 2), dtype=torch.float32, device=AP.device
            )
            grid_main = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]), split_k)
            ctpmv_splitk_kernel[grid_main](
                AP_real,
                xin_real,
                partial,
                n,
                _mode_key(uplo, trans_flag, unit) | (conj << 8),
                SPLIT_K=split_k,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
            )
            BLOCK_N = 1024
            grid_red = (triton.cdiv(n, BLOCK_N),)
            ctpmv_reduce_kernel[grid_red](
                partial,
                xin_real,
                x_real,
                n,
                incx,
                SPLIT_K=split_k,
                UNIT=unit,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
        else:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
            ctpmv_kernel[grid](
                AP_real,
                xin_real,
                x_real,
                n,
                incx,
                _mode_key(uplo, trans_flag, unit) | (conj << 8),
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
            )

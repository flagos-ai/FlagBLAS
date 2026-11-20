# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.utils import libentry, libtuner

__all__ = [
    "grouped_bfgemm_kernel",
    "group_bfgemm",
    "grouped_hgemm_kernel",
    "group_hgemm",
    "grouped_tf32gemm_kernel",
    "group_tf32gemm",
]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size", "stride_bk"],
)
@triton.jit
def grouped_bfgemm_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr = 0,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    if GROUP_N != 0:
        GROUP_N_EFFECTIVE: tl.constexpr = min(GROUP_N, (N + BLOCK_N - 1) // BLOCK_N)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        pe = window * num_n_tiles + tile_n
        sb = pe // (GROUP_N_EFFECTIVE * NUM_WM)
        r = pe % (GROUP_N_EFFECTIVE * NUM_WM)
        if (sb + 1) * GROUP_N_EFFECTIVE <= num_n_tiles:
            tile_n = sb * GROUP_N_EFFECTIVE + r % GROUP_N_EFFECTIVE
            window = r // GROUP_N_EFFECTIVE
        else:
            group_n = num_n_tiles - sb * GROUP_N_EFFECTIVE
            tile_n = sb * GROUP_N_EFFECTIVE + r % group_n
            window = r // group_n
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // BLOCK_M
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_full, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, K // BLOCK_K, num_stages=2, loop_unroll_factor=1):
            b = tl.load(b_ptrs, eviction_policy="evict_last")
            a = tl.load(a_ptrs)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        out_rows = (offs + tl.arange(0, BLOCK_M)).to(tl.int64)
        if BLOCK_M >= 256:
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_ptrs = (
                group_out
                + out_rows[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            if N % BLOCK_N == 0:
                tl.store(out_ptrs, output0.to(group_out.dtype.element_ty))
                tl.store(
                    out_ptrs + BLOCK_N // 2 * stride_on,
                    output1.to(group_out.dtype.element_ty),
                )
            else:
                tl.store(
                    out_ptrs,
                    output0.to(group_out.dtype.element_ty),
                    mask=offs_n_half[None, :] < N,
                )
                tl.store(
                    out_ptrs + BLOCK_N // 2 * stride_on,
                    output1.to(group_out.dtype.element_ty),
                    mask=offs_n_half[None, :] + BLOCK_N // 2 < N,
                )
        else:
            out_ptrs = (
                group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
            )
            if N % BLOCK_N == 0:
                tl.store(out_ptrs, accumulator.to(group_out.dtype.element_ty))
            else:
                tl.store(
                    out_ptrs,
                    accumulator.to(group_out.dtype.element_ty),
                    mask=offs_n[None, :] < N,
                )
    rem = end - start - num_full * BLOCK_M
    if rem == 0 or window != num_full % NUM_WM:
        return
    if N % BLOCK_N == 0 and K % BLOCK_K == 0:
        offs = start + num_full * BLOCK_M
        if rem <= BLOCK_M // 2:
            BLOCK_H: tl.constexpr = BLOCK_M // 2
            offs_m = offs + tl.arange(0, BLOCK_H)
            a_base = group_A + offs.to(tl.int64) * stride_am
            a_ptrs = (
                a_base
                + tl.arange(0, BLOCK_H)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, K // BLOCK_K, num_stages=2):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs, mask=offs_m[:, None] < end, other=0)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            out_rows = offs_m.to(tl.int64)
            if BLOCK_M >= 256:
                output = tl.reshape(accumulator, (BLOCK_H, 2, BLOCK_N // 2))
                output = tl.permute(output, (0, 2, 1))
                output0, output1 = tl.split(output)
                offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
                out_ptrs = (
                    group_out
                    + out_rows[:, None] * stride_om
                    + offs_n_half[None, :] * stride_on
                )
                tl.store(
                    out_ptrs,
                    output0.to(group_out.dtype.element_ty),
                    mask=offs_m[:, None] < end,
                )
                tl.store(
                    out_ptrs + BLOCK_N // 2 * stride_on,
                    output1.to(group_out.dtype.element_ty),
                    mask=offs_m[:, None] < end,
                )
            else:
                out_ptrs = (
                    group_out
                    + out_rows[:, None] * stride_om
                    + offs_n[None, :] * stride_on
                )
                tl.store(
                    out_ptrs,
                    accumulator.to(group_out.dtype.element_ty),
                    mask=offs_m[:, None] < end,
                )
            return
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, K // BLOCK_K, num_stages=2):
            b = tl.load(b_ptrs, eviction_policy="evict_last")
            a = tl.load(a_ptrs, mask=offs_m[:, None] < end, other=0)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        out_rows = offs_m.to(tl.int64)
        if BLOCK_M >= 256:
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_ptrs = (
                group_out
                + out_rows[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            tl.store(
                out_ptrs,
                output0.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
        else:
            out_ptrs = (
                group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
            )
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
        return


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size", "stride_bk"],
)
@triton.jit
def grouped_bfgemm_generic_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // BLOCK_M
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_full, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
            b = tl.load(
                b_ptrs,
                mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                other=0,
                eviction_policy="evict_last",
            )
            a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        out_rows = offs_m.to(tl.int64)
        if BLOCK_M >= 256:
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_ptrs = (
                group_out
                + out_rows[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            tl.store(
                out_ptrs,
                output0.to(group_out.dtype.element_ty),
                mask=offs_n_half[None, :] < N,
            )
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=offs_n_half[None, :] + BLOCK_N // 2 < N,
            )
        else:
            out_ptrs = (
                group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
            )
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=offs_n[None, :] < N,
            )
    rem = end - start - num_full * BLOCK_M
    if rem == 0 or window != num_full % NUM_WM:
        return
    BLOCK_H: tl.constexpr = BLOCK_M if K >= 1024 else BLOCK_M // 2
    for h in tl.static_range(0, 1 if K >= 1024 else 2):
        if h * BLOCK_H < rem:
            offs = start + num_full * BLOCK_M + h * BLOCK_H
            offs_m = offs + tl.arange(0, BLOCK_H)
            a_base = group_A + offs.to(tl.int64) * stride_am
            a_ptrs = (
                a_base
                + tl.arange(0, BLOCK_H)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < end) & (offs_k[None, :] + k * BLOCK_K < K),
                    other=0,
                )
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            out_rows = offs_m.to(tl.int64)
            if BLOCK_M >= 256:
                output = tl.reshape(accumulator, (BLOCK_H, 2, BLOCK_N // 2))
                output = tl.permute(output, (0, 2, 1))
                output0, output1 = tl.split(output)
                offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
                out_ptrs = (
                    group_out
                    + out_rows[:, None] * stride_om
                    + offs_n_half[None, :] * stride_on
                )
                tl.store(
                    out_ptrs,
                    output0.to(group_out.dtype.element_ty),
                    mask=(offs_m[:, None] < end) & (offs_n_half[None, :] < N),
                )
                tl.store(
                    out_ptrs + BLOCK_N // 2 * stride_on,
                    output1.to(group_out.dtype.element_ty),
                    mask=(offs_m[:, None] < end)
                    & (offs_n_half[None, :] + BLOCK_N // 2 < N),
                )
            else:
                out_ptrs = (
                    group_out
                    + out_rows[:, None] * stride_om
                    + offs_n[None, :] * stride_on
                )
                tl.store(
                    out_ptrs,
                    accumulator.to(group_out.dtype.element_ty),
                    mask=(offs_m[:, None] < end) & (offs_n[None, :] < N),
                )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size", "stride_bk"],
)
@triton.jit
def grouped_bfgemm_bp_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    offs_n = tile_n * BLOCK_N
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        a_ptrs = tl.make_block_ptr(
            base=group_A + offs.to(tl.int64) * stride_am,
            shape=(end - offs, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_ptrs = tl.make_block_ptr(
            base=b_base,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            offsets=(offs_n, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(0, 1),
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
            b = tl.trans(
                tl.load(b_ptrs, boundary_check=(0, 1), eviction_policy="evict_last")
            )
            a = tl.load(a_ptrs, boundary_check=(0, 1))
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
            b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
        out_ptrs = tl.make_block_ptr(
            base=group_out + offs.to(tl.int64) * stride_om,
            shape=(end - offs, N),
            strides=(stride_om, stride_on),
            offsets=(0, offs_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(
            out_ptrs, accumulator.to(group_out.dtype.element_ty), boundary_check=(0, 1)
        )


@libentry()
@libtuner(
    configs=[
        config
        for config in runtime.get_tuned_config("group_bfgemm")
        if config.kwargs["BLOCK_M"] == 256
    ],
    key=["M", "N", "K", "group_size", "stride_bk"],
)
@triton.jit
def grouped_bfgemm_wide_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr = 0,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    if GROUP_N != 0:
        GROUP_N_EFFECTIVE: tl.constexpr = min(GROUP_N, (N + BLOCK_N - 1) // BLOCK_N)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        pe = window * num_n_tiles + tile_n
        sb = pe // (GROUP_N_EFFECTIVE * NUM_WM)
        r = pe % (GROUP_N_EFFECTIVE * NUM_WM)
        if (sb + 1) * GROUP_N_EFFECTIVE <= num_n_tiles:
            tile_n = sb * GROUP_N_EFFECTIVE + r % GROUP_N_EFFECTIVE
            window = r // GROUP_N_EFFECTIVE
        else:
            group_n = num_n_tiles - sb * GROUP_N_EFFECTIVE
            tile_n = sb * GROUP_N_EFFECTIVE + r % group_n
            window = r // group_n
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // BLOCK_M
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_full, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(0, K // BLOCK_K, num_stages=2):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_rows = (offs + tl.arange(0, BLOCK_M)).to(tl.int64)
        output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        output = tl.permute(output, (0, 2, 1))
        output0, output1 = tl.split(output)
        offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n_half[None, :] * stride_on
        )
        if N % BLOCK_N == 0:
            tl.store(out_ptrs, output0.to(group_out.dtype.element_ty))
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
            )
        else:
            tl.store(
                out_ptrs,
                output0.to(group_out.dtype.element_ty),
                mask=offs_n_half[None, :] < N,
            )
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=offs_n_half[None, :] + BLOCK_N // 2 < N,
            )


@libentry()
@triton.jit
def grouped_bfgemm_wide_fused_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_om: tl.constexpr,
    NUM_WM: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // 256
    offs_k = tl.arange(0, 64)
    b_base = group_B + expert.to(tl.int64) * stride_be
    if window == NUM_WM:
        rem = end - start - num_full * 256
        tile_n = N // 256 - 1 - tile_n
        offs_n = tile_n * 256 + tl.arange(0, 256)
        if rem > 0:
            if rem <= 128:
                offs = start + num_full * 256
                offs_m = offs + tl.arange(0, 128)
                a_rows = tl.minimum(offs_m, end - 1)
                a_ptrs = (
                    group_A
                    + offs.to(tl.int64) * stride_am
                    + (a_rows - offs)[:, None] * stride_am
                    + offs_k[None, :]
                )
                b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :]
                accumulator = tl.zeros((128, 256), dtype=tl.float32)
                for k in tl.range(0, K // 64, num_stages=2):
                    b = tl.load(b_ptrs, eviction_policy="evict_last")
                    a = tl.load(a_ptrs)
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += 64
                    b_ptrs += 64 * stride_bk
                output = tl.reshape(accumulator, (128, 2, 128))
                output = tl.permute(output, (0, 2, 1))
                output0, output1 = tl.split(output)
                out_ptrs = (
                    group_out
                    + offs_m.to(tl.int64)[:, None] * stride_om
                    + (tile_n * 256 + tl.arange(0, 128))[None, :]
                )
                tl.store(
                    out_ptrs,
                    output0.to(group_out.dtype.element_ty),
                    mask=offs_m[:, None] < end,
                )
                tl.store(
                    out_ptrs + 128,
                    output1.to(group_out.dtype.element_ty),
                    mask=offs_m[:, None] < end,
                )
                return
            offs = start + num_full * 256
            offs_m = offs + tl.arange(0, 256)
            a_rows = tl.minimum(offs_m, end - 1)
            a_ptrs = (
                group_A
                + offs.to(tl.int64) * stride_am
                + (a_rows - offs)[:, None] * stride_am
                + offs_k[None, :]
            )
            b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :]
            accumulator = tl.zeros((256, 256), dtype=tl.float32)
            for k in tl.range(0, K // 64, num_stages=2):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += 64
                b_ptrs += 64 * stride_bk
            output = tl.reshape(accumulator, (256, 2, 128))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            out_ptrs = (
                group_out
                + offs_m.to(tl.int64)[:, None] * stride_om
                + (tile_n * 256 + tl.arange(0, 128))[None, :]
            )
            tl.store(
                out_ptrs,
                output0.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
            tl.store(
                out_ptrs + 128,
                output1.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
        return
    num_n = N // 256
    pe = window * num_n + tile_n
    sb = pe // (8 * NUM_WM)
    r = pe % (8 * NUM_WM)
    if (sb + 1) * 8 <= num_n:
        tile_n = sb * 8 + r % 8
        window = r // 8
    else:
        group_n = num_n - sb * 8
        tile_n = sb * 8 + r % group_n
        window = r // group_n
    offs_n = tile_n * 256 + tl.arange(0, 256)
    for w in tl.range(window, num_full, NUM_WM, num_stages=1):
        offs = start + w * 256
        a_ptrs = (
            group_A
            + offs.to(tl.int64) * stride_am
            + tl.arange(0, 256)[:, None] * stride_am
            + offs_k[None, :]
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :]
        accumulator = tl.zeros((256, 256), dtype=tl.float32)
        for k in tl.range(0, K // 64, num_stages=2):
            b = tl.load(b_ptrs, eviction_policy="evict_last")
            a = tl.load(a_ptrs)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += 64
            b_ptrs += 64 * stride_bk
        output = tl.reshape(accumulator, (256, 2, 128))
        output = tl.permute(output, (0, 2, 1))
        output0, output1 = tl.split(output)
        out_ptrs = (
            group_out
            + (offs + tl.arange(0, 256)).to(tl.int64)[:, None] * stride_om
            + (tile_n * 256 + tl.arange(0, 128))[None, :]
        )
        tl.store(out_ptrs, output0.to(group_out.dtype.element_ty))
        tl.store(out_ptrs + 128, output1.to(group_out.dtype.element_ty))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size", "stride_bk"],
)
@triton.jit
def grouped_bfgemm_aligned_tail_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // 256
    rem = end - start - num_full * 256
    if rem == 0:
        return
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    if rem <= BLOCK_M // 2:
        offs = start + num_full * 256
        offs_m = offs + tl.arange(0, BLOCK_M // 2)
        a_rows = tl.minimum(offs_m, end - 1)
        a_ptrs = (
            group_A
            + offs.to(tl.int64) * stride_am
            + (a_rows - offs)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, K // BLOCK_K, num_stages=2):
            b = tl.load(b_ptrs, eviction_policy="evict_last")
            a = tl.load(a_ptrs)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        output = tl.reshape(accumulator, (BLOCK_M // 2, 2, BLOCK_N // 2))
        output = tl.permute(output, (0, 2, 1))
        output0, output1 = tl.split(output)
        out_ptrs = (
            group_out
            + offs_m.to(tl.int64)[:, None] * stride_om
            + offs_n_half[None, :] * stride_on
        )
        tl.store(
            out_ptrs,
            output0.to(group_out.dtype.element_ty),
            mask=offs_m[:, None] < end,
        )
        tl.store(
            out_ptrs + BLOCK_N // 2 * stride_on,
            output1.to(group_out.dtype.element_ty),
            mask=offs_m[:, None] < end,
        )
        return
    for h in tl.static_range(0, 256 // BLOCK_M):
        if h * BLOCK_M < rem:
            offs = start + num_full * 256 + h * BLOCK_M
            offs_m = offs + tl.arange(0, BLOCK_M)
            a_rows = tl.minimum(offs_m, end - 1)
            a_ptrs = (
                group_A
                + offs.to(tl.int64) * stride_am
                + (a_rows - offs)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, K // BLOCK_K, num_stages=2):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            out_ptrs = (
                group_out
                + offs_m.to(tl.int64)[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            tl.store(
                out_ptrs,
                output0.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size", "stride_bk"],
)
@triton.jit
def grouped_bfgemm_tail_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // 256
    rem = end - start - num_full * 256
    if rem == 0:
        return
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for h in tl.static_range(0, 256 // BLOCK_M):
        if h * BLOCK_M < rem:
            offs = start + num_full * 256 + h * BLOCK_M
            offs_m = offs + tl.arange(0, BLOCK_M)
            a_base = group_A + offs.to(tl.int64) * stride_am
            if N % BLOCK_N == 0 and K % BLOCK_K == 0:
                a_rows = tl.minimum(offs_m, end - 1)
                a_ptrs = (
                    a_base
                    + (a_rows - offs)[:, None] * stride_am
                    + offs_k[None, :] * stride_ak
                )
            else:
                a_ptrs = (
                    a_base
                    + tl.arange(0, BLOCK_M)[:, None] * stride_am
                    + offs_k[None, :] * stride_ak
                )
            b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if N % BLOCK_N == 0 and K % BLOCK_K == 0:
                for k in tl.range(0, K // BLOCK_K, num_stages=2):
                    b = tl.load(b_ptrs, eviction_policy="evict_last")
                    a = tl.load(a_ptrs)
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            else:
                for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_n[None, :] < N)
                        & (offs_k[:, None] + k * BLOCK_K < K),
                        other=0,
                        eviction_policy="evict_last",
                    )
                    a = tl.load(
                        a_ptrs,
                        mask=(offs_m[:, None] < end)
                        & (offs_k[None, :] + k * BLOCK_K < K),
                        other=0,
                    )
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            out_rows = offs_m.to(tl.int64)
            if BLOCK_M >= 256:
                output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
                output = tl.permute(output, (0, 2, 1))
                output0, output1 = tl.split(output)
                offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
                out_ptrs = (
                    group_out
                    + out_rows[:, None] * stride_om
                    + offs_n_half[None, :] * stride_on
                )
                if N % BLOCK_N == 0:
                    tl.store(
                        out_ptrs,
                        output0.to(group_out.dtype.element_ty),
                        mask=offs_m[:, None] < end,
                    )
                    tl.store(
                        out_ptrs + BLOCK_N // 2 * stride_on,
                        output1.to(group_out.dtype.element_ty),
                        mask=offs_m[:, None] < end,
                    )
                else:
                    tl.store(
                        out_ptrs,
                        output0.to(group_out.dtype.element_ty),
                        mask=(offs_m[:, None] < end) & (offs_n_half[None, :] < N),
                    )
                    tl.store(
                        out_ptrs + BLOCK_N // 2 * stride_on,
                        output1.to(group_out.dtype.element_ty),
                        mask=(offs_m[:, None] < end)
                        & (offs_n_half[None, :] + BLOCK_N // 2 < N),
                    )
            else:
                out_ptrs = (
                    group_out
                    + out_rows[:, None] * stride_om
                    + offs_n[None, :] * stride_on
                )
                tl.store(
                    out_ptrs,
                    accumulator.to(group_out.dtype.element_ty),
                    mask=(offs_m[:, None] < end) & (offs_n[None, :] < N),
                )


def group_bfgemm(group_A, group_B, group_list, group_out):
    assert group_A.ndim == 2 and group_B.ndim == 3 and group_list.ndim == 1
    M, K = group_A.shape
    group_size, BK, N = group_B.shape
    assert BK == K
    assert group_list.numel() == group_size
    assert group_out.shape == (M, N)
    assert group_A.stride(1) == 1 and group_out.stride(1) == 1
    assert (group_B.stride(1) == N and group_B.stride(2) == 1) or (
        group_B.stride(1) == 1 and group_B.stride(2) == K
    )
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
    if K <= 96:
        grouped_bfgemm_bp_kernel[grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
            use_in_thread_transpose=False,
        )
    elif K == 2048 and N == 7168 and group_size in (16, 17):
        num_wm = 9 if group_size == 16 else 13
        grouped_bfgemm_wide_fused_kernel[(N // 256, num_wm + 1, group_size)](
            group_A,
            group_B,
            group_list,
            group_out,
            N,
            K,
            group_A.stride(0),
            group_B.stride(0),
            group_B.stride(1),
            group_out.stride(0),
            NUM_WM=num_wm,
            num_warps=16,
            num_stages=2,
            use_in_thread_transpose=True,
            sched_latency="mmac5-ds10",
        )
    elif N > 4096 and K >= 1024:
        tail_kernel = (
            grouped_bfgemm_aligned_tail_kernel
            if N % 256 == 0 and K % 64 == 0
            else grouped_bfgemm_tail_kernel
        )
        tail_grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), 1, group_size)
        if K == 2048:
            grouped_bfgemm_wide_kernel[grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                GROUP_N=8,
                kpack=1,
                use_in_thread_transpose=True,
                sched_latency="mmac5-ds10",
            )
            tail_kernel[tail_grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                kpack=1,
                use_in_thread_transpose=True,
                sched_latency="mmac5-ds10",
            )
        else:
            grouped_bfgemm_wide_kernel[grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                kpack=2 if K not in (1536, 7168) else 1,
                use_in_thread_transpose=False,
            )
            tail_kernel[tail_grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                kpack=2 if K not in (1536, 7168) else 1,
                use_in_thread_transpose=False,
            )
    elif N % 256 == 0 and K % 64 == 0:
        if K == 2048 and group_size == 32 and N == 1536:
            grouped_bfgemm_kernel[grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                GROUP_N=6,
                kpack=2,
                use_in_thread_transpose=False,
            )
        elif (
            (K == 384 and N == 2048)
            or (K in (1536, 7168) and N == 4096)
            or (K == 2048 and N == 256)
        ):
            grouped_bfgemm_kernel[grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                GROUP_N=8,
                kpack=1,
                use_in_thread_transpose=True,
            )
        elif K == 2048 and group_size == 128 and N == 1536:
            grouped_bfgemm_kernel[grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                GROUP_N=0,
                kpack=2,
                use_in_thread_transpose=False,
                sched_latency="mmac5-ds10",
            )
        else:
            grouped_bfgemm_kernel[grid](
                group_A,
                group_B,
                group_list,
                group_out,
                M,
                N,
                K,
                group_size,
                *group_A.stride(),
                *group_B.stride(),
                *group_out.stride(),
                GROUP_N=0,
                kpack=2 if K >= 512 and K not in (1536, 7168) else 1,
                use_in_thread_transpose=K == 128 and N == 2048,
            )
    else:
        grouped_bfgemm_generic_kernel[grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
            kpack=2
            if K >= 512 and N >= 256 and N != 384 and K not in (1536, 7168)
            else 1,
            use_in_thread_transpose=False,
        )
    return group_out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_hgemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_hgemm_kernel(
    group_A,
    group_B,
    group_out,
    group_list,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // BLOCK_M
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * (K * stride_bk)
    for w in tl.range(window, num_full, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(0, K // BLOCK_K, num_stages=2):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * stride_bk
        out_ptrs = (
            group_out + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
        )
        if N % BLOCK_N == 0:
            tl.store(out_ptrs, accumulator.to(tl.float16))
        else:
            tl.store(out_ptrs, accumulator.to(tl.float16), mask=offs_n[None, :] < N)
    rem = end - start - num_full * BLOCK_M
    if rem > 0 and window == num_full % NUM_WM:
        for h in tl.static_range(0, 2):
            if h * (BLOCK_M // 2) < rem:
                offs = start + num_full * BLOCK_M + h * (BLOCK_M // 2)
                offs_m = offs + tl.arange(0, BLOCK_M // 2)
                a_base = group_A + offs.to(tl.int64) * stride_am
                a_ptrs = (
                    a_base
                    + tl.arange(0, BLOCK_M // 2)[:, None] * stride_am
                    + offs_k[None, :]
                )
                b_ptrs = (
                    b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                )
                accumulator = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32)
                for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_n[None, :] < N)
                        & (offs_k[:, None] + k * BLOCK_K < K),
                        other=0,
                        eviction_policy="evict_last",
                    )
                    a = tl.load(
                        a_ptrs,
                        mask=(offs_m[:, None] < end)
                        & (offs_k[None, :] + k * BLOCK_K < K),
                        other=0,
                    )
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K
                    b_ptrs += BLOCK_K * stride_bk
                out_ptrs = (
                    group_out
                    + offs_m.to(tl.int64)[:, None] * stride_om
                    + offs_n[None, :]
                )
                tl.store(
                    out_ptrs,
                    accumulator.to(tl.float16),
                    mask=(offs_m[:, None] < end) & (offs_n[None, :] < N),
                )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_hgemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_hgemm_bp_kernel(
    group_A,
    group_B,
    group_out,
    group_list,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    offs_n = tile_n * BLOCK_N
    b_base = group_B + expert.to(tl.int64) * (K * stride_bk)
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        a_ptrs = tl.make_block_ptr(
            base=group_A + offs.to(tl.int64) * stride_am,
            shape=(end - offs, K),
            strides=(stride_am, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_ptrs = tl.make_block_ptr(
            base=b_base,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            offsets=(offs_n, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(0, 1),
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
            b = tl.trans(
                tl.load(b_ptrs, boundary_check=(0, 1), eviction_policy="evict_last")
            )
            a = tl.load(a_ptrs, boundary_check=(0, 1))
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
            b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
        out_ptrs = tl.make_block_ptr(
            base=group_out + offs.to(tl.int64) * stride_om,
            shape=(end - offs, N),
            strides=(stride_om, 1),
            offsets=(0, offs_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(out_ptrs, accumulator.to(tl.float16), boundary_check=(0, 1))


def group_hgemm(group_A, group_B, group_C, group_list, group_out):
    M, K = group_A.shape
    N = group_B.shape[1]
    group_size = group_list.numel()
    assert group_B.shape == (group_size * K, N)
    assert group_C.shape == (M, N) and group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
    if (K == 2048 and N >= 256 and N != 384) or (K in (1536, 4096, 7168) and N >= 3072):
        grouped_hgemm_bp_kernel[grid](
            group_A,
            group_B,
            group_out,
            group_list,
            M,
            N,
            K,
            group_size,
            group_A.stride(0),
            group_B.stride(0),
            group_B.stride(1),
            group_out.stride(0),
        )
    else:
        grouped_hgemm_kernel[grid](
            group_A,
            group_B,
            group_out,
            group_list,
            M,
            N,
            K,
            group_size,
            group_A.stride(0),
            group_B.stride(0),
            group_B.stride(1),
            group_out.stride(0),
        )
    return group_out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_tf32gemm_kernel(
    group_A,
    group_B,
    group_out,
    group_list,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_full = (end - start) // BLOCK_M
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * (N * stride_bn)
    for w in tl.range(window, num_full, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(0, K // BLOCK_K, num_stages=2 if BLOCK_K != 64 else 1):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=2 if BLOCK_K != 64 else 1
            ):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                accumulator = tl.dot(a, b, acc=accumulator)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * stride_bk
        out_ptrs = (
            group_out + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
        )
        if N % BLOCK_N == 0:
            tl.store(out_ptrs, accumulator)
        else:
            tl.store(out_ptrs, accumulator, mask=offs_n[None, :] < N)
    rem = end - start - num_full * BLOCK_M
    if rem > 0 and window == num_full % NUM_WM:
        for h in tl.static_range(0, 2):
            if h * (BLOCK_M // 2) < rem:
                offs = start + num_full * BLOCK_M + h * (BLOCK_M // 2)
                offs_m = offs + tl.arange(0, BLOCK_M // 2)
                a_base = group_A + offs.to(tl.int64) * stride_am
                a_ptrs = (
                    a_base
                    + tl.arange(0, BLOCK_M // 2)[:, None] * stride_am
                    + offs_k[None, :]
                )
                b_ptrs = (
                    b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                )
                accumulator = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32)
                for k in tl.range(
                    0, tl.cdiv(K, BLOCK_K), num_stages=2 if BLOCK_K != 64 else 1
                ):
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_n[None, :] < N)
                        & (offs_k[:, None] + k * BLOCK_K < K),
                        other=0,
                        eviction_policy="evict_last",
                    )
                    a = tl.load(
                        a_ptrs,
                        mask=(offs_m[:, None] < end)
                        & (offs_k[None, :] + k * BLOCK_K < K),
                        other=0,
                    )
                    accumulator = tl.dot(a, b, acc=accumulator)
                    a_ptrs += BLOCK_K
                    b_ptrs += BLOCK_K * stride_bk
                out_ptrs = (
                    group_out
                    + offs_m.to(tl.int64)[:, None] * stride_om
                    + offs_n[None, :]
                )
                tl.store(
                    out_ptrs,
                    accumulator,
                    mask=(offs_m[:, None] < end) & (offs_n[None, :] < N),
                )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_tf32gemm_bp_kernel(
    group_A,
    group_B,
    group_out,
    group_list,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    offs_n = tile_n * BLOCK_N
    b_base = group_B + expert.to(tl.int64) * (N * stride_bn)
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        a_ptrs = tl.make_block_ptr(
            base=group_A + offs.to(tl.int64) * stride_am,
            shape=(end - offs, K),
            strides=(stride_am, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_ptrs = tl.make_block_ptr(
            base=b_base,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, offs_n),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2, loop_unroll_factor=2):
            b = tl.load(b_ptrs, boundary_check=(0, 1), eviction_policy="evict_last")
            a = tl.load(a_ptrs, boundary_check=(0, 1))
            accumulator = tl.dot(a, b, acc=accumulator)
            a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
            b_ptrs = tl.advance(b_ptrs, (BLOCK_K, 0))
        out_ptrs = tl.make_block_ptr(
            base=group_out + offs.to(tl.int64) * stride_om,
            shape=(end - offs, N),
            strides=(stride_om, 1),
            offsets=(0, offs_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(out_ptrs, accumulator, boundary_check=(0, 1))


def group_tf32gemm(group_A, group_B, group_C, group_list, group_out):
    M, K = group_A.shape
    N = group_out.shape[1]
    group_size = group_list.numel()
    assert group_C.shape == (M, N) and group_out.shape == (M, N)
    assert group_B.shape == (group_size * N, K)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
    if (K in (1536, 4096, 7168) and N >= 3072) or (K == 2048 and N > 4096):
        grouped_tf32gemm_bp_kernel[grid](
            group_A,
            group_B,
            group_out,
            group_list,
            M,
            N,
            K,
            group_size,
            group_A.stride(0),
            group_B.stride(1),
            group_B.stride(0),
            group_out.stride(0),
        )
    else:
        grouped_tf32gemm_kernel[grid](
            group_A,
            group_B,
            group_out,
            group_list,
            M,
            N,
            K,
            group_size,
            group_A.stride(0),
            group_B.stride(1),
            group_B.stride(0),
            group_out.stride(0),
        )
    return group_out

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

__all__ = ["grouped_bfgemm_kernel", "group_bfgemm"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
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
    PIPE_STAGES: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // 256 * (256 // BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
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
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
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
        output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        output = tl.permute(output, (0, 2, 1))
        output0, output1 = tl.split(output)
        offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n_half[None, :] * stride_on
        )
        out_mask = (out_rows[:, None] < end.to(tl.int64)) & (offs_n_half[None, :] < N)
        tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
        tl.store(
            out_ptrs + BLOCK_N // 2 * stride_on,
            output1.to(group_out.dtype.element_ty),
            mask=out_mask & (offs_n_half[None, :] + BLOCK_N // 2 < N),
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_bfgemm_direct_store_kernel(
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
    PIPE_STAGES: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // 256 * (256 // BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
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
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_base = group_out + offs.to(tl.int64) * stride_om
        out_ptrs = (
            out_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        if N % BLOCK_N == 0:
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
        else:
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=(offs_m[:, None] < end) & (offs_n[None, :] < N),
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_bfgemm_swz_kernel(
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
    GROUP_N: tl.constexpr = 4,
    PIPE_STAGES: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    nt = tl.cdiv(N, BLOCK_N)
    per_expert = nt * NUM_WM
    expert = pid // per_expert
    pe = pid % per_expert
    sb = pe // (GROUP_N * NUM_WM)
    r = pe % (GROUP_N * NUM_WM)
    tile_n = sb * GROUP_N + r % GROUP_N
    window = r // GROUP_N
    if tile_n < nt:
        end = tl.load(group_list + expert).to(tl.int32)
        start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
        num_tiles = tl.cdiv(end - start, BLOCK_M)
        if expert == group_size - 1:
            num_tiles = (end - start) // 256 * (256 // BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        b_base = group_B + expert.to(tl.int64) * stride_be
        for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
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
            if N % BLOCK_N == 0 and K % BLOCK_K == 0:
                for k in tl.range(
                    0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
                ):
                    b = tl.load(b_ptrs, eviction_policy="evict_last")
                    a = tl.load(a_ptrs)
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            else:
                for k in tl.range(
                    0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
                ):
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_n[None, :] < N)
                        & (offs_k[:, None] + k * BLOCK_K < K),
                        other=0,
                        eviction_policy="evict_last",
                    )
                    a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_rows = offs_m.to(tl.int64)
            out_ptrs = (
                group_out
                + out_rows[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            out_mask = (out_rows[:, None] < end.to(tl.int64)) & (
                offs_n_half[None, :] < N
            )
            tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=out_mask & (offs_n_half[None, :] + BLOCK_N // 2 < N),
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm_tail"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_bfgemm_tail_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    tile_m = tl.program_id(1)
    expert = group_size - 1
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    end = tl.load(group_list + expert).to(tl.int32)
    offs = start + (end - start) // 256 * 256 + tile_m * BLOCK_M
    if offs < end:
        offs_m = offs + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = (
            group_A
            + offs_m[:, None].to(tl.int64) * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            group_B
            + expert.to(tl.int64) * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(0, K // BLOCK_K, num_stages=1, loop_unroll_factor=1):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs, mask=offs_m[:, None] < end, other=0)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=1, loop_unroll_factor=1
            ):
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
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        tl.store(
            out_ptrs,
            accumulator.to(group_out.dtype.element_ty),
            mask=(out_rows[:, None] < end.to(tl.int64)) & (offs_n[None, :] < N),
        )


def group_bfgemm(group_A, group_B, group_list, group_out):
    assert group_A.ndim == 2 and group_B.ndim == 3 and group_list.ndim == 1
    M, K = group_A.shape
    group_size, BK, N = group_B.shape
    assert BK == K
    assert group_list.numel() == group_size
    assert group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
    if K <= 768 and N >= 2048:
        _grouped_bfgemm_direct_store_kernel[grid](
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
        )
    elif N >= 7168:
        swz_grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]) * meta["NUM_WM"] * group_size,
        )
        _grouped_bfgemm_swz_kernel[swz_grid](
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
        )
    tail_grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(256, meta["BLOCK_M"]),
    )
    _grouped_bfgemm_tail_kernel[tail_grid](
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
    )
    return group_out

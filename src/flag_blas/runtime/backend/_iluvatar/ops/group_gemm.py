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

import torch
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


@triton.jit
def _grouped_gemm_schedule_kernel(
    group_list,
    schedule,
    group_size: tl.constexpr,
    MAX_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    expert = tl.program_id(0)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    end = tl.load(group_list + expert).to(tl.int32)
    schedule_start = 0
    for previous in tl.range(0, expert, loop_unroll_factor=1):
        previous_start = tl.load(
            group_list + previous - 1, mask=previous > 0, other=0
        ).to(tl.int32)
        previous_end = tl.load(group_list + previous).to(tl.int32)
        schedule_start += tl.cdiv(previous_end - previous_start, BLOCK_M)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // BLOCK_M
    for local_tile in tl.range(0, num_tiles, loop_unroll_factor=1):
        slot = (schedule_start + local_tile) * 2
        row = start + local_tile * BLOCK_M
        tl.store(schedule + slot, row)
        tl.store(schedule + slot + 1, expert)
    if expert == group_size - 1:
        for slot in tl.range(
            schedule_start + num_tiles, MAX_TILES, loop_unroll_factor=1
        ):
            tl.store(schedule + slot * 2, -1)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm_balanced"),
    key=["N", "K", "group_size"],
)
@triton.jit
def _grouped_bfgemm_balanced_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    schedule,
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
    MAX_TILES: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for task in tl.range(window, MAX_TILES, NUM_WM, loop_unroll_factor=1):
        offs = tl.load(schedule + task * 2)
        if offs >= 0:
            expert = tl.load(schedule + task * 2 + 1)
            end = tl.load(group_list + expert).to(tl.int32)
            offs_m = offs + tl.arange(0, BLOCK_M)
            a_base = group_A + offs.to(tl.int64) * stride_am
            a_ptrs = (
                a_base
                + tl.arange(0, BLOCK_M)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = (
                group_B
                + expert.to(tl.int64) * stride_be
                + offs_k[:, None] * stride_bk
                + offs_n[None, :] * stride_bn
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_base = group_out + offs.to(tl.int64) * stride_om
            out_ptrs = (
                out_base
                + tl.arange(0, BLOCK_M)[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            out_mask = offs_m[:, None] < end
            tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=out_mask,
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
    if K == 2048 and (
        (N in (768, 1024) and group_size == 128)
        or (N == 1536 and group_size in (64, 128))
        or (N == 256 and group_size == 512)
    ):
        max_tiles = triton.cdiv(M, 256) + group_size - 1
        schedule = torch.empty(
            (max_tiles, 2), dtype=torch.int32, device=group_list.device
        )
        _grouped_gemm_schedule_kernel[(group_size,)](
            group_list,
            schedule,
            group_size,
            max_tiles,
            256,
        )
        _grouped_bfgemm_balanced_kernel[
            lambda meta: (triton.cdiv(N, 256), meta["NUM_WM"], 1)
        ](
            group_A,
            group_B,
            group_list,
            group_out,
            schedule,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
            max_tiles,
            BLOCK_M=256,
            BLOCK_N=256,
            BLOCK_K=64,
            PIPE_STAGES=1,
        )
    elif (K <= 768 and N >= 2048) or N == 192:
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_hgemm_kernel(
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
    configs=runtime.get_tuned_config("group_hgemm_balanced"),
    key=["N", "K", "group_size"],
)
@triton.jit
def _grouped_hgemm_balanced_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    schedule,
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
    MAX_TILES: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for task in tl.range(window, MAX_TILES, NUM_WM, loop_unroll_factor=1):
        offs = tl.load(schedule + task * 2)
        if offs >= 0:
            expert = tl.load(schedule + task * 2 + 1)
            end = tl.load(group_list + expert).to(tl.int32)
            offs_m = offs + tl.arange(0, BLOCK_M)
            a_base = group_A + offs.to(tl.int64) * stride_am
            a_ptrs = (
                a_base
                + tl.arange(0, BLOCK_M)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = (
                group_B
                + expert.to(tl.int64) * stride_be
                + offs_k[:, None] * stride_bk
                + offs_n[None, :] * stride_bn
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_base = group_out + offs.to(tl.int64) * stride_om
            out_ptrs = (
                out_base
                + tl.arange(0, BLOCK_M)[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            out_mask = offs_m[:, None] < end
            tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=out_mask,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_hgemm_direct_store_kernel(
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
    configs=[
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "NUM_WM": 8,
                "PIPE_STAGES": 2,
            },
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "NUM_WM": 8,
                "PIPE_STAGES": 1,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "NUM_WM": 16,
                "PIPE_STAGES": 2,
            },
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "NUM_WM": 4,
                "PIPE_STAGES": 2,
            },
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "NUM_WM": 4,
                "PIPE_STAGES": 4,
            },
            num_warps=16,
            num_stages=1,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 512,
                "BLOCK_K": 32,
                "NUM_WM": 2,
                "PIPE_STAGES": 1,
            },
            num_warps=16,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_hgemm_swz_kernel(
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
def _grouped_hgemm_tail_kernel(
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


def group_hgemm(group_A, group_B, group_list, group_out):
    assert group_A.ndim == 2 and group_B.ndim == 3 and group_list.ndim == 1
    M, K = group_A.shape
    group_size, BK, N = group_B.shape
    assert BK == K
    assert group_list.numel() == group_size
    assert group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
    if K == 2048 and (
        (N in (192, 768, 1024) and group_size == 128)
        or (N == 1536 and group_size in (64, 128))
        or (N == 256 and group_size == 512)
    ):
        block_m = 128 if N == 192 else 256
        block_n = 64 if N == 192 else 256
        pipe_stages = 2 if N in (192, 768, 256) else 1
        max_tiles = triton.cdiv(M, block_m) + group_size - 1
        schedule = torch.empty(
            (max_tiles, 2), dtype=torch.int32, device=group_list.device
        )
        _grouped_gemm_schedule_kernel[(group_size,)](
            group_list,
            schedule,
            group_size,
            max_tiles,
            block_m,
        )
        _grouped_hgemm_balanced_kernel[
            lambda meta: (triton.cdiv(N, block_n), meta["NUM_WM"], 1)
        ](
            group_A,
            group_B,
            group_list,
            group_out,
            schedule,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
            max_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=64,
            PIPE_STAGES=pipe_stages,
        )
    elif K <= 768 and N >= 2048:
        _grouped_hgemm_direct_store_kernel[grid](
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
        _grouped_hgemm_swz_kernel[swz_grid](
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
        grouped_hgemm_kernel[grid](
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
    _grouped_hgemm_tail_kernel[tail_grid](
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_tf32gemm_kernel(
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
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
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
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
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
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_tf32gemm_direct_store_kernel(
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
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
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
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
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
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_tf32gemm_swz_kernel(
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
                    accumulator = tl.dot(
                        a,
                        b,
                        acc=accumulator,
                        input_precision="tf32",
                        out_dtype=tl.float32,
                    )
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
                    accumulator = tl.dot(
                        a,
                        b,
                        acc=accumulator,
                        input_precision="tf32",
                        out_dtype=tl.float32,
                    )
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
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        )
    ],
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_tf32gemm_tail_kernel(
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
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
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
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
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


def group_tf32gemm(group_A, group_B, group_list, group_out):
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
        _grouped_tf32gemm_direct_store_kernel[grid](
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
        _grouped_tf32gemm_swz_kernel[swz_grid](
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
        grouped_tf32gemm_kernel[grid](
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
    _grouped_tf32gemm_tail_kernel[tail_grid](
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

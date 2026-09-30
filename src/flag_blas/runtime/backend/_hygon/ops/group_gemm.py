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
    USE_BP: tl.constexpr = 0,
    PIPE_STAGES: tl.constexpr = 2,
    UNROLL: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    if USE_BP:
        end = tl.load(group_list + expert).to(tl.int32)
        start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(
            tl.int32
        )
        num_full = (end - start) // BLOCK_M
        offs_bn = tile_n * BLOCK_N
        b_base = group_B + expert.to(tl.int64) * stride_be
        EVEN_KN: tl.constexpr = (N % BLOCK_N == 0) and (K % BLOCK_K == 0)
        for w in tl.range(window, num_full, NUM_WM, num_stages=1):
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
                offsets=(offs_bn, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if EVEN_KN:
                for k in tl.range(
                    0,
                    K // BLOCK_K,
                    num_stages=PIPE_STAGES,
                    loop_unroll_factor=UNROLL,
                ):
                    b = tl.trans(tl.load(b_ptrs, eviction_policy="evict_last"))
                    a = tl.load(a_ptrs)
                    accumulator = tl.dot(
                        a, b, acc=accumulator, out_dtype=tl.float32
                    )
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
            else:
                for k in tl.range(
                    0,
                    tl.cdiv(K, BLOCK_K),
                    num_stages=PIPE_STAGES,
                    loop_unroll_factor=1,
                ):
                    b = tl.trans(
                        tl.load(
                            b_ptrs,
                            boundary_check=(0, 1),
                            eviction_policy="evict_last",
                        )
                    )
                    a = tl.load(a_ptrs, boundary_check=(0, 1))
                    accumulator = tl.dot(
                        a, b, acc=accumulator, out_dtype=tl.float32
                    )
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
            out_ptrs = tl.make_block_ptr(
                base=group_out + offs.to(tl.int64) * stride_om,
                shape=(end - offs, N),
                strides=(stride_om, stride_on),
                offsets=(0, offs_bn),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            if EVEN_KN:
                tl.store(out_ptrs, accumulator.to(group_out.dtype.element_ty))
            else:
                tl.store(
                    out_ptrs,
                    accumulator.to(group_out.dtype.element_ty),
                    boundary_check=(0, 1),
                )
        rem = end - start - num_full * BLOCK_M
        if rem > 0 and window == num_full % NUM_WM:
            offs = start + num_full * BLOCK_M
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
                offsets=(offs_bn, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(
                0,
                tl.cdiv(K, BLOCK_K),
                num_stages=PIPE_STAGES,
                loop_unroll_factor=1,
            ):
                b = tl.trans(
                    tl.load(
                        b_ptrs,
                        boundary_check=(0, 1),
                        eviction_policy="evict_last",
                    )
                )
                a = tl.load(a_ptrs, boundary_check=(0, 1))
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
            out_ptrs = tl.make_block_ptr(
                base=group_out + offs.to(tl.int64) * stride_om,
                shape=(end - offs, N),
                strides=(stride_om, stride_on),
                offsets=(0, offs_bn),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                boundary_check=(0, 1),
            )
        return
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
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=UNROLL
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
        out_rows = (offs + tl.arange(0, BLOCK_M)).to(tl.int64)
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
    if rem > 0 and window == num_full % NUM_WM:
        BLOCK_H: tl.constexpr = BLOCK_M // 2
        for h in tl.static_range(0, 2):
            if h * BLOCK_H < rem:
                offs = start + num_full * BLOCK_M + h * BLOCK_H
                offs_m = offs + tl.arange(0, BLOCK_H)
                a_base = group_A + offs.to(tl.int64) * stride_am
                a_ptrs = (
                    a_base
                    + tl.arange(0, BLOCK_H)[:, None] * stride_am
                    + offs_k[None, :] * stride_ak
                )
                b_ptrs = (
                    b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                )
                accumulator = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
                for k in tl.range(
                    0,
                    tl.cdiv(K, BLOCK_K),
                    num_stages=PIPE_STAGES,
                    loop_unroll_factor=1,
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
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
                out_rows = offs_m.to(tl.int64)
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
    assert group_A.stride(1) == 1 and group_B.stride(2) == 1
    assert group_out.stride(1) == 1
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]),
        meta["NUM_WM"],
        group_size,
    )
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
    group_C,
    group_out,
    group_list,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_om: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_BP: tl.constexpr = 0,
    PIPE_STAGES: tl.constexpr = 2,
    UNROLL: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    if USE_BP:
        end = tl.load(group_list + expert).to(tl.int32)
        start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(
            tl.int32
        )
        num_full = (end - start) // BLOCK_M
        offs_bn = tile_n * BLOCK_N
        b_base = group_B + expert.to(tl.int64) * (K * stride_bk)
        EVEN_KN: tl.constexpr = (N % BLOCK_N == 0) and (K % BLOCK_K == 0)
        for w in tl.range(window, num_full, NUM_WM, num_stages=1):
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
                offsets=(offs_bn, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if EVEN_KN:
                for k in tl.range(
                    0,
                    K // BLOCK_K,
                    num_stages=PIPE_STAGES,
                    loop_unroll_factor=UNROLL,
                ):
                    b = tl.trans(tl.load(b_ptrs, eviction_policy="evict_last"))
                    a = tl.load(a_ptrs)
                    accumulator = tl.dot(
                        a, b, acc=accumulator, out_dtype=tl.float32
                    )
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
            else:
                for k in tl.range(
                    0,
                    tl.cdiv(K, BLOCK_K),
                    num_stages=PIPE_STAGES,
                    loop_unroll_factor=1,
                ):
                    b = tl.trans(
                        tl.load(
                            b_ptrs,
                            boundary_check=(0, 1),
                            eviction_policy="evict_last",
                        )
                    )
                    a = tl.load(a_ptrs, boundary_check=(0, 1))
                    accumulator = tl.dot(
                        a, b, acc=accumulator, out_dtype=tl.float32
                    )
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
            result = accumulator * alpha
            c_rows = offs.to(tl.int64)
            if beta != 0.0:
                c_ptrs = tl.make_block_ptr(
                    base=group_C + c_rows * stride_cm,
                    shape=(end - offs, N),
                    strides=(stride_cm, 1),
                    offsets=(0, offs_bn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                if EVEN_KN:
                    c = tl.load(c_ptrs)
                else:
                    c = tl.load(c_ptrs, boundary_check=(0, 1))
                result += c.to(tl.float32) * beta
            out_ptrs = tl.make_block_ptr(
                base=group_out + c_rows * stride_om,
                shape=(end - offs, N),
                strides=(stride_om, 1),
                offsets=(0, offs_bn),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            if EVEN_KN:
                tl.store(out_ptrs, result.to(tl.float16))
            else:
                tl.store(out_ptrs, result.to(tl.float16), boundary_check=(0, 1))
        rem = end - start - num_full * BLOCK_M
        if rem > 0 and window == num_full % NUM_WM:
            offs = start + num_full * BLOCK_M
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
                offsets=(offs_bn, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(
                0,
                tl.cdiv(K, BLOCK_K),
                num_stages=PIPE_STAGES,
                loop_unroll_factor=1,
            ):
                b = tl.trans(
                    tl.load(
                        b_ptrs,
                        boundary_check=(0, 1),
                        eviction_policy="evict_last",
                    )
                )
                a = tl.load(a_ptrs, boundary_check=(0, 1))
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
            result = accumulator * alpha
            if beta != 0.0:
                c_ptrs = tl.make_block_ptr(
                    base=group_C + offs.to(tl.int64) * stride_cm,
                    shape=(end - offs, N),
                    strides=(stride_cm, 1),
                    offsets=(0, offs_bn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                c = tl.load(c_ptrs, boundary_check=(0, 1))
                result += c.to(tl.float32) * beta
            out_ptrs = tl.make_block_ptr(
                base=group_out + offs.to(tl.int64) * stride_om,
                shape=(end - offs, N),
                strides=(stride_om, 1),
                offsets=(0, offs_bn),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            tl.store(out_ptrs, result.to(tl.float16), boundary_check=(0, 1))
        return
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
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=UNROLL
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K
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
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * stride_bk
        result = accumulator * alpha
        c_rows = offs_m.to(tl.int64)
        if beta != 0.0:
            c_ptrs = group_C + c_rows[:, None] * stride_cm + offs_n[None, :]
            if N % BLOCK_N == 0:
                c = tl.load(c_ptrs)
            else:
                c = tl.load(c_ptrs, mask=offs_n[None, :] < N, other=0)
            result += c.to(tl.float32) * beta
        out_ptrs = group_out + c_rows[:, None] * stride_om + offs_n[None, :]
        if N % BLOCK_N == 0:
            tl.store(out_ptrs, result.to(tl.float16))
        else:
            tl.store(out_ptrs, result.to(tl.float16), mask=offs_n[None, :] < N)
    rem = end - start - num_full * BLOCK_M
    if rem > 0 and window == num_full % NUM_WM:
        BLOCK_H: tl.constexpr = BLOCK_M // 2
        for h in tl.static_range(0, 2):
            if h * BLOCK_H < rem:
                offs = start + num_full * BLOCK_M + h * BLOCK_H
                offs_m = offs + tl.arange(0, BLOCK_H)
                a_base = group_A + offs.to(tl.int64) * stride_am
                a_ptrs = (
                    a_base
                    + tl.arange(0, BLOCK_H)[:, None] * stride_am
                    + offs_k[None, :]
                )
                b_ptrs = (
                    b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                )
                accumulator = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
                for k in tl.range(
                    0,
                    tl.cdiv(K, BLOCK_K),
                    num_stages=PIPE_STAGES,
                    loop_unroll_factor=1,
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
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K
                    b_ptrs += BLOCK_K * stride_bk
                result = accumulator * alpha
                c_rows = offs_m.to(tl.int64)
                store_mask = (offs_m[:, None] < end) & (offs_n[None, :] < N)
                if beta != 0.0:
                    c_ptrs = group_C + c_rows[:, None] * stride_cm + offs_n[None, :]
                    c = tl.load(c_ptrs, mask=store_mask, other=0)
                    result += c.to(tl.float32) * beta
                out_ptrs = group_out + c_rows[:, None] * stride_om + offs_n[None, :]
                tl.store(out_ptrs, result.to(tl.float16), mask=store_mask)


def group_hgemm(
    group_A,
    group_B,
    group_C,
    group_list,
    group_out,
    alpha=1.0,
    beta=0.0,
):
    M, K = group_A.shape
    N = group_B.shape[1]
    group_size = group_list.numel()
    assert group_B.shape == (group_size * K, N)
    assert group_C.shape == (M, N) and group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]),
        meta["NUM_WM"],
        group_size,
    )
    grouped_hgemm_kernel[grid](
        group_A,
        group_B,
        group_C,
        group_out,
        group_list,
        M,
        N,
        K,
        group_size,
        group_A.stride(0),
        group_B.stride(0),
        group_B.stride(1),
        group_C.stride(0),
        group_out.stride(0),
        alpha=alpha,
        beta=beta,
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
    group_C,
    group_out,
    group_list,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_om: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_BP: tl.constexpr = 0,
    PIPE_STAGES: tl.constexpr = 2,
    UNROLL: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    if USE_BP:
        end = tl.load(group_list + expert).to(tl.int32)
        start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(
            tl.int32
        )
        num_tiles = tl.cdiv(end - start, BLOCK_M)
        offs_bn = tile_n * BLOCK_N
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
                offsets=(0, offs_bn),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(
                0,
                tl.cdiv(K, BLOCK_K),
                num_stages=PIPE_STAGES,
                loop_unroll_factor=UNROLL,
            ):
                b = tl.load(
                    b_ptrs, boundary_check=(0, 1), eviction_policy="evict_last"
                )
                a = tl.load(a_ptrs, boundary_check=(0, 1))
                accumulator = tl.dot(a, b, acc=accumulator)
                a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                b_ptrs = tl.advance(b_ptrs, (BLOCK_K, 0))
            result = accumulator * alpha
            if beta != 0.0:
                c_ptrs = tl.make_block_ptr(
                    base=group_C + offs.to(tl.int64) * stride_cm,
                    shape=(end - offs, N),
                    strides=(stride_cm, 1),
                    offsets=(0, offs_bn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                c = tl.load(c_ptrs, boundary_check=(0, 1))
                result += c * beta
            out_ptrs = tl.make_block_ptr(
                base=group_out + offs.to(tl.int64) * stride_om,
                shape=(end - offs, N),
                strides=(stride_om, 1),
                offsets=(0, offs_bn),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            tl.store(out_ptrs, result, boundary_check=(0, 1))
        return
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
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=UNROLL
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator)
                a_ptrs += BLOCK_K
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
                accumulator = tl.dot(a, b, acc=accumulator)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * stride_bk
        result = accumulator * alpha
        c_rows = offs_m.to(tl.int64)
        if beta != 0.0:
            c_ptrs = group_C + c_rows[:, None] * stride_cm + offs_n[None, :]
            if N % BLOCK_N == 0:
                c = tl.load(c_ptrs)
            else:
                c = tl.load(c_ptrs, mask=offs_n[None, :] < N, other=0)
            result += c * beta
        out_ptrs = group_out + c_rows[:, None] * stride_om + offs_n[None, :]
        if N % BLOCK_N == 0:
            tl.store(out_ptrs, result)
        else:
            tl.store(out_ptrs, result, mask=offs_n[None, :] < N)
    rem = end - start - num_full * BLOCK_M
    if rem > 0 and window == num_full % NUM_WM:
        BLOCK_H: tl.constexpr = BLOCK_M // 2
        for h in tl.static_range(0, 2):
            if h * BLOCK_H < rem:
                offs = start + num_full * BLOCK_M + h * BLOCK_H
                offs_m = offs + tl.arange(0, BLOCK_H)
                a_base = group_A + offs.to(tl.int64) * stride_am
                a_ptrs = (
                    a_base
                    + tl.arange(0, BLOCK_H)[:, None] * stride_am
                    + offs_k[None, :]
                )
                b_ptrs = (
                    b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                )
                accumulator = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
                for k in tl.range(
                    0,
                    tl.cdiv(K, BLOCK_K),
                    num_stages=PIPE_STAGES,
                    loop_unroll_factor=1,
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
                result = accumulator * alpha
                c_rows = offs_m.to(tl.int64)
                store_mask = (offs_m[:, None] < end) & (offs_n[None, :] < N)
                if beta != 0.0:
                    c_ptrs = group_C + c_rows[:, None] * stride_cm + offs_n[None, :]
                    c = tl.load(c_ptrs, mask=store_mask, other=0)
                    result += c * beta
                out_ptrs = group_out + c_rows[:, None] * stride_om + offs_n[None, :]
                tl.store(out_ptrs, result, mask=store_mask)


def group_tf32gemm(
    group_A,
    group_B,
    group_C,
    group_list,
    group_out,
    alpha=1.0,
    beta=0.0,
):
    M, K = group_A.shape
    N = group_out.shape[1]
    group_size = group_list.numel()
    assert group_C.shape == (M, N) and group_out.shape == (M, N)
    assert group_B.shape == (group_size * N, K)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]),
        meta["NUM_WM"],
        group_size,
    )
    grouped_tf32gemm_kernel[grid](
        group_A,
        group_B,
        group_C,
        group_out,
        group_list,
        M,
        N,
        K,
        group_size,
        group_A.stride(0),
        group_B.stride(1),
        group_B.stride(0),
        group_C.stride(0),
        group_out.stride(0),
        alpha=alpha,
        beta=beta,
    )
    return group_out

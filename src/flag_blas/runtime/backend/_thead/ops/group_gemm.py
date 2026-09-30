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
    "grouped_tf32gemm_tn_kernel",
    "grouped_tf32gemm_transpose_b_kernel",
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
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in range(window, num_tiles, NUM_WM):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        row_mask = offs_m < end
        ld_rows = tl.minimum(tl.arange(0, BLOCK_M), end - 1 - offs)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = a_base + ld_rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in range(0, K // BLOCK_K):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(
                    a_ptrs,
                    mask=(offs_k[None, :] + k * BLOCK_K < K),
                    other=0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                )
                accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        out_mask = row_mask[:, None] & (offs_n[None, :] < N)
        tl.store(out_ptrs, accumulator.to(group_out.dtype.element_ty), mask=out_mask)


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
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in range(window, num_tiles, NUM_WM):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        row_mask = offs_m < end
        ld_rows = tl.minimum(tl.arange(0, BLOCK_M), end - 1 - offs)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = a_base + ld_rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in range(0, K // BLOCK_K):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(
                    a_ptrs,
                    mask=(offs_k[None, :] + k * BLOCK_K < K),
                    other=0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                )
                accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        out_mask = row_mask[:, None] & (offs_n[None, :] < N)
        tl.store(out_ptrs, accumulator.to(group_out.dtype.element_ty), mask=out_mask)


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
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in range(window, num_tiles, NUM_WM):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        row_mask = offs_m < end
        ld_rows = tl.minimum(tl.arange(0, BLOCK_M), end - 1 - offs)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = a_base + ld_rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in range(0, K // BLOCK_K):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator = tl.dot(
                    a, b, accumulator, out_dtype=tl.float32, input_precision="tf32"
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(
                    a_ptrs,
                    mask=(offs_k[None, :] + k * BLOCK_K < K),
                    other=0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                )
                accumulator = tl.dot(
                    a, b, accumulator, out_dtype=tl.float32, input_precision="tf32"
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        out_mask = row_mask[:, None] & (offs_n[None, :] < N)
        tl.store(out_ptrs, accumulator.to(group_out.dtype.element_ty), mask=out_mask)


@triton.jit
def grouped_tf32gemm_transpose_b_kernel(
    group_B,
    group_B_t,
    n_off,
    n_span,
    N,
    K,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_te: tl.constexpr,
    stride_tn: tl.constexpr,
    stride_tk: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    expert = tl.program_id(2)
    offs_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = n_off + offs_local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_local[:, None] < n_span) & (offs_k[None, :] < K)
    src = (
        group_B
        + expert.to(tl.int64) * stride_be
        + offs_k[None, :] * stride_bk
        + offs_n[:, None] * stride_bn
    )
    dst = (
        group_B_t
        + expert.to(tl.int64) * stride_te
        + offs_n[:, None] * stride_tn
        + offs_k[None, :] * stride_tk
    )
    val = tl.load(src, mask=mask, other=0)
    tl.store(dst, val, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm_tn"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_tf32gemm_tn_kernel(
    group_A,
    group_B_t,
    group_list,
    group_out,
    M,
    n_off,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EXPERTS_PER_CTA: tl.constexpr,
):
    # Block-pointer addressing: raw int64 pointer tiles push the register
    # allocator to its 256-thread limit on wide-BLOCK_N tiles (measured
    # spills), while block pointers defer address arithmetic to the load
    # and keep the wide tiles viable.
    # EXPERTS_PER_CTA > 1 makes a CTA walk several experts' M-tiles, so the
    # per-tile CTA prologue amortizes across tiles (few-expert shapes have
    # ~1 tile per window otherwise); EXPERTS_PER_CTA = 1 keeps one CTA per
    # expert (grid z axis = group_size).
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    e_lo = tl.program_id(2) * EXPERTS_PER_CTA
    e_hi = tl.minimum(e_lo + EXPERTS_PER_CTA, group_size)
    offs_bn = n_off + tile_n * BLOCK_N
    for expert in range(e_lo, e_hi):
        end = tl.load(group_list + expert).to(tl.int32)
        start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
        num_tiles = tl.cdiv(end - start, BLOCK_M)
        a_base = group_A + start.to(tl.int64) * stride_am
        b_base = group_B_t + expert.to(tl.int64) * stride_be
        out_base = group_out + start.to(tl.int64) * stride_om
        for w in range(window, num_tiles, NUM_WM):
            a_ptrs = tl.make_block_ptr(
                base=a_base,
                shape=(end - start, K),
                strides=(stride_am, stride_ak),
                offsets=(w * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0),
            )
            b_ptrs = tl.make_block_ptr(
                base=b_base,
                shape=(N, K),
                strides=(stride_bn, stride_bk),
                offsets=(offs_bn, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs, boundary_check=(0, 1), padding_option="zero")
                b = tl.load(b_ptrs, boundary_check=(0, 1), padding_option="zero")
                accumulator = tl.dot(
                    a,
                    b.T,
                    accumulator,
                    out_dtype=tl.float32,
                    input_precision="tf32",
                )
                a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))
            out_ptrs = tl.make_block_ptr(
                base=out_base,
                shape=(end - start, N),
                strides=(stride_om, stride_on),
                offsets=(w * BLOCK_M, offs_bn),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            c = accumulator.to(group_out.dtype.element_ty)
            tl.store(out_ptrs, c, boundary_check=(0, 1))


def group_tf32gemm(group_A, group_B, group_list, group_out):
    assert group_A.ndim == 2 and group_B.ndim == 3 and group_list.ndim == 1
    M, K = group_A.shape
    group_size, BK, N = group_B.shape
    assert BK == K
    assert group_list.numel() == group_size
    assert group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    # The n-major B layout feeds the tf32 MMA noticeably better than the
    # k-major direct path on wide-N shapes; transposing B first is a net
    # win there (measured on core shapes), so keep the direct path for
    # narrow-N and tiny-K shapes. (A side-stream strip-pipelined transpose
    # was measured slower on PPU: the DRAM contention costs more than the
    # overlap saves.) A B tensor whose K axis is already contiguous is
    # n-major storage already and is consumed directly, skipping the
    # transpose entirely.
    if (2048 <= K <= 4096 and N >= 1024) or (K >= 512 and N >= 2048):
        if group_B.stride(1) == 1:
            # n-major storage viewed as [E, K, N]: the kernel's [E, N, K]
            # strides need the K/N axes swapped.
            group_B_t = group_B
            b_strides = (group_B.stride(0), group_B.stride(2), group_B.stride(1))
        else:
            group_B_t = torch.empty(
                (group_size, N, K), dtype=group_B.dtype, device=group_B.device
            )
            grouped_tf32gemm_transpose_b_kernel[
                (triton.cdiv(N, 64), triton.cdiv(K, 64), group_size)
            ](
                group_B,
                group_B_t,
                0,
                N,
                N,
                K,
                *group_B.stride(),
                *group_B_t.stride(),
                BLOCK_N=64,
                BLOCK_K=64,
                num_warps=4,
                num_stages=2,
            )
            b_strides = tuple(group_B_t.stride())
        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            meta["NUM_WM"],
            triton.cdiv(group_size, meta["EXPERTS_PER_CTA"]),
        )
        grouped_tf32gemm_tn_kernel[grid](
            group_A,
            group_B_t,
            group_list,
            group_out,
            M,
            0,
            N,
            K,
            group_size,
            *group_A.stride(),
            *b_strides,
            *group_out.stride(),
        )
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
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
    return group_out

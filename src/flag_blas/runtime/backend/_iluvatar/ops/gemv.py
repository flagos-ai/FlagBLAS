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

import importlib
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils import triton_lang_extension as tle

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

SPLITK_M_THRESHOLD = 64
SPLITK_K_THRESHOLD = 4096

_ILUVATAR_SGEMV_T64_MIN_K = 65536
_ILUVATAR_SGEMV_T64_MAX_K = 131072
_ILUVATAR_SGEMV_T64_SPLITS = 64
_ILUVATAR_SGEMV_N_SMALL_MIN_M = 4096
_ILUVATAR_SGEMV_T_COALESCED_MIN_DIM = 512
_ILUVATAR_SGEMV_T_COALESCED_MAX_M = 4096
_ILUVATAR_SGEMV_T_COALESCED_MAX_N = 1024

_common = importlib.import_module("flag_blas.ops.level2.gemv")


@libentry()
@triton.jit
def sgemv_n_small_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    BETA_IS_ZERO: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    y_ptrs = y_ptr + rows

    if N_CONST == 1:
        a = tl.load(a_ptr + rows, mask=row_mask, other=0.0)
        x = tl.load(x_ptr)
        acc = a * x
    else:
        ks = tl.arange(0, BLOCK_SIZE_K)
        k_mask = ks < N_CONST
        a = tl.load(
            a_ptr + rows[:, None] * N_CONST + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        x = tl.load(x_ptr + ks, mask=k_mask, other=0.0)
        acc = tl.sum(a * x[None, :], axis=1)

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * old_y
    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@triton.jit
def sgemv_t_coalesced_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    reduce_k,
    stride_ak,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < out_m
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_offsets = tl.max_contiguous(
        tl.multiple_of(k_offsets, BLOCK_SIZE_K), BLOCK_SIZE_K
    )
    a_ptrs = a_ptr + k_offsets[:, None] * stride_ak + rows[None, :]
    x_ptrs = x_ptr + k_offsets
    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_start in range(0, reduce_k, BLOCK_SIZE_K):
        ks = k_start + k_offsets
        k_mask = ks < reduce_k
        a = tl.trans(
            tl.load(
                a_ptrs,
                mask=k_mask[:, None] & row_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
        )
        x = tl.load(
            x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last"
        )
        acc_2d += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        x_ptrs += BLOCK_SIZE_K

    acc = tl.sum(acc_2d, axis=1)
    y_ptrs = y_ptr + rows
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * old_y
    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@triton.jit
def sgemv_t_splitk_partial_iluvatar_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_m,
    reduce_k,
    stride_ak,
    INCX,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < out_m
    chunk_k = (reduce_k + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, reduce_k)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (k_begin + k_offsets)[:, None] * stride_ak + rows[None, :]
    x_ptrs = x_ptr + (k_begin + k_offsets) * INCX
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_offsets
        k_mask = ks < k_end
        mask = k_mask[:, None] & row_mask[None, :]
        a = tl.trans(tl.load(a_ptrs, mask=mask, other=0.0))
        x = tl.load(x_ptrs, mask=k_mask, other=0.0)
        acc += tl.sum(a * x[None, :], axis=1)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        x_ptrs += BLOCK_SIZE_K * INCX

    tl.store(partial_ptr + pid_k * out_m + rows, acc, mask=row_mask)


@libentry()
@triton.jit
def sgemv_t_splitk_2d_partial_iluvatar_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_m,
    reduce_k,
    stride_ak,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < out_m
    chunk_k = (reduce_k + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, reduce_k)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (k_begin + k_offsets)[:, None] * stride_ak + rows[None, :]
    x_ptrs = x_ptr + k_begin + k_offsets
    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_offsets
        k_mask = ks < k_end
        a = tl.trans(
            tl.load(
                a_ptrs,
                mask=k_mask[:, None] & row_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
        )
        x = tl.load(
            x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last"
        )
        acc_2d += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        x_ptrs += BLOCK_SIZE_K

    acc = tl.sum(acc_2d, axis=1)
    tl.store(partial_ptr + pid_k * out_m + rows, acc, mask=row_mask)


@libentry()
@triton.jit
def sgemv_t_aligned_split2_partial_iluvatar_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_m,
    reduce_k,
    stride_ak,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    chunk_k = reduce_k // 2
    k_begin = pid_k * chunk_k
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (k_begin + k_offsets)[:, None] * stride_ak + rows[None, :]
    x_ptrs = x_ptr + k_begin + k_offsets
    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for _ in range(0, chunk_k, BLOCK_SIZE_K):
        a = tl.trans(tl.load(a_ptrs, eviction_policy="evict_first"))
        x = tl.load(x_ptrs, eviction_policy="evict_last")
        acc_2d += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        x_ptrs += BLOCK_SIZE_K

    acc = tl.sum(acc_2d, axis=1)
    tl.store(partial_ptr + pid_k * out_m + rows, acc)


@libentry()
@triton.jit
def sgemv_splitk_reduce_iluvatar_kernel(
    partial_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < out_m
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for split_begin in range(0, SPLIT_COUNT, BLOCK_SPLITS):
        splits = split_begin + split_offsets
        values = tl.load(
            partial_ptr + rows[:, None] + splits[None, :] * out_m,
            mask=row_mask[:, None],
            other=0.0,
        )
        acc += tl.sum(values, axis=1)

    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * old_y
    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@triton.jit
def sgemv_split2_reduce_aligned_iluvatar_kernel(
    partial_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    acc = tl.load(partial_ptr + rows) + tl.load(partial_ptr + out_m + rows)
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        result = alpha * acc + beta * tl.load(y_ptr + rows)
    tl.store(y_ptr + rows, result)


@libentry()
@triton.jit
def hgemv_t_aligned_splitk_partial_iluvatar_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_m,
    reduce_k,
    stride_ak,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M
    )
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_offsets = tl.max_contiguous(
        tl.multiple_of(k_offsets, BLOCK_SIZE_K), BLOCK_SIZE_K
    )
    chunk_k = reduce_k // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    a_ptrs = a_ptr + (k_begin + k_offsets)[:, None] * stride_ak + rows[None, :]
    x_ptrs = x_ptr + k_begin + k_offsets
    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for _ in range(0, chunk_k, BLOCK_SIZE_K):
        a = tl.trans(
            tl.load(a_ptrs, eviction_policy="evict_first")
        ).to(tl.float32)
        x = tl.load(x_ptrs, eviction_policy="evict_last").to(tl.float32)
        acc_2d += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        x_ptrs += BLOCK_SIZE_K

    acc = tl.sum(acc_2d, axis=1)
    tl.store(partial_ptr + pid_k * out_m + rows, acc)


@libentry()
@triton.jit
def hgemv_t_aligned_splitk_reduce_iluvatar_kernel(
    partial_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    BETA_IS_ZERO: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M
    )
    split_offsets = tl.arange(0, SPLIT_COUNT)
    partials = tl.load(
        partial_ptr + split_offsets[:, None] * out_m + rows[None, :]
    )
    acc = tl.sum(partials, axis=0)
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        result = alpha * acc + beta * tl.load(y_ptr + rows).to(tl.float32)
    tl.store(y_ptr + rows, result.to(tl.float16))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemv_t_tcu_iluvatar"),
    key=["out_n", "reduce_k"],
    restore_value=["y_ptr"],
)
@triton.jit
def hgemv_t_tcu_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_n,
    reduce_k,
    stride_ak,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """FP16 TCU GEMV with FP32 accumulation; TF32 is not involved."""
    rows = tle.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    dot_rows = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in tl.range(
        0, reduce_k, BLOCK_SIZE_K, num_stages=PIPE_STAGES
    ):
        ks = k_start + ks0
        a = tl.load(
            a_ptr + ks[:, None] * stride_ak + rows[None, :],
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptr + ks, eviction_policy="evict_last")
        x_tile = tl.where(dot_rows[:, None] == 0, x[None, :], 0.0)
        # Both dot operands are FP16, so TF32 is not applicable. The default
        # CoreX lowering is substantially faster than its explicit IEEE path.
        acc = tl.dot(x_tile, a, acc, out_dtype=tl.float32)

    result = tl.sum(acc, axis=0)
    if BETA_IS_ZERO:
        result *= alpha
    else:
        old_y = tl.load(y_ptr + rows).to(tl.float32)
        result = alpha * result + beta * old_y
    tl.store(y_ptr + rows, result.to(tl.float16))


@libentry()
@triton.jit
def hgemv_t_highsplit_partial_iluvatar_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_n,
    reduce_k,
    stride_ak,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid_n = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    chunk_k = (reduce_k + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, reduce_k)
    a_ptrs = a_ptr + (k_begin + ks0)[:, None] * stride_ak + rows[None, :]
    x_ptrs = x_ptr + k_begin + ks0
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in tl.range(
        0, chunk_k, BLOCK_SIZE_K, num_stages=PIPE_STAGES
    ):
        ks = k_begin + k_offset + ks0
        k_mask = ks < k_end
        a = tl.trans(
            tl.load(
                a_ptrs,
                mask=k_mask[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
        ).to(tl.float32)
        x = tl.load(
            x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        acc += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        x_ptrs += BLOCK_SIZE_K

    tl.store(
        partial_ptr + pid_k * out_n + rows, tl.sum(acc, axis=1)
    )


@libentry()
@triton.jit
def hgemv_t_stream_split_partial_iluvatar_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_n,
    reduce_k,
    stride_ak,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    UNROLL_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """Long-K split path with one FP32 accumulator per output."""
    pid_n = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    chunk_k = (reduce_k + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, reduce_k)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k_offset in tl.range(
        0, chunk_k, UNROLL_K, num_stages=PIPE_STAGES
    ):
        for j in tl.static_range(0, UNROLL_K):
            k = k_begin + k_offset + j
            k_mask = k < k_end
            a = tl.load(
                a_ptr + k * stride_ak + rows,
                mask=k_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            x = tl.load(
                x_ptr + k,
                mask=k_mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            acc += a * x

    tl.store(partial_ptr + pid_k * out_n + rows, acc)


@libentry()
@triton.jit
def hgemv_t_highsplit_reduce_iluvatar_kernel(
    partial_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_n,
    BETA_IS_ZERO: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for split_base in range(0, SPLIT_COUNT, BLOCK_SPLITS):
        splits = split_base + split_offsets
        values = tl.load(
            partial_ptr + splits[:, None] * out_n + rows[None, :],
            mask=splits[:, None] < SPLIT_COUNT,
            other=0.0,
        )
        acc += tl.sum(values, axis=0)
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptr + rows).to(tl.float32)
        result = alpha * acc + beta * old_y
    tl.store(y_ptr + rows, result.to(tl.float16))


@libentry()
@triton.jit
def bfgemv_t_tcu_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_n,
    reduce_k,
    stride_ak,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    dot_rows = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in tl.range(
        0, reduce_k, BLOCK_SIZE_K, num_stages=PIPE_STAGES
    ):
        ks = k_start + ks0
        a = tl.load(
            a_ptr + ks[:, None] * stride_ak + rows[None, :],
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptr + ks, eviction_policy="evict_last")
        x_tile = tl.where(
            dot_rows[:, None] == 0, x[None, :], 0.0
        )
        acc = tl.dot(x_tile, a, acc, out_dtype=tl.float32)

    result = tl.sum(acc, axis=0)
    if BETA_IS_ZERO:
        result *= alpha
    else:
        old_y = tl.load(y_ptr + rows).to(tl.float32)
        result = alpha * result + beta * old_y
    tl.store(y_ptr + rows, result.to(tl.bfloat16))


@libentry()
@triton.jit
def bfgemv_n_tcu_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    reduce_k,
    stride_am,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M
    )
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    dot_cols = tl.arange(0, 16)
    acc = tl.zeros((BLOCK_SIZE_M, 16), dtype=tl.float32)
    for k_start in tl.range(
        0, reduce_k, BLOCK_SIZE_K, num_stages=PIPE_STAGES
    ):
        ks = k_start + ks0
        a = tl.load(
            a_ptr + rows[:, None] * stride_am + ks[None, :],
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptr + ks, eviction_policy="evict_last")
        x_tile = tl.where(
            dot_cols[None, :] == 0, x[:, None], 0.0
        )
        acc = tl.dot(a, x_tile, acc, out_dtype=tl.float32)

    result = tl.sum(acc, axis=1)
    if BETA_IS_ZERO:
        result *= alpha
    else:
        old_y = tl.load(y_ptr + rows).to(tl.float32)
        result = alpha * result + beta * old_y
    tl.store(y_ptr + rows, result.to(tl.bfloat16))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemv_n_aligned"),
    key=["m", "n"],
    restore_value=["y_ptr"],
)
@triton.jit
def bfgemv_n_aligned_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    stride_am,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """Mask-free aligned BF16 GEMV with FP32 accumulation."""
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M
    )
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    ks0 = tl.max_contiguous(
        tl.multiple_of(ks0, BLOCK_SIZE_K), BLOCK_SIZE_K
    )
    a_ptrs = a_ptr + rows[:, None] * stride_am + ks0[None, :]
    x_ptrs = x_ptr + ks0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in tl.range(
        0, n, BLOCK_SIZE_K, num_stages=PIPE_STAGES
    ):
        a = tl.load(a_ptrs, eviction_policy="evict_first").to(tl.float32)
        x = tl.load(x_ptrs, eviction_policy="evict_last").to(tl.float32)
        acc += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K
        x_ptrs += BLOCK_SIZE_K
    result = tl.sum(acc, axis=1)
    if BETA_IS_ZERO:
        result *= alpha
    else:
        old_y = tl.load(y_ptr + rows).to(tl.float32)
        result = alpha * result + beta * old_y
    tl.store(y_ptr + rows, result.to(tl.bfloat16))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bfgemv_t_aligned"),
    key=["m", "n"],
    restore_value=["y_ptr"],
)
@triton.jit
def bfgemv_t_aligned_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    stride_ak,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """Mask-free aligned BF16 transposed GEMV with FP32 accumulation."""
    rows = tle.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    ks0 = tl.max_contiguous(
        tl.multiple_of(ks0, BLOCK_SIZE_K), BLOCK_SIZE_K
    )
    a_ptrs = a_ptr + ks0[:, None] * stride_ak + rows[None, :]
    x_ptrs = x_ptr + ks0
    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in tl.range(0, m, BLOCK_SIZE_K, num_stages=PIPE_STAGES):
        a = tl.load(a_ptrs, eviction_policy="evict_first").to(tl.float32)
        x = tl.load(x_ptrs, eviction_policy="evict_last").to(tl.float32)
        acc += a * x[:, None]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        x_ptrs += BLOCK_SIZE_K
    result = tl.sum(acc, axis=0)
    if BETA_IS_ZERO:
        result *= alpha
    else:
        old_y = tl.load(y_ptr + rows).to(tl.float32)
        result = alpha * result + beta * old_y
    tl.store(y_ptr + rows, result.to(tl.bfloat16))


@libentry()
@triton.jit
def bfgemv_t_stream_split_partial_iluvatar_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_n,
    reduce_k,
    stride_ak,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    UNROLL_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid_n = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rows = tl.max_contiguous(
        tl.multiple_of(rows, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    chunk_k = (reduce_k + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, reduce_k)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_offset in tl.range(
        0, chunk_k, UNROLL_K, num_stages=PIPE_STAGES
    ):
        for j in tl.static_range(0, UNROLL_K):
            k = k_begin + k_offset + j
            valid = k < k_end
            a = tl.load(
                a_ptr + k * stride_ak + rows,
                mask=valid,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            x = tl.load(
                x_ptr + k,
                mask=valid,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            acc += a * x
    tl.store(partial_ptr + pid_k * out_n + rows, acc)


@libentry()
@triton.jit
def bfgemv_split_reduce_iluvatar_kernel(
    partial_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_size,
    BETA_IS_ZERO: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split_base in range(0, SPLIT_COUNT, BLOCK_SPLITS):
        splits = split_base + split_offsets
        values = tl.load(
            partial_ptr + splits[:, None] * out_size + rows[None, :],
            mask=splits[:, None] < SPLIT_COUNT,
            other=0.0,
        )
        acc += tl.sum(values, axis=0)
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptr + rows).to(tl.float32)
        result = alpha * acc + beta * old_y
    tl.store(y_ptr + rows, result.to(tl.bfloat16))


def _use_iluvatar_sgemv_n_small(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_N
        and m >= _ILUVATAR_SGEMV_N_SMALL_MIN_M
        and 1 <= n <= 4
        and lda == n
        and incx == 1
        and incy == 1
    )


def _launch_iluvatar_sgemv_n_small(
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    x: torch.Tensor,
    beta: ScalarType,
    y: torch.Tensor,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.float32
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert A.device == x.device == y.device
    assert x.numel() >= n
    assert y.numel() >= m

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if n == 1:
        block_m, block_k, num_warps, num_stages = 256, 1, 4, 4
    elif n == 2:
        block_m, block_k, num_warps, num_stages = 64, 2, 1, 4
    elif n == 3:
        block_m, block_k, num_warps, num_stages = 512, 4, 8, 1
    else:
        block_m, block_k, num_warps, num_stages = 128, 4, 8, 4

    grid = (triton.cdiv(m, block_m),)
    with torch_device_fn.device(A.device):
        sgemv_n_small_iluvatar_kernel[grid](
            A,
            x,
            y,
            alpha,
            beta,
            m,
            beta == 0.0,
            N_CONST=n,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _use_iluvatar_sgemv_t64_splitk(trans: int, m: int, n: int) -> bool:
    """Use 64-way Split-K for the Iluvatar tall-by-64 transposed path."""
    return (
        trans == CUBLAS_OP_T
        and n == 64
        and _ILUVATAR_SGEMV_T64_MIN_K <= m < _ILUVATAR_SGEMV_T64_MAX_K
    )


def _use_iluvatar_sgemv_t_coalesced(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    """Use the contiguous-output loading layout for medium transposed GEMV."""
    return (
        trans == CUBLAS_OP_T
        and _ILUVATAR_SGEMV_T_COALESCED_MIN_DIM <= m
        <= _ILUVATAR_SGEMV_T_COALESCED_MAX_M
        and _ILUVATAR_SGEMV_T_COALESCED_MIN_DIM <= n
        <= _ILUVATAR_SGEMV_T_COALESCED_MAX_N
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_sgemv_t_large_splitk(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    """Use two-way Split-K for bandwidth-bound large transposed GEMV."""
    return (
        trans == CUBLAS_OP_T
        and m >= 8192
        and n >= 28672
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_sgemv_t_aligned_splitk(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    """Use the searched mask-free path for the aligned bandwidth-bound case."""
    return (
        trans == CUBLAS_OP_T
        and m == 3584
        and n == 18944
        and lda == n
        and incx == 1
        and incy == 1
    )


def sgemv(
    trans: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    """Iluvatar SGEMV overrides for specialized performance paths."""
    if _use_iluvatar_sgemv_n_small(trans, m, n, lda, incx, incy):
        _launch_iluvatar_sgemv_n_small(m, n, alpha, A, x, beta, y)
        return

    use_t_coalesced = _use_iluvatar_sgemv_t_coalesced(
        trans, m, n, lda, incx, incy
    )
    use_t_large_splitk = _use_iluvatar_sgemv_t_large_splitk(
        trans, m, n, lda, incx, incy
    )
    use_t_aligned_splitk = _use_iluvatar_sgemv_t_aligned_splitk(
        trans, m, n, lda, incx, incy
    )
    use_t64_splitk = _use_iluvatar_sgemv_t64_splitk(trans, m, n)
    if (
        not use_t_coalesced
        and not use_t_aligned_splitk
        and not use_t_large_splitk
        and not use_t64_splitk
    ):
        _common.sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        return

    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.float32
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert incx > 0 and incy > 0
    assert lda >= n
    assert x.numel() >= 1 + (m - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if use_t_coalesced:
        if m <= 512:
            num_stages = 4
        elif m <= 1024:
            num_stages = 3
        else:
            num_stages = 1
        grid = (triton.cdiv(n, 16),)
        with torch_device_fn.device(A.device):
            sgemv_t_coalesced_iluvatar_kernel[grid](
                A,
                x,
                y,
                alpha,
                beta,
                n,
                m,
                lda,
                beta == 0.0,
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_K=64,
                num_warps=16,
                num_stages=num_stages,
            )
        return

    if use_t_aligned_splitk:
        partial = torch.empty((2, n), dtype=torch.float32, device=A.device)
        partial_grid = (n // 64, 2)
        reduce_grid = (n // 32,)
        with torch_device_fn.device(A.device):
            sgemv_t_aligned_split2_partial_iluvatar_kernel[partial_grid](
                A,
                x,
                partial,
                n,
                m,
                lda,
                BLOCK_SIZE_M=64,
                BLOCK_SIZE_K=16,
                num_warps=16,
                num_stages=1,
            )
            sgemv_split2_reduce_aligned_iluvatar_kernel[reduce_grid](
                partial,
                y,
                alpha,
                beta,
                n,
                beta == 0.0,
                BLOCK_SIZE_M=32,
                num_warps=2,
                num_stages=1,
            )
        return

    if use_t_large_splitk:
        num_k_splits = 2
        partial = torch.empty(
            (num_k_splits, n), dtype=torch.float32, device=A.device
        )
        partial_grid = (triton.cdiv(n, 64), num_k_splits)
        reduce_grid = (triton.cdiv(n, 64),)
        with torch_device_fn.device(A.device):
            sgemv_t_splitk_2d_partial_iluvatar_kernel[partial_grid](
                A,
                x,
                partial,
                n,
                m,
                lda,
                SPLIT_COUNT=num_k_splits,
                BLOCK_SIZE_M=64,
                BLOCK_SIZE_K=16,
                num_warps=16,
                num_stages=2,
            )
            sgemv_splitk_reduce_iluvatar_kernel[reduce_grid](
                partial,
                y,
                alpha,
                beta,
                n,
                incy,
                beta == 0.0,
                SPLIT_COUNT=num_k_splits,
                BLOCK_SIZE_M=64,
                BLOCK_SPLITS=2,
                num_warps=4,
                num_stages=1,
            )
        return

    num_k_splits = _ILUVATAR_SGEMV_T64_SPLITS
    partial = torch.empty(
        (num_k_splits, n), dtype=torch.float32, device=A.device
    )
    partial_grid = (triton.cdiv(n, 64), num_k_splits)
    reduce_grid = (triton.cdiv(n, 32),)
    with torch_device_fn.device(A.device):
        sgemv_t_splitk_partial_iluvatar_kernel[partial_grid](
            A,
            x,
            partial,
            n,
            m,
            lda,
            incx,
            SPLIT_COUNT=num_k_splits,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_K=256,
            num_warps=4,
            num_stages=1,
        )
        sgemv_splitk_reduce_iluvatar_kernel[reduce_grid](
            partial,
            y,
            alpha,
            beta,
            n,
            incy,
            beta == 0.0,
            SPLIT_COUNT=num_k_splits,
            BLOCK_SIZE_M=32,
            BLOCK_SPLITS=16,
            num_warps=2,
            num_stages=1,
        )


def _use_iluvatar_hgemv_t_aligned_splitk(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    """Use the mask-free split-K path for searched regular FP16 shapes."""
    return (
        trans == CUBLAS_OP_T
        and m == 3584
        and n == 3584
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_hgemv_t_highsplit(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m == 4096
        and n == 1024
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_hgemv_t_stream_split(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_T
        and (
            (m == 18944 and n == 3584)
            or (m == 14336 and n == 4096)
        )
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_hgemv_t_tcu(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    """Use the FP16 TCU path selected from the vendor GEMV launch layout."""
    return (
        trans == CUBLAS_OP_T
        and m >= 1024
        and n >= 4096
        and m % 32 == 0
        and n % 128 == 0
        and not (n <= 4096 and m >= 2 * n)
        and lda == n
        and incx == 1
        and incy == 1
    )


def hgemv(
    trans: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    """Iluvatar HGEMV override for regular transposed bandwidth-bound shapes."""
    use_t_aligned_splitk = _use_iluvatar_hgemv_t_aligned_splitk(
        trans, m, n, lda, incx, incy
    )
    use_t_highsplit = _use_iluvatar_hgemv_t_highsplit(
        trans, m, n, lda, incx, incy
    )
    use_t_stream_split = _use_iluvatar_hgemv_t_stream_split(
        trans, m, n, lda, incx, incy
    )
    use_t_tcu = _use_iluvatar_hgemv_t_tcu(
        trans, m, n, lda, incx, incy
    )
    if (
        not use_t_aligned_splitk
        and not use_t_highsplit
        and not use_t_stream_split
        and not use_t_tcu
    ):
        _common.hgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        return

    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.float16
    assert x.dtype == torch.float16
    assert y.dtype == torch.float16
    assert A.device == x.device == y.device

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if use_t_highsplit:
        split_count = 144
        partial = torch.empty(
            (split_count, n), dtype=torch.float32, device=A.device
        )
        with torch_device_fn.device(A.device):
            hgemv_t_highsplit_partial_iluvatar_kernel[(8, split_count)](
                A,
                x,
                partial,
                n,
                m,
                lda,
                SPLIT_COUNT=split_count,
                BLOCK_SIZE_N=128,
                BLOCK_SIZE_K=16,
                PIPE_STAGES=4,
                num_warps=1,
                num_stages=1,
            )
            hgemv_t_highsplit_reduce_iluvatar_kernel[(16,)](
                partial,
                y,
                alpha,
                beta,
                n,
                beta == 0.0,
                SPLIT_COUNT=split_count,
                BLOCK_SIZE_N=64,
                BLOCK_SPLITS=16,
                num_warps=16,
                num_stages=1,
            )
        return

    if use_t_stream_split:
        split_count = 42 if m == 18944 else 40
        partial = torch.empty(
            (split_count, n), dtype=torch.float32, device=A.device
        )
        with torch_device_fn.device(A.device):
            hgemv_t_stream_split_partial_iluvatar_kernel[
                (n // 128, split_count)
            ](
                A,
                x,
                partial,
                n,
                m,
                lda,
                SPLIT_COUNT=split_count,
                BLOCK_SIZE_N=128,
                UNROLL_K=4,
                PIPE_STAGES=1,
                num_warps=1,
                num_stages=1,
            )
            hgemv_t_highsplit_reduce_iluvatar_kernel[(n // 64,)](
                partial,
                y,
                alpha,
                beta,
                n,
                beta == 0.0,
                SPLIT_COUNT=split_count,
                BLOCK_SIZE_N=64,
                BLOCK_SPLITS=16,
                num_warps=16,
                num_stages=1,
            )
        return

    if use_t_tcu:
        grid = lambda meta: (n // meta["BLOCK_SIZE_N"],)
        with torch_device_fn.device(A.device):
            hgemv_t_tcu_iluvatar_kernel[grid](
                A,
                x,
                y,
                alpha,
                beta,
                n,
                m,
                lda,
                beta == 0.0,
            )
        return

    split_count = 4
    block_m, block_k = 64, 16
    num_warps, num_stages = 8, 2
    reduce_block_m = block_m
    reduce_warps, reduce_stages = 4, 1

    partial = torch.empty(
        (split_count, n), dtype=torch.float32, device=A.device
    )
    partial_grid = (n // block_m, split_count)
    reduce_grid = (n // reduce_block_m,)
    with torch_device_fn.device(A.device):
        hgemv_t_aligned_splitk_partial_iluvatar_kernel[partial_grid](
            A,
            x,
            partial,
            n,
            m,
            lda,
            SPLIT_COUNT=split_count,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        hgemv_t_aligned_splitk_reduce_iluvatar_kernel[reduce_grid](
            partial,
            y,
            alpha,
            beta,
            n,
            beta == 0.0,
            SPLIT_COUNT=split_count,
            BLOCK_SIZE_M=reduce_block_m,
            num_warps=reduce_warps,
            num_stages=reduce_stages,
        )


def _use_iluvatar_bfgemv_t_stream_split(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_T
        and (m, n)
        in {
            (18944, 3584),
            (14336, 4096),
        }
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_bfgemv_t_tcu(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m >= 1024
        and n >= 3584
        and m % 32 == 0
        and n % 32 == 0
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_bfgemv_t_aligned(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_T
        and (m, n) in {(1024, 1024), (4096, 1024)}
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_bfgemv_n_tcu(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_N
        and (m, n) in {(18944, 3584), (14336, 4096)}
        and lda == n
        and incx == 1
        and incy == 1
    )


def _use_iluvatar_bfgemv_n_aligned(
    trans: int, m: int, n: int, lda: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_N
        and (m, n)
        in {
            (3584, 3584),
            (4096, 1024),
            (4096, 4096),
            (6144, 16384),
            (7168, 7168),
            (7168, 18432),
            (8192, 8192),
            (16384, 6144),
            (16384, 16384),
            (18432, 7168),
            (18432, 18432),
            (28672, 8192),
            (53248, 16384),
            (3584, 18944),
        }
        and lda == n
        and incx == 1
        and incy == 1
    )


def bfgemv(
    trans: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    """Iluvatar BF16 GEMV overrides for aligned performance paths."""
    use_t_stream = _use_iluvatar_bfgemv_t_stream_split(
        trans, m, n, lda, incx, incy
    )
    use_t_tcu = _use_iluvatar_bfgemv_t_tcu(
        trans, m, n, lda, incx, incy
    )
    use_t_aligned = _use_iluvatar_bfgemv_t_aligned(
        trans, m, n, lda, incx, incy
    )
    use_n_tcu = _use_iluvatar_bfgemv_n_tcu(
        trans, m, n, lda, incx, incy
    )
    use_n_aligned = _use_iluvatar_bfgemv_n_aligned(
        trans, m, n, lda, incx, incy
    )
    if (
        not use_t_stream
        and not use_t_tcu
        and not use_t_aligned
        and not use_n_tcu
        and not use_n_aligned
    ):
        _common.bfgemv(
            trans, m, n, alpha, A, lda, x, incx, beta, y, incy
        )
        return

    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert y.dtype == torch.bfloat16
    assert A.device == x.device == y.device

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if use_t_stream:
        if (m, n) == (18944, 3584):
            split_count, unroll_k, pipe_stages = 48, 8, 4
        else:
            split_count, unroll_k, pipe_stages = 48, 8, 1
        partial = torch.empty(
            (split_count, n), dtype=torch.float32, device=A.device
        )
        with torch_device_fn.device(A.device):
            bfgemv_t_stream_split_partial_iluvatar_kernel[
                (n // 128, split_count)
            ](
                A,
                x,
                partial,
                n,
                m,
                lda,
                SPLIT_COUNT=split_count,
                BLOCK_SIZE_N=128,
                UNROLL_K=unroll_k,
                PIPE_STAGES=pipe_stages,
                num_warps=1,
                num_stages=1,
            )
            bfgemv_split_reduce_iluvatar_kernel[(n // 64,)](
                partial,
                y,
                alpha,
                beta,
                n,
                beta == 0.0,
                SPLIT_COUNT=split_count,
                BLOCK_SIZE=64,
                BLOCK_SPLITS=16,
                num_warps=16,
                num_stages=1,
            )
        return

    if use_t_aligned:
        grid = lambda meta: (n // meta["BLOCK_SIZE_N"],)
        with torch_device_fn.device(A.device):
            bfgemv_t_aligned_iluvatar_kernel[grid](
                A,
                x,
                y,
                alpha,
                beta,
                m,
                n,
                lda,
                beta == 0.0,
            )
        return

    if use_n_tcu:
        pipe_stages = 4 if n == 4096 else 2
        grid = (m // 128,)
        with torch_device_fn.device(A.device):
            bfgemv_n_tcu_iluvatar_kernel[grid](
                A,
                x,
                y,
                alpha,
                beta,
                m,
                n,
                lda,
                beta == 0.0,
                BLOCK_SIZE_M=128,
                BLOCK_SIZE_K=32,
                PIPE_STAGES=pipe_stages,
                num_warps=4,
                num_stages=1,
            )
        return

    if use_n_aligned:
        grid = lambda meta: (m // meta["BLOCK_SIZE_M"],)
        with torch_device_fn.device(A.device):
            bfgemv_n_aligned_iluvatar_kernel[grid](
                A,
                x,
                y,
                alpha,
                beta,
                m,
                n,
                lda,
                beta == 0.0,
            )
        return

    if m <= 4096:
        block_n, num_warps = 32, 2
        pipe_stages = 2 if n == 3584 else 1
    elif (m, n) == (16384, 6144):
        block_n, num_warps, pipe_stages = 32, 2, 1
    elif m >= 2 * n:
        block_n, num_warps, pipe_stages = 32, 1, 3
    elif m >= 16384 and n >= 16384 and n % 64 == 0:
        block_n, num_warps, pipe_stages = 64, 4, 4
    else:
        block_n, num_warps, pipe_stages = 32, 2, 3
    grid = (n // block_n,)
    with torch_device_fn.device(A.device):
        bfgemv_t_tcu_iluvatar_kernel[grid](
            A,
            x,
            y,
            alpha,
            beta,
            n,
            m,
            lda,
            beta == 0.0,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=32,
            PIPE_STAGES=pipe_stages,
            num_warps=num_warps,
            num_stages=1,
        )


@triton.jit
def _fp8_bits_to_float32(raw, IS_E4M3: tl.constexpr):
    """Decode FP8 bytes without exposing an FP8 pointer to the CoreX compiler."""
    raw_i = raw.to(tl.int32)
    sign = tl.where((raw_i & 0x80) != 0, -1.0, 1.0)

    if IS_E4M3:
        exponent = (raw_i >> 3) & 0x0F
        mantissa = raw_i & 0x07
        normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
            exponent.to(tl.float32) - 7.0
        )
        subnormal = mantissa.to(tl.float32) * (1.0 / 512.0)
        value = tl.where(exponent == 0, subnormal, normal)
        value = tl.where(
            (exponent == 15) & (mantissa == 7), float("nan"), value
        )
    else:
        exponent = (raw_i >> 2) & 0x1F
        mantissa = raw_i & 0x03
        normal = (1.0 + mantissa.to(tl.float32) * 0.25) * tl.exp2(
            exponent.to(tl.float32) - 15.0
        )
        subnormal = mantissa.to(tl.float32) * (1.0 / 65536.0)
        value = tl.where(exponent == 0, subnormal, normal)
        special = tl.where(mantissa == 0, float("inf"), float("nan"))
        value = tl.where(exponent == 31, special, value)

    return sign * value


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8_gemv"),
    key=["m", "n"],
    restore_value=["y_ptr"],
)
@triton.jit
def fp8_gemv_iluvatar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    STRIDE_AK,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    A_IS_E4M3: tl.constexpr,
    X_IS_E4M3: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + k_offsets_init[:, None] * STRIDE_AK + row_offsets[None, :]
    x_ptrs = x_ptr + k_offsets_init * INCX

    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    step_a = BLOCK_SIZE_K * STRIDE_AK
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = k_mask[:, None] & row_mask[None, :]

        a_raw = tl.load(a_ptrs, mask=a_mask, other=0, eviction_policy="evict_first")
        a_block = tl.trans(_fp8_bits_to_float32(a_raw, A_IS_E4M3))
        x_raw = tl.load(x_ptrs, mask=k_mask, other=0, eviction_policy="evict_last")
        x_block = _fp8_bits_to_float32(x_raw, X_IS_E4M3)

        acc_2d += a_block * x_block[None, :]
        a_ptrs += step_a
        x_ptrs += step_x

    acc = tl.sum(acc_2d, axis=1)
    y_ptrs = y_ptr + row_offsets * INCY

    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        y_vals = tl.load(y_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * y_vals

    tl.store(y_ptrs, result.to(y_ptr.dtype.element_ty), mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8_gemv_splitk"),
    key=["m", "n"],
    restore_value=["y_ptr"],
)
@triton.jit
def fp8_gemv_iluvatar_splitk_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m,
    n,
    STRIDE_AK,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    alpha: tl.float32,
    num_k_splits,
    A_IS_E4M3: tl.constexpr,
    X_IS_E4M3: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    chunk_k = (n + num_k_splits - 1) // num_k_splits
    k_begin = pid_k * chunk_k
    k_chunk_end = tl.minimum(k_begin + chunk_k, n)
    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = (
        a_ptr + (k_begin + k_offsets_init)[:, None] * STRIDE_AK + row_offsets[None, :]
    )
    x_ptrs = x_ptr + (k_begin + k_offsets_init) * INCX
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    step_a = BLOCK_SIZE_K * STRIDE_AK
    step_x = BLOCK_SIZE_K * INCX

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        k_offsets = k_begin + k_offset + k_offsets_init
        k_mask = k_offsets < k_chunk_end
        a_mask = k_mask[:, None] & row_mask[None, :]

        a_raw = tl.load(a_ptrs, mask=a_mask, other=0, eviction_policy="evict_first")
        a_block = tl.trans(_fp8_bits_to_float32(a_raw, A_IS_E4M3))
        x_raw = tl.load(x_ptrs, mask=k_mask, other=0, eviction_policy="evict_last")
        x_block = _fp8_bits_to_float32(x_raw, X_IS_E4M3)
        acc += tl.sum(a_block * x_block[None, :], axis=1)

        a_ptrs += step_a
        x_ptrs += step_x

    y_ptrs = y_ptr + row_offsets * INCY
    tl.atomic_add(
        y_ptrs, (acc * alpha).to(y_ptr.dtype.element_ty), mask=row_mask, sem="relaxed"
    )


def fp8_gemv(
    trans: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    """Iluvatar FP8 GEMV using byte loads to avoid unsupported masked FP8 loads."""
    if m == 0 or n == 0:
        return

    assert trans == CUBLAS_OP_T, "FP8 gemv only supports trans=CUBLAS_OP_T (TN format)"
    assert A.stride(1) == 1, "A must be row-major contiguous in the inner dimension"
    assert lda == A.stride(0), "lda must match the row stride of A"
    assert A.dtype in FP8_DTYPES, f"A must be FP8 dtype, got {A.dtype}"
    assert x.dtype in FP8_DTYPES, f"x must be FP8 dtype, got {x.dtype}"
    assert y.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert A.device == x.device == y.device
    assert m % 16 == 0, f"Matrix dimension m ({m}) must be a multiple of 16"
    assert n % 16 == 0, f"Matrix dimension n ({n}) must be a multiple of 16"
    assert lda % 16 == 0, f"lda ({lda}) must be a multiple of 16"
    assert A.data_ptr() % 16 == 0, "Pointer to A must be 16-byte aligned"
    assert x.data_ptr() % 16 == 0, "Pointer to x must be 16-byte aligned"
    assert y.data_ptr() % 16 == 0, "Pointer to y must be 16-byte aligned"

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    # Both FP8 formats occupy one byte. Passing uint8 views keeps the original
    # storage/strides but prevents CoreX from rejecting fp8e4nv in the signature.
    a_bytes = A.view(torch.uint8)
    x_bytes = x.view(torch.uint8)
    a_is_e4m3 = A.dtype == torch.float8_e4m3fn
    x_is_e4m3 = x.dtype == torch.float8_e4m3fn

    if n <= SPLITK_M_THRESHOLD and m >= SPLITK_K_THRESHOLD:
        num_k_splits = min(triton.cdiv(m, 2048), 128)
        if n <= 4 and num_k_splits > 32:
            num_k_splits = min(triton.cdiv(m, 8192), 32)
        elif n <= 16 and num_k_splits > 32:
            num_k_splits = min(triton.cdiv(m, 4096), 32)

        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)

        grid = lambda meta: (
            triton.cdiv(n, meta["BLOCK_SIZE_M"]),
            num_k_splits,
        )
        with torch_device_fn.device(A.device):
            fp8_gemv_iluvatar_splitk_kernel[grid](
                a_bytes,
                x_bytes,
                y,
                n,
                m,
                lda,
                incx,
                incy,
                alpha,
                num_k_splits,
                a_is_e4m3,
                x_is_e4m3,
            )
    else:
        beta_is_zero = beta == 0.0
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            fp8_gemv_iluvatar_kernel[grid](
                a_bytes,
                x_bytes,
                y,
                alpha,
                beta,
                n,
                m,
                lda,
                incx,
                incy,
                beta_is_zero,
                a_is_e4m3,
                x_is_e4m3,
            )

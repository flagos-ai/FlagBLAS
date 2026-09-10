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

from typing import Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


@triton.jit
def _bfgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr,
    SKIP_FULL: tl.constexpr,
    FULL_GRID_M: tl.constexpr,
    FULL_GRID_N: tl.constexpr,
    CACHE: tl.constexpr,
    N_MAJOR_ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m
        pid_m = pid - pid_n * grid_m
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

    if SKIP_FULL and pid_m < FULL_GRID_M and pid_n < FULL_GRID_N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if CHECK_BOUNDS:
        is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
        is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K

        for ki in range(k_full_iters):
            offs_k = ki * BLOCK_K + offs_k_base
            if TRANS_A:
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
            else:
                a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

            if is_full_m and is_full_n:
                a = tl.load(a_ptrs, cache_modifier=CACHE)
                b = tl.load(b_ptrs, cache_modifier=CACHE)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_m[:, None] < m, other=0.0, cache_modifier=CACHE
                )
                b = tl.load(
                    b_ptrs, mask=offs_n[None, :] < n, other=0.0, cache_modifier=CACHE
                )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

        if k_remainder > 0:
            offs_k = k_full_iters * BLOCK_K + offs_k_base
            if TRANS_A:
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
            else:
                a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
            b_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=CACHE)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=CACHE)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        if UNROLL >= 4:
            k_unroll = BLOCK_K * 4
            k_full = (k // k_unroll) * k_unroll
            for k_start in range(0, k_full, k_unroll):
                offs_k0 = k_start + offs_k_base
                offs_k1 = k_start + BLOCK_K + offs_k_base
                offs_k2 = k_start + 2 * BLOCK_K + offs_k_base
                offs_k3 = k_start + 3 * BLOCK_K + offs_k_base
                if TRANS_A:
                    a0_ptrs = a_ptr + offs_k0[None, :] * lda + offs_m[:, None]
                    a1_ptrs = a_ptr + offs_k1[None, :] * lda + offs_m[:, None]
                    a2_ptrs = a_ptr + offs_k2[None, :] * lda + offs_m[:, None]
                    a3_ptrs = a_ptr + offs_k3[None, :] * lda + offs_m[:, None]
                else:
                    a0_ptrs = a_ptr + offs_m[:, None] * lda + offs_k0[None, :]
                    a1_ptrs = a_ptr + offs_m[:, None] * lda + offs_k1[None, :]
                    a2_ptrs = a_ptr + offs_m[:, None] * lda + offs_k2[None, :]
                    a3_ptrs = a_ptr + offs_m[:, None] * lda + offs_k3[None, :]
                if TRANS_B:
                    b0_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k0[:, None]
                    b1_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k1[:, None]
                    b2_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k2[:, None]
                    b3_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k3[:, None]
                else:
                    b0_ptrs = b_ptr + offs_k0[:, None] * ldb + offs_n[None, :]
                    b1_ptrs = b_ptr + offs_k1[:, None] * ldb + offs_n[None, :]
                    b2_ptrs = b_ptr + offs_k2[:, None] * ldb + offs_n[None, :]
                    b3_ptrs = b_ptr + offs_k3[:, None] * ldb + offs_n[None, :]
                a0 = tl.load(a0_ptrs, cache_modifier=CACHE)
                b0 = tl.load(b0_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a0, b0, acc, out_dtype=tl.float32, allow_tf32=False)
                a1 = tl.load(a1_ptrs, cache_modifier=CACHE)
                b1 = tl.load(b1_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a1, b1, acc, out_dtype=tl.float32, allow_tf32=False)
                a2 = tl.load(a2_ptrs, cache_modifier=CACHE)
                b2 = tl.load(b2_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a2, b2, acc, out_dtype=tl.float32, allow_tf32=False)
                a3 = tl.load(a3_ptrs, cache_modifier=CACHE)
                b3 = tl.load(b3_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a3, b3, acc, out_dtype=tl.float32, allow_tf32=False)
            for k_start in range(k_full, k, BLOCK_K):
                offs_k = k_start + offs_k_base
                if TRANS_A:
                    a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
                else:
                    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
                if TRANS_B:
                    b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
                else:
                    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
                a = tl.load(a_ptrs, cache_modifier=CACHE)
                b = tl.load(b_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        else:
            for k_start in range(0, k, BLOCK_K):
                offs_k = k_start + offs_k_base
                if TRANS_A:
                    a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
                else:
                    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
                if TRANS_B:
                    b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
                else:
                    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
                a = tl.load(a_ptrs, cache_modifier=CACHE)
                b = tl.load(b_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if ALPHA_IS_ONE:
        result = acc
    else:
        result = alpha * acc
    if CHECK_BOUNDS:
        c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)
    else:
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.bfloat16))


@triton.jit
def _bfgemm_nn_2048_square_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    grid_m = 2048 // BLOCK_M
    grid_n = 2048 // BLOCK_N
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in tl.static_range(0, 2048, BLOCK_K):
            offs_k = k_start + offs_k_base
            a = tl.load(
                a_ptr + offs_m[:, None] * 2048 + offs_k[None, :], cache_modifier=".cg"
            )
            b = tl.load(
                b_ptr + offs_k[:, None] * 2048 + offs_n[None, :], cache_modifier=".cg"
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        tl.store(c_ptr + offs_m[:, None] * 2048 + offs_n[None, :], acc.to(tl.bfloat16))


@triton.jit
def _bfgemm_nn_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    NUM_SMS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_MOD: tl.constexpr,
):
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            if CACHE_MOD == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            elif CACHE_MOD == 1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, cache_modifier=".cg")
                b = tl.load(b_ptrs, cache_modifier=".cg")
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        if BLOCK_M == 256 and BLOCK_N == 256:
            # Two-step store for 256x256 bf16 tiles: chaining the RNE
            # conversion inside tl.store inflates the register peak (142 vs
            # 96 regs) and costs ~35% throughput on this backend; a separate
            # value keeps the epilogue lean. 128x128 keeps the chained form
            # (measured equivalent there, 35 regs).
            out = (alpha * acc).to(tl.bfloat16)
            tl.store(c_ptr + offs_m[:, None] * ldc + offs_n[None, :], out)
        else:
            tl.store(
                c_ptr + offs_m[:, None] * ldc + offs_n[None, :],
                (alpha * acc).to(tl.bfloat16),
            )


@triton.jit
def _bfgemm_nt_native_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    NUM_SMS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_MOD: tl.constexpr,
):
    """Persistent NT kernel that reads B in its native (n, k) layout (ldb == k),
    i.e. C = A @ B^T without materializing B^T first. 2026-09-02: same structure
    as the NN persistent kernel; the only difference is the B-tile addressing
    (offs_k has stride 1, offs_n walks ldb). On GPU1 this wins on the NT shapes
    that used to pay a B-transpose copy (-> NN) or a thermal-sensitive
    transposed-dot kernel (configs in _select_bfgemm_nt_native_persistent_config)."""
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * ldb
            if CACHE_MOD == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            elif CACHE_MOD == 1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, cache_modifier=".cg")
                b = tl.load(b_ptrs, cache_modifier=".cg")
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        if BLOCK_M == 256 and BLOCK_N == 256:
            # Two-step store: same register-lean epilogue as the NN 256x256 path.
            out = (alpha * acc).to(tl.bfloat16)
            tl.store(c_ptr + offs_m[:, None] * ldc + offs_n[None, :], out)
        else:
            tl.store(
                c_ptr + offs_m[:, None] * ldc + offs_n[None, :],
                (alpha * acc).to(tl.bfloat16),
            )


@triton.jit
def _bfgemm_tntt_native_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    NUM_SMS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_MOD: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """Persistent TN/TT kernel for transa == T that reads A^T and B in their
    native row-major layouts (A is stored (k, m), lda == m), so neither the
    one-time operand copy (pretranspose -> NN) nor the per-tile tl.trans of the
    transposed-dot kernel is needed: an A^T tile is loaded as (M, K) with stride
    1 along M (the column-major-style leading read this backend favors), the
    accumulator keeps the ordinary (M, N) orientation, and C is stored directly.
    LAYOUT == 0: TN (C = A^T @ B, B stored (k, n), ldb == n); LAYOUT == 1: TT
    (C = A^T @ B^T, B stored (n, k), ldb == k). 2026-09-03: same-process
    3-round interleaved official-param A/B on GPU1 wins on all whitelisted
    shapes (see the config selectors); same persistent grid-stride /
    two-step-store family as the NN and NT-native persistent kernels."""
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            # (M, K) A^T tile: stride 1 along M, stride lda along K.
            a_ptrs = a_ptr + offs_m[:, None] + offs_k[None, :] * lda
            if LAYOUT == 1:  # TT: B (n, k) row-major, inner k contiguous
                b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * ldb
            else:  # TN: B (k, n) row-major, inner n contiguous
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            if CACHE_MOD == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            elif CACHE_MOD == 1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, cache_modifier=".cg")
                b = tl.load(b_ptrs, cache_modifier=".cg")
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        if BLOCK_M == 256 and BLOCK_N == 256:
            # Two-step store: same register-lean epilogue as the NN 256x256 path.
            out = (alpha * acc).to(tl.bfloat16)
            tl.store(c_ptr + offs_m[:, None] * ldc + offs_n[None, :], out)
        else:
            tl.store(
                c_ptr + offs_m[:, None] * ldc + offs_n[None, :],
                (alpha * acc).to(tl.bfloat16),
            )


@triton.jit
def _bfgemm_nn_pipe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    NUM_SMS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_MOD: tl.constexpr,
    NS: tl.constexpr,
):
    # Persistent variant with an explicit software-pipelined K loop
    # (tl.range(..., num_stages=NS)). Found on GPU1 to beat the plain
    # persistent kernel on several K-heavy core shapes.
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in tl.range(0, k, BLOCK_K, num_stages=NS):
            offs_k = k_start + offs_k_base
            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            if CACHE_MOD == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            elif CACHE_MOD == 1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, cache_modifier=".cg")
                b = tl.load(b_ptrs, cache_modifier=".cg")
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        tl.store(
            c_ptr + offs_m[:, None] * ldc + offs_n[None, :],
            (alpha * acc).to(tl.bfloat16),
        )


@triton.jit
def _bfgemm_nn_blockptr_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    N_MAJOR_ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m
        pid_m = pid - pid_n * grid_m
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        tl.store(c_block_ptr, (alpha * acc).to(tl.bfloat16), boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        tl.store(
            c_block_ptr,
            (alpha * acc + beta * c_vals).to(tl.bfloat16),
            boundary_check=(0, 1),
        )


@triton.jit
def _bfgemm_nn_blockptr_fast_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    N_MAJOR_ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m
        pid_m = pid - pid_n * grid_m
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if ALPHA_IS_ONE:
        result = acc
    else:
        result = alpha * acc
    if not BETA_IS_ZERO:
        result += beta * tl.load(c_ptrs).to(tl.float32)
    tl.store(c_ptrs, result.to(tl.bfloat16))


@triton.jit
def _bfgemm_nn_blockptr_ab1_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    N_MAJOR_ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m
        pid_m = pid - pid_n * grid_m
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


@triton.jit
def _bfgemm_nn_static_ab1_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N_MAJOR_ORDER: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m
        pid_m = pid - pid_n * grid_m
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.static_range(0, K, BLOCK_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            cache_modifier=".cg",
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * N + offs_n[None, :],
            cache_modifier=".cg",
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


@triton.jit
def _bfgemm_nn_m2_blockptr_fast_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    N_MAJOR_ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_m2 = grid_m // 2
    grid_n = tl.cdiv(n, BLOCK_N)
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m2
        pid_m2 = pid - pid_n * grid_m2
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m2 - group_id * GROUP_M, GROUP_M)
        pid_m2 = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
    pid_m0 = pid_m2 * 2
    pid_m1 = pid_m0 + 1

    a0_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m0 * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    a1_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m1 * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k, BLOCK_K):
        a0 = tl.load(a0_block_ptr)
        a1 = tl.load(a1_block_ptr)
        b = tl.load(b_block_ptr)
        acc0 = tl.dot(a0, b, acc0, out_dtype=tl.float32, allow_tf32=False)
        acc1 = tl.dot(a1, b, acc1, out_dtype=tl.float32, allow_tf32=False)
        a0_block_ptr = tl.advance(a0_block_ptr, (0, BLOCK_K))
        a1_block_ptr = tl.advance(a1_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c0_ptrs = c_ptr + (pid_m0 * BLOCK_M + offs_m)[:, None] * ldc + offs_n[None, :]
    c1_ptrs = c_ptr + (pid_m1 * BLOCK_M + offs_m)[:, None] * ldc + offs_n[None, :]
    result0 = alpha * acc0
    result1 = alpha * acc1
    if not BETA_IS_ZERO:
        result0 += beta * tl.load(c0_ptrs).to(tl.float32)
        result1 += beta * tl.load(c1_ptrs).to(tl.float32)
    tl.store(c0_ptrs, result0.to(tl.bfloat16))
    tl.store(c1_ptrs, result1.to(tl.bfloat16))


@triton.jit
def _bfgemm_nn_descriptor_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(k, BLOCK_K)):
        a = a_desc.load([offs_m, kk * BLOCK_K])
        b = b_desc.load([kk * BLOCK_K, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    if alpha == 1.0:
        result = acc
    else:
        result = alpha * acc
    if not BETA_IS_ZERO:
        result += beta * c_desc.load([offs_m, offs_n]).to(tl.float32)
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@triton.jit
def _bfgemm_nn_splitk_kernel(
    a_ptr,
    b_ptr,
    partial_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_TILES_PER_SPLIT: tl.constexpr,
    N_MAJOR_ORDER: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    if N_MAJOR_ORDER:
        pid_n = pid // grid_m
        pid_m = pid - pid_n * grid_m
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ki in range(0, K_TILES_PER_SPLIT):
        offs_k = (split_id * K_TILES_PER_SPLIT + ki) * BLOCK_K + offs_k_base
        a = tl.load(
            a_ptr + offs_m[:, None] * lda + offs_k[None, :],
            cache_modifier=".cg",
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * ldb + offs_n[None, :],
            cache_modifier=".cg",
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    partial_ptrs = (
        partial_ptr + split_id * m * n + offs_m[:, None] * n + offs_n[None, :]
    )
    tl.store(partial_ptrs, acc)


@triton.jit
def _bfgemm_nn_splitk_reduce_kernel(
    partial_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    ldc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    partial_offsets = offs_m[:, None] * n + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for split_id in range(0, SPLIT_K):
        acc += tl.load(partial_ptr + split_id * m * n + partial_offsets)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    tl.store(c_ptrs, (alpha * acc).to(tl.bfloat16))


@triton.jit
def _bfgemm_tt_transpose_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    CACHE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            a_ptr + offs_k[:, None] * lda + offs_m[None, :],
            cache_modifier=CACHE,
        )
        b = tl.load(
            b_ptr + offs_n[:, None] * ldb + offs_k[None, :],
            cache_modifier=CACHE,
        )
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    acc = tl.trans(acc_t)
    if ALPHA_IS_ONE:
        result = acc
    else:
        result = alpha * acc
    if not BETA_IS_ZERO:
        result += beta * tl.load(c_ptrs).to(tl.float32)
    tl.store(c_ptrs, result.to(tl.bfloat16))


@triton.jit
def _bfgemm_tn_transpose_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    CACHE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """TN variant of the transposed-dot kernel.

    For TN (C = A^T @ B) with A stored (k, m) and B stored (k, n) both
    row-major, loading A as [BLOCK_K, BLOCK_M] is fully coalesced (inner
    dimension m is contiguous), whereas the standard kernel's [BLOCK_M,
    BLOCK_K] A tile gathers along k.  The result is accumulated in the
    transposed orientation C^T = B^T @ A and transposed once before store.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            a_ptr + offs_k[:, None] * lda + offs_m[None, :],
            cache_modifier=CACHE,
        )
        b = tl.load(
            b_ptr + offs_k[None, :] * ldb + offs_n[:, None],
            cache_modifier=CACHE,
        )
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    acc = tl.trans(acc_t)
    if ALPHA_IS_ONE:
        result = acc
    else:
        result = alpha * acc
    if not BETA_IS_ZERO:
        result += beta * tl.load(c_ptrs).to(tl.float32)
    tl.store(c_ptrs, result.to(tl.bfloat16))


@triton.jit
def _bfgemm_tn_transpose_dot_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    NUM_SMS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_MOD: tl.constexpr,
):
    """Persistent TN transposed-dot kernel (C = A^T @ B, A stored (k, m)).

    Same arithmetic as _bfgemm_tn_transpose_dot_kernel but each program keeps
    a persistent tile via grid-stride looping; wins for extreme-aspect shapes
    (e.g. 8192x256x2048) where the plain transpose-dot launch loses to the
    A-pretranspose -> NN path on GPU1.
    """
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
            b_ptrs = b_ptr + offs_k[None, :] * ldb + offs_n[:, None]
            if CACHE_MOD == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            elif CACHE_MOD == 1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, cache_modifier=".cg")
                b = tl.load(b_ptrs, cache_modifier=".cg")
            acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)
        c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
        tl.store(c_ptrs, (alpha * tl.trans(acc_t)).to(tl.bfloat16))


@triton.jit
def _bfgemm_tt_transpose_dot_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    NUM_SMS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_MOD: tl.constexpr,
):
    """Persistent TT transposed-dot kernel (C = A^T @ B^T, A stored (k, m),
    B stored (n, k)). Same structure as the TN persistent kernel but B tiles
    are loaded row-contiguous (B is (n, k), ldb = k). Wins over the
    transpose-both -> NN path on GPU1 for extreme-aspect shapes."""
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
            b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
            if CACHE_MOD == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            elif CACHE_MOD == 1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, cache_modifier=".cg")
                b = tl.load(b_ptrs, cache_modifier=".cg")
            acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)
        c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
        tl.store(c_ptrs, (alpha * tl.trans(acc_t)).to(tl.bfloat16))


@triton.jit
def _bfgemm_nt_transpose_dot_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    NUM_SMS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_MOD: tl.constexpr,
):
    """Persistent NT transposed-dot kernel (C = A @ B^T, A stored (m, k),
    B stored (n, k), lda == ldb == k).

    A tiles are loaded DIRECTLY as [BLOCK_K, BLOCK_M] via the gather address
    offs_m * lda + offs_k (A[mm, kk] sits at mm * lda + kk), so no tl.trans of
    the A tile is needed; B tiles [N, K] are row-contiguous. The result is
    accumulated in the transposed orientation C^T = B @ A^T and transposed once
    before store. On GPU1 this beats the B-pretranspose -> NN path for shapes
    where A fits in cache (gather is L2-resident)."""
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n
    offs_k_base = tl.arange(0, BLOCK_K)
    tiles_per_program = tl.cdiv(num_tiles - start_pid, GRID_STRIDE)
    for idx in range(0, tiles_per_program):
        tile_id = start_pid + idx * GRID_STRIDE
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + offs_k_base
            a_ptrs = a_ptr + offs_m[None, :] * lda + offs_k[:, None]
            b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
            if CACHE_MOD == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            elif CACHE_MOD == 1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, cache_modifier=".cg")
                b = tl.load(b_ptrs, cache_modifier=".cg")
            acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)
        c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
        tl.store(c_ptrs, (alpha * tl.trans(acc_t)).to(tl.bfloat16))


def _select_bfgemm_nn_descriptor_config(m: int, n: int, k: int):
    return None


def _select_bfgemm_nn_2048_square_persistent_config(m: int, n: int, k: int):
    # Disabled: the generic blockptr path (128, 128, 64, nw=16, gm=2, ns=1,
    # nmo=0) beats the dedicated 2048^3 persistent kernel (0.970 vs 0.754).
    return None


def _select_bfgemm_nn_persistent_config(m: int, n: int, k: int):
    # bf16-tuned on GPU1 (2026-08-27): only shapes where an official core run
    # confirmed the persistent 128x128 kernel beats the blockptr path. The
    # fp16 256x256 persistent tiles regress 20-35% on bf16, so 128x128 only.
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages, wave_count,
    # cache_mod).
    # h2h (official-param, same-process cublas gemmEx) + official-subset
    # arbitration (2 rounds) on GPU1. Rejected candidates that looked good in
    # h2h but lost official arbitration: 512x16384x4096 g8ns2 / pers-g4w8,
    # 2048x12288x4096 g4ns1, 16384x2048x2048 g2ns1, 256x8192x2048 pers-g2w4.
    # wave 8 (full tile grid) wins for the skinny shapes.
    # 2026-08-31: the 128x128 persistent entries for the skinny shapes used to
    # shadow the 256x256 two-step-store configs below (dead code). Cross-op
    # sweep + official subset arbitration confirmed 256x256 wins ~11-17%, so
    # the 128x128 entries were removed and the 256x256 configs are now live.
    # Big squares: 256x256 persistent with a two-step store epilogue. The
    # chained (alpha*acc).to(bf16) store inflates the register peak (142 vs 96
    # regs) and costs ~35% throughput on 256x256 bf16 tiles; the kernel's
    # constexpr two-step branch (see _bfgemm_nn_persistent_kernel) keeps the
    # epilogue lean. Official subset arbitration: NN +0.04~+0.15 on the four
    # squares, no regression on 128x128 persistent shapes.
    if m == 2048 and n == 2048 and k == 2048:
        return 256, 256, 64, 16, 4, 1, 2, 2
    if m == 4096 and n == 4096 and k == 4096:
        return 256, 256, 64, 16, 4, 3, 2, 1
    if m == 8192 and n == 8192 and k == 8192:
        return 256, 256, 64, 16, 4, 1, 8, 2
    if m == 16384 and n == 16384 and k == 16384:
        return 256, 256, 64, 16, 4, 2, 2, 0
    # 2026-08-31: the 256x256 two-step-store persistent kernel also wins on
    # the skinny/model NN shapes (same-harness sweep +0.08~+0.30 vs the
    # 128x128 blockptr/persistent paths, output bit-identical to production;
    # official subset arbitration base/var x2).
    if m == 512 and n == 16384 and k == 4096:
        return 256, 256, 64, 16, 4, 1, 2, 2
    if m == 16384 and n == 512 and k == 4096:
        return 256, 256, 64, 16, 4, 1, 4, 2
    if m == 8192 and n == 256 and k == 2048:
        return 256, 256, 64, 16, 8, 1, 4, 2
    if m == 32768 and n == 1024 and k == 1024:
        return 256, 256, 64, 16, 2, 1, 8, 0
    if m == 2048 and n == 2048 and k == 16384:
        return 256, 256, 64, 16, 8, 1, 4, 2
    if m == 2048 and n == 12288 and k == 4096:
        return 256, 256, 64, 16, 2, 1, 8, 0
    if m == 4096 and n == 24576 and k == 8192:
        return 256, 256, 64, 16, 4, 1, 4, 2
    # 2026-08-31 round 2: same-harness sweep (official do_bench, 2-round
    # alternation, allclose vs production maxdiff=0) confirmed the 256x256
    # two-step-store persistent kernel also wins on the remaining 128x128
    # blockptr NN underperformers. Subset arbitration base/var x2 follows.
    if m == 2048 and n == 16384 and k == 2048:
        return 256, 256, 64, 16, 4, 1, 2, 2
    if m == 16384 and n == 2048 and k == 2048:
        return 256, 256, 64, 16, 4, 1, 2, 2
    if m == 2048 and n == 11008 and k == 4096:
        return 256, 256, 64, 16, 2, 1, 8, 0
    if m == 2048 and n == 4096 and k == 11008:
        return 256, 256, 64, 16, 2, 1, 4, 2
    if m == 4096 and n == 8192 and k == 28672:
        return 256, 256, 64, 16, 4, 1, 4, 2
    if m == 8192 and n == 28672 and k == 8192:
        # bf16-tuned 2026-09-03: 13 interleaved official-param A/B rounds on
        # GPU1 -> wave_count 4 (grid_stride 64) beats wave 8 by ~-1.4%
        # (cand ~49.7 vs prod ~50.5 ms); wv2/+regressed, wv16/cache_mod
        # neutral-or-worse at official arbitration. Isolation puts this shape
        # at ~0.91 vs cublas; official 0.73 runs are run-tail heat inflation.
        return 256, 256, 64, 16, 4, 1, 4, 2
    return None


def _select_bfgemm_nn_pipe_config(m: int, n: int, k: int):
    # Disabled for bf16: these pipe configs use 256x256 tiles, which regress
    # ~20-35% on bf16 (same root cause as the persistent configs; the fp16
    # pipe/persistent tuning does not transfer to bf16 on this backend). The
    # shapes fall through to the 128x128 blockptr/generic paths instead.
    return None


def _select_bfgemm_nn_static_ab1_config(m: int, n: int, k: int):
    return None


def _select_bfgemm_nn_ab1_blockptr_config(m: int, n: int, k: int):
    return None


def _select_bfgemm_nn_m2_blockptr_config(m: int, n: int, k: int):
    return None


def _select_bfgemm_nn_pointer_config(m: int, n: int, k: int):
    return None


def _select_bfgemm_nn_splitk_config(m: int, n: int, k: int):
    return None


def _select_bfgemm_nn_blockptr_config(m: int, n: int, k: int):
    # Exact-shape blockptr fast path configs, from batch sweeps on GPU1.
    # Tuple: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages, n_major_order).
    # Several shapes are dispatched to the persistent path first; these blockptr
    # entries act as fallback when persistent preconditions do not hold.
    if m == 2048 and n == 2048 and k == 2048:
        return 128, 128, 64, 16, 2, 1, False
    if m == 4096 and n == 4096 and k == 4096:
        return 128, 128, 64, 16, 8, 1, False
    if m == 16384 and n == 16384 and k == 16384:
        return 128, 128, 64, 16, 4, 2, False
    if m == 8192 and n == 256 and k == 2048:
        return 128, 128, 64, 16, 4, 1, False
    if m == 8192 and n == 8192 and k == 8192:
        return 128, 128, 64, 16, 8, 2, False
    if m == 16384 and n == 512 and k == 4096:
        return 128, 128, 64, 16, 1, 1, False
    if m == 512 and n == 16384 and k == 4096:
        return 128, 128, 64, 16, 2, 1, True
    if m == 2048 and n == 12288 and k == 4096:
        return 128, 128, 64, 16, 4, 2, False
    if m == 2048 and n == 11008 and k == 4096:
        return 128, 128, 64, 16, 8, 1, False
    if m == 2048 and n == 4096 and k == 11008:
        return 128, 128, 64, 16, 4, 1, False
    if m == 4096 and n == 24576 and k == 8192:
        return 128, 128, 64, 16, 8, 2, False
    if m == 4096 and n == 8192 and k == 28672:
        return 128, 128, 64, 16, 4, 2, False
    if m == 8192 and n == 28672 and k == 8192:
        return 128, 128, 64, 16, 4, 2, False
    if m == 16384 and n == 2048 and k == 2048:
        return 128, 128, 64, 16, 2, 2, False
    if m == 2048 and n == 16384 and k == 2048:
        return 128, 128, 64, 16, 8, 1, False
    if m == 2048 and n == 2048 and k == 16384:
        return 128, 128, 64, 16, 4, 1, False
    if m == 32768 and n == 1024 and k == 1024:
        return 128, 128, 64, 16, 2, 1, False
    return None


def _select_bfgemm_config(m: int, n: int, k: int, transa: int, transb: int):
    """Select (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages).

    Configurations are derived from extensive sweeps on Iluvatar BI-V150.
    """
    # ---- Smallest square ----
    if m == 64 and n == 64 and k == 64:
        return 64, 64, 64, 4, 8, 3, 1

    # ---- Tall / skinny (small m) ----
    if m <= 64:
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Short / wide (small n) ----
    if n <= 64:
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Small squares (max dim <= 512) ----
    if max(m, n, k) <= 512:
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Exact core-shape fixes for previously under-threshold cases ----
    # NN configurations derived from corrected-layout sweeps on Iluvatar BI-V150.
    # (128, 128, 64, nw=16) is the consistent winner; this backend does not
    # support multi-stage shared-memory pipelining (num_stages and tl.range
    # num_stages are both no-ops), so group_m (4 or 8) is the main occupancy
    # lever. nw=8 wins only for a couple of M-asymmetric shapes.
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
        if m == 2048 and n == 2048 and k == 2048:
            return 128, 128, 64, 16, 2, 4, 1
        if m == 4096 and n == 4096 and k == 4096:
            return 128, 128, 64, 16, 4, 1, 1
        if m == 8192 and n == 8192 and k == 8192:
            return 128, 128, 64, 16, 4, 4, 1
        if m == 16384 and n == 16384 and k == 16384:
            return 256, 128, 64, 16, 4, 1, 1
        if m == 2048 and n == 12288 and k == 4096:
            return 128, 128, 64, 16, 4, 1, 2
        if m == 2048 and n == 11008 and k == 4096:
            return 128, 128, 64, 16, 8, 1, 1
        if m == 2048 and n == 4096 and k == 11008:
            return 128, 128, 64, 16, 4, 1, 2
        if m == 4096 and n == 24576 and k == 8192:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 4096 and n == 8192 and k == 28672:
            return 128, 128, 64, 16, 4, 1, 1
        if m == 8192 and n == 28672 and k == 8192:
            return 128, 128, 64, 16, 4, 2, 2
        if m == 16384 and n == 2048 and k == 2048:
            return 128, 128, 64, 16, 2, 1, 1
        if m == 2048 and n == 16384 and k == 2048:
            return 128, 128, 64, 16, 4, 1, 2
        if m == 2048 and n == 2048 and k == 16384:
            return 128, 128, 64, 16, 4, 1, 1
        if m == 32768 and n == 1024 and k == 1024:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 4096 and n == 128 and k == 1024:
            return 128, 128, 128, 16, 16, 3, 4
        if m == 8192 and n == 256 and k == 2048:
            return 128, 128, 64, 16, 4, 1, 2
        if m == 16384 and n == 512 and k == 4096:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 512 and n == 16384 and k == 4096:
            return 128, 128, 64, 16, 2, 4, 1
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
        if m == 2048 and n == 2048 and k == 2048:
            return 128, 128, 64, 16, 2, 2, 1
        if m == 4096 and n == 4096 and k == 4096:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 8192 and n == 8192 and k == 8192:
            return 128, 128, 64, 16, 4, 5, 1
        if m == 16384 and n == 16384 and k == 16384:
            return 128, 128, 64, 16, 8, 3, 1
        if m == 2048 and n == 16384 and k == 2048:
            return 128, 128, 64, 8, 16, 6, 1
        if m == 16384 and n == 2048 and k == 2048:
            return 128, 128, 64, 16, 4, 10, 1
        if m == 16384 and n == 512 and k == 4096:
            return 128, 128, 64, 16, 2, 2, 1
        if m == 512 and n == 16384 and k == 4096:
            return 256, 256, 64, 8, 8, 8, 1
        if m == 2048 and n == 2048 and k == 16384:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 4096 and n == 24576 and k == 8192:
            return 128, 128, 64, 16, 4, 3, 4
        if m == 2048 and n == 11008 and k == 4096:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 2048 and n == 12288 and k == 4096:
            return 128, 128, 64, 16, 4, 3, 4
        if m == 8192 and n == 28672 and k == 8192:
            return 128, 128, 64, 16, 4, 4, 1
        if m == 8192 and n == 256 and k == 2048:
            return 128, 128, 64, 16, 2, 2, 1
        if m == 256 and n == 8192 and k == 2048:
            return 128, 128, 64, 16, 2, 2, 1
        if m == 4096 and n == 8192 and k == 28672:
            return 128, 128, 64, 16, 4, 6, 1
        if m == 32768 and n == 1024 and k == 1024:
            return 128, 128, 64, 16, 4, 3, 1
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
        if m == 128 and n == 4096 and k == 1024:
            return 128, 128, 128, 16, 4, 2, 1
        if m == 256 and n == 8192 and k == 2048:
            return 128, 128, 64, 16, 8, 3, 1
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_T:
        if m == 128 and n == 4096 and k == 1024:
            return 128, 128, 128, 16, 8, 4, 1

    # ---- 128-ish narrow shapes ----
    if m == 128:
        return 64, 128, 64, 8, 8, 4, 1
    if n == 128:
        if transa == CUBLAS_OP_T:
            return 128, 64, 64, 8, 8, 4, 1
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Medium / large shapes (max dim <= 2048, e.g. 1024^3 / 2048^3) ----
    if max(m, n, k) <= 2048:
        return 128, 128, 64, 16, 4, 3, 1

    # ---- Default large ----
    return 128, 128, 64, 16, 4, 3, 1


def _select_bfgemm_tt_transpose_dot_config(m: int, n: int, k: int):
    if m == 256 and n == 8192 and k == 2048:
        return 128, 128, 64, 16, 16, 2
    if m == 512 and n == 16384 and k == 4096:
        return 128, 128, 64, 16, 8, 2
    if m == 16384 and n == 512 and k == 4096:
        return 128, 128, 64, 16, 4, 3
    if m == 4096 and n == 4096 and k == 4096:
        return 128, 128, 64, 16, 8, 2
    return None


def _select_bfgemm_tn_transpose_dot_config(m: int, n: int, k: int):
    """Per-shape (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages)
    for the TN transposed-dot kernel, from a sweep on Iluvatar BI-V150.

    The transposed-dot kernel loads A fully coalesced (A is stored (k, m)
    for TN), which wins for most large shapes; it is disabled for shapes
    where the one-time fp32 accumulator transpose is more expensive than
    the strided-A savings (e.g. big tiles / extreme aspect ratios).
    """
    if m == 2048 and n == 2048 and k == 2048:
        return 128, 128, 64, 16, 2, 2
    if m == 4096 and n == 4096 and k == 4096:
        return 128, 128, 64, 16, 8, 4
    if m == 8192 and n == 8192 and k == 8192:
        return 128, 128, 64, 16, 4, 6
    if m == 16384 and n == 16384 and k == 16384:
        return 128, 128, 64, 16, 8, 3
    if m == 16384 and n == 512 and k == 4096:
        return 128, 128, 64, 16, 2, 2
    if m == 2048 and n == 2048 and k == 16384:
        return 128, 128, 64, 16, 8, 2
    if m == 2048 and n == 12288 and k == 4096:
        return 128, 128, 64, 16, 4, 3
    if m == 8192 and n == 256 and k == 2048:
        return 128, 128, 64, 16, 2, 10
    if m == 256 and n == 8192 and k == 2048:
        return 128, 128, 64, 16, 2, 2
    if m == 4096 and n == 24576 and k == 8192:
        return 128, 128, 64, 16, 4, 4
    if m == 2048 and n == 11008 and k == 4096:
        return 128, 128, 64, 16, 8, 3
    if m == 8192 and n == 28672 and k == 8192:
        return 128, 128, 64, 16, 4, 4
    return None


def _select_bfgemm_tn_transpose_dot_persistent_config(m: int, n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, wave_count, cache_mod)
    for the persistent TN transposed-dot kernel. 2026-08-26 official core run
    (GPU1): the head-to-head sweep's "full-ps" numbers were measured with the
    A-pretranspose copy OUTSIDE the timed region, but in bfgemm() the copy is
    inside do_bench, so the pretranspose -> NN path is much slower than the
    sweep suggested (official TN 8192x256x2048: 0.858 via pretranspose vs
    0.876 baseline here; 2048^3: 0.807 vs 0.794), so these shapes are
    intercepted again to keep the persistent kernel."""
    # 2026-08-27: retune from sweep td data: 32768x1024x1024 wave 4->8
    # (td-w8 0.970 vs td-comm 0.899), 2048x2048x16384 cm 0->1 (td-cm1 0.823 vs
    # td-comm <0.799). 8192x256x2048 keeps the persistent intercept (an
    # attempt to route it to the non-persistent transpose-dot kernel regressed
    # official 0.874 -> 0.857).
    # 2026-08-27 (round 2): 16384x512x4096 gm 4->2 (sweep-C td-gm2 0.863 vs
    # comm 0.828; official subset 0.851 vs 0.792, full core neutral 0.791).
    if m == 8192 and n == 256 and k == 2048:
        return 128, 128, 64, 16, 2, 4, 0
    if m == 16384 and n == 512 and k == 4096:
        return 128, 128, 64, 16, 2, 16, 0
    if m == 256 and n == 8192 and k == 2048:
        return 128, 128, 64, 16, 4, 4, 0
    if m == 32768 and n == 1024 and k == 1024:
        # 2026-08-31: gm 8->2, wave 8->4 (same-process interleaved A/B d=0.963)
        return 128, 128, 64, 16, 2, 4, 0
    if m == 2048 and n == 2048 and k == 16384:
        return 128, 128, 64, 16, 4, 16, 1
    if m == 2048 and n == 2048 and k == 2048:
        return 128, 128, 64, 16, 2, 4, 0
    return None


def _select_bfgemm_tt_transpose_dot_persistent_config(m: int, n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, wave_count, cache_mod)
    for the persistent TT transposed-dot kernel. 2026-08-26 official core run
    (GPU1) exposed the sweep's "full-ps" artifact: the transpose-both copy was
    outside the timed region in the sweep but inside do_bench in bfgemm(), so
    the pretranspose -> NN path collapsed (official TT 256x8192x2048 0.444 /
    512x16384x4096 0.564 / 16384x512x4096 0.606 / 2048^3 0.715 /
    2048x2048x16384 0.726 vs 0.802 / 0.801 / 0.800 / 0.771 / 0.829 baseline),
    so the intercepts are restored. 2026-08-26: 512x16384x4096 cache_mod 0->1
    (sweep td-cm1 0.937 vs td-comm 0.913; official 0.800 vs 0.772).
    2026-08-27: 256x8192x2048 w4->w8 and 16384x512x4096 gm4->gm2 (sweep-E td
    data; official 0.841 vs 0.799 / 0.865 vs 0.801); added wide-N intercepts
    that used to go pretranspose -> NN (sweep td 0.85-0.89 vs official
    pretranspose 0.73-0.83; official 0.873 / 0.817 / 0.850 / 0.840 / 0.853 /
    0.854 for 2048x12288x4096 / 2048x11008x4096 / 2048x4096x11008 /
    4096x24576x8192 / 8192x28672x8192 / 2048x16384x2048). 2048x2048x16384
    keeps wave 4 (w8 regressed official 0.829 -> 0.766). 8192x28672x8192
    keeps pretranspose -> NN (td kernel context-sensitive: two full core runs
    0.629 / 0.637 vs 0.755 pretranspose baseline, so the added intercept was
    removed)."""
    if m == 8192 and n == 256 and k == 2048:
        return 128, 128, 64, 16, 4, 4, 0
    # 2026-08-27 (round 4, sweep5 official-param do_bench): 512x16384x4096
    # cm1 (8,4,1) -> w8 (8,8,0) 0.792 vs 0.748; 2048^3 w4 (8,4,0) -> w16
    # (8,16,0) 0.880 vs 0.774. Older warmup=500 sweeps mis-ranked these.
    if m == 256 and n == 8192 and k == 2048:
        return 128, 128, 64, 16, 2, 8, 0
    if m == 512 and n == 16384 and k == 4096:
        # 2026-09-01 (GPU2 retune): wave 8->8, gm 8->4 (td (4,8,0) d=1.056 vs
        # (8,8,0), same-process official-param).
        return 128, 128, 64, 16, 4, 8, 0
    if m == 16384 and n == 512 and k == 4096:
        return 128, 128, 64, 16, 2, 16, 0
    if m == 2048 and n == 2048 and k == 16384:
        return 128, 128, 64, 16, 8, 4, 0
    if m == 32768 and n == 1024 and k == 1024:
        return 128, 128, 64, 16, 8, 4, 0
    if m == 2048 and n == 2048 and k == 2048:
        # 2026-08-31: gm 8->2, wave 16->8, cm 0->1 (interleaved A/B d=0.959)
        return 128, 128, 64, 16, 2, 8, 1
    if m == 2048 and n == 11008 and k == 4096:
        # 2026-08-31: wave 16->8, cm 0->1 (interleaved A/B d=0.979)
        return 128, 128, 64, 16, 8, 8, 1
    if m == 2048 and n == 4096 and k == 11008:
        return 128, 128, 64, 16, 4, 8, 0
    if m == 2048 and n == 12288 and k == 4096:
        # 2026-09-01: add intercept (was pretranspose-both -> NN). Full-core
        # 47-shape A/B flag 3474->3318us; single-shape A/B 3555->3122us
        # (-12.2%), bit-exact. Kept; the other five 09-01 intercepts were
        # reverted (td kernel context-sensitive in full-core, see NT below).
        return 128, 128, 64, 16, 8, 4, 0
    # 2026-08-31: 4096x24576x8192 / 2048x16384x2048 intercepts removed. The
    # 256x256 two-step-store NN persistent kernel now beats the td kernel on
    # the pretranspose -> NN route (same-process A/B: 0.86 / 0.86 vs 0.79 /
    # 0.81), so they fall through to the pretranspose branch again.
    # 8192x28672x8192 keeps pretranspose -> NN: the persistent td kernel is
    # context-sensitive (subset bench 0.81-0.85, full core run 0.629 vs
    # pretranspose 0.755 baseline).
    # 2026-09-01 (GPU2 clean full-core retune): 2048x16384x2048 td (8,4,0)
    # d=1.042 in single-shape probe but full-core regressed flag 2.134->2.191
    # (sp 0.798->0.778); reverted, keeps pretranspose -> NN.
    return None


def _select_bfgemm_nt_transpose_dot_persistent_config(m: int, n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, wave_count, cache_mod)
    for the persistent NT transposed-dot kernel (C = A @ B^T). 2026-08-26
    official core run (GPU1) exposed the sweep's "full-ps" artifact (the
    B-pretranspose copy was outside the timed region in the sweep but inside
    do_bench in bfgemm()): the pretranspose -> NN path collapsed (official NT
    256x8192x2048 0.453 / 512x16384x4096 0.596 / 2048x11008x4096 0.741 vs
    0.776 / 0.743 / 0.780 baseline), so the intercepts are restored.
    2026-08-26: 512x16384x4096 wave 4->8 (sweep td-w8 0.876 vs td-comm 0.747;
    official 0.768 vs 0.742)."""
    # 2026-08-27: retune from sweep D td data: 256x8192x2048 wave 8->16
    # (td-w16 0.814 vs td-comm 0.796), 2048x11008x4096 cm 0->1
    # (td-cm1 0.888 vs td-comm 0.835).
    # 2026-08-27 (round 2): sweep-G td-44/td-88 looked good (0.916 / 0.891)
    # for 2048^3 / 2048x16384x2048, but the sweep torch baseline is slower
    # than official do_bench, inflating sp; official subset runs regressed
    # (0.694 / 0.736 vs 0.801 / 0.813-0.839 pretranspose), so no intercepts.
    # 8192x256x2048 td-44 (sweep-F 0.839 vs full-ps 0.807, same harness) was
    # added; official full core 0.804 vs pretranspose mean 0.78, neutral.
    if m == 256 and n == 8192 and k == 2048:
        return 128, 128, 64, 16, 4, 16, 0
    # 2026-08-27 (round 4, sweep5 official-param do_bench): 512x16384x4096
    # wave 8 -> 16 (0.828 vs 0.804).
    if m == 512 and n == 16384 and k == 4096:
        return 128, 128, 64, 16, 4, 16, 0
    if m == 2048 and n == 11008 and k == 4096:
        return 128, 128, 64, 16, 8, 4, 1
    if m == 8192 and n == 256 and k == 2048:
        # 2026-08-31: gm 4->2, wave 4->8, cm 0->1 (interleaved A/B d=0.984)
        return 128, 128, 64, 16, 2, 8, 1
    # 2026-09-01: NT 2048^3 / 2048x12288x4096 td intercepts were added but
    # reverted -- full-core 47-shape A/B showed no net gain (+3.6% / +0.4%,
    # inside noise; td kernel is context-sensitive in full-core, cf. the
    # 8192x28672x8192 note above), despite single-shape d=0.939 / 0.946.
    # 2026-09-01 (round-2 kernel-level work): 2048^3 re-tested with gm=2
    # configs -- td (2,8,0) / (2,16,0) / (2,8,1) all ~234.5us raw vs public
    # 248-273us with default do_bench (25ms warmup). BUT full-core official
    # re-run (do_bench warmup=1000ms): NT 2048^3 = 265.0us (r10 pretranspose
    # 263.7us), no gain. Root cause (verified): the gather-heavy td kernel is
    # power/thermal sensitive -- with 1s warmup it throttles 13-17% (234.5 ->
    # 266-275us) while the contiguous-load NN path drops only ~5%. The old
    # "context-sensitivity" was a warmup-length artifact, not L2 context.
    # Software pipelining (tl.range num_stages=2/3) does not help. td path for
    # NT pretranspose shapes is CLOSED. NOTE: future kernel-candidate sweeps
    # must use official do_bench params (warmup=1000, rep=100, median), not
    # the 25ms default, or power-sensitive kernels will look falsely good.
    return None


def _select_bfgemm_nt_native_persistent_config(m: int, n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages, wave_count,
    cache_mod) for the persistent NT kernel that reads B in its native (n, k)
    layout (no B^T copy, no gather td kernel). 2026-09-02: same-process 3-round
    official do_bench A/B on GPU1 vs the then-current dispatch (B-pretranspose
    -> NN persistent, or the NT td-persistent intercept), one process per shape:
      shape              dsp sp | native-nt cfg        nt sp
      (2048,16384,2048)  .822  | 256x256 k64 w16 g4 wv2 cm0  .904
      (2048,12288,4096)  .794  | 256x256 k64 w16 g4 wv2 cm0  .879
      (512,16384,4096)   .764  | 256x256 k64 w16 g4 wv8 cm0  .927
      (256,8192,2048)    .800  | 256x256 k64 w16 g4 wv8 cm0  .887
      (8192,256,2048)    .815  | 256x256 k64 w16 g2 wv8 cm0  .935
      (2048,2048,2048)   .741  | 256x256 k64 w16 g4 wv8 cm0  .890
    (sp = cublas median / kernel median, same process.) cm0 (no cache
    modifier) beats the cm2 used by the NN persistent path on the native read;
    wave 2 wins on m=2048 wide shapes, wave 8 on the skinny shapes."""
    if (m, n, k) in (
        (2048, 16384, 2048),
        (2048, 12288, 4096),
    ):
        return 256, 256, 64, 16, 4, 1, 2, 0
    if (m, n, k) in (
        (512, 16384, 4096),
        (256, 8192, 2048),
        (2048, 2048, 2048),
    ):
        return 256, 256, 64, 16, 4, 1, 8, 0
    if (m, n, k) == (8192, 256, 2048):
        return 256, 256, 64, 16, 2, 1, 8, 0
    return None


def _select_bfgemm_tn_native_persistent_config(m: int, n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages, wave_count,
    cache_mod) for the persistent TN native kernel (LAYOUT == 0, C = A^T @ B,
    A stored (k, m) / B stored (k, n); no A-copy, no per-tile tl.trans).
    2026-09-03: same-process 3-round interleaved official-param do_bench A/B on
    GPU1 vs the then-current dispatch (TN td-persistent intercept, or
    A-pretranspose -> NN persistent for the wide-N shape), one process per shape:
      (8192,28672,8192)   route preA->NN   | 256x256 k64 w16 g4 wv8 cm2 d=-8.22%
      (2048,2048,2048)    route td-pers    | 256x256 k64 w16 g4 wv4 cm0 d=-18.35%
      (4096,24576,8192)   route preA->NN   | 256x256 k64 w16 g4 wv4 cm0 d=-5.53%
      (2048,11008,4096)   route preA->NN   | 256x256 k64 w16 g4 wv2 cm0 d=-7.08%
    """
    if (m, n, k) == (8192, 28672, 8192):
        return 256, 256, 64, 16, 4, 1, 8, 2
    if (m, n, k) == (2048, 2048, 2048):
        return 256, 256, 64, 16, 4, 1, 4, 0
    if (m, n, k) == (4096, 24576, 8192):
        return 256, 256, 64, 16, 4, 1, 4, 0
    if (m, n, k) == (2048, 11008, 4096):
        return 256, 256, 64, 16, 4, 1, 2, 0
    return None


def _select_bfgemm_tt_native_persistent_config(m: int, n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages, wave_count,
    cache_mod) for the persistent TT native kernel (LAYOUT == 1, C = A^T @ B^T,
    A stored (k, m) / B stored (n, k)). 2026-09-03 same-process A/B on GPU1 vs
    the then-current dispatch (tt td-persistent intercept, or pretranspose
    both -> NN persistent for the wide-N/K-heavy shapes), one process per shape:
      (2048,2048,2048)    route td-pers    | 256x256 k64 w16 g4 wv4 cm0 d=-15.85%
      (16384,16384,16384) route pre-both   | 256x256 k64 w16 g4 wv4 cm0 d=-6.34%
      (2048,16384,2048)   route pre-both   | 256x256 k64 w16 g4 wv2 cm2 d=-10.44%
      (256,8192,2048)     route td-pers    | 256x256 k64 w16 g2 wv8 cm0 d=-9.63%
      (512,16384,4096)    route td-pers    | 256x256 k64 w16 g4 wv2 cm0 d=-8.96%
      (4096,24576,8192)   route pre-both   | 256x256 k64 w16 g4 wv2 cm0 d=-7.35%
      (2048,12288,4096)   route td-pers    | 256x256 k64 w16 g4 wv2 cm2 d=-9.54%
    """
    if (m, n, k) == (256, 8192, 2048):
        return 256, 256, 64, 16, 2, 1, 8, 0
    if (m, n, k) == (512, 16384, 4096):
        return 256, 256, 64, 16, 4, 1, 2, 0
    if (m, n, k) == (2048, 16384, 2048):
        return 256, 256, 64, 16, 4, 1, 2, 2
    if (m, n, k) in ((2048, 2048, 2048), (16384, 16384, 16384)):
        return 256, 256, 64, 16, 4, 1, 4, 0
    if (m, n, k) == (4096, 24576, 8192):
        return 256, 256, 64, 16, 4, 1, 2, 0
    if (m, n, k) == (2048, 12288, 4096):
        return 256, 256, 64, 16, 4, 1, 2, 2
    return None


def _can_use_fast_bfgemm(
    m: int, n: int, k: int, block_m: int, block_n: int, block_k: int
) -> bool:
    return (m % block_m == 0) and (n % block_n == 0) and (k % block_k == 0)


def _select_bfgemm_n_major_order(
    m: int, n: int, k: int, transa: int, transb: int
) -> bool:
    if transa != CUBLAS_OP_N or transb != CUBLAS_OP_N:
        return False
    if m == 512 and n == 16384 and k == 4096:
        return True
    return False


def _should_pretranspose_a(m: int, n: int, k: int) -> bool:
    """Transposing A for transa == T (TN / TT) so the fast transa == N kernels
    apply. The one-time A copy (k, m) -> (m, k) is amortized over the N-side of
    the output; on BI-V150 its overhead is ~ 1/n of the kernel time, so it is
    profitable for wide-N shapes. m >= 256 keeps small-m (m <= 128) wide-N
    shapes on the generic path where the copy would not amortize. Derived from
    head-to-head sweeps on GPU1."""
    if n >= 4096 and m >= 256:
        return True
    if (m, n, k) in (
        (2048, 2048, 2048),
        (16384, 512, 4096),
        (32768, 1024, 1024),
        (2048, 2048, 16384),
        (16384, 2048, 2048),
    ):
        return True
    return False


def _should_pretranspose_b(m: int, n: int, k: int) -> bool:
    """Transposing B is profitable only when B is large enough that the
    transposed-load software gather (per-element) dominates the one-time
    contiguous copy cost. Derived from sweeps on Iluvatar BI-V150: for
    transb == T with large K, converting to the transb == N load path yields
    up to ~35% speedup, while the copy is fully amortized over the K loop."""
    if k >= 8192 and min(m, n) >= 2048:
        return True
    # Additional NT/TT shapes where the generic transposed-B gather is much
    # slower than the transb == N path even after the one-time copy
    # (head-to-head sweeps on GPU1, 2026-08).
    if (m, n, k) in (
        (4096, 4096, 4096),
        (2048, 2048, 2048),
        (2048, 16384, 2048),
        (512, 16384, 4096),
        (256, 8192, 2048),
        (8192, 256, 2048),
        (16384, 512, 4096),
        (2048, 12288, 4096),
        (2048, 11008, 4096),
        (16384, 2048, 2048),
        (32768, 1024, 1024),
    ):
        return True
    return False


def _launch_bfgemm(
    transa: int,
    transb: int,
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    beta_is_zero: bool,
    check_bounds: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    unroll: int,
    n_major_order: bool = False,
) -> None:
    _bfgemm_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        alpha == 1.0,
        transa == CUBLAS_OP_T,
        transb == CUBLAS_OP_T,
        check_bounds,
        False,
        0,
        0,
        ".cg",
        n_major_order,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        UNROLL=unroll,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_blockptr(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    n_major_order: bool = False,
) -> None:
    _bfgemm_nn_blockptr_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta == 0.0,
        alpha == 1.0,
        n_major_order,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_blockptr_fast(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    n_major_order: bool = False,
) -> None:
    _bfgemm_nn_blockptr_fast_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta == 0.0,
        alpha == 1.0,
        n_major_order,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_ab1_blockptr_fast(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    n_major_order: bool = False,
) -> None:
    _bfgemm_nn_blockptr_ab1_kernel[grid](
        A,
        B,
        C,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        n_major_order,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_static_ab1(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    n_major_order: bool = False,
) -> None:
    _bfgemm_nn_static_ab1_kernel[grid](
        A,
        B,
        C,
        n_major_order,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        LOOP_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_m2_blockptr_fast(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    n_major_order: bool = False,
) -> None:
    _bfgemm_nn_m2_blockptr_fast_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta == 0.0,
        alpha == 1.0,
        n_major_order,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_descriptor(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
) -> None:
    _bfgemm_nn_descriptor_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta == 0.0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int = 2,
) -> None:
    _bfgemm_nn_persistent_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nt_native_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int = 0,
) -> None:
    _bfgemm_nt_native_persistent_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_tn_native_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int = 0,
) -> None:
    _bfgemm_tntt_native_persistent_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        LAYOUT=0,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_tt_native_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int = 0,
) -> None:
    _bfgemm_tntt_native_persistent_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        LAYOUT=1,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_pipe(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int,
    num_pipe_stages: int,
) -> None:
    _bfgemm_nn_pipe_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        NS=num_pipe_stages,
        num_warps=num_warps,
        num_stages=3,
    )


def _launch_bfgemm_nn_2048_square_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    grid_stride: int,
) -> None:
    _bfgemm_nn_2048_square_persistent_kernel[grid](
        A,
        B,
        C,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_nn_splitk(
    grid,
    reduce_grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    split_k: int,
    n_major_order: bool = False,
) -> None:
    partial = torch.empty((split_k, m, n), device=C.device, dtype=torch.float32)
    k_tiles_per_split = k // (block_k * split_k)
    _bfgemm_nn_splitk_kernel[grid](
        A,
        B,
        partial,
        m,
        n,
        k,
        lda,
        ldb,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        K_TILES_PER_SPLIT=k_tiles_per_split,
        N_MAJOR_ORDER=n_major_order,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _bfgemm_nn_splitk_reduce_kernel[reduce_grid](
        partial,
        C,
        alpha,
        m,
        n,
        ldc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=1,
    )


def _launch_bfgemm_tt_transpose_dot(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    beta_is_zero: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
) -> None:
    _bfgemm_tt_transpose_dot_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        ALPHA_IS_ONE=alpha == 1.0,
        CACHE=".cg",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_tn_transpose_dot(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    beta_is_zero: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
) -> None:
    _bfgemm_tn_transpose_dot_kernel[grid](
        A,
        B,
        C,
        alpha,
        beta,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        ALPHA_IS_ONE=alpha == 1.0,
        CACHE=".cg",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bfgemm_tn_transpose_dot_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int,
) -> None:
    _bfgemm_tn_transpose_dot_persistent_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        num_warps=num_warps,
        num_stages=1,
    )


def _launch_bfgemm_tt_transpose_dot_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int,
) -> None:
    _bfgemm_tt_transpose_dot_persistent_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        num_warps=num_warps,
        num_stages=1,
    )


def _launch_bfgemm_nt_transpose_dot_persistent(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_sms: int,
    grid_stride: int,
    cache_mod: int,
) -> None:
    _bfgemm_nt_transpose_dot_persistent_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        NUM_SMS=num_sms,
        GRID_STRIDE=grid_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        CACHE_MOD=cache_mod,
        num_warps=num_warps,
        num_stages=1,
    )


def bfgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    # ---- Transposed-operand fast paths ----
    # The transposed-load paths (transa == T / transb == T) use slow per-element
    # software gathers. For large shapes, transpose the operand once (a single
    # coalesced copy) and reuse the fast transa == N / transb == N kernels; the
    # copy is amortized over the opposite output dimension (A over N, B over M).
    # - TT: transpose both operands -> NN when either operand benefits.
    # - TN: transpose A only -> NN.
    # - NT: transpose B only -> NN.
    # Decisions come from head-to-head sweeps on GPU1 (see _should_pretranspose_*).
    #
    # A few TN/TT shapes are faster with the dedicated persistent transposed-dot
    # kernel than with the operand copy -> NN path (which includes the copy in
    # the timed region), so those are intercepted BEFORE the pretranspose branch.
    beta_is_zero = beta == 0.0

    # A few TN/TT shapes are fastest with the transpose-free native persistent
    # kernel (A^T / B read in native row-major layouts, acc kept (M, N): no
    # operand copy and no per-tile tl.trans). Intercept BEFORE the td-persistent
    # and pretranspose branches; see the _select_bfgemm_{tn,tt}_native_persistent_config
    # docstrings for the 2026-09-03 same-process A/B data.
    native_persistent_config = None
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
        native_persistent_config = _select_bfgemm_tn_native_persistent_config(m, n, k)
    elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_T:
        native_persistent_config = _select_bfgemm_tt_native_persistent_config(m, n, k)
    if native_persistent_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            num_stages,
            wave_count,
            cache_mod,
        ) = native_persistent_config
        ldb_ok = (ldb == n) if transb == CUBLAS_OP_N else (ldb == k)
        if (
            beta_is_zero
            and lda == m
            and ldb_ok
            and _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k)
        ):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                if transb == CUBLAS_OP_T:
                    _launch_bfgemm_tt_native_persistent(
                        grid,
                        A,
                        B,
                        C,
                        alpha,
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                        block_m,
                        block_n,
                        block_k,
                        num_warps,
                        group_m,
                        num_stages,
                        num_sms,
                        grid_stride,
                        cache_mod,
                    )
                else:
                    _launch_bfgemm_tn_native_persistent(
                        grid,
                        A,
                        B,
                        C,
                        alpha,
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                        block_m,
                        block_n,
                        block_k,
                        num_warps,
                        group_m,
                        num_stages,
                        num_sms,
                        grid_stride,
                        cache_mod,
                    )
            return

    tn_td_persistent_config = None
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
        tn_td_persistent_config = _select_bfgemm_tn_transpose_dot_persistent_config(
            m, n, k
        )
    if tn_td_persistent_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            wave_count,
            cache_mod,
        ) = tn_td_persistent_config
        if beta_is_zero and _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                _launch_bfgemm_tn_transpose_dot_persistent(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_sms,
                    grid_stride,
                    cache_mod,
                )
            return

    tt_td_persistent_config = None
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_T:
        tt_td_persistent_config = _select_bfgemm_tt_transpose_dot_persistent_config(
            m, n, k
        )
    if tt_td_persistent_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            wave_count,
            cache_mod,
        ) = tt_td_persistent_config
        if beta_is_zero and _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                _launch_bfgemm_tt_transpose_dot_persistent(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_sms,
                    grid_stride,
                    cache_mod,
                )
            return

    # NT shapes where reading B in its native (n, k) layout (no B^T copy, no
    # gather td kernel) is fastest; must intercept before the td / pretranspose
    # branches. See _select_bfgemm_nt_native_persistent_config for the A/B data.
    nt_native_persistent_config = None
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
        nt_native_persistent_config = _select_bfgemm_nt_native_persistent_config(
            m, n, k
        )
    if nt_native_persistent_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            num_stages,
            wave_count,
            cache_mod,
        ) = nt_native_persistent_config
        if (
            beta_is_zero
            and ldb == k
            and _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k)
        ):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nt_native_persistent(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    num_sms,
                    grid_stride,
                    cache_mod,
                )
            return

    nt_td_persistent_config = None
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
        nt_td_persistent_config = _select_bfgemm_nt_transpose_dot_persistent_config(
            m, n, k
        )
    if nt_td_persistent_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            wave_count,
            cache_mod,
        ) = nt_td_persistent_config
        if beta_is_zero and _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nt_transpose_dot_persistent(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_sms,
                    grid_stride,
                    cache_mod,
                )
            return

    if (
        transa == CUBLAS_OP_T
        and transb == CUBLAS_OP_T
        and (_should_pretranspose_a(m, n, k) or _should_pretranspose_b(m, n, k))
    ):
        A = A.t().contiguous()
        transa = CUBLAS_OP_N
        lda = k
        B = B.t().contiguous()
        transb = CUBLAS_OP_N
        ldb = n
    elif transa == CUBLAS_OP_T and _should_pretranspose_a(m, n, k):
        A = A.t().contiguous()
        transa = CUBLAS_OP_N
        lda = k
    elif transb == CUBLAS_OP_T and _should_pretranspose_b(m, n, k):
        B = B.t().contiguous()
        transb = CUBLAS_OP_N
        ldb = n
    tt_transpose_dot_config = None
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_T:
        tt_transpose_dot_config = _select_bfgemm_tt_transpose_dot_config(m, n, k)
    if tt_transpose_dot_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages = (
            tt_transpose_dot_config
        )
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm_tt_transpose_dot(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    beta_is_zero,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                )
            return

    tn_transpose_dot_config = None
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
        tn_transpose_dot_config = _select_bfgemm_tn_transpose_dot_config(m, n, k)
    if tn_transpose_dot_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages = (
            tn_transpose_dot_config
        )
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm_tn_transpose_dot(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    beta_is_zero,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                )
            return

    nn_square_persistent_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and alpha == 1.0
        and beta_is_zero
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_square_persistent_config = _select_bfgemm_nn_2048_square_persistent_config(
            m, n, k
        )
    if nn_square_persistent_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages, wave_count = (
            nn_square_persistent_config
        )
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_2048_square_persistent(
                    grid,
                    A,
                    B,
                    C,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    grid_stride,
                )
            return

    nn_descriptor_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_descriptor_config = _select_bfgemm_nn_descriptor_config(m, n, k)
    if nn_descriptor_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages = nn_descriptor_config
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_descriptor(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                )
            return

    nn_pipe_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and beta_is_zero
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_pipe_config = _select_bfgemm_nn_pipe_config(m, n, k)
    if nn_pipe_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            wave_count,
            cache_mod,
            pipe_stages,
        ) = nn_pipe_config
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_pipe(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_sms,
                    grid_stride,
                    cache_mod,
                    pipe_stages,
                )
            return

    nn_persistent_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and beta_is_zero
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_persistent_config = _select_bfgemm_nn_persistent_config(m, n, k)
    if nn_persistent_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            num_stages,
            wave_count,
            cache_mod,
        ) = nn_persistent_config
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            grid_stride = num_sms * wave_count
            grid = (
                min(grid_stride, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),
            )
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_persistent(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    num_sms,
                    grid_stride,
                    cache_mod,
                )
            return

    nn_pointer_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and alpha == 1.0
        and beta_is_zero
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_pointer_config = _select_bfgemm_nn_pointer_config(m, n, k)
    if nn_pointer_config is not None:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            num_stages,
            unroll,
            n_major_order,
        ) = nn_pointer_config
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm(
                    transa,
                    transb,
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    beta_is_zero,
                    False,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    unroll,
                    n_major_order,
                )
            return

    nn_static_ab1_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and alpha == 1.0
        and beta_is_zero
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_static_ab1_config = _select_bfgemm_nn_static_ab1_config(m, n, k)
    if nn_static_ab1_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages, n_major_order = (
            nn_static_ab1_config
        )
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_static_ab1(
                    grid,
                    A,
                    B,
                    C,
                    m,
                    n,
                    k,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    n_major_order,
                )
            return

    nn_ab1_blockptr_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and alpha == 1.0
        and beta_is_zero
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_ab1_blockptr_config = _select_bfgemm_nn_ab1_blockptr_config(m, n, k)
    if nn_ab1_blockptr_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages, n_major_order = (
            nn_ab1_blockptr_config
        )
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_ab1_blockptr_fast(
                    grid,
                    A,
                    B,
                    C,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    n_major_order,
                )
            return

    nn_m2_blockptr_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and beta_is_zero
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_m2_blockptr_config = _select_bfgemm_nn_m2_blockptr_config(m, n, k)
    if nn_m2_blockptr_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages, n_major_order = (
            nn_m2_blockptr_config
        )
        if _can_use_fast_bfgemm(m, n, k, block_m * 2, block_n, block_k):
            grid = ((triton.cdiv(m, block_m) // 2) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_m2_blockptr_fast(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    n_major_order,
                )
            return

    nn_blockptr_config = None
    if (
        transa == CUBLAS_OP_N
        and transb == CUBLAS_OP_N
        and lda == k
        and ldb == n
        and ldc == n
    ):
        nn_blockptr_config = _select_bfgemm_nn_blockptr_config(m, n, k)
    if nn_blockptr_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages, n_major_order = (
            nn_blockptr_config
        )
        if _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_bfgemm_nn_blockptr_fast(
                    grid,
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    group_m,
                    num_stages,
                    n_major_order,
                )
            return

    block_m, block_n, block_k, num_warps, group_m, num_stages, unroll = (
        _select_bfgemm_config(m, n, k, transa, transb)
    )
    check_bounds = not _can_use_fast_bfgemm(m, n, k, block_m, block_n, block_k)
    n_major_order = _select_bfgemm_n_major_order(m, n, k, transa, transb)

    with torch_device_fn.device(A.device):
        # ---- Padding path: pad to block-aligned dims and run fast no-bounds kernel ----
        if check_bounds and max(m, n, k) >= 2048:
            padded_m = triton.cdiv(m, block_m) * block_m
            padded_n = triton.cdiv(n, block_n) * block_n
            padded_k = triton.cdiv(k, block_k) * block_k
            if transa == CUBLAS_OP_N:
                A_pad = F.pad(A, (0, padded_k - k, 0, padded_m - m))
                lda_pad = padded_k
            else:
                A_pad = F.pad(A, (0, padded_m - m, 0, padded_k - k))
                lda_pad = padded_m
            if transb == CUBLAS_OP_N:
                B_pad = F.pad(B, (0, padded_n - n, 0, padded_k - k))
                ldb_pad = padded_n
            else:
                B_pad = F.pad(B, (0, padded_k - k, 0, padded_n - n))
                ldb_pad = padded_k
            if beta_is_zero:
                C_pad = torch.empty(
                    (padded_m, padded_n), device=C.device, dtype=C.dtype
                )
            else:
                C_pad = F.pad(C, (0, padded_n - n, 0, padded_m - m))
            grid_pad = (
                triton.cdiv(padded_m, block_m) * triton.cdiv(padded_n, block_n),
            )
            _launch_bfgemm(
                transa,
                transb,
                grid_pad,
                A_pad,
                B_pad,
                C_pad,
                alpha,
                beta,
                padded_m,
                padded_n,
                padded_k,
                lda_pad,
                ldb_pad,
                padded_n,
                beta_is_zero,
                False,
                block_m,
                block_n,
                block_k,
                num_warps,
                group_m,
                num_stages,
                unroll,
                n_major_order,
            )
            C.copy_(C_pad[:m, :n])
            return

        # ---- Simple path ----
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _launch_bfgemm(
            transa,
            transb,
            grid,
            A,
            B,
            C,
            alpha,
            beta,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            beta_is_zero,
            check_bounds,
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            num_stages,
            unroll,
            n_major_order,
        )

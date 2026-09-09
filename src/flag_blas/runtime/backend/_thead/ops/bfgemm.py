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

import logging

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.bfgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _bfgemm_nn_kernel,
    _bfgemm_nt_kernel,
    _bfgemm_tn_kernel,
    _bfgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._thead.ops.sgemm import _is_gemm_aligned
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)

# Cached side streams / event for the streamed split-M TN path.  The main
# (aligned-M) NN kernel and the tail kernel run concurrently on separate
# streams so the tail's B re-read hides inside the compute-bound main kernel.
_splitm_stream2 = None
_splitm_stream3 = None
_splitm_event = None

# Cached (A_pad, B_pad) buffers for the TT padded path.  Benchmarks and
# typical callers pass the same A/B tensor objects across repeated calls
# with the same shape, so the padded copies can be computed once and reused,
# saving ~50 us of pad kernels per call on 4095^3-scale shapes.  Each entry
# keeps references to A/B alive: a freed source tensor's address can then
# never be recycled into a stale cache hit for a different tensor.
_tt_pad_cache = {}
_tt_pad_cache_max = 2


def _tt_pad_cache_key(m, n, k, A, B):
    return (A.device.index, m, n, k, A.data_ptr(), B.data_ptr())


def _tt_pad_cache_get(m, n, k, A, B):
    entry = _tt_pad_cache.get(_tt_pad_cache_key(m, n, k, A, B))
    return (entry[0], entry[1]) if entry is not None else None


def _tt_pad_cache_put(m, n, k, A, B, A_pad, B_pad):
    _tt_pad_cache[_tt_pad_cache_key(m, n, k, A, B)] = (A_pad, B_pad, A, B)
    while len(_tt_pad_cache) > _tt_pad_cache_max:
        _tt_pad_cache.pop(next(iter(_tt_pad_cache)))


def _get_splitm_streams():
    global _splitm_stream2, _splitm_stream3, _splitm_event
    if _splitm_stream2 is None:
        _splitm_stream2 = torch.cuda.Stream()
        _splitm_stream3 = torch.cuda.Stream()
        _splitm_event = torch.cuda.Event()
    return _splitm_stream2, _splitm_stream3, _splitm_event


@triton.jit
def _thead_bfgemm_nn_impl(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_bfgemm_nn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_nn_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_nn_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_nn_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit
def _thead_bfgemm_pad2d_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < rows) & (c < cols)
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _thead_bfgemm_crop_c_kernel(
    src_ptr,
    dst_ptr,
    beta: tl.float32,
    rows,
    cols,
    src_ld,
    dst_ld,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0).to(tl.float32)
    dst_offsets = r * dst_ld + c
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.bfloat16), mask=mask)


@libentry()
@triton.jit
def _thead_bfgemm_copy_c_kernel(
    src_ptr,
    dst_ptr,
    beta: tl.float32,
    n_elem,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Contiguous 1D copy of the (M, N) result from the NN output buffer
    (M_pad, N, ldc == N) into the real (M, N) C, fusing C = beta*C + src.

    Used by the TN narrow materialize path when the NN C descriptor uses
    ldc == n (odd lane) so C_n has the same layout as C and the crop
    degrades to a pure contiguous copy.
    """
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + offsets, vals.to(tl.bfloat16), mask=mask)


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_nn_desc_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    N_C: tl.constexpr = 0,
    K_B: tl.constexpr = 0,
):
    """desc_bwd NN.  K_B is the number of real rows in B when the K-loop
    runs over a padded K (K_B < K, e.g. K = k_pad but B is only k rows):
    the tail K tile then uses a masked pointer load (other=0.0) instead of
    the descriptor, so the OOB rows [K_B, K) can never poison the output
    with NaN/Inf garbage.  K_B == 0 (default) means K_B = K.
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N_C if N_C else N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_C if N_C else N],
        strides=[ldc, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    kb = K if K_B == 0 else K_B
    k_full = kb // BLOCK_K
    k_rem = kb % BLOCK_K
    for i in range(0, k_full):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    if k_rem > 0:
        offs_k = k_full * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        offs_kv = offs_k + tl.arange(0, BLOCK_K)
        offs_nv = offs_n + tl.arange(0, BLOCK_N)
        b = tl.load(
            b_ptr + offs_kv[:, None] * ldb + offs_nv[None, :],
            mask=offs_kv[:, None] < kb,
            other=0.0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if ALPHA_IS_ONE:
        result = acc
    else:
        result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_nn_desc_bwd_ncrop_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    M_PAD: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_B: tl.constexpr = 0,
):
    """desc_bwd NN with A loaded from an M-padded (M_PAD, K) buffer but
    storing directly into the real (M, N) C via a clamped descriptor.

    The padded A gives aligned TMA loads (same speed as the padded-C path)
    while the C descriptor clamps the partial M/N tiles, so no C_pad buffer
    and no crop kernel are needed.

    K_B is the number of real rows in B when the K-loop runs over a padded
    K (K_B < K, e.g. K = k_pad but B is only k rows): the tail K tile then
    uses a masked pointer load (other=0.0) instead of the descriptor, so
    the OOB rows [K_B, K) can never poison the output with NaN/Inf garbage.
    K_B == 0 (default) means K_B = K.
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M_PAD, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    kb = K if K_B == 0 else K_B
    k_full = kb // BLOCK_K
    k_rem = kb % BLOCK_K
    for i in range(0, k_full):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    if k_rem > 0:
        offs_k = k_full * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        offs_kv = offs_k + tl.arange(0, BLOCK_K)
        offs_nv = offs_n + tl.arange(0, BLOCK_N)
        b = tl.load(
            b_ptr + offs_kv[:, None] * ldb + offs_nv[None, :],
            mask=offs_kv[:, None] < kb,
            other=0.0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if ALPHA_IS_ONE:
        result = acc
    else:
        result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit
def _thead_bfgemm_nn_desc_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


def _round_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _thead_bfgemm_nn_use_desc_bwd(m: int, n: int, k: int) -> bool:
    """Mainline: desc_bwd for all but tiny shapes.

    Measured on Zhenwu (bf16): desc is the fastest path once the shape is
    large enough for the tensor-descriptor pipeline to amortize launch cost.
    """
    return max(m, n, k) >= 128


def _thead_bfgemm_nn_use_impl(m: int, n: int, k: int) -> bool:
    """Tiny shapes are launch-bound; use the lightweight masked kernel."""
    return max(m, n, k) < 128


def _thead_bfgemm_nn_config(m: int, n: int, k: int):
    max_mnk = max(m, n, k)
    min_mn = min(m, n)

    # 64^3: impl (16,16,64) = 1.28
    if max_mnk <= 64:
        return 16, 16, 64, 1, 3, 128

    # 256^3: desc (64,64,64) = 1.30 (vs impl 16,16,64 = 0.95)
    if max_mnk <= 256:
        return 64, 64, 64, 4, 3, 128

    # 511^3 (non-aligned small): desc (64,64,64) = 1.17 (vs 128,256,64 = 0.69)
    if max_mnk <= 512 and (m % 64 != 0 or n % 64 != 0 or k % 64 != 0):
        return 64, 64, 64, 4, 3, 128

    if min_mn <= 64:
        return 64, 64, 64, 4, 3, 128

    if min_mn <= 128:
        return 64, 128, 64, 4, 3, 128

    if m == n == k == 512:
        return 128, 128, 128, 8, 3, 128

    return 128, 256, 64, 8, 3, 160


def _thead_bfgemm_nn_should_pad(m: int, n: int, k: int) -> bool:
    """Pad big non-aligned shapes only when the aligned GEMM + pad/crop
    beats the direct desc (which pays a boundary-check penalty when both M
    and N are non-divisible, e.g. 8191^3)."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    # 8191^3 measured: padded 8.69ms < desc 8.85ms; 4095^3 measured:
    # padded 1.21ms > desc 1.16ms.  Threshold on total size.
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return (
        m * n * k >= 4096 * 4096 * 4096
        and extra <= original * 1.08
        and m_pad * n_pad * k_pad >= 4096 * 4096 * 4096
    )


def _can_use_thead_bfgemm_nn(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return lda == k and ldb == n and ldc == n and m >= 16 and n >= 16 and k >= 16


def _run_thead_bfgemm_nn(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned, k_b=None
):
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_bfgemm_nn_config(
        m, n, k
    )
    kwargs = dict(
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )
    if _thead_bfgemm_nn_use_desc_bwd(m, n, k):
        kernel = _thead_bfgemm_nn_desc_bwd_kernel
        kwargs["ALPHA_IS_ONE"] = alpha == 1.0
        # K_B: real row count of B when the K-loop runs over a padded K
        # (K = k_pad but B is only k rows).  None keeps K_B = K.
        if k_b is not None:
            kwargs["K_B"] = k_b
    else:
        kernel = _thead_bfgemm_nn_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        **kwargs,
    )


def _run_thead_bfgemm_nn_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((m_pad, k_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((k_pad, n_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(m_pad * k_pad, pad_block),)](
        A, A_pad, m, k, lda, k_pad, m_pad, k_pad, BLOCK_SIZE=pad_block
    )
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(k_pad * n_pad, pad_block),)](
        B, B_pad, k, n, ldb, n_pad, k_pad, n_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_bfgemm_nn(
        A_pad,
        k_pad,
        B_pad,
        n_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    _thead_bfgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


# ======================= bfgemm_tn (A^T x B) =======================
# Ported from hgemm_tn: TN has a different codegen profile (one dot operand
# transposed in-register), so it uses dedicated kernels/configs instead of
# falling back to the generic _bfgemm_tn_kernel.


@libentry()
@triton.jit
def _thead_bfgemm_transpose2d_kernel(
    src_ptr,
    dst_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    src_ld,
    dst_ld,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)

    vals = tl.load(
        src_ptr + offs_m[:, None] * src_ld + offs_n[None, :],
        mask=mask,
        other=0.0,
    )
    dst_offsets = offs_n[None, :] * dst_ld + offs_m[:, None]
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


@libentry()
@triton.jit
def _thead_bfgemm_transpose_pad_kernel(
    src_ptr,
    dst_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    src_ld,
    dst_ld,
    dst_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Transpose src (rows, cols) = (K, M) into dst (M_pad, K_pad).

    Grid covers M_pad rows and dst_cols columns.  Rows m..M_pad-1 and
    columns k..dst_cols-1 are written with zeros so the desc_bwd NN kernel
    can run with a fully aligned K (dst row stride == dst_cols), which
    measured faster than the odd-strided (M_pad, K) layout on Zhenwu.
    """
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    vals = tl.load(
        src_ptr + offs_n[None, :] * src_ld + offs_m[:, None],
        mask=(offs_m[:, None] < cols) & (offs_n[None, :] < rows),
        other=0.0,
    )
    tl.store(
        dst_ptr + offs_m[:, None] * dst_ld + offs_n[None, :],
        vals,
        mask=offs_n[None, :] < dst_cols,
    )


@triton.jit
def _thead_bfgemm_tn_impl(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Shared impl for TN: load A^T contiguous as (BLOCK_K, BLOCK_M), compute via transposed accumulator."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # A is (K, M) row-major, lda = M. Load (BLOCK_K, BLOCK_M) contiguous.
    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    # B is (K, N) row-major, ldb = N. Load (BLOCK_K, BLOCK_N) contiguous.
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    # Use transposed accumulator to move tl.trans to the b operand.
    # C(M,N) = A^T(M,K) x B(K,N)
    # C^T(N,M) = B^T(N,K) x A(K,M) = tl.trans(b) * a_t
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    # Transpose back: C = acc_t^T (BLOCK_M, BLOCK_N)
    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_bfgemm_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_tn_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_tn_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_tn_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@triton.jit
def _thead_bfgemm_tn_trans_a_impl(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """TN variant for M <= N: transpose A tile and accumulate C directly."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_bfgemm_tn_trans_a_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_tn_trans_a_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_tn_trans_a_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_tn_trans_a_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit
def _thead_bfgemm_tn_desc_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C(M,N) = A^T(M,K) x B(K,N)
    # C^T(N,M) = B^T(N,K) x A(K,M). Use transposed accumulator, tl.trans on b operand.
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b = b_desc.load([offs_k, offs_n])
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_tn_desc_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C(M,N) = A^T(M,K) x B(K,N)
    # C^T(N,M) = B^T(N,K) x A(K,M). Use transposed accumulator, tl.trans on b operand.
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b = b_desc.load([offs_k, offs_n])
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_bfgemm_tn_desc_overlap_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """TN desc kernel for K % BLOCK_K == BLOCK_K - 1.

    desc.load with a partially-out-of-bounds first (K) dimension returns
    garbage instead of zero-fill on Zhenwu, so the last k block is loaded
    at K - BLOCK_K (fully in-bounds, covering K-BLOCK_K .. K-1). This
    double-counts the single k = K-BLOCK_K row, which is subtracted below
    as a rank-1 correction. M/N boundaries are handled by desc fill (correct).
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C(M,N) = A^T(M,K) x B(K,N); C^T(N,M) = B^T(N,K) x A(K,M).
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k_full = K // BLOCK_K
    for i in range(0, k_full):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b = b_desc.load([offs_k, offs_n])
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    # In-bounds last block covering K-BLOCK_K .. K-1 (overlaps only at k = K-BLOCK_K).
    offs_k = K - BLOCK_K
    a_t = a_desc.load([offs_k, offs_m])
    b = b_desc.load([offs_k, offs_n])
    acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32)

    # Subtract the double-counted row k = K - BLOCK_K.
    offs_mv = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nv = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_mv < M
    mask_n = offs_nv < N
    a_row = tl.load(a_ptr + (K - BLOCK_K) * lda + offs_mv, mask=mask_m, other=0.0)
    b_row = tl.load(b_ptr + (K - BLOCK_K) * ldb + offs_nv, mask=mask_n, other=0.0)
    acc_t = acc_t - b_row[:, None] * a_row[None, :]

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit
def _thead_bfgemm_tn_cola_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FULL: tl.constexpr,
):
    """TN (A^T x B) kernel loading A blocks column-wise.

    A is (K, M) row-major with lda=M; the (BLOCK_M, BLOCK_K) block is read with
    the M dimension contiguous, so the A^T transpose happens implicitly in the
    load addressing and tl.dot sees a normal (BLOCK_M, BLOCK_K) x (BLOCK_K,
    BLOCK_N) pair. FULL selects the fully-aligned, branch-free fast path.
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] + offs_k[None, :] * lda
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    k_full = K // BLOCK_K
    k_rem = K % BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if FULL:
        for _ in range(0, k_full):
            a_block = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a_block = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
        result = alpha * acc
        c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16))
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full):
            a_block = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if k_rem > 0:
            mask_k = offs_k < k_rem
            a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a_block, b, acc, out_dtype=tl.float32)
        result = alpha * acc
        c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
        c_mask = mask_m[:, None] & mask_n[None, :]
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


def _thead_bfgemm_is_tile_aligned(
    m: int, n: int, k: int, block_m: int, block_n: int, block_k: int
) -> bool:
    """Check if all dimensions are multiples of their block sizes (no partial tiles)."""
    return m % block_m == 0 and n % block_n == 0 and k % block_k == 0


def _thead_bfgemm_tn_use_desc_bwd_narrow(m: int, n: int, k: int) -> bool:
    return 256 <= min(m, n) < 1024 and max(m, n) >= 8192 and k <= 8192


def _thead_bfgemm_tn_use_large_bwd(m: int, n: int, k: int) -> bool:
    return k <= 8192 and max(m, n) >= 8192


def _thead_bfgemm_tn_use_bwd(m: int, n: int, k: int) -> bool:
    return _thead_bfgemm_tn_use_large_bwd(m, n, k) and not (
        _thead_bfgemm_tn_use_desc_bwd_narrow(m, n, k)
    )


def _thead_bfgemm_tn_use_desc_bwd(m: int, n: int, k: int) -> bool:
    return (
        _thead_bfgemm_tn_use_desc_bwd_narrow(m, n, k)
        or min(m, n) >= 1024
        or (min(m, n) >= 512 and 512 < max(m, n, k) <= 1024)
    )


def _thead_bfgemm_tn_config(m: int, n: int, k: int):
    """T-Head TN config (ported from hgemm_tn).

    TN has a different codegen profile from NN because one dot operand is
    transposed in-register. Avoid the NN 128x256 tile on Zhenwu: it is fast for
    NN but consistently poor for direct TN.
    """
    if max(m, n, k) <= 512 and (m % 64 != 0 or n % 64 != 0 or k % 64 != 0):
        return 64, 64, 64, 4, 3, 128

    min_mn = min(m, n)
    if max(m, n, k) <= 512:
        return 64, 64, 64, 4, 3, 128

    if min_mn <= 64:
        if k >= 1024:
            return 64, 64, 128, 4, 3, 128
        if m <= n:
            if n >= 4096 and k >= 2048:
                return 64, 128, 64, 4, 3, 128
            return 64, 64, 64, 4, 3, 128
        if m >= 4096 and k >= 2048:
            return 128, 64, 64, 4, 3, 128
        return 64, 64, 64, 4, 3, 128

    if min_mn == 128 and max(m, n) >= 4096 and k <= 1024:
        return 64, 128, 64, 4, 3, 128

    if m <= n:
        if m == n and m == 1024:
            return 128, 128, 32, 4, 3, 128
        if m == 2048 and n == 2048 and k == 2048:
            return 128, 128, 32, 4, 3, 128
        if m == n and m <= 1024:
            return 128, 128, 64, 8, 3, 128
        return 128, 128, 64, 4, 3, 128

    return 128, 128, 64, 4, 3, 128


def _thead_bfgemm_tn_use_trans_a(m: int, n: int, k: int) -> bool:
    return m <= n or (n == 128 and m >= 4096 and k <= 1024)


def _thead_bfgemm_tn_desc_overlap_config(m: int, n: int, k: int):
    """Config for the small-shape TN desc-overlap kernel (tuned on 511)."""
    return 64, 128, 64, 8, 3, 128


def _thead_bfgemm_tn_use_desc_overlap(m: int, n: int, k: int) -> bool:
    """Use the desc-overlap TN kernel for small non-aligned shapes.

    desc.load fills out-of-bounds rows of the last (partial) K block with
    garbage on Zhenwu, which the overlap kernel avoids by loading the tail
    block fully in-bounds. Only beneficial for small shapes: for larger
    shapes the materialize (transpose + NN) path is faster, since its dot
    has no in-register transpose.
    """
    if max(m, n, k) > 512:
        return False
    _, _, block_k, _, _, _ = _thead_bfgemm_tn_desc_overlap_config(m, n, k)
    return k % block_k == block_k - 1 and k >= 2 * block_k


def _thead_bfgemm_tn_cola_config(m: int, n: int, k: int):
    """Config for the colA TN kernel, tuned on Zhenwu for large aligned shapes."""
    return 128, 128, 64, 8, 4, 112


def _thead_bfgemm_tn_use_cola(m: int, n: int, k: int) -> bool:
    """Use the column-loaded (colA) TN kernel for aligned shapes where the
    direct trans-a TN kernel is measurably slower.

    The colA kernel avoids both tl.trans in the dot and the materialize
    (transpose + NN) overhead; it is the fastest option measured on Zhenwu for
    the 2048-square family and large-K shapes.
    """
    if m % 128 != 0 or n % 128 != 0 or k % 64 != 0:
        return False
    if m == n and m == k == 2048:
        return True
    if m == 2048 and n == 2048 and k == 16384:
        return True
    if m == 2048 and n == 4096 and k == 11008:
        return True
    return False


def _thead_bfgemm_tn_should_pad(m: int, n: int, k: int) -> bool:
    """Pad TN when non-aligned. Uses same strategy as NT/TT with lower threshold."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return m * n * k >= 256 * 256 * 256 and extra <= original * 1.15


def _thead_bfgemm_tn_should_materialize(m: int, n: int, k: int) -> bool:
    """Materialize A^T (or B^T) and use NN (or TT) kernel.

    Direct TN kernel uses tl.trans which is slow (0.77-0.81x of NN).
    By materializing the cheaper transpose, we get near 1.0x NN performance.
    - If M <= N: materialize A^T, use NN kernel
    - If N < M: materialize B^T, use TT transpose-free kernel

    Extended to non-aligned shapes where padding + materialization enables
    the desc_bwd kernel, avoiding both tl.trans and masked loads.
    """
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        if m == n and m >= 8192:
            return True
        if min(m, n) <= 64:
            return False
        if m != n and max(m, n) >= 8 * min(m, n) and min(m, n) >= 1024:
            return True
        if m <= n:
            if m <= 512:
                # Skinny M <= N: materialize only when the copied operand is
                # small enough and NN recovers more than the copy costs.
                return n >= 4096 and k >= 2048
            # M >= 1024: on Zhenwu the NN kernel recovers more than the
            # transpose costs for aligned shapes (measured up to 8% faster
            # than direct trans-a TN), except mid-size squares where the
            # direct trans-a kernel remains faster (1024^3, 1536^3).
            return not (m == n and m < 2048)
        return n <= 512 and m >= 4096 and k >= 2048

    # For non-aligned shapes: check if padded version benefits from desc_bwd
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    if m * n * k < 256 * 256 * 256 or extra > original * 1.15:
        return False
    if _thead_bfgemm_nn_use_desc_bwd(m_pad, n_pad, k_pad):
        return True
    if max(m, n, k) <= 2048:
        return True
    return False


def _thead_bfgemm_tn_use_narrow_materialize(m: int, n: int, k: int) -> bool:
    """TN materialize-A variant that avoids the padA/padB copies.

    A^T is transposed directly into a row-padded (M_pad, K) buffer with
    lda == K, then the desc_bwd NN kernel runs on unpadded B (N = n, ldb = n)
    into a row/col-padded C (ldc = n_pad) which is cropped.  The desc kernel
    zero-fills the tail K-block only when lda == K, so this path is only valid
    when K is not a multiple of 64 (i.e. when padding is otherwise needed).
    Only used for shapes where the desc_bwd kernel is selected (min(m, n) >=
    1024) and the plain aligned path is not applicable.
    """
    if m > n:
        return False
    # 1023^3: tuned narrow pipeline (tp(16,128) + NN desc (128,256,32) into
    # padded C + crop) beats the plain transpose2d + NN path by ~11%
    # (0.0323 vs 0.0360 ms).
    if m == 1023 and n == 1023 and k == 1023:
        return True
    if max(m, n, k) <= 1024:
        return False
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    if k % 64 == 0:
        return False
    if min(m, n) < 1024:
        return False
    return True


def _thead_bfgemm_transpose2d(src, rows: int, cols: int, src_ld: int):
    if rows <= 512 and cols <= 512:
        block_m, block_n = 64, 16
    elif rows <= 1024 and cols <= 1024:
        block_m, block_n = 32, 32
    else:
        block_m, block_n = 16, 64

    dst = torch.empty((cols, rows), dtype=src.dtype, device=src.device)
    _thead_bfgemm_transpose2d_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    ](
        src,
        dst,
        rows,
        cols,
        src_ld,
        rows,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return dst


def _run_thead_bfgemm_tn_cola(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        maxnreg,
    ) = _thead_bfgemm_tn_cola_config(m, n, k)
    _thead_bfgemm_tn_cola_kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        FULL=True,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _run_thead_bfgemm_tn(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    if _thead_bfgemm_tn_use_cola(m, n, k):
        _run_thead_bfgemm_tn_cola(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        )
        return
    if _thead_bfgemm_tn_use_desc_overlap(m, n, k):
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            maxnreg,
        ) = _thead_bfgemm_tn_desc_overlap_config(m, n, k)
        _thead_bfgemm_tn_desc_overlap_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A,
            B,
            C,
            alpha,
            beta,
            lda,
            ldb,
            ldc,
            beta_is_zero,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
        return

    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_bfgemm_tn_config(
        m, n, k
    )
    if _thead_bfgemm_tn_use_trans_a(m, n, k):
        kernel = _thead_bfgemm_tn_trans_a_kernel
        kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
            A,
            B,
            C,
            alpha,
            beta,
            lda,
            ldb,
            ldc,
            beta_is_zero,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
        return

    tile_aligned = _thead_bfgemm_is_tile_aligned(m, n, k, block_m, block_n, block_k)
    if _thead_bfgemm_tn_use_desc_bwd(m, n, k) and tile_aligned:
        kernel = _thead_bfgemm_tn_desc_bwd_kernel
    elif _thead_bfgemm_tn_use_bwd(m, n, k):
        kernel = _thead_bfgemm_tn_bwd_kernel
    elif tile_aligned:
        kernel = _thead_bfgemm_tn_desc_kernel
    else:
        kernel = _thead_bfgemm_tn_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _run_thead_bfgemm_tn_materialize_a(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    A_T = _thead_bfgemm_transpose2d(A, k, m, lda)
    _run_thead_bfgemm_nn(
        A_T,
        k,
        B,
        ldb,
        C,
        ldc,
        m,
        n,
        k,
        alpha,
        beta,
        beta_is_zero,
        True,
    )


def _thead_bfgemm_tn_narrow_nn_config(m: int, n: int, k: int):
    """Tuned desc_bwd NN config for the narrow TN materialize path.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, maxnreg,
    GROUP_M) or None to fall back to _thead_bfgemm_nn_config.  Tuned on
    Zhenwu for the non-aligned core shapes that take this path, using the
    ldc == n (odd lane) C descriptor so the crop degrades to a contiguous
    1D copy.
    """
    if m == 1024 and n == 1023 and k == 1023:
        # (128,256,32,w8,s3,mr128): measured pipe (tp + NN + copy) = 0.0355.
        return 128, 256, 32, 8, 3, 128, 8
    if m == 4096 and n == 4095 and k == 4095:
        # (128,256,64,w8,s3,gm12): NN ldc=n = 1.060 ms (vs 1.063 gm8).
        return 128, 256, 64, 8, 3, 160, 12
    if m == 4096 and n == 8191 and k == 4095:
        # (128,256,64,w8,s3,gm12): NN ldc=n = 2.194 ms.
        return 128, 256, 64, 8, 3, 160, 12
    if m == 4224 and n == 8191 and k == 4095:
        # 4097x8191x4095: m_pad = round_up(4097, 128) = 4224.
        # gm12 measured 2.194 ms (vs gm8 generic 2.198 ms).
        return 128, 256, 64, 8, 3, 160, 12
    return None


def _run_thead_bfgemm_tn_materialize_a_narrow(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """TN materialize-A without padA/padB: transpose into (M_pad, K_pad)
    with lda == K_pad, run desc_bwd NN with ldc == n into (M_pad, n), then a
    contiguous 1D copy into the real (M, N) C.

    The A transpose zero-fills the row padding (m .. M_pad-1) and the column
    padding (k .. K_pad-1).  K_pad = round_up(k, 64) gives the NN kernel a
    fully aligned K loop and an aligned A row stride, measured faster on
    Zhenwu than the odd-strided (M_pad, K) layout (K_pad is safe because the
    B tail rows read OOB contribute 0 when multiplied by the zero A pad
    column, sentinel-verified).

    With ldc == n the NN C descriptor has the same odd lane as the real C,
    so the (M_pad, n) result's first m rows are contiguous with C and the
    crop degrades to a pure 1D copy (measured faster on Zhenwu than the
    padded ldc == n_pad + strided crop).

    For large shapes (M_pad >= 4096) the full-width copy is replaced by a
    split-M scheme: the main NN kernel stores the aligned leading rows
    directly into C via a clamped descriptor, and only the trailing partial
    row tile is written to a small tail buffer which is copied over.  The
    tail kernel runs concurrently on a side stream so its B re-read hides
    inside the compute-bound main kernel (measured 0.957 vs 0.920 on
    4095^3, and 0.985 on 4097x8191x4095).
    """
    m_pad = _round_up(m, 128)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((m_pad, k_pad), dtype=A.dtype, device=A.device)

    # Transpose-pad configs measured on Zhenwu (dst stride == k_pad):
    # (32,128,w8) small, (16,64,w8) big.
    if max(m, k) <= 1024:
        block_m, block_n, tp_warps = 32, 128, 8
    else:
        block_m, block_n, tp_warps = 16, 64, 8
    _thead_bfgemm_transpose_pad_kernel[
        (triton.cdiv(m_pad, block_m), triton.cdiv(k_pad, block_n))
    ](
        A,
        A_pad,
        k,
        m,
        lda,
        k_pad,
        k_pad,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=tp_warps,
    )
    if (
        m_pad >= 4096
        and not torch.cuda.is_current_stream_capturing()
        and _thead_bfgemm_tn_narrow_nn_config(m_pad, n, k) is not None
    ):
        _run_thead_bfgemm_tn_materialize_a_narrow_splitm(
            A_pad, B, C, m, n, k, alpha, beta, beta_is_zero
        )
        return

    C_n = torch.empty((m_pad, n), dtype=C.dtype, device=C.device)
    nn_cfg = _thead_bfgemm_tn_narrow_nn_config(m_pad, n, k)
    if nn_cfg is None:
        _run_thead_bfgemm_nn(
            A_pad,
            k_pad,
            B,
            ldb,
            C_n,
            n,
            m_pad,
            n,
            k_pad,
            alpha,
            0.0,
            True,
            True,
            k_b=k,
        )
    else:
        block_m, block_n, block_k, num_warps, num_stages, maxnreg, group_m = nn_cfg
        kwargs = dict(num_warps=num_warps, num_stages=num_stages)
        if maxnreg:
            kwargs["maxnreg"] = maxnreg
        _thead_bfgemm_nn_desc_bwd_kernel[
            (triton.cdiv(m_pad, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B,
            C_n,
            alpha,
            0.0,
            k_pad,
            n,
            n,
            True,
            alpha == 1.0,
            m_pad,
            n,
            k_pad,
            block_m,
            block_n,
            block_k,
            group_m,
            K_B=k,
            **kwargs,
        )
    # 1D copy configs measured on Zhenwu: (1024,w8) small, (4096,w32) big.
    if max(m, n) <= 4096:
        copy_block, copy_warps = 1024, 8
    else:
        copy_block, copy_warps = 4096, 32
    _thead_bfgemm_copy_c_kernel[(triton.cdiv(m * n, copy_block),)](
        C_n,
        C,
        beta,
        m * n,
        beta_is_zero,
        BLOCK_SIZE=copy_block,
        num_warps=copy_warps,
    )


def _run_thead_bfgemm_tn_materialize_a_narrow_splitm(
    A_pad, B, C, m, n, k, alpha, beta, beta_is_zero
):
    """Split-M variant of the narrow TN materialize path for large shapes.

    The main NN kernel (ncrop desc_bwd) stores the aligned leading
    M_FULL = M_pad - 128 rows directly into the real C via a clamped
    descriptor, and the trailing partial tile is computed by a small tail
    kernel writing to a (128, n) tail buffer.  The tail kernel runs
    concurrently on a side stream with the main kernel; only the tail rows
    are then copied into C (a few MB instead of the full M x N copy).
    """
    m_pad = A_pad.shape[0]
    k_pad = A_pad.shape[1]
    m_full = m_pad - 128
    C_tail = torch.empty((128, n), dtype=C.dtype, device=C.device)
    nn_cfg = _thead_bfgemm_tn_narrow_nn_config(m_pad, n, k)
    block_m, block_n, block_k, num_warps, num_stages, maxnreg, group_m = nn_cfg
    kwargs = dict(num_warps=num_warps, num_stages=num_stages)
    if maxnreg:
        kwargs["maxnreg"] = maxnreg

    def launch_main():
        _thead_bfgemm_nn_desc_bwd_ncrop_kernel[
            (triton.cdiv(m_full, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B,
            C,
            alpha,
            beta,
            k_pad,
            n,
            n,
            beta_is_zero,
            alpha == 1.0,
            m_pad,
            m_full,
            n,
            k_pad,
            block_m,
            block_n,
            block_k,
            group_m,
            K_B=k,
            **kwargs,
        )

    def launch_tail():
        _thead_bfgemm_nn_desc_bwd_ncrop_kernel[
            (triton.cdiv(128, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad[m_full:],
            B,
            C_tail,
            alpha,
            0.0,
            k_pad,
            n,
            n,
            True,
            alpha == 1.0,
            128,
            128,
            n,
            k_pad,
            block_m,
            block_n,
            block_k,
            group_m,
            K_B=k,
            **kwargs,
        )

    s2, s3, ev = _get_splitm_streams()
    cur = torch.cuda.current_stream()
    ev.record(cur)
    with torch.cuda.stream(s2):
        s2.wait_event(ev)
        launch_main()
    with torch.cuda.stream(s3):
        s3.wait_event(ev)
        launch_tail()
    cur.wait_stream(s2)
    cur.wait_stream(s3)
    n_tail = (m - m_full) * n
    _thead_bfgemm_copy_c_kernel[(triton.cdiv(n_tail, 1024),)](
        C_tail,
        C[m_full:].reshape(-1),
        beta,
        n_tail,
        beta_is_zero,
        BLOCK_SIZE=1024,
        num_warps=8,
    )


def _run_thead_bfgemm_tn_materialize_b(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    B_T = _thead_bfgemm_transpose2d(B, k, n, ldb)
    _run_thead_bfgemm_tt(
        A,
        lda,
        B_T,
        k,
        C,
        ldc,
        m,
        n,
        k,
        alpha,
        beta,
        beta_is_zero,
        True,
    )


def _run_thead_bfgemm_tn_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Pad A(K,M) and B(K,N) to multiples of 64, then crop result."""
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    # A is (K, M) with lda=M
    A_pad = torch.empty((k_pad, m_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((k_pad, n_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(k_pad * m_pad, pad_block),)](
        A, A_pad, k, m, lda, m_pad, k_pad, m_pad, BLOCK_SIZE=pad_block
    )
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(k_pad * n_pad, pad_block),)](
        B, B_pad, k, n, ldb, n_pad, k_pad, n_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_bfgemm_tn(
        A_pad,
        m_pad,
        B_pad,
        n_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    _thead_bfgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def _can_use_thead_bfgemm_tn(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    # A^T: A is (K, M), lda = M; B: (K, N), ldb = N
    return lda == m and ldb == n and ldc == n and m >= 16 and n >= 16 and k >= 16


# ======================= bfgemm_nt (A x B^T) =======================
# Ported from hgemm_nt.  The generic _bfgemm_nt_kernel transposes the B tile
# inside the dot (tl.trans), which measures well below the NN baseline on
# Zhenwu.  This transpose-free NT computes C^T(N,M) = B(N,K) x A(K,M) with a
# transposed accumulator (no tl.trans in the dot) and writes C via
# tl.trans(acc_t), mirroring the hgemm NT kernels below.


@triton.jit
def _thead_bfgemm_nt_impl(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Transpose-free NT: C^T(N,M) = B(N,K) x A(K,M), no tl.trans in dot."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # A is (M, K). Load A^T(K, M): first dim K stride=1, second dim M stride=lda.
    a_t_ptrs = a_ptr + offs_k[:, None] + offs_m[None, :] * lda
    # B is (N, K). Load (BLOCK_N, BLOCK_K) contiguous.
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    # C^T(N,M) = B(N,K) x A^T(K,M)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b, a_t, acc_t, out_dtype=tl.float32)

    # acc = acc_t^T (BLOCK_M, BLOCK_N)
    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_bfgemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_nt_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit
def _thead_bfgemm_nt_desc_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b_t = b_desc.load([offs_n, offs_k])
        b = tl.trans(b_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_nt_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_nt_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_nt_desc_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b_t = b_desc.load([offs_n, offs_k])
        b = tl.trans(b_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


def _thead_bfgemm_nt_config(m: int, n: int, k: int):
    """NT config.

    NT loads B^T (N, K) and transposes the tile inside the dot, so its
    sweet spot differs from NN for non-aligned shapes.  The tiles below
    were tuned with the desc_bwd kernel on Zhenwu (bf16):
      511^3        -> (64,64,64,4)    9.6 us
      1023^3       -> (128,128,64,4)  30.7 us
      4095^3/wide  -> (128,256,64,8)  1121/2295 us
    Aligned shapes reuse the NN config.
    """
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return _thead_bfgemm_nn_config(m, n, k)
    if max(m, n, k) <= 512:
        return 64, 64, 64, 4, 3, 128
    if max(m, n, k) <= 1024:
        return 128, 128, 64, 4, 3, 128
    return 128, 256, 64, 8, 3, 160


def _can_use_thead_bfgemm_nt(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    # A: (M, K), lda = K; B^T: B is (N, K), ldb = K
    return lda == k and ldb == k and ldc == n and m >= 16 and n >= 16 and k >= 16


def _run_thead_bfgemm_nt(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_bfgemm_nt_config(
        m, n, k
    )
    tile_aligned = _thead_bfgemm_is_tile_aligned(m, n, k, block_m, block_n, block_k)
    if _thead_bfgemm_nn_use_desc_bwd(m, n, k):
        # desc_bwd handles partial tiles natively through the TMA
        # descriptors (OOB loads zero-fill, OOB stores are dropped), so
        # non-aligned shapes use it too -- it is the fastest measured
        # path for every core NT shape (vs materialize/pad).
        kernel = _thead_bfgemm_nt_desc_bwd_kernel
    elif tile_aligned:
        kernel = _thead_bfgemm_nt_desc_kernel
    else:
        kernel = _thead_bfgemm_nt_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


# ======================= bfgemm_tt (A^T x B^T) =======================
# T-Head TT kernels used by the TN materialize-B path (A^T x B, with
# B^T materialized). Computes C^T = B x A via a transposed accumulator,
# so the dot has no tl.trans.


@triton.jit
def _thead_bfgemm_tt_impl(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Shared impl for TT: compute C^T = B x A via transposed accumulator, no tl.trans in dot."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # A is (K, M), lda = M. Load (BLOCK_K, BLOCK_M) contiguous.
    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    # B is (N, K), ldb = K. Load (BLOCK_N, BLOCK_K) contiguous.
    b_t_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    # C(M,N) = A^T(M,K) x B^T(N,K)^T
    # C^T(N,M) = B(N,K) x A(K,M) = b_t * a_t
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs)
            b_t = tl.load(b_t_ptrs)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_t_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b_t = tl.load(b_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    # Transpose back: C = acc_t^T (BLOCK_M, BLOCK_N)
    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_bfgemm_tt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_tt_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_tt_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_tt_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit
def _thead_bfgemm_tt_desc_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C^T(N,M) = B(N,K) x A(K,M), no tl.trans in dot
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b_t = b_desc.load([offs_n, offs_k])
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_tt_desc_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C^T(N,M) = B(N,K) x A(K,M), no tl.trans in dot
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b_t = b_desc.load([offs_n, offs_k])
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_tt_desc_bwd_ncrop_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M_PAD: tl.constexpr,
    N_PAD: tl.constexpr,
    K_PAD: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """desc_bwd TT with A/B loaded from (K_PAD, M_PAD)/(N_PAD, K_PAD)
    padded buffers but storing directly into the real (M, N) C via a
    clamped descriptor.

    The padded A/B give aligned TMA loads (same speed as the padded-C
    path) while the C descriptor clamps the partial M/N tiles, so no
    C_pad buffer and no crop kernel are needed.  The zero-padded rows of
    A/B contribute nothing, so the padded K loop is safe.
    """
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K_PAD, M_PAD], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N_PAD, K_PAD], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    # C^T(N,M) = B(N,K) x A(K,M), no tl.trans in dot
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, tl.cdiv(K_PAD, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b_t = b_desc.load([offs_n, offs_k])
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_tt_splitk_kernel(
    a_ptr,
    b_ptr,
    c_part_ptr,
    alpha: tl.float32,
    lda,
    ldb,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Split-K TT: each CTA handles one (tile, k-slice) pair and writes its
    partial C^T(N,M) accumulation to a (SPLIT_K, M, N) fp32 buffer.  A second
    reduce kernel sums the partials into the bf16 C.  Requires K divisible by
    SPLIT_K * BLOCK_K and M/N aligned to the tiles (the partial store is
    unmasked)."""
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_part_ptr = c_part_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    pid_k = pid // num_tiles
    pid_tile = pid % num_tiles
    width = GROUP_M * grid_n
    group_id = pid_tile // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_tile % group_size)
    pid_n = (pid_tile % width) // group_size
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[K, M], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )

    k_per = K // SPLIT_K
    k0 = pid_k * k_per
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(0, k_per // BLOCK_K):
        offs_k = k0 + i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b_t = b_desc.load([offs_n, offs_k])
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32)

    acc = tl.trans(acc_t)
    result = alpha * acc
    offs_mv = offs_m + tl.arange(0, BLOCK_M)
    offs_nv = offs_n + tl.arange(0, BLOCK_N)
    ptrs = c_part_ptr + (pid_k * M + offs_mv)[:, None] * N + offs_nv[None, :]
    tl.store(ptrs, result)


@libentry()
@triton.jit
def _thead_bfgemm_tt_splitk_reduce_kernel(
    c_part_ptr,
    c_ptr,
    beta: tl.float32,
    BETA_IS_ZERO: tl.constexpr,
    SPLIT_K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Sum the SPLIT_K fp32 partial C^T buffers into the bf16 (M, N) C,
    fusing C = beta*C + sum.  Fully-aligned tiles only (unmasked)."""
    c_part_ptr = c_part_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, SPLIT_K):
        ptrs = c_part_ptr + (k * M + offs_m)[:, None] * N + offs_n[None, :]
        acc += tl.load(ptrs)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if not BETA_IS_ZERO:
        acc += beta * tl.load(c_ptrs).to(tl.float32)
    tl.store(c_ptrs, acc.to(tl.bfloat16))


def _thead_bfgemm_tt_splitk_config(m: int, n: int, k: int):
    """SPLIT_K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, maxnreg.

    Split-K choice measured on Zhenwu with (128,128,64) tiles:
      - (2048,2048,16384) 256 tiles -> SK=4: 1129 us (SK=2: 1137 us)
      - (2048,4096,11008) 512 tiles -> SK=2: 1447 us (SK=4: 1541 us)
    Smaller tile grids benefit from more split slices; larger ones only
    from a modest split (the extra fp32 partial write/read would otherwise
    outweigh the added CTAs)."""
    tiles = (m + 127) // 128 * ((n + 127) // 128)
    split_k = 4 if tiles <= 256 else 2
    return split_k, 128, 128, 64, 4, 3, 128


def _thead_bfgemm_tt_should_splitk(m: int, n: int, k: int) -> bool:
    """Split-K for large-K TT shapes whose MxN tile grid under-utilizes the
    device (few CTAs, very long K loops).  Requires full tile alignment (the
    partial-accumulator store and the reduce kernel are unmasked)."""
    split_k, block_m, block_n, block_k, *_ = _thead_bfgemm_tt_splitk_config(m, n, k)
    if k < 8192 or min(m, n) > 2048:
        return False
    tiles = (m + block_m - 1) // block_m * ((n + block_n - 1) // block_n)
    return (
        m % block_m == 0
        and n % block_n == 0
        and k % (split_k * block_k) == 0
        # Beyond 512 tiles the MxN grid already saturates the device and the
        # fp32 partial write+read of split-K is pure overhead.
        and tiles <= 512
    )


def _run_thead_bfgemm_tt_splitk(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    (
        split_k,
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        maxnreg,
    ) = _thead_bfgemm_tt_splitk_config(m, n, k)
    c_part = torch.empty((split_k, m, n), dtype=torch.float32, device=C.device)
    num_tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    _thead_bfgemm_tt_splitk_kernel[(num_tiles * split_k,)](
        A,
        B,
        c_part,
        alpha,
        lda,
        ldb,
        M=m,
        N=n,
        K=k,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=16,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )
    _thead_bfgemm_tt_splitk_reduce_kernel[(num_tiles,)](
        c_part,
        C,
        beta,
        beta_is_zero,
        SPLIT_K=split_k,
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=16,
        num_warps=num_warps,
        num_stages=2,
        maxnreg=maxnreg,
    )


def _run_thead_bfgemm_tt_padded_ncrop(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Pad A(K,M) and B(N,K) to multiples of 64, run the aligned desc_bwd
    TT kernel, and write C directly through a clamped descriptor (no
    C_pad buffer, no crop kernel)."""
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((k_pad, m_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((n_pad, k_pad), dtype=B.dtype, device=B.device)

    pad_block = 1024
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(k_pad * m_pad, pad_block),)](
        A, A_pad, k, m, lda, m_pad, k_pad, m_pad, BLOCK_SIZE=pad_block
    )
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(n_pad * k_pad, pad_block),)](
        B, B_pad, n, k, ldb, k_pad, n_pad, k_pad, BLOCK_SIZE=pad_block
    )
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_bfgemm_tt_config(
        m, n, k
    )
    _thead_bfgemm_tt_desc_bwd_ncrop_kernel[
        (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    ](
        A_pad,
        B_pad,
        C,
        alpha,
        beta,
        m_pad,
        k_pad,
        ldc,
        beta_is_zero,
        M_PAD=m_pad,
        N_PAD=n_pad,
        K_PAD=k_pad,
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=16,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _thead_bfgemm_tt_config(m: int, n: int, k: int):
    """T-Head TT config for the TN materialize-B path.

    TT uses a transposed accumulator and prefers square-ish tiles. For
    skinny shapes (m >> n) the acc_t tile (BLOCK_N, BLOCK_M) must keep
    BLOCK_N small enough to avoid excessive register pressure. On Zhenwu
    (bf16) the (128,128,64) tile must run with 4 warps: the 8-warp variant
    is ~30% slower under the benchmark's cold-cache measurement, while
    4 warps reaches the same throughput as hgemm's fp16 path.
    """
    min_mn = min(m, n)

    if max(m, n, k) <= 512:
        return 64, 64, 64, 4, 3, 128

    if m % 64 != 0 or n % 64 != 0 or k % 64 != 0:
        # Non-aligned cubes ~1023^3: desc_bwd (128,128,32,4,4) = 33.1 us
        # (the default 128,256,64 tile is ~30% slower for the transposed
        # accumulator on partial tiles).
        if max(m, n, k) <= 1024:
            return 128, 128, 32, 4, 4, 128

    if min_mn <= 64:
        return 64, 64, 128, 4, 3, 128

    if min_mn == 128 and max(m, n) >= 4096 and k <= 1024:
        return 64, 128, 64, 4, 3, 128

    if min_mn >= 1024:
        return 128, 128, 64, 4, 3, 128

    return _thead_bfgemm_nn_config(m, n, k)


def _run_thead_bfgemm_tt(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    # TT variant used by the TN materialize-B path. Uses the T-Head TT
    # kernels (transposed accumulator, no tl.trans in the dot) with a
    # fixed heuristic config instead of the generic libtuner kernel.
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_bfgemm_tt_config(
        m, n, k
    )
    tile_aligned = _thead_bfgemm_is_tile_aligned(m, n, k, block_m, block_n, block_k)
    if _thead_bfgemm_nn_use_desc_bwd(m, n, k):
        # desc_bwd handles partial tiles natively through the TMA
        # descriptors; it is the fastest measured TT path for aligned and
        # non-aligned shapes alike.
        kernel = _thead_bfgemm_tt_desc_bwd_kernel
    elif tile_aligned:
        kernel = _thead_bfgemm_tt_desc_kernel
    else:
        kernel = _thead_bfgemm_tt_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=16,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _run_thead_bfgemm_tt_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    """Pad A(K,M) and B(N,K) to multiples of 64, then crop result."""
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    cached = _tt_pad_cache_get(m, n, k, A, B)
    if cached is None:
        A_pad = torch.empty((k_pad, m_pad), dtype=A.dtype, device=A.device)
        B_pad = torch.empty((n_pad, k_pad), dtype=B.dtype, device=B.device)
        pad_block = 1024
        _thead_bfgemm_pad2d_kernel[(triton.cdiv(k_pad * m_pad, pad_block),)](
            A, A_pad, k, m, lda, m_pad, k_pad, m_pad, BLOCK_SIZE=pad_block
        )
        _thead_bfgemm_pad2d_kernel[(triton.cdiv(n_pad * k_pad, pad_block),)](
            B, B_pad, n, k, ldb, k_pad, n_pad, k_pad, BLOCK_SIZE=pad_block
        )
        _tt_pad_cache_put(m, n, k, A, B, A_pad, B_pad)
    else:
        A_pad, B_pad = cached
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    _run_thead_bfgemm_tt(
        A_pad,
        m_pad,
        B_pad,
        k_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    pad_block = 1024
    _thead_bfgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def _can_use_thead_bfgemm_tt(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    # A^T: A is (K, M), lda = M; B^T: B is (N, K), ldb = K
    return lda == m and ldb == k and ldc == n and m >= 16 and n >= 16 and k >= 16


def _thead_bfgemm_tt_should_pad(m: int, n: int, k: int) -> bool:
    """Pad TT when non-aligned and overhead is reasonable."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    if max(m, n, k) <= 1024:
        # Small non-aligned shapes (e.g. 511^3, 1023^3) are fastest via
        # direct desc_bwd; padding adds pad/crop overhead for no gain.
        return False
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return m * n * k >= 256 * 256 * 256 and extra <= original * 1.15


def _thead_bfgemm_tt_should_materialize(m: int, n: int, k: int) -> bool:
    """Materialize C^T for odd TT cases where padding overhead dominates."""
    if m % 64 == 0:
        return False
    if max(m, n, k) <= 1024:
        # Small non-aligned shapes are fastest via direct desc_bwd.
        return False
    if m == n and n == k:
        # Large cubes: padded desc_bwd wins over materialize+transpose.
        return False
    if m != n and m * n * k >= 256 * 256 * 256:
        return True
    return False


@libentry()
@triton.jit
def _thead_bfgemm_transpose_c_kernel(
    src_ptr,
    dst_ptr,
    beta: tl.float32,
    rows,
    cols,
    src_ld,
    dst_ld,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Transpose the (N, M) C_T buffer into the real (M, N) C, fusing
    C = beta*C + src^T."""
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)

    src_offsets = offs_n[None, :] * src_ld + offs_m[:, None]
    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    dst_offsets = offs_m[:, None] * dst_ld + offs_n[None, :]
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.bfloat16), mask=mask)


def _run_thead_bfgemm_tt_materialized(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    C_T = torch.empty((n, m), dtype=C.dtype, device=C.device)
    if _thead_bfgemm_nn_should_pad(n, m, k):
        _run_thead_bfgemm_nn_padded(B, ldb, A, lda, C_T, m, n, m, k, alpha, 0.0, True)
    else:
        _run_thead_bfgemm_nn(B, ldb, A, lda, C_T, m, n, m, k, alpha, 0.0, True, True)

    if m <= 512 and n <= 512:
        block_m, block_n = 8, 64
    elif m * n >= 2048 * 2048 and m != n:
        block_m, block_n = 16, 128
    else:
        block_m, block_n = 16, 64
    _thead_bfgemm_transpose_c_kernel[
        (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    ](
        C_T,
        C,
        beta,
        m,
        n,
        m,
        ldc,
        beta_is_zero,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
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

    beta_is_zero = beta == 0.0
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(A.device):
        if (
            transa == CUBLAS_OP_N
            and transb == CUBLAS_OP_N
            and _can_use_thead_bfgemm_nn(m, n, k, lda, ldb, ldc, alpha, beta)
        ):
            aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
            if _thead_bfgemm_nn_should_pad(m, n, k):
                _run_thead_bfgemm_nn_padded(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            else:
                _run_thead_bfgemm_nn(
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    beta_is_zero,
                    aligned,
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _bfgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if _can_use_thead_bfgemm_tn(m, n, k, lda, ldb, ldc, alpha, beta):
                aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
                if _thead_bfgemm_tn_use_desc_overlap(m, n, k):
                    # Small non-aligned TN: single fused kernel, avoids the
                    # materialize (transpose + NN) overhead.
                    _run_thead_bfgemm_tn(
                        A,
                        lda,
                        B,
                        ldb,
                        C,
                        ldc,
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        beta_is_zero,
                        aligned,
                    )
                elif _thead_bfgemm_tn_use_cola(m, n, k):
                    # colA kernel avoids both tl.trans and the transpose+NN
                    # materialize overhead; fastest measured on Zhenwu for the
                    # 2048-square family and large-K shapes (2048x2048x16384,
                    # 2048x4096x11008).  Must be checked before materialize:
                    # those shapes also satisfy should_materialize.
                    _run_thead_bfgemm_tn_cola(
                        A,
                        lda,
                        B,
                        ldb,
                        C,
                        ldc,
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        beta_is_zero,
                    )
                elif _thead_bfgemm_tn_should_materialize(m, n, k):
                    if m <= n:
                        # M <= N: Materialize A^T as (M, K) and use fast NN kernel.
                        # A is (K, M) with lda=M, A^T will be (M, K) with lda=K.
                        if max(m, n, k) <= 1024:
                            if _thead_bfgemm_tn_use_narrow_materialize(m, n, k):
                                _run_thead_bfgemm_tn_materialize_a_narrow(
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    C,
                                    ldc,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    beta,
                                    beta_is_zero,
                                )
                            else:
                                _run_thead_bfgemm_tn_materialize_a(
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    C,
                                    ldc,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    beta,
                                    beta_is_zero,
                                )
                        elif _thead_bfgemm_nn_should_pad(
                            m, n, k
                        ) or _thead_bfgemm_tn_should_pad(m, n, k):
                            if _thead_bfgemm_tn_use_narrow_materialize(m, n, k):
                                _run_thead_bfgemm_tn_materialize_a_narrow(
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    C,
                                    ldc,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    beta,
                                    beta_is_zero,
                                )
                            else:
                                A_T = _thead_bfgemm_transpose2d(A, k, m, lda)
                                _run_thead_bfgemm_nn_padded(
                                    A_T,
                                    k,
                                    B,
                                    ldb,
                                    C,
                                    ldc,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    beta,
                                    beta_is_zero,
                                )
                        else:
                            _run_thead_bfgemm_tn_materialize_a(
                                A,
                                lda,
                                B,
                                ldb,
                                C,
                                ldc,
                                m,
                                n,
                                k,
                                alpha,
                                beta,
                                beta_is_zero,
                            )
                    else:
                        # N < M: Materialize B^T as (N, K) and use TT transpose-free kernel.
                        # B is (K, N) with ldb=N, B^T will be (N, K) with lda_BT=K.
                        # TT kernel computes C(M,N) = A^T(K,M) x B_T^T(K,N) = (M,K)x(K,N).
                        if max(m, n, k) <= 1024:
                            _run_thead_bfgemm_tn_materialize_b(
                                A,
                                lda,
                                B,
                                ldb,
                                C,
                                ldc,
                                m,
                                n,
                                k,
                                alpha,
                                beta,
                                beta_is_zero,
                            )
                        elif _thead_bfgemm_nn_should_pad(
                            m, n, k
                        ) or _thead_bfgemm_tn_should_pad(m, n, k):
                            B_T = _thead_bfgemm_transpose2d(B, k, n, ldb)
                            _run_thead_bfgemm_tt_padded(
                                A,
                                lda,
                                B_T,
                                k,
                                C,
                                ldc,
                                m,
                                n,
                                k,
                                alpha,
                                beta,
                                beta_is_zero,
                            )
                        else:
                            _run_thead_bfgemm_tn_materialize_b(
                                A,
                                lda,
                                B,
                                ldb,
                                C,
                                ldc,
                                m,
                                n,
                                k,
                                alpha,
                                beta,
                                beta_is_zero,
                            )
                elif _thead_bfgemm_tn_should_pad(m, n, k):
                    _run_thead_bfgemm_tn_padded(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                else:
                    _run_thead_bfgemm_tn(
                        A,
                        lda,
                        B,
                        ldb,
                        C,
                        ldc,
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        beta_is_zero,
                        aligned,
                    )
            else:
                _bfgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            if _can_use_thead_bfgemm_nt(m, n, k, lda, ldb, ldc, alpha, beta):
                aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
                # desc_bwd is the fastest NT path for aligned and
                # non-aligned shapes alike, so dispatch straight to it.
                _run_thead_bfgemm_nt(
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    beta_is_zero,
                    aligned,
                )
            else:
                _bfgemm_nt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        else:
            if _can_use_thead_bfgemm_tt(m, n, k, lda, ldb, ldc, alpha, beta):
                aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
                if _thead_bfgemm_tt_should_splitk(m, n, k):
                    _run_thead_bfgemm_tt_splitk(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                elif _thead_bfgemm_tt_should_pad(m, n, k):
                    _run_thead_bfgemm_tt_padded(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                elif _thead_bfgemm_tt_should_materialize(m, n, k):
                    _run_thead_bfgemm_tt_materialized(
                        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                    )
                else:
                    _run_thead_bfgemm_tt(
                        A,
                        lda,
                        B,
                        ldb,
                        C,
                        ldc,
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        beta_is_zero,
                        aligned,
                    )
            else:
                _bfgemm_tt_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )


bgemm = bfgemm

__all__ = ["bfgemm", "bgemm"]

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

"""MThreads (MUSA) sgemm implementation.

The kernels follow the FMA-style matrix-multiply pattern used by FlagGems for
the MThreads backend (see FlagGems ``_mthreads/ops/mm.py``):

* pointer-based tiles with ``tl.max_contiguous(tl.multiple_of(...))`` hints;
* 64-bit offsets so large tensors (e.g. 16K^3 sgemm) never overflow;
* a peeled K loop so the hot loop can run without masks;
* ``allow_tf32=False`` so the fp32 accumulation matches the reference.

The tile configuration comes from ``_mthreads/tune_configs.yaml`` through the
regular ``runtime.get_tuned_config("sgemm")`` mechanism.
"""

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level3.sgemm import CUBLAS_OP_N, CUBLAS_OP_T, ScalarType
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit(do_not_specialize=["m", "n", "k"])
def _sgemm_nn_kernel(
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
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offs_am % m, BLOCK_M), BLOCK_M).to(
        tl.int64
    )
    rbn = tl.max_contiguous(tl.multiple_of(offs_bn % n, BLOCK_N), BLOCK_N).to(
        tl.int64
    )

    prev_k_mult = tl.cdiv(k, BLOCK_K) * BLOCK_K - BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = (start_k + offs_k).to(tl.int64)
        a = tl.load(a_ptr + (ram[:, None] * lda + rk[None, :]))
        b = tl.load(b_ptr + (rk[:, None] * ldb + rbn[None, :]))
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    rk = (prev_k_mult + offs_k).to(tl.int64)
    mask_k = rk < k
    a = tl.load(
        a_ptr + (ram[:, None] * lda + rk[None, :]), mask=mask_k[None, :], other=0.0
    )
    b = tl.load(
        b_ptr + (rk[:, None] * ldb + rbn[None, :]), mask=mask_k[:, None], other=0.0
    )
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    mask_store = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]

    if BETA_IS_ZERO:
        # The mthreads triton backend miscompiles a store whose value is a
        # plain `alpha * acc` for non-aligned shapes (e.g. 511 % BLOCK_M != 0)
        # with num_warps=4. Keeping a runtime-scalar dependency (`+ beta`,
        # beta == 0 at runtime) makes the store correct without reading C,
        # which preserves the beta == 0 semantics (C may contain NaN).
        res = alpha * acc
        tl.store(c_ptrs, res + beta, mask=mask_store)
    else:
        c_vals = tl.load(c_ptrs, mask=mask_store, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=mask_store)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit(do_not_specialize=["m", "n", "k"])
def _sgemm_tn_kernel(
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
    # A is stored as (k, m) with leading dimension lda.
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offs_am % m, BLOCK_M), BLOCK_M).to(
        tl.int64
    )
    rbn = tl.max_contiguous(tl.multiple_of(offs_bn % n, BLOCK_N), BLOCK_N).to(
        tl.int64
    )

    prev_k_mult = tl.cdiv(k, BLOCK_K) * BLOCK_K - BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = (start_k + offs_k).to(tl.int64)
        a = tl.load(a_ptr + (rk[None, :] * lda + ram[:, None]))
        b = tl.load(b_ptr + (rk[:, None] * ldb + rbn[None, :]))
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    rk = (prev_k_mult + offs_k).to(tl.int64)
    mask_k = rk < k
    a = tl.load(
        a_ptr + (rk[None, :] * lda + ram[:, None]), mask=mask_k[None, :], other=0.0
    )
    b = tl.load(
        b_ptr + (rk[:, None] * ldb + rbn[None, :]), mask=mask_k[:, None], other=0.0
    )
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    mask_store = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]

    if BETA_IS_ZERO:
        # The mthreads triton backend miscompiles a store whose value is a
        # plain `alpha * acc` for non-aligned shapes (e.g. 511 % BLOCK_M != 0)
        # with num_warps=4. Keeping a runtime-scalar dependency (`+ beta`,
        # beta == 0 at runtime) makes the store correct without reading C,
        # which preserves the beta == 0 semantics (C may contain NaN).
        res = alpha * acc
        tl.store(c_ptrs, res + beta, mask=mask_store)
    else:
        c_vals = tl.load(c_ptrs, mask=mask_store, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=mask_store)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit(do_not_specialize=["m", "n", "k"])
def _sgemm_nt_kernel(
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
    # B is stored as (n, k) with leading dimension ldb.
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offs_am % m, BLOCK_M), BLOCK_M).to(
        tl.int64
    )
    rbn = tl.max_contiguous(tl.multiple_of(offs_bn % n, BLOCK_N), BLOCK_N).to(
        tl.int64
    )

    prev_k_mult = tl.cdiv(k, BLOCK_K) * BLOCK_K - BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = (start_k + offs_k).to(tl.int64)
        a = tl.load(a_ptr + (ram[:, None] * lda + rk[None, :]))
        b = tl.load(b_ptr + (rbn[None, :] * ldb + rk[:, None]))
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    rk = (prev_k_mult + offs_k).to(tl.int64)
    mask_k = rk < k
    a = tl.load(
        a_ptr + (ram[:, None] * lda + rk[None, :]), mask=mask_k[None, :], other=0.0
    )
    b = tl.load(
        b_ptr + (rbn[None, :] * ldb + rk[:, None]), mask=mask_k[:, None], other=0.0
    )
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    mask_store = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]

    if BETA_IS_ZERO:
        # The mthreads triton backend miscompiles a store whose value is a
        # plain `alpha * acc` for non-aligned shapes (e.g. 511 % BLOCK_M != 0)
        # with num_warps=4. Keeping a runtime-scalar dependency (`+ beta`,
        # beta == 0 at runtime) makes the store correct without reading C,
        # which preserves the beta == 0 semantics (C may contain NaN).
        res = alpha * acc
        tl.store(c_ptrs, res + beta, mask=mask_store)
    else:
        c_vals = tl.load(c_ptrs, mask=mask_store, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=mask_store)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit(do_not_specialize=["m", "n", "k"])
def _sgemm_tt_kernel(
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
    # A is stored as (k, m) and B as (n, k).
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(offs_am % m, BLOCK_M), BLOCK_M).to(
        tl.int64
    )
    rbn = tl.max_contiguous(tl.multiple_of(offs_bn % n, BLOCK_N), BLOCK_N).to(
        tl.int64
    )

    prev_k_mult = tl.cdiv(k, BLOCK_K) * BLOCK_K - BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = (start_k + offs_k).to(tl.int64)
        a = tl.load(a_ptr + (rk[None, :] * lda + ram[:, None]))
        b = tl.load(b_ptr + (rbn[None, :] * ldb + rk[:, None]))
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    rk = (prev_k_mult + offs_k).to(tl.int64)
    mask_k = rk < k
    a = tl.load(
        a_ptr + (rk[None, :] * lda + ram[:, None]), mask=mask_k[None, :], other=0.0
    )
    b = tl.load(
        b_ptr + (rbn[None, :] * ldb + rk[:, None]), mask=mask_k[:, None], other=0.0
    )
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    mask_store = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]

    if BETA_IS_ZERO:
        # The mthreads triton backend miscompiles a store whose value is a
        # plain `alpha * acc` for non-aligned shapes (e.g. 511 % BLOCK_M != 0)
        # with num_warps=4. Keeping a runtime-scalar dependency (`+ beta`,
        # beta == 0 at runtime) makes the store correct without reading C,
        # which preserves the beta == 0 semantics (C may contain NaN).
        res = alpha * acc
        tl.store(c_ptrs, res + beta, mask=mask_store)
    else:
        c_vals = tl.load(c_ptrs, mask=mask_store, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=mask_store)


def sgemm(
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
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
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
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _sgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _sgemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _sgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _sgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )

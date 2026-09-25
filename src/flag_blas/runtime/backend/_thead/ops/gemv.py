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
import struct
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils import triton_lang_extension as tle

ScalarType = Union[float, int, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2
_NARROW_T_MIN_K = 65536
_NARROW_T_MAX_OUTPUT = 4
_NARROW_T_SPLITS = 32
_LARGE_LOWP_T_MIN_K = 8192
_LARGE_LOWP_T_MIN_OUTPUT = 3584
_NARROW_LOWP_T_MIN_K = 65536
_NARROW_LOWP_T_MAX_OUTPUT = 2
_THEAD_SM_COUNT = 64
_K1_T_MIN_OUTPUT = 65536

_common = importlib.import_module("flag_blas.ops.level2.gemv")
_ZGEMV_NARROW_N_SPLITS = 32
_ZGEMV_NARROW_N_KERNEL = _common.zgemv_n_splitk_kernel.fn.fn
_ZGEMV_N_KERNEL = _common.zgemv_n_kernel.fn.fn
_ZGEMV_TALL_N_KERNEL = _common.zgemv_n_small_kernel.fn.fn
_CGEMV_N_KERNEL = _common.cgemv_n_kernel.fn.fn
_CGEMV_TC_KERNEL = _common.cgemv_kernel.fn.fn


@triton.jit
def cgemv_n_unrolled_thead_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    BETA_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    k0 = tl.arange(0, BLOCK)
    a_packed = A.to(tl.pointer_type(tl.int64))
    x_packed = X.to(tl.pointer_type(tl.int64))
    acc_real = tl.full((BLOCK,), 0, tl.float32)
    acc_imag = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(0, (N + BLOCK - 1) // BLOCK):
        k = j * BLOCK + k0
        a = tl.load(a_packed + row * N + k, mask=k < N, other=0)
        x = tl.load(x_packed + k, mask=k < N, other=0)
        a_real = a.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        x_real = x.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        acc_real += a_real * x_real - a_imag * x_imag
        acc_imag += a_real * x_imag + a_imag * x_real
    sum_real = tl.sum(acc_real, 0)
    sum_imag = tl.sum(acc_imag, 0)
    out_real = AR * sum_real - AI * sum_imag
    out_imag = AR * sum_imag + AI * sum_real
    if not BETA_ZERO:
        old_real = tl.load(Y + 2 * row)
        old_imag = tl.load(Y + 2 * row + 1)
        out_real += BR * old_real - BI * old_imag
        out_imag += BR * old_imag + BI * old_real
    tl.store(Y + 2 * row, out_real)
    tl.store(Y + 2 * row + 1, out_imag)


@triton.jit
def cgemv_n_square_thead_kernel(
    A,
    X,
    Y,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    BETA_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
    STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    k0 = tl.arange(0, BLOCK)
    a_packed = A.to(tl.pointer_type(tl.int64))
    x_packed = X.to(tl.pointer_type(tl.int64))
    acc_real = tl.full((BLOCK,), 0, tl.float32)
    acc_imag = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.range(0, (N + BLOCK - 1) // BLOCK, num_stages=STAGES):
        k = j * BLOCK + k0
        a = tl.load(a_packed + row * N + k, mask=k < N, other=0)
        x = tl.load(x_packed + k, mask=k < N, other=0)
        a_real = a.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        x_real = x.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        acc_real += a_real * x_real - a_imag * x_imag
        acc_imag += a_real * x_imag + a_imag * x_real
    sum_real = tl.sum(acc_real, 0)
    sum_imag = tl.sum(acc_imag, 0)
    out_real = AR * sum_real - AI * sum_imag
    out_imag = AR * sum_imag + AI * sum_real
    if not BETA_ZERO:
        old_real = tl.load(Y + 2 * row)
        old_imag = tl.load(Y + 2 * row + 1)
        out_real += BR * old_real - BI * old_imag
        out_imag += BR * old_imag + BI * old_real
    tl.store(Y + 2 * row, out_real)
    tl.store(Y + 2 * row + 1, out_imag)


@triton.jit
def cgemv_n_fma_thead_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    BETA_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k0 = tl.arange(0, BLOCK_K)
    a_packed = A.to(tl.pointer_type(tl.int64))
    x_packed = X.to(tl.pointer_type(tl.int64))
    acc_real = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    acc_imag = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for start in range(0, N, BLOCK_K):
        k = start + k0
        matrix_index = rows[:, None] * N + k[None, :]
        matrix_mask = (rows[:, None] < M) & (k[None, :] < N)
        a = tl.load(
            a_packed + matrix_index,
            matrix_mask,
            other=0,
            eviction_policy="evict_first",
        )
        x = tl.load(
            x_packed + k,
            k < N,
            other=0,
            eviction_policy="evict_last",
        )
        a_real = a.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        x_real = x.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        acc_real = tl.fma(a_real, x_real[None, :], acc_real)
        acc_real = tl.fma(-a_imag, x_imag[None, :], acc_real)
        acc_imag = tl.fma(a_real, x_imag[None, :], acc_imag)
        acc_imag = tl.fma(a_imag, x_real[None, :], acc_imag)
    sum_real = tl.sum(acc_real, 1)
    sum_imag = tl.sum(acc_imag, 1)
    out_real = AR * sum_real - AI * sum_imag
    out_imag = AR * sum_imag + AI * sum_real
    if not BETA_ZERO:
        old_real = tl.load(Y + 2 * rows, rows < M, other=0)
        old_imag = tl.load(Y + 2 * rows + 1, rows < M, other=0)
        out_real += BR * old_real - BI * old_imag
        out_imag += BR * old_imag + BI * old_real
    tl.store(Y + 2 * rows, out_real, rows < M)
    tl.store(Y + 2 * rows + 1, out_imag, rows < M)


@triton.jit
def cgemv_tc_wide_thead_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    CONJ: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    # Put adjacent output columns on adjacent lanes for the T/C wide matrix.
    out = tl.program_id(0) * 32 + tl.arange(0, 32)
    k0 = tl.arange(0, 128)
    a_packed = A.to(tl.pointer_type(tl.int64))
    x_packed = X.to(tl.pointer_type(tl.int64))
    acc_real = tl.zeros((128, 32), tl.float32)
    acc_imag = tl.zeros((128, 32), tl.float32)
    for start in range(0, M, 128):
        k = start + k0
        matrix_index = k[:, None] * N + out[None, :]
        matrix_mask = (k[:, None] < M) & (out[None, :] < N)
        a = tl.load(
            a_packed + matrix_index,
            matrix_mask,
            other=0,
            eviction_policy="evict_first",
        )
        x = tl.load(
            x_packed + k,
            k < M,
            other=0,
            eviction_policy="evict_last",
        )
        a_real = a.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        x_real = x.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        acc_real = tl.fma(a_real, x_real[:, None], acc_real)
        if CONJ:
            acc_real = tl.fma(a_imag, x_imag[:, None], acc_real)
            acc_imag = tl.fma(a_real, x_imag[:, None], acc_imag)
            acc_imag = tl.fma(-a_imag, x_real[:, None], acc_imag)
        else:
            acc_real = tl.fma(-a_imag, x_imag[:, None], acc_real)
            acc_imag = tl.fma(a_real, x_imag[:, None], acc_imag)
            acc_imag = tl.fma(a_imag, x_real[:, None], acc_imag)
    sum_real = tl.sum(acc_real, 0)
    sum_imag = tl.sum(acc_imag, 0)
    result_real = AR * sum_real - AI * sum_imag
    result_imag = AR * sum_imag + AI * sum_real
    if not BETA_ZERO:
        old_real = tl.load(Y + 2 * out, out < N, other=0)
        old_imag = tl.load(Y + 2 * out + 1, out < N, other=0)
        result_real += BR * old_real - BI * old_imag
        result_imag += BR * old_imag + BI * old_real
    tl.store(Y + 2 * out, result_real, out < N)
    tl.store(Y + 2 * out + 1, result_imag, out < N)


@triton.jit
def cgemv_tc_tall_scale_thead_kernel(
    Y,
    N: tl.constexpr,
    BR: tl.float32,
    BI: tl.float32,
    BETA_ZERO: tl.constexpr,
):
    out = tl.arange(0, 128)
    if BETA_ZERO:
        tl.store(Y + 2 * out, 0.0, out < N)
        tl.store(Y + 2 * out + 1, 0.0, out < N)
    else:
        old_real = tl.load(Y + 2 * out, out < N, other=0)
        old_imag = tl.load(Y + 2 * out + 1, out < N, other=0)
        tl.store(Y + 2 * out, BR * old_real - BI * old_imag, out < N)
        tl.store(Y + 2 * out + 1, BR * old_imag + BI * old_real, out < N)


@triton.jit
def cgemv_tc_tall_split_thead_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    AR: tl.float32,
    AI: tl.float32,
    CONJ: tl.constexpr,
):
    # Each split reads adjacent output columns and accumulates into y directly.
    out = tl.program_id(0) * 32 + tl.arange(0, 32)
    chunk = ((M + 80 * 16 - 1) // (80 * 16)) * 16
    k0 = tl.arange(0, 16)
    a_packed = A.to(tl.pointer_type(tl.int64))
    x_packed = X.to(tl.pointer_type(tl.int64))
    acc_real = tl.zeros((16, 32), tl.float32)
    acc_imag = tl.zeros((16, 32), tl.float32)
    for step in range(0, chunk, 16):
        k = tl.program_id(1) * chunk + step + k0
        matrix_index = k[:, None] * N + out[None, :]
        matrix_mask = (k[:, None] < M) & (out[None, :] < N)
        a = tl.load(
            a_packed + matrix_index,
            matrix_mask,
            other=0,
            eviction_policy="evict_first",
        )
        x = tl.load(
            x_packed + k,
            k < M,
            other=0,
            eviction_policy="evict_last",
        )
        a_real = a.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        x_real = x.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if CONJ:
            acc_real += a_real * x_real[:, None] + a_imag * x_imag[:, None]
            acc_imag += a_real * x_imag[:, None] - a_imag * x_real[:, None]
        else:
            acc_real += a_real * x_real[:, None] - a_imag * x_imag[:, None]
            acc_imag += a_real * x_imag[:, None] + a_imag * x_real[:, None]
    total_real = tl.sum(acc_real, 0)
    total_imag = tl.sum(acc_imag, 0)
    result_real = AR * total_real - AI * total_imag
    result_imag = AR * total_imag + AI * total_real
    tl.atomic_add(Y + 2 * out, result_real, out < N, sem="relaxed")
    tl.atomic_add(Y + 2 * out + 1, result_imag, out < N, sem="relaxed")


@triton.jit
def zgemv_n_unrolled_thead_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    AR: tl.int64,
    AI: tl.int64,
    BR: tl.int64,
    BI: tl.int64,
    BETA_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    acc_real = tl.full((BLOCK,), 0, tl.float64)
    acc_imag = tl.full((BLOCK,), 0, tl.float64)
    for j in tl.static_range(0, (N + BLOCK - 1) // BLOCK):
        k = j * BLOCK + offsets
        ar = tl.load(A + 2 * (row * N + k), k < N, other=0)
        ai = tl.load(A + 2 * (row * N + k) + 1, k < N, other=0)
        xr = tl.load(X + 2 * k, k < N, other=0)
        xi = tl.load(X + 2 * k + 1, k < N, other=0)
        acc_real += ar * xr - ai * xi
        acc_imag += ar * xi + ai * xr
    sum_real = tl.sum(acc_real, 0)
    sum_imag = tl.sum(acc_imag, 0)
    alpha_real = AR.to(tl.float64, bitcast=True)
    alpha_imag = AI.to(tl.float64, bitcast=True)
    out_real = alpha_real * sum_real - alpha_imag * sum_imag
    out_imag = alpha_real * sum_imag + alpha_imag * sum_real
    if not BETA_ZERO:
        beta_real = BR.to(tl.float64, bitcast=True)
        beta_imag = BI.to(tl.float64, bitcast=True)
        old_real = tl.load(Y + 2 * row)
        old_imag = tl.load(Y + 2 * row + 1)
        out_real += beta_real * old_real - beta_imag * old_imag
        out_imag += beta_real * old_imag + beta_imag * old_real
    tl.store(Y + 2 * row, out_real)
    tl.store(Y + 2 * row + 1, out_imag)


@triton.jit
def zgemv_tc_pairload_thead_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    AR: tl.int64,
    AI: tl.int64,
    BR: tl.int64,
    BI: tl.int64,
    CONJ: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    out = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k0 = tl.arange(0, BLOCK_K)
    component = tl.arange(0, 2)
    acc_real = tl.zeros((BLOCK_M, BLOCK_K), tl.float64)
    acc_imag = tl.zeros((BLOCK_M, BLOCK_K), tl.float64)
    for start in range(0, M, BLOCK_K):
        k = start + k0
        x_pair = tl.load(
            X + 2 * k[:, None] + component[None, :],
            k[:, None] < M,
            other=0,
        )
        x_real, x_imag = tl.split(x_pair)
        a_index = k[None, :] * N + out[:, None]
        a_mask = (k[None, :] < M) & (out[:, None] < N)
        a_pair = tl.load(
            A + 2 * a_index[:, :, None] + component[None, None, :],
            a_mask[:, :, None],
            other=0,
        )
        a_real, a_imag = tl.split(a_pair)
        if CONJ:
            acc_real += a_real * x_real[None, :] + a_imag * x_imag[None, :]
            acc_imag += a_real * x_imag[None, :] - a_imag * x_real[None, :]
        else:
            acc_real += a_real * x_real[None, :] - a_imag * x_imag[None, :]
            acc_imag += a_real * x_imag[None, :] + a_imag * x_real[None, :]
    total_real = tl.sum(acc_real, 1)
    total_imag = tl.sum(acc_imag, 1)
    alpha_real = AR.to(tl.float64, bitcast=True)
    alpha_imag = AI.to(tl.float64, bitcast=True)
    result_real = alpha_real * total_real - alpha_imag * total_imag
    result_imag = alpha_real * total_imag + alpha_imag * total_real
    if not BETA_ZERO:
        beta_real = BR.to(tl.float64, bitcast=True)
        beta_imag = BI.to(tl.float64, bitcast=True)
        old_real = tl.load(Y + 2 * out, out < N, other=0)
        old_imag = tl.load(Y + 2 * out + 1, out < N, other=0)
        result_real += beta_real * old_real - beta_imag * old_imag
        result_imag += beta_real * old_imag + beta_imag * old_real
    tl.store(Y + 2 * out, result_real, out < N)
    tl.store(Y + 2 * out + 1, result_imag, out < N)


# Small, verified regions only; keep the shared autotuner configs untouched.
_sg_small_t = runtime.get_tuned_config("thead_sgemv_t_fullk")
_SMALL_T_CONFIGS = {
    torch.float32: {c.kwargs["BLOCK_SIZE_K"]: c for c in _sg_small_t},
    torch.float64: {
        256: runtime.get_tuned_config("thead_dgemv_t_fullk")[0],
        # Same tile as the verified FP32 1024 region; no duplicate YAML config.
        1024: next(c for c in _sg_small_t if c.kwargs["BLOCK_SIZE_K"] == 1024),
    },
}
_LOWP_N_CONFIG = runtime.get_tuned_config("thead_lowp_gemv_n_medium")[0]
_LOWP_T_K4_CONFIG = runtime.get_tuned_config("thead_lowp_gemv_t_k4")[0]
_LOWP_DIRECT_KERNELS = {
    (dtype, trans): libentry()(getattr(_common, name + suffix).jit_function)
    for dtype, name in ((torch.float16, "hgemv"), (torch.bfloat16, "bfgemv"))
    for trans, suffix in ((CUBLAS_OP_N, "_n_kernel"), (CUBLAS_OP_T, "_t_kernel"))
}


@libentry()
@triton.jit
def sgemv_t_runtime_k_thead_kernel(
    A,
    X,
    Y,
    alpha,
    beta,
    M,
    K,
    LDA,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Dispatch guarantees K <= BLOCK_SIZE_K; the actual K remains a runtime
    # argument, with masked loads (also valid for non-power-of-two lengths).
    cols = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ks = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(
        A + ks[:, None] * LDA + cols[None, :],
        (ks[:, None] < K) & (cols[None, :] < M),
        other=0,
    )
    x = tl.load(X + ks * INCX, ks < K, other=0)
    result = alpha * tl.sum(a * x[:, None], 0)
    if not BETA_ZERO:
        result += beta * tl.load(Y + cols * INCY, cols < M, other=0)
    tl.store(Y + cols * INCY, result, cols < M)


@libentry()
@triton.jit
def gemv_t_fullk_thead_kernel(
    A,
    X,
    Y,
    alpha,
    beta,
    M: tl.constexpr,
    K: tl.constexpr,
    LDA,
    INCX,
    INCY,
    BETA_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    FP64: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ks = tl.arange(0, BLOCK_SIZE_K)
    if FP64:
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float64)
        # Zero bit patterns can be inferred as int32 at the Python boundary.
        aa = alpha.to(tl.int64).to(tl.float64, bitcast=True)
        bb = beta.to(tl.int64).to(tl.float64, bitcast=True)
    else:
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        aa = alpha
        bb = beta
    for start in range(0, K, BLOCK_SIZE_K):
        k = start + ks
        a = tl.load(
            A + k[:, None] * LDA + cols[None, :],
            (k[:, None] < K) & (cols[None, :] < M),
            other=0,
        )
        x = tl.load(X + k * INCX, k < K, other=0)
        acc += a * x[:, None]
    result = aa * tl.sum(acc, 0)
    if not BETA_ZERO:
        result += bb * tl.load(Y + cols * INCY, cols < M, other=0)
    tl.store(Y + cols * INCY, result, cols < M)


def _try_direct_gemv(dtype, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if incx != 1 or incy != 1:
        return False
    real = dtype in _SMALL_T_CONFIGS
    if real:
        config = (
            _SMALL_T_CONFIGS[dtype].get(m) if trans == CUBLAS_OP_T and m == n else None
        )
        kernel = (
            sgemv_t_runtime_k_thead_kernel
            if dtype == torch.float32
            else gemv_t_fullk_thead_kernel
        )
    else:
        if trans == CUBLAS_OP_N and m == n == 1024:
            config = _LOWP_N_CONFIG
        elif trans == CUBLAS_OP_T and m == 4 and n == 131072:
            config = _LOWP_T_K4_CONFIG
        else:
            return False
        kernel = _LOWP_DIRECT_KERNELS[(dtype, trans)]
    if config is None:
        return False
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == x.dtype == y.dtype == dtype
    assert A.device == x.device == y.device
    assert lda >= n
    out_m, reduce_k = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert x.numel() >= reduce_k and y.numel() >= out_m
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_value = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return True
    aa, bb = alpha_value, beta_value
    if dtype == torch.float64:
        aa = struct.unpack("=q", struct.pack("=d", aa))[0]
        bb = struct.unpack("=q", struct.pack("=d", bb))[0]
    extra = {"FP64": True} if dtype == torch.float64 else {}
    with torch_device_fn.device(A.device):
        kernel[(triton.cdiv(out_m, config.kwargs["BLOCK_SIZE_M"]),)](
            A,
            x,
            y,
            aa,
            bb,
            out_m,
            reduce_k,
            lda,
            incx,
            incy,
            beta_value == 0.0,
            **config.all_kwargs(),
            **extra,
        )
    return True


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemv_t_narrow_thead"),
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "num_k_splits"],
    restore_value=["y_ptr"],
)
@triton.jit
def sgemv_t_narrow_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m,
    n,
    STRIDE_AK,
    INCX,
    INCY,
    alpha: tl.float32,
    num_k_splits,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + num_k_splits - 1) // num_k_splits
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + (k_begin + ks0)[None, :] * STRIDE_AK
    x_ptrs = x_ptr + (k_begin + ks0) * INCX
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + ks0
        k_mask = ks < k_end
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")
        acc += tl.sum(a * x[None, :], axis=1)
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K * INCX
    tl.atomic_add(
        y_ptr + rows * INCY,
        acc * alpha,
        mask=row_mask,
        sem="relaxed",
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dgemv_t_narrow_thead"),
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "num_k_splits"],
    restore_value=["y_ptr"],
)
@triton.jit
def dgemv_t_narrow_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m,
    n,
    STRIDE_AK,
    INCX,
    INCY,
    alpha_int: tl.int64,
    num_k_splits,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    alpha = alpha_int.to(tl.float64, bitcast=True)
    chunk_k = (n + num_k_splits - 1) // num_k_splits
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + (k_begin + ks0)[None, :] * STRIDE_AK
    x_ptrs = x_ptr + (k_begin + ks0) * INCX
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + ks0
        k_mask = ks < k_end
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")
        acc += tl.sum(a * x[None, :], axis=1)
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K * INCX
    tl.atomic_add(
        y_ptr + rows * INCY,
        acc * alpha,
        mask=row_mask,
        sem="relaxed",
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lowp_gemv_t_thead"),
    key=["out_m", "reduce_k", "STRIDE_AK", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def lowp_gemv_t_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    reduce_k,
    STRIDE_AK,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < out_m
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + ks0[None, :] * STRIDE_AK
    x_ptrs = x_ptr + ks0 * INCX
    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k_start in range(0, reduce_k, BLOCK_SIZE_K):
        ks = k_start + ks0
        k_mask = ks < reduce_k
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = tl.load(
            x_ptrs,
            mask=k_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        acc_2d += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K * INCX
    acc = tl.sum(acc_2d, axis=1)
    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * old_y
    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lowp_gemv_t_narrow_split_thead"),
    key=["out_m", "reduce_k", "STRIDE_AK", "num_k_splits"],
)
@triton.jit
def lowp_gemv_t_narrow_split_thead_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_m,
    reduce_k,
    STRIDE_AK,
    num_k_splits,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < out_m
    chunk_k = (reduce_k + num_k_splits - 1) // num_k_splits
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, reduce_k)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + (k_begin + ks0)[None, :] * STRIDE_AK
    x_ptrs = x_ptr + k_begin + ks0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + ks0
        k_mask = ks < k_end
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last").to(
            tl.float32
        )
        acc += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K
    sums = tl.sum(acc, axis=1)
    tl.store(partial_ptr + pid_k * out_m + rows, sums, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemv_t_k1_thead"),
    key=["out_m", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def sgemv_t_k1_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_m
    scale = tl.load(x_ptr).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y_ptrs = y_ptr + offsets * INCY
    if BETA_IS_ZERO:
        result = alpha * a * scale
    else:
        old_y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        result = alpha * a * scale + beta * old_y
    tl.store(y_ptrs, result, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lowp_gemv_t_k1_thead"),
    key=["out_m", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def lowp_gemv_t_k1_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_m
    scale = tl.load(x_ptr).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y_ptrs = y_ptr + offsets * INCY
    if BETA_IS_ZERO:
        result = alpha * a * scale
    else:
        old_y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        result = alpha * a * scale + beta * old_y
    tl.store(y_ptrs, result, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dgemv_n3_thead"),
    key=["m", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def dgemv_n3_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    m,
    STRIDE_AM,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    REDUCE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    ks = tl.arange(0, BLOCK_SIZE_K)
    k_mask = ks < REDUCE_K
    a = tl.load(
        a_ptr + rows[:, None] * STRIDE_AM + ks[None, :],
        mask=row_mask[:, None] & k_mask[None, :],
        other=0.0,
    )
    x = tl.load(x_ptr + ks * INCX, mask=k_mask, other=0.0)
    acc = tl.sum(a * x[None, :], axis=1)
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * old_y
    tl.store(y_ptrs, result, mask=row_mask)


def _use_narrow_t_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m >= _NARROW_T_MIN_K
        and 0 < n <= _NARROW_T_MAX_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_large_lowp_t_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m >= _LARGE_LOWP_T_MIN_K
        and n >= _LARGE_LOWP_T_MIN_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_narrow_lowp_t_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m >= _NARROW_LOWP_T_MIN_K
        and 0 < n <= _NARROW_LOWP_T_MAX_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_t_k1_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m == 1
        and n >= _K1_T_MIN_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_dgemv_n3_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return trans == CUBLAS_OP_N and m >= 65536 and n == 3 and incx == 1 and incy == 1


def _prepare_narrow_t(
    trans: int,
    m: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T
    assert incx > 0 and incy > 0
    assert lda >= n
    assert x.numel() >= 1 + (m - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy


def _scale_y(y: torch.Tensor, beta: float) -> None:
    if beta == 0.0:
        y.zero_()
    elif beta != 1.0:
        y.mul_(beta)


def _gemv_t_k1(
    kernel,
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
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T and m == 1
    assert incx == 1 and incy == 1
    assert lda >= n
    assert x.numel() >= 1
    assert y.numel() >= n

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device):
        kernel[grid](
            A,
            x,
            y,
            alpha_value,
            beta_value,
            n,
            incy,
            beta_value == 0.0,
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
    if _try_direct_gemv(
        torch.float32, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    ):
        return
    if _use_t_k1_path(trans, m, n, incx, incy):
        return _gemv_t_k1(
            sgemv_t_k1_thead_kernel,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
            torch.float32,
        )
    if not _use_narrow_t_path(trans, m, n, incx, incy):
        return _common.sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    _prepare_narrow_t(trans, m, n, A, lda, x, incx, y, incy, torch.float32)
    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    _scale_y(y, beta_value)
    grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE_M"]),
        _NARROW_T_SPLITS,
    )
    with torch_device_fn.device(A.device):
        sgemv_t_narrow_thead_kernel[grid](
            A,
            x,
            y,
            n,
            m,
            lda,
            incx,
            incy,
            alpha_value,
            _NARROW_T_SPLITS,
        )


def dgemv(
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
    if _try_direct_gemv(
        torch.float64, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    ):
        return
    if _use_dgemv_n3_path(trans, m, n, incx, incy):
        assert A.is_contiguous()
        assert x.is_contiguous()
        assert y.is_contiguous()
        assert A.dtype == torch.float64
        assert x.dtype == torch.float64
        assert y.dtype == torch.float64
        assert A.device == x.device == y.device
        assert lda >= n
        assert x.numel() >= 3
        assert y.numel() >= m
        alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
        beta_value = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
        if alpha_value == 0.0:
            _scale_y(y, beta_value)
            return
        alpha_int = struct.unpack("=q", struct.pack("=d", alpha_value))[0]
        beta_int = struct.unpack("=q", struct.pack("=d", beta_value))[0]
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            dgemv_n3_thead_kernel[grid](
                A,
                x,
                y,
                alpha_int,
                beta_int,
                m,
                lda,
                incx,
                incy,
                beta_value == 0.0,
                3,
            )
        return
    if not _use_narrow_t_path(trans, m, n, incx, incy):
        return _common.dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    _prepare_narrow_t(trans, m, n, A, lda, x, incx, y, incy, torch.float64)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_value = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    alpha_int = struct.unpack("=q", struct.pack("=d", alpha_value))[0]
    _scale_y(y, beta_value)
    grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE_M"]),
        _NARROW_T_SPLITS,
    )
    with torch_device_fn.device(A.device):
        dgemv_t_narrow_thead_kernel[grid](
            A,
            x,
            y,
            n,
            m,
            lda,
            incx,
            incy,
            alpha_int,
            _NARROW_T_SPLITS,
        )


def _lowp_gemv(
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
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T
    assert incx > 0 and incy > 0
    assert lda >= n
    assert x.numel() >= 1 + (m - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    beta_is_zero = beta_value == 0.0
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
    with torch_device_fn.device(A.device):
        lowp_gemv_t_thead_kernel[grid](
            A,
            x,
            y,
            alpha_value,
            beta_value,
            n,
            m,
            lda,
            incx,
            incy,
            beta_is_zero,
        )


def _lowp_gemv_narrow_split(
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
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T
    assert incx == 1 and incy == 1
    assert lda >= n
    assert x.numel() >= m
    assert y.numel() >= n

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    num_k_splits = min(triton.cdiv(m, 1024), _THEAD_SM_COUNT // n)
    partial = torch.empty(num_k_splits, n, dtype=torch.float32, device=A.device)
    split_grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE_M"]),
        num_k_splits,
    )
    reduce_grid = (triton.cdiv(n, 32),)
    reduce_kernel = (
        _common.hgemv_splitk_reduce_kernel
        if dtype == torch.float16
        else _common.bfgemv_splitk_reduce_kernel
    )
    with torch_device_fn.device(A.device):
        lowp_gemv_t_narrow_split_thead_kernel[split_grid](
            A,
            x,
            partial,
            n,
            m,
            lda,
            num_k_splits,
        )
        reduce_kernel[reduce_grid](
            partial,
            y,
            alpha_value,
            beta_value,
            n,
            num_k_splits,
            incy,
            beta_value == 0.0,
            BLOCK_SIZE_M=32,
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
    if _try_direct_gemv(
        torch.float16, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    ):
        return
    if _use_t_k1_path(trans, m, n, incx, incy):
        return _gemv_t_k1(
            lowp_gemv_t_k1_thead_kernel,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
            torch.float16,
        )
    if _use_narrow_lowp_t_path(trans, m, n, incx, incy):
        return _lowp_gemv_narrow_split(
            trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.float16
        )
    if not _use_large_lowp_t_path(trans, m, n, incx, incy):
        return _common.hgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    return _lowp_gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.float16)


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
    if _try_direct_gemv(
        torch.bfloat16, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    ):
        return
    if _use_t_k1_path(trans, m, n, incx, incy):
        return _gemv_t_k1(
            lowp_gemv_t_k1_thead_kernel,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
            torch.bfloat16,
        )
    if _use_narrow_lowp_t_path(trans, m, n, incx, incy):
        return _lowp_gemv_narrow_split(
            trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.bfloat16
        )
    if not _use_large_lowp_t_path(trans, m, n, incx, incy):
        return _common.bfgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    return _lowp_gemv(
        trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.bfloat16
    )


def zgemv(
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
    # These N families were measured with the benchmark's KERNEL timer and
    # checked against a CPU reference before dispatch was enabled.
    narrow_n = 63 <= m <= 64 and 65535 <= n <= 65536
    medium_n = 4095 <= m <= 4096 and 1023 <= n <= 1024
    square_n = m == n and 3583 <= m <= 3584
    square_4k_n = 4095 <= m <= 4096 and 4095 <= n <= 4096
    tall_n = 65535 <= m <= 65536 and n == 64
    square_tc = (
        trans in (CUBLAS_OP_T, CUBLAS_OP_C)
        and m == n
        and (3583 <= m <= 3584 or 4095 <= m <= 4096)
    )
    wide_tc = trans in (CUBLAS_OP_T, CUBLAS_OP_C) and m == 4095 and n == 14335
    direct_tc = square_tc or wide_tc
    if not (
        (
            (
                trans == CUBLAS_OP_N
                and (narrow_n or medium_n or square_n or square_4k_n or tall_n)
            )
            or direct_tc
        )
        and lda == n
        and incx == incy == 1
    ):
        return _common.zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == x.dtype == y.dtype == torch.complex128
    assert A.device == x.device == y.device
    assert x.numel() >= (m if direct_tc else n)
    assert y.numel() >= (n if direct_tc else m)

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else beta
    if alpha_value == 0:
        return _common.zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    alpha_real_int = _common._float64_to_int(float(alpha_value.real))
    alpha_imag_int = _common._float64_to_int(float(alpha_value.imag))

    if direct_tc:
        beta_real_int = _common._float64_to_int(float(beta_value.real))
        beta_imag_int = _common._float64_to_int(float(beta_value.imag))
        block_m, block_k = (32, 32) if wide_tc else (16, 64)
        with torch_device_fn.device(A.device):
            zgemv_tc_pairload_thead_kernel[(triton.cdiv(n, block_m),)](
                A_real,
                x_real,
                y_real,
                m,
                n,
                alpha_real_int,
                alpha_imag_int,
                beta_real_int,
                beta_imag_int,
                trans == CUBLAS_OP_C,
                beta_value == 0,
                block_m,
                block_k,
                num_warps=2,
                num_stages=1,
            )
        return

    if square_4k_n:
        beta_real_int = _common._float64_to_int(float(beta_value.real))
        beta_imag_int = _common._float64_to_int(float(beta_value.imag))
        block, warps = (256, 4) if n == 4095 else (128, 2)
        with torch_device_fn.device(A.device):
            zgemv_n_unrolled_thead_kernel[(m,)](
                A_real,
                x_real,
                y_real,
                m,
                n,
                alpha_real_int,
                alpha_imag_int,
                beta_real_int,
                beta_imag_int,
                beta_value == 0,
                block,
                num_warps=warps,
                num_stages=1,
            )
        return

    if tall_n:
        beta_real_int = _common._float64_to_int(float(beta_value.real))
        beta_imag_int = _common._float64_to_int(float(beta_value.imag))
        with torch_device_fn.device(A.device):
            _ZGEMV_TALL_N_KERNEL[(triton.cdiv(m, 2),)](
                A_real,
                x_real,
                y_real,
                alpha_real_int,
                alpha_imag_int,
                beta_real_int,
                beta_imag_int,
                m,
                lda,
                incx,
                incy,
                beta_value == 0,
                64,
                BLOCK_SIZE_M=2,
                num_warps=1,
                num_stages=1,
            )
        return

    if medium_n or square_n:
        beta_real_int = _common._float64_to_int(float(beta_value.real))
        beta_imag_int = _common._float64_to_int(float(beta_value.imag))
        block_k = 256 if square_n else 128
        with torch_device_fn.device(A.device):
            _ZGEMV_N_KERNEL[(m,)](
                A_real,
                x_real,
                y_real,
                alpha_real_int,
                alpha_imag_int,
                beta_real_int,
                beta_imag_int,
                m,
                n,
                lda,
                incx,
                incy,
                beta_value == 0,
                BLOCK_SIZE_M=1,
                BLOCK_SIZE_K=block_k,
                num_warps=2,
                num_stages=1,
            )
        return

    with torch_device_fn.device(A.device):
        if beta_value == 0:
            y.zero_()
        elif beta_value != 1:
            y.mul_(beta_value)
        _ZGEMV_NARROW_N_KERNEL[(triton.cdiv(m, 2), _ZGEMV_NARROW_N_SPLITS)](
            A_real,
            x_real,
            y_real,
            m,
            n,
            lda,
            incx,
            incy,
            alpha_real_int,
            alpha_imag_int,
            _ZGEMV_NARROW_N_SPLITS,
            BLOCK_SIZE_M=2,
            BLOCK_SIZE_K=128,
            num_warps=2,
            num_stages=1,
        )


def cgemv(
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
    # Reuse independently measured Triton configs for these shape families.
    medium_n = 4095 <= m <= 4096 and 1023 <= n <= 1024
    square_n = (
        trans == CUBLAS_OP_N and m == n and (3583 <= m <= 3584 or 4095 <= m <= 4096)
    )
    tall_n = 65535 <= m <= 65536 and n == 64
    wide_odd_n = 12288 <= m <= 14336 and n == 4095
    wide_tc = (
        trans in (CUBLAS_OP_T, CUBLAS_OP_C)
        and 4095 <= m <= 4096
        and 14335 <= n <= 14336
    )
    square_tc = trans in (CUBLAS_OP_T, CUBLAS_OP_C) and m == n and 7167 <= m <= 7168
    small_wide_tc = trans in (CUBLAS_OP_T, CUBLAS_OP_C) and m == 1023 and n == 4095
    small_wide_c = trans == CUBLAS_OP_C and m == 1024 and n == 4096
    wide_odd_tc = wide_odd_n and trans in (CUBLAS_OP_T, CUBLAS_OP_C)
    tall_tc = trans in (CUBLAS_OP_T, CUBLAS_OP_C) and 65535 <= m <= 65536 and n == 64
    direct_tc = (
        wide_tc or square_tc or small_wide_tc or small_wide_c or wide_odd_tc or tall_tc
    )
    if not (
        (
            (trans == CUBLAS_OP_N and (medium_n or square_n or tall_n or wide_odd_n))
            or direct_tc
        )
        and lda == n
        and incx == incy == 1
    ):
        return _common.cgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == x.dtype == y.dtype == torch.complex64
    assert A.device == x.device == y.device
    assert x.numel() >= (m if direct_tc else n)
    assert y.numel() >= (n if direct_tc else m)

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else beta
    if alpha_value == 0:
        return _common.cgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    with torch_device_fn.device(A.device):
        if square_n:
            block, warps, stages = (
                (128, 1, 2) if m == 3583 else (256, 2, 2) if m == 4095 else (128, 1, 1)
            )
            cgemv_n_square_thead_kernel[(m,)](
                A_real,
                x_real,
                y_real,
                n,
                float(alpha_value.real),
                float(alpha_value.imag),
                float(beta_value.real),
                float(beta_value.imag),
                beta_value == 0,
                block,
                stages,
                num_warps=warps,
                num_stages=1,
            )
            return
        if tall_tc:
            if beta_value != 1:
                cgemv_tc_tall_scale_thead_kernel[(1,)](
                    y_real,
                    n,
                    float(beta_value.real),
                    float(beta_value.imag),
                    beta_value == 0,
                    num_warps=4,
                    num_stages=1,
                )
            cgemv_tc_tall_split_thead_kernel[(triton.cdiv(n, 32), 80)](
                A_real,
                x_real,
                y_real,
                m,
                n,
                float(alpha_value.real),
                float(alpha_value.imag),
                trans == CUBLAS_OP_C,
                num_warps=4,
                num_stages=1,
            )
            return
        if wide_odd_n and trans == CUBLAS_OP_N:
            cgemv_n_fma_thead_kernel[(triton.cdiv(m, 4),)](
                A_real,
                x_real,
                y_real,
                m,
                n,
                float(alpha_value.real),
                float(alpha_value.imag),
                float(beta_value.real),
                float(beta_value.imag),
                beta_value == 0,
                4,
                512,
                num_warps=4,
                num_stages=1,
            )
            return
        if wide_odd_tc:
            cgemv_tc_wide_thead_kernel[(triton.cdiv(n, 32),)](
                A_real,
                x_real,
                y_real,
                m,
                n,
                float(alpha_value.real),
                float(alpha_value.imag),
                float(beta_value.real),
                float(beta_value.imag),
                trans == CUBLAS_OP_C,
                beta_value == 0,
                num_warps=4,
                num_stages=1,
            )
            return
        if direct_tc:
            block_m = 8 if small_wide_tc else (16 if square_tc and m == 7167 else 32)
            _CGEMV_TC_KERNEL[(triton.cdiv(n, block_m),)](
                A_real,
                x_real,
                y_real,
                float(alpha_value.real),
                float(alpha_value.imag),
                float(beta_value.real),
                float(beta_value.imag),
                n,
                m,
                1,
                lda,
                incx,
                incy,
                int(trans == CUBLAS_OP_C),
                beta_value == 0,
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_K=64,
                num_warps=4 if small_wide_tc or small_wide_c else 2,
                num_stages=1,
            )
            return
        if tall_n:
            _CGEMV_N_KERNEL[(triton.cdiv(m, 2),)](
                A_real,
                x_real,
                y_real,
                float(alpha_value.real),
                float(alpha_value.imag),
                float(beta_value.real),
                float(beta_value.imag),
                m,
                n,
                lda,
                incx,
                incy,
                beta_value == 0,
                BLOCK_SIZE_M=2,
                BLOCK_SIZE_K=128,
                num_warps=1,
                num_stages=1,
            )
            return
        if n == 1023:
            cgemv_n_unrolled_thead_kernel[(m,)](
                A_real,
                x_real,
                y_real,
                m,
                n,
                float(alpha_value.real),
                float(alpha_value.imag),
                float(beta_value.real),
                float(beta_value.imag),
                beta_value == 0,
                128,
                num_warps=4,
                num_stages=1,
            )
            return
        _CGEMV_N_KERNEL[(m,)](
            A_real,
            x_real,
            y_real,
            float(alpha_value.real),
            float(alpha_value.imag),
            float(beta_value.real),
            float(beta_value.imag),
            m,
            n,
            lda,
            incx,
            incy,
            beta_value == 0,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_K=256,
            num_warps=1,
            num_stages=1,
        )

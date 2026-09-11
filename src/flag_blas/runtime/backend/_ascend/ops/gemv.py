import importlib
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ScalarType = Union[float, int, complex, torch.Tensor]

_common = importlib.import_module("flag_blas.ops.level2.gemv")


@triton.jit
def _sgemv_n_short_k1_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    STRIDE_AM,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    a_values = tl.load(a_ptr + rows * STRIDE_AM, mask=row_mask, other=0.0)
    x_value = tl.load(x_ptr)
    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        result = alpha * a_values * x_value
    else:
        y_values = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * a_values * x_value + beta * y_values
    tl.store(y_ptrs, result, mask=row_mask)


sgemv_n_wide_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_wide_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_common.sgemv_n_kernel.jit_function)
)

sgemv_n_four_to_one_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_four_to_one_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_common.sgemv_n_kernel.jit_function)
)

sgemv_n_small_square_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_small_square_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_common.sgemv_n_kernel.jit_function)
)

sgemv_n_short_k1_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_short_k1_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_sgemv_n_short_k1_jit)
)


def _select_sgemv_n_path(m, n):
    if m <= 4 and n >= 65536:
        return "wide_regular"
    if 4095 <= m <= 4096 and 1023 <= n <= 1024:
        return "four_to_one"
    if m == n and 63 <= m <= 1024:
        return "small_square"
    if m >= 65536 and n == 1:
        return "short_k1"
    if m <= 64 and n >= 4096:
        return "splitk"
    return "regular"


def _launch_sgemv_n(
    entry_tag,
    kernel,
    A,
    x,
    y,
    alpha,
    beta,
    m,
    n,
    lda,
    incx,
    incy,
    beta_is_zero,
):
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
    kernel[grid](A, x, y, alpha, beta, m, n, lda, incx, incy, beta_is_zero)


def _launch_on_tensor_device(tensor, launch):
    device_index = tensor.device.index
    if device_index is None or device_index == torch_device_fn.current_device():
        launch()
        return
    with torch_device_fn.device(tensor.device):
        launch()


def _scale_complex_y(y, length, stride, beta_real, beta_imag):
    logical_y = torch.view_as_real(y).as_strided((length, 2), (stride * 2, 1))
    if beta_real == 0.0 and beta_imag == 0.0:
        logical_y.zero_()
    elif beta_real != 1.0 or beta_imag != 0.0:
        values = logical_y.clone()
        logical_y[:, 0].copy_(beta_real * values[:, 0] - beta_imag * values[:, 1])
        logical_y[:, 1].copy_(beta_real * values[:, 1] + beta_imag * values[:, 0])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cgemv_ascend"),
    key=[
        "m",
        "n",
        "STRIDE_AM",
        "STRIDE_AN",
        "INCX",
        "INCY",
        "CONJ",
        "BETA_IS_ZERO",
    ],
    restore_value=["y_ptr"],
)
@triton.jit
def cgemv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    INCX,
    INCY,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    k_init = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_start in range(0, n, BLOCK_SIZE_K):
        ks = k_start + k_init
        k_mask = ks < n
        mask = row_mask[:, None] & k_mask[None, :]
        a_elem = rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AN
        a_off = a_elem * 2
        x_off = ks * INCX * 2
        a_real = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        a_imag = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        x_real = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + x_off + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)
    out_real = alpha_real * acc_real - alpha_imag * acc_imag
    out_imag = alpha_real * acc_imag + alpha_imag * acc_real
    y_off = rows * INCY * 2
    if not BETA_IS_ZERO:
        y_real = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        out_real += beta_real * y_real - beta_imag * y_imag
        out_imag += beta_real * y_imag + beta_imag * y_real
    tl.store(y_ptr + y_off, out_real, mask=row_mask)
    tl.store(y_ptr + y_off + 1, out_imag, mask=row_mask)


@triton.jit
def _cgemv_ktranspose_splitk_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_offsets
        k_mask = ks < k_end
        load_mask = k_mask[:, None] & row_mask[None, :]
        a_elem = ks[:, None] * STRIDE_AN + rows[None, :] * STRIDE_AM
        a_off = a_elem * 2
        a_real = tl.trans(tl.load(a_ptr + a_off, mask=load_mask, other=0.0))
        a_imag = tl.trans(tl.load(a_ptr + a_off + 1, mask=load_mask, other=0.0))
        x_real = tl.load(x_ptr + ks * 2, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + ks * 2 + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)
    result_real = alpha_real * acc_real - alpha_imag * acc_imag
    result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    y_off = rows * 2
    tl.atomic_add(y_ptr + y_off, result_real, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, result_imag, mask=row_mask, sem="relaxed")


@libentry()
@triton.jit
def cgemv_t_partial_kernel_ascend(
    a_ptr,
    x_ptr,
    partial_ptr,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_offsets
        k_mask = ks < k_end
        load_mask = k_mask[:, None] & row_mask[None, :]
        a_elem = ks[:, None] * STRIDE_AN + rows[None, :] * STRIDE_AM
        a_off = a_elem * 2
        a_real = tl.trans(tl.load(a_ptr + a_off, mask=load_mask, other=0.0))
        a_imag = tl.trans(tl.load(a_ptr + a_off + 1, mask=load_mask, other=0.0))
        x_real = tl.load(x_ptr + ks * 2, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + ks * 2 + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    partial_elem = pid_k * m + rows
    partial_off = partial_elem * 2
    tl.store(
        partial_ptr + partial_off,
        tl.sum(acc_real_2d, axis=1),
        mask=row_mask,
    )
    tl.store(
        partial_ptr + partial_off + 1,
        tl.sum(acc_imag_2d, axis=1),
        mask=row_mask,
    )


@libentry()
@triton.jit
def cgemv_t_reduce_kernel_ascend(
    partial_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    BETA_IS_ZERO: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    splits = tl.arange(0, BLOCK_SPLITS)
    row_mask = rows < m
    mask = row_mask[:, None] & (splits[None, :] < SPLIT_COUNT)
    partial_elem = splits[None, :] * m + rows[:, None]
    partial_off = partial_elem * 2
    sum_real = tl.sum(tl.load(partial_ptr + partial_off, mask=mask, other=0.0), axis=1)
    sum_imag = tl.sum(
        tl.load(partial_ptr + partial_off + 1, mask=mask, other=0.0), axis=1
    )
    out_real = alpha_real * sum_real - alpha_imag * sum_imag
    out_imag = alpha_real * sum_imag + alpha_imag * sum_real
    y_off = rows * 2
    if not BETA_IS_ZERO:
        y_real = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        out_real += beta_real * y_real - beta_imag * y_imag
        out_imag += beta_real * y_imag + beta_imag * y_real
    tl.store(y_ptr + y_off, out_real, mask=row_mask)
    tl.store(y_ptr + y_off + 1, out_imag, mask=row_mask)


@libentry()
@triton.jit
def cgemv_t_small_kernel_ascend(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_start in range(0, n, BLOCK_SIZE_K):
        ks = k_start + k_offsets
        k_mask = ks < n
        load_mask = k_mask[:, None] & row_mask[None, :]
        a_elem = ks[:, None] * STRIDE_AN + rows[None, :] * STRIDE_AM
        a_off = a_elem * 2
        a_real = tl.trans(tl.load(a_ptr + a_off, mask=load_mask, other=0.0))
        a_imag = tl.trans(tl.load(a_ptr + a_off + 1, mask=load_mask, other=0.0))
        x_real = tl.load(x_ptr + ks * 2, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + ks * 2 + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    sum_real = tl.sum(acc_real_2d, axis=1)
    sum_imag = tl.sum(acc_imag_2d, axis=1)
    out_real = alpha_real * sum_real - alpha_imag * sum_imag
    out_imag = alpha_real * sum_imag + alpha_imag * sum_real
    y_off = rows * 2
    if not BETA_IS_ZERO:
        y_real = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        out_real += beta_real * y_real - beta_imag * y_imag
        out_imag += beta_real * y_imag + beta_imag * y_real
    tl.store(y_ptr + y_off, out_real, mask=row_mask)
    tl.store(y_ptr + y_off + 1, out_imag, mask=row_mask)


cgemv_n_k1_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_k1_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_n_shortk_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_shortk_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_n_small_tile_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_small_tile_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_n_k1024_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_k1024_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_t_splitk_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_t_splitk_ascend"),
        key=[
            "m",
            "n",
            "STRIDE_AM",
            "STRIDE_AN",
            "CONJ",
            "SPLIT_COUNT",
        ],
        restore_value=["y_ptr"],
    )(_cgemv_ktranspose_splitk_jit)
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cgemv_splitk_ascend"),
    key=[
        "m",
        "n",
        "STRIDE_AM",
        "STRIDE_AN",
        "CONJ",
        "SPLIT_COUNT",
    ],
    restore_value=["y_ptr"],
)
@triton.jit
def cgemv_splitk_kernel_ascend(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    k_init = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_init
        k_mask = ks < k_end
        mask = row_mask[:, None] & k_mask[None, :]
        a_elem = rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AN
        a_off = a_elem * 2
        x_off = ks * 2
        a_real = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        a_imag = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        x_real = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + x_off + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)
    result_real = alpha_real * acc_real - alpha_imag * acc_imag
    result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    y_off = rows * 2
    tl.atomic_add(y_ptr + y_off, result_real, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, result_imag, mask=row_mask, sem="relaxed")


def _cgemv_split_count(eff_m, eff_n, trans):
    if trans == CUBLAS_OP_N and 3 <= eff_m <= 4 and eff_n >= 131071:
        return 128
    if eff_n <= 512:
        return 8
    if eff_n <= 4096:
        return 16
    if eff_m >= 32:
        return 64
    return min(triton.cdiv(eff_n, 2048), 64)


def _cgemv_t_split_count(eff_n):
    if eff_n <= 512:
        return 8
    if eff_n <= 1024:
        return 16
    if 2049 <= eff_n <= 3584:
        return 32
    if 4095 <= eff_n <= 4096:
        return 24
    if 7167 <= eff_n <= 7168:
        return 24
    if eff_n == 18944:
        return 28
    return 16


def _use_cgemv_t_workspace(eff_m, eff_n, split_count):
    if eff_m <= 1024 or split_count == 32:
        return True
    if eff_m == 4095 and eff_n == 4095:
        return True
    if 1023 <= eff_n <= 1024 and eff_m >= 4095:
        return True
    if 4095 <= eff_n <= 4096 and eff_m >= 8192:
        return True
    if 7167 <= eff_n <= 7168 and 7167 <= eff_m <= 7168:
        return True
    return eff_n == 18944 and eff_m == 3584


def _use_cgemv_t_small(eff_m, eff_n):
    return 255 <= eff_m <= 256 and 255 <= eff_n <= 256


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
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.complex64
    assert x.dtype == torch.complex64
    assert y.dtype == torch.complex64
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert incx > 0 and incy > 0
    assert lda >= n
    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    beta_real = float(beta.real) if isinstance(beta, complex) else float(beta)
    beta_imag = float(beta.imag) if isinstance(beta, complex) else 0.0

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        eff_m, eff_n = m, n
        stride_am, stride_an = lda, 1
    else:
        len_x, len_y = m, n
        eff_m, eff_n = n, m
        stride_am, stride_an = 1, lda
    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy
    if alpha_real == 0.0 and alpha_imag == 0.0:
        _scale_complex_y(y, len_y, incy, beta_real, beta_imag)
        return

    beta_is_zero = beta_real == 0.0 and beta_imag == 0.0
    with torch_device_fn.device(A.device):
        use_n_shortk = (
            trans == CUBLAS_OP_N and m >= 65536 and n <= 64 and incx == 1 and incy == 1
        )
        if use_n_shortk:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            kernel = (
                cgemv_n_k1_kernel_ascend if n == 1 else cgemv_n_shortk_kernel_ascend
            )
            grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
            kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                m,
                n,
                stride_am,
                stride_an,
                1,
                1,
                CONJ=False,
                BETA_IS_ZERO=beta_is_zero,
            )
            return

        use_n_small_tile = (
            trans == CUBLAS_OP_N
            and 511 <= m <= 512
            and 511 <= n <= 512
            and incx == 1
            and incy == 1
        )
        use_n_k1024 = (
            trans == CUBLAS_OP_N
            and 1023 <= n <= 1024
            and (1023 <= m <= 1024 or 4095 <= m <= 4096)
            and incx == 1
            and incy == 1
        )
        if use_n_small_tile or use_n_k1024:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            kernel = (
                cgemv_n_small_tile_kernel_ascend
                if use_n_small_tile
                else cgemv_n_k1024_kernel_ascend
            )
            grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
            kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                m,
                n,
                stride_am,
                stride_an,
                1,
                1,
                CONJ=False,
                BETA_IS_ZERO=beta_is_zero,
            )
            return

        use_t_small = (
            trans != CUBLAS_OP_N
            and _use_cgemv_t_small(eff_m, eff_n)
            and incx == 1
            and incy == 1
        )
        if use_t_small:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            grid = (triton.cdiv(eff_m, 16),)
            cgemv_t_small_kernel_ascend[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                eff_m,
                eff_n,
                stride_am,
                stride_an,
                CONJ=trans == CUBLAS_OP_C,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_K=256,
                num_warps=1,
                num_stages=1,
            )
            return

        use_long_splitk = eff_m <= 64 and eff_n >= 65536 and incx == 1 and incy == 1
        use_t_contiguous_splitk = (
            trans != CUBLAS_OP_N
            and 511 <= eff_m
            and 511 <= eff_n
            and incx == 1
            and incy == 1
        )
        use_splitk = use_long_splitk or use_t_contiguous_splitk
        if use_splitk:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            if use_t_contiguous_splitk:
                split_count = _cgemv_t_split_count(eff_n)
                if _use_cgemv_t_workspace(eff_m, eff_n, split_count):
                    partial = torch.empty(
                        (split_count, eff_m, 2),
                        dtype=torch.float32,
                        device=A.device,
                    )
                    partial_grid = (triton.cdiv(eff_m, 32), split_count)
                    cgemv_t_partial_kernel_ascend[partial_grid](
                        A_real,
                        x_real,
                        partial,
                        eff_m,
                        eff_n,
                        stride_am,
                        stride_an,
                        CONJ=trans == CUBLAS_OP_C,
                        SPLIT_COUNT=split_count,
                        BLOCK_SIZE_M=32,
                        BLOCK_SIZE_K=64,
                        num_warps=1,
                        num_stages=1,
                    )
                    reduce_grid = (triton.cdiv(eff_m, 32),)
                    cgemv_t_reduce_kernel_ascend[reduce_grid](
                        partial,
                        y_real,
                        alpha_real,
                        alpha_imag,
                        beta_real,
                        beta_imag,
                        eff_m,
                        BETA_IS_ZERO=beta_is_zero,
                        SPLIT_COUNT=split_count,
                        BLOCK_SIZE_M=32,
                        BLOCK_SPLITS=32,
                        num_warps=1,
                        num_stages=1,
                    )
                    return
                kernel = cgemv_t_splitk_kernel_ascend
            else:
                split_count = _cgemv_split_count(eff_m, eff_n, trans)
                kernel = cgemv_splitk_kernel_ascend
            _scale_complex_y(y, len_y, incy, beta_real, beta_imag)
            grid = lambda meta: (
                triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),
                split_count,
            )
            kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                eff_m,
                eff_n,
                stride_am,
                stride_an,
                CONJ=trans == CUBLAS_OP_C,
                SPLIT_COUNT=split_count,
            )
            return

        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        y_real = torch.view_as_real(y)
        kernel_incx = incx
        kernel_incy = incy
        if incx != 1:
            x_real = x_real.as_strided((len_x, 2), (incx * 2, 1)).clone()
            kernel_incx = 1
        logical_y_real = None
        if incy != 1:
            logical_y_real = y_real.as_strided((len_y, 2), (incy * 2, 1))
            y_real = logical_y_real.clone()
            kernel_incy = 1
        grid = lambda meta: (triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),)
        cgemv_kernel[grid](
            A_real,
            x_real,
            y_real,
            alpha_real,
            alpha_imag,
            beta_real,
            beta_imag,
            eff_m,
            eff_n,
            stride_am,
            stride_an,
            kernel_incx,
            kernel_incy,
            CONJ=trans == CUBLAS_OP_C,
            BETA_IS_ZERO=beta_is_zero,
        )
        if logical_y_real is not None:
            logical_y_real.copy_(y_real)


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
    if m == 0 or n == 0:
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

    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy
    if incy != 1:
        logical_y = y.as_strided((len_y,), (incy,))
        y_contiguous = logical_y.clone()
        sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y_contiguous, 1)
        logical_y.copy_(y_contiguous)
        return
    if incx != 1:
        x_contiguous = x.as_strided((len_x,), (incx,)).clone()
        sgemv(trans, m, n, alpha, A, lda, x_contiguous, 1, beta, y, incy)
        return

    if trans != CUBLAS_OP_N:
        _common.sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    path = _select_sgemv_n_path(m, n)
    if path == "splitk":
        num_k_splits = min(triton.cdiv(n, 2048), 128)
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]), num_k_splits)

        def launch():
            _common.sgemv_n_splitk_kernel[grid](
                A, x, y, m, n, lda, incx, incy, alpha, num_k_splits
            )

    else:
        if path == "wide_regular":
            kernel = sgemv_n_wide_kernel
        elif path == "four_to_one":
            kernel = sgemv_n_four_to_one_kernel
        elif path == "small_square":
            kernel = sgemv_n_small_square_kernel
        elif path == "short_k1":
            kernel = sgemv_n_short_k1_kernel
        else:
            kernel = _common.sgemv_n_kernel
        beta_is_zero = beta == 0.0

        def launch():
            _launch_sgemv_n(
                path,
                kernel,
                A,
                x,
                y,
                alpha,
                beta,
                m,
                n,
                lda,
                incx,
                incy,
                beta_is_zero,
            )

    _launch_on_tensor_device(A, launch)

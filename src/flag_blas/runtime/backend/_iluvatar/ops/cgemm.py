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

from flag_blas.ops.level3.cgemm import (
    ScalarType,
    _cgemm_dot_kernel,
    _complex_scalar_parts,
    _validate_cgemm_args,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._iluvatar.ops.sgemm import sgemm as _sgemm_iluvatar

_CGEMM_WORKSPACE = {"key": None, "buffers": None}
_CGEMM_AUG_WORKSPACE = {"key": None, "buffers": None, "pack_key": None}


@triton.jit
def _cgemm_split_sum_op_kernel(
    src,
    dst_r,
    dst_i,
    dst_sum,
    total,
    cols: tl.constexpr,
    ld: tl.constexpr,
    TRANS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    row = offsets // cols
    col = offsets - row * cols
    if TRANS == 0:
        src_offsets = row * ld + col
    else:
        src_offsets = col * ld + row
    real = tl.load(src + 2 * src_offsets, mask=mask, other=0.0)
    imag = tl.load(src + 2 * src_offsets + 1, mask=mask, other=0.0)
    tl.store(dst_r + offsets, real, mask=mask)
    tl.store(dst_i + offsets, imag, mask=mask)
    tl.store(dst_sum + offsets, real + imag, mask=mask)


@triton.jit
def _cgemm_split_sum_trans_tile_kernel(
    src,
    dst_r,
    dst_i,
    dst_sum,
    rows: tl.constexpr,
    cols: tl.constexpr,
    ld: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    src_offsets = offs_c[:, None] * ld + offs_r[None, :]
    src_mask = (offs_c[:, None] < cols) & (offs_r[None, :] < rows)
    real_t = tl.load(src + 2 * src_offsets, mask=src_mask, other=0.0)
    imag_t = tl.load(src + 2 * src_offsets + 1, mask=src_mask, other=0.0)
    real = tl.trans(real_t)
    imag = tl.trans(imag_t)

    dst_offsets = offs_r[:, None] * cols + offs_c[None, :]
    dst_mask = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
    tl.store(dst_r + dst_offsets, real, mask=dst_mask)
    tl.store(dst_i + dst_offsets, imag, mask=dst_mask)
    tl.store(dst_sum + dst_offsets, real + imag, mask=dst_mask)


@triton.jit
def _cgemm_split_sum2_op_kernel(
    src_a,
    src_b,
    dst_ar,
    dst_ai,
    dst_as,
    dst_br,
    dst_bi,
    dst_bs,
    total_a,
    total_b,
    cols_a: tl.constexpr,
    cols_b: tl.constexpr,
    lda: tl.constexpr,
    ldb: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    mask_a = offsets < total_a
    row_a = offsets // cols_a
    col_a = offsets - row_a * cols_a
    if TRANS_A == 0:
        a_src = row_a * lda + col_a
    else:
        a_src = col_a * lda + row_a
    ar = tl.load(src_a + 2 * a_src, mask=mask_a, other=0.0)
    ai = tl.load(src_a + 2 * a_src + 1, mask=mask_a, other=0.0)
    tl.store(dst_ar + offsets, ar, mask=mask_a)
    tl.store(dst_ai + offsets, ai, mask=mask_a)
    tl.store(dst_as + offsets, ar + ai, mask=mask_a)

    mask_b = offsets < total_b
    row_b = offsets // cols_b
    col_b = offsets - row_b * cols_b
    if TRANS_B == 0:
        b_src = row_b * ldb + col_b
    else:
        b_src = col_b * ldb + row_b
    br = tl.load(src_b + 2 * b_src, mask=mask_b, other=0.0)
    bi = tl.load(src_b + 2 * b_src + 1, mask=mask_b, other=0.0)
    tl.store(dst_br + offsets, br, mask=mask_b)
    tl.store(dst_bi + offsets, bi, mask=mask_b)
    tl.store(dst_bs + offsets, br + bi, mask=mask_b)


@triton.jit
def _cgemm_merge_3m_kernel(dst, prod_r, prod_i, prod_sum, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    real_prod = tl.load(prod_r + offsets, mask=mask, other=0.0)
    imag_prod = tl.load(prod_i + offsets, mask=mask, other=0.0)
    sum_prod = tl.load(prod_sum + offsets, mask=mask, other=0.0)
    tl.store(dst + 2 * offsets, real_prod - imag_prod, mask=mask)
    tl.store(dst + 2 * offsets + 1, sum_prod - real_prod - imag_prod, mask=mask)


@triton.jit
def _cgemm_aug_pack_kernel(
    a_ptr,
    b_ptr,
    a_aug,
    b_aug,
    total_a: tl.constexpr,
    total_b: tl.constexpr,
    size: tl.constexpr,
    lda: tl.constexpr,
    ldb: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    mask_a = offsets < total_a
    a_row = offsets // (2 * size)
    a_col2 = offsets - a_row * (2 * size)
    a_k = a_col2 % size
    a_is_imag = a_col2 >= size
    if TRANS_A == 0:
        a_src = a_row * lda + a_k
    else:
        a_src = a_k * lda + a_row
    ar = tl.load(a_ptr + 2 * a_src, mask=mask_a, other=0.0)
    ai = tl.load(a_ptr + 2 * a_src + 1, mask=mask_a, other=0.0)
    aval = tl.where(a_is_imag, ai, ar)
    tl.store(a_aug + offsets, aval, mask=mask_a)

    mask_b = offsets < total_b
    b_row2 = offsets // (2 * size)
    b_col2 = offsets - b_row2 * (2 * size)
    b_k = b_row2 % size
    b_n = b_col2 // 2
    b_is_imag_col = (b_col2 % 2) == 1
    b_bottom = b_row2 >= size
    if TRANS_B == 0:
        b_src = b_k * ldb + b_n
    else:
        b_src = b_n * ldb + b_k
    br = tl.load(b_ptr + 2 * b_src, mask=mask_b, other=0.0)
    bi = tl.load(b_ptr + 2 * b_src + 1, mask=mask_b, other=0.0)
    top_val = tl.where(b_is_imag_col, bi, br)
    bottom_val = tl.where(b_is_imag_col, br, -bi)
    bval = tl.where(b_bottom, bottom_val, top_val)
    tl.store(b_aug + offsets, bval, mask=mask_b)


@triton.jit
def _cgemm_3m_nomask_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    SIZE: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    prod_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, SIZE, BLOCK_K):
        cur_k = k_start + offs_k_base
        if TRANS_A == 0:
            a_elem = offs_m[:, None] * SIZE + cur_k[None, :]
        else:
            a_elem = cur_k[None, :] * SIZE + offs_m[:, None]

        if TRANS_B == 0:
            b_elem = cur_k[:, None] * SIZE + offs_n[None, :]
        else:
            b_elem = offs_n[None, :] * SIZE + cur_k[:, None]

        ar = tl.load(a_ptr + 2 * a_elem)
        ai = tl.load(a_ptr + 2 * a_elem + 1)
        br = tl.load(b_ptr + 2 * b_elem)
        bi = tl.load(b_ptr + 2 * b_elem + 1)

        prod_r += tl.dot(ar, br, out_dtype=tl.float32, input_precision="tf32x3")
        prod_i += tl.dot(ai, bi, out_dtype=tl.float32, input_precision="tf32x3")
        prod_sum += tl.dot(
            ar + ai,
            br + bi,
            out_dtype=tl.float32,
            input_precision="tf32x3",
        )

    c_elem = offs_m[:, None] * SIZE + offs_n[None, :]
    tl.store(c_ptr + 2 * c_elem, prod_r - prod_i)
    tl.store(c_ptr + 2 * c_elem + 1, prod_sum - prod_r - prod_i)


def _get_cgemm_workspace(A: torch.Tensor, m: int, n: int, k: int):
    key = (A.device, m, n, k)
    if _CGEMM_WORKSPACE["key"] != key:
        _CGEMM_WORKSPACE["key"] = key
        _CGEMM_WORKSPACE["buffers"] = (
            torch.empty((m, k), device=A.device, dtype=torch.float32),
            torch.empty((m, k), device=A.device, dtype=torch.float32),
            torch.empty((m, k), device=A.device, dtype=torch.float32),
            torch.empty((k, n), device=A.device, dtype=torch.float32),
            torch.empty((k, n), device=A.device, dtype=torch.float32),
            torch.empty((k, n), device=A.device, dtype=torch.float32),
            torch.empty((m, n), device=A.device, dtype=torch.float32),
            torch.empty((m, n), device=A.device, dtype=torch.float32),
            torch.empty((m, n), device=A.device, dtype=torch.float32),
        )
    return _CGEMM_WORKSPACE["buffers"]


def _get_cgemm_aug_workspace(A: torch.Tensor, size: int):
    key = (A.device, size)
    if _CGEMM_AUG_WORKSPACE["key"] != key:
        _CGEMM_AUG_WORKSPACE["key"] = key
        _CGEMM_AUG_WORKSPACE["buffers"] = (
            torch.empty((size, 2 * size), device=A.device, dtype=torch.float32),
            torch.empty((2 * size, 2 * size), device=A.device, dtype=torch.float32),
        )
        _CGEMM_AUG_WORKSPACE["pack_key"] = None
    return _CGEMM_AUG_WORKSPACE["buffers"]


def _tensor_version(t: torch.Tensor):
    return getattr(t, "_version", None)


def _launch_cgemm_aug_sgemm(
    transa: int,
    transb: int,
    size: int,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> None:
    a_aug, b_aug = _get_cgemm_aug_workspace(A, size)
    pack_key = (
        A.data_ptr(),
        _tensor_version(A),
        B.data_ptr(),
        _tensor_version(B),
        transa,
        transb,
        lda,
        ldb,
        size,
    )
    if _CGEMM_AUG_WORKSPACE["pack_key"] != pack_key:
        A_real = torch.view_as_real(A).reshape(-1)
        B_real = torch.view_as_real(B).reshape(-1)
        total_a = size * 2 * size
        total_b = 2 * size * 2 * size
        block = 256
        grid = (triton.cdiv(max(total_a, total_b), block),)
        _cgemm_aug_pack_kernel[grid](
            A_real,
            B_real,
            a_aug,
            b_aug,
            total_a,
            total_b,
            size,
            lda,
            ldb,
            transa,
            transb,
            BLOCK=block,
        )
        _CGEMM_AUG_WORKSPACE["pack_key"] = pack_key

    C_real = torch.view_as_real(C).reshape(-1)
    _sgemm_iluvatar(
        0,
        0,
        size,
        2 * size,
        2 * size,
        1.0,
        a_aug,
        2 * size,
        b_aug,
        2 * size,
        0.0,
        C_real,
        2 * ldc,
    )


def _try_cgemm_aug_sgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha_is_one: bool,
    beta_is_zero: bool,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    if not (alpha_is_one and beta_is_zero):
        return False
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if not (m == n == k and m in (128, 256)):
        return False
    if not (lda == m and ldb == n and ldc == n):
        return False
    _launch_cgemm_aug_sgemm(transa, transb, m, A, lda, B, ldb, C, ldc)
    return True


def _use_cgemm_tiled_pack(transa: int, transb: int, m: int, n: int, k: int) -> bool:
    if not (m == n == k):
        return False
    if transa == 0 and transb == 0:
        return False
    if m == 512:
        return True
    if m == 1024 and transa == 1 and transb == 1:
        return True
    if m == 1536 and transa == 1:
        return True
    return False


def _select_cgemm_pack_trans_tile(transa: int, transb: int, size: int):
    if size == 512:
        return 32, 16
    if size == 1536 and transa == 1 and transb == 0:
        return 16, 16
    if transa == 1 and transb == 1:
        return 16, 16
    return 32, 16


def _select_cgemm_pack_block(transa: int, transb: int, size: int) -> int:
    if size == 1536 and transa == 1 and transb == 1:
        return 128
    return 256


def _split_cgemm_pack_operand(
    src: torch.Tensor,
    dst_r: torch.Tensor,
    dst_i: torch.Tensor,
    dst_sum: torch.Tensor,
    rows: int,
    cols: int,
    ld: int,
    trans: int,
    pack_block: int,
    tile_r: int,
    tile_c: int,
) -> None:
    if trans == 0:
        grid = (triton.cdiv(rows * cols, pack_block),)
        _cgemm_split_sum_op_kernel[grid](
            src,
            dst_r,
            dst_i,
            dst_sum,
            rows * cols,
            cols,
            ld,
            0,
            BLOCK=pack_block,
        )
    else:
        grid = (triton.cdiv(rows, tile_r), triton.cdiv(cols, tile_c))
        _cgemm_split_sum_trans_tile_kernel[grid](
            src,
            dst_r,
            dst_i,
            dst_sum,
            rows,
            cols,
            ld,
            BLOCK_R=tile_r,
            BLOCK_C=tile_c,
            num_warps=4,
            num_stages=3,
        )


def _launch_cgemm_pack_sgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
) -> None:
    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_cgemm_workspace(A, m, n, k)
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    pack_block = _select_cgemm_pack_block(transa, transb, m)
    merge_grid = (triton.cdiv(m * n, pack_block),)

    if _use_cgemm_tiled_pack(transa, transb, m, n, k):
        tile_r, tile_c = _select_cgemm_pack_trans_tile(transa, transb, m)
        _split_cgemm_pack_operand(
            A_real, Ar, Ai, As, m, k, lda, transa, pack_block, tile_r, tile_c
        )
        _split_cgemm_pack_operand(
            B_real, Br, Bi, Bs, k, n, ldb, transb, pack_block, tile_r, tile_c
        )
    else:
        split_grid = (triton.cdiv(max(m * k, k * n), pack_block),)
        _cgemm_split_sum2_op_kernel[split_grid](
            A_real,
            B_real,
            Ar,
            Ai,
            As,
            Br,
            Bi,
            Bs,
            m * k,
            k * n,
            k,
            n,
            lda,
            ldb,
            transa,
            transb,
            BLOCK=pack_block,
        )

    _sgemm_iluvatar(0, 0, m, n, k, 1.0, Ar, k, Br, n, 0.0, prod_r, n)
    _sgemm_iluvatar(0, 0, m, n, k, 1.0, Ai, k, Bi, n, 0.0, prod_i, n)
    _sgemm_iluvatar(0, 0, m, n, k, 1.0, As, k, Bs, n, 0.0, prod_sum, n)
    _cgemm_merge_3m_kernel[merge_grid](
        C_real, prod_r, prod_i, prod_sum, m * n, BLOCK=pack_block
    )


def _try_cgemm_pack_sgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha_is_one: bool,
    beta_is_zero: bool,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    if not (alpha_is_one and beta_is_zero):
        return False
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if ldc != n:
        return False
    if max(m, n, k) < 511:
        if not (m == n == k == 256 and transa == 0 and transb == 0):
            return False
    _launch_cgemm_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C)
    return True


def _select_iluvatar_cgemm_3m_config(transa: int, transb: int, size: int):
    if size == 64:
        if transa == 1 and transb == 0:
            return 16, 16, 32, 8, 4, 3
        if transa == 0 and transb == 1:
            return 16, 16, 64, 8, 1, 3
        if transa == 1 and transb == 1:
            return 16, 16, 64, 8, 1, 3
        return 16, 16, 32, 8, 4, 3
    if size == 128:
        if transa == 0 and transb == 0:
            return 32, 16, 32, 8, 4, 3
        if transa == 1 and transb == 0:
            return 32, 16, 32, 8, 2, 3
        if transa == 0 and transb == 1:
            return 16, 16, 64, 4, 2, 3
        return 32, 32, 64, 8, 1, 3
    if size == 256:
        if transa == 1 and transb == 0:
            return 64, 32, 32, 8, 4, 3
        if transa == 0 and transb == 1:
            return 32, 64, 32, 8, 2, 3
        if transa == 1 and transb == 1:
            return 32, 32, 32, 4, 1, 3
        return 32, 64, 32, 8, 2, 3
    if size == 512 and transa == 1 and transb == 1:
        return 32, 64, 32, 4, 4, 3
    return None


def _try_cgemm_3m_nomask(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha_is_one: bool,
    beta_is_zero: bool,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    if not (alpha_is_one and beta_is_zero):
        return False
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if not (m == n == k and lda == m and ldb == n and ldc == n):
        return False
    config = _select_iluvatar_cgemm_3m_config(transa, transb, m)
    if config is None:
        return False

    block_m, block_n, block_k, num_warps, group_m, num_stages = config
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _cgemm_3m_nomask_kernel[grid](
        A_real,
        B_real,
        C_real,
        m,
        transa,
        transb,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return True


def _select_iluvatar_cgemm_config(transa: int, transb: int, m: int, n: int, k: int):
    if m == n == k:
        if m <= 64:
            if transa == 1:
                return 32, 16, 32, 8, 2
            return 16, 16, 32, 8, 2
        if m == 128 and transa == 1:
            return 32, 16, 32, 8, 2
        if m == 256:
            if transa == 0 and transb in (0, 1):
                return 32, 64, 32, 8, 4
            if transa == 1 and transb == 0:
                return 64, 32, 32, 8, 4

    max_dim = max(m, n, k)
    if max_dim <= 128:
        return 16, 16, 32, 4, 1
    if max_dim <= 384:
        return 32, 32, 32, 4, 1
    return 32, 64, 32, 4, 4


def cgemm(
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
    _validate_cgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha_r, alpha_i = _complex_scalar_parts(alpha)
    beta_r, beta_i = _complex_scalar_parts(beta)

    if m == 0 or n == 0 or k == 0 or (alpha_r == 0.0 and alpha_i == 0.0):
        if beta_r == 0.0 and beta_i == 0.0:
            C.zero_()
        elif not (beta_r == 1.0 and beta_i == 0.0):
            C.mul_(complex(beta_r, beta_i))
        return

    beta_is_zero = beta_r == 0.0 and beta_i == 0.0
    alpha_is_one = alpha_r == 1.0 and alpha_i == 0.0

    with torch_device_fn.device(A.device):
        if _try_cgemm_aug_sgemm(
            transa,
            transb,
            m,
            n,
            k,
            alpha_is_one,
            beta_is_zero,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
        ):
            return
        if _try_cgemm_pack_sgemm(
            transa,
            transb,
            m,
            n,
            k,
            alpha_is_one,
            beta_is_zero,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
        ):
            return
        if _try_cgemm_3m_nomask(
            transa,
            transb,
            m,
            n,
            k,
            alpha_is_one,
            beta_is_zero,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
        ):
            return

    block_m, block_n, block_k, num_warps, group_m = _select_iluvatar_cgemm_config(
        transa, transb, m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    with torch_device_fn.device(A.device):
        _cgemm_dot_kernel[grid](
            A_real,
            B_real,
            C_real,
            alpha_r,
            alpha_i,
            beta_r,
            beta_i,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            transa,
            transb,
            beta_is_zero,
            alpha_is_one,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
        )

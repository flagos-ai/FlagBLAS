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

from flag_blas.ops.level3.cgemm import (
    ScalarType,
    _cgemm_dot_kernel,
    _complex_scalar_parts,
    _validate_cgemm_args,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._nvidia.hopper.ops.gemm import sgemm as _sgemm_hopper
from flag_blas.runtime.dispatch import StaticDispatch

logger = logging.getLogger(__name__)

_CGEMM_WORKSPACE = {"key": None, "buffers": None}
_CGEMM_STREAMS = {"key": None, "streams": None}
_CGEMM_GRAPH_PACK = {"key": None, "graph": None}
_CGEMM_GRAPH_PAD_PACK = {"key": None, "graph": None}


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
def _cgemm_3m_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    lda: tl.constexpr,
    ldb: tl.constexpr,
    ldc: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    prod_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < k
        if TRANS_A == 0:
            a_elem = offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_elem = cur_k[None, :] * lda + offs_m[:, None]

        if TRANS_B == 0:
            b_elem = cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_elem = offs_n[None, :] * ldb + cur_k[:, None]

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        ar = tl.load(a_ptr + 2 * a_elem, mask=a_mask, other=0.0)
        ai = tl.load(a_ptr + 2 * a_elem + 1, mask=a_mask, other=0.0)
        br = tl.load(b_ptr + 2 * b_elem, mask=b_mask, other=0.0)
        bi = tl.load(b_ptr + 2 * b_elem + 1, mask=b_mask, other=0.0)

        prod_r += tl.dot(ar, br, out_dtype=tl.float32, input_precision="tf32x3")
        prod_i += tl.dot(ai, bi, out_dtype=tl.float32, input_precision="tf32x3")
        prod_sum += tl.dot(
            ar + ai,
            br + bi,
            out_dtype=tl.float32,
            input_precision="tf32x3",
        )

    c_elem = offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptr + 2 * c_elem, prod_r - prod_i, mask=c_mask)
    tl.store(c_ptr + 2 * c_elem + 1, prod_sum - prod_r - prod_i, mask=c_mask)


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
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    prod_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, SIZE, BLOCK_K):
        cur_k = k_start + offs_k
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


@triton.jit
def _cgemm_3m_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < SIZE
    mask_n = offs_n < SIZE

    prod_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prod_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, SIZE, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < SIZE
        a_elem = cur_k[:, None] * SIZE + offs_m[None, :]
        b_elem = cur_k[:, None] * SIZE + offs_n[None, :]
        if HAS_MASK:
            a_mask = mask_k[:, None] & mask_m[None, :]
            b_mask = mask_k[:, None] & mask_n[None, :]
            ar_t = tl.load(a_ptr + 2 * a_elem, mask=a_mask, other=0.0)
            ai_t = tl.load(a_ptr + 2 * a_elem + 1, mask=a_mask, other=0.0)
            br = tl.load(b_ptr + 2 * b_elem, mask=b_mask, other=0.0)
            bi = tl.load(b_ptr + 2 * b_elem + 1, mask=b_mask, other=0.0)
        else:
            ar_t = tl.load(a_ptr + 2 * a_elem)
            ai_t = tl.load(a_ptr + 2 * a_elem + 1)
            br = tl.load(b_ptr + 2 * b_elem)
            bi = tl.load(b_ptr + 2 * b_elem + 1)

        ar = tl.trans(ar_t)
        ai = tl.trans(ai_t)
        prod_r += tl.dot(ar, br, out_dtype=tl.float32, input_precision="tf32x3")
        prod_i += tl.dot(ai, bi, out_dtype=tl.float32, input_precision="tf32x3")
        prod_sum += tl.dot(
            ar + ai,
            br + bi,
            out_dtype=tl.float32,
            input_precision="tf32x3",
        )

    c_elem = offs_m[:, None] * SIZE + offs_n[None, :]
    if HAS_MASK:
        c_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptr + 2 * c_elem, prod_r - prod_i, mask=c_mask)
        tl.store(c_ptr + 2 * c_elem + 1, prod_sum - prod_r - prod_i, mask=c_mask)
    else:
        tl.store(c_ptr + 2 * c_elem, prod_r - prod_i)
        tl.store(c_ptr + 2 * c_elem + 1, prod_sum - prod_r - prod_i)


@triton.jit
def _cgemm_split_sum2_pad_op_kernel(
    src_a,
    src_b,
    dst_ar,
    dst_ai,
    dst_as,
    dst_br,
    dst_bi,
    dst_bs,
    src_size: tl.constexpr,
    pad_size: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    row = offsets // pad_size
    col = offsets - row * pad_size
    mask = (row < src_size) & (col < src_size)

    if TRANS_A == 0:
        a_src = row * src_size + col
    else:
        a_src = col * src_size + row
    if TRANS_B == 0:
        b_src = row * src_size + col
    else:
        b_src = col * src_size + row

    ar = tl.load(src_a + 2 * a_src, mask=mask, other=0.0)
    ai = tl.load(src_a + 2 * a_src + 1, mask=mask, other=0.0)
    br = tl.load(src_b + 2 * b_src, mask=mask, other=0.0)
    bi = tl.load(src_b + 2 * b_src + 1, mask=mask, other=0.0)
    tl.store(dst_ar + offsets, ar)
    tl.store(dst_ai + offsets, ai)
    tl.store(dst_as + offsets, ar + ai)
    tl.store(dst_br + offsets, br)
    tl.store(dst_bi + offsets, bi)
    tl.store(dst_bs + offsets, br + bi)


@triton.jit
def _cgemm_merge_pad_kernel(
    dst,
    prod_r,
    prod_i,
    prod_sum,
    src_size: tl.constexpr,
    pad_size: tl.constexpr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    dst_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = dst_offsets < total
    row = dst_offsets // src_size
    col = dst_offsets - row * src_size
    src_offsets = row * pad_size + col
    real_prod = tl.load(prod_r + src_offsets, mask=mask, other=0.0)
    imag_prod = tl.load(prod_i + src_offsets, mask=mask, other=0.0)
    sum_prod = tl.load(prod_sum + src_offsets, mask=mask, other=0.0)
    tl.store(dst + 2 * dst_offsets, real_prod - imag_prod, mask=mask)
    tl.store(dst + 2 * dst_offsets + 1, sum_prod - real_prod - imag_prod, mask=mask)


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


def _get_cgemm_streams(device):
    if _CGEMM_STREAMS["key"] != device:
        _CGEMM_STREAMS["key"] = device
        _CGEMM_STREAMS["streams"] = tuple(
            torch.cuda.Stream(device=device) for _ in range(3)
        )
    return _CGEMM_STREAMS["streams"]


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

    split_grid_a = (triton.cdiv(m * k, 1024),)
    split_grid_b = (triton.cdiv(k * n, 1024),)
    merge_grid = (triton.cdiv(m * n, 1024),)

    _cgemm_split_sum_op_kernel[split_grid_a](
        A_real, Ar, Ai, As, m * k, k, lda, transa, BLOCK=1024
    )
    _cgemm_split_sum_op_kernel[split_grid_b](
        B_real, Br, Bi, Bs, k * n, n, ldb, transb, BLOCK=1024
    )

    current_stream = torch.cuda.current_stream(device=A.device)
    stream_r, stream_i, stream_sum = _get_cgemm_streams(A.device)
    stream_r.wait_stream(current_stream)
    stream_i.wait_stream(current_stream)
    stream_sum.wait_stream(current_stream)

    with torch.cuda.stream(stream_r):
        _sgemm_hopper(0, 0, m, n, k, 1.0, Ar, k, Br, n, 0.0, prod_r, n)
    with torch.cuda.stream(stream_i):
        _sgemm_hopper(0, 0, m, n, k, 1.0, Ai, k, Bi, n, 0.0, prod_i, n)
    with torch.cuda.stream(stream_sum):
        _sgemm_hopper(0, 0, m, n, k, 1.0, As, k, Bs, n, 0.0, prod_sum, n)

    current_stream.wait_stream(stream_r)
    current_stream.wait_stream(stream_i)
    current_stream.wait_stream(stream_sum)
    _cgemm_merge_3m_kernel[merge_grid](
        C_real, prod_r, prod_i, prod_sum, m * n, BLOCK=1024
    )


def _try_cgemm_pack_sgemm(
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
    ldc: int,
) -> bool:
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if ldc != n or max(m, n, k) < 256:
        return False
    _launch_cgemm_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C)
    return True


def _try_cgemm_pack_sgemm_graph(
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
    ldc: int,
) -> bool:
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if not (m == n == k and lda == m and ldb == n and ldc == n):
        return False
    if m not in (1023, 1024):
        return False

    key = (transa, transb, m, A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _CGEMM_GRAPH_PACK["key"] != key:
        _launch_cgemm_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_cgemm_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_cgemm_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C)

        _CGEMM_GRAPH_PACK["key"] = key
        _CGEMM_GRAPH_PACK["graph"] = graph

    _CGEMM_GRAPH_PACK["graph"].replay()
    return True


def _select_cgemm_3m_config(transa: int, transb: int, m: int):
    if m == 256:
        if transa == 0 and transb == 0:
            return 8, 64, 16, 4, 8, 4, 96
        if transa == 1 and transb == 0:
            return 32, 16, 32, 4, 1, 3, None
        if transb == 1:
            return 16, 32, 64, 4, 1, 3, None
        return 16, 32, 32, 4, 1, 3, None
    if m == 511:
        return 16, 64, 32, 4, 1, 3, None
    if m == 512:
        if transa == 0 and transb == 0:
            return 16, 32, 16, 4, 2, 3, None
        if transa == 1 and transb == 0:
            return 64, 16, 32, 4, 1, 3, None
        return 16, 64, 32, 4, 1, 3, None
    return 32, 64, 32, 4, 2, 3, None


def _try_cgemm_3m(
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
    ldc: int,
) -> bool:
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if not (m == n == k and lda == m and ldb == n and ldc == n):
        return False
    if m not in (256, 511, 512):
        return False

    (
        block_m,
        block_n,
        block_k,
        num_warps,
        group_m,
        num_stages,
        maxnreg,
    ) = _select_cgemm_3m_config(transa, transb, m)
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    launch_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    if m in (256, 512) and m % block_m == 0 and m % block_n == 0 and m % block_k == 0:
        _cgemm_3m_nomask_kernel[grid](
            A_real,
            B_real,
            C_real,
            m,
            transa,
            transb,
            **launch_kwargs,
        )
    else:
        _cgemm_3m_kernel[grid](
            A_real,
            B_real,
            C_real,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            transa,
            transb,
            **launch_kwargs,
        )
    return True


def _launch_cgemm_pad_pack_sgemm(
    src_size: int,
    pad_size: int,
    transa: int,
    transb: int,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> None:
    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_cgemm_workspace(
        A, pad_size, pad_size, pad_size
    )
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    split_grid = (triton.cdiv(pad_size * pad_size, 1024),)
    merge_grid = (triton.cdiv(src_size * src_size, 1024),)

    _cgemm_split_sum2_pad_op_kernel[split_grid](
        A_real,
        B_real,
        Ar,
        Ai,
        As,
        Br,
        Bi,
        Bs,
        src_size,
        pad_size,
        transa,
        transb,
        BLOCK=1024,
    )

    current_stream = torch.cuda.current_stream(device=A.device)
    stream_r, stream_i, stream_sum = _get_cgemm_streams(A.device)
    stream_r.wait_stream(current_stream)
    stream_i.wait_stream(current_stream)
    stream_sum.wait_stream(current_stream)

    with torch.cuda.stream(stream_r):
        _sgemm_hopper(
            0,
            0,
            pad_size,
            pad_size,
            pad_size,
            1.0,
            Ar,
            pad_size,
            Br,
            pad_size,
            0.0,
            prod_r,
            pad_size,
        )
    with torch.cuda.stream(stream_i):
        _sgemm_hopper(
            0,
            0,
            pad_size,
            pad_size,
            pad_size,
            1.0,
            Ai,
            pad_size,
            Bi,
            pad_size,
            0.0,
            prod_i,
            pad_size,
        )
    with torch.cuda.stream(stream_sum):
        _sgemm_hopper(
            0,
            0,
            pad_size,
            pad_size,
            pad_size,
            1.0,
            As,
            pad_size,
            Bs,
            pad_size,
            0.0,
            prod_sum,
            pad_size,
        )

    current_stream.wait_stream(stream_r)
    current_stream.wait_stream(stream_i)
    current_stream.wait_stream(stream_sum)
    _cgemm_merge_pad_kernel[merge_grid](
        C_real,
        prod_r,
        prod_i,
        prod_sum,
        src_size,
        pad_size,
        src_size * src_size,
        BLOCK=1024,
    )


def _try_cgemm_pad_pack_sgemm_graph(
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
    ldc: int,
) -> bool:
    if transa not in (0, 1) or transb not in (0, 1):
        return False
    if not (m == n == k and lda == m and ldb == n and ldc == n):
        return False
    if m not in (511, 512, 1023):
        return False
    if m != 1023 and transb != 0:
        return False

    pad_size = 1024 if m == 1023 else 512
    key = (transa, transb, m, A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _CGEMM_GRAPH_PAD_PACK["key"] != key:
        _launch_cgemm_pad_pack_sgemm(m, pad_size, transa, transb, A, B, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_cgemm_pad_pack_sgemm(m, pad_size, transa, transb, A, B, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_cgemm_pad_pack_sgemm(m, pad_size, transa, transb, A, B, C)

        _CGEMM_GRAPH_PAD_PACK["key"] = key
        _CGEMM_GRAPH_PAD_PACK["graph"] = graph

    _CGEMM_GRAPH_PAD_PACK["graph"].replay()
    return True


def _select_cgemm_hopper_config(transa: int, transb: int, m: int, n: int, k: int):
    max_dim = max(m, n, k)
    min_dim = min(m, n, k)

    if max_dim <= 32:
        return 16, 16, 16, 4, 1, None
    if max_dim <= 128:
        return 16, 32, 32, 4, 4, None
    if max_dim <= 512:
        if transa == 0 and transb == 0:
            return 32, 64, 32, 4, 4, None
        return 64, 32, 32, 4, 4, None
    if max_dim <= 1024:
        if transa == 0 and transb == 0:
            return 32, 64, 32, 4, 8, None
        if transa == 0:
            return 32, 64, 32, 4, 8, None
        return 64, 32, 32, 4, 8, None

    if transa == 0 and transb == 0:
        if min_dim >= 4096:
            return 64, 64, 32, 4, 4, None
        return 32, 64, 32, 4, 8, None

    if transa == 0:
        if min_dim >= 4096:
            return 64, 64, 32, 4, 4, None
        return 32, 64, 32, 4, 8, None

    if transb == 0:
        if min_dim >= 4096:
            return 64, 64, 32, 4, 4, None
        return 64, 32, 32, 4, 8, None

    if min_dim >= 4096:
        return 64, 64, 32, 4, 4, None
    return 64, 32, 32, 4, 8, None


# ---------------------------------------------------------------------------
# Module-level condition predicates for cgemm StaticDispatch
# ---------------------------------------------------------------------------


def _cgemm_can_3m_preferred(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        alpha_is_one
        and beta_is_zero
        and transa in (0, 1)
        and transb in (0, 1)
        and m == n == k
        and lda == m
        and ldb == n
        and ldc == n
        and m in (256, 511, 512)
    )


def _cgemm_can_pad_pack_sgemm_graph(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        alpha_is_one
        and beta_is_zero
        and transa in (0, 1)
        and transb in (0, 1)
        and m == n == k
        and lda == m
        and ldb == n
        and ldc == n
        and (m == 1023 or (m in (511, 512) and transb == 0))
    )


def _cgemm_can_pack_sgemm_graph(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        alpha_is_one
        and beta_is_zero
        and transa in (0, 1)
        and transb in (0, 1)
        and m == n == k
        and lda == m
        and ldb == n
        and ldc == n
        and m in (1023, 1024)
    )


def _cgemm_can_pack_sgemm(
    m, n, k, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        alpha_is_one
        and beta_is_zero
        and transa in (0, 1)
        and transb in (0, 1)
        and ldc == n
        and max(m, n, k) >= 256
    )


def _cgemm_is_default(**_kw):
    return True


# ---------------------------------------------------------------------------
# Module-level factory functions for cgemm StaticDispatch
# ---------------------------------------------------------------------------


def _cgemm_build_3m(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_cgemm_3m(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)


def _cgemm_build_pad_pack_sgemm_graph(
    transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw
):
    return lambda: _try_cgemm_pad_pack_sgemm_graph(
        transa, transb, m, n, k, A, lda, B, ldb, C, ldc
    )


def _cgemm_build_pack_sgemm_graph(
    transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw
):
    return lambda: _try_cgemm_pack_sgemm_graph(
        transa, transb, m, n, k, A, lda, B, ldb, C, ldc
    )


def _cgemm_build_pack_sgemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_cgemm_pack_sgemm(
        transa, transb, m, n, k, A, lda, B, ldb, C, ldc
    )


def _cgemm_build_dot_kernel(
    transa,
    transb,
    m,
    n,
    k,
    alpha_r,
    alpha_i,
    beta_r,
    beta_i,
    A,
    lda,
    B,
    ldb,
    C,
    ldc,
    beta_is_zero,
    alpha_is_one,
):
    (
        block_m,
        block_n,
        block_k,
        num_warps,
        group_m,
        maxnreg,
    ) = _select_cgemm_hopper_config(transa, transb, m, n, k)
    launch_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    return lambda: _cgemm_dot_kernel[grid](
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
        **launch_kwargs,
    )


_CGEMM_DISPATCH = StaticDispatch(
    [
        (_cgemm_can_pad_pack_sgemm_graph, _cgemm_build_pad_pack_sgemm_graph),
        (_cgemm_can_3m_preferred, _cgemm_build_3m),
        (_cgemm_can_pack_sgemm_graph, _cgemm_build_pack_sgemm_graph),
        (_cgemm_can_pack_sgemm, _cgemm_build_pack_sgemm),
        (_cgemm_is_default, _cgemm_build_dot_kernel),
    ]
)


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

    context = dict(
        transa=transa,
        transb=transb,
        m=m,
        n=n,
        k=k,
        alpha_r=alpha_r,
        alpha_i=alpha_i,
        beta_r=beta_r,
        beta_i=beta_i,
        A=A,
        lda=lda,
        B=B,
        ldb=ldb,
        C=C,
        ldc=ldc,
        beta_is_zero=beta_is_zero,
        alpha_is_one=alpha_is_one,
    )
    with torch_device_fn.device(A.device):
        runner = _CGEMM_DISPATCH.lookup_and_build(
            m,
            n,
            k,
            aligned=True,
            context=context,
            transa=transa,
            transb=transb,
            A=A,
            B=B,
            C=C,
            lda=lda,
            ldb=ldb,
            ldc=ldc,
            alpha_is_one=alpha_is_one,
            beta_is_zero=beta_is_zero,
        )
        runner()

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

from flag_blas.ops.level3.zgemm import (
    ScalarType,
    _complex_scalar_parts,
    _validate_zgemm_args,
    _zgemm_dot_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._nvidia.hopper.ops.dgemm import (
    _dgemm_dot_kernel,
    _select_dgemm_dot_config,
)
from flag_blas.runtime.dispatch import StaticDispatch

logger = logging.getLogger(__name__)


_ZGEMM_NN_WORKSPACE = {"key": None, "buffers": None}
_ZGEMM_NN_GRAPH_256 = {"key": None, "graph": None}
_ZGEMM_TRANS_GRAPH_256 = {"key": None, "graph": None}
_ZGEMM_NN_GRAPH_512 = {"key": None, "graph": None}
_ZGEMM_NN_GRAPH_PAD = {"key": None, "graph": None}
_ZGEMM_TRANS_GRAPH_PAD = {"key": None, "graph": None}
_ZGEMM_TRANS_GRAPH_PACK = {"key": None, "graph": None}
_ZGEMM_TRANS_GRAPH_SPLIT = {"key": None, "graph": None}
_ZGEMM_NN_STREAMS_512 = {"key": None, "streams": None}


@triton.jit
def _zgemm_split_sum_kernel(src, dst_r, dst_i, dst_sum, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    real = tl.load(src + 2 * offsets, mask=mask, other=0.0)
    imag = tl.load(src + 2 * offsets + 1, mask=mask, other=0.0)
    tl.store(dst_r + offsets, real, mask=mask)
    tl.store(dst_i + offsets, imag, mask=mask)
    tl.store(dst_sum + offsets, real + imag, mask=mask)


@triton.jit
def _zgemm_split_sum_op_kernel(
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
def _zgemm_merge_3m_kernel(dst, prod_r, prod_i, prod_sum, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    real_prod = tl.load(prod_r + offsets, mask=mask, other=0.0)
    imag_prod = tl.load(prod_i + offsets, mask=mask, other=0.0)
    sum_prod = tl.load(prod_sum + offsets, mask=mask, other=0.0)
    tl.store(dst + 2 * offsets, real_prod - imag_prod, mask=mask)
    tl.store(dst + 2 * offsets + 1, sum_prod - real_prod - imag_prod, mask=mask)


@triton.jit
def _zgemm_nn_128_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(128, BLOCK_M)
    grid_n = tl.cdiv(128, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k_start in range(0, 128, BLOCK_K):
        cur_k = k_start + offs_k
        a_elem = offs_m[:, None] * 128 + cur_k[None, :]
        b_elem = cur_k[:, None] * 128 + offs_n[None, :]
        ar = tl.load(a_ptr + 2 * a_elem)
        ai = tl.load(a_ptr + 2 * a_elem + 1)
        br = tl.load(b_ptr + 2 * b_elem)
        bi = tl.load(b_ptr + 2 * b_elem + 1)

        acc_r += tl.dot(ar, br, out_dtype=tl.float64, allow_tf32=False) - tl.dot(
            ai, bi, out_dtype=tl.float64, allow_tf32=False
        )
        acc_i += tl.dot(ar, bi, out_dtype=tl.float64, allow_tf32=False) + tl.dot(
            ai, br, out_dtype=tl.float64, allow_tf32=False
        )

    c_elem = offs_m[:, None] * 128 + offs_n[None, :]
    tl.store(c_ptr + 2 * c_elem, acc_r)
    tl.store(c_ptr + 2 * c_elem + 1, acc_i)


def _try_zgemm_nn_128(
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
    if m != 128 or n != 128 or k != 128 or lda != 128 or ldb != 128 or ldc != 128:
        return False

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(128, 8) * triton.cdiv(128, 16),)
    _zgemm_nn_128_kernel[grid](
        A_real,
        B_real,
        C_real,
        BLOCK_M=8,
        BLOCK_N=16,
        BLOCK_K=16,
        GROUP_M=1,
        num_warps=2,
        num_stages=4,
    )
    return True


@triton.jit
def _zgemm_128_nomask_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(128, BLOCK_M)
    grid_n = tl.cdiv(128, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k_start in range(0, 128, BLOCK_K):
        cur_k = k_start + offs_k
        if TRANS_A == 0:
            a_elem = offs_m[:, None] * 128 + cur_k[None, :]
        else:
            a_elem = cur_k[None, :] * 128 + offs_m[:, None]
        if TRANS_B == 0:
            b_elem = cur_k[:, None] * 128 + offs_n[None, :]
        else:
            b_elem = offs_n[None, :] * 128 + cur_k[:, None]
        ar = tl.load(a_ptr + 2 * a_elem)
        ai = tl.load(a_ptr + 2 * a_elem + 1)
        br = tl.load(b_ptr + 2 * b_elem)
        bi = tl.load(b_ptr + 2 * b_elem + 1)

        acc_r += tl.dot(ar, br, out_dtype=tl.float64, allow_tf32=False) - tl.dot(
            ai, bi, out_dtype=tl.float64, allow_tf32=False
        )
        acc_i += tl.dot(ar, bi, out_dtype=tl.float64, allow_tf32=False) + tl.dot(
            ai, br, out_dtype=tl.float64, allow_tf32=False
        )

    c_elem = offs_m[:, None] * 128 + offs_n[None, :]
    tl.store(c_ptr + 2 * c_elem, acc_r)
    tl.store(c_ptr + 2 * c_elem + 1, acc_i)


def _try_zgemm_trans_128(
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
    if transa == 0 and transb == 0:
        return False
    if transa == 2 or transb == 2:
        return False
    if m != 128 or n != 128 or k != 128 or lda != 128 or ldb != 128 or ldc != 128:
        return False

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(128, 8) * triton.cdiv(128, 16),)
    _zgemm_128_nomask_kernel[grid](
        A_real,
        B_real,
        C_real,
        TRANS_A=transa,
        TRANS_B=transb,
        BLOCK_M=8,
        BLOCK_N=16,
        BLOCK_K=16,
        GROUP_M=1,
        num_warps=2,
        num_stages=4,
    )
    return True


@triton.jit
def _zgemm_256_mma884_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_m = pid // 32
    tile_n = pid - tile_m * 32

    lane_vec = tl.arange(0, 32)
    dummy = lane_vec.to(tl.int32)
    lane = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %laneid;",
        constraints="=r,r",
        args=[dummy],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    row = lane // 4
    tid = lane % 4
    m = tile_m * 8 + row
    n0 = tile_n * 8 + tid * 2
    n1 = n0 + 1

    acc_p1_0 = tl.full((32,), 0.0, dtype=tl.float64)
    acc_p1_1 = tl.full((32,), 0.0, dtype=tl.float64)
    acc_p2_0 = tl.full((32,), 0.0, dtype=tl.float64)
    acc_p2_1 = tl.full((32,), 0.0, dtype=tl.float64)
    acc_p3_0 = tl.full((32,), 0.0, dtype=tl.float64)
    acc_p3_1 = tl.full((32,), 0.0, dtype=tl.float64)

    for k_start in range(0, 256, 4):
        cur_k = k_start + tid
        if TRANS_A == 0:
            a_elem = m * 256 + cur_k
        else:
            a_elem = cur_k * 256 + m
        if TRANS_B == 0:
            b_elem = cur_k * 256 + (tile_n * 8 + row)
        else:
            b_elem = (tile_n * 8 + row) * 256 + cur_k
        ar = tl.load(a_ptr + 2 * a_elem)
        ai = tl.load(a_ptr + 2 * a_elem + 1)
        br = tl.load(b_ptr + 2 * b_elem)
        bi = tl.load(b_ptr + 2 * b_elem + 1)
        a_sum = ar + ai
        b_sum = br + bi

        acc_p1_0, acc_p1_1 = tl.inline_asm_elementwise(
            asm="mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            constraints="=d,=d,d,d,d,d",
            args=[ar, br, acc_p1_0, acc_p1_1],
            dtype=(tl.float64, tl.float64),
            is_pure=True,
            pack=1,
        )
        acc_p2_0, acc_p2_1 = tl.inline_asm_elementwise(
            asm="mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            constraints="=d,=d,d,d,d,d",
            args=[ai, bi, acc_p2_0, acc_p2_1],
            dtype=(tl.float64, tl.float64),
            is_pure=True,
            pack=1,
        )
        acc_p3_0, acc_p3_1 = tl.inline_asm_elementwise(
            asm="mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            constraints="=d,=d,d,d,d,d",
            args=[a_sum, b_sum, acc_p3_0, acc_p3_1],
            dtype=(tl.float64, tl.float64),
            is_pure=True,
            pack=1,
        )

    out_r0 = acc_p1_0 - acc_p2_0
    out_r1 = acc_p1_1 - acc_p2_1
    out_i0 = acc_p3_0 - acc_p1_0 - acc_p2_0
    out_i1 = acc_p3_1 - acc_p1_1 - acc_p2_1
    c_elem0 = m * 256 + n0
    c_elem1 = m * 256 + n1
    tl.store(c_ptr + 2 * c_elem0, out_r0)
    tl.store(c_ptr + 2 * c_elem0 + 1, out_i0)
    tl.store(c_ptr + 2 * c_elem1, out_r1)
    tl.store(c_ptr + 2 * c_elem1 + 1, out_i1)


def _launch_zgemm_256_mma884(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, transa: int, transb: int
) -> None:
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    if transa == 0 and transb == 0:
        num_stages, maxnreg = 4, 80
    elif transa == 0 and transb == 1:
        num_stages, maxnreg = 2, 72
    elif transa == 1 and transb == 0:
        num_stages, maxnreg = 5, 72
    else:
        num_stages, maxnreg = 1, 72
    _zgemm_256_mma884_kernel[(32 * 32,)](
        A_real,
        B_real,
        C_real,
        TRANS_A=transa,
        TRANS_B=transb,
        num_warps=1,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _launch_zgemm_nn_256_mma884(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> None:
    _launch_zgemm_256_mma884(A, B, C, 0, 0)


@triton.jit
def _zgemm_nn_256_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(256, BLOCK_M)
    grid_n = tl.cdiv(256, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k_start in range(0, 256, BLOCK_K):
        cur_k = k_start + offs_k
        a_elem = offs_m[:, None] * 256 + cur_k[None, :]
        b_elem = cur_k[:, None] * 256 + offs_n[None, :]
        ar = tl.load(a_ptr + 2 * a_elem)
        ai = tl.load(a_ptr + 2 * a_elem + 1)
        br = tl.load(b_ptr + 2 * b_elem)
        bi = tl.load(b_ptr + 2 * b_elem + 1)

        acc_r += tl.dot(ar, br, out_dtype=tl.float64, allow_tf32=False) - tl.dot(
            ai, bi, out_dtype=tl.float64, allow_tf32=False
        )
        acc_i += tl.dot(ar, bi, out_dtype=tl.float64, allow_tf32=False) + tl.dot(
            ai, br, out_dtype=tl.float64, allow_tf32=False
        )

    c_elem = offs_m[:, None] * 256 + offs_n[None, :]
    tl.store(c_ptr + 2 * c_elem, acc_r)
    tl.store(c_ptr + 2 * c_elem + 1, acc_i)


def _try_zgemm_nn_256(
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
    if m != 256 or n != 256 or k != 256 or lda != 256 or ldb != 256 or ldc != 256:
        return False

    key = (A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _ZGEMM_NN_GRAPH_256["key"] != key:
        _launch_zgemm_nn_256_mma884(A, B, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_zgemm_nn_256_mma884(A, B, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_zgemm_nn_256_mma884(A, B, C)

        _ZGEMM_NN_GRAPH_256["key"] = key
        _ZGEMM_NN_GRAPH_256["graph"] = graph

    _ZGEMM_NN_GRAPH_256["graph"].replay()
    return True


def _try_zgemm_trans_256(
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
    if transa == 0 and transb == 0:
        return False
    if transa == 2 or transb == 2:
        return False
    if m != 256 or n != 256 or k != 256 or lda != 256 or ldb != 256 or ldc != 256:
        return False

    key = (transa, transb, A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _ZGEMM_TRANS_GRAPH_256["key"] != key:
        _launch_zgemm_256_mma884(A, B, C, transa, transb)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_zgemm_256_mma884(A, B, C, transa, transb)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_zgemm_256_mma884(A, B, C, transa, transb)

        _ZGEMM_TRANS_GRAPH_256["key"] = key
        _ZGEMM_TRANS_GRAPH_256["graph"] = graph

    _ZGEMM_TRANS_GRAPH_256["graph"].replay()
    return True


@triton.jit
def _zgemm_split_sum2_512_kernel(
    src_a,
    src_b,
    dst_ar,
    dst_ai,
    dst_as,
    dst_br,
    dst_bi,
    dst_bs,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    ar = tl.load(src_a + 2 * offsets)
    ai = tl.load(src_a + 2 * offsets + 1)
    br = tl.load(src_b + 2 * offsets)
    bi = tl.load(src_b + 2 * offsets + 1)
    tl.store(dst_ar + offsets, ar)
    tl.store(dst_ai + offsets, ai)
    tl.store(dst_as + offsets, ar + ai)
    tl.store(dst_br + offsets, br)
    tl.store(dst_bi + offsets, bi)
    tl.store(dst_bs + offsets, br + bi)


@triton.jit
def _zgemm_merge_512_kernel(dst, prod_r, prod_i, prod_sum, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    real_prod = tl.load(prod_r + offsets)
    imag_prod = tl.load(prod_i + offsets)
    sum_prod = tl.load(prod_sum + offsets)
    tl.store(dst + 2 * offsets, real_prod - imag_prod)
    tl.store(dst + 2 * offsets + 1, sum_prod - real_prod - imag_prod)


@triton.jit
def _dgemm_nn_512_nomask_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(512, BLOCK_M)
    grid_n = tl.cdiv(512, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k_start in range(0, 512, BLOCK_K):
        cur_k = k_start + offs_k
        if TRANS_A == 0:
            a_ptrs = a_ptr + offs_m[:, None] * 512 + cur_k[None, :]
        else:
            a_ptrs = a_ptr + cur_k[None, :] * 512 + offs_m[:, None]
        if TRANS_B == 0:
            b_ptrs = b_ptr + cur_k[:, None] * 512 + offs_n[None, :]
        else:
            b_ptrs = b_ptr + offs_n[None, :] * 512 + cur_k[:, None]
        a_vals = tl.load(a_ptrs)
        b_vals = tl.load(b_ptrs)
        acc += tl.dot(a_vals, b_vals, out_dtype=tl.float64, allow_tf32=False)

    tl.store(c_ptr + offs_m[:, None] * 512 + offs_n[None, :], acc)


def _launch_dgemm_nn_512_nomask(a, b, c, transa: int = 0, transb: int = 0) -> None:
    if transa == 0 and transb == 0:
        block_m, block_n, block_k, num_warps, group_m = 64, 32, 32, 4, 1
    elif transa == 0 and transb == 1:
        block_m, block_n, block_k, num_warps, group_m = 32, 64, 16, 4, 2
    elif transa == 1 and transb == 0:
        block_m, block_n, block_k, num_warps, group_m = 64, 32, 32, 4, 2
    else:
        block_m, block_n, block_k, num_warps, group_m = 64, 32, 32, 4, 1
    grid = (triton.cdiv(512, block_m) * triton.cdiv(512, block_n),)
    _dgemm_nn_512_nomask_kernel[grid](
        a,
        b,
        c,
        TRANS_A=transa,
        TRANS_B=transb,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=3,
    )


def _get_zgemm_nn_streams_512(device):
    if _ZGEMM_NN_STREAMS_512["key"] != device:
        _ZGEMM_NN_STREAMS_512["key"] = device
        _ZGEMM_NN_STREAMS_512["streams"] = tuple(
            torch.cuda.Stream(device=device) for _ in range(3)
        )
    return _ZGEMM_NN_STREAMS_512["streams"]


def _launch_zgemm_nn_pack_dgemm_512(
    m: int,
    n: int,
    k: int,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> None:
    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_zgemm_nn_workspace(
        A, m, n, k
    )
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(512 * 512, 1024),)

    _zgemm_split_sum2_512_kernel[grid](
        A_real, B_real, Ar, Ai, As, Br, Bi, Bs, BLOCK=1024
    )

    current_stream = torch.cuda.current_stream(device=A.device)
    stream_r, stream_i, stream_sum = _get_zgemm_nn_streams_512(A.device)
    stream_r.wait_stream(current_stream)
    stream_i.wait_stream(current_stream)
    stream_sum.wait_stream(current_stream)

    with torch.cuda.stream(stream_r):
        _launch_dgemm_nn_512_nomask(Ar, Br, prod_r)
    with torch.cuda.stream(stream_i):
        _launch_dgemm_nn_512_nomask(Ai, Bi, prod_i)
    with torch.cuda.stream(stream_sum):
        _launch_dgemm_nn_512_nomask(As, Bs, prod_sum)

    current_stream.wait_stream(stream_r)
    current_stream.wait_stream(stream_i)
    current_stream.wait_stream(stream_sum)
    _zgemm_merge_512_kernel[grid](C_real, prod_r, prod_i, prod_sum, BLOCK=1024)


def _try_zgemm_nn_pack_dgemm_512(
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
    if m != 512 or n != 512 or k != 512 or lda != 512 or ldb != 512 or ldc != 512:
        return False

    key = (A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _ZGEMM_NN_GRAPH_512["key"] != key:
        _launch_zgemm_nn_pack_dgemm_512(m, n, k, A, B, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_zgemm_nn_pack_dgemm_512(m, n, k, A, B, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_zgemm_nn_pack_dgemm_512(m, n, k, A, B, C)

        _ZGEMM_NN_GRAPH_512["key"] = key
        _ZGEMM_NN_GRAPH_512["graph"] = graph

    _ZGEMM_NN_GRAPH_512["graph"].replay()
    return True


@triton.jit
def _zgemm_split_sum2_pad_kernel(
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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    row = offsets // pad_size
    col = offsets - row * pad_size
    mask = (row < src_size) & (col < src_size)
    src_offsets = row * src_size + col
    ar = tl.load(src_a + 2 * src_offsets, mask=mask, other=0.0)
    ai = tl.load(src_a + 2 * src_offsets + 1, mask=mask, other=0.0)
    br = tl.load(src_b + 2 * src_offsets, mask=mask, other=0.0)
    bi = tl.load(src_b + 2 * src_offsets + 1, mask=mask, other=0.0)
    tl.store(dst_ar + offsets, ar)
    tl.store(dst_ai + offsets, ai)
    tl.store(dst_as + offsets, ar + ai)
    tl.store(dst_br + offsets, br)
    tl.store(dst_bi + offsets, bi)
    tl.store(dst_bs + offsets, br + bi)


@triton.jit
def _zgemm_split_sum2_pad_op_kernel(
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
def _zgemm_merge_pad_kernel(
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


@triton.jit
def _dgemm_nn_1024_nomask_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(1024, BLOCK_M)
    grid_n = tl.cdiv(1024, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k_start in range(0, 1024, BLOCK_K):
        cur_k = k_start + offs_k
        if TRANS_A == 0:
            a_ptrs = a_ptr + offs_m[:, None] * 1024 + cur_k[None, :]
        else:
            a_ptrs = a_ptr + cur_k[None, :] * 1024 + offs_m[:, None]
        if TRANS_B == 0:
            b_ptrs = b_ptr + cur_k[:, None] * 1024 + offs_n[None, :]
        else:
            b_ptrs = b_ptr + offs_n[None, :] * 1024 + cur_k[:, None]
        a_vals = tl.load(a_ptrs)
        b_vals = tl.load(b_ptrs)
        acc += tl.dot(a_vals, b_vals, out_dtype=tl.float64, allow_tf32=False)

    tl.store(c_ptr + offs_m[:, None] * 1024 + offs_n[None, :], acc)


def _launch_dgemm_nn_1024_nomask(a, b, c, transa: int = 0, transb: int = 0) -> None:
    if transa == 0 and transb == 0:
        block_m, block_n, block_k, num_warps, group_m = 64, 128, 32, 4, 1
    elif transa == 0 and transb == 1:
        block_m, block_n, block_k, num_warps, group_m = 64, 64, 16, 4, 8
    elif transa == 1 and transb == 0:
        block_m, block_n, block_k, num_warps, group_m = 128, 64, 32, 4, 1
    else:
        block_m, block_n, block_k, num_warps, group_m = 64, 128, 16, 4, 4
    grid = (triton.cdiv(1024, block_m) * triton.cdiv(1024, block_n),)
    _dgemm_nn_1024_nomask_kernel[grid](
        a,
        b,
        c,
        TRANS_A=transa,
        TRANS_B=transb,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=3,
    )


def _launch_zgemm_nn_pad_pack_dgemm(
    src_size: int,
    pad_size: int,
    transa: int,
    transb: int,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> None:
    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_zgemm_nn_workspace(
        A, pad_size, pad_size, pad_size
    )
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    split_grid = (triton.cdiv(pad_size * pad_size, 1024),)
    merge_grid = (triton.cdiv(src_size * src_size, 1024),)

    use_op_pack = pad_size == 512 and transa == 1 and transb == 0
    if use_op_pack:
        _zgemm_split_sum2_pad_op_kernel[split_grid](
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
    else:
        _zgemm_split_sum2_pad_kernel[split_grid](
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
            BLOCK=1024,
        )

    current_stream = torch.cuda.current_stream(device=A.device)
    stream_r, stream_i, stream_sum = _get_zgemm_nn_streams_512(A.device)
    stream_r.wait_stream(current_stream)
    stream_i.wait_stream(current_stream)
    stream_sum.wait_stream(current_stream)

    dgemm_transa = 0 if use_op_pack else transa
    dgemm_transb = 0 if use_op_pack else transb
    if pad_size == 512:
        with torch.cuda.stream(stream_r):
            _launch_dgemm_nn_512_nomask(Ar, Br, prod_r, dgemm_transa, dgemm_transb)
        with torch.cuda.stream(stream_i):
            _launch_dgemm_nn_512_nomask(Ai, Bi, prod_i, dgemm_transa, dgemm_transb)
        with torch.cuda.stream(stream_sum):
            _launch_dgemm_nn_512_nomask(As, Bs, prod_sum, dgemm_transa, dgemm_transb)
    else:
        with torch.cuda.stream(stream_r):
            _launch_dgemm_nn_1024_nomask(Ar, Br, prod_r, dgemm_transa, dgemm_transb)
        with torch.cuda.stream(stream_i):
            _launch_dgemm_nn_1024_nomask(Ai, Bi, prod_i, dgemm_transa, dgemm_transb)
        with torch.cuda.stream(stream_sum):
            _launch_dgemm_nn_1024_nomask(As, Bs, prod_sum, dgemm_transa, dgemm_transb)

    current_stream.wait_stream(stream_r)
    current_stream.wait_stream(stream_i)
    current_stream.wait_stream(stream_sum)
    _zgemm_merge_pad_kernel[merge_grid](
        C_real,
        prod_r,
        prod_i,
        prod_sum,
        src_size,
        pad_size,
        src_size * src_size,
        BLOCK=1024,
    )


def _try_zgemm_nn_pad_pack_dgemm(
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
    if m == 511 and n == 511 and k == 511 and lda == 511 and ldb == 511 and ldc == 511:
        src_size, pad_size = 511, 512
    elif (
        m == 1023
        and n == 1023
        and k == 1023
        and lda == 1023
        and ldb == 1023
        and ldc == 1023
    ):
        src_size, pad_size = 1023, 1024
    else:
        return False

    key = (src_size, A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _ZGEMM_NN_GRAPH_PAD["key"] != key:
        _launch_zgemm_nn_pad_pack_dgemm(src_size, pad_size, 0, 0, A, B, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_zgemm_nn_pad_pack_dgemm(src_size, pad_size, 0, 0, A, B, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_zgemm_nn_pad_pack_dgemm(src_size, pad_size, 0, 0, A, B, C)

        _ZGEMM_NN_GRAPH_PAD["key"] = key
        _ZGEMM_NN_GRAPH_PAD["graph"] = graph

    _ZGEMM_NN_GRAPH_PAD["graph"].replay()
    return True


def _try_zgemm_trans_pad_pack_dgemm(
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
    if transa == 0 and transb == 0:
        return False
    if transa == 2 or transb == 2:
        return False
    if not (m == n == k and lda == m and ldb == n and ldc == n):
        return False
    if m in (511, 512):
        src_size, pad_size = m, 512
    elif m in (1023, 1024):
        src_size, pad_size = m, 1024
    else:
        return False

    key = (transa, transb, src_size, A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _ZGEMM_TRANS_GRAPH_PAD["key"] != key:
        _launch_zgemm_nn_pad_pack_dgemm(src_size, pad_size, transa, transb, A, B, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_zgemm_nn_pad_pack_dgemm(src_size, pad_size, transa, transb, A, B, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_zgemm_nn_pad_pack_dgemm(src_size, pad_size, transa, transb, A, B, C)

        _ZGEMM_TRANS_GRAPH_PAD["key"] = key
        _ZGEMM_TRANS_GRAPH_PAD["graph"] = graph

    _ZGEMM_TRANS_GRAPH_PAD["graph"].replay()
    return True


def _get_zgemm_nn_workspace(A: torch.Tensor, m: int, n: int, k: int):
    key = (A.device, m, n, k)
    if _ZGEMM_NN_WORKSPACE["key"] != key:
        _ZGEMM_NN_WORKSPACE["key"] = key
        _ZGEMM_NN_WORKSPACE["buffers"] = (
            torch.empty((m, k), device=A.device, dtype=torch.float64),
            torch.empty((m, k), device=A.device, dtype=torch.float64),
            torch.empty((m, k), device=A.device, dtype=torch.float64),
            torch.empty((k, n), device=A.device, dtype=torch.float64),
            torch.empty((k, n), device=A.device, dtype=torch.float64),
            torch.empty((k, n), device=A.device, dtype=torch.float64),
            torch.empty((m, n), device=A.device, dtype=torch.float64),
            torch.empty((m, n), device=A.device, dtype=torch.float64),
            torch.empty((m, n), device=A.device, dtype=torch.float64),
        )
    return _ZGEMM_NN_WORKSPACE["buffers"]


def _launch_dgemm_nn(a, b, c, alpha: float, beta: float, m: int, n: int, k: int):
    block_m, block_n, block_k, num_warps, group_m, maxnreg = _select_dgemm_dot_config(
        0, 0, m, n, k
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    launch_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg
    _dgemm_dot_kernel[grid](
        a,
        b,
        c,
        alpha,
        beta,
        m,
        n,
        k,
        k,
        n,
        n,
        0,
        0,
        beta == 0.0,
        alpha == 1.0,
        **launch_kwargs,
    )


def _launch_dgemm_real_config(
    a, b, c, m: int, n: int, k: int, transa: int, transb: int, config
) -> None:
    block_m, block_n, block_k, num_warps, group_m, maxnreg = config
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    launch_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": num_warps,
    }
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg
    _dgemm_dot_kernel[grid](
        a,
        b,
        c,
        1.0,
        0.0,
        m,
        n,
        k,
        m,
        n,
        n,
        transa,
        transb,
        True,
        True,
        **launch_kwargs,
    )


def _launch_zgemm_trans_split_dgemm_1536(
    transa: int,
    transb: int,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> None:
    m = n = k = 1536
    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_zgemm_nn_workspace(
        A, m, n, k
    )
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(m * n, 1024),)

    _zgemm_split_sum_kernel[grid](A_real, Ar, Ai, As, m * k, BLOCK=1024)
    _zgemm_split_sum_kernel[grid](B_real, Br, Bi, Bs, k * n, BLOCK=1024)

    if transa == 0 and transb == 1:
        config = (128, 64, 16, 4, 1, None)
    else:
        config = (128, 64, 32, 4, 1, None)

    current_stream = torch.cuda.current_stream(device=A.device)
    stream_r, stream_i, stream_sum = _get_zgemm_nn_streams_512(A.device)
    stream_r.wait_stream(current_stream)
    stream_i.wait_stream(current_stream)
    stream_sum.wait_stream(current_stream)

    with torch.cuda.stream(stream_r):
        _launch_dgemm_real_config(Ar, Br, prod_r, m, n, k, transa, transb, config)
    with torch.cuda.stream(stream_i):
        _launch_dgemm_real_config(Ai, Bi, prod_i, m, n, k, transa, transb, config)
    with torch.cuda.stream(stream_sum):
        _launch_dgemm_real_config(As, Bs, prod_sum, m, n, k, transa, transb, config)

    current_stream.wait_stream(stream_r)
    current_stream.wait_stream(stream_i)
    current_stream.wait_stream(stream_sum)
    _zgemm_merge_3m_kernel[grid](C_real, prod_r, prod_i, prod_sum, m * n, BLOCK=1024)


def _try_zgemm_trans_split_dgemm_1536(
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
    if transa == 0 and transb == 0:
        return False
    if transa == 2 or transb == 2:
        return False
    if not (m == n == k == 1536 and lda == 1536 and ldb == 1536 and ldc == 1536):
        return False

    key = (transa, transb, A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _ZGEMM_TRANS_GRAPH_SPLIT["key"] != key:
        _launch_zgemm_trans_split_dgemm_1536(transa, transb, A, B, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_zgemm_trans_split_dgemm_1536(transa, transb, A, B, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_zgemm_trans_split_dgemm_1536(transa, transb, A, B, C)

        _ZGEMM_TRANS_GRAPH_SPLIT["key"] = key
        _ZGEMM_TRANS_GRAPH_SPLIT["graph"] = graph

    _ZGEMM_TRANS_GRAPH_SPLIT["graph"].replay()
    return True


def _launch_zgemm_trans_pack_dgemm(
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
    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_zgemm_nn_workspace(
        A, m, n, k
    )
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    split_grid_a = (triton.cdiv(m * k, 1024),)
    split_grid_b = (triton.cdiv(k * n, 1024),)
    merge_grid = (triton.cdiv(m * n, 1024),)

    _zgemm_split_sum_op_kernel[split_grid_a](
        A_real, Ar, Ai, As, m * k, k, lda, transa, BLOCK=1024
    )
    _zgemm_split_sum_op_kernel[split_grid_b](
        B_real, Br, Bi, Bs, k * n, n, ldb, transb, BLOCK=1024
    )

    current_stream = torch.cuda.current_stream(device=A.device)
    stream_r, stream_i, stream_sum = _get_zgemm_nn_streams_512(A.device)
    stream_r.wait_stream(current_stream)
    stream_i.wait_stream(current_stream)
    stream_sum.wait_stream(current_stream)

    with torch.cuda.stream(stream_r):
        _launch_dgemm_nn(Ar, Br, prod_r, 1.0, 0.0, m, n, k)
    with torch.cuda.stream(stream_i):
        _launch_dgemm_nn(Ai, Bi, prod_i, 1.0, 0.0, m, n, k)
    with torch.cuda.stream(stream_sum):
        _launch_dgemm_nn(As, Bs, prod_sum, 1.0, 0.0, m, n, k)

    current_stream.wait_stream(stream_r)
    current_stream.wait_stream(stream_i)
    current_stream.wait_stream(stream_sum)
    _zgemm_merge_3m_kernel[merge_grid](
        C_real, prod_r, prod_i, prod_sum, m * n, BLOCK=1024
    )


def _try_zgemm_trans_pack_dgemm(
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
    if transa == 0 and transb == 0:
        return False
    if transa == 2 or transb == 2:
        return False
    if max(m, n, k) < 1536:
        return False
    if not (m == n == k and lda == m and ldb == n and ldc == n):
        return False

    key = (transa, transb, m, A.device, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if _ZGEMM_TRANS_GRAPH_PACK["key"] != key:
        _launch_zgemm_trans_pack_dgemm(transa, transb, m, n, k, A, lda, B, ldb, C)
        torch_device_fn.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=A.device)
        stream.wait_stream(torch.cuda.current_stream(device=A.device))
        with torch.cuda.stream(stream):
            _launch_zgemm_trans_pack_dgemm(transa, transb, m, n, k, A, lda, B, ldb, C)
        torch.cuda.current_stream(device=A.device).wait_stream(stream)
        torch_device_fn.synchronize()

        with torch.cuda.graph(graph):
            _launch_zgemm_trans_pack_dgemm(transa, transb, m, n, k, A, lda, B, ldb, C)

        _ZGEMM_TRANS_GRAPH_PACK["key"] = key
        _ZGEMM_TRANS_GRAPH_PACK["graph"] = graph

    _ZGEMM_TRANS_GRAPH_PACK["graph"].replay()
    return True


def _try_zgemm_nn_pack_dgemm(
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
    max_dim = max(m, n, k)
    if (max_dim < 1023 and max_dim != 512) or lda != k or ldb != n or ldc != n:
        return False

    Ar, Ai, As, Br, Bi, Bs, prod_r, prod_i, prod_sum = _get_zgemm_nn_workspace(
        A, m, n, k
    )
    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    split_grid_a = (triton.cdiv(m * k, 1024),)
    split_grid_b = (triton.cdiv(k * n, 1024),)
    merge_grid = (triton.cdiv(m * n, 1024),)

    _zgemm_split_sum_kernel[split_grid_a](A_real, Ar, Ai, As, m * k, BLOCK=1024)
    _zgemm_split_sum_kernel[split_grid_b](B_real, Br, Bi, Bs, k * n, BLOCK=1024)

    _launch_dgemm_nn(Ar, Br, prod_r, 1.0, 0.0, m, n, k)
    _launch_dgemm_nn(Ai, Bi, prod_i, 1.0, 0.0, m, n, k)
    _launch_dgemm_nn(As, Bs, prod_sum, 1.0, 0.0, m, n, k)

    _zgemm_merge_3m_kernel[merge_grid](
        C_real, prod_r, prod_i, prod_sum, m * n, BLOCK=1024
    )
    return True


def _select_zgemm_hopper_config(transa: int, transb: int, m: int, n: int, k: int):
    max_dim = max(m, n, k)
    min_dim = min(m, n, k)

    if max_dim <= 32:
        return 16, 16, 16, 4, 1, None
    if max_dim <= 128:
        return 16, 16, 32, 2, 8, None
    if max_dim <= 256:
        return 16, 32, 32, 2, 8, None
    if max_dim == 511:
        if transa == 0 and transb == 0:
            return 16, 64, 16, 4, 8, None
        return 64, 32, 16, 4, 4, None
    if max_dim <= 512:
        if transa == 0:
            return 32, 64, 16, 4, 4, None
        return 64, 32, 16, 4, 8, None
    if max_dim == 1023:
        return 64, 32, 16, 4, 8, None
    if max_dim <= 1024:
        if transa == 0 and transb == 0:
            return 64, 64, 16, 4, 1, None
        if transa == 0:
            return 32, 64, 16, 4, 8, None
        return 64, 32, 16, 4, 8, None

    if transa == 0 and transb == 0:
        if max_dim <= 2048:
            return 64, 64, 16, 4, 2, None
        return 64, 64, 16, 4, 1, None

    if transa == 0 and transb in (1, 2):
        if max_dim <= 2048:
            return 32, 64, 16, 4, 4, None
        if max_dim <= 4096:
            return 32, 64, 16, 4, 8, None
        if min_dim >= 6144:
            return 64, 64, 16, 4, 1, None
        return 32, 64, 16, 4, 8, None

    if transa in (1, 2) and transb == 0:
        if max_dim <= 1536:
            return 64, 32, 16, 4, 2, None
        if max_dim <= 4096:
            return 64, 32, 16, 4, 1, None
        if min_dim >= 6144:
            return 64, 64, 16, 4, 1, None
        return 64, 32, 16, 4, 1, None

    if max_dim <= 1536:
        return 64, 32, 16, 4, 2, None
    if max_dim <= 2048:
        return 64, 32, 16, 4, 8, None
    if max_dim <= 4096:
        return 64, 32, 16, 4, 4, None
    if min_dim >= 6144:
        return 64, 64, 16, 4, 1, None
    return 64, 32, 16, 4, 4, None


# ---------------------------------------------------------------------------
# Module-level condition predicates for zgemm StaticDispatch
# ---------------------------------------------------------------------------
def _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero, **_kw):
    return alpha_is_one and beta_is_zero


def _zgemm_is_nn(transa, transb, **_kw):
    return transa == 0 and transb == 0


def _zgemm_is_nonconj_trans(transa, transb, **_kw):
    return not (transa == 0 and transb == 0) and transa != 2 and transb != 2


def _zgemm_can_nn_128(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nn(transa, transb)
        and m == 128
        and n == 128
        and k == 128
        and lda == 128
        and ldb == 128
        and ldc == 128
    )


def _zgemm_can_nn_256(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nn(transa, transb)
        and m == 256
        and n == 256
        and k == 256
        and lda == 256
        and ldb == 256
        and ldc == 256
    )


def _zgemm_can_trans_128(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nonconj_trans(transa, transb)
        and m == 128
        and n == 128
        and k == 128
        and lda == 128
        and ldb == 128
        and ldc == 128
    )


def _zgemm_can_trans_256(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nonconj_trans(transa, transb)
        and m == 256
        and n == 256
        and k == 256
        and lda == 256
        and ldb == 256
        and ldc == 256
    )


def _zgemm_can_trans_pad_pack(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nonconj_trans(transa, transb)
        and m == n == k
        and lda == m
        and ldb == n
        and ldc == n
        and m in (511, 512, 1023, 1024)
    )


def _zgemm_can_trans_split_1536(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nonconj_trans(transa, transb)
        and m == n == k == 1536
        and lda == 1536
        and ldb == 1536
        and ldc == 1536
    )


def _zgemm_can_trans_pack(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nonconj_trans(transa, transb)
        and max(m, n, k) >= 1536
        and m == n == k
        and lda == m
        and ldb == n
        and ldc == n
    )


def _zgemm_can_nn_pad_pack(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nn(transa, transb)
        and (
            (
                m == 511
                and n == 511
                and k == 511
                and lda == 511
                and ldb == 511
                and ldc == 511
            )
            or (
                m == 1023
                and n == 1023
                and k == 1023
                and lda == 1023
                and ldb == 1023
                and ldc == 1023
            )
        )
    )


def _zgemm_can_nn_pack_512(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nn(transa, transb)
        and m == 512
        and n == 512
        and k == 512
        and lda == 512
        and ldb == 512
        and ldc == 512
    )


def _zgemm_can_nn_pack(
    m, n, k, lda, ldb, ldc, transa, transb, alpha_is_one, beta_is_zero, **_kw
):
    max_dim = max(m, n, k)
    return (
        _zgemm_is_alpha_one_beta_zero(alpha_is_one, beta_is_zero)
        and _zgemm_is_nn(transa, transb)
        and not (max_dim < 1023 and max_dim != 512)
        and lda == k
        and ldb == n
        and ldc == n
    )


def _zgemm_is_default(**_kw):
    return True


# ---------------------------------------------------------------------------
# Module-level factory functions for zgemm StaticDispatch
# ---------------------------------------------------------------------------
def _zgemm_build_nn_128(m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_nn_128(m, n, k, A, lda, B, ldb, C, ldc)


def _zgemm_build_nn_256(m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_nn_256(m, n, k, A, lda, B, ldb, C, ldc)


def _zgemm_build_trans_128(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_trans_128(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)


def _zgemm_build_trans_256(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_trans_256(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)


def _zgemm_build_trans_pad_pack(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_trans_pad_pack_dgemm(
        transa, transb, m, n, k, A, lda, B, ldb, C, ldc
    )


def _zgemm_build_trans_split_1536(
    transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw
):
    return lambda: _try_zgemm_trans_split_dgemm_1536(
        transa, transb, m, n, k, A, lda, B, ldb, C, ldc
    )


def _zgemm_build_trans_pack(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_trans_pack_dgemm(
        transa, transb, m, n, k, A, lda, B, ldb, C, ldc
    )


def _zgemm_build_nn_pad_pack(m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_nn_pad_pack_dgemm(m, n, k, A, lda, B, ldb, C, ldc)


def _zgemm_build_nn_pack_512(m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_nn_pack_dgemm_512(m, n, k, A, lda, B, ldb, C, ldc)


def _zgemm_build_nn_pack(m, n, k, A, lda, B, ldb, C, ldc, **_kw):
    return lambda: _try_zgemm_nn_pack_dgemm(m, n, k, A, lda, B, ldb, C, ldc)


def _zgemm_build_dot_kernel(
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
    ) = _select_zgemm_hopper_config(transa, transb, m, n, k)
    launch_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": num_warps,
    }
    if transa == 0 and transb == 0 and max(m, n, k) == 4096:
        launch_kwargs["num_stages"] = 4
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    return lambda: _zgemm_dot_kernel[grid](
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


_ZGEMM_DISPATCH = StaticDispatch(
    [
        (_zgemm_can_nn_128, _zgemm_build_nn_128),
        (_zgemm_can_nn_256, _zgemm_build_nn_256),
        (_zgemm_can_trans_128, _zgemm_build_trans_128),
        (_zgemm_can_trans_256, _zgemm_build_trans_256),
        (_zgemm_can_trans_pad_pack, _zgemm_build_trans_pad_pack),
        (_zgemm_can_trans_split_1536, _zgemm_build_trans_split_1536),
        (_zgemm_can_trans_pack, _zgemm_build_trans_pack),
        (_zgemm_can_nn_pad_pack, _zgemm_build_nn_pad_pack),
        (_zgemm_can_nn_pack_512, _zgemm_build_nn_pack_512),
        (_zgemm_can_nn_pack, _zgemm_build_nn_pack),
        (_zgemm_is_default, _zgemm_build_dot_kernel),
    ]
)


def zgemm(
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
    _validate_zgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

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
        runner = _ZGEMM_DISPATCH.lookup_and_build(
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

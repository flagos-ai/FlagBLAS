import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_blas import runtime
from flag_blas.ops.level3.group_gemm import (
    grouped_bfgemm_kernel,
    grouped_hgemm_kernel,
    grouped_launch,
    grouped_mm_kernel,
    grouped_tf32gemm_kernel,
)
from flag_blas.utils import libentry, libtuner


def supports_tma(device=None):
    return torch.cuda.get_device_capability(device)[0] >= 9


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M_val = nargs["BLOCK_M"]
    BLOCK_N_val = nargs["BLOCK_N"]
    BLOCK_K_val = nargs["BLOCK_K"]
    nargs["a_desc"].block_shape = [BLOCK_M_val, BLOCK_K_val]
    nargs["b_desc"].block_shape = [BLOCK_K_val, BLOCK_N_val]
    nargs["c_desc"].block_shape = [BLOCK_M_val, BLOCK_N_val]


def _get_group_mm_configs():
    configs = runtime.get_tuned_config("group_mm")
    for config in configs:
        config.pre_hook = matmul_tma_set_block_size_hook
    return configs


@libentry()
@libtuner(configs=runtime.get_tuned_config("group_bfgemm"), key=["M", "N", "K"])
@triton.jit
def grouped_bfgemm_tma_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(group_ldas + g)
            ldb = tl.load(group_ldbs + g)
            ldc = tl.load(group_ldcs + g)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.bfloat16))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gk, gn],
                strides=[ldb, 1],
                block_shape=[BLOCK_K, BLOCK_N],
            )
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_K])
                    b = b_desc.load([kk * BLOCK_K, offs_bn])
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                if beta == 0.0:
                    accumulator = accumulator * alpha
                else:
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
                    c_desc = tl.make_tensor_descriptor(
                        c_ptr,
                        shape=[gm, gn],
                        strides=[ldc, 1],
                        block_shape=[BLOCK_M, BLOCK_N],
                    )
                    ori_c = c_desc.load([offs_cm, offs_cn])
                    accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(out_desc.dtype)
                out_desc.store([offs_cm, offs_cn], c)

                tile_idx += total_grid

        last_problem_end = current_problem_end


@libentry()
@libtuner(configs=runtime.get_tuned_config("group_bfgemm"), key=["M", "N", "K"])
@triton.jit
def grouped_bfgemm_small_m_tma_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    total_tiles = group_size * num_n_tiles
    loop_count = (total_tiles - tile_idx + total_grid - 1) // total_grid

    for _ in tl.range(loop_count):
        g = tile_idx // num_n_tiles
        tile_n_idx = tile_idx - g * num_n_tiles
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        lda = tl.load(group_ldas + g)
        ldb = tl.load(group_ldbs + g)
        ldc = tl.load(group_ldcs + g)

        a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
        b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
        out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.bfloat16))

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[gm, gk],
            strides=[lda, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[gk, gn],
            strides=[ldb, 1],
            block_shape=[BLOCK_K, BLOCK_N],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[gm, gn],
            strides=[ldc, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        offs_bn = tile_n_idx * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(gk, BLOCK_K)):
            a = a_desc.load([0, kk * BLOCK_K])
            b = b_desc.load([kk * BLOCK_K, offs_bn])
            accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

        if beta == 0.0:
            accumulator = accumulator * alpha
        else:
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            ori_c = c_desc.load([0, offs_bn])
            accumulator = ori_c * beta + accumulator * alpha

        c = accumulator.to(out_desc.dtype)
        out_desc.store([0, offs_bn], c)
        tile_idx += total_grid


@libentry()
@libtuner(configs=runtime.get_tuned_config("group_hgemm"), key=["M", "N", "K"])
@triton.jit
def grouped_hgemm_tma_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(group_ldas + g)
            ldb = tl.load(group_ldbs + g)
            ldc = tl.load(group_ldcs + g)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.float16))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gk, gn],
                strides=[ldb, 1],
                block_shape=[BLOCK_K, BLOCK_N],
            )
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_K])
                    b = b_desc.load([kk * BLOCK_K, offs_bn])
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                if beta == 0.0:
                    accumulator = accumulator * alpha
                else:
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
                    c_desc = tl.make_tensor_descriptor(
                        c_ptr,
                        shape=[gm, gn],
                        strides=[ldc, 1],
                        block_shape=[BLOCK_M, BLOCK_N],
                    )
                    ori_c = c_desc.load([offs_cm, offs_cn])
                    accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(out_desc.dtype)
                out_desc.store([offs_cm, offs_cn], c)

                tile_idx += total_grid

        last_problem_end = current_problem_end


@libentry()
@libtuner(configs=runtime.get_tuned_config("group_hgemm"), key=["M", "N", "K"])
@triton.jit
def grouped_hgemm_small_m_tma_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    total_tiles = group_size * num_n_tiles
    loop_count = (total_tiles - tile_idx + total_grid - 1) // total_grid

    for _ in tl.range(loop_count):
        g = tile_idx // num_n_tiles
        tile_n_idx = tile_idx - g * num_n_tiles
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        lda = tl.load(group_ldas + g)
        ldb = tl.load(group_ldbs + g)
        ldc = tl.load(group_ldcs + g)

        a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
        b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
        out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.float16))

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[gm, gk],
            strides=[lda, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[gk, gn],
            strides=[ldb, 1],
            block_shape=[BLOCK_K, BLOCK_N],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[gm, gn],
            strides=[ldc, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        offs_bn = tile_n_idx * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(gk, BLOCK_K)):
            a = a_desc.load([0, kk * BLOCK_K])
            b = b_desc.load([kk * BLOCK_K, offs_bn])
            accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

        if beta == 0.0:
            accumulator = accumulator * alpha
        else:
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            ori_c = c_desc.load([0, offs_bn])
            accumulator = ori_c * beta + accumulator * alpha

        c = accumulator.to(out_desc.dtype)
        out_desc.store([0, offs_bn], c)
        tile_idx += total_grid


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"), key=["M", "N", "K"]
)
@triton.jit
def grouped_tf32gemm_tma_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(group_ldas + g)
            ldb = tl.load(group_ldbs + g)
            ldc = tl.load(group_ldcs + g)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float32))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float32))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.float32))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_K])
                    accumulator = tl.dot(
                        a,
                        b.T,
                        acc=accumulator,
                        out_dtype=tl.float32,
                        input_precision="tf32",
                    )

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                if beta == 0.0:
                    accumulator = accumulator * alpha
                else:
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float32))
                    c_desc = tl.make_tensor_descriptor(
                        c_ptr,
                        shape=[gm, gn],
                        strides=[ldc, 1],
                        block_shape=[BLOCK_M, BLOCK_N],
                    )
                    ori_c = c_desc.load([offs_cm, offs_cn])
                    accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(out_desc.dtype)
                out_desc.store([offs_cm, offs_cn], c)

                tile_idx += total_grid

        last_problem_end = current_problem_end


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"), key=["M", "N", "K"]
)
@triton.jit
def grouped_tf32gemm_small_m_tma_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    total_tiles = group_size * num_n_tiles
    loop_count = (total_tiles - tile_idx + total_grid - 1) // total_grid

    for _ in tl.range(loop_count):
        g = tile_idx // num_n_tiles
        tile_n_idx = tile_idx - g * num_n_tiles
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        lda = tl.load(group_ldas + g)
        ldb = tl.load(group_ldbs + g)
        ldc = tl.load(group_ldcs + g)

        a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float32))
        b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float32))
        out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.float32))

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[gm, gk],
            strides=[lda, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[gn, gk],
            strides=[ldb, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[gm, gn],
            strides=[ldc, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        offs_bn = tile_n_idx * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(gk, BLOCK_K)):
            a = a_desc.load([0, kk * BLOCK_K])
            b = b_desc.load([offs_bn, kk * BLOCK_K])
            accumulator = tl.dot(
                a,
                b.T,
                acc=accumulator,
                out_dtype=tl.float32,
                input_precision="tf32",
            )

        if beta == 0.0:
            accumulator = accumulator * alpha
        else:
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float32))
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            ori_c = c_desc.load([0, offs_bn])
            accumulator = ori_c * beta + accumulator * alpha

        c = accumulator.to(out_desc.dtype)
        out_desc.store([0, offs_bn], c)
        tile_idx += total_grid


@libentry()
@libtuner(
    configs=_get_group_mm_configs(), key=["M", "N", "K"]
)
@triton.jit
def grouped_mm_tma_kernel(
    a_desc,
    b_desc,
    c_desc,
    C,
    offs,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    group_end = 0

    for group_idx in tl.range(num_groups):
        group_end = tl.load(offs + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, m, N, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = group_start + tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N
                offs_bk = group_idx * K

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    a = a_desc.load([offs_am, k * BLOCK_K])
                    b = b_desc.load([offs_bk + k * BLOCK_K, offs_bn])
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                c = accumulator.to(c_desc.dtype)

                if offs_am + BLOCK_M <= group_end:
                    c_desc.store([offs_am, offs_bn], c)
                else:
                    offs_cm = offs_am + tl.arange(0, BLOCK_M)
                    offs_cn = offs_bn + tl.arange(0, BLOCK_N)
                    c_ptrs = (
                        C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                    )
                    c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, c, mask=c_mask)

                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


def group_bfgemm(
    group_out,
    d_a_ptrs,
    d_b_ptrs,
    d_c_ptrs,
    d_output_ptrs,
    d_m_sizes,
    d_n_sizes,
    d_k_sizes,
    d_ldas,
    d_ldbs,
    d_ldcs,
    group_size,
    M,
    N,
    K,
    alpha,
    beta,
    use_small_m=False,
):
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_available = supports_tma(group_out.device)
    if tma_available and use_small_m:
        grouped_bfgemm_small_m_tma_kernel[(num_sms,)](
            M,
            N,
            K,
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            d_output_ptrs,
            d_m_sizes,
            d_n_sizes,
            d_k_sizes,
            d_ldas,
            d_ldbs,
            d_ldcs,
            group_size,
            alpha=alpha,
            beta=beta,
        )
        return group_out

    kernel = grouped_bfgemm_tma_kernel if tma_available else grouped_bfgemm_kernel
    kernel[(num_sms,)](
        M,
        N,
        K,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_output_ptrs,
        d_m_sizes,
        d_n_sizes,
        d_k_sizes,
        d_ldas,
        d_ldbs,
        d_ldcs,
        group_size,
        alpha=alpha,
        beta=beta,
    )
    return group_out


def group_hgemm(
    group_out,
    d_a_ptrs,
    d_b_ptrs,
    d_c_ptrs,
    d_output_ptrs,
    d_m_sizes,
    d_n_sizes,
    d_k_sizes,
    d_ldas,
    d_ldbs,
    d_ldcs,
    group_size,
    M,
    N,
    K,
    alpha,
    beta,
    use_small_m=False,
):
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_available = supports_tma(group_out.device)
    if tma_available and use_small_m:
        grouped_hgemm_small_m_tma_kernel[(num_sms,)](
            M,
            N,
            K,
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            d_output_ptrs,
            d_m_sizes,
            d_n_sizes,
            d_k_sizes,
            d_ldas,
            d_ldbs,
            d_ldcs,
            group_size,
            alpha=alpha,
            beta=beta,
        )
        return group_out

    kernel = grouped_hgemm_tma_kernel if tma_available else grouped_hgemm_kernel
    kernel[(num_sms,)](
        M,
        N,
        K,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_output_ptrs,
        d_m_sizes,
        d_n_sizes,
        d_k_sizes,
        d_ldas,
        d_ldbs,
        d_ldcs,
        group_size,
        alpha=alpha,
        beta=beta,
    )
    return group_out


def group_tf32gemm(
    group_out,
    d_a_ptrs,
    d_b_ptrs,
    d_c_ptrs,
    d_output_ptrs,
    d_m_sizes,
    d_n_sizes,
    d_k_sizes,
    d_ldas,
    d_ldbs,
    d_ldcs,
    group_size,
    M,
    N,
    K,
    alpha,
    beta,
    use_small_m=False,
):
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_available = supports_tma(group_out.device)
    if tma_available and use_small_m:
        grouped_tf32gemm_small_m_tma_kernel[(num_sms,)](
            M,
            N,
            K,
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            d_output_ptrs,
            d_m_sizes,
            d_n_sizes,
            d_k_sizes,
            d_ldas,
            d_ldbs,
            d_ldcs,
            group_size,
            alpha=alpha,
            beta=beta,
        )
        return group_out

    kernel = grouped_tf32gemm_tma_kernel if tma_available else grouped_tf32gemm_kernel
    kernel[(num_sms,)](
        M,
        N,
        K,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_output_ptrs,
        d_m_sizes,
        d_n_sizes,
        d_k_sizes,
        d_ldas,
        d_ldbs,
        d_ldcs,
        group_size,
        alpha=alpha,
        beta=beta,
    )
    return group_out


def group_mm(A: torch.Tensor, B: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape

    num_groups, _BK, N = B.shape
    strideBK, strideBN = B.stride(1), B.stride(2)

    assert num_groups == offs.numel()
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    C = A.new_empty(M, N)
    if not supports_tma(A.device):
        grouped_mm_kernel[(num_sms,)](
            A,
            B,
            C,
            offs,
            num_groups,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            strideBK,
            strideBN,
            C.stride(0),
            C.stride(1),
        )
        return C

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(A, A.shape, A.stride(), dummy_block)
    b_desc = TensorDescriptor(B, [num_groups * K, N], [strideBK, strideBN], dummy_block)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), dummy_block)

    grouped_mm_tma_kernel[(num_sms,)](
        a_desc,
        b_desc,
        c_desc,
        C,
        offs,
        num_groups,
        M,
        N,
        K,
        C.stride(0),
        C.stride(1),
    )

    return C

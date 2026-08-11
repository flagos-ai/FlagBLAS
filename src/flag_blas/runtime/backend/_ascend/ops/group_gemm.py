import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level3.group_gemm import grouped_launch
from flag_blas.runtime.backend._ascend.utils import CORE_NUM
from flag_blas.utils import libentry, libtuner
from flag_blas.utils import triton_lang_extension as tle


def _get_num_aicore():
    try:
        import triton.runtime.driver as driver

        properties = driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        return max(1, int(properties["num_aicore"]))
    except (ImportError, AttributeError, RuntimeError, KeyError, TypeError, ValueError):
        return max(1, int(CORE_NUM) // 2)


def _use_n_chunk(N, K):
    return K == 2048 and N == 7168


@libentry()
@libtuner(configs=runtime.get_tuned_config("group_bfgemm"), key=["M", "N", "K"])
@triton.jit
def grouped_bfgemm_kernel(
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_A,
    group_B,
    group_list,
    group_out,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_idx = tle.program_id(0)
    total_grid = tle.num_programs(0)
    last_problem_end = tile_idx * 0
    group_start = (tile_idx * 0).to(tl.int64)
    a_descriptor = tl.make_tensor_descriptor(
        group_A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    for group_idx in range(group_size):
        group_end = tl.load(group_list + group_idx).to(tl.int64)
        group_m = group_end - group_start

        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm,
                    group_m,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    GROUP_M,
                )

                row_start = group_start + tile_m_idx * BLOCK_M
                local_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_m = group_start + local_m
                offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)
                mask_m = local_m < group_m
                mask_n = offs_n < N

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in range(0, tl.cdiv(K, BLOCK_K)):
                    k_offsets = k * BLOCK_K + offs_k
                    mask_k = k_offsets < K
                    b_ptrs = (
                        group_B
                        + group_idx * K * N
                        + k_offsets[:, None] * N
                        + offs_n[None, :]
                    )
                    if K % BLOCK_K == 0:
                        a = tl.load_tensor_descriptor(
                            a_descriptor,
                            [
                                row_start.to(tl.int32),
                                k * BLOCK_K,
                            ],
                        )
                    else:
                        a_ptrs = group_A + offs_m[:, None] * K + k_offsets[None, :]
                        a = tl.load(
                            a_ptrs,
                            mask=mask_m[:, None] & mask_k[None, :],
                            other=0.0,
                        )
                    b = tl.load(
                        b_ptrs,
                        mask=mask_k[:, None] & mask_n[None, :],
                        other=0.0,
                    )
                    accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

                out_ptrs = group_out + offs_m[:, None] * N + offs_n[None, :]
                tl.store(
                    out_ptrs,
                    accumulator.to(tl.bfloat16),
                    mask=mask_m[:, None] & mask_n[None, :],
                )
                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


@libentry()
@triton.jit
def grouped_bfgemm_n_chunk_kernel(
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_A,
    group_B,
    group_list,
    group_out,
    group_size,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    core_idx = tle.program_id(0)
    total_grid = tle.num_programs(0)
    task_idx = core_idx
    last_problem_end = core_idx * 0
    group_start = (core_idx * 0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    a_descriptor = tl.make_tensor_descriptor(
        group_A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    for group_idx in range(group_size):
        group_end = tl.load(group_list + group_idx)
        group_m = group_end - group_start
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        for n_chunk_start in range(0, N, 2048):
            chunk_n = min(2048, N - n_chunk_start)
            num_n_tiles = tl.cdiv(chunk_n, BLOCK_N)
            current_problem_end = last_problem_end + num_m_tiles * num_n_tiles
            if task_idx >= last_problem_end and task_idx < current_problem_end:
                loop_count = (current_problem_end - task_idx + total_grid - 1) // total_grid
                for _ in tl.range(loop_count):
                    task_idx_in_chunk = task_idx - last_problem_end
                    m_tile = task_idx_in_chunk // num_n_tiles
                    n_tile = task_idx_in_chunk % num_n_tiles
                    row_start = group_start + m_tile * BLOCK_M
                    local_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
                    offs_m = group_start + local_m
                    offs_n = n_chunk_start + n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask_m = local_m < group_m

                    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    for k in range(0, tl.cdiv(K, BLOCK_K)):
                        k_offsets = k * BLOCK_K + offs_k
                        b_ptrs = (
                            group_B
                            + group_idx * K * N
                            + k_offsets[:, None] * N
                            + offs_n[None, :]
                        )
                        a = tl.load_tensor_descriptor(
                            a_descriptor,
                            [row_start.to(tl.int32), k * BLOCK_K],
                        )
                        b = tl.load(b_ptrs)
                        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

                    out_ptrs = group_out + offs_m[:, None] * N + offs_n[None, :]
                    tl.store(
                        out_ptrs,
                        accumulator.to(tl.bfloat16),
                        mask=mask_m[:, None],
                    )
                    task_idx += total_grid
            last_problem_end = current_problem_end
        group_start = group_end


def group_bfgemm(group_A, group_B, group_list, group_out):
    M, K = group_A.shape
    group_size, _, N = group_B.shape

    num_aicores = _get_num_aicore()
    if not _use_n_chunk(N, K):
        grouped_bfgemm_kernel[(num_aicores,)](
            M,
            N,
            K,
            group_A,
            group_B,
            group_list,
            group_out,
            group_size,
            sync_solver=False,
        )
        return group_out

    grouped_bfgemm_n_chunk_kernel[(num_aicores,)](
        M,
        N,
        K,
        group_A,
        group_B,
        group_list,
        group_out,
        group_size,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        num_stages=2,
        num_warps=8,
        unit_flag=True,
        sync_solver=False,
    )
    return group_out

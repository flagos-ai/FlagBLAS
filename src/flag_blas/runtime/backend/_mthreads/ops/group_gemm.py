import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_blas import runtime
from flag_blas.ops.level3.group_gemm import grouped_launch
from flag_blas.utils import libentry, libtuner


GROUP_BFGEMM_LLC_OPTIONS = ("-mtgpu-enable-preisel-sinking=1 " "-mtgpu-enable-int-hidden=1 " "-mtgpu-use-sl=1 " "-mtgpu-enable-postra-sched=1 " "-mtgpu-reorder-scheduling=1")


def matmul_tma_set_block_size_hook(nargs):
    block_m = nargs.get("BLOCK_M", 128)
    block_k = nargs.get("BLOCK_K", 64)
    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_k, nargs["BLOCK_N"]]


for config in runtime.get_tuned_config("group_bfgemm"):
    config.pre_hook = matmul_tma_set_block_size_hook
    config.kwargs["enable_backend_opt"] = True
    config.kwargs["llc_options"] = GROUP_BFGEMM_LLC_OPTIONS


@libentry()
@libtuner(configs=runtime.get_tuned_config("group_bfgemm"), key=["M", "N", "K", "group_size"])
@triton.jit
def grouped_bfgemm_tma_kernel(M, N: tl.constexpr, K: tl.constexpr, a_desc, b_desc, group_list, group_out, group_size: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    for group_idx in tl.range(group_size):
        group_end = tl.load(group_list + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        current_problem_end = last_problem_end + num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = tl.cdiv(current_problem_end - tile_idx, total_grid)
            for _ in tl.range(loop_count):
                group_tile_idx = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(group_tile_idx, m, N, BLOCK_M, BLOCK_N, GROUP_M)
                offs_am = group_start + tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N
                offs_bk = group_idx * K

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    a = a_desc.load([offs_am, k * BLOCK_K])
                    b = b_desc.load([offs_bk + k * BLOCK_K, offs_bn])
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                c = accumulator.to(tl.bfloat16)
                offs_cm = offs_am + tl.arange(0, BLOCK_M)
                offs_cn = offs_bn + tl.arange(0, BLOCK_N)
                c_ptrs = group_out + offs_am.to(tl.int64) * N + offs_bn + N * tl.arange(0, BLOCK_M)[:, None] + tl.arange(0, BLOCK_N)[None, :]
                c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)

                tile_idx += total_grid
        last_problem_end = current_problem_end
        group_start = group_end


@libentry()
@libtuner(configs=runtime.get_tuned_config("group_bfgemm_m256"), key=["M", "N", "K", "group_size"])
@triton.jit
def grouped_bfgemm_tma_m256_kernel(M, N: tl.constexpr, K: tl.constexpr, a_desc, b_desc, group_list, group_out, group_size: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    block_m: tl.constexpr = BLOCK_M
    block_k: tl.constexpr = BLOCK_K
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    for group_idx in tl.range(group_size):
        group_end = tl.load(group_list + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, block_m)
        current_problem_end = last_problem_end + num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = tl.cdiv(current_problem_end - tile_idx, total_grid)
            for _ in tl.range(loop_count):
                group_tile_idx = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(group_tile_idx, m, N, block_m, BLOCK_N, GROUP_M)
                offs_am = group_start + tile_m_idx * block_m
                offs_bn = tile_n_idx * BLOCK_N
                offs_bk = group_idx * K
                accumulator_lo = tl.zeros((128, BLOCK_N), dtype=tl.float32)
                accumulator_hi = tl.zeros((128, BLOCK_N), dtype=tl.float32)
                for k in tl.range(0, tl.cdiv(K, block_k)):
                    b = b_desc.load([offs_bk + k * block_k, offs_bn])
                    a_lo = a_desc.load([offs_am, k * block_k])
                    a_hi = a_desc.load([offs_am + 128, k * block_k])
                    accumulator_lo = tl.dot(a_lo, b, acc=accumulator_lo, out_dtype=tl.float32)
                    accumulator_hi = tl.dot(a_hi, b, acc=accumulator_hi, out_dtype=tl.float32)
                offs_m = tl.arange(0, 128)
                offs_n = tl.arange(0, BLOCK_N)
                c_tile_base = group_out + offs_am.to(tl.int64) * N + offs_bn
                c_ptrs = c_tile_base + N * offs_m[:, None] + offs_n[None, :]
                tl.store(c_ptrs, accumulator_lo.to(tl.bfloat16), mask=(offs_am + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
                tl.store(c_ptrs + 128 * N, accumulator_hi.to(tl.bfloat16), mask=(offs_am + 128 + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
                tile_idx += total_grid
        last_problem_end = current_problem_end
        group_start = group_end


def matmul_tma_m256_set_block_size_hook(nargs):
    nargs["a_desc"].block_shape = [128, nargs["BLOCK_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_K"], nargs["BLOCK_N"]]


for config in runtime.get_tuned_config("group_bfgemm_m256"):
    config.pre_hook = matmul_tma_m256_set_block_size_hook
    config.kwargs["enable_backend_opt"] = True
    config.kwargs["llc_options"] = GROUP_BFGEMM_LLC_OPTIONS


@triton.jit
def grouped_bfgemm_tma_m256_k64_kernel(M, N: tl.constexpr, K: tl.constexpr, a_desc, b_desc, group_list, group_out, group_size: tl.constexpr):
    block_m: tl.constexpr = 256
    block_k: tl.constexpr = 64
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, 256)
    group_start = 0
    last_problem_end = 0
    for group_idx in tl.range(group_size):
        group_end = tl.load(group_list + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, block_m)
        current_problem_end = last_problem_end + num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = tl.cdiv(current_problem_end - tile_idx, total_grid)
            for _ in tl.range(loop_count):
                group_tile_idx = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(group_tile_idx, m, N, block_m, 256, 1)
                offs_am = group_start + tile_m_idx * block_m
                offs_bn = tile_n_idx * 256
                offs_bk = group_idx * K
                accumulator_lo = tl.zeros((128, 256), dtype=tl.float32)
                accumulator_hi = tl.zeros((128, 256), dtype=tl.float32)
                for tile_k in tl.range(0, tl.cdiv(K, block_k)):
                    b = b_desc.load([offs_bk + tile_k * block_k, offs_bn])
                    a_lo = a_desc.load([offs_am, tile_k * block_k])
                    a_hi = a_desc.load([offs_am + 128, tile_k * block_k])
                    accumulator_lo = tl.dot(a_lo, b, acc=accumulator_lo, out_dtype=tl.float32)
                    accumulator_hi = tl.dot(a_hi, b, acc=accumulator_hi, out_dtype=tl.float32)
                offs_m = tl.arange(0, 128)
                offs_n = tl.arange(0, 256)
                c_tile_base = group_out + offs_am.to(tl.int64) * N + offs_bn
                c_ptrs = c_tile_base + N * offs_m[:, None] + offs_n[None, :]
                tl.store(c_ptrs, accumulator_lo.to(tl.bfloat16), mask=(offs_am + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
                tl.store(c_ptrs + 128 * N, accumulator_hi.to(tl.bfloat16), mask=(offs_am + 128 + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
                tile_idx += total_grid
        last_problem_end = current_problem_end
        group_start = group_end


@triton.jit
def grouped_bfgemm_tma_schedule_m256_kernel(M, N: tl.constexpr, K: tl.constexpr, a_desc, b_desc, group_out, b_row_starts, c_row_ends, c_row_starts, num_m_blocks):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, 256)
    max_tiles = num_m_blocks * num_n_tiles
    loop_count = tl.cdiv(max_tiles - tile_idx, total_grid)
    for _ in tl.range(0, loop_count):
        tile_m_idx = tile_idx // num_n_tiles
        tile_n_idx = tile_idx - tile_m_idx * num_n_tiles
        offs_am = tl.load(c_row_starts + tile_m_idx).to(tl.int32)
        group_end = tl.load(c_row_ends + tile_m_idx).to(tl.int32)
        offs_bn = tile_n_idx * 256
        offs_bk = tl.load(b_row_starts + tile_m_idx).to(tl.int32)
        accumulator_lo = tl.zeros((128, 256), dtype=tl.float32)
        accumulator_hi = tl.zeros((128, 256), dtype=tl.float32)
        for tile_k in tl.range(0, tl.cdiv(K, 64)):
            b = b_desc.load([offs_bk + tile_k * 64, offs_bn])
            a_lo = a_desc.load([offs_am, tile_k * 64])
            a_hi = a_desc.load([offs_am + 128, tile_k * 64])
            accumulator_lo = tl.dot(a_lo, b, acc=accumulator_lo, out_dtype=tl.float32)
            accumulator_hi = tl.dot(a_hi, b, acc=accumulator_hi, out_dtype=tl.float32)
        offs_m = tl.arange(0, 128)
        offs_n = tl.arange(0, 256)
        c_tile_base = group_out + offs_am.to(tl.int64) * N + offs_bn
        c_ptrs = c_tile_base + N * offs_m[:, None] + offs_n[None, :]
        if N % 256 == 0:
            tl.store(c_ptrs, accumulator_lo.to(tl.bfloat16), mask=offs_am + offs_m[:, None] < group_end)
            tl.store(c_ptrs + 128 * N, accumulator_hi.to(tl.bfloat16), mask=offs_am + 128 + offs_m[:, None] < group_end)
        else:
            tl.store(c_ptrs, accumulator_lo.to(tl.bfloat16), mask=(offs_am + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
            tl.store(c_ptrs + 128 * N, accumulator_hi.to(tl.bfloat16), mask=(offs_am + 128 + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
        tile_idx += total_grid


@triton.jit
def grouped_bfgemm_tma_schedule_m256_n128_kernel(M, N: tl.constexpr, K: tl.constexpr, a_desc, b_desc, group_out, b_row_starts, c_row_ends, c_row_starts, num_m_blocks):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, 128)
    max_tiles = num_m_blocks * num_n_tiles
    loop_count = tl.cdiv(max_tiles - tile_idx, total_grid)
    for _ in tl.range(0, loop_count):
        tile_m_idx = tile_idx // num_n_tiles
        tile_n_idx = tile_idx - tile_m_idx * num_n_tiles
        offs_am = tl.load(c_row_starts + tile_m_idx).to(tl.int32)
        group_end = tl.load(c_row_ends + tile_m_idx).to(tl.int32)
        offs_bn = tile_n_idx * 128
        offs_bk = tl.load(b_row_starts + tile_m_idx).to(tl.int32)
        accumulator_lo = tl.zeros((128, 128), dtype=tl.float32)
        accumulator_hi = tl.zeros((128, 128), dtype=tl.float32)
        for tile_k in tl.range(0, tl.cdiv(K, 64)):
            b = b_desc.load([offs_bk + tile_k * 64, offs_bn])
            a_lo = a_desc.load([offs_am, tile_k * 64])
            a_hi = a_desc.load([offs_am + 128, tile_k * 64])
            accumulator_lo = tl.dot(a_lo, b, acc=accumulator_lo, out_dtype=tl.float32)
            accumulator_hi = tl.dot(a_hi, b, acc=accumulator_hi, out_dtype=tl.float32)
        offs_m = tl.arange(0, 128)
        offs_n = tl.arange(0, 128)
        c_tile_base = group_out + offs_am.to(tl.int64) * N + offs_bn
        c_ptrs = c_tile_base + N * offs_m[:, None] + offs_n[None, :]
        tl.store(c_ptrs, accumulator_lo.to(tl.bfloat16), mask=(offs_am + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
        tl.store(c_ptrs + 128 * N, accumulator_hi.to(tl.bfloat16), mask=(offs_am + 128 + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
        tile_idx += total_grid


@triton.jit
def grouped_bfgemm_tma_expert_m256_kernel(M, N: tl.constexpr, K: tl.constexpr, a_desc, b_desc, group_list, group_out, group_size: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    block_m: tl.constexpr = 256
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tiles_per_expert = tl.cdiv(M, group_size * block_m)
    tiles_per_group = tiles_per_expert * num_n_tiles
    pid = tl.program_id(axis=0)
    expert_idx = pid // tiles_per_group
    if expert_idx < group_size:
        group_pid = pid - expert_idx * tiles_per_group
        tile_m_start = group_pid // num_n_tiles
        tile_n_idx = group_pid - tile_m_start * num_n_tiles
        group_end = tl.load(group_list + expert_idx).to(tl.int32)
        group_start = tl.load(group_list + expert_idx - 1, mask=expert_idx > 0, other=0).to(tl.int32)
        expert_m = group_end - group_start
        num_m_tiles = tl.cdiv(expert_m, block_m)
        for tile_m_idx in tl.range(tile_m_start, num_m_tiles, tiles_per_expert):
            offs_am = group_start + tile_m_idx * block_m
            offs_bn = tile_n_idx * BLOCK_N
            offs_bk = expert_idx * K
            accumulator_lo = tl.zeros((128, BLOCK_N), dtype=tl.float32)
            accumulator_hi = tl.zeros((128, BLOCK_N), dtype=tl.float32)
            for tile_k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                b = b_desc.load([offs_bk + tile_k * BLOCK_K, offs_bn])
                a_lo = a_desc.load([offs_am, tile_k * BLOCK_K])
                a_hi = a_desc.load([offs_am + 128, tile_k * BLOCK_K])
                accumulator_lo = tl.dot(a_lo, b, acc=accumulator_lo, out_dtype=tl.float32)
                accumulator_hi = tl.dot(a_hi, b, acc=accumulator_hi, out_dtype=tl.float32)
            offs_m = tl.arange(0, 128)
            offs_n = tl.arange(0, BLOCK_N)
            c_ptrs = group_out + (offs_am + offs_m[:, None]).to(tl.int64) * N + offs_bn + offs_n[None, :]
            tl.store(c_ptrs, accumulator_lo.to(tl.bfloat16), mask=(offs_am + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))
            tl.store(c_ptrs + 128 * N, accumulator_hi.to(tl.bfloat16), mask=(offs_am + 128 + offs_m[:, None] < group_end) & (offs_bn + offs_n[None, :] < N))


@triton.jit
def grouped_bfgemm_schedule_offsets_kernel(group_list, group_block_offsets, block_chunk_counts, group_size: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_M: tl.constexpr):
    expert_idx = tl.program_id(axis=0) * BLOCK_E + tl.arange(0, BLOCK_E)
    valid = expert_idx < group_size
    group_end = tl.load(group_list + expert_idx, mask=valid, other=0).to(tl.int32)
    group_start = tl.load(group_list + expert_idx - 1, mask=valid & (expert_idx > 0), other=0).to(tl.int32)
    block_count = tl.where(valid, tl.cdiv(group_end - group_start, BLOCK_M), 0)
    block_end = tl.cumsum(block_count, axis=0)
    tl.store(group_block_offsets + expert_idx, block_end - block_count, mask=valid)
    tl.store(block_chunk_counts + tl.program_id(axis=0), tl.sum(block_count, axis=0))


@triton.jit
def grouped_bfgemm_schedule_chunk_offsets_kernel(block_chunk_counts, block_chunk_offsets, total_blocks, num_chunks: tl.constexpr, BLOCK_C: tl.constexpr):
    chunk_idx = tl.arange(0, BLOCK_C)
    valid = chunk_idx < num_chunks
    chunk_count = tl.load(block_chunk_counts + chunk_idx, mask=valid, other=0).to(tl.int32)
    chunk_end = tl.cumsum(chunk_count, axis=0)
    tl.store(block_chunk_offsets + chunk_idx, chunk_end - chunk_count, mask=valid)
    tl.store(total_blocks, tl.sum(chunk_count, axis=0))


@triton.jit
def grouped_bfgemm_schedule_fill_kernel(group_list, group_block_offsets, block_chunk_offsets, b_row_starts, c_row_ends, c_row_starts, K: tl.constexpr, group_size: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_M: tl.constexpr):
    expert_idx = tl.program_id(axis=0)
    group_end = tl.load(group_list + expert_idx).to(tl.int32)
    group_start = tl.load(group_list + expert_idx - 1, mask=expert_idx > 0, other=0).to(tl.int32)
    block_count = tl.cdiv(group_end - group_start, BLOCK_M)
    block_offset = tl.load(group_block_offsets + expert_idx).to(tl.int32) + tl.load(block_chunk_offsets + expert_idx // BLOCK_E).to(tl.int32)
    for block_idx in tl.range(0, block_count):
        schedule_idx = block_offset + block_idx
        tl.store(b_row_starts + schedule_idx, expert_idx * K)
        tl.store(c_row_ends + schedule_idx, group_end)
        tl.store(c_row_starts + schedule_idx, group_start + block_idx * BLOCK_M)


@triton.jit
def grouped_bfgemm_pack_a_kernel(group_A, packed_A, c_row_starts, total_blocks, M, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    schedule_idx = tl.program_id(axis=0)
    tile_k = tl.program_id(axis=1)
    num_blocks = tl.load(total_blocks).to(tl.int32)
    if schedule_idx < num_blocks:
        source_row = tl.load(c_row_starts + schedule_idx).to(tl.int32)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
        values = tl.load(group_A + (source_row + offs_m[:, None]) * K + offs_k[None, :], mask=(source_row + offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        tl.store(packed_A + (schedule_idx * BLOCK_M + offs_m[:, None]) * K + offs_k[None, :], values, mask=offs_k[None, :] < K)


@triton.jit
def _grouped_bfgemm_rasterization_2d_column(block_idx, grid_x: tl.constexpr, grid_y, PANEL_WIDTH: tl.constexpr):
    panel_size = PANEL_WIDTH * grid_y
    panel_idx = block_idx // panel_size
    panel_offset = block_idx % panel_size
    if grid_x % PANEL_WIDTH == 0:
        width: tl.constexpr = PANEL_WIDTH
    else:
        full_panels_size = grid_x // PANEL_WIDTH * panel_size
        residual_panel_width: tl.constexpr = grid_x % PANEL_WIDTH
        width = tl.where(block_idx >= full_panels_size, residual_panel_width, PANEL_WIDTH)
    row_idx = panel_offset // width
    mini_x = panel_offset % width
    mini_x = tl.where(row_idx % 2 == 1, width - 1 - mini_x, mini_x)
    row_idx = tl.where(panel_idx % 2 == 1, grid_y - 1 - row_idx, row_idx)
    col_idx = panel_idx * PANEL_WIDTH + mini_x
    return col_idx, row_idx


@triton.jit
def _grouped_bfgemm_tle_ws_consumer(a_reader, b_reader, valid, tile_m, tile_n, c_row_starts, c_row_ends, group_out, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PIPELINE_STAGES: tl.constexpr):
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    if valid:
        row_start = tl.load(c_row_starts + tile_m).to(tl.int32)
        group_end = tl.load(c_row_ends + tile_m).to(tl.int32)
        offs_m = row_start + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_iter in tl.range(0, k_tiles, num_stages=1, loop_unroll_factor=k_tiles):
            a_wait = a_reader.wait(k_iter)
            b_wait = b_reader.wait(k_iter)
            accumulator = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, accumulator)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            a_reader.release(k_iter)
            b_reader.release(k_iter)
        c_ptrs = group_out + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        if N % BLOCK_N != 0:
            tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=(offs_m[:, None] < group_end) & (offs_n[None, :] < N))
        elif row_start + BLOCK_M <= group_end:
            tl.store(c_ptrs, accumulator.to(tl.bfloat16))
        else:
            tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=offs_m[:, None] < group_end)


@triton.jit
def _grouped_bfgemm_tle_ws_producer(a_writer, b_writer, valid, tile_m, tile_n, a_desc, b_desc, b_row_starts, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    if valid:
        row_start = tile_m * BLOCK_M
        b_row_start = tl.load(b_row_starts + tile_m).to(tl.int32)
        for k_iter in tl.range(0, k_tiles, num_stages=1, loop_unroll_factor=k_tiles):
            offset_k = k_iter * BLOCK_K
            a_slot = a_writer.acquire(k_iter)
            b_slot = b_writer.acquire(k_iter)
            tle.gpu.copy(a_desc, a_slot.a, (BLOCK_M, BLOCK_K), (row_start, offset_k))
            tle.gpu.copy(b_desc, b_slot.b, (BLOCK_K, BLOCK_N), (b_row_start + offset_k, tile_n * BLOCK_N))
            a_writer.commit(k_iter)
            b_writer.commit(k_iter)


@triton.jit
def grouped_bfgemm_tle_ws_kernel(M, N: tl.constexpr, K: tl.constexpr, a_desc, b_desc, group_list, group_out, group_size: tl.constexpr, b_row_starts, c_row_ends, c_row_starts, num_m_blocks, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PANEL_WIDTH: tl.constexpr, PIPELINE_STAGES: tl.constexpr):
    a_smem = tle.gpu.alloc((PIPELINE_STAGES, BLOCK_M, BLOCK_K), dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    b_smem = tle.gpu.alloc((PIPELINE_STAGES, BLOCK_K, BLOCK_N), dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    a_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="group_bfgemm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="group_bfgemm_b", b=b_smem)
    num_n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)
    block_idx = tl.program_id(axis=0)
    tile_n, tile_m = _grouped_bfgemm_rasterization_2d_column(block_idx, num_n_blocks, num_m_blocks, PANEL_WIDTH)
    tle.gpu.warp_specialize([(_grouped_bfgemm_tle_ws_consumer, (a_pipe.reader(), b_pipe.reader(), True, tile_m, tile_n, c_row_starts, c_row_ends, group_out, N, K, BLOCK_M, BLOCK_N, BLOCK_K, PIPELINE_STAGES)), (_grouped_bfgemm_tle_ws_producer, (a_pipe.writer(), b_pipe.writer(), True, tile_m, tile_n, a_desc, b_desc, b_row_starts, K, BLOCK_M, BLOCK_N, BLOCK_K))], worker_num_warps=[4], worker_num_regs=[24])


@triton.jit
def _grouped_bfgemm_tle_ws_n192_consumer(a_reader, b0_reader, b1_reader, tile_m, c_row_starts, c_row_ends, group_out, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, PIPELINE_STAGES: tl.constexpr):
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    row_start = tl.load(c_row_starts + tile_m).to(tl.int32)
    group_end = tl.load(c_row_ends + tile_m).to(tl.int32)
    accumulator_0 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    accumulator_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    for k_iter in tl.range(0, k_tiles, num_stages=1, loop_unroll_factor=k_tiles):
        a_wait = a_reader.wait(k_iter)
        b0_wait = b0_reader.wait(k_iter)
        b1_wait = b1_reader.wait(k_iter)
        accumulator_0 = tle.gpu.wgmma(a_wait.slot.a, b0_wait.slot.b, accumulator_0)
        accumulator_1 = tle.gpu.wgmma(a_wait.slot.a, b1_wait.slot.b, accumulator_1)
        accumulator_0 = tle.gpu.wgmma_wait(0, accumulator_0)
        accumulator_1 = tle.gpu.wgmma_wait(0, accumulator_1)
        a_reader.release(k_iter)
        b0_reader.release(k_iter)
        b1_reader.release(k_iter)
    offs_m = row_start + tl.arange(0, BLOCK_M)
    c_ptrs_0 = group_out + offs_m[:, None].to(tl.int64) * 192 + tl.arange(0, 128)[None, :]
    c_ptrs_1 = group_out + offs_m[:, None].to(tl.int64) * 192 + 128 + tl.arange(0, 64)[None, :]
    tl.store(c_ptrs_0, accumulator_0.to(tl.bfloat16), mask=offs_m[:, None] < group_end)
    tl.store(c_ptrs_1, accumulator_1.to(tl.bfloat16), mask=offs_m[:, None] < group_end)


@triton.jit
def _grouped_bfgemm_tle_ws_n192_producer(a_writer, b0_writer, b1_writer, tile_m, a_desc, b0_desc, b1_desc, b_row_starts, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    row_start = tile_m * BLOCK_M
    b_row_start = tl.load(b_row_starts + tile_m).to(tl.int32)
    for k_iter in tl.range(0, k_tiles, num_stages=1, loop_unroll_factor=k_tiles):
        offset_k = k_iter * BLOCK_K
        a_slot = a_writer.acquire(k_iter)
        b0_slot = b0_writer.acquire(k_iter)
        b1_slot = b1_writer.acquire(k_iter)
        tle.gpu.copy(a_desc, a_slot.a, (BLOCK_M, BLOCK_K), (row_start, offset_k))
        tle.gpu.copy(b0_desc, b0_slot.b, (BLOCK_K, 128), (b_row_start + offset_k, 0))
        tle.gpu.copy(b1_desc, b1_slot.b, (BLOCK_K, 64), (b_row_start + offset_k, 128))
        a_writer.commit(k_iter)
        b0_writer.commit(k_iter)
        b1_writer.commit(k_iter)


@triton.jit
def grouped_bfgemm_tle_ws_n192_kernel(M, K: tl.constexpr, a_desc, b0_desc, b1_desc, group_out, b_row_starts, c_row_ends, c_row_starts, num_m_blocks, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, PIPELINE_STAGES: tl.constexpr):
    a_smem = tle.gpu.alloc((PIPELINE_STAGES, BLOCK_M, BLOCK_K), dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    b0_smem = tle.gpu.alloc((PIPELINE_STAGES, BLOCK_K, 128), dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    b1_smem = tle.gpu.alloc((PIPELINE_STAGES, BLOCK_K, 64), dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    a_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="group_bfgemm_n192_a", a=a_smem)
    b0_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="group_bfgemm_n192_b0", b=b0_smem)
    b1_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="group_bfgemm_n192_b1", b=b1_smem)
    tile_m = tl.program_id(axis=0)
    tle.gpu.warp_specialize([(_grouped_bfgemm_tle_ws_n192_consumer, (a_pipe.reader(), b0_pipe.reader(), b1_pipe.reader(), tile_m, c_row_starts, c_row_ends, group_out, K, BLOCK_M, BLOCK_K, PIPELINE_STAGES)), (_grouped_bfgemm_tle_ws_n192_producer, (a_pipe.writer(), b0_pipe.writer(), b1_pipe.writer(), tile_m, a_desc, b0_desc, b1_desc, b_row_starts, K, BLOCK_M, BLOCK_K))], worker_num_warps=[4], worker_num_regs=[24])


@triton.jit
def grouped_hgemm_kernel(
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
    SPLIT_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    group_stride = SPLIT_M * num_n_tiles
    group_idx = pid // group_stride
    group_pid = pid - group_idx * group_stride
    split_m = group_pid // num_n_tiles
    tile_n = group_pid - split_m * num_n_tiles
    group_start = tl.load(
        group_list + group_idx - 1,
        mask=group_idx > 0,
        other=0,
    ).to(tl.int64)
    group_end = tl.load(group_list + group_idx).to(tl.int64)
    group_m = group_end - group_start
    num_m_tiles = tl.cdiv(group_m, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    local_m = split_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_start = group_start + split_m * BLOCK_M
    offs_m = group_start + local_m
    mask_m = local_m < group_m
    if split_m < num_m_tiles:
        a_ptrs = tl.make_block_ptr(
            base=group_A,
            shape=(M, K),
            strides=(K, 1),
            offsets=(row_start.to(tl.int32), 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_ptrs = tl.make_block_ptr(
            base=group_B + group_idx * K * N,
            shape=(K, N),
            strides=(N, 1),
            offsets=(0, tile_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for tile_k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(b_ptrs, boundary_check=(0, 1), padding_option="zero")
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
            a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
            b_ptrs = tl.advance(b_ptrs, (BLOCK_K, 0))

        tl.store(
            group_out + offs_m[:, None] * N + offs_n[None, :],
            accumulator.to(tl.float16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def group_bfgemm(group_A, group_B, group_list, group_out):
    M, K = group_A.shape
    group_size, _, N = group_B.shape
    num_aicores = torch.musa.get_device_properties(group_A.device).multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(group_A, group_A.shape, group_A.stride(), dummy_block)
    b_desc = TensorDescriptor(group_B, [group_size * K, N], [group_B.stride(1), group_B.stride(2)], dummy_block)
    if K >= 64 and (N % 256 == 0 or N in (192, 384)) and 16 <= group_size <= 128:
        block_m = 256
        max_blocks = triton.cdiv(M, block_m) + group_size - 1
        block_e = 128
        num_chunks = triton.cdiv(group_size, block_e)
        group_block_offsets = torch.empty(group_size, dtype=torch.int32, device=group_A.device)
        block_chunk_counts = torch.empty(num_chunks, dtype=torch.int32, device=group_A.device)
        block_chunk_offsets = torch.empty(num_chunks, dtype=torch.int32, device=group_A.device)
        b_row_starts = torch.empty(max_blocks, dtype=torch.int32, device=group_A.device)
        c_row_ends = torch.empty(max_blocks, dtype=torch.int32, device=group_A.device)
        c_row_starts = torch.empty(max_blocks, dtype=torch.int32, device=group_A.device)
        total_blocks = torch.empty((), dtype=torch.int32, device=group_A.device)
        packed_A = torch.empty((max_blocks * block_m, K), dtype=group_A.dtype, device=group_A.device)
        grouped_bfgemm_schedule_offsets_kernel[(num_chunks,)](group_list, group_block_offsets, block_chunk_counts, group_size, BLOCK_E=block_e, BLOCK_M=block_m, num_warps=1)
        grouped_bfgemm_schedule_chunk_offsets_kernel[(1,)](block_chunk_counts, block_chunk_offsets, total_blocks, num_chunks, BLOCK_C=triton.next_power_of_2(num_chunks), num_warps=1)
        schedule_blocks = int(total_blocks.item())
        grouped_bfgemm_schedule_fill_kernel[(group_size,)](group_list, group_block_offsets, block_chunk_offsets, b_row_starts, c_row_ends, c_row_starts, K, group_size, BLOCK_E=block_e, BLOCK_M=block_m, num_warps=1)
        grouped_bfgemm_pack_a_kernel[(schedule_blocks, triton.cdiv(K, 64))](group_A, packed_A, c_row_starts, total_blocks, M, K, BLOCK_M=block_m, BLOCK_K=64, num_warps=8)
        packed_a_desc = TensorDescriptor(packed_A, packed_A.shape, packed_A.stride(), dummy_block)
        b_n64_desc = TensorDescriptor(group_B, [group_size * K, N], [group_B.stride(1), group_B.stride(2)], dummy_block)
        block_n = 256
        block_k = 32
        if N == 192:
            pipeline_stages = 6
        elif N == 384:
            pipeline_stages = 5
        elif (K == 768 and group_size >= 32) or (K == 512 and 32 <= group_size <= 64):
            pipeline_stages = 4
        elif K in (384, 512, 768):
            pipeline_stages = 3
        elif K == 192:
            pipeline_stages = 6
        else:
            pipeline_stages = 4
        if K == 2048 and N == 1536 and (group_size >= 64 or group_size == 16):
            panel_width = 6
        elif K == 192 and N >= 4096:
            panel_width = 16
        elif K <= 192 or (group_size <= 17 and K == 4096 and N >= 3072):
            panel_width = 8
        elif K <= 768 or (group_size <= 17 and N >= 4096) or (K == 2048 and N == 1024 and group_size >= 64):
            panel_width = 4
        elif K == 2048 and N == 768:
            panel_width = 3
        else:
            panel_width = 2
        consumer_warps = 16
        packed_a_desc.block_shape = [block_m, block_k]
        if N == 192:
            b_desc.block_shape = [block_k, 128]
            b_n64_desc.block_shape = [block_k, 64]
            grouped_bfgemm_tle_ws_n192_kernel[(schedule_blocks,)](M, K, packed_a_desc, b_desc, b_n64_desc, group_out, b_row_starts, c_row_ends, c_row_starts, schedule_blocks, BLOCK_M=block_m, BLOCK_K=block_k, PIPELINE_STAGES=pipeline_stages, num_warps=consumer_warps, num_stages=pipeline_stages)
        else:
            b_desc.block_shape = [block_k, block_n]
            max_tiles = schedule_blocks * triton.cdiv(N, block_n)
            grouped_bfgemm_tle_ws_kernel[(max_tiles,)](M, N, K, packed_a_desc, b_desc, group_list, group_out, group_size, b_row_starts, c_row_ends, c_row_starts, schedule_blocks, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, PANEL_WIDTH=panel_width, PIPELINE_STAGES=pipeline_stages, num_warps=consumer_warps, num_stages=pipeline_stages)
    elif group_size == 512 and K >= 64 and N >= 128:
        block_m = 256
        max_blocks = triton.cdiv(M, block_m) + group_size - 1
        block_e = 128
        num_chunks = triton.cdiv(group_size, block_e)
        group_block_offsets = torch.empty(group_size, dtype=torch.int32, device=group_A.device)
        block_chunk_counts = torch.empty(num_chunks, dtype=torch.int32, device=group_A.device)
        block_chunk_offsets = torch.empty(num_chunks, dtype=torch.int32, device=group_A.device)
        b_row_starts = torch.empty(max_blocks, dtype=torch.int32, device=group_A.device)
        c_row_ends = torch.empty(max_blocks, dtype=torch.int32, device=group_A.device)
        c_row_starts = torch.empty(max_blocks, dtype=torch.int32, device=group_A.device)
        total_blocks = torch.empty((), dtype=torch.int32, device=group_A.device)
        grouped_bfgemm_schedule_offsets_kernel[(num_chunks,)](group_list, group_block_offsets, block_chunk_counts, group_size, BLOCK_E=block_e, BLOCK_M=block_m, num_warps=1)
        grouped_bfgemm_schedule_chunk_offsets_kernel[(1,)](block_chunk_counts, block_chunk_offsets, total_blocks, num_chunks, BLOCK_C=triton.next_power_of_2(num_chunks), num_warps=1)
        schedule_blocks = int(total_blocks.item())
        grouped_bfgemm_schedule_fill_kernel[(group_size,)](group_list, group_block_offsets, block_chunk_offsets, b_row_starts, c_row_ends, c_row_starts, K, group_size, BLOCK_E=block_e, BLOCK_M=block_m, num_warps=1)
        a_desc.block_shape = [128, 64]
        if N == 128:
            b_desc.block_shape = [64, 128]
            grouped_bfgemm_tma_schedule_m256_n128_kernel[(num_aicores,)](M, N, K, a_desc, b_desc, group_out, b_row_starts, c_row_ends, c_row_starts, schedule_blocks, num_warps=8, num_stages=3, enable_backend_opt=True, llc_options=GROUP_BFGEMM_LLC_OPTIONS)
        else:
            b_desc.block_shape = [64, 256]
            grouped_bfgemm_tma_schedule_m256_kernel[(num_aicores,)](M, N, K, a_desc, b_desc, group_out, b_row_starts, c_row_ends, c_row_starts, schedule_blocks, num_warps=16, num_stages=2 if K == 64 else 3, enable_backend_opt=True, llc_options=GROUP_BFGEMM_LLC_OPTIONS)
    elif group_size <= 17:
        a_desc.block_shape = [128, 64]
        b_desc.block_shape = [64, 256]
        tiles_per_expert = triton.cdiv(M, group_size * 256)
        grouped_bfgemm_tma_expert_m256_kernel[(group_size * tiles_per_expert * triton.cdiv(N, 256),)](M, N, K, a_desc, b_desc, group_list, group_out, group_size, BLOCK_N=256, BLOCK_K=64, num_warps=16, num_stages=3, enable_backend_opt=True, llc_options=GROUP_BFGEMM_LLC_OPTIONS)
    elif K == 64 and N >= 128 and group_size >= 32:
        a_desc.block_shape = [128, 64]
        b_desc.block_shape = [64, 256]
        grouped_bfgemm_tma_m256_k64_kernel[(num_aicores,)](M, N, K, a_desc, b_desc, group_list, group_out, group_size, num_warps=32, num_stages=1, enable_backend_opt=True, llc_options=GROUP_BFGEMM_LLC_OPTIONS)
    elif K >= 64 and N >= 128 and (group_size >= 32 or K <= 768):
        grouped_bfgemm_tma_m256_kernel[(num_aicores,)](M, N, K, a_desc, b_desc, group_list, group_out, group_size)
    else:
        grouped_bfgemm_tma_kernel[(num_aicores,)](
            M,
            N,
            K,
            a_desc,
            b_desc,
            group_list,
            group_out,
            group_size,
        )
    return group_out


def group_hgemm(group_A, group_B, group_list, group_out):
    M, K = group_A.shape
    group_size, _, N = group_B.shape
    split_m = 32
    grouped_hgemm_kernel[
        lambda meta: (group_size * split_m * triton.cdiv(N, meta["BLOCK_N"]),)
    ](
        M,
        N,
        K,
        group_A,
        group_B,
        group_list,
        group_out,
        group_size,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        SPLIT_M=split_m,
        num_warps=4,
        num_stages=2,
    )
    return group_out


def group_tf32gemm(group_A, group_B, group_list, group_out):
    raise RuntimeError(
        "MThreads Triton backend does not support the FP32 dot lowering required by group_tf32gemm"
    )


def group_mm(A: torch.Tensor, B: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    output = A.new_empty((A.shape[0], B.shape[2]))
    start = 0
    for group_idx, end in enumerate(offs.tolist()):
        output[start:end].copy_(torch.mm(A[start:end], B[group_idx]))
        start = end
    return output

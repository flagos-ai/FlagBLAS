from typing import Generator

import numpy as np
import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils

# ---------------------------------------------------------------------------
# Shape space
# ---------------------------------------------------------------------------
GROUP_GEMM_CONFIGS = [
    (2048, 128, 1536),
    (768, 128, 2048),
    (2048, 128, 768),
    (384, 128, 2048),
    (2048, 128, 384),
    (192, 128, 2048),
    (2048, 128, 192),
    (96, 128, 2048),
    (2048, 64, 1536),
    (768, 64, 2048),
    (2048, 32, 1536),
    (768, 32, 2048),
    (2048, 16, 1536),
    (768, 16, 2048),
    (4096, 128, 384),
    (192, 128, 4096),
    (4096, 16, 3072),
    (1536, 16, 4096),
    (7168, 16, 4096),
    (2048, 16, 7168),
    (7168, 17, 4096),
    (2048, 17, 7168),
    (2048, 512, 256),
    (128, 512, 2048),
    (2048, 512, 128),
    (64, 512, 2048),
    (2048, 128, 1024), 
    (512, 128, 2048),
    (2048, 64, 1024),
    (512, 64, 2048),
]


M_VALUES = list(range(1, 33)) + [64, 128, 256, 512, 1024, 2048, 4096]

# ---------------------------------------------------------------------------
# cuBLAS enums
# ---------------------------------------------------------------------------
CUBLAS_OP_N = 0
CUDA_R_16BF = 14
CUBLAS_COMPUTE_32F_FAST_16BF = 75


def _get_grouped_gemm_fn():
    """Return ``cublasGemmGroupedBatchedEx``, preferring cupy's binding."""
    try:
        from cupy_backends.cuda.libs import cublas

        for name in ("gemmGroupedBatchedEx", "cublasGemmGroupedBatchedEx"):
            fn = getattr(cublas, name, None)
            if fn is not None:
                return fn
    except ImportError:
        pass

    import ctypes

    _lib = ctypes.CDLL("libcublas.so.12")

    def _fallback(
        handle,
        transa,
        transb,
        m_arr,
        n_arr,
        k_arr,
        alpha_ptr,
        d_a_ptrs_ptr,
        atype,
        lda,
        d_b_ptrs_ptr,
        btype,
        ldb,
        beta_ptr,
        d_c_ptrs_ptr,
        ctype,
        ldc,
        group_count,
        group_size,
        compute_type,
    ):
        return _lib.cublasGemmGroupedBatchedEx(
            ctypes.c_void_p(handle),
            transa.ctypes.data_as(ctypes.c_void_p),
            transb.ctypes.data_as(ctypes.c_void_p),
            m_arr.ctypes.data_as(ctypes.c_void_p),
            n_arr.ctypes.data_as(ctypes.c_void_p),
            k_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(alpha_ptr),
            d_a_ptrs_ptr,
            ctypes.c_int(atype),
            lda.ctypes.data_as(ctypes.c_void_p),
            d_b_ptrs_ptr,
            ctypes.c_int(btype),
            ldb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(beta_ptr),
            d_c_ptrs_ptr,
            ctypes.c_int(ctype),
            ldc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(group_count),
            group_size.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(compute_type),
        )

    return _fallback


_grouped_gemm_ex = _get_grouped_gemm_fn()


def _build_offs_table(e, m, n, k):
    offs = []
    for g in range(e):
        start_M = g * m
        start_K = g * k
        offs.append([m, n, k, start_M, start_K, start_M])
    return offs


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def cublas_group_gemm_baseline(
    group_A,
    group_B,
    group_C,
    offs_table,
    alpha,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    """cuBLAS grouped GEMM via ``cublasGemmGroupedBatchedEx``.

    All groups share identical (m, n, k) per *offs_table*, so the call uses
    a single GEMM configuration with batch size *e*.

    Row-major trick: the packed tensors are row-major while cuBLAS expects
    column-major.  We swap A↔B so that cuBLAS computes::

        C = alpha * B^T @ A^T + beta * C^T   (col-major)

    which is equivalent to::

        C = alpha * A @ B + beta * C         (row-major)
    """
    e = len(offs_table)
    if e == 0:
        return torch.empty_like(group_C)

    m_g, n_g, k_g = offs_table[0][0], offs_table[0][1], offs_table[0][2]

    out = torch.empty_like(group_C)

    a_ptrs = []  # cuBLAS "A" → B data (swapped)
    b_ptrs = []  # cuBLAS "B" → A data (swapped)
    c_ptrs = []  # cuBLAS "C" → output

    for entry in offs_table:
        _mg, _ng, _kg, start_M, start_K, start_C = entry
        a_ptrs.append(group_B[start_K : start_K + _kg, :_ng].data_ptr())
        b_ptrs.append(group_A[start_M : start_M + _mg, :_kg].data_ptr())
        c_ptrs.append(out[start_C : start_C + _mg, :_ng].data_ptr())

    # Host arrays — one entry (all groups share the same dims).
    transa = np.array([CUBLAS_OP_N], dtype=np.int32)
    transb = np.array([CUBLAS_OP_N], dtype=np.int32)
    m_arr  = np.array([n_g], dtype=np.int32)    # n because of A↔B swap
    n_arr  = np.array([m_g], dtype=np.int32)    # m because of A↔B swap
    k_arr  = np.array([k_g], dtype=np.int32)
    lda_arr = np.array([n_g], dtype=np.int32)   # ld of B row-major = n_g
    ldb_arr = np.array([k_g], dtype=np.int32)   # ld of A row-major = k_g
    ldc_arr = np.array([n_g], dtype=np.int32)   # ld of C row-major = n_g
    batch   = np.array([e], dtype=np.int32)

    device = group_A.device
    d_a_ptrs = torch.tensor(a_ptrs, dtype=torch.int64, device=device)
    d_b_ptrs = torch.tensor(b_ptrs, dtype=torch.int64, device=device)
    d_c_ptrs = torch.tensor(c_ptrs, dtype=torch.int64, device=device)

    _grouped_gemm_ex(
        handle,
        transa,
        transb,
        m_arr,
        n_arr,
        k_arr,
        alpha_ptr,
        d_a_ptrs.data_ptr(),
        CUDA_R_16BF,
        lda_arr,
        d_b_ptrs.data_ptr(),
        CUDA_R_16BF,
        ldb_arr,
        beta_ptr,
        d_c_ptrs.data_ptr(),
        CUDA_R_16BF,
        ldc_arr,
        1,         # group_count
        batch,
        CUBLAS_COMPUTE_32F_FAST_16BF,
    )

    return out


def gems_group_gemm_wrapper(group_A, group_B, group_C, offs_table, alpha, beta, **kwargs):
    return flag_blas.group_gemm(
        group_A, group_B, group_C, offs_table, alpha=alpha, beta=beta
    )


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
class GroupGemmBenchmark(Benchmark):

    def __init__(self, *args, alpha=1.0, beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [
            (k, e, n, m)
            for (k, e, n) in GROUP_GEMM_CONFIGS
            for m in M_VALUES
        ]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        import cupy as cp
        from cupy_backends.cuda.libs import cublas

        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        cublas.setMathMode(handle, 0)
        torch.backends.cuda.matmul.allow_tf32 = False

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        scale = 1.0
        for (k, e, n, m) in self.shapes:
            group_A = torch.randn(e * m, k, dtype=cur_dtype, device=self.device) * scale
            group_B = torch.randn(e * k, n, dtype=cur_dtype, device=self.device) * scale
            group_C = torch.randn(e * m, n, dtype=cur_dtype, device=self.device) * scale
            offs_table = _build_offs_table(e, m, n, k)

            yield group_A, group_B, group_C, offs_table, {
                "alpha": self.alpha,
                "beta": self.beta,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }

    def get_tflops(self, op, *args, **kwargs):
        offs_table = args[3]
        total_flops = 0
        for entry in offs_table:
            m_g, n_g, k_g = entry[0], entry[1], entry[2]
            total_flops += 2 * m_g * n_g * k_g
        return total_flops

    def get_gbps(self, args, latency):
        group_A, group_B, group_C = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(group_A)
            + shape_utils.size_in_bytes(group_B)
            + 2 * shape_utils.size_in_bytes(group_C)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.group_gemm
def test_perf_group_gemm_bf16():
    bench = GroupGemmBenchmark(
        op_name="group_gemm",
        torch_op=cublas_group_gemm_baseline,
        gems_op=gems_group_gemm_wrapper,
        dtypes=[torch.bfloat16],
        alpha=1.0,
        beta=0.0,
    )
    bench.run()

import random

import pytest
import torch

import flag_blas

from . import accuracy_utils as utils
from .conftest import TO_CPU

IS_ASCEND = flag_blas.vendor_name == "ascend"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_ASCEND:
    torch_npu = pytest.importorskip("torch_npu")
elif IS_MTHREADS:
    from flag_blas.runtime.backend._mthreads.ops.group_gemm import group_bfgemm
else:
    import ctypes
    import ctypes.util

    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    from flag_blas.ops import CUBLAS_OP_N


def load_cublas():
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)
    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on the system.")


_cublas = load_cublas() if not IS_ASCEND and not IS_MTHREADS else None


def _cublasGemmGroupedBatchedEx(
    handle,
    transa,
    transb,
    m_arr,
    n_arr,
    k_arr,
    alpha,
    a_array,
    a_type,
    lda,
    b_array,
    b_type,
    ldb,
    beta,
    c_array,
    c_type,
    ldc,
    group_count,
    group_size,
    compute_type,
):
    return _cublas.cublasGemmGroupedBatchedEx(
        ctypes.c_void_p(handle),
        ctypes.c_void_p(transa.data_ptr()),
        ctypes.c_void_p(transb.data_ptr()),
        ctypes.c_void_p(m_arr.data_ptr()),
        ctypes.c_void_p(n_arr.data_ptr()),
        ctypes.c_void_p(k_arr.data_ptr()),
        ctypes.c_void_p(alpha),
        ctypes.c_void_p(a_array),
        ctypes.c_int(a_type),
        ctypes.c_void_p(lda.data_ptr()),
        ctypes.c_void_p(b_array),
        ctypes.c_int(b_type),
        ctypes.c_void_p(ldb.data_ptr()),
        ctypes.c_void_p(beta),
        ctypes.c_void_p(c_array),
        ctypes.c_int(c_type),
        ctypes.c_void_p(ldc.data_ptr()),
        ctypes.c_int(group_count),
        ctypes.c_void_p(group_size.data_ptr()),
        ctypes.c_int(compute_type),
    )


if not IS_ASCEND and not IS_MTHREADS:
    cublas.cublasGemmGroupedBatchedEx = _cublasGemmGroupedBatchedEx


CUDA_R_16F = 2
CUDA_R_16BF = 14
CUBLAS_COMPUTE_32F = 0


def _build_offs_table(k, e, n, m_list):
    offs = []
    start_M = 0
    start_K = 0
    for g in range(e):
        mg = m_list[g]
        offs.append([mg, n, k, start_M, start_K, start_M])
        start_M += mg
        start_K += k
    return offs


def _build_triton_arrays(group_A, group_B, group_C, offs_table):
    group_size = len(offs_table)
    M, N = group_C.shape
    K = group_A.shape[1]
    group_out = torch.empty((M, N), device=group_A.device, dtype=group_A.dtype)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    out_addrs = []
    m_sizes = []
    n_sizes = []
    k_sizes = []
    ldas = []
    ldbs = []
    ldcs = []

    for i in range(group_size):
        mg, ng, kg = offs_table[i][0], offs_table[i][1], offs_table[i][2]
        A_g = group_A[offs_table[i][3]]
        B_g = group_B[offs_table[i][4]]
        C_g = group_C[offs_table[i][5]]
        out_g = group_out[offs_table[i][5]]
        m_sizes.append(mg)
        n_sizes.append(ng)
        k_sizes.append(kg)
        ldas.append(kg)
        ldbs.append(ng)
        ldcs.append(ng)
        A_addrs.append(A_g.data_ptr())
        B_addrs.append(B_g.data_ptr())
        C_addrs.append(C_g.data_ptr())
        out_addrs.append(out_g.data_ptr())

    d_a_ptrs = torch.tensor(A_addrs, device=group_A.device)
    d_b_ptrs = torch.tensor(B_addrs, device=group_A.device)
    d_c_ptrs = torch.tensor(C_addrs, device=group_A.device)
    d_output_ptrs = torch.tensor(out_addrs, device=group_A.device)
    d_m_sizes = torch.tensor(m_sizes, dtype=torch.int32, device=group_A.device)
    d_n_sizes = torch.tensor(n_sizes, dtype=torch.int32, device=group_A.device)
    d_k_sizes = torch.tensor(k_sizes, dtype=torch.int32, device=group_A.device)
    d_ldas = torch.tensor(ldas, dtype=torch.int32, device=group_A.device)
    d_ldbs = torch.tensor(ldbs, dtype=torch.int32, device=group_A.device)
    d_ldcs = torch.tensor(ldcs, dtype=torch.int32, device=group_A.device)

    return (
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
    )


def cublas_group_gemm_reference(group_A, group_B, group_C, offs_table, alpha, beta):
    e = len(offs_table)
    if e == 0:
        return torch.empty_like(group_C)

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    cublas.setMathMode(handle, 0)
    torch.backends.cuda.matmul.allow_tf32 = False

    if group_A.dtype == torch.float16:
        cu_dtype = CUDA_R_16F
    else:
        cu_dtype = CUDA_R_16BF

    out = group_C.clone().contiguous()

    a_ptrs = []
    b_ptrs = []
    c_ptrs = []
    m_arr_list = []
    n_arr_list = []
    k_arr_list = []
    lda_list = []
    ldb_list = []
    ldc_list = []

    for entry in offs_table:
        mg, ng, kg, start_M, start_K, start_C = entry
        a_ptrs.append(group_B[start_K : start_K + kg, :].data_ptr())
        b_ptrs.append(group_A[start_M : start_M + mg, :].data_ptr())
        c_ptrs.append(out[start_C : start_C + mg, :].data_ptr())
        m_arr_list.append(ng)
        n_arr_list.append(mg)
        k_arr_list.append(kg)
        lda_list.append(ng)
        ldb_list.append(kg)
        ldc_list.append(ng)

    transa = torch.tensor([CUBLAS_OP_N] * e, dtype=torch.int32)
    transb = torch.tensor([CUBLAS_OP_N] * e, dtype=torch.int32)
    m_arr = torch.tensor(m_arr_list, dtype=torch.int32)
    n_arr = torch.tensor(n_arr_list, dtype=torch.int32)
    k_arr = torch.tensor(k_arr_list, dtype=torch.int32)
    lda_arr = torch.tensor(lda_list, dtype=torch.int32)
    ldb_arr = torch.tensor(ldb_list, dtype=torch.int32)
    ldc_arr = torch.tensor(ldc_list, dtype=torch.int32)
    batch = torch.tensor([1] * e, dtype=torch.int32)
    alpha_arr = torch.full((e,), alpha, dtype=torch.float32)
    beta_arr = torch.full((e,), beta, dtype=torch.float32)

    device = group_A.device
    d_a_ptrs = torch.tensor(a_ptrs, dtype=torch.int64, device=device)
    d_b_ptrs = torch.tensor(b_ptrs, dtype=torch.int64, device=device)
    d_c_ptrs = torch.tensor(c_ptrs, dtype=torch.int64, device=device)

    cublas.cublasGemmGroupedBatchedEx(
        handle,
        transa,
        transb,
        m_arr,
        n_arr,
        k_arr,
        alpha_arr.data_ptr(),
        d_a_ptrs.data_ptr(),
        cu_dtype,
        lda_arr,
        d_b_ptrs.data_ptr(),
        cu_dtype,
        ldb_arr,
        beta_arr.data_ptr(),
        d_c_ptrs.data_ptr(),
        cu_dtype,
        ldc_arr,
        e,
        batch,
        CUBLAS_COMPUTE_32F,
    )

    return out


@pytest.mark.group_gemm
@pytest.mark.parametrize("k,e,n", utils.GROUP_GEMM_SHAPES)
def test_accuracy_group_gemm(k, e, n):
    device = flag_blas.device
    alpha, beta = 1.5, 0.5
    scale = k**-0.5

    m_list = [random.randint(1, 4096) for _ in range(e)]
    total_M = sum(m_list)
    if IS_ASCEND:
        group_A = torch.randn(total_M, k, dtype=torch.bfloat16, device=device) * scale
        group_B = torch.randn(e, k, n, dtype=torch.bfloat16, device=device) * scale
        group_list = torch.tensor(m_list, dtype=torch.int64, device=device).cumsum(0)
        group_ref = torch_npu.npu_grouped_matmul(
            [group_A],
            [group_B],
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=0,
            output_dtype=group_A.dtype,
        )[0]
        group_out = torch.empty_like(group_ref)
        group_out = flag_blas.group_bfgemm(group_A, group_B, group_list, group_out)
        if TO_CPU:
            group_out = group_out.cpu()
            group_ref = group_ref.cpu()
        utils.blas_assert_close(group_out, group_ref, torch.bfloat16, reduce_dim=k)
        return

    if IS_MTHREADS:
        group_A = torch.randn(total_M, k, dtype=torch.bfloat16, device=device)
        group_B = torch.randn(e, k, n, dtype=torch.bfloat16, device=device)
        group_list = torch.tensor(m_list, dtype=torch.int32, device=device).cumsum(0)
        ref = torch.cat(
            [
                torch.mm(group_A[start:end], group_B[group_idx])
                for group_idx, (start, end) in enumerate(
                    zip([0] + group_list[:-1].tolist(), group_list.tolist())
                )
            ],
            dim=0,
        )
        out = flag_blas.group_bfgemm(group_A, group_B, group_list, torch.empty_like(ref))
        utils.blas_assert_close(out, ref, torch.bfloat16, reduce_dim=k, atol=2e-4)
        return

    total_K = e * k
    group_A = (
        torch.randn(total_M, k, dtype=torch.bfloat16, device=device) * scale
    ).contiguous()
    group_B = (
        torch.randn(total_K, n, dtype=torch.bfloat16, device=device) * scale
    ).contiguous()
    group_C = (
        torch.randn(total_M, n, dtype=torch.bfloat16, device=device) * scale
    ).contiguous()
    offs_table = _build_offs_table(k, e, n, m_list)

    if TO_CPU:
        ref_A = group_A.to("cpu").to(torch.float64)
        ref_B = group_B.to("cpu").to(torch.float64)
        ref_C = group_C.to("cpu").clone().to(torch.float64)
        for entry in offs_table:
            m_g, n_g, k_g, start_M, start_K, start_C = entry
            A_sub = ref_A[start_M : start_M + m_g, :k_g]
            B_sub = ref_B[start_K : start_K + k_g, :n_g]
            res = torch.matmul(A_sub, B_sub)
            if beta == 0.0:
                ref_C[start_C : start_C + m_g, :n_g] = alpha * res
            else:
                ref_C[start_C : start_C + m_g, :n_g] = (
                    alpha * res + beta * ref_C[start_C : start_C + m_g, :n_g]
                )
        ref = ref_C.to(torch.bfloat16)
    else:
        ref = cublas_group_gemm_reference(
            group_A, group_B, group_C, offs_table, alpha, beta
        )

    out = flag_blas.group_bfgemm(
        *_build_triton_arrays(group_A, group_B, group_C, offs_table),
        alpha=alpha,
        beta=beta,
    )
    utils.blas_assert_close(out, ref, torch.bfloat16, reduce_dim=k)


@pytest.mark.group_gemm
@pytest.mark.skipif(
    IS_ASCEND or IS_MTHREADS, reason="Hopper-only alpha/beta interface"
)
def test_group_gemm_alpha_zero():
    m, k, e, n = 16, 64, 4, 128
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device).contiguous()
    B = torch.randn(e * k, n, dtype=dtype, device=device).contiguous()
    C = torch.randn(e * m, n, dtype=dtype, device=device).contiguous()
    C_orig = C.clone()
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    out = flag_blas.group_bfgemm(
        *_build_triton_arrays(A, B, C, offs_table), alpha=0.0, beta=2.0
    )

    if TO_CPU:
        utils.blas_assert_close(out, (C_orig * 2.0).to("cpu"), dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(out, C_orig * 2.0, dtype, reduce_dim=k)


@pytest.mark.group_gemm
@pytest.mark.skipif(
    IS_ASCEND or IS_MTHREADS, reason="Hopper-only alpha/beta interface"
)
def test_group_gemm_beta_zero():
    m, k, e, n = 8, 32, 3, 64
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device).contiguous()
    B = torch.randn(e * k, n, dtype=dtype, device=device).contiguous()
    C_zeros = torch.zeros(e * m, n, dtype=dtype, device=device).contiguous()
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    ref = cublas_group_gemm_reference(A, B, C_zeros, offs_table, 1.0, 0.0)
    out = flag_blas.group_bfgemm(
        *_build_triton_arrays(A, B, C_zeros, offs_table), alpha=1.0, beta=0.0
    )

    if TO_CPU:
        utils.blas_assert_close(out, ref.to("cpu"), dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(out, ref, dtype, reduce_dim=k, atol=2e-4)


@pytest.mark.group_gemm
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 1.5)]
)
@pytest.mark.skipif(
    IS_ASCEND or IS_MTHREADS, reason="Hopper-only alpha/beta interface"
)
def test_group_gemm_alpha_beta(alpha, beta):
    m, k, e, n = 32, 128, 2, 128
    dtype, device = torch.bfloat16, flag_blas.device
    scale = k**-0.5
    A = (torch.randn(e * m, k, dtype=dtype, device=device) * scale).contiguous()
    B = (torch.randn(e * k, n, dtype=dtype, device=device) * scale).contiguous()
    C = (torch.randn(e * m, n, dtype=dtype, device=device) * scale).contiguous()
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    ref = cublas_group_gemm_reference(A, B, C, offs_table, alpha, beta)
    out = flag_blas.group_bfgemm(
        *_build_triton_arrays(A, B, C, offs_table), alpha=alpha, beta=beta
    )

    if TO_CPU:
        utils.blas_assert_close(out, ref.to("cpu"), dtype, reduce_dim=k)
    else:
        utils.blas_assert_close(out, ref, dtype, reduce_dim=k)

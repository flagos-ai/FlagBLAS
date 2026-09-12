import ctypes
import ctypes.util
import random
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils

IS_ASCEND = flag_blas.device == "npu"

if IS_ASCEND:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip(
            "requires FlagBLAS with an available NPU backend",
            allow_module_level=True,
        )
    torch.npu.matmul.allow_hf32 = True
    torch.npu.matmul.cube_math_type = torch.npu.CubeMathType.USE_HF32
    if (
        not torch.npu.matmul.allow_hf32
        or torch.npu.matmul.cube_math_type != torch.npu.CubeMathType.USE_HF32
    ):
        raise RuntimeError("Failed to enable Ascend HF32 matmul mode.")
    from flag_blas.runtime.backend._ascend.ops.group_gemm import grouped_tf32gemm_kernel
else:
    if flag_blas.device != "cuda":
        pytest.skip(
            "requires FlagBLAS with an available CUDA or NPU backend",
            allow_module_level=True,
        )
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    from flag_blas.ops import CUBLAS_OP_N
    from flag_blas.runtime.backend._nvidia.hopper.ops.group_gemm import (
        grouped_tf32gemm_kernel,
        grouped_tf32gemm_small_m_tma_kernel,
        grouped_tf32gemm_tma_kernel,
        supports_tma,
    )


if not IS_ASCEND:

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

    _cublas = load_cublas()

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

    cublas.cublasGemmGroupedBatchedEx = _cublasGemmGroupedBatchedEx


SEED = 50
CUDA_R_32F = 0
CUBLAS_COMPUTE_32F_FAST_TF32 = 77
CUBLAS_COMPUTE_32F = 68
ACL_FLOAT = 0
ACL_INT64 = 9
ACL_FORMAT_ND = 2

if IS_ASCEND:

    def load_opapi():
        lib_names = [
            "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libopapi.so",
            "libopapi.so",
        ]
        found_path = ctypes.util.find_library("opapi")
        if found_path:
            lib_names.insert(0, found_path)
        for name in lib_names:
            try:
                return ctypes.cdll.LoadLibrary(name)
            except OSError:
                continue
        raise RuntimeError("Unable to find libopapi.so on the system.")

    _opapi = load_opapi()
    _ACL_PTR = ctypes.c_void_p
    _ACL_INT64_PTR = ctypes.POINTER(ctypes.c_int64)
    _opapi.aclCreateTensor.argtypes = [
        _ACL_INT64_PTR,
        ctypes.c_uint64,
        ctypes.c_int,
        _ACL_INT64_PTR,
        ctypes.c_int64,
        ctypes.c_int,
        _ACL_INT64_PTR,
        ctypes.c_uint64,
        _ACL_PTR,
    ]
    _opapi.aclCreateTensor.restype = _ACL_PTR
    _opapi.aclCreateTensorList.argtypes = [ctypes.POINTER(_ACL_PTR), ctypes.c_uint64]
    _opapi.aclCreateTensorList.restype = _ACL_PTR
    _opapi.aclSetAclOpExecutorRepeatable.argtypes = [_ACL_PTR]
    _opapi.aclSetAclOpExecutorRepeatable.restype = ctypes.c_int
    _opapi.aclnnGroupedMatmulV5GetWorkspaceSize.argtypes = [
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        _ACL_PTR,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(_ACL_PTR),
    ]
    _opapi.aclnnGroupedMatmulV5GetWorkspaceSize.restype = ctypes.c_int
    _opapi.aclnnGroupedMatmulV5.argtypes = [
        _ACL_PTR,
        ctypes.c_uint64,
        _ACL_PTR,
        _ACL_PTR,
    ]
    _opapi.aclnnGroupedMatmulV5.restype = ctypes.c_int


def cublas_group_gemm(
    group_A,
    group_B,
    group_C,
    offs_table,
    transa,
    transb,
    cu_m_arr,
    cu_n_arr,
    cu_k_arr,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    a_cublas,
    b_cublas,
    c_cublas,
    alpha_arr,
    beta_arr,
    batch,
    cu_dtype,
    compute_type,
    out_cublas,
    handle,
    a_flag,
    b_flag,
    c_flag,
    out_flag_ptrs,
    m_flag,
    n_flag,
    k_flag,
    lda_flag,
    ldb_flag,
    ldc_flag,
    group_size,
    M,
    N,
    K,
    out_flag,
    alpha,
    beta,
    **kwargs,
):
    cublas.cublasGemmGroupedBatchedEx(
        handle,
        transa,
        transb,
        cu_m_arr,
        cu_n_arr,
        cu_k_arr,
        alpha_arr.data_ptr(),
        a_cublas.data_ptr(),
        cu_dtype,
        lda_cublas,
        b_cublas.data_ptr(),
        cu_dtype,
        ldb_cublas,
        beta_arr.data_ptr(),
        c_cublas.data_ptr(),
        cu_dtype,
        ldc_cublas,
        group_size,
        batch,
        compute_type,
    )
    return out_cublas


def gems_group_gemm_wrapper(
    group_A,
    group_B,
    group_C,
    offs_table,
    transa,
    transb,
    cu_m_arr,
    cu_n_arr,
    cu_k_arr,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    a_cublas,
    b_cublas,
    c_cublas,
    alpha_arr,
    beta_arr,
    batch,
    cu_dtype,
    compute_type,
    out_cublas,
    handle,
    a_flag,
    b_flag,
    c_flag,
    out_flag_ptrs,
    m_flag,
    n_flag,
    k_flag,
    lda_flag,
    ldb_flag,
    ldc_flag,
    group_size,
    M,
    N,
    K,
    out_flag,
    alpha,
    beta,
    **kwargs,
):
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_available = supports_tma(out_flag.device)
    if tma_available and kwargs.get("use_small_m", False):
        kernel = grouped_tf32gemm_small_m_tma_kernel
    elif tma_available:
        kernel = grouped_tf32gemm_tma_kernel
    else:
        kernel = grouped_tf32gemm_kernel

    kernel[(num_sms,)](
        M,
        N,
        K,
        a_flag,
        b_flag,
        c_flag,
        out_flag_ptrs,
        m_flag,
        n_flag,
        k_flag,
        lda_flag,
        ldb_flag,
        ldc_flag,
        group_size,
        alpha=alpha,
        beta=beta,
    )
    return out_flag


class GroupGemmBenchmark(Benchmark):
    def __init__(self, *args, alpha=1.0, beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        cublas.setMathMode(handle, 0)

        scale = 1.0
        random.seed(SEED)
        for k, e, n in self.shapes:
            m_list = [random.randint(1, 4096) for _ in range(e)]
            total_M = sum(m_list)
            total_K = e * k

            group_A = (
                torch.randn(total_M, k, dtype=cur_dtype, device=self.device) * scale
            )
            group_B = (
                torch.randn(total_K, n, dtype=cur_dtype, device=self.device) * scale
            )
            group_B_T = torch.empty((e * n, k), dtype=cur_dtype, device=self.device)
            group_C = (
                torch.randn(total_M, n, dtype=cur_dtype, device=self.device) * scale
            )

            offs = []
            start_M = 0
            start_K = 0
            for g in range(e):
                mg = m_list[g]
                offs.append([mg, n, k, start_M, start_K, start_M])
                start_M += mg
                start_K += k

            out_cublas = torch.empty_like(group_C)
            out_flag = torch.empty_like(group_C)

            cu_a_ptrs = []
            cu_b_ptrs = []
            cu_c_ptrs = []
            cu_m_list = []
            cu_n_list = []
            cu_k_list = []
            cu_lda_list = []
            cu_ldb_list = []
            cu_ldc_list = []

            flag_a_ptrs = []
            flag_b_ptrs = []
            flag_c_ptrs = []
            flag_out_ptrs = []
            flag_m_list = []
            flag_n_list = []
            flag_k_list = []
            flag_lda_list = []
            flag_ldb_list = []
            flag_ldc_list = []

            start_BT_offs = 0
            for entry in offs:
                mg, ng, kg, start_M_offs, start_K_offs, start_C_offs = entry

                cu_a_ptrs.append(
                    group_B[start_K_offs : start_K_offs + kg, :].data_ptr()
                )
                group_B_T[start_BT_offs : start_BT_offs + ng, :].copy_(
                    group_B[start_K_offs : start_K_offs + kg, :].T
                )
                cu_b_ptrs.append(
                    group_A[start_M_offs : start_M_offs + mg, :].data_ptr()
                )
                cu_c_ptrs.append(
                    out_cublas[start_C_offs : start_C_offs + mg, :].data_ptr()
                )
                cu_m_list.append(ng)
                cu_n_list.append(mg)
                cu_k_list.append(kg)
                cu_lda_list.append(ng)
                cu_ldb_list.append(kg)
                cu_ldc_list.append(ng)

                flag_a_ptrs.append(
                    group_A[start_M_offs : start_M_offs + mg, :].data_ptr()
                )
                flag_b_ptrs.append(
                    group_B_T[start_BT_offs : start_BT_offs + ng, :].data_ptr()
                )
                flag_c_ptrs.append(
                    group_C[start_C_offs : start_C_offs + mg, :].data_ptr()
                )
                flag_out_ptrs.append(
                    out_flag[start_C_offs : start_C_offs + mg, :].data_ptr()
                )
                flag_m_list.append(mg)
                flag_n_list.append(ng)
                flag_k_list.append(kg)
                flag_lda_list.append(kg)
                flag_ldb_list.append(kg)
                flag_ldc_list.append(ng)
                start_BT_offs += ng

            yield group_A, group_B, group_C, offs, {
                "transa": torch.tensor([CUBLAS_OP_N] * e, dtype=torch.int32),
                "transb": torch.tensor([CUBLAS_OP_N] * e, dtype=torch.int32),
                "cu_m_arr": torch.tensor(cu_m_list, dtype=torch.int32),
                "cu_n_arr": torch.tensor(cu_n_list, dtype=torch.int32),
                "cu_k_arr": torch.tensor(cu_k_list, dtype=torch.int32),
                "lda_cublas": torch.tensor(cu_lda_list, dtype=torch.int32),
                "ldb_cublas": torch.tensor(cu_ldb_list, dtype=torch.int32),
                "ldc_cublas": torch.tensor(cu_ldc_list, dtype=torch.int32),
                "a_cublas": torch.tensor(
                    cu_a_ptrs, dtype=torch.int64, device=self.device
                ),
                "b_cublas": torch.tensor(
                    cu_b_ptrs, dtype=torch.int64, device=self.device
                ),
                "c_cublas": torch.tensor(
                    cu_c_ptrs, dtype=torch.int64, device=self.device
                ),
                "alpha_arr": torch.full((e,), self.alpha, dtype=torch.float32),
                "beta_arr": torch.full((e,), self.beta, dtype=torch.float32),
                "batch": torch.tensor([1] * e, dtype=torch.int32),
                "cu_dtype": CUDA_R_32F,
                "compute_type": CUBLAS_COMPUTE_32F_FAST_TF32,
                "out_cublas": out_cublas,
                "a_flag": torch.tensor(
                    flag_a_ptrs, dtype=torch.int64, device=self.device
                ),
                "b_flag": torch.tensor(
                    flag_b_ptrs, dtype=torch.int64, device=self.device
                ),
                "c_flag": torch.tensor(
                    flag_c_ptrs, dtype=torch.int64, device=self.device
                ),
                "out_flag_ptrs": torch.tensor(
                    flag_out_ptrs, dtype=torch.int64, device=self.device
                ),
                "m_flag": torch.tensor(
                    flag_m_list, dtype=torch.int32, device=self.device
                ),
                "n_flag": torch.tensor(
                    flag_n_list, dtype=torch.int32, device=self.device
                ),
                "k_flag": torch.tensor(
                    flag_k_list, dtype=torch.int32, device=self.device
                ),
                "lda_flag": torch.tensor(
                    flag_lda_list, dtype=torch.int32, device=self.device
                ),
                "ldb_flag": torch.tensor(
                    flag_ldb_list, dtype=torch.int32, device=self.device
                ),
                "ldc_flag": torch.tensor(
                    flag_ldc_list, dtype=torch.int32, device=self.device
                ),
                "group_size": e,
                "M": total_M,
                "N": n,
                "K": k,
                "out_flag": out_flag,
                "alpha": self.alpha,
                "beta": self.beta,
                "use_small_m": max(flag_m_list) <= 64,
                "handle": handle,
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

    def validate_results(self, torch_result, gems_result, reduce_dim, tolerance=1e-3):
        torch_cpu = torch_result.cpu()
        gems_cpu = gems_result.cpu()

        try:
            flag_blas.testing.assert_close(
                gems_cpu,
                torch_cpu,
                torch_cpu.dtype,
                equal_nan=False,
                reduce_dim=reduce_dim,
                atol=tolerance,
            )
        except AssertionError:
            max_abs_diff = torch.max(torch.abs(torch_cpu - gems_cpu))
            max_rel_diff = torch.max(
                torch.abs((torch_cpu - gems_cpu) / (torch.abs(torch_cpu) + 1e-9))
            )
            raise AssertionError(
                f"Results differ beyond tolerance {tolerance}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {torch_cpu.shape}"
            )


class AscendGroupGemmBenchmark(GroupGemmBenchmark):
    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def get_input_iter(self, cur_dtype) -> Generator:
        random.seed(SEED)
        for k, e, n in self.shapes:
            m_list = [random.randint(1, 4096) for _ in range(e)]
            M = sum(m_list)
            group_A = torch.randn((M, k), dtype=cur_dtype, device=self.device)
            group_B = torch.randn((e, k, n), dtype=cur_dtype, device=self.device)
            group_list = torch.tensor(
                m_list, dtype=torch.int64, device=self.device
            ).cumsum(0)
            out_aclnn = torch.empty((M, n), dtype=cur_dtype, device=self.device)
            group_out = torch.empty_like(out_aclnn)

            acl_tensors = []
            acl_tensor_meta = []
            for tensor in (group_A, group_B, group_list, out_aclnn):
                dims = (ctypes.c_int64 * tensor.dim())(*tensor.shape)
                strides = (ctypes.c_int64 * tensor.dim())(*tensor.stride())
                acl_dtype = ACL_INT64 if tensor.dtype == torch.int64 else ACL_FLOAT
                acl_tensor = _opapi.aclCreateTensor(
                    dims,
                    tensor.dim(),
                    acl_dtype,
                    strides,
                    tensor.storage_offset(),
                    ACL_FORMAT_ND,
                    dims,
                    tensor.dim(),
                    _ACL_PTR(tensor.data_ptr()),
                )
                if not acl_tensor:
                    raise RuntimeError("aclCreateTensor failed.")
                acl_tensors.append(acl_tensor)
                acl_tensor_meta.append((dims, strides))

            acl_tensor_lists = []
            acl_tensor_list_meta = []
            for tensor_idx in (0, 1, 3):
                tensor_array = (_ACL_PTR * 1)(acl_tensors[tensor_idx])
                tensor_list = _opapi.aclCreateTensorList(tensor_array, 1)
                if not tensor_list:
                    raise RuntimeError("aclCreateTensorList failed.")
                acl_tensor_lists.append(tensor_list)
                acl_tensor_list_meta.append(tensor_array)

            aclnn_workspace_size = ctypes.c_uint64()
            aclnn_executor = _ACL_PTR()
            status = _opapi.aclnnGroupedMatmulV5GetWorkspaceSize(
                acl_tensor_lists[0],
                acl_tensor_lists[1],
                None,
                None,
                None,
                None,
                None,
                None,
                acl_tensors[2],
                None,
                None,
                None,
                3,
                0,
                0,
                0,
                None,
                acl_tensor_lists[2],
                None,
                None,
                ctypes.byref(aclnn_workspace_size),
                ctypes.byref(aclnn_executor),
            )
            if status != 0:
                raise RuntimeError(
                    "aclnnGroupedMatmulV5GetWorkspaceSize failed with "
                    f"status {status}."
                )
            status = _opapi.aclSetAclOpExecutorRepeatable(aclnn_executor)
            if status != 0:
                raise RuntimeError(
                    f"aclSetAclOpExecutorRepeatable failed with status {status}."
                )
            aclnn_workspace = torch.empty(
                aclnn_workspace_size.value,
                dtype=torch.uint8,
                device=self.device,
            )
            aclnn_workspace_ptr = (
                _ACL_PTR(aclnn_workspace.data_ptr())
                if aclnn_workspace_size.value
                else None
            )
            num_aicores = (
                torch.npu.get_device_properties("npu").multi_processor_count // 2
            )
            yield group_A, group_B, group_list, {
                "group_out": group_out,
                "group_size": e,
                "M": M,
                "N": n,
                "K": k,
                "num_aicores": num_aicores,
                "aclnn_workspace_ptr": aclnn_workspace_ptr,
                "aclnn_workspace_size": aclnn_workspace_size.value,
                "aclnn_executor": aclnn_executor,
                "group_out_aclnn": out_aclnn,
            }

    def get_tflops(self, op, *args, **kwargs):
        group_A, group_B = args[0], args[1]
        return 2 * group_A.shape[0] * group_B.shape[1] * group_B.shape[2]

    def get_gbps(self, args, latency):
        group_A, group_B, group_list = args[0], args[1], args[2]
        output_size = group_A.shape[0] * group_B.shape[2] * group_A.element_size()
        io_amount = (
            shape_utils.size_in_bytes(group_A)
            + shape_utils.size_in_bytes(group_B)
            + shape_utils.size_in_bytes(group_list)
            + output_size
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def validate_results(self, torch_result, gems_result, reduce_dim, tolerance=1e-3):
        torch_cpu = torch_result.cpu()
        gems_cpu = gems_result.cpu()
        try:
            flag_blas.testing.assert_close(
                gems_cpu,
                torch_cpu,
                torch_cpu.dtype,
                equal_nan=False,
                reduce_dim=reduce_dim,
                atol=tolerance,
            )
        except AssertionError:
            max_abs_diff = torch.max(torch.abs(torch_cpu - gems_cpu))
            max_rel_diff = torch.max(
                torch.abs((torch_cpu - gems_cpu) / (torch.abs(torch_cpu) + 1e-9))
            )
            raise AssertionError(
                f"Results differ beyond tolerance {tolerance}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {torch_cpu.shape}"
            )


def aclnn_group_gemm(
    group_A,
    group_B,
    group_list,
    group_out,
    group_size,
    M,
    N,
    K,
    num_aicores,
    aclnn_workspace_ptr,
    aclnn_workspace_size,
    aclnn_executor,
    group_out_aclnn,
    **kwargs,
):
    status = _opapi.aclnnGroupedMatmulV5(
        aclnn_workspace_ptr,
        aclnn_workspace_size,
        aclnn_executor,
        torch.npu.current_stream()._as_parameter_,
    )
    if status != 0:
        raise RuntimeError(f"aclnnGroupedMatmulV5 failed with status {status}.")
    return group_out_aclnn


def ascend_gems_group_gemm_wrapper(
    group_A,
    group_B,
    group_list,
    group_out,
    group_size,
    M,
    N,
    K,
    num_aicores,
    aclnn_workspace_ptr,
    aclnn_workspace_size,
    aclnn_executor,
    group_out_aclnn,
    **kwargs,
):
    grouped_tf32gemm_kernel[(num_aicores,)](
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


@pytest.mark.group_gemm
def test_perf_group_gemm_tf32():
    if IS_ASCEND:
        bench = AscendGroupGemmBenchmark(
            op_name="group_gemm",
            torch_op=aclnn_group_gemm,
            gems_op=ascend_gems_group_gemm_wrapper,
            dtypes=[torch.float32],
        )
        bench.init_user_config()
        for cur_dtype in bench.to_bench_dtypes:
            for A, B, group_list, kwargs in bench.get_input_iter(cur_dtype):
                torch_result = aclnn_group_gemm(A, B, group_list, **kwargs)
                gems_result = ascend_gems_group_gemm_wrapper(A, B, group_list, **kwargs)
                bench.validate_results(torch_result, gems_result, 1, tolerance=1e-3)
        bench.run()
    else:
        bench = GroupGemmBenchmark(
            op_name="group_gemm",
            torch_op=cublas_group_gemm,
            gems_op=gems_group_gemm_wrapper,
            dtypes=[torch.float32],
        )
        bench.init_user_config()
        for cur_dtype in bench.to_bench_dtypes:
            for A, B, C, offs, kwargs in bench.get_input_iter(cur_dtype):
                torch_result = cublas_group_gemm(A, B, C.clone(), offs, **kwargs)
                gems_result = gems_group_gemm_wrapper(A, B, C.clone(), offs, **kwargs)
                k = kwargs.get("K", 0)
                bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
        bench.run()

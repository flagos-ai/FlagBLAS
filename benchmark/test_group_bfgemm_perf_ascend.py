import ctypes
import ctypes.util
import random
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.runtime.backend._ascend.ops.group_gemm import (
    _get_num_aicore,
    _use_n_chunk,
    grouped_bfgemm_kernel,
    grouped_bfgemm_n_chunk_kernel,
)
from flag_blas.utils import shape_utils


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
_opapi.aclCreateTensorList.argtypes = [
    ctypes.POINTER(_ACL_PTR),
    ctypes.c_uint64,
]
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


SEED = 50
ACL_BF16 = 27
ACL_INT64 = 9
ACL_FORMAT_ND = 2


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
    _opapi.aclnnGroupedMatmulV5(
        aclnn_workspace_ptr,
        aclnn_workspace_size,
        aclnn_executor,
        torch.npu.current_stream()._as_parameter_,
    )
    return group_out_aclnn


def gems_group_gemm_wrapper(
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


class GroupGemmBenchmark(Benchmark):
    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def get_input_iter(self, cur_dtype) -> Generator:
        scale = 1.0
        random.seed(SEED)
        for k, e, n in self.shapes:
            m_list = [random.randint(1, 4096) for _ in range(e)]
            M = sum(m_list)
            group_A = torch.randn((M, k), dtype=cur_dtype, device=self.device) * scale
            group_B = (
                torch.randn((e, k, n), dtype=cur_dtype, device=self.device) * scale
            )
            group_list = torch.tensor(
                m_list, dtype=torch.int64, device=self.device
            ).cumsum(0)
            num_aicores = _get_num_aicore()

            out_aclnn = torch.empty((M, n), dtype=cur_dtype, device=self.device)
            group_out = torch.empty_like(out_aclnn)

            acl_tensors = []
            acl_tensor_meta = []
            for tensor in (group_A, group_B, group_list, out_aclnn):
                dims = (ctypes.c_int64 * tensor.dim())(*tensor.shape)
                strides = (ctypes.c_int64 * tensor.dim())(*tensor.stride())
                acl_dtype = ACL_INT64 if tensor.dtype == torch.int64 else ACL_BF16
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

    def validate_results(self, torch_result, gems_result, reduce_dim, tolerance=1e-2):
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


@pytest.mark.group_gemm
def test_perf_group_gemm_bf16():
    bench = GroupGemmBenchmark(
        op_name="group_gemm",
        torch_op=aclnn_group_gemm,
        gems_op=gems_group_gemm_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.init_user_config()
    for cur_dtype in bench.to_bench_dtypes:
        for A, B, group_list, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = aclnn_group_gemm(A, B, group_list, **kwargs)
            gems_result = gems_group_gemm_wrapper(A, B, group_list, **kwargs)
            bench.validate_results(torch_result, gems_result, 1, tolerance=1e-2)
    bench.run()

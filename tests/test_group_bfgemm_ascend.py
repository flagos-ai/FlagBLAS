import random

import pytest
import torch

import flag_blas

from . import accuracy_utils as utils
from .conftest import TO_CPU

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or flag_blas.device != "npu",
    reason="requires FlagBLAS with an available NPU backend",
)

torch_npu = pytest.importorskip("torch_npu")

@pytest.mark.group_gemm
@pytest.mark.parametrize(
    "k,e,n",
    utils.GROUP_GEMM_SHAPES,
)
def test_accuracy_group_gemm(k, e, n):
    scale = k**-0.5
    m_list = [random.randint(1, 4096) for _ in range(e)]
    M = sum(m_list)
    group_A = torch.randn(M, k, dtype=torch.bfloat16, device=flag_blas.device) * scale
    group_B = (
        torch.randn(e, k, n, dtype=torch.bfloat16, device=flag_blas.device) * scale
    )
    group_list = torch.tensor(
        m_list, dtype=torch.int64, device=flag_blas.device
    ).cumsum(0)
    group_out = torch.empty((M, n), dtype=torch.bfloat16, device=flag_blas.device)

    group_ref = torch_npu.npu_grouped_matmul(
        [group_A],
        [group_B],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=0,
        output_dtype=group_A.dtype,
    )[0]
    group_out = flag_blas.group_bfgemm(group_A, group_B, group_list, group_out)

    if TO_CPU:
        group_out = group_out.cpu()
        group_ref = group_ref.cpu()
    utils.blas_assert_close(group_out, group_ref, torch.bfloat16, reduce_dim=k)

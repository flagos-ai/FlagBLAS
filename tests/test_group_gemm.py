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

import random

import pytest
import torch

import flag_blas

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

DTYPES = [torch.bfloat16, torch.float16]

M_MIN, M_MAX = 1, 4096


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


def _compute_reference(group_A, group_B, group_C, offs_table, alpha, beta):
    dtype = group_A.dtype
    ref_out = group_C.clone().to(torch.float32)

    A_f32 = group_A.to(torch.float32)
    B_f32 = group_B.to(torch.float32)

    for entry in offs_table:
        m_g, n_g, k_g, start_M, start_K, start_C = entry

        A_sub = A_f32[start_M : start_M + m_g, :k_g]
        B_sub = B_f32[start_K : start_K + k_g, :n_g]

        res = torch.matmul(A_sub, B_sub)

        if beta == 0.0:
            ref_out[start_C : start_C + m_g, :n_g] = alpha * res
        else:
            ref_out[start_C : start_C + m_g, :n_g] = (
                alpha * res + beta * ref_out[start_C : start_C + m_g, :n_g]
            )

    return ref_out.to(dtype)


@pytest.mark.group_gemm
@pytest.mark.parametrize("config_idx", range(len(GROUP_GEMM_CONFIGS)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_group_gemm(config_idx, dtype):
    k, e, n = GROUP_GEMM_CONFIGS[config_idx]
    device = flag_blas.device
    alpha, beta = 1.5, 0.5
    scale = k**-0.5

    m_list = [random.randint(M_MIN, M_MAX) for _ in range(e)]
    total_M = sum(m_list)
    total_K = e * k

    group_A = torch.randn(total_M, k, dtype=dtype, device=device) * scale
    group_B = torch.randn(total_K, n, dtype=dtype, device=device) * scale
    group_C = torch.randn(total_M, n, dtype=dtype, device=device) * scale

    offs_table = _build_offs_table(k, e, n, m_list)
    ref = _compute_reference(group_A, group_B, group_C, offs_table, alpha, beta)

    out = flag_blas.group_gemm(
        group_A, group_B, group_C, offs_table, alpha=alpha, beta=beta
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
def test_group_gemm_alpha_zero():
    m, k, e, n = 16, 64, 4, 128
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device)
    B = torch.randn(e * k, n, dtype=dtype, device=device)
    C = torch.randn(e * m, n, dtype=dtype, device=device)
    C_orig = C.clone()
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    out = flag_blas.group_gemm(A, B, C, offs_table, alpha=0.0, beta=2.0)

    torch.testing.assert_close(out, C_orig * 2.0, rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
def test_group_gemm_beta_zero():
    m, k, e, n = 8, 32, 3, 64
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device)
    B = torch.randn(e * k, n, dtype=dtype, device=device)
    C_zeros = torch.zeros(e * m, n, dtype=dtype, device=device)
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    ref = _compute_reference(A, B, C_zeros, offs_table, 1.0, 0.0)
    out = flag_blas.group_gemm(A, B, C_zeros, offs_table, alpha=1.0, beta=0.0)

    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 1.5)]
)
def test_group_gemm_alpha_beta(alpha, beta):
    m, k, e, n = 32, 128, 2, 128
    dtype, device = torch.bfloat16, flag_blas.device
    scale = k**-0.5
    A = torch.randn(e * m, k, dtype=dtype, device=device) * scale
    B = torch.randn(e * k, n, dtype=dtype, device=device) * scale
    C = torch.randn(e * m, n, dtype=dtype, device=device) * scale
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    ref = _compute_reference(A, B, C, offs_table, alpha, beta)
    out = flag_blas.group_gemm(A, B, C, offs_table, alpha=alpha, beta=beta)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

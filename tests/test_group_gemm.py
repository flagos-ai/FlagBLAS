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

M_VALUES = list(range(1, 33)) + [64, 128, 256, 512, 1024, 2048, 4096]
DTYPES = [torch.bfloat16]

def _build_offs_table(e, m, n, k):
    offs = []
    for g in range(e):
        start_M = g * m
        start_K = g * k
        offs.append([m, n, k, start_M, start_K, start_M])
    return offs

def _compute_reference(group_A, group_B, group_C, offs_table, alpha, beta):
    device = group_A.device
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
            ref_out[start_C : start_C + m_g, :n_g] = alpha * res + beta * ref_out[start_C : start_C + m_g, :n_g]
            
    return ref_out.to(dtype)

@pytest.mark.group_gemm
@pytest.mark.parametrize("config_idx", range(len(GROUP_GEMM_CONFIGS)))
@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_group_gemm(config_idx, m, dtype):
    k, e, n = GROUP_GEMM_CONFIGS[config_idx]
    device = flag_blas.device
    alpha, beta = 1.5, 0.5
    scale = k ** -0.5

    group_A = torch.randn(e * m, k, dtype=dtype, device=device) * scale
    group_B = torch.randn(e * k, n, dtype=dtype, device=device) * scale
    group_C = torch.randn(e * m, n, dtype=dtype, device=device) * scale

    offs_table = _build_offs_table(e, m, n, k)
    ref = _compute_reference(group_A, group_B, group_C, offs_table, alpha, beta)

    out = flag_blas.group_gemm(group_A, group_B, group_C, offs_table, alpha=alpha, beta=beta)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

@pytest.mark.group_gemm
def test_group_gemm_alpha_zero():
    m, k, e, n = 16, 64, 4, 128
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device)
    B = torch.randn(e * k, n, dtype=dtype, device=device)
    C = torch.randn(e * m, n, dtype=dtype, device=device)
    C_orig = C.clone()
    offs_table = _build_offs_table(e, m, n, k)

    out = flag_blas.group_gemm(A, B, C, offs_table, alpha=0.0, beta=2.0)

    torch.testing.assert_close(out, C_orig * 2.0, rtol=1e-2, atol=1e-2)

@pytest.mark.group_gemm
def test_group_gemm_beta_zero():
    m, k, e, n = 8, 32, 3, 64
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device)
    B = torch.randn(e * k, n, dtype=dtype, device=device)
    C_zeros = torch.zeros(e * m, n, dtype=dtype, device=device)
    offs_table = _build_offs_table(e, m, n, k)

    ref = _compute_reference(A, B, C_zeros, offs_table, 1.0, 0.0)
    out = flag_blas.group_gemm(A, B, C_zeros, offs_table, alpha=1.0, beta=0.0)

    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

@pytest.mark.group_gemm
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 1.5)])
def test_group_gemm_alpha_beta(alpha, beta):
    m, k, e, n = 32, 128, 2, 128
    dtype, device = torch.bfloat16, flag_blas.device
    scale = k ** -0.5
    A = torch.randn(e * m, k, dtype=dtype, device=device) * scale
    B = torch.randn(e * k, n, dtype=dtype, device=device) * scale
    C = torch.randn(e * m, n, dtype=dtype, device=device) * scale
    offs_table = _build_offs_table(e, m, n, k)

    ref = _compute_reference(A, B, C, offs_table, alpha, beta)
    out = flag_blas.group_gemm(A, B, C, offs_table, alpha=alpha, beta=beta)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
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

"""MThreads (MUSA) zgemm implementation (fp64 complex) -- Triton kernel.

MUSA has no complex and no fp64 MMA, so a complex128 GEMM is expressed as
four real fp64 ``tl.dot`` passes with ``input_precision="ieee"`` (the backend
lowers fp64 dots to an FMA loop). The accumulators and the alpha/beta epilogue
are fp64, which keeps the measured error at the fp64 rounding level and inside
the 1e-7 tolerance for complex128.

Layout mirrors ``cgemm``: a contiguous complex128 buffer is ``(rows, cols, 2)``
in fp64, so the kernel walks the real/imaginary planes with a stride of two.
The fp64 register pressure means the tiles stay small (a 32x32 output tile
with two fp64 accumulators already uses 16 KiB per program).
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.zgemm import (
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    ScalarType,
    _validate_zgemm_args,
)
from flag_blas.runtime import torch_device_fn


def _pick_config(m, n, k):
    """Heuristic tile selection for complex128 on MTT S5000.

    Returns ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)``.  Measured
    on 512**3: 64x64 with 8 warps reaches 0.06 TFLOPS, four times the 32x32
    tile.  complex128 needs four fp64 dots and two fp64 accumulators, so the
    tile cannot grow further without spilling registers; a single stage keeps
    the four fp64 operand tiles at 64 KiB of shared memory.
    """
    if m * n * k <= 128 ** 3:
        return 32, 32, 32, 4, 2
    grid64 = triton.cdiv(m, 64) * triton.cdiv(n, 64)
    if grid64 >= 16:
        return 64, 64, 32, 8, 1
    return 32, 32, 32, 4, 2


@triton.jit
def _zgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    lda2,
    ldb2,
    ldc2,
    alpha_r,
    alpha_i,
    beta_r,
    beta_i,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Row-major complex128 ``C = alpha * A @ B + beta * C`` on packed operands.

    The ``*2`` leading dimensions are the fp64 row strides of the real views,
    so element ``(i, j)`` of a complex matrix lives at ``i * ld2 + 2 * j``
    (real) and ``+1`` (imaginary).
    """
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = tl.max_contiguous(tl.multiple_of(pid_m * BM + tl.arange(0, BM), BM), BM)
    rn = tl.max_contiguous(tl.multiple_of(pid_n * BN + tl.arange(0, BN), BN), BN)
    rk = tl.arange(0, BK)

    acc_r = tl.zeros((BM, BN), dtype=tl.float64)
    acc_i = tl.zeros((BM, BN), dtype=tl.float64)
    for kk in range(0, k, BK):
        koff = kk + rk
        ap = a_ptr + rm[:, None] * lda2 + (koff * 2)[None, :]
        bp = b_ptr + (koff * ldb2)[:, None] + rn[None, :] * 2
        a_r = tl.load(ap)
        a_i = tl.load(ap + 1)
        b_r = tl.load(bp)
        b_i = tl.load(bp + 1)
        acc_r = tl.dot(a_r, b_r, acc_r, out_dtype=tl.float64, input_precision="ieee")
        acc_r = tl.dot(-a_i, b_i, acc_r, out_dtype=tl.float64, input_precision="ieee")
        acc_i = tl.dot(a_r, b_i, acc_i, out_dtype=tl.float64, input_precision="ieee")
        acc_i = tl.dot(a_i, b_r, acc_i, out_dtype=tl.float64, input_precision="ieee")

    y_r = alpha_r * acc_r - alpha_i * acc_i
    y_i = alpha_r * acc_i + alpha_i * acc_r

    cp = c_ptr + rm[:, None] * ldc2 + rn[None, :] * 2
    if BETA_ZERO:
        tl.store(cp, y_r)
        tl.store(cp + 1, y_i)
    else:
        c_r = tl.load(cp)
        c_i = tl.load(cp + 1)
        tl.store(cp, y_r + beta_r * c_r - beta_i * c_i)
        tl.store(cp + 1, y_i + beta_r * c_i + beta_i * c_r)


def _operand(A, op, rows, cols, ld):
    """Return a packed row-major complex ``(rows, cols)`` tensor holding ``op(A)``.

    For ``op == N`` the stored matrix is already ``(rows, cols)``; for T/C the
    operand is stored transposed as ``(cols, rows)``.  C additionally
    conjugates, materialised here instead of inside a vendor call.
    """
    if op == CUBLAS_OP_N:
        if ld == cols:
            return A
        return A.as_strided((rows, cols), (ld, 1)).contiguous()
    stored = A.as_strided((cols, rows), (ld, 1))
    out = stored.t().contiguous()
    if op == CUBLAS_OP_C:
        out = torch.conj(out).resolve_conj()
    return out


def _pad(x, rows, cols, pad_rows, pad_cols):
    """Zero-pad a packed complex tensor.  The pad lanes never reach ``C``."""
    if rows == pad_rows and cols == pad_cols:
        return x
    out = torch.zeros((pad_rows, pad_cols), dtype=x.dtype, device=x.device)
    out[:rows, :cols] = x
    return out


def zgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    _validate_zgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha_c = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_c = complex(beta.item() if isinstance(beta, torch.Tensor) else beta)

    if m == 0 or n == 0 or k == 0 or alpha_c == 0.0:
        if beta_c == 0.0:
            C.zero_()
        elif beta_c != 1.0:
            C.mul_(beta_c)
        return

    bm, bn, bk, num_warps, num_stages = _pick_config(m, n, k)
    pm = ((m + bm - 1) // bm) * bm
    pn = ((n + bn - 1) // bn) * bn
    pk = ((k + bk - 1) // bk) * bk
    beta_is_zero = beta_c == 0.0
    c_in_place = (pm == m) and (pn == n) and (ldc == n) and C.is_contiguous()

    def launch():
        a_p = _pad(_operand(A, transa, m, k, lda), m, k, pm, pk)
        b_p = _pad(_operand(B, transb, k, n, ldb), k, n, pk, pn)
        a_v = torch.view_as_real(a_p)
        b_v = torch.view_as_real(b_p)
        if c_in_place:
            c_buf, c_v = C, torch.view_as_real(C)
        else:
            c_buf = torch.zeros((pm, pn), dtype=C.dtype, device=C.device)
            if not beta_is_zero:
                c_buf[:m, :n] = C.as_strided((m, n), (ldc, 1))
            c_v = torch.view_as_real(c_buf)
        grid = (triton.cdiv(pm, bm) * triton.cdiv(pn, bn),)
        _zgemm_kernel[grid](
            a_v,
            b_v,
            c_v,
            pm,
            pn,
            pk,
            a_v.stride(0),
            b_v.stride(0),
            c_v.stride(0),
            alpha_c.real,
            alpha_c.imag,
            beta_c.real,
            beta_c.imag,
            beta_is_zero,
            bm,
            bn,
            bk,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if not c_in_place:
            if ldc == n:
                C.copy_(c_buf[:m, :n])
            else:
                C.as_strided((m, n), (ldc, 1)).copy_(c_buf[:m, :n])

    device = A.device
    if device.index is None or torch_device_fn.current_device() == device.index:
        launch()
    else:
        with torch_device_fn.device(device):
            launch()

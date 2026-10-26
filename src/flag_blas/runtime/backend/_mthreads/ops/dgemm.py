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

"""MThreads (MUSA) dgemm implementation (fp64).

MUSA has no fp64 matrix-multiply unit, so the products have to run as fp64 FMA
on the ALU.  A layered probe on MTT S5000 showed where that leaves us: the
fp64 FMA peak (register-resident, 8 independent accumulators) is ~7.0 TFLOPS,
but ``tl.dot`` with ``input_precision="ieee"`` tops out at 2.24 TFLOPS even
with the operands hoisted out of the k-loop and no global traffic -- the dot
lowering itself reaches only ~32% of the ALU (TTGIR shows each k iteration
round-tripping both operands through shared memory into a ``dot_op`` layout),
and a kernel built on it already sits at 95% of that ceiling.  So the Triton
kernel below skips ``tt.dot`` and accumulates with explicit outer products of
1-D operand vectors, which reaches ~3.9 TFLOPS (2.73 / 3.78 / 3.87 / 3.69 /
3.56 TFLOPS on 512^3 / 1024^3 / 2048^3 / 4096^3 / 8192^3, 1.7-1.95x the dot
form).  The vendor library still reaches ~10-14 TFLOPS.

Large problems therefore route to a hand-written MUSA kernel compiled with
``mcc`` (see ``_dgemm_musa``): leaving Triton lifts the code-generation ceiling
and gets 9.1 TFLOPS at 4096^3 and 9.4 at 8192^3, 2.6x the Triton kernel.  The
hand path stages its panels with the TME copy engine when the target and a
host-built tensor descriptor are available, and with plain vectorised loads
otherwise.  Every step is best-effort -- without ``mcc``, or if a compile
fails, it is skipped and the Triton kernel runs instead.  Below roughly 4e8
flops the Triton kernel is the faster of the two.

Computing in *real* fp64 costs that factor: emulating fp64 on the fp16/tf32
tensor cores runs 3-8x the vendor library but only carries ~1e-5 relative
accuracy, which the dgemm harness (atol ``1e-10*K``, rtol 1e-7) rejects.
Both kernels keep the accumulator and the epilogue in fp64, so they match the
vendor result to ~1e-13.

Operands that are transposed, or whose leading dimension is padded, are staged
into packed row-major buffers first; that staging is a plain tensor copy.
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.dgemm import (
    CUBLAS_OP_N,
    ScalarType,
    _validate_dgemm_args,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._mthreads.ops import _dgemm_musa

# Below this much work the Triton kernel is faster; above it the hand-written
# MUSA kernel wins (see _dgemm_musa for the measurements).
_MUSA_MIN_WORK = 4.0e8


def _launch_musa(a_p, b_p, c_buf, pm, pn, pk, alpha_f, beta_f, beta_is_zero):
    """Try the hand-written MUSA kernels.  Returns True if one of them ran.

    The TME-staged kernel is tried first (it stages the panels on the copy
    engine instead of the load/store pipe and is ~10% faster on large
    problems); when it is unavailable -- no PH1 target, no descriptor encoder,
    or no tile dividing the padded extents -- the LDG-staged kernel runs.  A
    ``False`` return means the caller should use the Triton kernel instead.
    """
    if pm * pn * pk <= _MUSA_MIN_WORK:
        return False
    if _dgemm_musa.launch_tme(a_p, b_p, c_buf, pm, pn, pk, alpha_f, beta_f,
                              beta_is_zero):
        return True
    launcher = _dgemm_musa.get_launcher()
    if launcher is None:
        return False
    rc = launcher(
        a_p.data_ptr(),
        b_p.data_ptr(),
        c_buf.data_ptr(),
        pm,
        pn,
        pk,
        alpha_f,
        beta_f,
        1 if beta_is_zero else 0,
    )
    return rc == 0


def _pick_config(m, n, k):
    """Heuristic tile selection for fp64 on MTT S5000.

    Returns ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)``.

    A narrow ``32x16x4`` tile with 4 warps is the fastest configuration at
    every size measured: 2.73 / 3.78 / 3.87 / 3.69 / 3.56 TFLOPS on
    512^3 / 1024^3 / 2048^3 / 4096^3 / 8192^3, 1.7-1.95x the equivalent
    ``tl.dot`` kernel.  The fp64 accumulator costs 8 bytes/element and the
    fp64 ALU is issue-bound, so anything wider or deeper loses to register
    pressure (64x64x4 -> 1.16, 32x32x16 -> 0.33 TFLOPS).  ``num_stages=1``
    throughout; the operands are re-loaded per k step and a second stage
    measured slower.
    """
    return 32, 16, 4, 4, 1


@triton.jit
def _dgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha: tl.float64,
    beta: tl.float64,
    BETA_ZERO: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Row-major fp64 ``C = alpha * A @ B + beta * C``.

    Shapes are expected to be tile-divisible; the caller pads them.
    """
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float64)
    for kk in range(0, k, BK):
        for i in tl.static_range(BK):
            a_col = tl.load(a_ptr + rm * lda + (kk + i))
            b_row = tl.load(b_ptr + (kk + i) * ldb + rn)
            acc += a_col[:, None] * b_row[None, :]

    y = acc * alpha
    cp = c_ptr + rm[:, None] * ldc + rn[None, :]
    if BETA_ZERO:
        tl.store(cp, y)
    else:
        tl.store(cp, y + beta * tl.load(cp))


def _operand(A, op, rows, cols, ld):
    """Return a packed row-major ``(rows, cols)`` tensor holding ``op(A)``.

    For ``op == N`` the stored matrix is already ``(rows, cols)``; for T the
    operand is stored transposed as ``(cols, rows)``.
    """
    if op == CUBLAS_OP_N:
        if ld == cols:
            return A
        return A.as_strided((rows, cols), (ld, 1)).contiguous()
    return A.as_strided((cols, rows), (ld, 1)).t().contiguous()


def _pad(x, rows, cols, pad_rows, pad_cols):
    """Zero-pad a packed operand.  The pad lanes never reach ``C``."""
    if rows == pad_rows and cols == pad_cols:
        return x
    out = torch.zeros((pad_rows, pad_cols), dtype=x.dtype, device=x.device)
    out[:rows, :cols] = x
    return out


def dgemm(
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
    _validate_dgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha_f = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_f = float(beta.item() if isinstance(beta, torch.Tensor) else beta)

    if m == 0 or n == 0 or k == 0 or alpha_f == 0.0:
        if beta_f == 0.0:
            C.zero_()
        elif beta_f != 1.0:
            C.mul_(beta_f)
        return

    bm, bn, bk, num_warps, num_stages = _pick_config(m, n, k)
    pm = ((m + bm - 1) // bm) * bm
    pn = ((n + bn - 1) // bn) * bn
    pk = ((k + bk - 1) // bk) * bk
    beta_is_zero = beta_f == 0.0
    # ``ldc == n`` keeps C usable in place; anything else (padded or
    # non-contiguous C) is computed in a packed buffer and copied back.
    c_in_place = (pm == m) and (pn == n) and (ldc == n) and C.is_contiguous()

    def launch():
        a_p = _pad(_operand(A, transa, m, k, lda), m, k, pm, pk)
        b_p = _pad(_operand(B, transb, k, n, ldb), k, n, pk, pn)
        if c_in_place:
            c_buf = C
        else:
            c_buf = torch.zeros((pm, pn), dtype=C.dtype, device=C.device)
            if not beta_is_zero:
                c_buf[:m, :n] = C.as_strided((m, n), (ldc, 1))
        grid = (triton.cdiv(pm, bm) * triton.cdiv(pn, bn),)
        if not _launch_musa(a_p, b_p, c_buf, pm, pn, pk, alpha_f, beta_f,
                            beta_is_zero):
            _dgemm_kernel[grid](
                a_p,
                b_p,
                c_buf,
                pm,
                pn,
                pk,
                a_p.stride(0),
                b_p.stride(0),
                c_buf.stride(0),
                alpha_f,
                beta_f,
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

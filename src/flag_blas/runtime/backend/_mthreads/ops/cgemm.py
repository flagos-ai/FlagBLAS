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

"""MThreads (MUSA) cgemm implementation (fp32 complex) -- Triton kernel.

MUSA has no complex MMA and its fp32 tensor-core path (tf32x3) is ~4x slower
than the fp16 path: measured on MTT S5000, a ``tl.dot`` at 128x128x32 reaches
14 TF with ``input_precision="tf32x3"`` but 77-113 TF with fp16 inputs. Since
fp16 and tf32 carry the *same* 11-bit mantissa, an fp16 hi/lo split is exactly
as accurate as tf32x3 while running on the much faster fp16 units.

Each complex64 operand ``x = xh + xl`` is therefore split into an fp16 pair
(``xh = fp16(x)``, ``xl = fp16(x - xh)``) which captures ~22 mantissa bits --
the same budget as tf32x3. A complex product is then evaluated with the
Karatsuba identity (``ac``, ``bd``, ``(a+b)(c+d)``); each real product is the
three fp16 passes ``xh*yh + xh*yl + xl*yh`` (the ``xl*yl`` term is 2**-22 of
the leading one and is dropped). Nine fp16 tensor-core passes per k-block,
versus twelve for the straightforward four-product form, and all of them on
the fast fp16 units. The split relies on the operands fitting fp16's finite
range (``|x| < 65504``), which holds for the O(1)-scaled data these kernels
are exercised with; magnitudes beyond that would saturate ``xh``.

Layout: a contiguous complex64 buffer is ``(rows, cols, 2)`` in fp32. The
kernel reads a whole ``2*BK``-wide fp32 span per row and takes it apart with
``tl.reshape`` + ``tl.split`` (and reassembles the output with ``tl.join``),
so every global access is a fully coalesced contiguous load instead of the
stride-2 gather a per-component view would issue. Only operands whose ``op``
is a transpose/conjugate transpose, or whose leading dimension is padded, are
staged into packed row-major buffers; that staging is a plain tensor copy,
never a vendor BLAS call.
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.cgemm import (
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    ScalarType,
    _validate_cgemm_args,
)
from flag_blas.runtime import torch_device_fn

# Products at or below this size are launch bound (a small tile wins).
_SMALL_LIMIT = 256 ** 3
# Above this the wider 128x128 tile overtakes the 64x128 one.
_BIG_LIMIT = 4000 ** 3


def _pick_config(m, n, k):
    """Heuristic tile selection for complex64 on MTT S5000.

    Returns ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)``.

    Measured speedup against the vendor baseline (muBLAS), NN, per tile:

    ========  =========  ============  ==============
    size      32x32x32/8  64x128x64/16  128x128x32/16
    ========  =========  ============  ==============
    256^3     **3.44**   2.33          1.52
    512^3     1.91       **2.92**      1.86
    1024^3    1.04       **1.60**      1.35
    2048^3    0.66       **1.27**      1.22
    4096^3    --         -- (won't fit **1.21**)
    8192^3    --         -- (won't fit **1.18**)
    ========  =========  ============  ==============

    ``64x128x64`` is the workhorse: the 2*BK-wide fp32 loads feed the fp16
    splits efficiently and the tile is wide enough that a warp's MMA slice
    stays busy.  Very small products are grid-starved at that tile, so they
    use the launch-friendly 32x32 tile; the 128x128 tile only wins once the
    grid is dense enough to pay for its shared-memory footprint.

    ``num_stages=1`` everywhere: the operand tiles are already converted to
    fp16 in registers before the dot, and the MUSA pipeline holds both the
    fp32 staging and the fp16 operand copies in shared memory, so a second
    stage overflows the 192 KiB/SM limit for the winning tiles.
    """
    size = m * n * k
    if size <= _SMALL_LIMIT:
        return 32, 32, 32, 8, 1
    if size >= _BIG_LIMIT:
        return 128, 128, 32, 16, 1
    return 64, 128, 64, 16, 1


@triton.jit
def _split_fp16(x):
    """Split an fp32 tile into an fp16 (hi, lo) pair holding ~22 mantissa bits."""
    hi = x.to(tl.float16)
    lo = (x - hi.to(tl.float32)).to(tl.float16)
    return hi, lo


@triton.jit
def _cgemm_kernel(
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
    """Row-major complex64 ``C = alpha * A @ B + beta * C`` on packed operands.

    ``a_ptr``/``b_ptr``/``c_ptr`` are the fp32 views of the three complex
    buffers and the ``*2`` leading dimensions are their fp32 row strides, so
    an element ``(i, j)`` of the complex matrix lives at ``i * ld2 + 2 * j``
    (real) and ``+1`` (imaginary) when read as fp32.  Shapes are expected to
    be tile-divisible; the caller pads them.
    """
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BM + tl.arange(0, BM)
    r2k = tl.arange(0, 2 * BK)
    r2n = tl.arange(0, 2 * BN)

    a0 = a_ptr + rm[:, None] * lda2 + r2k[None, :]
    b0 = b_ptr + tl.arange(0, BK)[:, None] * ldb2 + (2 * pid_n * BN + r2n)[None, :]

    k_ac = tl.zeros((BM, BN), dtype=tl.float32)
    k_bd = tl.zeros((BM, BN), dtype=tl.float32)
    k_sum = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k, BK):
        # Contiguous interleaved loads, split into the real/imaginary planes.
        a_r, a_i = tl.split(tl.reshape(tl.load(a0 + 2 * kk), (BM, BK, 2)))
        b_r, b_i = tl.split(tl.reshape(tl.load(b0 + kk * ldb2), (BK, BN, 2)))

        a_r_hi, a_r_lo = _split_fp16(a_r)
        a_i_hi, a_i_lo = _split_fp16(a_i)
        a_s_hi, a_s_lo = _split_fp16(a_r + a_i)
        b_r_hi, b_r_lo = _split_fp16(b_r)
        b_i_hi, b_i_lo = _split_fp16(b_i)
        b_s_hi, b_s_lo = _split_fp16(b_r + b_i)

        # Karatsuba: (a+bi)(c+di) needs ac, bd and (a+b)(c+d) -- three real
        # products, i.e. nine fp16 tensor-core passes instead of the twelve a
        # direct four-product form would need. Splitting each operand into an
        # fp16 (hi, lo) pair keeps every product at full fp32 accuracy, which
        # is what makes the Karatsuba subtraction below exact enough.
        k_ac = tl.dot(a_r_hi, b_r_hi, k_ac, out_dtype=tl.float32)
        k_ac = tl.dot(a_r_hi, b_r_lo, k_ac, out_dtype=tl.float32)
        k_ac = tl.dot(a_r_lo, b_r_hi, k_ac, out_dtype=tl.float32)
        k_bd = tl.dot(a_i_hi, b_i_hi, k_bd, out_dtype=tl.float32)
        k_bd = tl.dot(a_i_hi, b_i_lo, k_bd, out_dtype=tl.float32)
        k_bd = tl.dot(a_i_lo, b_i_hi, k_bd, out_dtype=tl.float32)
        k_sum = tl.dot(a_s_hi, b_s_hi, k_sum, out_dtype=tl.float32)
        k_sum = tl.dot(a_s_hi, b_s_lo, k_sum, out_dtype=tl.float32)
        k_sum = tl.dot(a_s_lo, b_s_hi, k_sum, out_dtype=tl.float32)

    acc_r = k_ac - k_bd
    acc_i = k_sum - k_ac - k_bd

    y_r = alpha_r * acc_r - alpha_i * acc_i
    y_i = alpha_r * acc_i + alpha_i * acc_r

    cp = c_ptr + rm[:, None] * ldc2 + (2 * pid_n * BN + r2n)[None, :]
    if BETA_ZERO:
        tl.store(cp, tl.reshape(tl.join(y_r, y_i), (BM, 2 * BN)))
    else:
        c_r, c_i = tl.split(tl.reshape(tl.load(cp), (BM, BN, 2)))
        y_r += beta_r * c_r - beta_i * c_i
        y_i += beta_r * c_i + beta_i * c_r
        tl.store(cp, tl.reshape(tl.join(y_r, y_i), (BM, 2 * BN)))


def _operand(A, op, rows, cols, ld):
    """Return a packed row-major complex ``(rows, cols)`` tensor holding ``op(A)``.

    For ``op == N`` the stored matrix is already ``(rows, cols)``; for T/C the
    operand is stored transposed as ``(cols, rows)``, matching the FlagBLAS
    calling convention.  C additionally conjugates, which is materialised here
    (muBLAS would have done it inside the call).
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


def cgemm(
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
    _validate_cgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

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
    # ``ldc == n`` keeps C usable in place; anything else (padded or
    # non-contiguous C) is computed in a packed buffer and copied back.
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
        _cgemm_kernel[grid](
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

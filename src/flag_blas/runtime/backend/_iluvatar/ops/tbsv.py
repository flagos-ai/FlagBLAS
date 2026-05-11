import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_DIAG_UNIT = 1

_TBSV_KEY = ["n", "k_bucket", "mode_key"]
_TBSV_RESTORE = ["x_ptr"]


def _band_bucket(k: int) -> int:
    if k <= 1:
        return 1
    b = 1
    while b < k and b < 1024:
        b <<= 1
    return b


def _mode_key(uplo: int, trans: int, unit: int) -> int:
    return (uplo << 4) | (trans << 2) | unit


def _pick_bs(n: int, k: int) -> int:
    """Panel size for the blocked algorithm.

    Larger BS amortises outer-loop overhead and produces fatter
    inter-panel GEMV updates, but also unrolls the panel-solve more,
    raising register pressure. BS=8 is a robust default for ivcore11
    (warp=64): the panel solve stays cheap, and BLOCK_K can still
    grow to absorb the work."""
    if n <= 1:
        return 1
    return 8


# --------------------------------------------------------------------------
# Panel-blocked TBSV kernel.
#
# Algorithm (taking Upper/NoTrans as the canonical example):
#   for each panel jp..jp+BS (processed right-to-left):
#     1. Load x[jp:jp+BS] into a register vector `xp` (size BS).
#     2. Panel solve: BS-sized triangular system solved entirely in
#        registers (BS reductions over a BS-wide vector). No GMEM
#        store/load happens during the panel solve, so there is no
#        chance for the compiler/HW to read a stale x value.
#     3. Store the solved `xp` back to x[jp:jp+BS].
#     4. Panel update: for i in [jp-k, jp), do
#            x[i] -= sum_{jj=0..BS-1} U[i, jp+jj] * xp[jj]
#        as a (BS, BLOCK_K)-tile GEMV. This is a pure axpy on
#        disjoint i positions, no inner RAW.
#     5. tl.debug_barrier() forces all panel writes to be visible
#        before the next panel's xp load.
#
# This wins on Iluvatar BI-V150 because:
#   * Outer iterations shrink by BS× (less serial latency to amortise).
#   * The inner update is a fat (BS, BLOCK_K) tile, which keeps the
#     64-lane warps busy and lets BLOCK_K grow.
#   * Cross-iteration RAW only crosses one barrier, never a software
#     pipeline boundary.
# --------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("stbsv"),
    key=_TBSV_KEY,
    restore_value=_TBSV_RESTORE,
)
@triton.jit
def stbsv_kernel(
    a_ptr,
    x_ptr,
    n,
    k,
    LDA,
    INCX,
    k_bucket,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_bs = tl.arange(0, BS)
    offs_k = tl.arange(0, BLOCK_K)
    n_panels = (n + BS - 1) // BS

    # ====================================================================
    # Lower / NoTrans :  L x = b      (forward sub, panels left -> right)
    # Storage:  L[i,j]  (i>=j, i-j<=k)  =  A[(i-j) + j*LDA]
    # Diagonal: L[j,j]                  =  A[j*LDA]
    # ====================================================================
    if (UPLO == 0) and (TRANS == 0):
        for pc in tl.range(0, n_panels):
            jp = pc * BS
            panel_valid = (jp + offs_bs) < n
            xp = tl.load(x_ptr + (jp + offs_bs) * INCX, mask=panel_valid, other=0.0)

            # --- Panel solve (entirely in registers) ---
            for jj_r in tl.static_range(BS):
                jj = jj_r
                j = jp + jj
                one_hot = offs_bs == jj
                if not UNIT:
                    diag_jj = tl.load(a_ptr + j * LDA, mask=j < n, other=1.0)
                    xj_old = tl.sum(xp * one_hot.to(tl.float32), axis=0)
                    xj_new = xj_old / diag_jj
                    xp = tl.where(one_hot, xj_new, xp)
                else:
                    xj_new = tl.sum(xp * one_hot.to(tl.float32), axis=0)
                # Propagate xp[jj] downward inside the panel.
                # L[jp+ii, jp+jj] = A[(ii-jj) + (jp+jj)*LDA]
                d = offs_bs - jj
                a_off = d + j * LDA
                a_mask = (d >= 1) & (d <= k) & ((jp + offs_bs) < n)
                A_col = tl.load(a_ptr + a_off, mask=a_mask, other=0.0)
                xp = xp - A_col * xj_new

            tl.store(x_ptr + (jp + offs_bs) * INCX, xp, mask=panel_valid)

            # --- Panel update: rows i in [jp+BS, jp+BS+k) ---
            # x[i] -= sum_{jj} L[i, jp+jj] * xp[jj]
            # L[i, jp+jj] = A[(i - (jp+jj)) + (jp+jj)*LDA]
            for kb in tl.range(0, k, BLOCK_K):
                i_vec = jp + BS + kb + offs_k
                i_mask = (i_vec < n) & (i_vec < jp + BS + k)
                xi = tl.load(x_ptr + i_vec * INCX, mask=i_mask, other=0.0)
                j_2d = (jp + offs_bs)[:, None]
                i_2d = i_vec[None, :]
                d_2d = i_2d - j_2d
                a_off_2d = d_2d + j_2d * LDA
                a_mask_2d = (d_2d >= 1) & (d_2d <= k) & (j_2d < n) & i_mask[None, :]
                A_tile = tl.load(a_ptr + a_off_2d, mask=a_mask_2d, other=0.0)
                contrib = tl.sum(A_tile * xp[:, None], axis=0)
                xi = xi - contrib
                tl.store(x_ptr + i_vec * INCX, xi, mask=i_mask)

            tl.debug_barrier()

    # ====================================================================
    # Upper / NoTrans :  U x = b      (back sub, panels right -> left)
    # Storage:  U[i,j]  (i<=j, j-i<=k)  =  A[(k-(j-i)) + j*LDA]
    # Diagonal: U[j,j]                  =  A[k + j*LDA]
    # ====================================================================
    elif (UPLO == 1) and (TRANS == 0):
        for pc in tl.range(0, n_panels):
            jp = (n_panels - 1 - pc) * BS
            panel_valid = (jp + offs_bs) < n
            xp = tl.load(x_ptr + (jp + offs_bs) * INCX, mask=panel_valid, other=0.0)

            # --- Panel solve (in registers, backward over jj) ---
            for jj_r in tl.static_range(BS):
                jj = BS - 1 - jj_r
                j = jp + jj
                one_hot = offs_bs == jj
                d = offs_bs - jj  # = ii - jj
                a_off = (k - d) + (jp + offs_bs) * LDA
                a_mask = (d >= 1) & (d <= k) & ((jp + offs_bs) < n)
                A_row = tl.load(a_ptr + a_off, mask=a_mask, other=0.0)
                dot = tl.sum(A_row * xp, axis=0)
                xj_old = tl.sum(xp * one_hot.to(tl.float32), axis=0)
                if not UNIT:
                    diag_jj = tl.load(a_ptr + k + j * LDA, mask=j < n, other=1.0)
                    xj_new = (xj_old - dot) / diag_jj
                else:
                    xj_new = xj_old - dot
                xp = tl.where(one_hot, xj_new, xp)

            tl.store(x_ptr + (jp + offs_bs) * INCX, xp, mask=panel_valid)

            # --- Panel update: rows i in [jp-k, jp) ---
            # x[i] -= sum_{jj} U[i, jp+jj] * xp[jj]
            # U[i, jp+jj] = A[(k - ((jp+jj)-i)) + (jp+jj)*LDA]
            for kb in tl.range(0, k, BLOCK_K):
                i_vec = jp - k + kb + offs_k
                i_mask = (i_vec >= 0) & (i_vec < jp)
                xi = tl.load(x_ptr + i_vec * INCX, mask=i_mask, other=0.0)
                j_2d = (jp + offs_bs)[:, None]
                i_2d = i_vec[None, :]
                d_2d = j_2d - i_2d  # = (jp+jj) - i, >=1
                a_off_2d = (k - d_2d) + j_2d * LDA
                a_mask_2d = (d_2d >= 1) & (d_2d <= k) & (j_2d < n) & i_mask[None, :]
                A_tile = tl.load(a_ptr + a_off_2d, mask=a_mask_2d, other=0.0)
                contrib = tl.sum(A_tile * xp[:, None], axis=0)
                xi = xi - contrib
                tl.store(x_ptr + i_vec * INCX, xi, mask=i_mask)

            tl.debug_barrier()

    # ====================================================================
    # Upper / Trans :  U^T x = b      (forward sub, panels left -> right)
    # Reads U from the same upper-banded layout but in the transposed
    # access pattern.
    # ====================================================================
    elif (UPLO == 1) and (TRANS == 1):
        for pc in tl.range(0, n_panels):
            jp = pc * BS
            panel_valid = (jp + offs_bs) < n
            xp = tl.load(x_ptr + (jp + offs_bs) * INCX, mask=panel_valid, other=0.0)

            # --- Panel solve ---
            for jj_r in tl.static_range(BS):
                jj = jj_r
                j = jp + jj
                one_hot = offs_bs == jj
                # U[jp+ii, j] = A[(k - (jj - ii)) + j*LDA] for ii < jj
                d = jj - offs_bs  # = jj - ii (>0 when ii<jj)
                a_off = (k - d) + j * LDA
                a_mask = (offs_bs < jj) & (d <= k) & (j < n)
                A_row = tl.load(a_ptr + a_off, mask=a_mask, other=0.0)
                dot = tl.sum(A_row * xp, axis=0)
                xj_old = tl.sum(xp * one_hot.to(tl.float32), axis=0)
                if not UNIT:
                    diag_jj = tl.load(a_ptr + k + j * LDA, mask=j < n, other=1.0)
                    xj_new = (xj_old - dot) / diag_jj
                else:
                    xj_new = xj_old - dot
                xp = tl.where(one_hot, xj_new, xp)

            tl.store(x_ptr + (jp + offs_bs) * INCX, xp, mask=panel_valid)

            # --- Panel update: rows i in [jp+BS, jp+BS+k) ---
            # x[i] -= sum_{jj} U[jp+jj, i] * xp[jj]
            # U[jp+jj, i] = A[(k - (i - (jp+jj))) + i*LDA]
            for kb in tl.range(0, k, BLOCK_K):
                i_vec = jp + BS + kb + offs_k
                i_mask = (i_vec < n) & (i_vec < jp + BS + k)
                xi = tl.load(x_ptr + i_vec * INCX, mask=i_mask, other=0.0)
                j_2d = (jp + offs_bs)[:, None]
                i_2d = i_vec[None, :]
                d_2d = i_2d - j_2d  # = i - (jp+jj)
                a_off_2d = (k - d_2d) + i_2d * LDA
                a_mask_2d = (d_2d >= 1) & (d_2d <= k) & (j_2d < n) & i_mask[None, :]
                A_tile = tl.load(a_ptr + a_off_2d, mask=a_mask_2d, other=0.0)
                contrib = tl.sum(A_tile * xp[:, None], axis=0)
                xi = xi - contrib
                tl.store(x_ptr + i_vec * INCX, xi, mask=i_mask)

            tl.debug_barrier()

    # ====================================================================
    # Lower / Trans :  L^T x = b      (back sub, panels right -> left)
    # ====================================================================
    else:
        for pc in tl.range(0, n_panels):
            jp = (n_panels - 1 - pc) * BS
            panel_valid = (jp + offs_bs) < n
            xp = tl.load(x_ptr + (jp + offs_bs) * INCX, mask=panel_valid, other=0.0)

            # --- Panel solve (backward over jj) ---
            for jj_r in tl.static_range(BS):
                jj = BS - 1 - jj_r
                j = jp + jj
                one_hot = offs_bs == jj
                # L[jp+ii, jp+jj] = A[(ii-jj) + (jp+jj)*LDA] for ii > jj
                d = offs_bs - jj
                a_off = d + j * LDA
                a_mask = (d >= 1) & (d <= k) & ((jp + offs_bs) < n)
                A_col = tl.load(a_ptr + a_off, mask=a_mask, other=0.0)
                dot = tl.sum(A_col * xp, axis=0)
                xj_old = tl.sum(xp * one_hot.to(tl.float32), axis=0)
                if not UNIT:
                    diag_jj = tl.load(a_ptr + j * LDA, mask=j < n, other=1.0)
                    xj_new = (xj_old - dot) / diag_jj
                else:
                    xj_new = xj_old - dot
                xp = tl.where(one_hot, xj_new, xp)

            tl.store(x_ptr + (jp + offs_bs) * INCX, xp, mask=panel_valid)

            # --- Panel update: rows i in [jp-k, jp) ---
            # x[i] -= sum_{jj} L[jp+jj, i] * xp[jj]
            # L[jp+jj, i] = A[((jp+jj) - i) + i*LDA]
            for kb in tl.range(0, k, BLOCK_K):
                i_vec = jp - k + kb + offs_k
                i_mask = (i_vec >= 0) & (i_vec < jp)
                xi = tl.load(x_ptr + i_vec * INCX, mask=i_mask, other=0.0)
                j_2d = (jp + offs_bs)[:, None]
                i_2d = i_vec[None, :]
                d_2d = j_2d - i_2d  # = (jp+jj) - i
                a_off_2d = d_2d + i_2d * LDA
                a_mask_2d = (d_2d >= 1) & (d_2d <= k) & (j_2d < n) & i_mask[None, :]
                A_tile = tl.load(a_ptr + a_off_2d, mask=a_mask_2d, other=0.0)
                contrib = tl.sum(A_tile * xp[:, None], axis=0)
                xi = xi - contrib
                tl.store(x_ptr + i_vec * INCX, xi, mask=i_mask)

            tl.debug_barrier()


# --------------------------------------------------------------------------
# Argument validation (unchanged from the original)
# --------------------------------------------------------------------------
def _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok):
    assert A.is_contiguous() and x.is_contiguous()
    assert A.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
    allowed = (
        [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
        if complex_ok
        else [CUBLAS_OP_N, CUBLAS_OP_T]
    )
    assert trans in allowed
    assert diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    assert incx > 0
    assert n >= 0 and k >= 0
    assert lda >= k + 1
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= n * lda


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def stbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a real single-precision triangular banded system in-place."""
    assert A.dtype == torch.float32 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    bs = _pick_bs(n, k)

    with torch_device_fn.device(A.device):
        grid = (1,)
        stbsv_kernel[grid](
            A,
            x,
            n,
            k,
            lda,
            incx,
            _band_bucket(k + 1),
            _mode_key(uplo, trans_flag, unit),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            BS=bs,
        )

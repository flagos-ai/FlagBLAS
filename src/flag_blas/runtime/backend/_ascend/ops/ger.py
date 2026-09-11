import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.ger import _check_ger_common, _scalar_to_complex_parts
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_MAX_CORE_DIM = 65535


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cger"),
    key=["m", "n", "LDA", "INCX", "INCY"],
    restore_value=["A_ptr"],
)
@triton.jit
def cger_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    m,
    n,
    INCX,
    INCY,
    LDA,
    CONJ_Y: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles_n = tl.cdiv(n, BLOCK_SIZE_N)
    tile_count = tl.cdiv(m, BLOCK_SIZE_M) * tiles_n
    program_count = tl.num_programs(0)

    while tile_id < tile_count:
        pid_m = tile_id // tiles_n
        pid_n = tile_id % tiles_n
        rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        row_mask = rows < m
        col_mask = cols < n

        x_off = rows * INCX * 2
        y_off = cols * INCY * 2
        x_real = tl.load(x_ptr + x_off, mask=row_mask, other=0.0)
        x_imag = tl.load(x_ptr + x_off + 1, mask=row_mask, other=0.0)
        y_real = tl.load(y_ptr + y_off, mask=col_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=col_mask, other=0.0)
        if CONJ_Y:
            y_imag = -y_imag

        ax_real = alpha_real * x_real - alpha_imag * x_imag
        ax_imag = alpha_real * x_imag + alpha_imag * x_real
        update_real = (
            ax_real[:, None] * y_real[None, :] - ax_imag[:, None] * y_imag[None, :]
        )
        update_imag = (
            ax_real[:, None] * y_imag[None, :] + ax_imag[:, None] * y_real[None, :]
        )

        elem_off = rows[:, None] * LDA + cols[None, :]
        a_off = elem_off * 2
        mask = row_mask[:, None] & col_mask[None, :]
        tl.atomic_add(A_ptr + a_off, update_real, mask=mask, sem="relaxed")
        tl.atomic_add(A_ptr + a_off + 1, update_imag, mask=mask, sem="relaxed")
        tile_id += program_count


def _grid(m, n):
    def grid(meta):
        tiles_m = triton.cdiv(m, meta["BLOCK_SIZE_M"])
        tiles_n = triton.cdiv(n, meta["BLOCK_SIZE_N"])
        return (min(tiles_m * tiles_n, _MAX_CORE_DIM),)

    return grid


def _cger(m, n, alpha, x, incx, y, incy, A, lda, conj_y):
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.complex64):
        return
    alpha_real, alpha_imag = _scalar_to_complex_parts(alpha)
    if alpha_real == 0.0 and alpha_imag == 0.0:
        return

    with torch_device_fn.device(A.device):
        cger_kernel[_grid(m, n)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            torch.view_as_real(A),
            alpha_real,
            alpha_imag,
            m,
            n,
            incx,
            incy,
            lda,
            CONJ_Y=conj_y,
        )


def cgeru(m, n, alpha, x, incx, y, incy, A, lda):
    _cger(m, n, alpha, x, incx, y, incy, A, lda, False)


def cgerc(m, n, alpha, x, incx, y, incy, A, lda):
    _cger(m, n, alpha, x, incx, y, incy, A, lda, True)

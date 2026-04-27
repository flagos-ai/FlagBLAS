import ctypes
import ctypes.util
import math
import pytest
import torch
import cupy as cp
import flag_blas

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_DIAG_UNIT = 1

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
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = load_cublas()


def cublas_stbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0:
        return

    handle = cp.cuda.device.get_cublas_handle()
    func = _cublas.cublasStbsv_v2

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasStbsv_v2 execution failed with error code: {status}")


# --------------------------------------------------------------------------
# Test grid (kept compact - the parametrize space is the cartesian product
# of n x k x uplo x trans x diag, so we use a moderate size set).
# --------------------------------------------------------------------------
STBSV_SIZES = [1, 2, 32, 63, 64, 128, 256, 512, 1024, 4096]
STBSV_KS = [0, 1, 4, 16, 64]
STBSV_STRIDE_SIZES = [64, 127, 256]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
TRANS_MODES = [CUBLAS_OP_N, CUBLAS_OP_T]
DIAG_MODES = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
STRIDES = [(1,), (2,)]


def make_triangular_banded(n, k, lda, uplo, dtype, device, unit_diag=False):
    """Generate a strongly diagonally-dominant triangular banded matrix.

    Diagonal magnitude is set to ~ 2*(k+1) so that the system is
    well-conditioned and the solve is numerically stable across all sizes.
    """
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    diag_floor = 2.0 * (k + 1) + 1.0

    if uplo == CUBLAS_FILL_MODE_UPPER:
        # Row k holds the diagonal; rows 0..k-1 hold superdiagonals
        for j in range(n):
            i_min = max(0, j - k)
            cnt = j - i_min + 1
            if cnt > 0:
                vals = torch.randn(cnt, dtype=dtype, device=device) * 0.1
                if unit_diag:
                    vals[-1] = 1.0
                else:
                    sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                    vals[-1] = sign * (diag_floor + torch.rand(1).item())
                A[j, k + i_min - j:k + 1] = vals
    else:
        # Row 0 holds the diagonal; rows 1..k hold subdiagonals
        for j in range(n):
            i_max = min(n - 1, j + k)
            cnt = i_max - j + 1
            if cnt > 0:
                vals = torch.randn(cnt, dtype=dtype, device=device) * 0.1
                if unit_diag:
                    vals[0] = 1.0
                else:
                    sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                    vals[0] = sign * (diag_floor + torch.rand(1).item())
                A[j, 0:cnt] = vals
    return A.contiguous()


def _stbsv_tol(dtype, n, k):
    # Triangular solve compounds error roughly with (k+1)*n.
    K = max(1, n)
    if dtype == torch.float32:
        return min(max(1e-4, 5e-6 * math.sqrt(K)), 5e-2)
    if dtype == torch.float64:
        return min(max(1e-12, 5e-14 * math.sqrt(K)), 1e-9)
    raise ValueError(f"Unsupported dtype {dtype}")


def _effective_k(n, k):
    return min(k, max(0, n - 1))


# --------------------------------------------------------------------------
# Accuracy: contiguous x, full sweep of (uplo, trans, diag)
# --------------------------------------------------------------------------
@pytest.mark.stbsv
@pytest.mark.parametrize("n", STBSV_SIZES)
@pytest.mark.parametrize("k", STBSV_KS)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_stbsv(n, k, uplo, trans, diag):
    k = _effective_k(n, k)
    dtype = torch.float32
    lda = k + 1 + 2          # add padding to exercise non-trivial lda

    A = make_triangular_banded(
        n, k, lda, uplo, dtype, flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_stbsv_reference(uplo, trans, diag, n, k, A, lda, ref_x, 1)
    flag_blas.ops.stbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _stbsv_tol(dtype, n, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


# --------------------------------------------------------------------------
# Accuracy: strided x
# --------------------------------------------------------------------------
@pytest.mark.stbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2])
def test_accuracy_stbsv_stride(n, k, uplo, trans, incx):
    k = _effective_k(n, k)
    dtype = torch.float32
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_stbsv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    flag_blas.ops.stbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _stbsv_tol(dtype, n, k)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
@pytest.mark.stbsv
def test_stbsv_n_zero():
    A = torch.empty((0, 1), dtype=torch.float32, device=flag_blas.device)
    x = torch.empty((0,), dtype=torch.float32, device=flag_blas.device)
    flag_blas.ops.stbsv(
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        0, 0, A, 1, x, 1,
    )
    assert x.numel() == 0


@pytest.mark.stbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
def test_stbsv_k_zero(uplo, trans):
    """k=0 => purely diagonal solve."""
    n, k = 256, 0
    lda = 1
    dtype = torch.float32
    diag = CUBLAS_DIAG_NON_UNIT

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_x = x.clone()

    cublas_stbsv_reference(uplo, trans, diag, n, k, A, lda, ref_x, 1)
    flag_blas.ops.stbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    torch.testing.assert_close(x, ref_x, rtol=1e-5, atol=1e-5)


@pytest.mark.stbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
def test_stbsv_unit_diag_ignored(uplo, trans):
    """When DIAG=UNIT the implementation must ignore the diagonal entries
    of A (cuBLAS does the same). Poisoning them with NaN must not affect
    the result."""
    n, k = 128, 8
    lda = k + 1 + 1
    dtype = torch.float32

    A_clean = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device, unit_diag=True)
    A_dirty = A_clean.clone()
    diag_row = k if uplo == CUBLAS_FILL_MODE_UPPER else 0
    A_dirty[:, diag_row] = float("nan")

    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    x_clean = x.clone()
    x_dirty = x.clone()

    flag_blas.ops.stbsv(uplo, trans, CUBLAS_DIAG_UNIT, n, k, A_clean, lda, x_clean, 1)
    flag_blas.ops.stbsv(uplo, trans, CUBLAS_DIAG_UNIT, n, k, A_dirty, lda, x_dirty, 1)

    tol = _stbsv_tol(dtype, n, k)
    torch.testing.assert_close(x_dirty, x_clean, rtol=tol, atol=tol)


@pytest.mark.stbsv
def test_stbsv_solve_then_multiply_roundtrip():
    """A * (A^{-1} * x) ≈ x  -- a self-consistency check independent of
    the cuBLAS reference."""
    n, k = 512, 16
    lda = k + 1
    dtype = torch.float32
    uplo = CUBLAS_FILL_MODE_LOWER

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x_orig = torch.randn(n, dtype=dtype, device=flag_blas.device)

    # Solve A * y = x_orig => y stored in x_buf
    x_buf = x_orig.clone()
    flag_blas.ops.stbsv(uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x_buf, 1)

    # Re-multiply: compute A * y and compare with x_orig
    # We do the multiply in dense-matrix form for the reference.
    A_dense = torch.zeros(n, n, dtype=dtype, device=flag_blas.device)
    for j in range(n):
        for i in range(j, min(n, j + k + 1)):
            A_dense[i, j] = A[j, i - j]
    Ay = A_dense @ x_buf

    tol = _stbsv_tol(dtype, n, k)
    torch.testing.assert_close(Ay, x_orig, rtol=tol, atol=tol)

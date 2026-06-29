import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.utils import shape_utils

STBSV_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    12288,
    16384,
]

STBSV_KS = [1, 4, 16, 64, 256]


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


def cublas_stbsv_baseline(
    A,
    x,
    uplo,
    trans,
    diag,
    n,
    k,
    lda,
    incx,
    handle,
    c_func,
    **kwargs,
):
    if n == 0:
        return x
    status = c_func(
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
        raise RuntimeError(f"cublasStbsv_v2 failed with status code: {status}")
    return x


def gems_stbsv_wrapper(A, x, uplo, trans, diag, n, k, lda, incx, **kwargs):
    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, incx)
    return x


def _make_triangular_banded(n, k, lda, uplo, dtype, device):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    A = torch.randn((n, lda), dtype=dtype, device=device) * 0.1
    diag_floor = 2.0 * (k + 1) + 1.0
    cols = torch.arange(lda, device=device).view(1, lda)
    j = torch.arange(n, device=device).view(n, 1)

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = (cols >= torch.clamp(k - j, min=0)) & (cols <= k)
        diag_col = k
    else:
        valid = cols <= torch.clamp(n - 1 - j, max=k)
        diag_col = 0

    A = A.masked_fill(~valid, 0.0)
    A[:, diag_col] = diag_floor
    return A.contiguous()


def _stored_band_nnz(n, k):
    if n <= 0:
        return 0
    if k >= n - 1:
        return n * (n + 1) // 2
    return (k + 1) * (k + 2) // 2 + (n - k - 1) * (k + 1)


class StbsvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.trans = trans
        self.diag = diag
        self.ks = STBSV_KS

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in STBSV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            c_func = _cublas.cublasStbsv_v2
        elif cur_dtype == torch.float64:
            c_func = _cublas.cublasDtbsv_v2
        elif cur_dtype == torch.complex64:
            c_func = _cublas.cublasCtbsv_v2
        elif cur_dtype == torch.complex128:
            c_func = _cublas.cublasZtbsv_v2
        else:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")

        seen = set()
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            for k_req in self.ks:
                k = min(k_req, max(0, n - 1))
                key = (n, k)
                if key in seen:
                    continue
                seen.add(key)
                lda = k + 1
                A = _make_triangular_banded(
                    n, k, lda, self.uplo, cur_dtype, self.device
                )
                x = torch.randn(n, dtype=cur_dtype, device=self.device)

                yield A, x.clone(), {
                    "uplo": self.uplo,
                    "trans": self.trans,
                    "diag": self.diag,
                    "n": n,
                    "k": k,
                    "lda": lda,
                    "incx": 1,
                    "handle": handle,
                    "c_func": c_func,
                }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
        # ~2 flops per stored band element (1 mul + 1 add) for the
        # off-diagonal updates, plus ~n divisions for the diagonal.
        # The off-diagonals dominate.
        nnz = _stored_band_nnz(n, k)
        return 2 * nnz

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        n = x.numel()
        k = A.shape[-1] - 1
        stored = _stored_band_nnz(n, k)
        a_bytes = stored * A.element_size()
        # x is read and written exactly once for each unknown.
        io_amount = a_bytes + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


# --------------------------------------------------------------------------
# Top-level perf entry points
# --------------------------------------------------------------------------
@pytest.mark.stbsv
def test_perf_stbsv():
    bench = StbsvBenchmark(
        op_name="stbsv",
        torch_op=cublas_stbsv_baseline,
        gems_op=gems_stbsv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    bench.run()


@pytest.mark.stbsv
def test_perf_stbsv_upper():
    bench = StbsvBenchmark(
        op_name="stbsv_upper",
        torch_op=cublas_stbsv_baseline,
        gems_op=gems_stbsv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    bench.run()


@pytest.mark.stbsv
def test_perf_stbsv_trans():
    bench = StbsvBenchmark(
        op_name="stbsv_trans",
        torch_op=cublas_stbsv_baseline,
        gems_op=gems_stbsv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    bench.run()


@pytest.mark.stbsv
def test_perf_stbsv_upper_trans():
    bench = StbsvBenchmark(
        op_name="stbsv_upper_trans",
        torch_op=cublas_stbsv_baseline,
        gems_op=gems_stbsv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    bench.run()

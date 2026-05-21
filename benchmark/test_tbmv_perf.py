import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.utils import shape_utils

TBMV_SIZES = [
    64,
    127,
    128,
    255,
    256,
    1024,
    4096,
    8192,
    16384,
]

TBMV_KS = [1, 4, 32, 48, 128, 512]


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

_CUBLAS_TBMV_FUNCS = {
    torch.float32: _cublas.cublasStbmv_v2,
    torch.float64: _cublas.cublasDtbmv_v2,
    torch.complex64: _cublas.cublasCtbmv_v2,
    torch.complex128: _cublas.cublasZtbmv_v2,
}


def cublas_tbmv_baseline(
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
        raise RuntimeError(f"cublasXtbmv_v2 failed with status code: {status}")
    return x


def _gems_wrapper(op):
    def _impl(A, x, uplo, trans, diag, n, k, lda, incx, handle, **kwargs):
        op(uplo, trans, diag, n, k, A, lda, x, incx)
        return x

    return _impl


gems_stbmv_wrapper = _gems_wrapper(flag_blas.ops.stbmv)
gems_dtbmv_wrapper = _gems_wrapper(flag_blas.ops.dtbmv)
gems_ctbmv_wrapper = _gems_wrapper(flag_blas.ops.ctbmv)
gems_ztbmv_wrapper = _gems_wrapper(flag_blas.ops.ztbmv)


def _generate_triangular_banded(n, k, lda, uplo, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        for j in range(n):
            i_min = max(0, j - k)
            cnt = j - i_min + 1
            if cnt > 0:
                vals = torch.randn(cnt, dtype=dtype, device=device) * 0.1
                A[j, k + i_min - j : k + 1] = vals
    else:
        for j in range(n):
            i_max = min(n - 1, j + k)
            cnt = i_max - j + 1
            if cnt > 0:
                vals = torch.randn(cnt, dtype=dtype, device=device) * 0.1
                A[j, 0:cnt] = vals
    return A.contiguous()


def _triangular_banded_nnz(n, k):
    if n <= 0:
        return 0
    if k >= n - 1:
        return n * (n + 1) // 2
    return (k + 1) * (k + 2) // 2 + (n - k - 1) * (k + 1)


class TbmvBenchmark(Benchmark):
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
        self.ks = TBMV_KS

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in TBMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_TBMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func = _CUBLAS_TBMV_FUNCS[cur_dtype]

        seen = set()
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            for k_req in self.ks:
                k = min(k_req, max(0, n - 1))
                # Skip near-dense triangular cases; they are better covered by TRMV.
                if k * 2 >= n:
                    continue
                key = (n, k)
                if key in seen:
                    continue
                seen.add(key)
                lda = k + 1
                A = _generate_triangular_banded(
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
        nnz = _triangular_banded_nnz(n, k)
        A = args[0]
        if A.dtype in (torch.complex64, torch.complex128):
            return 8 * nnz
        return 2 * nnz

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        n = x.numel()
        k = A.shape[-1] - 1
        nnz = _triangular_banded_nnz(n, k)
        a_bytes = nnz * A.element_size()
        io_amount = a_bytes + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["k"] + 1

    def clone_correctness_inputs(self, args, kwargs):
        A, x = args
        ref_args = (A, x.clone())
        blas_args = (A, x.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.stbmv
def test_perf_stbmv():
    bench = TbmvBenchmark(
        op_name="stbmv",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_stbmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.stbmv
def test_perf_stbmv_upper():
    bench = TbmvBenchmark(
        op_name="stbmv_upper",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_stbmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.stbmv
def test_perf_stbmv_trans():
    bench = TbmvBenchmark(
        op_name="stbmv_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_stbmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtbmv
def test_perf_stbmv_upper_trans():
    bench = TbmvBenchmark(
        op_name="stbmv_upper_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_stbmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.stbmv
def test_perf_stbmv_unit():
    bench = TbmvBenchmark(
        op_name="stbmv_unit",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_stbmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtbmv
def test_perf_dtbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="dtbmv",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_dtbmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtbmv
def test_perf_dtbmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="dtbmv_upper",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_dtbmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtbmv
def test_perf_dtbmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="dtbmv_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_dtbmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctbmv
def test_perf_dtbmv_upper_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="dtbmv_upper_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_dtbmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtbmv
def test_perf_dtbmv_unit():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="dtbmv_unit",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_dtbmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctbmv
def test_perf_ctbmv():
    bench = TbmvBenchmark(
        op_name="ctbmv",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ctbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctbmv
def test_perf_ctbmv_upper():
    bench = TbmvBenchmark(
        op_name="ctbmv_upper",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ctbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctbmv
def test_perf_ctbmv_trans():
    bench = TbmvBenchmark(
        op_name="ctbmv_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ctbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctbmv
def test_perf_ctbmv_conj():
    bench = TbmvBenchmark(
        op_name="ctbmv_conj",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ctbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ctbmv_upper_trans():
    bench = TbmvBenchmark(
        op_name="ctbmv_upper_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ctbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctbmv
def test_perf_ctbmv_upper_conj():
    bench = TbmvBenchmark(
        op_name="ctbmv_upper_conj",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ctbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctbmv
def test_perf_ctbmv_unit():
    bench = TbmvBenchmark(
        op_name="ctbmv_unit",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ctbmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ztbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="ztbmv",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ztbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ztbmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="ztbmv_upper",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ztbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ztbmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="ztbmv_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ztbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ztbmv_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="ztbmv_conj",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ztbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ztbmv_upper_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="ztbmv_upper_trans",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ztbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ztbmv_upper_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="ztbmv_upper_conj",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ztbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztbmv
def test_perf_ztbmv_unit():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TbmvBenchmark(
        op_name="ztbmv_unit",
        torch_op=cublas_tbmv_baseline,
        gems_op=gems_ztbmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)

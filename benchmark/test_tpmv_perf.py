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

TPMV_SIZES = [
    31,
    127,
    128,
    192,
    256,
    384,
    512,
    768,
    1023,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    10000,
    12288,
    16384,
]


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

_CUBLAS_TPMV_FUNCS = {
    torch.float32: _cublas.cublasStpmv_v2,
    torch.float64: _cublas.cublasDtpmv_v2,
    torch.complex64: _cublas.cublasCtpmv_v2,
    torch.complex128: _cublas.cublasZtpmv_v2,
}


def cublas_tpmv_baseline(
    AP,
    x,
    uplo,
    trans,
    diag,
    n,
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
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtpmv_v2 failed with status code: {status}")
    return x


def _gems_wrapper(op):
    def _impl(AP, x, uplo, trans, diag, n, incx, handle, **kwargs):
        op(uplo, trans, diag, n, AP, x, incx)
        return x

    return _impl


gems_stpmv_wrapper = _gems_wrapper(flag_blas.stpmv)
gems_dtpmv_wrapper = _gems_wrapper(flag_blas.dtpmv)
gems_ctpmv_wrapper = _gems_wrapper(flag_blas.ctpmv)
gems_ztpmv_wrapper = _gems_wrapper(flag_blas.ztpmv)


def _generate_packed_triangular(n, dtype, device):
    return (
        torch.randn(n * (n + 1) // 2, dtype=dtype, device=device) * 0.1
    ).contiguous()


class TpmvBenchmark(Benchmark):
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

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in TPMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_TPMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func = _CUBLAS_TPMV_FUNCS[cur_dtype]

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP = _generate_packed_triangular(n, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)

            yield AP, x.clone(), {
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "n": n,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        nnz = n * (n + 1) // 2
        AP = args[0]
        if AP.dtype in (torch.complex64, torch.complex128):
            return 4 * nnz
        return nnz

    def get_gbps(self, args, latency):
        AP, x = args[0], args[1]
        io_amount = shape_utils.size_in_bytes(AP) + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]

    def clone_correctness_inputs(self, args, kwargs):
        AP, x = args
        ref_args = (AP, x.clone())
        blas_args = (AP, x.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.stpmv
def test_perf_stpmv():
    bench = TpmvBenchmark(
        op_name="stpmv",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_stpmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.stpmv
def test_perf_stpmv_upper():
    bench = TpmvBenchmark(
        op_name="stpmv_upper",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_stpmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.stpmv
def test_perf_stpmv_trans():
    bench = TpmvBenchmark(
        op_name="stpmv_trans",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_stpmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.stpmv
def test_perf_stpmv_upper_trans():
    bench = TpmvBenchmark(
        op_name="stpmv_upper_trans",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_stpmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.stpmv
def test_perf_stpmv_unit():
    bench = TpmvBenchmark(
        op_name="stpmv_unit",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_stpmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtpmv
def test_perf_dtpmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="dtpmv",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_dtpmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtpmv
def test_perf_dtpmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="dtpmv_upper",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_dtpmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtpmv
def test_perf_dtpmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="dtpmv_trans",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_dtpmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtpmv
def test_perf_dtpmv_upper_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="dtpmv_upper_trans",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_dtpmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtpmv
def test_perf_dtpmv_unit():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="dtpmv_unit",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_dtpmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctpmv
def test_perf_ctpmv():
    bench = TpmvBenchmark(
        op_name="ctpmv",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ctpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctpmv
def test_perf_ctpmv_upper():
    bench = TpmvBenchmark(
        op_name="ctpmv_upper",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ctpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctpmv
def test_perf_ctpmv_trans():
    bench = TpmvBenchmark(
        op_name="ctpmv_trans",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ctpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctpmv
def test_perf_ctpmv_conj():
    bench = TpmvBenchmark(
        op_name="ctpmv_conj",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ctpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctpmv
def test_perf_ctpmv_upper_conj():
    bench = TpmvBenchmark(
        op_name="ctpmv_upper_conj",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ctpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctpmv
def test_perf_ctpmv_unit():
    bench = TpmvBenchmark(
        op_name="ctpmv_unit",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ctpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztpmv
def test_perf_ztpmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="ztpmv",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ztpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztpmv
def test_perf_ztpmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="ztpmv_upper",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ztpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztpmv
def test_perf_ztpmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="ztpmv_trans",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ztpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztpmv
def test_perf_ztpmv_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="ztpmv_conj",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ztpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztpmv
def test_perf_ztpmv_upper_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="ztpmv_upper_conj",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ztpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztpmv
def test_perf_ztpmv_unit():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TpmvBenchmark(
        op_name="ztpmv_unit",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ztpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)

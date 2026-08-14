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

TRMV_SIZES = [
    31,
    127,
    128,
    192,
    256,
    384,
    512,
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

_CUBLAS_TRMV_FUNCS = {
    torch.float32: _cublas.cublasStrmv_v2,
    torch.float64: _cublas.cublasDtrmv_v2,
    torch.complex64: _cublas.cublasCtrmv_v2,
    torch.complex128: _cublas.cublasZtrmv_v2,
}


def cublas_trmv_baseline(
    A,
    x,
    uplo,
    trans,
    diag,
    n,
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
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtrmv_v2 failed with status code: {status}")
    return x


def _gems_wrapper(op):
    def _impl(A, x, uplo, trans, diag, n, lda, incx, handle, **kwargs):
        op(uplo, trans, diag, n, A, lda, x, incx)
        return x

    return _impl


gems_strmv_wrapper = _gems_wrapper(flag_blas.ops.strmv)
gems_dtrmv_wrapper = _gems_wrapper(flag_blas.ops.dtrmv)
gems_ctrmv_wrapper = _gems_wrapper(flag_blas.ops.ctrmv)
gems_ztrmv_wrapper = _gems_wrapper(flag_blas.ops.ztrmv)


def _generate_triangular_A(n, lda, uplo, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    vals = torch.randn(n, n, dtype=dtype, device=device) * 0.1
    if uplo == CUBLAS_FILL_MODE_UPPER:
        for j in range(n):
            A[j, : j + 1] = vals[: j + 1, j]
    else:
        for j in range(n):
            A[j, j:n] = vals[j:n, j]
    return A.contiguous()


class TrmvBenchmark(Benchmark):
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
        self.shapes = [(n,) for n in TRMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_TRMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func = _CUBLAS_TRMV_FUNCS[cur_dtype]

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = n
            A = _generate_triangular_A(n, lda, self.uplo, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)

            yield A, x.clone(), {
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "n": n,
                "lda": lda,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        nnz = n * (n + 1) // 2
        A = args[0]
        if A.dtype in (torch.complex64, torch.complex128):
            return 4 * nnz
        return nnz

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        n = x.numel()
        a_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = a_bytes + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]

    def clone_correctness_inputs(self, args, kwargs):
        A, x = args
        ref_args = (A, x.clone())
        blas_args = (A, x.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.strmv
def test_perf_strmv():
    bench = TrmvBenchmark(
        op_name="strmv",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_strmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.strmv
def test_perf_strmv_upper():
    bench = TrmvBenchmark(
        op_name="strmv_upper",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_strmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.strmv
def test_perf_strmv_trans():
    bench = TrmvBenchmark(
        op_name="strmv_trans",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_strmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.strmv
def test_perf_strmv_upper_trans():
    bench = TrmvBenchmark(
        op_name="strmv_upper_trans",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_strmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.strmv
def test_perf_strmv_unit():
    bench = TrmvBenchmark(
        op_name="strmv_unit",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_strmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtrmv
def test_perf_dtrmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="dtrmv",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_dtrmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtrmv
def test_perf_dtrmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="dtrmv_upper",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_dtrmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtrmv
def test_perf_dtrmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="dtrmv_trans",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_dtrmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtrmv
def test_perf_dtrmv_upper_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="dtrmv_upper_trans",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_dtrmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dtrmv
def test_perf_dtrmv_unit():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="dtrmv_unit",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_dtrmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctrmv
def test_perf_ctrmv():
    bench = TrmvBenchmark(
        op_name="ctrmv",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ctrmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctrmv
def test_perf_ctrmv_upper():
    bench = TrmvBenchmark(
        op_name="ctrmv_upper",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ctrmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctrmv
def test_perf_ctrmv_trans():
    bench = TrmvBenchmark(
        op_name="ctrmv_trans",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ctrmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctrmv
def test_perf_ctrmv_conj():
    bench = TrmvBenchmark(
        op_name="ctrmv_conj",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ctrmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctrmv
def test_perf_ctrmv_upper_conj():
    bench = TrmvBenchmark(
        op_name="ctrmv_upper_conj",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ctrmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ctrmv
def test_perf_ctrmv_unit():
    bench = TrmvBenchmark(
        op_name="ctrmv_unit",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ctrmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztrmv
def test_perf_ztrmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="ztrmv",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ztrmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztrmv
def test_perf_ztrmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="ztrmv_upper",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ztrmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztrmv
def test_perf_ztrmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="ztrmv_trans",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ztrmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztrmv
def test_perf_ztrmv_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="ztrmv_conj",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ztrmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztrmv
def test_perf_ztrmv_upper_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="ztrmv_upper_conj",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ztrmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ztrmv
def test_perf_ztrmv_unit():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrmvBenchmark(
        op_name="ztrmv_unit",
        torch_op=cublas_trmv_baseline,
        gems_op=gems_ztrmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)

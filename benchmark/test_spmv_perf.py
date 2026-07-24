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
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

SPMV_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
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

_CUBLAS_SPMV_FUNCS = {
    torch.float32: (_cublas.cublasSspmv_v2, ctypes.c_float),
    torch.float64: (_cublas.cublasDspmv_v2, ctypes.c_double),
}


def cublas_spmv_baseline(
    AP,
    x,
    y,
    uplo,
    n,
    alpha,
    incx,
    beta,
    incy,
    handle,
    c_func,
    alpha_c,
    beta_c,
    **kwargs,
):
    if n == 0:
        return y
    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXspmv_v2 failed with status code: {status}")
    return y


def _gems_wrapper(op):
    def _impl(AP, x, y, uplo, n, alpha, incx, beta, incy, handle, **kwargs):
        op(uplo, n, alpha, AP, x, incx, beta, y, incy)
        return y

    return _impl


gems_sspmv_wrapper = _gems_wrapper(flag_blas.ops.sspmv)
gems_dspmv_wrapper = _gems_wrapper(flag_blas.ops.dspmv)


def _generate_packed_sym(n, uplo, dtype, device):
    return torch.randn(n * (n + 1) // 2, dtype=dtype, device=device)


class SpmvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SPMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_SPMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func, ctor = _CUBLAS_SPMV_FUNCS[cur_dtype]
        alpha_c = ctor(self.alpha)
        beta_c = ctor(self.beta)

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP = _generate_packed_sym(n, self.uplo, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)

            yield AP, x, y, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
                "beta_c": beta_c,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 2 * n * n

    def get_gbps(self, args, latency):
        AP, x, y = args[0], args[1], args[2]
        a_bytes = AP.numel() * AP.element_size()
        io_amount = (
            a_bytes + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs.get("n", 0))

    def clone_correctness_inputs(self, args, kwargs):
        AP, x, y = args
        ref_args = (AP, x, y.clone())
        blas_args = (AP, x, y.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.sspmv
def test_perf_sspmv():
    bench = SpmvBenchmark(
        op_name="sspmv",
        torch_op=cublas_spmv_baseline,
        gems_op=gems_sspmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sspmv
def test_perf_sspmv_upper():
    bench = SpmvBenchmark(
        op_name="sspmv_upper",
        torch_op=cublas_spmv_baseline,
        gems_op=gems_sspmv_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dspmv
def test_perf_dspmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SpmvBenchmark(
        op_name="dspmv",
        torch_op=cublas_spmv_baseline,
        gems_op=gems_dspmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dspmv
def test_perf_dspmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SpmvBenchmark(
        op_name="dspmv_upper",
        torch_op=cublas_spmv_baseline,
        gems_op=gems_dspmv_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)

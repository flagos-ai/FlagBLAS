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

from typing import Generator

import numpy as np
import pytest
import torch

import flag_blas
from benchmark.attri_util import L1_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    import ctypes

    from tests.hipblas_reference import (
        check_hipblas_status,
        get_hipblas_context,
        get_hipblas_library,
    )
elif flag_blas.vendor_name == "mthreads":
    from benchmark.mublas_compat import cp, cublas
elif flag_blas.vendor_name != "ascend":
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


def blas_sscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.sscal(n, alpha, x, incx)
    return x


def blas_dscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.dscal(n, alpha, x, incx)
    return x


def blas_cscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.cscal(n, alpha, x, incx)
    return x


def blas_zscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.zscal(n, alpha, x, incx)
    return x


def blas_csscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.csscal(n, alpha, x, incx)
    return x


def blas_zdscal_wrapper(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
    flag_blas.zdscal(n, alpha, x, incx)
    return x


if IS_HYGON:

    def _resolve_hipblas_scal(op_name):
        symbol = {
            "sscal": "hipblasSscal",
            "dscal": "hipblasDscal",
            "cscal": "hipblasCscal_v2",
            "zscal": "hipblasZscal_v2",
            "csscal": "hipblasCsscal_v2",
            "zdscal": "hipblasZdscal_v2",
        }[op_name]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_scal_baseline(
        x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None, c_func=None
    ):
        check_hipblas_status(
            c_func(handle, n, alpha_ptr, x.data_ptr(), incx), "hipBLAS SCAL"
        )
        return x

    def _hygon_scal_wrapper(op):
        def wrapper(
            x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None, c_func=None
        ):
            return op(
                x, alpha=alpha, incx=incx, n=n, handle=handle, alpha_ptr=alpha_ptr
            )

        return wrapper

    cublas_sscal = hipblas_scal_baseline
    blas_sscal_wrapper = _hygon_scal_wrapper(blas_sscal_wrapper)
    cublas_dscal = hipblas_scal_baseline
    blas_dscal_wrapper = _hygon_scal_wrapper(blas_dscal_wrapper)
    cublas_cscal = hipblas_scal_baseline
    blas_cscal_wrapper = _hygon_scal_wrapper(blas_cscal_wrapper)
    cublas_zscal = hipblas_scal_baseline
    blas_zscal_wrapper = _hygon_scal_wrapper(blas_zscal_wrapper)
    cublas_csscal = hipblas_scal_baseline
    blas_csscal_wrapper = _hygon_scal_wrapper(blas_csscal_wrapper)
    cublas_zdscal = hipblas_scal_baseline
    blas_zdscal_wrapper = _hygon_scal_wrapper(blas_zdscal_wrapper)

else:

    def cublas_sscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
        cublas.sscal(handle, n, alpha_ptr, x.data_ptr(), incx)
        return x

    def cublas_dscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
        cublas.dscal(handle, n, alpha_ptr, x.data_ptr(), incx)
        return x

    def cublas_cscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
        cublas.cscal(handle, n, alpha_ptr, x.data_ptr(), incx)
        return x

    def cublas_zscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
        cublas.zscal(handle, n, alpha_ptr, x.data_ptr(), incx)
        return x

    def cublas_csscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
        cublas.csscal(handle, n, alpha_ptr, x.data_ptr(), incx)
        return x

    def cublas_zdscal(x, alpha=1e-5, incx=1, n=None, handle=None, alpha_ptr=None):
        cublas.zdscal(handle, n, alpha_ptr, x.data_ptr(), incx)
        return x


class ScalBenchmark(Benchmark):
    def __init__(self, *args, alpha=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_scal(self.op_name.split("_", 1)[0])
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            alpha_np = np.array(self.alpha, dtype=np.float32)
        elif cur_dtype == torch.float64:
            alpha_np = np.array(self.alpha, dtype=np.float64)
        elif cur_dtype == torch.complex64:
            alpha_np = np.array(self.alpha, dtype=np.complex64)
        elif cur_dtype == torch.complex128:
            alpha_np = np.array(self.alpha, dtype=np.complex128)
        else:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp.numel()
            if n > 0:
                if IS_HYGON:
                    _, handle = get_hipblas_context(inp)
                kwargs = {
                    "alpha": self.alpha,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }
                if IS_HYGON:
                    kwargs["c_func"] = c_func
                yield inp, kwargs

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = 2 * shape_utils.size_in_bytes(inp)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        if not x.dtype.is_complex:
            return n
        # Real-times-complex scaling needs two multiplies, complex-times-complex six.
        real_alpha = self.op_name.split("_", 1)[0] in ("csscal", "zdscal")
        return (2 if real_alpha else 6) * n


class ScalStrideBenchmark(Benchmark):
    def __init__(self, *args, incx=1, alpha=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_scal(self.op_name.split("_", 1)[0])
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype == torch.float32:
            alpha_np = np.array(self.alpha, dtype=np.float32)
        elif cur_dtype == torch.float64:
            alpha_np = np.array(self.alpha, dtype=np.float64)
        elif cur_dtype == torch.complex64:
            alpha_np = np.array(self.alpha, dtype=np.complex64)
        elif cur_dtype == torch.complex128:
            alpha_np = np.array(self.alpha, dtype=np.complex128)
        else:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")

        alpha_ptr = alpha_np.ctypes.data

        for shape in self.shapes:
            n = shape[0]
            inp = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            if n > 0:
                if IS_HYGON:
                    _, handle = get_hipblas_context(inp)
                kwargs = {
                    "alpha": self.alpha,
                    "incx": self.incx,
                    "n": n,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                }
                if IS_HYGON:
                    kwargs["c_func"] = c_func
                yield inp, kwargs

    def get_gbps(self, args, latency):
        inp = args[0]
        n = (inp.numel() + self.incx - 1) // self.incx
        element_size = inp.element_size()
        io_amount = 2 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        if not x.dtype.is_complex:
            return n
        # Real-times-complex scaling needs two multiplies, complex-times-complex six.
        real_alpha = self.op_name.split("_", 1)[0] in ("csscal", "zdscal")
        return (2 if real_alpha else 6) * n


if IS_HYGON:
    ScalBenchmark.correctness_reference = "hipBLAS"
    ScalStrideBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.sscal
def test_perf_sscal():
    bench = ScalBenchmark(
        op_name="sscal",
        torch_op=cublas_sscal,
        blas_op=blas_sscal_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dscal
def test_perf_dscal():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = ScalBenchmark(
        op_name="dscal",
        torch_op=cublas_dscal,
        blas_op=blas_dscal_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cscal
def test_perf_cscal():
    bench = ScalBenchmark(
        op_name="cscal",
        torch_op=cublas_cscal,
        blas_op=blas_cscal_wrapper,
        dtypes=[torch.complex64],
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zscal
def test_perf_zscal():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")
    bench = ScalBenchmark(
        op_name="zscal",
        torch_op=cublas_zscal,
        blas_op=blas_zscal_wrapper,
        dtypes=[torch.complex128],
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.csscal
def test_perf_csscal():
    bench = ScalBenchmark(
        op_name="csscal",
        torch_op=cublas_csscal,
        blas_op=blas_csscal_wrapper,
        dtypes=[torch.complex64],
        alpha=0.01,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zdscal
def test_perf_zdscal():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")
    bench = ScalBenchmark(
        op_name="zdscal",
        torch_op=cublas_zdscal,
        blas_op=blas_zdscal_wrapper,
        dtypes=[torch.complex128],
        alpha=0.01,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sscal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_sscal_stride(incx):
    bench = ScalStrideBenchmark(
        op_name=f"sscal_stride_incx{incx}",
        torch_op=cublas_sscal,
        blas_op=blas_sscal_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dscal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_dscal_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = ScalStrideBenchmark(
        op_name=f"dscal_stride_incx{incx}",
        torch_op=cublas_dscal,
        blas_op=blas_dscal_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cscal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_cscal_stride(incx):
    bench = ScalStrideBenchmark(
        op_name=f"cscal_stride_incx{incx}",
        torch_op=cublas_cscal,
        blas_op=blas_cscal_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zscal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_zscal_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")
    bench = ScalStrideBenchmark(
        op_name=f"zscal_stride_incx{incx}",
        torch_op=cublas_zscal,
        blas_op=blas_zscal_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        alpha=0.01 + 0.01j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.csscal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_csscal_stride(incx):
    bench = ScalStrideBenchmark(
        op_name=f"csscal_stride_incx{incx}",
        torch_op=cublas_csscal,
        blas_op=blas_csscal_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        alpha=0.01,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zdscal
@pytest.mark.parametrize("incx", [2, 3, 5])
def test_perf_zdscal_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64/complex128")
    bench = ScalStrideBenchmark(
        op_name=f"zdscal_stride_incx{incx}",
        torch_op=cublas_zdscal,
        blas_op=blas_zdscal_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        alpha=0.01,
    )
    run_correctness_then_benchmark(bench)

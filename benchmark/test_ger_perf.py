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
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_HYGON:
    import atexit
    import ctypes
    import ctypes.util

    class HipComplex(ctypes.Structure):
        _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

    class HipDoubleComplex(ctypes.Structure):
        _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    GER_BENCH_OPS = {
        "sger": (torch.float32, "hipblasSger", ctypes.c_float, 1e-5),
        "dger": (torch.float64, "hipblasDger", ctypes.c_double, 1e-5),
        "cgeru": (
            torch.complex64,
            "hipblasCgeru_v2",
            HipComplex,
            1e-5 + 2e-5j,
        ),
        "cgerc": (
            torch.complex64,
            "hipblasCgerc_v2",
            HipComplex,
            1e-5 + 2e-5j,
        ),
        "zgeru": (
            torch.complex128,
            "hipblasZgeru_v2",
            HipDoubleComplex,
            1e-5 + 2e-5j,
        ),
        "zgerc": (
            torch.complex128,
            "hipblasZgerc_v2",
            HipDoubleComplex,
            1e-5 + 2e-5j,
        ),
    }

    def _check_hipblas_status(status, operation):
        if status != 0:
            raise RuntimeError(f"{operation} failed with hipBLAS status {status}")

    def _load_hipblas():
        global _HIPBLAS_LIBRARY
        if _HIPBLAS_LIBRARY is None:
            library_name = ctypes.util.find_library("hipblas")
            if library_name is None:
                raise RuntimeError("Unable to find the hipBLAS shared library")
            library = ctypes.CDLL(library_name)
            library.hipblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            library.hipblasCreate.restype = ctypes.c_int
            library.hipblasDestroy.argtypes = [ctypes.c_void_p]
            library.hipblasDestroy.restype = ctypes.c_int
            library.hipblasSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            library.hipblasSetStream.restype = ctypes.c_int
            library.hipblasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
            library.hipblasSetPointerMode.restype = ctypes.c_int
            _HIPBLAS_LIBRARY = library
        return _HIPBLAS_LIBRARY

    def _prepare_hipblas(device):
        library = _load_hipblas()
        torch_device = torch.device(device)
        device_index = torch_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        handle = _HIPBLAS_HANDLES.get(device_index)
        if handle is None:
            with torch.cuda.device(device_index):
                handle = ctypes.c_void_p()
                _check_hipblas_status(
                    library.hipblasCreate(ctypes.byref(handle)), "hipblasCreate"
                )
                _check_hipblas_status(
                    library.hipblasSetPointerMode(handle, 0),
                    "hipblasSetPointerMode",
                )
            _HIPBLAS_HANDLES[device_index] = handle
        stream = torch.cuda.current_stream(device).cuda_stream
        _check_hipblas_status(
            library.hipblasSetStream(handle, ctypes.c_void_p(stream)),
            "hipblasSetStream",
        )
        return library, handle

    def _resolve_hipblas_ger(library, symbol):
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function

    def _make_hipblas_scalar(scalar_type, value):
        if scalar_type in (HipComplex, HipDoubleComplex):
            complex_value = complex(value)
            return scalar_type(complex_value.real, complex_value.imag)
        return scalar_type(float(value))

    def _make_hipblas_ger_args(handle, m, n, alpha_ptr, x, incx, y, incy, A, lda):
        return (
            handle,
            m,
            n,
            alpha_ptr,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.c_void_p(y.data_ptr()),
            incy,
            ctypes.c_void_p(A.data_ptr()),
            lda,
        )

    def _destroy_hipblas_handles():
        if _HIPBLAS_LIBRARY is None:
            return
        for handle in tuple(_HIPBLAS_HANDLES.values()):
            try:
                _HIPBLAS_LIBRARY.hipblasDestroy(handle)
            except Exception:
                pass
        _HIPBLAS_HANDLES.clear()

    atexit.register(_destroy_hipblas_handles)
elif IS_MTHREADS:
    from benchmark.mublas_compat import cp, cublas

    GER_BENCH_OPS = {
        "sger": (torch.float32, cublas.sger, np.float32, 1e-5),
        "dger": (torch.float64, cublas.dger, np.float64, 1e-5),
        "cgeru": (torch.complex64, cublas.cgeru, np.complex64, 1e-5 + 2e-5j),
        "cgerc": (torch.complex64, cublas.cgerc, np.complex64, 1e-5 + 2e-5j),
        "zgeru": (torch.complex128, cublas.zgeru, np.complex128, 1e-5 + 2e-5j),
        "zgerc": (torch.complex128, cublas.zgerc, np.complex128, 1e-5 + 2e-5j),
    }
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    GER_BENCH_OPS = {
        "sger": (torch.float32, cublas.sger, np.float32, 1e-5),
        "dger": (torch.float64, cublas.dger, np.float64, 1e-5),
        "cgeru": (torch.complex64, cublas.cgeru, np.complex64, 1e-5 + 2e-5j),
        "cgerc": (torch.complex64, cublas.cgerc, np.complex64, 1e-5 + 2e-5j),
        "zgeru": (torch.complex128, cublas.zgeru, np.complex128, 1e-5 + 2e-5j),
        "zgerc": (torch.complex128, cublas.zgerc, np.complex128, 1e-5 + 2e-5j),
    }


def cublas_ger(
    A,
    x,
    y,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    incy,
    handle,
    alpha_ptr,
    op_name,
    c_func,
    **kwargs,
):
    c_func(
        handle,
        m,
        n,
        alpha_ptr,
        x.data_ptr(),
        incx,
        y.data_ptr(),
        incy,
        A.data_ptr(),
        lda_col,
    )
    return A


def hipblas_ger(
    A,
    x,
    y,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    incy,
    handle,
    alpha_ptr,
    op_name,
    c_func,
    vendor_args,
    **kwargs,
):
    status = c_func(*vendor_args)
    if status != 0:
        raise RuntimeError(f"{op_name} failed with hipBLAS status {status}")
    return A


def flag_blas_ger_wrapper(
    A,
    x,
    y,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    incy,
    handle,
    alpha_ptr,
    op_name,
    gems_func,
    **kwargs,
):
    gems_func(m, n, alpha, x, incx, y, incy, A_row, lda_row)
    return A_row


class GerBenchmark(Benchmark):
    def __init__(self, *args, ger_op_name, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ger_op_name = ger_op_name
        self.alpha = GER_BENCH_OPS[ger_op_name][3]
        self.incx = incx
        self.incy = incy
        if "gbps" not in self.metrics:
            self.metrics.append("gbps")

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            (64, 64),
            (256, 256),
            (1024, 1024),
            (4096, 4096),
            (1024, 4096),
            (4096, 1024),
            (127, 255),
            (1023, 4095),
            (4095, 1023),
        ]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            _, symbol, scalar_type, _ = GER_BENCH_OPS[self.ger_op_name]
            c_func = _resolve_hipblas_ger(library, symbol)
            alpha_value = _make_hipblas_scalar(scalar_type, self.alpha)
            alpha_ptr = ctypes.byref(alpha_value)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            _, c_func, np_dtype, _ = GER_BENCH_OPS[self.ger_op_name]
            alpha_value = np.array(self.alpha, dtype=np_dtype)
            alpha_ptr = alpha_value.ctypes.data
        gems_func = getattr(flag_blas, self.ger_op_name)

        for m, n in self.shapes:
            A_col = torch.randn(n, m, dtype=cur_dtype, device=self.device).t()
            A_row = A_col.contiguous()
            x = torch.randn(m * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            vendor_args = None
            if IS_HYGON:
                vendor_args = _make_hipblas_ger_args(
                    handle,
                    m,
                    n,
                    alpha_ptr,
                    x,
                    self.incx,
                    y,
                    self.incy,
                    A_col,
                    m,
                )

            yield A_col, x, y, {
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_row": A_row,
                "lda_col": m,
                "lda_row": n,
                "incx": self.incx,
                "incy": self.incy,
                "handle": handle,
                "alpha_value": alpha_value,
                "alpha_ptr": alpha_ptr,
                "op_name": self.ger_op_name,
                "c_func": c_func,
                "gems_func": gems_func,
                "vendor_args": vendor_args,
            }

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        m, n = A.shape
        element_size = A.element_size()
        io_amount = 2 * m * n * element_size + m * element_size + n * element_size
        if self.incx != 1:
            io_amount += shape_utils.size_in_bytes(x) - m * element_size
        if self.incy != 1:
            io_amount += shape_utils.size_in_bytes(y) - n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return 1

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        A_col_ref = A.t().contiguous().t()
        A_row_blas = A_col_ref.contiguous()
        ref_x = x.clone()
        ref_y = y.clone()

        ref_kwargs = dict(kwargs)
        ref_kwargs["A_row"] = A_row_blas.clone()
        if IS_HYGON:
            ref_kwargs["vendor_args"] = _make_hipblas_ger_args(
                kwargs["handle"],
                kwargs["m"],
                kwargs["n"],
                kwargs["alpha_ptr"],
                ref_x,
                kwargs["incx"],
                ref_y,
                kwargs["incy"],
                A_col_ref,
                kwargs["lda_col"],
            )

        blas_kwargs = dict(kwargs)
        blas_kwargs["A_row"] = A_row_blas

        return (
            (A_col_ref, ref_x, ref_y),
            ref_kwargs,
            (A.clone(), x.clone(), y.clone()),
            blas_kwargs,
        )


def _run_ger_benchmark(op_name):
    dtype = GER_BENCH_OPS[op_name][0]
    if dtype in (torch.float64, torch.complex128):
        if not flag_blas.runtime.device.support_fp64:
            pytest.skip("Device does not support float64")

    bench = GerBenchmark(
        op_name=op_name,
        torch_op=hipblas_ger if IS_HYGON else cublas_ger,
        gems_op=flag_blas_ger_wrapper,
        dtypes=[dtype],
        ger_op_name=op_name,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sger
def test_perf_sger():
    _run_ger_benchmark("sger")


@pytest.mark.dger
def test_perf_dger():
    _run_ger_benchmark("dger")


@pytest.mark.cgeru
def test_perf_cgeru():
    _run_ger_benchmark("cgeru")


@pytest.mark.cgerc
def test_perf_cgerc():
    _run_ger_benchmark("cgerc")


@pytest.mark.zgeru
def test_perf_zgeru():
    _run_ger_benchmark("zgeru")


@pytest.mark.zgerc
def test_perf_zgerc():
    _run_ger_benchmark("zgerc")

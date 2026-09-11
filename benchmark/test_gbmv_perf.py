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

import numpy as np
import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_HYGON:
    import atexit
elif IS_MTHREADS:
    from benchmark.mublas_compat import cp, cublas
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

GBMV_BANDS = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 7),
    (32, 32),
    (128, 128),
    (256, 256),
]

GBMV_SHAPES = [
    (255, 255),
    (256, 256),
    (1023, 1023),
    (1024, 1024),
    (4095, 4095),
    (4096, 4096),
    (16384, 16384),
    (127, 255),
    (4096, 16384),
    (16384, 4096),
    (10000, 10000),
]


def load_cublas():
    if IS_MTHREADS:
        from benchmark.mublas_compat import load_mublas

        return load_mublas()
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)
    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on the system.")


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


if IS_HYGON:
    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    _HIPBLAS_GBMV_FUNCS = {
        torch.float32: ("hipblasSgbmv", ctypes.c_float, None),
        torch.float64: ("hipblasDgbmv", ctypes.c_double, None),
        torch.complex64: ("hipblasCgbmv_v2", cuComplex, True),
        torch.complex128: ("hipblasZgbmv_v2", cuDoubleComplex, True),
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

    def _destroy_hipblas_handles():
        if _HIPBLAS_LIBRARY is None:
            return
        for handle in tuple(_HIPBLAS_HANDLES.values()):
            try:
                _HIPBLAS_LIBRARY.hipblasDestroy(handle)
            except Exception:
                pass
        _HIPBLAS_HANDLES.clear()

    def _resolve_hipblas_gbmv(library, dtype):
        try:
            symbol, scalar_type, is_complex = _HIPBLAS_GBMV_FUNCS[dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Hygon GBMV benchmark dtype: {dtype}"
            ) from error
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(scalar_type),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(scalar_type),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function, scalar_type, is_complex

    atexit.register(_destroy_hipblas_handles)


_cublas = None if IS_HYGON else load_cublas()

_CUBLAS_GBMV_FUNCS = (
    {}
    if IS_HYGON
    else {
        torch.float32: (_cublas.cublasSgbmv_v2, ctypes.c_float, None),
        torch.float64: (_cublas.cublasDgbmv_v2, ctypes.c_double, None),
        torch.complex64: (_cublas.cublasCgbmv_v2, cuComplex, True),
        torch.complex128: (_cublas.cublasZgbmv_v2, cuDoubleComplex, True),
    }
)


def _make_scalar(ctor, is_complex, value):
    if is_complex:
        return ctor(value.real, value.imag)
    return ctor(value)


if IS_HYGON:

    def cublas_gbmv_baseline(
        AB,
        x,
        y,
        trans,
        m,
        n,
        kl,
        ku,
        alpha,
        lda,
        incx,
        beta,
        incy,
        handle,
        alpha_ptr,
        beta_ptr,
        c_func,
        vendor_args,
        hip_trans,
        **kwargs,
    ):
        status = c_func(*vendor_args)
        _check_hipblas_status(status, "hipBLAS GBMV")
        return y

else:

    def cublas_gbmv_baseline(
        AB,
        x,
        y,
        trans,
        m,
        n,
        kl,
        ku,
        alpha,
        lda,
        incx,
        beta,
        incy,
        handle,
        alpha_ptr,
        beta_ptr,
        c_func,
        vendor_args,
        alpha_c,
        beta_c,
        **kwargs,
    ):
        if m == 0 or n == 0:
            return y

        status = c_func(*vendor_args)
        if status != 0:
            raise RuntimeError(f"cublasXgbmv_v2 failed with status code: {status}")
        return y


def _gems_wrapper(op):
    def _impl(
        AB,
        x,
        y,
        trans,
        m,
        n,
        kl,
        ku,
        alpha,
        lda,
        incx,
        beta,
        incy,
        handle,
        alpha_ptr,
        beta_ptr,
        **kwargs,
    ):
        op(trans, m, n, kl, ku, alpha, AB, lda, x, incx, beta, y, incy)
        return y

    return _impl


gems_sgbmv_wrapper = _gems_wrapper(flag_blas.sgbmv)
gems_dgbmv_wrapper = _gems_wrapper(flag_blas.dgbmv)
gems_cgbmv_wrapper = _gems_wrapper(flag_blas.cgbmv)
gems_zgbmv_wrapper = _gems_wrapper(flag_blas.zgbmv)


def _generate_banded_AB(m, n, kl, ku, lda, dtype, device):
    """Generate equivalent row-major and column-major band storage."""
    # Current TorchMUSA does not implement vectorized IndexPut for complex
    # tensors.  Build the small band-storage reference on the host in that
    # case and transfer it once; this affects only benchmark setup, not the
    # measured BLAS call.
    storage_device = "cpu" if IS_MTHREADS else device
    row_AB = torch.zeros((m, lda), dtype=dtype, device=storage_device)
    column_AB = torch.zeros((n, lda), dtype=dtype, device=storage_device)
    for d in range(-ku, kl + 1):
        j_min = max(0, -d)
        j_max = min(n, m - d)
        if j_min < j_max:
            j_idx = torch.arange(j_min, j_max, device=storage_device)
            i_idx = j_idx + d
            if dtype.is_complex:
                vals = torch.randn(len(j_idx), dtype=dtype, device=storage_device)
            else:
                vals = torch.randn(len(j_idx), dtype=dtype, device=storage_device) * 0.1
            row_AB[i_idx, kl - d] = vals
            column_AB[j_idx, ku + d] = vals
    return row_AB.contiguous().to(device), column_AB.contiguous().to(device)


class GbmvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        trans=CUBLAS_OP_N,
        alpha=1.5,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trans = trans
        self.alpha = alpha
        self.beta = beta
        self.bands = GBMV_BANDS
        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = GBMV_SHAPES
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, ctor, is_complex = _resolve_hipblas_gbmv(library, cur_dtype)
            hip_trans = {
                CUBLAS_OP_N: 111,
                CUBLAS_OP_T: 112,
                CUBLAS_OP_C: 113,
            }[self.trans]
            alpha_c = _make_scalar(ctor, is_complex, self.alpha)
            beta_c = _make_scalar(ctor, is_complex, self.beta)
            alpha_ptr = ctypes.byref(alpha_c)
            beta_ptr = ctypes.byref(beta_c)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

            if cur_dtype == torch.float32:
                np_dtype = np.float32
            elif cur_dtype == torch.float64:
                np_dtype = np.float64
            elif cur_dtype == torch.complex64:
                np_dtype = np.complex64
            elif cur_dtype == torch.complex128:
                np_dtype = np.complex128
            else:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")

            alpha_np = np.array(self.alpha, dtype=np_dtype)
            beta_np = np.array(self.beta, dtype=np_dtype)
            alpha_ptr = alpha_np.ctypes.data
            beta_ptr = beta_np.ctypes.data

            if cur_dtype not in _CUBLAS_GBMV_FUNCS:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")
            c_func, ctor, is_complex = _CUBLAS_GBMV_FUNCS[cur_dtype]
            alpha_c = _make_scalar(ctor, is_complex, self.alpha)
            beta_c = _make_scalar(ctor, is_complex, self.beta)

        seen_configs = set()

        for m, n in self.shapes:
            for kl, ku in self.bands:
                actual_kl = min(kl, max(0, m - 1))
                actual_ku = min(ku, max(0, n - 1))

                config_key = (m, n, actual_kl, actual_ku)
                if config_key in seen_configs:
                    continue
                seen_configs.add(config_key)

                lda = actual_kl + actual_ku + 1
                AB, column_AB = _generate_banded_AB(
                    m, n, actual_kl, actual_ku, lda, cur_dtype, self.device
                )

                x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)
                x = torch.randn(x_len, dtype=cur_dtype, device=self.device)
                y = torch.randn(y_len, dtype=cur_dtype, device=self.device)
                output = y.clone()

                if IS_HYGON:
                    vendor_args = (
                        handle,
                        ctypes.c_int(hip_trans),
                        ctypes.c_int(m),
                        ctypes.c_int(n),
                        ctypes.c_int(actual_kl),
                        ctypes.c_int(actual_ku),
                        alpha_ptr,
                        ctypes.c_void_p(column_AB.data_ptr()),
                        ctypes.c_int(lda),
                        ctypes.c_void_p(x.data_ptr()),
                        ctypes.c_int(1),
                        beta_ptr,
                        ctypes.c_void_p(output.data_ptr()),
                        ctypes.c_int(1),
                    )
                else:
                    vendor_args = (
                        ctypes.c_void_p(handle),
                        ctypes.c_int(self.trans),
                        ctypes.c_int(m),
                        ctypes.c_int(n),
                        ctypes.c_int(actual_kl),
                        ctypes.c_int(actual_ku),
                        ctypes.byref(alpha_c),
                        ctypes.c_void_p(column_AB.data_ptr()),
                        ctypes.c_int(lda),
                        ctypes.c_void_p(x.data_ptr()),
                        ctypes.c_int(1),
                        ctypes.byref(beta_c),
                        ctypes.c_void_p(output.data_ptr()),
                        ctypes.c_int(1),
                    )

                call_kwargs = {
                    "trans": self.trans,
                    "m": m,
                    "n": n,
                    "kl": actual_kl,
                    "ku": actual_ku,
                    "alpha": self.alpha,
                    "lda": lda,
                    "incx": 1,
                    "beta": self.beta,
                    "incy": 1,
                    "handle": handle,
                    "alpha_ptr": alpha_ptr,
                    "beta_ptr": beta_ptr,
                    "c_func": c_func,
                    "vendor_args": vendor_args,
                    "column_AB": column_AB,
                }
                if IS_HYGON:
                    call_kwargs["hip_trans"] = hip_trans
                else:
                    call_kwargs["alpha_c"] = alpha_c
                    call_kwargs["beta_c"] = beta_c
                yield AB, x, output, call_kwargs

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        kl = kwargs.get("kl", 0)
        ku = kwargs.get("ku", 0)
        nnz = 0
        for d in range(-ku, kl + 1):
            j_min = max(0, -d)
            j_max = min(n, m - d)
            nnz += max(0, j_max - j_min)
        AB = args[0]
        if AB.dtype in [torch.complex64, torch.complex128]:
            return 8 * nnz
        return 2 * nnz

    def get_gbps(self, args, latency):
        AB, x, y = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(AB)
            + shape_utils.size_in_bytes(x)
            + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        input_len = kwargs["n"] if kwargs["trans"] == CUBLAS_OP_N else kwargs["m"]
        return max(1, min(input_len, kwargs["kl"] + kwargs["ku"] + 1))

    def clone_correctness_inputs(self, args, kwargs):
        AB, x, y = args
        ref_y = y.clone()
        ref_args = (AB, x, ref_y)
        blas_args = (AB, x, y.clone())
        ref_kwargs = kwargs.copy()
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[12] = ctypes.c_void_p(ref_y.data_ptr())
        ref_kwargs["vendor_args"] = tuple(vendor_args)
        return ref_args, ref_kwargs, blas_args, kwargs


@pytest.mark.sgbmv
def test_perf_sgbmv():
    bench = GbmvBenchmark(
        op_name="sgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_sgbmv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_N,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sgbmv
def test_perf_sgbmv_trans():
    bench = GbmvBenchmark(
        op_name="sgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_sgbmv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_T,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dgbmv
def test_perf_dgbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="dgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_dgbmv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_N,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dgbmv
def test_perf_dgbmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="dgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_dgbmv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_T,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cgbmv
def test_perf_cgbmv():
    bench = GbmvBenchmark(
        op_name="cgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_cgbmv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cgbmv
def test_perf_cgbmv_trans():
    bench = GbmvBenchmark(
        op_name="cgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_cgbmv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cgbmv
def test_perf_cgbmv_conj():
    bench = GbmvBenchmark(
        op_name="cgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_cgbmv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zgbmv
def test_perf_zgbmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="zgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_zgbmv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zgbmv
def test_perf_zgbmv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="zgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_zgbmv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zgbmv
def test_perf_zgbmv_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GbmvBenchmark(
        op_name="zgbmv",
        torch_op=cublas_gbmv_baseline,
        gems_op=gems_zgbmv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)

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

import atexit
import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

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
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = None if IS_HYGON else load_cublas()
_cublas_handle = None


def _get_cublas_handle():
    if IS_MTHREADS:
        from benchmark.mublas_compat import get_mublas_handle
        return get_mublas_handle()
    global _cublas_handle
    if _cublas_handle is None:
        handle = ctypes.c_void_p()
        status = _cublas.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        _cublas_handle = handle.value
    status = _cublas.cublasSetStream_v2(
        ctypes.c_void_p(_cublas_handle),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"cublasSetStream_v2 failed with status code: {status}")
    return _cublas_handle


_HIPBLAS_LIBRARY = None
_HIPBLAS_HANDLES = {}
_HIPBLAS_TBSV_FUNCS = {
    torch.float32: "hipblasStbsv",
    torch.float64: "hipblasDtbsv",
    torch.complex64: "hipblasCtbsv_v2",
    torch.complex128: "hipblasZtbsv_v2",
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
                library.hipblasSetPointerMode(handle, 0), "hipblasSetPointerMode"
            )
        _HIPBLAS_HANDLES[device_index] = handle
    stream = torch.cuda.current_stream(device).cuda_stream
    _check_hipblas_status(
        library.hipblasSetStream(handle, ctypes.c_void_p(stream)),
        "hipblasSetStream",
    )
    return library, handle


def _resolve_hipblas_tbsv(library, dtype):
    function = getattr(library, _HIPBLAS_TBSV_FUNCS[dtype])
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    return function


def _destroy_hipblas_handles():
    if _HIPBLAS_LIBRARY is None:
        return
    for handle in tuple(_HIPBLAS_HANDLES.values()):
        try:
            _HIPBLAS_LIBRARY.hipblasDestroy(handle)
        except Exception:
            pass
    _HIPBLAS_HANDLES.clear()


if IS_HYGON:
    atexit.register(_destroy_hipblas_handles)


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
    c_func,
    vendor_args,
    reference_result,
    **kwargs,
):
    if n == 0:
        return x
    status = c_func(*vendor_args)
    if IS_HYGON:
        _check_hipblas_status(status, "hipBLAS TBSV")
    elif status != 0:
        raise RuntimeError(f"cuBLAS TBSV failed with status code: {status}")
    return reference_result


def gems_stbsv_wrapper(A, x, uplo, trans, diag, n, k, lda, incx, **kwargs):
    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, incx)
    return x


def _gems_wrapper(op):
    def _impl(A, x, uplo, trans, diag, n, k, lda, incx, **kwargs):
        op(uplo, trans, diag, n, k, A, lda, x, incx)
        return x

    return _impl


GEMS_TBSV_WRAPPERS = {
    "stbsv": gems_stbsv_wrapper,
    "dtbsv": _gems_wrapper(flag_blas.dtbsv),
    "ctbsv": _gems_wrapper(flag_blas.ctbsv),
    "ztbsv": _gems_wrapper(flag_blas.ztbsv),
}


def _make_triangular_banded(n, k, lda, uplo, dtype, device):
    if n == 0:
        empty = torch.zeros((n, lda), dtype=dtype, device=device).contiguous()
        return empty, empty.clone()

    A = torch.zeros((n, lda), dtype=dtype, device=device)
    column_A = torch.zeros((n, lda), dtype=dtype, device=device)
    diag_floor = 2.0 * (k + 1) + 1.0
    rows = torch.arange(n, device=device).view(n, 1)
    bands = torch.arange(k + 1, device=device).view(1, k + 1)
    values = torch.randn((n, k + 1), dtype=dtype, device=device) * 0.1
    if uplo == CUBLAS_FILL_MODE_UPPER:
        columns = rows + bands
        diag_col = 0
    else:
        columns = rows + bands - k
        diag_col = k
    valid = (columns >= 0) & (columns < n)
    A[:, : k + 1] = values.masked_fill(~valid, 0.0)
    A[:, diag_col] = diag_floor
    column_bands = (k - bands).expand(n, k + 1)
    column_A[columns.expand(n, k + 1)[valid], column_bands[valid]] = A[:, : k + 1][
        valid
    ]
    return A.contiguous(), column_A.contiguous()


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
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func = _resolve_hipblas_tbsv(library, cur_dtype)
        else:
            handle = _get_cublas_handle()
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
                A, reference_A = _make_triangular_banded(
                    n, k, lda, self.uplo, cur_dtype, self.device
                )
                x = torch.randn(n, dtype=cur_dtype, device=self.device)
                reference_x = x.clone()
                if IS_HYGON:
                    vendor_args = (
                        handle,
                        121 if self.uplo == CUBLAS_FILL_MODE_UPPER else 122,
                        111 + self.trans,
                        131 + self.diag,
                        n,
                        k,
                        ctypes.c_void_p(reference_A.data_ptr()),
                        lda,
                        ctypes.c_void_p(reference_x.data_ptr()),
                        1,
                    )
                else:
                    vendor_args = (
                        ctypes.c_void_p(handle),
                        ctypes.c_int(self.uplo),
                        ctypes.c_int(self.trans),
                        ctypes.c_int(self.diag),
                        ctypes.c_int(n),
                        ctypes.c_int(k),
                        ctypes.c_void_p(reference_A.data_ptr()),
                        ctypes.c_int(lda),
                        ctypes.c_void_p(reference_x.data_ptr()),
                        ctypes.c_int(1),
                    )

                yield A, x, {
                    "uplo": self.uplo,
                    "trans": self.trans,
                    "diag": self.diag,
                    "n": n,
                    "k": k,
                    "lda": lda,
                    "incx": 1,
                    "c_func": c_func,
                    "reference_A": reference_A,
                    "reference_x": reference_x,
                    "reference_result": reference_x,
                    "vendor_args": vendor_args,
                }

    def clone_correctness_inputs(self, args, kwargs):
        A, x = args
        reference_x = x.clone()
        ref_kwargs = kwargs.copy()
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[8] = ctypes.c_void_p(reference_x.data_ptr())
        ref_kwargs.update(
            reference_x=reference_x,
            reference_result=reference_x,
            vendor_args=tuple(vendor_args),
        )
        return (A, x.clone()), ref_kwargs, (A, x.clone()), kwargs

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
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

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["k"] + 1


# --------------------------------------------------------------------------
# Top-level perf entry points
# --------------------------------------------------------------------------
def _run_tbsv_variant(op_name, dtype, uplo, trans):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    bench = StbsvBenchmark(
        op_name=op_name,
        torch_op=cublas_stbsv_baseline,
        gems_op=GEMS_TBSV_WRAPPERS[op_name],
        dtypes=[dtype],
        uplo=uplo,
        trans=trans,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


TBSV_PERF_CASES = [
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztbsv,
    ),
]


@pytest.mark.parametrize("op_name,dtype,uplo,trans", TBSV_PERF_CASES)
def test_perf_tbsv(op_name, dtype, uplo, trans):
    _run_tbsv_variant(op_name, dtype, uplo, trans)

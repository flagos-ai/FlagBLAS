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

import pytest
import torch

import flag_blas

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_HYGON:
    import atexit
elif IS_MTHREADS:
    from benchmark.mublas_compat import cp, cublas
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

from benchmark.performance_utils import (  # noqa: E402
    Benchmark,
    run_correctness_then_benchmark,
)
from flag_blas.ops import (  # noqa: E402
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.utils import shape_utils  # noqa: E402

TRSV_SIZES = [64, 256, 512, 1024, 2048, 4096, 8192]


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

_CUBLAS_TRSV_FUNCS = (
    {}
    if IS_HYGON
    else {
        torch.float32: _cublas.cublasStrsv_v2,
        torch.float64: _cublas.cublasDtrsv_v2,
        torch.complex64: _cublas.cublasCtrsv_v2,
        torch.complex128: _cublas.cublasZtrsv_v2,
    }
)


if IS_HYGON:
    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    _HIPBLAS_TRSV_FUNCS = {
        torch.float32: "hipblasStrsv",
        torch.float64: "hipblasDtrsv",
        torch.complex64: "hipblasCtrsv_v2",
        torch.complex128: "hipblasZtrsv_v2",
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

    def _resolve_hipblas_trsv(library, dtype):
        try:
            symbol = _HIPBLAS_TRSV_FUNCS[dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Hygon TRSV benchmark dtype: {dtype}"
            ) from error
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
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

    atexit.register(_destroy_hipblas_handles)


if IS_HYGON:

    def cublas_trsv_baseline(
        A,
        x,
        uplo,
        trans,
        diag,
        n,
        lda,
        incx,
        c_func,
        vendor_args,
        reference_result,
        **kwargs,
    ):
        status = c_func(*vendor_args)
        _check_hipblas_status(status, "hipBLAS TRSV")
        return reference_result

else:

    def cublas_trsv_baseline(
        A,
        x,
        uplo,
        trans,
        diag,
        n,
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
        if status != 0:
            raise RuntimeError(f"cublasXtrsv_v2 failed with status code: {status}")
        return reference_result


def _gems_wrapper(op):
    def _impl(A, x, uplo, trans, diag, n, lda, incx, handle=None, **kwargs):
        op(uplo, trans, diag, n, A, lda, x, incx)
        return x

    return _impl


gems_strsv_wrapper = _gems_wrapper(flag_blas.strsv)
gems_dtrsv_wrapper = _gems_wrapper(flag_blas.dtrsv)
gems_ctrsv_wrapper = _gems_wrapper(flag_blas.ctrsv)
gems_ztrsv_wrapper = _gems_wrapper(flag_blas.ztrsv)


def _run_trsv_benchmark(
    op_name,
    wrapper,
    dtype,
    uplo,
    trans,
    diag,
    require_fp64=False,
):
    if require_fp64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrsvBenchmark(
        op_name=op_name,
        torch_op=cublas_trsv_baseline,
        gems_op=wrapper,
        dtypes=[dtype],
        uplo=uplo,
        trans=trans,
        diag=diag,
    )
    run_correctness_then_benchmark(bench)


def _generate_triangular_A(n, lda, uplo, diag, dtype, device):
    vals = torch.randn(n, n, dtype=dtype, device=device) * 0.02
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    row_idx = torch.arange(n, device=device).view(1, n)
    col_idx = torch.arange(n, device=device).view(n, 1)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = row_idx <= col_idx
    else:
        valid = row_idx >= col_idx
    A[:, :n] = vals.masked_fill(~valid, 0.0)
    if diag == CUBLAS_DIAG_NON_UNIT:
        diag_vals = torch.diagonal(vals).clone()
        if dtype.is_complex:
            diag_vals = diag_vals + (2.0 + 0.25j)
        else:
            diag_vals = diag_vals + 2.0
        idx = torch.arange(n, device=device)
        A[idx, idx] = diag_vals
    return A.contiguous()


class TrsvBenchmark(Benchmark):
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
        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in TRSV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        reference_uplo = (
            CUBLAS_FILL_MODE_LOWER
            if self.uplo == CUBLAS_FILL_MODE_UPPER
            else CUBLAS_FILL_MODE_UPPER
        )
        reference_trans = CUBLAS_OP_T if self.trans == CUBLAS_OP_N else CUBLAS_OP_N
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func = _resolve_hipblas_trsv(library, cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            if cur_dtype not in _CUBLAS_TRSV_FUNCS:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")
            c_func = _CUBLAS_TRSV_FUNCS[cur_dtype]
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = n
            A = _generate_triangular_A(
                n, lda, self.uplo, self.diag, cur_dtype, self.device
            )
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            reference_x = x.clone()
            if self.trans == CUBLAS_OP_C:
                reference_x.copy_(reference_x.conj())
                reference_result = reference_x.conj()
            else:
                reference_result = reference_x
            if IS_HYGON:
                vendor_args = (
                    handle,
                    121 if reference_uplo == CUBLAS_FILL_MODE_UPPER else 122,
                    111 + reference_trans,
                    131 + self.diag,
                    n,
                    ctypes.c_void_p(A.data_ptr()),
                    lda,
                    ctypes.c_void_p(reference_x.data_ptr()),
                    1,
                )
            else:
                vendor_args = (
                    ctypes.c_void_p(handle),
                    ctypes.c_int(reference_uplo),
                    ctypes.c_int(reference_trans),
                    ctypes.c_int(self.diag),
                    ctypes.c_int(n),
                    ctypes.c_void_p(A.data_ptr()),
                    ctypes.c_int(lda),
                    ctypes.c_void_p(reference_x.data_ptr()),
                    ctypes.c_int(1),
                )
            call_kwargs = {
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "n": n,
                "lda": lda,
                "incx": 1,
                "c_func": c_func,
                "vendor_args": vendor_args,
                "reference_x": reference_x,
                "reference_result": reference_result,
            }
            yield A, x, call_kwargs

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs["n"])

    def clone_correctness_inputs(self, args, kwargs):
        A, x = args
        reference_x = x.clone()
        if kwargs["trans"] == CUBLAS_OP_C:
            reference_x.copy_(reference_x.conj())
            reference_result = reference_x.conj()
        else:
            reference_result = reference_x
        ref_args = (A, x.clone())
        blas_args = (A, x.clone())
        ref_kwargs = kwargs.copy()
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[7] = ctypes.c_void_p(reference_x.data_ptr())
        ref_kwargs.update(
            vendor_args=tuple(vendor_args),
            reference_x=reference_x,
            reference_result=reference_result,
        )
        return ref_args, ref_kwargs, blas_args, kwargs

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


@pytest.mark.strsv
def test_perf_strsv():
    _run_trsv_benchmark(
        op_name="strsv",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strsv
def test_perf_strsv_upper():
    _run_trsv_benchmark(
        op_name="strsv",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strsv
def test_perf_strsv_trans():
    _run_trsv_benchmark(
        op_name="strsv",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strsv
def test_perf_strsv_unit():
    _run_trsv_benchmark(
        op_name="strsv",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv():
    _run_trsv_benchmark(
        op_name="dtrsv",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv_upper():
    _run_trsv_benchmark(
        op_name="dtrsv",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv_trans():
    _run_trsv_benchmark(
        op_name="dtrsv",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv_unit():
    _run_trsv_benchmark(
        op_name="dtrsv",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
        require_fp64=True,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv():
    _run_trsv_benchmark(
        op_name="ctrsv",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_upper():
    _run_trsv_benchmark(
        op_name="ctrsv",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_trans():
    _run_trsv_benchmark(
        op_name="ctrsv",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_conj():
    _run_trsv_benchmark(
        op_name="ctrsv",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_unit():
    _run_trsv_benchmark(
        op_name="ctrsv",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv():
    _run_trsv_benchmark(
        op_name="ztrsv",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_upper():
    _run_trsv_benchmark(
        op_name="ztrsv",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_trans():
    _run_trsv_benchmark(
        op_name="ztrsv",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_conj():
    _run_trsv_benchmark(
        op_name="ztrsv",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_unit():
    _run_trsv_benchmark(
        op_name="ztrsv",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
        require_fp64=True,
    )

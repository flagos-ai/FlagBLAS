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

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_HYGON:
    import atexit
elif IS_MTHREADS:
    from benchmark.mublas_compat import cp, cublas
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

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


if IS_HYGON:
    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    _HIPBLAS_TPMV_FUNCS = {
        torch.float32: "hipblasStpmv",
        torch.float64: "hipblasDtpmv",
        torch.complex64: "hipblasCtpmv_v2",
        torch.complex128: "hipblasZtpmv_v2",
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

    def _resolve_hipblas_tpmv(library, dtype):
        try:
            symbol = _HIPBLAS_TPMV_FUNCS[dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Hygon TPMV benchmark dtype: {dtype}"
            ) from error
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function

    atexit.register(_destroy_hipblas_handles)


_cublas = None if IS_HYGON else load_cublas()

_CUBLAS_TPMV_FUNCS = (
    {}
    if IS_HYGON
    else {
        torch.float32: _cublas.cublasStpmv_v2,
        torch.float64: _cublas.cublasDtpmv_v2,
        torch.complex64: _cublas.cublasCtpmv_v2,
        torch.complex128: _cublas.cublasZtpmv_v2,
    }
)


if IS_HYGON:

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
        reference_AP,
        reference_x,
        reference_result,
        hip_uplo,
        hip_trans,
        hip_diag,
        **kwargs,
    ):
        status = c_func(*kwargs["vendor_args"])
        _check_hipblas_status(status, "hipBLAS TPMV")
        return reference_result

else:

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
        reference_AP,
        reference_x,
        reference_result,
        reference_uplo,
        reference_trans,
        **kwargs,
    ):
        status = c_func(*kwargs["vendor_args"])
        if status != 0:
            raise RuntimeError(f"cublasXtpmv_v2 failed with status code: {status}")
        return reference_result


def _gems_wrapper(op):
    def _impl(AP, x, uplo, trans, diag, n, incx, handle=None, **kwargs):
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
        self.correctness_reference = "hipBLAS" if IS_HYGON else ("muBLAS" if IS_MTHREADS else "cuBLAS")

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in TPMV_SIZES]
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
            c_func = _resolve_hipblas_tpmv(library, cur_dtype)
            hip_uplo = 121 if reference_uplo == CUBLAS_FILL_MODE_UPPER else 122
            hip_trans = {CUBLAS_OP_N: 111, CUBLAS_OP_T: 112}[reference_trans]
            hip_diag = 132 if self.diag == CUBLAS_DIAG_UNIT else 131
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            if cur_dtype not in _CUBLAS_TPMV_FUNCS:
                raise ValueError(f"Unsupported dtype: {cur_dtype}")
            c_func = _CUBLAS_TPMV_FUNCS[cur_dtype]

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP = _generate_packed_triangular(n, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            reference_x = x.clone()
            if self.trans == CUBLAS_OP_C:
                reference_x.copy_(reference_x.conj())
                reference_result = reference_x.conj()
            else:
                reference_result = reference_x
            vendor_args = (
                (
                    handle,
                    hip_uplo,
                    hip_trans,
                    hip_diag,
                    n,
                    ctypes.c_void_p(AP.data_ptr()),
                    ctypes.c_void_p(reference_x.data_ptr()),
                    1,
                )
                if IS_HYGON
                else (
                    ctypes.c_void_p(handle),
                    ctypes.c_int(reference_uplo),
                    ctypes.c_int(reference_trans),
                    ctypes.c_int(self.diag),
                    ctypes.c_int(n),
                    ctypes.c_void_p(AP.data_ptr()),
                    ctypes.c_void_p(reference_x.data_ptr()),
                    ctypes.c_int(1),
                )
            )
            kwargs = {
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "n": n,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
                "reference_AP": AP,
                "reference_x": reference_x,
                "reference_result": reference_result,
                "vendor_args": vendor_args,
                "reference_uplo": reference_uplo,
                "reference_trans": reference_trans,
            }
            if IS_HYGON:
                kwargs.update(
                    hip_uplo=hip_uplo,
                    hip_trans=hip_trans,
                    hip_diag=hip_diag,
                )
            yield AP, x.clone(), kwargs

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
        reference_x = x.clone()
        if kwargs["trans"] == CUBLAS_OP_C:
            reference_x.copy_(reference_x.conj())
            reference_result = reference_x.conj()
        else:
            reference_result = reference_x
        ref_kwargs = kwargs.copy()
        ref_kwargs.update(
            reference_x=reference_x,
            reference_result=reference_result,
        )
        vendor_args = list(kwargs["vendor_args"])
        vendor_args[6] = ctypes.c_void_p(reference_x.data_ptr())
        ref_kwargs["vendor_args"] = tuple(vendor_args)
        ref_args = (AP, x.clone())
        blas_args = (AP, x.clone())
        return ref_args, ref_kwargs, blas_args, kwargs


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
        op_name="stpmv",
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
        op_name="stpmv",
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
        op_name="stpmv",
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
        op_name="stpmv",
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
        op_name="dtpmv",
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
        op_name="dtpmv",
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
        op_name="dtpmv",
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
        op_name="dtpmv",
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
        op_name="ctpmv",
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
        op_name="ctpmv",
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
        op_name="ctpmv",
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
        op_name="ctpmv",
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
        op_name="ctpmv",
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
        op_name="ztpmv",
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
        op_name="ztpmv",
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
        op_name="ztpmv",
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
        op_name="ztpmv",
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
        op_name="ztpmv",
        torch_op=cublas_tpmv_baseline,
        gems_op=gems_ztpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )
    run_correctness_then_benchmark(bench)

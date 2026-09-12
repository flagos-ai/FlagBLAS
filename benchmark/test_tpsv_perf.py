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

TPSV_SIZES = [
    64,
    96,
    127,
    128,
    129,
    192,
    255,
    256,
    257,
    384,
    511,
    512,
    513,
    768,
    1023,
    1024,
    1025,
    1536,
    2048,
    3072,
    4096,
]


def _load_cublas():
    if IS_MTHREADS:
        from benchmark.mublas_compat import load_mublas

        return load_mublas()
    names = ["libcublas.so.13"]
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so")


_cublas = None if IS_HYGON else _load_cublas()
_cublas_handle = None
_CUBLAS_TPSV_FUNCS = (
    {}
    if IS_HYGON
    else {
        torch.float32: _cublas.cublasStpsv_v2,
        torch.float64: _cublas.cublasDtpsv_v2,
        torch.complex64: _cublas.cublasCtpsv_v2,
        torch.complex128: _cublas.cublasZtpsv_v2,
    }
)


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
_HIPBLAS_TPSV_FUNCS = {
    torch.float32: "hipblasStpsv",
    torch.float64: "hipblasDtpsv",
    torch.complex64: "hipblasCtpsv_v2",
    torch.complex128: "hipblasZtpsv_v2",
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


def _resolve_hipblas_tpsv(library, dtype):
    function = getattr(library, _HIPBLAS_TPSV_FUNCS[dtype])
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


def cublas_tpsv_baseline(
    AP,
    x,
    uplo,
    trans,
    diag,
    n,
    incx,
    handle,
    c_func,
    reference_x,
    reference_result,
    **kwargs,
):
    if n == 0:
        return reference_result
    status = c_func(*kwargs["vendor_args"])
    if IS_HYGON:
        _check_hipblas_status(status, "hipBLAS TPSV")
    elif status != 0:
        raise RuntimeError(f"cuBLAS TPSV failed with status code: {status}")
    return reference_result


def _gems_wrapper(op):
    def _impl(AP, x, uplo, trans, diag, n, incx, **kwargs):
        op(uplo, trans, diag, n, AP, x, incx)
        return x

    return _impl


GEMS_TPSV_WRAPPERS = {
    "stpsv": _gems_wrapper(flag_blas.stpsv),
    "dtpsv": _gems_wrapper(flag_blas.dtpsv),
    "ctpsv": _gems_wrapper(flag_blas.ctpsv),
    "ztpsv": _gems_wrapper(flag_blas.ztpsv),
}


def _row_major_diag_offsets(n, uplo, device):
    rows = torch.arange(n, dtype=torch.int64, device=device)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        return rows * n - rows * (rows - 1) // 2
    return rows * (rows + 3) // 2


def _make_case(n, dtype, uplo, diag, device):
    AP = torch.randn(n * (n + 1) // 2, dtype=dtype, device=device) * 0.02
    if diag == CUBLAS_DIAG_NON_UNIT:
        AP[_row_major_diag_offsets(n, uplo, device)] = (
            (2.0 + 0.25j) if dtype.is_complex else 2.0
        )
    x = torch.randn(n, dtype=dtype, device=device)
    return AP.contiguous(), x.contiguous()


class TpsvBenchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in TPSV_SIZES]
    DEFAULT_SHAPE_DESC = "N"

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
        self.shapes = [(n,) for n in TPSV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func = _resolve_hipblas_tpsv(library, cur_dtype)
        else:
            handle = _get_cublas_handle()
            c_func = _CUBLAS_TPSV_FUNCS[cur_dtype]
        for shape in self.shapes:
            n = shape[0]
            AP, x = _make_case(n, cur_dtype, self.uplo, self.diag, self.device)
            reference_uplo = (
                CUBLAS_FILL_MODE_LOWER
                if self.uplo == CUBLAS_FILL_MODE_UPPER
                else CUBLAS_FILL_MODE_UPPER
            )
            reference_trans = CUBLAS_OP_T if self.trans == CUBLAS_OP_N else CUBLAS_OP_N
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
                    ctypes.c_void_p(AP.data_ptr()),
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
                    ctypes.c_void_p(AP.data_ptr()),
                    ctypes.c_void_p(reference_x.data_ptr()),
                    ctypes.c_int(1),
                )
            yield AP, x.clone(), {
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "n": n,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
                "reference_x": reference_x,
                "reference_result": reference_result,
                "vendor_args": vendor_args,
            }

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs["n"])

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
        return (AP, x.clone()), ref_kwargs, (AP, x.clone()), kwargs

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return n * (n + 1)

    def get_gbps(self, args, latency):
        AP, x = args[0], args[1]
        io_amount = shape_utils.size_in_bytes(AP) + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


def _run_tpsv(op_name, dtype, uplo, trans, diag=CUBLAS_DIAG_NON_UNIT):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    bench = TpsvBenchmark(
        op_name=op_name,
        torch_op=cublas_tpsv_baseline,
        gems_op=GEMS_TPSV_WRAPPERS[op_name],
        dtypes=[dtype],
        uplo=uplo,
        trans=trans,
        diag=diag,
    )
    run_correctness_then_benchmark(bench)


TPSV_PERF_CASES = [
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztpsv,
    ),
]


@pytest.mark.parametrize("op_name,dtype,uplo,trans", TPSV_PERF_CASES)
def test_perf_tpsv(op_name, dtype, uplo, trans):
    _run_tpsv(op_name, dtype, uplo, trans)

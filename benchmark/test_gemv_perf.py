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
from flag_blas.ops import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"
IS_ASCEND = flag_blas.vendor_name == "ascend"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

pytestmark = pytest.mark.skipif(
    IS_ASCEND,
    reason="GEMV vendor-reference benchmarks are unavailable on Ascend",
)
if IS_HYGON:
    import atexit
    import ctypes
    import ctypes.util
elif IS_MTHREADS:
    from benchmark.mublas_compat import cp, cublas
elif not IS_ASCEND:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

_HUGE_NONCONTIG_COPY_BYTES = 2 * 1024 * 1024 * 1024
_CHUNKED_COPY_BYTES = 256 * 1024 * 1024


if IS_HYGON:

    class HipComplex(ctypes.Structure):
        _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

    class HipDoubleComplex(ctypes.Structure):
        _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}
    _HIPBLAS_GEMV_FUNCS = {
        torch.float32: ("hipblasSgemv", ctypes.c_float),
        torch.float64: ("hipblasDgemv", ctypes.c_double),
        torch.complex64: ("hipblasCgemv_v2", HipComplex),
        torch.complex128: ("hipblasZgemv_v2", HipDoubleComplex),
    }
    _HIPBLAS_GEMV_OPERATIONS = {
        CUBLAS_OP_N: 111,
        CUBLAS_OP_T: 112,
        CUBLAS_OP_C: 113,
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

    def _resolve_hipblas_gemv(library, dtype):
        try:
            symbol, scalar_type = _HIPBLAS_GEMV_FUNCS[dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Hygon GEMV benchmark dtype: {dtype}"
            ) from error
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function, scalar_type

    def _resolve_hipblas_gemm_ex(library):
        function = library.hipblasGemmEx_v2
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function

    def _make_hipblas_scalar(scalar_type, value):
        if scalar_type in (HipComplex, HipDoubleComplex):
            complex_value = complex(value)
            return scalar_type(complex_value.real, complex_value.imag)
        return scalar_type(float(value))

    atexit.register(_destroy_hipblas_handles)


def _needs_iluvatar_sgemv_chunked_noncontig_copy(tensor):
    # Iluvatar currently corrupts very large non-contiguous float32 copies
    # when materializing them in one contiguous conversion. This appears in
    # sgemv perf inputs and the FP8 GEMV float32 cuBLAS reference matrix.
    # Keep the workaround narrowly scoped so normal GEMV paths still use the
    # faster single-copy path.
    return (
        flag_blas.vendor_name == "iluvatar"
        and tensor.dtype == torch.float32
        and tensor.ndim == 2
        and not tensor.is_contiguous()
        and tensor.numel() * tensor.element_size() >= _HUGE_NONCONTIG_COPY_BYTES
    )


def _chunked_2d_copy(tensor, *, device, dtype):
    out = torch.empty(tensor.shape, dtype=dtype, device=device)
    row_bytes = tensor.shape[1] * torch.empty((), dtype=dtype).element_size()
    rows_per_chunk = max(1, _CHUNKED_COPY_BYTES // max(1, row_bytes))
    for row_start in range(0, tensor.shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, tensor.shape[0])
        out[row_start:row_end].copy_(
            tensor[row_start:row_end].to(device=device, dtype=dtype)
        )
    return out


def _sgemv_contiguous_matrix(tensor):
    if _needs_iluvatar_sgemv_chunked_noncontig_copy(tensor):
        return _chunked_2d_copy(tensor, device=tensor.device, dtype=tensor.dtype)
    return tensor.contiguous()


def cublas_sgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.sgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def hipblas_sgemv_baseline(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    c_func,
    hip_trans,
):
    if m == 0 or n == 0:
        return y
    status = c_func(
        handle,
        hip_trans,
        m,
        n,
        ctypes.byref(alpha_ptr),
        ctypes.c_void_p(A.data_ptr()),
        lda_col,
        ctypes.c_void_p(x.data_ptr()),
        incx,
        ctypes.byref(beta_ptr),
        ctypes.c_void_p(y.data_ptr()),
        incy,
    )
    _check_hipblas_status(status, "hipBLAS GEMV")
    return y


hipblas_dgemv_baseline = hipblas_sgemv_baseline
hipblas_cgemv_baseline = hipblas_sgemv_baseline
hipblas_zgemv_baseline = hipblas_sgemv_baseline


def hipblas_low_precision_gemv_baseline(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle=None,
    alpha_ptr=None,
    beta_ptr=None,
    cuda_type=None,
    c_func=None,
    hip_trans=None,
    gemm_m=None,
    gemm_k=None,
):
    if m == 0 or n == 0:
        return y
    status = c_func(
        handle,
        hip_trans,
        111,
        gemm_m,
        1,
        gemm_k,
        ctypes.byref(alpha_ptr),
        ctypes.c_void_p(A_row.data_ptr()),
        cuda_type,
        lda_row,
        ctypes.c_void_p(x.data_ptr()),
        cuda_type,
        gemm_k,
        ctypes.byref(beta_ptr),
        ctypes.c_void_p(y.data_ptr()),
        cuda_type,
        gemm_m,
        2,
        160,
    )
    _check_hipblas_status(status, "hipblasGemmEx_v2")
    return y


def cublas_dgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.dgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def cublas_cgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.cgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def cublas_zgemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.zgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A.data_ptr(),
        lda_col,
        x.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def cublas_half_gemv(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    cuda_type,
):
    CUDA_R_32F = 0
    if trans == CUBLAS_OP_N:
        transA = cublas.CUBLAS_OP_T
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = m, 1, n
        lda_c, ldb_c, ldc_c = lda_row, n, m
    else:
        transA = cublas.CUBLAS_OP_N
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = n, 1, m
        lda_c, ldb_c, ldc_c = lda_row, m, n

    cublas.gemmEx(
        handle,
        transA,
        transB,
        m_c,
        n_c,
        k_c,
        alpha_ptr,
        A_row.data_ptr(),
        cuda_type,
        lda_c,
        x.data_ptr(),
        cuda_type,
        ldb_c,
        beta_ptr,
        y.data_ptr(),
        cuda_type,
        ldc_c,
        CUDA_R_32F,
        0,
    )
    return y


def gems_sgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    **kwargs,
):
    flag_blas.sgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_dgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    **kwargs,
):
    flag_blas.dgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_cgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    **kwargs,
):
    flag_blas.cgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_zgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    **kwargs,
):
    flag_blas.zgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_hgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle=None,
    alpha_ptr=None,
    beta_ptr=None,
    cuda_type=None,
    **kwargs,
):
    flag_blas.hgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


def gems_bfgemv_wrapper(
    A,
    x,
    y,
    trans,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    beta,
    incy,
    handle=None,
    alpha_ptr=None,
    beta_ptr=None,
    cuda_type=None,
    **kwargs,
):
    flag_blas.bfgemv(trans, m, n, alpha, A_row, lda_row, x, incx, beta, y, incy)
    return y


class GemvBenchmark(Benchmark):
    def __init__(self, *args, trans=CUBLAS_OP_N, alpha=1.5, beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.trans = trans
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        shapes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (3584, 3584),
            (4096, 4096),
            (7168, 7168),
            (8192, 8192),
            (16384, 16384),
            (18432, 18432),
            (1024, 4096),
            (3584, 18944),
            (4096, 14336),
            (6144, 16384),
            (7168, 18432),
            (8192, 28672),
            (16384, 53248),
            (4096, 1024),
            (18944, 3584),
            (14336, 4096),
            (16384, 6144),
            (18432, 7168),
            (28672, 8192),
            (53248, 16384),
            (63, 63),
            (127, 127),
            (255, 255),
            (511, 511),
            (1023, 1023),
            (3583, 3583),
            (4095, 4095),
            (7167, 7167),
            (8191, 8191),
            (1023, 4095),
            (4095, 14335),
            (4095, 1023),
            (14335, 4095),
            # Extreme shapes
            (1, 65536),
            (2, 65536),
            (3, 131071),
            (4, 131072),
            (64, 65536),
            (65536, 1),
            (65536, 2),
            (131071, 3),
            (131072, 4),
            (65536, 64),
        ]
        self.shapes = shapes
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            library, handle = _prepare_hipblas(self.device)
            c_func, scalar_type = _resolve_hipblas_gemv(library, cur_dtype)
            hip_trans = _HIPBLAS_GEMV_OPERATIONS[self.trans]
            alpha_ptr = _make_hipblas_scalar(scalar_type, self.alpha)
            beta_ptr = _make_hipblas_scalar(scalar_type, self.beta)
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
            alpha_np = np.array(self.alpha, dtype=np_dtype)
            beta_np = np.array(self.beta, dtype=np_dtype)
            alpha_ptr = alpha_np.ctypes.data
            beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n = shape

            A_col = torch.randn(n, m, dtype=cur_dtype, device=self.device).t()
            A_row = _sgemv_contiguous_matrix(A_col)

            x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)
            x = torch.randn(x_len, dtype=cur_dtype, device=self.device)
            y = torch.randn(y_len, dtype=cur_dtype, device=self.device)

            kwargs = {
                "trans": self.trans,
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_row": A_row,
                "lda_col": m,
                "lda_row": n,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }
            if IS_HYGON:
                kwargs.update(c_func=c_func, hip_trans=hip_trans)
            yield A_col, x, y.clone(), kwargs

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        A = args[0]
        if A.dtype in [torch.complex64, torch.complex128]:
            return 8 * m * n
        return 2 * m * n

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(A)
            + shape_utils.size_in_bytes(x)
            + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"] if kwargs["trans"] == CUBLAS_OP_N else kwargs["m"]

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        ref_args = (A, x, y.clone())
        blas_args = (A, x, y.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.sgemv
def test_perf_sgemv():
    bench = GemvBenchmark(
        op_name="sgemv",
        torch_op=hipblas_sgemv_baseline if IS_HYGON else cublas_sgemv,
        gems_op=gems_sgemv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_N,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sgemv
def test_perf_sgemv_trans():
    bench = GemvBenchmark(
        op_name="sgemv_trans",
        torch_op=hipblas_sgemv_baseline if IS_HYGON else cublas_sgemv,
        gems_op=gems_sgemv_wrapper,
        dtypes=[torch.float32],
        trans=CUBLAS_OP_T,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dgemv
def test_perf_dgemv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="dgemv",
        torch_op=hipblas_dgemv_baseline if IS_HYGON else cublas_dgemv,
        gems_op=gems_dgemv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_N,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dgemv
def test_perf_dgemv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="dgemv_trans",
        torch_op=hipblas_dgemv_baseline if IS_HYGON else cublas_dgemv,
        gems_op=gems_dgemv_wrapper,
        dtypes=[torch.float64],
        trans=CUBLAS_OP_T,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cgemv
def test_perf_cgemv():
    if IS_ASCEND:
        pytest.skip("Ascend cgemv vendor-reference benchmark is not available")
    bench = GemvBenchmark(
        op_name="cgemv",
        torch_op=hipblas_cgemv_baseline if IS_HYGON else cublas_cgemv,
        gems_op=gems_cgemv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cgemv
def test_perf_cgemv_trans():
    if IS_ASCEND:
        pytest.skip("Ascend cgemv vendor-reference benchmark is not available")
    bench = GemvBenchmark(
        op_name="cgemv_trans",
        torch_op=hipblas_cgemv_baseline if IS_HYGON else cublas_cgemv,
        gems_op=gems_cgemv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.cgemv
def test_perf_cgemv_conj():
    if IS_ASCEND:
        pytest.skip("Ascend cgemv vendor-reference benchmark is not available")
    bench = GemvBenchmark(
        op_name="cgemv_conj",
        torch_op=hipblas_cgemv_baseline if IS_HYGON else cublas_cgemv,
        gems_op=gems_cgemv_wrapper,
        dtypes=[torch.complex64],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zgemv
def test_perf_zgemv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="zgemv",
        torch_op=hipblas_zgemv_baseline if IS_HYGON else cublas_zgemv,
        gems_op=gems_zgemv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_N,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zgemv
def test_perf_zgemv_trans():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="zgemv_trans",
        torch_op=hipblas_zgemv_baseline if IS_HYGON else cublas_zgemv,
        gems_op=gems_zgemv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_T,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zgemv
def test_perf_zgemv_conj():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GemvBenchmark(
        op_name="zgemv_conj",
        torch_op=hipblas_zgemv_baseline if IS_HYGON else cublas_zgemv,
        gems_op=gems_zgemv_wrapper,
        dtypes=[torch.complex128],
        trans=CUBLAS_OP_C,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


class HalfGemvBenchmark(GemvBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            if cur_dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    f"Unsupported Hygon low-precision GEMV dtype: {cur_dtype}"
                )
            library, handle = _prepare_hipblas(self.device)
            c_func = _resolve_hipblas_gemm_ex(library)
            alpha_ptr = ctypes.c_float(float(self.alpha))
            beta_ptr = ctypes.c_float(float(self.beta))
            cuda_type = 2 if cur_dtype == torch.float16 else 14
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            alpha_np = np.array(self.alpha, dtype=np.float32)
            beta_np = np.array(self.beta, dtype=np.float32)
            alpha_ptr = alpha_np.ctypes.data
            beta_ptr = beta_np.ctypes.data
            cuda_type = 2 if cur_dtype == torch.float16 else 14

        for shape in self.shapes:
            m, n = shape

            A_row = torch.randn(m, n, dtype=cur_dtype, device=self.device).contiguous()

            x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)
            x = torch.randn(x_len, dtype=cur_dtype, device=self.device)
            y = torch.randn(y_len, dtype=cur_dtype, device=self.device)

            kwargs = {
                "trans": self.trans,
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_row": A_row,
                "lda_col": m,
                "lda_row": n,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
                "cuda_type": cuda_type,
            }
            if IS_HYGON:
                if self.trans == CUBLAS_OP_N:
                    hip_trans = 112
                    gemm_m, gemm_k = m, n
                else:
                    hip_trans = 111
                    gemm_m, gemm_k = n, m
                kwargs.update(
                    c_func=c_func,
                    hip_trans=hip_trans,
                    gemm_m=gemm_m,
                    gemm_k=gemm_k,
                )
            yield A_row, x, y.clone(), kwargs


@pytest.mark.hgemv
def test_perf_hgemv():
    bench = HalfGemvBenchmark(
        op_name="hgemv",
        torch_op=(
            hipblas_low_precision_gemv_baseline if IS_HYGON else cublas_half_gemv
        ),
        gems_op=gems_hgemv_wrapper,
        dtypes=[torch.float16],
        trans=CUBLAS_OP_N,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.hgemv
def test_perf_hgemv_trans():
    bench = HalfGemvBenchmark(
        op_name="hgemv",
        torch_op=(
            hipblas_low_precision_gemv_baseline if IS_HYGON else cublas_half_gemv
        ),
        gems_op=gems_hgemv_wrapper,
        dtypes=[torch.float16],
        trans=CUBLAS_OP_T,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.bfgemv
def test_perf_bfgemv():
    bench = HalfGemvBenchmark(
        op_name="bfgemv",
        torch_op=(
            hipblas_low_precision_gemv_baseline if IS_HYGON else cublas_half_gemv
        ),
        gems_op=gems_bfgemv_wrapper,
        dtypes=[torch.bfloat16],
        trans=CUBLAS_OP_N,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.bfgemv
def test_perf_bfgemv_trans():
    bench = HalfGemvBenchmark(
        op_name="bfgemv",
        torch_op=(
            hipblas_low_precision_gemv_baseline if IS_HYGON else cublas_half_gemv
        ),
        gems_op=gems_bfgemv_wrapper,
        dtypes=[torch.bfloat16],
        trans=CUBLAS_OP_T,
    )
    run_correctness_then_benchmark(bench)


def cublas_sgemv_fp8_baseline(
    A_fp8,
    x_fp8,
    y,
    trans,
    m,
    n,
    alpha,
    A_col_f32,
    x_f32_ref,
    lda,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
):
    cublas.sgemv(
        handle,
        trans,
        m,
        n,
        alpha_ptr,
        A_col_f32.data_ptr(),
        lda,
        x_f32_ref.data_ptr(),
        incx,
        beta_ptr,
        y.data_ptr(),
        incy,
    )
    return y


def gems_fp8_gemv_wrapper(
    A_fp8,
    x_fp8,
    y,
    trans,
    m,
    n,
    alpha,
    A_col_f32,
    x_f32_ref,
    lda,
    incx,
    beta,
    incy,
    handle,
    alpha_ptr,
    beta_ptr,
    **kwargs,
):
    flag_blas.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, incx, beta, y, incy)
    return y


class Fp8GemvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        trans=CUBLAS_OP_N,
        alpha=1.5,
        beta=0.5,
        fp8_dtype=torch.float8_e4m3fn,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trans = trans
        self.alpha = alpha
        self.beta = beta
        self.fp8_dtype = fp8_dtype

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        shapes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (3584, 3584),
            (4096, 4096),
            (7168, 7168),
            (8192, 8192),
            (16384, 16384),
            (18432, 18432),
            (1024, 4096),
            (3584, 18944),
            (4096, 14336),
            (6144, 16384),
            (7168, 18432),
            (8192, 28672),
            (16384, 53248),
            (4096, 1024),
            (18944, 3584),
            (14336, 4096),
            (16384, 6144),
            (18432, 7168),
            (28672, 8192),
            (53248, 16384),
            (64, 65536),
            (65536, 64),
        ]
        self.shapes = [(m, n) for m, n in shapes if m % 16 == 0 and n % 16 == 0]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        self.alpha_np = np.array(self.alpha, dtype=np.float32)
        self.beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = self.alpha_np.ctypes.data
        beta_ptr = self.beta_np.ctypes.data

        for shape in self.shapes:
            m, n = shape
            x_len, y_len = (n, m) if self.trans == CUBLAS_OP_N else (m, n)

            A_f32 = torch.randn(m, n, dtype=torch.float32, device=self.device) * 0.1
            A_fp8 = A_f32.to(self.fp8_dtype)
            A_col_f32 = _sgemv_contiguous_matrix(A_fp8.float().t()).t()

            x_f32 = torch.randn(x_len, dtype=torch.float32, device=self.device) * 0.1
            x_fp8 = x_f32.to(self.fp8_dtype)
            x_f32_ref = x_fp8.float()

            y = torch.randn(y_len, dtype=torch.float32, device=self.device)

            yield A_fp8, x_fp8, y.clone(), {
                "trans": self.trans,
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_col_f32": A_col_f32,
                "x_f32_ref": x_f32_ref,
                "lda": m,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        return 2 * m * n

    def get_gbps(self, args, latency):
        A_fp8, x_fp8, y = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(A_fp8)
            + shape_utils.size_in_bytes(x_fp8)
            + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"] if kwargs["trans"] == CUBLAS_OP_N else kwargs["m"]

    def clone_correctness_inputs(self, args, kwargs):
        A_fp8, x_fp8, y = args
        ref_args = (A_fp8, x_fp8, y.clone())
        blas_args = (A_fp8, x_fp8, y.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.fp8gemv
def test_perf_fp8_gemv_e4m3_vs_sgemv_trans():
    if flag_blas.vendor_name == "iluvatar":
        pytest.skip("FP8 GEMV vendor baseline is unavailable on Iluvatar")
    if IS_HYGON:
        pytest.skip("FP8 GEMV cuBLAS baseline is unavailable on Hygon")
    bench = Fp8GemvBenchmark(
        op_name="fp8_gemv",
        torch_op=cublas_sgemv_fp8_baseline,
        gems_op=gems_fp8_gemv_wrapper,
        dtypes=[torch.float8_e4m3fn],
        trans=CUBLAS_OP_T,
        fp8_dtype=torch.float8_e4m3fn,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.fp8gemv
def test_perf_fp8_gemv_e5m2_vs_sgemv_trans():
    if flag_blas.vendor_name == "iluvatar":
        pytest.skip("FP8 GEMV vendor baseline is unavailable on Iluvatar")
    if IS_HYGON:
        pytest.skip("FP8 GEMV cuBLAS baseline is unavailable on Hygon")
    bench = Fp8GemvBenchmark(
        op_name="fp8_gemv",
        torch_op=cublas_sgemv_fp8_baseline,
        gems_op=gems_fp8_gemv_wrapper,
        dtypes=[torch.float8_e5m2],
        trans=CUBLAS_OP_T,
        fp8_dtype=torch.float8_e5m2,
    )
    run_correctness_then_benchmark(bench)

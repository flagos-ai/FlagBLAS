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
# Keep GEMV coverage across data types when updating a backend implementation.

import ctypes

import numpy as np
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
from flag_blas.ops import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
from .conftest import TO_CPU

IS_ASCEND = flag_blas.vendor_name == "ascend"
IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_HYGON or IS_MTHREADS:
    from .vendor_blas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )
elif not (IS_ASCEND or IS_MTHREADS):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

ASCEND_CPU_ONLY = pytest.mark.skipif(
    not IS_ASCEND or not TO_CPU,
    reason="Ascend --ref cpu vendor regression",
)
ASCEND_CPU_OR_HYGON = pytest.mark.skipif(
    not ((IS_ASCEND and TO_CPU) or IS_HYGON),
    reason="Ascend CPU-reference or Hygon regression",
)
HYGON_ONLY = pytest.mark.skipif(not IS_HYGON, reason="Hygon regression")
HYGON_FP8_UNSUPPORTED = pytest.mark.skipif(
    IS_HYGON,
    reason="Hygon does not support FP8 GEMV",
)


def _probe_ascend_fp8_dtype(dtype_name):
    if not IS_ASCEND:
        return True, ""

    fp8_dtype = getattr(torch, dtype_name, None)
    if fp8_dtype is None:
        return (
            False,
            f"Ascend FP8 capability unavailable: torch.{dtype_name} is missing",
        )
    try:
        fp8_value = torch.zeros(1, dtype=torch.float32, device=flag_blas.device).to(
            fp8_dtype
        )
        fp8_value.to(device="cpu", dtype=torch.float64).contiguous()
        torch.zeros(0, dtype=fp8_dtype, device=flag_blas.device)
    except (AttributeError, RuntimeError, TypeError) as exc:
        detail = str(exc).splitlines()[0]
        return (
            False,
            f"Ascend FP8 capability unavailable for torch.{dtype_name}: {detail}",
        )

    return True, ""


def _require_ascend_fp8_dtype(dtype_name):
    supported, reason = _probe_ascend_fp8_dtype(dtype_name)
    if not supported:
        pytest.skip(reason)


@pytest.fixture(scope="module")
def _require_ascend_fp8_e4m3():
    _require_ascend_fp8_dtype("float8_e4m3fn")


@pytest.fixture(scope="module")
def _require_ascend_fp8_e5m2():
    _require_ascend_fp8_dtype("float8_e5m2")


ASCEND_FP8_E4M3_REQUIRED = pytest.mark.usefixtures("_require_ascend_fp8_e4m3")
ASCEND_FP8_E5M2_REQUIRED = pytest.mark.usefixtures("_require_ascend_fp8_e5m2")

GEMV_SHAPES = [
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

FP8_GEMV_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (4096, 4096),
    (8192, 8192),
    (1024, 4096),
    (4096, 14336),
    (8192, 28672),
    (4096, 1024),
    (14336, 4096),
    (28672, 8192),
    (64, 65536),
    (65536, 64),
]

STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]

_HUGE_NONCONTIG_COPY_BYTES = 2 * 1024 * 1024 * 1024
_CHUNKED_COPY_BYTES = 256 * 1024 * 1024


def gemv_randn(shape, dtype, device):
    if IS_ASCEND and dtype == torch.complex64:
        if isinstance(shape, int):
            shape = (shape,)
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


def _needs_iluvatar_sgemv_chunked_noncontig_copy(tensor):
    # Iluvatar currently corrupts very large non-contiguous float32 copies
    # (for example transposed GEMV matrices above 2 GiB) when using one
    # monolithic contiguous/to-CPU conversion. Keep this workaround scoped to
    # sgemv-sized float32 matrices so other GEMV tests keep the original path.
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


def _sgemv_cpu_blas_tensor(tensor):
    if _needs_iluvatar_sgemv_chunked_noncontig_copy(tensor):
        return _chunked_2d_copy(tensor.detach(), device="cpu", dtype=torch.float64)
    return to_cpu_blas_tensor(tensor)


def prepare_fp8_gemv_data(m, n, incx, incy, fp8_dtype, y_dtype, device):
    A_f32 = torch.randn(m, n, dtype=torch.float32, device=device)
    A_fp8 = A_f32.to(fp8_dtype)

    x_f32 = torch.randn(m * incx, dtype=torch.float32, device=device)
    x_fp8 = x_f32.to(fp8_dtype)

    y = torch.randn(n * incy, dtype=y_dtype, device=device)
    if TO_CPU:
        return A_fp8, None, x_fp8, None, y, None

    A_col_f32 = A_fp8.float().t().contiguous().t()
    x_f32_ref = x_fp8.float()

    return A_fp8, A_col_f32, x_fp8, x_f32_ref, y, None


def cublas_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return

    dtype = A.dtype
    if dtype == torch.float32:
        func, np_dtype = cublas.sgemv, np.float32
    elif dtype == torch.float64:
        func, np_dtype = cublas.dgemv, np.float64
    elif dtype == torch.complex64:
        func, np_dtype = cublas.cgemv, np.complex64
    elif dtype == torch.complex128:
        func, np_dtype = cublas.zgemv, np.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    alpha_np = np.asarray(alpha, dtype=np_dtype)
    beta_np = np.asarray(beta, dtype=np_dtype)

    func(
        handle,
        trans,
        m,
        n,
        alpha_np.ctypes.data,
        A.data_ptr(),
        lda,
        x.data_ptr(),
        incx,
        beta_np.ctypes.data,
        y.data_ptr(),
        incy,
    )


def hipblas_gemv_operation(trans):
    operations = {CUBLAS_OP_N: 111, CUBLAS_OP_T: 112, CUBLAS_OP_C: 113}
    if trans not in operations:
        raise ValueError(f"Unsupported GEMV transpose mode: {trans}")
    return operations[trans]


def hipblas_sgemv_reference(
    trans,
    m,
    n,
    alpha,
    A,
    lda,
    x,
    incx,
    beta,
    y,
    incy,
):
    if m == 0 or n == 0:
        return y

    hip_trans = hipblas_gemv_operation(trans)
    alpha_value = ctypes.c_float(float(alpha))
    beta_value = ctypes.c_float(float(beta))
    library, handle = get_hipblas_context(A)
    function = library.hipblasSgemv
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
    check_hipblas_status(
        function(
            handle,
            hip_trans,
            m,
            n,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        "hipblasSgemv",
    )
    return y


def hipblas_dgemv_reference(
    trans,
    m,
    n,
    alpha,
    A,
    lda,
    x,
    incx,
    beta,
    y,
    incy,
):
    if m == 0 or n == 0:
        return y

    hip_trans = hipblas_gemv_operation(trans)
    alpha_value = ctypes.c_double(float(alpha))
    beta_value = ctypes.c_double(float(beta))
    library, handle = get_hipblas_context(A)
    function = library.hipblasDgemv
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
    check_hipblas_status(
        function(
            handle,
            hip_trans,
            m,
            n,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        "hipblasDgemv",
    )
    return y


def hipblas_cgemv_reference(
    trans,
    m,
    n,
    alpha,
    A,
    lda,
    x,
    incx,
    beta,
    y,
    incy,
):
    if m == 0 or n == 0:
        return y

    alpha_value = HipComplex(float(alpha.real), float(alpha.imag))
    beta_value = HipComplex(float(beta.real), float(beta.imag))
    library, handle = get_hipblas_context(A)
    function = library.hipblasCgemv_v2
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
    check_hipblas_status(
        function(
            handle,
            hipblas_gemv_operation(trans),
            m,
            n,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        "hipblasCgemv_v2",
    )
    return y


def hipblas_zgemv_reference(
    trans,
    m,
    n,
    alpha,
    A,
    lda,
    x,
    incx,
    beta,
    y,
    incy,
):
    if m == 0 or n == 0:
        return y

    alpha_value = HipDoubleComplex(float(alpha.real), float(alpha.imag))
    beta_value = HipDoubleComplex(float(beta.real), float(beta.imag))
    library, handle = get_hipblas_context(A)
    function = library.hipblasZgemv_v2
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
    check_hipblas_status(
        function(
            handle,
            hipblas_gemv_operation(trans),
            m,
            n,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y.data_ptr()),
            incy,
        ),
        "hipblasZgemv_v2",
    )
    return y


def hipblas_low_precision_gemv_reference(
    trans,
    m,
    n,
    alpha,
    A,
    lda,
    x,
    incx,
    beta,
    y,
    incy,
):
    if m == 0 or n == 0:
        return y

    if A.dtype == torch.float16:
        data_type = 2
    elif A.dtype == torch.bfloat16:
        data_type = 14
    else:
        raise ValueError(f"Unsupported low-precision GEMV dtype: {A.dtype}")

    assert x.dtype == A.dtype
    assert y.dtype == A.dtype

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x_work = x[: 1 + (x_len - 1) * incx : incx].contiguous()
    if incy == 1:
        y_work = y
    else:
        y_work = y[: 1 + (y_len - 1) * incy : incy].contiguous()

    library, handle = get_hipblas_context(A)
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

    if IS_MTHREADS and A.dtype == torch.float16:
        alpha_value = ctypes.c_uint16(
            np.asarray(alpha, dtype=np.float16).view(np.uint16).item()
        )
        beta_value = ctypes.c_uint16(
            np.asarray(beta, dtype=np.float16).view(np.uint16).item()
        )
    else:
        alpha_value = ctypes.c_float(float(alpha))
        beta_value = ctypes.c_float(float(beta))
    if trans == CUBLAS_OP_N:
        trans_a = 112
        gemm_m, gemm_k = m, n
    else:
        trans_a = 111
        gemm_m, gemm_k = n, m

    check_hipblas_status(
        function(
            handle,
            trans_a,
            111,
            gemm_m,
            1,
            gemm_k,
            ctypes.byref(alpha_value),
            ctypes.c_void_p(A.data_ptr()),
            data_type,
            lda,
            ctypes.c_void_p(x_work.data_ptr()),
            data_type,
            gemm_k,
            ctypes.byref(beta_value),
            ctypes.c_void_p(y_work.data_ptr()),
            data_type,
            gemm_m,
            (64 if A.dtype == torch.float16 else 68) if IS_MTHREADS else 2,
            0 if IS_MTHREADS else 160,
        ),
        "hipblasGemmEx_v2",
    )

    if incy != 1:
        y[: 1 + (y_len - 1) * incy : incy].copy_(y_work)
    return y


def cpu_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return to_cpu_blas_tensor(y)

    ref_A = _sgemv_cpu_blas_tensor(A)
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    if beta == 0:
        len_y = m if trans == CUBLAS_OP_N else n
        logical_y = ref_y.as_strided((len_y,), (incy,))
        nan_value = (
            complex(float("nan"), float("nan")) if y.is_complex() else float("nan")
        )
        logical_y.fill_(nan_value)
    func = cpu_blas.zgemv if ref_A.dtype.is_complex else cpu_blas.dgemv

    yout = func(
        alpha,
        ref_A.numpy(),
        ref_x.numpy(),
        beta=beta,
        y=ref_y.numpy(),
        incx=incx,
        incy=incy,
        trans=trans,
        overwrite_y=1,
    )
    return torch.from_numpy(yout)


def gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if TO_CPU:
        return cpu_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    ref_y = y.clone()
    if IS_HYGON or IS_MTHREADS:
        if A.dtype == torch.float32:
            hipblas_sgemv_reference(
                trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy
            )
        elif A.dtype == torch.float64:
            hipblas_dgemv_reference(
                trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy
            )
        elif A.dtype == torch.complex64:
            hipblas_cgemv_reference(
                trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy
            )
        elif A.dtype == torch.complex128:
            hipblas_zgemv_reference(
                trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy
            )
        elif A.dtype in (torch.float16, torch.bfloat16):
            hipblas_low_precision_gemv_reference(
                trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy
            )
        else:
            raise ValueError(f"Unsupported vendor GEMV reference dtype: {A.dtype}")
    elif A.dtype in (torch.float16, torch.bfloat16):
        cupy_half_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    else:
        cublas_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, ref_y, incy)
    return ref_y


@pytest.mark.parametrize(
    "trans,dtype,incx,incy,beta",
    [
        (CUBLAS_OP_N, torch.float32, 2, 3, 0.0),
        (CUBLAS_OP_T, torch.float32, 3, 2, 0.25),
        (CUBLAS_OP_N, torch.complex64, 2, 3, 0.0),
        (CUBLAS_OP_T, torch.complex64, 3, 2, 0.25 - 0.5j),
        (CUBLAS_OP_C, torch.complex64, 2, 3, 0.25 - 0.5j),
    ],
)
def test_cpu_gemv_reference(trans, dtype, incx, incy, beta):
    m, n = 3, 2
    dense_dtype = torch.complex128 if dtype.is_complex else torch.float64
    A = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]], dtype=dense_dtype)
    if dtype.is_complex:
        A = A + 1j * torch.tensor(
            [[0.25, -0.5], [1.0, 0.75], [-0.25, 0.5]], dtype=torch.float64
        )
    input_len, output_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x_values = torch.tensor([0.75, -1.25, 2.0][:input_len], dtype=dense_dtype)
    if dtype.is_complex:
        x_values = x_values + 1j * torch.tensor(
            [0.5, -0.75, 0.25][:input_len], dtype=torch.float64
        )
    y_values = torch.tensor([-0.5, 1.5, -2.0][:output_len], dtype=dense_dtype)
    if dtype.is_complex:
        y_values = y_values + 1j * torch.tensor(
            [0.25, 0.5, -1.0][:output_len], dtype=torch.float64
        )

    x = torch.full((1 + (input_len - 1) * incx,), 37, dtype=dtype)
    y = torch.full((1 + (output_len - 1) * incy,), 41, dtype=dtype)
    x[::incx] = x_values.to(dtype)
    y[::incy] = y_values.to(dtype)
    if beta == 0:
        y[::incy] = (
            complex(float("nan"), float("nan")) if dtype.is_complex else float("nan")
        )
    alpha = 1.25 + 0.5j if dtype.is_complex else 1.25

    actual = cpu_gemv_reference(
        trans, m, n, alpha, A.to(dtype), n, x, incx, beta, y, incy
    )
    if trans == CUBLAS_OP_N:
        op_A = A
    elif trans == CUBLAS_OP_T:
        op_A = A.T
    else:
        op_A = A.mH
    expected = y.to(dense_dtype)
    expected[::incy] = alpha * (op_A @ x_values) + beta * y_values

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.sgemv
@ASCEND_CPU_ONLY
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_sgemv_alpha_zero_stride(trans, beta):
    m, n, incy = 3, 2, 2
    dtype = torch.float32
    A = torch.ones((m, n), dtype=dtype, device=flag_blas.device)
    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.ones(len_x, dtype=dtype, device=flag_blas.device)
    y = torch.arange(len_y * incy, dtype=dtype, device=flag_blas.device)
    expected = y.clone()
    expected[::incy].mul_(beta)
    expected = to_reference(expected)

    flag_blas.sgemv(trans, m, n, 0.0, A, n, x, 1, beta, y, incy)

    blas_assert_close(y, expected, dtype, reduce_dim=1)


@pytest.mark.cgemv
@ASCEND_CPU_OR_HYGON
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_cgemv_alpha_zero_stride(trans, beta):
    m, n, incy = 3, 2, 2
    dtype = torch.complex64
    A = torch.view_as_complex(
        torch.ones((m, n, 2), dtype=torch.float32, device=flag_blas.device)
    )
    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.view_as_complex(
        torch.ones((len_x, 2), dtype=torch.float32, device=flag_blas.device)
    )
    y = torch.view_as_complex(
        torch.arange(
            len_y * incy * 2, dtype=torch.float32, device=flag_blas.device
        ).reshape(len_y * incy, 2)
    )
    expected = y.clone()
    expected[::incy].mul_(beta)
    expected = to_reference(expected)

    flag_blas.cgemv(trans, m, n, 0.0, A, n, x, 1, beta, y, incy)

    blas_assert_close(y, expected, dtype, reduce_dim=1)


@pytest.mark.zgemv
@HYGON_ONLY
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_zgemv_alpha_zero_stride(trans, beta):
    m, n, incy = 3, 2, 3
    dtype = torch.complex128
    nan = complex(float("nan"), float("nan"))
    sentinel = 41.0 + 13.0j
    A = torch.full((m, n), nan, dtype=dtype, device=flag_blas.device)
    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.full((len_x,), nan, dtype=dtype, device=flag_blas.device)
    y = torch.full(
        (1 + (len_y - 1) * incy + 2,),
        sentinel,
        dtype=dtype,
        device=flag_blas.device,
    )
    y_values = torch.randn(len_y, dtype=dtype, device=flag_blas.device)
    y[0 : len_y * incy : incy] = y_values
    if beta == 0.0:
        y[0 : len_y * incy : incy] = nan

    expected = y.clone()
    if beta == 0.0:
        expected[0 : len_y * incy : incy] = 0.0
    else:
        expected[0 : len_y * incy : incy] = beta * y_values
    expected = to_reference(expected)

    flag_blas.zgemv(trans, m, n, 0.0, A, n, x, 1, beta, y, incy)

    blas_assert_close(y, expected, dtype, reduce_dim=1)


@pytest.mark.sgemv
@ASCEND_CPU_ONLY
@pytest.mark.parametrize("m,n,trans", [(1, 4096, CUBLAS_OP_N), (4096, 1, CUBLAS_OP_T)])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_sgemv_splitk_stride_tail(m, n, trans, beta):
    dtype, alpha, incx, incy = torch.float32, 1.5, 2, 2
    A = torch.randn((m, n), dtype=dtype, device=flag_blas.device)
    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.full((len_x * incx,), 37.0, dtype=dtype, device=flag_blas.device)
    x_values = torch.randn(len_x, dtype=dtype, device=flag_blas.device)
    x[::incx] = x_values
    y = torch.full((len_y * incy,), 41.0, dtype=dtype, device=flag_blas.device)
    y_values = torch.randn(len_y, dtype=dtype, device=flag_blas.device)
    y[::incy] = y_values
    if beta == 0.0:
        y[::incy] = float("nan")

    op_A = A if trans == CUBLAS_OP_N else A.T
    expected = to_reference(y, upcast=True)
    product = op_A.to(torch.float64).to("cpu") @ x_values.to(torch.float64).to("cpu")
    old_y = y_values.to(torch.float64).to("cpu")
    expected[::incy] = alpha * product + beta * old_y

    flag_blas.sgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    blas_assert_close(y, expected, dtype, reduce_dim=len_x)


@pytest.mark.cgemv
@HYGON_ONLY
@pytest.mark.parametrize(
    "m,n,trans",
    [
        (1, 4096, CUBLAS_OP_N),
        (4096, 1, CUBLAS_OP_T),
        (4096, 1, CUBLAS_OP_C),
    ],
)
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_cgemv_splitk_stride_tail(m, n, trans, beta):
    dtype, alpha, incx, incy = torch.complex64, 1.5 + 0.5j, 2, 2
    A = torch.randn((m, n), dtype=dtype, device=flag_blas.device)
    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.full((len_x * incx,), 37.0 + 11.0j, dtype=dtype, device=flag_blas.device)
    x_values = torch.randn(len_x, dtype=dtype, device=flag_blas.device)
    x[::incx] = x_values
    y = torch.full((len_y * incy,), 41.0 + 13.0j, dtype=dtype, device=flag_blas.device)
    y_values = torch.randn(len_y, dtype=dtype, device=flag_blas.device)
    y[::incy] = y_values
    if beta == 0.0:
        y[::incy] = complex(float("nan"), float("nan"))

    if trans == CUBLAS_OP_N:
        op_A = A
    elif trans == CUBLAS_OP_T:
        op_A = A.T
    else:
        op_A = A.mH
    expected = to_reference(y, upcast=True)
    product = op_A.to(torch.complex128).to("cpu") @ x_values.to(torch.complex128).to(
        "cpu"
    )
    old_y = y_values.to(torch.complex128).to("cpu")
    expected[::incy] = alpha * product + beta * old_y

    flag_blas.cgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    blas_assert_close(y, expected, dtype, reduce_dim=len_x)


@pytest.mark.zgemv
@HYGON_ONLY
@pytest.mark.parametrize(
    "m,n,trans",
    [
        (1, 4096, CUBLAS_OP_N),
        (4096, 1, CUBLAS_OP_T),
        (4096, 1, CUBLAS_OP_C),
    ],
)
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_zgemv_splitk_stride_tail(m, n, trans, beta):
    dtype, alpha, incx, incy = torch.complex128, 1.5 + 0.5j, 2, 2
    sentinel_x = 37.0 + 11.0j
    sentinel_y = 41.0 + 13.0j
    A = torch.randn((m, n), dtype=dtype, device=flag_blas.device)
    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.full(
        (1 + (len_x - 1) * incx + 2,),
        sentinel_x,
        dtype=dtype,
        device=flag_blas.device,
    )
    x_values = torch.randn(len_x, dtype=dtype, device=flag_blas.device)
    x[0 : len_x * incx : incx] = x_values
    y = torch.full(
        (1 + (len_y - 1) * incy + 2,),
        sentinel_y,
        dtype=dtype,
        device=flag_blas.device,
    )
    y_values = torch.randn(len_y, dtype=dtype, device=flag_blas.device)
    y[0 : len_y * incy : incy] = y_values
    if beta == 0.0:
        y[0 : len_y * incy : incy] = complex(float("nan"), float("nan"))

    if trans == CUBLAS_OP_N:
        op_A = A
    elif trans == CUBLAS_OP_T:
        op_A = A.T
    else:
        op_A = A.mH
    expected = to_reference(y.clone(), upcast=True)
    product = op_A.to("cpu") @ x_values.to("cpu")
    old_y = y_values.to("cpu")
    expected[0 : len_y * incy : incy] = alpha * product + beta * old_y

    flag_blas.zgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    blas_assert_close(y, expected, dtype, reduce_dim=len_x)


def fp8_gemv_reference(trans, m, n, alpha, A, A_col_ref, x, x_ref, incx, beta, y, incy):
    if TO_CPU:
        return cpu_gemv_reference(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    ref_y = y.float().clone()
    if IS_MTHREADS:
        hipblas_sgemv_reference(
            trans, m, n, alpha, A_col_ref, m, x_ref, incx, beta, ref_y, incy
        )
    else:
        cublas_gemv_reference(
            trans, m, n, alpha, A_col_ref, m, x_ref, incx, beta, ref_y, incy
        )
    return ref_y


def cupy_half_gemv_reference(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if m == 0 or n == 0:
        return

    x_len = n if trans == CUBLAS_OP_N else m
    y_len = m if trans == CUBLAS_OP_N else n

    x_contig = x[::incx][:x_len].contiguous()
    y_contig = y[::incy][:y_len].contiguous()

    CUDA_R_32F = 0
    CUDA_R_16F = 2
    CUDA_R_16BF = 14

    dtype = A.dtype
    if dtype == torch.float16:
        cuda_type = CUDA_R_16F
    elif dtype == torch.bfloat16:
        cuda_type = CUDA_R_16BF
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    alpha_np = np.array([alpha], dtype=np.float32)
    beta_np = np.array([beta], dtype=np.float32)

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    if trans == CUBLAS_OP_N:
        transA = cublas.CUBLAS_OP_T
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = m, 1, n
        lda_c, ldb_c, ldc_c = lda, n, m
    else:
        transA = cublas.CUBLAS_OP_N
        transB = cublas.CUBLAS_OP_N
        m_c, n_c, k_c = n, 1, m
        lda_c, ldb_c, ldc_c = lda, m, n

    cublas.gemmEx(
        handle,
        transA,
        transB,
        m_c,
        n_c,
        k_c,
        alpha_np.ctypes.data,
        A.data_ptr(),
        cuda_type,
        lda_c,
        x_contig.data_ptr(),
        cuda_type,
        ldb_c,
        beta_np.ctypes.data,
        y_contig.data_ptr(),
        cuda_type,
        ldc_c,
        CUDA_R_32F,
        0,
    )

    y[::incy][:y_len].copy_(y_contig)


@pytest.mark.sgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_sgemv(m, n, trans, beta):
    dtype, alpha = torch.float32, 1.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = _sgemv_contiguous_matrix(A_col)

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.sgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.sgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_sgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.float32, 2.0, 0.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.sgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.dgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dgemv_stride(m, n, trans, incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha, beta = torch.float64, 2.0, 0.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.dgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-14 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.dgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_dgemv(m, n, trans, beta):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha = torch.float64, 1.5

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.dgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-13 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.cgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_cgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.complex64, 2.0 + 0.5j, 0.5 + 0.25j

    A_col = gemv_randn((n, m), dtype, flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = gemv_randn(x_len * incx, dtype, flag_blas.device)
    y = gemv_randn(y_len * incy, dtype, flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.cgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.cgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_accuracy_cgemv(m, n, trans, beta):
    dtype, alpha = torch.complex64, 1.5 + 0.5j

    A_col = gemv_randn((n, m), dtype, flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = gemv_randn(x_len, dtype, flag_blas.device)
    y = gemv_randn(y_len, dtype, flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.cgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-5 * (x_len**0.5), 1e-3)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.zgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zgemv_stride(m, n, trans, incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha, beta = torch.complex128, 2.0 + 0.5j, 0.5 + 0.25j

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, incx, beta, y, incy)
    flag_blas.zgemv(trans, m, n, alpha, A_row, n, x, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-14 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.zgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C])
@pytest.mark.parametrize("beta", [0.0, 0.5 + 0.25j])
def test_accuracy_zgemv(m, n, trans, beta):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("No FP64")

    dtype, alpha = torch.complex128, 1.5 + 0.5j

    A_col = torch.randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A_col, m, x, 1, beta, y, 1)
    flag_blas.zgemv(trans, m, n, alpha, A_row, n, x, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)
    else:
        tol = min(1e-14 * (x_len**0.5), 1e-11)
        torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)


@pytest.mark.hgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_hgemv(m, n, trans, beta):
    dtype, alpha = torch.float16, 1.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, 1, beta, y, 1)
    flag_blas.hgemv(trans, m, n, alpha, A, n, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)


@pytest.mark.hgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_hgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.float16, 2.0, 0.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, incx, beta, y, incy)
    flag_blas.hgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)


@pytest.mark.bfgemv
@pytest.mark.parametrize("m,n", GEMV_SHAPES)
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_bfgemv(m, n, trans, beta):
    dtype, alpha = torch.bfloat16, 1.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len, dtype=dtype, device=flag_blas.device)

    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, 1, beta, y, 1)
    flag_blas.bfgemv(trans, m, n, alpha, A, n, x, 1, beta, y, 1)

    blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)


@pytest.mark.bfgemv
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("trans", [CUBLAS_OP_N, CUBLAS_OP_T])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_bfgemv_stride(m, n, trans, incx, incy):
    dtype, alpha, beta = torch.bfloat16, 2.0, 0.5

    A = torch.randn(m, n, dtype=dtype, device=flag_blas.device).contiguous()

    x_len, y_len = (n, m) if trans == CUBLAS_OP_N else (m, n)
    x = torch.randn(x_len * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(y_len * incy, dtype=dtype, device=flag_blas.device)
    ref_y = gemv_reference(trans, m, n, alpha, A, n, x, incx, beta, y, incy)
    flag_blas.bfgemv(trans, m, n, alpha, A, n, x, incx, beta, y, incy)

    blas_assert_close(y, ref_y, dtype, reduce_dim=x_len)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E4M3_REQUIRED
@pytest.mark.parametrize("m,n", FP8_GEMV_SHAPES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_fp8_gemv_e4m3(m, n, beta):
    fp8_dtype = torch.float8_e4m3fn
    alpha = 1.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, 1, 1, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, 1, beta, y, 1
    )
    flag_blas.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E4M3_REQUIRED
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_fp8_gemv_e4m3_stride(m, n, incx, incy):
    fp8_dtype = torch.float8_e4m3fn
    alpha, beta = 2.0, 0.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, incx, incy, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, incx, beta, y, incy
    )
    flag_blas.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E5M2_REQUIRED
@pytest.mark.parametrize("m,n", FP8_GEMV_SHAPES)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_accuracy_fp8_gemv_e5m2(m, n, beta):
    fp8_dtype = torch.float8_e5m2
    alpha = 1.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, 1, 1, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, 1, beta, y, 1
    )
    flag_blas.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E5M2_REQUIRED
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_fp8_gemv_e5m2_stride(m, n, incx, incy):
    fp8_dtype = torch.float8_e5m2
    alpha, beta = 2.0, 0.5
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, incx, incy, fp8_dtype, torch.float32, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, incx, beta, y, incy
    )
    flag_blas.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, incx, beta, y, incy)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E4M3_REQUIRED
@pytest.mark.parametrize("m,n", [(256, 256), (1024, 1024), (4096, 4096)])
@pytest.mark.parametrize("y_dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fp8_gemv_output_dtype(m, n, y_dtype):
    fp8_dtype = torch.float8_e4m3fn
    alpha, beta = 1.0, 0.0
    trans = CUBLAS_OP_T

    A_fp8, A_col_f32, x_fp8, x_f32_ref, y, ref_y = prepare_fp8_gemv_data(
        m, n, 1, 1, fp8_dtype, y_dtype, flag_blas.device
    )

    ref_y = fp8_gemv_reference(
        trans, m, n, alpha, A_fp8, A_col_f32, x_fp8, x_f32_ref, 1, beta, y, 1
    )
    flag_blas.fp8_gemv(trans, m, n, alpha, A_fp8, n, x_fp8, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, y_dtype, reduce_dim=m)
    else:
        rtol = 1e-3 if y_dtype == torch.float16 else 1.6e-2
        atol = (3e-3 if y_dtype == torch.float16 else 3e-2) * (m**0.5)
        torch.testing.assert_close(y.float(), ref_y, rtol=rtol, atol=atol)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E4M3_REQUIRED
def test_fp8_gemv_alpha_zero():
    m, n = 128, 256
    fp8_dtype = torch.float8_e4m3fn

    A_f32 = torch.randn(m, n, dtype=torch.float32, device=flag_blas.device)
    A_fp8 = A_f32.to(fp8_dtype)

    x_f32 = torch.randn(m, dtype=torch.float32, device=flag_blas.device)
    x_fp8 = x_f32.to(fp8_dtype)

    y = torch.randn(n, dtype=torch.float32, device=flag_blas.device)
    y_orig = y.clone()

    flag_blas.fp8_gemv(CUBLAS_OP_T, m, n, 0.0, A_fp8, n, x_fp8, 1, 2.0, y, 1)

    if TO_CPU:
        blas_assert_close(y, to_reference(y_orig * 2.0, upcast=True), torch.float32)
    else:
        torch.testing.assert_close(y, y_orig * 2.0)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E4M3_REQUIRED
def test_fp8_gemv_beta_zero():
    m, n = 128, 256
    fp8_dtype = torch.float8_e4m3fn

    A_f32 = torch.randn(m, n, dtype=torch.float32, device=flag_blas.device)
    A_fp8 = A_f32.to(fp8_dtype)

    x_f32 = torch.randn(m, dtype=torch.float32, device=flag_blas.device)
    x_fp8 = x_f32.to(fp8_dtype)

    y_nan = torch.full((n,), float("nan"), dtype=torch.float32, device=flag_blas.device)
    y_zero = torch.zeros(n, dtype=torch.float32, device=flag_blas.device)
    if TO_CPU:
        ref_y_nan = fp8_gemv_reference(
            CUBLAS_OP_T, m, n, 1.0, A_fp8, None, x_fp8, None, 1, 0.0, y_nan, 1
        )

    flag_blas.fp8_gemv(CUBLAS_OP_T, m, n, 1.0, A_fp8, n, x_fp8, 1, 0.0, y_nan, 1)
    flag_blas.fp8_gemv(CUBLAS_OP_T, m, n, 1.0, A_fp8, n, x_fp8, 1, 0.0, y_zero, 1)

    if TO_CPU:
        blas_assert_close(y_nan, ref_y_nan, torch.float32, reduce_dim=m)
        blas_assert_close(
            y_nan, to_reference(y_zero, upcast=True), torch.float32, reduce_dim=m
        )
    else:
        torch.testing.assert_close(y_nan, y_zero)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E4M3_REQUIRED
def test_fp8_gemv_empty():
    fp8_dtype = torch.float8_e4m3fn
    device = flag_blas.device

    m, n = 0, 64
    A_fp8 = torch.zeros(m, n, dtype=fp8_dtype, device=device)
    x_fp8 = torch.zeros(m, dtype=fp8_dtype, device=device)

    y = torch.randn(n, dtype=torch.float32, device=device)
    y_orig = y.clone()

    flag_blas.fp8_gemv(CUBLAS_OP_T, m, n, 1.0, A_fp8, n, x_fp8, 1, 0.0, y, 1)
    if TO_CPU:
        blas_assert_close(y, to_reference(y_orig, upcast=True), torch.float32)
    else:
        torch.testing.assert_close(y, y_orig)


@pytest.mark.fp8gemv
@pytest.mark.fp8_gemv
@HYGON_FP8_UNSUPPORTED
@ASCEND_FP8_E4M3_REQUIRED
@pytest.mark.parametrize("m,n", [(256, 256), (1024, 1024)])
def test_accuracy_fp8_gemv_mixed_dtype(m, n):
    alpha, beta = 1.0, 0.0
    device = flag_blas.device
    trans = CUBLAS_OP_T

    A_f32 = torch.randn(m, n, dtype=torch.float32, device=device)
    A_fp8_e4m3 = A_f32.to(torch.float8_e4m3fn)

    A_col_f32 = None if TO_CPU else A_fp8_e4m3.float().t().contiguous().t()

    x_f32 = torch.randn(m, dtype=torch.float32, device=device)
    x_fp8_e4m3 = x_f32.to(torch.float8_e4m3fn)
    x_f32_ref = None if TO_CPU else x_fp8_e4m3.float()

    y = torch.zeros(n, dtype=torch.float32, device=device)

    ref_y = fp8_gemv_reference(
        trans,
        m,
        n,
        alpha,
        A_fp8_e4m3,
        A_col_f32,
        x_fp8_e4m3,
        x_f32_ref,
        1,
        beta,
        y,
        1,
    )
    flag_blas.fp8_gemv(trans, m, n, alpha, A_fp8_e4m3, n, x_fp8_e4m3, 1, beta, y, 1)

    if TO_CPU:
        blas_assert_close(y, ref_y, torch.float32, reduce_dim=m)
    else:
        rtol = 1e-3
        atol = 3e-3 * (m**0.5)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)

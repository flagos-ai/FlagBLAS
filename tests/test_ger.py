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

import numpy as np
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor, to_reference
from .conftest import TO_CPU

IS_ASCEND = flag_blas.vendor_name == "ascend"
IS_HYGON = flag_blas.vendor_name == "hygon"
IS_MTHREADS = flag_blas.vendor_name == "mthreads"

if IS_HYGON or IS_MTHREADS:
    import ctypes

    from .vendor_blas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )
elif not IS_ASCEND and not IS_MTHREADS:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

GER_OPS = {
    "sger": (torch.float32, np.float32, 1.5),
    "dger": (torch.float64, np.float64, 1.5),
    "cgeru": (torch.complex64, np.complex64, 1.5 - 0.5j),
    "cgerc": (torch.complex64, np.complex64, 1.5 - 0.5j),
    "zgeru": (torch.complex128, np.complex128, 1.5 - 0.5j),
    "zgerc": (torch.complex128, np.complex128, 1.5 - 0.5j),
}

# Parametrize cases carrying per-op precision markers (sger/dger/cgeru/...),
# so `pytest -m "<op>"` (as used by tools/run_tests.py) selects the matching cases.
GER_OPS_CASES = [pytest.param(op, marks=getattr(pytest.mark, op)) for op in GER_OPS]

GER_PERF_SHAPES = [
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

GER_EDGE_SHAPES = [
    (128, 256),
    (257, 129),
    (1, 128),
    (128, 1),
    (1, 1),
    (0, 64),
    (64, 0),
]

GER_SHAPES = GER_PERF_SHAPES + GER_EDGE_SHAPES

STRIDE_SHAPES = [(64, 128), (127, 65), (1, 256), (256, 1)]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 3)]


def ger_randn(*shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


def _skip_if_unsupported(dtype):
    if dtype in (torch.float64, torch.complex128):
        if not flag_blas.runtime.device.support_fp64:
            pytest.skip("Device does not support float64")


def cublas_ger_reference(op_name, m, n, alpha, x, incx, y, incy, A, lda):
    if m == 0 or n == 0:
        return

    _, np_dtype, _ = GER_OPS[op_name]
    func = getattr(cublas, op_name)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    alpha_np = np.asarray(alpha, dtype=np_dtype)
    func(
        handle,
        m,
        n,
        alpha_np.ctypes.data,
        x.data_ptr(),
        incx,
        y.data_ptr(),
        incy,
        A.data_ptr(),
        lda,
    )


def hipblas_ger_reference(op_name, m, n, alpha, x, incx, y, incy, A, lda):
    if m == 0 or n == 0:
        return

    library, handle = get_hipblas_context(A)
    symbols = {
        "sger": ("hipblasSger", ctypes.c_float),
        "dger": ("hipblasDger", ctypes.c_double),
        "cgeru": ("hipblasCgeru_v2", HipComplex),
        "cgerc": ("hipblasCgerc_v2", HipComplex),
        "zgeru": ("hipblasZgeru_v2", HipDoubleComplex),
        "zgerc": ("hipblasZgerc_v2", HipDoubleComplex),
    }
    symbol, scalar_type = symbols[op_name]
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
    if op_name.startswith(("c", "z")):
        alpha_value = complex(alpha)
        scalar = scalar_type(alpha_value.real, alpha_value.imag)
    else:
        scalar = scalar_type(alpha)
    check_hipblas_status(
        function(
            handle,
            m,
            n,
            ctypes.byref(scalar),
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.c_void_p(y.data_ptr()),
            incy,
            ctypes.c_void_p(A.data_ptr()),
            lda,
        ),
        symbol,
    )


def cpu_ger_reference(op_name, m, n, alpha, x, incx, y, incy, A, lda):
    if m == 0 or n == 0:
        return to_cpu_blas_tensor(A)

    ref_A = to_cpu_blas_tensor(A)
    ref_x = to_cpu_blas_tensor(x)[::incx][:m].contiguous()
    ref_y = to_cpu_blas_tensor(y)[::incy][:n].contiguous()
    cpu_op_name = {
        "sger": "dger",
        "dger": "dger",
        "cgeru": "zgeru",
        "zgeru": "zgeru",
        "cgerc": "zgerc",
        "zgerc": "zgerc",
    }[op_name]
    func = getattr(cpu_blas, cpu_op_name)
    out = func(
        alpha,
        ref_x.numpy(),
        ref_y.numpy(),
        incx=1,
        incy=1,
        a=ref_A.numpy(),
        overwrite_a=1,
    )
    return torch.from_numpy(out)


def ger_reference(op_name, m, n, alpha, x, incx, y, incy, A, lda):
    if TO_CPU:
        return cpu_ger_reference(op_name, m, n, alpha, x, incx, y, incy, A, lda)

    ref_A = A.clone()
    if IS_HYGON or IS_MTHREADS:
        hipblas_ger_reference(op_name, m, n, alpha, x, incx, y, incy, ref_A, lda)
    else:
        cublas_ger_reference(op_name, m, n, alpha, x, incx, y, incy, ref_A, lda)
    return ref_A


def run_ger_case(op_name, m, n, alpha, incx=1, incy=1):
    dtype = GER_OPS[op_name][0]
    _skip_if_unsupported(dtype)

    A_col = ger_randn(n, m, dtype=dtype, device=flag_blas.device).t()
    A_row = A_col.contiguous()
    x = ger_randn(max(1, m) * incx, dtype=dtype, device=flag_blas.device)
    y = ger_randn(max(1, n) * incy, dtype=dtype, device=flag_blas.device)

    ref_A = ger_reference(op_name, m, n, alpha, x, incx, y, incy, A_col, m)
    getattr(flag_blas, op_name)(m, n, alpha, x, incx, y, incy, A_row, n)

    blas_assert_close(A_row, ref_A.contiguous(), dtype)


@pytest.mark.ger
@pytest.mark.parametrize("op_name", GER_OPS_CASES)
def test_ger_exports(op_name):
    assert hasattr(flag_blas.ops, op_name)
    assert hasattr(flag_blas, op_name)
    if IS_HYGON and op_name.startswith(("c", "z")):
        assert getattr(flag_blas, op_name) is not getattr(flag_blas.ops, op_name)
    elif not IS_ASCEND:
        assert getattr(flag_blas, op_name) is getattr(flag_blas.ops, op_name)


@pytest.mark.ger
@pytest.mark.parametrize("op_name", GER_OPS_CASES)
@pytest.mark.parametrize("m,n", GER_SHAPES)
def test_accuracy_ger(op_name, m, n):
    alpha = GER_OPS[op_name][2]
    run_ger_case(op_name, m, n, alpha=alpha)


@pytest.mark.ger
@pytest.mark.parametrize("op_name", GER_OPS_CASES)
@pytest.mark.parametrize("m,n", STRIDE_SHAPES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_ger_stride(op_name, m, n, incx, incy):
    alpha = GER_OPS[op_name][2]
    run_ger_case(op_name, m, n, alpha=alpha, incx=incx, incy=incy)


@pytest.mark.ger
@pytest.mark.parametrize("op_name", GER_OPS_CASES)
@pytest.mark.parametrize("m,n", [(64, 64), (127, 65), (1, 128), (128, 1)])
def test_accuracy_ger_alpha_zero(op_name, m, n):
    run_ger_case(op_name, m, n, alpha=0.0)


@pytest.mark.ger
@pytest.mark.parametrize("op_name", GER_OPS)
def test_ger_row_major_padded_layout(op_name):
    dtype, _, alpha = GER_OPS[op_name]
    _skip_if_unsupported(dtype)
    m, n, lda = 3, 2, 5
    x = ger_randn(m, dtype=dtype, device=flag_blas.device)
    y = ger_randn(n, dtype=dtype, device=flag_blas.device)
    A = ger_randn(m, lda, dtype=dtype, device=flag_blas.device)
    expected = A.clone()
    rhs = y.conj() if op_name.endswith("gerc") else y
    expected[:, :n] += alpha * x[:, None] * rhs[None, :]

    getattr(flag_blas, op_name)(m, n, alpha, x, 1, y, 1, A, lda)

    blas_assert_close(A, to_reference(expected), dtype, reduce_dim=1)


@pytest.mark.ger
@pytest.mark.parametrize(
    "op_u,op_c,dtype,alpha",
    [
        pytest.param(
            "cgeru",
            "cgerc",
            torch.complex64,
            0.75 + 0.25j,
            marks=(getattr(pytest.mark, "cgeru"), getattr(pytest.mark, "cgerc")),
        ),
        pytest.param(
            "zgeru",
            "zgerc",
            torch.complex128,
            0.75 + 0.25j,
            marks=(getattr(pytest.mark, "zgeru"), getattr(pytest.mark, "zgerc")),
        ),
    ],
)
def test_accuracy_ger_conjugate_difference(op_u, op_c, dtype, alpha):
    _skip_if_unsupported(dtype)
    m, n = 5, 7
    x = ger_randn(m, dtype=dtype, device=flag_blas.device)
    y = ger_randn(n, dtype=dtype, device=flag_blas.device)
    A_u = ger_randn(m, n, dtype=dtype, device=flag_blas.device)
    A_c = A_u.clone()
    if TO_CPU:
        ref_x = to_cpu_blas_tensor(x)
        ref_y = to_cpu_blas_tensor(y)
        ref_u = to_cpu_blas_tensor(A_u) + alpha * ref_x[:, None] * ref_y[None, :]
        ref_c = to_cpu_blas_tensor(A_c) + alpha * ref_x[:, None] * ref_y.conj()[None, :]
    else:
        ref_u = A_u + alpha * x[:, None] * y[None, :]
        ref_c = A_c + alpha * x[:, None] * y.conj()[None, :]

    getattr(flag_blas, op_u)(m, n, alpha, x, 1, y, 1, A_u, n)
    getattr(flag_blas, op_c)(m, n, alpha, x, 1, y, 1, A_c, n)

    blas_assert_close(A_u, to_reference(ref_u), dtype)
    blas_assert_close(A_c, to_reference(ref_c), dtype)
    if IS_ASCEND:
        # torch.equal does not support complex tensors on Ascend.
        assert not torch.equal(torch.view_as_real(A_u), torch.view_as_real(A_c))
    else:
        assert not torch.equal(A_u, A_c)

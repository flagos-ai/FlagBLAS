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

import numpy as np
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

if flag_blas.vendor_name in {"hygon", "mthreads"}:
    from .vendor_blas_reference import check_hipblas_status, get_hipblas_context

from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor
from .conftest import TO_CPU

pytestmark = pytest.mark.syr

SYR_SIZES = [
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
SYR_UPLOS = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
SYR_STRIDES = [(2, CUBLAS_FILL_MODE_UPPER, 64), (3, CUBLAS_FILL_MODE_LOWER, 128)]


def syr_randn(*shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


def check_fp64_support(dtype):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")


class _ComplexFloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


def _load_cublas():
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


_cublas = None
_cublas_handle = None


def _ensure_cublas():
    global _cublas
    if _cublas is None:
        _cublas = _load_cublas()
    return _cublas


def _get_cublas_handle():
    global _cublas_handle
    cublas = _ensure_cublas()
    if _cublas_handle is None:
        _cublas_handle = ctypes.c_void_p()
        status = cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        status = cublas.cublasSetPointerMode_v2(_cublas_handle, 0)
        if status != 0:
            raise RuntimeError(
                f"cublasSetPointerMode_v2 failed with status code: {status}"
            )
    return _cublas_handle


def _make_inputs(n, incx, dtype, device):
    check_fp64_support(dtype)
    lda = n
    x_len = 1 + (n - 1) * incx
    if dtype.is_complex:
        x = syr_randn(x_len, dtype=dtype, device=device)
        A = syr_randn(lda, n, dtype=dtype, device=device)
    else:
        x = torch.randn(x_len, dtype=dtype, device=device)
        A = torch.randn((lda, n), dtype=dtype, device=device)
    return x, A


def _row_to_column_full(A, n, lda):
    column_A = torch.zeros((n, lda), dtype=A.dtype, device=A.device)
    column_A[:, :n] = A[:n, :n].T
    return column_A


def _cpu_ref(name, uplo, n, alpha, x, incx, A, lda):
    x_cpu = to_cpu_blas_tensor(x)
    ref = to_cpu_blas_tensor(A)
    logical_A = ref[:n, :n].numpy().copy(order="F")
    lower = int(uplo == CUBLAS_FILL_MODE_LOWER)
    if name in ("ssyr", "dsyr"):
        updated = cpu_blas.dsyr(
            float(alpha),
            x_cpu.numpy(),
            a=logical_A,
            lower=lower,
            incx=incx,
            overwrite_a=1,
        )
    else:
        updated = cpu_blas.zsyr(
            complex(alpha),
            x_cpu.numpy(),
            a=logical_A,
            lower=lower,
            incx=incx,
            overwrite_a=1,
        )
    ref[:n, :n] = torch.from_numpy(updated.copy(order="C"))
    return ref


def _cublas_ref(name, uplo, n, alpha, x, incx, A, lda):
    cublas = _ensure_cublas()
    ref = A.clone()
    column_A = _row_to_column_full(ref, n, lda)
    if name == "ssyr":
        func, scalar = cublas.cublasSsyr_v2, ctypes.c_float(alpha)
    elif name == "dsyr":
        func, scalar = cublas.cublasDsyr_v2, ctypes.c_double(alpha)
    elif name == "csyr":
        value = complex(alpha)
        func, scalar = cublas.cublasCsyr_v2, _ComplexFloat(value.real, value.imag)
    else:
        value = complex(alpha)
        func, scalar = cublas.cublasZsyr_v2, _ComplexDouble(value.real, value.imag)
    status = func(
        _get_cublas_handle(),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(scalar),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(column_A.data_ptr()),
        ctypes.c_int(lda),
    )
    if status != 0:
        raise RuntimeError(f"cublasXsyr_v2 failed with status code: {status}")
    ref[:n, :n] = column_A[:, :n].T
    return ref


def _hipblas_syr_reference(name, uplo, n, alpha, x, incx, A, lda):
    ref = A.clone()
    if n == 0:
        return ref
    column_A = _row_to_column_full(ref, n, lda)
    library, handle = get_hipblas_context(column_A)
    symbols = {
        "ssyr": ("hipblasSsyr", ctypes.c_float),
        "dsyr": ("hipblasDsyr", ctypes.c_double),
        "csyr": ("hipblasCsyr_v2", _ComplexFloat),
        "zsyr": ("hipblasZsyr_v2", _ComplexDouble),
    }
    symbol, scalar_type = symbols[name]
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
    ]
    function.restype = ctypes.c_int
    if name in ("csyr", "zsyr"):
        alpha_value = complex(alpha)
        scalar = scalar_type(alpha_value.real, alpha_value.imag)
    else:
        scalar = scalar_type(alpha)
    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    check_hipblas_status(
        function(
            handle,
            hip_uplo,
            n,
            ctypes.byref(scalar),
            ctypes.c_void_p(x.data_ptr()),
            incx,
            ctypes.c_void_p(column_A.data_ptr()),
            lda,
        ),
        symbol,
    )
    ref[:n, :n] = column_A[:, :n].T
    return ref


def _reference(name, uplo, n, alpha, x, incx, A, lda):
    if TO_CPU:
        return _cpu_ref(name, uplo, n, alpha, x, incx, A, lda)
    if flag_blas.vendor_name in {"hygon", "mthreads"}:
        return _hipblas_syr_reference(name, uplo, n, alpha, x, incx, A, lda)
    return _cublas_ref(name, uplo, n, alpha, x, incx, A, lda)


def _run_syr(name, dtype, alpha, uplo, n, incx):
    device = flag_blas.device
    x, A = _make_inputs(n, incx, dtype, device)
    lda = n
    ref = _reference(name, uplo, n, alpha, x, incx, A, lda)
    getattr(flag_blas, name)(uplo, n, alpha, x, incx, A, lda)
    blas_assert_close(A, ref, dtype, reduce_dim=1)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.ssyr
def test_accuracy_ssyr_sizes(uplo, n):
    _run_syr("ssyr", torch.float32, 0.75, uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.ssyr
def test_accuracy_ssyr_stride(incx, uplo, n):
    _run_syr("ssyr", torch.float32, 0.75, uplo, n, incx)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.dsyr
def test_accuracy_dsyr_sizes(uplo, n):
    _run_syr("dsyr", torch.float64, 0.75, uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.dsyr
def test_accuracy_dsyr_stride(incx, uplo, n):
    _run_syr("dsyr", torch.float64, 0.75, uplo, n, incx)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.csyr
def test_accuracy_csyr_sizes(uplo, n):
    _run_syr("csyr", torch.complex64, np.complex64(0.5 + 0.25j), uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.csyr
def test_accuracy_csyr_stride(incx, uplo, n):
    _run_syr("csyr", torch.complex64, np.complex64(0.5 + 0.25j), uplo, n, incx)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.zsyr
def test_accuracy_zsyr_sizes(uplo, n):
    _run_syr("zsyr", torch.complex128, np.complex128(0.5 + 0.25j), uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.zsyr
def test_accuracy_zsyr_stride(incx, uplo, n):
    _run_syr("zsyr", torch.complex128, np.complex128(0.5 + 0.25j), uplo, n, incx)


def _make_regression_inputs(n, incx, dtype, lda):
    check_fp64_support(dtype)
    x_len = 1 + (n - 1) * incx if n > 0 else 0
    x = syr_randn(x_len, dtype=dtype, device=flag_blas.device)
    A = syr_randn(n, lda, dtype=dtype, device=flag_blas.device)
    return x, A.contiguous()


def _regression_reference(name, uplo, n, alpha, x, incx, A, lda):
    return _reference(name, uplo, n, alpha, x, incx, A, lda)


@pytest.mark.parametrize(
    "name,dtype", [("ssyr", torch.float32), ("dsyr", torch.float64)]
)
def test_syr_accepts_scalar_cuda_tensor_alpha(name, dtype):
    n, lda = 7, 10
    x, A = _make_regression_inputs(n, 1, dtype, lda)
    alpha = torch.tensor(0.75, dtype=dtype, device=flag_blas.device)
    ref = _regression_reference(
        name,
        CUBLAS_FILL_MODE_LOWER,
        n,
        alpha.item(),
        x,
        1,
        A.clone(),
        lda,
    )

    getattr(flag_blas, name)(CUBLAS_FILL_MODE_LOWER, n, alpha, x, 1, A, lda)

    blas_assert_close(A, ref, dtype, reduce_dim=1)


@pytest.mark.parametrize(
    "name,dtype,alpha",
    [
        ("dsyr", torch.float64, 0.12345678901234568),
        (
            "zsyr",
            torch.complex128,
            complex(0.12345678901234568, -0.2345678912345679),
        ),
    ],
)
def test_syr_preserves_double_scalar_precision(name, dtype, alpha):
    n = 31
    x, A = _make_regression_inputs(n, 1, dtype, n)
    ref = _regression_reference(
        name, CUBLAS_FILL_MODE_UPPER, n, alpha, x, 1, A.clone(), n
    )

    getattr(flag_blas, name)(CUBLAS_FILL_MODE_UPPER, n, alpha, x, 1, A, n)

    blas_assert_close(A, ref, dtype, reduce_dim=1)


def test_syr_n_zero_is_noop():
    x = torch.empty(0, dtype=torch.float32, device=flag_blas.device)
    A = torch.empty((0, 1), dtype=torch.float32, device=flag_blas.device)

    result = flag_blas.ssyr(CUBLAS_FILL_MODE_LOWER, 0, 0.75, x, 1, A, 1)

    assert result is A


def test_syr_rejects_noncontiguous_matrix():
    n = 8
    x = torch.randn(n, device=flag_blas.device)
    A = torch.randn((n, 2 * n), device=flag_blas.device)[:, ::2]

    with pytest.raises(AssertionError):
        flag_blas.ssyr(CUBLAS_FILL_MODE_LOWER, n, 0.75, x, 1, A, n)


@pytest.mark.skipif(flag_blas.vendor_name != "hygon", reason="Hygon only")
def test_syr_root_api_uses_hygon_backend():
    for name in ("ssyr", "dsyr", "csyr", "zsyr"):
        assert getattr(flag_blas, name).__module__ == "_hygon.ops.syr"


SYR_BALANCED_SIZES = (1, 2, 7, 16, 17, 33, 127)
SYR_VARIANTS = [
    pytest.param("ssyr", torch.float32, 0.75, id="ssyr"),
    pytest.param("dsyr", torch.float64, 0.12345678901234568, id="dsyr"),
    pytest.param("csyr", torch.complex64, complex(0.5, -0.25), id="csyr"),
    pytest.param(
        "zsyr",
        torch.complex128,
        complex(0.12345678901234568, -0.2345678912345679),
        id="zsyr",
    ),
]


@pytest.mark.parametrize("name,dtype,alpha", SYR_VARIANTS)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.parametrize("n", SYR_BALANCED_SIZES)
@pytest.mark.parametrize("incx", [1, 2, 3])
@pytest.mark.parametrize("lda_pad", [0, 3])
def test_accuracy_syr_balanced(name, dtype, alpha, uplo, n, incx, lda_pad):
    check_fp64_support(dtype)
    lda = n + lda_pad
    x, A = _make_regression_inputs(n, incx, dtype, lda)
    x_before = x.clone()
    ref = _regression_reference(name, uplo, n, alpha, x, incx, A.clone(), lda)

    getattr(flag_blas, name)(uplo, n, alpha, x, incx, A, lda)

    blas_assert_close(A, ref, dtype, reduce_dim=1)
    if flag_blas.vendor_name == "ascend":
        torch.testing.assert_close(x.cpu(), x_before.cpu())
    else:
        torch.testing.assert_close(x, x_before)

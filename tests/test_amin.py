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

import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import (
    AMIN_SHAPES,
    L1_STRIDES,
    blas_assert_equal,
    to_cpu_blas_tensor,
    to_reference,
)
from .conftest import TO_CPU

IS_HYGON = flag_blas.vendor_name == "hygon"
if IS_HYGON:
    import ctypes

    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


def cpu_amin_reference(n, x, incx, result):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert result.numel() == 1, "result must be a single-element tensor"
    assert result.dtype == torch.int32, "result must be torch.int32"

    if n == 0:
        result.zero_()
        return

    ref_x = to_cpu_blas_tensor(x)
    ref_np = ref_x.numpy()
    dtype = ref_x.dtype
    if dtype == torch.float64:
        func = cpu_blas.dasum
    elif dtype == torch.complex128:
        func = cpu_blas.dzasum
    else:
        raise ValueError(f"Unsupported dtype for CPU BLAS: {dtype}")

    min_val = float("inf")
    min_idx = 0
    for idx in range(n):
        val = func(ref_np, n=1, offx=idx * incx, incx=1)
        if val < min_val:
            min_val = val
            min_idx = idx
    result.fill_(min_idx + 1)


def amin_reference(n, x, incx, result):
    if TO_CPU or (flag_blas.vendor_name == "thead" and x.is_complex()):
        # PPU complex AMIN is unavailable; validate against the CPU reference.
        if not TO_CPU:
            print(
                "[correctness] T-Head complex AMIN uses CPU reference; vendor interface unavailable"
            )
        ref_result = torch.zeros(result.shape, dtype=result.dtype, device="cpu")
        cpu_amin_reference(n, x, incx, ref_result)
        return ref_result if TO_CPU else ref_result.to(result.device)

    ref_x = to_reference(x)
    ref_result = to_reference(result).clone()
    cublas_amin_reference(n, ref_x, incx, ref_result)
    return ref_result


def call_amin(op_name, n, x, incx, result):
    if op_name == "samin":
        flag_blas.samin(n, x, incx, result)
    elif op_name == "damin":
        flag_blas.damin(n, x, incx, result)
    elif op_name == "camin":
        flag_blas.camin(n, x, incx, result)
    elif op_name == "zamin":
        flag_blas.zamin(n, x, incx, result)
    else:
        raise ValueError(f"Unsupported amin op: {op_name}")


if IS_HYGON:

    def _resolve_hipblas_amin(dtype):
        symbol = {
            torch.float32: "hipblasIsamin",
            torch.float64: "hipblasIdamin",
            torch.complex64: "hipblasIcamin_v2",
            torch.complex128: "hipblasIzamin_v2",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_amin_reference(n, x, incx, result):
        assert x.dim() == 1, "x must be 1-dimensional"
        assert result.numel() == 1, "result must be a single-element tensor"
        assert result.dtype == torch.int32, "result must be torch.int32"
        if n <= 0:
            result.zero_()
            return
        _, handle = get_l1_hipblas_context(x)
        function = _resolve_hipblas_amin(x.dtype)
        check_hipblas_status(
            function(handle, n, x.data_ptr(), incx, result.data_ptr()), "hipBLAS AMIN"
        )

    cublas_amin_reference = hipblas_amin_reference


else:

    def cublas_amin_reference(n, x, incx, result):
        assert x.dim() == 1, "x must be 1-dimensional"
        assert result.numel() == 1, "result must be a single-element tensor"
        assert result.dtype == torch.int32, "result must be torch.int32"

        if n == 0:
            result.zero_()
            return

        dtype = x.dtype
        if dtype == torch.float32:
            func = cublas.isamin
        elif dtype == torch.float64:
            func = cublas.idamin
        elif dtype == torch.complex64:
            func = cublas.icamin
        elif dtype == torch.complex128:
            func = cublas.izamin
        else:
            raise ValueError(f"Unsupported dtype for cuBLAS: {dtype}")

        handle = cp.cuda.device.get_cublas_handle()
        if flag_blas.vendor_name == "thead" and not x.is_complex():
            # PPU real AMIN requires host memory for its output index.
            host_result = torch.empty(1, dtype=torch.int32, device="cpu")
            previous_mode = cublas.getPointerMode(handle)
            try:
                cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
                func(handle, n, x.data_ptr(), incx, host_result.data_ptr())
            finally:
                cublas.setPointerMode(handle, previous_mode)
            result.copy_(host_result)
            return
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        func(handle, n, x.data_ptr(), incx, result.data_ptr())


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype,values,expected,func_name",
    [
        pytest.param(
            torch.float32, [3.0, -1.0, 2.0], 2, "dasum", marks=pytest.mark.samin
        ),
        pytest.param(
            torch.float64, [3.0, -1.0, 2.0], 2, "dasum", marks=pytest.mark.damin
        ),
        pytest.param(
            torch.complex64,
            [3.0 + 4.0j, 1.0 - 1.0j, 2.0 + 0.0j],
            2,
            "dzasum",
            marks=pytest.mark.camin,
        ),
        pytest.param(
            torch.complex128,
            [3.0 + 4.0j, 1.0 - 1.0j, 2.0 + 0.0j],
            2,
            "dzasum",
            marks=pytest.mark.zamin,
        ),
    ],
)
def test_cpu_amin_reference_uses_scipy_blas(
    monkeypatch, dtype, values, expected, func_name
):
    calls = []

    def forbidden_argmin(*args, **kwargs):
        raise RuntimeError("torch.argmin should not be used in CPU amin reference")

    def fake_asum(x, n=None, offx=0, incx=1):
        calls.append((n, offx, incx))
        val = x[offx]
        return float(abs(val.real) + abs(val.imag))

    monkeypatch.setattr(torch, "argmin", forbidden_argmin)
    monkeypatch.setattr(cpu_blas, func_name, fake_asum)

    x = torch.tensor(values, dtype=dtype)
    result = torch.zeros(1, dtype=torch.int32)

    cpu_amin_reference(3, x, 1, result)

    assert result.item() == expected
    assert calls == [(1, 0, 1), (1, 1, 1), (1, 2, 1)]


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.samin),
        pytest.param(torch.float64, marks=pytest.mark.damin),
    ],
)
@pytest.mark.parametrize("shape", AMIN_SHAPES)
@pytest.mark.parametrize("incx", L1_STRIDES)
def test_accuracy_amin_real(dtype, shape, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amin_reference(n, x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.samin(n, x, incx, result)
    else:
        flag_blas.damin(n, x, incx, result)

    blas_assert_equal(result, ref_result)


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.camin),
        pytest.param(torch.complex128, marks=pytest.mark.zamin),
    ],
)
@pytest.mark.parametrize("shape", AMIN_SHAPES)
@pytest.mark.parametrize("incx", L1_STRIDES)
def test_accuracy_amin_complex(dtype, shape, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)

    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amin_reference(n, x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.camin(n, x, incx, result)
    else:
        flag_blas.zamin(n, x, incx, result)

    blas_assert_equal(result, ref_result)


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.samin),
        pytest.param(torch.float64, marks=pytest.mark.damin),
        pytest.param(torch.complex64, marks=pytest.mark.camin),
        pytest.param(torch.complex128, marks=pytest.mark.zamin),
    ],
)
def test_accuracy_amin_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    n = 0

    if dtype == torch.float32:
        flag_blas.samin(n, x, 1, result)
    elif dtype == torch.float64:
        flag_blas.damin(n, x, 1, result)
    elif dtype == torch.complex64:
        flag_blas.camin(n, x, 1, result)
    else:
        flag_blas.zamin(n, x, 1, result)

    assert result.item() == 0
    assert result.dtype == torch.int32
    assert result.device == x.device


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.samin),
        pytest.param(torch.float64, marks=pytest.mark.damin),
    ],
)
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_amin_different_n_real(dtype, n, vec_size):
    """Test n smaller than allocated tensor length; verify only n elements considered."""
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amin_reference(n, x, 1, ref_result)

    if dtype == torch.float32:
        flag_blas.samin(n, x, 1, result)
    else:
        flag_blas.damin(n, x, 1, result)

    blas_assert_equal(result, ref_result)
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.camin),
        pytest.param(torch.complex128, marks=pytest.mark.zamin),
    ],
)
@pytest.mark.parametrize(
    "n,vec_size", [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)]
)
def test_accuracy_amin_different_n_complex(dtype, n, vec_size):
    """Test n smaller than allocated tensor length for complex types."""
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amin_reference(n, x, 1, ref_result)

    if dtype == torch.complex64:
        flag_blas.camin(n, x, 1, result)
    else:
        flag_blas.zamin(n, x, 1, result)

    blas_assert_equal(result, ref_result)
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.samin),
        pytest.param(torch.float64, marks=pytest.mark.damin),
    ],
)
@pytest.mark.parametrize(
    "n,vec_size,incx",
    [
        (5, 20, 2),
        (5, 20, 3),
        (10, 50, 2),
        (10, 100, 5),
    ],
)
def test_accuracy_amin_different_n_with_stride_real(dtype, n, vec_size, incx):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amin_reference(n, x, incx, ref_result)

    if dtype == torch.float32:
        flag_blas.samin(n, x, incx, result)
    else:
        flag_blas.damin(n, x, incx, result)

    blas_assert_equal(result, ref_result)
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.complex64, marks=pytest.mark.camin),
        pytest.param(torch.complex128, marks=pytest.mark.zamin),
    ],
)
@pytest.mark.parametrize(
    "n,vec_size,incx",
    [
        (5, 20, 2),
        (5, 20, 3),
        (10, 50, 2),
        (10, 100, 5),
    ],
)
def test_accuracy_amin_different_n_with_stride_complex(dtype, n, vec_size, incx):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)
    result = torch.zeros(1, dtype=torch.int32, device=flag_blas.device)

    ref_result = amin_reference(n, x, incx, ref_result)

    if dtype == torch.complex64:
        flag_blas.camin(n, x, incx, result)
    else:
        flag_blas.zamin(n, x, incx, result)

    blas_assert_equal(result, ref_result)
    if n > 0:
        assert 1 <= result.item() <= n, f"Index {result.item()} out of range [1, {n}]"

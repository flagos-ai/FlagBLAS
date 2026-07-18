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

import flag_blas

from .accuracy_utils import (
    ABS_SHAPES,
    blas_assert_close,
    blas_assert_equal,
    to_reference,
)


def torch_abs_reference(n, x, y):
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"

    if n <= 0:
        return

    ref_x = x[:n]
    if ref_x.device.type == "cpu":
        if ref_x.is_complex():
            ref_x = ref_x.to(torch.complex128)
        else:
            ref_x = ref_x.to(torch.float64)
    y[:n].copy_(torch.abs(ref_x).to(y.dtype))


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", ABS_SHAPES)
def test_accuracy_abs_real(dtype, shape):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    ref_x = to_reference(x)
    ref_y = torch.empty(n, dtype=dtype, device=ref_x.device)
    y = torch.empty(n, dtype=dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, ref_y)

    if dtype == torch.float32:
        flag_blas.ops.sabs(n, x, y)
    else:
        flag_blas.ops.dabs(n, x, y)

    blas_assert_equal(y[:n], ref_y[:n])


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", ABS_SHAPES)
def test_accuracy_abs_complex(dtype, shape):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    ref_x = to_reference(x)
    ref_y = torch.empty(n, dtype=result_dtype, device=ref_x.device)
    y = torch.empty(n, dtype=result_dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, ref_y)

    if dtype == torch.complex64:
        flag_blas.ops.cabs(n, x, y)
    else:
        flag_blas.ops.zabs(n, x, y)

    blas_assert_close(y[:n], ref_y[:n], result_dtype)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_accuracy_abs_empty_tensor(dtype):
    if (
        dtype in [torch.float64, torch.complex128]
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        y = torch.empty(0, dtype=torch.float32, device=flag_blas.device)
        flag_blas.ops.sabs(0, x, y)
    elif dtype == torch.float64:
        y = torch.empty(0, dtype=torch.float64, device=flag_blas.device)
        flag_blas.ops.dabs(0, x, y)
    elif dtype == torch.complex64:
        y = torch.empty(0, dtype=torch.float32, device=flag_blas.device)
        flag_blas.ops.cabs(0, x, y)
    else:
        y = torch.empty(0, dtype=torch.float64, device=flag_blas.device)
        flag_blas.ops.zabs(0, x, y)

    assert y.numel() == 0
    assert y.device == x.device


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,vec_size",
    [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)],
)
def test_accuracy_abs_different_n_real(dtype, n, vec_size):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)

    ref_x = to_reference(x)
    ref_y = torch.empty(vec_size, dtype=dtype, device=ref_x.device)
    y = torch.empty(vec_size, dtype=dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, ref_y)

    if dtype == torch.float32:
        flag_blas.ops.sabs(n, x, y)
    else:
        flag_blas.ops.dabs(n, x, y)

    blas_assert_equal(y[:n], ref_y[:n])


@pytest.mark.abs
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "n,vec_size",
    [(1, 10), (5, 10), (10, 10), (10, 20), (100, 1000)],
)
def test_accuracy_abs_different_n_complex(dtype, n, vec_size):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    result_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

    ref_x = to_reference(x)
    ref_y = torch.empty(vec_size, dtype=result_dtype, device=ref_x.device)
    y = torch.empty(vec_size, dtype=result_dtype, device=flag_blas.device)

    torch_abs_reference(n, ref_x, ref_y)

    if dtype == torch.complex64:
        flag_blas.ops.cabs(n, x, y)
    else:
        flag_blas.ops.zabs(n, x, y)

    blas_assert_close(y[:n], ref_y[:n], result_dtype)

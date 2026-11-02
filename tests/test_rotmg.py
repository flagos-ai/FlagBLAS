import ctypes
import ctypes.util
import glob
import os
import random

import pytest
import scipy
import torch

import flag_blas

from .accuracy_utils import blas_assert_close, to_reference
from .conftest import TO_CPU

IS_HYGON = flag_blas.vendor_name == "hygon"
if IS_HYGON:
    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library


ROTMG_CASES = [
    (-1.0, 2.0, 3.0, 4.0),
    (1.0, 0.0, 2.0, 3.0),
    (1.0, 2.0, 3.0, 0.0),
    (2.0, 1.0, 3.0, 1.0),
    (1.0, -2.0, 3.0, 4.0),
    (1.0, 2.0, 1.0, 1.0),
]
# Shared with the performance benchmark: retain ordinary, zero and sign cases.
ROTMG_CASES += [
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (1.0, -1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0, 1.0),
]
_rotmg_rng = random.Random(20261013)
for _ in range(24):
    ROTMG_CASES.append(tuple(_rotmg_rng.uniform(-10.0, 10.0) for _ in range(4)))

ROTMG_OPS = ["srotmg", "drotmg"]
CUBLAS_POINTER_MODE_DEVICE = 1


def load_cpu_blas():
    lib_paths = glob.glob(
        os.path.join(
            os.path.dirname(scipy.__file__),
            "..",
            "scipy.libs",
            "libscipy_openblas*.so",
        )
    )
    for path in lib_paths:
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError:
            continue
    raise RuntimeError("Cannot find SciPy OpenBLAS library")


def load_cublas():
    if flag_blas.vendor_name == "mthreads":
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
    raise RuntimeError("Cannot find libcublas.so in the system")


_cpu_blas_lib = load_cpu_blas()
_cublas = None
_cublas_handle = None


def _normalize_rotmg_param(param):
    param = list(param)
    flag = param[0]
    if flag == -2.0:
        param[1:] = [0.0, 0.0, 0.0, 0.0]
    elif flag == 0.0:
        param[1] = 0.0
        param[4] = 0.0
    elif flag == 1.0:
        param[2] = 0.0
        param[3] = 0.0
    return param


def _get_cublas():
    global _cublas
    if _cublas is not None:
        return _cublas

    _cublas = load_cublas()
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for rotmg_name in ("cublasSrotmg_v2", "cublasDrotmg_v2"):
        rotmg_func = getattr(_cublas, rotmg_name)
        rotmg_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        rotmg_func.restype = ctypes.c_int
    return _cublas


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is not None:
        return _cublas_handle

    cublas_lib = _get_cublas()
    _cublas_handle = ctypes.c_void_p()
    status = cublas_lib.cublasCreate_v2(ctypes.byref(_cublas_handle))
    if status != 0:
        raise RuntimeError(f"cuBLAS handle creation failed, error code: {status}")
    status = cublas_lib.cublasSetPointerMode_v2(
        _cublas_handle, CUBLAS_POINTER_MODE_DEVICE
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS pointer mode setup failed, error code: {status}")
    return _cublas_handle


def _scipy_rotmg_reference(dtype, d1, d2, x1, y1):
    c_type = ctypes.c_float if dtype == torch.float32 else ctypes.c_double
    d1_ref = c_type(d1)
    d2_ref = c_type(d2)
    x1_ref = c_type(x1)
    y1_ref = c_type(y1)
    param_ref = (c_type * 5)()
    func = (
        _cpu_blas_lib.scipy_srotmg_
        if dtype == torch.float32
        else _cpu_blas_lib.scipy_drotmg_
    )
    func(
        ctypes.byref(d1_ref),
        ctypes.byref(d2_ref),
        ctypes.byref(x1_ref),
        ctypes.byref(y1_ref),
        param_ref,
    )
    return (
        d1_ref.value,
        d2_ref.value,
        x1_ref.value,
        y1_ref.value,
        _normalize_rotmg_param(param_ref),
    )


if IS_HYGON:

    def _resolve_hipblas_rotmg(dtype):
        symbol = {
            torch.float32: "hipblasSrotmg",
            torch.float64: "hipblasDrotmg",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [ctypes.c_void_p] * 6
        function.restype = ctypes.c_int
        return function

    def _hipblas_rotmg_reference(dtype, d1_val, d2_val, x1_val, y1_val):
        d1 = torch.tensor([d1_val], dtype=dtype, device=flag_blas.device)
        d2 = torch.tensor([d2_val], dtype=dtype, device=flag_blas.device)
        x1 = torch.tensor([x1_val], dtype=dtype, device=flag_blas.device)
        y1 = torch.tensor([y1_val], dtype=dtype, device=flag_blas.device)
        param = torch.empty(5, dtype=dtype, device=flag_blas.device)
        _, handle = get_l1_hipblas_context(d1)
        function = _resolve_hipblas_rotmg(dtype)
        check_hipblas_status(
            function(
                handle,
                d1.data_ptr(),
                d2.data_ptr(),
                x1.data_ptr(),
                y1.data_ptr(),
                param.data_ptr(),
            ),
            "hipBLAS ROTMG",
        )
        return (
            d1.cpu()[0].item(),
            d2.cpu()[0].item(),
            x1.cpu()[0].item(),
            y1.cpu()[0].item(),
            _normalize_rotmg_param(param.cpu().tolist()),
        )

    _cublas_rotmg_reference = _hipblas_rotmg_reference

else:

    def _cublas_rotmg_reference(dtype, d1_val, d2_val, x1_val, y1_val):
        d1 = torch.tensor([d1_val], dtype=dtype, device=flag_blas.device)
        d2 = torch.tensor([d2_val], dtype=dtype, device=flag_blas.device)
        x1 = torch.tensor([x1_val], dtype=dtype, device=flag_blas.device)
        y1 = torch.tensor([y1_val], dtype=dtype, device=flag_blas.device)
        param = torch.empty(5, dtype=dtype, device=flag_blas.device)
        func = (
            _get_cublas().cublasSrotmg_v2
            if dtype == torch.float32
            else _get_cublas().cublasDrotmg_v2
        )
        status = func(
            _get_cublas_handle(),
            ctypes.c_void_p(d1.data_ptr()),
            ctypes.c_void_p(d2.data_ptr()),
            ctypes.c_void_p(x1.data_ptr()),
            ctypes.c_void_p(y1.data_ptr()),
            ctypes.c_void_p(param.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(f"cuBLAS rotmg execution failed, error code: {status}")
        return (
            d1.cpu()[0].item(),
            d2.cpu()[0].item(),
            x1.cpu()[0].item(),
            y1.cpu()[0].item(),
            _normalize_rotmg_param(param.cpu().tolist()),
        )


def _rotmg_reference(dtype, d1, d2, x1, y1):
    if TO_CPU:
        return _scipy_rotmg_reference(dtype, d1, d2, x1, y1)
    return _cublas_rotmg_reference(dtype, d1, d2, x1, y1)


def _reference_tensor(values, dtype):
    return to_reference(torch.tensor(values, dtype=dtype, device=flag_blas.device))


@pytest.mark.rotmg
@pytest.mark.parametrize(
    "op_name", [pytest.param(op, marks=getattr(pytest.mark, op)) for op in ROTMG_OPS]
)
def test_accuracy_rotmg_flag_blas_api_symbols(op_name):
    assert callable(getattr(flag_blas, op_name))


@pytest.mark.rotmg
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srotmg),
        pytest.param(torch.float64, marks=pytest.mark.drotmg),
    ],
)
@pytest.mark.parametrize("d1_val,d2_val,x1_val,y1_val", ROTMG_CASES)
def test_accuracy_rotmg(dtype, d1_val, d2_val, x1_val, y1_val):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.tensor([d1_val], dtype=dtype, device=flag_blas.device)
    d2 = torch.tensor([d2_val], dtype=dtype, device=flag_blas.device)
    x1 = torch.tensor([x1_val], dtype=dtype, device=flag_blas.device)
    y1 = torch.tensor([y1_val], dtype=dtype, device=flag_blas.device)
    param = torch.empty(5, dtype=dtype, device=flag_blas.device)

    ref_d1, ref_d2, ref_x1, ref_y1, ref_param = _rotmg_reference(
        dtype,
        float(torch.tensor(d1_val, dtype=dtype).item()),
        float(torch.tensor(d2_val, dtype=dtype).item()),
        float(torch.tensor(x1_val, dtype=dtype).item()),
        float(torch.tensor(y1_val, dtype=dtype).item()),
    )

    if dtype == torch.float32:
        flag_blas.srotmg(d1, d2, x1, y1, param)
    else:
        flag_blas.drotmg(d1, d2, x1, y1, param)

    blas_assert_close(d1, _reference_tensor([ref_d1], dtype), dtype)
    blas_assert_close(d2, _reference_tensor([ref_d2], dtype), dtype)
    blas_assert_close(x1, _reference_tensor([ref_x1], dtype), dtype)
    blas_assert_close(y1, _reference_tensor([ref_y1], dtype), dtype)
    blas_assert_close(param, _reference_tensor(ref_param, dtype), dtype)


@pytest.mark.rotmg
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srotmg),
        pytest.param(torch.float64, marks=pytest.mark.drotmg),
    ],
)
def test_rotmg_rejects_bad_shapes(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    d2 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    x1 = torch.randn(2, dtype=dtype, device=flag_blas.device)
    y1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    param = torch.empty(5, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError):
        if dtype == torch.float32:
            flag_blas.srotmg(d1, d2, x1, y1, param)
        else:
            flag_blas.drotmg(d1, d2, x1, y1, param)


@pytest.mark.rotmg
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srotmg),
        pytest.param(torch.float64, marks=pytest.mark.drotmg),
    ],
)
def test_accuracy_rotmg_flag_blas_api(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.tensor([1.0], dtype=dtype, device=flag_blas.device)
    d2 = torch.tensor([0.0], dtype=dtype, device=flag_blas.device)
    x1 = torch.tensor([2.0], dtype=dtype, device=flag_blas.device)
    y1 = torch.tensor([3.0], dtype=dtype, device=flag_blas.device)
    param = torch.empty(5, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        flag_blas.srotmg(d1, d2, x1, y1, param)
    else:
        flag_blas.drotmg(d1, d2, x1, y1, param)

    expected_param = _reference_tensor([-2.0, 0.0, 0.0, 0.0, 0.0], dtype)
    blas_assert_close(d1, _reference_tensor([1.0], dtype), dtype)
    blas_assert_close(d2, _reference_tensor([0.0], dtype), dtype)
    blas_assert_close(x1, _reference_tensor([2.0], dtype), dtype)
    blas_assert_close(param, expected_param, dtype)


@pytest.mark.rotmg
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srotmg),
        pytest.param(torch.float64, marks=pytest.mark.drotmg),
    ],
)
def test_rotmg_rejects_bad_param_shape(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    d2 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    x1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    y1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    param = torch.empty(4, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError, match="param must have exactly five elements"):
        if dtype == torch.float32:
            flag_blas.srotmg(d1, d2, x1, y1, param)
        else:
            flag_blas.drotmg(d1, d2, x1, y1, param)


@pytest.mark.srotmg
@pytest.mark.rotmg
def test_srotmg_rejects_mismatched_param_dtype():
    d1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    d2 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    x1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    y1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    param = torch.empty(5, dtype=torch.float64, device=flag_blas.device)

    with pytest.raises(AssertionError, match="param must be float32"):
        flag_blas.srotmg(d1, d2, x1, y1, param)


@pytest.mark.srotmg
@pytest.mark.rotmg
def test_srotmg_rejects_non_scalar_d1():
    d1 = torch.randn(2, dtype=torch.float32, device=flag_blas.device)
    d2 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    x1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    y1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    param = torch.empty(5, dtype=torch.float32, device=flag_blas.device)

    with pytest.raises(AssertionError, match="d1 must have exactly one element"):
        flag_blas.srotmg(d1, d2, x1, y1, param)

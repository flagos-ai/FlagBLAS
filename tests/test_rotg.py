import ctypes
import ctypes.util
import glob
import math
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


ROTG_REAL_CASES = [
    (3.0, 4.0),
    (4.0, 3.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 0.0),
    (-3.0, 4.0),
    (3.0, -4.0),
    (-3.0, -4.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (1e-8, 1e-8),
    (1e-20, 1.0),
    (1.0, 1e-20),
    (1e8, 1e8),
    (1e20, -1e20),
]

ROTG_COMPLEX_CASES = [
    (1.0 + 2.0j, 3.0 + 4.0j),
    (0.0 + 0.0j, 1.0 + 0.0j),
    (0.0 + 0.0j, 0.0 + 1.0j),
    (1.0 + 0.0j, 0.0 + 0.0j),
    (0.0 + 1.0j, 0.0 + 0.0j),
    (0.0 + 0.0j, 0.0 + 0.0j),
    (-2.0 + 3.0j, 4.0 - 5.0j),
    (-2.0 - 3.0j, -4.0 + 5.0j),
    (3.0 + 0.0j, 0.0 + 4.0j),
    (0.0 + 3.0j, 4.0 + 0.0j),
    (1.0 - 1.0j, -1.0 + 1.0j),
    (1e-8 + 1e-8j, 1e-8 - 1e-8j),
    (1e-20 + 0.0j, 1.0 + 0.0j),
    (1.0 + 0.0j, 1e-20 + 0.0j),
    (1e8 + 0.0j, 0.0 + 1e8j),
    (1e20 + 1e20j, -1e20 + 1e20j),
]

ROTG_REAL_CASES += [
    (1e-30, 1e-30),
    (1e30, -1e30),
    (2.0**-80, 2.0**-79),
    (2.0**80, -(2.0**79)),
    (0.0, 1e30),
    (-1e30, 0.0),
    (1e30, 1e-30),
    (1e-30, -1e30),
]
ROTG_COMPLEX_CASES += [
    (1e-30 + 1e-30j, -1e-30 + 1e-30j),
    (1e30 + 1e30j, -1e30 + 1e30j),
    (1.0 + 0.0j, -1.0 + 0.0j),
    (0.0 - 1.0j, 0.0 + 1.0j),
    (1.0 + 0.0j, 0.0 - 1.0j),
    (-1.0 + 0.0j, 0.0 + 1.0j),
    (1e30 + 1e-30j, 1e-30 - 1e30j),
    (1e-30 + 1e-30j, 1e30 - 1e30j),
]
_rotg_rng = random.Random(20261013)
for _ in range(32):
    ROTG_REAL_CASES.append(tuple(_rotg_rng.uniform(-10.0, 10.0) for _ in range(2)))
    _v = [_rotg_rng.uniform(-10.0, 10.0) for _ in range(4)]
    ROTG_COMPLEX_CASES.append((complex(*_v[:2]), complex(*_v[2:])))


# Keep CPU and device references on the same non-divergent complex cases.
ROTG_COMPLEX_CASES_BY_DTYPE = {
    dtype: [
        (a, b)
        for a, b in ROTG_COMPLEX_CASES
        if not (a == 0.0j and (b == 0.0j or b.imag != 0.0))
        and not (dtype == torch.complex64 and (abs(a) > 1e19 or abs(b) > 1e19))
        and not (
            dtype == torch.complex64 and (a, b) == (1e-30 + 1e-30j, -1e-30 + 1e-30j)
        )
    ]
    for dtype in (torch.complex64, torch.complex128)
}

ROTG_OPS = ["srotg", "drotg", "crotg", "zrotg"]
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


_cpu_blas_lib = load_cpu_blas()
_cublas = None
_cublas_handle = None


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def load_cublas():
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


def _get_cublas():
    global _cublas
    if _cublas is not None:
        return _cublas

    _cublas = load_cublas()
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for rotg_name in (
        "cublasSrotg_v2",
        "cublasDrotg_v2",
        "cublasCrotg_v2",
        "cublasZrotg_v2",
    ):
        rotg_func = getattr(_cublas, rotg_name)
        rotg_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        rotg_func.restype = ctypes.c_int
    return _cublas


def _get_cublas_handle():
    global _cublas_handle
    cublas_lib = _get_cublas()
    if _cublas_handle is not None:
        return _cublas_handle

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


def _call_rotg(dtype, a, b, c, s):
    if dtype == torch.float32:
        flag_blas.srotg(a, b, c, s)
    elif dtype == torch.float64:
        flag_blas.drotg(a, b, c, s)
    elif dtype == torch.complex64:
        flag_blas.crotg(a, b, c, s)
    else:
        flag_blas.zrotg(a, b, c, s)


def _rotg_real_dtype(dtype):
    return torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64


def _scipy_rotg_reference(dtype, a, b):
    if dtype == torch.float32:
        a_ref = ctypes.c_float(a)
        b_ref = ctypes.c_float(b)
        c_ref = ctypes.c_float(0.0)
        s_ref = ctypes.c_float(0.0)
        _cpu_blas_lib.scipy_srotg_(
            ctypes.byref(a_ref),
            ctypes.byref(b_ref),
            ctypes.byref(c_ref),
            ctypes.byref(s_ref),
        )
        return a_ref.value, b_ref.value, c_ref.value, s_ref.value

    if dtype == torch.float64:
        a_ref = ctypes.c_double(a)
        b_ref = ctypes.c_double(b)
        c_ref = ctypes.c_double(0.0)
        s_ref = ctypes.c_double(0.0)
        _cpu_blas_lib.scipy_drotg_(
            ctypes.byref(a_ref),
            ctypes.byref(b_ref),
            ctypes.byref(c_ref),
            ctypes.byref(s_ref),
        )
        return a_ref.value, b_ref.value, c_ref.value, s_ref.value

    if dtype == torch.complex64:
        a_ref = cuComplex(a.real, a.imag)
        b_ref = cuComplex(b.real, b.imag)
        c_ref = ctypes.c_float(0.0)
        s_ref = cuComplex(0.0, 0.0)
        _cpu_blas_lib.scipy_crotg_(
            ctypes.byref(a_ref),
            ctypes.byref(b_ref),
            ctypes.byref(c_ref),
            ctypes.byref(s_ref),
        )
        return (
            complex(a_ref.x, a_ref.y),
            complex(b_ref.x, b_ref.y),
            c_ref.value,
            complex(s_ref.x, s_ref.y),
        )

    a_ref = cuDoubleComplex(a.real, a.imag)
    b_ref = cuDoubleComplex(b.real, b.imag)
    c_ref = ctypes.c_double(0.0)
    s_ref = cuDoubleComplex(0.0, 0.0)
    _cpu_blas_lib.scipy_zrotg_(
        ctypes.byref(a_ref),
        ctypes.byref(b_ref),
        ctypes.byref(c_ref),
        ctypes.byref(s_ref),
    )
    return (
        complex(a_ref.x, a_ref.y),
        complex(b_ref.x, b_ref.y),
        c_ref.value,
        complex(s_ref.x, s_ref.y),
    )


if IS_HYGON:

    def _resolve_hipblas_rotg(dtype):
        symbol = {
            torch.float32: "hipblasSrotg",
            torch.float64: "hipblasDrotg",
            torch.complex64: "hipblasCrotg_v2",
            torch.complex128: "hipblasZrotg_v2",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [ctypes.c_void_p] * 5
        function.restype = ctypes.c_int
        return function

    def _hipblas_rotg_reference(dtype, a_val, b_val):
        a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
        b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
        c = torch.zeros(1, dtype=_rotg_real_dtype(dtype), device=flag_blas.device)
        s = torch.zeros(1, dtype=dtype, device=flag_blas.device)
        _, handle = get_l1_hipblas_context(a)
        function = _resolve_hipblas_rotg(dtype)
        check_hipblas_status(
            function(handle, a.data_ptr(), b.data_ptr(), c.data_ptr(), s.data_ptr()),
            "hipBLAS ROTG",
        )
        return (
            a.cpu()[0].item(),
            b.cpu()[0].item(),
            c.cpu()[0].item(),
            s.cpu()[0].item(),
        )

    _cublas_rotg_reference = _hipblas_rotg_reference

else:

    def _cublas_rotg_reference(dtype, a_val, b_val):
        real_dtype = _rotg_real_dtype(dtype)
        a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
        b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
        c = torch.zeros(1, dtype=real_dtype, device=flag_blas.device)
        s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

        if dtype == torch.float32:
            func = _get_cublas().cublasSrotg_v2
        elif dtype == torch.float64:
            func = _get_cublas().cublasDrotg_v2
        elif dtype == torch.complex64:
            func = _get_cublas().cublasCrotg_v2
        else:
            func = _get_cublas().cublasZrotg_v2

        status = func(
            _get_cublas_handle(),
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_void_p(s.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(f"cuBLAS rotg execution failed, error code: {status}")
        return (
            a.cpu()[0].item(),
            b.cpu()[0].item(),
            c.cpu()[0].item(),
            s.cpu()[0].item(),
        )


def _rotg_reference(dtype, a, b):
    if TO_CPU:
        if dtype.is_complex:
            # CPU math uses the cuBLAS/hipBLAS ROTG output convention. Newer
            # OpenBLAS chooses a different phase when a == 0, and its CROTG can
            # overflow for finite large inputs. No device reference is used here.
            a = complex(torch.tensor(a, dtype=dtype).item())
            b = complex(torch.tensor(b, dtype=dtype).item())
            abs_a = math.hypot(a.real, a.imag)
            if abs_a == 0.0:
                return b, b, 0.0, 1.0 + 0.0j
            norm = math.hypot(abs_a, math.hypot(b.real, b.imag))
            phase = a / abs_a
            return phase * norm, b, abs_a / norm, phase * b.conjugate() / norm
        r, z, c, s = _scipy_rotg_reference(dtype, a, b)
        # cuBLAS/hipBLAS encode z as 1 when c rounds to zero. OpenBLAS
        # may return infinity from 1/c; r, c and s are still checked normally.
        if c == 0.0 and r != 0.0:
            z = 1.0
        return r, z, c, s
    return _cublas_rotg_reference(dtype, a, b)


def _reference_tensor(values, dtype):
    return to_reference(torch.tensor(values, dtype=dtype, device=flag_blas.device))


@pytest.mark.rotg
@pytest.mark.parametrize(
    "op_name", [pytest.param(op, marks=getattr(pytest.mark, op)) for op in ROTG_OPS]
)
def test_accuracy_rotg_flag_blas_api_symbols(op_name):
    assert callable(getattr(flag_blas, op_name))


@pytest.mark.rotg
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srotg),
        pytest.param(torch.float64, marks=pytest.mark.drotg),
    ],
)
@pytest.mark.parametrize("a_val,b_val", ROTG_REAL_CASES)
def test_accuracy_rotg_real(dtype, a_val, b_val):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
    b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_a, ref_b, ref_c, ref_s = _rotg_reference(dtype, a_val, b_val)
    if dtype == torch.float32:
        flag_blas.srotg(a, b, c, s)
    else:
        flag_blas.drotg(a, b, c, s)

    expected = _reference_tensor([ref_a, ref_b, ref_c, ref_s], dtype)
    actual = torch.stack((a[0], b[0], c[0], s[0]))
    blas_assert_close(actual, expected, dtype)


@pytest.mark.srotg
@pytest.mark.rotg
def test_accuracy_rotg_flag_blas_api():
    a = torch.tensor([3.0], dtype=torch.float32, device=flag_blas.device)
    b = torch.tensor([4.0], dtype=torch.float32, device=flag_blas.device)
    c = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)
    s = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)

    flag_blas.srotg(a, b, c, s)

    blas_assert_close(a, _reference_tensor([5.0], torch.float32), torch.float32)
    blas_assert_close(b, _reference_tensor([5.0 / 3.0], torch.float32), torch.float32)
    blas_assert_close(c, _reference_tensor([0.6], torch.float32), torch.float32)
    blas_assert_close(s, _reference_tensor([0.8], torch.float32), torch.float32)


@pytest.mark.rotg
@pytest.mark.parametrize(
    "dtype,a_val,b_val",
    [
        pytest.param(dtype, a, b, marks=getattr(pytest.mark, op))
        for dtype, op in ((torch.complex64, "crotg"), (torch.complex128, "zrotg"))
        for a, b in ROTG_COMPLEX_CASES_BY_DTYPE[dtype]
    ],
)
def test_accuracy_rotg_complex(dtype, a_val, b_val):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
    b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=real_dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_a, ref_b, ref_c, ref_s = _rotg_reference(dtype, a_val, b_val)
    if dtype == torch.complex64:
        flag_blas.crotg(a, b, c, s)
    else:
        flag_blas.zrotg(a, b, c, s)

    blas_assert_close(a, _reference_tensor([ref_a], dtype), dtype)
    blas_assert_close(b, _reference_tensor([ref_b], dtype), dtype)
    blas_assert_close(c, _reference_tensor([ref_c], real_dtype), real_dtype)
    blas_assert_close(s, _reference_tensor([ref_s], dtype), dtype)


@pytest.mark.rotg
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.srotg),
        pytest.param(torch.float64, marks=pytest.mark.drotg),
        pytest.param(torch.complex64, marks=pytest.mark.crotg),
        pytest.param(torch.complex128, marks=pytest.mark.zrotg),
    ],
)
def test_rotg_rejects_non_scalar_inputs(dtype):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    real_dtype = _rotg_real_dtype(dtype)
    a = torch.ones(2, dtype=dtype, device=flag_blas.device)
    b = torch.ones(1, dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=real_dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError, match="a must have exactly one element"):
        _call_rotg(dtype, a, b, c, s)


@pytest.mark.srotg
@pytest.mark.rotg
def test_srotg_rejects_mismatched_input_dtype():
    a = torch.ones(1, dtype=torch.float32, device=flag_blas.device)
    b = torch.ones(1, dtype=torch.float64, device=flag_blas.device)
    c = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)
    s = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)

    with pytest.raises(AssertionError, match="b must be float32"):
        flag_blas.srotg(a, b, c, s)


@pytest.mark.crotg
@pytest.mark.rotg
def test_crotg_rejects_mismatched_c_dtype():
    a = torch.ones(1, dtype=torch.complex64, device=flag_blas.device)
    b = torch.ones(1, dtype=torch.complex64, device=flag_blas.device)
    c = torch.zeros(1, dtype=torch.float64, device=flag_blas.device)
    s = torch.zeros(1, dtype=torch.complex64, device=flag_blas.device)

    with pytest.raises(AssertionError, match="c must be float32"):
        flag_blas.crotg(a, b, c, s)

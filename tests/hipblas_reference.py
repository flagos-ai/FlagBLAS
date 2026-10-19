import atexit
import ctypes
import ctypes.util
from threading import RLock

import torch

HIPBLAS_POINTER_MODE_HOST = 0

# DTK hipBLAS follows the rocBLAS convention for the operation enum
# (HIPBLAS_OP_N/T/C = 111/112/113), while FlagBLAS APIs use cuBLAS-style
# enums (CUBLAS_OP_N/T/C = 0/1/2). Map between them when calling hipBLAS.
HIPBLAS_OP_N = 111
HIPBLAS_OP_T = 112
HIPBLAS_OP_C = 113


def to_hipblas_op(op):
    """Map a cuBLAS-style operation enum (CUBLAS_OP_N/T/C) to hipBLAS."""
    return {0: HIPBLAS_OP_N, 1: HIPBLAS_OP_T, 2: HIPBLAS_OP_C}[op]


class HipComplex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class HipDoubleComplex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


_LIBRARY = None
_HANDLES = {}
_LOCK = RLock()


def check_hipblas_status(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed with hipBLAS status {status}")


def _configure_library(library):
    library.hipblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.hipblasCreate.restype = ctypes.c_int
    library.hipblasDestroy.argtypes = [ctypes.c_void_p]
    library.hipblasDestroy.restype = ctypes.c_int
    library.hipblasSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.hipblasSetStream.restype = ctypes.c_int
    library.hipblasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.hipblasSetPointerMode.restype = ctypes.c_int
    # GEMM entry points share the same argument layout
    # (handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    # only the scalar type pointed by alpha/beta differs (fp32 / fp64).
    _gemm_argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,  # transa
        ctypes.c_int,  # transb
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_int,  # k
        ctypes.c_void_p,  # alpha
        ctypes.c_void_p,  # A
        ctypes.c_int,  # lda
        ctypes.c_void_p,  # B
        ctypes.c_int,  # ldb
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # C
        ctypes.c_int,  # ldc
    ]
    for symbol in (
        "hipblasSgemm",
        "hipblasDgemm",
        "hipblasCgemm",
        "hipblasZgemm",
    ):
        func = getattr(library, symbol)
        func.argtypes = _gemm_argtypes
        func.restype = ctypes.c_int


def get_hipblas_library():
    global _LIBRARY
    if _LIBRARY is None:
        with _LOCK:
            if _LIBRARY is None:
                library_name = ctypes.util.find_library("hipblas")
                if library_name is None:
                    raise RuntimeError("Unable to find the hipBLAS shared library")
                library = ctypes.CDLL(library_name)
                _configure_library(library)
                _LIBRARY = library
    return _LIBRARY


def get_hipblas_handle(device_index):
    handle = _HANDLES.get(device_index)
    if handle is not None:
        return handle

    with _LOCK:
        handle = _HANDLES.get(device_index)
        if handle is None:
            library = get_hipblas_library()
            with torch.cuda.device(device_index):
                handle = ctypes.c_void_p()
                check_hipblas_status(
                    library.hipblasCreate(ctypes.byref(handle)), "hipblasCreate"
                )
                check_hipblas_status(
                    library.hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST),
                    "hipblasSetPointerMode",
                )
            _HANDLES[device_index] = handle
    return handle


def _destroy_handles():
    if _LIBRARY is None:
        return
    for handle in tuple(_HANDLES.values()):
        try:
            _LIBRARY.hipblasDestroy(handle)
        except Exception:
            pass
    _HANDLES.clear()


atexit.register(_destroy_handles)


def get_hipblas_context(tensor):
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    library = get_hipblas_library()
    handle = get_hipblas_handle(device_index)
    stream = torch.cuda.current_stream(tensor.device).cuda_stream
    check_hipblas_status(
        library.hipblasSetStream(handle, ctypes.c_void_p(stream)),
        "hipblasSetStream",
    )
    return library, handle


def hipblas_dgemm(
    handle,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A,
    lda,
    B,
    ldb,
    beta,
    C,
    ldc,
):
    """fp64 GEMM reference via hipBLAS (pointer mode: host scalars).

    ``transa``/``transb`` are cuBLAS-style enums (0/1/2); the mapping to the
    hipBLAS op enum (111/112/113) happens here.
    """
    library = get_hipblas_library()
    alpha_c = ctypes.c_double(alpha)
    beta_c = ctypes.c_double(beta)
    check_hipblas_status(
        library.hipblasDgemm(
            handle,
            to_hipblas_op(transa),
            to_hipblas_op(transb),
            m,
            n,
            k,
            ctypes.byref(alpha_c),
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(B.data_ptr()),
            ldb,
            ctypes.byref(beta_c),
            ctypes.c_void_p(C.data_ptr()),
            ldc,
        ),
        "hipblasDgemm",
    )
    return C


def hipblas_cgemm(
    handle,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A,
    lda,
    B,
    ldb,
    beta,
    C,
    ldc,
):
    """complex64 GEMM reference via hipBLAS (pointer mode: host scalars).

    ``alpha``/``beta`` are Python complex numbers, packed into the
    ``hipblasComplex`` host struct. ``transa``/``transb`` are cuBLAS-style
    enums (0/1/2); the mapping to the hipBLAS op enum happens here.
    """
    library = get_hipblas_library()
    alpha_c = HipComplex(float(alpha.real), float(alpha.imag))
    beta_c = HipComplex(float(beta.real), float(beta.imag))
    check_hipblas_status(
        library.hipblasCgemm(
            handle,
            to_hipblas_op(transa),
            to_hipblas_op(transb),
            m,
            n,
            k,
            ctypes.byref(alpha_c),
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(B.data_ptr()),
            ldb,
            ctypes.byref(beta_c),
            ctypes.c_void_p(C.data_ptr()),
            ldc,
        ),
        "hipblasCgemm",
    )
    return C


def hipblas_zgemm(
    handle,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A,
    lda,
    B,
    ldb,
    beta,
    C,
    ldc,
):
    """complex128 GEMM reference via hipBLAS (pointer mode: host scalars).

    ``alpha``/``beta`` are Python complex numbers, packed into the
    ``hipblasDoubleComplex`` host struct. ``transa``/``transb`` are
    cuBLAS-style enums (0/1/2); the mapping to the hipBLAS op enum happens here.
    """
    library = get_hipblas_library()
    alpha_c = HipDoubleComplex(float(alpha.real), float(alpha.imag))
    beta_c = HipDoubleComplex(float(beta.real), float(beta.imag))
    check_hipblas_status(
        library.hipblasZgemm(
            handle,
            to_hipblas_op(transa),
            to_hipblas_op(transb),
            m,
            n,
            k,
            ctypes.byref(alpha_c),
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(B.data_ptr()),
            ldb,
            ctypes.byref(beta_c),
            ctypes.c_void_p(C.data_ptr()),
            ldc,
        ),
        "hipblasZgemm",
    )
    return C

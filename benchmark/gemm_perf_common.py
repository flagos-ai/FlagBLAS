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

import itertools
import struct
from typing import Generator, List, Tuple

import numpy as np
import torch

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.utils import shape_utils

CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14

# T-Head (Zhenwu) cuBLAS rejects a legacy cudaDataType passed where
# cublasGemmEx expects a cublasComputeType_t (CUBLAS_STATUS_NOT_SUPPORTED).
# Use the proper cublasComputeType_t / cublasGemmAlgo_t enums there; other
# vendors keep the original arguments.
if flag_blas.vendor_name == "thead":
    from cupy_backends.cuda.libs import cublas

    GEMM_COMPUTE_TYPE = cublas.CUBLAS_COMPUTE_32F
    GEMM_ALGO = cublas.CUBLAS_GEMM_DEFAULT
else:
    GEMM_COMPUTE_TYPE = CUDA_R_32F
    GEMM_ALGO = 0

HIPBLAS_POINTER_MODE_HOST = 0

# DTK hipBLAS follows the rocBLAS convention for the operation enum
# (HIPBLAS_OP_N/T/C = 111/112/113), while FlagBLAS APIs use cuBLAS-style
# enums (CUBLAS_OP_N/T/C = 0/1/2). Map between them when calling hipBLAS.
_HIPBLAS_OP_MAP = {0: 111, 1: 112, 2: 113}

IS_HYGON = flag_blas.vendor_name == "hygon"

if IS_HYGON:
    import atexit
    import ctypes
    import ctypes.util

    _HIPBLAS_LIBRARY = None
    _HIPBLAS_HANDLES = {}

    class _HipComplex(ctypes.Structure):
        # hipblasComplex (host scalar for hipblasCgemm's alpha/beta)
        _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

    class _HipDoubleComplex(ctypes.Structure):
        # hipblasDoubleComplex (host scalar for hipblasZgemm's alpha/beta)
        _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

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
            library.hipblasSgemm.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
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
            library.hipblasSgemm.restype = ctypes.c_int
            # hipblasDgemm (fp64 in/out) shares the same signature as
            # hipblasSgemm; only the scalar type pointed by alpha/beta
            # differs (fp64 instead of fp32).
            library.hipblasDgemm.argtypes = library.hipblasSgemm.argtypes
            library.hipblasDgemm.restype = ctypes.c_int
            # hipblasHgemm (fp16 in/out, fp32 accumulate) shares the same
            # (handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
            # beta, C, ldc) signature as hipblasSgemm.
            library.hipblasHgemm.argtypes = library.hipblasSgemm.argtypes
            library.hipblasHgemm.restype = ctypes.c_int
            # hipblasCgemm (complex64 in/out) shares the same (handle, transa,
            # transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) signature as
            # hipblasSgemm; alpha/beta point to hipblasComplex host structs.
            library.hipblasCgemm.argtypes = library.hipblasSgemm.argtypes
            library.hipblasCgemm.restype = ctypes.c_int
            # hipblasZgemm (complex128 in/out) shares the same signature; alpha/
            # beta point to hipblasDoubleComplex host structs.
            library.hipblasZgemm.argtypes = library.hipblasSgemm.argtypes
            library.hipblasZgemm.restype = ctypes.c_int
            # hipblasGemmEx_v2: like cuBLAS gemmEx, alpha/beta are scalars in
            # the compute type and the A/B/C types are selected via hipDataType.
            library.hipblasGemmEx_v2.argtypes = [
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
            library.hipblasGemmEx_v2.restype = ctypes.c_int
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
                    library.hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST),
                    "hipblasSetPointerMode",
                )
            _HIPBLAS_HANDLES[device_index] = handle
        stream = torch.cuda.current_stream(device).cuda_stream
        _check_hipblas_status(
            library.hipblasSetStream(handle, ctypes.c_void_p(stream)),
            "hipblasSetStream",
        )
        return library, handle

    def _hipblas_sgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A,
        lda,
        B,
        ldb,
        beta_ptr,
        C,
        ldc,
    ):
        library = _load_hipblas()
        _check_hipblas_status(
            library.hipblasSgemm(
                handle,
                _HIPBLAS_OP_MAP[transa],
                _HIPBLAS_OP_MAP[transb],
                m,
                n,
                k,
                alpha_ptr,
                ctypes.c_void_p(A.data_ptr()),
                lda,
                ctypes.c_void_p(B.data_ptr()),
                ldb,
                beta_ptr,
                ctypes.c_void_p(C.data_ptr()),
                ldc,
            ),
            "hipblasSgemm",
        )
        return C

    def _hipblas_dgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A,
        lda,
        B,
        ldb,
        beta_ptr,
        C,
        ldc,
    ):
        library = _load_hipblas()
        _check_hipblas_status(
            library.hipblasDgemm(
                handle,
                _HIPBLAS_OP_MAP[transa],
                _HIPBLAS_OP_MAP[transb],
                m,
                n,
                k,
                alpha_ptr,
                ctypes.c_void_p(A.data_ptr()),
                lda,
                ctypes.c_void_p(B.data_ptr()),
                ldb,
                beta_ptr,
                ctypes.c_void_p(C.data_ptr()),
                ldc,
            ),
            "hipblasDgemm",
        )
        return C

    def _hipblas_hgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A,
        lda,
        B,
        ldb,
        beta_ptr,
        C,
        ldc,
    ):
        library = _load_hipblas()
        _check_hipblas_status(
            library.hipblasHgemm(
                handle,
                _HIPBLAS_OP_MAP[transa],
                _HIPBLAS_OP_MAP[transb],
                m,
                n,
                k,
                alpha_ptr,
                ctypes.c_void_p(A.data_ptr()),
                lda,
                ctypes.c_void_p(B.data_ptr()),
                ldb,
                beta_ptr,
                ctypes.c_void_p(C.data_ptr()),
                ldc,
            ),
            "hipblasHgemm",
        )
        return C

    def _hipblas_gemm_ex_v2(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A,
        a_type,
        lda,
        B,
        b_type,
        ldb,
        beta_ptr,
        C,
        c_type,
        ldc,
        compute_type,
        algo=0,
    ):
        library = _load_hipblas()
        _check_hipblas_status(
            library.hipblasGemmEx_v2(
                handle,
                _HIPBLAS_OP_MAP[transa],
                _HIPBLAS_OP_MAP[transb],
                m,
                n,
                k,
                alpha_ptr,
                ctypes.c_void_p(A.data_ptr()),
                a_type,
                lda,
                ctypes.c_void_p(B.data_ptr()),
                b_type,
                ldb,
                beta_ptr,
                ctypes.c_void_p(C.data_ptr()),
                c_type,
                ldc,
                compute_type,
                algo,
            ),
            "hipblasGemmEx_v2",
        )
        return C

    def _destroy_hipblas_handles():
        if _HIPBLAS_LIBRARY is None:
            return
        for handle in tuple(_HIPBLAS_HANDLES.values()):
            try:
                _HIPBLAS_LIBRARY.hipblasDestroy(handle)
            except Exception:
                pass
        _HIPBLAS_HANDLES.clear()

    atexit.register(_destroy_hipblas_handles)
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


GEMM_SHAPES = [
    (511, 511, 511),
    (1023, 1023, 1023),
    (2048, 12288, 4096),
    (2048, 11008, 4096),
    (2048, 4096, 11008),
    (4096, 24576, 8192),
    (4096, 8192, 28672),
    (8192, 28672, 8192),
    (16384, 2048, 2048),
    (2048, 16384, 2048),
    (2048, 2048, 16384),
    (32768, 1024, 1024),
    (4095, 4095, 4095),
    (8191, 8191, 8191),
    (4097, 8191, 4095),
]


def model_shapes() -> List[Tuple[int, int, int]]:
    """
    Generate shapes with m ranging from 1 to 32 (step 1), then 64, 128, 256, 512, 1024, 2048, 4096.
    """
    m_values = list(range(1, 33)) + [64, 128, 256, 512, 1024, 2048, 4096]

    NK = [
        [6144, 4096],
        [4096, 4096],
        [24576, 4096],
        [4096, 12288],
        [5120, 5120],
        [5120, 4096],
        [25600, 5120],
        [5120, 12800],
        [2560, 5120],
        [5120, 2048],
        [12800, 5210],
        [5120, 6400],
        [5120, 2048],
        [2048, 4096],
        [2560, 2048],
        [2048, 1024],
        [1152, 4096],
        [4096, 1024],
        [4096, 7168],
        [7168, 2048],
        [2304, 2048],
        [1152, 2048],
        [2048, 1024],
        [2048, 512],
        [3072, 2048],
        [1536, 2048],
    ]

    return [(m, n, k) for m, (n, k) in itertools.product(m_values, NK)]


def cublas_sgemm(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    if IS_HYGON:
        _hipblas_sgemm(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A_col,
            lda_cublas,
            B_col,
            ldb_cublas,
            beta_ptr,
            C_col,
            ldc_cublas,
        )
    else:
        cublas.sgemm(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A_col.data_ptr(),
            lda_cublas,
            B_col.data_ptr(),
            ldb_cublas,
            beta_ptr,
            C_col.data_ptr(),
            ldc_cublas,
        )
    return C_col


def gems_sgemm_wrapper(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.sgemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        lda_flag,
        B_row,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )
    return C_row


def cublas_hgemm(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    if IS_HYGON:
        # hipblasHgemm is the only DTK hipBLAS entry point that accepts fp16
        # arrays (hipblasGemmEx/_v2 reject fp16 on this backend). Its alpha and
        # beta scalars are fp16; get_input_iter packs them accordingly. Note
        # this entry point is used only as the native latency baseline: it does
        # not accumulate in fp32, so accuracy is checked against
        # hgemm_reference() instead.
        return _hipblas_hgemm(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A_col,
            lda_cublas,
            B_col,
            ldb_cublas,
            beta_ptr,
            C_col,
            ldc_cublas,
        )
    cublas.gemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        CUDA_R_16F,
        lda_cublas,
        B_col.data_ptr(),
        CUDA_R_16F,
        ldb_cublas,
        beta_ptr,
        C_col.data_ptr(),
        CUDA_R_16F,
        ldc_cublas,
        GEMM_COMPUTE_TYPE,
        GEMM_ALGO,
    )
    return C_col


def cublas_dgemm(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    if IS_HYGON:
        # hipblasDgemm reads the alpha/beta host scalars as fp64; the
        # get_input_iter hygon branch packs them into c_double when
        # alpha_dtype is np.float64.
        return _hipblas_dgemm(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A_col,
            lda_cublas,
            B_col,
            ldb_cublas,
            beta_ptr,
            C_col,
            ldc_cublas,
        )
    cublas.dgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        lda_cublas,
        B_col.data_ptr(),
        ldb_cublas,
        beta_ptr,
        C_col.data_ptr(),
        ldc_cublas,
    )
    return C_col


def gems_hgemm_wrapper(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.hgemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        lda_flag,
        B_row,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )
    return C_row


def hgemm_reference(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    """fp32-accumulation accuracy reference for fp16 GEMM.

    FlagBLAS hgemm kernels accumulate in fp32 (same semantics as cuBLAS
    gemmEx with CUDA_R_16F inputs / CUDA_R_32F compute, used on NVIDIA below).
    On Hygon/DTK the only hipBLAS fp16 entry point, hipblasHgemm, accumulates in
    reduced precision and deviates up to ~10% relative to an fp32 accumulator on
    large K, so it cannot act as the accuracy oracle. Instead the fp16 operands
    (exactly representable in fp32) are multiplied on-device in fp32.
    """
    if IS_HYGON:
        A = A_row.float()
        B = B_row.float()
        if transa == CUBLAS_OP_T:
            A = A.t()
        if transb == CUBLAS_OP_T:
            B = B.t()
        acc = alpha * torch.matmul(A, B)
        if beta != 0:
            acc += beta * C_row.float()
        return acc.to(torch.float16)
    return cublas_hgemm(
        A_col,
        B_col,
        C_col,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        B_row,
        C_row,
        lda_cublas,
        ldb_cublas,
        ldc_cublas,
        lda_flag,
        ldb_flag,
        ldc_flag,
        beta,
        handle,
        alpha_ptr,
        beta_ptr,
    )


def cublas_bfgemm(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    if IS_HYGON:
        # DTK hipBLAS has no working bf16 gemm entry point: hipblasGemmEx and
        # hipblasGemmEx_v2 reject every (aType, bType, cType, computeType)
        # combination tested (status HIPBLAS_STATUS_INVALID_ENUM / NOT_SUPPORTED,
        # including plain fp32), and there is no legacy hipblasBgemm. torch's
        # bf16 matmul dispatches to the vendor-tuned rocBLAS bf16 Tensile
        # kernel (ISA936, fp32 accumulate), which is exactly the kernel a
        # working hipBLAS bf16 gemmEx would select, so it is used here as the
        # native latency baseline.
        A = A_row
        B = B_row
        if transa == CUBLAS_OP_T:
            A = A.t()
        if transb == CUBLAS_OP_T:
            B = B.t()
        if alpha == 1.0 and beta == 0.0:
            # BLAS reference computes in place: C <- A @ B (no alpha/beta
            # kernel to launch). Skip the identity alpha multiply that would
            # otherwise add an extra full-tensor kernel per call.
            C_row.copy_(torch.matmul(A, B))
        else:
            out = alpha * torch.matmul(A, B)
            if beta != 0:
                out = out + beta * C_row
            C_row.copy_(out)
        return C_row
    cublas.gemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        CUDA_R_16BF,
        lda_cublas,
        B_col.data_ptr(),
        CUDA_R_16BF,
        ldb_cublas,
        beta_ptr,
        C_col.data_ptr(),
        CUDA_R_16BF,
        ldc_cublas,
        GEMM_COMPUTE_TYPE,
        GEMM_ALGO,
    )
    return C_col


def cublas_bfgemm_reference(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    """fp32-accumulation accuracy reference for bf16 GEMM.

    FlagBLAS bfgemm kernels accumulate in fp32 (the cuBLAS gemmEx semantic
    used on NVIDIA below). On Hygon/DTK there is no hipBLAS bf16 gemm entry
    point, so the bf16 operands (exactly representable in fp32) are multiplied
    on-device in fp32 as the accuracy oracle.
    """
    if IS_HYGON:
        A = A_row.float()
        B = B_row.float()
        if transa == CUBLAS_OP_T:
            A = A.t()
        if transb == CUBLAS_OP_T:
            B = B.t()
        acc = alpha * torch.matmul(A, B)
        if beta != 0:
            acc += beta * C_row.float()
        return acc.to(torch.bfloat16)
    C_fp32 = torch.empty_strided(
        C_col.shape, C_col.stride(), dtype=torch.float32, device=C_col.device
    )
    C_fp32.copy_(C_col)

    cublas.gemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha_ptr,
        A_col.data_ptr(),
        CUDA_R_16BF,
        lda_cublas,
        B_col.data_ptr(),
        CUDA_R_16BF,
        ldb_cublas,
        beta_ptr,
        C_fp32.data_ptr(),
        CUDA_R_32F,
        ldc_cublas,
        GEMM_COMPUTE_TYPE,
        GEMM_ALGO,
    )
    C_col.copy_(C_fp32.to(torch.bfloat16))
    return C_col


def gems_bfgemm_wrapper(
    A_col,
    B_col,
    C_col,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A_row,
    B_row,
    C_row,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    beta,
    handle,
    alpha_ptr,
    beta_ptr,
):
    flag_blas.bfgemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A_row,
        lda_flag,
        B_row,
        ldb_flag,
        beta,
        C_row,
        ldc_flag,
    )
    return C_row


class GemmBenchmark(Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def __init__(
        self,
        *args,
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
        alpha=1.0,
        beta=0.0,
        alpha_dtype=np.float32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transa = transa
        self.transb = transb
        self.alpha = alpha
        self.beta = beta
        self.alpha_dtype = alpha_dtype

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        """
        Return additional shapes for COMPREHENSIVE benchmark level.
        These include shapes from real-world LLMs and special large-k cases.
        """
        return GEMM_SHAPES + model_shapes()

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            _, handle = _prepare_hipblas(self.device)
            if cur_dtype == torch.float16:
                # hipblasHgemm reads alpha/beta as fp16 (hipblasHalf) scalars,
                # so pack the host scalars into IEEE-754 binary16.
                alpha_np = ctypes.c_ushort(
                    int.from_bytes(struct.pack("<e", float(self.alpha)), "little")
                )
                beta_np = ctypes.c_ushort(
                    int.from_bytes(struct.pack("<e", float(self.beta)), "little")
                )
            elif self.alpha_dtype == np.float64:
                # hipblasDgemm reads alpha/beta as fp64 host scalars.
                alpha_np = ctypes.c_double(float(self.alpha))
                beta_np = ctypes.c_double(float(self.beta))
            elif self.alpha_dtype == np.complex64:
                # hipblasCgemm reads alpha/beta as hipblasComplex host structs.
                alpha_np = _HipComplex(float(self.alpha.real), float(self.alpha.imag))
                beta_np = _HipComplex(float(self.beta.real), float(self.beta.imag))
            elif self.alpha_dtype == np.complex128:
                # hipblasZgemm reads alpha/beta as hipblasDoubleComplex host
                # structs.
                alpha_np = _HipDoubleComplex(
                    float(self.alpha.real), float(self.alpha.imag)
                )
                beta_np = _HipDoubleComplex(
                    float(self.beta.real), float(self.beta.imag)
                )
            else:
                alpha_np = ctypes.c_float(float(self.alpha))
                beta_np = ctypes.c_float(float(self.beta))
            alpha_ptr = ctypes.byref(alpha_np)
            beta_ptr = ctypes.byref(beta_np)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            cublas.setMathMode(handle, 0)
            torch.backends.cuda.matmul.allow_tf32 = False

            alpha_np = np.array(self.alpha, dtype=self.alpha_dtype)
            beta_np = np.array(self.beta, dtype=self.alpha_dtype)
            alpha_ptr = alpha_np.ctypes.data
            beta_ptr = beta_np.ctypes.data

        for shape in self.shapes:
            m, n, k = shape
            if self.transa == CUBLAS_OP_N:
                A_col = torch.randn(k, m, dtype=cur_dtype, device=self.device).t()
                lda_cublas, lda_flag = m, k
            else:
                A_col = torch.randn(m, k, dtype=cur_dtype, device=self.device).t()
                lda_cublas, lda_flag = k, m
            A_row = A_col.contiguous()

            if self.transb == CUBLAS_OP_N:
                B_col = torch.randn(n, k, dtype=cur_dtype, device=self.device).t()
                ldb_cublas, ldb_flag = k, n
            else:
                B_col = torch.randn(k, n, dtype=cur_dtype, device=self.device).t()
                ldb_cublas, ldb_flag = n, k
            B_row = B_col.contiguous()

            C_col = torch.randn(n, m, dtype=cur_dtype, device=self.device).t()
            C_row = C_col.contiguous()
            ldc_cublas, ldc_flag = m, n

            yield A_col, B_col, C_col.clone(), {
                "transa": self.transa,
                "transb": self.transb,
                "m": m,
                "n": n,
                "k": k,
                "alpha": self.alpha,
                "A_row": A_row,
                "B_row": B_row,
                "C_row": C_row,
                "lda_cublas": lda_cublas,
                "ldb_cublas": ldb_cublas,
                "ldc_cublas": ldc_cublas,
                "lda_flag": lda_flag,
                "ldb_flag": ldb_flag,
                "ldc_flag": ldc_flag,
                "beta": self.beta,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
        return 2 * m * n * k

    def get_gbps(self, args, latency):
        A, B, C = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(A)
            + shape_utils.size_in_bytes(B)
            + 2 * shape_utils.size_in_bytes(C)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def validate_results(self, torch_result, gems_result, reduce_dim, tolerance=1e-5):
        """
        Compare whether the two result tensors are equal within the specified tolerance.
        If the error exceeds the specified tolerance, throw an AssertionError.
        """
        torch_cpu = torch_result.cpu()
        gems_cpu = gems_result.cpu()

        try:
            flag_blas.testing.assert_close(
                gems_cpu,
                torch_cpu,
                torch_cpu.dtype,
                equal_nan=False,
                reduce_dim=reduce_dim,
                atol=tolerance,
            )
        except AssertionError as e:
            max_abs_diff = torch.max(torch.abs(torch_cpu - gems_cpu))
            max_rel_diff = torch.max(
                torch.abs((torch_cpu - gems_cpu) / (torch.abs(torch_cpu) + 1e-9))
            )
            raise AssertionError(
                f"{e} Results differ beyond tolerance {tolerance}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {torch_cpu.shape}"
            )

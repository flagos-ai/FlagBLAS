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
import itertools
import os
from typing import Generator, List, Tuple

import numpy as np
import torch

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.ops import CUBLAS_OP_N
from flag_blas.utils import shape_utils

# Multi-vendor GEMM reference: use the official BLAS library of the running
# vendor when available, otherwise fall back to a torch.matmul reference.
#
#     vendor      device_name   reference library     loader
#     ------------------------------------------------------------
#     nvidia      cuda          cuBLAS                cupy
#     iluvatar    corex         cuBLAS-compatible     ctypes libcublas (best effort)
#     mthreads    musa          muBLAS                ctypes libmublas
#     others      -             none yet              torch.matmul fallback
_CUBLAS_VENDORS = ("nvidia", "iluvatar")

# GEMM entry names per dtype: (cupy name, ctypes name-suffix).
_GEMM_NAMES = {
    torch.float32: ("sgemm", "Sgemm"),
    torch.float64: ("dgemm", "Dgemm"),
    torch.complex64: ("cgemm", "Cgemm"),
    torch.complex128: ("zgemm", "Zgemm"),
}

try:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    HAS_CUBLAS = True
except ImportError:
    cp = None
    cublas = None
    HAS_CUBLAS = False


def _load_lib(names, search_dir=None):
    """Load the first shared library that can be resolved."""
    if search_dir is not None:
        for name in names:
            path = os.path.join(search_dir, name)
            if os.path.exists(path):
                try:
                    return ctypes.cdll.LoadLibrary(path)
                except OSError:
                    continue
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    return None


def _lib_hint(vendor):
    """Directory where the vendor BLAS library usually lives."""
    if vendor == "mthreads":
        return os.path.join(os.environ.get("MUSA_HOME", "/usr/local/musa"), "lib")
    if vendor in _CUBLAS_VENDORS:
        cuda_home = os.environ.get("CUDA_HOME")
        return os.path.join(cuda_home, "lib64") if cuda_home else None
    return None


# cuBLAS-compatible library via ctypes (best effort on non-cupy CUDA vendors,
# e.g. iluvatar), or muBLAS for mthreads.
_ctypes_blas = None  # (lib, function-prefix)
if flag_blas.vendor_name in _CUBLAS_VENDORS and not HAS_CUBLAS:
    _lib = _load_lib(
        ["libcublas.so", "libcublas.so.12", "libcublas.so.11"],
        _lib_hint(flag_blas.vendor_name),
    )
    if _lib is not None:
        _ctypes_blas = (_lib, "cublas")
elif flag_blas.vendor_name == "mthreads":
    _lib = _load_lib(
        ["libmublas.so", "libmublas.so.1"], _lib_hint(flag_blas.vendor_name)
    )
    if _lib is not None:
        _ctypes_blas = (_lib, "mublas")




def _clear_musa_sticky_error():
    """Consume the sticky error muBLAS leaves on the MUSA context.

    When called through ctypes, muBLAS may hit driver APIs the installed MUSA
    driver does not implement (e.g. on large non-transposed shapes) and leave
    a sticky musaErrorNotSupported on the context. The error is only reported
    to the *next* torch_musa API call (e.g. a clone), which would otherwise
    fail spuriously. Reading musaGetLastError clears it.
    """
    global _musart_last_error_fn
    if _musart_last_error_fn is None:
        _musart_last_error_fn = ctypes.cdll.LoadLibrary("libmusart.so.5")
        _musart_last_error_fn.musaGetLastError.restype = ctypes.c_int
    _musart_last_error_fn.musaGetLastError()


_musart_last_error_fn = None
def native_blas_available():
    """Whether an official BLAS library is usable on this vendor."""
    return HAS_CUBLAS or _ctypes_blas is not None


def _create_handle(lib, create_names):
    """Create a native BLAS handle through ctypes (best effort)."""
    for name in create_names:
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p]
        handle = ctypes.c_void_p()
        if fn(ctypes.byref(handle)) == 0:
            return handle.value
    return None


def get_blas_handle():
    """Native BLAS handle of the current vendor (None if unavailable)."""
    if HAS_CUBLAS:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        cublas.setMathMode(handle, 0)
        return handle
    if _ctypes_blas is not None:
        lib, prefix = _ctypes_blas
        return _create_handle(lib, [f"{prefix}Create_v2", f"{prefix}Create"])
    return None


def _config_gemm_fn(lib, name):
    fn = getattr(lib, name, None)
    if fn is None:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [
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
    return fn


def _native_op(code):
    """Map a cuBLAS op code (N=0/T=1/C=2) to the native BLAS codes.

    cuBLAS-compatible libraries (iluvatar) reuse 0/1/2, but muBLAS uses
    MUBLAS_OP_N=111/MUBLAS_OP_T=112/MUBLAS_OP_C=113.
    """
    if _ctypes_blas is not None and _ctypes_blas[1] == "mublas":
        return {CUBLAS_OP_N: 111, 1: 112, 2: 113}[code]
    return code


def native_gemm(handle, dtype, transa, transb, m, n, k, alpha_ptr, A, lda, B,
                ldb, beta_ptr, C, ldc):
    """Native GEMM (column-major, cuBLAS semantics) for the current vendor.

    ``A``/``B``/``C`` are torch tensors already in column-major (cuBLAS) layout.
    Raises ``RuntimeError`` when no native BLAS library is available; callers
    should fall back to :func:`torch_gemm_reference`.
    """
    if HAS_CUBLAS:
        fn = getattr(cublas, _GEMM_NAMES[dtype][0])
        fn(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A.data_ptr(),
            lda,
            B.data_ptr(),
            ldb,
            beta_ptr,
            C.data_ptr(),
            ldc,
        )
        return
    if _ctypes_blas is None:
        raise RuntimeError(
            f"No native BLAS library available for vendor {flag_blas.vendor_name!r}"
        )
    lib, prefix = _ctypes_blas
    fn = _config_gemm_fn(lib, f"{prefix}{_GEMM_NAMES[dtype][1]}")
    if fn is None:
        raise RuntimeError(
            f"Vendor library {prefix} does not export "
            f"{prefix}{_GEMM_NAMES[dtype][1]}"
        )
    fn(
        ctypes.c_void_p(handle),
        ctypes.c_int(_native_op(transa)),
        ctypes.c_int(_native_op(transb)),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_void_p(alpha_ptr),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_int(ldb),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(ldc),
    )
    if prefix == "mublas":
        _clear_musa_sticky_error()


def torch_gemm_reference(A_row, B_row, C_row, transa, transb, alpha, beta):
    """Reference GEMM implemented with torch.matmul (universal fallback)."""
    a = A_row if transa == CUBLAS_OP_N else A_row.t()
    b = B_row if transb == CUBLAS_OP_N else B_row.t()
    C_row.copy_(alpha * torch.matmul(a, b) + beta * C_row)
    return C_row


def disable_tf32():
    """Turn off TF32 so fp32 references accumulate in true fp32."""
    if flag_blas.vendor_name == "mthreads":
        if hasattr(torch.backends, "mudnn"):
            torch.backends.mudnn.allow_tf32 = False
        return
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False

CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14

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
    """fp32 GEMM reference: official BLAS of the running vendor when available
    (cuBLAS / muBLAS), otherwise a torch.matmul-based fallback."""
    if native_blas_available():
        native_gemm(
            handle,
            torch.float32,
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
        return C_col
    return torch_gemm_reference(A_row, B_row, C_row, transa, transb, alpha, beta)


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
    if not HAS_CUBLAS:
        return torch_gemm_reference(A_row, B_row, C_row, transa, transb, alpha, beta)
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
        CUDA_R_32F,
        0,
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
    if not HAS_CUBLAS:
        return torch_gemm_reference(A_row, B_row, C_row, transa, transb, alpha, beta)
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
        CUDA_R_32F,
        0,
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
    if not HAS_CUBLAS:
        return torch_gemm_reference(A_row, B_row, C_row, transa, transb, alpha, beta)
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
        CUDA_R_32F,
        0,
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
        # Native BLAS handle of the running vendor (cuBLAS / muBLAS), or None
        # when the vendor has no official library and we fall back to
        # torch.matmul.
        handle = get_blas_handle()
        disable_tf32()

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

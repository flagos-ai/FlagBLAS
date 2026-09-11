"""Small cuBLAS-shaped adapter for the TorchMUSA muBLAS runtime.

The benchmark reference implementations use the cuBLAS Python/ctypes API.  A
CUDA CuPy installation cannot be used on a MUSA process, so this module keeps
the same call surface while resolving symbols from ``libmublas`` and using the
TorchMUSA BLAS handle for the current stream.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

import torch


MUBLAS_POINTER_MODE_HOST = 0
MUBLAS_POINTER_MODE_DEVICE = 1
_MUBLAS_GEMM_EX_SYMBOL = (
    "_Z12mublasGemmExP15_mublasHandle_t17mublasOperation_tS1_iiiPKvS3_"
    "14musaDataType_tiS3_S4_iS3_PvS4_i19mublasComputeType_t16mublasGemmAlgo_t"
)


def _handle_value():
    from torch_musa.core import current_blas_handle

    return int(current_blas_handle())


def _stream_value(device=None):
    if device is None:
        device = torch.musa.current_device()
    stream = torch.musa.current_stream(device)
    return int(stream.musa_stream)


class _MuBLASLibrary:
    """Resolve both cuBLAS and hipBLAS spellings against muBLAS symbols."""

    def __init__(self, raw):
        self._raw = raw

    def __getattr__(self, name):
        symbol = name
        for prefix in ("cublas", "hipblas"):
            if symbol.startswith(prefix):
                symbol = "mublas" + symbol[len(prefix) :]
                break
        else:
            if symbol.startswith("mublas"):
                return getattr(self._raw, symbol)
            # CuPy's cublas module exposes short spellings such as ``sgemv``;
            # the shared library exports the conventional ``mublasSgemv``.
            if symbol and symbol[0].isalpha():
                symbol = "mublas" + symbol[0].upper() + symbol[1:]
        if symbol.endswith("_v2"):
            symbol = symbol[:-3]
        if symbol == "mublasGemmEx":
            symbol = _MUBLAS_GEMM_EX_SYMBOL
        function = getattr(self._raw, symbol)
        # The benchmark's CuPy calls pass data/scalar pointers as Python
        # integers. ctypes defaults integer arguments to C ``int`` and
        # truncates 64-bit device addresses. Distinguish ordinary BLAS enum/
        # dimension values from device/host pointers by their magnitude.
        def call(*args):
            values = list(args)
            lname = name.lower()
            def enum_value(value):
                return value.value if isinstance(value, ctypes._SimpleCData) else value

            # cuBLAS uses zero-based enum values while muBLAS follows the
            # cuBLAS ABI constants (MUBLAS_OP_N=111, ...).  The benchmark
            # inputs intentionally use the public CUBLAS_* constants.
            if len(values) > 1 and any(token in lname for token in ("gemv", "gbmv")):
                raw = enum_value(values[1])
                values[1] = {0: 111, 1: 112, 2: 113}.get(raw, raw)
            if len(values) > 1 and any(
                token in lname
                for token in ("hemv", "hbmv", "symv", "spmv", "sbmv", "her", "syr", "spr", "hpr")
            ):
                raw = enum_value(values[1])
                values[1] = {0: 122, 1: 121}.get(raw, raw)
            if len(values) > 1 and any(
                token in lname
                for token in ("trmv", "trsv", "tbmv", "tbsv", "tpmv", "tpsv")
            ):
                raw = enum_value(values[1])
                values[1] = {0: 122, 1: 121}.get(raw, raw)
                if len(values) > 2:
                    raw = enum_value(values[2])
                    values[2] = {0: 111, 1: 112, 2: 113}.get(raw, raw)
                if len(values) > 3:
                    raw = enum_value(values[3])
                    values[3] = {0: 131, 1: 132}.get(raw, raw)
            args = tuple(values)
            argtypes = []
            converted = []
            for value in args:
                if isinstance(value, int):
                    if abs(value) < (1 << 31):
                        argtypes.append(ctypes.c_int)
                        converted.append(ctypes.c_int(value))
                    else:
                        argtypes.append(ctypes.c_void_p)
                        converted.append(ctypes.c_void_p(value))
                else:
                    target = getattr(value, "_obj", None)
                    if target is not None:
                        argtypes.append(ctypes.c_void_p)
                        converted.append(ctypes.c_void_p(ctypes.addressof(target)))
                    elif isinstance(value, ctypes.c_void_p):
                        argtypes.append(ctypes.c_void_p)
                        converted.append(value)
                    else:
                        argtypes.append(type(value))
                        converted.append(value)
            function.argtypes = argtypes
            function.restype = ctypes.c_int
            return function(*converted)

        return call


_LIBRARY = None


def load_mublas():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    names = []
    found = ctypes.util.find_library("mublas")
    if found:
        names.append(found)
    names.extend(["libmublas.so"])
    musa_home = os.environ.get("MUSA_HOME") or os.environ.get("MUSA_PATH")
    if musa_home:
        names.append(os.path.join(musa_home, "lib", "libmublas.so"))
        names.append(os.path.join(musa_home, "lib64", "libmublas.so"))
    errors = []
    for name in names:
        try:
            _LIBRARY = _MuBLASLibrary(ctypes.CDLL(name))
            return _LIBRARY
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Unable to load libmublas.so (" + "; ".join(errors) + ")")


def get_mublas_handle(device=None):
    handle = ctypes.c_void_p(_handle_value())
    library = load_mublas()
    status = library.mublasSetStream(handle, ctypes.c_void_p(_stream_value(device)))
    if status != 0:
        raise RuntimeError(f"mublasSetStream failed with status {status}")
    return handle.value


class _CublasCompat:
    CUBLAS_POINTER_MODE_HOST = MUBLAS_POINTER_MODE_HOST
    CUBLAS_POINTER_MODE_DEVICE = MUBLAS_POINTER_MODE_DEVICE
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2

    def __getattr__(self, name):
        if name in {"setPointerMode", "setPointerMode_v2"}:
            return lambda handle, mode: load_mublas().mublasSetPointerMode(
                ctypes.c_void_p(int(handle)), mode
            )
        if name in {"setStream", "setStream_v2"}:
            return lambda handle, stream: load_mublas().mublasSetStream(
                ctypes.c_void_p(int(handle)), ctypes.c_void_p(int(stream))
            )
        return getattr(load_mublas(), name)


cublas = _CublasCompat()


class _MusaDevice:
    @staticmethod
    def get_cublas_handle():
        return get_mublas_handle()


class _MusaCuda:
    device = _MusaDevice()


# A tiny CuPy-compatible namespace used by the legacy TRSV benchmark.  It
# deliberately exposes only the handle accessor needed by that benchmark;
# data remains in TorchMUSA tensors and no CuPy package is imported.
cp = type("_CompatCuPy", (), {"cuda": _MusaCuda()})()


class _DeviceCompat:
    @staticmethod
    def get_cublas_handle():
        return get_mublas_handle()


class _CudaCompat:
    device = _DeviceCompat()


class _CuPyCompat:
    cuda = _CudaCompat()


cp = _CuPyCompat()


__all__ = ["cp", "cublas", "load_mublas", "get_mublas_handle"]

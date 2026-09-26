# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Saved H100 cuBLAS timings for T-Head Level 2 equivalent benchmarks.

Score = (H100 resource peak / PPU resource capacity) *
        (H100 cuBLAS latency / PPU FlagBLAS latency).
Memory uses both vendors' nominal bandwidth. Vector mode uses measured PPU
FP32/FP64 rates because no reliable published nominal vector specification
is available; it is a diagnostic scenario, not the default GEMV score.
Logical FLOP and byte counts are displayed only; they do not enter the score.
"""

import gc
import json
import math
import os
from pathlib import Path

import torch

import flag_blas
from benchmark.attri_util import BenchmarkMetrics, BenchmarkResult
from benchmark.conftest import Config, emit_record_logger
from benchmark.level2_metrics import level2_workload

_ROOT = Path(__file__).resolve().parents[1]
_BASE = _ROOT / "benchmark/baselines"
_THEAD_GEMV_OPS = {
    "cgemv",
    "cgemv_trans",
    "cgemv_conj",
    "zgemv",
    "zgemv_trans",
    "zgemv_conj",
}
_THEAD_GBMV_OPS = {"sgbmv", "dgbmv", "cgbmv", "zgbmv"}
_THEAD_SYMV_OPS = {"ssymv", "dsymv", "csymv", "zsymv"}
_THEAD_SBMV_OPS = {"ssbmv", "dsbmv"}
_THEAD_HBMV_OPS = {"chbmv", "zhbmv"}
_THEAD_SPMV_OPS = {"sspmv", "dspmv"}
_THEAD_HPMV_OPS = {"chpmv", "zhpmv"}
_THEAD_TRSV_OPS = {"strsv", "dtrsv", "ctrsv", "ztrsv"}
_THEAD_TBSV_OPS = {"stbsv", "dtbsv", "ctbsv", "ztbsv"}
_THEAD_TPSV_OPS = {"stpsv", "dtpsv", "ctpsv", "ztpsv"}
_THEAD_HEMV_OPS = {"chemv", "zhemv"}
_THEAD_TRMV_OPS = {"strmv", "dtrmv", "ctrmv", "ztrmv"}
_THEAD_TBMV_OPS = {"stbmv", "dtbmv", "ctbmv", "ztbmv"}
_THEAD_TPMV_OPS = {"stpmv", "dtpmv", "ctpmv", "ztpmv"}
_THEAD_HPR_OPS = {"chpr", "zhpr"}
_THEAD_HPR2_OPS = {"chpr2", "zhpr2"}
_THEAD_HER_OPS = {"cher", "zher"}
_THEAD_HER2_OPS = {"cher2", "zher2"}
_THEAD_SYR_OPS = {"csyr", "zsyr"}
_THEAD_GER_OPS = {"cgeru", "cgerc", "zgeru", "zgerc"}


def _scalar(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    return value


class THeadGemvReference:
    def __init__(self, ops=None, reference_label="c/zgemv"):
        self.reference_label = reference_label
        self.bottleneck = os.environ.get("FLAGBLAS_L2_BOTTLENECK", "memory")
        self.threshold = float(os.environ.get("FLAGBLAS_L2_THRESHOLD", "0.8"))
        if self.bottleneck not in ("memory", "vector"):
            raise ValueError("T-Head L2 bottleneck must be memory or vector")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("T-Head L2 threshold must be finite and positive")
        self.hardware = json.loads((_BASE / "hardware.json").read_text())["thead"]
        self.cases = {}
        selected_ops = _THEAD_GEMV_OPS if ops is None else ops
        for row in json.loads((_BASE / "h100.json").read_text()):
            if row["op"] not in selected_ops:
                continue
            key = self._key(row)
            if key in self.cases:
                raise ValueError(f"Duplicate H100 L2 baseline case: {key}")
            if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
                raise ValueError(f"Invalid H100 L2 latency: {key}")
            self.cases[key] = row
        if not self.cases:
            raise ValueError("H100 L2 baseline is empty")

    @staticmethod
    def _key(row):
        return (
            row["op"],
            row["dtype"],
            row["m"],
            row["n"],
            row["trans"],
            tuple(row["alpha"]),
            tuple(row["beta"]),
        )

    def lookup(self, op, dtype, kwargs):
        row = dict(
            op=op,
            dtype=str(dtype),
            m=kwargs["m"],
            n=kwargs["n"],
            trans=kwargs["trans"],
            alpha=_scalar(kwargs["alpha"]),
            beta=_scalar(kwargs["beta"]),
        )
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 L2 baseline case: {key}")
        return self.cases[key]

    def factor(self, dtype):
        resource = (
            "memory"
            if self.bottleneck == "memory"
            else (
                "vector_fp64"
                if dtype in (torch.float64, torch.complex128)
                else "vector_fp32"
            )
        )
        values = self.hardware[resource]
        if resource == "memory":
            denominator, basis = values["thead_vendor_theoretical"], "nominal"
        else:
            denominator, basis = values["thead_measured"], "measured_fallback"
        return resource, values["h100_theoretical"] / denominator, basis

    def description_lines(self):
        memory = self.hardware["memory"]
        nominal = memory["thead_vendor_theoretical"]
        attained = max(memory["thead_measured"], memory["validation_card1_measured"])
        nominal_factor = memory["h100_theoretical"] / nominal
        lines = [
            "[conversion] T-Head PPU-ZW810E; reference=H100 SXM saved cuBLAS "
            f"{self.reference_label}",
            "[hardware] source=H100 NVIDIA specification; PPU Alibaba Cloud product "
            "specification and 2026-09-23 single-card measurements",
            f"[hardware] memory: H100_nominal={memory['h100_theoretical']:.9g} TB/s; "
            f"PPU_nominal={nominal:.9g} TB/s; PPU_streaming_measured="
            f"{memory['thead_measured']:.9g}/{memory['validation_card1_measured']:.9g} "
            f"TB/s (cards 0/1); best={attained:.9g} TB/s; "
            f"attained={100 * attained / nominal:.2f}% of PPU nominal; "
            f"gap={100 * (1 - attained / nominal):.2f}%",
            f"[conversion] memory: K_nominal={memory['h100_theoretical']:.9g}/"
            f"{nominal:.9g}={nominal_factor:.9g}; "
            f"time_ratio_limit=K/{self.threshold:g}="
            f"{nominal_factor / self.threshold:.9g} (default scoring)",
        ]
        for name in ("vector_fp32", "vector_fp64"):
            values = self.hardware[name]
            factor = values["h100_theoretical"] / values["thead_measured"]
            lines.append(
                f"[hardware] {name}: H100_nominal={values['h100_theoretical']:.9g} "
                f"TFLOPS; PPU_nominal=unpublished; PPU_FMA_measured="
                f"{values['thead_measured']:.9g}/"
                f"{values['validation_card1_measured']:.9g} TFLOPS (cards 0/1); "
                f"K_measured_fallback={factor:.9g} (vector scenario only)"
            )
        for name in ("tensor_fp16", "tensor_bf16"):
            values = self.hardware[name]
            factor = values["h100_theoretical_dense"] / values["thead_measured_gemm"]
            lines.append(
                f"[hardware] {name}_dense: H100_nominal_dense="
                f"{values['h100_theoretical_dense']:.9g} TFLOPS "
                "(half of sparse-listed specification); PPU_nominal=unpublished; "
                f"PPU_GEMM_measured={values['thead_measured_gemm']:.9g} TFLOPS; "
                f"K_measured_fallback={factor:.9g} (not used for default scoring)"
            )
        lines.extend(
            (
                "[conversion] score=K*T_H100/T_PPU; logical FLOP/byte rates are display only",
                f"[conversion] selected={self.bottleneck} (assumed); "
                f"PASS iff score > {self.threshold:g}; "
                "no measured case bottleneck claim",
            )
        )
        return lines

    def describe(self):
        for line in self.description_lines():
            print(line)


class THeadGbmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four GBMV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_GBMV_OPS, reference_label="s/d/c/zgbmv")

    @staticmethod
    def _key(row):
        def scalar(value):
            return tuple(value) if isinstance(value, list) else value

        return (
            row["op"],
            row["dtype"],
            row["trans"],
            row["m"],
            row["n"],
            row["kl"],
            row["ku"],
            row["lda"],
            row["incx"],
            row["incy"],
            scalar(row["alpha"]),
            scalar(row["beta"]),
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{
                name: kwargs[name]
                for name in ("trans", "m", "n", "kl", "ku", "lda", "incx", "incy")
            },
            "alpha": _scalar(kwargs["alpha"]),
            "beta": _scalar(kwargs["beta"]),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 GBMV baseline case: {key}")
        return self.cases[key]


class THeadSymvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four SYMV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_SYMV_OPS, reference_label="s/d/c/zsymv")

    @staticmethod
    def _key(row):
        def scalar(value):
            return tuple(value) if isinstance(value, list) else value

        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo, row["lda"],
            row["incx"], row["incy"], scalar(row["alpha"]), scalar(row["beta"]),
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "uplo", "lda", "incx", "incy"
            )},
            "alpha": _scalar(kwargs["alpha"]),
            "beta": _scalar(kwargs["beta"]),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 SYMV baseline case: {key}")
        return self.cases[key]

class THeadSbmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for real row-banded SBMV."""

    def __init__(self):
        super().__init__(ops=_THEAD_SBMV_OPS, reference_label="s/dsbmv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], row["k"], uplo,
            row["lda"], row["incx"], row["incy"], row["alpha"], row["beta"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "k", "uplo", "lda", "incx", "incy", "alpha", "beta"
            )},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 SBMV baseline case: {key}")
        return self.cases[key]

class THeadHbmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-banded HBMV."""

    def __init__(self):
        super().__init__(ops=_THEAD_HBMV_OPS, reference_label="c/zhbmv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], row["k"], uplo,
            row["lda"], row["incx"], row["incy"],
            tuple(row["alpha"]), tuple(row["beta"]),
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "k", "uplo", "lda", "incx", "incy"
            )},
            "alpha": _scalar(complex(kwargs["alpha"])),
            "beta": _scalar(complex(kwargs["beta"])),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 HBMV baseline case: {key}")
        return self.cases[key]

class THeadSpmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for real row-packed SPMV."""

    def __init__(self):
        super().__init__(ops=_THEAD_SPMV_OPS, reference_label="s/dspmv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo,
            row["incx"], row["incy"], row["alpha"], row["beta"],
            row["packed_layout"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "uplo", "incx", "incy", "alpha", "beta", "packed_layout"
            )},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 SPMV baseline case: {key}")
        return self.cases[key]

class THeadHpmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-packed HPMV."""

    def __init__(self):
        super().__init__(ops=_THEAD_HPMV_OPS, reference_label="c/zhpmv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo,
            row["incx"], row["incy"], tuple(row["alpha"]), tuple(row["beta"]),
            row["packed_layout"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "uplo", "incx", "incy", "packed_layout"
            )},
            "alpha": _scalar(complex(kwargs["alpha"])),
            "beta": _scalar(complex(kwargs["beta"])),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 HPMV baseline case: {key}")
        return self.cases[key]

class THeadTrsvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four row-major TRSV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_TRSV_OPS, reference_label="s/d/c/ztrsv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo,
            row["trans"], row["diag"], row["lda"], row["incx"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "uplo", "trans", "diag", "lda", "incx"
            )},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 TRSV baseline case: {key}")
        return self.cases[key]

class THeadTbsvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four row-banded TBSV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_TBSV_OPS, reference_label="s/d/c/ztbsv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], row["k"], uplo,
            row["trans"], row["diag"], row["lda"], row["incx"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "k", "uplo", "trans", "diag", "lda", "incx"
            )},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 TBSV baseline case: {key}")
        return self.cases[key]

class THeadTpsvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four row-packed TPSV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_TPSV_OPS, reference_label="s/d/c/ztpsv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo,
            row["trans"], row["diag"], row["incx"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in ("n", "uplo", "trans", "diag", "incx")},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 TPSV baseline case: {key}")
        return self.cases[key]

class THeadHemvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-major full HEMV."""

    def __init__(self):
        super().__init__(ops=_THEAD_HEMV_OPS, reference_label="c/zhemv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo, row["lda"],
            row["incx"], row["incy"], tuple(row["alpha"]), tuple(row["beta"]),
            row["matrix_layout"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "uplo", "lda", "incx", "incy", "matrix_layout"
            )},
            "alpha": _scalar(complex(kwargs["alpha"])),
            "beta": _scalar(complex(kwargs["beta"])),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 HEMV baseline case: {key}")
        return self.cases[key]

class THeadTrmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four row-major TRMV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_TRMV_OPS, reference_label="s/d/c/ztrmv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo,
            row["trans"], row["diag"], row["lda"], row["incx"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "uplo", "trans", "diag", "lda", "incx"
            )},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 TRMV baseline case: {key}")
        return self.cases[key]


class THeadTbmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four row-banded TBMV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_TBMV_OPS, reference_label="s/d/c/ztbmv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], row["k"], uplo,
            row["trans"], row["diag"], row["lda"], row["incx"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in (
                "n", "k", "uplo", "trans", "diag", "lda", "incx"
            )},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 TBMV baseline case: {key}")
        return self.cases[key]

class THeadTpmvReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for all four row-packed TPMV dtypes."""

    def __init__(self):
        super().__init__(ops=_THEAD_TPMV_OPS, reference_label="s/d/c/ztpmv")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"], row["dtype"], row["n"], uplo,
            row["trans"], row["diag"], row["incx"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in ("n", "uplo", "trans", "diag", "incx")},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 TPMV baseline case: {key}")
        return self.cases[key]


class THeadHprReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-packed HPR."""

    def __init__(self):
        super().__init__(ops=_THEAD_HPR_OPS, reference_label="c/zhpr")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"],
            row["dtype"],
            row["n"],
            uplo,
            row["incx"],
            row["alpha"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in ("n", "uplo", "incx", "alpha")},
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 HPR baseline case: {key}")
        return self.cases[key]


class THeadHpr2Reference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-packed HPR2."""

    def __init__(self):
        super().__init__(ops=_THEAD_HPR2_OPS, reference_label="c/zhpr2")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"],
            row["dtype"],
            row["n"],
            uplo,
            row["incx"],
            row["incy"],
            tuple(row["alpha"]),
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{name: kwargs[name] for name in ("n", "uplo", "incx", "incy")},
            "alpha": _scalar(complex(kwargs["alpha"])),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 HPR2 baseline case: {key}")
        return self.cases[key]


class THeadHerReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-major full HER."""

    def __init__(self):
        super().__init__(ops=_THEAD_HER_OPS, reference_label="c/zher")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"],
            row["dtype"],
            row["n"],
            uplo,
            row["lda"],
            row["incx"],
            row["alpha"],
            row["matrix_layout"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{
                name: kwargs[name]
                for name in ("n", "uplo", "lda", "incx", "alpha", "matrix_layout")
            },
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 HER baseline case: {key}")
        return self.cases[key]


class THeadHer2Reference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-major full HER2."""

    def __init__(self):
        super().__init__(ops=_THEAD_HER2_OPS, reference_label="c/zher2")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"],
            row["dtype"],
            row["n"],
            uplo,
            row["lda"],
            row["incx"],
            row["incy"],
            tuple(row["alpha"]),
            row["matrix_layout"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{
                name: kwargs[name]
                for name in ("n", "uplo", "lda", "incx", "incy", "matrix_layout")
            },
            "alpha": _scalar(complex(kwargs["alpha"])),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 HER2 baseline case: {key}")
        return self.cases[key]


class THeadSyrReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-major full SYR."""

    def __init__(self):
        super().__init__(ops=_THEAD_SYR_OPS, reference_label="c/zsyr")

    @staticmethod
    def _key(row):
        uplo = {"lower": 0, "upper": 1}.get(row["uplo"], row["uplo"])
        return (
            row["op"],
            row["dtype"],
            row["n"],
            uplo,
            row["lda"],
            row["incx"],
            tuple(row["alpha"]),
            row["matrix_layout"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{
                name: kwargs[name]
                for name in ("n", "uplo", "lda", "incx", "matrix_layout")
            },
            "alpha": _scalar(complex(kwargs["alpha"])),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 SYR baseline case: {key}")
        return self.cases[key]


class THeadGerReference(THeadGemvReference):
    """Exact saved H100 cuBLAS reference for complex row-major full GER."""

    def __init__(self):
        super().__init__(ops=_THEAD_GER_OPS, reference_label="c/zgeru, c/zgerc")

    @staticmethod
    def _key(row):
        return (
            row["op"],
            row["dtype"],
            row["m"],
            row["n"],
            row["lda"],
            row["incx"],
            row["incy"],
            tuple(row["alpha"]),
            row["matrix_layout"],
        )

    def lookup(self, op, dtype, kwargs):
        row = {
            "op": op,
            "dtype": str(dtype),
            **{
                name: kwargs[name]
                for name in ("m", "n", "lda", "incx", "incy", "matrix_layout")
            },
            "alpha": _scalar(complex(kwargs["alpha"])),
        }
        key = self._key(row)
        if key not in self.cases:
            raise ValueError(f"Missing exact H100 GER baseline case: {key}")
        return self.cases[key]


def run_thead_gemv(bench):
    if flag_blas.vendor_name != "thead":
        raise RuntimeError("T-Head equivalent GEMV path used on another platform")
    if Config.query:
        return bench.run()
    bench.init_user_config()
    reference = THeadGemvReference()
    reference.describe()
    print(
        f"[correctness] {bench.op_name}: HGGC complex GEMV unsupported; "
        "CPU correctness is verified separately, not by this performance timer"
    )
    for dtype in bench.to_bench_dtypes:
        resource, factor, basis = reference.factor(dtype)
        metrics, records = [], []
        for inputs in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            saved = reference.lookup(bench.op_name, dtype, kwargs)
            workload = level2_workload("gemv", args, kwargs)
            metric = BenchmarkMetrics()
            metric.shape_detail = bench.record_shapes(*args, **kwargs)
            metric.latency = float(bench.get_latency(bench.gems_op, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError("T-Head GEMV latency must be finite and positive")
            metric.latency_base = saved["latency_ms"]
            metric.gbps_base = saved["gbps"]
            metric.tflops = workload["flops"] / metric.latency / 1e9
            metric.gbps = workload["bytes"] / metric.latency / 1e6
            raw_ratio = metric.latency_base / metric.latency
            equivalent_score = factor * raw_ratio
            metric.speedup = equivalent_score
            records.append(
                dict(
                    op=bench.op_name,
                    dtype=str(dtype),
                    m=kwargs["m"],
                    n=kwargs["n"],
                    trans=kwargs["trans"],
                    alpha=_scalar(kwargs["alpha"]),
                    beta=_scalar(kwargs["beta"]),
                    h100_ms=metric.latency_base,
                    ppu_ms=metric.latency,
                    raw_ratio=raw_ratio,
                    conversion_factor=factor,
                    equivalent_score=equivalent_score,
                    performance_status=(
                        "PASS"
                        if equivalent_score > reference.threshold
                        else "UNDERPERFORM"
                    ),
                    resource=resource,
                    normalization_basis=basis,
                    threshold=reference.threshold,
                    useful_bytes=workload["bytes"],
                    flops=workload["flops"],
                    useful_gbps=metric.gbps,
                    useful_tflops=metric.tflops,
                    baseline_id="h100",
                )
            )
            metrics.append(metric)
            gc.collect()
        result = BenchmarkResult(
            level=Config.bench_level.value,
            op_name=bench.op_name,
            dtype=str(dtype),
            mode=Config.mode.value,
            result=metrics,
        )
        print(
            "Metrics: level2-scalar-useful-v1; thead_l2=true; reference=h100_saved; "
            "Torch=H100 saved cuBLAS, Gems=PPU FlagBLAS; "
            "Gems Speedup=equivalent score; "
            "algorithmic TFLOPS / useful GB/s (not measured HBM traffic)"
        )
        print(result, end="")
        print("[thead-l2-results] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    gc.collect()
    flag_blas.runtime.torch_device_fn.empty_cache()


def run_thead_gbmv(bench):
    if flag_blas.vendor_name != "thead":
        raise RuntimeError("T-Head equivalent GBMV path used on another platform")
    if Config.query:
        return bench.run()
    bench.init_user_config()
    reference = THeadGbmvReference()
    reference.describe()
    print(
        f"[correctness] {bench.op_name}: HGGC GBMV unsupported; "
        "CPU correctness is verified separately, not by this performance timer"
    )
    for dtype in bench.to_bench_dtypes:
        resource, factor, basis = reference.factor(dtype)
        metrics, records = [], []
        for inputs in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            saved = reference.lookup(bench.op_name, dtype, kwargs)
            workload = level2_workload("gbmv", args, kwargs)
            metric = BenchmarkMetrics()
            metric.shape_detail = bench.record_shapes(*args, **kwargs)
            metric.latency = float(bench.get_latency(bench.gems_op, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError("T-Head GBMV latency must be finite and positive")
            metric.latency_base = saved["latency_ms"]
            metric.gbps_base = saved["gbps"]
            metric.tflops = workload["flops"] / metric.latency / 1e9
            metric.gbps = workload["bytes"] / metric.latency / 1e6
            raw_ratio = metric.latency_base / metric.latency
            equivalent_score = factor * raw_ratio
            metric.speedup = equivalent_score
            records.append(
                dict(
                    op=bench.op_name,
                    dtype=str(dtype),
                    **{
                        name: kwargs[name]
                        for name in (
                            "trans",
                            "m",
                            "n",
                            "kl",
                            "ku",
                            "lda",
                            "incx",
                            "incy",
                        )
                    },
                    alpha=_scalar(kwargs["alpha"]),
                    beta=_scalar(kwargs["beta"]),
                    h100_ms=metric.latency_base,
                    ppu_ms=metric.latency,
                    raw_ratio=raw_ratio,
                    conversion_factor=factor,
                    equivalent_score=equivalent_score,
                    performance_status=(
                        "PASS"
                        if equivalent_score > reference.threshold
                        else "UNDERPERFORM"
                    ),
                    resource=resource,
                    normalization_basis=basis,
                    threshold=reference.threshold,
                    useful_bytes=workload["bytes"],
                    flops=workload["flops"],
                    useful_gbps=metric.gbps,
                    useful_tflops=metric.tflops,
                    baseline_id="h100",
                )
            )
            metrics.append(metric)
            gc.collect()
        result = BenchmarkResult(
            level=Config.bench_level.value,
            op_name=bench.op_name,
            dtype=str(dtype),
            mode=Config.mode.value,
            result=metrics,
        )
        print(
            "Metrics: level2-scalar-useful-v1; thead_l2=true; reference=h100_saved; "
            "Torch=H100 saved cuBLAS, Gems=PPU FlagBLAS; "
            "Gems Speedup=equivalent score; "
            "algorithmic TFLOPS / useful GB/s (not measured HBM traffic)"
        )
        print(result, end="")
        print("[thead-l2-results] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    gc.collect()
    flag_blas.runtime.torch_device_fn.empty_cache()


def run_thead_symv(bench):
    return _run_thead_l2(
        bench, "symv", THeadSymvReference,
        ("n", "uplo", "lda", "incx", "incy", "alpha", "beta"),
    )

def run_thead_sbmv(bench):
    return _run_thead_l2(
        bench, "sbmv", THeadSbmvReference,
        ("n", "k", "uplo", "lda", "incx", "incy", "alpha", "beta"),
    )

def run_thead_hbmv(bench):
    return _run_thead_l2(
        bench, "hbmv", THeadHbmvReference,
        ("n", "k", "uplo", "lda", "incx", "incy", "alpha", "beta"),
    )

def run_thead_spmv(bench):
    return _run_thead_l2(
        bench, "spmv", THeadSpmvReference,
        ("n", "uplo", "incx", "incy", "alpha", "beta", "packed_layout"),
    )

def run_thead_hpmv(bench):
    return _run_thead_l2(
        bench, "hpmv", THeadHpmvReference,
        ("n", "uplo", "incx", "incy", "alpha", "beta", "packed_layout"),
    )

def run_thead_trsv(bench):
    return _run_thead_l2(
        bench, "trsv", THeadTrsvReference,
        ("n", "uplo", "trans", "diag", "lda", "incx"),
    )

def run_thead_tbsv(bench):
    return _run_thead_l2(
        bench, "tbsv", THeadTbsvReference,
        ("n", "k", "uplo", "trans", "diag", "lda", "incx"),
    )

def run_thead_tpsv(bench):
    return _run_thead_l2(
        bench, "tpsv", THeadTpsvReference,
        ("n", "uplo", "trans", "diag", "incx"),
    )

def run_thead_hemv(bench):
    return _run_thead_l2(
        bench, "hemv", THeadHemvReference,
        ("n", "uplo", "lda", "incx", "incy", "alpha", "beta", "matrix_layout"),
    )

def run_thead_trmv(bench):
    return _run_thead_l2(
        bench, "trmv", THeadTrmvReference,
        ("n", "uplo", "trans", "diag", "lda", "incx"),
    )


def run_thead_tbmv(bench):
    return _run_thead_l2(
        bench, "tbmv", THeadTbmvReference,
        ("n", "k", "uplo", "trans", "diag", "lda", "incx"),
    )

def run_thead_tpmv(bench):
    return _run_thead_l2(
        bench, "tpmv", THeadTpmvReference,
        ("n", "uplo", "trans", "diag", "incx"),
    )


def run_thead_hpr(bench):
    return _run_thead_l2(
        bench,
        "hpr",
        THeadHprReference,
        ("n", "uplo", "incx", "alpha"),
    )


def run_thead_hpr2(bench):
    return _run_thead_l2(
        bench,
        "hpr2",
        THeadHpr2Reference,
        ("n", "uplo", "incx", "incy", "alpha"),
    )


def run_thead_her(bench):
    return _run_thead_l2(
        bench,
        "her",
        THeadHerReference,
        ("n", "uplo", "lda", "incx", "alpha", "matrix_layout"),
    )


def run_thead_her2(bench):
    return _run_thead_l2(
        bench,
        "her2",
        THeadHer2Reference,
        ("n", "uplo", "lda", "incx", "incy", "alpha", "matrix_layout"),
    )


def run_thead_syr(bench):
    return _run_thead_l2(
        bench,
        "syr",
        THeadSyrReference,
        ("n", "uplo", "lda", "incx", "alpha", "matrix_layout"),
    )


def run_thead_ger(bench):
    return _run_thead_l2(
        bench,
        "ger",
        THeadGerReference,
        ("m", "n", "lda", "incx", "incy", "alpha", "matrix_layout"),
    )


def _run_thead_l2(bench, family, reference_type, case_fields):
    if flag_blas.vendor_name != "thead":
        raise RuntimeError(
            f"T-Head equivalent {family.upper()} path used on another platform"
        )
    if Config.query:
        return bench.run()
    bench.init_user_config()
    reference = reference_type()
    reference.describe()
    print(
        f"[correctness] {bench.op_name}: HGGC {family.upper()} unsupported; "
        "CPU correctness is verified separately, not by this performance timer"
    )
    for dtype in bench.to_bench_dtypes:
        resource, factor, basis = reference.factor(dtype)
        metrics, records = [], []
        for inputs in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            saved = reference.lookup(bench.op_name, dtype, kwargs)
            workload = level2_workload(family, args, kwargs)
            metric = BenchmarkMetrics()
            metric.shape_detail = bench.record_shapes(*args, **kwargs)
            metric.latency = float(bench.get_latency(bench.gems_op, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError(
                    f"T-Head {family.upper()} latency must be finite and positive"
                )
            metric.latency_base = saved["latency_ms"]
            metric.gbps_base = saved["gbps"]
            metric.tflops = workload["flops"] / metric.latency / 1e9
            metric.gbps = workload["bytes"] / metric.latency / 1e6
            raw_ratio = metric.latency_base / metric.latency
            equivalent_score = factor * raw_ratio
            metric.speedup = equivalent_score
            records.append(
                dict(
                    op=bench.op_name,
                    dtype=str(dtype),
                    **{name: saved[name] for name in case_fields},
                    h100_ms=metric.latency_base,
                    ppu_ms=metric.latency,
                    raw_ratio=raw_ratio,
                    conversion_factor=factor,
                    equivalent_score=equivalent_score,
                    performance_status=(
                        "PASS"
                        if equivalent_score > reference.threshold
                        else "UNDERPERFORM"
                    ),
                    resource=resource,
                    normalization_basis=basis,
                    threshold=reference.threshold,
                    useful_bytes=workload["bytes"],
                    flops=workload["flops"],
                    useful_gbps=metric.gbps,
                    useful_tflops=metric.tflops,
                    baseline_id="h100",
                )
            )
            metrics.append(metric)
            gc.collect()
        result = BenchmarkResult(
            level=Config.bench_level.value,
            op_name=bench.op_name,
            dtype=str(dtype),
            mode=Config.mode.value,
            result=metrics,
        )
        print(
            "Metrics: level2-scalar-useful-v1; thead_l2=true; reference=h100_saved; "
            "Torch=H100 saved cuBLAS, Gems=PPU FlagBLAS; "
            "Gems Speedup=equivalent score; "
            "algorithmic TFLOPS / useful GB/s (not measured HBM traffic)"
        )
        print(result, end="")
        print("[thead-l2-results] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    gc.collect()
    flag_blas.runtime.torch_device_fn.empty_cache()

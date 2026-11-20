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

"""Saved H100 cuBLAS timings for T-Head Level 3 GroupGEMM equivalent benchmarks.

Score = (H100 tensor peak / measured PPU tensor GEMM rate) *
        (H100 cuBLAS grouped-GEMM latency / PPU FlagBLAS latency).
The tensor resource is selected by the benchmark dtype, like the L2 vector
resource: float32 -> tensor_tf32, bfloat16 -> tensor_bf16, float16 ->
tensor_fp16. GroupGEMM is compute-bound, so no memory bottleneck scenario
is offered. Logical FLOP and byte counts are displayed only; they do not
enter the score.
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

_ROOT = Path(__file__).resolve().parents[1]
_BASE = _ROOT / "benchmark/baselines"


class THeadGroupGemmReference:
    def __init__(self, op_name, reference_file=None, hardware_file=None):
        self.op_name = op_name
        self.mode = os.environ.get("FLAGBLAS_L3_REFERENCE", "h100")
        self.threshold = float(os.environ.get("FLAGBLAS_L3_THRESHOLD", "0.9"))
        if self.mode not in ("none", "h100", "legacy"):
            raise ValueError("FLAGBLAS_L3_REFERENCE must be none, h100 or legacy")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("T-Head L3 threshold must be finite and positive")
        override = os.environ.get("FLAGBLAS_L3_CONVERSION_FACTOR")
        self.factor_override = float(override) if override is not None else None
        if self.factor_override is not None and (
            not math.isfinite(self.factor_override) or self.factor_override <= 0
        ):
            raise ValueError(
                "FLAGBLAS_L3_CONVERSION_FACTOR must be finite and positive"
            )
        if self.mode == "none":
            return

        self.path = (
            Path(reference_file)
            if reference_file
            else _ROOT
            / os.environ.get(
                "FLAGBLAS_L3_REFERENCE_FILE", "benchmark/baselines/h100_l3.json"
            )
        )
        self.baseline_id = self.path.name.rsplit(".", 1)[0]
        self.cases = {}
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        if not rows:
            raise ValueError("H100 L3 GroupGEMM baseline is empty")
        for row in rows:
            if row.get("op") != self.op_name:
                continue
            key = self._key(row)
            if key in self.cases:
                raise ValueError(f"Duplicate H100 L3 GroupGEMM baseline case: {key}")
            latency = float(row["latency_ms"])
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"Invalid H100 L3 GroupGEMM latency: {key}")
            self.cases[key] = row
        if not self.cases:
            raise ValueError(
                f"H100 L3 GroupGEMM baseline has no rows for op "
                f"{self.op_name!r}: {self.path}"
            )

        hardware_path = (
            Path(hardware_file) if hardware_file else _BASE / "hardware.json"
        )
        self.hardware = json.loads(hardware_path.read_text(encoding="utf-8"))["thead"]

    @staticmethod
    def _key(row):
        return (
            row["op"],
            row["dtype"],
            int(row["K"]),
            int(row["E"]),
            int(row["N"]),
        )

    def lookup(self, op, dtype, k, e, n):
        key = (op, str(dtype), int(k), int(e), int(n))
        if key not in self.cases:
            raise ValueError(
                f"Missing exact H100 L3 GroupGEMM baseline case: {key} "
                f"(baseline={self.baseline_id})"
            )
        return self.cases[key]

    def factor(self, dtype):
        resource = (
            "tensor_tf32"
            if dtype == torch.float32
            else ("tensor_bf16" if dtype == torch.bfloat16 else "tensor_fp16")
        )
        values = self.hardware[resource]
        if self.factor_override is not None:
            factor, basis = self.factor_override, "overridden"
        else:
            factor = values["h100_theoretical_dense"] / values["thead_measured_gemm"]
            basis = "measured_gemm"
        return resource, factor, basis

    def _reference_api(self):
        reference_api = "saved H100 grouped-GEMM timings"
        provenance = self.path.with_name(
            self.path.name.replace(".json", ".provenance.json")
        )
        if provenance.is_file():
            try:
                meta = json.loads(provenance.read_text(encoding="utf-8"))
                ref = meta.get("reference", {})
                if ref.get("library") and ref.get("api"):
                    reference_api = (
                        f"{ref['library']} {ref['api']} (saved H100 timings)"
                    )
            except (ValueError, OSError):
                pass
        return reference_api

    def description_lines(self):
        lines = [
            "[conversion] T-Head PPU-ZW810E; reference=H100 SXM saved cuBLAS "
            f"grouped GEMM ({self.op_name})",
            f"[conversion] baseline={self.baseline_id}; file={self.path}; "
            f"reference={self._reference_api()}",
        ]
        for name in ("tensor_tf32", "tensor_fp16", "tensor_bf16"):
            values = self.hardware[name]
            factor = values["h100_theoretical_dense"] / values["thead_measured_gemm"]
            lines.append(
                f"[hardware] {name}_dense: H100_nominal_dense="
                f"{values['h100_theoretical_dense']:.9g} TFLOPS "
                "(half of sparse-listed specification); PPU_nominal=unpublished; "
                f"PPU_GEMM_measured={values['thead_measured_gemm']:.9g} TFLOPS; "
                f"K_measured_gemm={factor:.9g} (selected by dtype for scoring)"
            )
        lines.append(
            "[conversion] selected=tensor (compute-bound, fixed); "
            f"PASS iff equivalent_score > {self.threshold:g}; "
            "equivalent_score = K * T_H100_saved / T_PPU_measured"
        )
        if self.factor_override is not None:
            lines.append(
                "[conversion] K overridden by FLAGBLAS_L3_CONVERSION_FACTOR="
                f"{self.factor_override:.9g}"
            )
        return lines

    def describe(self):
        if self.mode == "none":
            print("[conversion] reference=none; equivalent metrics not evaluated")
            return
        for line in self.description_lines():
            print(line)


class THeadGroupGemmTf32Reference(THeadGroupGemmReference):
    """Exact saved H100 cuBLAS reference for TF32 GroupGEMM."""

    def __init__(self, reference_file=None, hardware_file=None):
        super().__init__(
            op_name="group_tf32gemm",
            reference_file=reference_file,
            hardware_file=hardware_file,
        )


class THeadGroupBfGemmReference(THeadGroupGemmReference):
    """Exact saved H100 cuBLAS reference for BF16 GroupGEMM."""

    def __init__(self, reference_file=None, hardware_file=None):
        super().__init__(
            op_name="group_bfgemm",
            reference_file=reference_file,
            hardware_file=hardware_file,
        )


class THeadGroupHGemmReference(THeadGroupGemmReference):
    """Exact saved H100 cuBLAS reference for FP16 GroupGEMM."""

    def __init__(self, reference_file=None, hardware_file=None):
        super().__init__(
            op_name="group_hgemm",
            reference_file=reference_file,
            hardware_file=hardware_file,
        )


def run_thead_group_tf32gemm(bench):
    return _run_thead_l3(bench, THeadGroupGemmTf32Reference)


def run_thead_group_bfgemm(bench):
    return _run_thead_l3(bench, THeadGroupBfGemmReference)


def run_thead_group_hgemm(bench):
    return _run_thead_l3(bench, THeadGroupHGemmReference)


def _run_thead_l3(bench, reference_type):
    if flag_blas.vendor_name != "thead":
        raise RuntimeError("T-Head L3 GroupGEMM path used on another platform")
    if Config.query:
        return bench.run()
    bench.init_user_config()
    reference = reference_type()
    reference.describe()
    if reference.mode == "none":
        return bench.run()
    print(
        f"[correctness] {bench.op_name}: PPU vendor grouped-GEMM correctness is "
        "verified by the benchmark before this timer, not inside it"
    )
    for dtype in bench.to_bench_dtypes:
        resource, factor, basis = reference.factor(dtype)
        metrics, records = [], []
        for inputs in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            group_a, group_b, group_list = args[0], args[1], args[2]
            k, e, n = kwargs["K"], kwargs["group_size"], kwargs["N"]
            total_m = kwargs["M"]
            saved = reference.lookup(reference.op_name, dtype, k, e, n)
            metric = BenchmarkMetrics()
            metric.shape_detail = bench.record_shapes(*args, **kwargs)
            metric.latency = float(bench.get_latency(bench.gems_op, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError("T-Head GroupGEMM latency must be finite and positive")
            metric.latency_base = saved["latency_ms"]
            metric.gbps_base = saved["gbps"]
            flops = 2 * total_m * n * k
            metric.tflops = flops / metric.latency / 1e9
            io_bytes = (
                group_a.numel() * group_a.element_size()
                + group_b.numel() * group_b.element_size()
                + group_list.numel() * group_list.element_size()
                + total_m * n * group_a.element_size()
            )
            metric.gbps = io_bytes / metric.latency / 1e6
            raw_ratio = metric.latency_base / metric.latency
            equivalent_score = factor * raw_ratio
            metric.speedup = equivalent_score
            records.append(
                dict(
                    op=reference.op_name,
                    dtype=str(dtype),
                    K=k,
                    E=e,
                    N=n,
                    M=total_m,
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
                    flops=flops,
                    useful_tflops=metric.tflops,
                    baseline_id=reference.baseline_id,
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
            "Metrics: l3-groupgemm-equivalent-v1; thead_l3=true; "
            "reference=h100_saved; Torch=H100 saved grouped-GEMM, "
            "Gems=PPU FlagBLAS; Gems Speedup=equivalent score "
            f"({resource} normalized)"
        )
        print(result, end="")
        print("[thead-l3-results] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    gc.collect()
    flag_blas.runtime.torch_device_fn.empty_cache()

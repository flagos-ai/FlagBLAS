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

import gc
import json
import math
import os
from pathlib import Path

import flag_blas
from benchmark.attri_util import BenchmarkMetrics, BenchmarkResult
from benchmark.conftest import Config, emit_record_logger


_PROJECT = Path(__file__).resolve().parents[1]
_BASE = Path(__file__).resolve().parent / "baselines"
_CASE_FIELDS = (
    "op",
    "dtype",
    "compute_dtype",
    "K",
    "E",
    "N",
    "M",
    "alpha",
    "beta",
)
_PERFORMANCE_FIELDS = ("latency_ms", "tflops", "gbps")
_RESOURCES = {
    "fp16": "tensor_fp16",
    "bf16": "tensor_bf16",
    "tf32": "tensor_tf32",
}
_COMPUTE_OPS = {
    "fp16": "group_hgemm",
    "bf16": "group_bfgemm",
    "tf32": "group_tf32gemm",
}
_OP_COMPUTE_DTYPES = {op: name for name, op in _COMPUTE_OPS.items()}


class HygonGroupGemmReference:
    def __init__(self):
        self.mode = os.environ.get("FLAGBLAS_GROUP_GEMM_REFERENCE", "h100")
        self.threshold = float(os.environ.get("FLAGBLAS_GROUP_GEMM_THRESHOLD", "0.9"))
        if self.mode not in ("none", "h100"):
            raise ValueError("GroupGEMM reference must be none or h100")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("GroupGEMM threshold must be finite and positive")
        self.path = None
        self.baseline_id = None
        self.cases = {}
        self.hardware = None
        if self.mode == "none":
            return

        relative = os.environ.get(
            "FLAGBLAS_GROUP_GEMM_REFERENCE_FILE",
            "benchmark/baselines/h100_l3.json",
        )
        self.path = _PROJECT / relative
        self.baseline_id = self.path.stem
        rows = json.loads(self.path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError("H100 GroupGEMM baseline is empty")
        for saved in rows:
            saved = dict(saved)
            if "compute_dtype" not in saved:
                try:
                    saved["compute_dtype"] = _OP_COMPUTE_DTYPES[saved["op"]]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown H100 GroupGEMM baseline operator: {saved['op']}"
                    ) from exc
            saved.setdefault("alpha", 1.0)
            saved.setdefault("beta", 0.0)
            missing = set(_CASE_FIELDS + _PERFORMANCE_FIELDS) - set(saved)
            if missing:
                raise ValueError(
                    f"Incomplete H100 GroupGEMM baseline case: {sorted(missing)}"
                )
            key = self.key(saved)
            if key in self.cases:
                raise ValueError(f"Duplicate H100 GroupGEMM baseline case: {key}")
            row = {name: float(saved[name]) for name in _PERFORMANCE_FIELDS}
            if any(not math.isfinite(value) or value <= 0 for value in row.values()):
                raise ValueError(f"Invalid H100 GroupGEMM reference performance: {key}")
            self.cases[key] = row
        if not self.cases:
            raise ValueError("H100 GroupGEMM baseline has no matching cases")
        self.hardware = json.loads((_BASE / "hardware.json").read_text())["hygon"]

    @staticmethod
    def key(case):
        return (
            case["op"],
            case["dtype"],
            case["compute_dtype"],
            case["K"],
            case["E"],
            case["N"],
            case["M"],
            case["alpha"],
            case["beta"],
        )

    def lookup(self, op, dtype, compute_dtype, kwargs):
        if self.mode == "none":
            return None
        case = {
            "op": _COMPUTE_OPS[compute_dtype],
            "dtype": str(dtype),
            "compute_dtype": compute_dtype,
            "K": kwargs["K"],
            "E": kwargs["group_size"],
            "N": kwargs["N"],
            "M": kwargs["M"],
            "alpha": kwargs["alpha"],
            "beta": kwargs["beta"],
        }
        key = self.key(case)
        row = self.cases.get(key)
        if row is None:
            raise ValueError(f"Missing exact H100 GroupGEMM baseline case: {key}")
        return row

    def nvidia_utilizations(self, compute_dtype, saved):
        if saved is None:
            return None, None
        tensor_peak = self.hardware[_RESOURCES[compute_dtype]][
            "h100_theoretical_dense"
        ]
        memory_peak = self.hardware["memory"]["h100_theoretical"]
        return saved["tflops"] / tensor_peak, saved["gbps"] / (memory_peak * 1000)

    def factor(self, compute_dtype, saved=None):
        resource = _RESOURCES[compute_dtype]
        if self.mode == "none":
            return resource, None, "not_evaluated"
        if saved is not None:
            tensor_util, memory_util = self.nvidia_utilizations(compute_dtype, saved)
            if memory_util > tensor_util:
                resource = "memory"
        values = self.hardware[resource]
        if resource == "memory":
            h100_peak = values["h100_theoretical"]
            hygon_peak = values["hygon_measured"]
            basis = "h100_nominal_over_hygon_measured_memory"
        else:
            h100_peak = values["h100_theoretical_dense"]
            hygon_peak = values["hygon_measured_gemm"]
            basis = "h100_nominal_dense_over_hygon_measured_dense"
        factor = h100_peak / hygon_peak
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("GroupGEMM conversion factor must be finite and positive")
        return resource, factor, basis

    def describe(self):
        if self.mode == "none":
            print(
                "[conversion] reference=none; H100 latency/speed_ratio=null; "
                "performance threshold not evaluated"
            )
            return
        print(
            f"[conversion] baseline={self.baseline_id}; file={self.path}; "
            "reference=H100 saved cuBLAS grouped GEMM timings"
        )
        for compute_dtype, resource in _RESOURCES.items():
            values = self.hardware[resource]
            factor = values["h100_theoretical_dense"] / values["hygon_measured_gemm"]
            print(
                f"[conversion] {compute_dtype}: resource={resource}; "
                f"H100_dense={values['h100_theoretical_dense']:.9g} TFLOPS; "
                f"Hygon_dense={values['hygon_measured_gemm']:.9g} TFLOPS; "
                f"K={factor:.12g}"
            )
        memory = self.hardware["memory"]
        print(
            f"[conversion] memory: H100_nominal={memory['h100_theoretical']:.9g} TB/s; "
            f"Hygon_measured={memory['hygon_measured']:.9g} TB/s; "
            f"K={memory['h100_theoretical'] / memory['hygon_measured']:.12g}"
        )
        print(
            "[conversion] speed_ratio=K*T_H100/T_Hygon; "
            f"PASS iff speed_ratio >= {self.threshold:g}; "
            "per-case NVIDIA bottleneck=max(tensor/memory utilization) from "
            "saved algorithmic rates (no-ncu fallback)"
        )
        print(
            "[conversion] Torch columns are raw H100 saved values; "
            "Gems Speedup is equivalent speed_ratio, not direct speedup"
        )


def run_hygon_group_gemm(bench, compute_dtype):
    if flag_blas.vendor_name != "hygon":
        raise RuntimeError("Hygon equivalent GroupGEMM path used on another platform")
    if compute_dtype not in _RESOURCES:
        raise ValueError(f"Unsupported GroupGEMM compute dtype: {compute_dtype}")
    if Config.query:
        return bench.run()
    bench.init_user_config()
    reference = HygonGroupGemmReference()
    reference.describe()
    for dtype in bench.to_bench_dtypes:
        metrics, records = [], []
        for inputs in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            saved = reference.lookup(bench.op_name, dtype, compute_dtype, kwargs)
            resource, factor, basis = reference.factor(compute_dtype, saved)
            tensor_util, memory_util = reference.nvidia_utilizations(
                compute_dtype, saved
            )
            metric = BenchmarkMetrics()
            metric.shape_detail = bench.record_shapes(*args, **kwargs)
            operation = bench.blas_op or bench.gems_op
            metric.latency = float(bench.get_latency(operation, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError("Hygon GroupGEMM latency must be finite and positive")
            flops = bench.get_tflops(operation, *args, **kwargs)
            metric.tflops = flops / metric.latency / 1e9
            metric.gbps = bench.get_gbps(args, latency=metric.latency)
            useful_bytes = metric.gbps * metric.latency * 1e6
            raw_ratio = None
            if saved is not None:
                metric.latency_base = saved["latency_ms"]
                metric.gbps_base = saved["gbps"]
                raw_ratio = metric.latency_base / metric.latency
                metric.speedup = factor * raw_ratio
            records.append(
                {
                    "op": bench.op_name,
                    "dtype": str(dtype),
                    "compute_dtype": compute_dtype,
                    "K": kwargs["K"],
                    "E": kwargs["group_size"],
                    "N": kwargs["N"],
                    "M": kwargs["M"],
                    "m_list": kwargs["m_list"],
                    "alpha": kwargs["alpha"],
                    "beta": kwargs["beta"],
                    "h100_ms": metric.latency_base,
                    "hygon_ms": metric.latency,
                    "raw_ratio": raw_ratio,
                    "conversion_factor": factor,
                    "equivalent_score": metric.speedup,
                    "performance_status": "NO_REFERENCE"
                    if saved is None
                    else "PASS"
                    if metric.speedup >= reference.threshold
                    else "UNDERPERFORM",
                    "resource": resource,
                    "bottleneck_assumption": "not_evaluated"
                    if saved is None
                    else "h100_algorithmic_sol_fallback",
                    "nvidia_tensor_utilization": tensor_util,
                    "nvidia_memory_utilization": memory_util,
                    "normalization_basis": basis,
                    "threshold": reference.threshold,
                    "flops": flops,
                    "useful_bytes": useful_bytes,
                    "useful_gbps": metric.gbps,
                    "useful_tflops": metric.tflops,
                    "h100_tflops": saved["tflops"] if saved else None,
                    "baseline_id": reference.baseline_id,
                }
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
        reference_label = (
            "H100 saved cuBLAS grouped GEMM" if reference.mode == "h100" else "null"
        )
        print(
            "Metrics: grouped-gemm-algorithmic-v1; hygon_equivalent=true; "
            f"reference={reference.mode}; Torch={reference_label}; "
            "Gems=Hygon FlagBLAS; Gems Speedup=equivalent speed_ratio; "
            "algorithmic TFLOPS / useful GB/s (not measured HBM traffic)"
        )
        print(result, end="")
        print("[hygon-group-gemm-results] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    gc.collect()
    flag_blas.runtime.torch_device_fn.empty_cache()

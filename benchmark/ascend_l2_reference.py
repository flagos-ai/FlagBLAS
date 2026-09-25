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

import torch
import triton

import flag_blas
from benchmark.attri_util import BenchmarkMetrics, BenchmarkResult, BenchMode
from benchmark.conftest import Config, emit_record_logger
from benchmark.level2_metrics import level2_workload
from benchmark.performance_utils import Benchmark


def randn(size, *, dtype, device):
    if not dtype.is_complex:
        return torch.randn(size, dtype=dtype, device=device)
    shape = (size,) if isinstance(size, int) else tuple(size)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    parts = torch.randn((*shape, 2), dtype=real_dtype, device=device)
    parts.mul_(2**-0.5)
    return torch.view_as_complex(parts)


class AscendL2Reference:
    """Saved cuBLAS timings and explicit resource-normalization assumptions."""

    def __init__(self):
        self.mode = os.environ.get("FLAGBLAS_L2_REFERENCE", "h100")
        self.bottleneck = os.environ.get("FLAGBLAS_L2_BOTTLENECK", "memory")
        self.threshold = float(os.environ.get("FLAGBLAS_L2_THRESHOLD", "0.8"))
        if self.mode not in ("none", "h100"):
            raise ValueError("L2 reference must be none or h100")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("L2 threshold must be finite and positive")
        self.cases = {}
        self.case_fields = {}
        self.factor = None
        self.path = None
        self.baseline_id = None
        self.calibration = None
        if self.mode == "none":
            return
        project = Path(__file__).resolve().parents[1]
        relative = os.environ.get(
            "FLAGBLAS_L2_REFERENCE_FILE",
            "benchmark/baselines/h100.json",
        )
        self.path = project / relative
        self.baseline_id = self.path.stem
        rows = json.loads(self.path.read_text())
        if not rows:
            raise ValueError("L2 baseline is empty")
        for saved in rows:
            fields = tuple(sorted(set(saved) - {"latency_ms", "gbps", "tflops"}))
            op = saved["op"]
            if self.case_fields.setdefault(op, fields) != fields:
                raise ValueError(f"Inconsistent baseline parameters for {op}")
            key = self.key(saved)
            if key in self.cases:
                raise ValueError(f"Duplicate L2 baseline case: {key}")
            row = {
                field: float(saved[field]) for field in ("latency_ms", "gbps", "tflops")
            }
            if any(not math.isfinite(value) or value <= 0 for value in row.values()):
                raise ValueError(f"Invalid L2 reference performance: {key}")
            self.cases[key] = row
        self.calibration = json.loads(
            (project / "benchmark/baselines/hardware.json").read_text()
        )["ascend"]
        # Tensor entries are explanatory scenarios, not the default L2 scoring basis.
        if self.bottleneck not in ("memory", "vector"):
            raise ValueError("L2 supports memory or vector assumptions")
        resource = self.calibration[self.bottleneck]
        self.factor = resource["h100_theoretical"] / resource["ascend_measured"]
        override = os.environ.get("FLAGBLAS_L2_CONVERSION_FACTOR")
        self.overridden = override is not None
        if override is not None:
            self.factor = float(override)
        if not math.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("L2 conversion factor must be finite and positive")

    def key(self, case):
        def normalize(name, value):
            if name == "uplo":
                return {"lower": 0, "upper": 1}.get(value, value)
            if isinstance(value, complex):
                return (value.real, value.imag)
            if isinstance(value, (list, tuple)):
                return tuple(normalize("", v) for v in value)
            return value

        return tuple(
            (name, normalize(name, case[name])) for name in self.case_fields[case["op"]]
        )

    def lookup(self, op, dtype, kwargs):
        if self.mode == "none":
            return None
        if op not in self.case_fields:
            raise ValueError(f"Missing H100 L2 baseline operator: {op}")
        key = self.key(dict(kwargs, op=op, dtype=str(dtype)))
        row = self.cases.get(key)
        if row is None:
            raise ValueError(f"Missing H100 L2 baseline case: {key}")
        return row

    def describe(self):
        if self.mode == "none":
            print(
                "[conversion] reference=none; official metrics/speedup=null; performance threshold not evaluated"
            )
            return
        print(
            f"[conversion] baseline={self.baseline_id}; file={self.path}; reference=cuBLAS saved timings"
        )
        print(
            "[conversion] H100 SXM theoretical / Ascend910B4-1 measured resource capacity"
        )
        for name, resource in self.calibration.items():
            factor = resource["h100_theoretical"] / resource["ascend_measured"]
            print(
                f"[conversion] scenario={name}: K={resource['h100_theoretical']:.9g} / "
                f"{resource['ascend_measured']:.12g} {resource['unit']} = {factor:.12g}; "
                f"time_ratio_limit=K/{self.threshold:g}={factor / self.threshold:.12g} "
                "(conditional scenario, not measured case bottleneck)"
            )
        print(
            f"[conversion] selected={self.bottleneck} (assumed); K={self.factor:.12g}; override={self.overridden}"
        )
        print(
            f"[conversion] raw_speedup=T_NV/T_Ascend; speedup=K*raw_speedup; "
            f"PASS iff speedup > {self.threshold:g}; "
            f"equivalently T_Ascend < {self.factor / self.threshold:.12g} * T_NV"
        )
        print(
            "[conversion] Torch Latency/GBPS are raw H100 cuBLAS values; "
            "Gems Speedup is resource-normalized, not raw speedup"
        )


class AscendL2Benchmark(Benchmark):
    """Use saved reference timings with the normal FlagBLAS timer on Ascend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = list(self.DEFAULT_METRICS)
        self.to_bench_metrics = list(self.metrics)

    def run(self):
        if Config.query:
            return super().run()
        self.init_user_config()
        reference = AscendL2Reference()
        reference.describe()
        if Config.mode == BenchMode.KERNEL:
            # do_bench estimates iteration counts around cache.zero_() as well
            # as the operation. Ascend's first cache clear initializes runtime
            # state; including that setup can reduce warmup and sampling to a
            # single iteration. Initialize the timer's cache-clear path first,
            # without executing the operation or changing the measured region.
            driver = triton.runtime.driver.active
            cache = driver.get_empty_cache_for_benchmark()
            driver.clear_cache(cache)
            driver.get_device_interface().synchronize()
            del cache
            print("[timing] cache-clear setup initialized before do_bench estimation")
        for dtype in self.to_bench_dtypes:
            metrics = []
            records = []
            for inputs in self.get_input_iter(dtype):
                args, kwargs = self.unpack_to_args_kwargs(inputs)
                metric = BenchmarkMetrics()
                metric.shape_detail = self.record_shapes(*args, **kwargs)
                workload = level2_workload(self.metric_family, args, kwargs)
                # Only the FlagBLAS call is timed, using the original timer.
                operation = self.blas_op or self.gems_op
                metric.latency = float(self.get_latency(operation, *args, **kwargs))
                if not math.isfinite(metric.latency) or metric.latency <= 0:
                    raise ValueError("L2 latency must be finite and positive")
                metric.tflops = workload["flops"] / metric.latency / 1e9
                metric.gbps = workload["bytes"] / metric.latency / 1e6
                saved = reference.lookup(self.op_name, dtype, kwargs)
                raw_speedup = None
                if saved is not None:
                    metric.latency_base = saved["latency_ms"]
                    metric.gbps_base = saved["gbps"]
                    raw_speedup = metric.latency_base / metric.latency
                    metric.speedup = reference.factor * raw_speedup
                records.append(
                    {
                        "op": self.op_name,
                        "shape": str(
                            tuple(kwargs[k] for k in ("m", "n") if k in kwargs)
                        ),
                        "dtype": str(dtype),
                        "attrs": ", ".join(
                            f"{k}={v}"
                            for k, v in kwargs.items()
                            if isinstance(v, (str, int, float, complex))
                        ),
                        "triton_ms": metric.latency,
                        "cutensor_ms": metric.latency_base,
                        "speedup": metric.speedup,
                        "raw_speedup": raw_speedup,
                        "tflops": metric.tflops,
                        "gbps": metric.gbps,
                        "gbps_base": metric.gbps_base,
                        "tflops_base": saved["tflops"] if saved else None,
                        "flops": workload["flops"],
                        "useful_bytes": workload["bytes"],
                        "conversion_factor": reference.factor,
                        "bottleneck_assumption": reference.bottleneck,
                        "threshold": reference.threshold,
                        "performance_status": "NO_REFERENCE"
                        if metric.speedup is None
                        else "PASS"
                        if metric.speedup > reference.threshold
                        else "UNDERPERFORM",
                        "baseline_id": reference.baseline_id,
                        "ascend_l2": True,
                    }
                )
                metrics.append(metric)
                gc.collect()
            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            self._print_ascend_result(result)
            print("[ascend-l2-results] " + json.dumps(records, allow_nan=False))
            emit_record_logger(result.to_json())
        gc.collect()
        flag_blas.runtime.torch_device_fn.empty_cache()

    @staticmethod
    def _print_ascend_result(result):
        print(
            f"\nOperator: {result.op_name}  Performance Test "
            f"(dtype={result.dtype}, mode={result.mode},level={result.level})"
        )
        print(
            "Metrics: level2-scalar-useful-v1; ascend_l2=true; "
            f"reference={'h100_saved' if any(m.latency_base is not None for m in result.result) else 'null'}; "
            "algorithmic TFLOPS / useful GB/s (not measured HBM traffic)"
        )
        # Preserve the existing six metric columns and their order.
        labels = [
            "Torch Latency (ms)",
            "Gems Latency (ms)",
            "Gems Speedup",
            "TFLOPS",
            "Torch GBPS",
            "Gems GBPS",
        ]
        print(
            f"{'Status':<10}"
            + "".join(f"{v:>20}" for v in labels)
            + "          Size Detail"
        )
        print("-" * 150)
        for metric in result.result:
            values = [
                metric.latency_base,
                metric.latency,
                metric.speedup,
                metric.tflops,
                metric.gbps_base,
                metric.gbps,
            ]
            fields = ["null" if v is None else f"{v:.6f}" for v in values]
            print(
                f"{'SUCCESS':<10}"
                + "".join(f"{v:>20}" for v in fields)
                + f"          {metric.shape_detail}"
            )

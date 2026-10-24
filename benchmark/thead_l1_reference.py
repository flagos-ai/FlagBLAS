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

"""Saved H100 cuBLAS references for unavailable PPU L1 vendor paths.

Vector workloads default to the same nominal-bandwidth latency conversion as
T-Head L2: score = K * T_H100 / T_PPU. This is an assumed memory scenario, not
an observed bottleneck. Scalar ROTG uses K=1 and reports an unscaled latency
comparison: bulk bandwidth/FMA peaks cannot characterize a scalar launch.
"""

import gc
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import torch

import flag_blas
from benchmark.attri_util import BenchmarkMetrics, BenchmarkResult, BenchMode
from benchmark.conftest import Config, emit_record_logger

_BASE = Path(__file__).resolve().parent / "baselines"
_EQUIVALENT_OPS = {
    "camax",
    "zamax",
    "camin",
    "zamin",
    "caxpy",
    "zaxpy",
    "ccopy",
    "zcopy",
    "crot",
    "zrot",
    "crotg",
    "zrotg",
    "cscal",
    "zscal",
    "cswap",
    "zswap",
    "scasum",
    "dzasum",
    "scnrm2",
    "dznrm2",
    "cdotc",
    "zdotc",
    "cdotu",
    "zdotu",
    "csscal",
    "zdscal",
}
_FAMILIES = {
    **{op: op[1:] for op in _EQUIVALENT_OPS},
    "scasum": "asum",
    "dzasum": "asum",
    "scnrm2": "nrm2",
    "dznrm2": "nrm2",
    "csscal": "scal",
    "zdscal": "scal",
}


def needs_h100_l1_reference(op):
    return op.split("_", 1)[0] in _EQUIVALENT_OPS


def _number(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    return value


def case_parameters(bench, args, kwargs):
    """Keep every logical parameter; exclude runtime addresses and handles."""
    op = bench.op_name.split("_", 1)[0]
    family = _FAMILIES[op]
    if family == "rotg":
        # record_shapes retains inputs before the in-place timing mutates them.
        return {
            name: _number(complex(value)) for name, value in bench.scalar_case.items()
        }
    params = {"n": kwargs["n"], "incx": kwargs.get("incx", 1)}
    if family in ("axpy", "copy", "rot", "swap", "dotc", "dotu"):
        params["incy"] = kwargs.get("incy", 1)
    if family in ("axpy", "scal"):
        alpha = kwargs["alpha"]
        params["alpha"] = (
            _number(complex(alpha)) if op not in ("csscal", "zdscal") else alpha
        )
    if family == "rot":
        params.update(c=kwargs["c"], s=_number(complex(kwargs["s"])))
    return params


def _key(op, dtype, params):
    return op, str(dtype), json.dumps(params, sort_keys=True, allow_nan=False)


@lru_cache(maxsize=1)
def _saved_cases():
    cases = {}
    for row in json.loads((_BASE / "h100_l1.json").read_text()):
        key = _key(row["op"], row["dtype"], row["params"])
        if key in cases:
            raise ValueError(f"Duplicate H100 L1 baseline: {key}")
        if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
            raise ValueError(f"Invalid H100 L1 latency: {key}")
        cases[key] = row
    return cases


def lookup(bench, dtype, args, kwargs):
    key = _key(bench.op_name, dtype, case_parameters(bench, args, kwargs))
    try:
        return _saved_cases()[key]
    except KeyError:
        raise ValueError(f"Missing exact H100 L1 baseline: {key}") from None


def _factor(hardware, op, dtype, bottleneck):
    if op in ("crotg", "zrotg"):
        return "latency", 1.0, "unscaled_scalar_latency"
    resource = (
        "memory"
        if bottleneck == "memory"
        else (
            "vector_fp64"
            if dtype in (torch.float64, torch.complex128)
            else "vector_fp32"
        )
    )
    values = hardware[resource]
    denominator = (
        values["thead_vendor_theoretical"]
        if resource == "memory"
        else values["thead_measured"]
    )
    return (
        resource,
        values["h100_theoretical"] / denominator,
        ("nominal" if resource == "memory" else "measured_fallback"),
    )


def run_thead_l1(bench):
    if flag_blas.vendor_name != "thead":
        raise RuntimeError("T-Head L1 reference used on another platform")
    if Config.query:
        return bench.run()
    if Config.mode != BenchMode.KERNEL:
        raise ValueError("Saved H100 L1 timings require --mode kernel")
    bench.init_user_config()
    bottleneck = os.environ.get("FLAGBLAS_L1_BOTTLENECK", "memory")
    threshold = float(os.environ.get("FLAGBLAS_L1_THRESHOLD", "0.8"))
    if (
        bottleneck not in ("memory", "vector")
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("Invalid T-Head L1 bottleneck or threshold")
    hardware = json.loads((_BASE / "hardware.json").read_text())["thead"]
    print(
        "[conversion] T-Head L1: reference=H100 SXM saved cuBLAS; "
        "score=K*T_H100/T_PPU; threshold=" + str(threshold)
    )
    print(
        "[conversion] memory uses both nominal peaks; vector uses measured PPU FMA; "
        "selected=" + bottleneck + " (assumed, not profiled); "
        "scalar ROTG: K=1 (unscaled latency)"
    )
    memory = hardware["memory"]
    print(
        f"[hardware] {hardware['device']}; calibration={hardware['calibration_date']}; "
        f"memory H100_nominal={memory['h100_theoretical']} TB/s; "
        f"PPU_nominal={memory['thead_vendor_theoretical']} TB/s; "
        f"PPU_measured={memory['thead_measured']} TB/s (diagnostic only)"
    )
    for resource in ("vector_fp32", "vector_fp64"):
        values = hardware[resource]
        print(
            f"[hardware] {resource}: H100_nominal={values['h100_theoretical']} TFLOPS; "
            f"PPU_measured={values['thead_measured']} TFLOPS; PPU_nominal=unpublished"
        )
    print(
        f"[correctness] {bench.op_name}: official L1 reference unavailable for this "
        "interface; use --ref cpu for correctness separately, outside this timer"
    )
    op = bench.blas_op or bench.gems_op
    for dtype in bench.to_bench_dtypes:
        metrics, records = [], []
        resource, factor, basis = _factor(
            hardware, bench.op_name.split("_", 1)[0], dtype, bottleneck
        )
        print(
            f"[conversion] {bench.op_name}: resource={resource}; K={factor:.9g}; basis={basis}"
        )
        for inputs in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            saved = lookup(bench, dtype, args, kwargs)
            metric = BenchmarkMetrics()
            metric.shape_detail = bench.record_shapes(*args, **kwargs)
            # No vendor function, cloning, CPU work or lookup is inside the timer.
            metric.latency = float(bench.get_latency(op, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError("Invalid PPU L1 latency")
            metric.latency_base = saved["latency_ms"]
            metric.speedup = factor * metric.latency_base / metric.latency
            if resource != "latency":
                metric.gbps = bench.get_gbps(args, latency=metric.latency)
                metric.gbps_base = bench.get_gbps(args, latency=metric.latency_base)
                metric.tflops = (
                    bench.get_tflops(op, *args, **kwargs) / metric.latency / 1e9
                )
            records.append(
                dict(
                    op=bench.op_name,
                    dtype=str(dtype),
                    params=saved["params"],
                    h100_ms=metric.latency_base,
                    ppu_ms=metric.latency,
                    raw_ratio=metric.latency_base / metric.latency,
                    conversion_factor=factor,
                    equivalent_score=metric.speedup,
                    resource=resource,
                    normalization_basis=basis,
                    threshold=threshold,
                    performance_status="PASS"
                    if metric.speedup >= threshold
                    else "UNDERPERFORM",
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
            "Metrics: thead_l1=true; Torch=H100 saved cuBLAS; Gems=PPU FlagBLAS; "
            "Gems Speedup=converted latency ratio (ROTG unscaled); rates=logical useful work"
        )
        print(result, end="")
        print("[thead-l1-results] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    flag_blas.runtime.torch_device_fn.empty_cache()

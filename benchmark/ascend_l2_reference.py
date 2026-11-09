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


ASCEND_L1_FAMILIES = {
    name: family
    for family, names in {
        "amax": ("isamax", "idamax", "icamax", "izamax"),
        "amin": ("isamin", "idamin", "icamin", "izamin"),
        "asum": ("sasum", "dasum", "scasum", "dzasum"),
        "axpy": ("saxpy", "daxpy", "caxpy", "zaxpy"),
        "copy": ("scopy", "dcopy", "ccopy", "zcopy"),
        "dot": ("sdot", "ddot"),
        "dotc": ("cdotc", "zdotc"),
        "dotu": ("cdotu", "zdotu"),
        "nrm2": ("snrm2", "dnrm2", "scnrm2", "dznrm2"),
        "rot": ("srot", "drot", "csrot", "zdrot"),
        "rotg": ("srotg", "drotg", "crotg", "zrotg"),
        "rotm": ("srotm", "drotm"),
        "rotmg": ("srotmg", "drotmg"),
        "scal": ("sscal", "dscal", "cscal", "zscal", "csscal", "zdscal"),
        "swap": ("sswap", "dswap", "cswap", "zswap"),
    }.items()
    for name in names
}


def _l1_key(op, dtype, params):
    return op, str(dtype), json.dumps(params, sort_keys=True, allow_nan=False)


def _l1_parameters(bench, family, kwargs):
    if family == "rotg":
        return {
            name: [complex(value).real, complex(value).imag]
            for name, value in bench.scalar_case.items()
        }
    if family == "rotmg":
        return dict(bench.scalar_case)
    params = {"n": kwargs["n"], "incx": kwargs.get("incx", 1)}
    if family in ("axpy", "copy", "dot", "dotc", "dotu", "rot", "rotm", "swap"):
        params["incy"] = kwargs.get("incy", 1)
    if family in ("axpy", "scal"):
        value = complex(kwargs["alpha"])
        params["alpha"] = (
            value.real
            if bench.op_name.split("_", 1)[0] in ("csscal", "zdscal")
            else [value.real, value.imag]
        )
    if family == "rot":
        params.update(c=kwargs["c"], s=kwargs["s"])
    if family == "rotm":
        params["flag"] = bench.flag
    return params


def _l1_scalar(value, dtype, device):
    if dtype.is_complex:
        value = complex(value)
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        return torch.view_as_complex(
            torch.tensor([[value.real, value.imag]], dtype=real_dtype, device=device)
        )
    return torch.tensor([value], dtype=dtype, device=device)


def _l1_inputs(bench, family, dtype):
    """Keep Ascend input preparation independent of CUDA vendor handles."""
    incx, incy = getattr(bench, "incx", 1), getattr(bench, "incy", 1)
    real_dtype = (
        torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    )
    for shape in bench.shapes:
        if family == "rotg":
            if dtype.is_complex:
                a, b = complex(*shape[:2]), complex(*shape[2:])
            else:
                a, b = shape
            bench.scalar_case = {"a": str(a), "b": str(b)}
            yield (
                _l1_scalar(a, dtype, bench.device),
                _l1_scalar(b, dtype, bench.device),
                _l1_scalar(0, real_dtype, bench.device),
                _l1_scalar(0, dtype, bench.device),
                {},
            )
            continue
        if family == "rotmg":
            bench.scalar_case = dict(zip(("d1", "d2", "x1", "y1"), shape))
            values = [_l1_scalar(v, dtype, bench.device) for v in shape]
            yield (*values, torch.zeros(5, dtype=dtype, device=bench.device), {})
            continue
        n = math.prod(shape)
        x = randn(n * incx, dtype=dtype, device=bench.device)
        kwargs = {"n": n, "incx": incx}
        if family in ("amax", "amin", "asum", "nrm2"):
            result_dtype = torch.int32 if family in ("amax", "amin") else real_dtype
            kwargs["result"] = torch.zeros(1, dtype=result_dtype, device=bench.device)
            yield x, kwargs
            continue
        if family == "scal":
            kwargs["alpha"] = bench.alpha
            yield x, kwargs
            continue
        y = randn(n * incy, dtype=dtype, device=bench.device)
        kwargs["incy"] = incy
        if family in ("dot", "dotc", "dotu"):
            yield x, y, _l1_scalar(0, dtype, bench.device), kwargs
        elif family == "rotm":
            param = torch.tensor(
                [bench.flag, 0.8, 0.2, -0.3, 0.7], dtype=dtype, device=bench.device
            )
            yield x, y, param, kwargs
        else:
            if family == "axpy":
                kwargs["alpha"] = bench.alpha
            if family == "rot":
                kwargs.update(c=bench.c, s=bench.s)
            yield x, y, kwargs


def run_ascend_l1(bench):
    """Use the official benchmark timer with exact saved H100 L1 cases."""
    if flag_blas.vendor_name != "ascend":
        raise RuntimeError("Ascend L1 reference used on another platform")
    # Keep stride and scalar parameters in each case, not in the operator name.
    bench.op_name = bench.op_name.split("_", 1)[0]
    if Config.query:
        return bench.run()
    if Config.mode != BenchMode.KERNEL:
        raise ValueError("Saved H100 L1 timings require --mode kernel")
    bench.init_user_config()
    family = ASCEND_L1_FAMILIES[bench.op_name.split("_", 1)[0]]
    mode = os.environ.get("FLAGBLAS_L1_REFERENCE", "h100")
    bottleneck = os.environ.get("FLAGBLAS_L1_BOTTLENECK", "memory")
    threshold = float(os.environ.get("FLAGBLAS_L1_THRESHOLD", "0.8"))
    if mode not in ("none", "h100") or bottleneck not in ("memory", "vector"):
        raise ValueError("Invalid Ascend L1 reference or resource assumption")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("L1 threshold must be finite and positive")
    project = Path(__file__).resolve().parents[1]
    cases = {}
    if mode == "h100":
        path = project / os.environ.get(
            "FLAGBLAS_L1_REFERENCE_FILE", "benchmark/baselines/h100_l1.json"
        )
        for row in json.loads(path.read_text()):
            key = _l1_key(row["op"], row["dtype"], row["params"])
            if (
                key in cases
                or not math.isfinite(row["latency_ms"])
                or row["latency_ms"] <= 0
            ):
                raise ValueError(f"Invalid or duplicate H100 L1 reference: {key}")
            cases[key] = row
    hardware = json.loads((project / "benchmark/baselines/hardware.json").read_text())[
        "ascend"
    ]
    scalar = family in ("rotg", "rotmg") or (family == "rotm" and bench.flag == -2.0)
    resource = "latency" if scalar else bottleneck
    factor = (
        1.0
        if scalar
        else hardware[bottleneck]["h100_theoretical"]
        / hardware[bottleneck]["ascend_measured"]
    )
    override = os.environ.get("FLAGBLAS_L1_CONVERSION_FACTOR")
    if override is not None:
        factor = float(override)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("L1 conversion factor must be finite and positive")
    print(
        f"[conversion] Ascend L1: reference={mode}; score=K*T_H100/T_Ascend; "
        f"resource={resource}; K={factor:.12g}; threshold={threshold}; "
        "ROTG/ROTMG default to unscaled scalar latency"
    )
    for name in ("memory", "vector"):
        value = hardware[name]
        print(
            f"[hardware] {name}: H100_nominal={value['h100_theoretical']}; "
            f"Ascend_measured={value['ascend_measured']} {value['unit']}; "
            "same calibration as Ascend L2"
        )
    print(
        f"[correctness] {bench.op_name}: official L1 reference unavailable; "
        "validate separately with tests/run_tests.py --ref cpu, outside this timer"
    )
    driver = triton.runtime.driver.active
    cache = driver.get_empty_cache_for_benchmark()
    driver.clear_cache(cache)
    driver.get_device_interface().synchronize()
    del cache
    operation = bench.blas_op or bench.gems_op
    for dtype in bench.to_bench_dtypes:
        metrics, records = [], []
        for inputs in _l1_inputs(bench, family, dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            params = _l1_parameters(bench, family, kwargs)
            saved = None
            if mode == "h100":
                key = _l1_key(bench.op_name, dtype, params)
                if key not in cases:
                    raise ValueError(f"Missing exact H100 L1 baseline: {key}")
                saved = cases[key]
            metric = BenchmarkMetrics()
            record_kwargs = dict(kwargs)
            if family in ("rotg", "rotmg"):
                record_kwargs["scalar_case"] = bench.scalar_case
            metric.shape_detail = bench.record_shapes(*args, **record_kwargs)
            metric.latency = float(bench.get_latency(operation, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError("L1 latency must be finite and positive")
            if not scalar:
                metric.gbps = bench.get_gbps(args, latency=metric.latency)
                metric.tflops = (
                    bench.get_tflops(operation, *args, **kwargs) / metric.latency / 1e9
                )
            if saved is not None:
                metric.latency_base = saved["latency_ms"]
                metric.speedup = factor * metric.latency_base / metric.latency
                if not scalar:
                    metric.gbps_base = bench.get_gbps(args, latency=metric.latency_base)
            records.append(
                dict(
                    op=bench.op_name,
                    dtype=str(dtype),
                    params=params,
                    shape=str((params["n"],)) if "n" in params else "(1,)",
                    attrs=", ".join(f"{k}={v}" for k, v in params.items() if k != "n"),
                    triton_ms=metric.latency,
                    cutensor_ms=metric.latency_base,
                    speedup=metric.speedup,
                    raw_speedup=metric.latency_base / metric.latency if saved else None,
                    conversion_factor=factor,
                    bottleneck_assumption=resource,
                    threshold=threshold,
                    performance_status="NO_REFERENCE"
                    if saved is None
                    else "PASS"
                    if metric.speedup > threshold
                    else "UNDERPERFORM",
                    tflops=metric.tflops,
                    gbps=metric.gbps,
                    gbps_base=metric.gbps_base,
                    tflops_base=None,
                    ascend_l1=True,
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
        print("Metrics: ascend_l1=true; Torch=H100 saved cuBLAS; Gems=Ascend FlagBLAS")
        print(result, end="")
        print("[ascend-l1-results] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    flag_blas.runtime.torch_device_fn.empty_cache()

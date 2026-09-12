import os
import re
import json
import sys
import argparse
import html
import shlex
import subprocess
import statistics
import threading
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import flag_blas


OUTPUT_BASE_DIR = os.environ.get(
    "FLAGBLAS_OUTPUT_BASE_DIR", "/home/zhc/workspace"
)
OUTPUT_DATE_FORMAT = "%Y%m%d"
OUTPUT_LOG_DIR_PREFIX = "perf_test"

DEVICE_TYPE = flag_blas.device
DEVICE_LABEL = "NPU" if DEVICE_TYPE == "npu" else "GPU"
if flag_blas.vendor_name == "hygon":
    VISIBLE_DEVICE_ENV = "HIP_VISIBLE_DEVICES"
else:
    VISIBLE_DEVICE_ENV = {
        "cuda": "CUDA_VISIBLE_DEVICES",
        # Moore Threads exposes GPU selection through MUSA_VISIBLE_DEVICES.
        # Keeping this mapping here (instead of relying on CUDA_VISIBLE_DEVICES)
        # is important because TorchMUSA ignores the CUDA variable in many
        # runtime versions.
        "musa": "MUSA_VISIBLE_DEVICES",
        "npu": "ASCEND_RT_VISIBLE_DEVICES",
    }.get(DEVICE_TYPE)
VISIBLE_DEVICE_ENVS = (
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES",
    "MUSA_VISIBLE_DEVICES",
)

_CASE_LINE_RE = re.compile(r"(?:^|::\S+\s+)(PASSED|FAILED|ERROR|SKIPPED|XPASS|XFAIL)\b")
_COLLECTED_RE = re.compile(r"(\d+)\s+tests?\s+collected")
_METRIC_VALUE_RE = re.compile(
    r"(N/A|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)"
)
_SHAPE_RE = re.compile(r"torch\.Size\(\[(.*?)\]\)")

ATTR_KEYS = (
    "trans",
    "transa",
    "transb",
    "uplo",
    "diag",
    "m",
    "n",
    "k",
    "kl",
    "ku",
    "lda",
    "lda_col",
    "lda_row",
    "incx",
    "incy",
)

TRANS_NAMES = {"0": "N", "1": "T", "2": "C"}
UPLO_NAMES = {"0": "L", "1": "U"}
DIAG_NAMES = {"0": "nonunit", "1": "unit"}

try:
    from benchmark.attri_util import DEFAULT_ITER_COUNT, DEFAULT_WARMUP_COUNT
except ModuleNotFoundError:
    from attri_util import DEFAULT_ITER_COUNT, DEFAULT_WARMUP_COUNT

try:
    import torch
    torch_version = torch.__version__
except ImportError:
    torch_version = "N/A"

try:
    import triton
    triton_version = triton.__version__
    # triton_version = "N/A"
except ImportError:
    triton_version = "N/A"

LEVEL_TESTS = {
    "L1": [
        "test_abs_perf.py",
        "test_amax_perf.py",
        "test_amin_perf.py",
        "test_asum_perf.py",
        "test_axpy_perf.py",
        "test_copy_perf.py",
        "test_dot_perf.py",
        "test_dotc_perf.py",
        "test_dotu_perf.py",
        "test_nrm2_perf.py",
        "test_rot_perf.py",
        "test_scal_perf.py",
        "test_swap_perf.py",
    ],
    "L2": [
        "test_gbmv_perf.py",
        "test_gemv_perf.py",
        "test_ger_perf.py",
        "test_hbmv_perf.py",
        "test_hemv_perf.py",
        "test_her_perf.py",
        "test_her2_perf.py",
        "test_hpmv_perf.py",
        "test_hpr_perf.py",
        "test_hpr2_perf.py",
        "test_sbmv_perf.py",
        "test_spmv_perf.py",
        "test_spr_perf.py",
        "test_spr2_perf.py",
        "test_symv_perf.py",
        "test_syr_perf.py",
        "test_syr2_perf.py",
        "test_tbmv_perf.py",
        "test_tbsv_perf.py",
        "test_tpmv_perf.py",
        "test_tpsv_perf.py",
        "test_trmv_perf.py",
        "test_trsv_perf.py",
    ],
    "L3": [
        "test_gemm_perf.py",
    ],
}

# Suffix matching lets the performance report collapse dtype-specific names
# (sgemv/dgemv/cgemv, ...) into one BLAS operator family (gemv).  This is the
# aggregate requested for Level-2 runs while retaining the detailed variant
# table below it.
LEVEL2_OPERATOR_FAMILIES = tuple(
    name.removeprefix("test_").removesuffix("_perf.py")
    for name in LEVEL_TESTS["L2"]
)


def parse_device_list(value):
    device_list = [item.strip() for item in value.split(",") if item.strip()]
    if not device_list:
        raise ValueError("At least one device must be specified")
    if any(not item.isdigit() for item in device_list):
        raise ValueError("Device IDs must be non-negative integers")
    return device_list


def resolve_devices(args, parser):
    if DEVICE_TYPE == "npu":
        if args.gpus is not None:
            parser.error("--gpu/--gpus is only valid on CUDA; use --npu on Ascend")
        value = args.npus or "0"
    elif DEVICE_TYPE in ("cuda", "musa"):
        if args.npus is not None:
            parser.error(
                "--npu/--npus is only valid on Ascend; use --gpu/--gpus on CUDA/MUSA"
            )
        value = args.gpus or "0"
    else:
        parser.error(f"Unsupported runner device type: {DEVICE_TYPE}")
    try:
        return parse_device_list(value)
    except ValueError as exc:
        parser.error(str(exc))


def clear_visible_device_env(env):
    for name in VISIBLE_DEVICE_ENVS:
        env.pop(name, None)


def make_worker_env(base_env, device_id):
    env = base_env.copy()
    clear_visible_device_env(env)
    env[VISIBLE_DEVICE_ENV] = device_id
    return env


def create_log_root(now):
    output_root = os.path.join(OUTPUT_BASE_DIR, now.strftime(OUTPUT_DATE_FORMAT))
    os.makedirs(output_root, exist_ok=True)
    used = []
    for name in os.listdir(output_root):
        match = re.match(r"^(\d{3})_", name)
        if match:
            used.append(int(match.group(1)))
    index = max(used, default=0) + 1
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    while True:
        log_root = os.path.join(
            output_root, f"{index:03d}_{OUTPUT_LOG_DIR_PREFIX}_{timestamp}"
        )
        try:
            os.mkdir(log_root)
            return output_root, log_root
        except FileExistsError:
            index += 1

def strip_ansi(text):
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)

def get_prefix(dtype_str):
    mapping = {
        'float32': 's', 'float64': 'd', 'complex64': 'c', 'complex128': 'z',
        'half': 'h', 'float16': 'h', 'bfloat16': 'bf',
        'float8_e4m3fn': 'fp8', 'float8_e5m2': 'fp8',
        'float8_e4m3fnuz': 'fp8', 'float8_e5m2fnuz': 'fp8', 'float8': 'fp8'
    }
    if dtype_str in mapping:
        return mapping[dtype_str]
    for k in sorted(mapping.keys(), key=len, reverse=True):
        if k in dtype_str:
            return mapping[k]
    return ''

def normalize_test_name(name):
    name = name.strip()
    if not name:
        return ""
    if name.endswith(".py"):
        return name
    if name.startswith("test_"):
        return name + ".py"
    return f"test_{name}_perf.py"


def operator_family(name):
    """Map a dtype-qualified benchmark name to its BLAS family."""
    for family in LEVEL2_OPERATOR_FAMILIES:
        if name == family or name.endswith(family):
            return family
    return name

def expand_test_selectors(selectors):
    selected = []
    for selector in selectors:
        selector = selector.strip()
        if not selector:
            continue
        level = selector.upper()
        if level in LEVEL_TESTS:
            selected.extend(LEVEL_TESTS[level])
        else:
            selected.append(normalize_test_name(selector))
    return selected


def resolve_test_selection(all_available, level_value=None, include_value=None):
    """Resolve levels and includes, treating include as a filter on level."""
    selected_sets = []
    for option_name, option_value in (
        ("--level", level_value),
        ("--include", include_value),
    ):
        if not option_value:
            continue
        selectors = option_value.split(",")
        expanded = expand_test_selectors(selectors)
        missing = sorted(set(expanded) - set(all_available))
        if missing:
            raise ValueError(
                f"Unknown benchmark tests from {option_name}: "
                f"{', '.join(missing)}"
            )
        selected_sets.append(set(expanded))

    if not selected_sets:
        return list(all_available)

    selected = set.intersection(*selected_sets)
    if not selected:
        raise ValueError(
            "No benchmark tests match the requested --level/--include filters"
        )
    return [name for name in all_available if name in selected]

def _format_attr_value(key, value):
    value = value.strip()
    if key in ("trans", "transa", "transb"):
        return TRANS_NAMES.get(value, value)
    if key == "uplo":
        return UPLO_NAMES.get(value, value)
    if key == "diag":
        return DIAG_NAMES.get(value, value)
    return value


_MN_RE = re.compile(r"'m':\s*(\d+)\b")
_N_RE = re.compile(r"'n':\s*(\d+)\b")


def _extract_shape(detail):
    """Build the case shape from the operation's problem dimensions.

    Prefer the explicit ``m``/``n`` attrs (the real problem size, e.g. the B
    matrix for TRMM/GEMM) over the first ``torch.Size`` in the detail line,
    which is just the first tensor argument (e.g. the square triangular matrix
    A of size k×k) and does not reflect the actual problem shape.
    """
    m_match = _MN_RE.search(detail)
    n_match = _N_RE.search(detail)
    if m_match and n_match:
        return f"({m_match.group(1)}, {n_match.group(1)})"
    shape_match = _SHAPE_RE.search(detail)
    return f"({shape_match.group(1)})" if shape_match else "N/A"


def _extract_case_attrs(detail):
    attrs = []
    for key in ATTR_KEYS:
        m = re.search(rf"'{key}':\s*([^,}}]+)", detail)
        if not m:
            continue
        value = _format_attr_value(key, m.group(1))
        attrs.append(f"{key}={value}")
    return ", ".join(attrs) if attrs else "-"


def _parse_metric_line(line):
    text = line.strip()
    if not text:
        return None
    parts = text.split(None, 1)
    if len(parts) != 2 or parts[0] not in ("SUCCESS", "FAILURE", "FAILED"):
        return None

    status, rest = parts
    values = []
    while rest and len(values) < 8:
        m = _METRIC_VALUE_RE.match(rest)
        if not m:
            break
        values.append(m.group(1))
        rest = m.group(2).lstrip()
    if len(values) < 3:
        return None
    return status, values, rest


def collect_case_counts(bench_dir, test_files, base_env, markexpr=None):
    counts = {}
    env = base_env.copy()
    clear_visible_device_env(env)
    for t in test_files:
        cmd = ["pytest", "--collect-only", "-q", t]
        if markexpr:
            cmd.extend(["-m", markexpr])
        try:
            result = subprocess.run(
                cmd, cwd=bench_dir, env=env,
                capture_output=True, text=True, timeout=120,
            )
            text = result.stdout + result.stderr
            n = 0
            for line in text.splitlines():
                m = _COLLECTED_RE.search(line)
                if m:
                    n = int(m.group(1))
                    break
            counts[t] = n
        except Exception:
            counts[t] = 0
    return counts


def collect_test_cases(bench_dir, test_files, base_env, markexpr=None):
    env = base_env.copy()
    clear_visible_device_env(env)
    cmd = ["pytest", "--collect-only", "-q", *test_files]
    if markexpr:
        cmd.extend(["-m", markexpr])
    result = subprocess.run(
        cmd, cwd=bench_dir, env=env, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    selected = set(test_files)
    test_cases = []
    for line in result.stdout.splitlines():
        if "::" not in line:
            continue
        path, suffix = line.strip().split("::", 1)
        test_file = os.path.basename(path)
        if test_file in selected:
            test_cases.append(f"{test_file}::{suffix}")
    return test_cases


def shard_test_cases(test_cases, num_workers):
    shards = [[] for _ in range(num_workers)]
    for index, test_case in enumerate(test_cases):
        shards[(index // num_workers) % num_workers].append(test_case)
    return shards


class WorkerState:
    __slots__ = ("label", "total", "completed", "files_total",
                 "files_done", "current", "done", "elapsed", "returncode")

    def __init__(self, label, total, files_total):
        self.label = label
        self.total = total
        self.completed = 0
        self.files_total = files_total
        self.files_done = 0
        self.current = ""
        self.done = False
        self.elapsed = 0.0
        self.returncode = 0


def _format_bar(completed, total, width=18):
    if total == 0:
        return "[" + "-" * width + "]  --%"
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = completed * 100 // total
    return f"[{bar}] {pct:>3}%"


def _render_lines(states, lock):
    with lock:
        lines = []
        for s in states:
            status = "done" if s.done else (s.current or "idle")
            if len(status) > 18:
                status = status[:17] + "…"
            counts = f"{s.completed}/{s.total}"
            files = f"f{s.files_done}/{s.files_total}"
            lines.append(
                f"{s.label:<6} {_format_bar(s.completed, s.total)} "
                f"{counts:>9} {files:<5} {s.elapsed:6.1f}s {status}"
            )
        return lines


def _progress_renderer(states, lock, stop_event, refresh, use_ansi):
    if use_ansi:
        n = len(states)
        sys.stdout.write("\n" * n)
        sys.stdout.flush()
        while not stop_event.wait(timeout=refresh):
            sys.stdout.write(f"\x1b[{n}A")
            for line in _render_lines(states, lock):
                sys.stdout.write("\x1b[2K" + line + "\n")
            sys.stdout.flush()
        sys.stdout.write(f"\x1b[{n}A")
        for line in _render_lines(states, lock):
            sys.stdout.write("\x1b[2K" + line + "\n")
        sys.stdout.flush()
    else:
        seen = [(-1, -1, "") for _ in states]
        while not stop_event.wait(timeout=refresh):
            with lock:
                snapshot = [(i, s.completed, s.total, s.files_done,
                             s.files_total, s.current, s.done)
                            for i, s in enumerate(states)]
            for i, completed, total, fdone, ftotal, current, done in snapshot:
                key = (completed, fdone, "done" if done else current)
                if key != seen[i]:
                    seen[i] = key
                    status = "done" if done else (current or "idle")
                    print(
                        f"  {states[i].label}: cases {completed}/{total} "
                        f"files {fdone}/{ftotal} ({status})",
                        flush=True,
                    )


def _run_perf_worker(
    state, assigned, env, bench_dir, log_root, args, worker_index, test_to_logs, lock
):
    started = time.monotonic()
    grouped = {}
    for nodeid in assigned:
        grouped.setdefault(nodeid.split("::", 1)[0], []).append(nodeid)

    for test, nodeids in grouped.items():
        stem = test.replace(".py", "")
        log_path = os.path.join(log_root, f"worker_{worker_index:02d}_{stem}.log")
        with lock:
            state.current = test
            test_to_logs.setdefault(test, []).append(log_path)
        pytest_cmd = [
            "pytest", *nodeids, "-v", "-s", "--color=yes",
            f"--warmup={args.warmup}", f"--iter={args.iterations}",
        ]
        if args.skip_correctness:
            pytest_cmd.append("--skip_correctness")
        if args.markexpr:
            pytest_cmd.extend(["-m", args.markexpr])
        with open(log_path, "w", encoding="utf-8", buffering=1) as logf:
            logf.write(f"Worker: {state.label}\n")
            logf.write(f"Visible device env: {VISIBLE_DEVICE_ENV}="
                       f"{env.get(VISIBLE_DEVICE_ENV, '')}\n")
            logf.write(f"Command: {shlex.join(pytest_cmd)}\n\n")
            with subprocess.Popen(
                pytest_cmd, env=env, cwd=bench_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    logf.write(line)
                    clean = strip_ansi(line)
                    if _CASE_LINE_RE.search(clean):
                        with lock:
                            state.completed += 1
                            state.elapsed = time.monotonic() - started
                returncode = proc.wait()
            logf.write(f"\nReturn code: {returncode}\n")
        with lock:
            state.returncode = state.returncode or returncode
            state.files_done += 1
            state.elapsed = time.monotonic() - started
        if returncode != 0:
            break
    with lock:
        state.current = ""
        state.done = True
        state.elapsed = time.monotonic() - started


def parse_detailed_logs(log_path, filename):
    if not os.path.exists(log_path): return {}
    core_op = re.search(r'test_(\w+)_perf\.py', filename).group(1) if re.search(r'test_(\w+)_perf\.py', filename) else ""
    with open(log_path, "r", encoding="utf-8") as f:
        content = strip_ansi(f.read())

    blocks = re.split(r'Operator:\s+', content)
    unified_results = {}

    for block in blocks[1:]:
        lines = block.splitlines()
        header = lines[0]
        dtype_raw = re.search(r'dtype=torch\.(\w+)', header)
        if not dtype_raw: continue
        dtype_str = f"torch.{dtype_raw.group(1)}"

        prefix = get_prefix(dtype_raw.group(1))
        unified_name = f"{prefix}{core_op}" if prefix else core_op

        if unified_name not in unified_results:
            unified_results[unified_name] = {'cases': [], 'pass': 0, 'fail': 0}

        for line in lines:
            row = _parse_metric_line(line)
            if row:
                status, values, detail = row
                shape_str = _extract_shape(detail)
                attrs = _extract_case_attrs(detail)

                if status != "SUCCESS":
                    unified_results[unified_name]['fail'] += 1
                    continue

                metrics = {
                    'op': unified_name,
                    'shape': shape_str,
                    'dtype': dtype_str,
                    'attrs': attrs,
                    'triton_ms': float(values[1]),
                    'cutensor_ms': float(values[0]),
                    'speedup': float(values[2])
                }
                unified_results[unified_name]['cases'].append(metrics)
                unified_results[unified_name]['pass'] += 1
    return unified_results

def generate_html_report(all_ops_dict, log_root, timestamp):
    report_path = os.path.join(log_root, "performance_report.html")

    summary_data = []
    detail_rows = []
    pass_count = 0
    fail_count = 0

    for name, data in all_ops_dict.items():
        if data['cases']:
            avg_s = statistics.mean([c['speedup'] for c in data['cases']])
            summary_data.append([name, avg_s])
            detail_rows.extend(data['cases'])
            if data['fail'] == 0: pass_count += 1
            else: fail_count += 1
        else:
            fail_count += 1

    summary_data.sort(key=lambda x: x[0])
    speedups = [i[1] for i in summary_data]
    total_ops = len(summary_data)

    avg_s = statistics.mean(speedups) if speedups else 0
    med_s = statistics.median(speedups) if speedups else 0
    min_s, max_s = (min(speedups), max(speedups)) if speedups else (0, 0)

    c_low = len([i for i in summary_data if i[1] < 1.0])
    c_high = len([i for i in summary_data if i[1] >= 1.0])
    p_low = (c_low / total_ops * 100) if total_ops else 0
    p_high = (c_high / total_ops * 100) if total_ops else 0

    high_perf_ops = [i for i in summary_data if i[1] > 1.6]
    low_perf_ops = [i for i in summary_data if i[1] < 1.0]

    python_v = sys.version.split()[0]

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>FlagBLAS 全量算子测试详细报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 40px 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; color: white; margin-bottom: 40px; }}
        .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
        .env-info {{ display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; margin-top: 20px; }}
        .env-item {{ display: flex; align-items: center; gap: 8px; color: rgba(255,255,255,0.9); }}
        .card {{ background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); margin-bottom: 30px; overflow: hidden; }}
        .card-header {{ background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); color: white; padding: 20px 30px; font-size: 1.3rem; font-weight: 600; }}
        .card-body {{ padding: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat-item {{ background: linear-gradient(135deg, #f6f8fc 0%, #eef2f7 100%); border-radius: 12px; padding: 25px; text-align: center; transition: transform 0.3s ease; }}
        .stat-value {{ font-size: 2.5rem; font-weight: 700; color: #5a67d8; margin-bottom: 8px; }}
        .stat-value.success {{ color: #38a169; }}
        .stat-label {{ color: #718096; font-size: 0.95rem; }}
        .summary-box {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
        .summary-item {{ text-align: center; padding: 20px; background: #f7fafc; border-radius: 10px; }}
        .summary-item .value {{ font-size: 1.8rem; font-weight: 700; color: #2d3748; }}
        .distribution-chart {{ display: flex; height: 40px; border-radius: 8px; overflow: hidden; margin: 20px 0; }}
        .dist-segment {{ display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9rem; }}
        .dist-low {{ background: #f56565; }} .dist-high {{ background: #48bb78; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 15px 20px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f7fafc; color: #4a5568; text-transform: uppercase; font-size: 0.85rem; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }}
        .badge-success {{ background: #c6f6d5; color: #22543d; }}
        .badge-danger {{ background: #fed7d7; color: #742a2a; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .table-wrap {{ overflow-x: auto; }}
        .footer {{ text-align: center; color: white; margin-top: 30px; padding-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FlagBLAS 全量算子测试详细报告</h1>
            <div class="env-info">
                <div class="env-item"><span>Python: {python_v}</span></div>
                <div class="env-item"><span>Torch: {torch_version}</span></div>
                <div class="env-item"><span>Triton: {triton_version}</span></div>
                <div class="env-item"><span>Commit: N/A</span></div>
                <div class="env-item"><span>{total_ops} 个算子</span></div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">1. 概览</div>
            <div class="card-body">
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-value">{total_ops}</div><div class="stat-label">总算子数量</div></div>
                    <div class="stat-item"><div class="stat-value success">{pass_count}</div><div class="stat-label">correctness+perf 通过</div></div>
                    <div class="stat-item"><div class="stat-value">{fail_count}</div><div class="stat-label">精度测试失败</div></div>
                    <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">无精度测试用例</div></div>
                    <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">无性能结果</div></div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">2. 加速比统计（基于平均加速比）</div>
            <div class="card-body">
                <div class="summary-box">
                    <div class="summary-item"><div class="value">{med_s:.6f}</div><div class="label">中位数</div></div>
                    <div class="summary-item"><div class="value">{avg_s:.6f}</div><div class="label">平均值</div></div>
                    <div class="summary-item"><div class="value">{min_s:.6f}</div><div class="label">最小值</div></div>
                    <div class="summary-item"><div class="value">{max_s:.6f}</div><div class="label">最大值</div></div>
                </div>
                <h3 style="font-size: 1.1rem; color: #4a5568; margin-bottom: 20px;">加速比分布</h3>
                <div class="distribution-chart">
                    <div class="dist-segment dist-low" style="flex: {p_low};">{p_low:.2f}%</div>
                    <div class="dist-segment dist-high" style="flex: {p_high};">{p_high:.2f}%</div>
                </div>
                <table style="margin-top: 30px;">
                    <thead><tr><th>区间</th><th>数量</th><th>占比</th></tr></thead>
                    <tbody>
                        <tr><td><span class="badge badge-danger">&lt; 1.0</span></td><td>{c_low}</td><td>{p_low:.2f}%</td></tr>
                        <tr><td><span class="badge badge-success">&ge; 1.0</span></td><td>{c_high}</td><td>{p_high:.2f}%</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">3. 算子加速比柱状图</div>
            <div class="card-body"><div style="height: 400px;"><canvas id="speedupChart"></canvas></div></div>
        </div>

        <div class="two-col">
            <div class="card">
                <div class="card-header">4. 需关注算子（加速比 &lt; 1.0）</div>
                <div class="card-body"><table><thead><tr><th>算子名</th><th>加速比</th></tr></thead><tbody>
                {"".join([f"<tr><td>{n}</td><td><span class='badge badge-danger'>{s:.6f}</span></td></tr>" for n,s in low_perf_ops])}
                </tbody></table></div>
            </div>
            <div class="card">
                <div class="card-header">5. 高性能算子（加速比 &gt; 1.6）</div>
                <div class="card-body"><table><thead><tr><th>算子名</th><th>加速比</th></tr></thead><tbody>
                {"".join([f"<tr><td>{n}</td><td><span class='badge badge-success'>{s:.6f}</span></td></tr>" for n,s in high_perf_ops])}
                </tbody></table></div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">6. 各数据规模性能明细</div>
            <div class="card-body"><div class="table-wrap"><table><thead><tr><th>算子</th><th>shape</th><th>dtype</th><th>attrs</th><th>triton_ms</th><th>reference_ms</th><th>speedup</th></tr></thead><tbody>
                {"".join([f"<tr><td>{html.escape(c['op'])}</td><td>{html.escape(c['shape'])}</td><td>{html.escape(c['dtype'])}</td><td>{html.escape(c.get('attrs', '-'))}</td><td>{c['triton_ms']:.3f} ms</td><td>{c['cutensor_ms']:.3f} ms</td><td>{c['speedup']:.6f}x</td></tr>" for c in detail_rows])}
            </tbody></table></div></div>
        </div>

        <div class="footer"><p>生成时间: {timestamp}</p></div>
    </div>

    <script>
        const allData = {json.dumps(summary_data)};
        const ctx = document.getElementById('speedupChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: allData.map(item => item[0]),
                datasets: [{{
                    label: '加速比',
                    data: allData.map(item => item[1]),
                    backgroundColor: allData.map(item => item[1] > 1.0 ? 'rgba(72, 187, 120, 0.8)' : 'rgba(245, 101, 101, 0.8)'),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'speedup' }} }},
                    x: {{ ticks: {{ maxRotation: 60, minRotation: 45 }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path

def main():
    parser = argparse.ArgumentParser()
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--gpu", "--gpus", dest="gpus", type=str, default=None)
    device_group.add_argument("--npu", "--npus", dest="npus", type=str, default=None)
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel benchmark workers per device",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="L2",
        help="Comma-separated BLAS levels to run (default: L2). Example: --level L2 or --level L1,L3",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Comma-separated tests or levels to run. Examples: gemm, test_gemm_perf.py, L2",
    )
    parser.add_argument(
        "--generate-html-report",
        action="store_true",
        help="Generate performance_report.html after tests finish. Disabled by default.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_COUNT)
    parser.add_argument("--iter", dest="iterations", type=int, default=DEFAULT_ITER_COUNT)
    parser.add_argument(
        "--skip_correctness",
        action="store_true",
        default=False,
        help="Skip correctness checks that run before benchmark measurements.",
    )
    parser.add_argument("-m", "--markexpr", type=str, default=None)
    parser.add_argument(
        "--underperf-threshold",
        type=float,
        default=0.9,
        help="Speedup threshold below which a case is flagged as underperforming (default: 0.9)",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iterations <= 0:
        parser.error("--iter must be > 0")
    if args.parallel < 1:
        parser.error("--parallel must be >= 1")

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(bench_dir)
    device_list = resolve_devices(args, parser)

    now = datetime.now()
    timestamp_full = now.strftime("%Y-%m-%d %H:%M:%S")
    output_root, log_root = create_log_root(now)

    all_available = sorted([f for f in os.listdir(bench_dir) if f.startswith("test_") and f.endswith(".py") and f != "run_perf.py"])
    try:
        all_tests = resolve_test_selection(
            all_available,
            level_value=args.level,
            include_value=args.include,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"[*] Root: {project_root}")
    print(f"[*] Output Root: {output_root}")
    print(f"[*] Device Type: {DEVICE_TYPE}")
    print(f"[*] {DEVICE_LABEL}s: {', '.join(device_list)}")
    print(f"[*] Parallel workers per {DEVICE_LABEL}: {args.parallel}")
    if args.level:
        print(f"[*] Level: {args.level}")
    if args.include:
        print(f"[*] Include: {args.include}")
    print(f"[*] Tests: {', '.join(all_tests)}")
    if args.markexpr:
        print(f"[*] Marker: {args.markexpr}")
    print(f"[*] Warmup: {args.warmup}")
    print(f"[*] Iterations: {args.iterations}")
    print(f"[*] Skip Correctness: {args.skip_correctness}")
    print(f"[*] Logs: {log_root}\n")

    workers = [
        (device_id, local_id)
        for device_id in device_list
        for local_id in range(args.parallel)
    ]
    num_workers = len(workers)
    base_env = os.environ.copy()
    source_pythonpath = os.pathsep.join([SRC_DIR, PROJECT_ROOT])
    base_env["PYTHONPATH"] = source_pythonpath + (
        ":" + base_env.get("PYTHONPATH", "") if base_env.get("PYTHONPATH") else ""
    )
    print(f"[*] Collecting case counts for {len(all_tests)} test files...")
    test_cases = collect_test_cases(bench_dir, all_tests, base_env, args.markexpr)
    grand_total = len(test_cases)
    shards = shard_test_cases(test_cases, num_workers)
    print(f"[*] Collected {grand_total} cases across {len(all_tests)} files")

    test_to_logs = {}
    states = []
    threads = []
    state_lock = threading.Lock()

    for i, (device_id, local_id) in enumerate(workers):
        assigned = shards[i]
        label = (
            f"{DEVICE_LABEL} {device_id} w{local_id + 1}/{args.parallel}"
            if args.parallel > 1 else f"{DEVICE_LABEL} {device_id}"
        )
        total_cases = len(assigned)
        state = WorkerState(label=label, total=total_cases, files_total=len({nodeid.split("::", 1)[0] for nodeid in assigned}))
        states.append(state)
        if not assigned:
            state.done = True
            continue
        env = make_worker_env(base_env, device_id)
        print(
            f"[{label}] assigned {total_cases} cases across "
            f"{state.files_total} files"
        )
        t = threading.Thread(
            target=_run_perf_worker,
            args=(state, assigned, env, bench_dir, log_root, args, i, test_to_logs, state_lock),
            daemon=True,
        )
        threads.append(t)

    print()
    use_ansi = sys.stdout.isatty()
    stop_event = threading.Event()
    renderer = threading.Thread(
        target=_progress_renderer,
        args=(states, state_lock, stop_event, 0.3 if use_ansi else 1.0, use_ansi),
        daemon=True,
    )
    renderer.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_event.set()
    renderer.join()

    for log_paths in test_to_logs.values():
        for log_path in log_paths:
            if not os.path.exists(log_path):
                continue
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                cleaned = strip_ansi(f.read())
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

    final_summary = {}
    for test in all_tests:
        for log_path in test_to_logs.get(test, []):
            file_results = parse_detailed_logs(log_path, test)
            for op_key, data in file_results.items():
                if op_key not in final_summary:
                    final_summary[op_key] = {"cases": [], "pass": 0, "fail": 0}
                final_summary[op_key]["cases"].extend(data["cases"])
                final_summary[op_key]["pass"] += data["pass"]
                final_summary[op_key]["fail"] += data["fail"]

    # Aggregate all dtype variants into one row per Level-2 operator family.
    # Keep successful case timings so the reported averages are arithmetic
    # means over the same cases that produced the detailed speedup rows.
    family_summary = {}
    for op_name, data in final_summary.items():
        family = operator_family(op_name)
        row = family_summary.setdefault(
            family, {"cases": [], "pass": 0, "fail": 0}
        )
        row["cases"].extend(data["cases"])
        row["pass"] += data["pass"]
        row["fail"] += data["fail"]

    op_name_w = max(
        (len(n) for n in final_summary.keys()),
        default=len("Operator (Unified)"),
    )
    op_name_w = max(op_name_w, len("Operator (Unified)"))
    cases_w = max(len("Cases"), 6)
    pass_w = max(len("Pass"), 6)
    fail_w = max(len("Fail"), 6)
    speedup_w = max(len("Avg Speedup"), 8)

    summary_header = (
        f"{'Operator (Unified)':<{op_name_w}} | "
        f"{'Cases':<{cases_w}} | "
        f"{'Pass':<{pass_w}} | "
        f"{'Fail':<{fail_w}} | "
        f"{'Avg Speedup':<{speedup_w}}"
    )
    summary_lines = [summary_header, "-" * len(summary_header)]
    for name in sorted(final_summary.keys()):
        d = final_summary[name]
        avg_s = statistics.mean([c['speedup'] for c in d['cases']]) if d['cases'] else 1.0
        summary_lines.append(
            f"{name:<{op_name_w}} | "
            f"{len(d['cases'])+d['fail']:<{cases_w}} | "
            f"{d['pass']:<{pass_w}} | "
            f"{d['fail']:<{fail_w}} | "
            f"{avg_s:<{speedup_w}.4f}"
        )

    print()
    for line in summary_lines:
        print(line)

    family_lines = []
    family_name_w = max(
        (len(name) for name in family_summary),
        default=len("Operator Family"),
    )
    family_name_w = max(family_name_w, len("Operator Family"))
    family_header = (
        f"{'Operator Family':<{family_name_w}} | {'Cases':>8} | "
        f"{'Pass':>8} | {'Fail':>8} | {'Avg Triton(ms)':>15} | "
        f"{'Avg Reference(ms)':>18} | {'Avg Speedup':>12}"
    )
    family_lines.extend([
                         f"Average Performance by BLAS Operator Family ({args.level or 'all levels'})",
                         family_header,
                         "-" * len(family_header)])
    for family in sorted(family_summary):
        data = family_summary[family]
        cases = data["cases"]
        avg_triton = statistics.mean(c["triton_ms"] for c in cases) if cases else 0.0
        avg_reference = statistics.mean(c["cutensor_ms"] for c in cases) if cases else 0.0
        avg_speedup = statistics.mean(c["speedup"] for c in cases) if cases else 0.0
        family_lines.append(
            f"{family:<{family_name_w}} | {len(cases) + data['fail']:>8} | "
            f"{data['pass']:>8} | {data['fail']:>8} | {avg_triton:>15.3f} | "
            f"{avg_reference:>18.3f} | {avg_speedup:>12.4f}"
        )
    family_lines.append("-" * len(family_header))
    print()
    for line in family_lines:
        print(line)

    threshold = args.underperf_threshold
    underperf_cases = []
    for name in sorted(final_summary.keys()):
        for c in final_summary[name]['cases']:
            if c['speedup'] < threshold:
                underperf_cases.append(c)
    underperf_cases.sort(key=lambda c: c['speedup'])

    total_cases = sum(len(d['cases']) for d in final_summary.values())
    underperf_lines = [
        f"Underperforming cases (speedup < {threshold}): "
        f"{len(underperf_cases)} / {total_cases}",
    ]
    if underperf_cases:
        op_w = max(len("Op"), max(len(c['op']) for c in underperf_cases))
        shape_w = max(len("Shape"), max(len(c['shape']) for c in underperf_cases))
        dtype_w = max(len("Dtype"), max(len(c['dtype']) for c in underperf_cases))
        attrs_w = max(len("Attrs"), max(len(c.get('attrs', '-')) for c in underperf_cases))
        triton_w = max(len("Triton(ms)"), 10)
        reference_column = "Reference(ms)"
        cublas_w = max(len(reference_column), 10)
        up_speedup_w = max(len("Speedup"), 8)

        header = (
            f"{'Op':<{op_w}} | "
            f"{'Shape':<{shape_w}} | "
            f"{'Dtype':<{dtype_w}} | "
            f"{'Attrs':<{attrs_w}} | "
            f"{'Triton(ms)':>{triton_w}} | "
            f"{reference_column:>{cublas_w}} | "
            f"{'Speedup':>{up_speedup_w}}"
        )
        total_width = len(header)
        underperf_lines.append("-" * total_width)
        underperf_lines.append(header)
        underperf_lines.append("-" * total_width)
        for c in underperf_cases:
            underperf_lines.append(
                f"{c['op']:<{op_w}} | "
                f"{c['shape']:<{shape_w}} | "
                f"{c['dtype']:<{dtype_w}} | "
                f"{c.get('attrs', '-'):<{attrs_w}} | "
                f"{c['triton_ms']:>{triton_w}.3f} | "
                f"{c['cutensor_ms']:>{cublas_w}.3f} | "
                f"{c['speedup']:>{up_speedup_w}.4f}"
            )
        op_counts = {}
        for c in underperf_cases:
            op_counts[c['op']] = op_counts.get(c['op'], 0) + 1
        by_op_w = max(len(n) for n in op_counts.keys())
        underperf_lines.append("-" * total_width)
        underperf_lines.append("By op:")
        for op_name in sorted(op_counts.keys()):
            total_for_op = len(final_summary[op_name]['cases']) if op_name in final_summary else 0
            underperf_lines.append(f"  {op_name:<{by_op_w}} {op_counts[op_name]} / {total_for_op}")
    else:
        underperf_lines.append("(none — all cases meet threshold)")

    print()
    for line in underperf_lines:
        print(line)

    summary_log_path = os.path.join(log_root, "summary.log")
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write(f"Generated: {timestamp_full}\n")
        if args.level:
            f.write(f"Level: {args.level}\n")
        f.write(f"Warmup: {args.warmup}  Iterations: {args.iterations}\n")
        f.write(f"Skip Correctness: {args.skip_correctness}\n")
        if args.markexpr:
            f.write(f"Marker: {args.markexpr}\n")
        f.write(f"Device Type: {DEVICE_TYPE}\n")
        f.write(f"{DEVICE_LABEL}s: {','.join(device_list)}\n")
        f.write(f"Parallel workers per {DEVICE_LABEL}: {args.parallel}\n")
        f.write(f"Tests: {', '.join(all_tests)}\n\n")
        f.write("\n".join(summary_lines) + "\n\n")
        f.write("\n".join(family_lines) + "\n\n")
        f.write("\n".join(underperf_lines) + "\n")
    print(f"\n[*] Summary written to: {summary_log_path}")

    underperf_log_path = os.path.join(log_root, "underperforming.log")
    with open(underperf_log_path, "w", encoding="utf-8") as f:
        f.write(f"Generated: {timestamp_full}\n")
        f.write(f"Threshold: speedup < {threshold}\n\n")
        f.write("\n".join(underperf_lines) + "\n")
    print(f"[*] Underperforming cases written to: {underperf_log_path}")

    if args.generate_html_report:
        report_file = generate_html_report(final_summary, log_root, timestamp_full)
        print(f"\n[√] 性能报告已生成: {report_file}")
    else:
        print(f"\n[*] HTML report generation disabled. Use --generate-html-report to enable it.")

    if any(state.returncode != 0 for state in states):
        print(f"\n[!] One or more benchmark workers failed. Logs: {log_root}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

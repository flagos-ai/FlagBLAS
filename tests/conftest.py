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

import json
import logging
import os
from datetime import datetime

import pytest
import torch

import flag_blas

device = flag_blas.device

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"test_detail_and_result_{timestamp}.json"

DEFAULT_L1_N_START = 100
DEFAULT_L1_N_END = 1000
DEFAULT_L1_N_STEP = 100
L1_n_start = DEFAULT_L1_N_START
L1_n_end = DEFAULT_L1_N_END
L1_n_step = DEFAULT_L1_N_STEP
TO_CPU = False
QUICK_MODE = False
RECORD_LOG = False
RECORD_JSON = False
REPORT_FILE = "accuracy_result.json"


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default=device,
        required=False,
        choices=[device, "cpu"],
        help="device to run reference tests on",
    )
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        required=False,
        help="run tests on quick mode",
    )
    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log", "json"],
        help="record test results in log/json files or not",
    )
    parser.addoption(
        "--output",
        help="path to the result file",
    )

    parser.addoption(
        "--L1_n_start",
        default=DEFAULT_L1_N_START,
        help="Number of reps for each benchmark run.",
    )

    parser.addoption(
        "--L1_n_end",
        default=DEFAULT_L1_N_END,
        help="Number of reps for each benchmark run.",
    )

    parser.addoption(
        "--L1_n_step",
        default=DEFAULT_L1_N_STEP,
        help="Number of reps for each benchmark run.",
    )


def pytest_configure(config):
    global TO_CPU
    TO_CPU = config.getoption("--ref") == "cpu"
    ref_backend = "CPU (--ref cpu)" if TO_CPU else f"{device} (--ref {device})"
    print(f"[correctness] reference backend: {ref_backend}", flush=True)

    global QUICK_MODE
    QUICK_MODE = config.getoption("--quick")

    global RECORD_LOG
    RECORD_LOG = config.getoption("--record") == "log"

    global RECORD_JSON
    RECORD_JSON = config.getoption("--record") == "json"

    global REPORT_FILE
    if RECORD_JSON:
        report_file = config.getoption("--output")
        if report_file:
            REPORT_FILE = report_file

    global L1_n_start
    L1_n_start = config.getoption("--L1_n_start")

    global L1_n_end
    L1_n_end = config.getoption("--L1_n_end")

    global L1_n_step
    L1_n_step = config.getoption("--L1_n_step")

    if RECORD_LOG:
        global RUNTEST_INFO, BUILTIN_MARKS, REGISTERED_MARKERS
        RUNTEST_INFO = {}
        BUILTIN_MARKS = {
            "parametrize",
            "skip",
            "skipif",
            "xfail",
            "usefixtures",
            "filterwarnings",
            "timeout",
            "tryfirst",
            "trylast",
        }
        REGISTERED_MARKERS = {
            marker.split(":")[0].strip() for marker in config.getini("markers")
        }
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )


def pytest_runtest_teardown(item, nextitem):
    if not RECORD_LOG:
        return
    if hasattr(item, "callspec"):
        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKERS
        ]
        if len(op_marks) > 0:
            params = str(item.callspec.params)
            for op_mark in op_marks:
                if op_mark not in RUNTEST_INFO:
                    RUNTEST_INFO[op_mark] = [params]
                else:
                    RUNTEST_INFO[op_mark].append(params)
        else:
            func_name = item.function.__name__
            logging.warning("There is no mark at {}".format(func_name))


def pytest_sessionfinish(session, exitstatus):
    if RECORD_LOG:
        logging.info(json.dumps(RUNTEST_INFO, indent=2))


test_results = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    test_results[item.nodeid] = {"params": None, "result": None, "opname": None}
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params

    test_results[item.nodeid]["params"] = param_values
    # get all mark
    all_marks = [mark.name for mark in item.iter_markers()]
    # exclude marks，such as parametrize、skipif and so on
    exclude_marks = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "inplace"}
    operator_marks = [mark for mark in all_marks if mark not in exclude_marks]
    test_results[item.nodeid]["opname"] = operator_marks


def get_skipped_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    elif isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    else:
        return str(report.longrepr)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == "setup":
        if report.outcome == "skipped":
            reason = get_skipped_reason(report)
            test_results[report.nodeid]["result"] = "skipped"
            test_results[report.nodeid]["skipped_reason"] = reason

    elif report.when == "call":
        test_results[report.nodeid]["result"] = report.outcome
        if report.outcome == "skipped":
            reason = get_skipped_reason(report)
            test_results[report.nodeid]["skipped_reason"] = reason
        else:
            test_results[report.nodeid]["skipped_reason"] = None


def pytest_terminal_summary(terminalreporter):
    if not RECORD_JSON:
        return

    data = test_results
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(test_results)
        data = existing_data

    with open(REPORT_FILE, "w") as json_file:
        json.dump(data, json_file, indent=4, default=str)

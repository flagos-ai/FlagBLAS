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

"""Filter T-Head Level-2 benchmarks to supported reference modes."""


_L2_PERF_FILES = {
    "test_gbmv_perf.py",
    "test_gemv_perf.py",
    "test_ger_perf.py",
    "test_hbmv_perf.py",
    "test_hemv_perf.py",
    "test_her2_perf.py",
    "test_her_perf.py",
    "test_hpmv_perf.py",
    "test_hpr2_perf.py",
    "test_hpr_perf.py",
    "test_sbmv_perf.py",
    "test_spmv_perf.py",
    "test_spr2_perf.py",
    "test_spr_perf.py",
    "test_symv_perf.py",
    "test_syr2_perf.py",
    "test_syr_perf.py",
    "test_tbmv_perf.py",
    "test_tbsv_perf.py",
    "test_tpmv_perf.py",
    "test_tpsv_perf.py",
    "test_trmv_perf.py",
    "test_trsv_perf.py",
}


def keep_thead_l2_perf_node(nodeid: str) -> bool:
    path, _, test_name = nodeid.partition("::")
    filename = path.rsplit("/", 1)[-1]
    if filename not in _L2_PERF_FILES:
        return True
    if filename == "test_gbmv_perf.py":
        return test_name.startswith(
            (
                "test_perf_sgbmv",
                "test_perf_dgbmv",
                "test_perf_cgbmv",
                "test_perf_zgbmv",
            )
        )
    if filename == "test_gemv_perf.py":
        return test_name.startswith(
            (
                "test_perf_sgemv",
                "test_perf_dgemv",
                "test_perf_hgemv",
                "test_perf_bfgemv",
                "test_perf_cgemv",
                "test_perf_zgemv",
            )
        )
    if filename == "test_ger_perf.py":
        return test_name in {
            "test_perf_sger",
            "test_perf_dger",
            "test_perf_cgeru",
            "test_perf_cgerc",
            "test_perf_zgeru",
            "test_perf_zgerc",
        }
    if filename == "test_hpr_perf.py":
        return test_name.startswith(("test_perf_chpr", "test_perf_zhpr"))
    if filename == "test_hpr2_perf.py":
        return test_name.startswith(("test_perf_chpr2", "test_perf_zhpr2"))
    if filename == "test_her_perf.py":
        return test_name.startswith(("test_perf_her[cher-", "test_perf_her[zher-"))
    if filename == "test_her2_perf.py":
        return test_name.startswith(("test_perf_cher2", "test_perf_zher2"))
    if filename == "test_spr_perf.py":
        return test_name.startswith(("test_perf_sspr", "test_perf_dspr"))
    if filename == "test_spr2_perf.py":
        return test_name.startswith(("test_perf_sspr2", "test_perf_dspr2"))
    if filename == "test_syr_perf.py":
        return any(f"[{op}-" in test_name for op in ("ssyr", "dsyr", "csyr", "zsyr"))
    if filename == "test_syr2_perf.py":
        return test_name.startswith(("test_perf_ssyr2", "test_perf_dsyr2"))
    return False

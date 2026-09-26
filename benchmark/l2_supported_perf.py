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

"""Exclude unsupported T-Head Level-2 benchmarks."""


_THEAD_UNSUPPORTED_L2_TEST_PREFIXES = {
    "test_gemv_perf.py": ("test_perf_fp8_gemv",),
    "test_syr2_perf.py": ("test_perf_csyr2", "test_perf_zsyr2"),
}


def keep_thead_l2_perf_node(nodeid: str) -> bool:
    path, _, test_name = nodeid.partition("::")
    filename = path.rsplit("/", 1)[-1]
    unsupported = _THEAD_UNSUPPORTED_L2_TEST_PREFIXES.get(filename, ())
    return not test_name.startswith(unsupported)

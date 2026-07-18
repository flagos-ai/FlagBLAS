#!/bin/bash


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

VENDOR=${1:?"Usage: bash tools/run_tests.sh <vendor>"}
export BLAS_VENDOR=$VENDOR

echo "===================================================="
echo "Running FlagBLAS tests | Backend: $BLAS_VENDOR"
echo "===================================================="

export CUDA_VISIBLE_DEVICES=0

source tools/run_command.sh

echo "----------------------------------------------------"
echo "Running Pytest unit tests..."

run_command pytest -s tests/test_scal.py
run_command pytest -s tests/test_asum.py
run_command pytest -s tests/test_amax.py
run_command pytest -s tests/test_axpy.py
run_command pytest -s tests/test_rot.py
run_command pytest -s tests/test_gemv.py
run_command pytest -s tests/test_gemm.py

echo "===================================================="
echo "All tests passed!"
echo "===================================================="

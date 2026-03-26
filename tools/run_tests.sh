#!/bin/bash

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

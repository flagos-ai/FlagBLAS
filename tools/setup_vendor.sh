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

VENDOR=$1

SUPPORTED_VENDORS=(
  "nvidia"
  "iluvatar"
  "ascend"
  "hygon"
)
export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple"
# Public mirror backing non-vendor packages (pytest, numpy, ...), mirroring
# FlagGems: `--default-index <vendor>` + `--index <mirror>` resolve vendor
# wheels (e.g. torch==2.9.1+musa5.2.0) and plain PyPI packages together.
export MIRROR="https://mirrors.aliyun.com/pypi/simple"

valid_vendor() {
  needle=$1
  for item in "${SUPPORTED_VENDORS[@]}" ; do
    [ "$item" == "$needle" ] && return 0
  done
  return 1
}

[ "$#" -eq 1 ] || { echo "Usage: source tools/setup_vendor.sh <vendor>"; exit 1; }
valid_vendor "$VENDOR" || { echo "Invalid vendor: $VENDOR"; exit 1; }

# Source environment variables if not already set
if [ -z "$BLAS_VENDOR" ]; then
  source tools/set-env.sh "$VENDOR"
fi

echo "Installing FlagBLAS for ${VENDOR} ..."

case $VENDOR in
  nvidia)
    # Install PyTorch and Triton with CUDA support
    uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128
    # Install FlagBLAS in editable mode

    uv pip uninstall triton
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
    python3.12 -m pip install flagtree===0.5.0 $RES
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  iluvatar)
    # Install PyTorch with Corex support
    uv pip install \
      "torch>=2.6.0"

    # Install FlagBLAS in editable mode
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  ascend)
    # Install PyTorch (CPU build) and torch-npu for Ascend NPU
    uv pip install torch==2.10.0+cpu torch-npu==2.10.0 \
        --index-url https://resource.flagos.net/repository/flagos-pypi-ascend/simple

    # Install FlagTree compiler for Ascend
    uv pip uninstall triton || true
    uv pip install flagtree==0.6.0+ascend3.5 \
        --index-url https://resource.flagos.net/repository/flagos-pypi-ascend/simple

    # Install FlagBLAS in editable mode
    uv pip install -e .
    uv pip install ".[test]"
    ;;
  hygon)
    # Install PyTorch for Hygon DCU (ROCm/HIP).
    # The flagos-pypi-hygon index only hosts vendor wheels, so add a general
    # PyPI mirror (same one FlagGems uses) to resolve torch's transitive deps.
    # --index-strategy unsafe-best-match is required: under uv's default
    # first-index strategy, torch is found on the aliyun mirror and the DTK
    # build from the flagos index is never considered.
    UV_INDEX_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/simple"
    UV_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"

    uv pip install torch==2.9.0+das.opt1.dtk2604 \
        --index-url ${UV_INDEX_URL} \
        --extra-index-url ${UV_EXTRA_INDEX_URL} \
        --index-strategy unsafe-best-match || {
          echo "::error title=hygon torch install failed::uv pip install torch==2.9.0+das.opt1.dtk2604 (indexes: ${UV_INDEX_URL}, ${UV_EXTRA_INDEX_URL})"
          exit 1
        }
    echo "::warning title=hygon setup::torch installed"

    # Install FlagTree compiler for Hygon DCU
    uv pip uninstall triton || true
    uv pip install flagtree==0.5.1+hcu3.1 \
        --index-url ${UV_INDEX_URL} \
        --extra-index-url ${UV_EXTRA_INDEX_URL} \
        --index-strategy unsafe-best-match || {
          echo "::error title=hygon flagtree install failed::uv pip install flagtree==0.5.1+hcu3.1"
          exit 1
        }
    echo "::warning title=hygon setup::flagtree installed"

    # Install FlagBLAS without touching the DTK-patched torch. pyproject.toml
    # declares `torch>=2.6.0`; without --no-deps the dependency resolver
    # replaces the DTK build with the newest CUDA torch from the extra index.
    if ! uv pip install -e . --no-deps --no-build-isolation \
         --index-url ${UV_EXTRA_INDEX_URL} 2>&1 | tee /tmp/flagblas-install.log; then
      echo "::error title=hygon flagblas install failed::$(tail -8 /tmp/flagblas-install.log | tr '\n' ' ' | head -c 1500)"
      exit 1
    fi
    echo "::warning title=hygon setup::flagblas installed"

    # Test deps. `cupy-cuda12x` is excluded: it is NVIDIA-only and would pull
    # a CUDA runtime that conflicts with the DTK stack.
    # sqlalchemy/packaging/pybind11 are FlagBLAS runtime deps that were skipped
    # by the --no-deps install above (sqlalchemy is imported at module load time
    # via flag_blas.utils.models).
    # numpy must stay on 1.x: the DTK-patched torch 2.9.0 is built against the
    # numpy 1.x C API ("_ARRAY_API not found" under numpy 2.x).
    if ! uv pip install pytest numpy\<2 scipy distro gitpython pyyaml coverage pytest-md-report \
         sqlalchemy packaging pybind11 \
         --index-url ${UV_EXTRA_INDEX_URL} 2>&1 | tee /tmp/hygon-testdeps.log; then
      echo "::error title=hygon test deps install failed::$(tail -8 /tmp/hygon-testdeps.log | tr '\n' ' ' | head -c 1500)"
      exit 1
    fi
    echo "::warning title=hygon setup::testdeps installed"

    # Sanity check: make sure the DTK-patched torch survived the installs above.
    # NOTE: torch.__version__ drops the +das.opt1.dtk2604 local tag (it reports
    # "2.9.0"), so check the installed distribution version instead.
    set +e
    python - <<'PYEOF'
import sys, importlib.metadata, traceback
try:
    dist = importlib.metadata.version("torch")
    print("hygon torch dist:", dist)
    assert dist.startswith("2.9.0+das.opt1.dtk2604"), \
        f"unexpected torch distribution: {dist}"
    import torch
    print("torch.__version__:", torch.__version__)
    print("torch.version.hip:", getattr(torch.version, "hip", None))
except Exception:
    tb = traceback.format_exc()
    print(tb)
    print("::error title=hygon torch sanity check failed::" + tb.replace("%", "%25").replace("\n", "%0A"))
    sys.exit(1)
PYEOF
    SANITY_RC=$?
    set -e
    if [ $SANITY_RC -ne 0 ]; then
      echo "::error title=hygon torch sanity check failed::sanity check failed with rc=${SANITY_RC}"
      exit 1
    fi
    echo "::warning title=hygon setup::sanity check passed"

    # Mirror FlagGems' env_source: bake the DTK environment into the venv so
    # that every `source .venv/bin/activate` also loads the DTK runtime.
    # Otherwise torch.cuda init fails at import time ("Found no NVIDIA driver")
    # because the DTK libs are missing from LD_LIBRARY_PATH.
    if [ -n "$DTK_ENV" ]; then
      printf '\n# Source Hygon DTK environment (required by DTK-patched PyTorch)\n[ -f "%s" ] && source "%s" || true\n' "$DTK_ENV" "$DTK_ENV" >> .venv/bin/activate
      echo "Baked DTK environment into .venv/bin/activate: $DTK_ENV"
    fi
    ;;
  mthreads)
    # Install FlagBLAS with the full MUSA stack (MThreads S2000/S3000/S5000,
    # MUSA 5.2) in one go, FlagGems-style: the vendor index (default) serves
    # torch==2.9.1+musa5.2.0 / torch_musa / mkl, while the public mirror backs
    # the remaining test dependencies.
    uv pip install -e ".[mthreads-musa520,test]" \
      --default-index "${FLAGOS_PYPI}" \
      --index "${MIRROR}"

    # Install FlagTree (Triton-compatible compiler with MUSA backend)
    uv pip uninstall triton
    uv pip install flagtree===0.6.0+mthreads3.6 \
      --default-index "${FLAGOS_PYPI}" \
      --index "${MIRROR}"
    ;;

esac

echo "FlagBLAS installation for ${VENDOR} completed."

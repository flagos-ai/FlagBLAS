#!/bin/bash

VENDOR=$1

SUPPORTED_VENDORS=(
  "nvidia"
  "iluvatar"
)
export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple"

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
esac

echo "FlagBLAS installation for ${VENDOR} completed."

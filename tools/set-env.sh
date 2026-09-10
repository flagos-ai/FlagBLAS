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

SUPPORTED_VENDORS=(
  "nvidia"
  "iluvatar"
  "ascend"
  "hygon"
)

valid_vendor() {
  needle=$1
  for item in "${SUPPORTED_VENDORS[@]}" ; do
    [ "$item" == "$needle" ] && return 0
  done
  return 1
}

# Validate argument count
[ "$#" -eq 1 ] || { echo "Please specify <VENDOR>"; exit 1; }

VENDOR=${1}
valid_vendor "$VENDOR"
if [ "$?" != 0 ]; then
    echo "Invalid vendor '${VENDOR}' specified ..."
    echo "Please specify one of: ${SUPPORTED_VENDORS[@]}"
    exit 1
fi

export BLAS_VENDOR=$VENDOR

case $VENDOR in
  nvidia)
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
    ;;
  iluvatar)
    # Locate the real CoreX install. FlagGems' runners use the canonical
    # unversioned /usr/local/corex, but some bare runners only ship the
    # versioned dir (/usr/local/corex-4.4.0) without a symlink, so fall back
    # to globbing. The corex PyTorch wheels link against the CUDA-10.2
    # runtime shipped inside CoreX; that lib dir must be on LD_LIBRARY_PATH or
    # `import torch` dies with "undefined symbol: cudaProfilerInitialize"
    # (the system libcudart is CUDA 12+ and no longer exports that symbol).
    export COREX_ROOT=${COREX_ROOT:-}
    for _cr in /usr/local/corex /usr/local/corex-*; do
      if [ -d "$_cr" ] && { [ -d "$_cr/bin" ] || [ -d "$_cr/lib" ] || [ -d "$_cr/lib64" ]; }; then
        export COREX_ROOT="$_cr"
        break
      fi
    done
    if [ -z "$COREX_ROOT" ]; then
      echo "WARNING: no corex install found under /usr/local/corex*; corex torch will not work"
    else
      export PATH="${COREX_ROOT}/bin:${PATH}"
      for _cd in "${COREX_ROOT}/lib64" "${COREX_ROOT}/lib"; do
        if [ -d "$_cd" ]; then
          case ":${LD_LIBRARY_PATH:-}:" in
            *":$_cd:"*) ;;
            *) export LD_LIBRARY_PATH="$_cd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
          esac
        fi
      done
      # FlagGems backends.yaml sets CPATH for iluvatar as well.
      if [ -d /usr/local/cuda-10.2/include ]; then
        export CPATH=/usr/local/cuda-10.2/include
      fi
    fi
    ;;
  ascend)
    if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
      source /usr/local/Ascend/cann/set_env.sh || true
    fi
    ;;
  hygon)
    # Locate and source the Hygon DTK environment. The DTK-patched PyTorch
    # needs the DTK runtime libs (LD_LIBRARY_PATH) to detect the DCUs at
    # torch import time, so this must run before any python/torch invocation.
    export DTK_ENV=""
    for f in /opt/dtk-26.04/env.sh /opt/dtk/env.sh /usr/local/dtk/env.sh /opt/dtk-*/env.sh /usr/local/dtk-*/env.sh; do
      if [ -f "$f" ]; then
        export DTK_ENV="$f"
        source "$f" || true
        echo "Sourced Hygon DTK environment: $f"
        break
      fi
    done
    if [ -z "$DTK_ENV" ]; then
      echo "WARNING: no DTK env.sh found under /opt/dtk-26.04, /opt/dtk*, /usr/local/dtk*. torch will not see the DCUs."
    fi
    # Explicitly ensure the DTK/hyhal library dirs are on LD_LIBRARY_PATH,
    # in case env.sh does not cover them (torch._C._cuda_init needs them to
    # find the DCU driver).
    if [ -n "$DTK_ENV" ]; then
      DTK_ROOT="${DTK_ENV%/env.sh}"
      for d in "${DTK_ROOT}/lib" "${DTK_ROOT}/lib64" /opt/hyhal/lib /opt/hyhal/lib64; do
        if [ -d "$d" ]; then
          case ":$LD_LIBRARY_PATH:" in
            *":$d:"*) ;;
            *) export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
          esac
        fi
      done
    fi
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    ;;
esac

echo "Environment configured for vendor: ${VENDOR} (BLAS_VENDOR=${BLAS_VENDOR})"

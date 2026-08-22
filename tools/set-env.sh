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
  "mthreads"
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
    export COREX_ROOT=${COREX_ROOT:-/usr/local/corex}
    export PATH="${COREX_ROOT}/bin:${PATH}"
    export LD_LIBRARY_PATH="${COREX_ROOT}/lib:${LD_LIBRARY_PATH}"
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
  mthreads)
    export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
    export PATH="${MUSA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH}"
    if [ -n "${VIRTUAL_ENV}" ]; then
      export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
      SITE_PACKAGES="${VIRTUAL_ENV}/lib/python3.10/site-packages"
      # torch_musa._MUSAC.so links against the libtorch*.so inside the wheel
      export LD_LIBRARY_PATH="${SITE_PACKAGES}/torch/lib:${LD_LIBRARY_PATH}"
      export LD_LIBRARY_PATH="${SITE_PACKAGES}/triton/_C:${LD_LIBRARY_PATH}"
    fi
    # musart 5.2 no longer exports the mu* prefixed symbols that mublas 1.13.0
    # still references (e.g. muThreadExchangeStreamCaptureMode). Preload the real
    # driver (which exports 3 of the 4) plus a tiny shim for the remaining one.
    MUSA_COMPAT_SO="${MUSA_HOME}/lib/libmusart_compat.so"
    MUSA_DRIVER_SO="/usr/lib/x86_64-linux-gnu/libmusa.so.4.3.5"
    if [ -f "${MUSA_COMPAT_SO}" ] && [ -f "${MUSA_DRIVER_SO}" ]; then
      export LD_PRELOAD="${MUSA_DRIVER_SO}:${MUSA_COMPAT_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
    ;;
esac

echo "Environment configured for vendor: ${VENDOR} (BLAS_VENDOR=${BLAS_VENDOR})"

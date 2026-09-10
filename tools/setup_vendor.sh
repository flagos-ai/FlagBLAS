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
    # Install FlagTree compiler (plain build, no CUDA). The flagtree wheel
    # bundles the `triton` package that flag_blas imports at runtime, so it
    # must actually be installed or import fails later.
    # Version aligned with FlagGems' nvidia backends (flagtree==0.6.1);
    # `===` pins the exact plain 0.6.1 build (the hosted index also serves
    # vendor-tagged 0.6.1+<backend>3.6 wheels).
    uv pip uninstall triton || true
    # Use `uv pip` (not `python3.12 -m pip`): the venv is created by `uv venv`,
    # which does not seed pip, so `-m pip` always fails with
    # "No module named pip" and flagtree is never installed.
    uv pip install flagtree===0.6.1 \
        --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    uv pip install -e .
    uv pip install ".[test,nvidia-cuda128]"
    ;;

  iluvatar)
    # Corex PyTorch is published with a local version tag (+corex.4.4.0) that
    # only exists on the flagos-pypi-iluvatar index, so the `[iluvatar]` extra
    # can never resolve against plain PyPI. Dropping the index flags makes the
    # whole `[test,iluvatar]` resolve fail, which silently also skips the
    # `test` extra -- leaving the venv without torch and without coverage.
    # The flagos index only hosts vendor wheels, so add a general PyPI mirror
    # for torch's transitive deps. --index-strategy unsafe-best-match is
    # required: under uv's default first-index strategy a plain torch from the
    # mirror is picked and the corex build is never considered.
    UV_INDEX_URL="https://resource.flagos.net/repository/flagos-pypi-iluvatar/simple"
    UV_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"

    uv pip install -e .
    if ! uv pip install ".[test,iluvatar]" \
         --index-url ${UV_INDEX_URL} \
         --extra-index-url ${UV_EXTRA_INDEX_URL} \
         --index-strategy unsafe-best-match 2>&1 | tee /tmp/iluvatar-deps.log; then
      echo "::error title=iluvatar deps install failed::$(tail -8 /tmp/iluvatar-deps.log | tr '\n' ' ' | head -c 1500)"
      exit 1
    fi

    # FlagTree (which bundles the `triton` package flag_blas imports) is built
    # from source instead of installed from the prebuilt `flagtree` wheel:
    # every iluvatar wheel on the mirror (0.5.1+iluvatar3.1, 0.6.0/0.6.1/
    # 0.6.2a3+iluvatar3.6) is built on Ubuntu 24.04, so its libtriton.so
    # needs a newer runtime than this Ubuntu 22.04 runner provides ("version
    # `GLIBC_2.38' not found", then "version `GLIBCXX_3.4.32' not found" once
    # a glibc 2.34 wheel is picked). Building the iluvatar backend from source
    # links libtriton.so against the runner's own toolchain. torch is
    # installed first (above) because FlagTree's build probes it, and
    # setup.py downloads the iluvatar LLVM + plugin into ~/.flagtree/iluvatar
    # when the network is reachable.
    uv pip uninstall triton flagtree || true

    # `uv venv` does not seed pip, but FlagTree's documented source build runs
    # `python3 -m pip install . --no-build-isolation`; the pip that runs and
    # the pip that setup.py shells out to must exist. setuptools is pinned <82
    # because FlagTree's build-system.requires is `setuptools>=79.0.1,<82`
    # and --no-build-isolation validates it against the venv -- setup.sh
    # installs the newest setuptools (>=82), which would abort the build.
    uv pip install pip wheel "setuptools>=79.0.1,<82"

    FLAGTREE_SRC=${FLAGTREE_SRC:-${HOME}/FlagTree}
    if [ -d "${FLAGTREE_SRC}/.git" ]; then
      git -C "${FLAGTREE_SRC}" fetch --depth 1 origin main
      git -C "${FLAGTREE_SRC}" checkout -f FETCH_HEAD
    else
      git clone --depth 1 https://github.com/flagos-ai/FlagTree.git "${FLAGTREE_SRC}"
    fi
    if [ ! -f "${FLAGTREE_SRC}/setup.py" ]; then
      echo "::error title=flagtree checkout failed::${FLAGTREE_SRC} has no setup.py (clone/fetch of https://github.com/flagos-ai/FlagTree.git main failed)"
      exit 1
    fi
    echo "FlagTree source: ${FLAGTREE_SRC} @ $(git -C "${FLAGTREE_SRC}" rev-parse --short HEAD)"

    # setup.py fetches the iluvatar LLVM toolchain (~1.5 GiB) with urllib while
    # generating package metadata; on this runner that aborts the build before
    # it starts ("The download failed, probably due to network problems!",
    # setup_tools/utils/tools.py, 4 retries, no backoff). Pre-fetch the tarball
    # with curl into the directory FlagTree's cache looks for, so check_file()
    # finds it and the build skips its own download.
    #
    # The runner reaches the network through https_proxy and that proxy answers
    # every request for this KS3 bucket with HTTP 500 ("curl: (22) The
    # requested URL returned error: 500"). Which of the possible causes it is
    # (host missing from the proxy's allow-list, CONNECT refused, ...) cannot
    # be told from here, so try the alternatives in turn and keep the whole log
    # for the failure annotation.
    FLAGTREE_LLVM_URL=${FLAGTREE_LLVM_URL:-https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm22-x86_64_v0.6.1.tar.gz}

    # The proxy replies "Tunnel connection failed: 500 Internal Server Error"
    # for the hosts below even though they are reachable directly (the KS3
    # fetch works with --noproxy, and so does the CUDA toolchain URL). FlagTree
    # downloads with urllib, which honours no_proxy, so exempting these hosts
    # here covers both this script's curl and the build's own downloads --
    # without it build_ext dies with "<urlopen error Tunnel connection failed:
    # 500 Internal Server Error>" while fetching cuda_nvcc.
    no_proxy="${no_proxy:+${no_proxy},}baai-cp-web.ks3-cn-beijing.ksyuncs.com,developer.download.nvidia.com"
    export no_proxy
    export NO_PROXY="${no_proxy}"

    LLVM_DIR="${HOME}/.flagtree/iluvatar/iluvatar-llvm22-x86_64"
    # bin/clang rather than a bare -d: FlagTree's check_file() only tests for
    # the directory, so a half-extracted cache from an aborted run would be
    # accepted silently and the build would fail much later on a broken
    # toolchain.
    if [ ! -x "${LLVM_DIR}/bin/clang" ]; then
      mkdir -p "$(dirname "${LLVM_DIR}")"
      : > /tmp/flagtree-llvm-fetch.log
      fetched=""
      for attempt in proxy direct proxy-http; do
        url="${FLAGTREE_LLVM_URL}"
        extra=()
        case "${attempt}" in
          direct) extra=(--noproxy '*') ;;
          # Plain HTTP is proxied as an absolute-URI request instead of
          # CONNECT, which some proxies allow where CONNECT is blocked.
          proxy-http) url="${url/https:/http:}" ;;
        esac
        echo "===== curl [${attempt}] ${url} =====" >> /tmp/flagtree-llvm-fetch.log
        rm -f /tmp/iluvatar-llvm22.tar.gz
        if curl -v -fsSL --retry 2 --retry-delay 5 --connect-timeout 30 "${extra[@]}" \
             -o /tmp/iluvatar-llvm22.tar.gz "${url}" \
             >> /tmp/flagtree-llvm-fetch.log 2>&1; then
          fetched="${attempt}"
          break
        fi
      done
      if [ -z "${fetched}" ]; then
        echo "----- LLVM fetch log (tail) -----"
        tail -25 /tmp/flagtree-llvm-fetch.log
        echo "----- end of tail (full log on the runner: /tmp/flagtree-llvm-fetch.log) -----"
        # -v puts the proxy's response and curl's last line next to each other,
        # so the end of the log is the part worth quoting.
        echo "::error title=iluvatar LLVM download failed::no curl attempt reached ${FLAGTREE_LLVM_URL} [proxy=${https_proxy:-${http_proxy:-<none>}}] -> $(tail -6 /tmp/flagtree-llvm-fetch.log | tr '\n' ' ' | tail -c 900)"
        exit 1
      fi
      echo "iluvatar LLVM downloaded with curl mode: ${fetched}"
      mkdir -p "${LLVM_DIR}"
      tar xzf /tmp/iluvatar-llvm22.tar.gz -C "${LLVM_DIR}" --strip-components=1
      rm -f /tmp/iluvatar-llvm22.tar.gz
    fi
    echo "iluvatar LLVM: ${LLVM_DIR} ($(du -sh "${LLVM_DIR}" 2>/dev/null | cut -f1))"

    # FLAGTREE_BACKEND selects the iluvatar backend, MAX_JOBS the native build
    # parallelism (FlagTree's setup.py reads both). The verbose build output
    # goes to a log file (it is huge) and its tail is echoed verbatim on
    # failure, with the *end* of that tail (the exception line) repeated in the
    # annotation: squeezing the whole tail into the annotation truncated the
    # exception away, and the raw log itself is not readable without a login.
    if ! ( cd "${FLAGTREE_SRC}/" \
           && export FLAGTREE_BACKEND=iluvatar MAX_JOBS="${MAX_JOBS:-32}" \
           && python3 -m pip install . --no-build-isolation -v ) \
         > /tmp/flagtree-build.log 2>&1; then
      echo "----- last 30 lines of /tmp/flagtree-build.log -----"
      tail -30 /tmp/flagtree-build.log
      echo "----- end of tail (full log on the runner: /tmp/flagtree-build.log) -----"
      # pip's summary is the very last thing in the log and the traceback sits
      # a few lines above it, so take a window and keep its end.
      echo "::error title=flagtree source build failed::$(tail -25 /tmp/flagtree-build.log | tr '\n' ' ' | tail -c 1800)"
      exit 1
    fi
    tail -3 /tmp/flagtree-build.log

    # Sanity check: the corex torch must be importable, must see the Iluvatar
    # device, and the flagtree-built triton must load -- otherwise the test
    # step fails later with a confusing ModuleNotFoundError / import error.
    # glibc/libstdc++ and the resolved flagtree distribution are printed
    # because a triton linked against a newer toolchain than the runner's is
    # the failure mode behind "import triton" errors.
    set +e
    python - <<'PYEOF'
import importlib.metadata, traceback
try:
    dist = importlib.metadata.version("torch")
    print("iluvatar torch dist:", dist)
    assert "+corex" in dist, f"unexpected torch distribution: {dist}"
    import torch
    torch.cuda.init()
    print("torch.cuda available:", torch.cuda.is_available(),
          "| count:", torch.cuda.device_count())
    assert torch.cuda.device_count() > 0, "no iluvatar device visible to torch"
    import platform
    print("glibc:", platform.libc_ver())
    print("flagtree dist:", importlib.metadata.version("flagtree"))
    import triton
    print("triton:", triton.__version__, "from", triton.__file__)
    import triton._C.libtriton as libtriton
    print("libtriton:", libtriton.__file__)
except Exception:
    tb = traceback.format_exc()
    print(tb)
    print("::error title=iluvatar runtime sanity check failed::" + tb.replace("%", "%25").replace("\n", "%0A"))
    raise SystemExit(1)
PYEOF
    SANITY_RC=$?
    set -e
    if [ $SANITY_RC -ne 0 ]; then
      echo "::error title=iluvatar runtime sanity check failed::sanity check failed with rc=${SANITY_RC}"
      exit 1
    fi
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
esac

echo "FlagBLAS installation for ${VENDOR} completed."

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

declare -A PYTHON_SUPPORTED=(
  ["nvidia"]="3.12"
  ["iluvatar"]="3.12"
  ["ascend"]="3.11"
  ["hygon"]="3.10"
  ["mthreads"]="3.10"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

valid_vendor() {
  needle=$1
  for item in "${SUPPORTED_VENDORS[@]}" ; do
    [ "$item" == "$needle" ] && return 0
  done
  return 1
}

# Validate argument count
[ "$#" -eq 1 ] || { echo "Please specify <VENDOR>"; exit 1; }

# Validate vendor name
VENDOR=${1}
valid_vendor "$VENDOR"
if [ "$?" != 0 ]; then
    echo "Invalid vendor '${VENDOR}' specified ..."
    echo "Please specify one of: ${SUPPORTED_VENDORS[@]}"
    exit 1
fi
printf "Checking vendor ... ${VENDOR} $GREEN[OK]$NC\n"

# Source environment setup
source tools/set-env.sh "$VENDOR"

# Detect or install uv (FlagGems-style: standalone binary, no pip required)
UV_VERSION="0.11.22"
UV_MIRROR="https://resource.flagos.net/repository/flagos-filestore/utils"

printf "Checking uv ... "
export PATH="${HOME}/.local/bin:${PATH}"
if command -v uv &>/dev/null; then
  printf "uv $(uv --version | cut -d ' ' -f 2) $GREEN[OK]$NC\n"
else
  printf "${RED}NOT FOUND${NC}, installing ... "
  ARCH=$(uname -m)
  mkdir -p "$HOME/.local/bin"
  curl -sSf "${UV_MIRROR}/uv-${ARCH}-${UV_VERSION}-linux-gnu.tar.gz" \
    | tar xz -C "$HOME/.local/bin" 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv &>/dev/null || { printf "$RED[FAILED]$NC\n"; exit 1; }
  printf "$GREEN[OK]$NC\n"
fi
# Persist PATH for subsequent GitHub Actions steps
[ -n "${GITHUB_PATH:-}" ] && echo "$HOME/.local/bin" >> "$GITHUB_PATH"

# Provision the exact Python version via uv managed builds (FlagGems-style),
# so setup does not depend on any preinstalled system Python / pyenv.
expected_version=${PYTHON_SUPPORTED[$VENDOR]}
printf "Installing Python ${expected_version} ... "
uv python install "${expected_version}" --python-preference only-managed -q
if [ "$?" != 0 ]; then
  printf "$RED[FAILED]$NC\n"
  exit 1
fi
printf "$GREEN[OK]$NC\n"

# Start installation
printf "Installing FlagBLAS for ${VENDOR}\n"

printf "Creating virtual environment ... "
uv venv .venv --python "${expected_version}" --python-preference only-managed -q -c
if [ "$?" != 0 ]; then
  printf "$RED[FAILED]$NC\n"
  exit 1
else
  printf "$GREEN[OK]$NC\n"
  source .venv/bin/activate
fi

printf "Python: $(python --version) $GREEN[OK]$NC\n"

# Install build tools
printf "Installing build tools ... "
uv pip install \
  "setuptools>=64.0" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0"

if [ "$?" != 0 ]; then
  printf "$RED[FAILED]$NC\n"
  exit 1
else
  printf "$GREEN[OK]$NC\n"
fi

# Vendor-specific installation steps
source tools/setup_vendor.sh "$VENDOR"

[ "$?" == 0 ] || { echo "Failed to setup FlagBLAS"; exit 1; }

echo "FlagBLAS setup for ${VENDOR} completed successfully."

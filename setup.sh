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
)

declare -A PYTHON_SUPPORTED=(
  ["nvidia"]="3.12"
  ["iluvatar"]="3.12"
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

printf "Detecting pyenv ... "
pyenv_version=$(pyenv --version 2>/dev/null | awk '{print $NF}')
if [ "$?" != 0 ]; then
  printf "NOT FOUND $GREEN[OK]$NC\n"
else
  printf "${pyenv_version} $GREEN[OK]$NC\n"

  if [ x"$PYENV_ROOT" == x ]; then
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"
  fi
fi

# Validate Python version
printf "Checking Python version ... "
python_version=$(python --version 2>/dev/null | awk '{print $NF}')
expected_version=${PYTHON_SUPPORTED[$VENDOR]}
if [[ "$python_version" == *"$expected_version"* ]]; then
  printf "${python_version} $GREEN[OK]$NC\n"
else
  printf "${python_version}, expecting '${expected_version}.*' $RED[FAILED]$NC\n"
  exit 1
fi

# Validate uv install
printf "Checking uv ... "
uv_version=$(uv --version 2>/dev/null | cut -d ' ' -f 2)
if [ -n "$uv_version" ];  then
  printf "uv ${uv_version} ${GREEN}[OK]${NC}\n"
else
  printf "${RED}NOT FOUND${NC}\n"
  printf "Installing/upgrading pip and uv ... "
  pip install uv || exit 1
fi

# Start installation
printf "Installing FlagBLAS for ${VENDOR}\n"

printf "Creating virtual environment ... "
uv venv -q -c
if [ "$?" != 0 ]; then
  printf "$RED[FAILED]$NC\n"
  exit 1
else
  printf "$GREEN[OK]$NC\n"
  source .venv/bin/activate
fi

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

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

# Configuration parameters
mem_threshold=30000     # Minimum free memory required (MB)
sleep_time=120          # Wait time between retries (seconds)
max_wait=600            # Maximum total wait time (seconds)

# Locate ixsmi without relying on PATH: CI runners may not put the corex
# bin directory on PATH in non-interactive shells.
find_ixsmi() {
    local cmd
    cmd=$(command -v ixsmi 2>/dev/null || true)
    if [ -z "$cmd" ]; then
        for dir in /usr/local/corex/bin /usr/local/corex-*/bin; do
            if [ -x "$dir/ixsmi" ]; then
                cmd="$dir/ixsmi"
                break
            fi
        done
    fi
    if [ -z "$cmd" ]; then
        echo "Error: ixsmi not found (searched PATH and /usr/local/corex*/bin)." >&2
        return 1
    fi
    echo "$cmd"
}

IXSMI=$(find_ixsmi) || exit 1

# Add the corex library directory matching the found ixsmi (and the common
# unversioned one, if present) so that ixsmi can find its libraries.
corex_lib="$(dirname "$(dirname "$IXSMI")")/lib"
for lib_dir in /usr/local/corex/lib "$corex_lib"; do
    if [ -d "$lib_dir" ]; then
        export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
    fi
done

# Get the number of GPUs
gpu_list=$("$IXSMI" --query-gpu=name --format=csv,noheader 2>&1)
rc=$?
if [ $rc -ne 0 ]; then
    echo "Error: failed to run $IXSMI (exit code $rc):" >&2
    echo "$gpu_list" >&2
    exit 1
fi
gpu_count=$(printf '%s\n' "$gpu_list" | sed '/^[[:space:]]*$/d' | wc -l)

if [ "$gpu_count" -eq 0 ]; then
    echo "No Iluvatar GPUs detected. Please ensure you have Iluvatar GPUs installed and properly configured."
    echo "Debug: $IXSMI returned the following output:" >&2
    echo "$gpu_list" >&2
    exit 1
fi

echo "Detected $gpu_count Iluvatar GPU(s)."

"$IXSMI"

waited_time=0
while true; do
    memory_usage=$("$IXSMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)
    rc=$?
    memory_total=$("$IXSMI" --query-gpu=memory.total --format=csv,noheader,nounits 2>&1)
    rc=$((rc + $?))

    if [ $rc -ne 0 ]; then
        echo "Failed to query GPU memory information."
        echo "$memory_usage" >&2
        echo "$memory_total" >&2
        exit 1
    fi

    IFS=$'\n' read -d '' -r -a memory_usage_array <<< "$memory_usage"
    IFS=$'\n' read -d '' -r -a memory_total_array <<< "$memory_total"

    available_gpus=()

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for ((i=0; i<$gpu_count; i++)); do
        used_i=${memory_usage_array[$i]}
        total_i=${memory_total_array[$i]}
        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -ge $mem_threshold ]; then
            available_gpus+=($i)
        fi
    done

    if [ ${#available_gpus[@]} -gt 0 ]; then
        AVAILABLE_GPUS=$(IFS=,; echo "${available_gpus[*]}")
        echo "Available GPUs: ${AVAILABLE_GPUS}"
        break
    fi

    echo "No GPU has sufficient memory, waiting for $sleep_time seconds..."
    sleep $sleep_time
    waited_time=$((waited_time + sleep_time))
    if [ $waited_time -ge $max_wait ]; then
        echo "Error: Timed out waiting for available GPU."
        exit 1
    fi
done

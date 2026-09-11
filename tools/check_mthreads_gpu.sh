#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

command -v mthreads-gmi >/dev/null 2>&1 || {
  echo "Error: mthreads-gmi command not found. Install the Moore Threads runtime."
  exit 1
}
mthreads-gmi >/dev/null

python - <<'PYEOF'
import torch

musa = getattr(torch, "musa", None)
if musa is None or not musa.is_available() or musa.device_count() < 1:
    raise SystemExit("No TorchMUSA device is available")
print(f"Detected {musa.device_count()} MUSA device(s)")
PYEOF

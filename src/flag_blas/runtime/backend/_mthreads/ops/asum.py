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

import torch

from .nrm2 import _common_asum, _run_reduction


def sasum(n, x, incx, result):
    _run_reduction(n, x, incx, result, torch.float32, torch.float32, square=False)


def dasum(n, x, incx, result):
    _run_reduction(n, x, incx, result, torch.float64, torch.float64, square=False)


def scasum(n, x, incx, result):
    _run_reduction(n, x, incx, result, torch.complex64, torch.float32, square=False)


def dzasum(n, x, incx, result):
    if n > 8192:
        return _common_asum.dzasum(n, x, incx, result)
    _run_reduction(n, x, incx, result, torch.complex128, torch.float64, square=False)

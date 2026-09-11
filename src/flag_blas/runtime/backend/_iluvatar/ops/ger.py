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

import importlib

import torch
import triton

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn


_common = importlib.import_module("flag_blas.ops.level2.ger")


def _match_config(configs, block_m, block_n):
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_SIZE_M"] == block_m
        and config.kwargs["BLOCK_SIZE_N"] == block_n
    ]


def _prune_sger_configs(configs, named_args, **kwargs):
    m, n = named_args["m"], named_args["n"]
    if m * n <= 4096:
        return _match_config(configs, 64, 32)
    if n > 1024:
        return _match_config(configs, 1, 2048)
    return configs


def _prune_cger_configs(configs, named_args, **kwargs):
    m, n = named_args["m"], named_args["n"]
    if m * n <= 4096:
        return _match_config(configs, 16, 64)
    if n >= 1024:
        return _match_config(configs, 1, 1024)
    return configs


sger_kernel = triton.autotune(
    configs=runtime.get_tuned_config("sger_iluvatar"),
    key=["m", "n", "LDA", "INCX", "INCY"],
    restore_value=["A_ptr"],
    prune_configs_by={"early_config_prune": _prune_sger_configs},
)(_common.sger_kernel.fn.fn)

cger_kernel = triton.autotune(
    configs=runtime.get_tuned_config("cger_iluvatar"),
    key=["m", "n", "LDA", "INCX", "INCY", "CONJ_Y"],
    restore_value=["A_ptr"],
    prune_configs_by={"early_config_prune": _prune_cger_configs},
)(_common.cger_kernel.fn.fn)


def sger(m, n, alpha, x, incx, y, incy, A, lda):
    if not _common._check_ger_common(
        m, n, x, incx, y, incy, A, lda, torch.float32
    ):
        return
    alpha = _common._scalar_to_float(alpha)
    if alpha == 0.0:
        return
    with torch_device_fn.device(A.device):
        sger_kernel[_common._grid(m, n)](
            x, y, A, alpha, m, n, incx, incy, lda
        )


def _cger(m, n, alpha, x, incx, y, incy, A, lda, conj_y):
    if not _common._check_ger_common(
        m, n, x, incx, y, incy, A, lda, torch.complex64
    ):
        return
    alpha_real, alpha_imag = _common._scalar_to_complex_parts(alpha)
    if alpha_real == 0.0 and alpha_imag == 0.0:
        return
    with torch_device_fn.device(A.device):
        cger_kernel[_common._grid(m, n)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            torch.view_as_real(A),
            alpha_real,
            alpha_imag,
            m,
            n,
            incx,
            incy,
            lda,
            conj_y,
        )


def cgeru(m, n, alpha, x, incx, y, incy, A, lda):
    _cger(m, n, alpha, x, incx, y, incy, A, lda, False)


def cgerc(m, n, alpha, x, incx, y, incy, A, lda):
    _cger(m, n, alpha, x, incx, y, incy, A, lda, True)

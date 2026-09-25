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

"""Compatibility helpers for different Triton builds.

FlagBLAS's ``_thead`` (T-Head PPU) kernels use vendor-only ``triton.jit``
keyword arguments such as ``ppu_hint`` to select forward/backward codegen
specialization. Those keywords are only understood by the vendor Triton build;
other Triton builds (e.g. FlagTree's unified multi-backend Triton) raise
``TypeError`` when they meet them.

``patch_triton_jit`` wraps ``triton.jit`` so that vendor-only keywords are
forwarded when the active Triton supports them, and silently dropped otherwise.
It must be called before any kernel module using such keywords is imported.
"""

import inspect
import logging

import triton

logger = logging.getLogger(__name__)

# Vendor-only ``triton.jit`` keywords that are not part of the upstream API.
_VENDOR_ONLY_JIT_KWARGS = ("ppu_hint",)

_patched = False
_reported = set()


def patch_triton_jit():
    """Make ``triton.jit`` tolerate vendor-only keyword arguments.

    The patch is a no-op when the active ``triton.jit`` already accepts every
    vendor-only keyword or accepts ``**kwargs``.
    """
    global _patched
    if _patched:
        return

    jit = triton.jit
    try:
        params = inspect.signature(jit).parameters
    except (TypeError, ValueError):
        # Signature not inspectable; leave triton.jit untouched.
        _patched = True
        return

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        _patched = True
        return

    vendor_kwargs = {k for k in _VENDOR_ONLY_JIT_KWARGS if k not in params}
    if not vendor_kwargs:
        _patched = True
        return

    def wrapped(fn=None, **kwargs):
        for name in vendor_kwargs & kwargs.keys():
            if name not in _reported:
                _reported.add(name)
                logger.info(
                    "Active triton.jit does not support %r; ignoring it "
                    "(FlagBLAS kernels fall back to default codegen).",
                    name,
                )
            kwargs.pop(name)
        if fn is not None:
            return jit(fn, **kwargs)
        return lambda func: jit(func, **kwargs)

    triton.jit = wrapped
    _patched = True

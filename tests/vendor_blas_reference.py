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

import flag_blas

if flag_blas.vendor_name == "mthreads":
    from .mublas_reference import (
        MuComplex as HipComplex,
        MuDoubleComplex as HipDoubleComplex,
        check_mublas_status as check_hipblas_status,
        get_mublas_context as get_hipblas_context,
    )
else:
    from .hipblas_reference import (  # noqa: F401
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
    )

__all__ = [
    "HipComplex",
    "HipDoubleComplex",
    "check_hipblas_status",
    "get_hipblas_context",
]

import math

import triton

from flag_blas.runtime.shape_classifier import ShapeClassifier

_THIN_M_THRESHOLD = 64
_THIN_N_THRESHOLD = 64
_SPLITK_K_THRESHOLD = 256
_THIN_GRID_LIMIT = 32
_LARGE_MN_THRESHOLD = 1024


def _is_thin(m, n, k):
    if k < _SPLITK_K_THRESHOLD:
        return False
    if min(m, n) > _THIN_M_THRESHOLD:
        return False
    grid_mn = triton.cdiv(m, 128) * triton.cdiv(n, 32)
    return grid_mn < _THIN_GRID_LIMIT


def _is_large_mn(m, n, k):
    return m > _LARGE_MN_THRESHOLD and n > _LARGE_MN_THRESHOLD and k > _LARGE_MN_THRESHOLD


GEMM_SHAPE_RULES = {
    "thin_mn_large_k": lambda m, n, k, **_kw: _is_thin(m, n, k),
    "large_mn": lambda m, n, k, **_kw: _is_large_mn(m, n, k),
}

GEMM_CLASSIFIER = ShapeClassifier(rules=GEMM_SHAPE_RULES, default="balanced")


def gemm_shape_cache_key(**shape_args):
    m = shape_args.get("m", 0)
    n = shape_args.get("n", 0)
    k = shape_args.get("k", 0)
    shape_class = GEMM_CLASSIFIER.classify(**shape_args)

    if shape_class == "thin_mn_large_k":
        return (
            "thin_mn_large_k",
            m,
            n,
            math.ceil(math.log2(max(k, 1))),
        )
    elif shape_class == "large_mn":
        return (
            "large_mn",
            math.ceil(m / 512),
            math.ceil(n / 512),
            math.ceil(k / 512),
        )
    else:
        return (
            "balanced",
            math.ceil(math.log2(max(m, 1))),
            math.ceil(math.log2(max(n, 1))),
            math.ceil(math.log2(max(k, 1))),
        )

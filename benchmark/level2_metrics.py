"""Logical work and useful-byte models shared by all Level 2 benchmarks.

GEMV/GBMV/SYMV/... families use the Level 2 counts described below. The GEMM
family (used when an Ascend GEMM is compared with the saved H100 baseline) uses
the plain 2*m*n*k / A+B+2C model so both sides share one numerator.

These are algorithmic rates, not hardware instruction or DRAM counters. A real
add/multiply/divide costs 1 FLOP, a complex multiply 6 and add 2. Complex division
costs 11 (the textbook real/imaginary formula); conjugation, assignment, casts
and address arithmetic are not counted. Fused multiply-add counts as 2.

MV uses row dot products followed by alpha/beta scaling. Rank updates scale
vectors once outside the outer product (the shorter vector for GER); Hermitian
diagonals use real arithmetic. Triangular solves use scalar substitution. Thus
these counts include linear terms and can differ from BLAS tables retaining
only leading terms, or from the extra work of a particular blocked kernel.

Useful bytes count each logical input element once and each output write once,
including output reads when needed. Padding, stride gaps, unused matrix halves,
band corner slots and implicit unit diagonals are excluded. Hermitian diagonal
storage is charged as a complex element. Caches, repeated loads, temporary
buffers and memory transaction granularity are deliberately not modeled.

This is a shared FlagBLAS/reference comparison convention, not an NVIDIA or
AMD official benchmark formula. Both latencies use the same FLOP numerator.
"""

METRIC_CONVENTION = "level2-scalar-useful-v1"

MV = frozenset({"gemv", "gbmv", "symv", "hemv", "spmv", "hpmv", "sbmv", "hbmv"})
TRIANGULAR = frozenset({"trmv", "tpmv", "tbmv", "trsv", "tpsv", "tbsv"})
RANK1 = frozenset({"syr", "spr", "her", "hpr"})
RANK2 = frozenset({"syr2", "spr2", "her2", "hpr2"})
# Level 3 GEMM: one multiply plus one add per element of the k-reduction, i.e.
# 2*m*n*k FLOPs (alpha/beta scaling is not modeled, same as the GEMM benchmark
# tolerances). Useful bytes are A + B + read/write of C.
GEMM = frozenset({"gemm"})
FAMILIES = MV | TRIANGULAR | RANK1 | RANK2 | GEMM | {"ger"}


def triangle_elements(n, k=None):
    if n <= 0:
        return 0
    k = n - 1 if k is None else min(max(k, 0), n - 1)
    return n * (k + 1) - k * (k + 1) // 2


def general_band_elements(m, n, kl, ku):
    """Count valid matrix entries, for rectangular matrices as well as square."""
    return sum(
        max(0, min(n, m - d) - max(0, -d))
        for d in range(-min(ku, max(0, n - 1)), min(kl, max(0, m - 1)) + 1)
    )


def _dimensions(family, parameters):
    if family not in FAMILIES:
        raise ValueError(f"Unknown Level 2 operator: {family}")
    n = parameters["n"]
    m = parameters["m"] if family in {"gemv", "gbmv", "ger"} | GEMM else n
    return m, n


def _mv_work(family, m, n, parameters):
    if family == "gbmv":
        kl, ku = parameters["kl"], parameters["ku"]
        entries = general_band_elements(m, n, kl, ku)
        active_rows, active_cols = min(m, n + kl), min(n, m + ku)
    elif family in {"sbmv", "hbmv"}:
        entries = 2 * triangle_elements(n, parameters["k"]) - n
        active_rows = active_cols = n
    else:
        entries = m * n
        active_rows, active_cols = m, n
    transposed = family in {"gemv", "gbmv"} and parameters.get("trans", 0) != 0
    return (entries, n, m, active_cols) if transposed else (entries, m, n, active_rows)


def level2_flops(family, args, parameters):
    """Return FLOPs per call, not FLOPs/s. All dimensions come from BLAS args."""
    m, n = _dimensions(family, parameters)
    if m <= 0 or n <= 0:
        return 0
    if family in GEMM:
        k = parameters.get("k", 0)
        return 2 * m * n * k if k > 0 else 0
    complex_data = args[0].dtype.is_complex
    multiply, add = (6, 2) if complex_data else (1, 1)
    mac = multiply + add
    alpha, beta = parameters.get("alpha", 1), parameters.get("beta", 0)

    if family in MV:
        entries, out_len, _, active_rows = _mv_work(family, m, n, parameters)
        flops = 0
        if alpha != 0:
            flops = multiply * entries + add * (entries - active_rows)
            if family in {"hemv", "hpmv", "hbmv"}:
                flops -= 4 * n  # Real diagonal times complex x costs 2, not 6.
            if alpha != 1:
                flops += multiply * out_len
        if beta not in (0, 1):
            flops += multiply * out_len
        if alpha != 0 and beta != 0:
            flops += add * out_len
        return flops

    if family in TRIANGULAR:
        stored = triangle_elements(
            n, parameters["k"] if family.startswith("tb") else None
        )
        off_diagonal = stored - n
        unit = parameters.get("diag", 0) == 1
        if family.endswith("sv"):
            division = 11 if complex_data else 1
            return mac * off_diagonal + (0 if unit else division * n)
        # Unit diagonal starts each row with x[i], then adds off-diagonal terms.
        return mac * off_diagonal + (0 if unit else multiply * n)

    if alpha == 0:
        return 0
    scale = 0 if alpha == 1 else multiply
    if family == "ger":
        return mac * m * n + scale * min(m, n)
    triangle = triangle_elements(n)
    if family in {"her", "hpr"}:
        # Scale x by real alpha; real diagonal product+update costs 4 FLOPs.
        return 8 * (triangle - n) + 4 * n + (0 if alpha == 1 else 2 * n)
    if family in {"her2", "hpr2"}:
        # Diagonal: A_ii += 2*Re((alpha*x_i)*conj(y_i)), costing 3+1+1.
        return 16 * (triangle - n) + 5 * n + 2 * scale * n
    updates = 2 if family in RANK2 else 1
    return updates * (mac * triangle + scale * n)


def _io_tensors(family, args, parameters, *, reference=False):
    """Logical storage used by each side, excluding conversion temporaries."""
    if reference and family == "gemv" and "A_col_f32" in parameters:
        return (parameters["A_col_f32"], parameters["x_f32_ref"], args[2])
    return args


def level2_bytes(family, args, parameters, *, reference=False):
    """Return useful bytes per call; reference=True handles FP32-vs-FP8 GEMV."""
    m, n = _dimensions(family, parameters)
    if m <= 0 or n <= 0:
        return 0
    if family in GEMM:
        k = parameters.get("k", 0)
        if k <= 0:
            return 0
        a, b, c = args[0], args[1], args[2]
        return (
            m * k * a.element_size()
            + k * n * b.element_size()
            + 2 * m * n * c.element_size()
        )
    matrix, x = _io_tensors(family, args, parameters, reference=reference)[:2]
    matrix_size, x_size = matrix.element_size(), x.element_size()
    alpha, beta = parameters.get("alpha", 1), parameters.get("beta", 0)
    if family in MV:
        entries, out_len, in_len, _ = _mv_work(family, m, n, parameters)
        if alpha == 0 and beta == 1:
            return 0
        if family in {"symv", "hemv", "spmv", "hpmv"}:
            entries = triangle_elements(n)
        elif family in {"sbmv", "hbmv"}:
            entries = triangle_elements(n, parameters["k"])
        inputs = 0 if alpha == 0 else entries * matrix_size + in_len * x_size
        return inputs + out_len * args[2].element_size() * (1 + (beta != 0))
    if family in TRIANGULAR:
        entries = triangle_elements(
            n, parameters["k"] if family.startswith("tb") else None
        )
        if parameters.get("diag", 0) == 1:
            entries -= n
        return entries * matrix_size + 2 * n * x_size
    if alpha == 0:
        return 0
    if family == "ger":
        return 2 * m * n * matrix_size + m * x_size + n * args[2].element_size()
    result = 2 * triangle_elements(n) * matrix_size + n * x_size
    if family in RANK2:
        result += n * args[2].element_size()
    return result


def level2_workload(family, args, parameters):
    """Auditable per-call numerators shared by both timed implementations."""
    io = _io_tensors(family, args, parameters)
    ref_io = _io_tensors(family, args, parameters, reference=True)
    dtypes = [str(t.dtype) for t in io]
    ref_dtypes = [str(t.dtype) for t in ref_io]
    return {
        "convention": METRIC_CONVENTION,
        "flops": level2_flops(family, args, parameters),
        "bytes": level2_bytes(family, args, parameters),
        "bytes_base": level2_bytes(family, args, parameters, reference=True),
        "io_dtypes": dtypes,
        "io_dtypes_base": ref_dtypes,
        "comparison": (
            "same-storage-dtypes" if dtypes == ref_dtypes else "mixed-storage-dtypes"
        ),
    }


class Level2MetricsMixin:
    """Use before Benchmark in the MRO; subclasses specify metric_family."""

    metric_convention = METRIC_CONVENTION

    def get_metric_workload(self, args, kwargs):
        return level2_workload(self.metric_family, args, kwargs)

    def get_tflops(self, op, *args, **kwargs):
        return level2_flops(self.metric_family, args, kwargs)

    def get_gbps(self, args, latency, *, parameters, reference=False):
        if latency <= 0:
            raise ValueError("Bandwidth requires a positive latency in milliseconds")
        return (
            level2_bytes(self.metric_family, args, parameters, reference=reference)
            / latency
            / 1e6
        )

    def get_bandwidth(self, args, kwargs, latency, *, reference=False):
        return self.get_gbps(args, latency, parameters=kwargs, reference=reference)

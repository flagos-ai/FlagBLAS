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

"""Hand-written MUSA (mcc) fp64 GEMM kernel for large shapes.

Why this exists
---------------
The Triton fp64 kernel in ``dgemm.py`` accumulates with explicit outer products
and reaches ~3.9 TFLOPS; ``tl.dot`` with ``input_precision="ieee"`` only reaches
2.1 TFLOPS because its lowering round-trips both operands through shared memory
every k iteration.  Neither can go much further: Triton's fp64 code generation
tops out well short of the ALU (a register-fed fp64 FMA microbenchmark compiled
by mcc measures 13.6 TFLOPS on this part, the vendor library 13-14).

Writing the kernel in MUSA C++ instead clears the Triton code-generation
ceiling.  Two things turned out to matter, both measured on MTT S5000:

* **Column ownership.**  Giving thread ``tx`` the contiguous N-range
  ``[tx*TN, tx*TN+TN)`` makes its ``double2`` chunks sit ``TN`` doubles apart,
  so a warp's ``LDS.128`` touches ~4x the shared-memory bytes it actually needs
  and the kernel becomes load/store bound.  Spreading each thread's column pairs
  across the tile (``tx + j*BDX``) makes consecutive ``tx`` read consecutive
  chunks: +45% on the inner loop.
* **Register blocking and the shared-memory wall.**  Shared-memory reads cost
  roughly 128 bytes/cycle/SM, so the quantity that matters is bytes per FLOP,
  ``4*(1/TM + 1/TN)``.  A 4x4 tile spends 2.0 B/FLOP and caps out near 6.8
  TFLOPS, an 8x4 tile spends 1.5 but runs into the 32-accumulator issue limit
  at ~7.1, and only 8x8 (1.0 B/FLOP) gets under the wall.  8x8 means 64 fp64
  accumulators = 128 registers, and with the k loop fully unrolled the compiler
  software-pipelines the operand loads, overruns the register file and spills
  ~500 bytes, which collapses it to under 1 TFLOPS.  Leaving that one loop
  rolled (``#pragma unroll 1``) keeps the kernel at ~200 registers and lifts the
  inner loop from 7.5 to 10.6 TFLOPS.

Together those give 5.7 TFLOPS with the LDG staging, and 9.1 / 9.4 TFLOPS at
4096^3 / 8192^3 once ``_TME_SRC`` stages the panels with the TME copy engine
instead -- 2.6x the Triton kernel and ~67% of the vendor library.  (The PH1 part
has no fp64 tensor core, its ``sqmma`` unit only covers fp16/fp32/int8, so the
vendor number is plain SIMT fp64 FMA against a measured 13.6 TFLOPS ALU peak.)
The TME variant needs the PH1 target and a host-built 64-byte tensor descriptor,
so it lives in its own object file and is skipped whenever either is missing --
the LDG kernel then runs unchanged.

Large problems route here, smaller ones stay on Triton (measured: 64x32/4x4
wins below ~4e9 flops).

The kernel is compiled on first use with ``mcc`` and cached; every step is
best-effort and any failure falls back to the Triton implementation.
"""

import ctypes
import hashlib
import os
import warnings
import shutil
import subprocess
import threading

# ---------------------------------------------------------------------------
# MUSA source.  Kept as a string so it ships with the package without needing
# extra packaging rules or a build step.
# ---------------------------------------------------------------------------

_SRC = r"""
// MUSA fp64 DGEMM: double-buffered, vectorised staging, adaptive tile.
//   C = alpha * A * B + beta * C   (row-major; M % BTM == 0, N % BTN == 0,
//                                    K % BKT == 0 -- the caller pads)
//
// Column ownership is *interleaved*: thread tx owns the N column pairs
// {tx + j*BDX | j < TN/2} rather than the contiguous range [tx*TN, tx*TN+TN).
// With the contiguous mapping a warp's LDS.128 requests are spread TN doubles
// apart, so each one touches ~4x the shared-memory bytes it actually needs and
// the kernel runs into the load/store pipe; interleaving makes consecutive tx
// read consecutive double2 chunks and measured +45% on the inner loop.
//
// Register blocking is deliberately kept at TM*TN = 16 or 32.  fp64 values take
// two registers each, so the 8x8 tile the vendor kernel uses (64 accumulators =
// 128 registers) hits the 255-register ceiling and spills; TM=TN=4 (94 regs) and
// TM=8/TN=4 (148 regs) stay clean.
#include <musa_runtime.h>

#ifndef NBUF
#define NBUF 2
#endif

#define NTHR(BTM, BTN, TM, TN) (((BTN) / (TN)) * ((BTM) / (TM)))

template <int BTM, int BTN, int BKT, int TM, int TN>
__device__ __forceinline__ void stage_panels(const double *__restrict__ A,
                                             const double *__restrict__ B,
                                             double *pa, double *pb,
                                             int brow, int bcol, int k0,
                                             int K, int N, int tid)
{
    constexpr int NT = NTHR(BTM, BTN, TM, TN);
    constexpr int APANEL = BTM * BKT;
    constexpr int BPANEL = BKT * BTN;

#pragma unroll
    for (int t = 0; t < (APANEL / 2 + NT - 1) / NT; ++t) {
        const int p = t * NT + tid;
        if (p * 2 < APANEL) {
            const int r = (p * 2) / BKT;
            const int c = (p * 2) % BKT;
            *(double2 *)&pa[r * BKT + c] =
                *(const double2 *)&A[(brow + r) * K + k0 + c];
        }
    }
#pragma unroll
    for (int t = 0; t < (BPANEL / 2 + NT - 1) / NT; ++t) {
        const int p = t * NT + tid;
        if (p * 2 < BPANEL) {
            const int r = (p * 2) / BTN;
            const int c = (p * 2) % BTN;
            *(double2 *)&pb[r * BTN + c] =
                *(const double2 *)&B[(k0 + r) * N + bcol + c];
        }
    }
}

template <int BTM, int BTN, int BKT, int TM, int TN>
__global__ void __launch_bounds__(NTHR(BTM, BTN, TM, TN)) k_gemm(
    const double *__restrict__ A, const double *__restrict__ B,
    double *__restrict__ C, int M, int N, int K, double alpha, double beta,
    int zero_beta)
{
    constexpr int BDX = BTN / TN;
    constexpr int NT = NTHR(BTM, BTN, TM, TN);
    constexpr int APANEL = BTM * BKT;
    constexpr int BPANEL = BKT * BTN;

    extern __shared__ __align__(16) double smem[];
    double *const apanel = smem;
    double *const bpanel = smem + NBUF * APANEL;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BDX + tx;
    const int brow = blockIdx.y * BTM;
    const int bcol = blockIdx.x * BTN;
    const int nkb = K / BKT;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    stage_panels<BTM, BTN, BKT, TM, TN>(A, B, apanel, bpanel, brow, bcol, 0,
                                        K, N, tid);
    __syncthreads();

    for (int kb = 0; kb < nkb; ++kb) {
        const int cur = kb % NBUF;
        const int nxt = (cur + 1) % NBUF;
        if (kb + 1 < nkb)
            stage_panels<BTM, BTN, BKT, TM, TN>(
                A, B, apanel + nxt * APANEL, bpanel + nxt * BPANEL,
                brow, bcol, (kb + 1) * BKT, K, N, tid);

        const double *ap = apanel + cur * APANEL;
        const double *bp = bpanel + cur * BPANEL;
// Fully unrolling this loop makes the compiler software-pipeline the
        // operand loads and blow the register budget (8x8 then spills 500+ bytes
        // and collapses).  Leaving it rolled keeps TM*TN=64 at ~200 registers.
#pragma unroll 1
        for (int kk = 0; kk < BKT; kk += 2) {
            double2 av[TM];
            double2 c0[TN / 2], c1[TN / 2];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                av[i] = *(const double2 *)&ap[(ty * TM + i) * BKT + kk];
#pragma unroll
            for (int j = 0; j < TN / 2; ++j) {
                c0[j] = *(const double2 *)&bp[kk * BTN + (tx + j * BDX) * 2];
                c1[j] = *(const double2 *)&bp[(kk + 1) * BTN + (tx + j * BDX) * 2];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const double x0 = av[i].x, x1 = av[i].y;
#pragma unroll
                for (int j = 0; j < TN / 2; ++j) {
                    acc[i][2 * j] += x0 * c0[j].x;
                    acc[i][2 * j + 1] += x0 * c0[j].y;
                    acc[i][2 * j] += x1 * c1[j].x;
                    acc[i][2 * j + 1] += x1 * c1[j].y;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = brow + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
            const int col = (tx + j * BDX) * 2;
            double2 v;
            v.x = alpha * acc[i][2 * j];
            v.y = alpha * acc[i][2 * j + 1];
            if (!zero_beta) {
                const double2 c = *(const double2 *)&C[r * N + bcol + col];
                v.x += beta * c.x;
                v.y += beta * c.y;
            }
            *(double2 *)&C[r * N + bcol + col] = v;
        }
    }
}

typedef void (*kfn_t)(const double *, const double *, double *, int, int, int,
                      double, double, int);

template <int BTM, int BTN, int BKT, int TM, int TN>
static void launch_t(const double *A, const double *B, double *C, int M, int N,
                     int K, double alpha, double beta, int zero_beta)
{
    constexpr int want = NBUF * (BTM * BKT + BKT * BTN) * (int)sizeof(double);
    static bool done = false;
    if (!done) {
        done = true;
        musaFuncSetAttribute((const void *)(kfn_t)k_gemm<BTM, BTN, BKT, TM, TN>,
                             musaFuncAttributeMaxDynamicSharedMemorySize, want);
    }
    dim3 block(BTN / TN, BTM / TM);
    dim3 grid(N / BTN, M / BTM);
    k_gemm<BTM, BTN, BKT, TM, TN><<<grid, block, want>>>(A, B, C, M, N, K,
                                                         alpha, beta, zero_beta);
}

struct Cand {
    int btm, btn;
    kfn_t fn;
};

// Measured on MTT S5000 (TFLOPS at 4096^3 / 8192^3): 128x256 TM8/TN4 gives
// 5.10/5.17, 256x128 TM8/TN8 4.92/5.04, 256x128 TM8/TN4 4.43/4.56, 64x64 TM4/TN4
// 3.79/3.64.  Below ~2e10 flops the wide tiles lose to the extra parallelism of
// the small tile (1024^3 / 2048^3: 64x64 2.83/3.73).
static const Cand LADDER_L[] = {
    {64, 128, launch_t<64, 128, 8, 8, 8>},
    {128, 128, launch_t<128, 128, 8, 8, 8>},
    {128, 256, launch_t<128, 256, 8, 8, 8>},
    {256, 128, launch_t<256, 128, 8, 8, 8>},
    {64, 64, launch_t<64, 64, 8, 8, 8>},
    {64, 64, launch_t<64, 64, 8, 4, 4>},
    {64, 32, launch_t<64, 32, 8, 4, 4>},
    {32, 32, launch_t<32, 32, 8, 4, 4>},
};
static const Cand LADDER_S[] = {
    {64, 32, launch_t<64, 32, 8, 4, 4>},
    {64, 64, launch_t<64, 64, 8, 4, 4>},
    {32, 32, launch_t<32, 32, 8, 4, 4>},
    {64, 64, launch_t<64, 64, 8, 8, 8>},
    {32, 16, launch_t<32, 16, 8, 4, 2>},
};

extern "C" int dgemm_musa_launch(const double *A, const double *B, double *C,
                                 int M, int N, int K, double alpha, double beta,
                                 int zero_beta)
{
    const bool large = ((double)M * (double)N * (double)K) > 4.0e9;
    const Cand *lad = large ? LADDER_L : LADDER_S;
    const int n = large ? (int)(sizeof(LADDER_L) / sizeof(LADDER_L[0]))
                        : (int)(sizeof(LADDER_S) / sizeof(LADDER_S[0]));
    for (int i = 0; i < n; ++i) {
        if (M % lad[i].btm == 0 && N % lad[i].btn == 0 && K % 8 == 0) {
            lad[i].fn(A, B, C, M, N, K, alpha, beta, zero_beta);
            return (int)musaGetLastError();
        }
    }
    return -999;
}
"""

# ---------------------------------------------------------------------------
# TME (tensor memory engine) staged variant.  Kept in its own translation unit
# because the builtins only exist for the PH1 target, so this is compiled with
# ``--offload-arch=mp_31`` and silently skipped when that is unavailable.
# ---------------------------------------------------------------------------

_TME_SRC = r"""
// fp64 DGEMM whose shared-memory staging is done by the TME copy engine.
//
// The descriptor is a 64-byte opaque blob built on the host; it is passed *by
// value* so it lands in the kernel's parameter space, and the device operand is
// the address of that copy.  Both operands are described as int32 (two per
// double), so one double column is two descriptor columns and tile geometry is
// given as {inner, outer} = {2*cols, rows} at coords {2*col, row}.
//
// Three barriers are used with a single __syncthreads() per k block: the load
// for slot (kb+1)%3 is issued at the top of iteration kb, and slot s is only
// reused two iterations after its last consumer.  With two slots that would
// race (the issuing thread can outrun the other warps still reading the slot).
#include <musa_runtime.h>

typedef int int2v __attribute__((vector_size(8)));
struct __align__(64) TmeDesc { unsigned long long q[8]; };

#define TME_NBUF 3
#define NTHR(BTM, BTN, TM, TN) (((BTN) / (TN)) * ((BTM) / (TM)))
#define AS3(p) ((__attribute__((address_space(3))) void *)(p))

template <int BTM, int BTN, int BKT, int TM, int TN>
__global__ void __launch_bounds__(NTHR(BTM, BTN, TM, TN)) k_gemm_tme(
    TmeDesc dA, TmeDesc dB, double *__restrict__ C, int M, int N, int K,
    double alpha, double beta, int zero_beta)
{
    constexpr int BDX = BTN / TN;
    constexpr int NT = NTHR(BTM, BTN, TM, TN);
    constexpr int APANEL = BTM * BKT;
    constexpr int BPANEL = BKT * BTN;
    constexpr int ABYTES = APANEL * 8;
    constexpr int BBYTES = BPANEL * 8;

    extern __shared__ __align__(128) double smem[];
    double *const apanel = smem;
    double *const bpanel = smem + TME_NBUF * APANEL;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BDX + tx;
    const int brow = blockIdx.y * BTM;
    const int bcol = blockIdx.x * BTN;
    const int nkb = K / BKT;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

#pragma unroll
    for (int b = 1; b <= TME_NBUF; ++b) __musa_async_bar_record(b);
    __syncthreads();
    if (tid == 0) {
        __musa_async_init_arrival(1, 1, 0);
        __musa_async_add_trans(1, ABYTES + BBYTES);
        __musa_tme_ld_tile_2d(1, AS3(apanel), (long)&dA, (int2v){2 * BKT, BTM},
                              (int2v){0, brow}, 0, 3, 1, 0);
        __musa_tme_ld_tile_2d(1, AS3(bpanel), (long)&dB, (int2v){2 * BTN, BKT},
                              (int2v){2 * bcol, 0}, 0, 3, 1, 0);
        __musa_async_arrive_none_phaseid(1);
    }

    for (int kb = 0; kb < nkb; ++kb) {
        const int cur = kb % TME_NBUF, nxt = (cur + 1) % TME_NBUF;
        if (kb + 1 < nkb && tid == 0) {
            const int k0 = (kb + 1) * BKT;
            const int nb = nxt + 1;
            __musa_async_init_arrival(nb, 1, 0);
            __musa_async_add_trans(nb, ABYTES + BBYTES);
            __musa_tme_ld_tile_2d(nb, AS3(apanel + nxt * APANEL), (long)&dA,
                                  (int2v){2 * BKT, BTM}, (int2v){2 * k0, brow},
                                  0, 3, 1, 0);
            __musa_tme_ld_tile_2d(nb, AS3(bpanel + nxt * BPANEL), (long)&dB,
                                  (int2v){2 * BTN, BKT}, (int2v){2 * bcol, k0},
                                  0, 3, 1, 0);
            __musa_async_arrive_none_phaseid(nb);
        }
        __musa_async_wait(cur + 1, 0);
        __syncthreads();

        const double *ap = apanel + cur * APANEL;
        const double *bp = bpanel + cur * BPANEL;
// Fully unrolling this loop makes the compiler software-pipeline the
        // operand loads and blow the register budget (8x8 then spills 500+ bytes
        // and collapses).  Leaving it rolled keeps TM*TN=64 at ~200 registers.
#pragma unroll 1
        for (int kk = 0; kk < BKT; kk += 2) {
            double2 av[TM];
            double2 c0[TN / 2], c1[TN / 2];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                av[i] = *(const double2 *)&ap[(ty * TM + i) * BKT + kk];
#pragma unroll
            for (int j = 0; j < TN / 2; ++j) {
                c0[j] = *(const double2 *)&bp[kk * BTN + (tx + j * BDX) * 2];
                c1[j] = *(const double2 *)&bp[(kk + 1) * BTN + (tx + j * BDX) * 2];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const double x0 = av[i].x, x1 = av[i].y;
#pragma unroll
                for (int j = 0; j < TN / 2; ++j) {
                    acc[i][2 * j] += x0 * c0[j].x;
                    acc[i][2 * j + 1] += x0 * c0[j].y;
                    acc[i][2 * j] += x1 * c1[j].x;
                    acc[i][2 * j + 1] += x1 * c1[j].y;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = brow + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
            const int col = (tx + j * BDX) * 2;
            double2 v;
            v.x = alpha * acc[i][2 * j];
            v.y = alpha * acc[i][2 * j + 1];
            if (!zero_beta) {
                const double2 c = *(const double2 *)&C[r * N + bcol + col];
                v.x += beta * c.x;
                v.y += beta * c.y;
            }
            *(double2 *)&C[r * N + bcol + col] = v;
        }
    }
}

typedef void (*kfn_t)(TmeDesc, TmeDesc, double *, int, int, int, double, double,
                      int);

template <int BTM, int BTN, int BKT, int TM, int TN>
static void launch_t(TmeDesc dA, TmeDesc dB, double *C, int M, int N, int K,
                     double alpha, double beta, int zero_beta)
{
    constexpr int want = TME_NBUF * (BTM * BKT + BKT * BTN) * (int)sizeof(double);
    static bool done = false;
    if (!done) {
        done = true;
        musaFuncSetAttribute((const void *)(kfn_t)k_gemm_tme<BTM, BTN, BKT, TM, TN>,
                             musaFuncAttributeMaxDynamicSharedMemorySize, want);
    }
    dim3 block(BTN / TN, BTM / TM);
    dim3 grid(N / BTN, M / BTM);
    k_gemm_tme<BTM, BTN, BKT, TM, TN><<<grid, block, want>>>(dA, dB, C, M, N, K,
                                                             alpha, beta,
                                                             zero_beta);
}

struct CandK {
    int btm, btn, bkt;
    kfn_t fn;
};

// Tile selection is banded by problem size.
//
// Measured end-to-end on MTT S5000 (ratio vs muBLAS, 1024^3 / 1536^3 / 2048^3
// / 3072^3 / 4096^3 / 8192^3):
//
//   64x128 8x8 : 0.97 / 0.63 / 0.63 / 0.71 / 0.69 / 0.67
//   64x64  8x4 : 1.43 / 0.85 / 0.78 / 0.69 / 0.69 / 0.66
//   64x64  4x4 : 1.09 / 0.70 / 0.61 / 0.54 / 0.54 / -
//
// Up to ~2048^3 the 8x4 tile wins by a wide margin: its 32 fp64 accumulators
// cost only 64 registers, so blocks stay resident and the grid is short enough
// that occupancy/latency hiding dominates the extra 1.5 byte/FLOP it spends.
// Past ~1.5e10 flops the grid is long enough that the 8x8 tile's lower
// byte/FLOP rate wins instead.  (Wider tiles -- 8x12/12x8/16x8 -- are not an
// option: they need >256 registers once the 2-register fp64 accumulators are
// counted and spill, collapsing to <1.6 TFLOPS.  Software-pipelining the kk
// loop fails the same way.  The inner loop saturates near 10.6 TFLOPS, ~78% of
// the measured 13.6 TFLOPS fp64 ALU peak, so ~0.70 of muBLAS is the ceiling of
// this kernel shape.)
static const CandK LADDER_TL[] = {
    {64, 128, 8, launch_t<64, 128, 8, 8, 8>},
    {128, 128, 8, launch_t<128, 128, 8, 8, 8>},
    {128, 256, 8, launch_t<128, 256, 8, 8, 8>},
    {256, 128, 8, launch_t<256, 128, 8, 8, 8>},
    {64, 64, 8, launch_t<64, 64, 8, 8, 8>},
    {64, 64, 8, launch_t<64, 64, 8, 4, 4>},
    {64, 32, 8, launch_t<64, 32, 8, 4, 4>},
    {32, 32, 8, launch_t<32, 32, 8, 4, 4>},
};
static const CandK LADDER_TM[] = {
    {64, 64, 8, launch_t<64, 64, 8, 8, 4>},
    {64, 32, 8, launch_t<64, 32, 8, 8, 4>},
    {64, 128, 8, launch_t<64, 128, 8, 8, 8>},
    {64, 64, 8, launch_t<64, 64, 8, 8, 8>},
    {64, 64, 8, launch_t<64, 64, 8, 4, 4>},
    {64, 32, 8, launch_t<64, 32, 8, 4, 4>},
    {32, 32, 8, launch_t<32, 32, 8, 4, 4>},
    {32, 16, 8, launch_t<32, 16, 8, 4, 2>},
};

// Boundary between the mid band (8x4 tile) and the large band (8x8 tile).
#define TME_MID_WORK 1.5e10

template <int I>
static void copy_desc(TmeDesc &d, const void *p) { __builtin_memcpy(&d, p, 64); }

extern "C" int dgemm_musa_launch_tme(const void *da, const void *db, double *C,
                                     int M, int N, int K, double alpha,
                                     double beta, int zero_beta)
{
    TmeDesc dA, dB;
    copy_desc<0>(dA, da);
    copy_desc<0>(dB, db);
    const double work = (double)M * (double)N * (double)K;
    const CandK *lad;
    int n;
    if (work > TME_MID_WORK) {
        lad = LADDER_TL;
        n = (int)(sizeof(LADDER_TL) / sizeof(LADDER_TL[0]));
    } else {
        lad = LADDER_TM;
        n = (int)(sizeof(LADDER_TM) / sizeof(LADDER_TM[0]));
    }
    for (int i = 0; i < n; ++i) {
        if (M % lad[i].btm == 0 && N % lad[i].btn == 0 &&
            K % lad[i].bkt == 0) {
            lad[i].fn(dA, dB, C, M, N, K, alpha, beta, zero_beta);
            return (int)musaGetLastError();
        }
    }
    return -999;
}
"""

_lock = threading.Lock()
_launcher = None
_tried = False

# ``None`` while the mcc kernels are usable; otherwise a short explanation of
# why they are not.  This matters because the operator silently drops to the
# much slower Triton kernel otherwise, which is indistinguishable from a
# performance regression.
_unavailable = None
_warned = False


def _mark_unavailable(why, warn=True):
    global _unavailable, _warned
    if _unavailable is None:
        _unavailable = why
    if warn and not _warned:
        _warned = True
        warnings.warn(
            "mthreads dgemm: hand-written mcc kernels are not in use (%s); "
            "large problems will fall back to the Triton kernel" % why,
            RuntimeWarning, stacklevel=3)


def unavailable_reason():
    """Return a short explanation if the mcc kernels cannot run, else None."""
    return _unavailable


def _env_disabled():
    return os.environ.get("FLAGBLAS_MTHREADS_DISABLE_MCC", "") not in ("", "0")


def _musa_home():
    for var in ("MUSA_HOME", "MUSA_PATH", "MUSA_ROOT"):
        v = os.environ.get(var)
        if v and os.path.isdir(v):
            return v
    return "/usr/local/musa"


def _cache_dir():
    base = os.environ.get("FLAGBLAS_MCC_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "flagblas", "mcc")
    os.makedirs(base, exist_ok=True)
    return base


def _compile(src_text, tag, arch=None):
    """Compile one MUSA translation unit into a cached shared object.

    Returns the ``.so`` path, or ``None`` if anything is missing (no mcc,
    compile error, ...).  ``arch`` pins ``--offload-arch``; the TME unit needs
    that because its builtins only exist for the PH1 (mp_31) target.
    """
    home = _musa_home()
    # Take the compiler from the same toolkit as the headers and libraries we
    # pass below.  Mixing them -- a 4.3.5 mcc from PATH with 5.2 headers --
    # makes the TME unit fail to build and silently drops the whole fast path.
    mcc = os.path.join(home, "bin", "mcc")
    if not os.path.exists(mcc):
        mcc = shutil.which("mcc")
    if not mcc or not os.path.exists(mcc):
        return None
    key = hashlib.sha256(
        ("%s|%s|%s" % (src_text, arch, home)).encode()).hexdigest()[:16]
    so = os.path.join(_cache_dir(), "%s_%s.so" % (tag, key))
    if os.path.exists(so):
        return so
    src = so + ".mu"
    with open(src, "w") as fh:
        fh.write(src_text)
    # The rpath matters: the object links against libmusart.so.5, which only
    # exists in the toolkit that produced it.  Without it a dlopen in a process
    # whose LD_LIBRARY_PATH points at a different toolkit fails, and the whole
    # fast path silently degrades to the Triton kernel.
    cmd = [mcc, "-shared", "-fPIC", "-O3", "-o", so + ".tmp", src,
           "-I%s/include" % home, "-L%s/lib" % home,
           "-Wl,-rpath,%s/lib" % home, "-lmusart"]
    if arch:
        cmd.append("--offload-arch=%s" % arch)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not os.path.exists(so + ".tmp"):
        return None
    os.replace(so + ".tmp", so)
    return so


def get_launcher():
    """Return a callable ``(A, B, C, M, N, K, alpha, beta, zero_beta)`` that
    launches the MUSA kernel, or ``None`` if the fast path is unavailable."""
    global _launcher, _tried
    with _lock:
        if _tried:
            return _launcher
        _tried = True
        if _env_disabled():
            _mark_unavailable("FLAGBLAS_MTHREADS_DISABLE_MCC is set",
                              warn=False)
        else:
            so = _compile(_SRC, "dgemm_musa",
                          os.environ.get("FLAGBLAS_MCC_ARCH"))
            if not so:
                _mark_unavailable("mcc not found or the kernel failed to "
                                  "compile")
            if so:
                try:
                    lib = ctypes.CDLL(so)
                    fn = lib.dgemm_musa_launch
                    fn.restype = ctypes.c_int
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p] + [ctypes.c_int] * 3 + \
                                  [ctypes.c_double, ctypes.c_double,
                                   ctypes.c_int]
                    _launcher = fn
                except OSError:
                    _launcher = None
    return _launcher


# ---------------------------------------------------------------------------
# TME host side
# ---------------------------------------------------------------------------

_tme_lock = threading.Lock()
_tme_fn = None
_tme_tried = False
_desc_enc = None
_desc_enc_tried = False

# TME builtins only exist for the PH1 target; the device here is a PH1 part.
_TME_ARCH = "mp_31"
_TME_MIN_WORK = 4.0e8


def _get_desc_encoder():
    """Return ``(ptr, rows, cols) -> 64-byte descriptor`` or ``None``.

    A TME descriptor is an opaque 64-byte blob the device reads directly, and
    the only encoder reachable here is Triton's ``fill_tme_descriptor`` (the
    MUSA driver entry point for it is not exported by the 4.3.5 runtime).
    fp64 operands are described as int32 with the inner dimension doubled,
    which matches the geometry ``k_gemm_tme`` requests.
    """
    global _desc_enc, _desc_enc_tried
    with _tme_lock:
        if _desc_enc_tried:
            return _desc_enc
        _desc_enc_tried = True
        try:
            import triton

            fill = triton.runtime.driver.active.utils.fill_tme_descriptor

            def make(ptr, rows, cols):
                desc = fill(ptr, [rows, cols], [8, 64], 4)
                buf = ctypes.create_string_buffer(64)
                ctypes.memmove(buf, desc.tme_desc_cpu_ptr(), 64)
                return buf
        except Exception:
            _desc_enc = None
        else:
            _desc_enc = make
    return _desc_enc


def _get_tme_launcher():
    """Return the TME kernel entry point, or ``None`` if it cannot be built."""
    global _tme_fn, _tme_tried
    with _tme_lock:
        if _tme_tried:
            return _tme_fn
        _tme_tried = True
        if not _env_disabled():
            arch = os.environ.get("FLAGBLAS_MCC_ARCH") or _TME_ARCH
            so = _compile(_TME_SRC, "dgemm_musa_tme", arch)
            if not so:
                _mark_unavailable(
                    "TME kernel failed to build with the toolkit at %s"
                    % _musa_home())
            if so:
                try:
                    lib = ctypes.CDLL(so)
                    fn = lib.dgemm_musa_launch_tme
                    fn.restype = ctypes.c_int
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p] + [ctypes.c_int] * 3 + \
                                  [ctypes.c_double, ctypes.c_double,
                                   ctypes.c_int]
                    _tme_fn = fn
                except OSError as exc:
                    _mark_unavailable("TME kernel failed to load: %s" % exc)
                    _tme_fn = None
    return _tme_fn


def launch_tme(a_p, b_p, c_buf, pm, pn, pk, alpha_f, beta_f, beta_is_zero):
    """Stage both panels with the TME copy engine.  Returns False if the path
    is unavailable, in which case the caller should use the LDG kernel."""
    if pm * pn * pk <= _TME_MIN_WORK:
        return False
    if _env_disabled():
        # explicit request, so record it but do not warn about it
        _mark_unavailable("FLAGBLAS_MTHREADS_DISABLE_MCC is set", warn=False)
        return False
    make = _get_desc_encoder()
    fn = _get_tme_launcher()
    if make is None or fn is None:
        _mark_unavailable("TME staging unavailable: no tensor-descriptor "
                          "encoder")
        return False
    try:
        da = make(a_p.data_ptr(), pm, 2 * pk)
        db = make(b_p.data_ptr(), pk, 2 * pn)
    except Exception:
        return False
    rc = fn(da, db, c_buf.data_ptr(), pm, pn, pk, alpha_f, beta_f,
            1 if beta_is_zero else 0)
    return rc == 0

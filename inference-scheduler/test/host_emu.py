"""Run a generated project on the HOST: the generated inference.c and
test/test_inference.c are compiled unchanged against

  * software models of the VectorOPKernel, MatmulKernel and ConvKernel
    drivers — the AXI-Lite register semantics of
    kernels/vectorop/include/VectorOP.h, kernels/matmul/include/MatmulKernel.h
    and kernels/conv/include/ConvKernel.h (ap_fixed<16,8>: exact products /
    sums, AP_TRN floor + AP_SAT on the result, C-style division; packed-B
    tile-major layout; ConvKernel's packed tile-major weights with the §2.34
    half last tile, depthwise stride, word-padded bias; NCHW x / y with
    implicit zero padding; whole-word tail writes zeroed) executed
    synchronously at Start;
  * a malloc-backed inference_buf implementation (phys == virt), whose
    buffers report ``cached`` = EMU_BUF_CACHED (default 1, the board's
    default: host ops work in place in the buffers; 0 exercises the
    non-cacheable staging path).  sync_to_device / sync_from_device are
    no-ops here (``test_host_ops.TestCoherency`` checks the emitted sync
    sequence instead) — unless ``incoherent=True`` (-DEMU_INCOHERENT): then
    every allocation has a separate "DDR" copy that the kernels read and
    write (phys) next to the CPU's copy (virt), a flush writes its whole range
    into DDR and an invalidate reloads the CPU copy from DDR — so a missing
    or too-narrow sync (e.g. a DMA state's row range) makes a kernel read
    stale data or the CPU read a stale result, and the outputs differ.  (Every
    flush of the generated code follows CPU stores to its whole range — a
    host op's output, host_store's chunks and gaps, an input, a state's rows
    — and a stored line is dirty even where a byte keeps its stale value: a
    slot a kernel wrote and only kernels read is never invalidated, so its
    CPU copy is stale when a host op reuses it.)

The harness fills the inputs, runs inference_run() and compares every output
bit for bit with the scheduler simulation's expected arrays, exactly as on
the board — so a PASS proves the generated host-op C code, its staging /
layout handling and the kernel call parameters all agree with _simulate.py.
Pool models are not supported (no software model).
"""

import os
import re
import shutil
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BUF_EMU = r"""
#include "inference.h"
#include <stdlib.h>
#include <string.h>
#ifndef EMU_BUF_CACHED
#define EMU_BUF_CACHED 1
#endif
Data_t  *inference_buf_ptr(inference_buf_t *b)          { return (Data_t *)b->virt; }
uint64_t inference_buf_phys(const inference_buf_t *b)   { return b->phys; }
unsigned inference_buf_count(const inference_buf_t *b)  { return b->count; }
void inference_buf_init_view(inference_buf_t *v, inference_buf_t *base,
                             unsigned off, unsigned count)
{
    memset(v, 0, sizeof *v);
    v->virt  = (char *)base->virt + (size_t)off * INFERENCE_BYTES_PER_ELEM;
    v->phys  = base->phys + (uint64_t)off * INFERENCE_BYTES_PER_ELEM;
    v->count = count;
    v->cached = base->cached;
}
int  inference_buf_pool_init(void)   { return 0; }
void inference_buf_pool_deinit(void) {}
#ifdef EMU_INCOHERENT
/* CPU copy (virt) and DDR copy (phys: what the kernels access). */
typedef struct { unsigned char *cpu, *ddr; size_t bytes; } emu_alloc_t;
static emu_alloc_t emu_allocs[4096];
static unsigned    emu_nalloc;
static emu_alloc_t *emu_find(const void *p)
{
    unsigned i;
    for (i = 0; i < emu_nalloc; i++)
        if (emu_allocs[i].cpu && (const unsigned char *)p >= emu_allocs[i].cpu
                && (const unsigned char *)p < emu_allocs[i].cpu + emu_allocs[i].bytes)
            return &emu_allocs[i];
    return NULL;
}
#endif
inference_buf_t *inference_buf_alloc(unsigned n)
{
    size_t bytes = ((size_t)n * INFERENCE_BYTES_PER_ELEM + 63u) & ~(size_t)63u;
    inference_buf_t *b = (inference_buf_t *)calloc(1, sizeof *b);
    void *m = NULL;
    if (!b || posix_memalign(&m, 64, bytes ? bytes : 64)) { free(b); return NULL; }
    memset(m, 0xA5, bytes);            /* poison: nothing may rely on zeroed memory */
    b->virt = m; b->phys = (uint64_t)(uintptr_t)m; b->count = n;
    b->refcount = 1u; b->is_owner = 1u; b->cached = EMU_BUF_CACHED;
#ifdef EMU_INCOHERENT
    {
        emu_alloc_t *a = NULL;
        unsigned i;
        for (i = 0; i < emu_nalloc && !a; i++)
            if (!emu_allocs[i].cpu) a = &emu_allocs[i];
        if (!a && emu_nalloc < 4096u) a = &emu_allocs[emu_nalloc++];
        if (!a) { free(m); free(b); return NULL; }
        a->bytes = bytes ? bytes : 64;
        a->cpu = (unsigned char *)m;
        a->ddr = (unsigned char *)malloc(a->bytes);
        if (!a->ddr) return NULL;
        memset(a->ddr, 0xA5, a->bytes);
        b->phys = (uint64_t)(uintptr_t)a->ddr;
    }
#endif
    return b;
}
void inference_buf_retain(inference_buf_t *b)  { if (b && b->is_owner) b->refcount++; }
void inference_buf_release(inference_buf_t *b)
{
    if (b && b->is_owner && --b->refcount == 0u) {
#ifdef EMU_INCOHERENT
        emu_alloc_t *a = emu_find(b->virt);
        if (a) { free(a->ddr); memset(a, 0, sizeof *a); }
#endif
        free(b->virt); free(b);
    }
}
void inference_buf_free(inference_buf_t *b) { inference_buf_release(b); }
#ifdef EMU_INCOHERENT
void inference_buf_sync_to_device(inference_buf_t *b)
{
    emu_alloc_t *a = emu_find(b->virt);
    size_t o, n = (size_t)b->count * INFERENCE_BYTES_PER_ELEM;
    if (!a) return;
    o = (size_t)((unsigned char *)b->virt - a->cpu);
    if (o + n > a->bytes) n = a->bytes - o;
    memcpy(a->ddr + o, a->cpu + o, n);
}
void inference_buf_sync_from_device(inference_buf_t *b)
{
    emu_alloc_t *a = emu_find(b->virt);
    size_t o, n = (size_t)b->count * INFERENCE_BYTES_PER_ELEM;
    if (!a) return;
    o = (size_t)((unsigned char *)b->virt - a->cpu);
    if (o + n > a->bytes) n = a->bytes - o;
    memcpy(a->cpu + o, a->ddr + o, n);
}
#else
void inference_buf_sync_to_device(inference_buf_t *b)   { (void)b; }
void inference_buf_sync_from_device(inference_buf_t *b) { (void)b; }
#endif
@BUF_FLOAT_CONVERSIONS@
void inference_buf_fill_float(inference_buf_t *b, const float *s, unsigned n)
{ unsigned i; for (i = 0; i < n; i++) ((Data_t *)b->virt)[i] = buf_from_float(s[i]); }
void inference_buf_read_float(const inference_buf_t *b, float *d, unsigned n)
{ unsigned i; for (i = 0; i < n; i++) d[i] = buf_to_float(((const Data_t *)b->virt)[i]); }
"""

def buf_emu_source(dtype=None) -> str:
    """inference_buf_emu.c for a project whose Data_t is ``dtype`` (default
    ap_fixed<16,8>): the float fill / read helpers use its conversions."""
    from src.dtype import AP_FIXED_16_8
    d = dtype if dtype is not None else AP_FIXED_16_8
    return _BUF_EMU.replace("@BUF_FLOAT_CONVERSIONS@", d.c_buf_float_conversions())


_COMMON = r"""
#pragma once
#include <stdint.h>
#include <string.h>
typedef uint64_t u64;
static inline int16_t emu_sat(int64_t v)
{ return (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v)); }
static inline int64_t emu_floor_shift(int64_t v, int s)   /* floor(v / 2^s) */
{ return v >= 0 ? (v >> s) : -((-v + ((int64_t)1 << s) - 1) >> s); }
"""

_VOP = r"""
#include "emu_common.h"
#include <math.h>
/* The activation unit's register (the generated code guards plain alpha
 * writes with this macro, xvectoropkernel_hw.h of IPs that have it). */
#define XVECTOROPKERNEL_CTRL_ADDR_ALPHA_DATA 0x64
#define XVECTOROPKERNEL_CTRL_ADDR_SMX_CM_DATA 0x6c
typedef struct { u64 a, b, c; uint32_t size, op, outer, a_inc, b_inc, act, alpha,
                 smx_cm, smx_cfg, smx_mask; } XVectoropkernel;
static inline int XVectoropkernel_Initialize(XVectoropkernel *p, const char *n)
{ (void)n; memset(p, 0, sizeof *p); return 0; }
#define VOP_SET(f) static inline void XVectoropkernel_Set_##f(XVectoropkernel *p, u64 v) { p->f = (uint32_t)v; }
static inline void XVectoropkernel_Set_a(XVectoropkernel *p, u64 v) { p->a = v; }
static inline void XVectoropkernel_Set_b(XVectoropkernel *p, u64 v) { p->b = v; }
static inline void XVectoropkernel_Set_c(XVectoropkernel *p, u64 v) { p->c = v; }
VOP_SET(size) VOP_SET(op) VOP_SET(outer) VOP_SET(a_inc) VOP_SET(b_inc) VOP_SET(act) VOP_SET(alpha)
VOP_SET(smx_cm) VOP_SET(smx_cfg) VOP_SET(smx_mask)
static inline uint32_t XVectoropkernel_Get_alpha(XVectoropkernel *p) { return p->alpha; }
static inline uint32_t XVectoropkernel_Get_smx_cm(XVectoropkernel *p) { return p->smx_cm; }
/* The softmax unit (ops 10 / 11): VectorOP.h smx_vector — the integer
 * specification of src/vectorop_smx.py; EMU_SMX_TAB is its table. */
static const int32_t emu_smx_tab[4096] = { EMU_SMX_TAB };
static void emu_smx_vector(const int16_t *x, size_t xs, unsigned n, unsigned v, uint32_t cm,
                           uint32_t cfg, int16_t *p, size_t ps)
{
    static int64_t e[2048];
    const unsigned cs = cfg & 63u, fp = (cfg >> 8) & 31u;
    int64_t m = INT64_MIN, S = 0;
    uint64_t R;
    unsigned j;
    cm &= 0xFFFFFFu;
    if (v > n) v = n;
    for (j = 0; j < v; j++) if (x[j * xs] > m) m = x[j * xs];
    for (j = 0; j < v; j++) {
        const uint64_t y = ((uint64_t)(m - x[j * xs]) * cm) >> cs, sh = y >> 12;
        e[j] = sh >= 63u ? 0 : (int64_t)emu_smx_tab[y & 4095u] >> sh;   /* 64-bit: sh may exceed 31 */
        S += e[j];
    }
    R = S > 0 ? ((uint64_t)1 << 40) / (uint64_t)S : 0u;
    for (j = 0; j < n; j++) {
        uint64_t q = 0;
        if (j < v) q = ((uint64_t)e[j] * R + ((uint64_t)1 << (39u - fp))) >> (40u - fp);
        p[j * ps] = (int16_t)(q > 32767u ? 32767u : q);
    }
}
static inline unsigned emu_smx_valid(unsigned q, unsigned n, uint32_t mask)
{
    const unsigned v = (mask & 0xFFFFu) + ((mask >> 16) ? q % (mask >> 16) : 0u);
    return v < n ? v : n;
}
/* LeakyReLU / SiLU / GELU / GELU tanh (acts 3..6): VectorOP.cpp act_fn in
 * double, rounded to nearest, ties to even, saturated. */
static inline int64_t emu_act(int64_t r, uint32_t act, uint32_t alpha)
{
    const double x = (double)r / 256.0;
    double y;
    switch (act) {
    case 3: y = x >= 0.0 ? x : x * ((double)(alpha & 0xFFFFu) / 65536.0); break;
    case 4: y = x / (1.0 + exp(-x)); break;
    case 5: y = x * (0.5 * (1.0 + erf(x / sqrt(2.0)))); break;
    case 6: y = x * (0.5 * (1.0 + tanh(sqrt(2.0 / 3.14159265358979323846)
                                       * (x + 0.044715 * (x * x * x))))); break;
    default: return r;
    }
    return emu_sat((int64_t)nearbyint(y * 256.0));
}
static inline int XVectoropkernel_IsDone(XVectoropkernel *p) { (void)p; return 1; }
static inline int XVectoropkernel_Release(XVectoropkernel *p) { (void)p; return 0; }
/* c[o*(a_inc+b_inc) + i] = act(op(a[o*a_inc+i], b[o*b_inc+i])) — an
 * activation op (6..9) applies its own activation (op - 3) instead of act;
 * the last 8-lane word of every run is written whole, tail lanes = 0. */
static inline void XVectoropkernel_Start(XVectoropkernel *p)
{
    const int16_t *a = (const int16_t *)(uintptr_t)p->a;
    const int16_t *b = (const int16_t *)(uintptr_t)p->b;
    int16_t *c = (int16_t *)(uintptr_t)p->c;
    unsigned o, i, words = (p->size + 7u) / 8u, c_inc = p->a_inc + p->b_inc;
    if (p->op == 10u || p->op == 11u) {
        /* softmax: row mode (rows at a_inc -> rows at b_inc) or column mode (s[size]
         * [outer] at row stride a_inc -> P[outer & ~15][size] at b_inc); every output
         * word written whole, tail lanes 0 */
        const int col = p->op == 11u;
        const unsigned rows = col ? (p->outer & ~15u) : p->outer;
        for (o = 0; o < rows; o++) {
            int16_t *out = c + (size_t)o * p->b_inc;
            if (col)
                emu_smx_vector(a + o, p->a_inc, p->size, emu_smx_valid(o, p->size, p->smx_mask),
                               p->smx_cm, p->smx_cfg, out, 1);
            else
                emu_smx_vector(a + (size_t)o * p->a_inc, 1, p->size,
                               emu_smx_valid(o, p->size, p->smx_mask), p->smx_cm, p->smx_cfg, out, 1);
            for (i = p->size; i < words * 8u; i++) out[i] = 0;
        }
        return;
    }
    const uint32_t jact = (p->op >= 6u && p->op <= 9u) ? p->op - 3u : p->act;
    for (o = 0; o < p->outer; o++)
        for (i = 0; i < words * 8u; i++) {
            int64_t x = 0, y = 0, r;
            if (i < p->size) {
                x = a[(size_t)o * p->a_inc + i];
                if (p->op <= 3u) y = b[(size_t)o * p->b_inc + i];
            }
            switch (p->op) {
            case 0: r = x + y; break;
            case 1: r = x - y; break;
            case 2: r = emu_floor_shift(x * y, 8); break;
            case 3: r = y ? (x * 256) / y : 0; break;          /* C division: trunc */
            case 4: r = x > 0 ? x : 0; break;
            case 5: r = x > 0 ? (x < 1536 ? x : 1536) : 0; break;
            default: r = x; break;                              /* 6..9, >= 10: pass */
            }
            r = emu_sat(r);
            if (jact == 1u && r < 0) r = 0;
            if (jact == 2u) r = r < 0 ? 0 : (r > 1536 ? 1536 : r);
            r = emu_act(r, jact, p->alpha);
            if (i >= p->size) r = 0;
            c[(size_t)o * c_inc + i] = (int16_t)r;
        }
}
"""

def vop_source() -> str:
    """The emulated xvectoropkernel.h: _VOP with the softmax unit's table
    (every emulator that builds a generated project writes this one)."""
    return _VOP.replace("EMU_SMX_TAB", _smx_tab())


def _smx_tab() -> str:
    from src.vectorop_smx import TAB
    return ", ".join(str(int(v)) for v in TAB)


_MM = r"""
#include "emu_common.h"
#include <stdlib.h>
#ifndef EMU_TILE_M
#error EMU_TILE_M
#endif
typedef struct { u64 a, b, c, a_to_b; uint32_t n, k, m, batch, a_batch_stride, b_batch_stride,
                 c_batch_stride, b_packed, gemv_kw; } XMatmulkernel;
static inline int XMatmulkernel_Initialize(XMatmulkernel *p, const char *n)
{ (void)n; memset(p, 0, sizeof *p); return 0; }
static inline void XMatmulkernel_Set_a(XMatmulkernel *p, u64 v) { p->a = v; }
static inline void XMatmulkernel_Set_b(XMatmulkernel *p, u64 v) { p->b = v; }
static inline void XMatmulkernel_Set_c(XMatmulkernel *p, u64 v) { p->c = v; }
static inline void XMatmulkernel_Set_a_to_b(XMatmulkernel *p, u64 v) { p->a_to_b = v; }
#define MM_SET(f) static inline void XMatmulkernel_Set_##f(XMatmulkernel *p, u64 v) { p->f = (uint32_t)v; }
MM_SET(n) MM_SET(k) MM_SET(m) MM_SET(batch) MM_SET(a_batch_stride) MM_SET(b_batch_stride)
MM_SET(c_batch_stride) MM_SET(b_packed) MM_SET(gemv_kw)
static inline int XMatmulkernel_IsDone(XMatmulkernel *p) { (void)p; return 1; }
static inline int XMatmulkernel_Release(XMatmulkernel *p) { (void)p; return 0; }
/* Element offset of B[kk][j] in the GEMV image of kernel width kw
 * (MatmulKernel.h matmul_gemv_index; kw = 1: row-major). */
static inline size_t emu_gemv_index(unsigned kk, unsigned j, unsigned m, unsigned kw)
{
    unsigned c;
    if (kw <= 1u) return (size_t)kk * m + j;
    c = (kk / (16u * kw)) * 16u + kk % 16u;
    return ((size_t)c * m + j) * kw + (kk / 16u) % kw;
}
/* C = sat(floor(A.B / 256)); B row-major [k][m], packed tile-major
 * [ceil(m/T)][k][T], or the GEMV image of kernel width gemv_kw, per batch
 * slice (MatmulKernel.h).  GEMV: the hardware reaches B through a_to_b on
 * port a too; b itself is the same address. */
static inline void XMatmulkernel_Start(XMatmulkernel *p)
{
    unsigned bt, r, j, kk;
    if (p->gemv_kw && p->a + p->a_to_b != p->b) abort();
    for (bt = 0; bt < p->batch; bt++) {
        const int16_t *A = (const int16_t *)(uintptr_t)p->a + (size_t)bt * p->a_batch_stride;
        const int16_t *B = (const int16_t *)(uintptr_t)p->b + (size_t)bt * p->b_batch_stride;
        int16_t *C = (int16_t *)(uintptr_t)p->c + (size_t)bt * p->c_batch_stride;
        for (r = 0; r < p->n; r++)
            for (j = 0; j < p->m; j++) {
                int64_t acc = 0;
                for (kk = 0; kk < p->k; kk++) {
                    int64_t bv = p->gemv_kw
                        ? B[emu_gemv_index(kk, j, p->m, p->gemv_kw)]
                        : p->b_packed
                        ? B[((size_t)(j / EMU_TILE_M) * p->k + kk) * EMU_TILE_M + j % EMU_TILE_M]
                        : B[(size_t)kk * p->m + j];
                    acc += (int64_t)A[(size_t)r * p->k + kk] * bv;
                }
                C[(size_t)r * p->m + j] = emu_sat(emu_floor_shift(acc, 8));
            }
    }
}
"""

_CONV = r"""
#include "emu_common.h"
#include <stdlib.h>
#ifndef EMU_CONV_TILE_IC
#error EMU_CONV_TILE_IC
#endif
typedef struct { u64 x, weight, bias, y;
                 uint32_t batch, in_ch, in_h, in_w, out_ch, out_h, out_w, kh, kw,
                          stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left,
                          has_bias, is_depthwise; } XConvkernel;
static inline int XConvkernel_Initialize(XConvkernel *p, const char *n)
{ (void)n; memset(p, 0, sizeof *p); return 0; }
static inline void XConvkernel_Set_x(XConvkernel *p, u64 v)      { p->x = v; }
static inline void XConvkernel_Set_weight(XConvkernel *p, u64 v) { p->weight = v; }
static inline void XConvkernel_Set_bias(XConvkernel *p, u64 v)   { p->bias = v; }
static inline void XConvkernel_Set_y(XConvkernel *p, u64 v)      { p->y = v; }
#define CV_SET(f) static inline void XConvkernel_Set_##f(XConvkernel *p, u64 v) { p->f = (uint32_t)v; }
CV_SET(batch) CV_SET(in_ch) CV_SET(in_h) CV_SET(in_w) CV_SET(out_ch) CV_SET(out_h)
CV_SET(out_w) CV_SET(kh) CV_SET(kw) CV_SET(stride_h) CV_SET(stride_w) CV_SET(dilation_h)
CV_SET(dilation_w) CV_SET(pad_top) CV_SET(pad_left) CV_SET(has_bias) CV_SET(is_depthwise)
static inline int XConvkernel_IsDone(XConvkernel *p) { (void)p; return 1; }
static inline int XConvkernel_Release(XConvkernel *p) { (void)p; return 0; }
/* ConvKernel.h: y[n][m][oh][ow] = sat(floor((bias[m] + sum x * w) / 256)) with
 * x NCHW (zero outside [0,in_h) x [0,in_w)), the standard weight packed
 * tile-major [M][ceil(C/T)][kh][kw][lanes] (lanes = T, the last tile 8 when
 * it holds <= 8 channels), the depthwise one [M][roundup(kh*kw, 8)]. */
static inline void XConvkernel_Start(XConvkernel *p)
{
    const unsigned T = EMU_CONV_TILE_IC, KK = p->kh * p->kw;
    const unsigned tiles = (p->in_ch + T - 1u) / T;
    const unsigned last  = (p->in_ch - (tiles - 1u) * T) <= 8u ? 8u : T;
    const size_t per_m = p->is_depthwise ? (size_t)((KK + 7u) / 8u * 8u)
                                         : (size_t)KK * ((tiles - 1u) * T + last);
    const unsigned P = p->out_h * p->out_w;
    const int16_t *X = (const int16_t *)(uintptr_t)p->x;
    const int16_t *W = (const int16_t *)(uintptr_t)p->weight;
    const int16_t *B = (const int16_t *)(uintptr_t)p->bias;
    int16_t *Y = (int16_t *)(uintptr_t)p->y;
    int64_t *acc = (int64_t *)malloc((size_t)(P ? P : 1u) * sizeof *acc);
    unsigned n, m, c, khi, kwi, oh, ow;
    for (n = 0; n < p->batch; n++)
        for (m = 0; m < p->out_ch; m++) {
            const int64_t b0 = p->has_bias ? (int64_t)B[m] * 256 : 0;
            for (oh = 0; oh < P; oh++) acc[oh] = b0;
            for (c = 0; c < (p->is_depthwise ? 1u : p->in_ch); c++) {
                const unsigned ch = p->is_depthwise ? m : c;
                const unsigned ict = c / T;
                const unsigned lanes = ict + 1u == tiles ? last : T;
                const int16_t *xc = X + ((size_t)n * p->in_ch + ch) * p->in_h * p->in_w;
                for (khi = 0; khi < p->kh; khi++)
                    for (kwi = 0; kwi < p->kw; kwi++) {
                        const size_t wi = p->is_depthwise
                            ? (size_t)m * per_m + khi * p->kw + kwi
                            : (size_t)m * per_m + (size_t)ict * KK * T
                              + (size_t)(khi * p->kw + kwi) * lanes + c % T;
                        const int64_t wv = W[wi];
                        if (!wv) continue;
                        for (oh = 0; oh < p->out_h; oh++) {
                            const int ih = (int)(oh * p->stride_h + khi * p->dilation_h)
                                         - (int)p->pad_top;
                            if (ih < 0 || ih >= (int)p->in_h) continue;
                            for (ow = 0; ow < p->out_w; ow++) {
                                const int iw = (int)(ow * p->stride_w + kwi * p->dilation_w)
                                             - (int)p->pad_left;
                                if (iw < 0 || iw >= (int)p->in_w) continue;
                                acc[oh * p->out_w + ow] += wv * xc[(size_t)ih * p->in_w + iw];
                            }
                        }
                    }
            }
            for (oh = 0; oh < P; oh++)
                Y[((size_t)n * p->out_ch + m) * P + oh] = emu_sat(emu_floor_shift(acc[oh], 8));
        }
    free(acc);
}
"""


def which_cc():
    return shutil.which("cc") or shutil.which("gcc")


def build_and_run(cg, workdir, timeout=600, cached=True, threads=None, min_elems=None,
                  incoherent=False):
    """Write the project of CodeGenerator ``cg`` into ``workdir``, compile it
    against the software kernels and run test_inference.  Returns
    (returncode, combined output).

    ``cached``    buffers report a cacheable mapping (in-place host ops) or not
                  (staging through s_host_stage);
    ``threads``   INFERENCE_HOST_THREADS for the run (None: the default, 4);
    ``min_elems`` -DINFERENCE_HOST_MIN_ELEMS (1 splits even tiny host ops
                  over the threads);
    ``incoherent`` -DEMU_INCOHERENT: separate CPU / DDR copies, the syncs
                  move the data (see the module docstring)."""
    from src._conv_hw_config import CONV_TILE_IC
    from src._matmul_hw_config import MATMUL_TILE_M
    for kd in cg._active_kernels:
        if kd.name not in ("VectorOPKernel", "MatmulKernel", "ConvKernel"):
            raise RuntimeError(f"host emulation has no model of {kd.name}")
    inc, src, tst, emu = (os.path.join(workdir, d) for d in ("include", "src", "test", "emu"))
    for d in (inc, src, tst, emu):
        os.makedirs(d, exist_ok=True)

    def w(path, text):
        with open(path, "w") as f:
            f.write(text)

    w(os.path.join(inc, "inference.h"), cg.generate_header())
    w(os.path.join(src, "inference.c"), cg.generate_source())
    w(os.path.join(tst, "test_inference.c"), cg.generate_test())
    w(os.path.join(emu, "inference_buf_emu.c"), buf_emu_source(cg._dtype))
    w(os.path.join(emu, "emu_common.h"), _COMMON)
    w(os.path.join(emu, "xvectoropkernel.h"), vop_source())
    w(os.path.join(emu, "xmatmulkernel.h"), _MM)
    w(os.path.join(emu, "xconvkernel.h"), _CONV)
    shutil.copy(os.path.join(_ROOT, "runtime", "inference_prof.h"), inc)
    if cg.large_weight_tensors:
        os.makedirs(os.path.join(workdir, "weights"), exist_ok=True)
        for t in cg.large_weight_tensors:
            with open(os.path.join(workdir, "weights", f"{t.c_name}.dat"), "wb") as f:
                f.write(cg.generate_weight_dat(t))
    host_tables = cg.host_table_files() if hasattr(cg, "host_table_files") else []
    if host_tables:
        os.makedirs(os.path.join(workdir, "weights"), exist_ok=True)
        for name, tb in host_tables:
            with open(os.path.join(workdir, "weights", f"{name}.dat"), "wb") as f:
                f.write(tb.dat_bytes())
    if cg.large_expected_tensors:
        os.makedirs(os.path.join(workdir, "expected"), exist_ok=True)
        for t in cg.large_expected_tensors:
            with open(os.path.join(workdir, "expected", f"{t.c_name}.dat"), "wb") as f:
                f.write(cg.generate_expected_dat(t))

    exe = os.path.join(workdir, "test_inference")
    cmd = [which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
           "-Wno-unused-function", "-Wno-error=parentheses",
           f"-DEMU_TILE_M={MATMUL_TILE_M}", f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}",
           f"-DEMU_BUF_CACHED={1 if cached else 0}",
           *(["-DEMU_INCOHERENT"] if incoherent else []),
           *([f"-DINFERENCE_HOST_MIN_ELEMS={min_elems}u"] if min_elems else []), "-pthread",
           f'-DINFERENCE_WEIGHTS_DIR="{workdir}"', f'-DINFERENCE_EXPECTED_DIR="{workdir}"',
           "-I", inc, "-I", emu,
           os.path.join(src, "inference.c"), os.path.join(emu, "inference_buf_emu.c"),
           os.path.join(tst, "test_inference.c"), "-lm", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return r.returncode, "COMPILE FAILED\n" + r.stdout + r.stderr
    env = dict(os.environ)
    if threads is not None:
        env["INFERENCE_HOST_THREADS"] = str(threads)
    r = subprocess.run([exe], capture_output=True, text=True, timeout=timeout, cwd=workdir,
                       env=env)
    return r.returncode, r.stdout + r.stderr


def failures(output):
    return re.findall(r"^FAIL .*$", output, re.M)

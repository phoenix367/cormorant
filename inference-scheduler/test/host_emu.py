"""Run a generated project on the HOST: the generated inference.c and
test/test_inference.c are compiled unchanged against

  * software models of the VectorOPKernel and MatmulKernel drivers — the
    AXI-Lite register semantics of kernels/vectorop/include/VectorOP.h and
    kernels/matmul/include/MatmulKernel.h (ap_fixed<16,8>: exact products /
    sums, AP_TRN floor + AP_SAT on the result, C-style division; packed-B
    tile-major layout; whole-word tail writes zeroed) executed synchronously
    at Start;
  * a malloc-backed inference_buf implementation (phys == virt), whose
    buffers report ``cached`` = EMU_BUF_CACHED (default 1, the board's
    default: host ops work in place in the buffers; 0 exercises the
    non-cacheable staging path).  sync_to_device / sync_from_device are
    no-ops here (``test_host_ops.TestCoherency`` checks the emitted sync
    sequence instead).

The harness fills the inputs, runs inference_run() and compares every output
bit for bit with the scheduler simulation's expected arrays, exactly as on
the board — so a PASS proves the generated host-op C code, its staging /
layout handling and the kernel call parameters all agree with _simulate.py.
Conv / Pool models are not supported (no software model).
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
inference_buf_t *inference_buf_alloc(unsigned n)
{
    size_t bytes = ((size_t)n * INFERENCE_BYTES_PER_ELEM + 63u) & ~(size_t)63u;
    inference_buf_t *b = (inference_buf_t *)calloc(1, sizeof *b);
    void *m = NULL;
    if (!b || posix_memalign(&m, 64, bytes ? bytes : 64)) { free(b); return NULL; }
    memset(m, 0xA5, bytes);            /* poison: nothing may rely on zeroed memory */
    b->virt = m; b->phys = (uint64_t)(uintptr_t)m; b->count = n;
    b->refcount = 1u; b->is_owner = 1u; b->cached = EMU_BUF_CACHED;
    return b;
}
void inference_buf_retain(inference_buf_t *b)  { if (b && b->is_owner) b->refcount++; }
void inference_buf_release(inference_buf_t *b)
{
    if (b && b->is_owner && --b->refcount == 0u) { free(b->virt); free(b); }
}
void inference_buf_free(inference_buf_t *b) { inference_buf_release(b); }
void inference_buf_sync_to_device(inference_buf_t *b)   { (void)b; }
void inference_buf_sync_from_device(inference_buf_t *b) { (void)b; }
void inference_buf_fill_float(inference_buf_t *b, const float *s, unsigned n)
{ unsigned i; for (i = 0; i < n; i++) ((Data_t *)b->virt)[i] = (Data_t)s[i]; }
void inference_buf_read_float(const inference_buf_t *b, float *d, unsigned n)
{ unsigned i; for (i = 0; i < n; i++) d[i] = (float)((const Data_t *)b->virt)[i]; }
"""

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
typedef struct { u64 a, b, c; uint32_t size, op, outer, a_inc, b_inc, act; } XVectoropkernel;
static inline int XVectoropkernel_Initialize(XVectoropkernel *p, const char *n)
{ (void)n; memset(p, 0, sizeof *p); return 0; }
#define VOP_SET(f) static inline void XVectoropkernel_Set_##f(XVectoropkernel *p, u64 v) { p->f = (uint32_t)v; }
static inline void XVectoropkernel_Set_a(XVectoropkernel *p, u64 v) { p->a = v; }
static inline void XVectoropkernel_Set_b(XVectoropkernel *p, u64 v) { p->b = v; }
static inline void XVectoropkernel_Set_c(XVectoropkernel *p, u64 v) { p->c = v; }
VOP_SET(size) VOP_SET(op) VOP_SET(outer) VOP_SET(a_inc) VOP_SET(b_inc) VOP_SET(act)
static inline int XVectoropkernel_IsDone(XVectoropkernel *p) { (void)p; return 1; }
/* c[o*(a_inc+b_inc) + i] = act(op(a[o*a_inc+i], b[o*b_inc+i])); the last
 * 8-lane word of every run is written whole, tail lanes = op(0,0) = 0. */
static inline void XVectoropkernel_Start(XVectoropkernel *p)
{
    const int16_t *a = (const int16_t *)(uintptr_t)p->a;
    const int16_t *b = (const int16_t *)(uintptr_t)p->b;
    int16_t *c = (int16_t *)(uintptr_t)p->c;
    unsigned o, i, words = (p->size + 7u) / 8u, c_inc = p->a_inc + p->b_inc;
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
            default: r = x > 0 ? (x < 1536 ? x : 1536) : 0; break;
            }
            r = emu_sat(r);
            if (p->act == 1u && r < 0) r = 0;
            if (p->act == 2u) r = r < 0 ? 0 : (r > 1536 ? 1536 : r);
            if (i >= p->size) r = 0;
            c[(size_t)o * c_inc + i] = (int16_t)r;
        }
}
"""

_MM = r"""
#include "emu_common.h"
#ifndef EMU_TILE_M
#error EMU_TILE_M
#endif
typedef struct { u64 a, b, c; uint32_t n, k, m, batch, a_batch_stride, b_batch_stride,
                 c_batch_stride, b_packed; } XMatmulkernel;
static inline int XMatmulkernel_Initialize(XMatmulkernel *p, const char *n)
{ (void)n; memset(p, 0, sizeof *p); return 0; }
static inline void XMatmulkernel_Set_a(XMatmulkernel *p, u64 v) { p->a = v; }
static inline void XMatmulkernel_Set_b(XMatmulkernel *p, u64 v) { p->b = v; }
static inline void XMatmulkernel_Set_c(XMatmulkernel *p, u64 v) { p->c = v; }
#define MM_SET(f) static inline void XMatmulkernel_Set_##f(XMatmulkernel *p, u64 v) { p->f = (uint32_t)v; }
MM_SET(n) MM_SET(k) MM_SET(m) MM_SET(batch) MM_SET(a_batch_stride) MM_SET(b_batch_stride)
MM_SET(c_batch_stride) MM_SET(b_packed)
static inline int XMatmulkernel_IsDone(XMatmulkernel *p) { (void)p; return 1; }
/* C = sat(floor(A.B / 256)); B row-major [k][m] or packed tile-major
 * [ceil(m/T)][k][T] per batch slice (MatmulKernel.h). */
static inline void XMatmulkernel_Start(XMatmulkernel *p)
{
    unsigned bt, r, j, kk;
    for (bt = 0; bt < p->batch; bt++) {
        const int16_t *A = (const int16_t *)(uintptr_t)p->a + (size_t)bt * p->a_batch_stride;
        const int16_t *B = (const int16_t *)(uintptr_t)p->b + (size_t)bt * p->b_batch_stride;
        int16_t *C = (int16_t *)(uintptr_t)p->c + (size_t)bt * p->c_batch_stride;
        for (r = 0; r < p->n; r++)
            for (j = 0; j < p->m; j++) {
                int64_t acc = 0;
                for (kk = 0; kk < p->k; kk++) {
                    int64_t bv = p->b_packed
                        ? B[((size_t)(j / EMU_TILE_M) * p->k + kk) * EMU_TILE_M + j % EMU_TILE_M]
                        : B[(size_t)kk * p->m + j];
                    acc += (int64_t)A[(size_t)r * p->k + kk] * bv;
                }
                C[(size_t)r * p->m + j] = emu_sat(emu_floor_shift(acc, 8));
            }
    }
}
"""


def which_cc():
    return shutil.which("cc") or shutil.which("gcc")


def build_and_run(cg, workdir, timeout=600, cached=True):
    """Write the project of CodeGenerator ``cg`` into ``workdir``, compile it
    against the software kernels and run test_inference.  Returns
    (returncode, combined output).

    ``cached``  buffers report a cacheable mapping (in-place host ops) or not
                (staging through s_host_stage)."""
    from src._matmul_hw_config import MATMUL_TILE_M
    for kd in cg._active_kernels:
        if kd.name not in ("VectorOPKernel", "MatmulKernel"):
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
    w(os.path.join(emu, "inference_buf_emu.c"), _BUF_EMU)
    w(os.path.join(emu, "emu_common.h"), _COMMON)
    w(os.path.join(emu, "xvectoropkernel.h"), _VOP)
    w(os.path.join(emu, "xmatmulkernel.h"), _MM)
    shutil.copy(os.path.join(_ROOT, "runtime", "inference_prof.h"), inc)
    if cg.large_weight_tensors:
        os.makedirs(os.path.join(workdir, "weights"), exist_ok=True)
        for t in cg.large_weight_tensors:
            with open(os.path.join(workdir, "weights", f"{t.c_name}.dat"), "wb") as f:
                f.write(cg.generate_weight_dat(t))
    if cg.large_expected_tensors:
        os.makedirs(os.path.join(workdir, "expected"), exist_ok=True)
        for t in cg.large_expected_tensors:
            with open(os.path.join(workdir, "expected", f"{t.c_name}.dat"), "wb") as f:
                f.write(cg.generate_expected_dat(t))

    exe = os.path.join(workdir, "test_inference")
    cmd = [which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
           "-Wno-unused-function", "-Wno-error=parentheses",
           f"-DEMU_TILE_M={MATMUL_TILE_M}", f"-DEMU_BUF_CACHED={1 if cached else 0}",
           f'-DINFERENCE_WEIGHTS_DIR="{workdir}"', f'-DINFERENCE_EXPECTED_DIR="{workdir}"',
           "-I", inc, "-I", emu,
           os.path.join(src, "inference.c"), os.path.join(emu, "inference_buf_emu.c"),
           os.path.join(tst, "test_inference.c"), "-lm", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return r.returncode, "COMPILE FAILED\n" + r.stdout + r.stderr
    r = subprocess.run([exe], capture_output=True, text=True, timeout=timeout, cwd=workdir)
    return r.returncode, r.stdout + r.stderr


def failures(output):
    return re.findall(r"^FAIL .*$", output, re.M)

/* bench_vectorop.c — VectorOPKernel latency / GB/s benchmark
 *
 * args: instance label op size outer a_inc b_inc iters [warmup]
 * output (stdout): one JSON line
 */
#include "inference.h"
#include "xvectoropkernel.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_once(XVectoropkernel *k,
                     uint64_t ap, uint64_t bp, uint64_t cp,
                     unsigned op, unsigned sz, unsigned outer,
                     unsigned ai, unsigned bi)
{
    XVectoropkernel_Set_a(k, ap);
    XVectoropkernel_Set_b(k, bp);
    XVectoropkernel_Set_c(k, cp);
    XVectoropkernel_Set_size(k, sz);
    XVectoropkernel_Set_op(k, op);
    XVectoropkernel_Set_outer(k, outer);
    XVectoropkernel_Set_a_inc(k, ai);
    XVectoropkernel_Set_b_inc(k, bi);
    XVectoropkernel_Set_act(k, 0);      /* no fused activation; the register keeps the last program's value */
    XVectoropkernel_Start(k);
    while (!XVectoropkernel_IsDone(k)) {}
}

int main(int argc, char **argv)
{
    if (argc < 9) {
        fprintf(stderr,
            "usage: %s instance label op size outer a_inc b_inc iters [warmup]\n",
            argv[0]);
        return 1;
    }
    const char *inst    = argv[1];
    const char *label   = argv[2];
    unsigned op     = (unsigned)strtoul(argv[3], NULL, 0);
    unsigned sz     = (unsigned)strtoul(argv[4], NULL, 0);
    unsigned outer  = (unsigned)strtoul(argv[5], NULL, 0);
    unsigned ai     = (unsigned)strtoul(argv[6], NULL, 0);
    unsigned bi     = (unsigned)strtoul(argv[7], NULL, 0);
    unsigned iters  = (unsigned)strtoul(argv[8], NULL, 0);
    unsigned warmup = (argc > 9) ? (unsigned)strtoul(argv[9], NULL, 0) : 10u;

    if (inference_buf_pool_init() != 0) return 1;

    XVectoropkernel k;
    if (XVectoropkernel_Initialize(&k, inst) != 0) {
        fprintf(stderr, "bench_vectorop: init '%s' failed\n", inst); return 1;
    }

    unsigned a_n = (ai > 0u) ? outer * sz : sz;
    unsigned b_n = (bi > 0u) ? outer * sz : sz;
    unsigned c_n = outer * sz;

    inference_buf_t *ba = inference_buf_alloc(a_n);
    inference_buf_t *bb = inference_buf_alloc(b_n);
    inference_buf_t *bc = inference_buf_alloc(c_n);
    if (!ba || !bb || !bc) {
        fprintf(stderr, "bench_vectorop: alloc failed\n"); return 1;
    }
    memset(inference_buf_ptr(ba), 1, (size_t)a_n * INFERENCE_BYTES_PER_ELEM);
    memset(inference_buf_ptr(bb), 1, (size_t)b_n * INFERENCE_BYTES_PER_ELEM);
    inference_buf_sync_to_device(ba);
    inference_buf_sync_to_device(bb);

    uint64_t ap = inference_buf_phys(ba);
    uint64_t bp = inference_buf_phys(bb);
    uint64_t cp = inference_buf_phys(bc);

    unsigned w;
    for (w = 0; w < warmup; w++) {
        run_once(&k, ap, bp, cp, op, sz, outer, ai, bi);
        inference_buf_sync_from_device(bc);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned i;
    for (i = 0; i < iters; i++) {
        run_once(&k, ap, bp, cp, op, sz, outer, ai, bi);
        inference_buf_sync_from_device(bc);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms  = (double)(t1.tv_sec  - t0.tv_sec ) * 1e3
               + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
    double lat = ms / (double)iters;
    /* unary ops (RELU, RELU6) touch 2 ports; binary ops touch 3 */
    unsigned ports = (op >= 4u) ? 2u : 3u;
    double gbs = (double)(ports * c_n * INFERENCE_BYTES_PER_ELEM)
                 / (lat * 1e-3) / 1e9;

    printf("{\"kernel\":\"VectorOPKernel\",\"label\":\"%s\","
           "\"op\":%u,\"size\":%u,\"outer\":%u,"
           "\"a_inc\":%u,\"b_inc\":%u,\"iters\":%u,"
           "\"lat_ms\":%.4f,\"gbs\":%.3f}\n",
           label, op, sz, outer, ai, bi, iters, lat, gbs);

    inference_buf_free(ba); inference_buf_free(bb); inference_buf_free(bc);
    XVectoropkernel_Release(&k);
    inference_buf_pool_deinit();
    return 0;
}

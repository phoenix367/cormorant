/* bench_matmul.c — MatmulKernel latency / GOps/s benchmark
 *
 * args: instance label n k m batch a_batch_stride b_batch_stride b_packed iters [warmup]
 *
 * b_packed = 1 benchmarks MatmulKernel's tile-major packed B layout
 * (MATMUL_OPTIMISATION §3b): the B buffer is then k x roundup(m, 16) per
 * slice and b_batch_stride is in packed elements.
 */
#include "inference.h"
#include "xmatmulkernel.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_once(XMatmulkernel *k,
                     uint64_t ap, uint64_t bp, uint64_t cp,
                     unsigned n, unsigned kd, unsigned m,
                     unsigned batch, unsigned as, unsigned bs,
                     unsigned b_packed)
{
    XMatmulkernel_Set_a(k, ap);
    XMatmulkernel_Set_b(k, bp);
    XMatmulkernel_Set_c(k, cp);
    XMatmulkernel_Set_n(k, n);
    XMatmulkernel_Set_k(k, kd);
    XMatmulkernel_Set_m(k, m);
    XMatmulkernel_Set_batch(k, batch);
    XMatmulkernel_Set_a_batch_stride(k, as);
    XMatmulkernel_Set_b_batch_stride(k, bs);
    XMatmulkernel_Set_c_batch_stride(k, n * m);
    XMatmulkernel_Set_b_packed(k, b_packed);
    XMatmulkernel_Start(k);
    while (!XMatmulkernel_IsDone(k)) {}
}

int main(int argc, char **argv)
{
    if (argc < 11) {
        fprintf(stderr,
            "usage: %s instance label n k m batch a_stride b_stride b_packed iters [warmup]\n",
            argv[0]);
        return 1;
    }
    const char *inst    = argv[1];
    const char *label   = argv[2];
    unsigned n     = (unsigned)strtoul(argv[3],  NULL, 0);
    unsigned kd    = (unsigned)strtoul(argv[4],  NULL, 0);
    unsigned m     = (unsigned)strtoul(argv[5],  NULL, 0);
    unsigned batch = (unsigned)strtoul(argv[6],  NULL, 0);
    unsigned as    = (unsigned)strtoul(argv[7],  NULL, 0);
    unsigned bs    = (unsigned)strtoul(argv[8],  NULL, 0);
    unsigned b_packed = (unsigned)strtoul(argv[9], NULL, 0);
    unsigned iters  = (unsigned)strtoul(argv[10], NULL, 0);
    unsigned warmup = (argc > 11) ? (unsigned)strtoul(argv[11], NULL, 0) : 10u;

    if (inference_buf_pool_init() != 0) return 1;

    XMatmulkernel k;
    if (XMatmulkernel_Initialize(&k, inst) != 0) {
        fprintf(stderr, "bench_matmul: init '%s' failed\n", inst); return 1;
    }

    unsigned a_n = (as > 0u) ? batch * n * kd : n * kd;
    const unsigned m_pk = b_packed ? ((m + 15u) / 16u) * 16u : m;   /* packed: m padded to kTileM */
    if (b_packed && bs > 0u) bs = kd * m_pk;                          /* stride in packed elements */
    unsigned b_n = (bs > 0u) ? batch * kd * m_pk : kd * m_pk;
    unsigned c_n = batch * n * m;

    inference_buf_t *ba = inference_buf_alloc(a_n);
    inference_buf_t *bb = inference_buf_alloc(b_n);
    inference_buf_t *bc = inference_buf_alloc(c_n);
    if (!ba || !bb || !bc) {
        fprintf(stderr, "bench_matmul: alloc failed\n"); return 1;
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
        run_once(&k, ap, bp, cp, n, kd, m, batch, as, bs, b_packed);
        inference_buf_sync_from_device(bc);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned i;
    for (i = 0; i < iters; i++) {
        run_once(&k, ap, bp, cp, n, kd, m, batch, as, bs, b_packed);
        inference_buf_sync_from_device(bc);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms  = (double)(t1.tv_sec  - t0.tv_sec ) * 1e3
               + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
    double lat = ms / (double)iters;
    /* 2 ops per MAC (multiply + accumulate); reported as GOps/s (not GFLOPS) */
    double gops = 2.0 * n * kd * m * batch / (lat * 1e-3) / 1e9;

    printf("{\"kernel\":\"MatmulKernel\",\"label\":\"%s\","
           "\"n\":%u,\"k\":%u,\"m\":%u,\"batch\":%u,"
           "\"a_str\":%u,\"b_str\":%u,\"b_packed\":%u,\"iters\":%u,"
           "\"lat_ms\":%.4f,\"gops\":%.4f}\n",
           label, n, kd, m, batch, as, bs, b_packed, iters, lat, gops);

    inference_buf_free(ba); inference_buf_free(bb); inference_buf_free(bc);
    XMatmulkernel_Release(&k);
    inference_buf_pool_deinit();
    return 0;
}

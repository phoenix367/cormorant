/* bench_pool.c — PoolingKernel latency / GB/s benchmark
 *
 * layout "OFF_X,OFF_Y": x and y at byte offsets inside one buffer (PS_PORTS_PLAN §11)
 * args: instance label batch channels in_h in_w pool_h pool_w sh sw
 *             pt pl dh dw pool_type lp_order count_include_pad iters [warmup]
 */
#include "inference.h"
#include "xpoolingkernel.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_once(XPoolingkernel *k,
                     uint64_t xp, uint64_t yp,
                     unsigned batch, unsigned ch,
                     unsigned ih, unsigned iw, unsigned oh, unsigned ow,
                     unsigned ph, unsigned pw, unsigned sh, unsigned sw,
                     unsigned pt, unsigned pl, unsigned dh, unsigned dw,
                     unsigned ptype, unsigned lpo, unsigned cip)
{
    XPoolingkernel_Set_x(k, xp);
    XPoolingkernel_Set_y(k, yp);
    XPoolingkernel_Set_batch(k, batch);
    XPoolingkernel_Set_channels(k, ch);
    XPoolingkernel_Set_in_h(k, ih);
    XPoolingkernel_Set_in_w(k, iw);
    XPoolingkernel_Set_out_h(k, oh);
    XPoolingkernel_Set_out_w(k, ow);
    XPoolingkernel_Set_pool_h(k, ph);
    XPoolingkernel_Set_pool_w(k, pw);
    XPoolingkernel_Set_stride_h(k, sh);
    XPoolingkernel_Set_stride_w(k, sw);
    XPoolingkernel_Set_pad_top(k, pt);
    XPoolingkernel_Set_pad_left(k, pl);
    XPoolingkernel_Set_dil_h(k, dh);
    XPoolingkernel_Set_dil_w(k, dw);
    XPoolingkernel_Set_pool_type(k, ptype);
    XPoolingkernel_Set_lp_order(k, lpo);
    XPoolingkernel_Set_count_include_pad(k, cip);
    XPoolingkernel_Start(k);
    while (!XPoolingkernel_IsDone(k)) {}
}

int main(int argc, char **argv)
{
    if (argc < 19) {
        fprintf(stderr,
            "usage: %s instance label batch channels in_h in_w pool_h pool_w "
            "sh sw pt pl dh dw pool_type lp_order cip iters [warmup [layout]]\n",
            argv[0]);
        return 1;
    }
    const char *inst  = argv[1];
    const char *label = argv[2];
    unsigned batch = (unsigned)strtoul(argv[3],  NULL, 0);
    unsigned ch    = (unsigned)strtoul(argv[4],  NULL, 0);
    unsigned ih    = (unsigned)strtoul(argv[5],  NULL, 0);
    unsigned iw    = (unsigned)strtoul(argv[6],  NULL, 0);
    unsigned ph    = (unsigned)strtoul(argv[7],  NULL, 0);
    unsigned pw    = (unsigned)strtoul(argv[8],  NULL, 0);
    unsigned sh    = (unsigned)strtoul(argv[9],  NULL, 0);
    unsigned sw    = (unsigned)strtoul(argv[10], NULL, 0);
    unsigned pt    = (unsigned)strtoul(argv[11], NULL, 0);
    unsigned pl    = (unsigned)strtoul(argv[12], NULL, 0);
    unsigned dh    = (unsigned)strtoul(argv[13], NULL, 0);
    unsigned dw    = (unsigned)strtoul(argv[14], NULL, 0);
    unsigned ptype = (unsigned)strtoul(argv[15], NULL, 0);
    unsigned lpo   = (unsigned)strtoul(argv[16], NULL, 0);
    unsigned cip   = (unsigned)strtoul(argv[17], NULL, 0);
    unsigned iters  = (unsigned)strtoul(argv[18], NULL, 0);
    unsigned warmup = (argc > 19) ? (unsigned)strtoul(argv[19], NULL, 0) : 10u;
    const char *layout = (argc > 20) ? argv[20] : NULL;

    unsigned oh = (ih + 2*pt - dh*(ph-1) - 1) / sh + 1;
    unsigned ow = (iw + 2*pl - dw*(pw-1) - 1) / sw + 1;

    if (inference_buf_pool_init() != 0) return 1;

    XPoolingkernel k;
    if (XPoolingkernel_Initialize(&k, inst) != 0) {
        fprintf(stderr, "bench_pool: init '%s' failed\n", inst); return 1;
    }

    unsigned x_n = batch * ch * ih * iw;
    unsigned y_n = batch * ch * oh * ow;

    inference_buf_t *bufs[2], *pool;
    unsigned counts[2] = { x_n, y_n };
    if (inference_buf_alloc_layout(layout, 2, counts, bufs, &pool) != 0) {
        fprintf(stderr, "bench_pool: alloc failed\n"); return 1;
    }
    inference_buf_t *bx = bufs[0], *by = bufs[1];
    memset(inference_buf_ptr(bx), 1, (size_t)x_n * INFERENCE_BYTES_PER_ELEM);
    inference_buf_sync_to_device(bx);

    uint64_t xp = inference_buf_phys(bx);
    uint64_t yp = inference_buf_phys(by);

    unsigned i;
    for (i = 0; i < warmup; i++) {
        run_once(&k, xp, yp, batch, ch, ih, iw, oh, ow,
                 ph, pw, sh, sw, pt, pl, dh, dw, ptype, lpo, cip);
        inference_buf_sync_from_device(by);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < iters; i++) {
        run_once(&k, xp, yp, batch, ch, ih, iw, oh, ow,
                 ph, pw, sh, sw, pt, pl, dh, dw, ptype, lpo, cip);
        inference_buf_sync_from_device(by);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms  = (double)(t1.tv_sec  - t0.tv_sec ) * 1e3
               + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
    double lat = ms / (double)iters;
    double gbs = (double)(x_n + y_n) * INFERENCE_BYTES_PER_ELEM
                 / (lat * 1e-3) / 1e9;

    printf("{\"kernel\":\"PoolingKernel\",\"label\":\"%s\","
           "\"batch\":%u,\"channels\":%u,\"in_h\":%u,\"in_w\":%u,"
           "\"out_h\":%u,\"out_w\":%u,"
           "\"pool_h\":%u,\"pool_w\":%u,\"pool_type\":%u,\"iters\":%u,"
           "\"lat_ms\":%.4f,\"gbs\":%.3f,\"phys\":%s}\n",
           label, batch, ch, ih, iw, oh, ow,
           ph, pw, ptype, iters, lat, gbs, inference_buf_phys_json(2, bufs));

    inference_buf_free_layout(2, bufs, pool);
    XPoolingkernel_Release(&k);
    inference_buf_pool_deinit();
    return 0;
}

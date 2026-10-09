/* bench_conv.c — ConvKernel latency / GOps/s benchmark
 *
 * layout "OFF_X,OFF_W,OFF_BIAS,OFF_Y": the buffers at byte offsets inside one buffer (PS_PORTS_PLAN §11)
 * args: instance label batch in_ch in_h in_w out_ch kh kw sh sw
 *             dh dw pt pl has_bias is_dw iters [warmup]
 */
#include "inference.h"
#include "xconvkernel.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_once(XConvkernel *k,
                     uint64_t xp, uint64_t wp, uint64_t bp2, uint64_t yp,
                     unsigned batch, unsigned ic, unsigned ih, unsigned iw,
                     unsigned oc, unsigned oh, unsigned ow,
                     unsigned kh, unsigned kw, unsigned sh, unsigned sw,
                     unsigned dh, unsigned dw, unsigned pt, unsigned pl,
                     unsigned hb, unsigned dw2)
{
    XConvkernel_Set_x(k, xp);
    XConvkernel_Set_weight(k, wp);
    XConvkernel_Set_bias(k, bp2);
    XConvkernel_Set_y(k, yp);
    XConvkernel_Set_batch(k, batch);
    XConvkernel_Set_in_ch(k, ic);
    XConvkernel_Set_in_h(k, ih);
    XConvkernel_Set_in_w(k, iw);
    XConvkernel_Set_out_ch(k, oc);
    XConvkernel_Set_out_h(k, oh);
    XConvkernel_Set_out_w(k, ow);
    XConvkernel_Set_kh(k, kh);
    XConvkernel_Set_kw(k, kw);
    XConvkernel_Set_stride_h(k, sh);
    XConvkernel_Set_stride_w(k, sw);
    XConvkernel_Set_dilation_h(k, dh);
    XConvkernel_Set_dilation_w(k, dw);
    XConvkernel_Set_pad_top(k, pt);
    XConvkernel_Set_pad_left(k, pl);
    XConvkernel_Set_has_bias(k, hb);
    XConvkernel_Set_is_depthwise(k, dw2);
    XConvkernel_Start(k);
    while (!XConvkernel_IsDone(k)) {}
}

int main(int argc, char **argv)
{
    if (argc < 19) {
        fprintf(stderr,
            "usage: %s instance label batch in_ch in_h in_w out_ch kh kw "
            "sh sw dh dw pt pl has_bias is_dw iters [warmup [layout]]\n", argv[0]);
        return 1;
    }
    const char *inst  = argv[1];
    const char *label = argv[2];
    unsigned batch = (unsigned)strtoul(argv[3],  NULL, 0);
    unsigned ic    = (unsigned)strtoul(argv[4],  NULL, 0);
    unsigned ih    = (unsigned)strtoul(argv[5],  NULL, 0);
    unsigned iw    = (unsigned)strtoul(argv[6],  NULL, 0);
    unsigned oc    = (unsigned)strtoul(argv[7],  NULL, 0);
    unsigned kh    = (unsigned)strtoul(argv[8],  NULL, 0);
    unsigned kw    = (unsigned)strtoul(argv[9],  NULL, 0);
    unsigned sh    = (unsigned)strtoul(argv[10], NULL, 0);
    unsigned sw    = (unsigned)strtoul(argv[11], NULL, 0);
    unsigned dh    = (unsigned)strtoul(argv[12], NULL, 0);
    unsigned dw    = (unsigned)strtoul(argv[13], NULL, 0);
    unsigned pt    = (unsigned)strtoul(argv[14], NULL, 0);
    unsigned pl    = (unsigned)strtoul(argv[15], NULL, 0);
    unsigned hb    = (unsigned)strtoul(argv[16], NULL, 0);
    unsigned idw   = (unsigned)strtoul(argv[17], NULL, 0);
    unsigned iters  = (unsigned)strtoul(argv[18], NULL, 0);
    unsigned warmup = (argc > 19) ? (unsigned)strtoul(argv[19], NULL, 0) : 10u;
    const char *layout = (argc > 20) ? argv[20] : NULL;

    unsigned oh = (ih + 2*pt - dh*(kh-1) - 1) / sh + 1;
    unsigned ow = (iw + 2*pl - dw*(kw-1) - 1) / sw + 1;

    if (inference_buf_pool_init() != 0) return 1;

    XConvkernel k;
    if (XConvkernel_Initialize(&k, inst) != 0) {
        fprintf(stderr, "bench_conv: init '%s' failed\n", inst); return 1;
    }

    unsigned x_n    = batch * ic * ih * iw;
    unsigned w_n    = idw ? (oc * kh * kw) : (oc * ic * kh * kw);
    unsigned bias_n = hb ? oc : 1u;
    unsigned y_n    = batch * oc * oh * ow;

    inference_buf_t *bufs[4], *pool;
    unsigned counts[4] = { x_n, w_n, bias_n, y_n };
    if (inference_buf_alloc_layout(layout, 4, counts, bufs, &pool) != 0) {
        fprintf(stderr, "bench_conv: alloc failed\n"); return 1;
    }
    inference_buf_t *bx = bufs[0], *bw = bufs[1], *bb = bufs[2], *by = bufs[3];
    memset(inference_buf_ptr(bx), 1, (size_t)x_n    * INFERENCE_BYTES_PER_ELEM);
    memset(inference_buf_ptr(bw), 1, (size_t)w_n    * INFERENCE_BYTES_PER_ELEM);
    memset(inference_buf_ptr(bb), 0, (size_t)bias_n * INFERENCE_BYTES_PER_ELEM);
    inference_buf_sync_to_device(bx);
    inference_buf_sync_to_device(bw);
    inference_buf_sync_to_device(bb);

    uint64_t xp  = inference_buf_phys(bx);
    uint64_t wp  = inference_buf_phys(bw);
    uint64_t bp2 = inference_buf_phys(bb);
    uint64_t yp  = inference_buf_phys(by);

    unsigned i;
    for (i = 0; i < warmup; i++) {
        run_once(&k, xp, wp, bp2, yp,
                 batch, ic, ih, iw, oc, oh, ow,
                 kh, kw, sh, sw, dh, dw, pt, pl, hb, idw);
        inference_buf_sync_from_device(by);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < iters; i++) {
        run_once(&k, xp, wp, bp2, yp,
                 batch, ic, ih, iw, oc, oh, ow,
                 kh, kw, sh, sw, dh, dw, pt, pl, hb, idw);
        inference_buf_sync_from_device(by);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms  = (double)(t1.tv_sec  - t0.tv_sec ) * 1e3
               + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
    double lat = ms / (double)iters;
    double macs = (double)batch * oc * oh * ow;
    macs *= idw ? (double)(kh * kw) : (double)(ic * kh * kw);
    /* 2 ops per MAC (multiply + accumulate); reported as GOps/s (not GFLOPS) */
    double gops = 2.0 * macs / (lat * 1e-3) / 1e9;

    printf("{\"kernel\":\"ConvKernel\",\"label\":\"%s\","
           "\"batch\":%u,\"in_ch\":%u,\"in_h\":%u,\"in_w\":%u,"
           "\"out_ch\":%u,\"out_h\":%u,\"out_w\":%u,"
           "\"kh\":%u,\"kw\":%u,\"sh\":%u,\"sw\":%u,"
           "\"is_dw\":%u,\"iters\":%u,\"lat_ms\":%.4f,\"gops\":%.4f,\"phys\":%s}\n",
           label, batch, ic, ih, iw, oc, oh, ow,
           kh, kw, sh, sw, idw, iters, lat, gops, inference_buf_phys_json(4, bufs));

    inference_buf_free_layout(4, bufs, pool);
    XConvkernel_Release(&k);
    inference_buf_pool_deinit();
    return 0;
}

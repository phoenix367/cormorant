/* calib_runner.c — batch timing of kernel calls for the performance models
 * (inference-scheduler/perf_calibrate.py, doc/plans/TACTICS_PLAN.md §4.3).
 *
 * usage: calib_runner CASES VOP_INST MM_INST CONV_INST POOL_INST [FILL]
 *
 * CASES: one case per line,
 *     id kernel iters warmup nbuf n_0 .. n_{nbuf-1} r_0 r_1 ...
 * kernel = V (VectorOPKernel), M (MatmulKernel), C (ConvKernel), P (PoolingKernel);
 * n_i = element counts of the call's buffers (sized by the host:
 * V a b c, M a b c, C x w bias y, P x y); r_i = the register values in the
 * order of src/perf_calls.py FIELDS.  A kernel whose instance is "-" is not
 * opened.  FILL (default 1) is the byte every buffer is filled with — a
 * second pass with another one checks that the timing does not depend on
 * the data.
 *
 * Each call is timed alone: the register writes, Start and the poll until
 * done (what the generated code pays per call), no cache maintenance.  One
 * JSON line per case: {"id","ok","iters","mean_us","min_us","max_us","sd_us"}.
 * A call that is not done after 20 s stops the batch (exit 2): the kernel is
 * wedged and later calls would be meaningless.
 */
#define _GNU_SOURCE
#include "inference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "xconvkernel.h"
#include "xmatmulkernel.h"
#include "xpoolingkernel.h"
#include "xvectoropkernel.h"

#define MAX_BUF  4
#define MAX_REG  24
#define TIMEOUT_S 20.0

static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec * 1e-3;
}

static XVectoropkernel s_v;
static XMatmulkernel   s_m;
static XConvkernel     s_c;
static XPoolingkernel  s_p;
static int s_open[4];

typedef struct {
    char     kernel;
    uint64_t addr[MAX_BUF];
    unsigned r[MAX_REG];
} call_t;

/* program + start; returns 0 */
static void start(const call_t *c)
{
    const unsigned *r = c->r;
    switch (c->kernel) {
    case 'V':   /* op size outer a_inc b_inc act */
        XVectoropkernel_Set_a(&s_v, c->addr[0]);
        XVectoropkernel_Set_b(&s_v, c->addr[1]);
        XVectoropkernel_Set_c(&s_v, c->addr[2]);
        XVectoropkernel_Set_size(&s_v, r[1]);
        XVectoropkernel_Set_op(&s_v, r[0]);
        XVectoropkernel_Set_outer(&s_v, r[2]);
        XVectoropkernel_Set_a_inc(&s_v, r[3]);
        XVectoropkernel_Set_b_inc(&s_v, r[4]);
        XVectoropkernel_Set_act(&s_v, r[5]);
#ifdef XVECTOROPKERNEL_CTRL_ADDR_ALPHA_DATA   /* IPs with the activation unit */
        XVectoropkernel_Set_alpha(&s_v, 0);   /* LeakyReLU slope: no effect on timing */
#endif
#ifdef XVECTOROPKERNEL_CTRL_ADDR_SMX_CM_DATA  /* IPs with the softmax unit: the softmax ops */
        if (r[0] == 10u || r[0] == 11u) {        /* only, as run_softmax() (the scale and */
            XVectoropkernel_Set_smx_cm(&s_v, 12102203u);   /* masks have no effect on timing) */
            XVectoropkernel_Set_smx_cfg(&s_v, 19u | (12u << 8));
            XVectoropkernel_Set_smx_mask(&s_v, r[1] & 0xFFFFu);
        }
#endif
        XVectoropkernel_Start(&s_v);
        break;
    case 'M':   /* n k m batch a_stride b_stride c_stride b_packed gemv_kw */
        XMatmulkernel_Set_a(&s_m, c->addr[0]);
        XMatmulkernel_Set_b(&s_m, c->addr[1]);
        XMatmulkernel_Set_c(&s_m, c->addr[2]);
        XMatmulkernel_Set_n(&s_m, r[0]);
        XMatmulkernel_Set_k(&s_m, r[1]);
        XMatmulkernel_Set_m(&s_m, r[2]);
        XMatmulkernel_Set_batch(&s_m, r[3]);
        XMatmulkernel_Set_a_batch_stride(&s_m, r[4]);
        XMatmulkernel_Set_b_batch_stride(&s_m, r[5]);
        XMatmulkernel_Set_c_batch_stride(&s_m, r[6]);
        XMatmulkernel_Set_b_packed(&s_m, r[7]);
        XMatmulkernel_Set_gemv_kw(&s_m, r[8]);
        XMatmulkernel_Set_a_to_b(&s_m, c->addr[1] - c->addr[0]);
        XMatmulkernel_Start(&s_m);
        break;
    case 'C':   /* batch in_ch in_h in_w out_ch out_h out_w kh kw sh sw dh dw pt pl has_bias is_dw */
        XConvkernel_Set_x(&s_c, c->addr[0]);
        XConvkernel_Set_weight(&s_c, c->addr[1]);
        XConvkernel_Set_bias(&s_c, r[15] ? c->addr[2] : (u64)0);
        XConvkernel_Set_y(&s_c, c->addr[3]);
        XConvkernel_Set_batch(&s_c, r[0]);
        XConvkernel_Set_in_ch(&s_c, r[1]);
        XConvkernel_Set_in_h(&s_c, r[2]);
        XConvkernel_Set_in_w(&s_c, r[3]);
        XConvkernel_Set_out_ch(&s_c, r[4]);
        XConvkernel_Set_out_h(&s_c, r[5]);
        XConvkernel_Set_out_w(&s_c, r[6]);
        XConvkernel_Set_kh(&s_c, r[7]);
        XConvkernel_Set_kw(&s_c, r[8]);
        XConvkernel_Set_stride_h(&s_c, r[9]);
        XConvkernel_Set_stride_w(&s_c, r[10]);
        XConvkernel_Set_dilation_h(&s_c, r[11]);
        XConvkernel_Set_dilation_w(&s_c, r[12]);
        XConvkernel_Set_pad_top(&s_c, r[13]);
        XConvkernel_Set_pad_left(&s_c, r[14]);
        XConvkernel_Set_has_bias(&s_c, r[15]);
        XConvkernel_Set_is_depthwise(&s_c, r[16]);
        XConvkernel_Start(&s_c);
        break;
    case 'P':   /* batch ch in_h in_w out_h out_w ph pw sh sw pt pl dh dw type lp cip */
        XPoolingkernel_Set_x(&s_p, c->addr[0]);
        XPoolingkernel_Set_y(&s_p, c->addr[1]);
        XPoolingkernel_Set_batch(&s_p, r[0]);
        XPoolingkernel_Set_channels(&s_p, r[1]);
        XPoolingkernel_Set_in_h(&s_p, r[2]);
        XPoolingkernel_Set_in_w(&s_p, r[3]);
        XPoolingkernel_Set_out_h(&s_p, r[4]);
        XPoolingkernel_Set_out_w(&s_p, r[5]);
        XPoolingkernel_Set_pool_h(&s_p, r[6]);
        XPoolingkernel_Set_pool_w(&s_p, r[7]);
        XPoolingkernel_Set_stride_h(&s_p, r[8]);
        XPoolingkernel_Set_stride_w(&s_p, r[9]);
        XPoolingkernel_Set_pad_top(&s_p, r[10]);
        XPoolingkernel_Set_pad_left(&s_p, r[11]);
        XPoolingkernel_Set_dil_h(&s_p, r[12]);
        XPoolingkernel_Set_dil_w(&s_p, r[13]);
        XPoolingkernel_Set_pool_type(&s_p, r[14]);
        XPoolingkernel_Set_lp_order(&s_p, r[15]);
        XPoolingkernel_Set_count_include_pad(&s_p, r[16]);
        XPoolingkernel_Start(&s_p);
        break;
    }
}

static int done(char k)
{
    switch (k) {
    case 'V': return (int)XVectoropkernel_IsDone(&s_v);
    case 'M': return (int)XMatmulkernel_IsDone(&s_m);
    case 'C': return (int)XConvkernel_IsDone(&s_c);
    default:  return (int)XPoolingkernel_IsDone(&s_p);
    }
}

/* one timed call (us); < 0 on timeout */
static double call(const call_t *c)
{
    const double t0 = now_us();
    start(c);
    while (!done(c->kernel))
        if (now_us() - t0 > TIMEOUT_S * 1e6)
            return -1.0;
    return now_us() - t0;
}

static int kidx(char k) { return k == 'V' ? 0 : k == 'M' ? 1 : k == 'C' ? 2 : k == 'P' ? 3 : -1; }

static int open_kernel(char k, char *const inst[4])
{
    const int i = kidx(k);
    int rc = -1;
    if (i < 0 || !strcmp(inst[i], "-")) return -1;
    if (s_open[i]) return 0;
    switch (k) {
    case 'V': rc = XVectoropkernel_Initialize(&s_v, inst[0]); break;
    case 'M': rc = XMatmulkernel_Initialize(&s_m, inst[1]); break;
    case 'C': rc = XConvkernel_Initialize(&s_c, inst[2]); break;
    case 'P': rc = XPoolingkernel_Initialize(&s_p, inst[3]); break;
    }
    if (rc == 0) s_open[i] = 1;
    return rc == 0 ? 0 : -1;
}

int main(int argc, char **argv)
{
    char  line[4096];
    FILE *f;
    int   fill;
    if (argc < 6) {
        fprintf(stderr, "usage: %s CASES VOP_INST MM_INST CONV_INST POOL_INST [FILL]\n", argv[0]);
        return 1;
    }
    fill = argc > 6 ? (int)strtol(argv[6], NULL, 0) & 0xff : 1;
    if (!(f = fopen(argv[1], "r"))) {
        perror(argv[1]);
        return 1;
    }
    if (inference_buf_pool_init() != 0) {
        fprintf(stderr, "calib_runner: buffer pool init failed\n");
        return 1;
    }
    while (fgets(line, sizeof line, f)) {
        char            id[128], kernel;
        unsigned        iters, warmup, nbuf, n[MAX_BUF], i, nreg = 0;
        inference_buf_t *buf[MAX_BUF] = { 0 };
        call_t          c;
        char           *p = line, *end;
        double          sum = 0.0, sum2 = 0.0, mn = 1e30, mx = 0.0;
        int             used = 0, ok = 1;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(p, "%127s %c %u %u %u%n", id, &kernel, &iters, &warmup, &nbuf, &used) != 5 ||
            nbuf > MAX_BUF || iters == 0) {
            fprintf(stderr, "calib_runner: bad line: %s", line);
            return 1;
        }
        p += used;
        for (i = 0; i < nbuf; i++) {
            n[i] = (unsigned)strtoul(p, &end, 10);
            p = end;
        }
        memset(&c, 0, sizeof c);
        c.kernel = kernel;
        for (;;) {
            unsigned long v = strtoul(p, &end, 10);
            if (end == p || nreg >= MAX_REG) break;
            c.r[nreg++] = (unsigned)v;
            p = end;
        }
        if (open_kernel(kernel, argv + 2) != 0) {
            printf("{\"id\":\"%s\",\"ok\":0,\"err\":\"kernel %c not available\"}\n", id, kernel);
            fflush(stdout);
            continue;
        }
        for (i = 0; i < nbuf && ok; i++) {
            buf[i] = inference_buf_alloc(n[i] ? n[i] : 1u);
            if (!buf[i]) { ok = 0; break; }
            memset(inference_buf_ptr(buf[i]), fill, (size_t)(n[i] ? n[i] : 1u) * INFERENCE_BYTES_PER_ELEM);
            inference_buf_sync_to_device(buf[i]);
            c.addr[i] = inference_buf_phys(buf[i]);
        }
        if (ok) {
            for (i = 0; i < warmup && ok; i++)
                ok = call(&c) >= 0.0;
            for (i = 0; i < iters && ok; i++) {
                const double t = call(&c);
                if (t < 0.0) { ok = 0; break; }
                sum += t; sum2 += t * t;
                mn = t < mn ? t : mn;
                mx = t > mx ? t : mx;
            }
            if (!ok) {
                printf("{\"id\":\"%s\",\"ok\":0,\"err\":\"timeout\"}\n", id);
                fflush(stdout);
                return 2;
            }
            {
                const double mean = sum / iters;
                const double var = iters > 1 ? (sum2 - sum * mean) / (iters - 1) : 0.0;
                printf("{\"id\":\"%s\",\"ok\":1,\"iters\":%u,\"mean_us\":%.3f,\"min_us\":%.3f,"
                       "\"max_us\":%.3f,\"sd_us\":%.3f}\n", id, iters, mean, mn, mx,
                       var > 0.0 ? sqrt(var) : 0.0);
            }
        } else {
            printf("{\"id\":\"%s\",\"ok\":0,\"err\":\"alloc\"}\n", id);
        }
        fflush(stdout);
        for (i = 0; i < nbuf; i++)
            if (buf[i]) inference_buf_free(buf[i]);
    }
    fclose(f);
    inference_buf_pool_deinit();
    return 0;
}

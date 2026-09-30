/*
 * tts_dp.c — see tts_dp.h.  Rows are time steps ([n][192], time-major), so
 * every stage is independent per row except the depthwise conv, which reads
 * its input's neighbours: each stage runs over row ranges on `threads`
 * threads, joined before the next.  Operation order = piper_vits.py
 * duration_predictor_seq (the specification), including the order of the
 * two operands of every product and sum.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tts_dp.h"

#define DC   192                      /* channels */
#define NB   10                       /* spline bins */
#define NP   29                       /* spline parameters per step: 3 * NB - 1 */
#define MAXT 8                        /* threads */

typedef struct {
    const double *sep_w, *sep_b, *g1, *b1, *pw, *pb, *g2, *b2;
} dds_layer_t;

typedef struct {
    const double *pre_w, *pre_b, *proj_w, *proj_b;
    dds_layer_t   l[3];
} dp_flow_t;

static struct {
    double       *w;                  /* every weight, float32 values as double */
    const double *pre_w, *pre_b, *proj_w, *proj_b, *m, *enl;
    dds_layer_t   convs[3];
    dp_flow_t     flow[3];            /* dp.flows.7, .5, .3 */
} s_dp;

/* ---- weights ------------------------------------------------------------ */

static const double *take(const double **p, size_t n)
{
    const double *r = *p;
    *p += n;
    return r;
}

static void take_dds(const double **p, dds_layer_t *l)
{
    int i;
    for (i = 0; i < 3; i++) {
        l[i].sep_w = take(p, DC * 3);
        l[i].sep_b = take(p, DC);
        l[i].g1 = take(p, DC);
        l[i].b1 = take(p, DC);
        l[i].pw = take(p, DC * DC);
        l[i].pb = take(p, DC);
        l[i].g2 = take(p, DC);
        l[i].b2 = take(p, DC);
    }
}

int tts_dp_load(const char *path, size_t expect_floats)
{
    FILE         *f = fopen(path, "rb");
    float        *raw;
    size_t        i, got;
    const double *p;
    int           k;
    if (!f)
        return -1;
    raw = malloc(expect_floats * sizeof(float));
    tts_dp_free();
    s_dp.w = malloc(expect_floats * sizeof(double));
    if (!raw || !s_dp.w) {
        fclose(f);
        free(raw);
        tts_dp_free();
        return -2;
    }
    got = fread(raw, sizeof(float), expect_floats, f);
    k = fgetc(f);
    fclose(f);
    if (got != expect_floats || k != EOF) {
        free(raw);
        tts_dp_free();
        return -3;                      /* a different dp.dat layout */
    }
    for (i = 0; i < expect_floats; i++)
        s_dp.w[i] = (double)raw[i];
    free(raw);
    p = s_dp.w;
    s_dp.pre_w = take(&p, DC * DC);
    s_dp.pre_b = take(&p, DC);
    take_dds(&p, s_dp.convs);
    s_dp.proj_w = take(&p, DC * DC);
    s_dp.proj_b = take(&p, DC);
    for (k = 0; k < 3; k++) {
        s_dp.flow[k].pre_w = take(&p, DC);
        s_dp.flow[k].pre_b = take(&p, DC);
        take_dds(&p, s_dp.flow[k].l);
        s_dp.flow[k].proj_w = take(&p, (size_t)NP * DC);
        s_dp.flow[k].proj_b = take(&p, NP);
    }
    s_dp.m = take(&p, 2);
    s_dp.enl = take(&p, 2);
    if ((size_t)(p - s_dp.w) != expect_floats) {
        tts_dp_free();
        return -3;
    }
    return 0;
}

void tts_dp_free(void)
{
    free(s_dp.w);
    memset(&s_dp, 0, sizeof s_dp);
}

int tts_dp_loaded(void) { return s_dp.w != NULL; }

/* ---- threads -------------------------------------------------------------- */

typedef void (*dp_fn)(void *arg, int r0, int r1);

typedef struct {
    dp_fn fn;
    void *arg;
    int   r0, r1;
} dp_job_t;

static void *dp_runner(void *p)
{
    dp_job_t *j = (dp_job_t *)p;
    j->fn(j->arg, j->r0, j->r1);
    return NULL;
}

/* fn over rows [0, n) split in `threads` ranges; the caller takes the first */
static void dp_parallel(dp_fn fn, void *arg, int n, int threads)
{
    pthread_t th[MAXT];
    dp_job_t  job[MAXT];
    int       started[MAXT] = {0};
    int       k;
    if (threads > MAXT)
        threads = MAXT;
    if (threads <= 1 || n < 4 * threads) {
        fn(arg, 0, n);
        return;
    }
    for (k = 1; k < threads; k++) {
        job[k].fn = fn;
        job[k].arg = arg;
        job[k].r0 = (int)((long)n * k / threads);
        job[k].r1 = (int)((long)n * (k + 1) / threads);
        started[k] = pthread_create(&th[k], NULL, dp_runner, &job[k]) == 0;
        if (!started[k])
            fn(arg, job[k].r0, job[k].r1);
    }
    fn(arg, 0, (int)((long)n / threads));
    for (k = 1; k < threads; k++)
        if (started[k])
            pthread_join(th[k], NULL);
}

/* ---- the operations (per row of DC channels) ------------------------------ */

static inline double gelu(double x)            /* piper_vits._gelu_seq */
{
    const double u = x * 0.7071067811865475;   /* 1.0 / math.sqrt(2.0) */
    const double a = fabs(u);
    double       t = 1.0 / (a * 0.3275911 + 1.0);
    double       p = ((((t * 1.061405429 - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
                      0.254829592) * t;
    const double sg = u > 0.0 ? 1.0 : (u < 0.0 ? -1.0 : 0.0);
    p = 1.0 - p * exp((-a) * a);
    return (x * 0.5) * (p * sg + 1.0);
}

/* LayerNorm over one row, then GELU, in place */
static void ln_gelu(double *v, const double *g, const double *b)
{
    double sum = 0.0, mean, acc = 0.0, sd;
    int    c;
    for (c = 0; c < DC; c++)
        sum += v[c];
    mean = sum / DC;
    for (c = 0; c < DC; c++) {
        const double d = v[c] - mean;
        acc += d * d;
    }
    sd = 1.0 / sqrt(acc / DC + 1e-5);
    for (c = 0; c < DC; c++)
        v[c] = gelu((v[c] - mean) * sd * g[c] + b[c]);
}

/* two rows at once, four outputs at a time: eight independent sums (the
 * in-order A53 hides the add latency; each weight load serves two rows) */
static void conv1x1_2rows(const double *restrict x0, const double *restrict x1, const double *restrict w,
                          const double *restrict b, int O, double *restrict y0, double *restrict y1)
{
    int o, c;
    for (o = 0; o + 4 <= O; o += 4) {
        const double *w0 = w + (size_t)o * DC, *w1 = w0 + DC, *w2 = w1 + DC, *w3 = w2 + DC;
        double        a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
        for (c = 0; c < DC; c++) {
            const double u = x0[c], v = x1[c];
            const double p = w0[c], q = w1[c], r = w2[c], s = w3[c];
            a0 += p * u;
            b0 += p * v;
            a1 += q * u;
            b1 += q * v;
            a2 += r * u;
            b2 += r * v;
            a3 += s * u;
            b3 += s * v;
        }
        y0[o] = a0 + b[o];
        y0[o + 1] = a1 + b[o + 1];
        y0[o + 2] = a2 + b[o + 2];
        y0[o + 3] = a3 + b[o + 3];
        y1[o] = b0 + b[o];
        y1[o + 1] = b1 + b[o + 1];
        y1[o + 2] = b2 + b[o + 2];
        y1[o + 3] = b3 + b[o + 3];
    }
    for (; o < O; o++) {
        const double *wo = w + (size_t)o * DC;
        double        a = 0.0, bb = 0.0;
        for (c = 0; c < DC; c++) {
            a += wo[c] * x0[c];
            bb += wo[c] * x1[c];
        }
        y0[o] = a + b[o];
        y1[o] = bb + b[o];
    }
}

/* y[o] = (sum_c w[o][c] * x[c], left to right) + b[o]; four outputs at a time */
static void conv1x1_row(const double *restrict x, const double *restrict w, const double *restrict b,
                        int O, double *restrict y)
{
    int o, c;
    for (o = 0; o + 4 <= O; o += 4) {
        const double *w0 = w + (size_t)o * DC, *w1 = w0 + DC, *w2 = w1 + DC, *w3 = w2 + DC;
        double        a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        for (c = 0; c < DC; c++) {
            const double xv = x[c];
            a0 += w0[c] * xv;
            a1 += w1[c] * xv;
            a2 += w2[c] * xv;
            a3 += w3[c] * xv;
        }
        y[o] = a0 + b[o];
        y[o + 1] = a1 + b[o + 1];
        y[o + 2] = a2 + b[o + 2];
        y[o + 3] = a3 + b[o + 3];
    }
    for (; o < O; o++) {
        const double *wo = w + (size_t)o * DC;
        double        a = 0.0;
        for (c = 0; c < DC; c++)
            a += wo[c] * x[c];
        y[o] = a + b[o];
    }
}

/* ---- the stages ------------------------------------------------------------- */

typedef struct {
    const double *x;                   /* [n][DC] input */
    double       *y;                   /* [n][O] output */
    const double *w, *b, *g, *gb;      /* 1x1 conv; g / gb: LayerNorm + GELU after (or NULL) */
    const double *res;                 /* the DDS residual: y = res + (...) (or NULL) */
    int           O, n, dil;
    const dds_layer_t *l;
} dp_stage_t;

/* y = x (1x1 conv) [-> LN -> GELU] [+ res] */
static void st_conv(void *p, int r0, int r1)
{
    const dp_stage_t *s = (const dp_stage_t *)p;
    int               t, c, k;
    for (t = r0; t < r1; t += 2) {
        const int two = t + 1 < r1;
        double   *y = s->y + (size_t)t * s->O;
        if (two)
            conv1x1_2rows(s->x + (size_t)t * DC, s->x + (size_t)(t + 1) * DC, s->w, s->b, s->O, y, y + s->O);
        else
            conv1x1_row(s->x + (size_t)t * DC, s->w, s->b, s->O, y);
        for (k = 0; k <= two; k++) {
            double *yk = y + (size_t)k * s->O;
            if (s->g)
                ln_gelu(yk, s->g, s->gb);
            if (s->res)
                for (c = 0; c < DC; c++)
                    yk[c] = s->res[(size_t)(t + k) * DC + c] + yk[c];
        }
    }
}

/* y = GELU(LN(depthwise kernel-3 conv of x, dilation dil)) */
static void st_dw(void *p, int r0, int r1)
{
    const dp_stage_t  *s = (const dp_stage_t *)p;
    const dds_layer_t *l = s->l;
    int                t, c, k;
    for (t = r0; t < r1; t++) {
        double *y = s->y + (size_t)t * DC;
        for (c = 0; c < DC; c++) {
            double acc = 0.0;
            for (k = 0; k < 3; k++) {
                const int    r  = t + (k - 1) * s->dil;
                const double xv = r >= 0 && r < s->n ? s->x[(size_t)r * DC + c] : 0.0;
                acc = acc + l->sep_w[c * 3 + k] * xv;
            }
            y[c] = acc + l->sep_b[c];
        }
        ln_gelu(y, l->g1, l->b1);
    }
}

/* the DDS stack on x [n][DC] in place (tmp: [n][DC] scratch x 2) */
static void dds(double *x, int n, const dds_layer_t *l, double *t1, double *t2, int threads)
{
    int i;
    for (i = 0; i < 3; i++) {
        dp_stage_t a = {0}, b = {0};
        a.x = x; a.y = t1; a.n = n; a.dil = i == 0 ? 1 : (i == 1 ? 3 : 9); a.l = &l[i];
        dp_parallel(st_dw, &a, n, threads);
        b.x = t1; b.y = t2; b.w = l[i].pw; b.b = l[i].pb; b.g = l[i].g2; b.gb = l[i].b2; b.res = x;
        b.O = DC; b.n = n;
        dp_parallel(st_conv, &b, n, threads);
        memcpy(x, t2, (size_t)n * DC * sizeof(double));
    }
}

/* ---- the spline (piper_vits.rq_spline_inverse / _spline_seq) ----------------- */

static void smax(const double *u, double *out)
{
    double m = u[0], s = 0.0;
    int    k;
    for (k = 1; k < NB; k++)
        m = u[k] > m ? u[k] : m;
    for (k = 0; k < NB; k++) {
        out[k] = exp(u[k] - m);
        s += out[k];
    }
    for (k = 0; k < NB; k++)
        out[k] = out[k] / s;
}

static double spline_inverse(double x, const double *hh)
{
    const double tb = 5.0, mbw = 1e-3, mbh = 1e-3, md = 1e-3;
    double       uw[NB], uh[NB], ud[NB + 1], sm[NB], cw[NB + 1], ch[NB + 1], wd[NB], ht[NB], der[NB + 1];
    double       acc, icw, ibw, ich, ih, ide, idl, idr, a, b, c, root;
    int          k, cnt, bi;
    if (!(x >= -tb && x <= tb))
        return x;                                  /* the linear tails */
    for (k = 0; k < NB; k++) {
        uw[k] = hh[k] / 13.856406460551018;        /* math.sqrt(192) */
        uh[k] = hh[NB + k] / 13.856406460551018;
    }
    ud[0] = ud[NB] = log(exp(1 - md) - 1);
    for (k = 1; k < NB; k++)
        ud[k] = hh[2 * NB + k - 1];
    smax(uw, sm);
    cw[0] = 0.0;
    acc = 0.0;
    for (k = 0; k < NB; k++) {
        acc = acc + (mbw + (1 - mbw * NB) * sm[k]);
        cw[k + 1] = acc;
    }
    for (k = 0; k <= NB; k++)
        cw[k] = 2 * tb * cw[k] - tb;
    cw[0] = -tb;
    cw[NB] = tb;
    for (k = 0; k < NB; k++)
        wd[k] = cw[k + 1] - cw[k];
    for (k = 0; k <= NB; k++)
        der[k] = md + (log1p(exp(-fabs(ud[k]))) + (ud[k] > 0.0 ? ud[k] : 0.0));
    smax(uh, sm);
    ch[0] = 0.0;
    acc = 0.0;
    for (k = 0; k < NB; k++) {
        acc = acc + (mbh + (1 - mbh * NB) * sm[k]);
        ch[k + 1] = acc;
    }
    for (k = 0; k <= NB; k++)
        ch[k] = 2 * tb * ch[k] - tb;
    ch[0] = -tb;
    ch[NB] = tb;
    for (k = 0; k < NB; k++)
        ht[k] = ch[k + 1] - ch[k];
    cnt = 0;
    for (k = 0; k <= NB; k++)
        cnt += x >= (k == NB ? ch[k] + 1e-6 : ch[k]);
    bi = cnt - 1;
    bi = bi < 0 ? 0 : (bi > NB - 1 ? NB - 1 : bi);
    icw = cw[bi];
    ibw = wd[bi];
    ich = ch[bi];
    ih = ht[bi];
    ide = ht[bi] / wd[bi];
    idl = der[bi];
    idr = der[bi + 1];
    a = (x - ich) * (idl + idr - 2 * ide) + ih * (ide - idl);
    b = ih * idl - (x - ich) * (idl + idr - 2 * ide);
    c = -ide * (x - ich);
    root = (2 * c) / (-b - sqrt(b * b - 4 * a * c));
    return root * ibw + icw;
}

/* ---- the predictor ------------------------------------------------------------- */

typedef struct {
    double       *h;
    const double *g, *z0, *pw, *pb;
    int           n;
} dp_pre_t;

/* h[t][c] = (w[c] * z0[t] + b[c]) + g[t][c]: the flow's 1 -> DC conv, then + the condition */
static void st_flow_pre(void *p, int r0, int r1)
{
    const dp_pre_t *s = (const dp_pre_t *)p;
    int             t, c;
    for (t = r0; t < r1; t++)
        for (c = 0; c < DC; c++)
            s->h[(size_t)t * DC + c] = (s->pw[c] * s->z0[t] + s->pb[c]) + s->g[(size_t)t * DC + c];
}

int tts_dp_run(const float *xin, int n, const double *z, double *logw, int threads)
{
    double    *x, *g, *t1, *t2, *hh, *z0, *z1, *tmp;
    int        t, c, k;
    dp_stage_t s = {0};
    if (!s_dp.w)
        return -1;
    if (n < 1)
        return -2;
    x = malloc((size_t)n * DC * sizeof(double));
    g = malloc((size_t)n * DC * sizeof(double));
    t1 = malloc((size_t)n * DC * sizeof(double));
    t2 = malloc((size_t)n * DC * sizeof(double));
    hh = malloc((size_t)n * NP * sizeof(double));
    z0 = malloc((size_t)n * sizeof(double));
    z1 = malloc((size_t)n * sizeof(double));
    if (!x || !g || !t1 || !t2 || !hh || !z0 || !z1) {
        free(x); free(g); free(t1); free(t2); free(hh); free(z0); free(z1);
        return -3;
    }
    for (t = 0; t < n; t++)                                  /* channel-major float -> rows */
        for (c = 0; c < DC; c++)
            t1[(size_t)t * DC + c] = (double)xin[(size_t)c * (size_t)n + (size_t)t];
    s.x = t1; s.y = x; s.w = s_dp.pre_w; s.b = s_dp.pre_b; s.O = DC; s.n = n;
    dp_parallel(st_conv, &s, n, threads);                    /* dp.pre */
    dds(x, n, s_dp.convs, t1, t2, threads);                  /* dp.convs */
    s.x = x; s.y = g; s.w = s_dp.proj_w; s.b = s_dp.proj_b;
    dp_parallel(st_conv, &s, n, threads);                    /* dp.proj: the condition g */
    memcpy(z0, z, (size_t)n * sizeof(double));
    memcpy(z1, z + n, (size_t)n * sizeof(double));
    for (k = 0; k < 3; k++) {                                /* dp.flows.7, .5, .3 */
        const dp_flow_t *fl = &s_dp.flow[k];
        dp_pre_t         pp;
        dp_stage_t       ps = {0};
        tmp = z0; z0 = z1; z1 = tmp;                         /* Flip */
        pp.h = x; pp.g = g; pp.z0 = z0; pp.pw = fl->pre_w; pp.pb = fl->pre_b; pp.n = n;
        dp_parallel(st_flow_pre, &pp, n, threads);
        dds(x, n, fl->l, t1, t2, threads);
        ps.x = x; ps.y = hh; ps.w = fl->proj_w; ps.b = fl->proj_b; ps.O = NP; ps.n = n;
        dp_parallel(st_conv, &ps, n, threads);
        for (t = 0; t < n; t++)
            z1[t] = spline_inverse(z1[t], hh + (size_t)t * NP);
    }
    tmp = z0; z0 = z1; z1 = tmp;                             /* the last Flip */
    for (t = 0; t < n; t++)
        logw[t] = (z0[t] - s_dp.m[0]) * s_dp.enl[0];
    free(x); free(g); free(t1); free(t2); free(hh); free(z0); free(z1);
    return 0;
}

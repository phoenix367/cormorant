// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * vjepa2_api.c — see vjepa2_api.h.  The model's constants and the run function
 * come from the generated test/vjepa2_glue.h; VJEPA2_API_WEIGHTS_DIR from the
 * CMake targets added by demo/video_classification/scripts/generate_vjepa2_project.py.
 * The frame sampling, the resize and the crop are the specification's
 * (demo/video_classification/scripts/vjepa2_q.py sample_frames, resample_taps,
 * resize_short, center_crop, patches), operation for operation.
 */

#define _POSIX_C_SOURCE 200809L   /* clock_gettime, fchdir, realpath */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "vjepa2_api.h"
#include "inference.h"
#include "vjepa2_glue.h"

/* No FMA contraction: the resize is the specification's separate double
 * multiplies and adds (GCC's GNU modes default to -ffp-contract=fast, which
 * fuses a*b + c on the A53 — a pixel at a rounding boundary then differs). */
#if defined(__clang__)
#  pragma clang fp contract(off)
#elif defined(__GNUC__)
#  pragma GCC optimize ("fp-contract=off")
#endif

#ifndef VJEPA2_API_WEIGHTS_DIR
#  define VJEPA2_API_WEIGHTS_DIR "."
#endif

#define VJ_PATCH_K   (3 * VJ_TUBELET * VJ_PATCH * VJ_PATCH)   /* a patch row: (channel, frame, row, column) */
#define VJ_GRID      (VJ_CROP / VJ_PATCH)
#define VJ_FRAME_SZ  ((size_t)VJ_CROP * VJ_CROP * 3u)

static int              s_open;
static char             s_err[512];
static inference_buf_t *s_rows[VJ_PIECES];               /* the entry's inputs: one per row piece */
static uint8_t          s_clip[VJ_FRAMES * VJ_CROP * VJ_CROP * 3];
static float            s_logits[VJ_CLASSES];
static vjepa2_stats_t   s_stats;

static void set_err(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_err, sizeof s_err, fmt, ap);
    va_end(ap);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int same_dir(const char *a, const char *b)
{
    char   ra[PATH_MAX], rb[PATH_MAX];
    size_t la, lb;
    if (realpath(a, ra) && realpath(b, rb))
        return strcmp(ra, rb) == 0;
    la = strlen(a);
    lb = strlen(b);
    while (la > 1u && a[la - 1u] == '/') la--;
    while (lb > 1u && b[lb - 1u] == '/') lb--;
    return la == lb && strncmp(a, b, la) == 0;
}

int vjepa2_open(const char *weights_dir)
{
    int rc, cwd = -1, p;

    s_err[0] = '\0';
    if (s_open)
        return 0;
    if (weights_dir && *weights_dir && !same_dir(weights_dir, VJEPA2_API_WEIGHTS_DIR)) {
        if (VJEPA2_API_WEIGHTS_DIR[0] == '/') {
            set_err("this build reads its weights from %s/weights (INFERENCE_WEIGHTS_DIR); "
                    "rebuild with -DINFERENCE_WEIGHTS_DIR=%s to use %s",
                    VJEPA2_API_WEIGHTS_DIR, weights_dir, weights_dir);
            return -2;
        }
        cwd = open(".", O_RDONLY);
        if (cwd < 0 || chdir(weights_dir) != 0) {
            set_err("chdir(%s): %s", weights_dir, strerror(errno));
            if (cwd >= 0)
                close(cwd);
            return -3;
        }
    }
    rc = vj_glue_init();
    if (cwd >= 0) {
        if (fchdir(cwd) != 0)
            fprintf(stderr, "vjepa2_api: warning: cannot restore the working directory: %s\n", strerror(errno));
        close(cwd);
    }
    if (rc != 0) {
        set_err("inference_init() failed (%d): check the UIO devices, CMA (CmaFree) and the weights under "
                "%s/weights", rc, weights_dir && *weights_dir ? weights_dir : VJEPA2_API_WEIGHTS_DIR);
        return -4;
    }
    for (p = 0; p < VJ_PIECES; p++) {
        s_rows[p] = inference_buf_alloc(vj_piece_rows[p] * VJ_PATCH_K);
        if (!s_rows[p]) {
            while (p-- > 0)
                inference_buf_free(s_rows[p]);
            inference_deinit();
            set_err("cannot allocate the patch rows (%u values)", (unsigned)(VJ_TOKENS * VJ_PATCH_K));
            return -5;
        }
    }
    memset(&s_stats, 0, sizeof s_stats);
    s_open = 1;
    return 0;
}

void vjepa2_close(void)
{
    int p;
    if (!s_open)
        return;
    for (p = 0; p < VJ_PIECES; p++) {
        inference_buf_free(s_rows[p]);
        s_rows[p] = NULL;
    }
    inference_deinit();
    s_open = 0;
}

const char *vjepa2_last_error(void)  { return s_err; }
int vjepa2_frames(void)              { return VJ_FRAMES; }
int vjepa2_crop(void)                { return VJ_CROP; }
int vjepa2_short_side(void)          { return VJ_SHORT; }
int vjepa2_classes(void)             { return VJ_CLASSES; }
const char *vjepa2_weights_dir(void) { return VJEPA2_API_WEIGHTS_DIR; }

const char *vjepa2_label(int c)
{
    return c >= 0 && c < VJ_CLASSES ? vj_labels[c] : NULL;
}

/* numpy.linspace(0, stop, num)[i]: i * (stop / (num - 1)) + 0, the last one stop. */
static double linspace(double stop, int num, int i)
{
    if (num > 1 && i == num - 1)
        return stop;
    return num > 1 ? (double)i * (stop / (double)(num - 1)) + 0.0 : 0.0;
}

/* vjepa2_q.sample_frames: 16 frames over the first 64 at step ~4 of a longer video,
 * else every ~4th frame padded with the last one; np.clip, then the truncation of astype. */
int vjepa2_sample_frames(int n, int *idx)
{
    const int step = 4, clip_len = VJ_FRAMES * step;
    int       i;
    if (n < 1 || !idx) {
        set_err("vjepa2_sample_frames: a video of %d frames", n);
        return -1;
    }
    if (n > clip_len) {
        for (i = 0; i < VJ_FRAMES; i++) {
            double v = linspace((double)clip_len, VJ_FRAMES, i);
            if (v > clip_len - 1) v = clip_len - 1;
            idx[i] = (int)v;
        }
        return 0;
    }
    {
        const int sample_len = (n < clip_len ? n : clip_len) - 1, num = sample_len / step;
        const int hi = sample_len - 1;       /* np.clip(idx, 0, sample_len - 1): above all bounds -> hi */
        for (i = 0; i < VJ_FRAMES; i++) {
            double v = i < num ? linspace((double)sample_len, num, i) : (double)sample_len;
            if (v < 0.0) v = 0.0;
            if (v > (double)hi) v = (double)hi;
            idx[i] = (int)v;
            if (idx[i] < 0) idx[i] += n;     /* hi = -1 (one frame): numpy's index -1, the last frame */
        }
    }
    return 0;
}

/* vjepa2_q.resample_taps: output i = sum_j w[i][j] x[lo[i] + j] (taps[i] of them). */
typedef struct {
    int     n_out, max_taps;
    int    *lo, *taps;
    double *w;
} taps_t;

static void taps_free(taps_t *t)
{
    free(t->lo);
    free(t->taps);
    free(t->w);
}

static int taps_make(int n_in, int n_out, taps_t *t)
{
    const double scale = (double)n_in / (double)n_out;
    const double fs = scale > 1.0 ? scale : 1.0, support = fs;
    int          i, k;
    t->n_out = n_out;
    t->max_taps = (int)(2.0 * support) + 3;
    t->lo = (int *)malloc((size_t)n_out * sizeof(int));
    t->taps = (int *)malloc((size_t)n_out * sizeof(int));
    t->w = (double *)malloc((size_t)n_out * (size_t)t->max_taps * sizeof(double));
    if (!t->lo || !t->taps || !t->w) {
        taps_free(t);
        return -1;
    }
    for (i = 0; i < n_out; i++) {
        const double c = ((double)i + 0.5) * scale;
        int          a = (int)(c - support + 0.5), b = (int)(c + support + 0.5);
        double       tot = 0.0, *w = t->w + (size_t)i * t->max_taps;
        if (a < 0) a = 0;
        if (b > n_in) b = n_in;
        if (b - a > t->max_taps) b = a + t->max_taps;         /* never: the filter's width */
        for (k = a; k < b; k++) {
            const double x = ((double)k - c + 0.5) / fs;
            const double v = 1.0 - fabs(x);
            w[k - a] = v > 0.0 ? v : 0.0;
            tot += w[k - a];
        }
        for (k = a; k < b; k++)
            w[k - a] = w[k - a] / tot;
        t->lo[i] = a;
        t->taps[i] = b - a;
    }
    return 0;
}

/* floor(v + 0.5) clipped to [0, 255] */
static double round_px(double v)
{
    const double r = floor(v + 0.5);
    return r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r);
}

/* vjepa2_q.make_clip's per-frame part: resize_short then center_crop — only the
 * cropped outputs computed (each output is its own sum of taps). */
int vjepa2_frame(const uint8_t *rgb, int h, int w, uint8_t *out)
{
    const int S = VJ_CROP, sh = VJ_SHORT;
    int       oh, ow, top, left, y, x, ch, j;
    taps_t    th = {0}, tv = {0};
    double   *mid;
    if (!rgb || !out || h < 1 || w < 1 || h > 16384 || w > 16384) {
        set_err("vjepa2_frame: a frame of %d x %d", h, w);
        return -1;
    }
    if (h <= w) {
        oh = sh;
        ow = (int)((double)((long long)sh * w) / (double)h);
    } else {
        oh = (int)((double)((long long)sh * h) / (double)w);
        ow = sh;
    }
    if (oh < S || ow < S) {
        set_err("vjepa2_frame: %d x %d resizes to %d x %d, below the %d crop", h, w, oh, ow, S);
        return -2;
    }
    top = (oh - S) / 2;
    left = (ow - S) / 2;
    if ((ow != w && taps_make(w, ow, &th)) || (oh != h && taps_make(h, oh, &tv))) {
        taps_free(&th);
        set_err("vjepa2_frame: out of memory");
        return -3;
    }
    mid = (double *)malloc((size_t)h * S * 3u * sizeof(double));     /* the horizontal pass: [h][S][3] */
    if (!mid) {
        taps_free(&th);
        taps_free(&tv);
        set_err("vjepa2_frame: out of memory");
        return -3;
    }
    for (y = 0; y < h; y++)
        for (x = 0; x < S; x++) {
            const int xo = left + x;
            for (ch = 0; ch < 3; ch++) {
                double v;
                if (ow == w) {
                    v = (double)rgb[((size_t)y * w + xo) * 3 + ch];
                } else {
                    const double *wt = th.w + (size_t)xo * th.max_taps;
                    const int     lo = th.lo[xo];
                    v = wt[0] * (double)rgb[((size_t)y * w + lo) * 3 + ch];
                    for (j = 1; j < th.taps[xo]; j++)
                        v = v + wt[j] * (double)rgb[((size_t)y * w + lo + j) * 3 + ch];
                    v = round_px(v);
                }
                mid[((size_t)y * S + x) * 3 + ch] = v;
            }
        }
    for (y = 0; y < S; y++) {
        const int yo = top + y;
        for (x = 0; x < S; x++)
            for (ch = 0; ch < 3; ch++) {
                double v;
                if (oh == h) {
                    v = mid[((size_t)yo * S + x) * 3 + ch];
                } else {
                    const double *wt = tv.w + (size_t)yo * tv.max_taps;
                    const int     lo = tv.lo[yo];
                    v = wt[0] * mid[((size_t)lo * S + x) * 3 + ch];
                    for (j = 1; j < tv.taps[yo]; j++)
                        v = v + wt[j] * mid[((size_t)(lo + j) * S + x) * 3 + ch];
                    v = round_px(v);
                }
                out[((size_t)y * S + x) * 3 + ch] = (uint8_t)v;
            }
    }
    free(mid);
    taps_free(&th);
    taps_free(&tv);
    return 0;
}

/* vjepa2_q.patches into the row pieces: token t = (frame pair f, row r, column k),
 * element ((c tub + j) P + y) P + x = clip[tub f + j][r P + y][k P + x][c]. */
static void patch_rows(const uint8_t *clip)
{
    const unsigned P = VJ_PATCH, g = VJ_GRID, tb = VJ_TUBELET;
    unsigned       p, t = 0u;
    for (p = 0u; p < VJ_PIECES; p++) {
        Data_t  *o = inference_buf_ptr(s_rows[p]);
        unsigned i;
        for (i = 0u; i < vj_piece_rows[p]; i++, t++) {
            const unsigned f = t / (g * g), r = (t / g) % g, k = t % g;
            unsigned       c, j, y, x;
            Data_t        *row = o + (size_t)i * VJ_PATCH_K;
            for (c = 0u; c < 3u; c++)
                for (j = 0u; j < tb; j++) {
                    const uint8_t *fr = clip + (size_t)(tb * f + j) * VJ_FRAME_SZ;
                    for (y = 0u; y < P; y++)
                        for (x = 0u; x < P; x++)
                            row[((c * tb + j) * P + y) * P + x] =
                                (Data_t)fr[((size_t)(r * P + y) * VJ_CROP + k * P + x) * 3u + c];
                }
        }
    }
}

int vjepa2_classify_clip(const uint8_t *clip, float *logits)
{
    double t0 = now_ms();
    if (!s_open) {
        set_err("vjepa2_classify_clip: the model is not open");
        return -1;
    }
    if (!clip || !logits) {
        set_err("vjepa2_classify_clip: clip and logits needed");
        return -2;
    }
    patch_rows(clip);
    vj_glue_clip(s_rows, s_logits);
    memcpy(logits, s_logits, sizeof s_logits);
    s_stats.run_ms = now_ms() - t0;
    s_stats.prep_ms = 0.0;
    s_stats.total_ms = s_stats.run_ms;
    return 0;
}

int vjepa2_classify(const uint8_t *frames, int n, int h, int w, float *logits)
{
    int    idx[VJ_FRAMES], i, rc;
    double t0 = now_ms(), prep;
    if (!s_open) {
        set_err("vjepa2_classify: the model is not open");
        return -1;
    }
    if (!frames || !logits || n < 1) {
        set_err("vjepa2_classify: %d frames; frames and logits needed", n);
        return -2;
    }
    rc = vjepa2_sample_frames(n, idx);
    if (rc)
        return rc;
    for (i = 0; i < VJ_FRAMES; i++) {
        rc = vjepa2_frame(frames + (size_t)idx[i] * h * w * 3u, h, w, s_clip + (size_t)i * VJ_FRAME_SZ);
        if (rc)
            return rc;
    }
    prep = now_ms() - t0;
    rc = vjepa2_classify_clip(s_clip, logits);
    s_stats.prep_ms = prep;
    s_stats.total_ms = now_ms() - t0;
    return rc;
}

void vjepa2_last_stats(vjepa2_stats_t *st)
{
    if (st)
        *st = s_stats;
}

// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * vla_api.c — see vla_api.h.  The model's constants and the run functions
 * come from the generated test/vla_glue.h; VLA_API_WEIGHTS_DIR from the
 * CMake targets added by demo/vla/scripts/generate_vla_project.py.  The
 * resize, the prefix rows and the Euler steps are the specification's
 * (demo/vla/scripts/smolvla_q.py resize_with_pad / pixels, SmolVLA.sample),
 * operation for operation.
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

#include "vla_api.h"
#include "inference.h"
#include "vla_glue.h"

/* No FMA contraction: the resize and the Euler update are the specification's
 * separate double multiplies and adds (GCC's GNU modes default to
 * -ffp-contract=fast, which fuses a*b + c on the A53 — the last bit of an
 * action then differs). */
#if defined(__clang__)
#  pragma clang fp contract(off)
#elif defined(__GNUC__)
#  pragma GCC optimize ("fp-contract=off")
#endif

#ifndef VLA_API_WEIGHTS_DIR
#  define VLA_API_WEIGHTS_DIR "."
#endif

#define VLA_PATCH_ELEMS (3 * VLA_PATCH * VLA_PATCH)

static int              s_open;
static char             s_err[512];
static inference_buf_t *s_patches;                       /* the vision entry's input */
static int              s_have[VLA_CAMERAS];             /* camera c's image tokens are set */
static uint8_t          s_rgb[VLA_IMAGE * VLA_IMAGE * 3];
static vla_stats_t      s_stats;

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
    char ra[PATH_MAX], rb[PATH_MAX];
    size_t la, lb;
    if (realpath(a, ra) && realpath(b, rb))
        return strcmp(ra, rb) == 0;
    la = strlen(a);
    lb = strlen(b);
    while (la > 1u && a[la - 1u] == '/') la--;
    while (lb > 1u && b[lb - 1u] == '/') lb--;
    return la == lb && strncmp(a, b, la) == 0;
}

int vla_open(const char *weights_dir)
{
    int rc, cwd = -1;

    s_err[0] = '\0';
    if (s_open)
        return 0;
    if (weights_dir && *weights_dir && !same_dir(weights_dir, VLA_API_WEIGHTS_DIR)) {
        if (VLA_API_WEIGHTS_DIR[0] == '/') {
            set_err("this build reads its weights from %s/weights (INFERENCE_WEIGHTS_DIR); "
                    "rebuild with -DINFERENCE_WEIGHTS_DIR=%s to use %s",
                    VLA_API_WEIGHTS_DIR, weights_dir, weights_dir);
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
    rc = vla_glue_init();
    if (cwd >= 0) {
        if (fchdir(cwd) != 0)
            fprintf(stderr, "vla_api: warning: cannot restore the working directory: %s\n", strerror(errno));
        close(cwd);
    }
    if (rc != 0) {
        set_err("inference_init() failed (%d): check the UIO devices, CMA (CmaFree) and the weights under "
                "%s/weights", rc, weights_dir && *weights_dir ? weights_dir : VLA_API_WEIGHTS_DIR);
        return -4;
    }
    s_patches = inference_buf_alloc(VLA_PATCHES * VLA_PATCH_ELEMS);
    if (!s_patches) {
        inference_deinit();
        set_err("cannot allocate the patches buffer (%u values)", (unsigned)(VLA_PATCHES * VLA_PATCH_ELEMS));
        return -5;
    }
    memset(s_have, 0, sizeof s_have);
    memset(&s_stats, 0, sizeof s_stats);
    s_open = 1;
    return 0;
}

void vla_close(void)
{
    if (!s_open)
        return;
    inference_buf_free(s_patches);
    s_patches = NULL;
    inference_deinit();
    s_open = 0;
}

const char *vla_last_error(void) { return s_err; }
int vla_cameras(void)            { return VLA_CAMERAS; }
int vla_image_size(void)         { return VLA_IMAGE; }
int vla_tok_max(void)            { return VLA_TOK_MAX; }
int vla_state_size(void)         { return VLA_STATE; }
int vla_action_size(void)        { return VLA_ACTION; }
int vla_state_dim(void)          { return VLA_STATE_DIM; }
int vla_action_dim(void)         { return VLA_ACTION_DIM; }
int vla_chunk(void)              { return VLA_CHUNK; }
int vla_steps(void)              { return VLA_STEPS; }
const char *vla_weights_dir(void) { return VLA_API_WEIGHTS_DIR; }

/* One axis of torch's bilinear interpolation (align_corners False, no antialias):
 * src = max((j + 0.5) * (n_in / n_out) - 0.5, 0), i0 = (int)src, i1 = min(i0 + 1, n_in - 1),
 * lam = src - i0 (smolvla_q.bilinear). */
static void axis(int n_in, int n_out, int j, int *i0, int *i1, double *lam)
{
    const double sc = (double)n_in / (double)n_out;
    double       src = ((double)j + 0.5) * sc - 0.5;
    if (src < 0.0) src = 0.0;
    *i0 = (int)src;
    *i1 = *i0 + 1 < n_in ? *i0 + 1 : n_in - 1;
    *lam = src - (double)*i0;
}

int vla_resize(const uint8_t *rgb, int h, int w, uint8_t *out)
{
    const int S = VLA_IMAGE;
    double    ratio, *row;
    int       rh, rw, y, x, ch;
    if (!rgb || !out || h < 1 || w < 1 || h > 16384 || w > 16384) {
        set_err("vla_resize: an image of %d x %d", h, w);
        return -1;
    }
    if (h == S && w == S) {
        memcpy(out, rgb, (size_t)S * S * 3u);
        return 0;
    }
    /* resize_with_pad: the aspect ratio kept, zero padding on the left and top */
    ratio = (double)w / S > (double)h / S ? (double)w / S : (double)h / S;
    rh = (int)((double)h / ratio);
    rw = (int)((double)w / ratio);
    if (rh < 1 || rw < 1 || rh > S || rw > S) {
        set_err("vla_resize: %d x %d -> %d x %d", h, w, rh, rw);
        return -2;
    }
    row = (double *)malloc((size_t)rh * w * 3u * sizeof(double));     /* the rows pass */
    if (!row) {
        set_err("vla_resize: out of memory");
        return -3;
    }
    memset(out, 0, (size_t)S * S * 3u);
    /* rows first: top[y][x] = img[y0][x] (1 - ly) + img[y1][x] ly over the source columns */
    for (y = 0; y < rh; y++) {
        int    y0, y1;
        double ly;
        axis(h, rh, y, &y0, &y1, &ly);
        for (x = 0; x < w; x++)
            for (ch = 0; ch < 3; ch++)
                row[((size_t)y * w + x) * 3 + ch] =
                    (double)rgb[((size_t)y0 * w + x) * 3 + ch] * (1.0 - ly)
                    + (double)rgb[((size_t)y1 * w + x) * 3 + ch] * ly;
    }
    for (y = 0; y < rh; y++)
        for (x = 0; x < rw; x++) {
            int    x0, x1;
            double lx;
            axis(w, rw, x, &x0, &x1, &lx);
            for (ch = 0; ch < 3; ch++) {
                const double v = row[((size_t)y * w + x0) * 3 + ch] * (1.0 - lx)
                               + row[((size_t)y * w + x1) * 3 + ch] * lx;
                const double r = nearbyint(v);          /* round half to even: numpy's round */
                out[((size_t)(S - rh + y) * S + (S - rw + x)) * 3 + ch] =
                    (uint8_t)(r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r));
            }
        }
    free(row);
    return 0;
}

int vla_set_image(int c, const uint8_t *rgb, int h, int w)
{
    const unsigned S = VLA_IMAGE, P = VLA_PATCH, n = S / P;
    Data_t        *x;
    unsigned       r, k, ch, ky, kx;
    double         t0 = now_ms();
    int            rc;
    if (!s_open) {
        set_err("vla_set_image: the model is not open");
        return -1;
    }
    if (c < 0 || c >= VLA_CAMERAS) {
        set_err("vla_set_image: camera %d (the library takes %d)", c, VLA_CAMERAS);
        return -2;
    }
    rc = vla_resize(rgb, h, w, s_rgb);
    if (rc)
        return rc;
    /* patch t = r * n + k, element (channel, row, column) of the patch: the raw
     * uint8 value (exponent 0; the normalisation is in the weights) */
    x = inference_buf_ptr(s_patches);
    for (r = 0u; r < n; r++)
        for (k = 0u; k < n; k++) {
            Data_t *o = x + (size_t)(r * n + k) * VLA_PATCH_ELEMS;
            for (ch = 0u; ch < 3u; ch++)
                for (ky = 0u; ky < P; ky++)
                    for (kx = 0u; kx < P; kx++)
                        o[(ch * P + ky) * P + kx] = (Data_t)s_rgb[((size_t)(r * P + ky) * S + k * P + kx) * 3u + ch];
        }
    vla_glue_vision(s_patches);
    vla_glue_image(c);
    s_have[c] = 1;
    s_stats.image_ms = now_ms() - t0;
    return 0;
}

int vla_act(int n_cameras, const int32_t *ids, int n_ids, const float *state, int n_state,
            const float *noise, float *actions)
{
    static int32_t s_ids[VLA_TOK_MAX];
    static float   s_state[VLA_STATE];
    static float   s_x[VLA_CHUNK * VLA_ACTION], s_v[VLA_CHUNK * VLA_ACTION];
    const double   dt = -1.0 / VLA_STEPS;
    int32_t        n, step, pos, nr, xpos, zero = 0;
    double         t0, t1;
    int            i, c;
    if (!s_open) {
        set_err("vla_act: the model is not open");
        return -1;
    }
    if (n_cameras < 1 || n_cameras > VLA_CAMERAS) {
        set_err("vla_act: %d cameras (1 .. %d)", n_cameras, VLA_CAMERAS);
        return -2;
    }
    for (c = 0; c < n_cameras; c++)
        if (!s_have[c]) {
            set_err("vla_act: camera %d has no image (vla_set_image)", c);
            return -3;
        }
    if (n_ids < 0 || n_ids > VLA_TOK_MAX || (n_ids && !ids) || n_state < 0 || n_state > VLA_STATE
        || (n_state && !state) || !noise || !actions) {
        set_err("vla_act: ids %d (<= %d), state %d (<= %d), noise and actions needed", n_ids, VLA_TOK_MAX,
                n_state, VLA_STATE);
        return -4;
    }
    memset(s_ids, 0, sizeof s_ids);
    memset(s_state, 0, sizeof s_state);
    if (n_ids)
        memcpy(s_ids, ids, (size_t)n_ids * sizeof(int32_t));
    if (n_state)
        memcpy(s_state, state, (size_t)n_state * sizeof(float));
    t0 = now_ms();
    n = n_cameras * VLA_IMG_TOKENS + n_ids + 1;                  /* the prefix's valid rows */
    vla_glue_prefix(n_cameras, s_ids, &n, s_state, &zero);
    t1 = now_ms();
    memcpy(s_x, noise, sizeof s_x);
    pos = n;
    nr = VLA_CHUNK;
    xpos = n - VLA_CHUNK;
    for (step = 0; step < VLA_STEPS; step++) {
        vla_glue_expert(s_x, &step, &pos, &nr, &xpos, &zero, s_v);
        for (i = 0; i < VLA_CHUNK * VLA_ACTION; i++)
            s_x[i] = (float)((double)s_x[i] + dt * (double)s_v[i]);
    }
    memcpy(actions, s_x, sizeof s_x);
    s_stats.prefix_ms = t1 - t0;
    s_stats.expert_ms = now_ms() - t1;
    s_stats.act_ms = now_ms() - t0;
    s_stats.cameras = n_cameras;
    s_stats.steps = VLA_STEPS;
    return 0;
}

static uint64_t splitmix64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void vla_noise(uint64_t seed, float *noise)
{
    uint64_t s = seed;
    int      i;
    for (i = 0; i < VLA_CHUNK * VLA_ACTION; i += 2) {
        const double u1 = ((double)(splitmix64(&s) >> 11) + 1.0) / 9007199254740993.0;   /* (0, 1] */
        const double u2 = (double)(splitmix64(&s) >> 11) / 9007199254740992.0;
        const double r = sqrt(-2.0 * log(u1));
        noise[i] = (float)(r * cos(6.283185307179586 * u2));
        if (i + 1 < VLA_CHUNK * VLA_ACTION)
            noise[i + 1] = (float)(r * sin(6.283185307179586 * u2));
    }
}

void vla_last_stats(vla_stats_t *st)
{
    if (st)
        *st = s_stats;
}

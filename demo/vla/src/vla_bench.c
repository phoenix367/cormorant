// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * vla_bench.c — the gate runner of libsmolvla (demo/vla/scripts/vla_board.py,
 * vla_host_emu.py): samples in, action chunks out, timing.
 *
 *   vla_bench -i samples.bin -o actions.bin [-R reps] [-r] [-w weights_dir]
 *
 * samples.bin: int32 count, then per sample: int32 cameras; per camera int32 h,
 * int32 w and h * w * 3 uint8 (RGB, row-major); int32 n_ids and the ids
 * (int32); int32 n_state and the state (float32, normalised); the noise
 * [vla_chunk()][vla_action_size()] (float32).
 * actions.bin: per sample the chunk [vla_chunk()][vla_action_size()] (float32).
 * Every sample runs ``reps`` times (the chunks must be identical); -r closes
 * and reopens the library after the samples and runs the first one again
 * (identical).  Result lines on stdout (JSON after the tag): VLA_OPEN,
 * VLA_SAMPLE (one per sample), VLA_REOPEN, VLA_SUMMARY.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "vla_api.h"

typedef struct {
    int       cams;
    int       h[8], w[8];
    uint8_t  *rgb[8];
    int       n_ids, n_state;
    int32_t  *ids;
    float    *state, *noise;
} sample_t;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int rd(FILE *f, void *p, size_t n)
{
    return fread(p, 1u, n, f) == n ? 0 : -1;
}

static int load(const char *path, sample_t **out, int *count)
{
    FILE     *f = fopen(path, "rb");
    int32_t   n, i, c, v;
    sample_t *s;
    const size_t nz = (size_t)vla_chunk() * (size_t)vla_action_size();
    if (!f || rd(f, &n, 4) || n < 1 || n > 100000) {
        fprintf(stderr, "vla_bench: cannot read %s\n", path);
        if (f) fclose(f);
        return -1;
    }
    s = (sample_t *)calloc((size_t)n, sizeof *s);
    for (i = 0; i < n; i++) {
        if (rd(f, &v, 4) || v < 1 || v > 8) goto bad;
        s[i].cams = v;
        for (c = 0; c < s[i].cams; c++) {
            if (rd(f, &s[i].h[c], 4) || rd(f, &s[i].w[c], 4) || s[i].h[c] < 1 || s[i].w[c] < 1) goto bad;
            s[i].rgb[c] = (uint8_t *)malloc((size_t)s[i].h[c] * (size_t)s[i].w[c] * 3u);
            if (rd(f, s[i].rgb[c], (size_t)s[i].h[c] * (size_t)s[i].w[c] * 3u)) goto bad;
        }
        if (rd(f, &s[i].n_ids, 4) || s[i].n_ids < 0 || s[i].n_ids > 4096) goto bad;
        s[i].ids = (int32_t *)malloc((size_t)s[i].n_ids * 4u + 4u);
        if (rd(f, s[i].ids, (size_t)s[i].n_ids * 4u)) goto bad;
        if (rd(f, &s[i].n_state, 4) || s[i].n_state < 0 || s[i].n_state > 4096) goto bad;
        s[i].state = (float *)malloc((size_t)s[i].n_state * 4u + 4u);
        if (rd(f, s[i].state, (size_t)s[i].n_state * 4u)) goto bad;
        s[i].noise = (float *)malloc(nz * 4u);
        if (rd(f, s[i].noise, nz * 4u)) goto bad;
    }
    fclose(f);
    *out = s;
    *count = n;
    return 0;
bad:
    fprintf(stderr, "vla_bench: %s: sample %d is malformed\n", path, i);
    fclose(f);
    return -1;
}

/* One sample: every camera's image, then the chunk; per-camera image times into img_ms. */
static int run(const sample_t *s, float *act, double *img_ms)
{
    vla_stats_t st;
    int         c, rc;
    for (c = 0; c < s->cams; c++) {
        rc = vla_set_image(c, s->rgb[c], s->h[c], s->w[c]);
        if (rc) {
            fprintf(stderr, "vla_set_image(%d): %d %s\n", c, rc, vla_last_error());
            return rc;
        }
        vla_last_stats(&st);
        if (img_ms) img_ms[c] = st.image_ms;
    }
    rc = vla_act(s->cams, s->ids, s->n_ids, s->state, s->n_state, s->noise, act);
    if (rc)
        fprintf(stderr, "vla_act: %d %s\n", rc, vla_last_error());
    return rc;
}

int main(int argc, char **argv)
{
    const char *in = NULL, *outp = NULL, *wd = NULL;
    int         reps = 1, reopen = 0, opt, n = 0, i, r, c, all_same = 1;
    sample_t   *s = NULL;
    FILE       *fo;
    float      *act, *act2, *first;
    size_t      na;
    double      t0, tot_img = 0.0, tot_act = 0.0;
    while ((opt = getopt(argc, argv, "i:o:R:rw:")) != -1) {
        switch (opt) {
        case 'i': in = optarg; break;
        case 'o': outp = optarg; break;
        case 'R': reps = atoi(optarg); break;
        case 'r': reopen = 1; break;
        case 'w': wd = optarg; break;
        default:
            fprintf(stderr, "usage: %s -i samples.bin -o actions.bin [-R reps] [-r] [-w weights_dir]\n", argv[0]);
            return 2;
        }
    }
    if (!in || !outp || reps < 1) {
        fprintf(stderr, "vla_bench: -i and -o are needed\n");
        return 2;
    }
    t0 = now_ms();
    if (vla_open(wd) != 0) {
        fprintf(stderr, "vla_open: %s\n", vla_last_error());
        return 1;
    }
    printf("VLA_OPEN: {\"ms\": %.1f, \"cameras\": %d, \"chunk\": %d, \"action\": %d, \"steps\": %d}\n",
           now_ms() - t0, vla_cameras(), vla_chunk(), vla_action_size(), vla_steps());
    if (load(in, &s, &n))
        return 1;
    na = (size_t)vla_chunk() * (size_t)vla_action_size();
    act = (float *)malloc(na * sizeof(float));
    act2 = (float *)malloc(na * sizeof(float));
    first = (float *)malloc(na * sizeof(float));
    fo = fopen(outp, "wb");
    if (!fo) {
        fprintf(stderr, "vla_bench: cannot write %s\n", outp);
        return 1;
    }
    for (i = 0; i < n; i++) {
        double      img[8] = {0}, best_act = 1e300;
        vla_stats_t st;
        int         same = 1;
        if (s[i].cams > vla_cameras()) {
            fprintf(stderr, "vla_bench: sample %d has %d cameras (the library takes %d)\n", i, s[i].cams,
                    vla_cameras());
            return 1;
        }
        for (r = 0; r < reps; r++) {
            if (run(&s[i], r ? act2 : act, img))
                return 1;
            vla_last_stats(&st);
            if (st.act_ms < best_act) best_act = st.act_ms;
            if (r && memcmp(act, act2, na * sizeof(float)) != 0) same = 0;
        }
        if (i == 0)
            memcpy(first, act, na * sizeof(float));
        fwrite(act, sizeof(float), na, fo);
        all_same &= same;
        printf("VLA_SAMPLE: {\"i\": %d, \"cameras\": %d, \"ids\": %d, \"image_ms\": [", i, s[i].cams, s[i].n_ids);
        for (c = 0; c < s[i].cams; c++) {
            printf("%s%.1f", c ? ", " : "", img[c]);
            tot_img += img[c];
        }
        printf("], \"prefix_ms\": %.1f, \"expert_ms\": %.1f, \"act_ms\": %.1f, \"best_act_ms\": %.1f, "
               "\"reps_identical\": %s}\n", st.prefix_ms, st.expert_ms, st.act_ms, best_act, same ? "true" : "false");
        tot_act += st.act_ms;
        fflush(stdout);
    }
    fclose(fo);
    if (reopen) {
        int same;
        vla_close();
        if (vla_open(wd) != 0) {
            fprintf(stderr, "vla_open (again): %s\n", vla_last_error());
            return 1;
        }
        if (run(&s[0], act2, NULL))
            return 1;
        same = memcmp(first, act2, na * sizeof(float)) == 0;
        all_same &= same;
        printf("VLA_REOPEN: {\"identical\": %s}\n", same ? "true" : "false");
    }
    printf("VLA_SUMMARY: {\"samples\": %d, \"image_ms_mean\": %.1f, \"act_ms_mean\": %.1f, \"identical\": %s}\n",
           n, tot_img / n, tot_act / n, all_same ? "true" : "false");
    vla_close();
    return all_same ? 0 : 3;
}

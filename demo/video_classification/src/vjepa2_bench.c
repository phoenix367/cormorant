// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * vjepa2_bench.c — the gate runner of libvjepa2 (demo/video_classification/
 * scripts/vjepa2_board.py, vjepa2_host_emu.py): decoded videos in, logits out,
 * timing.
 *
 *   vjepa2_bench -i videos.bin -o logits.bin [-R reps] [-r] [-w weights_dir]
 *
 * videos.bin: int32 count, then per video int32 n, int32 h, int32 w and
 * n * h * w * 3 uint8 (the decoded frames, RGB, row-major) — read one video at
 * a time.  logits.bin: per video the logits [vjepa2_classes()] (float32).
 * Every video runs ``reps`` times (the logits must be identical); -r closes
 * and reopens the library after the videos and runs the first one again
 * (identical).  Result lines on stdout (JSON after the tag): VJ_OPEN, VJ_VIDEO
 * (one per video), VJ_REOPEN, VJ_SUMMARY.  Built with -DINFERENCE_PROFILING=ON,
 * the first video once more under the per-layer profiler ("PROFILE_PHASE: clip",
 * then the profiler's "LAYERS_JSON: {...}").
 */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "vjepa2_api.h"
#if INFERENCE_PROFILING
#  include "inference.h"
#  include "inference_prof.h"
#endif

typedef struct {
    int      n, h, w;
    uint8_t *rgb;
} video_t;

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

/* The next video of the file (its frames replace the previous one's). */
static int next_video(FILE *f, video_t *v)
{
    int32_t n, h, w;
    size_t  sz;
    if (rd(f, &n, 4) || rd(f, &h, 4) || rd(f, &w, 4) || n < 1 || n > 100000 || h < 1 || w < 1 || h > 8192
        || w > 8192)
        return -1;
    sz = (size_t)n * (size_t)h * (size_t)w * 3u;
    free(v->rgb);
    v->rgb = (uint8_t *)malloc(sz);
    if (!v->rgb || rd(f, v->rgb, sz))
        return -1;
    v->n = n;
    v->h = h;
    v->w = w;
    return 0;
}

static int run(const video_t *v, float *logits)
{
    int rc = vjepa2_classify(v->rgb, v->n, v->h, v->w, logits);
    if (rc)
        fprintf(stderr, "vjepa2_classify: %d %s\n", rc, vjepa2_last_error());
    return rc;
}

static int argmax(const float *x, int n)
{
    int i, b = 0;
    for (i = 1; i < n; i++)
        if (x[i] > x[b]) b = i;
    return b;
}

int main(int argc, char **argv)
{
    const char *in = NULL, *outp = NULL, *wd = NULL;
    int         reps = 1, reopen = 0, opt, count, i, r, all_same = 1, C;
    FILE       *fi, *fo;
    float      *lg, *lg2, *first;
    video_t     v = {0}, v0 = {0};
    double      t0, tot_prep = 0.0, tot_run = 0.0;
    while ((opt = getopt(argc, argv, "i:o:R:rw:")) != -1) {
        switch (opt) {
        case 'i': in = optarg; break;
        case 'o': outp = optarg; break;
        case 'R': reps = atoi(optarg); break;
        case 'r': reopen = 1; break;
        case 'w': wd = optarg; break;
        default:
            fprintf(stderr, "usage: %s -i videos.bin -o logits.bin [-R reps] [-r] [-w weights_dir]\n", argv[0]);
            return 2;
        }
    }
    if (!in || !outp || reps < 1) {
        fprintf(stderr, "vjepa2_bench: -i and -o are needed\n");
        return 2;
    }
    t0 = now_ms();
    if (vjepa2_open(wd) != 0) {
        fprintf(stderr, "vjepa2_open: %s\n", vjepa2_last_error());
        return 1;
    }
    C = vjepa2_classes();
    printf("VJ_OPEN: {\"ms\": %.1f, \"frames\": %d, \"crop\": %d, \"classes\": %d}\n", now_ms() - t0,
           vjepa2_frames(), vjepa2_crop(), C);
    fi = fopen(in, "rb");
    if (!fi || rd(fi, &count, 4) || count < 1 || count > 100000) {
        fprintf(stderr, "vjepa2_bench: cannot read %s\n", in);
        return 1;
    }
#if INFERENCE_PROFILING
    if (inference_prof_init(inference_num_layers(), inference_layer_names_ptr()) != 0)
        fprintf(stderr, "warning: inference_prof_init failed\n");
#endif
    lg = (float *)malloc((size_t)C * sizeof(float));
    lg2 = (float *)malloc((size_t)C * sizeof(float));
    first = (float *)malloc((size_t)C * sizeof(float));
    fo = fopen(outp, "wb");
    if (!fo || !lg || !lg2 || !first) {
        fprintf(stderr, "vjepa2_bench: cannot write %s\n", outp);
        return 1;
    }
    for (i = 0; i < count; i++) {
        vjepa2_stats_t st;
        double         best = 1e300;
        int            same = 1;
        if (next_video(fi, &v)) {
            fprintf(stderr, "vjepa2_bench: %s: video %d is malformed\n", in, i);
            return 1;
        }
        for (r = 0; r < reps; r++) {
            if (run(&v, r ? lg2 : lg))
                return 1;
            vjepa2_last_stats(&st);
            if (st.total_ms < best) best = st.total_ms;
            if (r && memcmp(lg, lg2, (size_t)C * sizeof(float)) != 0) same = 0;
        }
        if (i == 0) {
            memcpy(first, lg, (size_t)C * sizeof(float));
            v0 = v;                                              /* kept for the profile / the reopen */
            v.rgb = NULL;
        }
        fwrite(lg, sizeof(float), (size_t)C, fo);
        all_same &= same;
        tot_prep += st.prep_ms;
        tot_run += st.run_ms;
        printf("VJ_VIDEO: {\"i\": %d, \"frames\": %d, \"h\": %d, \"w\": %d, \"top1\": %d, \"prep_ms\": %.1f, "
               "\"run_ms\": %.1f, \"total_ms\": %.1f, \"best_ms\": %.1f, \"reps_identical\": %s}\n", i,
               i ? v.n : v0.n, i ? v.h : v0.h, i ? v.w : v0.w, argmax(lg, C), st.prep_ms, st.run_ms, st.total_ms,
               best, same ? "true" : "false");
        fflush(stdout);
    }
    fclose(fi);
    fclose(fo);
#if INFERENCE_PROFILING
    inference_prof_reset();
    run(&v0, lg2);
    printf("PROFILE_PHASE: clip\n");
    inference_prof_dump_json(stdout);
    fflush(stdout);
#endif
    if (reopen) {
        int same;
        vjepa2_close();
        if (vjepa2_open(wd) != 0) {
            fprintf(stderr, "vjepa2_open (again): %s\n", vjepa2_last_error());
            return 1;
        }
        if (run(&v0, lg2))
            return 1;
        same = memcmp(first, lg2, (size_t)C * sizeof(float)) == 0;
        all_same &= same;
        printf("VJ_REOPEN: {\"identical\": %s}\n", same ? "true" : "false");
    }
    printf("VJ_SUMMARY: {\"videos\": %d, \"prep_ms_mean\": %.1f, \"run_ms_mean\": %.1f, \"identical\": %s}\n",
           count, tot_prep / count, tot_run / count, all_same ? "true" : "false");
    vjepa2_close();
    free(v.rgb);
    free(v0.rgb);
    return all_same ? 0 : 3;
}

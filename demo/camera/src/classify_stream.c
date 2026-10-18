// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * classify_stream.c — persistent KV260 inference host for the camera demo.
 *
 * Unlike classify_images.c (a one-shot batch over images.bin), this program
 * initialises the FPGA inference pipeline ONCE and then services an
 * open-ended stream of camera frames:
 *
 *   stdin   raw preprocessed frames — exactly BENCH_INPUT_NUMEL int16
 *           (ap_fixed<16,8>) elements in NCHW order, back to back, with no
 *           length prefix (every frame is the same fixed size).
 *   stdout  the JSON frame protocol — one line per frame:
 *             {"frame":N,"latency_ms":F,"top":[{"class_id":I,"prob":F,
 *                                               "logit":I}, ...]}
 *           preceded once, after init completes, by a handshake line:
 *             {"status":"ready","input_numel":N,"classes":C,"top_k":K}
 *   stderr  ALL human-readable diagnostics.
 *
 * The board-side camera_loop.py drives this as a subprocess: it writes one
 * frame to stdin and reads back one result line.  The blocking write/read
 * pair keeps exactly one frame in flight.  Closing stdin (EOF) is the clean
 * shutdown signal.
 *
 * IMPORTANT: stdout carries the line-delimited JSON protocol and nothing
 * else.  Every diagnostic / log message MUST go to stderr, or the driving
 * Python loop will fail to parse a result line.
 *
 * Host glue
 *   scripts/generate_project.py emits a model-specific bench_glue.h that
 *   supplies BENCH_INPUT_NUMEL / BENCH_OUTPUT_NUMEL / BENCH_NUM_CLASSES /
 *   BENCH_MODEL_NAME and the bench_inference_init() / bench_inference_run()
 *   shims (this is the same glue header the image_classification demo uses).
 *
 * Runtime arguments:
 *   ./classify_stream [warmup] [top_k]
 *     warmup  inferences on a zero frame before the ready handshake (default 1)
 *     top_k   predictions emitted per frame                (default BENCH_TOP_K)
 */

#define _POSIX_C_SOURCE 200809L   /* clock_gettime */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference.h"
#include "bench_glue.h"

#ifndef BENCH_TOP_K
#  define BENCH_TOP_K 5u
#endif
#ifndef BENCH_DEFAULT_WARMUP
#  define BENCH_DEFAULT_WARMUP 1u
#endif

/* ──────────────────────────────────────────────────────────────────────── */
/* Top-K selection                                                           */
/* ──────────────────────────────────────────────────────────────────────── */

typedef struct {
    float    prob;      /* softmax probability in [0, 1] */
    int      logit;     /* signed int16 ap_fixed<16,8> bits (raw, for debug) */
    unsigned idx;
} prediction_t;

static int cmp_prediction_desc(const void *a, const void *b) {
    float pa = ((const prediction_t *)a)->prob;
    float pb = ((const prediction_t *)b)->prob;
    if (pa < pb) return  1;
    if (pa > pb) return -1;
    unsigned ia = ((const prediction_t *)a)->idx;
    unsigned ib = ((const prediction_t *)b)->idx;
    return (ia > ib) - (ia < ib);
}

/*
 * Compute softmax over the model's int16 ap_fixed<16,8> logits and return the
 * top-K predictions sorted by probability descending.  The deployed graph has
 * its Softmax tail stripped, so the host applies the normalisation here.
 * Numerically stable: subtract the max logit before exp.
 */
static void top_k(const Data_t *logits, unsigned n_classes, unsigned k,
                  prediction_t *out) {
    prediction_t *all = (prediction_t *)malloc(n_classes * sizeof(*all));
    if (!all) { for (unsigned i = 0; i < k; ++i) out[i] = (prediction_t){0}; return; }

    float max_logit = -INFINITY;
    for (unsigned i = 0; i < n_classes; ++i) {
        int   raw = (int)(int16_t)logits[i];
        float v   = (float)raw / 256.0f;          /* ap_fixed<16,8> → float */
        all[i].prob  = v;                         /* stash logit value temp */
        all[i].logit = raw;
        all[i].idx   = i;
        if (v > max_logit) max_logit = v;
    }
    double sum = 0.0;
    for (unsigned i = 0; i < n_classes; ++i) {
        float e = expf(all[i].prob - max_logit);
        all[i].prob = e;
        sum += (double)e;
    }
    if (sum > 0.0) {
        float inv = (float)(1.0 / sum);
        for (unsigned i = 0; i < n_classes; ++i) all[i].prob *= inv;
    }
    qsort(all, n_classes, sizeof(*all), cmp_prediction_desc);
    for (unsigned i = 0; i < k; ++i) out[i] = all[i];
    free(all);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/*
 * Read exactly *n* bytes from *f* into *buf*, looping over short reads (the
 * driving Python writes one frame per write()/flush(), but a pipe may still
 * hand it over in fragments).  Returns n on success, 0 on a clean EOF at a
 * frame boundary (got nothing), -1 on a short read mid-frame or a stream error.
 */
static long read_full(FILE *f, void *buf, size_t n) {
    size_t got = 0;
    unsigned char *p = (unsigned char *)buf;
    while (got < n) {
        size_t r = fread(p + got, 1, n - got, f);
        if (r == 0) {
            if (feof(f))   return (got == 0) ? 0 : -1;
            if (ferror(f)) return -1;
        }
        got += r;
    }
    return (long)n;
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Main                                                                      */
/* ──────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    unsigned warmup = BENCH_DEFAULT_WARMUP;
    unsigned topk   = BENCH_TOP_K;
    if (argc > 1) warmup = (unsigned)strtoul(argv[1], NULL, 10);
    if (argc > 2) topk   = (unsigned)strtoul(argv[2], NULL, 10);
    if (topk == 0u || topk > BENCH_NUM_CLASSES) topk = BENCH_TOP_K;

    const size_t frame_bytes = (size_t)BENCH_INPUT_NUMEL * sizeof(Data_t);

    fprintf(stderr,
            "classify_stream: model=%s classes=%u input_numel=%u (%zu B) "
            "warmup=%u top_k=%u\n",
            BENCH_MODEL_NAME, (unsigned)BENCH_NUM_CLASSES,
            (unsigned)BENCH_INPUT_NUMEL, frame_bytes, warmup, topk);

    if (bench_inference_init() != 0) {
        fprintf(stderr, "error: inference_init failed\n");
        printf("{\"status\":\"error\",\"reason\":\"inference_init failed\"}\n");
        fflush(stdout);
        return 1;
    }

    inference_buf_t *in_buf  = inference_buf_alloc(BENCH_INPUT_NUMEL);
    inference_buf_t *out_buf = inference_buf_alloc(BENCH_OUTPUT_NUMEL);
    if (!in_buf || !out_buf) {
        fprintf(stderr, "error: inference_buf_alloc failed\n");
        printf("{\"status\":\"error\",\"reason\":\"buf_alloc failed\"}\n");
        fflush(stdout);
        return 1;
    }
    Data_t *in_ptr  = (Data_t *)inference_buf_ptr(in_buf);
    Data_t *out_ptr = (Data_t *)inference_buf_ptr(out_buf);

    /* Warm the driver + DDR caches on a zero frame so the first real frame's
     * latency is representative. */
    if (warmup > 0u) {
        memset(in_ptr, 0, frame_bytes);
        for (unsigned w = 0; w < warmup; ++w)
            bench_inference_run(in_buf, out_buf);
    }

    prediction_t *preds = (prediction_t *)malloc(sizeof(*preds) * topk);
    if (!preds) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }

    /* Handshake: tells camera_loop.py init succeeded and how big a frame is. */
    printf("{\"status\":\"ready\",\"input_numel\":%u,\"classes\":%u,"
           "\"top_k\":%u}\n",
           (unsigned)BENCH_INPUT_NUMEL, (unsigned)BENCH_NUM_CLASSES, topk);
    fflush(stdout);
    fprintf(stderr, "classify_stream: ready — streaming frames\n");

    unsigned long frame = 0;
    for (;;) {
        long r = read_full(stdin, in_ptr, frame_bytes);
        if (r == 0) break;                  /* clean EOF — shutdown */
        if (r < 0) {
            fprintf(stderr,
                    "error: short read on frame %lu — aborting\n", frame);
            break;
        }

        double t0 = now_ms();
        bench_inference_run(in_buf, out_buf);
        double dt = now_ms() - t0;

        top_k(out_ptr, BENCH_NUM_CLASSES, topk, preds);

        printf("{\"frame\":%lu,\"latency_ms\":%.3f,\"top\":[", frame, dt);
        for (unsigned k = 0; k < topk; ++k) {
            if (k) putchar(',');
            printf("{\"class_id\":%u,\"prob\":%.6f,\"logit\":%d}",
                   preds[k].idx, preds[k].prob, preds[k].logit);
        }
        printf("]}\n");
        fflush(stdout);
        frame++;
    }

    fprintf(stderr, "classify_stream: %lu frame(s) processed — shutting down\n",
            frame);
    free(preds);
    inference_buf_free(in_buf);
    inference_buf_free(out_buf);
    inference_deinit();
    return 0;
}

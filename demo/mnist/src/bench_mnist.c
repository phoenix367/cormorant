// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * bench_mnist.c — MNIST throughput / accuracy benchmark for KV260.
 *
 * Reads the MNIST test split (IDX-3 images + IDX-1 labels), feeds each
 * image into the FPGA inference pipeline as ap_fixed<16,8>, takes the
 * argmax of the output logits, and reports:
 *   - mean / p50 / p99 latency per image (ms)
 *   - throughput (images/s)
 *   - top-1 accuracy (%)
 *
 * Output is a single line of JSON suitable for parsing by the host script.
 *
 * Host glue
 *   The model-specific input/output buffer sizes and the inference_init()
 *   signature differ per model.  scripts/generate_project.py emits a small
 *   bench_glue.h that supplies:
 *     - BENCH_INPUT_NUMEL   / BENCH_OUTPUT_NUMEL
 *     - BENCH_NUM_CLASSES
 *     - bench_inference_init()  (forwards to inference_init with kernel UIO names)
 *     - bench_inference_run()   (forwards to inference_run)
 *
 * Compile-time options (override via -D):
 *   BENCH_DATA_DIR     directory containing the IDX files     (default ".")
 *   BENCH_INPUT_BIAS   integer added to each pixel byte before encoding
 *                      (default 0; set to e.g. -128 if model expects centred input)
 *
 * Runtime arguments:
 *   ./bench_mnist [iters] [warmup]
 *     iters   number of test images to evaluate (default: all = 10000, capped to dataset size)
 *     warmup  inferences run before timing starts             (default: 50)
 */

#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference.h"
#include "inference_prof.h"
#include "inference_ddr.h"
#include "bench_glue.h"

#ifndef BENCH_DATA_DIR
#  define BENCH_DATA_DIR "."
#endif
#ifndef BENCH_INPUT_BIAS
#  define BENCH_INPUT_BIAS  0
#endif

#define BENCH_IMAGE_BYTES   (28u * 28u)
#define BENCH_DEFAULT_WARMUP  50u

/*
 * Roughly how many progress updates to emit during the run.  20 → about one
 * line every 5 % of the iterations, which is enough resolution to estimate
 * remaining time without flooding the host's terminal.  Override with
 * -DBENCH_PROGRESS_BUCKETS=N at compile time, or set the BENCH_PROGRESS env
 * var to a non-zero integer (lines per run) at runtime.  0 disables.
 */
#ifndef BENCH_PROGRESS_BUCKETS
#  define BENCH_PROGRESS_BUCKETS  20u
#endif

/* ──────────────────────────────────────────────────────────────────────── */
/* Endianness helpers — IDX files are big-endian.                          */
/* ──────────────────────────────────────────────────────────────────────── */

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

/* ──────────────────────────────────────────────────────────────────────── */
/* IDX loaders                                                              */
/* ──────────────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t  count;
    uint32_t  rows;
    uint32_t  cols;
    uint8_t  *pixels;   /* count * rows * cols bytes */
} idx_images_t;

typedef struct {
    uint32_t  count;
    uint8_t  *labels;
} idx_labels_t;

static int load_idx_images(const char *path, idx_images_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t hdr[16];
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fprintf(stderr, "error: short read on %s header\n", path);
        fclose(f); return -1;
    }
    if (be32(hdr) != 0x00000803u) {
        fprintf(stderr, "error: %s: bad IDX-3 magic\n", path);
        fclose(f); return -1;
    }
    out->count = be32(hdr + 4);
    out->rows  = be32(hdr + 8);
    out->cols  = be32(hdr + 12);
    if (out->rows != 28u || out->cols != 28u) {
        fprintf(stderr, "error: %s: expected 28x28, got %ux%u\n",
                path, out->rows, out->cols);
        fclose(f); return -1;
    }
    size_t total = (size_t)out->count * out->rows * out->cols;
    out->pixels  = (uint8_t *)malloc(total);
    if (!out->pixels) { fclose(f); return -1; }
    if (fread(out->pixels, 1, total, f) != total) {
        fprintf(stderr, "error: short read on %s data\n", path);
        free(out->pixels); fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

static int load_idx_labels(const char *path, idx_labels_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t hdr[8];
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fprintf(stderr, "error: short read on %s header\n", path);
        fclose(f); return -1;
    }
    if (be32(hdr) != 0x00000801u) {
        fprintf(stderr, "error: %s: bad IDX-1 magic\n", path);
        fclose(f); return -1;
    }
    out->count  = be32(hdr + 4);
    out->labels = (uint8_t *)malloc(out->count);
    if (!out->labels) { fclose(f); return -1; }
    if (fread(out->labels, 1, out->count, f) != out->count) {
        fprintf(stderr, "error: short read on %s data\n", path);
        free(out->labels); fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Encoders / decoders                                                      */
/* ──────────────────────────────────────────────────────────────────────── */

/*
 * Encode a 28x28 uint8 image into an ap_fixed<16,8> buffer.
 *
 * Storage layout: bits = round(value * 256).  Mapping pixel byte p directly
 * to bits gives value = p / 256 ∈ [0, 0.996], which is close enough to the
 * [0, 1] range commonly used during training that the fixed-point quantised
 * networks classify correctly.  An optional BENCH_INPUT_BIAS shifts pixels
 * before the encode step (set to -128 for models trained on centred input).
 *
 * Padding bytes between BENCH_IMAGE_BYTES and BENCH_INPUT_NUMEL (broadcast
 * alignment slack) are zeroed.
 */
static void encode_image(const uint8_t *pixels, Data_t *dst) {
    for (size_t i = 0; i < BENCH_IMAGE_BYTES; ++i) {
        int v = (int)pixels[i] + BENCH_INPUT_BIAS;
        if (v < -32768) v = -32768;
        if (v >  32767) v =  32767;
        dst[i] = (uint16_t)(int16_t)v;
    }
    for (size_t i = BENCH_IMAGE_BYTES; i < BENCH_INPUT_NUMEL; ++i) {
        dst[i] = 0u;
    }
}

/*
 * Find the argmax over the first BENCH_NUM_CLASSES elements of the output.
 * Output values are int16_t bits of ap_fixed<16,8>; argmax is invariant under
 * the * 1/256 conversion, so we compare bits directly.
 */
static unsigned argmax_class(const Data_t *out) {
    int16_t best_v = (int16_t)out[0];
    unsigned best_i = 0;
    for (unsigned i = 1; i < BENCH_NUM_CLASSES; ++i) {
        int16_t v = (int16_t)out[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    return best_i;
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Latency stats                                                            */
/* ──────────────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Main                                                                     */
/* ──────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    unsigned want_iters = 0u;                 /* 0 = all */
    unsigned warmup     = BENCH_DEFAULT_WARMUP;
    if (argc > 1) want_iters = (unsigned)strtoul(argv[1], NULL, 10);
    if (argc > 2) warmup     = (unsigned)strtoul(argv[2], NULL, 10);

    const char *data_dir = BENCH_DATA_DIR;
    const char *env_dir  = getenv("BENCH_DATA_DIR");
    if (env_dir && *env_dir) data_dir = env_dir;

    char img_path[512], lbl_path[512];
    snprintf(img_path, sizeof(img_path), "%s/t10k-images-idx3-ubyte", data_dir);
    snprintf(lbl_path, sizeof(lbl_path), "%s/t10k-labels-idx1-ubyte", data_dir);

    idx_images_t imgs = {0};
    idx_labels_t lbls = {0};
    if (load_idx_images(img_path, &imgs) != 0) return 1;
    if (load_idx_labels(lbl_path, &lbls) != 0) return 1;
    if (imgs.count != lbls.count) {
        fprintf(stderr, "error: image/label count mismatch %u vs %u\n",
                imgs.count, lbls.count);
        return 1;
    }

    unsigned n_total = imgs.count;
    if (want_iters == 0u || want_iters > n_total) want_iters = n_total;
    if (warmup > want_iters) warmup = want_iters;

    fprintf(stderr,
            "bench_mnist: dataset=%u images, iters=%u, warmup=%u\n"
            "             input_numel=%u, output_numel=%u, classes=%u\n",
            n_total, want_iters, warmup,
            (unsigned)BENCH_INPUT_NUMEL, (unsigned)BENCH_OUTPUT_NUMEL,
            (unsigned)BENCH_NUM_CLASSES);

    if (bench_inference_init() != 0) {
        fprintf(stderr, "error: inference_init failed\n");
        return 1;
    }

#if INFERENCE_PROFILING
    /*
     * Per-layer profiler + whole-run DDR PMU counters — both opt-in at
     * compile time (cmake -DINFERENCE_PROFILING=ON).  DDR probing emits
     * its own warning if the host has no PMU; we just continue.
     */
    if (inference_prof_init(inference_num_layers(),
                            inference_layer_names_ptr()) != 0) {
        fprintf(stderr, "warning: inference_prof_init failed; "
                        "continuing without per-layer stats\n");
    } else {
        fprintf(stderr,
                "bench_mnist: per-layer profiling ENABLED (%u layers)\n",
                inference_num_layers());
    }
    (void)inference_ddr_init();   /* warns + degrades to "available:false" */
#endif

    inference_buf_t *in_buf  = inference_buf_alloc(BENCH_INPUT_NUMEL);
    inference_buf_t *out_buf = inference_buf_alloc(BENCH_OUTPUT_NUMEL);
    if (!in_buf || !out_buf) {
        fprintf(stderr, "error: inference_buf_alloc failed\n");
        return 1;
    }

    Data_t   *in_ptr  = (Data_t *)inference_buf_ptr(in_buf);
    Data_t   *out_ptr = (Data_t *)inference_buf_ptr(out_buf);

    double *latencies = (double *)malloc(sizeof(double) *
                                         (size_t)(want_iters - warmup));
    if (!latencies) {
        fprintf(stderr, "error: out of memory for latency array\n");
        return 1;
    }

    unsigned correct  = 0u;
    unsigned timed_n  = 0u;
    double   t_total  = 0.0;

    /* Decide progress cadence: prefer env var if set, else BENCH_PROGRESS_BUCKETS. */
    unsigned buckets = BENCH_PROGRESS_BUCKETS;
    const char *env_p = getenv("BENCH_PROGRESS");
    if (env_p && *env_p) buckets = (unsigned)strtoul(env_p, NULL, 10);
    unsigned progress_every = (buckets > 0u && want_iters > buckets)
                              ? (want_iters / buckets) : 0u;

    double t_start = now_ms();
    double t_timed_start = 0.0;

#if INFERENCE_PROFILING
    /* DDR counters span the same iterations as inference_prof — every
     * kernel call, warmup included.  Latency stats below still drop the
     * warmup window. */
    inference_ddr_start();
#endif

    for (unsigned i = 0; i < want_iters; ++i) {
        const uint8_t *pix = imgs.pixels + (size_t)i * BENCH_IMAGE_BYTES;
        encode_image(pix, in_ptr);

        if (i == warmup) t_timed_start = now_ms();

        double t0 = now_ms();
        bench_inference_run(in_buf, out_buf);
        double dt = now_ms() - t0;

        unsigned pred = argmax_class(out_ptr);
        if (pred == lbls.labels[i]) correct++;

#if INFERENCE_PROFILING
        /* Sample the DDR counters once per inference so backends with
         * narrow native counters (e.g. the 32-bit Xilinx APM byte
         * counters) can fold deltas into a 64-bit total before the
         * hardware wraps.  No-op if the active backend doesn't need
         * sampling, or if init failed. */
        inference_ddr_sample();
#endif

        /* Per-layer counters cover EVERY kernel call (warmup included) so
         * each layer's `calls` field equals want_iters — i.e. one bump per
         * input sample.  Per-image latency stats below still exclude the
         * warmup window so mean/p50/p99 reflect steady-state behaviour. */

        if (i >= warmup) {
            latencies[timed_n++] = dt;
            t_total += dt;
        }

        /* Periodic progress: print to stderr; the host parses these lines. */
        unsigned done = i + 1u;
        int is_last   = (done == want_iters);
        if (progress_every > 0u && (done % progress_every == 0u || is_last)) {
            double now      = now_ms();
            double pct      = 100.0 * (double)done / (double)want_iters;
            double cur_acc  = 100.0 * (double)correct / (double)done;
            double cur_mean = (timed_n > 0u) ? (t_total / (double)timed_n) : 0.0;
            double rate     = (now > t_start)
                              ? (1000.0 * (double)done / (now - t_start)) : 0.0;
            double eta_s    = (rate > 0.0)
                              ? ((double)(want_iters - done) / rate) : 0.0;
            fprintf(stderr,
                    "progress: %u/%u (%.1f%%) acc=%.2f%% mean=%.3fms "
                    "rate=%.1fips eta=%.1fs\n",
                    done, want_iters, pct, cur_acc, cur_mean, rate, eta_s);
            fflush(stderr);
        }
    }
    (void)t_timed_start;

#if INFERENCE_PROFILING
    inference_ddr_stop();
#endif

    qsort(latencies, timed_n, sizeof(double), cmp_double);
    double mean_ms = (timed_n > 0u) ? (t_total / (double)timed_n) : 0.0;
    double p50_ms  = (timed_n > 0u) ? latencies[timed_n / 2u]            : 0.0;
    double p99_idx = (timed_n > 0u) ? (double)(timed_n - 1u) * 0.99      : 0.0;
    double p99_ms  = (timed_n > 0u) ? latencies[(unsigned)p99_idx]       : 0.0;
    double tput    = (mean_ms > 0.0) ? (1000.0 / mean_ms) : 0.0;
    double acc     = (want_iters > 0u)
                     ? (100.0 * (double)correct / (double)want_iters) : 0.0;

    /* Single-line JSON for the host parser. */
    printf(
        "{\"model\":\"%s\",\"iters\":%u,\"warmup\":%u,\"timed\":%u,"
        "\"correct\":%u,\"accuracy_pct\":%.4f,"
        "\"mean_ms\":%.4f,\"p50_ms\":%.4f,\"p99_ms\":%.4f,"
        "\"throughput_ips\":%.2f}\n",
        BENCH_MODEL_NAME, want_iters, warmup, timed_n,
        correct, acc, mean_ms, p50_ms, p99_ms, tput);

#if INFERENCE_PROFILING
    /* Two extra JSON lines, each prefixed with its own marker — the host
     * parser keys off the marker to associate the stats with the model
     * summary above.  DDR_JSON is emitted unconditionally; when no PMU
     * was found it carries {"available":false,"reason":...} so the host
     * still records why the counters are missing. */
    inference_prof_dump_json(stdout);
    inference_ddr_dump_json (stdout);
    inference_prof_deinit();
    inference_ddr_deinit();
#endif

    free(latencies);
    free(imgs.pixels);
    free(lbls.labels);
    inference_deinit();
    return 0;
}

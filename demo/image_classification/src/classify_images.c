// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * classify_images.c — KV260 image-classification host for the
 *                     image_classification demo.
 *
 * Reads a flat preprocessed blob (NCHW int16 ap_fixed<16,8> per image),
 * a tab-separated manifest of <name>\t<byte_offset> lines, and a 1001-line
 * ImageNet label list.  Runs inference for every image and prints:
 *   - top-K predictions with class names and the raw int16 logit value
 *   - per-image latency
 *   - a single-line summary JSON (parsed by deploy_and_run.py)
 *
 * Host glue
 *   The model-specific buffer sizes and the inference_init signature
 *   differ per model; scripts/generate_project.py emits a small
 *   bench_glue.h that supplies BENCH_INPUT_NUMEL / BENCH_OUTPUT_NUMEL,
 *   BENCH_NUM_CLASSES (1001 for MobileNetV1), BENCH_MODEL_NAME, and
 *   bench_inference_init() / bench_inference_run() shims.
 *
 * Compile-time options (override via -D):
 *   BENCH_DATA_DIR     directory containing images.bin/manifest.txt/labels.txt
 *                      (default ".")
 *   BENCH_TOP_K        default top-K (override at runtime via the third arg)
 *   BENCH_LABEL_OFFSET added to each class index before indexing into the
 *                      shared 1001-line labels file (default 0).  Use 1 for
 *                      models whose output emits the 1000 ImageNet classes
 *                      without the synthetic "background" slot — i.e. the
 *                      ONNX Model Zoo MobileNetV2 / ResNet etc.
 *
 * Runtime arguments:
 *   ./classify_images [warmup] [top_k]
 *     warmup  inferences run before timing starts        (default: 1)
 *     top_k   number of top predictions to print         (default: BENCH_TOP_K)
 */

#define _POSIX_C_SOURCE 200809L   /* strdup + clock_gettime */

#include <errno.h>
#include <math.h>
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
#ifndef BENCH_TOP_K
#  define BENCH_TOP_K 5u
#endif
#ifndef BENCH_DEFAULT_WARMUP
#  define BENCH_DEFAULT_WARMUP 1u
#endif
#ifndef BENCH_LABEL_OFFSET
#  define BENCH_LABEL_OFFSET 0u
#endif

/* ──────────────────────────────────────────────────────────────────────── */
/* Tiny readers                                                              */
/* ──────────────────────────────────────────────────────────────────────── */

typedef struct {
    char    *name;       /* malloc'd, NUL-terminated */
    long     offset;     /* byte offset into images.bin */
} manifest_entry_t;

static int read_manifest(const char *path, manifest_entry_t **out_arr,
                          unsigned *out_n) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: open %s: %s\n", path, strerror(errno));
        return -1;
    }
    size_t cap = 16, n = 0;
    manifest_entry_t *arr = (manifest_entry_t *)calloc(cap, sizeof(*arr));
    if (!arr) { fclose(f); return -1; }

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        /* Strip trailing whitespace */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' ||
                           line[len - 1] == ' '  || line[len - 1] == '\t')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        char *tab = strchr(line, '\t');
        if (!tab) {
            fprintf(stderr, "warning: %s: skipping malformed line: %s\n",
                    path, line);
            continue;
        }
        *tab = '\0';
        long off = strtol(tab + 1, NULL, 10);

        if (n == cap) {
            cap *= 2;
            manifest_entry_t *grown =
                (manifest_entry_t *)realloc(arr, cap * sizeof(*arr));
            if (!grown) { free(arr); fclose(f); return -1; }
            arr = grown;
        }
        arr[n].name   = strdup(line);
        arr[n].offset = off;
        if (!arr[n].name) { free(arr); fclose(f); return -1; }
        n++;
    }
    fclose(f);
    *out_arr = arr;
    *out_n   = (unsigned)n;
    return 0;
}

static int read_labels(const char *path, char ***out_arr, unsigned *out_n) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: open %s: %s\n", path, strerror(errno));
        return -1;
    }
    size_t cap = 1024, n = 0;
    char **arr = (char **)calloc(cap, sizeof(*arr));
    if (!arr) { fclose(f); return -1; }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (n == cap) {
            cap *= 2;
            char **grown = (char **)realloc(arr, cap * sizeof(*arr));
            if (!grown) { free(arr); fclose(f); return -1; }
            arr = grown;
        }
        arr[n] = strdup(line);
        if (!arr[n]) { free(arr); fclose(f); return -1; }
        n++;
    }
    fclose(f);
    *out_arr = arr;
    *out_n   = (unsigned)n;
    return 0;
}

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
    /* tie-break by index for stable ordering */
    unsigned ia = ((const prediction_t *)a)->idx;
    unsigned ib = ((const prediction_t *)b)->idx;
    return (ia > ib) - (ia < ib);
}

/*
 * Compute softmax over the model's int16 ap_fixed<16,8> logits and return
 * the top-K predictions sorted by probability descending.
 *
 * The deployed graph has its Softmax tail stripped (the FPGA scheduler
 * doesn't materialise Softmax), so the on-board host applies the
 * normalisation here.  Numerically stable form: subtract max before exp.
 */
static void top_k(const Data_t *logits, unsigned n_classes, unsigned k,
                   prediction_t *out) {
    /* For 1001 classes a full sort is plenty fast (~30 us). */
    prediction_t *all =
        (prediction_t *)malloc(n_classes * sizeof(*all));

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

/* ──────────────────────────────────────────────────────────────────────── */
/* Misc                                                                      */
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
/* JSON escape (just enough for image names + label strings)                 */
/* ──────────────────────────────────────────────────────────────────────── */

static void json_escape(FILE *out, const char *s) {
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        unsigned char c = *p;
        switch (c) {
            case '"':  fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\b': fputs("\\b",  out); break;
            case '\f': fputs("\\f",  out); break;
            case '\n': fputs("\\n",  out); break;
            case '\r': fputs("\\r",  out); break;
            case '\t': fputs("\\t",  out); break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", c);
                else          fputc(c, out);
        }
    }
    fputc('"', out);
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Main                                                                     */
/* ──────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    unsigned warmup = BENCH_DEFAULT_WARMUP;
    unsigned topk   = BENCH_TOP_K;
    if (argc > 1) warmup = (unsigned)strtoul(argv[1], NULL, 10);
    if (argc > 2) topk   = (unsigned)strtoul(argv[2], NULL, 10);
    if (topk == 0u || topk > BENCH_NUM_CLASSES) topk = BENCH_TOP_K;

    const char *data_dir = BENCH_DATA_DIR;
    const char *env_dir  = getenv("BENCH_DATA_DIR");
    if (env_dir && *env_dir) data_dir = env_dir;

    char bin_path  [768];
    char man_path  [768];
    char lbl_path  [768];
    snprintf(bin_path, sizeof(bin_path), "%s/preprocessed/images.bin",  data_dir);
    snprintf(man_path, sizeof(man_path), "%s/preprocessed/manifest.txt", data_dir);
    snprintf(lbl_path, sizeof(lbl_path), "%s/labels/imagenet_1001_labels.txt",
             data_dir);

    manifest_entry_t *images = NULL;
    unsigned n_images = 0;
    if (read_manifest(man_path, &images, &n_images) != 0) return 1;
    if (n_images == 0u) {
        fprintf(stderr, "error: %s has zero images\n", man_path);
        return 1;
    }

    char **labels = NULL;
    unsigned n_labels = 0;
    if (read_labels(lbl_path, &labels, &n_labels) != 0) return 1;
    /* Indices land at [BENCH_LABEL_OFFSET, BENCH_LABEL_OFFSET + NUM_CLASSES) —
     * verify the labels file is wide enough.  For BENCH_LABEL_OFFSET=1 this
     * means the file must have at least NUM_CLASSES + 1 lines. */
    const unsigned labels_required = (unsigned)BENCH_NUM_CLASSES
                                     + (unsigned)BENCH_LABEL_OFFSET;
    if (n_labels < labels_required) {
        fprintf(stderr, "error: %s has %u lines, expected at least %u "
                        "(num_classes=%u + label_offset=%u)\n",
                lbl_path, n_labels, labels_required,
                (unsigned)BENCH_NUM_CLASSES, (unsigned)BENCH_LABEL_OFFSET);
        return 1;
    }

    FILE *fbin = fopen(bin_path, "rb");
    if (!fbin) {
        fprintf(stderr, "error: open %s: %s\n", bin_path, strerror(errno));
        return 1;
    }
    const size_t img_bytes = (size_t)BENCH_INPUT_NUMEL * sizeof(Data_t);

    fprintf(stderr,
            "classify_images: model=%s images=%u classes=%u warmup=%u top_k=%u\n"
            "                 input_numel=%u (%zu B)\n",
            BENCH_MODEL_NAME, n_images, (unsigned)BENCH_NUM_CLASSES,
            warmup, topk, (unsigned)BENCH_INPUT_NUMEL, img_bytes);

    if (bench_inference_init() != 0) {
        fprintf(stderr, "error: inference_init failed\n");
        return 1;
    }

#if INFERENCE_PROFILING
    if (inference_prof_init(inference_num_layers(),
                            inference_layer_names_ptr()) != 0) {
        fprintf(stderr, "warning: inference_prof_init failed; "
                        "continuing without per-layer stats\n");
    } else {
        fprintf(stderr,
                "classify_images: per-layer profiling ENABLED (%u layers)\n",
                inference_num_layers());
    }
    (void)inference_ddr_init();
#endif

    inference_buf_t *in_buf  = inference_buf_alloc(BENCH_INPUT_NUMEL);
    inference_buf_t *out_buf = inference_buf_alloc(BENCH_OUTPUT_NUMEL);
    if (!in_buf || !out_buf) {
        fprintf(stderr, "error: inference_buf_alloc failed\n");
        return 1;
    }
    Data_t *in_ptr  = (Data_t *)inference_buf_ptr(in_buf);
    Data_t *out_ptr = (Data_t *)inference_buf_ptr(out_buf);

    /* Optional warmup: re-runs inference on the first image so the
     * driver and DDR caches are hot before timing begins. */
    if (warmup > 0u) {
        if (fseek(fbin, images[0].offset, SEEK_SET) != 0 ||
            fread(in_ptr, 1, img_bytes, fbin) != img_bytes) {
            fprintf(stderr, "error: warmup read from %s failed\n", bin_path);
            return 1;
        }
        for (unsigned w = 0; w < warmup; ++w) {
            bench_inference_run(in_buf, out_buf);
        }
    }

    double *latencies = (double *)malloc(sizeof(double) * (size_t)n_images);
    prediction_t *preds = (prediction_t *)malloc(sizeof(*preds) * topk);
    if (!latencies || !preds) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }

#if INFERENCE_PROFILING
    inference_ddr_start();
#endif

    /* Build the per-image JSON array as we go.  Buffered to a temp file so
     * the final summary can be a single line at end-of-output. */
    FILE *jbuf = tmpfile();
    if (!jbuf) {
        fprintf(stderr, "error: tmpfile() failed\n");
        return 1;
    }
    fputc('[', jbuf);

    double t_total = 0.0;
    for (unsigned i = 0; i < n_images; ++i) {
        const manifest_entry_t *m = &images[i];
        if (fseek(fbin, m->offset, SEEK_SET) != 0) {
            fprintf(stderr, "error: seek to %ld in %s: %s\n",
                    m->offset, bin_path, strerror(errno));
            return 1;
        }
        if (fread(in_ptr, 1, img_bytes, fbin) != img_bytes) {
            fprintf(stderr, "error: short read on %s for %s\n",
                    bin_path, m->name);
            return 1;
        }

        double t0 = now_ms();
        bench_inference_run(in_buf, out_buf);
        double dt = now_ms() - t0;

#if INFERENCE_PROFILING
        inference_ddr_sample();
#endif

        top_k(out_ptr, BENCH_NUM_CLASSES, topk, preds);
        latencies[i] = dt;
        t_total += dt;

        /* Human-readable per-image report on stderr (host streams it live). */
        fprintf(stderr, "image: %s  latency=%.3f ms\n", m->name, dt);
        for (unsigned k = 0; k < topk; ++k) {
            const unsigned lidx = preds[k].idx + (unsigned)BENCH_LABEL_OFFSET;
            const char *lab = (lidx < n_labels) ? labels[lidx] : "<unknown>";
            fprintf(stderr, "  %u) [%4u] %-32.32s  prob=%6.2f%%  logit=%6d\n",
                    k + 1u, preds[k].idx, lab,
                    100.0f * preds[k].prob, preds[k].logit);
        }
        fflush(stderr);

        /* Per-image JSON record */
        if (i > 0) fputc(',', jbuf);
        fputs("{\"name\":", jbuf); json_escape(jbuf, m->name);
        fprintf(jbuf, ",\"latency_ms\":%.4f,\"top\":[", dt);
        for (unsigned k = 0; k < topk; ++k) {
            if (k > 0) fputc(',', jbuf);
            const unsigned lidx = preds[k].idx + (unsigned)BENCH_LABEL_OFFSET;
            const char *lab = (lidx < n_labels) ? labels[lidx] : "";
            fprintf(jbuf, "{\"class_id\":%u,\"label\":", preds[k].idx);
            json_escape(jbuf, lab);
            fprintf(jbuf, ",\"prob\":%.6f,\"logit\":%d}",
                    preds[k].prob, preds[k].logit);
        }
        fputs("]}", jbuf);
    }
    fputc(']', jbuf);

#if INFERENCE_PROFILING
    inference_ddr_stop();
#endif

    /* Latency stats */
    qsort(latencies, n_images, sizeof(double), cmp_double);
    double mean_ms = (n_images > 0u) ? (t_total / (double)n_images) : 0.0;
    double p50_ms  = (n_images > 0u) ? latencies[n_images / 2u]      : 0.0;
    double p99_idx = (n_images > 0u) ? (double)(n_images - 1u) * 0.99 : 0.0;
    double p99_ms  = (n_images > 0u) ? latencies[(unsigned)p99_idx]   : 0.0;
    double tput    = (mean_ms > 0.0) ? (1000.0 / mean_ms) : 0.0;

    /* Splice the per-image array out of the temp file. */
    long jbuf_len = ftell(jbuf);
    rewind(jbuf);
    char *jbuf_data = (char *)malloc((size_t)jbuf_len + 1);
    if (!jbuf_data) {
        fprintf(stderr, "error: out of memory for json buffer\n");
        return 1;
    }
    if (fread(jbuf_data, 1, (size_t)jbuf_len, jbuf) != (size_t)jbuf_len) {
        fprintf(stderr, "error: tmpfile read failed\n");
        return 1;
    }
    jbuf_data[jbuf_len] = '\0';
    fclose(jbuf);

    /* Single-line JSON summary (parsed by deploy_and_run.py). */
    printf(
        "{\"model\":\"%s\",\"images\":%u,\"warmup\":%u,\"top_k\":%u,"
        "\"mean_ms\":%.4f,\"p50_ms\":%.4f,\"p99_ms\":%.4f,"
        "\"throughput_ips\":%.2f,\"results\":%s}\n",
        BENCH_MODEL_NAME, n_images, warmup, topk,
        mean_ms, p50_ms, p99_ms, tput, jbuf_data);
    free(jbuf_data);

#if INFERENCE_PROFILING
    inference_prof_dump_json(stdout);
    inference_ddr_dump_json (stdout);
    inference_prof_deinit();
    inference_ddr_deinit();
#endif

    free(latencies);
    free(preds);
    for (unsigned i = 0; i < n_images; ++i) free(images[i].name);
    free(images);
    for (unsigned i = 0; i < n_labels; ++i) free(labels[i]);
    free(labels);
    fclose(fbin);
    inference_deinit();
    return 0;
}

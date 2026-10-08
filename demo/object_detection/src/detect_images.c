/*
 * detect_images.c — KV260 host of the object_detection demo (YOLOv5n).
 *
 * Reads a flat preprocessed blob (one letterboxed image per entry: NCHW int16
 * ap_fixed<16,8>, [3][640][640]) and a manifest of "<name>\t<byte_offset>"
 * lines, runs the FPGA graph (YOLOv5n up to its three Detect convs) on every
 * image and writes the three raw head maps (int16, P3 [255][80][80], P4
 * [255][40][40], P5 [255][20][20], in that order) of every image to
 * heads.bin.  The decode (sigmoid, grid, anchors) and NMS run on the host
 * (scripts/postprocess.py): their input is bit-exact with the scheduler's
 * simulation.
 *
 * Prints the per-image latency of inference_run() on stderr and one summary
 * JSON line on stdout (parsed by scripts/deploy_and_run.py).
 *
 * Built with -DINFERENCE_PROFILING=ON (deploy_and_run.py --profile) it also
 * prints the per-layer times (LAYERS_JSON) after the summary.
 *
 * Host glue: scripts/generate_project.py emits test/detect_glue.h with the
 * buffer sizes and the inference_init() / inference_run() shims.
 *
 *   ./detect_images [warmup] [heads.bin]
 *     warmup     inferences on the first image before timing (default 1)
 *     heads.bin  output path (default <BENCH_DATA_DIR>/heads.bin)
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference.h"
#include "inference_prof.h"
#include "detect_glue.h"

#ifndef BENCH_DATA_DIR
#  define BENCH_DATA_DIR "."
#endif

typedef struct {
    char *name;
    long  offset;
} entry_t;

static int read_manifest(const char *path, entry_t **out, unsigned *n_out)
{
    FILE    *f = fopen(path, "r");
    char     line[1024];
    entry_t *arr = NULL;
    unsigned n = 0u, cap = 0u;
    if (!f) {
        fprintf(stderr, "error: open %s: %s\n", path, strerror(errno));
        return -1;
    }
    while (fgets(line, sizeof line, f)) {
        char *tab = strchr(line, '\t');
        if (!tab)
            continue;
        *tab = '\0';
        if (n == cap) {
            cap = cap ? 2u * cap : 64u;
            arr = (entry_t *)realloc(arr, sizeof(*arr) * cap);
            if (!arr) {
                fclose(f);
                return -1;
            }
        }
        arr[n].name = strdup(line);
        arr[n].offset = strtol(tab + 1, NULL, 10);
        n++;
    }
    fclose(f);
    *out = arr;
    *n_out = n;
    return 0;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    const char *dir = getenv("BENCH_DATA_DIR");
    char        bin_path[768], man_path[768], out_path[768];
    unsigned    warmup = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 10) : 1u;
    entry_t    *im = NULL;
    unsigned    n = 0u, i, k;
    if (!dir || !*dir)
        dir = BENCH_DATA_DIR;
    snprintf(bin_path, sizeof bin_path, "%s/images.bin", dir);
    snprintf(man_path, sizeof man_path, "%s/manifest.txt", dir);
    snprintf(out_path, sizeof out_path, "%s", argc > 2 ? argv[2] : "");
    if (!out_path[0])
        snprintf(out_path, sizeof out_path, "%s/heads.bin", dir);
    if (read_manifest(man_path, &im, &n) != 0 || n == 0u) {
        fprintf(stderr, "error: no images in %s\n", man_path);
        return 1;
    }

    FILE *fin = fopen(bin_path, "rb"), *fout = fopen(out_path, "wb");
    if (!fin || !fout) {
        fprintf(stderr, "error: open %s / %s: %s\n", bin_path, out_path, strerror(errno));
        return 1;
    }
    const size_t in_bytes = (size_t)BENCH_INPUT_NUMEL * sizeof(Data_t);
    const unsigned out_numel[BENCH_NUM_OUTPUTS] = BENCH_OUTPUT_NUMELS;

    fprintf(stderr, "detect_images: model=%s images=%u warmup=%u\n", BENCH_MODEL_NAME, n, warmup);
    if (bench_inference_init() != 0) {
        fprintf(stderr, "error: inference_init failed\n");
        return 1;
    }
#if INFERENCE_PROFILING
    if (inference_prof_init(inference_num_layers(), inference_layer_names_ptr()) != 0)
        fprintf(stderr, "warning: inference_prof_init failed; no per-layer times\n");
#endif
    inference_buf_t *in = inference_buf_alloc(BENCH_INPUT_NUMEL);
    inference_buf_t *out[BENCH_NUM_OUTPUTS];
    for (k = 0u; k < BENCH_NUM_OUTPUTS; k++) {
        out[k] = inference_buf_alloc(out_numel[k]);
        if (!out[k]) {
            fprintf(stderr, "error: inference_buf_alloc failed\n");
            return 1;
        }
    }
    if (!in) {
        fprintf(stderr, "error: inference_buf_alloc failed\n");
        return 1;
    }
    Data_t *in_ptr = (Data_t *)inference_buf_ptr(in);

    if (warmup) {
        if (fseek(fin, im[0].offset, SEEK_SET) != 0 || fread(in_ptr, 1, in_bytes, fin) != in_bytes) {
            fprintf(stderr, "error: read %s\n", bin_path);
            return 1;
        }
        for (i = 0u; i < warmup; i++)
            bench_inference_run(in, out);
    }

    double *lat = (double *)malloc(sizeof(double) * n), total = 0.0;
    for (i = 0u; i < n; i++) {
        if (fseek(fin, im[i].offset, SEEK_SET) != 0 || fread(in_ptr, 1, in_bytes, fin) != in_bytes) {
            fprintf(stderr, "error: short read of %s for %s\n", bin_path, im[i].name);
            return 1;
        }
        const double t0 = now_ms();
        bench_inference_run(in, out);
        lat[i] = now_ms() - t0;
        total += lat[i];
        for (k = 0u; k < BENCH_NUM_OUTPUTS; k++) {
            if (fwrite(inference_buf_ptr(out[k]), sizeof(Data_t), out_numel[k], fout) != out_numel[k]) {
                fprintf(stderr, "error: write %s\n", out_path);
                return 1;
            }
        }
        fprintf(stderr, "image: %s  latency=%.3f ms\n", im[i].name, lat[i]);
    }
    fclose(fout);
    fclose(fin);

    double *s = (double *)malloc(sizeof(double) * n);
    memcpy(s, lat, sizeof(double) * n);
    qsort(s, n, sizeof(double), cmp_double);
    printf("{\"model\": \"%s\", \"images\": %u, \"mean_ms\": %.4f, \"p50_ms\": %.4f, \"min_ms\": %.4f, "
           "\"max_ms\": %.4f, \"fps\": %.2f, \"heads\": \"%s\", \"latencies_ms\": [",
           BENCH_MODEL_NAME, n, total / n, s[n / 2], s[0], s[n - 1], 1000.0 * n / total, out_path);
    for (i = 0u; i < n; i++)
        printf("%s%.4f", i ? ", " : "", lat[i]);
    printf("]}\n");
#if INFERENCE_PROFILING
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
#endif
    for (k = 0u; k < BENCH_NUM_OUTPUTS; k++)
        inference_buf_free(out[k]);
    inference_buf_free(in);
    inference_deinit();
    return 0;
}

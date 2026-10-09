/*
 * stereo_pairs.c — KV260 host of the stereo_depth demo (LightStereo-S).
 *
 * Reads a flat blob of prepared pairs (per pair: the left then the right
 * image, NCHW int16 at the input's exponent, [3][H][W] each) and a manifest of
 * "<name>\t<byte_offset>" lines, runs the network on every pair and writes
 * the disparity maps (float32 [H][W], pixels at the network's resolution) one
 * after another to disparity.bin.  Every number on the way is the FPGA
 * datapath's; the board's maps are bit-exact with the scheduler's simulation
 * (scripts/postprocess.py checks it).
 *
 * Prints the per-pair latency of inference_run() on stderr and one summary
 * JSON line on stdout (parsed by scripts/deploy_and_run.py); built with
 * -DINFERENCE_PROFILING=ON (deploy_and_run.py --profile) it also prints the
 * per-layer times (LAYERS_JSON) after the summary.
 *
 * Host glue: scripts/generate_project.py emits test/stereo_glue.h with the
 * buffer sizes and the inference_init() / inference_run() shims.
 *
 *   ./stereo_pairs [warmup] [disparity.bin]
 *     warmup         inferences on the first pair before timing (default 1)
 *     disparity.bin  output path (default <BENCH_DATA_DIR>/disparity.bin)
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
#include "stereo_glue.h"

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
            entry_t *grown;
            cap = cap ? 2u * cap : 64u;
            grown = (entry_t *)realloc(arr, sizeof(*arr) * cap);
            if (!grown) {
                free(arr);
                fclose(f);
                return -1;
            }
            arr = grown;
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

static int read_pair(FILE *f, long off, size_t bytes, Data_t *left, Data_t *right)
{
    return fseek(f, off, SEEK_SET) == 0 && fread(left, 1, bytes, f) == bytes && fread(right, 1, bytes, f) == bytes;
}

int main(int argc, char **argv)
{
    const char *dir = getenv("BENCH_DATA_DIR");
    char        bin_path[768], man_path[768], out_path[768];
    unsigned    warmup = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 10) : 1u;
    entry_t    *pr = NULL;
    unsigned    n = 0u, i;
    if (!dir || !*dir)
        dir = BENCH_DATA_DIR;
    snprintf(bin_path, sizeof bin_path, "%s/pairs.bin", dir);
    snprintf(man_path, sizeof man_path, "%s/manifest.txt", dir);
    snprintf(out_path, sizeof out_path, "%s", argc > 2 ? argv[2] : "");
    if (!out_path[0])
        snprintf(out_path, sizeof out_path, "%s/disparity.bin", dir);
    if (read_manifest(man_path, &pr, &n) != 0 || n == 0u) {
        fprintf(stderr, "error: no pairs in %s\n", man_path);
        return 1;
    }

    FILE *fin = fopen(bin_path, "rb"), *fout = fopen(out_path, "wb");
    if (!fin || !fout) {
        fprintf(stderr, "error: open %s / %s: %s\n", bin_path, out_path, strerror(errno));
        return 1;
    }
    const size_t in_bytes = (size_t)BENCH_IMAGE_NUMEL * sizeof(Data_t);

    fprintf(stderr, "stereo_pairs: model=%s pairs=%u warmup=%u\n", BENCH_MODEL_NAME, n, warmup);
    if (bench_inference_init() != 0) {
        fprintf(stderr, "error: inference_init failed\n");
        return 1;
    }
#if INFERENCE_PROFILING
    if (inference_prof_init(inference_num_layers(), inference_layer_names_ptr()) != 0)
        fprintf(stderr, "warning: inference_prof_init failed; no per-layer times\n");
#endif
    inference_buf_t *left = inference_buf_alloc(BENCH_IMAGE_NUMEL);
    inference_buf_t *right = inference_buf_alloc(BENCH_IMAGE_NUMEL);
    float           *disp = (float *)malloc(sizeof(float) * BENCH_DISP_NUMEL);
    if (!left || !right || !disp) {
        fprintf(stderr, "error: allocation failed\n");
        return 1;
    }
    Data_t *lp = (Data_t *)inference_buf_ptr(left), *rp = (Data_t *)inference_buf_ptr(right);

    if (warmup) {
        if (!read_pair(fin, pr[0].offset, in_bytes, lp, rp)) {
            fprintf(stderr, "error: read %s\n", bin_path);
            return 1;
        }
        for (i = 0u; i < warmup; i++)
            bench_inference_run(left, right, disp);
    }

    double *lat = (double *)malloc(sizeof(double) * n), total = 0.0;
    for (i = 0u; i < n; i++) {
        if (!read_pair(fin, pr[i].offset, in_bytes, lp, rp)) {
            fprintf(stderr, "error: short read of %s for %s\n", bin_path, pr[i].name);
            return 1;
        }
        const double t0 = now_ms();
        bench_inference_run(left, right, disp);
        lat[i] = now_ms() - t0;
        total += lat[i];
        if (fwrite(disp, sizeof(float), BENCH_DISP_NUMEL, fout) != BENCH_DISP_NUMEL) {
            fprintf(stderr, "error: write %s\n", out_path);
            return 1;
        }
        fprintf(stderr, "pair: %s  latency=%.3f ms\n", pr[i].name, lat[i]);
    }
    fclose(fout);
    fclose(fin);

    double *s = (double *)malloc(sizeof(double) * n);
    memcpy(s, lat, sizeof(double) * n);
    qsort(s, n, sizeof(double), cmp_double);
    printf("{\"model\": \"%s\", \"pairs\": %u, \"mean_ms\": %.4f, \"p50_ms\": %.4f, \"min_ms\": %.4f, "
           "\"max_ms\": %.4f, \"fps\": %.3f, \"disparity\": \"%s\", \"latencies_ms\": [",
           BENCH_MODEL_NAME, n, total / n, s[n / 2], s[0], s[n - 1], 1000.0 * n / total, out_path);
    for (i = 0u; i < n; i++)
        printf("%s%.4f", i ? ", " : "", lat[i]);
    printf("]}\n");
#if INFERENCE_PROFILING
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
#endif
    inference_buf_free(left);
    inference_buf_free(right);
    free(disp);
    inference_deinit();
    return 0;
}

/*
 * squad_bench.c — KV260 BERT-SQuAD runner for the bert_squad demo.
 *
 * Reads the tokenized SQuAD features written by scripts/prepare_inputs.py
 * (inputs.bin: int16 per example — input_ids[SEQ], segment_ids[SEQ],
 * input_mask[SEQ]), runs inference_run() once per example and writes the
 * raw logits (logits.bin: int16 ap_fixed<16,8> bits per example —
 * start[SEQ], end[SEQ]).  Span decoding, EM / F1 and the bit-exact check
 * against the scheduler simulation happen on the host
 * (scripts/deploy_and_run.py).
 *
 * The integer inputs are stored raw in Data_t (inference.h: "Integer
 * tensors ... hold the raw signed integer value"); unique_ids = the
 * example index and is checked on the pass-through output.
 *
 * Output: per-example progress on stderr, then one JSON line on stdout
 *   {"model":..,"examples":N,"warmup":W,"seq_len":S,"mean_ms":..,
 *    "min_ms":..,"max_ms":..,"p50_ms":..,"total_s":..,"uid_ok":N,
 *    "latencies_ms":[..]}
 * and, when built with -DINFERENCE_PROFILING=ON, the LAYERS_JSON: and
 * DDR_JSON: lines of the per-layer profiler (warm-up excluded).
 *
 * Model-specific glue (buffer order / sizes / roles, inference_init
 * arguments) comes from the generated bench_glue.h.
 *
 * usage: squad_bench [-i inputs.bin] [-o logits.bin] [-n N] [-w warmup]
 *        -n 0 (default) = every example in inputs.bin
 */

#define _POSIX_C_SOURCE 200809L   /* clock_gettime, getopt */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "inference.h"
#include "inference_prof.h"
#include "inference_ddr.h"
#include "bench_glue.h"

#define REC_ELEMS  (3u * BENCH_SEQ_LEN)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Copy one inputs.bin record into the model's input buffers (raw int16). */
static void fill_inputs(inference_buf_t *const *b, const int16_t *rec, unsigned uid)
{
    const int roles[3] = { BENCH_IDX_INPUT_IDS, BENCH_IDX_SEGMENT_IDS, BENCH_IDX_INPUT_MASK };
    unsigned r, i;
    for (r = 0; r < 3u; r++) {
        Data_t *p = inference_buf_ptr(b[roles[r]]);
        for (i = 0; i < BENCH_SEQ_LEN; i++)
            p[i] = (Data_t)rec[r * BENCH_SEQ_LEN + i];
    }
#if BENCH_IDX_UNIQUE_IDS >= 0
    inference_buf_ptr(b[BENCH_IDX_UNIQUE_IDS])[0] = (Data_t)(int16_t)uid;
#else
    (void)uid;
#endif
}

static unsigned argmax16(const int16_t *v, unsigned n)
{
    unsigned i, best = 0;
    for (i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char **argv)
{
    const char *in_path  = "inputs.bin";
    const char *out_path = "logits.bin";
    unsigned    n_req = 0, warmup = 1;
    int         c;

    while ((c = getopt(argc, argv, "i:o:n:w:")) != -1) {
        switch (c) {
        case 'i': in_path  = optarg; break;
        case 'o': out_path = optarg; break;
        case 'n': n_req  = (unsigned)strtoul(optarg, NULL, 10); break;
        case 'w': warmup = (unsigned)strtoul(optarg, NULL, 10); break;
        default:
            fprintf(stderr, "usage: %s [-i inputs.bin] [-o logits.bin] [-n N] [-w warmup]\n",
                    argv[0]);
            return 2;
        }
    }

    /* ---- inputs ------------------------------------------------------ */
    FILE *fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "error: open %s: %s\n", in_path, strerror(errno));
        return 1;
    }
    fseek(fin, 0, SEEK_END);
    long in_bytes = ftell(fin);
    rewind(fin);
    const size_t rec_bytes = (size_t)REC_ELEMS * sizeof(int16_t);
    if (in_bytes <= 0 || (size_t)in_bytes % rec_bytes != 0) {
        fprintf(stderr, "error: %s: %ld bytes is not a multiple of %zu (3 x %u int16)\n",
                in_path, in_bytes, rec_bytes, BENCH_SEQ_LEN);
        return 1;
    }
    unsigned n_avail = (unsigned)((size_t)in_bytes / rec_bytes);
    unsigned n = (n_req == 0u || n_req > n_avail) ? n_avail : n_req;
    int16_t *recs = (int16_t *)malloc((size_t)n * rec_bytes);
    if (!recs || fread(recs, rec_bytes, n, fin) != n) {
        fprintf(stderr, "error: reading %u records from %s failed\n", n, in_path);
        return 1;
    }
    fclose(fin);

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "error: open %s: %s\n", out_path, strerror(errno));
        return 1;
    }

    fprintf(stderr, "squad_bench: model=%s examples=%u (of %u) seq_len=%u warmup=%u\n",
            BENCH_MODEL_NAME, n, n_avail, BENCH_SEQ_LEN, warmup);

    /* ---- init -------------------------------------------------------- */
    double t_init = now_ms();
    if (bench_inference_init() != 0) {
        fprintf(stderr, "error: inference_init failed\n");
        return 1;
    }
    t_init = now_ms() - t_init;
    fprintf(stderr, "squad_bench: inference_init %.0f ms\n", t_init);

#if INFERENCE_PROFILING
    if (inference_prof_init(inference_num_layers(), inference_layer_names_ptr()) != 0)
        fprintf(stderr, "warning: inference_prof_init failed; no per-layer stats\n");
    else
        fprintf(stderr, "squad_bench: per-layer profiling ENABLED (%u layers)\n",
                inference_num_layers());
    (void)inference_ddr_init();
#endif

    inference_buf_t *bufs[BENCH_N_BUFS];
    unsigned k;
    for (k = 0; k < BENCH_N_BUFS; k++) {
        bufs[k] = inference_buf_alloc(bench_buf_numel[k]);
        if (!bufs[k]) {
            fprintf(stderr, "error: inference_buf_alloc(%u) failed for '%s'\n",
                    bench_buf_numel[k], bench_buf_name[k]);
            return 1;
        }
        memset(inference_buf_ptr(bufs[k]), 0, (size_t)bench_buf_numel[k] * sizeof(Data_t));
    }

    /* ---- warm-up (example 0) ----------------------------------------- */
    unsigned w;
    for (w = 0; w < warmup; w++) {
        double t0 = now_ms();
        fill_inputs(bufs, recs, 0u);
        bench_inference_run(bufs);
        fprintf(stderr, "warmup %u/%u: %.1f ms\n", w + 1u, warmup, now_ms() - t0);
    }
#if INFERENCE_PROFILING
    inference_prof_reset();
    inference_ddr_start();
#endif

    /* ---- timed run --------------------------------------------------- */
    double  *lat = (double *)malloc(sizeof(double) * (n ? n : 1u));
    int16_t *lg  = (int16_t *)malloc(sizeof(int16_t) * 2u * BENCH_SEQ_LEN);
    if (!lat || !lg) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }
    double   t_total = 0.0;
    unsigned uid_ok  = 0, i;
    for (i = 0; i < n; i++) {
        fill_inputs(bufs, recs + (size_t)i * REC_ELEMS, i);

        double t0 = now_ms();
        bench_inference_run(bufs);
        double dt = now_ms() - t0;

#if INFERENCE_PROFILING
        inference_ddr_sample();
#endif
        const Data_t *ps = inference_buf_ptr(bufs[BENCH_IDX_START_LOGITS]);
        const Data_t *pe = inference_buf_ptr(bufs[BENCH_IDX_END_LOGITS]);
        memcpy(lg, ps, sizeof(int16_t) * BENCH_SEQ_LEN);
        memcpy(lg + BENCH_SEQ_LEN, pe, sizeof(int16_t) * BENCH_SEQ_LEN);
        if (fwrite(lg, sizeof(int16_t), 2u * BENCH_SEQ_LEN, fout) != 2u * BENCH_SEQ_LEN) {
            fprintf(stderr, "error: write %s failed\n", out_path);
            return 1;
        }
        fflush(fout);

        int uid_out = -1;
#if BENCH_IDX_UNIQUE_IDS_OUT >= 0
        uid_out = (int)(int16_t)inference_buf_ptr(bufs[BENCH_IDX_UNIQUE_IDS_OUT])[0];
        uid_ok += (uid_out == (int)i);
#else
        uid_ok++;
#endif
        lat[i] = dt;
        t_total += dt;
        unsigned s = argmax16(lg, BENCH_SEQ_LEN), e = argmax16(lg + BENCH_SEQ_LEN, BENCH_SEQ_LEN);
        fprintf(stderr, "example %u/%u: latency=%.1f ms  argmax start=%u (%.2f) end=%u (%.2f)  uid=%d\n",
                i + 1u, n, dt, s, lg[s] / 256.0, e, lg[BENCH_SEQ_LEN + e] / 256.0, uid_out);
    }
    fclose(fout);

#if INFERENCE_PROFILING
    inference_ddr_stop();
#endif

    /* ---- summary ----------------------------------------------------- */
    double *sorted = (double *)malloc(sizeof(double) * (n ? n : 1u));
    if (!sorted) return 1;
    memcpy(sorted, lat, sizeof(double) * n);
    qsort(sorted, n, sizeof(double), cmp_double);
    double mean = n ? t_total / n : 0.0;
    printf("{\"model\":\"%s\",\"examples\":%u,\"warmup\":%u,\"seq_len\":%u,"
           "\"init_ms\":%.1f,\"mean_ms\":%.3f,\"min_ms\":%.3f,\"max_ms\":%.3f,"
           "\"p50_ms\":%.3f,\"total_s\":%.3f,\"uid_ok\":%u,\"latencies_ms\":[",
           BENCH_MODEL_NAME, n, warmup, BENCH_SEQ_LEN, t_init, mean,
           n ? sorted[0] : 0.0, n ? sorted[n - 1u] : 0.0, n ? sorted[n / 2u] : 0.0,
           t_total / 1000.0, uid_ok);
    for (i = 0; i < n; i++)
        printf("%s%.3f", i ? "," : "", lat[i]);
    printf("]}\n");
    fflush(stdout);

#if INFERENCE_PROFILING
    inference_prof_dump_json(stdout);
    inference_ddr_dump_json(stdout);
    inference_prof_deinit();
    inference_ddr_deinit();
#endif

    for (k = 0; k < BENCH_N_BUFS; k++)
        inference_buf_free(bufs[k]);
    free(sorted);
    free(lat);
    free(lg);
    free(recs);
    inference_deinit();
    return 0;
}

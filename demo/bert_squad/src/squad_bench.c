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
 * Model setup (inference_init with the UIO instances, the I/O buffers,
 * raw int16 inputs / Q8.8 logit bits) is bert_api.c — the same code the
 * chat server's libbert_squad.so runs; model-specific glue comes from the
 * generated bench_glue.h through it.
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
#include "bert_api.h"

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

/* One inputs.bin record (input_ids, segment_ids, input_mask) through the
 * model; logits -> lg[0 .. 2*seq). */
static void run_record(const int16_t *rec, unsigned seq, unsigned uid, int16_t *lg, int *uid_out)
{
    bert_run_ex(rec, rec + seq, rec + 2u * seq, (int)uid, lg, lg + seq, uid_out);
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
    const unsigned seq       = bert_seq_len();
    const size_t   rec_elems = 3u * (size_t)seq;
    const size_t   rec_bytes = rec_elems * sizeof(int16_t);
    if (in_bytes <= 0 || (size_t)in_bytes % rec_bytes != 0) {
        fprintf(stderr, "error: %s: %ld bytes is not a multiple of %zu (3 x %u int16)\n",
                in_path, in_bytes, rec_bytes, seq);
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
            bert_model_name(), n, n_avail, seq, warmup);

    /* ---- init (inference_init + I/O buffers) ------------------------- */
    double t_init = now_ms();
    if (bert_open(NULL) != 0) {
        fprintf(stderr, "error: %s\n", bert_last_error());
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

    int16_t *lg = (int16_t *)malloc(sizeof(int16_t) * 2u * seq);
    if (!lg) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }

    /* ---- warm-up (example 0) ----------------------------------------- */
    unsigned w;
    for (w = 0; w < warmup; w++) {
        double t0 = now_ms();
        run_record(recs, seq, 0u, lg, NULL);
        fprintf(stderr, "warmup %u/%u: %.1f ms\n", w + 1u, warmup, now_ms() - t0);
    }
#if INFERENCE_PROFILING
    inference_prof_reset();
    inference_ddr_start();
#endif

    /* ---- timed run --------------------------------------------------- */
    double  *lat = (double *)malloc(sizeof(double) * (n ? n : 1u));
    if (!lat) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }
    double   t_total = 0.0;
    unsigned uid_ok  = 0, i;
    for (i = 0; i < n; i++) {
        int uid_out = BERT_NO_UID;
        double t0 = now_ms();
        run_record(recs + (size_t)i * rec_elems, seq, i, lg, &uid_out);
        double dt = now_ms() - t0;

#if INFERENCE_PROFILING
        inference_ddr_sample();
#endif
        if (fwrite(lg, sizeof(int16_t), 2u * seq, fout) != 2u * seq) {
            fprintf(stderr, "error: write %s failed\n", out_path);
            return 1;
        }
        fflush(fout);

        uid_ok += (uid_out == BERT_NO_UID || uid_out == (int)i);
        lat[i] = dt;
        t_total += dt;
        unsigned s = argmax16(lg, seq), e = argmax16(lg + seq, seq);
        fprintf(stderr, "example %u/%u: latency=%.1f ms  argmax start=%u (%.2f) end=%u (%.2f)  uid=%d\n",
                i + 1u, n, dt, s, lg[s] / 256.0, e, lg[seq + e] / 256.0, uid_out);
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
           bert_model_name(), n, warmup, seq, t_init, mean,
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

    free(sorted);
    free(lat);
    free(lg);
    free(recs);
    bert_close();
    return 0;
}

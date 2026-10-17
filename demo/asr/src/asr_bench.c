/*
 * asr_bench.c — the runner of libmoonshine (asr_api.c): the host-emulation
 * and board gates of doc/plans/ASR_PLAN.md §5.
 *
 *   1. asr_open() (timed; CmaFree before / after from /proc/meminfo).
 *   2. For every utterance of utts.bin: asr_transcribe(), -R repetitions
 *      (every repetition must give the same ids), the best one's time split
 *      into encode and decode; the ids are appended to ids.bin for the
 *      host's bit-exact comparison with the specification
 *      (demo/asr/scripts/asr_host_emu.py), with an FNV-1a checksum printed
 *      per utterance.
 *   3. With -DINFERENCE_PROFILING=ON: the per-layer profile of the first
 *      part's encode of the first utterance in each bucket (phase
 *      "encode_<T>", the entry's name) and of every decode step of utterance
 *      0's first part (phase "decode"), outside the timed runs.
 *   4. Re-open (-r): asr_close(), CmaFree, asr_open() again, utterance 0
 *      transcribed again and compared id for id.
 *
 * utts.bin: int32 count, then per utterance int32 n and n int16 samples
 * (16 kHz mono).  ids.bin: per utterance int32 count, then count int32 ids.
 * Result lines on stdout: "ASR_OPEN: {...}", "ASR_UTT: {...}" per
 * utterance, "ASR_REOPEN: {...}", "ASR_SUMMARY: {...}", and with profiling
 * "PROFILE_PHASE: <entry>" each followed by the profiler's "LAYERS_JSON: {...}".
 *
 * usage: asr_bench [-w weights_dir] [-i utts.bin] [-o ids.bin] [-R 1] [-r]
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, getopt */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "asr_api.h"
#include "inference.h"
#include "inference_prof.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static long cma_free_kb(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    char  line[256];
    long  v = -1;
    if (!f)
        return -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "CmaFree: %ld kB", &v) == 1)
            break;
    fclose(f);
    return v;
}

static uint32_t fnv1a(const int32_t *x, size_t n)
{
    const unsigned char *p = (const unsigned char *)x;
    uint32_t h = 2166136261u;
    size_t   i;
    for (i = 0; i < n * sizeof(int32_t); i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

/* Room for every id of n samples: each part has fewer than 256 tokens. */
static int max_ids(int n)
{
    return (n / asr_max_samples() + 1) * 256;
}

#if INFERENCE_PROFILING
/* The profiles of step 3: the first part's encode in a bucket not profiled
 * yet; with decode, its greedy steps as asr_transcribe() runs them (until </s>
 * or 6.5 tokens per second). */
static void profile(const int16_t *pcm, int n, int *seen, int decode)
{
    int32_t start, len;
    int     T, i, tok, b, steps;
    if (asr_chunks(pcm, n, &start, &len, 1) != 1)
        return;
    steps = (int)((double)len / asr_sample_rate() * 6.5);
    if (steps < 1)                                   /* < 0.154 s: never encoded */
        return;
    inference_prof_reset();
    T = asr_encode(pcm + start, len);
    if (T < 0)
        return;
    for (b = 0; b < asr_num_buckets() && asr_bucket(b) != T; b++)
        ;
    if (!seen[b]) {
        seen[b] = 1;
        printf("PROFILE_PHASE: encode_%d\n", T);
        inference_prof_dump_json(stdout);
        fflush(stdout);
    }
    if (!decode)
        return;
    inference_prof_reset();
    for (i = 0, tok = asr_bos(); i < steps; i++) {
        tok = asr_decode_step(tok, i);
        if (tok < 0 || tok == asr_eos())
            break;
    }
    printf("PROFILE_PHASE: decode\n");
    inference_prof_dump_json(stdout);
    fflush(stdout);
}
#endif

int main(int argc, char **argv)
{
    const char *wdir = NULL, *in_path = "utts.bin", *out_path = "ids.bin";
    int         reps = 1, reopen = 0, opt, count = 0, u, r;
    int        *ns, *first = NULL, nfirst = 0;
    int16_t   **pcm;
    FILE       *fi, *fo;
    double      sum_ms = 0.0, sum_audio = 0.0, sum_enc = 0.0, sum_dec = 0.0;
    long        sum_steps = 0, sum_tokens = 0;

    while ((opt = getopt(argc, argv, "w:i:o:R:r")) != -1) {
        switch (opt) {
        case 'w': wdir = optarg; break;
        case 'i': in_path = optarg; break;
        case 'o': out_path = optarg; break;
        case 'R': reps = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
        case 'r': reopen = 1; break;
        default:
            fprintf(stderr, "usage: %s [-w weights_dir] [-i utts.bin] [-o ids.bin] [-R 1] [-r]\n", argv[0]);
            return 2;
        }
    }

    /* ---- inputs ------------------------------------------------------- */
    fi = fopen(in_path, "rb");
    if (!fi || fread(&count, sizeof count, 1, fi) != 1 || count <= 0) {
        fprintf(stderr, "error: cannot read %s\n", in_path);
        return 1;
    }
    ns = calloc((size_t)count, sizeof *ns);
    pcm = calloc((size_t)count, sizeof *pcm);
    for (u = 0; u < count; u++) {
        if (fread(&ns[u], sizeof(int), 1, fi) != 1 || ns[u] <= 0) {
            fprintf(stderr, "error: %s: utterance %d header\n", in_path, u);
            return 1;
        }
        pcm[u] = malloc((size_t)ns[u] * sizeof(int16_t));
        if (fread(pcm[u], sizeof(int16_t), (size_t)ns[u], fi) != (size_t)ns[u]) {
            fprintf(stderr, "error: %s: utterance %d is short\n", in_path, u);
            return 1;
        }
    }
    fclose(fi);

    /* ---- open --------------------------------------------------------- */
    long   cma0 = cma_free_kb();
    double t = now_ms();
    if (asr_open(wdir) != 0) {
        fprintf(stderr, "error: asr_open: %s\n", asr_last_error());
        return 1;
    }
    double open_ms = now_ms() - t;
    long   cma1 = cma_free_kb();
    fprintf(stderr, "asr_bench: %s, asr_open %.0f ms, CmaFree %ld -> %ld kB (%.1f MB used)\n",
            asr_model_name(), open_ms, cma0, cma1, (cma0 - cma1) / 1024.0);
    printf("ASR_OPEN: {\"model\":\"%s\",\"open_ms\":%.1f,\"cma_free_kb_before\":%ld,"
           "\"cma_free_kb_after\":%ld,\"sample_rate\":%d,\"max_samples\":%d,\"buckets\":[",
           asr_model_name(), open_ms, cma0, cma1, asr_sample_rate(), asr_max_samples());
    for (r = 0; r < asr_num_buckets(); r++)
        printf("%s%d", r ? "," : "", asr_bucket(r));
    printf("]}\n");
    fflush(stdout);
#if INFERENCE_PROFILING
    if (inference_prof_init(inference_num_layers(), inference_layer_names_ptr()) != 0)
        fprintf(stderr, "warning: inference_prof_init failed\n");
#endif

    /* ---- utterances --------------------------------------------------- */
    fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        return 1;
    }
    for (u = 0; u < count; u++) {
        int         cap = max_ids(ns[u]), n = 0, n2, same = 1;
        int32_t    *ids = malloc((size_t)cap * sizeof(int32_t)), *again = malloc((size_t)cap * sizeof(int32_t));
        asr_stats_t st, best;
        double      audio_s = (double)ns[u] / asr_sample_rate();
        memset(&best, 0, sizeof best);
        for (r = 0; r < reps; r++) {
            n2 = asr_transcribe(pcm[u], ns[u], r ? again : ids, cap);
            if (n2 < 0) {
                fprintf(stderr, "error: utterance %d: %s\n", u, asr_last_error());
                return 1;
            }
            asr_last_stats(&st);
            if (r == 0)
                n = n2;
            else if (n2 != n || memcmp(ids, again, (size_t)n * sizeof(int32_t)) != 0)
                same = 0;
            if (r == 0 || st.total_ms < best.total_ms)
                best = st;
        }
        fwrite(&n, sizeof n, 1, fo);
        fwrite(ids, sizeof(int32_t), (size_t)n, fo);
        printf("ASR_UTT: {\"i\":%d,\"samples\":%d,\"audio_s\":%.3f,\"parts\":%d,\"tokens\":%d,\"steps\":%d,"
               "\"encode_ms\":%.2f,\"decode_ms\":%.2f,\"best_ms\":%.2f,\"rtf\":%.4f,\"ms_per_step\":%.3f,"
               "\"reps\":%d,\"reps_identical\":%s,\"fnv\":%u}\n",
               u, ns[u], audio_s, best.parts, best.tokens, best.steps, best.encode_ms, best.decode_ms,
               best.total_ms, best.total_ms / 1000.0 / audio_s,
               best.steps ? best.decode_ms / best.steps : 0.0, reps, same ? "true" : "false",
               fnv1a(ids, (size_t)n));
        fflush(stdout);
        fprintf(stderr, "asr_bench: utterance %d: %.2f s audio, %d part(s), %d tokens in %.0f ms "
                        "(encode %.0f, decode %.0f), RTF %.3f\n",
                u, audio_s, best.parts, best.tokens, best.total_ms, best.encode_ms, best.decode_ms,
                best.total_ms / 1000.0 / audio_s);
        sum_ms += best.total_ms;
        sum_enc += best.encode_ms;
        sum_dec += best.decode_ms;
        sum_steps += best.steps;
        sum_tokens += best.tokens;
        sum_audio += audio_s;
        if (u == 0) {
            first = ids;
            nfirst = n;
        } else {
            free(ids);
        }
        free(again);
    }
    fclose(fo);

#if INFERENCE_PROFILING
    {
        int *seen = calloc((size_t)asr_num_buckets(), sizeof(int));
        for (u = 0; u < count; u++)
            profile(pcm[u], ns[u], seen, u == 0);
        free(seen);
    }
#endif

    /* ---- re-open ------------------------------------------------------ */
    if (reopen) {
        int      cap = max_ids(ns[0]), n;
        int32_t *ids = malloc((size_t)cap * sizeof(int32_t));
        long     cma2, cma3;
        asr_close();
        cma2 = cma_free_kb();
        if (asr_open(wdir) != 0) {
            fprintf(stderr, "error: re-open: %s\n", asr_last_error());
            return 1;
        }
        cma3 = cma_free_kb();
        n = asr_transcribe(pcm[0], ns[0], ids, cap);
        printf("ASR_REOPEN: {\"cma_free_kb_after_close\":%ld,\"cma_free_kb_after_open\":%ld,"
               "\"identical\":%s}\n", cma2, cma3,
               n == nfirst && memcmp(ids, first, (size_t)n * sizeof(int32_t)) == 0 ? "true" : "false");
        free(ids);
    }
    free(first);

    asr_close();
    printf("ASR_SUMMARY: {\"utterances\":%d,\"total_ms\":%.1f,\"encode_ms\":%.1f,\"decode_ms\":%.1f,"
           "\"steps\":%ld,\"tokens\":%ld,\"audio_s\":%.3f,\"rtf\":%.4f,\"cma_free_kb_after_close\":%ld}\n",
           count, sum_ms, sum_enc, sum_dec, sum_steps, sum_tokens, sum_audio,
           sum_audio > 0 ? sum_ms / 1000.0 / sum_audio : 0.0, cma_free_kb());
    for (u = 0; u < count; u++)
        free(pcm[u]);
    free(pcm);
    free(ns);
    return 0;
}

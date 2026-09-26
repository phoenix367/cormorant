/*
 * llm_bench.c — KV260 runner for libsmollm2 (llm_api.c): the phase-3 board
 * gate of doc/CHAT_PLAN.md.
 *
 *   1. llm_open() (timed; CmaFree before / after from /proc/meminfo).
 *   2. For every prompt of prompts.bin: llm_truncate(1), llm_prefill(prompt),
 *      then -k greedy steps (argmax, first maximum) of llm_decode(); every
 *      logits vector (1 + k per prompt, vocab float32 each) is appended to
 *      logits.bin for the host's bit-exact comparison with the scheduler
 *      simulation (demo/chat/scripts/llm_board.py), with an FNV-1a checksum
 *      of its bits and its argmax printed per step.
 *   3. Decode timing: ms per llm_decode() over the steps of every prompt;
 *      with -DINFERENCE_PROFILING=ON the per-layer profile of the decode
 *      steps (prompt 0) is printed as a LAYERS_JSON:decode line.
 *   4. Prefill timing (-P 16,64,256): llm_truncate(1) + llm_prefill of that
 *      many tokens (the first prompt's ids, repeated), -R repetitions,
 *      LAYERS_JSON:prefill_<n> per length when profiling.
 *   5. Re-open (-r): llm_close(), CmaFree, llm_open() again, the first
 *      prompt's prefill + 2 decode steps compared bit for bit with step 2's.
 *
 * prompts.bin: int32 count, then per prompt int32 n and n int32 token ids
 * (the ids AFTER the leading <|im_start|>, which is the cache's sink).
 * Result lines on stdout: "LLM_OPEN: {...}", "LLM_PROMPT: {...}" per prompt,
 * "LLM_PREFILL: {...}" per length, "LLM_REOPEN: {...}", "LLM_SUMMARY: {...}",
 * and with profiling "PROFILE_PHASE: <phase>" + the profiler's "LAYERS_JSON: {...}".
 *
 * usage: llm_bench [-w weights_dir] [-i prompts.bin] [-o logits.bin] [-k 32]
 *                  [-P 16,64,256] [-R 3] [-r]
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, getopt */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "llm_api.h"
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

static uint32_t fnv1a(const float *x, int n)
{
    const unsigned char *p = (const unsigned char *)x;
    uint32_t h = 2166136261u;
    size_t   i;
    for (i = 0; i < (size_t)n * sizeof(float); i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

static int argmax(const float *x, int n)
{
    int i, b = 0;
    for (i = 1; i < n; i++)
        if (x[i] > x[b]) b = i;
    return b;
}

static void prof_dump(const char *tag)
{
#if INFERENCE_PROFILING
    printf("PROFILE_PHASE: %s\n", tag);        /* the next LAYERS_JSON: line belongs to it */
    inference_prof_dump_json(stdout);
    fflush(stdout);
#else
    (void)tag;
#endif
}

static void prof_reset(void)
{
#if INFERENCE_PROFILING
    inference_prof_reset();
#endif
}

int main(int argc, char **argv)
{
    const char *wdir = NULL, *in_path = "prompts.bin", *out_path = "logits.bin";
    const char *plens = "";
    int         k = 32, reps = 3, reopen = 0, c;
    while ((c = getopt(argc, argv, "w:i:o:k:P:R:r")) != -1) {
        switch (c) {
        case 'w': wdir = optarg; break;
        case 'i': in_path = optarg; break;
        case 'o': out_path = optarg; break;
        case 'k': k = atoi(optarg); break;
        case 'P': plens = optarg; break;
        case 'R': reps = atoi(optarg); break;
        case 'r': reopen = 1; break;
        default:
            fprintf(stderr, "usage: %s [-w weights_dir] [-i prompts.bin] [-o logits.bin] [-k 32]"
                            " [-P 16,64,256] [-R 3] [-r]\n", argv[0]);
            return 2;
        }
    }

    /* ---- prompts ------------------------------------------------------ */
    FILE *fin = fopen(in_path, "rb");
    int32_t np = 0;
    if (!fin || fread(&np, 4, 1, fin) != 1 || np < 1) {
        fprintf(stderr, "error: %s: cannot read the prompt count\n", in_path);
        return 1;
    }
    int32_t **prompts = calloc((size_t)np, sizeof *prompts), *plen = calloc((size_t)np, 4);
    int p;
    for (p = 0; p < np; p++) {
        if (fread(&plen[p], 4, 1, fin) != 1 || plen[p] < 1) return 1;
        prompts[p] = malloc((size_t)plen[p] * 4);
        if (fread(prompts[p], 4, (size_t)plen[p], fin) != (size_t)plen[p]) return 1;
    }
    fclose(fin);

    /* ---- open --------------------------------------------------------- */
    long   cma0 = cma_free_kb();
    double t = now_ms();
    if (llm_open(wdir) != 0) {
        fprintf(stderr, "error: llm_open: %s\n", llm_last_error());
        return 1;
    }
    double open_ms = now_ms() - t;
    long   cma1 = cma_free_kb();
    const int V = llm_vocab_size();
    fprintf(stderr, "llm_bench: %s vocab %d context %d, llm_open %.0f ms, CmaFree %ld -> %ld kB"
                    " (%.1f MB used)\n", llm_model_name(), V, llm_context_size(), open_ms,
            cma0, cma1, (cma0 - cma1) / 1024.0);
#if INFERENCE_PROFILING
    if (inference_prof_init(inference_num_layers(), inference_layer_names_ptr()) != 0)
        fprintf(stderr, "warning: inference_prof_init failed\n");
#endif

    float *lg = malloc((size_t)V * sizeof(float));
    float *ref = malloc((size_t)V * 3u * sizeof(float));
    FILE  *fout = fopen(out_path, "wb");
    if (!lg || !ref || !fout) {
        fprintf(stderr, "error: output\n");
        return 1;
    }
    double dec_sum = 0.0;
    int    dec_n = 0;
    printf("LLM_OPEN: {\"model\":\"%s\",\"vocab\":%d,\"context\":%d,\"open_ms\":%.1f,"
           "\"cma_free_kb_before\":%ld,\"cma_free_kb_after\":%ld,\"buckets\":[",
           llm_model_name(), V, llm_context_size(), open_ms, cma0, cma1);
    for (p = 0; p < llm_num_buckets(); p++)
        printf("%s%d", p ? "," : "", llm_bucket(p));
    printf("]}\n");
    fflush(stdout);
    for (p = 0; p < np; p++) {
        int     s, tok;
        double  pre_ms;
        if (llm_truncate(1) != 0) return 1;
        t = now_ms();
        if (llm_prefill(prompts[p], plen[p], lg) != 0) {
            fprintf(stderr, "error: llm_prefill: %s\n", llm_last_error());
            return 1;
        }
        pre_ms = now_ms() - t;
        fwrite(lg, sizeof(float), (size_t)V, fout);
        if (p == 0) memcpy(ref, lg, (size_t)V * sizeof(float));
        tok = argmax(lg, V);
        printf("LLM_PROMPT: {\"i\":%d,\"n\":%d,\"prefill_ms\":%.2f,\"steps\":[{\"tok\":%d,\"fnv\":%u}",
               p, plen[p], pre_ms, tok, (unsigned)fnv1a(lg, V));
        fprintf(stderr, "prompt %d: %d tokens, prefill %.1f ms, next %d\n", p, plen[p], pre_ms, tok);
        if (p == 0) prof_reset();
        for (s = 0; s < k; s++) {
            double d;
            t = now_ms();
            if (llm_decode(tok, lg) != 0) {
                fprintf(stderr, "error: llm_decode: %s\n", llm_last_error());
                return 1;
            }
            d = now_ms() - t;
            dec_sum += d;
            dec_n++;
            fwrite(lg, sizeof(float), (size_t)V, fout);
            if (p == 0 && s < 2) memcpy(ref + (size_t)(s + 1) * V, lg, (size_t)V * sizeof(float));
            tok = argmax(lg, V);
            printf(",{\"tok\":%d,\"fnv\":%u,\"ms\":%.2f}", tok, (unsigned)fnv1a(lg, V), d);
        }
        printf("],\"pos\":%d}\n", llm_position());
        fflush(stdout);
        if (p == 0)
            prof_dump("decode");
        fprintf(stderr, "prompt %d: %d decode steps, %.1f ms/token so far\n", p, k,
                dec_n ? dec_sum / dec_n : 0.0);
    }
    fclose(fout);

    /* ---- prefill timing --------------------------------------------- */
    {
        const char *q = plens;
        while (*q) {
            int n = atoi(q), r;
            double best = 1e30, sum = 0.0;
            int32_t *ids = malloc((size_t)(n > 0 ? n : 1) * 4);
            for (r = 0; r < n; r++)
                ids[r] = prompts[0][r % plen[0]];
            for (r = 0; r < reps && n > 0; r++) {
                if (llm_truncate(1) != 0) return 1;
                prof_reset();
                t = now_ms();
                if (llm_prefill(ids, n, lg) != 0) {
                    fprintf(stderr, "error: llm_prefill(%d): %s\n", n, llm_last_error());
                    return 1;
                }
                t = now_ms() - t;
                sum += t;
                if (t < best) best = t;
            }
            if (n > 0) {
                char tag[32];
                snprintf(tag, sizeof tag, "prefill_%d", n);
                printf("LLM_PREFILL: {\"n\":%d,\"best_ms\":%.2f,\"mean_ms\":%.2f,\"reps\":%d}\n",
                       n, best, sum / reps, reps);
                fflush(stdout);
                fprintf(stderr, "prefill %d tokens: best %.1f ms, mean %.1f ms\n", n, best, sum / reps);
                prof_dump(tag);
            }
            free(ids);
            while (*q && *q != ',') q++;
            if (*q == ',') q++;
        }
    }

    /* ---- re-open ------------------------------------------------------ */
    if (reopen) {
        long cma2, cma3;
        int  ok = 1, s, tok;
        llm_close();
        cma2 = cma_free_kb();
        t = now_ms();
        if (llm_open(wdir) != 0) {
            fprintf(stderr, "error: re-open: %s\n", llm_last_error());
            return 1;
        }
        t = now_ms() - t;
        cma3 = cma_free_kb();
        if (llm_prefill(prompts[0], plen[0], lg) != 0) return 1;
        ok &= memcmp(lg, ref, (size_t)V * sizeof(float)) == 0;
        tok = argmax(lg, V);
        for (s = 0; s < 2 && s < k; s++) {
            if (llm_decode(tok, lg) != 0) return 1;
            ok &= memcmp(lg, ref + (size_t)(s + 1) * V, (size_t)V * sizeof(float)) == 0;
            tok = argmax(lg, V);
        }
        fprintf(stderr, "re-open: CmaFree after close %ld kB, after open %ld kB, open %.0f ms, %s\n",
                cma2, cma3, t, ok ? "logits identical" : "LOGITS DIFFER");
        printf("LLM_REOPEN: {\"cma_free_kb_after_close\":%ld,\"cma_free_kb_after_open\":%ld,"
               "\"open_ms\":%.1f,\"identical\":%s}\n", cma2, cma3, t, ok ? "true" : "false");
    }
    llm_close();
    printf("LLM_SUMMARY: {\"decode_ms_mean\":%.2f,\"decode_steps\":%d,"
           "\"cma_free_kb_after_close\":%ld}\n", dec_n ? dec_sum / dec_n : 0.0, dec_n, cma_free_kb());
    fprintf(stderr, "llm_bench: done, CmaFree after close %ld kB\n", cma_free_kb());
    return 0;
}

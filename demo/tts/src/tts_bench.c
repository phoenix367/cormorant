/*
 * tts_bench.c — KV260 runner for libpiper_tts (tts_api.c): the board gate of
 * doc/plans/TTS_PLAN.md §4.
 *
 *   1. tts_open() (timed; CmaFree before / after from /proc/meminfo).
 *   2. For every utterance of utts.bin: tts_synthesize_chunk() for each
 *      chunk, timed per chunk, -R repetitions (every repetition must give the
 *      same samples); the samples are appended to pcm.bin for the host's
 *      bit-exact comparison with the specification (demo/tts/scripts/
 *      tts_board.py), with an FNV-1a checksum printed per utterance.  With
 *      -DINFERENCE_PROFILING=ON the per-layer profile of utterance 0's first
 *      repetition is printed as a LAYERS_JSON:chunk line (all its chunks).
 *   3. Re-open (-r): tts_close(), CmaFree, tts_open() again, utterance 0
 *      synthesized again and compared bit for bit.
 *
 * utts.bin: int32 count, then per utterance int32 frames and 192 x frames
 * float32 (z_p, channel-major).  pcm.bin: the utterances' int16 samples,
 * frames x 256 each, one after the other.
 * Result lines on stdout: "TTS_OPEN: {...}", "TTS_UTT: {...}" per utterance,
 * "TTS_REOPEN: {...}", "TTS_SUMMARY: {...}", and with profiling
 * "PROFILE_PHASE: chunk" + the profiler's "LAYERS_JSON: {...}".
 *
 * usage: tts_bench [-w weights_dir] [-i utts.bin] [-o pcm.bin] [-R 1] [-r]
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, getopt */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tts_api.h"
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

static uint32_t fnv1a(const int16_t *x, size_t n)
{
    const unsigned char *p = (const unsigned char *)x;
    uint32_t h = 2166136261u;
    size_t   i;
    for (i = 0; i < n * sizeof(int16_t); i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

/* Synthesize one utterance chunk by chunk; per-chunk times into ms[]. */
static int synth(const float *zp, int frames, int16_t *pcm, double *ms)
{
    int k, n, total = 0;
    for (k = 0; k < tts_num_chunks(frames); k++) {
        double t = now_ms();
        n = tts_synthesize_chunk(zp, frames, k, pcm + total);
        ms[k] = now_ms() - t;
        if (n < 0)
            return n;
        total += n;
    }
    return total;
}

int main(int argc, char **argv)
{
    const char *wdir = NULL, *in_path = "utts.bin", *out_path = "pcm.bin";
    int         reps = 1, reopen = 0, opt, count = 0, u, r, k;
    int        *frames;
    float     **zp;
    FILE       *fi, *fo;
    double      sum_ms = 0.0, sum_audio = 0.0;

    while ((opt = getopt(argc, argv, "w:i:o:R:r")) != -1) {
        switch (opt) {
        case 'w': wdir = optarg; break;
        case 'i': in_path = optarg; break;
        case 'o': out_path = optarg; break;
        case 'R': reps = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
        case 'r': reopen = 1; break;
        default:
            fprintf(stderr, "usage: %s [-w weights_dir] [-i utts.bin] [-o pcm.bin] [-R 1] [-r]\n",
                    argv[0]);
            return 2;
        }
    }

    /* ---- inputs ------------------------------------------------------- */
    fi = fopen(in_path, "rb");
    if (!fi || fread(&count, sizeof count, 1, fi) != 1 || count <= 0) {
        fprintf(stderr, "error: cannot read %s\n", in_path);
        return 1;
    }
    frames = calloc((size_t)count, sizeof *frames);
    zp = calloc((size_t)count, sizeof *zp);
    for (u = 0; u < count; u++) {
        size_t n;
        if (fread(&frames[u], sizeof(int), 1, fi) != 1 || frames[u] <= 0) {
            fprintf(stderr, "error: %s: utterance %d header\n", in_path, u);
            return 1;
        }
        n = (size_t)tts_channels() * (size_t)frames[u];
        zp[u] = malloc(n * sizeof(float));
        if (fread(zp[u], sizeof(float), n, fi) != n) {
            fprintf(stderr, "error: %s: utterance %d is short\n", in_path, u);
            return 1;
        }
    }
    fclose(fi);

    /* ---- open --------------------------------------------------------- */
    long   cma0 = cma_free_kb();
    double t = now_ms();
    if (tts_open(wdir) != 0) {
        fprintf(stderr, "error: tts_open: %s\n", tts_last_error());
        return 1;
    }
    double open_ms = now_ms() - t;
    long   cma1 = cma_free_kb();
    fprintf(stderr, "tts_bench: %s, tts_open %.0f ms, CmaFree %ld -> %ld kB (%.1f MB used)\n",
            tts_model_name(), open_ms, cma0, cma1, (cma0 - cma1) / 1024.0);
    printf("TTS_OPEN: {\"model\":\"%s\",\"open_ms\":%.1f,\"cma_free_kb_before\":%ld,"
           "\"cma_free_kb_after\":%ld,\"sample_rate\":%d,\"hop\":%d,\"chunk_frames\":%d}\n",
           tts_model_name(), open_ms, cma0, cma1, tts_sample_rate(), tts_hop(), tts_chunk_frames());
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
    int16_t *first = NULL;
    for (u = 0; u < count; u++) {
        int      nk = tts_num_chunks(frames[u]);
        size_t   ns = (size_t)frames[u] * (size_t)tts_hop();
        int16_t *pcm = malloc(ns * sizeof(int16_t)), *again = malloc(ns * sizeof(int16_t));
        double  *ms = malloc((size_t)nk * sizeof(double)), *best = malloc((size_t)nk * sizeof(double));
        int      same = 1;
        double   tot = 0.0, best_tot = 0.0, audio_s = (double)ns / tts_sample_rate();
        for (r = 0; r < reps; r++) {
#if INFERENCE_PROFILING
            if (u == 0 && r == 0) inference_prof_reset();
#endif
            if (synth(zp[u], frames[u], r ? again : pcm, ms) != (int)ns) {
                fprintf(stderr, "error: utterance %d: %s\n", u, tts_last_error());
                return 1;
            }
#if INFERENCE_PROFILING
            if (u == 0 && r == 0) {
                printf("PROFILE_PHASE: chunk\n");
                inference_prof_dump_json(stdout);
                fflush(stdout);
            }
#endif
            if (r && memcmp(pcm, again, ns * sizeof(int16_t)) != 0)
                same = 0;
            for (tot = 0.0, k = 0; k < nk; k++) {
                tot += ms[k];
                if (r == 0 || ms[k] < best[k]) best[k] = ms[k];
            }
            if (r == 0 || tot < best_tot) best_tot = tot;
        }
        fwrite(pcm, sizeof(int16_t), ns, fo);
        printf("TTS_UTT: {\"i\":%d,\"frames\":%d,\"chunks\":%d,\"samples\":%zu,\"audio_s\":%.3f,"
               "\"best_ms\":%.1f,\"rtf\":%.4f,\"reps\":%d,\"reps_identical\":%s,\"fnv\":%u,"
               "\"chunk_ms\":[", u, frames[u], nk, ns, audio_s, best_tot, best_tot / 1000.0 / audio_s,
               reps, same ? "true" : "false", fnv1a(pcm, ns));
        for (k = 0; k < nk; k++)
            printf("%s%.1f", k ? "," : "", best[k]);
        printf("]}\n");
        fflush(stdout);
        fprintf(stderr, "tts_bench: utterance %d: %d frames (%.2f s audio) in %.0f ms, RTF %.3f\n",
                u, frames[u], audio_s, best_tot, best_tot / 1000.0 / audio_s);
        sum_ms += best_tot;
        sum_audio += audio_s;
        if (u == 0)
            first = pcm;
        else
            free(pcm);
        free(again);
        free(ms);
        free(best);
    }
    fclose(fo);

    /* ---- re-open ------------------------------------------------------ */
    if (reopen) {
        size_t   ns = (size_t)frames[0] * (size_t)tts_hop();
        int16_t *pcm = malloc(ns * sizeof(int16_t));
        double  *ms = malloc((size_t)tts_num_chunks(frames[0]) * sizeof(double));
        long     cma2, cma3;
        tts_close();
        cma2 = cma_free_kb();
        if (tts_open(wdir) != 0) {
            fprintf(stderr, "error: re-open: %s\n", tts_last_error());
            return 1;
        }
        cma3 = cma_free_kb();
        int n = synth(zp[0], frames[0], pcm, ms);
        printf("TTS_REOPEN: {\"cma_free_kb_after_close\":%ld,\"cma_free_kb_after_open\":%ld,"
               "\"identical\":%s}\n", cma2, cma3,
               n == (int)ns && memcmp(pcm, first, ns * sizeof(int16_t)) == 0 ? "true" : "false");
        free(pcm);
        free(ms);
    }
    free(first);

    tts_close();
    printf("TTS_SUMMARY: {\"utterances\":%d,\"total_ms\":%.1f,\"audio_s\":%.3f,\"rtf\":%.4f,"
           "\"cma_free_kb_after_close\":%ld}\n", count, sum_ms, sum_audio,
           sum_audio > 0 ? sum_ms / 1000.0 / sum_audio : 0.0, cma_free_kb());
    for (u = 0; u < count; u++)
        free(zp[u]);
    free(zp);
    free(frames);
    return 0;
}

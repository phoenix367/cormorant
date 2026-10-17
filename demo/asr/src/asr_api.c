/*
 * asr_api.c — see asr_api.h.  The buckets, the model's constants and the run
 * functions come from the generated test/asr_glue.h; ASR_API_WEIGHTS_DIR and
 * ASR_MODEL_NAME from the CMake targets added by
 * demo/asr/scripts/generate_asr_project.py.  The parts (asr_chunks), the
 * encoder's frame count and the token limit are the specification's —
 * demo/asr/scripts/moonshine_q.py chunks / transcribe and
 * inference-scheduler/src/asr_nodes.py stem_frames — integer for integer.
 */

#define _POSIX_C_SOURCE 200809L   /* clock_gettime, fchdir, realpath */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "asr_api.h"
#include "inference.h"
#include "asr_glue.h"

#ifndef ASR_API_WEIGHTS_DIR
#  define ASR_API_WEIGHTS_DIR "."
#endif
#ifndef ASR_MODEL_NAME
#  define ASR_MODEL_NAME "moonshine"
#endif

#define ASR_SAMPLE_RATE     16000
#define ASR_TOKENS_PER_S    6.5                       /* the model card's generation limit */
#define ASR_SPLIT_SEARCH    (2 * ASR_SAMPLE_RATE)     /* a boundary within +-2 s of the equal split ... */
#define ASR_SPLIT_WIN       1600                      /* ... at the centre of the quietest 0.1 s window */
#define ASR_BUCKET_SAMPLES(T) (384 * (T) + 511)

_Static_assert(ASR_MAX_SAMPLES == ASR_BUCKET_SAMPLES(ASR_MAX_FRAMES), "the largest bucket's samples");

/* inference.c (src/asr_nodes.py ASR_C): the logits the last argmax read */
extern const int16_t *asr_argmax_logits;
extern unsigned       asr_argmax_vocab;

static int         s_open;
static int         s_stepped;                         /* asr_last_logits(): a decode step since the last encode */
static char        s_err[512];
static int16_t     s_pcm[ASR_MAX_SAMPLES];            /* the bucket's samples, zero past n */
static int32_t     s_xpos = -1;                       /* the encoded audio's frames - 1; -1: none */
static asr_stats_t s_stats;

static void set_err(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_err, sizeof s_err, fmt, ap);
    va_end(ap);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int same_dir(const char *a, const char *b)
{
    char ra[PATH_MAX], rb[PATH_MAX];
    size_t la, lb;
    if (realpath(a, ra) && realpath(b, rb))
        return strcmp(ra, rb) == 0;
    la = strlen(a);
    lb = strlen(b);
    while (la > 1u && a[la - 1u] == '/') la--;
    while (lb > 1u && b[lb - 1u] == '/') lb--;
    return la == lb && strncmp(a, b, la) == 0;
}

int asr_open(const char *weights_dir)
{
    int rc, cwd = -1;

    s_err[0] = '\0';
    if (s_open)
        return 0;
    if (weights_dir && *weights_dir && !same_dir(weights_dir, ASR_API_WEIGHTS_DIR)) {
        if (ASR_API_WEIGHTS_DIR[0] == '/') {
            set_err("this build reads its weights from %s/weights (INFERENCE_WEIGHTS_DIR); "
                    "rebuild with -DINFERENCE_WEIGHTS_DIR=%s to use %s",
                    ASR_API_WEIGHTS_DIR, weights_dir, weights_dir);
            return -2;
        }
        cwd = open(".", O_RDONLY);
        if (cwd < 0 || chdir(weights_dir) != 0) {
            set_err("chdir(%s): %s", weights_dir, strerror(errno));
            if (cwd >= 0)
                close(cwd);
            return -3;
        }
    }
    rc = asr_glue_init();
    if (cwd >= 0) {
        if (fchdir(cwd) != 0)
            fprintf(stderr, "asr_api: warning: cannot restore the working directory: %s\n",
                    strerror(errno));
        close(cwd);
    }
    if (rc != 0) {
        set_err("inference_init() failed (%d): check the UIO device, CMA (CmaFree) and "
                "the weights under %s/weights", rc,
                weights_dir && *weights_dir ? weights_dir : ASR_API_WEIGHTS_DIR);
        return -4;
    }
    s_xpos = -1;
    s_open = 1;
    return 0;
}

void asr_close(void)
{
    if (!s_open)
        return;
    inference_deinit();
    s_xpos = -1;
    s_open = 0;
}

const char *asr_last_error(void)  { return s_err; }
int         asr_sample_rate(void) { return ASR_SAMPLE_RATE; }
int         asr_max_samples(void) { return ASR_MAX_SAMPLES; }
int         asr_num_buckets(void) { return ASR_N_BUCKETS; }
int         asr_bucket(int i)     { return i >= 0 && i < ASR_N_BUCKETS ? asr_buckets[i] : -1; }
int         asr_bos(void)         { return ASR_BOS; }
int         asr_eos(void)         { return ASR_EOS; }
int         asr_vocab(void)       { return ASR_VOCAB; }
int         asr_logit_frac(void)  { return ASR_LOGIT_FRAC; }
const char *asr_model_name(void)  { return ASR_MODEL_NAME; }
const char *asr_weights_dir(void) { return ASR_API_WEIGHTS_DIR; }

void asr_last_stats(asr_stats_t *st)
{
    if (st)
        *st = s_stats;
}

/* The valid encoder frames of n samples: conv1 (k 127, s 64), conv2 (k 7, s 3),
 * conv3 (k 3, s 2); 0 when the audio is too short (asr_nodes.stem_frames). */
static int stem_frames(long n)
{
    long t1 = n >= 127 ? (n - 127) / 64 + 1 : 0;
    long t2 = t1 >= 7 ? (t1 - 7) / 3 + 1 : 0;
    return t2 >= 3 ? (int)((t2 - 3) / 2 + 1) : 0;
}

/* The tokens of n samples, <s> included (moonshine_np.max_tokens, capped by the
 * decoder's positions as moonshine_q.transcribe). */
static int token_limit(int n)
{
    int l = (int)((double)n / ASR_SAMPLE_RATE * ASR_TOKENS_PER_S) + 1;
    return l < ASR_MAX_POS ? l : ASR_MAX_POS;
}

static int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }

/* Boundary i of k (moonshine_q.chunks): the centre of the first 0.1 s window of
 * least energy (the exact sum of the squared samples) within +-2 s of the
 * equal split, kept where every part stays <= M samples. */
static int64_t quiet_boundary(const int16_t *x, int64_t n, int64_t M, int64_t k, int64_t i, int64_t prev)
{
    int64_t ideal = i * n / k;
    int64_t lo = max64(max64(prev + 1, ideal - ASR_SPLIT_SEARCH), n - (k - i) * M);
    int64_t hi = min64(ideal + ASR_SPLIT_SEARCH, prev + M);
    int64_t s0 = max64(lo - ASR_SPLIT_WIN / 2, 0), s1 = max64(hi - ASR_SPLIT_WIN / 2, 0);
    int64_t last = min64(s1, n - ASR_SPLIT_WIN), s, j, e = 0, best, at = s0;

    if (last < s0)
        return min64(max64(ideal, lo), hi);
    for (j = s0; j < s0 + ASR_SPLIT_WIN; j++)
        e += (int64_t)x[j] * x[j];
    best = e;
    for (s = s0 + 1; s <= last; s++) {
        e += (int64_t)x[s + ASR_SPLIT_WIN - 1] * x[s + ASR_SPLIT_WIN - 1] - (int64_t)x[s - 1] * x[s - 1];
        if (e < best) {
            best = e;
            at = s;
        }
    }
    return min64(max64(at + ASR_SPLIT_WIN / 2, lo), hi);
}

static int num_parts(int n)
{
    return n <= ASR_MAX_SAMPLES ? 1 : (int)(((int64_t)n + ASR_MAX_SAMPLES - 1) / ASR_MAX_SAMPLES);
}

int asr_chunks(const int16_t *pcm, int n, int32_t *starts, int32_t *lens, int max_parts)
{
    int64_t prev = 0, b;
    int     k, i;

    if (!pcm || n < 1 || !starts || !lens || max_parts < 1) {
        set_err("asr_chunks: bad arguments (n %d, max_parts %d)", n, max_parts);
        return -2;
    }
    k = num_parts(n);
    for (i = 1; i <= k; i++) {
        b = i < k ? quiet_boundary(pcm, n, ASR_MAX_SAMPLES, k, i, prev) : n;
        if (i <= max_parts) {
            starts[i - 1] = (int32_t)prev;
            lens[i - 1] = (int32_t)(b - prev);
        }
        prev = b;
    }
    return k < max_parts ? k : max_parts;
}

int asr_encode(const int16_t *pcm, int n)
{
    int     b, nf, T;
    int32_t nn = n, pos = 0;

    if (!s_open) {
        set_err("asr_open() first");
        return -1;
    }
    if (!pcm || n < 1 || n > ASR_MAX_SAMPLES) {
        set_err("asr_encode: %d samples (1 .. %d)", n, ASR_MAX_SAMPLES);
        return -2;
    }
    nf = stem_frames(n);
    if (nf < 1) {
        set_err("asr_encode: %d samples make no encoder frame (895 at least)", n);
        return -3;
    }
    for (b = 0; b < ASR_N_BUCKETS - 1 && ASR_BUCKET_SAMPLES(asr_buckets[b]) < n; b++)
        ;
    T = asr_buckets[b];
    memcpy(s_pcm, pcm, (size_t)n * sizeof(int16_t));
    memset(s_pcm + n, 0, (size_t)(ASR_BUCKET_SAMPLES(T) - n) * sizeof(int16_t));
    s_stepped = 0;                        /* the encoder's buffers may share the logits' pool slot */
    asr_glue_encode(b, s_pcm, &nn, &pos);
    s_xpos = nf - 1;
    return T;
}

int asr_decode_step(int token, int pos)
{
    int32_t id = token, p = pos, xp = s_xpos, out = -1;

    if (!s_open) {
        set_err("asr_open() first");
        return -1;
    }
    if (s_xpos < 0) {
        set_err("asr_decode_step: asr_encode() first");
        return -3;
    }
    if (token < 0 || token >= ASR_VOCAB || pos < 0 || pos >= ASR_MAX_POS) {
        set_err("asr_decode_step: token %d (0 .. %d), position %d (0 .. %d)", token, ASR_VOCAB - 1, pos,
                ASR_MAX_POS - 1);
        return -2;
    }
    asr_glue_decode(&id, &p, &xp, &out);
    s_stepped = 1;
    return out;
}

int asr_last_logits(const int32_t *tokens, int n, int32_t *logits)
{
    int i;

    if (!s_open) {
        set_err("asr_open() first");
        return -1;
    }
    if (!tokens || !logits || n < 0) {
        set_err("asr_last_logits: bad arguments (n %d)", n);
        return -2;
    }
    if (!s_stepped || !asr_argmax_logits || asr_argmax_vocab != ASR_VOCAB) {
        set_err("asr_last_logits: asr_decode_step() first");
        return -3;
    }
    for (i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= ASR_VOCAB) {
            set_err("asr_last_logits: token %d (0 .. %d)", (int)tokens[i], ASR_VOCAB - 1);
            return -2;
        }
        logits[i] = asr_argmax_logits[tokens[i]];
    }
    return n;
}

int asr_max_steps(int n)
{
    return n < 1 ? 0 : token_limit(n) - 1;
}

/* One part: encode, then greedy steps from <s> until </s> or the token limit
 * (moonshine_q.transcribe); the ids appended at *count (stored while < max_ids). */
static int transcribe_part(const int16_t *pcm, int n, int32_t *ids, int max_ids, int *count)
{
    int    limit = token_limit(n), i, tok = ASR_BOS, next;
    double t;

    if (limit <= 1)                       /* < 0.154 s: no token, nothing encoded */
        return 0;
    t = now_ms();
    next = asr_encode(pcm, n);
    s_stats.encode_ms += now_ms() - t;
    if (next < 0)
        return next;
    t = now_ms();
    for (i = 0; i < limit - 1; i++) {
        next = asr_decode_step(tok, i);
        s_stats.steps++;
        if (next < 0)
            return next;
        if (next == ASR_EOS)
            break;
        if (*count < max_ids)
            ids[*count] = next;
        (*count)++;
        tok = next;
    }
    s_stats.decode_ms += now_ms() - t;
    return 0;
}

int asr_transcribe(const int16_t *pcm, int n, int32_t *ids, int max_ids)
{
    int32_t *starts, *lens;
    int      k, p, count = 0, rc = 0;
    double   t0 = now_ms();

    memset(&s_stats, 0, sizeof s_stats);
    if (!s_open) {
        set_err("asr_open() first");
        return -1;
    }
    if (!pcm || n < 1 || max_ids < 0 || (max_ids > 0 && !ids)) {
        set_err("asr_transcribe: bad arguments (n %d, max_ids %d)", n, max_ids);
        return -2;
    }
    k = num_parts(n);
    starts = malloc((size_t)k * sizeof *starts);
    lens = malloc((size_t)k * sizeof *lens);
    if (!starts || !lens) {
        free(starts);
        free(lens);
        set_err("asr_transcribe: out of memory (%d parts)", k);
        return -5;
    }
    asr_chunks(pcm, n, starts, lens, k);
    for (p = 0; p < k && rc == 0; p++)
        rc = transcribe_part(pcm + starts[p], lens[p], ids, max_ids, &count);
    free(starts);
    free(lens);
    s_stats.parts = k;
    s_stats.tokens = count;
    s_stats.total_ms = now_ms() - t0;
    if (rc < 0)
        return rc;
    return count < max_ids ? count : max_ids;
}

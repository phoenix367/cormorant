/*
 * llm_api.c — see llm_api.h.  The model-specific glue (entry functions,
 * vocabulary, context, prefill buckets, UIO instances) comes from the
 * generated llm_glue.h; LLM_API_WEIGHTS_DIR from the project's
 * INFERENCE_WEIGHTS_DIR (both set by demo/chat/scripts/generate_llm_project.py).
 *
 * Prefill split (identical to demo/chat/scripts/llm_project.py split_prefill,
 * which the bit-exactness check drives): the largest bucket the remaining
 * tokens fill completely, and the last remainder padded into the smallest
 * bucket (rows >= n are computed but write nothing into the cache and are
 * never read).  Every call to a prefill entry leaves the last valid row in
 * the h_last state; llm_prefill() then runs the head entry once.
 */

#define _POSIX_C_SOURCE 200809L   /* fchdir, realpath */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "llm_api.h"
#include "inference.h"
#include "llm_glue.h"

#ifndef LLM_API_WEIGHTS_DIR
#  define LLM_API_WEIGHTS_DIR "."
#endif

static int     s_open;
static int     s_pos;                       /* positions filled, incl. the sink */
static char    s_err[512];
static int32_t s_ids[LLM_MAX_BUCKET];

static void set_err(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_err, sizeof s_err, fmt, ap);
    va_end(ap);
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

int llm_open(const char *weights_dir)
{
    int rc, cwd = -1;

    s_err[0] = '\0';
    if (s_open)
        return 0;
    if (weights_dir && *weights_dir && !same_dir(weights_dir, LLM_API_WEIGHTS_DIR)) {
        if (LLM_API_WEIGHTS_DIR[0] == '/') {
            set_err("this build reads its weights from %s/weights (INFERENCE_WEIGHTS_DIR); "
                    "rebuild with -DINFERENCE_WEIGHTS_DIR=%s to use %s",
                    LLM_API_WEIGHTS_DIR, weights_dir, weights_dir);
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
    rc = llm_glue_init();
    if (cwd >= 0) {
        if (fchdir(cwd) != 0)
            fprintf(stderr, "llm_api: warning: cannot restore the working directory: %s\n",
                    strerror(errno));
        close(cwd);
    }
    if (rc != 0) {
        set_err("inference_init() failed (%d): check the UIO devices, CMA (CmaFree) and "
                "the weights under %s/weights", rc,
                weights_dir && *weights_dir ? weights_dir : LLM_API_WEIGHTS_DIR);
        return -4;
    }
    s_open = 1;
    s_pos = 1;                              /* the sink */
    return 0;
}

void llm_close(void)
{
    if (!s_open)
        return;
    inference_deinit();
    s_open = 0;
    s_pos = 0;
}

const char *llm_last_error(void)    { return s_err; }
int         llm_vocab_size(void)    { return LLM_VOCAB; }
int         llm_context_size(void)  { return LLM_CONTEXT; }
int         llm_position(void)      { return s_open ? s_pos : 0; }
const char *llm_model_name(void)    { return LLM_MODEL_NAME; }
const char *llm_weights_dir(void)   { return LLM_API_WEIGHTS_DIR; }
int         llm_num_buckets(void)   { return (int)LLM_N_BUCKETS; }

int llm_bucket(int i)
{
    return (i >= 0 && i < (int)LLM_N_BUCKETS) ? (int)llm_buckets[i] : -1;
}

int llm_truncate(int n)
{
    if (!s_open) {
        set_err("llm_truncate: the model is not open");
        return -1;
    }
    if (n < 1 || n > s_pos) {
        set_err("llm_truncate(%d): must be in [1, %d]", n, s_pos);
        return -2;
    }
    s_pos = n;
    return 0;
}

static int check_tokens(const int32_t *tokens, int n)
{
    int i;
    for (i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= LLM_VOCAB) {
            set_err("token %d at index %d outside [0, %d)", (int)tokens[i], i, LLM_VOCAB);
            return -1;
        }
    return 0;
}

int llm_prefill(const int32_t *tokens, int n, float *logits)
{
    int done = 0;
    if (!s_open) {
        set_err("llm_prefill: the model is not open");
        return -1;
    }
    if (n < 1 || !tokens || !logits) {
        set_err("llm_prefill: n = %d (>= 1 tokens and an output needed)", n);
        return -2;
    }
    if (s_pos + n > LLM_CONTEXT) {
        set_err("llm_prefill: %d + %d positions exceed the context (%d)", s_pos, n, LLM_CONTEXT);
        return -3;
    }
    if (check_tokens(tokens, n) != 0)
        return -4;
    while (done < n) {
        unsigned b = llm_buckets[0], k, j;
        int32_t  pos, nv;
        for (j = 0; j < LLM_N_BUCKETS; j++)
            if ((int)llm_buckets[j] <= n - done)
                b = llm_buckets[j];
        k = (unsigned)(n - done) < b ? (unsigned)(n - done) : b;
        memset(s_ids, 0, sizeof s_ids);
        memcpy(s_ids, tokens + done, (size_t)k * sizeof(int32_t));
        pos = (int32_t)s_pos;
        nv = (int32_t)k;
        llm_glue_prefill(b, s_ids, &pos, &nv);
        s_pos += (int)k;
        done += (int)k;
    }
    llm_glue_head(logits);
    return 0;
}

int llm_decode(int32_t token, float *logits)
{
    int32_t pos;
    if (!s_open) {
        set_err("llm_decode: the model is not open");
        return -1;
    }
    if (!logits) {
        set_err("llm_decode: no output");
        return -2;
    }
    if (s_pos + 1 > LLM_CONTEXT) {
        set_err("llm_decode: the context (%d positions) is full", LLM_CONTEXT);
        return -3;
    }
    if (check_tokens(&token, 1) != 0)
        return -4;
    pos = (int32_t)s_pos;
    llm_glue_decode(&token, &pos, logits);
    s_pos++;
    return 0;
}

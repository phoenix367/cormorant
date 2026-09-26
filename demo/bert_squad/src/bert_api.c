/*
 * bert_api.c — see bert_api.h.  Model-specific glue (buffer order / sizes /
 * roles, inference_init() arguments) comes from the generated bench_glue.h;
 * the UIO instance names from the INFERENCE_*_INSTANCE compile definitions
 * and BERT_API_WEIGHTS_DIR from the project's INFERENCE_WEIGHTS_DIR (both
 * set by the CMake targets that demo/bert_squad/scripts/generate_project.py
 * adds).
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

#include "bert_api.h"
#include "inference.h"
#include "bench_glue.h"

#ifndef BERT_API_WEIGHTS_DIR
#  define BERT_API_WEIGHTS_DIR "."
#endif

static inference_buf_t *s_bufs[BENCH_N_BUFS];
static int              s_open;
static char             s_err[512];

static void set_err(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_err, sizeof s_err, fmt, ap);
    va_end(ap);
}

/* Same directory?  realpath() when both exist, else the strings without
 * trailing slashes. */
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

static void free_bufs(void)
{
    unsigned k;
    for (k = 0; k < BENCH_N_BUFS; k++) {
        if (s_bufs[k])
            inference_buf_free(s_bufs[k]);
        s_bufs[k] = NULL;
    }
}

int bert_open(const char *weights_dir)
{
    int      rc, cwd = -1;
    unsigned k;

    s_err[0] = '\0';
    if (s_open)
        return 0;
    if (weights_dir && *weights_dir && !same_dir(weights_dir, BERT_API_WEIGHTS_DIR)) {
        if (BERT_API_WEIGHTS_DIR[0] == '/') {
            set_err("this build reads its weights from %s/weights (INFERENCE_WEIGHTS_DIR); "
                    "rebuild with -DINFERENCE_WEIGHTS_DIR=%s to use %s",
                    BERT_API_WEIGHTS_DIR, weights_dir, weights_dir);
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
    rc = bench_inference_init();
    if (cwd >= 0) {
        if (fchdir(cwd) != 0)
            fprintf(stderr, "bert_api: warning: cannot restore the working directory: %s\n",
                    strerror(errno));
        close(cwd);
    }
    if (rc != 0) {
        set_err("inference_init() failed (%d): check the UIO devices and the weights under %s/weights",
                rc, weights_dir && *weights_dir ? weights_dir : BERT_API_WEIGHTS_DIR);
        return -4;
    }
    for (k = 0; k < BENCH_N_BUFS; k++) {
        s_bufs[k] = inference_buf_alloc(bench_buf_numel[k]);
        if (!s_bufs[k]) {
            set_err("inference_buf_alloc(%u) failed for '%s' (%s)", bench_buf_numel[k],
                    bench_buf_name[k], bench_buf_is_int[k] ? "int16" : "Q8.8");
            free_bufs();
            inference_deinit();
            return -5;
        }
        memset(inference_buf_ptr(s_bufs[k]), 0, (size_t)bench_buf_numel[k] * sizeof(Data_t));
    }
    s_open = 1;
    return 0;
}

int bert_run_ex(const int16_t *ids, const int16_t *seg, const int16_t *mask,
                int uid, int16_t *start_logits, int16_t *end_logits, int *uid_out)
{
    const int      roles[3] = { BENCH_IDX_INPUT_IDS, BENCH_IDX_SEGMENT_IDS, BENCH_IDX_INPUT_MASK };
    const int16_t *src[3];
    unsigned       r, i;

    if (!s_open) {
        set_err("bert_run: the model is not open");
        return -1;
    }
    src[0] = ids;
    src[1] = seg;
    src[2] = mask;
    /* Integer inputs are raw int16 in Data_t (inference.h). */
    for (r = 0; r < 3u; r++) {
        Data_t *p = inference_buf_ptr(s_bufs[roles[r]]);
        for (i = 0; i < BENCH_SEQ_LEN; i++)
            p[i] = (Data_t)src[r][i];
    }
#if BENCH_IDX_UNIQUE_IDS >= 0
    inference_buf_ptr(s_bufs[BENCH_IDX_UNIQUE_IDS])[0] = (Data_t)(int16_t)uid;
#else
    (void)uid;
#endif

    bench_inference_run(s_bufs);

    memcpy(start_logits, inference_buf_ptr(s_bufs[BENCH_IDX_START_LOGITS]),
           sizeof(int16_t) * BENCH_SEQ_LEN);
    memcpy(end_logits, inference_buf_ptr(s_bufs[BENCH_IDX_END_LOGITS]),
           sizeof(int16_t) * BENCH_SEQ_LEN);
    if (uid_out) {
#if BENCH_IDX_UNIQUE_IDS_OUT >= 0
        *uid_out = (int)(int16_t)inference_buf_ptr(s_bufs[BENCH_IDX_UNIQUE_IDS_OUT])[0];
#else
        *uid_out = BERT_NO_UID;
#endif
    }
    return 0;
}

int bert_run(const int16_t *ids, const int16_t *seg, const int16_t *mask,
             int16_t *start_logits, int16_t *end_logits)
{
    return bert_run_ex(ids, seg, mask, 0, start_logits, end_logits, NULL);
}

void bert_close(void)
{
    if (!s_open)
        return;
    free_bufs();
    inference_deinit();
    s_open = 0;
}

unsigned bert_seq_len(void)
{
    return BENCH_SEQ_LEN;
}

const char *bert_model_name(void)
{
    return BENCH_MODEL_NAME;
}

const char *bert_weights_dir(void)
{
    return BERT_API_WEIGHTS_DIR;
}

const char *bert_last_error(void)
{
    return s_err;
}

/* fake_libsmollm2.c — the libsmollm2.so C API of doc/CHAT_PLAN.md §11 with
 * scripted logits, for testing the ctypes path of smollm2_backend.py on a
 * host without the FPGA.  Build:  cc -O2 -shared -fPIC -o libfakellm.so fake_libsmollm2.c
 *
 * llm_open(dir) reads dir/script.txt: line 1 = the generation prompt ids
 * (e.g. "1 520 9531 198"), line 2 = the reply ids.  After the last occurrence
 * of the generation prompt the next token is reply[k] (then <|im_end|> = 2);
 * logits are 0 except 30 at that token.  Context and vocabulary sizes come
 * from dir/sizes.txt ("vocab ctx") if present (default 49152 1024).
 * fake_llm_counters() exposes calls for the tests.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXS 4096

static int     s_open, s_vocab = 49152, s_ctx = 1024, s_pos;
static int32_t s_seq[MAXS];
static int32_t s_gen[64], s_ngen;
static int32_t s_reply[MAXS], s_nreply;
static char    s_err[256];
static long    s_counters[4];                /* opens, prefill tokens, decodes, truncates */

const char *llm_last_error(void) { return s_err; }
int         llm_vocab_size(void) { return s_open ? s_vocab : -1; }
int         llm_context_size(void) { return s_open ? s_ctx : -1; }
long       *fake_llm_counters(void) { return s_counters; }

static int read_ids(FILE *f, int32_t *out, int max)
{
    char  line[65536], *p, *end;
    int   n = 0;
    long  v;

    if (!fgets(line, sizeof line, f))
        return 0;
    for (p = line; n < max; p = end) {
        v = strtol(p, &end, 10);
        if (end == p)
            break;
        out[n++] = (int32_t)v;
    }
    return n;
}

int llm_open(const char *weights_dir)
{
    char  path[1024];
    FILE *f;

    if (s_open)
        return 0;
    if (!weights_dir) {
        snprintf(s_err, sizeof s_err, "fake: weights_dir required");
        return -2;
    }
    snprintf(path, sizeof path, "%s/script.txt", weights_dir);
    f = fopen(path, "r");
    if (!f) {
        snprintf(s_err, sizeof s_err, "fake: cannot open %s", path);
        return -3;
    }
    s_ngen = read_ids(f, s_gen, 64);
    s_nreply = read_ids(f, s_reply, MAXS);
    fclose(f);
    snprintf(path, sizeof path, "%s/sizes.txt", weights_dir);
    f = fopen(path, "r");
    if (f) {
        if (fscanf(f, "%d %d", &s_vocab, &s_ctx) != 2 || s_ctx > MAXS) {
            fclose(f);
            snprintf(s_err, sizeof s_err, "fake: bad sizes.txt");
            return -4;
        }
        fclose(f);
    }
    s_seq[0] = 1;                                /* <|im_start|>, the sink */
    s_pos = 1;
    s_open = 1;
    s_counters[0]++;
    s_err[0] = 0;
    return 0;
}

void llm_close(void)
{
    s_open = 0;
}

int llm_position(void)
{
    if (!s_open) {
        snprintf(s_err, sizeof s_err, "fake: not open");
        return -1;
    }
    return s_pos;
}

int llm_truncate(int n)
{
    if (!s_open || n < 1 || n > s_pos) {
        snprintf(s_err, sizeof s_err, "fake: llm_truncate(%d) at position %d", n, s_pos);
        return -1;
    }
    s_pos = n;
    s_counters[3]++;
    return 0;
}

static void next_logits(float *logits)
{
    int i, k, last = -1, next = 2;

    for (i = s_pos - s_ngen; i >= 0 && s_ngen > 0; i--)
        if (memcmp(&s_seq[i], s_gen, sizeof(int32_t) * (size_t)s_ngen) == 0) {
            last = i;
            break;
        }
    if (last >= 0) {
        k = s_pos - (last + s_ngen);
        if (k < s_nreply)
            next = s_reply[k];
    }
    memset(logits, 0, sizeof(float) * (size_t)s_vocab);
    logits[next] = 30.0f;
}

static int append(const int32_t *t, int n)
{
    int i;

    for (i = 0; i < n; i++)
        if (t[i] < 0 || t[i] >= s_vocab) {
            snprintf(s_err, sizeof s_err, "fake: token %d out of range", t[i]);
            return -1;
        }
    if (s_pos + n > s_ctx) {
        snprintf(s_err, sizeof s_err, "fake: context full (%d + %d > %d)", s_pos, n, s_ctx);
        return -2;
    }
    memcpy(&s_seq[s_pos], t, sizeof(int32_t) * (size_t)n);
    s_pos += n;
    return 0;
}

int llm_prefill(const int32_t *tokens, int n, float *logits)
{
    int rc;

    if (!s_open || n < 1 || !tokens || !logits) {
        snprintf(s_err, sizeof s_err, "fake: llm_prefill bad arguments");
        return -1;
    }
    rc = append(tokens, n);
    if (rc < 0)
        return rc;
    s_counters[1] += n;
    next_logits(logits);
    return 0;
}

int llm_decode(int32_t token, float *logits)
{
    int rc;

    if (!s_open || !logits) {
        snprintf(s_err, sizeof s_err, "fake: llm_decode bad arguments");
        return -1;
    }
    rc = append(&token, 1);
    if (rc < 0)
        return rc;
    s_counters[2]++;
    next_logits(logits);
    return 0;
}

/* sampler.c — see sampler.h.  Build:  cc -O2 -shared -fPIC -o libsampler.so sampler.c -lm */
#include "sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double  l;
    int32_t id;
} cand_t;

struct smp_state {
    int32_t  n;
    uint64_t rng;
    double  *l;       /* working logits */
    double  *e;       /* exp(l - max) of the candidate list */
    cand_t  *c;       /* candidate list */
    int32_t *cnt;     /* occurrences in `recent` (zero between calls) */
    int32_t *seen;    /* distinct ids of `recent` */
    uint8_t *brk;     /* DRY sequence breakers */
    int32_t *dml;     /* DRY: longest match per candidate (zero between calls) */
    int32_t *dseen;   /* DRY: candidates with a match, in first-seen order */
};

const char *smp_version(void)
{
    return "kv260-sampler 2 (dry)";
}

smp_state *smp_new(int32_t n_vocab)
{
    smp_state *s;

    if (n_vocab <= 0)
        return NULL;
    s = (smp_state *)calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->n = n_vocab;
    s->l = (double *)malloc(sizeof(double) * (size_t)n_vocab);
    s->e = (double *)malloc(sizeof(double) * (size_t)n_vocab);
    s->c = (cand_t *)malloc(sizeof(cand_t) * (size_t)n_vocab);
    s->cnt = (int32_t *)calloc((size_t)n_vocab, sizeof(int32_t));
    s->seen = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_vocab);
    s->brk = (uint8_t *)calloc((size_t)n_vocab, 1);
    s->dml = (int32_t *)calloc((size_t)n_vocab, sizeof(int32_t));
    s->dseen = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_vocab);
    if (!s->l || !s->e || !s->c || !s->cnt || !s->seen || !s->brk || !s->dml || !s->dseen) {
        smp_free(s);
        return NULL;
    }
    smp_seed(s, 0);
    return s;
}

void smp_free(smp_state *s)
{
    if (!s)
        return;
    free(s->l);
    free(s->e);
    free(s->c);
    free(s->cnt);
    free(s->seen);
    free(s->brk);
    free(s->dml);
    free(s->dseen);
    free(s);
}

void smp_seed(smp_state *s, uint64_t seed)
{
    s->rng = seed;
}

uint64_t smp_rng_next(smp_state *s)
{                                                   /* splitmix64 */
    uint64_t z = (s->rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double smp_rng_uniform(smp_state *s)
{
    return (double)(smp_rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* a before b in "descending logit, then ascending id" order */
static int before(const cand_t *a, const cand_t *b)
{
    return a->l > b->l || (a->l == b->l && a->id < b->id);
}

static int cmp_desc(const void *pa, const void *pb)
{
    const cand_t *a = (const cand_t *)pa, *b = (const cand_t *)pb;
    return before(a, b) ? -1 : (before(b, a) ? 1 : 0);
}

/* min-heap on "before": the root is the candidate that comes last */
static void sift_down(cand_t *h, int32_t k, int32_t i)
{
    for (;;) {
        int32_t lo = i, a = 2 * i + 1, b = a + 1;
        if (a < k && before(&h[lo], &h[a]))
            lo = a;
        if (b < k && before(&h[lo], &h[b]))
            lo = b;
        if (lo == i)
            return;
        cand_t t = h[i];
        h[i] = h[lo];
        h[lo] = t;
        i = lo;
    }
}

static void sift_up(cand_t *h, int32_t i)
{
    while (i > 0) {
        int32_t p = (i - 1) / 2;
        if (!before(&h[p], &h[i]))
            return;
        cand_t t = h[i];
        h[i] = h[p];
        h[p] = t;
        i = p;
    }
}

static int bad_args(const smp_state *s, const float *logits, const int32_t *recent,
                    int32_t n_recent, const int32_t *hist, int32_t n_hist, const smp_params *p)
{
    return !s || !logits || !p || n_recent < 0 || (n_recent > 0 && !recent)
        || n_hist < 0 || (n_hist > 0 && !hist);
}

int32_t smp_set_breakers(smp_state *s, const int32_t *ids, int32_t n)
{
    int32_t i, k = 0;

    if (!s || n < 0 || (n > 0 && !ids))
        return -1;
    memset(s->brk, 0, (size_t)s->n);
    for (i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < s->n && !s->brk[ids[i]]) {
            s->brk[ids[i]] = 1;
            k++;
        }
    return k;
}

/* step 1b: DRY over hist (see sampler.h) */
static void dry(smp_state *s, double *l, const int32_t *hist, int32_t n_hist, const smp_params *p)
{
    const int32_t n = s->n;
    int32_t       i, nd = 0, last, maxm, al;

    if (!(p->dry_multiplier > 0.0) || n_hist < 2)
        return;
    last = hist[n_hist - 1];
    if (last < 0 || last >= n || s->brk[last])
        return;
    maxm = p->dry_max_match > 0 ? p->dry_max_match : 50;
    al = p->dry_allowed_length > 0 ? p->dry_allowed_length : 1;
    for (i = 0; i + 1 < n_hist; i++) {
        int32_t nx, len = 1;
        if (hist[i] != last)
            continue;
        nx = hist[i + 1];
        if (nx < 0 || nx >= n || s->brk[nx])
            continue;
        while (len < maxm) {
            int32_t j = i - len, prev;
            if (j < 0)
                break;
            prev = hist[n_hist - 1 - len];
            if (hist[j] != prev || prev < 0 || prev >= n || s->brk[prev])
                break;
            len++;
        }
        if (s->dml[nx] == 0)
            s->dseen[nd++] = nx;
        if (len > s->dml[nx])
            s->dml[nx] = len;
    }
    for (i = 0; i < nd; i++) {
        int32_t t = s->dseen[i], len = s->dml[t];
        if (len >= al)
            l[t] = l[t] - p->dry_multiplier * pow(p->dry_base, (double)(len - al));
        s->dml[t] = 0;
    }
}

/* steps 1-6; returns the candidate count (list in s->c / s->e, total mass in *z),
 * or -(id + 1) for a greedy pick */
static int32_t filter(smp_state *s, const float *logits, const int32_t *recent, int32_t n_recent,
                      const int32_t *hist, int32_t n_hist, const smp_params *p, double *z)
{
    const int32_t n = s->n;
    double       *l = s->l, *e = s->e, m, acc;
    cand_t       *c = s->c;
    int32_t       i, nc, n_seen = 0, k = p->top_k;
    int           greedy, sorted = 0;

    for (i = 0; i < n; i++) {
        double v = (double)logits[i];
        l[i] = isnan(v) ? -INFINITY : v;
    }
    /* 1. penalties */
    if ((p->repetition_penalty > 0.0 && p->repetition_penalty != 1.0) || p->presence_penalty != 0.0
        || p->frequency_penalty != 0.0) {
        for (i = 0; i < n_recent; i++) {
            int32_t t = recent[i];
            if (t < 0 || t >= n)
                continue;
            if (s->cnt[t]++ == 0)
                s->seen[n_seen++] = t;
        }
        for (i = 0; i < n_seen; i++) {
            int32_t t = s->seen[i];
            double  v = l[t];
            if (p->repetition_penalty > 0.0 && p->repetition_penalty != 1.0)
                v = v > 0.0 ? v / p->repetition_penalty : v * p->repetition_penalty;
            v -= p->presence_penalty + p->frequency_penalty * (double)s->cnt[t];
            l[t] = v;
            s->cnt[t] = 0;
        }
    }
    /* 1b. DRY */
    dry(s, l, hist, n_hist, p);
    /* 2. greedy */
    greedy = !(p->temperature > 0.0) || k == 1;
    if (greedy) {
        int32_t best = 0;
        for (i = 1; i < n; i++)
            if (l[i] > l[best])
                best = i;
        return -(best + 1);
    }
    /* 3. temperature */
    for (i = 0; i < n; i++)
        l[i] = l[i] / p->temperature;
    /* 4. top-k */
    if (k > 0 && k < n) {
        for (i = 0; i < n; i++) {
            cand_t x;
            x.l = l[i];
            x.id = i;
            if (i < k) {
                c[i] = x;
                sift_up(c, i);
            } else if (before(&x, &c[0])) {
                c[0] = x;
                sift_down(c, k, 0);
            }
        }
        qsort(c, (size_t)k, sizeof(cand_t), cmp_desc);
        nc = k;
        sorted = 1;
        m = c[0].l;
    } else {
        nc = n;
        m = l[0];
        for (i = 0; i < n; i++) {
            c[i].l = l[i];
            c[i].id = i;
            if (l[i] > m)
                m = l[i];
        }
    }
    /* 5. softmax numerators */
    acc = 0.0;
    for (i = 0; i < nc; i++) {
        e[i] = exp(c[i].l - m);
        acc += e[i];
    }
    /* 6. top-p */
    if (p->top_p > 0.0 && p->top_p < 1.0) {
        double target = p->top_p * acc;
        if (!sorted) {
            double  thr = (1.0 - p->top_p) * acc / (double)n, kept = 0.0;
            int32_t nk = 0;
            for (i = 0; i < nc; i++)
                if (e[i] >= thr)
                    c[nk++] = c[i];
            qsort(c, (size_t)nk, sizeof(cand_t), cmp_desc);
            for (i = 0; i < nk; i++) {
                e[i] = exp(c[i].l - m);
                kept += e[i];
            }
            if (kept < target) {                    /* rounding: sort everything */
                for (i = 0; i < n; i++) {
                    c[i].l = l[i];
                    c[i].id = i;
                }
                qsort(c, (size_t)n, sizeof(cand_t), cmp_desc);
                for (i = 0; i < n; i++)
                    e[i] = exp(c[i].l - m);
                nk = n;
            }
            nc = nk;
        }
        acc = 0.0;
        for (i = 0; i < nc; i++) {
            acc += e[i];
            if (acc >= target) {
                i++;
                break;
            }
        }
        nc = i;
    }
    *z = acc;
    return nc;
}

int32_t smp_sample_ex(smp_state *s, const float *logits, const int32_t *recent, int32_t n_recent,
                      const int32_t *hist, int32_t n_hist, const smp_params *p)
{
    double  z, target, acc = 0.0;
    int32_t nc, i;

    if (bad_args(s, logits, recent, n_recent, hist, n_hist, p))
        return -1;
    nc = filter(s, logits, recent, n_recent, hist, n_hist, p, &z);
    if (nc < 0)
        return -nc - 1;
    target = smp_rng_uniform(s) * z;
    for (i = 0; i < nc; i++) {
        acc += s->e[i];
        if (acc > target)
            return s->c[i].id;
    }
    return s->c[nc - 1].id;
}

int32_t smp_sample(smp_state *s, const float *logits, const int32_t *recent, int32_t n_recent,
                   const smp_params *p)
{
    return smp_sample_ex(s, logits, recent, n_recent, recent, n_recent, p);
}

int32_t smp_candidates_ex(smp_state *s, const float *logits, const int32_t *recent, int32_t n_recent,
                          const int32_t *hist, int32_t n_hist, const smp_params *p,
                          int32_t *out_ids, double *out_probs, int32_t max_out)
{
    double  z;
    int32_t nc, i;

    if (bad_args(s, logits, recent, n_recent, hist, n_hist, p) || max_out < 0
        || (max_out > 0 && (!out_ids || !out_probs)))
        return -1;
    nc = filter(s, logits, recent, n_recent, hist, n_hist, p, &z);
    if (nc < 0) {
        if (max_out > 0) {
            out_ids[0] = -nc - 1;
            out_probs[0] = 1.0;
        }
        return 1;
    }
    for (i = 0; i < nc && i < max_out; i++) {
        out_ids[i] = s->c[i].id;
        out_probs[i] = s->e[i] / z;
    }
    return nc;
}

int32_t smp_candidates(smp_state *s, const float *logits, const int32_t *recent, int32_t n_recent,
                       const smp_params *p, int32_t *out_ids, double *out_probs, int32_t max_out)
{
    return smp_candidates_ex(s, logits, recent, n_recent, recent, n_recent, p, out_ids, out_probs,
                             max_out);
}

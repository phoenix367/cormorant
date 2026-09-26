/* sampler.h — next-token sampling for the KV260 chat server (libsampler.so).
 *
 * Operates on the float logits the decoder library returns (llm_decode /
 * llm_prefill, doc/CHAT_PLAN.md §11).  Everything is computed in double in a
 * fixed order, so demo/chat/sampler.py's pure-Python fallback gives the same
 * token for the same logits, parameters and seed (same libm).
 *
 * Pipeline (the order of Hugging Face generate's logits processors):
 *   1. penalties over `recent` (the caller's window of recent token ids):
 *        repetition_penalty r (CTRL / HF): l = l > 0 ? l / r : l * r, once per
 *          distinct token;
 *        presence_penalty a, frequency_penalty f (OpenAI):
 *          l -= a + f * count   for every token with count > 0;
 *   2. temperature <= 0 or top_k == 1: greedy — the first index of the maximum;
 *   3. l /= temperature;
 *   4. top_k > 0: keep the k largest (ties: lower index first), in descending
 *      order; otherwise all tokens in index order;
 *   5. e = exp(l - max), Z = sum of e in candidate order;
 *   6. 0 < top_p < 1: nucleus — candidates in descending order (ties: lower
 *      index first), the shortest prefix whose mass reaches top_p * Z (at
 *      least one token); Z is re-summed over it;
 *   7. u = uniform [0, 1) from the RNG, pick the first candidate at which the
 *      running sum of e exceeds u * Z (the last one if rounding leaves none).
 *
 * RNG: splitmix64 (one 64-bit state; seeded directly with the request's
 * seed), uniform = (next >> 11) * 2^-53.  Deterministic for a given seed.
 */
#ifndef KV260_SAMPLER_H
#define KV260_SAMPLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct smp_params {
    double  temperature;        /* <= 0: greedy */
    double  top_p;              /* <= 0 or >= 1: off */
    double  repetition_penalty; /* 1 (or <= 0): off */
    double  presence_penalty;   /* 0: off */
    double  frequency_penalty;  /* 0: off */
    int32_t top_k;              /* <= 0: off; 1: greedy */
    int32_t reserved;
} smp_params;

typedef struct smp_state smp_state;

smp_state *smp_new(int32_t n_vocab);          /* NULL on allocation failure */
void       smp_free(smp_state *s);
void       smp_seed(smp_state *s, uint64_t seed);
uint64_t   smp_rng_next(smp_state *s);        /* the raw generator (tests) */
double     smp_rng_uniform(smp_state *s);     /* [0, 1) */

/* The next token; < 0 on bad arguments. */
int32_t    smp_sample(smp_state *s, const float *logits, const int32_t *recent, int32_t n_recent,
                      const smp_params *p);

/* The candidate set after steps 1-6 (no RNG draw): ids and probabilities
 * (normalised over the set) in the order step 7 walks them.  Returns the set
 * size (at most max_out are written), 1 for greedy, < 0 on bad arguments. */
int32_t    smp_candidates(smp_state *s, const float *logits, const int32_t *recent,
                          int32_t n_recent, const smp_params *p, int32_t *out_ids,
                          double *out_probs, int32_t max_out);

const char *smp_version(void);

#ifdef __cplusplus
}
#endif
#endif

/*
 * asr_api.h — libmoonshine_<size>.so: Moonshine speech to text on the KV260
 * (doc/plans/ASR_PLAN.md §5).  16 kHz mono int16 PCM in, token ids out (the
 * caller decodes them with the model's tokenizer.json, which the generator
 * copies to weights/tokenizer.json).
 *
 * Built from the generated multi-entry inference project (entries
 * "encode_<T>" — T = 128 / 256 / 512 / 1024 encoder frames — and "decode")
 * by demo/asr/scripts/generate_asr_project.py.  asr_transcribe() splits audio
 * longer than asr_max_samples() at quiet points, encodes each part in the
 * smallest bucket that holds it and decodes greedily (until </s> or 6.5
 * tokens per second of audio); demo/asr/scripts/moonshine_q.py
 * transcribe_long is the specification, which it equals bit for bit.
 *
 * Every int function returns >= 0 on success, < 0 on error
 * (asr_last_error() says why).  Not thread-safe: one call at a time.
 */
#ifndef ASR_API_H
#define ASR_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open: DMA pool + weights + the kernel drivers.  weights_dir = the
 * directory holding weights/<name>.dat, or NULL for the one the library was
 * built for (asr_weights_dir()).  A second asr_open() without asr_close() is
 * a no-op. */
int         asr_open(const char *weights_dir);
void        asr_close(void);
const char *asr_last_error(void);

int         asr_sample_rate(void);       /* 16000 */
int         asr_max_samples(void);       /* one encoder pass: 384 x the largest bucket + 511 (24.6 s) */
int         asr_num_buckets(void);       /* encode_<T> entries ... */
int         asr_bucket(int i);           /* ... their frames, ascending */
int         asr_bos(void);
int         asr_eos(void);
int         asr_vocab(void);

/* Transcribe n samples (any length >= 1): writes at most max_ids token ids
 * (without <s> / </s>) and returns their count (the transcript may be longer
 * than max_ids: the count returned is then max_ids and asr_last_stats()
 * says how many there were).  Audio shorter than 0.154 s has no token. */
int         asr_transcribe(const int16_t *pcm, int n, int32_t *ids, int max_ids);

/* The parts asr_transcribe() splits n samples into: writes up to max_parts
 * (start, length) pairs and returns the count. */
int         asr_chunks(const int16_t *pcm, int n, int32_t *starts, int32_t *lens, int max_parts);

/* The steps, for tests and diagnostics: encode 895 <= n <= asr_max_samples()
 * samples (returns the bucket's frames; the cross caches are filled), then
 * greedy steps — the token at position pos -> the next token (the first step:
 * asr_bos() at 0). */
int         asr_encode(const int16_t *pcm, int n);
int         asr_decode_step(int token, int pos);

/* The decoder's scores, to check a phrase against the audio (the phrase fed
 * as the tokens, demo/chat/moonshine_backend.py phrase_decode): after
 * asr_decode_step(), the raw logits of n tokens (int16 at 2^-asr_logit_frac();
 * the step's return value is the first maximum) into logits; returns n.
 * Valid until the next asr_encode() or asr_transcribe(). */
int         asr_last_logits(const int32_t *tokens, int n, int32_t *logits);
int         asr_logit_frac(void);

/* The most decode steps asr_transcribe() runs for a part of n samples (6.5
 * tokens per second of audio, at most the decoder's positions - 1; 0 below
 * 0.154 s). */
int         asr_max_steps(int n);

/* The last asr_transcribe()'s timing (ms) and counts. */
typedef struct {
    double encode_ms, decode_ms, total_ms;
    int    parts, tokens, steps;
} asr_stats_t;
void        asr_last_stats(asr_stats_t *st);

const char *asr_model_name(void);
const char *asr_weights_dir(void);

#ifdef __cplusplus
}
#endif

#endif /* ASR_API_H */

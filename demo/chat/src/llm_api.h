/*
 * llm_api.h — libsmollm2.so: a Llama-family decoder (SmolLM2-135M-Instruct)
 * on the KV260's FPGA kernels.  The C API of doc/CHAT_PLAN.md §11.
 *
 * Built from the generated multi-entry inference project (decode step,
 * prefill buckets, head; one weight pool; the KV cache and the position-0
 * attention sink as persistent states) by
 * demo/chat/scripts/generate_llm_project.py; the chat server (demo/chat/,
 * phase 4) loads it with ctypes.
 *
 * Every int function returns >= 0 on success, < 0 on error
 * (llm_last_error() says why).  Not thread-safe: one call at a time — but
 * calls may come from different threads (no thread-local state).
 *
 * Position 0 of the cache is always <|im_start|> (the precomputed sink):
 * callers pass the conversation's token ids AFTER that leading token, and
 * reuse the cache across turns by llm_truncate() to the common prefix and
 * prefilling only the new tokens.  Logits are the LM head's output
 * converted from its fixed-point exponents to float.  No sampling here.
 */
#ifndef LLM_API_H
#define LLM_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open the model: DMA pool + weights, host tables, KV cache (sink row),
 * kernel drivers.  weights_dir = the directory holding weights/<name>.dat,
 * or NULL for the one the library was built for (INFERENCE_WEIGHTS_DIR,
 * see llm_weights_dir()).  A second llm_open() without llm_close() is a
 * no-op.  llm_open() may follow llm_close() (everything is re-created). */
int         llm_open(const char *weights_dir);

/* Release everything llm_open() created: the DMA pool (CMA), host tables,
 * threads, the UIO mappings of the kernel drivers and the XRT handle. */
void        llm_close(void);

const char *llm_last_error(void);
int         llm_vocab_size(void);                /* 49152 */
int         llm_context_size(void);              /* 1024 (cache positions incl. the sink) */
int         llm_position(void);                  /* positions filled, incl. the sink at 0 */
int         llm_truncate(int n);                 /* keep positions [0, n), n >= 1 (1 = sink only) */
int         llm_prefill(const int32_t *tokens, int n, float *logits);
                  /* append n >= 1 tokens (the library splits over its prefill buckets);
                     writes the next-token logits after the last one (vocab floats) */
int         llm_decode(int32_t token, float *logits);
                  /* append one token; next-token logits */

/* Extras (not in the §11 contract; used by llm_bench and diagnostics). */
const char *llm_model_name(void);                /* e.g. "smollm2-135m-instruct" */
const char *llm_weights_dir(void);               /* INFERENCE_WEIGHTS_DIR of the build */
int         llm_num_buckets(void);               /* prefill buckets (rows per call) ... */
int         llm_bucket(int i);                   /* ... ascending */

#ifdef __cplusplus
}
#endif

#endif /* LLM_API_H */

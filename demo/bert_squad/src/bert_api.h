/*
 * bert_api.h — small C API over the generated BERT-SQuAD inference project.
 *
 * Compiled into the board-side runner (squad_bench) and into the shared
 * library libbert_squad.so that the KV260 chat server (demo/chat/) loads
 * with ctypes.  It owns what squad_bench used to do by hand: inference_init()
 * with the configured UIO instances, the input / output buffers, filling the
 * integer inputs as raw int16 and reading the Q8.8 logits back as raw bits.
 * The generated inference API (inference.h) is unchanged.
 *
 * Not thread-safe: one caller at a time (the FPGA runs one inference at a
 * time anyway).  bert_open() and bert_close() are process-wide.
 */
#ifndef BERT_API_H
#define BERT_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open the model: DMA pool, weights, kernel instances, I/O buffers.
 *   weights_dir — directory holding weights/<name>.dat, or NULL for the one
 *                 the project was built for (bert_weights_dir()).  The
 *                 generated code resolves weight files at compile time
 *                 (INFERENCE_WEIGHTS_DIR), so a different directory is
 *                 accepted only when the build used a relative one, in
 *                 which case bert_open() chdir()s into weights_dir while
 *                 the weights load (call it before starting other threads).
 * Returns 0, or a negative code (bert_last_error() says why). */
int bert_open(const char *weights_dir);

/* One inference over one BERT_SEQ_LEN-token window.
 *   ids, seg, mask       — input_ids / segment_ids / input_mask (int16)
 *   start_logits,
 *   end_logits           — out: raw ap_fixed<16,8> bits (value = bits / 256)
 * Returns 0, or -1 when the model is not open. */
int bert_run(const int16_t *ids, const int16_t *seg, const int16_t *mask,
             int16_t *start_logits, int16_t *end_logits);

/* bert_run() with the pass-through unique id: uid goes into the
 * unique_ids input (when the model has one) and *uid_out (may be NULL)
 * receives the unique_ids output, or BERT_NO_UID when the model has none. */
#define BERT_NO_UID  (-65536)   /* outside the int16 range */
int bert_run_ex(const int16_t *ids, const int16_t *seg, const int16_t *mask,
                int uid, int16_t *start_logits, int16_t *end_logits, int *uid_out);

/* Release the buffers and the DMA pool (inference_deinit()). */
void bert_close(void);

unsigned    bert_seq_len(void);       /* tokens per window (256)            */
const char *bert_model_name(void);    /* e.g. "bertsquad-12-simplified"     */
const char *bert_weights_dir(void);   /* INFERENCE_WEIGHTS_DIR of the build */
const char *bert_last_error(void);    /* "" when there was none             */

#ifdef __cplusplus
}
#endif

#endif /* BERT_API_H */

/*
 * tts_api.h — libpiper_tts.so: the acoustic back end of Piper (VITS) text to
 * speech on the KV260 (doc/plans/TTS_PLAN.md §4): the reverse flow and the
 * HiFi-GAN decoder on ConvKernel + host ops, in chunks of 128 frames.
 *
 * Built from the generated single-entry inference project (entry "chunk")
 * by demo/tts/scripts/generate_tts_project.py.  The caller (the chat
 * server) runs the front end in numpy — phonemes -> ids, the text encoder,
 * the duration predictor, the length regulator and the noise — and hands
 * over z_p, the flow's input: frames x 192 values in channel-major order
 * (zp[c * frames + t]).  The library returns 16-bit PCM at
 * tts_sample_rate(), tts_hop() samples per frame.
 *
 * Chunk k covers the utterance frames [128k - 64, 128k + 192) and yields the
 * samples of frames [128k, 128k + 128): the 64 frames of context on either
 * side cover the receptive field of the flow and the decoder, so the
 * stitched chunks equal one long pass bit for bit (demo/tts/scripts/
 * piper_vits.py synthesize_chunked is the specification).  A caller streams
 * with tts_synthesize_chunk(k) for k = 0 .. tts_num_chunks(frames) - 1.
 *
 * The text encoder runs on the FPGA too (tts_encode, TTS_PLAN §6): phoneme
 * ids -> x, m_p, logs_p [192][n]; the stochastic duration predictor runs in
 * C on the host (tts_duration, TTS_PLAN §7): x + the noise -> logw.  The
 * caller keeps the alignment and the noise (piper_vits.front_end with
 * encoder=... and duration=...).
 *
 * Every int function returns >= 0 on success, < 0 on error
 * (tts_last_error() says why).  Not thread-safe: one call at a time.
 */
#ifndef TTS_API_H
#define TTS_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open: DMA pool + weights, kernel driver.  weights_dir = the directory
 * holding weights/<name>.dat, or NULL for the one the library was built for
 * (INFERENCE_WEIGHTS_DIR, see tts_weights_dir()).  A second tts_open()
 * without tts_close() is a no-op. */
int         tts_open(const char *weights_dir);
void        tts_close(void);
const char *tts_last_error(void);

int         tts_channels(void);          /* 192: z_p rows */
int         tts_sample_rate(void);       /* 22050 */
int         tts_hop(void);               /* 256 samples per frame */
int         tts_chunk_frames(void);      /* 128 frames (32768 samples) per chunk */
int         tts_num_chunks(int frames);  /* ceil(frames / 128) */

/* The text encoder on n phoneme ids (1 <= n <= tts_max_ids()): x, m_p and
 * logs_p [192][n] float32, channel-major (any may be NULL).  The ids are
 * padded to the smallest bucket >= n (the result does not depend on it);
 * returns that bucket. */
int         tts_encode(const int32_t *ids, int n, float *x, float *m_p, float *logs_p);
int         tts_max_ids(void);           /* 400 */

/* The duration predictor on x [192][n] (tts_encode's) and the noise z [2][n]
 * (standard normal * noise_w): logw [n] (float64; 4 host threads).  Bit for
 * bit piper_vits.duration_predictor_seq.  < 0 without weights/dp.dat. */
int         tts_duration(const float *x, int n, const double *z, double *logw);
int         tts_num_buckets(void);       /* encode_<T> entries ... */
int         tts_bucket(int i);           /* ... ascending */

/* Chunk k of an utterance of `frames` frames: writes
 * min(128, frames - 128k) * 256 samples to pcm and returns their count. */
int         tts_synthesize_chunk(const float *zp, int frames, int k, int16_t *pcm);

/* The whole utterance: frames * 256 samples. */
int         tts_synthesize(const float *zp, int frames, int16_t *pcm);

/* Extras (tts_bench and diagnostics). */
const char *tts_model_name(void);        /* "piper-lessac-medium" */
const char *tts_weights_dir(void);       /* INFERENCE_WEIGHTS_DIR of the build */

/* The chunk entry itself: zp [192][256] (zero outside [lo, hi)), the
 * utterance's frames in chunk coordinates [lo, hi); pcm: 192 * 256 samples
 * of the decoder window, flow frames [32, 224). */
int         tts_chunk_raw(const float *zp, int lo, int hi, int16_t *pcm);

#ifdef __cplusplus
}
#endif

#endif /* TTS_API_H */

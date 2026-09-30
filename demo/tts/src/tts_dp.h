/*
 * tts_dp.h — Piper's stochastic duration predictor in C, float64, on the
 * host (doc/plans/TTS_PLAN.md §7): the text encoder's x and the noise z ->
 * logw, bit for bit what demo/tts/scripts/piper_vits.py
 * duration_predictor_seq computes (every sum left to right, exp / log1p
 * from libm, no fused multiply-add: build with -ffp-contract=off).
 *
 * The weights come from weights/dp.dat (float32, piper_vits.dp_tensors()
 * order, written by generate_tts_project.py).  One instance per process.
 */
#ifndef TTS_DP_H
#define TTS_DP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Load dp.dat (expect_floats values); 0 on success. */
int  tts_dp_load(const char *path, size_t expect_floats);
void tts_dp_free(void);
int  tts_dp_loaded(void);

/* x [192][n] (channel-major float32), z [2][n] (the noise * noise_w) ->
 * logw [n]; `threads` worker threads (1 = the caller only).  0 on success. */
int  tts_dp_run(const float *x, int n, const double *z, double *logw, int threads);

#ifdef __cplusplus
}
#endif

#endif /* TTS_DP_H */

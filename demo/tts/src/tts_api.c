/*
 * tts_api.c — see tts_api.h.  The chunk geometry is src/piper.py's
 * (FLOW_FRAMES 256, OUT_FRAMES 128, DEC_OFF 32, HOP 256), checked against
 * the generated entry's buffer sizes at compile time; TTS_API_WEIGHTS_DIR,
 * TTS_MODEL_NAME and TTS_SAMPLE_RATE come from the CMake targets added by
 * demo/tts/scripts/generate_tts_project.py.
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

#include "tts_api.h"
#include "inference.h"

#ifndef TTS_API_WEIGHTS_DIR
#  define TTS_API_WEIGHTS_DIR "."
#endif
#ifndef TTS_MODEL_NAME
#  define TTS_MODEL_NAME "piper"
#endif
#ifndef TTS_SAMPLE_RATE
#  define TTS_SAMPLE_RATE 22050
#endif

#define TTS_CH          192
#define TTS_FLOW_FRAMES 256                                   /* frames per chunk pass */
#define TTS_OUT_FRAMES  128                                   /* valid frames per chunk */
#define TTS_OUT_OFF     ((TTS_FLOW_FRAMES - TTS_OUT_FRAMES) / 2)   /* 64 */
#define TTS_DEC_OFF     32                                    /* decoder window start */
#define TTS_HOP         256

_Static_assert(INFERENCE_ZP_SIZE == TTS_CH * TTS_FLOW_FRAMES, "zp is [192][256]");
_Static_assert(INFERENCE_PCM_SIZE == (TTS_FLOW_FRAMES - 2 * TTS_DEC_OFF) * TTS_HOP,
               "pcm is the decoder window");

static int     s_open;
static char    s_err[512];
static float   s_zp[TTS_CH * TTS_FLOW_FRAMES];
static int16_t s_pcm[INFERENCE_PCM_SIZE];

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

int tts_open(const char *weights_dir)
{
    int rc, cwd = -1;

    s_err[0] = '\0';
    if (s_open)
        return 0;
    if (weights_dir && *weights_dir && !same_dir(weights_dir, TTS_API_WEIGHTS_DIR)) {
        if (TTS_API_WEIGHTS_DIR[0] == '/') {
            set_err("this build reads its weights from %s/weights (INFERENCE_WEIGHTS_DIR); "
                    "rebuild with -DINFERENCE_WEIGHTS_DIR=%s to use %s",
                    TTS_API_WEIGHTS_DIR, weights_dir, weights_dir);
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
    rc = inference_init(INFERENCE_CONVKERNEL_INSTANCE);
    if (cwd >= 0) {
        if (fchdir(cwd) != 0)
            fprintf(stderr, "tts_api: warning: cannot restore the working directory: %s\n",
                    strerror(errno));
        close(cwd);
    }
    if (rc != 0) {
        set_err("inference_init() failed (%d): check the UIO device, CMA (CmaFree) and "
                "the weights under %s/weights", rc,
                weights_dir && *weights_dir ? weights_dir : TTS_API_WEIGHTS_DIR);
        return -4;
    }
    s_open = 1;
    return 0;
}

void tts_close(void)
{
    if (!s_open)
        return;
    inference_deinit();
    s_open = 0;
}

const char *tts_last_error(void)  { return s_err; }
int         tts_channels(void)    { return TTS_CH; }
int         tts_sample_rate(void) { return TTS_SAMPLE_RATE; }
int         tts_hop(void)         { return TTS_HOP; }
int         tts_chunk_frames(void){ return TTS_OUT_FRAMES; }
const char *tts_model_name(void)  { return TTS_MODEL_NAME; }
const char *tts_weights_dir(void) { return TTS_API_WEIGHTS_DIR; }

int tts_num_chunks(int frames)
{
    return frames > 0 ? (frames + TTS_OUT_FRAMES - 1) / TTS_OUT_FRAMES : 0;
}

int tts_chunk_raw(const float *zp, int lo, int hi, int16_t *pcm)
{
    int32_t l = lo, h = hi;
    if (!s_open) {
        set_err("tts_open() first");
        return -1;
    }
    inference_run_chunk(zp, &l, &h, pcm);
    return (int)INFERENCE_PCM_SIZE;
}

int tts_synthesize_chunk(const float *zp, int frames, int k, int16_t *pcm)
{
    int start, a, b, c, n;
    if (!s_open) {
        set_err("tts_open() first");
        return -1;
    }
    if (!zp || !pcm || frames <= 0 || k < 0 || k >= tts_num_chunks(frames)) {
        set_err("bad arguments: frames %d, chunk %d", frames, k);
        return -2;
    }
    /* chunk frames [start, start + 256) of the utterance; zero outside it */
    start = k * TTS_OUT_FRAMES - TTS_OUT_OFF;
    a = start > 0 ? start : 0;
    b = start + TTS_FLOW_FRAMES < frames ? start + TTS_FLOW_FRAMES : frames;
    memset(s_zp, 0, sizeof s_zp);
    for (c = 0; c < TTS_CH; c++)
        memcpy(&s_zp[c * TTS_FLOW_FRAMES + (a - start)], &zp[(size_t)c * (size_t)frames + (size_t)a],
               (size_t)(b - a) * sizeof(float));
    tts_chunk_raw(s_zp, -start, frames - start, s_pcm);
    n = frames - k * TTS_OUT_FRAMES;
    n = (n < TTS_OUT_FRAMES ? n : TTS_OUT_FRAMES) * TTS_HOP;
    memcpy(pcm, &s_pcm[(TTS_OUT_OFF - TTS_DEC_OFF) * TTS_HOP], (size_t)n * sizeof(int16_t));
    return n;
}

int tts_synthesize(const float *zp, int frames, int16_t *pcm)
{
    int k, n, total = 0;
    for (k = 0; k < tts_num_chunks(frames); k++) {
        n = tts_synthesize_chunk(zp, frames, k, pcm + total);
        if (n < 0)
            return n;
        total += n;
    }
    return total;
}

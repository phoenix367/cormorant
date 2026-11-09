// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * vjepa2_api.h — libvjepa2.so: the V-JEPA 2 video classifier
 * (facebook/vjepa2-vitl-fpc16-256-ssv2, Something-Something v2's 174 actions) on
 * the KV260 (doc/plans/VJEPA2_PLAN.md).  Decoded video frames in, the class
 * logits out.
 *
 * Built from the generated inference project (the entry "clip") by
 * demo/video_classification/scripts/generate_vjepa2_project.py.
 * demo/video_classification/scripts/vjepa2_q.py (VJEPA2.forward of
 * make_clip with the project's formats) is the specification, which
 * vjepa2_classify() equals bit for bit:
 *
 *   vjepa2_sample_frames()  the clip's 16 frames of a video of n frames (Meta's
 *                           evaluation sampler: over the first 64 at step ~4)
 *   vjepa2_frame()          one frame: the shorter side resized to 292
 *                           (bilinear, antialiased — PIL's filter), the centre
 *                           256 x 256, rounded to uint8
 *   vjepa2_classify_clip()  16 such frames -> the 2048 patch rows -> the encoder,
 *                           the attentive pooler and the classifier -> logits
 *
 * Every int function returns >= 0 on success, < 0 on error
 * (vjepa2_last_error() says why).  Not thread-safe: one call at a time.
 */
#ifndef VJEPA2_API_H
#define VJEPA2_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open: DMA pool + weights + the kernel drivers.  weights_dir = the directory
 * holding weights/<name>.dat, or NULL for the one the library was built for
 * (vjepa2_weights_dir()).  A second vjepa2_open() without vjepa2_close() is a
 * no-op. */
int         vjepa2_open(const char *weights_dir);
void        vjepa2_close(void);
const char *vjepa2_last_error(void);

int         vjepa2_frames(void);         /* frames per clip (16) */
int         vjepa2_crop(void);           /* the model's square frame (256) */
int         vjepa2_short_side(void);     /* the resize's shorter side (292) */
int         vjepa2_classes(void);        /* classes (174) */
const char *vjepa2_label(int c);         /* class c's name, NULL outside [0, classes) */

/* The indices of the clip's vjepa2_frames() frames in a video of n frames (n >= 1). */
int         vjepa2_sample_frames(int n, int *idx);

/* One frame rgb [h][w][3] uint8 -> the model's [S][S][3] (S = vjepa2_crop()). */
int         vjepa2_frame(const uint8_t *rgb, int h, int w, uint8_t *out);

/* A clip of vjepa2_frames() frames [S][S][3] (vjepa2_frame's) -> logits [vjepa2_classes()]. */
int         vjepa2_classify_clip(const uint8_t *clip, float *logits);

/* A decoded video: frames [n][h][w][3] uint8 -> logits [vjepa2_classes()]
 * (vjepa2_sample_frames, vjepa2_frame per selected frame, vjepa2_classify_clip). */
int         vjepa2_classify(const uint8_t *frames, int n, int h, int w, float *logits);

/* The last call's timing (ms). */
typedef struct {
    double prep_ms;                      /* the frames' resize and crop (vjepa2_classify) */
    double run_ms;                       /* the patch rows and the network */
    double total_ms;
} vjepa2_stats_t;
void        vjepa2_last_stats(vjepa2_stats_t *st);

const char *vjepa2_weights_dir(void);

#ifdef __cplusplus
}
#endif

#endif /* VJEPA2_API_H */

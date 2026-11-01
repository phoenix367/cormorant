// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * vla_api.h — libsmolvla.so: SmolVLA (lerobot/smolvla_base) on the KV260
 * (doc/plans/VLA_PLAN.md).  Camera frames, the task's token ids and the
 * robot's state in, a chunk of actions out.
 *
 * Built from the generated multi-entry inference project (entries "vision",
 * "image_<c>", "prefix_<C>" and "expert") by
 * demo/vla/scripts/generate_vla_project.py.  demo/vla/scripts/smolvla_q.py
 * (SmolVLA.sample with the project's formats) is the specification, which
 * vla_act() equals bit for bit:
 *
 *   vla_set_image(c, ...)  each camera: resize with padding to the encoder's
 *                          square (bilinear, the aspect ratio kept, black on
 *                          the left and the top), rounded to uint8; the vision
 *                          encoder -> camera c's 64 image tokens
 *   vla_act(C, ...)        the prefix (cameras 0 .. C-1, the ids, the state
 *                          row) through the VLM's 16 layers -> the caches; then
 *                          the flow: x = noise, ten Euler steps x += -0.1 v(x, t)
 *                          (float32), v from the action expert
 *
 * The state and the actions are the policy's normalised values (the caller
 * applies the dataset's mean / std), padded to vla_state_size() /
 * vla_action_size(); the robot's are the first vla_state_dim() /
 * vla_action_dim().  The ids are the tokenizer's of the task + "\n"
 * (weights/tokenizer.json, SmolVLM2's), at most vla_tok_max().
 *
 * Every int function returns >= 0 on success, < 0 on error
 * (vla_last_error() says why).  Not thread-safe: one call at a time.
 */
#ifndef VLA_API_H
#define VLA_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open: DMA pool + weights + the kernel drivers.  weights_dir = the
 * directory holding weights/<name>.dat, or NULL for the one the library was
 * built for (vla_weights_dir()).  A second vla_open() without vla_close() is
 * a no-op. */
int         vla_open(const char *weights_dir);
void        vla_close(void);
const char *vla_last_error(void);

int         vla_cameras(void);           /* cameras the library takes (the config's: 3) */
int         vla_image_size(void);        /* the vision encoder's square input (512) */
int         vla_tok_max(void);           /* language ids (48) */
int         vla_state_size(void);        /* the state vector (32, zero padded) */
int         vla_action_size(void);       /* the action vector (32) */
int         vla_state_dim(void);         /* the config's state values (6) */
int         vla_action_dim(void);        /* the config's action values (6) */
int         vla_chunk(void);             /* actions per chunk (50) */
int         vla_steps(void);             /* Euler steps (10) */

/* Camera c's frame: rgb [h][w][3] uint8, any size -> its image tokens (kept
 * until the next vla_set_image(c) or vla_close()). */
int         vla_set_image(int c, const uint8_t *rgb, int h, int w);

/* One chunk over cameras 0 .. n_cameras - 1 (each set by vla_set_image): ids
 * [n_ids] (n_ids <= vla_tok_max()), state [n_state] (n_state <=
 * vla_state_size(), the rest 0), noise [vla_chunk()][vla_action_size()] (x_1,
 * float32: e.g. vla_noise()) -> actions [vla_chunk()][vla_action_size()]. */
int         vla_act(int n_cameras, const int32_t *ids, int n_ids, const float *state, int n_state,
                    const float *noise, float *actions);

/* Standard normal noise [vla_chunk()][vla_action_size()] from a seed
 * (splitmix64 + Box-Muller): any noise works, this one is reproducible. */
void        vla_noise(uint64_t seed, float *noise);

/* The vision encoder's input of a frame: [S][S][3] uint8 (S = vla_image_size()). */
int         vla_resize(const uint8_t *rgb, int h, int w, uint8_t *out);

/* The last calls' timing (ms). */
typedef struct {
    double image_ms;                     /* the last vla_set_image() */
    double prefix_ms, expert_ms, act_ms; /* the last vla_act(): the prefix, the steps, both */
    int    cameras, steps;
} vla_stats_t;
void        vla_last_stats(vla_stats_t *st);

const char *vla_weights_dir(void);

#ifdef __cplusplus
}
#endif

#endif /* VLA_API_H */

// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * inference_ddr_backend.h — internal vtable for DDR-bandwidth backends.
 *
 * Each backend lives in its own translation unit under runtime/ddr/ and
 * exports a single `const ddr_backend_t` symbol that is added to the
 * registry in inference_ddr.c.  The dispatcher walks the registry at
 * init time, picking the first backend whose probe() returns 0.
 *
 * This header is internal — it is copied into the generated project's
 * src/ tree (NOT include/) and must not appear in any public header.
 */
#pragma once

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ddr_backend {
    /* Stable identifier — also the value matched by the
     * INFERENCE_DDR_BACKEND env var when the user wants to force a
     * specific backend. */
    const char *name;

    /* Open whatever device / fd / mapping this backend needs.
     * Return 0 on success, non-zero on any failure.  Backends are
     * expected to populate their own internal failure-reason string
     * which last_error() returns. */
    int          (*probe)(void);

    /* Start counting.  Called once after a successful probe. */
    int          (*start)(void);

    /* Optional periodic sampler — used by backends with narrow
     * counters (e.g. the 32-bit Xilinx APM byte counters) to fold
     * deltas into a 64-bit accumulator before they wrap.  May be
     * NULL when the backend has wide enough native counters. */
    int          (*sample)(void);

    /* Stop counting and snapshot final values. */
    int          (*stop)(void);

    /* Pull the accumulated read/write byte totals out of the backend.
     * Both pointers are non-NULL.  Return 0 on success. */
    int          (*read_counts)(uint64_t *r_bytes, uint64_t *w_bytes);

    /* Append backend-specific JSON fields (",\"key\":val,…") to f.
     * The dispatcher emits the common fields (available/backend/
     * read_bytes/write_bytes/duration_ns/…) itself.  May be NULL. */
    void         (*describe)(FILE *f);

    /* Most recent failure-reason string — used by the dispatcher when
     * probe() failed, or to annotate the {"available":false} JSON.
     * Must be valid for the lifetime of the program. */
    const char  *(*last_error)(void);

    /* Release any resources acquired in probe()/start().  Called on
     * inference_ddr_deinit() and again before re-probing. */
    void         (*deinit)(void);
} ddr_backend_t;

#ifdef __cplusplus
}
#endif

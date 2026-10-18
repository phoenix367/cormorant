// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * profiler_overlap_harness.c — drives inference_prof through an overlapping
 * BEGIN/END pattern that mirrors what the codegen emits for
 * parallel_two_chains.onnx.
 *
 * The interleaving is structured so that BEGIN(2) (Pool start) sits BEFORE
 * BEGIN(3) and BEGIN(4), and END(2) fires only after END(3).  Under the old
 * single-global profiler, BEGIN(3) clobbers BEGIN(2)'s start time and END(2)
 * would be silently dropped.  With per-layer start storage, all six layers
 * record their own intervals correctly.
 *
 * Built and run by test/test_profiler_overlap.py.
 */

#define INFERENCE_PROFILING 1
#include "inference_prof.h"

#include <stdio.h>
#include <time.h>

static const char *names[6] = {
    "convA", "reluA", "poolA",
    "convB", "reluB", "join",
};

/* Sleep helpers in milliseconds.  nanosleep is in POSIX 199309 (the same
 * feature-test macro inference_prof.c already uses), so no extra
 * _DEFAULT_SOURCE / _BSD_SOURCE plumbing is needed. */
static void ms(unsigned m)
{
    struct timespec req = {
        .tv_sec  = (time_t)(m / 1000u),
        .tv_nsec = (long)((m % 1000u) * 1000000ul),
    };
    nanosleep(&req, NULL);
}

int main(void)
{
    if (inference_prof_init(6, names) != 0) return 1;

    /* [0] convA: ~10 ms */
    inference_prof_begin(0);
    ms(10);
    inference_prof_end(0);

    /* [1] reluA: ~5 ms */
    inference_prof_begin(1);
    ms(5);
    inference_prof_end(1);

    /* [2] poolA STARTS — non-blocking; will be drained much later. */
    inference_prof_begin(2);

    /* [3] convB STARTS — runs ~15 ms in parallel with poolA. */
    inference_prof_begin(3);
    ms(15);
    inference_prof_end(3);

    /* [4] reluB STARTS — runs ~50 ms in parallel with poolA. */
    inference_prof_begin(4);
    ms(50);

    /* poolA finally drained: total wall-clock ~ 65 ms (15 + 50). */
    inference_prof_end(2);

    /* reluB drained: total wall-clock ~ 50 ms. */
    inference_prof_end(4);

    /* [5] join: ~5 ms */
    inference_prof_begin(5);
    ms(5);
    inference_prof_end(5);

    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

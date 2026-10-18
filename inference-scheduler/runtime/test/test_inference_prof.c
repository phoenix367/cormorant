// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * test_inference_prof.c — exercise individual scenarios of the
 *                        inference_prof.{h,c} runtime module.
 *
 * The companion pytest module (test/test_runtime_prof.py) compiles this
 * source against runtime/inference_prof.c, runs the resulting binary
 * once per scenario name, and parses the "LAYERS_JSON: {…}" line that
 * the profiler dumps to stdout.
 *
 * Each scenario lives in its own static t_* function; main() dispatches
 * by name from argv[1].  Adding a new scenario means appending another
 * entry to the TESTS table at the bottom and a matching test method on
 * the Python side.
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference_prof.h"

/* Sleep for ns nanoseconds — used to give begin/end pairs a measurable
 * duration so min/max/mean can be asserted by the Python harness. */
static void busy(long ns)
{
    struct timespec ts;
    ts.tv_sec  = ns / 1000000000L;
    ts.tv_nsec = ns % 1000000000L;
    nanosleep(&ts, NULL);
}

/* ---- scenarios ------------------------------------------------------- */

static int t_basic(void)
{
    static const char *const names[] = { "a", "b" };
    if (inference_prof_init(2u, names) != 0) return 1;
    for (int i = 0; i < 3; ++i) {
        inference_prof_begin(0u); busy(50000L);  inference_prof_end(0u);
        inference_prof_begin(1u); busy(100000L); inference_prof_end(1u);
    }
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

static int t_reset(void)
{
    static const char *const names[] = { "only" };
    if (inference_prof_init(1u, names) != 0) return 1;
    inference_prof_begin(0u); busy(50000L); inference_prof_end(0u);
    inference_prof_reset();
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

static int t_min_first_sample(void)
{
    /* Verifies that the first sample correctly drops min from
     * UINT64_MAX (the initial value) to the actual measurement. */
    static const char *const names[] = { "only" };
    if (inference_prof_init(1u, names) != 0) return 1;
    inference_prof_begin(0u); busy(50000L); inference_prof_end(0u);
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

static int t_escape(void)
{
    /* Names with characters that require JSON escaping.  No begin/end —
     * we only care about the dump_json output here. */
    static const char *const names[] = {
        "quote\"X",
        "back\\Y",
        "tab\tZ",
        "newline\nW",
        /* String-literal concatenation prevents the 'e' that follows from
         * being absorbed into the \x escape (\x consumes every trailing
         * hex digit). */
        "control\x01" "end",
    };
    if (inference_prof_init(5u, names) != 0) return 1;
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

static int t_empty(void)
{
    if (inference_prof_init(0u, NULL) != 0) return 1;
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

static int t_oob(void)
{
    /* Out-of-bounds layer indices must be silently ignored — they may
     * NOT crash, must not affect counters, and must not corrupt the
     * begin/end state for the valid layer. */
    static const char *const names[] = { "only" };
    if (inference_prof_init(1u, names) != 0) return 1;
    inference_prof_begin(99u);
    inference_prof_end  (99u);
    inference_prof_begin(0u); busy(50000L); inference_prof_end(0u);
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

static int t_mismatch(void)
{
    /* begin(0) followed by end(1) must be rejected without recording a
     * sample on either layer.  A subsequent clean begin/end pair must
     * still work, so we add one on layer 1 to verify the profiler is
     * still functional after the mismatch. */
    static const char *const names[] = { "first", "second" };
    if (inference_prof_init(2u, names) != 0) return 1;
    inference_prof_begin(0u);
    busy(50000L);
    inference_prof_end(1u);                /* mismatched — discarded */
    inference_prof_begin(1u); busy(50000L); inference_prof_end(1u);
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

static int t_double_init(void)
{
    /* Calling init() twice must release the first allocation, swap in
     * the new names, and reset all counters.  Old names must not leak
     * into the dump. */
    static const char *const names1[] = { "old0", "old1", "old2" };
    static const char *const names2[] = { "new0", "new1" };
    if (inference_prof_init(3u, names1) != 0) return 1;
    inference_prof_begin(0u); busy(50000L); inference_prof_end(0u);
    if (inference_prof_init(2u, names2) != 0) return 1;
    inference_prof_begin(0u); busy(50000L); inference_prof_end(0u);
    inference_prof_dump_json(stdout);
    inference_prof_deinit();
    return 0;
}

/* ---- dispatch -------------------------------------------------------- */

static const struct {
    const char *name;
    int (*fn)(void);
} TESTS[] = {
    { "basic",            t_basic            },
    { "reset",            t_reset            },
    { "min_first_sample", t_min_first_sample },
    { "escape",           t_escape           },
    { "empty",            t_empty            },
    { "oob",              t_oob              },
    { "mismatch",         t_mismatch         },
    { "double_init",      t_double_init      },
};

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <test-name>\n", argv[0]);
        return 2;
    }
    for (size_t i = 0; i < sizeof(TESTS) / sizeof(TESTS[0]); ++i) {
        if (strcmp(argv[1], TESTS[i].name) == 0) {
            return TESTS[i].fn();
        }
    }
    fprintf(stderr, "unknown test: %s\n", argv[1]);
    return 2;
}

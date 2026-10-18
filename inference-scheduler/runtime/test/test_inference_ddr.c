// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * test_inference_ddr.c — exercise the failure-path behaviour of the
 *                       inference_ddr module on hosts that don't have
 *                       a DDR PMU (i.e. anywhere except an actual
 *                       Zynq UltraScale+ board with the right driver).
 *
 * The success path is only meaningful on real hardware, so the
 * companion pytest module focuses on:
 *   - init() returns non-zero and prints a warning when no PMU is found
 *   - dump_json() emits {"available":false, "reason":"..."}
 *   - start/stop/deinit are silent no-ops after a failed init
 *   - calling init twice re-runs the probe and stays graceful
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference_ddr.h"

static int t_init_failure_dump(void)
{
    /* On a host without a DDR PMU init() should fail gracefully and
     * dump_json() should still emit a parseable JSON line. */
    int rc = inference_ddr_init();
    inference_ddr_dump_json(stdout);
    inference_ddr_deinit();
    /* The test passes regardless of init's return value — what matters
     * is that we got here without crashing.  rc is captured in the
     * trailing marker line so the Python harness can sanity-check it. */
    fprintf(stdout, "INIT_RC: %d\n", rc);
    return 0;
}

static int t_no_init_then_dump(void)
{
    /* Skip init entirely.  start/stop/sample/deinit must be safe;
     * dump_json must report unavailability. */
    inference_ddr_start();
    inference_ddr_sample();
    inference_ddr_stop();
    inference_ddr_dump_json(stdout);
    inference_ddr_deinit();
    return 0;
}

static int t_start_stop_after_failed_init(void)
{
    /* After a failed init, start/sample/stop are no-ops but must not
     * crash; the subsequent dump still reports unavailability. */
    (void)inference_ddr_init();
    inference_ddr_start();
    inference_ddr_sample();
    inference_ddr_stop();
    inference_ddr_start();   /* idempotent */
    inference_ddr_sample();
    inference_ddr_stop();
    inference_ddr_dump_json(stdout);
    inference_ddr_deinit();
    inference_ddr_deinit();  /* second deinit is harmless */
    return 0;
}

static int t_double_init(void)
{
    /* init() re-runs the probe each time and must not leak fds even
     * when the previous attempt failed. */
    (void)inference_ddr_init();
    (void)inference_ddr_init();
    inference_ddr_dump_json(stdout);
    inference_ddr_deinit();
    return 0;
}

static int t_disabled_via_env(void)
{
    /* INFERENCE_DDR_BACKEND=disabled must skip probing entirely AND
     * suppress the warning (the user explicitly opted out). */
    setenv("INFERENCE_DDR_BACKEND", "disabled", 1);
    int rc = inference_ddr_init();
    inference_ddr_dump_json(stdout);
    inference_ddr_deinit();
    unsetenv("INFERENCE_DDR_BACKEND");
    fprintf(stdout, "INIT_RC: %d\n", rc);
    return 0;
}

static int t_unknown_backend(void)
{
    /* INFERENCE_DDR_BACKEND=<unknown> must fail with a clear reason. */
    setenv("INFERENCE_DDR_BACKEND", "no_such_backend", 1);
    int rc = inference_ddr_init();
    inference_ddr_dump_json(stdout);
    inference_ddr_deinit();
    unsetenv("INFERENCE_DDR_BACKEND");
    fprintf(stdout, "INIT_RC: %d\n", rc);
    return 0;
}

static const struct {
    const char *name;
    int (*fn)(void);
} TESTS[] = {
    { "init_failure_dump",            t_init_failure_dump            },
    { "no_init_then_dump",            t_no_init_then_dump            },
    { "start_stop_after_failed_init", t_start_stop_after_failed_init },
    { "double_init",                  t_double_init                  },
    { "disabled_via_env",             t_disabled_via_env             },
    { "unknown_backend",              t_unknown_backend              },
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

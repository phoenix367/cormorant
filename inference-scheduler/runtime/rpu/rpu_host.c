/*
 * rpu_host.c — the gate of a generated project on the Cortex-R5 (RPU_PLAN
 * track B1): a Linux program that runs the project's library twice per
 * inference on the same inputs (the generated test's ramp, offset per run) —
 * first with the A53 driving the kernels (the reference), then with the R5
 * driving them (the firmware replays the recorded calls with
 * interrupt-driven waits, inference_rpu.h) — and compares every output bit
 * for bit.
 *
 *   rpu_host [--runs N] [--no-check] [--ring-kb N] [--auto]
 *
 * Default: the second run is forced onto the R5 (INFERENCE_RPU_MODE_R5, the
 * gate of the R5 path).  --auto lets the project's policy choose instead
 * (INFERENCE_RPU_RECOMMENDED, src/rpu_policy.py): a small model then stays on
 * the A53 in both runs.
 *
 * Exit 0: every run bit-exact (or --no-check); 1: a mismatch or an R5 error;
 * 2: a setup error.  Load / start the firmware with remoteproc first
 * (rpu_board.py does all of it).
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference.h"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* The ramp of test/test_inference.c: p[i] = (Data_t)(i & 0xFFFF), offset by the run. */
static void fill_ramp(Data_t *p, unsigned n, unsigned offset)
{
    unsigned i;
    for (i = 0u; i < n; i++) p[i] = (Data_t)((i + offset) & 0xFFFFu);
}

int main(int argc, char **argv)
{
    unsigned runs = 5u, ring_kb = 0u, i, r, on_r5 = 0u;
    int check = 1, rc, bad_runs = 0, arg, auto_mode = 0;
    static const unsigned sizes[] = INFERENCE_RPU_IO_SIZES;
    const unsigned n_in = INFERENCE_RPU_N_IN, n_io = INFERENCE_RPU_N_IN + INFERENCE_RPU_N_OUT;
    inference_buf_t *io[INFERENCE_RPU_N_IN + INFERENCE_RPU_N_OUT];
    Data_t *ref[INFERENCE_RPU_N_OUT];
    double t_run = 0.0, t_a53 = 0.0, t0;
    inference_rpu_stats_t st;
#if !INFERENCE_RPU_IO_DMA
    fprintf(stderr, "this model has host-memory I/O: rpu_host runs DMA I/O models only\n");
    return 2;
#endif
    for (arg = 1; arg < argc; arg++) {
        if (!strcmp(argv[arg], "--runs") && arg + 1 < argc) runs = (unsigned)atoi(argv[++arg]);
        else if (!strcmp(argv[arg], "--ring-kb") && arg + 1 < argc) ring_kb = (unsigned)atoi(argv[++arg]);
        else if (!strcmp(argv[arg], "--no-check")) check = 0;
        else if (!strcmp(argv[arg], "--auto")) auto_mode = 1;
        else { fprintf(stderr, "usage: rpu_host [--runs N] [--no-check] [--ring-kb N] [--auto]\n"); return 2; }
    }

    rc = inference_init(INFERENCE_RPU_LINUX_INIT_ARGS);        /* its policy attaches when the firmware runs */
    if (rc != 0) { fprintf(stderr, "inference_init failed: %d\n", rc); return 2; }
    if (ring_kb || inference_rpu_mode() < 0) {                  /* a ring size, or INFERENCE_RPU=0 in the environment */
        inference_rpu_detach();
        rc = inference_rpu_attach(ring_kb * 1024u);
        if (rc != 0) {
            fprintf(stderr, "inference_rpu_attach failed: %d (%s)\n", rc,
                    rc == -1 ? "no firmware at the control block: is remoteproc started?" :
                    rc == -2 ? "the firmware is in error or speaks another protocol version" :
                    rc == -3 ? "the firmware has no lane for a kernel of this project" : "no memory");
            return 2;
        }
    }
    printf("policy: %s (%u calls, %u batch(es), %u us predicted, %u us R5 overhead); this gate %s\n",
           INFERENCE_RPU_RECOMMENDED ? "the R5 drives the kernels" : "the A53 drives the kernels",
           INFERENCE_RPU_CALLS, INFERENCE_RPU_BATCHES, INFERENCE_RPU_PREDICTED_US, INFERENCE_RPU_OVERHEAD_US,
           auto_mode ? "follows it" : "forces the R5");
    for (i = 0u; i < n_io; i++) {
        io[i] = inference_buf_alloc(sizes[i]);
        if (!io[i]) { fprintf(stderr, "inference_buf_alloc failed\n"); return 2; }
        memset(inference_buf_ptr(io[i]), 0, (size_t)sizes[i] * INFERENCE_BYTES_PER_ELEM);
    }
    for (i = n_in; i < n_io; i++) {
        ref[i - n_in] = (Data_t *)calloc(sizes[i], INFERENCE_BYTES_PER_ELEM);
        if (!ref[i - n_in]) return 2;
    }
    inference_rpu_stats(&st);
    printf("rpu: firmware lanes 0x%x, interrupt self-test %s, %u batches so far\n",
           st.lanes, st.self_test_ok ? "ok" : "FAILED", st.batches);

    for (r = 0u; r < runs; r++) {
        unsigned bad = 0u;
        for (i = 0u; i < n_in; i++) fill_ramp(inference_buf_ptr(io[i]), sizes[i], r);
        if (check) {                                    /* the reference: the A53 drives the kernels */
            inference_rpu_set_mode(INFERENCE_RPU_MODE_A53);
            t0 = now_s();
            INFERENCE_RPU_RUN(io);
            t_a53 += now_s() - t0;
            for (i = n_in; i < n_io; i++)
                memcpy(ref[i - n_in], inference_buf_ptr(io[i]), (size_t)sizes[i] * INFERENCE_BYTES_PER_ELEM);
            for (i = n_in; i < n_io; i++)
                memset(inference_buf_ptr(io[i]), 0x5A, (size_t)sizes[i] * INFERENCE_BYTES_PER_ELEM);
        }
        inference_rpu_set_mode(auto_mode ? INFERENCE_RPU_MODE_AUTO : INFERENCE_RPU_MODE_R5);
        t0 = now_s();
        INFERENCE_RPU_RUN(io);                          /* the R5, or the policy's choice */
        t_run += now_s() - t0;
        if (inference_rpu_active()) on_r5++;
        if (inference_rpu_error()) { fprintf(stderr, "run %u: R5 error %d\n", r, inference_rpu_error()); return 1; }
        if (check) {
            for (i = n_in; i < n_io; i++) {
                const Data_t *a = inference_buf_ptr(io[i]), *b = ref[i - n_in];
                unsigned k;
                for (k = 0u; k < sizes[i]; k++)
                    if (a[k] != b[k]) {
                        if (bad < 8u) fprintf(stderr, "  output %u[%u]: run 0x%04x, A53 0x%04x\n", i - n_in, k, a[k], b[k]);
                        bad++;
                    }
            }
        }
        inference_rpu_stats(&st);
        printf("run %u: %s %.3f ms%s (%u batches, %u calls, %u waits so far)%s\n", r,
               inference_rpu_active() ? "R5" : "A53", (t_run / (r + 1u)) * 1e3, check ? "" : " (no check)",
               st.batches, st.calls, st.waits, check ? (bad ? "  MISMATCH" : "  bit-exact with the A53") : "");
        if (bad) bad_runs++;
    }
    inference_rpu_stats(&st);
    printf("summary: %u runs (%u driven by the R5), measured mean %.3f ms", runs, on_r5, t_run / runs * 1e3);
    if (check) printf(", A53 mean %.3f ms", t_a53 / runs * 1e3);
    printf("; batches %u, calls %u, waits %u; interrupts per lane:", st.batches, st.calls, st.waits);
    for (i = 0u; i < 5u; i++) printf(" %u", st.irq_count[i]);
    printf("; missed:");
    for (i = 0u; i < 5u; i++) printf(" %u", st.irq_missed[i]);
    printf("; interrupt self-test %s; %s\n", st.self_test_ok ? "ok" : "FAILED",
           bad_runs ? "MISMATCH" : (check ? "all runs bit-exact" : "unchecked"));
    inference_deinit();                                 /* detaches */
    return bad_runs ? 1 : 0;
}

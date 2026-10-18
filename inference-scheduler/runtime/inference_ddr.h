// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * inference_ddr.h — whole-run DDR bandwidth counters for generated
 *                   inference projects.
 *
 * Probes /sys/bus/event_source/devices/ for a DDR PMU exposed by the
 * Linux perf subsystem (e.g. xilinx_ddrc on Zynq UltraScale+) and opens
 * read/write event counters via perf_event_open(2).  Counters are
 * snapshotted around the timed measurement window; bandwidth is
 * reported by inference_ddr_dump_json() as a "DDR_JSON: {…}" line that
 * the host harness can parse alongside LAYERS_JSON.
 *
 * This module is opt-in via the same INFERENCE_PROFILING gate as the
 * per-layer profiler — when off, the API is omitted entirely.  When on
 * but no DDR PMU is available (no driver, wrong kernel, missing
 * permission), inference_ddr_init() prints a single warning to stderr
 * and returns non-zero; subsequent calls are silent no-ops, and
 * inference_ddr_dump_json() emits {"available":false,"reason":"…"} so
 * the host can record the diagnostic.
 *
 * Bare-metal target: stubbed; perf counters require Linux.
 *
 * Thread-safety: none.  start/stop pairs are strictly nested.
 */
#pragma once

#include <stdio.h>

#ifndef INFERENCE_PROFILING
#  define INFERENCE_PROFILING 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if INFERENCE_PROFILING

/* Probe sysfs and open perf counters.  Returns 0 on success, non-zero
 * on any failure (PMU absent, kernel CONFIG missing, EACCES, etc.).
 * On failure a one-line warning is written to stderr; the caller
 * should continue without DDR stats. */
int  inference_ddr_init(void);

/* Reset and enable the counters and capture a wall-clock baseline.
 * No-op if init failed.  Idempotent. */
void inference_ddr_start(void);

/* Optional periodic sampler.  Backends with narrow native counters
 * (e.g. the 32-bit Xilinx APM byte counters) need to be polled often
 * enough to fold each 4-GiB-bounded delta into a 64-bit total before
 * the hardware wraps.  Calling this from a per-iteration host loop
 * (e.g. once per inference_run) is sufficient at any reasonable
 * bandwidth.  No-op if the active backend doesn't need it, or if init
 * failed. */
void inference_ddr_sample(void);

/* Disable the counters, read out their final values, and capture the
 * elapsed wall-clock time. */
void inference_ddr_stop(void);

/* Emit one self-contained "DDR_JSON: {…}" line.  Includes a complete
 * snapshot when init succeeded; otherwise reports
 * {"available":false,"reason":"…"} so the host knows why nothing was
 * measured. */
void inference_ddr_dump_json(FILE *f);

/* Close perf fds.  Safe to call repeatedly or after a failed init. */
void inference_ddr_deinit(void);

#endif /* INFERENCE_PROFILING */

#ifdef __cplusplus
}
#endif

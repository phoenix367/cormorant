// Copyright 2025 GradeBuilder SL
// SPDX-License-Identifier: Apache-2.0

/*
 * inference_ddr.c — backend dispatcher for whole-run DDR-bandwidth
 *                  measurement.
 *
 * The dispatcher knows nothing about specific PMUs / register blocks —
 * it walks a registry of backends (see runtime/ddr/<name>.c) at init
 * time, picks the first one whose probe() succeeds, and forwards
 * start/sample/stop/read calls to it.  All the platform-specific code
 * lives in the backends.
 *
 * Backend selection priority:
 *   1. INFERENCE_DDR_BACKEND=<name>  → force one specific backend
 *      INFERENCE_DDR_BACKEND=disabled → skip probing entirely
 *   2. otherwise: walk the registry in declaration order
 *
 * Failure mode: if no backend probes successfully, we print a single
 * warning and dump_json() reports {"available":false,"reason":"…"}.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1

#include "inference_ddr.h"

#if INFERENCE_PROFILING

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

#if defined(__linux__)

#include <stdarg.h>
#include <time.h>

#include "inference_ddr_backend.h"

/* ---------------------------------------------------------------- */
/* Backend registry — most-specific first.  Each backend exports a  */
/* single const ddr_backend_t symbol from its .c file.              */
/* ---------------------------------------------------------------- */

extern const ddr_backend_t inference_ddr_backend_zuplus_apm;

static const ddr_backend_t *const BACKENDS[] = {
    &inference_ddr_backend_zuplus_apm,
    /* Future: &inference_ddr_backend_perf_event,
     *         &inference_ddr_backend_soft_apm,
     *         &inference_ddr_backend_versal_ddrmc,
     */
};

#define N_BACKENDS  (sizeof(BACKENDS) / sizeof(BACKENDS[0]))

/* ---------------------------------------------------------------- */
/* Dispatcher state                                                 */
/* ---------------------------------------------------------------- */

static const ddr_backend_t *g_backend     = NULL;
static int                  g_running     = 0;
static struct timespec      g_t_start;
static uint64_t             g_duration_ns = 0;
static char                 g_init_reason[256] = {0};

static void set_init_reason(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_init_reason, sizeof(g_init_reason), fmt, ap);
    va_end(ap);
}

int inference_ddr_init(void)
{
    inference_ddr_deinit();
    g_init_reason[0] = '\0';

    const char *forced = getenv("INFERENCE_DDR_BACKEND");
    if (forced && strcmp(forced, "disabled") == 0) {
        set_init_reason("disabled by INFERENCE_DDR_BACKEND=disabled");
        /* No warning when explicitly disabled — the user asked for it. */
        return -1;
    }

    const ddr_backend_t *last_tried = NULL;
    for (size_t i = 0; i < N_BACKENDS; ++i) {
        const ddr_backend_t *b = BACKENDS[i];
        if (forced && strcmp(forced, b->name) != 0) continue;
        last_tried = b;
        if (b->probe && b->probe() == 0) {
            g_backend = b;
            return 0;
        }
    }

    if (last_tried && last_tried->last_error) {
        set_init_reason("backend '%s' probe failed: %s",
                        last_tried->name,
                        last_tried->last_error()
                            ? last_tried->last_error()
                            : "(no detail)");
    } else if (forced) {
        set_init_reason("backend '%s' not registered", forced);
    } else {
        set_init_reason("no DDR backend probed successfully");
    }
    fprintf(stderr,
            "inference_ddr: warning — DDR counters unavailable: %s\n",
            g_init_reason);
    return -1;
}

void inference_ddr_start(void)
{
    if (!g_backend || g_running) return;
    if (g_backend->start && g_backend->start() != 0) return;
    clock_gettime(CLOCK_MONOTONIC, &g_t_start);
    g_running = 1;
}

void inference_ddr_sample(void)
{
    if (!g_backend || !g_running) return;
    if (g_backend->sample) g_backend->sample();
}

void inference_ddr_stop(void)
{
    if (!g_backend || !g_running) return;

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    if (g_backend->stop) g_backend->stop();

    g_duration_ns =
        (uint64_t)(t_end.tv_sec  - g_t_start.tv_sec)  * 1000000000ull
      + (uint64_t)(t_end.tv_nsec - g_t_start.tv_nsec);
    g_running = 0;
}

static void escape_json(FILE *f, const char *s)
{
    fputc('"', f);
    if (s) {
        for (const char *p = s; *p; ++p) {
            unsigned char c = (unsigned char)*p;
            switch (c) {
                case '"':  fputs("\\\"", f); break;
                case '\\': fputs("\\\\", f); break;
                case '\n': fputs("\\n",  f); break;
                case '\r': fputs("\\r",  f); break;
                case '\t': fputs("\\t",  f); break;
                default:
                    if (c < 0x20) fprintf(f, "\\u%04x", c);
                    else          fputc((int)c, f);
            }
        }
    }
    fputc('"', f);
}

void inference_ddr_dump_json(FILE *f)
{
    if (!f) return;
    if (!g_backend) {
        fputs("DDR_JSON: {\"available\":false,\"reason\":", f);
        escape_json(f, g_init_reason);
        fputs("}\n", f);
        fflush(f);
        return;
    }

    uint64_t r_bytes = 0, w_bytes = 0;
    if (g_backend->read_counts) {
        g_backend->read_counts(&r_bytes, &w_bytes);
    }

    double dur_s = (g_duration_ns > 0u)
                   ? (double)g_duration_ns / 1.0e9
                   : 0.0;
    double r_gbs = (dur_s > 0.0) ? (double)r_bytes / dur_s / 1.0e9 : 0.0;
    double w_gbs = (dur_s > 0.0) ? (double)w_bytes / dur_s / 1.0e9 : 0.0;

    fputs("DDR_JSON: {\"available\":true,\"backend\":", f);
    escape_json(f, g_backend->name);
    fprintf(f,
            ",\"read_bytes\":%" PRIu64
            ",\"write_bytes\":%" PRIu64
            ",\"duration_ns\":%" PRIu64
            ",\"read_gbs\":%.4f"
            ",\"write_gbs\":%.4f",
            r_bytes, w_bytes, g_duration_ns, r_gbs, w_gbs);
    if (g_backend->describe) g_backend->describe(f);
    fputs("}\n", f);
    fflush(f);
}

void inference_ddr_deinit(void)
{
    if (g_backend && g_backend->deinit) g_backend->deinit();
    g_backend     = NULL;
    g_running     = 0;
    g_duration_ns = 0;
}

#else  /* !__linux__ — bare-metal stub */

int inference_ddr_init(void)
{
    fprintf(stderr,
            "inference_ddr: warning — perf counters require Linux; "
            "DDR stats disabled on this target\n");
    return -1;
}

void inference_ddr_start (void)        { }
void inference_ddr_sample(void)        { }
void inference_ddr_stop  (void)        { }
void inference_ddr_deinit(void)        { }

void inference_ddr_dump_json(FILE *f)
{
    if (!f) return;
    fputs("DDR_JSON: {\"available\":false,"
          "\"reason\":\"bare-metal target — DDR counters unavailable\"}\n",
          f);
    fflush(f);
}

#endif /* __linux__ */

#endif /* INFERENCE_PROFILING */

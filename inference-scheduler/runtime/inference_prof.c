// Copyright 2025 GradeBuilder SL
// Modifications copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * inference_prof.c — implementation of the aggregate per-layer profiler
 *                    declared in inference_prof.h.
 *
 * Compiled only when INFERENCE_PROFILING evaluates to non-zero.  When the
 * macro is 0 (default) the file expands to nothing, so it is safe to add
 * this source unconditionally to a project's CMake target list.
 */

/* clock_gettime / CLOCK_MONOTONIC need POSIX 199309 visible before any
 * system header is pulled in. */
#define _POSIX_C_SOURCE 199309L

#include "inference_prof.h"

#if INFERENCE_PROFILING

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <time.h>
#endif

typedef struct {
    uint64_t calls;
    uint64_t total_ns;
    uint64_t min_ns;
    uint64_t max_ns;
    /* Timestamp of the most-recent PROF_BEGIN for this layer.  PROF_BEGIN/END
     * brackets for different layer indices may overlap freely (which they do
     * under cross-lane kernel parallelism — Conv ‖ Pool ‖ VectorOP), so each
     * layer needs its own start-time slot.  UINT64_MAX = no pending begin. */
    uint64_t begin_ns;
} prof_layer_t;

#define PROF_NO_BEGIN  UINT64_MAX

static prof_layer_t       *g_layers   = NULL;
static const char *const  *g_names    = NULL;
static unsigned            g_n_layers = 0u;

static uint64_t now_ns(void)
{
    struct timespec ts;
#ifdef __linux__
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    ts.tv_sec = 0; ts.tv_nsec = 0;     /* no wall clock on bare metal: the firmware times runs itself */
#endif
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int inference_prof_init(unsigned n_layers, const char *const *names)
{
    inference_prof_deinit();
    if (n_layers == 0u) {
        g_n_layers = 0u;
        g_names    = NULL;
        return 0;
    }
    g_layers = (prof_layer_t *)calloc(n_layers, sizeof(prof_layer_t));
    if (!g_layers) return -1;
    g_n_layers = n_layers;
    g_names    = names;
    inference_prof_reset();
    return 0;
}

void inference_prof_reset(void)
{
    if (!g_layers) return;
    for (unsigned i = 0; i < g_n_layers; ++i) {
        g_layers[i].calls    = 0u;
        g_layers[i].total_ns = 0u;
        g_layers[i].min_ns   = UINT64_MAX;
        g_layers[i].max_ns   = 0u;
        g_layers[i].begin_ns = PROF_NO_BEGIN;
    }
}

void inference_prof_begin(unsigned layer_idx)
{
    if (layer_idx >= g_n_layers) return;
    g_layers[layer_idx].begin_ns = now_ns();
}

void inference_prof_end(unsigned layer_idx)
{
    if (layer_idx >= g_n_layers) return;
    prof_layer_t *L = &g_layers[layer_idx];
    /* Defensive: an end without a matching begin (or a double-end after
     * the bracket was already closed) is silently dropped. */
    if (L->begin_ns == PROF_NO_BEGIN) return;
    uint64_t dt = now_ns() - L->begin_ns;
    L->begin_ns = PROF_NO_BEGIN;
    L->calls    += 1u;
    L->total_ns += dt;
    if (dt < L->min_ns) L->min_ns = dt;
    if (dt > L->max_ns) L->max_ns = dt;
}

static void escape_json(FILE *f, const char *s)
{
    if (!s) { fputs("null", f); return; }
    fputc('"', f);
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
    fputc('"', f);
}

void inference_prof_dump_json(FILE *f)
{
    if (!f) return;
    fputs("LAYERS_JSON: {\"layers\":[", f);
    for (unsigned i = 0; i < g_n_layers; ++i) {
        const prof_layer_t *L = &g_layers[i];
        double total_us = (double)L->total_ns / 1000.0;
        double mean_us  = (L->calls > 0u)
                          ? (total_us / (double)L->calls) : 0.0;
        double min_us   = (L->calls > 0u) ? (double)L->min_ns / 1000.0 : 0.0;
        double max_us   = (double)L->max_ns / 1000.0;
        if (i) fputc(',', f);
        fputs("{\"i\":", f); fprintf(f, "%u", i);
        fputs(",\"name\":", f); escape_json(f, g_names ? g_names[i] : NULL);
        fprintf(f,
                ",\"calls\":%llu,\"mean_us\":%.3f,\"min_us\":%.3f,"
                "\"max_us\":%.3f,\"total_us\":%.3f}",
                (unsigned long long)L->calls,
                mean_us, min_us, max_us, total_us);
    }
    fputs("]}\n", f);
    fflush(f);
}

void inference_prof_deinit(void)
{
    free(g_layers);
    g_layers   = NULL;
    g_names    = NULL;
    g_n_layers = 0u;
}

#else

/* INFERENCE_PROFILING == 0 — empty translation unit.  Provide a dummy
 * symbol so the linker has something to chew on when this file is the
 * only member of a static library archive on some toolchains. */
extern const int inference_prof_disabled;
const int inference_prof_disabled = 0;

#endif /* INFERENCE_PROFILING */

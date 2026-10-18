// Copyright 2025 GradeBuilder SL
// Modifications copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * ddr/zuplus_apm.c — ZU+ MPSoC AXI Performance Monitor backend.
 *
 * Talks to one of the four built-in APMs in Zynq UltraScale+ MPSoC
 * via /dev/mem.  Same Xilinx APM IP that vaitrace's apm.cpp programs
 * over UIO; we use /dev/mem because the KV260 cormorant overlay does
 * not expose the APMs as UIO devices.
 *
 *   APM block        Base addr     Watches                  Slots
 *   ------------------------------------------------------------------
 *   FPD APM          0xFD0B0000    FPD inner switch         8
 *   DDR APM          0xFD490000    DDRC AXI slave ports     6   ★ default
 *   LPD APM          0xFFA00000    LPD interconnect         8
 *   OCM APM          0xFFA10000    OCM interconnect         varies
 *
 * The DDR APM is the right block for measuring FPGA→DDR bandwidth on
 * KV260: every AFI / FPGA-HP transaction landing in DDR shows up on
 * one of its six slots, regardless of whether CCI is in the path or
 * not.  (The kernels use S_AXI_HPC0 / HPC1, which reach the DDRC
 * through the CCI-400, and S_AXI_HP0 / HP2, which go straight to it —
 * doc/ARCHITECTURE.md §4.5.)
 *
 * Each slot maps to one DDRC AXI port.  The five FPGA kernels'
 * m_axi_gmem masters land on different DDRC ports depending on the
 * bitstream wiring, so a single-slot measurement misses traffic.
 * This backend supports up to 5 slots in one run (10 metric counters
 * total → 2 metrics × 5 slots).
 *
 * Run-time configuration (env vars):
 *   INFERENCE_DDR_APM_BASE    physical base addr (default 0xFD490000 = DDR APM)
 *   INFERENCE_DDR_APM_SLOTS   comma-separated list, e.g. "3,4"  (up to 5 slots;
 *                              default "0" when unset)
 *
 * Counter sequence (mirrors vaitrace's start_collect()):
 *   For each watched slot i in [0, n_slots):
 *     set_metric(slot[i], READ_BYTE_COUNT,  counter=2*i)
 *     set_metric(slot[i], WRITE_BYTE_COUNT, counter=2*i+1)
 *   CTL |= MCNTR_RESET_MASK  ; CTL &= ~MCNTR_RESET_MASK
 *   CTL |= MCNTR_ENABLE_MASK
 *   … run inference …
 *   sample(): read MC[0..2N-1] (32-bit), fold deltas into 64-bit totals
 *   CTL &= ~MCNTR_ENABLE_MASK
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1

/* Public header — resolved via the inference target's PUBLIC
 * include directory (include/ in the generated project, runtime/ when
 * the runtime tests build it standalone). */
#include "inference_ddr.h"

#if INFERENCE_PROFILING && defined(__linux__)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "../inference_ddr_backend.h"

/* ---- Xilinx APM register layout — verbatim from vaitrace's
 *      apm_definitions.hpp (the original Xilinx PG037 definitions). */

#define XAPM_GCC_HIGH_OFFSET   0x0000
#define XAPM_GCC_LOW_OFFSET    0x0004
#define XAPM_MSR0_OFFSET       0x0044
#define XAPM_MSR1_OFFSET       0x0048
#define XAPM_MSR2_OFFSET       0x004C
#define XAPM_MC0_OFFSET        0x0100  /* MCn = MC0 + n * 0x10 */
#define XAPM_CTL_OFFSET        0x0300

#define XAPM_CR_MCNTR_RESET_MASK    0x00000002u
#define XAPM_CR_MCNTR_ENABLE_MASK   0x00000001u

#define XAPM_METRIC_WRITE_BYTE_COUNT  2
#define XAPM_METRIC_READ_BYTE_COUNT   3

/* ---- backend state ---- */

#define APM_DEFAULT_BASE   0xFD490000ULL    /* ZU+ DDR APM (UG1085 §36) */
#define APM_DEFAULT_SLOT   0u
#define APM_MAP_SIZE       0x1000
#define APM_MAX_SLOTS      5u               /* 10 counters / 2 metrics */
#define APM_MAX_COUNTERS   10u

#define MC_OFFSET(n)       (XAPM_MC0_OFFSET + (n) * 0x10)

typedef struct {
    int       fd;
    void     *base;
    uint64_t  base_addr;

    unsigned  n_slots;
    unsigned  slots[APM_MAX_SLOTS];

    int       running;
    /* Per-slot 64-bit accumulators built from periodic 32-bit reads. */
    uint64_t  read_total [APM_MAX_SLOTS];
    uint64_t  write_total[APM_MAX_SLOTS];
    uint32_t  last_read  [APM_MAX_SLOTS];
    uint32_t  last_write [APM_MAX_SLOTS];

    /* Diagnostics (exposed via describe()). */
    uint32_t  ctl_writeback;
    uint32_t  mc_at_start[APM_MAX_COUNTERS];
    uint32_t  mc_at_stop [APM_MAX_COUNTERS];

    char      err[192];
} state_t;

static state_t S = { .fd = -1 };

/* ---- mmio helpers ---- */

static inline uint32_t readreg(unsigned off)
{
    return *(volatile uint32_t *)((char *)S.base + off);
}

static inline void writereg(unsigned off, uint32_t v)
{
    *(volatile uint32_t *)((char *)S.base + off) = v;
}

/* Identical to vaitrace's APM::set_metrics_counter(): MSR0 holds the
 * (slot,metric) pair for counters 0-3, MSR1 for 4-7, MSR2 for 8-9.
 * Each entry is a 13-bit field — 5-bit metric in the low 5 bits,
 * 3-bit slot in bits [7:5].  We OR the new entry into the right byte
 * after masking the old value out. */
static void set_metric(unsigned slot, uint8_t metric, unsigned counter)
{
    uint32_t mask, off, reg;
    switch (counter % 4) {
        case 0:  mask = 0xffffff00u; break;
        case 1:  mask = 0xffff00ffu; break;
        case 2:  mask = 0xff00ffffu; break;
        default: mask = 0x00ffffffu;
    }
    if      (counter < 4) { off = XAPM_MSR0_OFFSET; }
    else if (counter < 8) { off = XAPM_MSR1_OFFSET; counter -= 4; }
    else                  { off = XAPM_MSR2_OFFSET; counter -= 8; }
    reg  = readreg(off);
    reg &= mask;
    reg |= ((uint32_t)metric) << (counter * 8);
    reg |= ((uint32_t)slot)   << (counter * 8 + 5);
    writereg(off, reg);
}

/* Parse a comma-separated slot list into S.slots[].  Returns the
 * number of slots accepted (capped at APM_MAX_SLOTS).  Empty string
 * or no parseable values → 0.
 *
 * If the user requested more slots than fit in the hardware counter
 * budget, prints a one-line warning to stderr listing the dropped
 * slot numbers — silent truncation would let two configs differ by
 * one missing slot with no visible cause. */
static unsigned parse_slot_list(const char *s)
{
    unsigned n = 0;
    unsigned dropped[16];
    unsigned n_dropped = 0;

    while (s && *s) {
        char *end = NULL;
        unsigned long v = strtoul(s, &end, 0);
        if (end != s) {
            if (n < APM_MAX_SLOTS) {
                S.slots[n++] = (unsigned)v;
            } else if (n_dropped < sizeof(dropped) / sizeof(dropped[0])) {
                dropped[n_dropped++] = (unsigned)v;
            }
        }
        if (!end || !*end) break;
        s = end + 1;
    }

    if (n_dropped > 0) {
        fprintf(stderr,
                "inference_ddr: warning — APM has 10 metric counters; "
                "monitoring %u slots saturates them.  Dropping slot(s):",
                APM_MAX_SLOTS);
        for (unsigned i = 0; i < n_dropped; ++i) {
            fprintf(stderr, " %u", dropped[i]);
        }
        fprintf(stderr,
                ".  Run again with a smaller list to cover those.\n");
    }
    return n;
}

/* ---- vtable methods ---- */

static int b_probe(void)
{
    /* Reset state — probe may be called multiple times. */
    if (S.fd >= 0) close(S.fd);
    if (S.base)    munmap(S.base, APM_MAP_SIZE);
    memset(&S, 0, sizeof(S));
    S.fd = -1;

    const char *env;

    env = getenv("INFERENCE_DDR_APM_BASE");
    S.base_addr = (env && *env)
                  ? (uint64_t)strtoull(env, NULL, 0)
                  : APM_DEFAULT_BASE;

    /* Slot selection: INFERENCE_DDR_APM_SLOTS is a comma-separated
     * list of DDRC ports to monitor.  Empty/unset → watch slot 0 only. */
    env = getenv("INFERENCE_DDR_APM_SLOTS");
    if (env && *env) {
        S.n_slots = parse_slot_list(env);
    }
    if (S.n_slots == 0) {
        S.slots[0] = APM_DEFAULT_SLOT;
        S.n_slots  = 1;
    }

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        snprintf(S.err, sizeof(S.err),
                 "open(/dev/mem) failed: %s "
                 "(needs CAP_SYS_RAWIO / root)",
                 strerror(errno));
        return -1;
    }
    void *base = mmap(NULL, APM_MAP_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, (off_t)S.base_addr);
    if (base == MAP_FAILED) {
        snprintf(S.err, sizeof(S.err),
                 "mmap(0x%" PRIx64 ", %u) failed: %s",
                 S.base_addr, (unsigned)APM_MAP_SIZE, strerror(errno));
        close(fd);
        return -1;
    }

    S.fd   = fd;
    S.base = base;

    /* Sanity check: MSR0 is always fully 32-bit-writable on any APM
     * with ≥4 metric counters (PG037 Table 2-3).  If writeback fails
     * the block is dead (clock or power gated), or this address is
     * not an APM at all. */
    volatile uint32_t orig = readreg(XAPM_MSR0_OFFSET);
    writereg(XAPM_MSR0_OFFSET, 0xa5a5a5a5u);
    volatile uint32_t back = readreg(XAPM_MSR0_OFFSET);
    writereg(XAPM_MSR0_OFFSET, orig);
    if (back != 0xa5a5a5a5u) {
        snprintf(S.err, sizeof(S.err),
                 "register writeback test failed at 0x%" PRIx64 " "
                 "(MSR0 wrote 0xa5a5a5a5, read 0x%08x) — block is likely "
                 "clock-gated or this address is not a live APM",
                 S.base_addr, back);
        munmap(base, APM_MAP_SIZE);
        close(fd);
        S.fd = -1; S.base = NULL;
        return -1;
    }
    return 0;
}

static int b_start(void)
{
    if (!S.base) return -1;

    /* Assign 2 counters per slot — counter 2*i = read, 2*i+1 = write. */
    for (unsigned i = 0; i < S.n_slots; ++i) {
        set_metric(S.slots[i], XAPM_METRIC_READ_BYTE_COUNT,  2u * i);
        set_metric(S.slots[i], XAPM_METRIC_WRITE_BYTE_COUNT, 2u * i + 1u);
    }

    uint32_t ctl = readreg(XAPM_CTL_OFFSET);
    writereg(XAPM_CTL_OFFSET, ctl |  XAPM_CR_MCNTR_RESET_MASK);
    writereg(XAPM_CTL_OFFSET, ctl & ~XAPM_CR_MCNTR_RESET_MASK);
    writereg(XAPM_CTL_OFFSET, ctl |  XAPM_CR_MCNTR_ENABLE_MASK);

    S.ctl_writeback = readreg(XAPM_CTL_OFFSET);

    for (unsigned i = 0; i < S.n_slots; ++i) {
        S.read_total [i] = 0u;
        S.write_total[i] = 0u;
        S.last_read  [i] = readreg(MC_OFFSET(2u * i));
        S.last_write [i] = readreg(MC_OFFSET(2u * i + 1u));
        S.mc_at_start[2u * i]      = S.last_read [i];
        S.mc_at_start[2u * i + 1u] = S.last_write[i];
    }
    S.running = 1;
    return 0;
}

static int b_sample(void)
{
    if (!S.running) return -1;
    for (unsigned i = 0; i < S.n_slots; ++i) {
        uint32_t r = readreg(MC_OFFSET(2u * i));
        uint32_t w = readreg(MC_OFFSET(2u * i + 1u));
        /* 32-bit unsigned subtraction wraps cleanly across overflow. */
        S.read_total [i] += (uint64_t)(uint32_t)(r - S.last_read [i]);
        S.write_total[i] += (uint64_t)(uint32_t)(w - S.last_write[i]);
        S.last_read  [i] = r;
        S.last_write [i] = w;
    }
    return 0;
}

static int b_stop(void)
{
    if (!S.running) return -1;
    /* Snapshot raw counters before final sample, so describe() can
     * report what the hardware actually showed at end-of-run. */
    for (unsigned i = 0; i < S.n_slots; ++i) {
        S.mc_at_stop[2u * i]      = readreg(MC_OFFSET(2u * i));
        S.mc_at_stop[2u * i + 1u] = readreg(MC_OFFSET(2u * i + 1u));
    }
    b_sample();   /* fold final delta */

    uint32_t ctl = readreg(XAPM_CTL_OFFSET);
    writereg(XAPM_CTL_OFFSET, ctl & ~XAPM_CR_MCNTR_ENABLE_MASK);
    S.running = 0;
    return 0;
}

static int b_read_counts(uint64_t *r_bytes, uint64_t *w_bytes)
{
    uint64_t r = 0, w = 0;
    for (unsigned i = 0; i < S.n_slots; ++i) {
        r += S.read_total [i];
        w += S.write_total[i];
    }
    *r_bytes = r;
    *w_bytes = w;
    return 0;
}

static void b_describe(FILE *f)
{
    fprintf(f,
            ",\"base_addr\":\"0x%" PRIx64 "\""
            ",\"ctl_writeback\":\"0x%08x\""
            ",\"slots\":[",
            S.base_addr, S.ctl_writeback);
    for (unsigned i = 0; i < S.n_slots; ++i) {
        fprintf(f,
                "%s{\"slot\":%u"
                ",\"read_bytes\":%" PRIu64 ",\"write_bytes\":%" PRIu64
                ",\"mc_read_at_stop\":%u,\"mc_write_at_stop\":%u}",
                i ? "," : "",
                S.slots[i],
                S.read_total [i], S.write_total[i],
                S.mc_at_stop[2u * i], S.mc_at_stop[2u * i + 1u]);
    }
    fputc(']', f);
}

static const char *b_last_error(void)
{
    return S.err[0] ? S.err : NULL;
}

static void b_deinit(void)
{
    if (S.base != NULL && S.base != MAP_FAILED) {
        munmap(S.base, APM_MAP_SIZE);
    }
    if (S.fd >= 0) close(S.fd);
    S.base = NULL;
    S.fd   = -1;
    S.running = 0;
}

const ddr_backend_t inference_ddr_backend_zuplus_apm = {
    .name        = "zuplus_apm",
    .probe       = b_probe,
    .start       = b_start,
    .sample      = b_sample,
    .stop        = b_stop,
    .read_counts = b_read_counts,
    .describe    = b_describe,
    .last_error  = b_last_error,
    .deinit      = b_deinit,
};

#endif /* INFERENCE_PROFILING && __linux__ */

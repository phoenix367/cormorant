// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * rpu_main.c — the kernel-driver firmware on the Cortex-R5 (RPU_PLAN track
 * B1).  It knows no model: the A53's generated library (src/inference_rpu.c)
 * records every kernel call's register block, its Start and the lane waits
 * into a command ring in DDR and submits batches through the control block
 * (rpu_ctrl.h) in the BTCM; the firmware replays them, waiting on the
 * kernels' interrupts (rpu_irq.c) instead of polling.
 *
 * Built by CMakeLists.txt here against the bitstream's R5 standalone BSP
 * (tools/rpu/make_r5_bsp.py): xparameters.h names the kernels and their PL
 * lines.  Loaded by remoteproc (board/kv260/rpu/rpu.dts, rpu_board.py).
 */
#include "rpu_ctrl.h"
#include "rpu_irq.h"
#include "xil_cache.h"
#include "xil_io.h"

#define CTRL_AP_CTRL  0x00u
#define CTRL_ARGS     0x10u

/* The control block: first in the BTCM (lscript.ld .rpu_ctrl). */
volatile struct rpu_ctrl rpu_ctrl __attribute__((section(".rpu_ctrl"), aligned(64))) = {
    .magic = 0u, .version = RPU_CTRL_VERSION, .state = RPU_STATE_BOOT,
};

enum { PROG_LOOP = 1, PROG_IRQ = 2, PROG_BATCH = 3, PROG_BATCH_DONE = 4 };

/* The R5's PMU cycle counter: how long the firmware has been idle.  The RPU runs
 * at 533 MHz on the KV260 (the BSP does not name the CPU clock), and the BSP's
 * cpu_init.S sets PMCR.D: the counter ticks once per 64 cycles.  Returns the
 * counts in RPU_IDLE_SPIN_US. */
#define R5_CYCLES_PER_US 533u
static uint32_t cycles_init(void)
{
    uint32_t v;
    __asm__ volatile("mrc p15, 0, %0, c9, c12, 0" : "=r"(v));    /* PMCR */
    __asm__ volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(v | 1u)); /* E: counters on */
    __asm__ volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(1u << 31));   /* PMCNTENSET: the cycle counter */
    return RPU_IDLE_SPIN_US * R5_CYCLES_PER_US / ((v & 8u) ? 64u : 1u);
}
static inline uint32_t cycles(void)
{
    uint32_t v;
    __asm__ volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(v));    /* PMCCNTR */
    return v;
}

static void rpu_batch(void)
{
    const uint32_t n = rpu_ctrl.batch_words;
    const uint32_t *w;
    uint32_t i = 0u, calls = 0u, waits = 0u;

    if (rpu_ctrl.ring_hi != 0u || rpu_ctrl.ring_lo == 0u || n * 4u > rpu_ctrl.ring_bytes) {
        rpu_ctrl.rc = (uint32_t)-5; return;
    }
    w = (const uint32_t *)(uintptr_t)rpu_ctrl.ring_lo;
    Xil_DCacheInvalidateRange((INTPTR)w, n * 4u);
    rpu_ctrl.state = RPU_STATE_RUNNING;
    rpu_ctrl.progress = PROG_BATCH;
    while (i < n) {
        const uint32_t hdr = w[i];
        rpu_ctrl.batch_pos = i++;
        const uint32_t lane = RPU_CMD_LANE(hdr);
        UINTPTR base;
        if (lane >= RPU_LANES || !(rpu_ctrl.lanes & (1u << lane))) { rpu_ctrl.rc = (uint32_t)-7; goto out; }
        base = rpu_lane_base(lane);
        switch (RPU_CMD_OP(hdr)) {
        case RPU_OP_REGS: {
            const uint32_t cnt = RPU_CMD_N(hdr);
            uint32_t r;
            if (i + cnt > n) { rpu_ctrl.rc = (uint32_t)-5; goto out; }
            for (r = 0u; r < cnt; r++) Xil_Out32(base + CTRL_ARGS + 4u * r, w[i + r]);
            i += cnt;
            break;
        }
        case RPU_OP_REGSO: {
            const uint32_t cnt = RPU_CMD_N(hdr);
            uint32_t r, first;
            if (i + 1u + cnt > n) { rpu_ctrl.rc = (uint32_t)-5; goto out; }
            first = w[i++];
            if (first + cnt > RPU_REG_WORDS_MAX) { rpu_ctrl.rc = (uint32_t)-6; goto out; }
            for (r = 0u; r < cnt; r++) Xil_Out32(base + CTRL_ARGS + 4u * (first + r), w[i + r]);
            i += cnt;
            break;
        }
        case RPU_OP_START:
            rpu_irq_start(lane);                    /* its interrupt armed first */
            Xil_Out32(base + CTRL_AP_CTRL, 1u);     /* ap_start (no auto-restart) */
            calls++;
            break;
        case RPU_OP_WAIT:
            rpu_ctrl.wait_lane = lane;
            rpu_irq_wait(lane);
            waits++;
            break;
        default:
            rpu_ctrl.rc = (uint32_t)-6; goto out;
        }
    }
    rpu_ctrl.rc = 0u;
out:
    rpu_irq_batch_end();
    rpu_ctrl.calls += calls;
    rpu_ctrl.waits += waits;
    rpu_ctrl.batches++;
    rpu_ctrl.state = RPU_STATE_READY;
    rpu_ctrl.progress = PROG_BATCH_DONE;
}

int main(void)
{
    uint32_t seen = 0u, idle_from, idle_spin;
    int rc;

    rpu_ctrl.done_seq = 0u;
    rpu_ctrl.cmd_seq = 0u;
    rpu_ctrl.batches = 0u; rpu_ctrl.calls = 0u; rpu_ctrl.waits = 0u;
    rpu_ctrl.progress = PROG_LOOP;
    rc = rpu_irq_init(&rpu_ctrl);
    rpu_ctrl.progress = PROG_IRQ;
    if (rc != 0) {
        rpu_ctrl.rc = (uint32_t)rc;
        rpu_ctrl.state = RPU_STATE_ERROR;
    } else {
        rpu_ctrl.rc = 0u;
        rpu_ctrl.state = RPU_STATE_READY;
    }
    rpu_ctrl.magic = RPU_CTRL_MAGIC;
    __asm__ volatile("dsb" ::: "memory");
    idle_spin = cycles_init();
    idle_from = cycles();

    /* Between commands: poll for RPU_IDLE_SPIN_US (the batches of one inference
     * pay no wake-up), then sleep until the library rings (rpu_irq_idle). */
    while (rpu_ctrl.state != RPU_STATE_ERROR) {
        uint32_t seq = rpu_ctrl.cmd_seq;
        if (seq == seen) {
            if (cycles() - idle_from > idle_spin) rpu_irq_idle(seen);
            __asm__ volatile("dsb" ::: "memory");
            continue;
        }
        seen = seq;
        switch (rpu_ctrl.cmd) {
        case RPU_CMD_BATCH:
            rpu_batch();
            break;
        case RPU_CMD_QUIT:
            rpu_ctrl.rc = 0u;
            rpu_ctrl.state = RPU_STATE_QUIT;
            break;
        default:
            rpu_ctrl.rc = (uint32_t)-1;
            break;
        }
        __asm__ volatile("dsb" ::: "memory");
        rpu_ctrl.done_seq = seq;
        __asm__ volatile("dsb" ::: "memory");
        /* the A53 sets `wake` before it sleeps and rechecks done_seq after: either
         * it sees done_seq or this sees `wake` (both sides fence in between) */
        if (rpu_ctrl.wake) {
            Xil_Out32(RPU_IPI_RPU0_TRIG, RPU_IPI_CH7_BIT);
            rpu_ctrl.doorbells++;
        }
        if (rpu_ctrl.state == RPU_STATE_QUIT) break;
        idle_from = cycles();
    }
    for (;;) { __asm__ volatile("wfi"); }
    return 0;
}

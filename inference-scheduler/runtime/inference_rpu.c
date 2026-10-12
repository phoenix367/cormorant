/*
 * inference_rpu.c — the recorder behind inference_rpu.h.  While the R5
 * drives a run, each driver instance's Ctrl_BaseAddress points at a shadow
 * register page, so the generated code's register writes land in memory;
 * every Start (a macro in the generated inference.c) copies that page's
 * argument registers into the command ring with a START command, every
 * kernel_wait() records a WAIT, and a flush hands the batch to the
 * Cortex-R5 firmware (runtime/rpu/rpu_main.c) through the control block in
 * its BTCM (rpu_ctrl.h).
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "inference.h"
#include "inference_rpu.h"
#include "rpu_ctrl.h"

#define SHADOW_WORDS     64u                /* 256 B: every kernel's control block */
#define DEFAULT_RING     (1u << 20)
#define FLUSH_TIMEOUT_S  60.0

static volatile struct rpu_ctrl *s_ctrl;    /* NULL = not attached */
static inference_buf_t *s_ring;
static uint32_t *s_ring_words;
static uint32_t  s_ring_bytes;
static uint32_t  s_words;                   /* recorded, not yet submitted */
static uint64_t  s_saved_base[16];          /* the real Ctrl_BaseAddress per kernel id */
static uint32_t  s_shadow[RPU_LANES][SHADOW_WORDS] __attribute__((aligned(64)));
static const inference_rpu_kernel_t *s_kernels;
static unsigned  s_n_kernels;
static int       s_error;
static uint32_t  s_flushes, s_words_total;
static int       s_enabled;                 /* the shadow pages are in place: calls are recorded */
static int       s_mode = INFERENCE_RPU_MODE_A53;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int inference_rpu_active(void) { return s_ctrl != NULL && s_enabled; }
int inference_rpu_mode(void)   { return s_ctrl ? s_mode : -1; }
int inference_rpu_error(void)  { return s_error; }

static volatile struct rpu_ctrl *map_ctrl(void)
{
#if defined(__aarch64__)
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    void *p;
    if (fd < 0) return NULL;
    p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RPU_CTRL_A53_ADDR);
    close(fd);
    return p == MAP_FAILED ? NULL : (volatile struct rpu_ctrl *)p;
#else
    return NULL;                            /* not the KV260's A53 (the host emulation) */
#endif
}

/* Map the control block and allocate the ring; no mode change.  The error
 * codes of inference_rpu_attach(). */
static int attach_core(unsigned ring_bytes)
{
    volatile struct rpu_ctrl *c;
    unsigned i;

    if (s_ctrl) return 0;
    s_kernels = inference_rpu_kernels(&s_n_kernels);
    if (s_n_kernels == 0u) return -1;       /* nothing to drive */
    if (s_n_kernels > 16u) return -3;
    c = map_ctrl();
    if (!c) return -1;
    if (c->magic != RPU_CTRL_MAGIC) { munmap((void *)c, 4096); return -1; }
    if (c->version != RPU_CTRL_VERSION || c->state == RPU_STATE_ERROR || c->state == RPU_STATE_QUIT) {
        munmap((void *)c, 4096); return -2;
    }
    for (i = 0u; i < s_n_kernels; i++)
        if (!(c->lanes & (1u << s_kernels[i].lane))) { munmap((void *)c, 4096); return -3; }
    if (ring_bytes == 0u) ring_bytes = DEFAULT_RING;
    ring_bytes = (ring_bytes + 4095u) & ~4095u;
    s_ring = inference_buf_alloc(ring_bytes / INFERENCE_BYTES_PER_ELEM);
    if (!s_ring || inference_buf_phys(s_ring) + ring_bytes > 0x100000000ull) {
        if (s_ring) inference_buf_free(s_ring);
        s_ring = NULL; munmap((void *)c, 4096); return -4;
    }
    s_ring_words = (uint32_t *)inference_buf_ptr(s_ring);
    s_ring_bytes = ring_bytes;
    s_words = 0u;
    s_error = 0;
    s_enabled = 0;
    s_ctrl = c;
    return 0;
}

/* The shadow pages in (on) or out (off).  A page starts as the kernel's
 * registers stand, so a register the generated code never writes keeps the
 * value the kernel holds; leaving the R5 first flushes what it recorded. */
static void enable(int on)
{
    unsigned i;
    if (!s_ctrl || on == s_enabled) return;
    if (on) {
        for (i = 0u; i < s_n_kernels; i++) {
            const inference_rpu_kernel_t *k = &s_kernels[i];
            const volatile uint32_t *real = (const volatile uint32_t *)(uintptr_t)*k->ctrl_base;
            unsigned w;
            s_saved_base[i] = *k->ctrl_base;
            for (w = 0u; w < SHADOW_WORDS; w++) s_shadow[k->lane][w] = 0u;
            for (w = 4u; w < 4u + k->reg_words; w++) s_shadow[k->lane][w] = real[w];
            *k->ctrl_base = (uint64_t)(uintptr_t)s_shadow[k->lane];
        }
        s_enabled = 1;
    } else {
        inference_rpu_flush();
        for (i = 0u; i < s_n_kernels; i++) *s_kernels[i].ctrl_base = s_saved_base[i];
        s_enabled = 0;
    }
}

int inference_rpu_set_mode(int mode)
{
    if (!s_ctrl || mode < INFERENCE_RPU_MODE_A53 || mode > INFERENCE_RPU_MODE_R5) return -1;
    s_mode = mode;
    if (mode == INFERENCE_RPU_MODE_A53) enable(0);
    else if (mode == INFERENCE_RPU_MODE_R5) enable(1);
    return 0;                               /* AUTO: the next run's entry decides */
}

int inference_rpu_attach(unsigned ring_bytes)
{
    int rc = attach_core(ring_bytes);
    if (rc != 0) return rc;
    return inference_rpu_set_mode(INFERENCE_RPU_MODE_R5);
}

void inference_rpu_detach(void)
{
    if (!s_ctrl) return;
    enable(0);
    munmap((void *)s_ctrl, 4096);
    s_ctrl = NULL;
    s_mode = INFERENCE_RPU_MODE_A53;
    inference_buf_free(s_ring);
    s_ring = NULL;
}

int inference_rpu_select(unsigned ring_bytes)
{
    const char *env = getenv("INFERENCE_RPU");
    int forced = 0, rc;
    if (env && (!strcmp(env, "0") || !strcmp(env, "off") || !strcmp(env, "a53"))) return 0;
    if (env && (!strcmp(env, "1") || !strcmp(env, "on") || !strcmp(env, "r5"))) forced = 1;
    rc = attach_core(ring_bytes);
    if (rc == -1) return 0;                 /* no firmware: the A53 drives, silently */
    if (rc != 0) {
        fprintf(stderr, "inference: the Cortex-R5 firmware is there but cannot be used (%s): "
                        "the A53 drives the kernels\n",
                rc == -2 ? "in error or another protocol version" :
                rc == -3 ? "no lane for a kernel of this project" : "no memory for the command ring");
        return rc;
    }
    inference_rpu_set_mode(forced ? INFERENCE_RPU_MODE_R5 : INFERENCE_RPU_MODE_AUTO);
    return 1;
}

void inference_rpu_enter(int recommended)
{
    if (!s_ctrl) return;
    if (s_mode == INFERENCE_RPU_MODE_AUTO) enable(recommended);
    else enable(s_mode == INFERENCE_RPU_MODE_R5);
}

void inference_rpu_flush(void)
{
    volatile struct rpu_ctrl *c = s_ctrl;
    uint32_t seq;
    double t0;
    if (!c || s_words == 0u) return;
    inference_buf_sync_to_device(s_ring);               /* the batch out of the A53's cache */
    /* the ring is named per batch: several libraries in one process share the
     * firmware (the chat server), each with its own ring */
    c->ring_lo = (uint32_t)inference_buf_phys(s_ring);
    c->ring_hi = (uint32_t)(inference_buf_phys(s_ring) >> 32);
    c->ring_bytes = s_ring_bytes;
    c->batch_words = s_words;
    c->cmd = RPU_CMD_BATCH;
    __sync_synchronize();
    seq = c->cmd_seq + 1u;
    c->cmd_seq = seq;
    __sync_synchronize();
    t0 = now_s();
    while (c->done_seq != seq) {
        if (now_s() - t0 > FLUSH_TIMEOUT_S) {
            fprintf(stderr, "inference_rpu: the R5 did not finish a batch of %u words (state %u, progress %u, "
                            "lane %u, fault %u @ 0x%08x)\n", s_words, c->state, c->progress, c->wait_lane,
                    c->fault, c->fault_addr);
            s_error = -1;
            s_words = 0u;
            return;
        }
    }
    if (c->rc != 0u) {
        fprintf(stderr, "inference_rpu: the firmware rejected a batch: rc %d\n", (int)c->rc);
        s_error = (int)c->rc;
    }
    s_flushes++;
    s_words_total += s_words;
    s_words = 0u;
}

static void room(uint32_t words)
{
    if (s_words + words > s_ring_bytes / 4u) inference_rpu_flush();
}

void inference_rpu_start(unsigned k)
{
    const inference_rpu_kernel_t *kd = &s_kernels[k];
    const uint32_t n = kd->reg_words;
    uint32_t *w;
    room(n + 2u);
    w = s_ring_words + s_words;
    w[0] = RPU_CMD_WORD(RPU_OP_REGS, kd->lane, n);
    memcpy(w + 1, &s_shadow[kd->lane][4], n * 4u);
    w[1 + n] = RPU_CMD_WORD(RPU_OP_START, kd->lane, 0u);
    s_words += n + 2u;
}

void inference_rpu_wait(unsigned k)
{
    room(1u);
    s_ring_words[s_words++] = RPU_CMD_WORD(RPU_OP_WAIT, s_kernels[k].lane, 0u);
}

int inference_rpu_stats(inference_rpu_stats_t *out)
{
    unsigned i;
    if (!s_ctrl) return -1;
    memset(out, 0, sizeof *out);
    out->batches = s_ctrl->batches;
    out->calls = s_ctrl->calls;
    out->waits = s_ctrl->waits;
    for (i = 0u; i < 8u; i++) { out->irq_count[i] = s_ctrl->irq_count[i]; out->irq_missed[i] = s_ctrl->irq_missed[i]; }
    out->self_test_ok = s_ctrl->gic[12];
    out->lanes = s_ctrl->lanes;
    out->flushes = s_flushes;
    out->words = s_words_total;
    return 0;
}

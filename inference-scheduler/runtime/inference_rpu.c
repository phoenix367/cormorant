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
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
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
#define FLUSH_TIMEOUT_S  60.0               /* INFERENCE_RPU_TIMEOUT_S */
#define SPIN_US_DEFAULT  50u                /* then sleep on the doorbell (INFERENCE_RPU_SPIN_US) */
#define IPI_TRIG         (0x00u / 4u)       /* IPI channel 7's registers (the rpu_ipi UIO map) */
#define IPI_ISR          (0x10u / 4u)
#define IPI_IER          (0x18u / 4u)

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
static uint32_t  s_sent[RPU_LANES][SHADOW_WORDS];   /* the argument registers the R5 last wrote */
static uint32_t  s_sent_valid;              /* bit per lane: s_sent holds what the kernel holds */
static int       s_mode = INFERENCE_RPU_MODE_A53;
static int       s_ipi_fd = -1;             /* the doorbell (board/kv260/rpu/rpu-ipi.dts), or -1: spin */
static volatile uint32_t *s_ipi;
static double    s_spin_s = SPIN_US_DEFAULT * 1e-6;
static uint32_t  s_sleeps;
static uint32_t  s_rings;                   /* submissions that rang a sleeping R5 */
static double    s_timeout_s = FLUSH_TIMEOUT_S;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int inference_rpu_active(void) { return s_ctrl != NULL && s_enabled; }
int inference_rpu_mode(void)   { return s_ctrl ? s_mode : -1; }
int inference_rpu_error(void)  { return s_error; }

/* 1 when a remote processor runs.  Its TCMs are powered only then: a read of
 * the control block with the R5 stopped (or never started since boot) is a
 * bus error (SIGBUS) — so the sysfs state comes first. */
static int r5_running(void)
{
    DIR *d = opendir("/sys/class/remoteproc");
    struct dirent *e;
    int running = 0;
    if (!d) return 0;
    while (!running && (e = readdir(d)) != NULL) {
        char path[288], state[32];
        FILE *f;
        if (strncmp(e->d_name, "remoteproc", 10) != 0) continue;
        snprintf(path, sizeof path, "/sys/class/remoteproc/%s/state", e->d_name);
        f = fopen(path, "r");
        if (!f) continue;
        state[0] = '\0';
        if (fgets(state, sizeof state, f)) running = strncmp(state, "running", 7) == 0;
        fclose(f);
    }
    closedir(d);
    return running;
}

static volatile struct rpu_ctrl *map_ctrl(void)
{
#if defined(__aarch64__)
    int fd;
    void *p;
    if (!r5_running()) return NULL;
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return NULL;
    p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RPU_CTRL_A53_ADDR);
    close(fd);
    return p == MAP_FAILED ? NULL : (volatile struct rpu_ctrl *)p;
#else
    return NULL;                            /* not the KV260's A53 (the host emulation) */
#endif
}

/* The doorbell: the UIO device named rpu_ipi (IPI channel 7), its RPU0 source
 * enabled.  Without it (an older overlay) the library spins on the control
 * block, as before. */
static void ipi_open(void)
{
#if defined(__aarch64__)
    DIR *d = opendir("/sys/class/uio");
    struct dirent *e;
    char path[288], name[64];
    const char *env = getenv("INFERENCE_RPU_SPIN_US");
    if (env) s_spin_s = 1e-6 * (double)strtoul(env, NULL, 10);
    env = getenv("INFERENCE_RPU_TIMEOUT_S");
    if (env && strtod(env, NULL) > 0.0) s_timeout_s = strtod(env, NULL);
    if (!d) return;
    while ((e = readdir(d)) != NULL) {
        FILE *f;
        if (strncmp(e->d_name, "uio", 3) != 0) continue;
        snprintf(path, sizeof path, "/sys/class/uio/%s/name", e->d_name);
        f = fopen(path, "r");
        if (!f) continue;
        name[0] = '\0';
        if (fgets(name, sizeof name, f)) name[strcspn(name, "\n")] = '\0';
        fclose(f);
        if (strcmp(name, "rpu_ipi") != 0) continue;
        snprintf(path, sizeof path, "/dev/%s", e->d_name);
        s_ipi_fd = open(path, O_RDWR | O_CLOEXEC);
        if (s_ipi_fd >= 0) {
            void *p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, s_ipi_fd, 0);
            if (p == MAP_FAILED) { close(s_ipi_fd); s_ipi_fd = -1; }
            else {
                s_ipi = (volatile uint32_t *)p;
                s_ipi[IPI_ISR] = RPU_IPI_SRC_RPU0;      /* a stale doorbell */
                s_ipi[IPI_IER] = RPU_IPI_SRC_RPU0;
            }
        }
        break;
    }
    closedir(d);
#endif
}

static void ipi_close(void)
{
    if (s_ipi) munmap((void *)s_ipi, 4096);
    if (s_ipi_fd >= 0) close(s_ipi_fd);
    s_ipi = NULL;
    s_ipi_fd = -1;
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
    s_ring = inference_buf_alloc_uncached(ring_bytes / INFERENCE_BYTES_PER_ELEM);   /* no cache clean per batch */
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
    ipi_open();
    /* the R5 sleeps between commands only while the libraries attached can ring it */
    c->a53_rings = s_ipi ? RPU_RINGS_MAGIC : 0u;
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#endif
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
        s_sent_valid = 0u;
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
    ipi_close();
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
    /* another library or the A53 may have driven the kernels since the last run:
     * every lane's first call of this run sends its whole block */
    s_sent_valid = 0u;
    if (s_mode == INFERENCE_RPU_MODE_AUTO) enable(recommended);
    else enable(s_mode == INFERENCE_RPU_MODE_R5);
}

/* A hung batch, for the log: the commands before the one the R5 executes
 * (decoded), and the waited lane's registers as the kernel holds them next to
 * what the generated code last wrote (its shadow page). */
static void dump_hang(uint32_t pos, uint32_t lane)
{
    static const char *ops[] = {"?", "REGS", "START", "WAIT", "REGSO"};
    uint32_t i = 0u, shown = 0u, start = 0u, k;
    /* walk from the batch start to `pos`, remembering the last 24 command starts */
    uint32_t hist[24];
    while (i < s_words && i <= pos) {
        const uint32_t h = s_ring_words[i], op = RPU_CMD_OP(h);
        hist[shown++ % 24u] = i;
        i += 1u + (op == RPU_OP_REGS ? RPU_CMD_N(h) : op == RPU_OP_REGSO ? 1u + RPU_CMD_N(h) : 0u);
    }
    start = shown > 24u ? shown - 24u : 0u;
    for (k = start; k < shown; k++) {
        const uint32_t at = hist[k % 24u], h = s_ring_words[at], op = RPU_CMD_OP(h), n = RPU_CMD_N(h);
        uint32_t j, first = op == RPU_OP_REGSO ? s_ring_words[at + 1u] : 0u;
        const uint32_t *v = s_ring_words + at + (op == RPU_OP_REGSO ? 2u : 1u);
        fprintf(stderr, "  [%5u] %-5s lane %u", at, op <= 4u ? ops[op] : "?", RPU_CMD_LANE(h));
        if (op == RPU_OP_REGS || op == RPU_OP_REGSO) {
            fprintf(stderr, " regs 0x%02x..:", 0x10u + 4u * first);
            for (j = 0u; j < n && j < 48u; j++) fprintf(stderr, " %x", v[j]);
        }
        fprintf(stderr, "\n");
    }
    for (k = 0u; k < s_n_kernels; k++) {
        const inference_rpu_kernel_t *kd = &s_kernels[k];
        const volatile uint32_t *real = (const volatile uint32_t *)(uintptr_t)s_saved_base[k];
        uint32_t w;
        if (kd->lane != lane) continue;
        fprintf(stderr, "  lane %u kernel: ap_ctrl %x gie %x ier %x isr %x\n  kernel :", lane,
                real[0], real[1], real[2], real[3]);
        for (w = 0u; w < kd->reg_words; w++) fprintf(stderr, " %x", real[4u + w]);
        fprintf(stderr, "\n  shadow :");
        for (w = 0u; w < kd->reg_words; w++) fprintf(stderr, " %x", s_shadow[lane][4u + w]);
        fprintf(stderr, "\n");
    }
}

void inference_rpu_flush(void)
{
    volatile struct rpu_ctrl *c = s_ctrl;
    uint32_t seq;
    double t0;
    if (!c || s_words == 0u) return;
    if (inference_buf_is_cached(s_ring)) {   /* (bare metal) the batch's words out of the cache */
        inference_buf_t view;
        inference_buf_init_view(&view, s_ring, 0u,
                                (s_words * 4u + INFERENCE_BYTES_PER_ELEM - 1u) / INFERENCE_BYTES_PER_ELEM);
        inference_buf_sync_to_device(&view);
    }
    /* the ring is named per batch: several libraries in one process share the
     * firmware (the chat server), each with its own ring */
    c->ring_lo = (uint32_t)inference_buf_phys(s_ring);
    c->ring_hi = (uint32_t)(inference_buf_phys(s_ring) >> 32);
    c->ring_bytes = s_ring_bytes;
    c->batch_words = s_words;
    c->cmd = RPU_CMD_BATCH;
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");      /* the ring's write-combined stores reach DDR first */
#else
    __sync_synchronize();
#endif
    seq = c->cmd_seq + 1u;
    c->cmd_seq = seq;
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");      /* cmd_seq out before r5_sleeping is read */
#else
    __sync_synchronize();
#endif
    /* the R5 sets r5_sleeping, fences, then rereads cmd_seq: it sees this
     * submission, or this sees it asleep and rings (rpu_ctrl.h, protocol v5) */
    if (s_ipi && c->r5_sleeping) {
        s_ipi[IPI_TRIG] = RPU_IPI_DST_RPU0;
        s_rings++;
    }
    t0 = now_s();
    /* spin a little (a short batch costs less than a wake-up), then sleep on the
     * doorbell: ask for it, recheck, unmask the interrupt, block */
    while (c->done_seq != seq && now_s() - t0 < s_spin_s) { }
    if (c->done_seq != seq && s_ipi) {
        for (;;) {
            struct pollfd pfd;
            uint32_t one = 1u, count;
            s_ipi[IPI_ISR] = RPU_IPI_SRC_RPU0;           /* a doorbell already consumed */
            c->wake = 1u;
#if defined(__aarch64__)
            __asm__ volatile("dsb sy" ::: "memory");
#endif
            if (c->done_seq == seq) break;
            if (write(s_ipi_fd, &one, sizeof one) != (ssize_t)sizeof one) break;   /* unmask */
            if (c->done_seq == seq) break;
            pfd.fd = s_ipi_fd; pfd.events = POLLIN; pfd.revents = 0;
            s_sleeps++;
            if (poll(&pfd, 1, (int)(s_timeout_s * 1000.0)) > 0)
                (void)!read(s_ipi_fd, &count, sizeof count);
            if (c->done_seq == seq || now_s() - t0 > s_timeout_s) break;
        }
        c->wake = 0u;
    }
    while (c->done_seq != seq) {
        if (now_s() - t0 > s_timeout_s) {
            fprintf(stderr, "inference_rpu: the R5 did not finish a batch of %u words (state %u, progress %u, "
                            "lane %u, fault %u @ 0x%08x, at word %u)\n", s_words, c->state, c->progress,
                    c->wait_lane, c->fault, c->fault_addr, c->batch_pos);
            dump_hang(c->batch_pos, c->wait_lane);
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

/* A Start: the lane's whole argument block on its first call of a run, then
 * only the runs of registers that changed (a gap of up to two unchanged words
 * is sent rather than split, a segment costs two header words). */
void inference_rpu_start(unsigned k)
{
    const inference_rpu_kernel_t *kd = &s_kernels[k];
    const uint32_t n = kd->reg_words, lane = kd->lane;
    const uint32_t *regs = &s_shadow[lane][4];
    uint32_t *sent = s_sent[lane];
    uint32_t *w;
    room(2u * n + 4u);
    w = s_ring_words + s_words;
    if (!(s_sent_valid & (1u << lane))) {
        w[0] = RPU_CMD_WORD(RPU_OP_REGS, lane, n);
        memcpy(w + 1, regs, n * 4u);
        memcpy(sent, regs, n * 4u);
        s_sent_valid |= 1u << lane;
        w += 1u + n;
    } else {
        uint32_t i = 0u;
        while (i < n) {
            uint32_t first, last, gap;
            if (regs[i] == sent[i]) { i++; continue; }
            first = last = i;
            for (i = i + 1u, gap = 0u; i < n && gap <= 2u; i++) {
                if (regs[i] != sent[i]) { last = i; gap = 0u; } else gap++;
            }
            i = last + 1u;
            w[0] = RPU_CMD_WORD(RPU_OP_REGSO, lane, last - first + 1u);
            w[1] = first;
            memcpy(w + 2, regs + first, (last - first + 1u) * 4u);
            memcpy(sent + first, regs + first, (last - first + 1u) * 4u);
            w += 2u + (last - first + 1u);
        }
    }
    w[0] = RPU_CMD_WORD(RPU_OP_START, lane, 0u);
    s_words = (uint32_t)(w + 1 - s_ring_words);
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
    out->sleeps = s_sleeps;
    out->doorbell = s_ipi != NULL;
    out->rings = s_rings;
    out->r5_sleeps = s_ctrl->r5_sleeps;
    out->r5_rings = s_ctrl->r5_rings;
    return 0;
}

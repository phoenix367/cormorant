// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/*
 * inference_irq.c — kernel waits on the kernels' interrupts while the A53
 * drives them (Linux, UIO): inference_irq.h.
 *
 * The UIO devices are opened here a second time (the drivers' own file
 * descriptors are private to them); every open file of a UIO device sees
 * each interrupt, and a write of 1 to any of them unmasks the line.
 */
#define _GNU_SOURCE
#include "inference_irq.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

#define LANES            8u
#define REG_GIE          1u     /* 32-bit words of the ap_ctrl_hs block: 0x04 */
#define REG_IER          2u     /* 0x08: bit 0 latches ap_done into the ISR */
#define REG_ISR          3u     /* 0x0C: bit 0 = done; a write toggles */
#define POLL_TIMEOUT_MS  100    /* a sleep wakes to read the ISR at least this often */
#define EARLY_MIN_US     50     /* a shorter timed sleep is not worth its syscall and wake-up */

typedef struct {
    int                fd;      /* the UIO device (valid while regs != NULL) */
    volatile uint32_t *regs;    /* the kernel's register window, or NULL: the lane polls */
    int                armed;   /* the last Start armed the interrupt: the wait sleeps */
    unsigned           expect_us;  /* the predicted length of the lane's calls (0: unknown) */
    double             t_start;    /* the armed Start's time (with expect_us) */
} lane_t;

static lane_t   s_lane[LANES];
static int      s_env_done;
static int      s_mode = INFERENCE_IRQ_MODE_DEFAULT;
static unsigned s_spin_us = INFERENCE_IRQ_SPIN_US_DEFAULT;
static int      s_early_us = INFERENCE_IRQ_EARLY_US_DEFAULT;
static unsigned long long s_waits, s_spun, s_sleeps, s_wakeups, s_timeouts, s_missed;
static unsigned long long s_early, s_early_timer, s_early_line;
static double   s_early_spin_s;

static void env_once(void)
{
    const char *e;
    if (s_env_done) return;
    s_env_done = 1;
    e = getenv("INFERENCE_WAIT");
    if (e && (!strcmp(e, "poll") || !strcmp(e, "0"))) s_mode = INFERENCE_WAIT_POLL;
    else if (e && (!strcmp(e, "irq") || !strcmp(e, "1"))) s_mode = INFERENCE_WAIT_IRQ;
    e = getenv("INFERENCE_IRQ_SPIN_US");
    if (e && *e) s_spin_us = (unsigned)strtoul(e, NULL, 10);
    e = getenv("INFERENCE_IRQ_EARLY_US");
    if (e && *e) s_early_us = !strcmp(e, "off") ? -1 : (int)strtol(e, NULL, 10);
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int inference_irq_open(unsigned k, const char *uio_name, void *ctrl)
{
    env_once();
    if (k >= LANES || !uio_name || !ctrl) return -1;
#if defined(__linux__) && defined(__aarch64__)     /* the KV260; never the host emulation */
    {
        DIR *d = opendir("/sys/class/uio");
        struct dirent *e;
        char path[288], name[64];
        int fd = -1;
        if (!d) return -1;
        while (fd < 0 && (e = readdir(d)) != NULL) {
            FILE *f;
            if (strncmp(e->d_name, "uio", 3) != 0) continue;
            snprintf(path, sizeof path, "/sys/class/uio/%s/name", e->d_name);
            if (!(f = fopen(path, "r"))) continue;
            name[0] = '\0';
            if (fgets(name, sizeof name, f)) name[strcspn(name, "\n")] = '\0';
            fclose(f);
            if (strcmp(name, uio_name) != 0) continue;
            snprintf(path, sizeof path, "/dev/%s", e->d_name);
            fd = open(path, O_RDWR | O_CLOEXEC);
        }
        closedir(d);
        if (fd < 0) return -1;
        if (s_lane[k].regs) close(s_lane[k].fd);
        s_lane[k].fd = fd;
        s_lane[k].regs = (volatile uint32_t *)ctrl;
        s_lane[k].armed = 0;
        return 0;
    }
#else
    return -1;
#endif
}

void inference_irq_close(void)
{
    unsigned k;
    for (k = 0u; k < LANES; k++) {
#if defined(__linux__)
        if (s_lane[k].regs) close(s_lane[k].fd);
#endif
        s_lane[k].regs = NULL;
        s_lane[k].armed = 0;
    }
}

int inference_irq_set_mode(int mode)
{
    int prev;
    env_once();
    prev = s_mode;
    if (mode == INFERENCE_WAIT_POLL || mode == INFERENCE_WAIT_IRQ) s_mode = mode;
    return prev;
}

int inference_irq_mode(void)
{
    env_once();
    return s_mode;
}

unsigned inference_irq_set_spin_us(unsigned us)
{
    unsigned prev;
    env_once();
    prev = s_spin_us;
    s_spin_us = us;
    return prev;
}

int inference_irq_set_early_us(int us)
{
    int prev;
    env_once();
    prev = s_early_us;
    s_early_us = us;
    return prev;
}

void inference_irq_expect(unsigned k, unsigned us)
{
    if (k < LANES) s_lane[k].expect_us = us;
}

void inference_irq_start(unsigned k)
{
    volatile uint32_t *r;
    uint32_t isr;
    if (k >= LANES) return;
    s_lane[k].armed = 0;
    r = s_lane[k].regs;
    if (!r || s_mode != INFERENCE_WAIT_IRQ) return;
    /* The lane is idle: let ap_done latch, clear the previous job's (this
     * library's, another's or the R5's) and see the clear land — after it the
     * bit can only be this job's.  The ap_start that follows goes through the
     * same device mapping, after these writes. */
    r[REG_IER] = 1u;
    isr = r[REG_ISR];
    if (isr) {
        unsigned n = 0u;
        r[REG_ISR] = isr;
        while ((r[REG_ISR] & 1u) && ++n < 1000u) { }
    }
    r[REG_GIE] = 1u;
    s_lane[k].armed = 1;
    if (s_lane[k].expect_us && s_early_us >= 0) s_lane[k].t_start = now_s();
}

#if defined(__linux__)
/* The early wake: with the line armed, sleep until s_early_us before the
 * call's predicted end.  Returns 1 when it slept until the timer (the caller
 * then reads the ISR), 0 otherwise. */
static int early_sleep(lane_t *l)
{
    volatile uint32_t *r = l->regs;
    const double target = l->t_start + 1e-6 * ((double)l->expect_us - (double)s_early_us);
    if (target - now_s() < 1e-6 * EARLY_MIN_US) return 0;
    s_early++;
    for (;;) {
        struct pollfd p;
        struct timespec ts;
        uint32_t one = 1u, count;
        double rem;
        int rc;
        if (r[REG_ISR] & 1u) return 0;
        if (write(l->fd, &one, sizeof one) != (ssize_t)sizeof one) return 0;   /* the normal path copes */
        if (r[REG_ISR] & 1u) return 0;
        rem = target - now_s();
        if (rem <= 0.0) return 1;
        ts.tv_sec = (time_t)rem;
        ts.tv_nsec = (long)((rem - (double)ts.tv_sec) * 1e9);
        p.fd = l->fd;
        p.events = POLLIN;
        p.revents = 0;
        rc = ppoll(&p, 1, &ts, NULL);
        if (rc > 0) {                                   /* the line: the job ended, or a stale event */
            (void)!read(l->fd, &count, sizeof count);
            if (r[REG_ISR] & 1u) { s_early_line++; return 0; }
            continue;
        }
        if (rc == 0) { s_early_timer++; return 1; }
        return 0;
    }
}
#endif

int inference_irq_wait(unsigned k)
{
    volatile uint32_t *r;
    if (k >= LANES || !s_lane[k].armed) return 0;
    s_lane[k].armed = 0;
    r = s_lane[k].regs;
    s_waits++;
#if defined(__linux__)
    if (s_lane[k].expect_us && s_early_us >= 0 && !(r[REG_ISR] & 1u) && early_sleep(&s_lane[k])) {
        const double t0 = now_s();
        while (!(r[REG_ISR] & 1u) && now_s() - t0 < 1e-6 * (double)s_spin_us) { }
        s_early_spin_s += now_s() - t0;
    }
#endif
    if (!(r[REG_ISR] & 1u)) {                           /* a short job costs less than a wake-up */
        const double t0 = now_s(), spin = 1e-6 * (double)s_spin_us;
        while (!(r[REG_ISR] & 1u) && now_s() - t0 < spin) { }
    }
    if (r[REG_ISR] & 1u) {
        s_spun++;
    } else {
#if defined(__linux__)
        for (;;) {
            struct pollfd p;
            uint32_t one = 1u, count;
            int rc;
            /* Unmask, then read: a job that ended before the unmask holds the
             * line high and fires at once, so the sleep cannot miss it. */
            if (write(s_lane[k].fd, &one, sizeof one) != (ssize_t)sizeof one) {
                while (!(r[REG_ISR] & 1u)) { }          /* no interrupt on this device: poll */
                s_lane[k].regs = NULL;                  /* … from now on */
                close(s_lane[k].fd);
                break;
            }
            if (r[REG_ISR] & 1u) break;
            p.fd = s_lane[k].fd;
            p.events = POLLIN;
            p.revents = 0;
            s_sleeps++;
            rc = poll(&p, 1, POLL_TIMEOUT_MS);
            if (rc > 0) {
                (void)!read(s_lane[k].fd, &count, sizeof count);   /* consume the event */
                s_wakeups++;
            } else if (rc == 0) {
                s_timeouts++;
                if (r[REG_ISR] & 1u) { s_missed++; break; }
            }
            if (r[REG_ISR] & 1u) break;
        }
#else
        while (!(r[REG_ISR] & 1u)) { }
#endif
    }
    r[REG_GIE] = 0u;                                    /* the line low until the next armed Start */
    return 1;
}

void inference_irq_stats(inference_irq_stats_t *out)
{
    unsigned k;
    env_once();
    memset(out, 0, sizeof *out);
    out->mode = (unsigned)s_mode;
    out->spin_us = s_spin_us;
    for (k = 0u; k < LANES; k++)
        if (s_lane[k].regs) out->lanes |= 1u << k;
    out->waits = s_waits;
    out->spun = s_spun;
    out->sleeps = s_sleeps;
    out->wakeups = s_wakeups;
    out->timeouts = s_timeouts;
    out->missed = s_missed;
    out->early_us = s_early_us;
    out->early = s_early;
    out->early_timer = s_early_timer;
    out->early_line = s_early_line;
    out->early_spin_us = 1e6 * s_early_spin_s;
}

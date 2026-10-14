/*
 * inference_irq.h — kernel waits on the kernels' interrupts while the A53
 * drives them (Linux, UIO).
 *
 * Each kernel's UIO device (fabric_vecop, fabric_matmul, …, the overlay
 * dts/kv260/cormorant.dts) carries the kernel's PL interrupt, and the
 * uio_pdrv_genirq driver masks the line when it fires.  The record of a
 * completion is the kernel's own ISR bit 0 (ap_done, latched while IER bit 0
 * is set) — as on the Cortex-R5 (runtime/rpu/rpu_irq.c):
 *
 *   inference_irq_start(k)  before a Start: clear lane k's ISR and see the
 *                           clear land, raise GIE
 *   inference_irq_wait(k)   read the ISR for INFERENCE_IRQ_SPIN_US, then:
 *                           unmask the line (a write of 1 to /dev/uioN), read
 *                           the ISR again, sleep in poll() until the line
 *                           fires; repeat until the ISR shows the job done;
 *                           drop GIE
 *
 * The level line cannot be lost between the read and the sleep: it is high
 * from ap_done until the next Start's clear, so an unmask after the job ended
 * fires at once.  GIE is up only from a Start to the end of its wait, so the
 * line stays low while the R5 (which arms its own lanes) or a polling library
 * drives the kernel.
 *
 * The generated code calls these only while the A53 drives (not while the R5
 * records, inference_rpu.h).  The mode comes from INFERENCE_WAIT in the
 * environment ("irq" or "poll"; default INFERENCE_IRQ_MODE_DEFAULT), the spin
 * from INFERENCE_IRQ_SPIN_US (default INFERENCE_IRQ_SPIN_US_DEFAULT): a short
 * call costs less than a sleep and its wake-up.  A lane without a UIO
 * interrupt — bare metal, the host emulation, an older overlay — polls.
 */
#ifndef INFERENCE_IRQ_H
#define INFERENCE_IRQ_H

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_WAIT_POLL  0   /* kernel_wait() reads the done bit in a loop */
#define INFERENCE_WAIT_IRQ   1   /* spin, then sleep on the kernel's interrupt */

#ifndef INFERENCE_IRQ_MODE_DEFAULT
#define INFERENCE_IRQ_MODE_DEFAULT     INFERENCE_WAIT_IRQ
#endif
#ifndef INFERENCE_IRQ_SPIN_US_DEFAULT
#define INFERENCE_IRQ_SPIN_US_DEFAULT  300u
#endif

/* inference_init(), after kernel k's driver mapped its registers: open the
 * UIO device uio_name for the interrupt.  ctrl is the driver's register
 * window (Ctrl_BaseAddress).  0: the lane can sleep; -1: it polls. */
int  inference_irq_open(unsigned k, const char *uio_name, void *ctrl);
/* inference_deinit(): close the devices (the registers are left alone). */
void inference_irq_close(void);

/* The wait mode (INFERENCE_WAIT_POLL / _IRQ); returns the previous one.  It
 * applies from the next Start: a call in flight is waited for as it began. */
int      inference_irq_set_mode(int mode);
int      inference_irq_mode(void);
/* The spin before a sleep, in microseconds; returns the previous value. */
unsigned inference_irq_set_spin_us(unsigned us);

/* For the generated code: before a Start of kernel k (kernel_id_t) — arms the
 * lane when it waits on its interrupt; and in kernel_wait(): 1 when it waited
 * for an armed lane, 0 when the caller polls. */
void inference_irq_start(unsigned k);
int  inference_irq_wait(unsigned k);

typedef struct {
    unsigned           mode;        /* INFERENCE_WAIT_* */
    unsigned           spin_us;
    unsigned           lanes;       /* bit k: kernel k has its UIO interrupt */
    unsigned long long waits;       /* waits on armed lanes */
    unsigned long long spun;        /* … done within the spin */
    unsigned long long sleeps;      /* poll() calls */
    unsigned long long wakeups;     /* … that returned with the interrupt */
    unsigned long long timeouts;    /* … that timed out (long jobs; or a line that never fired) */
    unsigned long long missed;      /* timeouts after which the job was done: an interrupt lost */
} inference_irq_stats_t;
void inference_irq_stats(inference_irq_stats_t *out);

#ifdef __cplusplus
}
#endif
#endif /* INFERENCE_IRQ_H */

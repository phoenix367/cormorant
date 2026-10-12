/*
 * inference_rpu.h — drive the kernels from the Cortex-R5 (RPU_PLAN track B1).
 *
 * The firmware on the R5 (runtime/rpu/) knows no model: while it drives a
 * run, this library records every kernel call it would have issued — the
 * register block, the Start, the lane waits — into a command ring and
 * submits it in batches; the R5 replays them with interrupt-driven waits.
 * The weights, the DMA buffers, the host ops and the outputs stay on the
 * A53.  A batch ends where the host must see the kernels' results: before a
 * host op and before the output sync at the end of inference_run*()
 * (INFERENCE_RPU_FLUSH in the generated code), or when the ring is full.
 *
 * Who drives (src/rpu_policy.py, doc/plans/RPU_PLAN.md §9.3): the generated
 * inference_init() calls inference_rpu_select(), which attaches when the
 * firmware runs; each inference_run*() then follows its entry's
 * INFERENCE_RPU_RECOMMENDED — small models (few milliseconds, many calls
 * per millisecond) stay on the A53, large ones go to the R5.  The
 * environment overrides it: INFERENCE_RPU=0 never attaches, INFERENCE_RPU=1
 * puts every run on the R5.  Without the firmware nothing changes.
 *
 * Linux on the KV260 only (src/inference_rpu.c): the control block is the
 * RPU's BTCM through /dev/mem (rpu_ctrl.h), the ring a DMA buffer of this
 * library.  Elsewhere (the host emulation) the firmware is never found.
 */
#ifndef INFERENCE_RPU_H
#define INFERENCE_RPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned  lane;        /* RPU_LANE_* (rpu_ctrl.h): the firmware's lane of this kernel */
    unsigned  reg_words;   /* its argument registers from offset 0x10, in 32-bit words */
    uint64_t *ctrl_base;   /* the driver instance's Ctrl_BaseAddress (swapped to a shadow page) */
} inference_rpu_kernel_t;

/* Generated (inference.c): the active kernels, indexed by kernel_id_t. */
const inference_rpu_kernel_t *inference_rpu_kernels(unsigned *count);

#define INFERENCE_RPU_MODE_A53   0   /* attached, the A53 drives every run */
#define INFERENCE_RPU_MODE_AUTO  1   /* each run follows its entry's INFERENCE_RPU_RECOMMENDED */
#define INFERENCE_RPU_MODE_R5    2   /* the R5 drives every run */

/* The policy's attach (the generated inference_init() calls it): attach when
 * the firmware runs, in INFERENCE_RPU_MODE_AUTO (INFERENCE_RPU=1:
 * INFERENCE_RPU_MODE_R5; INFERENCE_RPU=0: no attach).  ring_bytes 0 = the
 * default (1 MiB).  Returns 1 attached, 0 not (no firmware, INFERENCE_RPU=0),
 * <0 the firmware is there but unusable (printed; the A53 drives). */
int  inference_rpu_select(unsigned ring_bytes);

/* Attach (if not yet) and put every run on the R5: INFERENCE_RPU_MODE_R5.
 * 0 on success; -1 no firmware (control block without its magic), -2 the
 * firmware in error / another protocol version, -3 a kernel of this project
 * the bitstream's firmware has no lane for, -4 no memory for the ring. */
int  inference_rpu_attach(unsigned ring_bytes);
/* Back to the A53 for good: the ring freed, the control block unmapped
 * (inference_deinit() calls it). */
void inference_rpu_detach(void);

/* Change the mode of an attached library (-1 when not attached). */
int  inference_rpu_set_mode(int mode);
/* The mode, or -1 when not attached. */
int  inference_rpu_mode(void);
/* 1 while the kernel calls go to the R5 (the current or last run's choice). */
int  inference_rpu_active(void);

/* For the generated code: the top of every inference_run*() — the run's
 * driver, from the mode and the entry's recommendation. */
void inference_rpu_enter(int recommended);
/* Submit what was recorded and wait for the R5 to finish it (no-op when
 * nothing is pending). */
void inference_rpu_flush(void);
/* The last flush's result: 0 ok, -1 the R5 did not answer in time, or the
 * firmware's rc (rpu_ctrl.h). */
int  inference_rpu_error(void);
/* Record a Start / a wait of kernel k (kernel_id_t). */
void inference_rpu_start(unsigned k);
void inference_rpu_wait(unsigned k);

typedef struct {
    uint32_t batches, calls, waits;   /* since the firmware started (every library) */
    uint32_t irq_count[8];            /* interrupts per RPU lane */
    uint32_t irq_missed[8];           /* waits served without an interrupt (should be 0) */
    uint32_t self_test_ok;            /* the firmware's software-interrupt self-test */
    uint32_t lanes;                   /* the firmware's lane mask */
    uint32_t flushes;                 /* this library's batches submitted */
    uint32_t words;                   /* command words submitted */
} inference_rpu_stats_t;
int  inference_rpu_stats(inference_rpu_stats_t *out);   /* -1 when not attached */

#ifdef __cplusplus
}
#endif
#endif /* INFERENCE_RPU_H */

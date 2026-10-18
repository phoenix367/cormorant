// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

/* rpu_irq.h — the firmware's kernel lanes and interrupt-driven waits (rpu_irq.c). */
#ifndef RPU_IRQ_H
#define RPU_IRQ_H

#include <stdint.h>
#include "xil_types.h"

struct rpu_ctrl;

/* The GIC, the fault handlers and every kernel lane the BSP names: their
 * interrupt enables (GIE off until rpu_irq_start), the control block's lane
 * mask and diagnostics.  0 on success. */
int rpu_irq_init(volatile struct rpu_ctrl *ctrl);
/* A lane's interrupt is armed before its first Start of a batch (call it
 * before writing ap_start) and disarmed at the batch's end unless a Start is
 * still in flight: the lines are shared with the APU, which must not lose its
 * own completions to this core. */
void rpu_irq_start(uint32_t lane);
void rpu_irq_batch_end(void);
/* Block until the lane's most recent Start completed (WFI between interrupts). */
void rpu_irq_wait(uint32_t lane);
/* Between commands: sleep in WFI until the library rings (IPI channel 7 -> this
 * core) — unless a command came already (cmd_seq != seen), the library cannot
 * ring (a53_rings) or this core has no interrupt path.  Returns at once then. */
void rpu_irq_idle(uint32_t seen);
UINTPTR rpu_lane_base(uint32_t lane);

#endif

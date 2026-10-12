/* rpu_irq.h — the firmware's kernel lanes and interrupt-driven waits (rpu_irq.c). */
#ifndef RPU_IRQ_H
#define RPU_IRQ_H

#include <stdint.h>
#include "xil_types.h"

struct rpu_ctrl;

/* The GIC, the fault handlers and every kernel lane the BSP names: their
 * interrupt enables (GIE off until rpu_irq_arm), the control block's lane
 * mask and diagnostics.  0 on success. */
int rpu_irq_init(volatile struct rpu_ctrl *ctrl);
/* The kernels raise their lines only while a batch runs (they are shared
 * with the APU, which must not lose its own completions to this core). */
void rpu_irq_arm(int on);
/* Block until the lane's most recent Start completed (WFI between interrupts). */
void rpu_irq_wait(uint32_t lane);
UINTPTR rpu_lane_base(uint32_t lane);

#endif

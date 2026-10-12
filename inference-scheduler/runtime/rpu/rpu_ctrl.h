/*
 * rpu_ctrl.h — the protocol between a generated library on the A53 (Linux,
 * src/inference_rpu.c) and the kernel-driver firmware on the Cortex-R5
 * (runtime/rpu/rpu_main.c, RPU_PLAN track B1).  The firmware knows no model:
 * it replays a command stream the library records — the register block of
 * each kernel call, its Start and the lane waits — with interrupt-driven
 * waits; the library keeps the weights, the buffers, the host ops and the
 * outputs.
 *
 * Control block: the first thing in the RPU's BTCM (rpu/lscript.ld: 0x20000
 * for the R5, 0xFFE20000 for the A53 through /dev/mem).  Every field is a
 * 32-bit word so the two views agree without padding games.
 *
 * Command ring: a DMA buffer the library allocates (DDR, below 4 GiB) and
 * names in ring_lo / ring_hi / ring_bytes.  A batch is written from word 0,
 * cleaned from the A53's cache, its length put in batch_words, and
 * cmd = RPU_CMD_BATCH with cmd_seq += 1 submits it; the firmware executes it
 * (arming the kernels' interrupts around it) and answers done_seq = cmd_seq.
 * Each command is a header word followed by its payload:
 *
 *   RPU_OP_REGS  | lane << 8 | n << 16   then n words: the kernel's registers
 *                                         from offset 0x10 up (the arguments)
 *   RPU_OP_START | lane << 8              write ap_start
 *   RPU_OP_WAIT  | lane << 8              block until that lane's ap_done
 *
 * The lanes are the bitstream's kernels in the registry order (RPU_LANE_*);
 * the firmware takes their bases and PL lines from the BSP's xparameters.h.
 */
#ifndef RPU_CTRL_H
#define RPU_CTRL_H

#include <stdint.h>

#define RPU_CTRL_MAGIC     0x52505531u          /* "RPU1" */
#define RPU_CTRL_VERSION   3u
#define RPU_CTRL_A53_ADDR  0xFFE20000u           /* the BTCM as the A53 sees it */

#define RPU_LANE_VECTOROP  0u
#define RPU_LANE_MATMUL    1u
#define RPU_LANE_CONV      2u
#define RPU_LANE_POOL      3u
#define RPU_LANE_ROWOPS    4u
#define RPU_LANES          5u

#define RPU_OP_REGS        1u
#define RPU_OP_START       2u
#define RPU_OP_WAIT        3u
#define RPU_CMD_OP(w)      ((w) & 0xFFu)
#define RPU_CMD_LANE(w)    (((w) >> 8) & 0xFFu)
#define RPU_CMD_N(w)       ((w) >> 16)
#define RPU_CMD_WORD(op, lane, n)  ((uint32_t)(op) | ((uint32_t)(lane) << 8) | ((uint32_t)(n) << 16))

enum rpu_state { RPU_STATE_BOOT = 0, RPU_STATE_READY = 2, RPU_STATE_RUNNING = 3,
                 RPU_STATE_ERROR = 4, RPU_STATE_QUIT = 5 };
enum rpu_cmd   { RPU_CMD_NONE = 0, RPU_CMD_BATCH = 2, RPU_CMD_QUIT = 3 };

struct rpu_ctrl {
    uint32_t magic;            /* RPU_CTRL_MAGIC once the firmware runs */
    uint32_t version;          /* RPU_CTRL_VERSION */
    uint32_t state;            /* enum rpu_state */
    uint32_t cmd;              /* enum rpu_cmd, written by the A53 */
    uint32_t cmd_seq;          /* the A53 increments after writing cmd and its arguments */
    uint32_t done_seq;         /* the firmware sets it to cmd_seq when the command completed */
    uint32_t rc;               /* the command's result (0 = ok; -1 unknown command, -5 bad ring,
                                * -6 unknown op or lane, -7 a lane this bitstream lacks) */
    uint32_t ring_lo;          /* the command ring: physical address and size (A53 -> R5) */
    uint32_t ring_hi;
    uint32_t ring_bytes;
    uint32_t batch_words;      /* the batch at ring[0 .. batch_words) */
    uint32_t batches;          /* batches executed */
    uint32_t calls;            /* Starts issued */
    uint32_t waits;            /* waits served */
    uint32_t lanes;            /* bit i set: lane i exists in this firmware's bitstream */
    uint32_t irq_count[8];     /* interrupts taken per lane */
    uint32_t irq_missed[8];    /* waits that found the kernel done without its interrupt (rpu_irq.c) */
    uint32_t heartbeat;        /* incremented while a wait spins (the firmware is alive) */
    uint32_t reserved0;
    uint32_t fault;            /* an exception the firmware took: 1 undefined, 2 prefetch abort, 3 data abort */
    uint32_t fault_addr;       /* its fault address (DFAR / IFAR) */
    uint32_t wait_lane;        /* the lane the current / last wait is on */
    uint32_t wait_isr;         /* the kernel's ISR register as the fallback last read it */
    uint32_t wait_checks;      /* fallback checks made */
    uint32_t gic[16];          /* the RPU GIC as the firmware reads it after its setup (rpu_irq.c):
                                * 0 dist CTLR, 1 ISENABLER3, 2 IGROUPR3, 3 ISPENDR3, 4 ISACTIVER3, 5 ICFGR7,
                                * 6 targets 120..123, 7 priorities 120..123, 8 cpu CTLR, 9 PMR, 10 RPR,
                                * 11 HPPIR, 12 the SGI self-test's count (1 = the interrupt path works),
                                * 13 TYPER */
    uint32_t progress;         /* the firmware's last step (rpu_main.c) */
    uint32_t reserved[71];
};                             /* 512 bytes */

#endif /* RPU_CTRL_H */

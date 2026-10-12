/*
 * rpu_irq.c — the kernel lanes of the Cortex-R5 firmware and their
 * interrupt-driven waits (RPU_PLAN §4.2).  Every kernel's ap_done interrupt
 * (its control block's GIE / IER, the PL line from xparameters.h) is routed
 * through the RPU's GIC; a wait sleeps in WFI until the lane's interrupt was
 * taken.  The command stream issues at most one wait per Start on a lane
 * (the generated code's event stream drains a lane before reusing it), so
 * the per-lane counter consumed here stays in step with the Starts.
 */
#include "rpu_ctrl.h"
#include "rpu_irq.h"
#include "xil_exception.h"
#include "xil_io.h"
#include "xparameters.h"
#include "xscugic.h"

#define CTRL_GIE      0x04u
#define CTRL_IER      0x08u
#define CTRL_ISR      0x0Cu

typedef struct { uint32_t id; UINTPTR base; uint32_t spi; } rpu_kernel_t;

/* The bitstream's kernels, as the BSP names them (a lane this bitstream lacks
 * is absent: the control block's lane mask tells the library). */
static const rpu_kernel_t s_kernels[] = {
#ifdef XPAR_XVECTOROPKERNEL_0_BASEADDR
    { RPU_LANE_VECTOROP, XPAR_XVECTOROPKERNEL_0_BASEADDR, XPAR_FABRIC_XVECTOROPKERNEL_0_INTR },
#endif
#ifdef XPAR_XMATMULKERNEL_0_BASEADDR
    { RPU_LANE_MATMUL,   XPAR_XMATMULKERNEL_0_BASEADDR,   XPAR_FABRIC_XMATMULKERNEL_0_INTR },
#endif
#ifdef XPAR_XCONVKERNEL_0_BASEADDR
    { RPU_LANE_CONV,     XPAR_XCONVKERNEL_0_BASEADDR,     XPAR_FABRIC_XCONVKERNEL_0_INTR },
#endif
#ifdef XPAR_XPOOLINGKERNEL_0_BASEADDR
    { RPU_LANE_POOL,     XPAR_XPOOLINGKERNEL_0_BASEADDR,  XPAR_FABRIC_XPOOLINGKERNEL_0_INTR },
#endif
#ifdef XPAR_XROWOPSKERNEL_0_BASEADDR
    { RPU_LANE_ROWOPS,   XPAR_XROWOPSKERNEL_0_BASEADDR,   XPAR_FABRIC_XROWOPSKERNEL_0_INTR },
#endif
};
#define N_KERNELS (sizeof s_kernels / sizeof s_kernels[0])

static XScuGic s_gic;
static volatile uint32_t s_done[RPU_LANES];
static volatile struct rpu_ctrl *s_ctrl;
static UINTPTR s_base[RPU_LANES];

UINTPTR rpu_lane_base(uint32_t lane) { return s_base[lane]; }

static void kernel_isr(void *ref)
{
    const rpu_kernel_t *k = (const rpu_kernel_t *)ref;
    const u32 isr = Xil_In32(k->base + CTRL_ISR);   /* ap_done (and ap_ready): toggle-on-write clear */
    if (isr) {
        Xil_Out32(k->base + CTRL_ISR, isr);
        (void)Xil_In32(k->base + CTRL_ISR);         /* the posted write lands before the EOI: the level drops */
    }
    /* (ap_ctrl is not read here: its ap_done bit is clear-on-read, and a poller
     * elsewhere — the Linux library sharing the kernels — would miss it.)
     * Only an ap_done counts: a second interrupt can follow the clear while the
     * level is still seen high, and it must not wake a wait early. */
    if (isr & 1u) s_done[k->id]++;
    if (s_ctrl) s_ctrl->irq_count[k->id]++;
}

static void fault_handler(void *ref)
{
    uint32_t far = 0u;
    if ((uintptr_t)ref == 3u) __asm__ volatile("mrc p15, 0, %0, c6, c0, 0" : "=r"(far));   /* DFAR */
    if ((uintptr_t)ref == 2u) __asm__ volatile("mrc p15, 0, %0, c6, c0, 2" : "=r"(far));   /* IFAR */
    if (s_ctrl) { s_ctrl->fault = (uint32_t)(uintptr_t)ref; s_ctrl->fault_addr = far; s_ctrl->state = RPU_STATE_ERROR; }
    for (;;) { }
}

static volatile uint32_t s_sgi_count;
static void sgi_isr(void *ref) { (void)ref; s_sgi_count++; }

/* The RPU GIC's state into the control block (the A53 cannot read 0xF9000000). */
static void gic_dump(volatile uint32_t *g)
{
    const UINTPTR d = s_gic.Config->DistBaseAddress, c = s_gic.Config->CpuBaseAddress;
    g[0] = Xil_In32(d + 0x000u);  g[1] = Xil_In32(d + 0x100u + 12u); g[2] = Xil_In32(d + 0x080u + 12u);
    g[3] = Xil_In32(d + 0x200u + 12u); g[4] = Xil_In32(d + 0x300u + 12u); g[5] = Xil_In32(d + 0xC00u + 28u);
    g[6] = Xil_In32(d + 0x800u + 120u); g[7] = Xil_In32(d + 0x400u + 120u);
    g[8] = Xil_In32(c + 0x0u); g[9] = Xil_In32(c + 0x4u); g[10] = Xil_In32(c + 0x14u); g[11] = Xil_In32(c + 0x18u);
    g[12] = s_sgi_count; g[13] = Xil_In32(d + 0x004u);
}

int rpu_irq_init(volatile struct rpu_ctrl *ctrl)
{
    XScuGic_Config *cfg;
    unsigned i;
    int rc;

    s_ctrl = ctrl;
    ctrl->lanes = 0u;
    for (i = 0u; i < N_KERNELS; i++) ctrl->lanes |= 1u << s_kernels[i].id;
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_UNDEFINED_INT, fault_handler, (void *)1u);
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_PREFETCH_ABORT_INT, fault_handler, (void *)2u);
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_DATA_ABORT_INT, fault_handler, (void *)3u);
    for (i = 0u; i < N_KERNELS; i++) s_base[s_kernels[i].id] = s_kernels[i].base;
    cfg = XScuGic_LookupConfig(XPAR_XSCUGIC_0_BASEADDR);   /* the SDT driver: by base address */
    if (!cfg) return -10;
    /* The driver only configures the distributor (groups, priorities, targets) when
     * it finds it disabled; a previous firmware leaves it enabled.  The RPU GIC is
     * this core's alone (core 1 is unused), so disable it first. */
    Xil_Out32(cfg->DistBaseAddress + 0x0u, 0u);
    /* The GIC keeps its state across a remoteproc stop / start: an interrupt a
     * previous firmware acknowledged but never completed stays active and holds
     * the running priority, which masks everything at or below it.  Complete and
     * clear every active interrupt, and drain the pending ones. */
    for (i = 0u; i < 6u; i++) {
        u32 act = Xil_In32(cfg->DistBaseAddress + 0x300u + 4u * i), b;
        for (b = 0u; b < 32u; b++)
            if (act & (1u << b)) Xil_Out32(cfg->CpuBaseAddress + 0x10u, 32u * i + b);   /* EOIR */
        Xil_Out32(cfg->DistBaseAddress + 0x380u + 4u * i, act);                          /* ICACTIVER */
    }
    for (i = 0u; i < 32u; i++) {
        const u32 iar = Xil_In32(cfg->CpuBaseAddress + 0x0Cu);
        if ((iar & 0x3FFu) == 1023u) break;
        Xil_Out32(cfg->CpuBaseAddress + 0x10u, iar);
    }
    rc = XScuGic_CfgInitialize(&s_gic, cfg, cfg->CpuBaseAddress);
    if (rc != XST_SUCCESS) return -11;
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT, (Xil_ExceptionHandler)XScuGic_InterruptHandler, &s_gic);
    for (i = 0u; i < N_KERNELS; i++) {
        const u32 id = s_kernels[i].spi + 32u;      /* a PL line's SPI number -> the GIC interrupt id */
        XScuGic_SetPriorityTriggerType(&s_gic, id, 0xA0u, 0x1u);   /* active-high level */
        /* The driver leaves the distributor alone when it is already enabled (a
         * previous firmware, the other core), so the SPI's target CPU is set here. */
        XScuGic_InterruptMaptoCpu(&s_gic, 0u, id);
        rc = XScuGic_Connect(&s_gic, id, kernel_isr, (void *)&s_kernels[i]);
        if (rc != XST_SUCCESS) return -12;
        {   /* clear a stale ap_done / ap_ready (toggle on write: only the set bits) */
            const u32 isr = Xil_In32(s_kernels[i].base + CTRL_ISR);
            if (isr) Xil_Out32(s_kernels[i].base + CTRL_ISR, isr);
        }
        Xil_Out32(s_kernels[i].base + CTRL_IER, 1u);   /* ap_done; GIE: inference_irq_arm() around a run */
        Xil_Out32(s_kernels[i].base + CTRL_GIE, 0u);
        XScuGic_Enable(&s_gic, id);
    }
    Xil_ExceptionEnable();
    /* Self-test of the interrupt path: a software interrupt to this CPU. */
    XScuGic_Connect(&s_gic, 15u, sgi_isr, NULL);
    XScuGic_Enable(&s_gic, 15u);
    XScuGic_SoftwareIntr(&s_gic, 15u, 1u);          /* target list: CPU 0 */
    for (i = 0u; i < 100000u && s_sgi_count == 0u; i++) { }
    gic_dump(ctrl->gic);
    return 0;
}

#define WAIT_TIMEOUT_SPINS  (20u * 1000u * 1000u)   /* ~1 s of the loop below */

/* The kernels raise their interrupt lines only while the firmware runs an
 * batch: the lines are shared with the APU (the Linux library may drive the
 * same kernels between batches, and its completions must not be consumed here). */
void rpu_irq_arm(int on)
{
    unsigned i;
    for (i = 0u; i < N_KERNELS; i++) {
        if (on) {
            const u32 isr = Xil_In32(s_kernels[i].base + CTRL_ISR);
            if (isr) Xil_Out32(s_kernels[i].base + CTRL_ISR, isr);     /* a stale ap_done of another master */
            s_done[s_kernels[i].id] = 0u;
        }
        Xil_Out32(s_kernels[i].base + CTRL_GIE, on ? 1u : 0u);
        (void)Xil_In32(s_kernels[i].base + CTRL_GIE);
    }
}

void rpu_irq_wait(uint32_t k)
{
    uint32_t spins = 0u;
    while (s_done[k] == 0u) {
        if (s_sgi_count)                            /* the interrupt path is proven: sleep until one arrives */
            __asm__ volatile("wfi" ::: "memory");
        s_ctrl->heartbeat++;
        /* Fallback (diagnostics): the kernel's ap_done in its ISR register without
         * the interrupt having been taken — count it, clear it, go on. */
        if (++spins > WAIT_TIMEOUT_SPINS) {
            spins = 0u;
            const u32 isr = Xil_In32(s_base[k] + CTRL_ISR);
            s_ctrl->wait_isr = isr; s_ctrl->wait_checks++;
            if (isr & 1u) {
                Xil_Out32(s_base[k] + CTRL_ISR, isr);
                s_ctrl->irq_missed[k]++;
                return;
            }
        }
    }
    s_done[k]--;
}

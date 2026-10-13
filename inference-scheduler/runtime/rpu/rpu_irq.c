/*
 * rpu_irq.c — the kernel lanes of the Cortex-R5 firmware and their
 * interrupt-driven waits (RPU_PLAN §4.2, §9.4).
 *
 * The record of a completion is the kernel's own ISR bit 0: every control
 * block (kernels/<k>_rtl/rtl/<k>_ctrl_s_axi.sv) latches ap_done there (IER bit 0
 * set) until a write toggles it.  rpu_irq_start() clears it — and sees the
 * clear land — while the lane is idle, before the ap_start on the same write
 * channel, so after a Start the bit means "this job is done".  A wait reads
 * the bit; between reads it sleeps in WFI with the lane's GIC line unmasked
 * and the CPU's IRQs held off (no wake-up can be lost between the read and
 * the WFI: a pending interrupt wakes WFI even while CPSR.I is set).  The
 * interrupt handler only masks the line at the GIC and returns — it counts
 * nothing and never writes a kernel register.
 *
 * (The first two designs counted completions in the handler: a toggle-clear
 * of the ISR with a posted write could be overtaken by its own read-back and
 * count one completion twice, and a stale count from the previous job's still
 * high line masked the real one — waits returned early or never; and the
 * driver's enable / disable rewrote the routing register three lanes share.
 * The 135M / 360M prefill and SmolVLM hangs of 2026-10-13.)
 *
 * A kernel raises its line only while the R5 drives it (GIE): the lines are
 * shared with the APU, which may drive the same kernels between batches.
 *
 * Between commands the core sleeps the same way (rpu_irq_idle, protocol v5):
 * the record is the control block's cmd_seq, the wake-up the library's ring —
 * IPI channel 7 triggering this core's channel 1 — and the handler again only
 * masks the line.
 */
#include "rpu_ctrl.h"
#include "rpu_irq.h"
#include "xil_exception.h"
#include "xil_io.h"
#include "xil_mpu.h"
#include "xparameters.h"
#include "xscugic.h"

#define CTRL_GIE      0x04u
#define CTRL_IER      0x08u
#define CTRL_ISR      0x0Cu
#define GICD_ISENABLER 0x100u
#define GICD_ICENABLER 0x180u

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
static UINTPTR s_dist;                      /* the GIC distributor */
static volatile struct rpu_ctrl *s_ctrl;
static UINTPTR s_base[RPU_LANES];
static u32 s_irq_id[RPU_LANES];             /* the lane's GIC interrupt id */
static uint32_t s_armed;                    /* lanes with GIE on */
static uint32_t s_inflight;                 /* lanes started and not yet waited for */

UINTPTR rpu_lane_base(uint32_t lane) { return s_base[lane]; }

/* The GIC enable bits are write-1 set / clear registers: no read-modify-write
 * (the driver's XScuGic_Enable / _Disable also rewrite the target register,
 * one word for four interrupt ids). */
static inline void gic_unmask(u32 id) { Xil_Out32(s_dist + GICD_ISENABLER + 4u * (id >> 5), 1u << (id & 31u)); }
static inline void gic_mask(u32 id)   { Xil_Out32(s_dist + GICD_ICENABLER + 4u * (id >> 5), 1u << (id & 31u)); }

/* A kernel's line: mask it (its ISR stays set until the lane's next Start, so
 * the level would re-fire) and let the waiting code read the ISR. */
static void kernel_isr(void *ref)
{
    const rpu_kernel_t *k = (const rpu_kernel_t *)ref;
    gic_mask(k->spi + 32u);
    __asm__ volatile("dsb" ::: "memory");          /* masked before the EOI */
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

/* The library's ring (IPI channel 7 -> channel 1): mask the line, as for a kernel;
 * rpu_irq_idle clears the ISR before it unmasks again. */
static void ipi_isr(void *ref)
{
    (void)ref;
    gic_mask(RPU_IPI_RPU0_GIC);
    __asm__ volatile("dsb" ::: "memory");
    if (s_ctrl) s_ctrl->r5_rings++;
}

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
    /* The BSP maps the whole PL window (0x8000_0000, 1 GiB) strongly ordered: every
     * register write then waits for its completion across the LPD-FPD path.  Device
     * memory keeps the writes to one control block in order but posts them; a read
     * back still waits.  The kernels' blocks share one 1 MiB window (0xA000_0000 +
     * 64 KiB each). */
    {
        UINTPTR lo = ~(UINTPTR)0, hi = 0u;
        for (i = 0u; i < N_KERNELS; i++) {
            if (s_kernels[i].base < lo) lo = s_kernels[i].base;
            if (s_kernels[i].base + 0x1000u > hi) hi = s_kernels[i].base + 0x1000u;
        }
        ctrl->regs_window = 0u;
        if (N_KERNELS && (lo & ~(UINTPTR)0xFFFFFu) == ((hi - 1u) & ~(UINTPTR)0xFFFFFu)) {
            const UINTPTR win = lo & ~(UINTPTR)0xFFFFFu;
            if (Xil_SetMPURegion(win, 0x100000u, DEVICE_SHARED | PRIV_RW_USER_RW) == XST_SUCCESS)
                ctrl->regs_window = (uint32_t)win;
        }
    }
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_UNDEFINED_INT, fault_handler, (void *)1u);
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_PREFETCH_ABORT_INT, fault_handler, (void *)2u);
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_DATA_ABORT_INT, fault_handler, (void *)3u);
    for (i = 0u; i < N_KERNELS; i++) {
        s_base[s_kernels[i].id] = s_kernels[i].base;
        s_irq_id[s_kernels[i].id] = s_kernels[i].spi + 32u;   /* a PL line's SPI -> the GIC id */
    }
    cfg = XScuGic_LookupConfig(XPAR_XSCUGIC_0_BASEADDR);   /* the SDT driver: by base address */
    if (!cfg) return -10;
    s_dist = cfg->DistBaseAddress;
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
        /* the target CPU, once (the driver skips it when the distributor was enabled) */
        XScuGic_InterruptMaptoCpu(&s_gic, 0u, id);
        rc = XScuGic_Connect(&s_gic, id, kernel_isr, (void *)&s_kernels[i]);
        if (rc != XST_SUCCESS) return -12;
        {   /* clear a stale ap_done / ap_ready (toggle on write: only the set bits) */
            const u32 isr = Xil_In32(s_kernels[i].base + CTRL_ISR);
            if (isr) Xil_Out32(s_kernels[i].base + CTRL_ISR, isr);
        }
        Xil_Out32(s_kernels[i].base + CTRL_IER, 1u);   /* ap_done latches into ISR bit 0 */
        Xil_Out32(s_kernels[i].base + CTRL_GIE, 0u);   /* the line: rpu_irq_start() .. batch end */
        gic_mask(id);                                   /* unmasked by a wait */
    }
    /* The library's ring: only channel 7 as a source of this core's IPI channel. */
    Xil_Out32(RPU_IPI_RPU0_IDR, ~RPU_IPI_CH7_BIT);
    Xil_Out32(RPU_IPI_RPU0_ISR, RPU_IPI_CH7_BIT);
    Xil_Out32(RPU_IPI_RPU0_IER, RPU_IPI_CH7_BIT);
    XScuGic_SetPriorityTriggerType(&s_gic, RPU_IPI_RPU0_GIC, 0xA0u, 0x1u);
    XScuGic_InterruptMaptoCpu(&s_gic, 0u, RPU_IPI_RPU0_GIC);
    rc = XScuGic_Connect(&s_gic, RPU_IPI_RPU0_GIC, ipi_isr, NULL);
    if (rc != XST_SUCCESS) return -13;
    gic_mask(RPU_IPI_RPU0_GIC);                         /* unmasked by an idle sleep */
    ctrl->r5_sleeping = 0u;
    ctrl->r5_sleeps = 0u;
    ctrl->r5_rings = 0u;
    Xil_ExceptionEnable();
    /* Self-test of the interrupt path: a software interrupt to this CPU. */
    XScuGic_Connect(&s_gic, 15u, sgi_isr, NULL);
    XScuGic_Enable(&s_gic, 15u);
    XScuGic_SoftwareIntr(&s_gic, 15u, 1u);          /* target list: CPU 0 */
    for (i = 0u; i < 100000u && s_sgi_count == 0u; i++) { }
    gic_dump(ctrl->gic);
    return 0;
}

void rpu_irq_start(uint32_t lane)
{
    const uint32_t bit = 1u << lane;
    const UINTPTR b = s_base[lane];
    /* The lane is idle: clear its previous ap_done (this core's, or another
     * master's), and see the clear land — after it, ISR bit 0 can only be this
     * job's.  (The ap_start that follows is on the same write channel, so it lands
     * after the clear anyway; the read-back makes the wait's first read safe too.) */
    const u32 isr = Xil_In32(b + CTRL_ISR);
    if (isr) {
        unsigned n = 0u;
        Xil_Out32(b + CTRL_ISR, isr);
        __asm__ volatile("dsb" ::: "memory");
        while ((Xil_In32(b + CTRL_ISR) & 1u) && ++n < 1000u) { }
    }
    if (!(s_armed & bit)) {
        Xil_Out32(b + CTRL_GIE, 1u);
        s_armed |= bit;
    }
    s_inflight |= bit;
}

/* A lane started and not waited for keeps GIE across the batch's end (a host op
 * between two batches does not need every lane's result); the others drop it,
 * so the APU may drive them. */
void rpu_irq_batch_end(void)
{
    uint32_t idle = s_armed & ~s_inflight, lane;
    for (lane = 0u; idle; lane++, idle >>= 1)
        if (idle & 1u) {
            Xil_Out32(s_base[lane] + CTRL_GIE, 0u);
            __asm__ volatile("dsb" ::: "memory");
            (void)Xil_In32(s_base[lane] + CTRL_GIE);   /* landed before the A53 may drive it */
        }
    s_armed &= s_inflight;
}

void rpu_irq_wait(uint32_t k)
{
    const UINTPTR isr = s_base[k] + CTRL_ISR;
    const u32 id = s_irq_id[k];
    s_ctrl->wait_lane = k;
    if (!s_sgi_count) {                             /* no interrupt path: read the ISR */
        while (!(Xil_In32(isr) & 1u)) s_ctrl->wait_checks++;
        s_ctrl->irq_missed[k]++;                    /* (counts the waits served by reading) */
    } else {
        for (;;) {
            if (Xil_In32(isr) & 1u) break;
            __asm__ volatile("cpsid i" ::: "memory");   /* no IRQ between the check and the WFI */
            gic_unmask(id);
            __asm__ volatile("dsb" ::: "memory");
            if (Xil_In32(isr) & 1u) { __asm__ volatile("cpsie i" ::: "memory"); break; }
            __asm__ volatile("wfi" ::: "memory");       /* a pending IRQ wakes it, I set or not */
            __asm__ volatile("cpsie i" ::: "memory");   /* the handler runs here: masks the line */
            s_ctrl->heartbeat++;
        }
    }
    s_ctrl->wait_isr = Xil_In32(isr);
    s_inflight &= ~(1u << k);
}

void rpu_irq_idle(uint32_t seen)
{
    if (!s_sgi_count || s_ctrl->a53_rings != RPU_RINGS_MAGIC) return;   /* keep polling */
    __asm__ volatile("cpsid i" ::: "memory");       /* no IRQ between the checks and the WFI */
    Xil_Out32(RPU_IPI_RPU0_ISR, RPU_IPI_CH7_BIT);   /* a ring already served: cmd_seq is read below */
    s_ctrl->r5_sleeping = 1u;
    __asm__ volatile("dsb" ::: "memory");           /* the library sees it, or this sees its cmd_seq */
    if (s_ctrl->cmd_seq == seen && s_ctrl->a53_rings == RPU_RINGS_MAGIC) {
        gic_unmask(RPU_IPI_RPU0_GIC);
        __asm__ volatile("dsb" ::: "memory");
        if (s_ctrl->cmd_seq == seen) {
            s_ctrl->r5_sleeps++;
            __asm__ volatile("wfi" ::: "memory");   /* a pending IRQ wakes it, I set or not */
        }
    }
    s_ctrl->r5_sleeping = 0u;
    __asm__ volatile("cpsie i" ::: "memory");       /* the handler runs here: masks the line */
}

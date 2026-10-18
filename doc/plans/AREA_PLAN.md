# The kernels' LUTs — an area pass over the five kernels (2026-10-14)

**Status (2026-10-15): done — the FOLD design routes and was in production from 2026-10-15 to 2026-10-16 (its successors `33a799febc56` and, since 2026-10-17, the production `04259a7af917` build on it).**  Bitstream `de968ccc45da` (`/mnt/data/bitstreams/kv260_250_de968ccc45da/`): WNS +0.029 ns, WHS +0.010 ns at 4 ns, 89 547 LUTs (76.5 %) — 4 422 below the previous production `1cedaf2ed816` and 8 182 below the FOLD build that did not route — 108 713 FF, 136 BRAM tiles, 55 URAM, 841 DSP (§5).  The block-design simulation 112 / 112; on the board every model and library bit-exact and the 72 kernel benchmarks equal to `1cedaf2ed816`'s within the run-to-run noise (HOSTOPS_PLAN §11.7).  Steps 5's replay file and 6 stay open for the next design that needs room.

**History (2026-10-14):**  The design with the row-ops unit's
FOLD op (HOSTOPS_PLAN §11) no longer routes.  It places at 97 729 LUTs
(83.4 %) with 99.9 % of the CLBs occupied.  The router reports congestion
around the ConvKernel engine and the row-ops unit, and stays at WNS −2.18 ns
after its first rip-up iteration.  The production design closed 4 ns by
+0.020 ns at 93 969 LUTs (80.2 %).  The user asked for a LUT diet of the
kernels, Conv and Matmul first, at equal throughput, rather than a narrower
FOLD.  The goal is to land below the production LUT count with FOLD in,
then build and gate the bitstream.

## 1. Where the LUTs are (the placed FOLD design)

| kernel | LUTs | logic | LUTRAM | SRL | FF |
|---|---:|---:|---:|---:|---:|
| RowOpsKernel | 34 618 | 30 476 | 3 706 | 436 | 27 793 |
| ConvKernel | 18 272 | 12 124 | 3 758 | 2 390 | 39 733 |
| MatmulKernel | 16 341 | 9 427 | 6 892 | 22 | 15 278 |
| PoolingKernel | 12 358 | 8 912 | 3 404 | 42 | 11 827 |
| VectorOPKernel | 11 760 | 11 269 | 304 | 187 | 13 752 |
| interconnects, SmartConnect, register slices | ≈ 4 400 | | | | ≈ 8 600 |

The repeated building blocks cost the most:
- **FIFOs: 11.7 k LUTs over 54 instances.**  The 12 block-RAM FIFOs (Conv 2,
  Matmul 3, VectorOP 3, RowOps 3, Pool 1) each spend 300–560 LUTs and
  420–800 FFs on a three-register skid stage.
- **Burst / run generators: ≈ 5.5 k.**  60-bit word-address arithmetic.
- **AXI-Lite control slaves: ≈ 3.2 k.**  Every argument register is written
  through a byte-merge mux instead of the flip-flops' clock enables.

## 2. Steps

| step | what | kernels | measured (out of context, 3.333 ns) |
|---|---|---|---|
| 1 | the block-RAM FIFO's skid stage as a 4-entry LUTRAM ring (three used, the same read rule and latency) | all five | VectorOP −589 LUT / −1 155 FF, Matmul −545 / −1 209, Pool −194 / −402, RowOps −662 / −1 772, Conv −767 / −1 515: ≈ 200 LUTs and 400 FFs per FIFO |
| 2a | control registers written with per-byte clock enables (no merge mux) | all five | the control slaves −215 … −306 LUTs each (VectorOP 493 → 278, Matmul 544 → 297, Pool 653 → 347) |
| 2b | the address arithmetic at 48-bit byte / 44-bit word addresses (the PS ports' physical space is 40 bits; SMMU IOVAs 48); the AXI address outputs stay 64-bit, zero-extended | all five | 2a + 2b: VectorOP −425, Matmul −795, Pool −338; Matmul's run-generator paths that step 1 had pushed to −0.316 ns are back at +0.186 ns |
| 3 | Matmul: one 128-bit accumulator LUTRAM per half-group of 4 rows (shared addresses); one-entry AR / AW slices (the engines issue every other cycle at most) | Matmul | 15 001 → 14 460 LUT (−541), WNS +0.115 ns; the perf shapes' cycles unchanged |
| 4 | Conv: the patch beat mask in the line-buffer RAMs' output-register reset (no 512-bit AND) | Conv | with 2b: 17 505 → 16 598 LUT (−907), WNS +0.100 ns |
| 5 | Conv: the weight skew in the RAM addresses and the DSPs' A-port registers instead of SRLs (≈ −1.9 k SRL); the replay file served from the patch FIFO (≈ −1.1 k) | Conv | the skew: done (the DSP48E2 check 199 899 cycles, 0 mismatches; every perf shape and six weight-heavy MatMul-on-conv / FC shapes take the same cycles as before — the later bank release never binds); the replay file: open |
| 6 | Matmul: 3-bit handshake counters; the B chunk base walked once in the walker; the A buffer read at 16 bits | Matmul | planned |

The audits behind steps 3–6 are §3.  Every step keeps the kernels'
throughput: the ideal-memory cycle counts of each kernel's `--perf` shapes
are unchanged (Matmul's GEMM 64 × 576 × 576: 172 663 cycles, 172 665 before).

## 3. The Conv and Matmul audits (read-only reviews of the RTL, 2026-10-14)

**ConvKernel, ≈ 5.5–6.5 k LUTs in 12 changes:**
- **Weight skew (−1.7 … −1.9 k SRL, +1.8 k FF).**  `cv_engine` delays weight
  lane l by l + 1 cycles through SRLs: 16 columns × 15 lanes × 16 bits.
  Each column's cache is already four 64-bit RAMs, one per 4-lane group.
  Instead, delay group k's read address by 4k cycles, and let the DSP's
  A-port registers (AREG 2, the AD register) take the rest.  The patch then
  moves to the B port.
- **Replay file (−1.07 k).**  The two 64 × 256-bit LUTRAMs of m-tiles 1 …
  G − 1 go away; the patch FIFO serves those reads.
- **Beat mask (−300 … −400).**  Step 4.
- **Seed mux folded into `sw5` (−300).**
- **Bias RAM (−850 in 4 URAM, or −144 trimmed).**
- **The drain's rotation merged into the y writer's funnel shift (−250).**
- **Write-back pixel mux in DSPs (−350).**
- **The emitter's column arithmetic narrowed (−200).**
- **Control widths (−150) and the weight loader's push mux (−64 … −128).**
- **Not worth it:** the x row buffers (already RAM32M16-dense) and the patch
  FIFO in UltraRAM (8 URAM).

**MatmulKernel, ≈ 2.2 k LUTs in 7 changes:**
- **Handshake counters, 32 → 3 bits (−380).**
- **Control writes (−270).**  Step 2a.
- **Accumulator packing (−256).**  Step 3.
- **The A buffer read as 16 bits (−384).**
- **One-entry AR / AW slices (−250).**  Step 3.
- **The B chunk base in the walker (−370).**
- **The drain's lane sum in DSPs (−256).**
- **Inherent:** the accumulators (5.1 k LUTRAM).  64 MACs read and write a
  32-bit word every cycle, beyond any free block RAM width.  Halving their
  depth would cost throughput.

## 4. Verification of every step

- Each kernel's Verilator suite (`ctest -R Rtl`: fixtures and random jobs),
  plus extra random seeds at slow memory timing.
- Lint, and the drivers' register tables.
- The ideal-memory cycle counts (`--perf`).
- Out-of-context synthesis at 300 MHz (WNS ≥ 0).
- Then the in-context bitstream and the board gates of HOSTOPS_PLAN §11.6.

The testbenches place their buffers above 4 GiB (bit 35), so the 48-bit
address arithmetic is exercised across the 32-bit boundary.  They write the
control registers with all byte strobes; the per-byte enables are a
one-line `for` per register.

## 5. Results in context (bitstream `de968ccc45da`, 2026-10-15)

| | FOLD, before the pass | after the pass | production `1cedaf2ed816` |
|---|---:|---:|---:|
| LUTs (placed) | 97 729 (83.4 %) | **89 547 (76.5 %)** | 93 969 (80.2 %) |
| FFs | 116 378 | 108 713 | 113 020 |
| CLBs occupied | 14 627 (99.9 %) | 14 573 (99.5 %) | 14 618 (99.85 %) |
| DSP / BRAM tiles / URAM | 841 / 136 / 55 | 841 / 136 / 55 | 833 / 136 / 55 |
| routing | congested, WNS −2.18 ns after the first rip-up iteration (stopped) | WNS +0.029 ns, WHS +0.010 ns (6 rip-up iterations: −0.883 → −0.309 → −0.169 → −0.028 → +0.029) | +0.020 / +0.010 ns |

Per kernel (routed, `report_utilization -hierarchical`):

| kernel | LUTs before → after | of which |
|---|---:|---|
| ConvKernel | 18 272 → **14 827** (−3 445) | SRL 2 390 → 512 (the weight skew in the RAM addresses and the DSPs' A registers), FF 39 733 → 37 497 |
| MatmulKernel | 16 341 → **14 453** (−1 888) | LUTRAM 6 892 → 6 656 (the packed accumulators) |
| RowOpsKernel | 34 618 → **33 340** (−1 278) | |
| VectorOPKernel | 11 760 → **10 768** (−992) | |
| PoolingKernel | 12 358 → **11 856** (−502) | |

The post-synthesis count fell from 101 447 to 91 630 LUTs (−9 817).

**A build-script trap found on the way** (`hw/cormorant_hw_128/scripts/build.tcl`): an
implementation run that is killed mid-way stays "running" in the project and silently
blocks `reset_run synth_1`; the script then saw synthesis at 100 % and implemented the
old netlist twice (the same placement checksums and WNS as the build before).  It now
resets `impl_1` first and stops with an error when synthesis is still out of date; the
stale run directories were moved aside to start from clean runs.


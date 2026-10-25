# The kernels' LUTs — an area pass over the five kernels (2026-10-14)

**Status (2026-10-25): a second pass (§6) cuts the five kernels by 11.8 k LUTs and 11.6 k FFs at equal results and cycles; bitstream `89205c11fded` routes at 79 986 LUTs (68.3 %), WNS +0.031 ns, and on the board is bit-exact (159 / 159 models, the demos, Piper, SmolLM2-135M) at production's speed (§6.4) — not promoted.  Its first build `9446b3d462b6` failed MatmulKernel on the board: Vivado dropped a DSP reset (§6.4).**

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


## 6. The second pass (2026-10-24 / 25)

The production design `04259a7af917` (the row-ops unit's GATE and SCALE,
RPU_PLAN §14) placed at 91 805 LUTs (78.4 %) with 99.75 % of the CLBs
occupied, and closed 4 ns by +0.027 ns after nine rip-up iterations.  The
user asked for less resource use without any slowdown.  Five read-only
audits of the RTL (RowOps in two parts, Pool, VectorOP, and the open items
of §3 for Conv and Matmul) proposed the changes; each was kept only if it
held all of these:
- every result bit unchanged (each testbench's C++ oracle; the fixtures,
  the random jobs and extra seeds at random and slow memory timing);
- every cycle count unchanged: the per-case cycles of the fixtures and the
  random jobs at ideal memory and of the `--perf` shapes, compared case by
  case with the unmodified kernels;
- the register maps and the AXI behaviour unchanged;
- out-of-context WNS at 3.333 ns not below 0, and no extra BRAM or URAM.

### 6.1 What changed

- **RowOps** (`ro_pkg`, `ro_stream`, `ro_norm`): `rne_a` shifts 2x right in
  three 4:1 steps, its sticky bit the OR of the bits each step drops (no
  64-bit mask subtraction; 16 instances, −3.4 k LUT alone, and the sticky
  paths 13 → 5 levels); one rounding / saturation unit per stream lane
  (COPY / ADDX's sum carried to s17, rounded with MULACT / GATE / SCALE);
  the 32-bit gv shifters replaced by a word of the 64-bit shifts; MULACT's
  σ from the B-side table; NORM's parameters read 6 cycles after issue
  instead of carried, Horner n·S2, 89-bit scalar registers, P1 at 33 bits
  (−8 DSP).  (`ro_core`, `ro_fold`, `ro_xpose`, `ro_rope`, the ports,
  `ro_ctrl_s_axi`, `ro_param`): one output FIFO for ROPE / FOLD /
  TRANSPOSE; FOLD's sequencer at 33–47 bits; TRANSPOSE / FOLD write
  addresses at 48 / 44 bits; the burst generators' window and tile levels
  only where a geometry uses them; ROPE's cos / sin in four lane banks;
  descriptor beats written as taken and read asynchronously; the control
  slave's read-back from a LUTRAM shadow; one byte shifter in TRANSPOSE.
- **Conv**: the seed fold (one more register on the engine's accumulator
  read address, `acc_r3` gone); the drain's rotation onto the y alignment,
  so the y writer only picks each lane from the current or previous word;
  11-bit channel counts (`CHB`); one divider for the two line-buffer
  divisions; no column masks (lanes ≥ out_ch are don't-care: every word's
  first sweep seeds from the bias and the drain emits channels < mv only);
  `j` as wires from the frozen `q_*`; the bias RAM in two LUTRAM banks of
  64 + 16.
- **Pool**: 7-bit emitter columns; the finaliser's shifts narrowed; the
  writer stores each row at its DDR lane (no 8:1 rotation in the drain);
  `j` as wires; the beat padding through `rv`'s synchronous reset; MAX and
  the sums on one add chain.
- **VectorOP**: the softmax's Cs and P shifters keep the bits used (17 /
  15 and a flag); its 16 reciprocals in a LUTRAM; `vo_div` non-restoring
  (checked over every |a| << 8 × |b|); no block level in the b and c burst
  generators; LeakyReLU's rounding in the DSP (P = x·alpha + 2^15, a tie
  clears bit 16) and the activation's select merged into stage 5.
- **Matmul**: the handshake counters modulo 2^HS (HS = 4); the walker counts
  down and carries the B chunk base, which the run generators no longer
  compute; the drain's lane sum in one DSP48E2 per beat lane (+8 DSP; the
  tap sums kept in fabric — a DSP after it chained two ALUs in a cycle and
  missed 300 MHz by 0.16 ns; `mm_lane`'s drain register `keep`, §6.4).

Not done: Conv's replay file served from the patch FIFO (built cycle-exact,
≈ −20 LUT net with a 128-deep FIFO, WNS +0.017: reverted); Conv's weight
loader push mux (no gain, and it would rely on `cv_patch` zeroing lanes:
reverted); Matmul's A buffer read at 16 bits (needs an asymmetric block RAM)
and its one-adder run generator (on its worst paths); VectorOP's shared
softmax accumulators and ADD / SUB in the DSP pre-adder; RowOps' tile buffer
in RAMB18 (−2 k LUTRAM for 4 more BRAM tiles); Pool's lane contribution in
an instantiated DSP.

### 6.2 Results

Out of context (place and route at 3.333 ns, the production RTL in the same
flow):

| kernel | LUT | FF | DSP | WNS |
|---|---:|---:|---:|---:|
| RowOpsKernel | 35 323 → **27 150** (−8 173) | 28 667 → 22 290 | 133 → 125 | −0.027 → +0.002 |
| ConvKernel | 14 973 → **13 749** (−1 224) | 37 793 → 36 187 | 518 | +0.026 → +0.001 |
| VectorOPKernel | 10 746 → **9 570** (−1 176) | 12 196 → 10 721 | 35 | +0.221 → +0.210 |
| PoolingKernel | 11 826 → **10 652** (−1 174) | 11 572 → 10 481 | 29 | +0.145 → +0.184 |
| MatmulKernel | 14 460 → **13 954** (−506) | 13 329 → 11 904 | 130 → 138 | +0.115 → +0.006 |

In context, bitstream `89205c11fded` (`/mnt/data/bitstreams/kv260_250_89205c11fded/`):

| | production `04259a7af917` | `89205c11fded` |
|---|---:|---:|
| LUTs (placed) | 91 805 (78.4 %) | **79 986 (68.3 %)** |
| FFs | 111 099 | 99 469 |
| CLBs occupied | 14 604 (99.75 %) | 14 223 (97.2 %) |
| CARRY8 | 3 260 | 2 528 |
| DSP / BRAM tiles / URAM | 845 / 136 / 57 | 845 / 136 / 57 |
| routing | WNS +0.027 ns, nine rip-up iterations | WNS +0.031 ns, WHS +0.010 ns, two rip-up iterations |

Per kernel (routed, `report_utilization -hierarchical`): RowOps 35 337 →
27 400, Conv 15 120 → 13 689, VectorOP 10 771 → 9 562, Pool 11 837 →
10 697, Matmul 14 447 → 14 008 LUTs.  Post-synthesis 82 300 LUTs; no
`Synth 8-4767`.  (The first build, `9446b3d462b6` without the `keep`
of §6.4: 79 953 LUTs, 99 236 FFs, WNS +0.059 ns.)

Verification: each kernel's Verilator suite (`ctest -R Rtl`, 13 tests) and
lint; ≥ 2 extra seeds × random and slow timing per kernel (1 000–2 000
random jobs each, RowOps with 300 chains); the per-case cycle compare of all
five kernels identical; `neteq_matmul_rtl` (20 jobs since §6.4),
`neteq_vectorop_rtl`, `neteq_pool_rtl` 0 errors; a sweep of every softmax Cs × f_p in both modes;
LeakyReLU over every Q8.8 input at seven slopes; ConvRtlDsp; the block-design
simulation (`sim_hw_kv260`) 123 / 123.

### 6.3 Found on the way (not changed)

- **ROPE can hang on inputs outside its contract** (`B_EN = 0`, `HD < 2`,
  or `n mod HD ≠ 0`): it waits for B words that never come, or leaves A
  words in port A.  The testbench never makes these jobs; ROWOPS_RTL_KERNEL
  §1.4 promises no hang.
- **A pool W-tile wholly in the padding** (pads ≥ the window, or outputs
  past the input) wraps `run_len`: wrong values, bursts across 4 KiB, often
  a hang (235 of 400 such jobs), in the production RTL as well; the C++
  model returns garbage there too.  Models within ONNX Runtime's rule (pads
  < kernel) never reach it.

### 6.4 On the board (2026-10-25)

**The first build failed.**  On `9446b3d462b6`, 120 of the 159 models of
`run_remote_tests.py` passed; each of the 39 failures runs a MatMul on
MatmulKernel, its outputs offset by stale accumulator words (+19 LSB on
the first row of a block, up to ±10 on whole columns, the same garbage
across unrelated models).  The drain's lane-sum DSPs (§6.1) zero an idle
lane through the reset of their input registers (`ab5` → A2 / B2 with
RSTA / RSTB, `c5` → C with RSTC).  Vivado also absorbed `mm_lane`'s
reset-less `drd_data` as A1 / B1 (AREG = BREG = 2), and as A1 / A2 share
RSTA (B1 / B2 RSTB) it tied both resets to 0: when lane 0 had no K block,
its accumulators from an earlier job were added.  Verilator, the cycle compare,
`sim_hw_kv260` and `neteq_matmul_rtl` all passed — the RTL is right, and
the netlist check's 14 jobs never ran a K ≤ 16 job after two-lane jobs.
Its job list now does (jobs 7–9 and 17–19): 39 mismatches on that RTL's
netlist, 0 with the fix, `(* keep = "true" *)` on `drd_data` (AREG = BREG = 1,
RSTA / RSTB connected; `(C'+A':B')'` in the bitstream's synthesis).  A
read-only audit of every other DSP of the five kernels whose input register
is reset or zeroed (Pool's `pl_reduce` / `pl_core`, VectorOP's `vo_act` /
`vo_compute` / `vo_smx`, RowOps' `ro_norm` / `ro_stream` / `ro_fold`, Conv's
instantiated MAC chains and control DSPs) found no second case.  Note that
`run_remote_perf.py`'s cases check that a call runs, not its result.

**`89205c11fded`** (the fix; same day, production `04259a7af917` as the
control on the same boot, both loaded twice, interleaved):

| check | `89205c11fded` | production |
|---|---|---|
| `run_remote_tests.py` (159 models) | 159 / 159 | — |
| MNIST convnet / LeNet | 9 892 / 9 735 correct, 0.115 / 1.229 ms | same counts, 0.115 / 1.231 ms |
| MobileNet v1 / v2, ResNet-18 (top-5) | identical, 20.57 / 17.60 / 19.99 ms | 20.51 / 17.70 / 20.05 ms |
| BERT-SQuAD (50 examples) | 50 / 50 bit-exact, 267.1 ms | 50 / 50, 266.8 ms |
| Piper (`tts_board.py`) | PCM bit-exact; chunk 147.9 / 146.8 ms | 147.8 / 146.7 ms |
| SmolLM2-135M (`llm_board.py`, A53) | logits bit-exact 4 × 33; decode 47.17 ms/token, prefill 256: 399.1 ms | 47.14 ms/token, 399.7 ms |
| 72 kernel benchmarks (3 vs 5 runs) | geomean of the best latencies per kernel +0.1 … +0.7 %, no case beyond ±2 % | — |

Two 47 µs MatmulKernel cases, `batch4-64x64x64` and `batch4-A-bcast`,
measured 0.5–1.2 µs (1–2.6 %) above that day's production runs in every
run of the new bitstreams (0.0469–0.0478 against 0.0464–0.0470 ms), but
equal to production's own baseline of 2026-10-16 (0.0475 / 0.0478 ms).
The RTL is cycle-identical for exactly these geometries (and single-batch)
at ideal, random and slow memory, so it is not the kernels' logic.  No
baseline was recorded for `89205c11fded`; promoting it would need the
performance model (`perf-calibrate`) and the chat server's redeploy.

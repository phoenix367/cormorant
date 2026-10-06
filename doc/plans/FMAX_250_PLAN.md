# The whole design at 250 MHz: plan

**Status (2026-10-06):** done.  The four kernels, the interconnects and the
PS-PL AXI ports run at **250 MHz** (an MMCM in the block design); bitstream
`986cef4866a0` closes 4.000 ns with WNS +0.105 ns, is bit-exact on the board
everywhere (148 / 148 models, the demos, the chat / TTS gates) and is in
production: ResNet-18 47.5 → 20.5 ms, BERT 839 → 541 ms, SmolLM2-135M 10 →
18.3 tokens/s, Piper RTF 0.30 → 0.24 (§6).  Open: a refit of the engine
cost model at 250 MHz (§5, phase 4).

## 1. The problem

The routed design of bitstream `c2b2a6e5e50e` (hw_128 850cc88) meets its
10 ns constraint with WNS +1.956 ns: an 8.04 ns critical path, read as
"the design works at 124 MHz only", while each kernel alone closes at
3.333 ns (300 MHz) out of context.

## 2. Investigation — four hypotheses in parallel (2026-10-06)

| | hypothesis | how | result |
|---|---|---|---|
| A | the 124 MHz is an artefact of the 10 ns constraint: a timing-driven placer stops at positive slack | re-place and re-route the current netlist (post-opt checkpoint) at 4.000 ns (`place ExtraTimingOpt`, `phys_opt AggressiveExplore`, `route AggressiveExplore`, post-route `phys_opt`); a full re-synthesis with PL0 = 250 MHz, project strategy, then the same directives | **largely confirmed**: run 1 WNS **+0.004 ns**, 0 failing endpoints, WHS +0.010; the full flow with PL0 = 250 MHz WNS −0.061 ns with the project strategy, **0.000 ns** with the run-1 directives.  The design as it is closes 250 MHz — with no margin |
| B | the platform (clocking, interconnects, PS ports, resets) blocks 250 MHz | read the BD, the PS IP data, the board's clock tree, the routed checkpoint's reset and interconnect paths | no blocker: PS-PL AXI ports allow 333 MHz (−2LV), IOPLL / 4 gives 250 MHz exactly, an MMCM any value; no CDC.  But: the two data interconnects have **no register slices** (PS port ↔ crossbar ↔ kernel paths 4.5–5.8 ns at 10 ns), ConvKernel's reset is **unregistered** (one FF → 855 loads, 4.8 ns), MatmulKernel's and VectorOPKernel's job resets are combinational |
| C | ConvKernel's in-context nets are structurally too long | the routed checkpoint at 10 ns: every ConvKernel net with fanout ≥ 16, its loads' sites and bounding box | confirmed: the 48 URAMs fill the device's single URAM column over 4 clock regions, the 518 DSPs 10 columns over all 4 rows; the write-back control (528 loads, span 4.9 clock regions, 5.55 ns of route), the hold-register enables, the weight-cache write address / data, the seed / `wfb` controls, the bias write address and `urst` are one register and ≤ 1 LUT each, ≥ 94 % route |
| D | MatmulKernel / VectorOPKernel / PoolingKernel likewise | the same | MatmulKernel's operand fanout (`off_reg` → DSP A, 6.5 ns route), high-fanout LUT-RAM write addresses (Pool `u_em/waddr` 2 944 loads, Matmul `w3_reg` 1 440); see §3 |

Run 2's 14 failing endpoints (WNS −0.061 ns): MatmulKernel `g_port[1].u_gb`
`row_rem` → `w0` clock enables (7 levels, fanout 128, −0.061), VectorOPKernel's
`pm1 <= j_outer * nw` (`vo_core.sv:283`, a 32 × 32 multiply in two cascaded
DSPs plus carry logic: 3.4 ns of **logic**, 11 % route, −0.043 — the one path
placement cannot fix), MatmulKernel `u_xpf` `rptr` → DSP B (fanout 259,
−0.027), PoolingKernel `u_rd` → `u_wr` LUT-RAM write enable (7 levels,
−0.004).  Near-critical in runs 1 / 2b (slack < 0.3 ns, about 3 100–3 700
endpoints, median 88 % route): ConvKernel's weight-cache write into 16
URAMs (0.000 ns in 2b), the SmartConnect's AXI-Lite payload registers →
the kernels' control registers (distance, +0.001…0.045), MatmulKernel's
operand / tap selects into its DSPs and LUT-RAMs, PoolingKernel's line
buffer (`u_em/waddr`, 2 944 loads).  The PS ports, the crossbars and the
reset block never fail (≥ +0.012 ns); synthesis at 4 ns changes almost
nothing (same 688 DSP, 48 URAM, 103 BRAM, ~57 k LUT) — the strategy does.

What the 124 MHz figure measured: Vivado places and routes a design only as
well as its constraint asks.  At 10 ns the worst paths were one LUT and
7 ns of wire — a register in one clock region driving URAMs or DSPs four
regions away — because nothing asked for less.  At 4 ns the same netlist
meets timing, but the margin is zero, and the full flow (synthesis at
4 ns, project strategy) misses by 61 ps: the design closes by the tools'
effort, not by its structure.  An MMCM adds ~0.11 ns of jitter, and every
other change (a kernel, the scheduler's IP upgrade) re-rolls the
placement.  The fixes therefore aim at structural margin, following the
Model Composer guide's practice (UG1483 pp. 72–75: register the I/O, pipeline,
a register before an SRL, RAM output registers, DSP input / pipeline
registers; pp. 537–538: fanout pipeline registers per DSP column —
`Data_Path_Fanout`, `Control_Column_Fanout`, "Inter-Column Pipe Length").

## 3. Fixes

**Clock (hw_128 BD, `scripts/bd_kernel_clock.tcl`).**  An MMCM
(`clk_wiz_0`, clk_wiz 6.0, no input buffer) fed by `pl_clk0` at the boot
firmware's 100 MHz drives everything that was on `pl_clk0` (the four
kernels, both interconnects, the SmartConnect, `maxihpm0_fpd_aclk`,
`saxihpc0/1_fpd_aclk`, the reset block): 249.9975 MHz, 110 ps jitter.
`pl_resetn0` resets the MMCM, its `locked` holds `rst_ps8_0_99M`
(`dcm_locked`).  Chosen over setting PL0 to 250 (IOPLL / 4) because the
bitstream then defines its own clock: no loader or overlay sets PL0, and a
stale PL0 setting can neither slow a 250 MHz bitstream nor overclock a
100 MHz one; the bitstream id implies the clock (the performance models
are keyed by it).  The loader checks that `fclk0` is at 100 MHz.

**Interconnects (same script).**  Register slices ("Outer") on the nine
SIs and the MI of `axi_interconnect_0` (→ S_AXI_HPC0_FPD) and the three
SIs of `axi_mem_intercon` (→ the read-only S_AXI_HPC1_FPD, whose MI keeps
none: a slice there disagrees with the crossbar's READ_WRITE_MODE).

**ConvKernel (branch `fmax/conv-timing`).**  Register trees — one copy per
column group, then per chain or buffer — for the write-back, the
hold-register enables, the seeds and `wfb`, the weight-cache fill and
the patch lanes; register stages on the line-buffer, row-buffer,
transposer, bias and vector-FIFO ports; registered reset and per-unit job
resets; registered AXI-Lite and read-data inputs.  LAT 21 → 24 (HAZ 80
kept, checked at elaboration).  Bit-exact (63 fixtures, ~5 000 random
jobs, the DSP model check), +8…23 cycles per job (≤ 0.12 %), OOC WNS
+0.039 ns at 3.333 ns; FF 21.5 k → 39.6 k, LUT 18.4 k → 17.8 k.

**Implementation strategy (hw_128 `scripts/build.tcl`).**  impl_1 with
run 1's directives: `place_design ExtraTimingOpt`, `phys_opt_design
AggressiveExplore`, `route_design AggressiveExplore`, post-route
`phys_opt_design AggressiveExplore` (`set_impl_directives`).

**MatmulKernel, VectorOPKernel, PoolingKernel (branch `fmax/mvp-timing`).**
MatmulKernel: the MAC DSPs with AREG = BREG = 2, CREG, MREG, PREG (the beat
element moved to the 18-bit B port — on the 27-bit A port its sign bit drove
12 pins per DSP); the issue decision registered once and copied per group of
8 DSPs (valid, pop, init, accumulator addresses); the A-buffer address /
enable and the tap-FIFO read pointer registered and copied per beat lane;
the gearbox's end conditions precomputed; 2-entry register slices
(`mm_rs`) between the read FIFO, the gearbox and the lanes and on AR / AW /
W; drain `PIPE` 6 → 9.  VectorOPKernel: `outer · n_words` as three registered
16 × 16 partial products (one more configuration state) — the only
logic-bound path; register slices (`vo_rs`) on AR / AW / W, registered R /
B; the tail mask from a registered run-end flag; the burst generator's
carry / borrow from registers.  PoolingKernel: one more line-buffer write
stage (LX) with per-bank copies, registered read addresses per channel, the
reducer's stall decision per group of 4 lanes from local copies, a slice
into the reducer, registered row-buffer writes, slices on AR / AW / W.  All
three: registered job resets per unit (`start_q` one cycle later), registered
AXI-Lite inputs (a write takes effect with BVALID, a read answers one cycle
later), data registers off the reset.  Bit-exact (all fixtures, ~14 000
random jobs: 3 000 / 6 300 / 4 400), cycles +0.01…0.12 % (Matmul), +2…10 per
job (VectorOP), +0.02…1.5 % (Pool); OOC at 3.333 ns WNS +0.026 / +0.093 /
+0.109 ns; Matmul LUT 18.4 k → 16.3 k, FF 9.5 k → 15.3 k; VectorOP FF
6.5 k → 7.3 k; Pool FF 8.5 k → 11.7 k.

**Loader (`inference-scheduler/src/bitstream`).**  `parse_hwh_clocks` reads
the PL0 frequency and the kernels' `ap_clk` from the HWH; `upload_bitstream`
checks `/sys/devices/platform/fclk0/set_rate` against PL0 before
programming (sets it and re-reads when it differs, refuses when it cannot)
and again after the xclbin load.

## 4. Phases

0. **Investigation** — done (§2).
1. **Fixes and timing closure.**  The kernel branches bit-exact in
   Verilator, OOC at 3.333 ns; the BD script; one full build at 250 MHz
   with the fixed IPs and the run-1 directives on a copy of the project.
   Done when WNS ≥ +0.100 ns at 4.000 ns with the MMCM (the jitter
   included), WHS ≥ 0.
2. **Simulation.**  `sim_hw_kv260` with the MMCM (the testbench waits for
   `locked` and the reset release instead of a fixed delay), the
   behaviour tests (`behavior_test_<k>`: kernel cycles at the test stand's
   100 MHz, unchanged clock basis).
3. **Board.**  The loader's PL0 check, the new bitstream loaded,
   registers, `run_remote_tests` (148 models), the 60 kernel benchmarks,
   the demos and the chat / TTS gates bit-exact; speed against
   `c2b2a6e5e50e`.  Compute-bound layers should approach 2.5×; GEMV
   decode, VectorOP and pooling are HPC-port / DDR bound (4 GB/s per
   port at 250 MHz against 1.6, DDR4 2133 MT/s ≈ 17 GB/s shared with the
   CPUs), so less.
4. **Software at 250 MHz.**  The places that assume 100 MHz (cycle → time
   conversions, `CALL_OVERHEAD` and the board-fitted cost-model
   constants, `perf_fit`'s `clock_mhz`, the bitstream metadata, the
   conv-cycle-model's `PL_MHZ`, docs); a performance-model campaign and
   perf-regression baseline for the new bitstream; production.

## 5. Results

**Phase 1 — timing closure** (full builds on project copies, the MMCM and
register slices from `bd_kernel_clock.tcl`, the impl directives of
`build.tcl`):

| build | kernels | WNS / WHS at 4.000 ns | endpoints < 0.3 ns |
|---|---|---|---|
| baseline (agent A, PS clock 250 MHz, no fixes) | main | 0.000 / +0.010 (run-1 directives), −0.061 (project strategy) | 3 100–3 700 |
| int1 | ConvKernel fixes; Matmul / VectorOP / Pool from main | **+0.084** / +0.010 | — |
| int2 | all four (fmax/mvp-timing 1a96a14) | **+0.074** / +0.010 | 447 (21 below 0.1) |
| int3 | all four + VectorOP write issue, Pool read path (2318714) | **+0.157** / +0.010 | — |
| int4 | int3 with main's PoolingKernel (the board-proven one) | **+0.085** / +0.010 | — |
| int5 | int3 with the fixed PoolingKernel (688883b) | +0.017 / +0.010 | 319 |
| **int6** | int5 + `keep` on each kernel's registered reset, the control slaves' own copy; `MAX_FANOUT 32` on the HPC0 register slice's ready (`constrs_1/new/kernel_clock.xdc`) | **+0.105** / +0.010 | — |

int5's worst clusters were systematic: the HPC0 master-side register slice's
`s_ready` selecting ~150 payload bits (69 endpoints, +0.017 ns), and **one
reset register driving three kernels** — global synthesis of the block
design merged the four kernels' equivalent `rst <= !ap_rst_n` registers
(VectorOP's `rst_reg` → its own, MatmulKernel's and ConvKernel's control
slaves), undoing the per-kernel reset trees.  int6 keeps each kernel's
reset (`(* keep *)`; one register per kernel, 20–91 loads) and replicates the
slice's ready (9 copies); its worst paths are kernel logic (MatmulKernel's
prefetch reservation, 12 levels, +0.105 ns; Pool's configuration product
+0.135 ns).  int6 — bitstream `986cef4866a0`, 56.0 k LUT (47.8 %), 81.4 k FF
(34.7 %, 48.3 k before), 103 BRAM tiles, 48 URAM, 688 DSP — is the
candidate (`/mnt/data/bitstreams/kv260_250_986cef4866a0/`).

The MMCM's clock uncertainty at 250 MHz is 0.066 ns (the PS clock's was
0.160).  int2's worst paths: VectorOP's write-FIFO count → burst-generator
enables (9 levels, 0.074) and Pool's line-buffer read address → read
register (0.094) — both addressed in 2318714; int1's: VectorOP's `pm1`
multiply (0.084) and the old Pool configuration arithmetic (0.092).

**Phase 2 — simulation.**  The merged RTL suite (Verilator: TestConvRtl,
TestMatmulRtl, TestVectorOpRtl, TestPoolRtl, the driver checks, ConvRtlDsp)
9 / 9; `sim_hw_kv260` on int2 (MMCM, register slices, the four new IPs)
68 / 68, no ERROR — after the testbench waits for the reset release on the
BD's Verilog net (an index into the VHDL reset block's port never woke the
`wait`; the MMCM locks within 10 µs) and the PS VIPs' reset-width check is a
warning (the MMCM, reset with PL0, stops the port clocks during the pulse).

**Phase 3 — board, first pass (int2, 2026-10-06).**  The loader's PL0 check
(fclk0 99 999 999 Hz before and after the xclbin load), the 83 control
registers of the four kernels written and read back, the 60 kernel
benchmarks pass, **every one faster**:

| kernel benchmarks | 100 MHz (`c2b2a6e5e50e`) → 250 MHz |
|---|---|
| ConvKernel (3×3 / 1×1 / 5×5 / depthwise) | −54 … −60 % (3×3 64 ch 56² 2.295 → 0.926 ms) |
| MatmulKernel GEMM / FC / batched | −35 … −60 % (256³ 1.404 → 0.564 ms) |
| MatmulKernel GEMV kw 4 (memory-bound) | −42 / −46 % |
| VectorOPKernel | −44 … −59 % |
| PoolingKernel global / 7×7 | −37 … −42 % |
| PoolingKernel 2×2 / 3×3 windows | −6 … −22 % (short runs per row, DDR-latency bound) |

BERT 50 / 50 bit-exact, p50 840.5 → **545.5 ms** (−35 %; the host-CPU share
does not scale); MobileNet v1 / v2 41.2 / 38.1 → **22.0 / 20.5 ms**, results
unchanged.  **But run_remote_tests 128 / 148**: every model with a windowed
pooling call fails (e.g. `pool_maxpool_simple`: the first pixel pair of
every row of channels 2–3 reads −128, the max's initial value), so do
ResNet-18 (20.5 ms) and the MNIST models.  Discrimination on the board:
int1 — the old pool RTL at the same 250 MHz, MMCM and slices — passes all
9 pool models, int2 fails 6 of 9, twice, deterministically: the failure is
in the pool pipeline changes of 1a96a14, not the clock; the Verilator bench
passes the same job under every timing mode.

**The cause: a synthesis / simulation mismatch.**  1a96a14's `pl_reduce`
wrote its snapshot array `accd` (and the per-group control registers) from
several generate scopes.  Verilator and xsim simulate that as written;
Vivado does not — synthesis reports `Synth 8-4767 Trying to implement RAM
'g_lanes[1].accd_reg' in registers … Invalid write to RAM` (8 times in the
int2 / int3 synthesis logs, never in int4's) and builds something else.
Reproduced by running the RTL in lockstep with its own synthesised netlist
in xsim on the board's pool geometries: 1 872 mismatching words over 13
jobs, all windowed pools, the first two lanes of a row's output word holding
0x8000 (MAX) or 0 (AVG) — the board's pattern; main's pool netlist 0.
Fixed in 688883b (every register in one process of one generate scope; the
same for 2318714's `rv`), bit-exact, Pool OOC +0.110 ns.  The lockstep is
now a build check: `tools/neteq` and the targets `neteq_matmul_rtl`,
`neteq_vectorop_rtl`, `neteq_pool_rtl` (Vivado OOC synthesis of the IP
top, then the RTL-vs-netlist lockstep in xsim, 15–25 min, exit 1 on any
mismatch; MatmulKernel 14 jobs and VectorOPKernel 24 jobs: 0 mismatches) —
before every bitstream, and a grep of the synthesis log for `Synth 8-4767`.
The Pool testbench gained `--timing board` / `board2` (150–900 cycles of
latency, throttled AR, R bursts in pieces) and job sequences.

**Board, int5 and int6** (the fixed pool): 83 / 83 registers, 60 / 60
benchmarks, **148 / 148 models**, the 9 pool models, every demo result
unchanged, and the chat / TTS gates bit-exact — SmolLM2-135M / 360M and
SmolVLM logits (4 × 33 / 33, 2 images), Piper PCM, text encoder and
duration predictor — with the production libraries unchanged (the engine
choices do not depend on the clock).  §6 has int6's numbers.

**Board, int3** (2318714): the same — 83 / 83 registers, 60 / 60 benchmarks,
every non-pool model correct (D's new VectorOP write issue included), the
same 20 windowed-pool models wrong.  **Board, int4** (int3's IPs, main's
PoolingKernel): **148 / 148 models**, 83 / 83 registers, 60 / 60 benchmarks,
every demo result unchanged against the 100 MHz baseline (BERT 50 / 50
bit-exact):

| workload | 100 MHz (`c2b2a6e5e50e`) | 250 MHz (int4) |
|---|---|---|
| ResNet-18 | 47.46 ms | **20.52 ms (48.7 FPS)** |
| MobileNet v1 / v2 | 41.23 / 38.10 ms | **21.98 / 20.46 ms** |
| BERT-SQuAD (p50 of 50) | 839.2 ms | **541.7 ms** |
| MNIST convnet / LeNet | 0.2355 / 2.7015 ms | **0.1105 / 1.2491 ms** (98.92 / 97.35 %) |

**Phase 4 — software and integration (2026-10-06).**

- The kernel sources of the four branches (fmax/conv-timing d24d9eb,
  fmax/mvp-timing 1a96a14 · 2318714 · 688883b, the reset `keep`s) in the
  main tree, byte-identical to int6's; the four IPs re-packaged in
  `build_hw128`; `bd_kernel_clock.tcl` applied to the real `hw/cormorant_hw_128`
  project (the clk_wiz and 13 register-slice IPs, `constrs_1/new/kernel_clock.xdc`);
  `scripts/build.tcl` sets impl_1's directives; the block-design testbench
  waits for the MMCM.
- `upload_bitstream.py`: Step 5b checks `/sys/devices/platform/fclk0/set_rate`
  against the HWH's PL0 (sets it when another loader changed it, refuses
  when it cannot) and again after the xclbin load (`src/bitstream/hwh.py`
  `parse_hwh_clocks`, `board.py` `read_fclk_hz` / `set_fclk_hz`).
- The performance models record the kernel clock of their bitstream
  (`perf_fit.kernel_clock_mhz`: the HWH's `ap_clk` when the local bitstream
  is the one fitted, else the model file's value, else 100); `perf_calibrate`
  sizes its per-call estimates with the local HWH's clock;
  `perf_model.MHZ` (unused) removed; the conv-cycle-model's `PL_MHZ` 250.
- Not changed: the engine cost model.  Its board terms (`RTL_CONV_BOARD`,
  `RTL_COEF`, `CALL_OVERHEAD` in cycles) were fitted at 100 MHz; at 250 MHz
  a fixed host or DDR cost is 2.5× the cycles, so a refit from the
  `986cef4866a0` campaign may move some engine choices (MatMul on ConvKernel
  vs MatmulKernel) — a follow-up, with regenerated and re-gated libraries.
  Until then the reports' cost-model estimates stay labelled "@100 MHz".

## 6. The production bitstream at 250 MHz — `986cef4866a0` (2026-10-06)

int6, validated on the board end to end (83 / 83 registers, 60 / 60 kernel
benchmarks — every one faster —, **148 / 148 models**, the 9 pool models,
every demo result unchanged against `c2b2a6e5e50e`, the four chat / TTS
gates bit-exact), then made the production bitstream (the local
`bitstream_config_kv260.json` names `/mnt/data/bitstreams/kv260_250_986cef4866a0/`),
the chat server restarted on it with the libraries it had (the engine
choices do not depend on the clock), its performance model measured
(`perf_models/kv260/986cef4866a0.json`: 2011 exact calls after five
refinement rounds of 108 / 59 / 33 / 69 / 13, `clock_mhz` 250, repeat spread
median 0.099 %; the family fits loose — mm-gemv held-out p90 18.5 %,
conv-mm 64 % — until the cost model is refitted) and its perf-regression baseline
recorded (66 cases, against `c2b2a6e5e50e`: 0 regressions, 66 improved, 0
result changes).

| workload | 100 MHz (`c2b2a6e5e50e`) | 250 MHz (`986cef4866a0`) | |
|---|---|---|---|
| ResNet-18 | 47.46 ms (21.1 FPS) | **20.51 ms (48.8 FPS)** | 2.31× |
| MobileNet v1 / v2 | 41.23 / 38.10 ms | **21.97 / 20.48 ms** | 1.88× / 1.86× |
| MNIST convnet / LeNet | 0.2355 / 2.7015 ms | **0.1106 / 1.2488 ms** | 2.13× / 2.16× |
| BERT-SQuAD (p50 of 50) | 839.2 ms | **541.2 ms** | 1.55× (host ops) |
| SmolLM2-135M decode / prefill 16 / 256 | 99.8 ms / 243 / 1118 ms | **54.5 ms / 156 / 810 ms** | 1.83× / 1.56× / 1.38× |
| SmolLM2-360M decode / prefill 16 / 256 | 252.0 ms / 565 / 2621 ms | **137.4 ms / 336 / 2053 ms** | 1.83× / 1.68× / 1.28× |
| SmolVLM `llm_image` / decode | 3.37 s / 99.4 ms | **2.35 s / 54.0 ms** | 1.43× / 1.84× |
| Piper RTF (6.9 s utterance), chunk | 0.303, 428 ms | **0.243, 347 ms** | 1.25× |

The whole-model gains stay below the clock's 2.5×: the host CPU's share
(attention, softmax, layer norms, GELU, Piper's flow / gate ops) does not
scale, and the memory-bound work — GEMV decode through the HPC ports,
VectorOP, the spatial pools (DDR latency on short runs) — gains less than
the compute-bound convolutions and GEMMs (−54 … −60 % per call).

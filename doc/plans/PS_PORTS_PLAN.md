# PS port layout: kernel outputs on HP0, VectorOP b on HPC1

**Status (2026-10-08):**

- **Experiment 1, the kernels' outputs on `S_AXI_HP0_FPD` (§1–§4): dropped.**
  - Bitstream `4a820240ba9c` was bit-exact on the board but
    performance-neutral (±0.5 %).
  - The user chose to try experiment 2 instead.  The hw_128 block design is
    back at its committed state; `scripts/bd_hp0_outputs.tcl` stays as the
    record.
- **Experiment 2, VectorOPKernel's `gmem1` (b) on `S_AXI_HPC1_FPD` (§5): a
  gain.**
  - Bitstream `8599aa7a5f12` is bit-exact on the board.
  - Binary VectorOP calls are 24–38 % faster; no other kernel moves.
  - End to end: BERT 427 → 418 ms (−2.1 %), ResNet-18 −1.6 %, SmolVLM image
    −1.1…−1.6 %, Piper −1.3…−1.9 %; the LLM chat models are flat.
- **Promoted (2026-10-08): `8599aa7a5f12` is the production bitstream** (§5.5).
  - It has its own performance model (1862 calls) and perf-regression
    baseline.
  - The chat server runs on it.
  - `CALL_OVERHEAD` follows the new call floor (434 → 431 cycles, with
    `ONE` / `one` +3): no engine choice moves.

The user asked to add the MPSoC's `S_AXI_HP0_FPD` port to the block design
and move the kernels' outputs to it ("let's try").  When that was neutral,
the user asked to move VectorOPKernel's second read port to HPC1 instead.
Each experiment is decided by board measurements.
## 1. Where it stands (experiment 1)

`hw/cormorant_hw_128` (production bitstream `588d721997cb`, 250 MHz) has
two PS slave ports, both through the CCI-400:

| port | interconnect | masters |
|---|---|---|
| `S_AXI_HPC0_FPD` (SAXIGP0) | `axi_interconnect_0`, 9 SI | VectorOP gmem0 / gmem1 (a, b) / **gmem2 (c)**; Matmul gmem0 (A) / **gmem2 (C)**; Conv gmem0 (x) / **gmem3 (y)**; Pool gmem0 (x) / **gmem1 (y)** |
| `S_AXI_HPC1_FPD` (SAXIGP1) | `axi_mem_intercon`, 3 SI, read-only | Conv gmem1 (w) / gmem2 (b), Matmul gmem1 (B) |

Every RTL kernel declares its ports read-only or write-only (fact
`rtl.axi_masters`), so the four write masters (bold) are exactly the
kernels' outputs, and they share HPC0 with five readers.

The port is not shared with anything else, and the CCI adds latency to
every transfer.  At 100 MHz the second port (RESNET18_15FPS_PLAN §3.2) made
no difference.  At 250 MHz a port carries at most 4 GB/s per direction
(16 B × 250 MHz).  The 588d721997cb benchmarks come close to that:

| case | traffic | HPC0 read | HPC0 write |
|---|---|---:|---:|
| VectorOP ADD-256K | 2 reads + 1 write, 512 KB each | 3.8 GB/s | 1.9 GB/s |
| VectorOP RELU-64K | 1 read + 1 write | 3.4 GB/s | 3.4 GB/s |
| VectorOP RELU-bcast-8x16K | 1 read + 1 write | 3.6 GB/s | 3.6 GB/s |
| Pool AvgPool-2x2-3x3-32-112 | | 2.8 GB/s total | |

## 2. Design

The ZynqMP routes each PS port to the DDR controller (UG1085, "DDR memory
controller") as follows:

- HPC0 / HPC1 → CCI-400 → DDRC ports 1–2, shared with the APU;
- HP0 → FPD switch → DDRC port 3, shared with DisplayPort (idle: the board
  is headless);
- HP1 / HP2 → port 4;
- HP3 → port 5, shared with FPD DMA.

The new layout:

| port | interconnect | masters | mode |
|---|---|---|---|
| HPC0 | `axi_interconnect_0`, 5 SI | VectorOP a / b, Matmul A, Conv x, Pool x | read |
| HPC1 | `axi_mem_intercon`, 3 SI (unchanged) | Conv w / b, Matmul B | read |
| **HP0** (SAXIGP2, 128-bit) | **`axi_out_intercon`, 4 SI** | VectorOP c, Matmul C, Conv y, Pool y | write |

- **Script.**  The block design is changed by
  `hw/cormorant_hw_128/scripts/bd_hp0_outputs.tcl`, which is idempotent, as
  `bd_kernel_clock.tcl` is.  It does the following:
  - sets `PSU__USE__S_AXI_GP2` 1 and `PSU__SAXIGP2__DATA_WIDTH` 128;
  - clocks `saxihp0_fpd_aclk` from `clk_wiz_0/clk_out1`;
  - creates the new interconnect, with the clocks and resets of the other two;
  - moves the four writers and shrinks `axi_interconnect_0` to five SIs;
  - deletes the moved masters' stale `SEG_*HPC0*` segments before
    `assign_bd_address` (RESNET18_15FPS_PLAN §3.2 trap);
  - adds register slices on every SI;
  - ends with `validate_bd_design`.
- **Ordering.**  A kernel writes through HP0, and the next kernel (or the
  CPU after its cache invalidate) reads through HPC0 / CCI.  The kernels
  raise done only after every B response, the DDRC orders reads after
  writes across its ports, and the coherency is software-managed already
  (`PSU__AFI0_COHERENCY 0`).  The board suite's chained models are the
  check.
- **Loader.**  `upload_bitstream.py` already programs every
  `C_SAXIGP*_DATA_WIDTH` of the HWH (AFIFM2 = HP0 at `0xFD38_0000`).
  `read_kernel_regs.sh` gains the HP0 line.
- **Scheduler.**  The scheduler does not depend on the ports, so the
  libraries do not change.  A new bitstream needs its own performance model
  and perf-regression baseline before it becomes production.

**Expected.**
- Unary VectorOP and the pooling kernel write as much as they read, so they
  may gain up to the per-direction cap.
- Binary VectorOP stays bound by its two reads on HPC0.  A follow-up could
  split gmem1 to another port.
- The compute-bound 3×3 convolutions should not move.
- Less traffic through the CCI could shorten read latency.

## 3. Phases and gates

| phase | work | gate |
|---|---|---|
| 0 | this plan; `bd_hp0_outputs.tcl` applied to the BD | `validate_bd_design` clean; address map: writers on HP0 only |
| 1 | testbench: HP0 VIP reset check and slave profile | `sim_hw_kv260` 75 / 75 |
| 2 | `build_hw_kv260` at 250 MHz | WNS ≥ 0, WHS ≥ 0; archived in `/mnt/data/bitstreams/` |
| 3 | board: load, HP0 width 128; model suite; perf benchmarks against the `588d721997cb` baseline; demos; chat-model gates | 159 / 159, demos bit-exact; per-case speed table |
| 4 | decision: promote (perf model, baseline, docs, facts) or restore `588d721997cb` and record the result | — |

## 4. Results

### 4.1 Phases 0–1: block design and simulation (2026-10-08)

`scripts/bd_hp0_outputs.tcl` applied to the BD; a second run changes nothing
and `validate_bd_design` is clean.

- **Master-side register slices.**  The first run left
  `axi_interconnect_0`'s MI register slice in place.  That gave a critical
  warning (BD 41-237): the slice is READ_ONLY, its crossbar READ_WRITE.  This
  is the rule HPC1 already follows, so all three MIs had none in this design.
  For this design, `bd_kernel_clock.tcl` left the HPC0 MI slice out when
  `axi_out_intercon` existed, and `kernel_clock.xdc`'s replication of that
  slice's ready was commented out.  Both edits were reverted when the
  experiment was dropped.
- **LPS_OCM.**  Every port maps it at `0xFF00_0000`.  The moved masters keep
  the HPC0 segment, as HPC1's masters have since 2026-09-26, and Vivado
  excludes it.  The kernels address DDR only.

| gate | result |
|---|---|
| address map | writers: `HP0_DDR_LOW` / `HP0_DDR_HIGH` / `HP0_QSPI`; readers unchanged |
| `sim_hw_kv260` (HP0 VIP: reset check as a warning, BEST_CASE slave profile as HPC0) | 75 / 75 (3 min 34 s); PS VIP transactions — HP0: 191 writes, 0 reads; HPC0: 0 writes, 486 reads; HPC1: 0 writes, 63 reads |

### 4.2 Phase 2: the bitstream (2026-10-08)

| | `588d721997cb` (production) | `4a820240ba9c` (outputs on HP0) |
|---|---:|---:|
| WNS / WHS at 4 ns | +0.041 / +0.010 ns | **+0.117** / +0.010 ns |
| CLB LUTs | 62 347 (53.2 %) | 62 254 (53.2 %) |
| CLB registers | 87 703 | 87 356 |
| BRAM tiles / URAM / DSP | 123 / 48 / 712 | 123 / 48 / 712 |
| build | 73 min (kernel IPs re-synthesised) | 38 min (synthesis 9, implementation 29; the kernels' out-of-context runs reused) |

The HWH declares `C_SAXIGP2_DATA_WIDTH = 128`, which the loader programs.
Archived in `/mnt/data/bitstreams/kv260_250_4a820240ba9c/`.  The build
printed one critical warning: `kernel_clock.xdc`'s `set_property` matched no
object, because HPC0's MI slice was gone.  The constraint was then commented
out; it had been a no-op in this build.

### 4.3 Phase 3: the board (2026-10-08)

**Load.**  `upload_bitstream.py` programmed the PL and the xclbin, but
re-applying the `pl` overlay failed with `err=-22`.  `k26-starter-kits_image_1`
(the starter kit's overlay, applied after ours) overlaps it at `/axi/afi0`, so
`pl` is "not topmost" and can be neither removed nor re-applied.  This state
predates the experiment: the softmax loads hit the same error.  The old
`pl` overlay's nodes stay live.  They come from the same dtbo, so the UIO
names, addresses and interrupts are right.  All six AFIFM width fields read
0 (128-bit), HP0's included.  A reboot (or an `rmdir` of the starter-kit
overlay) clears it.

| gate | `588d721997cb` | `4a820240ba9c` (HP0) |
|---|---|---|
| `run_remote_tests` (159 models) | 159 / 159 | **159 / 159**, every output equal to the simulation (11.5 min) |
| `run_remote_perf` (63 cases), two runs | baseline | 63 / 63; per-kernel mean Δ VectorOP −0.6 %, Matmul −0.6 %, Conv +0.2 %, Pool +0.5 %; one consistent mover, Pool `AvgPool-2x2-3x3-32-112` +1.4 % (5 µs) |
| MNIST convnet / LeNet | 0.1105 / 1.2512 ms | 0.1105 / 1.2547 ms; 98.92 % / 97.35 % |
| MobileNet v1 / v2, ResNet-18 | 21.98 / 20.44 / 20.50 ms | 21.98 / 20.52 / 20.51 ms; top-1 unchanged |
| BERT (p50) | 427.2 ms | 426.7 ms; EM / F1 88.0 / 90.3, bit-exact |
| SmolLM2-135M prefill 16 / 64 / 256, decode | 152.0 / 276.1 / 736.7, 54.6 ms | 152.1 / 276.5 / 737.6, 54.3 ms; logits bit-exact (4 × 33) |
| SmolLM2-360M prefill 16 / 64 / 256, decode | 329.6 / 661.3 / 1925.6, 137.5 ms | 327.8 / 660.2 / 1926.0, 137.6 ms; bit-exact (4 × 33) |
| SmolVLM `llm_image` (2 images), decode | 1940.9 / 1926.8, 54.3 ms | 1933.3 / 1933.7, 54.3 ms; bit-exact (2 × 33) |
| Piper utterances 0 / 1 | 507.9 / 1403.4 ms | 508.4 / 1401.6 ms; PCM 0 mismatches |

**Why nothing moved.**
- **Writes on HPC0 did not slow the reads.**  The memory-bound cases were the
  hope: unary VectorOP, read 1 : write 1 at 3.4 GB/s each.  They stay where
  they were.  Most likely HPC0's writes never took bandwidth from its reads:
  the AXI read and write channels are independent through the port, and the
  CCI evidently had headroom for both.
- **The DDRC and the kernels' own pipelines set the limit.**  Binary VectorOP
  stays at 3.8 GB/s of reads, near the port's 4 GB/s.  HP0 does not change
  that, because its two reads are still on HPC0 (§2).
- **Pooling pays slightly.**  `AvgPool-2x2-3x3-32-112` is +1.4 %, probably a
  longer write path to DDRC port 3 for its short bursts.

The board runs `588d721997cb` again (restored with `upload_bitstream.py`:
id, FPGA state, UIO names and widths checked), the chat server is restarted
and answers.  Scratch directories `/root/hp0_scratch` removed.

**Phase 4: the decision is the user's.**
- **Keep HP0.**
  - It needs a performance model campaign for `4a820240ba9c`, a
    perf-regression baseline and the docs and facts that name the
    production bitstream.
  - What it buys: 0.076 ns more slack at 4 ns for future kernel changes.
    HPC0's write-side register slice was the design's worst path at
    250 MHz, and with read-only ports it is gone.
- **Drop HP0.**  `git checkout` the hw_128 block design; this plan and
  `bd_hp0_outputs.tcl` stay as the record.  **Chosen:** the user asked for
  experiment 2 instead (§5).
- **Next experiment.**  It would target the binary VectorOP reads: gmem1 (b)
  onto HPC1 or HP1.

## 5. Experiment 2: VectorOPKernel's b on HPC1

### 5.1 Design

A binary VectorOP (ADD / SUB / MUL / DIV, and the activation-fused forms)
reads a (gmem0) and b (gmem1) through HPC0.  At up to 3.8 GB/s for the two
together, that is near the port's 4 GB/s per direction at 250 MHz
(ADD-256K: 1 MB read in 0.275 ms).  The unit consumes 8 lanes of each
operand per cycle, 4 GB/s each.  With b on HPC1, each operand has a port.
HPC1 carries ConvKernel's weights and bias and MatmulKernel's B; those
overlap VectorOP only when the lanes run concurrently.

`hw/cormorant_hw_128/scripts/bd_vop_b_hpc1.tcl` is idempotent.  Starting from
the committed (production) block design, it changes these connections:

| change | from | to |
|---|---|---|
| VectorOPKernel `gmem1` (b) | `axi_interconnect_0/S01` (HPC0) | **`axi_mem_intercon/S03` (HPC1)**, with an SI register slice like the other three |
| PoolingKernel `gmem1` (y) | `axi_interconnect_0/S08` | `axi_interconnect_0/S01` (the freed slot) |
| `axi_interconnect_0` | 9 SIs | 8 SIs (the MI slice stays: HPC0 still reads and writes) |

Nothing else moves.  VectorOP b's HPC0 DDR / QSPI segments are deleted
before `assign_bd_address`.  The script takes `-ip-repo DIR` and upgrades
locked IPs first, as `sim.tcl` / `build.tcl` do.  After a `git checkout` of
the `.xpr` the project's IP repository path is the stale default, and the
locked kernel IPs block `validate_bd_design`.

**Expected.**
- Large binary VectorOP calls up to ~2× if the DDRC keeps up: two 4 GB/s
  reads plus a 4 GB/s write.
- End to end, the residual and bias adds of ResNet-18, MobileNet v2 and BERT,
  SmolVLM's GELU bias ADD and MUL, and Piper's decoder sums.
- MatmulKernel GEMV now shares HPC1 with VectorOP b, but only when the two
  lanes overlap.

### 5.2 Block design and simulation (2026-10-08)

The script ran on the restored block design; `validate_bd_design` is clean.
The two kinds of critical warning are known:
- the kernel IPs' upgrade (Coretcl 2-1280), whose instance widths
  `build.tcl` / `sim.tcl` reset;
- `HPC1_LPS_OCM` not assignable (BD 41-1359), as for HPC1's other masters.

Two traps on the way:
- **Stale generated outputs.**  After the HP0 build, `.gen` held the HP0
  interconnect.  The restored `axi_interconnect_0` was "locked" (stale
  generated content), so the `.gen/sources_1/bd/design_cormorant` tree had
  to go.  It is regenerated by the next `sim_hw` / build.
- **Stale IP repository path.**  The restored `.xpr` points the IP
  repository at the stale default, which locks all four kernel IPs.  Hence
  `-ip-repo`.

| gate | result |
|---|---|
| `sim_hw_kv260` | 75 / 75; PS VIP reads HPC0 486 → 467, HPC1 63 → 82 (the testbench's 19 binary VectorOP b reads moved), writes all on HPC0 (191) |

### 5.3 The bitstream (2026-10-08)

| | `588d721997cb` (production) | `4a820240ba9c` (exp. 1, HP0) | `8599aa7a5f12` (exp. 2, b on HPC1) |
|---|---:|---:|---:|
| WNS / WHS at 4 ns | +0.041 / +0.010 ns | +0.117 / +0.010 ns | **+0.114** / +0.010 ns |
| CLB LUTs | 62 347 | 62 254 | 62 400 |
| BRAM tiles / URAM / DSP | 123 / 48 / 712 | 123 / 48 / 712 | 123 / 48 / 712 |

No critical warning in the build.  Both port changes gained the same
~0.075 ns of slack.  So experiment 1's slack was not HP0's: it comes from
place-and-route variation, or from a smaller HPC0 crossbar (8 or 5 SIs
instead of 9).  Archived in `/mnt/data/bitstreams/kv260_250_8599aa7a5f12/`.

### 5.4 The board (2026-10-08)

The load hit the known overlay error (§4.3).  The board script then checked
the loaded id (`8599aa7a5f12`, fpga0 operating) and the AFIFM widths (all
128).  At the end it put `588d721997cb` back (id checked) and restarted the
chat server.

| gate | `588d721997cb` | `8599aa7a5f12` (b on HPC1) |
|---|---|---|
| `run_remote_tests` (159 models) | 159 / 159 | **159 / 159** (11.3 min) |
| VectorOP ADD-4K / 16K / 64K / 256K | 8.9 / 21.8 / 72.7 / 275.1 µs | **6.7 / 15.9 / 49.6 / 185.1 µs** (−24 / −27 / −32 / −33 %; ADD-256K 5.7 → 8.5 GB/s) |
| VectorOP MUL-16K / 64K | 21.6 / 72.4 µs | **16.2 / 50.5 µs** (−25 / −30 %) |
| VectorOP ADD / MUL-bcast-8x16K | 139.3 / 139.2 µs | **87.0 / 86.5 µs** (−38 %) |
| VectorOP unary (RELU, softmax), DIV, MUL-bcast-dw | | unchanged (no b, or DIV's 1 lane per cycle; dw replays b on chip) |
| MatmulKernel, ConvKernel, PoolingKernel (two runs) | | per-kernel mean Δ +0.4 / +0.3 / −0.1 %; no consistent mover — `FC-1x1280x1001-packed` 0.3598 → 0.3993 / 0.3605 ms, a one-run outlier (it read 0.4009 on `986cef4866a0`) |
| MNIST convnet / LeNet | 0.1105 / 1.2512 ms | 0.1105 / 1.2504 ms; 98.92 % / 97.35 % |
| MobileNet v1 / v2, ResNet-18 | 21.98 / 20.44 / 20.50 ms | 21.96 / 20.35 / **20.18 ms** (−1.6 %); top-1 unchanged |
| BERT (p50) | 427.2 ms | **418.0 ms** (−2.1 %); EM / F1 88.0 / 90.3, bit-exact (50 / 50 against the emulation) |
| SmolLM2-135M prefill 16 / 64 / 256, decode | 152.0 / 276.1 / 736.7, 54.6 ms | 152.7 / 276.8 / 736.2, 54.6 ms; logits bit-exact (4 × 33) |
| SmolLM2-360M prefill 16 / 64 / 256, decode | 329.6 / 661.3 / 1925.6, 137.5 ms | 328.4 / 661.1 / 1918.9, 137.5 ms; bit-exact |
| SmolVLM `llm_image` (2 images) | 1940.9 / 1926.8 ms | **1910.1 / 1905.1 ms** (−1.6 / −1.1 %); bit-exact (2 × 33) |
| Piper utterances 0 / 1 | 507.9 / 1403.4 ms | **498.3 / 1385.7 ms** (−1.9 / −1.3 %, every chunk −5 ms); PCM 0 mismatches |

**Reading.**
- **The read bound.**  The binary VectorOP was bound by its two reads
  sharing HPC0's read channel.  With one operand per port, ADD-256K moves
  1.5 MB at 8.5 GB/s: 2.8 GB/s per stream, 70 % of the unit's 4 GB/s per
  operand.  The rest is DDR and burst overheads.
- **End to end.**  The gain follows the workloads' share of binary VectorOP
  time: BERT's bias and residual adds, ResNet-18's residual adds, SmolVLM's
  GELU bias ADD and MUL, and Piper's decoder sums.
- **The chat models are flat.**  Their residual sums are host float32
  (`LlmResAdd`), so they issue no binary VectorOP.
- **No interference on HPC1.**  The MatmulKernel GEMV / packed cases, whose
  B also streams through HPC1, did not move.

### 5.5 Promotion (2026-10-08)

The user asked to promote `8599aa7a5f12`.  The steps were:
- **Board and models.**  Load the bitstream, run a performance model
  campaign (`perf_calibrate.py` cases / run / fit with refinement rounds; the
  chat server is down for the runs), and record the perf-regression baseline.
  The local `bitstream_config_kv260.json` then names the new archive.
- **Chat server.**  Restart it.  The libraries do not change: the scheduler
  does not see the ports.
- **Commits.**  Commit the hw_128 block design with `bd_vop_b_hpc1.tcl` and
  the CLAUDE.md / README port tables and timing.
- **Docs and facts.**  Update the docs and facts that name the production
  bitstream and its results.

Done:

| step | result |
|---|---|
| local `bitstream_config_kv260.json` | names `/mnt/data/bitstreams/kv260_250_8599aa7a5f12/` (the previous one saved there as `bitstream_config_kv260.before.json`); loaded (id and fpga0 checked, the known overlay error) |
| performance model `kv260/8599aa7a5f12` | `588d721997cb`'s converged case list (1597) plus the 261 ConvKernel calls of that campaign's earlier rounds, two passes (2 min 50 s); one refinement round (12 min host) added 4 → **1862 exact calls**, repeat spread median 0.052 %; family fits as `588d721997cb`'s |
| per call against `588d721997cb` | ConvKernel (1265) / MatmulKernel (414) median 0.00 %, p10 / p90 within ±0.2 %; VectorOP unary / softmax unchanged; binary down to −33 %; five small 2×2 / 3×3 stride-2 pooling calls +8…+17 % (1–2 µs; e.g. 14×14×32 → 7×7 13.9 → 16.2 µs, steady on the earlier 250 MHz bitstreams) — not in the pooling benchmarks or MNIST, unexplained |
| simulator (`perf_calibrate.py simulate`) against this bitstream's board runs | the CNNs, BERT, SmolLM2-135M, SmolVLM, Piper within ±2.2 % (MNIST convnet −5.1 % on 0.1 ms); SmolLM2-360M prefill 16 −12 %, 256 +4.6 % (as on `588d721997cb`: the host-op kinds' fits) |
| engine cost model (`tools/fit_cost_model.py fit 8599aa7a5f12`) | the constants as used: ConvKernel 18.6 / 48.3 % over 1265 calls, MatmulKernel 4.5 / 17.7 % over 414 (`588d721997cb`: 18.7 / 48.2, 4.6 / 17.7); kept, the call floor 434 → 431 cycles with `ONE` / `one` +3; `fit_cost_model.py diff` against the old constants: no shipped model's engine choice changes |
| perf-regression baseline `kv260-8599aa7a5f12.json` | the second benchmark pass of §5.4 (63 cases) and the demos of that run |
| chat server | restarted on `8599aa7a5f12` (the libraries unchanged); SmolLM2-135M / 360M, BERT and Piper answer |

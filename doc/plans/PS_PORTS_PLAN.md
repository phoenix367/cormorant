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
- **Experiment 3 (2026-10-09), the outputs on `S_AXI_HP0_FPD` on top of
  experiment 2's design (§6): dropped.**  Bitstream `5de9b4f80908` is
  bit-exact (159 / 159 models, demos) but a consistent small loss: the
  large binary VectorOP calls +3…9 %, the global pools +2…4 %, BERT
  418 → 421 ms (+0.8 %), ResNet-18 +0.7 %.  The user chose to try HP2
  instead.
- **Experiment 5 (2026-10-10), static read QoS 15 on HPC0 / HPC1 (§9): a
  real but small gain, no hardware change.**  Memory-bound binary VectorOP
  calls −4…6 %, the GEMV −4 %, two pools −2…5 %; end to end −0.1…−0.2 %
  (BERT 418.0 → 417.6 ms); bit-exact.  Registers restored to 0; whether to
  make it persistent in `upload_bitstream.py` is open.
- **Experiment 4 (2026-10-09), the same outputs on `S_AXI_HP2_FPD` (§7):
  the same as HP0.**  Bitstream `4a61fc965037` is bit-exact (159 / 159,
  demos) and a small loss of the same shape: the large binary VectorOP
  calls +2…7 %, the global pools +3 %, per-kernel means +0.1…+1.1 %; end to
  end BERT +0.15 %, ResNet-18 +0.25 %, MNIST +0.5…0.7 %.  §8 explains why:
  an HP port is one DDRC port behind the FPD switch, a CCI port is two.

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

## 6. Experiment 3: the outputs on HP0 again, on top of experiment 2 (2026-10-09)

The user asked to add a third interconnect, enable `S_AXI_HP0_FPD`, connect
the interconnect to it and move the kernels' outputs there — experiment 1's
layout, now on top of the production design (`8599aa7a5f12`, VectorOP b on
HPC1).  Experiment 1 was measured neutral when both VectorOP reads shared
HPC0; this run measures the combination.

### 6.1 Design

| port | interconnect | masters | mode |
|---|---|---|---|
| HPC0 | `axi_interconnect_0`, 4 SI (was 8) | VectorOP a, Matmul A, Conv x, Pool x | read |
| HPC1 | `axi_mem_intercon`, 4 SI (unchanged) | Conv w / b, Matmul B, VectorOP b | read |
| **HP0** (SAXIGP2, 128-bit) | **`axi_out_intercon`, 4 SI** | VectorOP c, Matmul C, Conv y, Pool y | write |

- `scripts/bd_hp0_outputs.tcl` updated: the HPC0 reader list no longer
  names VectorOP `gmem1` (it would have pulled b back from HPC1), and it
  takes `-ip-repo` and upgrades locked kernel IPs, as `bd_vop_b_hpc1.tcl`
  does.  Register slices on every SI of the three interconnects, none on
  the three MIs (every port is one-way: BD 41-237).
- `scripts/bd_kernel_clock.tcl`: with `axi_out_intercon` present its
  register-slice pass leaves HPC0's MI without a slice too (idempotent on
  this design).
- `constrs_1/new/kernel_clock.xdc`: the `MAX_FANOUT` on that MI slice's
  ready is commented out (`set_property` on no cells is an error; a guard
  with `if` is not XDC — Designutils 20-1307 in this build's synthesis, the
  line then ignored, which is the same result).
- Testbench: the HP0 PS VIP gets the reset-check-to-warn and the BEST_CASE
  slave profile, as HPC0.

### 6.2 Phases 0–1: block design and simulation (2026-10-09)

| gate | result |
|---|---|
| `bd_hp0_outputs.tcl` on the `8599aa7a5f12` design | `validate_bd_design` clean; the only critical warnings are the eight `*_LPS_OCM` BD 41-1359 (HPC1's four as before, HP0's four new) |
| address map | writers `HP0_DDR_LOW` / `HP0_DDR_HIGH` / `HP0_QSPI` only; VectorOP a, Matmul A, Conv x, Pool x on HPC0; Conv w / b, Matmul B, VectorOP b on HPC1 |
| `sim_hw_kv260` (HP0 VIP: reset check as a warning, BEST_CASE slave profile as HPC0) | 75 / 75 (3 min 36 s); PS VIP transactions — HP0: 191 writes, 0 reads; HPC0: 0 writes, 467 reads; HPC1: 0 writes, 82 reads (the reads as on `8599aa7a5f12`, the writes all moved); the user-width BD 41-237 warnings of HP0 are the ones HPC0 / HPC1 have |

### 6.3 Phase 2: the bitstream (2026-10-09)

| | `8599aa7a5f12` (production) | `5de9b4f80908` (outputs on HP0, b on HPC1) |
|---|---|---|
| synthesis | | 8 min; the XDC guard's `if` reported (Designutils 20-1307) and the line ignored — the slice it names does not exist; the line is commented out since |
| routed, after post-route `phys_opt_design` | WNS +0.114 ns, WHS +0.010 ns | **WNS +0.076 ns, WHS +0.010 ns** |
| placed utilization | | 62 196 LUT (53.1 %), 87 464 FF, 123 BRAM tiles (85.4 %), 48 URAM, 712 DSP |
| archive | | `/mnt/data/bitstreams/kv260_250_5de9b4f80908/` (bit, hwh, timing and utilization reports, build log) |

Experiment 1's build had gained slack from losing HPC0's write-side slice;
this one lost 0.038 ns against production — place-and-route variation
rather than a property of the layout.

### 6.4 Phase 3: the board (2026-10-09)

Loaded with `upload_bitstream.py --config bitstream_config_kv260.exp3.json`
(a local copy of the production config naming the archive; the chat server
stopped first): id `5de9b4f80908`, fpga0 operating, all six AFIFM width
fields 0 (128-bit), HP0's included; the overlay applied without the §4.3
error this time.

| gate | `8599aa7a5f12` | `5de9b4f80908` (HP0) |
|---|---|---|
| `run_remote_tests` (159 models) | 159 / 159 | **159 / 159** (every output equal to the simulation) |
| `run_remote_perf` (63 cases), two runs, against the `kv260-8599aa7a5f12` baseline | baseline | 63 / 63 correct; per-kernel mean Δ VectorOP **+0.9 %**, Matmul +0.8 %, Conv +0.1 %, Pool **+1.1 %** (medians +0.4 / +0.5 / +0.1 / +0.8 %) |
| VectorOP ADD-16K / 64K / 256K | 16.2 / 49.7 / 185.2 µs | 16.7–16.4 / **52.7–54.2** / **192.7–191.2** µs (+1…3 / **+6…9** / **+3…4 %**) |
| VectorOP MUL-16K / 64K, MUL-bcast-8x16K | 16.2 / 51.3, 87.6 µs | **16.7 / 53.0–53.2**, 89.2–90.1 µs (+3.1 / +3.3…3.7, +1.8…2.9 %) |
| VectorOP unary (RELU, RELU6, softmax), DIV, MUL-bcast-dw, ADD-1K | | unchanged (ADD-1K 5.0 → 4.6 µs, within the ~0.5 µs jitter of the shortest calls) |
| PoolingKernel GlobalMaxPool-7x7-256 / GlobalAvgPool-7x7-64 | 38.4 / 12.9 µs | 39.3–39.5 / 13.2–13.4 µs (+2.3…2.9 / +2.3…3.9 %) |
| MatmulKernel, ConvKernel | | no consistent mover above 2 % (16x16x16 +7.1 / +2.4 %: a 4 µs call) |
| MNIST convnet / LeNet | 0.1105 / 1.2504 ms | 0.1111 / 1.2579 ms (+0.5 / +0.6 %); 98.92 % / 97.35 % |
| MobileNet v1 / v2, ResNet-18 | 21.96 / 20.35 / 20.18 ms | 21.96 / 20.40 / 20.32 ms (+0.0 / +0.2 / +0.7 %); top-1 unchanged |
| BERT (p50) | 418.0 ms | **421.4 ms (+0.8 %)**; EM / F1 88.0 / 90.3, bit-exact |

**Reading.**
- **HP0 writes cost a little.**  Every write-heavy case is slower, none
  faster: the large binary VectorOP calls +3…9 %, the global pools +2…4 %,
  the demos +0.5…0.8 %.  The write path to DDRC port 3 through the FPD
  switch is longer than HPC0's through the CCI (experiment 1 saw the same
  sign on one pooling case, +1.4 %), and with b already on HPC1 nothing is
  left for the move to relieve: a's reads never contended with c's writes
  (§4.3).
- **Bit-exact throughout**: the 159 models, the demos' results and
  accuracies, the benchmark outputs.

**Phase 4: the decision (2026-10-09).**  Dropped: the user asked to move
the outputs from HP0 to HP2 instead (§7).  The board was put back on
`8599aa7a5f12` and the chat server restarted (healthy) before the HP2 build;
`bitstream_config_kv260.exp3.json` (local, untracked) names the HP0 archive.

## 7. Experiment 4: the outputs on HP2 (2026-10-09)

HP0's path (FPD switch → DDRC port 3, shared with DisplayPort) cost 0.5–1 %
end to end against HPC0's through the CCI.  `S_AXI_HP2_FPD` (SAXIGP4)
reaches the DDRC through the FPD switch's other leg, port 4 (shared with
HP1, unused here): the user asked to try it.

### 7.1 Design

The layout of §6.1 with `axi_out_intercon` → `S_AXI_HP2_FPD`:
`scripts/bd_hp0_outputs.tcl -port HP2` re-targets the interconnect's M00,
enables `S_AXI_GP4` at 128 bits, clocks `saxihp2_fpd_aclk` from the MMCM,
disconnects and disables HP0 (`PSU__USE__S_AXI_GP2` 0) and swaps the four
writers' address segments to `HP2_DDR_LOW` / `HP2_DDR_HIGH` / `HP2_QSPI`.
The testbench's reset check and slave profile name HP2; the loader already
programs every `C_SAXIGP*_DATA_WIDTH` of the HWH (AFIFM4 at `0xFD3A_0000`),
and `read_kernel_regs.sh` now prints HP1–HP3 too.

### 7.2 Phases 0–3

| gate | result |
|---|---|
| `bd_hp0_outputs.tcl -port HP2` on the §6 design | `validate_bd_design` clean; `HP0 released`; the eight `*_LPS_OCM` BD 41-1359 (HPC1's, HP2's); writers on HP2 segments only |
| `sim_hw_kv260` (HP2 VIP: reset check as a warning, BEST_CASE slave profile) | 75 / 75; PS VIP — HP2: 191 writes, 0 reads; HP0: nothing; HPC0: 467 reads; HPC1: 82 reads |
| `build_hw_kv260` | **WNS +0.021 ns, WHS +0.010 ns** (production +0.114, HP0 +0.076 — the same netlist but for the port, so place-and-route variation); 62 248 LUT, 123 BRAM, 712 DSP; bitstream **`4a61fc965037`**, archived in `/mnt/data/bitstreams/kv260_250_4a61fc965037/`; loaded with the local `bitstream_config_kv260.exp4.json` |
| load | id `4a61fc965037`, fpga0 operating, all twelve AFIFM width fields 0 (HP2 = AFIFM4 included); the overlay applied cleanly |
| `run_remote_tests` (159 models) | **159 / 159** |
| `run_remote_perf` (63 cases), two runs, against the `kv260-8599aa7a5f12` baseline | 63 / 63 correct; per-kernel mean Δ VectorOP **+0.8 %**, Matmul +0.6 %, Conv +0.1 %, Pool **+1.1 %** (HP0: +0.9 / +0.8 / +0.1 / +1.1) |
| VectorOP ADD-16K / 64K / 256K | 16.9–16.7 / **52.1–53.1** / **192.3–192.2** µs (+3…4 / **+5…7** / **+3.8 %**; HP0 +1…3 / +6…9 / +3…4) |
| VectorOP MUL-16K / 64K, ADD / MUL-bcast-8x16K, DIV-4K | +1…3 / +2.4, +2 / +2, +3.3 % |
| VectorOP ADD-1K / 4K | 5.0 → 4.6 / 6.9 → 6.6 µs (−7 / −4 %: the shortest calls, the write response path a little shorter?) |
| unary VectorOP, softmax, MUL-bcast-dw, ConvKernel | unchanged |
| PoolingKernel GlobalAvgPool-7x7-64 | +3…4 % (as HP0) |
| MatmulKernel 64x64x64 | 14.7 → 15.3–15.4 µs (+4…5 %; HP0 +6 / +1); the others within noise |
| MNIST convnet / LeNet | 0.1113 / 1.2567 ms (+0.7 / +0.5 %; HP0 +0.5 / +0.6); 98.92 % / 97.35 % |
| MobileNet v1 / v2, ResNet-18 | 21.97 / 20.36 / 20.23 ms (+0.05 / +0.03 / **+0.25 %**; HP0 +0.0 / +0.2 / +0.7) |
| BERT (p50) | **418.7 ms (+0.15 %**; HP0 +0.8 %); EM / F1 88.0 / 90.3, bit-exact |

**Reading.**  HP2 behaves like HP0: the memory-bound binary VectorOP calls
lose 2–7 %, the pools 3 %, everything else is flat, and the end-to-end
loss is a little smaller than HP0's (BERT +0.15 against +0.8 %) — within
the run-to-run spread of these demos rather than a property of port 4.
What the two experiments share is the cause (§8): an HP port is one DDRC
XPI behind the FPD switch, whereas a stream on HPC0 is spread over the two
CCI channels and rides their write credits.  The board runs
`8599aa7a5f12` again (id and fpga0 checked); the chat server is restarted
and answers.  The DDR APM sample of this run was lost (the sampler
produced no file; not repeated).

## 8. How the DDR controller takes the PS ports (2026-10-09)

Asked while the HP2 bitstream was building: how does the DDRC distribute
the PL ports' data over its channels?  From UG1085 (v1.8, chapters 15–17,
35) and the production board's registers.

**The six XPI ports.**  The DDRC has six 128-bit AXI port interfaces (XPI),
all synchronous to the controller clock — 533 MHz here (DDR4-2400 part run
at 2133 MT/s, `PSU__CRF_APB__DDR_CTRL` 533.33), i.e. 8.5 GB/s per port and
direction into a 64-bit DRAM bus of 17 GB/s peak.  Who reaches which port
(Table 16-4 / Figure 16-3):

| XPI | masters | path |
|---|---|---|
| 0 | LPD (RPU, PMU, CSU) | LPD switch |
| 1, 2 | CCI: APU, `S_AXI_ACE_FPD`, TCU — and **`S_AXI_HPC0_FPD`, `S_AXI_HPC1_FPD`** | CCI-400, two parallel channels, each with two QoS virtual networks |
| 3 | **`S_AXI_HP0_FPD`**, DisplayPort | FPD main switch |
| 4 | `S_AXI_HP1_FPD`, **`S_AXI_HP2_FPD`** (exclusive to the two) | FPD main switch |
| 5 | `S_AXI_HP3_FPD`, FPD DMA | FPD main switch |

The FPD main switch (`TOPSW_MAIN`) runs at 533 MHz too; the PL-side AFIFM
of every port is 128 bits at the PL clock (250 MHz: 4 GB/s per direction),
with an asynchronous crossing.

**The CCI spreads each HPC port over both channels.**  The TRM says the
CCI's two DDR master ports each reach all of DDR and that a region is
"allocated to one of the master ports" — an address split, not a per-slave
one.  Measured with the DDR APM (`0xFD0B0000`, six slots = the six XPIs,
read / write byte counters per slot, polled every 0.25 s) during a
`run_remote_perf` pass on `8599aa7a5f12`, where every write goes through
HPC0 and the reads through HPC0 and HPC1:

| XPI | read | write |
|---|---|---|
| 1 (CCI) | 6 971 MB — **51 %** | 4 469 MB — **50 %** |
| 2 (CCI) | 6 729 MB — 49 % | 4 340 MB — 50 % |
| 0, 3, 4 | 0 | 0 |

Writes from a single PL port land half on each channel: the CCI
interleaves by address at a fine grain, so HPC0 and HPC1 each see two
DDRC ports (17 GB/s of DDRC-side read capacity shared with the APU), while
an HP port has exactly one XPI (8.5 GB/s per direction; HP1 and HP2 share
theirs, HP0 shares with DisplayPort, idle on the headless board).  The idle
board moves ~9 MB/s of reads and ~13 MB/s of writes on the CCI channels
(the APU).

**The port arbiter treats our ports alike.**  The PA arbitrates the six
XPIs in tiers: (1) read / write direction — stay on the current direction
while it has credits and no port of the other direction has timed out,
reads before writes when equal, direction switches minimised; (2) port
timeouts (aging counters: `PCFGR`/`PCFGW` `*_port_priority`, enabled by
`*_port_aging_en`); (3) read class HPR over LPR / VPR (VPR and VPW become
top priority once their latency budget expires), writes BEW / VPW; (4) the
per-command AxQOS priority; (5) round-robin from the lowest port index.  On
the board every port has `PCFGR = PCFGW = 0x200f` (priority 15, aging off,
urgent on), the AFIFMs drive `AxQOS = 0` (`RDQOS` / `WRQOS` 0, issuing
capability 7 + 1 outstanding per direction), and the QoS maps put QoS 0 in
LPR / BEW on every port (`PCFGQOS0` ports 1–2: 0–3 LPR, 4–11 LPR, 12–15
HPR; ports 3–5: 0–3 LPR / BEW, 4–15 VPR / VPW with a 79-cycle timeout).
The DDR QoS controller's thresholds are 0 (`DDR_QOS_CTRL` `RD_LPR` /
`RD_HPR` / `WR_THRSLD`), so it never masks a port.  With aging off and all
traffic LPR / BEW at QoS 0, tiers 2–4 never differ between our ports: the
PA is a per-direction round-robin over the XPIs that have requests, under
the global read / write turnaround policy.

**What that means for the layouts measured here.**
- Moving the writes from HPC0 to an HP port does not change the DRAM-side
  read / write turnaround (one PA, one DRAM bus), and the CCI channels were
  never short of write credits — hence experiment 1's neutral result.
- A write stream on HP0 pays the FPD-switch path and a single XPI with a
  deeper queue ahead of it instead of two CCI channels: the +3…9 % on the
  large binary VectorOP calls, whose c stream (2.8 GB/s) then also competes
  in the PA with its own two read streams as a third requester instead of
  riding the CCI's write credits (experiment 3).
- Experiment 4 (HP2) tests whether port 4's leg of the switch behaves
  differently from port 3's; the arbiter and QoS settings are identical, so
  the expectation is "the same as HP0".
- A read port on the CCI effectively has two DDRC ports; the binary
  VectorOP's gain from HPC1 (experiment 2) came from the second AFIFM /
  CCI slave port (4 GB/s each at the PL clock), not from DDRC ports.
- The levers the TRM leaves: `AxQOS ≥ 12` on a CCI port makes its reads
  HPR (ahead of the APU's LPR reads); `AxQOS ≥ 4` on an HP port makes them
  VPR / VPW with an expiry — the AFIFM's `RDQOS` / `WRQOS` can set them
  statically per port without a bitstream change.  Not tried.

## 9. Experiment 5: static read QoS on the CCI ports (2026-10-10)

The lever §8 left: every PL port's AFIFM carries a static AxQOS for its
read and write channels (`RDQOS` / `WRQOS` at `+0x08` / `+0x1C`, `VALUE[3:0]`,
used while `RDCTRL` / `WRCTRL` `FABRIC_QOS_EN` (bit 2) is 0 — it is, and the
PS configuration sets the values to 0).  On DDRC ports 1–2 the read map
puts QoS 12–15 in the HPR class (`PCFGQOS0`), and the CCI's QVN sorts by
AxQOS into its low-latency virtual network.  The ECRTS 2021 study of the
ZU+ QoS features (Serrano-Cases et al., "Leveraging Hardware QoS to
Control Multicore Contention in the Zynq UltraScale+") confirms that the
PL FIFOs and the second-level switches relay the AFIFM's static value to
the memory controller, and that the CCI's two DDRC ports are 8 KB
address-interleaved (their APM measurement; ours in §8 agrees).  The user
asked to try it after the HP experiments.  No bitstream change: the
registers are written through `/dev/mem` on the production bitstream
`8599aa7a5f12` (chat server stopped), one `run_remote_perf` pass per
setting against the `kv260-8599aa7a5f12` baseline, then restored to 0.

| HPC0 + HPC1 `RDQOS` / `WRQOS` | 15 / 15 | 15 / 0 | 0 / 15 |
|---|---|---|---|
| VectorOP mean | **−2.0 %** | **−2.1 %** | −0.8 % |
| ADD-256K / 64K / 16K | −5.4 / −2.6 / −5.6 % | −5.4 / −3.6 / −4.3 % | −0.4 / +1.2 / −3.1 % |
| MUL-64K / 16K | −4.9 / −4.9 % | −6.0 / −4.3 % | −2.5 / −1.9 % |
| ADD / MUL-bcast-8x16K | −5.8 / −5.1 % | −5.2 / −5.0 % | −0.8 / −1.0 % |
| FC-1x512x1000-packed (GEMV) | −4.0 % | −3.9 % | −0.2 % |
| AvgPool-2x2-3x3-32-112 / MaxPool-3x3-56x56 | −5.0 / −2.1 % | −5.1 / −2.1 % | +0.1 / +2.3 % |
| unary VectorOP, softmax, ConvKernel | flat | flat | flat |
| 4–15 µs MatMul calls, DIV-4K | +2…10 % | +2…4 % | +0…10 % — the same calls jitter that much in a plain re-run of the baseline (`16x16x16` +9.5 %) |

The write QoS does nothing (the CCI ports' write map has one class), the
read QoS does it all, and it moves exactly the memory-bound calls the HP
ports made slower.  Why it helps with an idle APU: the port arbiter's
direction policy switches from writes back to reads as soon as an HPR
read port has credit (§8), the HPR reads have their own CAM share, and the
QVN's low-latency network keeps them out of the best-effort queue behind
the APU's traffic — a binary VectorOP streams reads and writes at once,
so its reads wait less behind its own writes.

A second pass with `RDQOS` 15 / `WRQOS` 0 repeated the kernel picture
(ADD-64K / 256K, MUL-64K, both bcasts and MaxPool-3x3-56x56 improved again;
`FC-1x1280x1001-packed` and `GlobalMaxPool-7x7-256` +2 % once — the two
known one-run outliers of §5.4 / §6.4).  End to end, same setting:

| demo | `8599aa7a5f12`, QoS 0 | `RDQOS` 15 on HPC0 + HPC1 |
|---|---|---|
| MNIST convnet / LeNet | 0.1105 / 1.2504 ms | 0.1111 / 1.2485 ms (+0.5 / −0.15 %); 98.92 % / 97.35 % |
| MobileNet v1 / v2, ResNet-18 | 21.96 / 20.35 / 20.18 ms | 21.94 / 20.32 / 20.13 ms (−0.1 / −0.2 / −0.2 %); top-1 unchanged |
| BERT (p50) | 418.0 ms | **417.6 ms (−0.1 %)**; EM / F1 88.0 / 90.3, bit-exact |

**Reading.**  A real 4–6 % on the memory-bound binary VectorOP calls, the
GEMV and two pooling shapes, bit-exact and free of any hardware change —
but 0.1–0.2 % end to end: after experiment 2 those calls are a small share
of every model's time (BERT's are ~5 % of 418 ms, ResNet-18's residual
adds ~4 %).  The APU-side cost of giving the PL HPR reads did not show
(the host-op phases of BERT and the demos' pre- / post-processing are
unchanged); a loaded APU (the chat server's host ops during decode) was not
measured.  The registers were restored to 0 and the chat server restarted.

**Making it persistent** would be an `upload_bitstream.py` step beside
the AFIFM width writes (a bitstream-config field such as
`"axi_qos": {"HPC0": {"rd": 15}, "HPC1": {"rd": 15}}`), plus a top-up of
the performance model's VectorOP / pooling calls and a new perf-regression
baseline — the user's call.

## 10. Experiment 6: the AFIFM issuing capability (2026-10-10)

The other per-port AFIFM knob: `RDISSUE` / `WRISSUE` (`+0x04` / `+0x18`),
the number of outstanding read / write commands the port accepts from the
PL.  The PS configuration leaves both at 7; a write of 0xFF reads back
0x1F, so the field is five bits.  The kernels declare 16 outstanding
bursts per port (VectorOP's reads: 64-beat bursts, 1 KB), so the AFIFM's 7
could cap them — but 8 × 1 KB in flight already covers 2 µs of latency at
4 GB/s, several times the CCI path's, so no gain was expected.  The user
asked to try the maximum.  Same method as §9: production bitstream,
registers through `/dev/mem`, one `run_remote_perf` pass per setting
against the `kv260-8599aa7a5f12` baseline, restored to 7 / 7 after.

| HPC0 + HPC1 `RDISSUE` / `WRISSUE` (+ `RDQOS`) | 15 / 7 | 15 / 15 | 15 / 15 + 15 | 31 / 31 | 31 / 31 + 15 |
|---|---|---|---|---|---|
| VectorOP mean | −0.7 % | −0.5 % | **−2.8 %** | −1.6 % | **−3.4 %** |
| ADD-64K / 256K, MUL-64K | +1.2 / +0.5 / −0.2 % | +3.6 / −0.2 / −1.6 % | −5.0 / −5.3 / −8.0 % | +3.0 / +0.1 / −4.1 % | −1.2 / −5.6 / −7.4 % |
| ADD / MUL-bcast-8x16K | −1.9 / −0.5 % | −1.1 / −1.3 % | −4.9 / −4.5 % | | |
| RELU-64K | 0.0 % | 0.0 % | −1.0 % | −3.3 % | −3.3 % |
| AvgPool-2x2-3x3-32-112 | −0.5 % | −0.5 % | −5.8 % | −0.6 % | −5.8 % |
| Matmul / Conv / Pool means | +1.1 / +0.4 / −0.4 % | +1.6 / +0.5 / −0.3 % | +0.6 / +0.2 / −1.0 % | **+2.2** / +0.2 / −0.2 % | **+2.6** / +0.2 / −1.3 % |

(`WRISSUE` is a four-bit field: 0x1F reads back 0xF.)

The issuing capability is not what limits these calls: 16 outstanding
instead of 8 moves nothing beyond the noise, and combined with the HPR
reads it reproduces §9's gain (−2.8 against −2.7 % on VectorOP) — the
priority does it all.  At the field's maximum, 31, the picture turns into
a trade: unary VectorOP gains (RELU-64K −3.3 %, the kernel's own 16
bursts now all in flight) but every GEMV / FC MatMul case loses 2–5 %
(`FC-1x512x1000` and its packed / gemv variants, `FC-1x1280x1001`,
`dw-12544x16x1`: MatmulKernel mean +2.2 %, +2.6 % with HPR) — the weight
stream's many outstanding 1 KB bursts on both CCI ports thrash the DDRC's
queues rather than hide latency.  Consistent with the bandwidth-delay
arithmetic: the latency was covered at 8, and the kernels' own prefetch
(16 bursts) is not the bound either.  Nothing to keep from this one;
`RDISSUE` 15 would be harmless but idle.

## 11. Experiment 7: a, b, c in different DRAM banks (2026-10-10)

The binary VectorOP's remaining gap to its ceiling (§9–§10: ~25 %, not the
port, the issuing depth, the arbiter class or the kernel's prefetch) leaves
the DRAM itself: three sequential streams whose rows collide in the same
banks.  The user asked to spread them.

**The board's DDR address map** (DDRC `ADDRMAP0–11`, read through
`/dev/mem`; uMCTL2 semantics, HIF address = byte address ≫ 3 on the
64-bit bus; `MSTR 0x81040010`: DDR4, one rank):

| byte address bits | field | from |
|---|---|---|
| 0–5 | within a 64-byte burst (BL8 × 8 B) | — |
| **6** | **bank group** (`ADDRMAP8` bg0 = 1 → HIF 3; bg1 unused: x16 devices, two groups) | consecutive 64 B bursts alternate groups |
| 7–13 | column (`ADDRMAP2–4`: col 2 → HIF 2, col 3–9 → HIF 4–10) | an **8 KB row** per bank (1 K HIF words) |
| **14–15** | **bank** (`ADDRMAP1` b0, b1 = 9 → HIF 11, 12; b2 unused: four banks per group) | |
| 16–31 | row (`ADDRMAP5/6/9/10/11`: HIF 13–28, 16 bits) | |

So a sequential stream walks one 8 KB row, then the next bank, and returns
to the same bank every 64 KB; two streams whose start addresses differ by a
multiple of 64 KB are in the **same bank with different rows at every
moment** — a row miss (precharge + activate) on every switch between them,
which the DDRC's 32-entry CAM can only partly batch away.  The benchmark's
three buffers are separate XRT allocations (CMA pages, which tend to be
large-aligned), so the natural layout is close to that worst case; the
scheduler's pools are 64-byte aligned with arbitrary sizes, so the models
see a mix.

**Probe.**  `bench_vectorop` takes an optional `layout OFF_A,OFF_B,OFF_C`
(byte offsets inside one buffer; `run_remote_perf.py` passes a case's
`"layout"` and records the physical addresses) — the three arrays 1 MB
(ADD-256K) / 256 KB (the 64K cases) apart, plus a bank phase on b and c:

| layout | b, c shifted by | banks of a / b / c at any moment |
|---|---|---|
| same-bank | 0, 0 | n, n, n |
| bank+1/+2 | 16 KB, 32 KB | n, n+1, n+2 |
| bank+2/+1 | 32 KB, 16 KB | n, n+2, n+1 |
| bank+1/+3 | 16 KB, 48 KB | n, n+1, n+3 |
| half-row | 8 KB, 24 KB | same bank, other half of the row (control: no bank change) |
| bg-flip | 64 B, 128 B | bank-group phase only (control) |

One `run_remote_perf` pass on `8599aa7a5f12` (QoS 0, issuing 7), 100 / 200
iterations per case, against the natural-allocation baseline numbers
(ADD-256K 185.2 µs, ADD-64K 49.7, MUL-64K 51.3).

**Result** (one pass; a at `0x37f00000` in every case, so the phases are
exact):

| layout | ADD-256K | ADD-64K | MUL-64K |
|---|---|---|---|
| natural (three allocations, the baseline) | 185.2 µs | 49.7 µs | 51.3 µs |
| same-bank | 184.9 (−0.2 %) | 50.2 (+1.0 %) | 50.5 (−1.6 %) |
| bg-flip (64 B, 128 B) | 183.2 (−1.1 %) | 50.3 (+1.2 %) | 49.7 (−3.1 %) |
| half-row (8 KB, 24 KB) | 158.1 (−14.6 %) | 43.0 (−13.5 %) | 43.3 (−15.6 %) |
| bank+1/+2 (16 KB, 32 KB) | 149.9 (−19.1 %) | 41.4 (−16.7 %) | 41.7 (−18.7 %) |
| bank+2/+1 (32 KB, 16 KB) | 149.0 (−19.5 %) | 40.9 (−17.7 %) | 40.7 (−20.7 %) |
| **bank+1/+3 (16 KB, 48 KB)** | **145.9 (−21.2 %)** | **40.0 (−19.5 %)** | **40.3 (−21.4 %)** |

- The natural allocation **is** the worst case (same-bank reproduces it):
  CMA hands out the 512 KB / 128 KB buffers at 1 MB / 256 KB spacing, every
  stream in the same bank as the others at every moment.
- Three distinct banks give **−20 %**: ADD-256K at 145.9 µs is 90 % of the
  kernel's ideal 132 µs (the 25 % gap of §9 was three quarters DRAM row
  conflicts); 1.5 MB in 146 µs = 10.8 GB/s of DRAM traffic, 3.6 GB/s per
  stream — now near the port's 4 GB/s per direction.
- The bank-group phase alone does nothing (the 64 B interleave already
  alternates groups within every stream), and the half-row shift — same
  bank, the other 8 KB half of the 16 KB bank-pair span — recovers two
  thirds of it, so part of the loss is the controller's same-row / hazard
  tracking between the streams, not only precharge-activate pairs.
- Nothing in hardware or registers changed; bit-exact (the benchmark
  checks nothing, but the addresses only move the data).

**What it means.**  The scheduler owns every buffer address (`src/schedule`
/ `codegen`: pool slots at `CHUNK_STRIDE`, 64-byte alignment; weights in
their own pool), and the HLS-era layout rules never looked at DRAM banks.
A bank-aware placement — give the operands of a kernel call start
addresses that differ in bits 14–15 (a 16 KB bank phase per operand) —
is a codegen-only change, bit-exact by construction.  The binary VectorOP
is the clearest customer (its share of BERT / ResNet-18 / SmolVLM / Piper
time is 4–8 %, so −20 % there is ~1–1.5 % end to end); whether the
ConvKernel's x / w / y and MatmulKernel's A / B / C streams lose the same
way is the next probe (their benches need the same `layout` option).

**The other kernels** (the same probe through `bench_matmul` / `bench_conv`
/ `bench_pool`: every case of `perf_config.json`, buffers 8 MB apart,
"same-bank" against "bank-spread" = +16 KB of bank phase per buffer, one
pass):

| kernel (cases) | spread vs natural, mean | spread vs same-bank, mean | largest consistent move |
|---|---|---|---|
| MatmulKernel (24) | −0.2 % | −0.5 % | `batch4-A-bcast` −4.1 % vs same-bank, −0.9 % vs natural; the GEMVs ±0.4 % |
| ConvKernel (10) | −0.1 % | −0.1 % | none above ±0.8 % |
| PoolingKernel (11) | 0.0 % | −1.0 % | `AvgPool-3x3-28x28` −2.4 / −3.0 %, `MaxPool-3x3-14x14` −5.1 % vs same-bank but +4.6 % vs natural (a 28 µs call: noise) |

Nothing like the VectorOP's 20 %: the ConvKernel is compute-bound and
reads its weights into the on-chip cache in bursts, the MatmulKernel's
GEMV path streams one operand (A is cached, C is small) and its tiled path
is DSP-bound, and the pooling kernel reads one stream and writes one at
the window-reduced rate — none of them keeps three full-rate sequential
streams open at once.  The DRAM-bank effect belongs to the **binary
VectorOP** (two reads + one write at 2.8 GB/s each), and in a model that
is the residual / bias / gating adds and muls.

**Scope of a scheduler change, then.**  Give the a, b, c of every binary
VectorOP call start addresses that differ in bits 14–15: the pool
allocator's slot bases get a 64 KB alignment plus a bank phase
(`(slot index mod 4) × 16 KB`), so tensors in different slots of a
VectorOP call land in different banks (weights broadcast with `b_inc` 0
replay on chip and do not matter; a constant elementwise operand lives in
the weight pool, which gets its own phase).  Cost: ≤ 64 KB of padding per
slot; bit-exact by construction; checked by the generated address table.
Worth ~20 % of the binary VectorOP's time: ~1–1.5 % end to end on BERT,
ResNet-18, SmolVLM and Piper (whose binary VectorOP share is 4–8 %), more
on graphs with many element-wise ops.

### 11.1 Implementation (2026-10-10)

`src/bank_phase.py` (`vectorop_groups`, `place_slots`), wired into the pool
layouts of `src/codegen/_core.py` (`_bank_groups`, `_place`,
`_compute_pool_layout` / `_compute_intermediate_layout(base_elems, fixed)`)
and `src/codegen/multi.py` (`pool_layout`: the weights against every
entry's groups, each entry's intermediates against its own and the weights');
`CodeGenerator(bank_phase=True)`, `MultiEntryGenerator(bank_phase=True)`,
`--bank-phase {on,off}`; `INFERENCE_BUF_POOL_SIZE_BYTES` covers the laid-out
pool.  Streams shorter than 16 KB are left alone (nothing to gain, padding
to pay — and the tiny test graphs keep their packing).  Documented in
`doc/scheduler/INFERENCE_SCHEDULER.md` §DRAM bank phases, the user guide's
option table and `SCHEDULER_DAG.md` §6; `test/test_bank_phase.py` (11 tests:
`place_slots`, the groups and their distinct phases on a synthetic graph,
`bank_phase=False` = the old packing, the growth bound, a multi-entry
project, the host emulation bit-exact).  ResNet-18: 25 VectorOP groups, all
at distinct phases, +4 KB of pool; MobileNet v2: 43 groups, +0 KB — the
coloured slots mostly differed in phase already, the placement only fixes
the ones that did not.  The scheduler suite: 1754 tests.

Board, `8599aa7a5f12`, the demos regenerated with the phases (QoS 0, the
AFIFM issuing at its default), against the `kv260-8599aa7a5f12` baseline:

| demo | baseline | with bank phases |
|---|---|---|
| MNIST convnet / LeNet | 0.1105 / 1.2504 ms | 0.1112 / 1.2491 ms (+0.6 / −0.1 %); 98.92 % / 97.35 % |
| MobileNet v1 / v2, ResNet-18 | 21.96 / 20.35 / 20.18 ms | 21.98 / 20.36 / 20.21 ms (+0.1 / +0.0 / +0.1 %); top-1 unchanged |
| BERT (p50) | 418.0 ms | **414.9 ms (−0.75 %)**; EM / F1 88.0 / 90.3, bit-exact |
| LightStereo-S 640 × 480 | 599.0 ms per pair | 598.1 ms (−0.15 %); disparity bit-exact with the simulation |

**Reading.**  BERT takes the gain its binary VectorOP share allows (its
residual and bias adds: −0.75 % of the inference); ResNet-18, the
MobileNets and the stereo model are flat because their pools already had
the operands of most VectorOP calls at different phases (ResNet-18's pool
grew 4 KB: the placement changed almost nothing) — the benchmark's natural
layout was the worst case because CMA hands separate buffers out 1 MB
apart, the scheduler's packed slots are not.  All bit-exact.  The feature
stays on: it costs nothing, fixes the layouts that do collide (BERT), and
makes the gain independent of how the slots happen to fall.

# Throughput plan — MatmulKernel, depthwise ConvKernel, VectorOPKernel

Date: 2026-09-25.  Status: **proposal, not yet executed.**
Baseline bitstream: main a7d0bce / 56fc2eb (hw_128 f432a24), PL clock 100 MHz,
all four kernels on one 128-bit `S_AXI_HPC0_FPD` port through
`axi_interconnect_0` (≈ 1.6 GB/s per direction).

Each track below was derived from a per-kernel cycle model that reproduces
the on-board `run_remote_perf.py` numbers of 2026-09-25 (matmul −2…−8 %,
VectorOP ±3 %, depthwise ±6 %), so the projections are model-based, not
guesses; every step still has to be measured on the RTL stand and the board
before the next one starts.

## 0. Where the time goes today (board, 100 MHz)

| Kernel | Case | Today | Bound by |
|---|---|---|---|
| Matmul | 256³ packed B | 18.5 ms / 1.8 GOps/s (57 % of the 3.2 GOps/s peak) | B block load (29 %) not overlapped with the K-reduction (57 %) |
| Matmul | FC 1×256×256 packed | 0.268 ms / 0.49 GOps/s | 3 of the 4 row lanes idle when n = 1 (61 % of time in a 25 %-utilised loop) |
| Matmul | FC 1×1280×1001 packed | 5.0 ms | same + B bandwidth |
| Conv depthwise | 3×3 64ch 56×56 | 9.0 ms / 0.40 GOps/s | sweep re-entered per pixel (15 cycles for 9 MACs, 44 %), 16-bit x loads serial with compute (28 %), 16-bit drain/writes (24 %) |
| Conv 1×1 | 64→128 56×56 | 18.1 ms / 2.8 GOps/s | sweep 78 %, drain 11–23 %, loads 9–18 % |
| VectorOP | ADD 256K | 2.65 ms / 0.59 GB/s | one 16-bit element per cycle per stream — its ceiling |
| VectorOP | MUL bcast 12544×16 | 6.1 ms | 33-cycle per-outer-iteration restart on 16-element runs |
| VectorOP | ADD 1K | 19 µs | ≈ 9 µs per-call overhead (register writes, polling, sync) |

Model share in the demos: MobileNet v2 (400 ms) — depthwise ≈ 117 ms,
Clip/Relu6 ≈ 65–70 ms, 1×1 convs ≈ 176 ms; ResNet-18 (372 ms) — Relu/Add
≈ 30–40 ms; MobileNet v1 (489 ms) — depthwise ≈ 97 ms.  MatMul only
appears as the classifiers (≤ 2 ms per demo), so its track matters for
transformer/FC workloads rather than the current demos.

## 1. Cross-cutting findings

**Clock.**  The implementation's worst paths are the reset network
(`rst_ps8_0_99M` → kernel FFs: 8.9 ns of pure routing, one LUT).  The worst
real datapath is ConvKernel's DSP accumulate chain (8.08 ns, DSP_ALU×4 +
two LUTs) and MatmulKernel's flow control (7.9 ns, 79 % route).  HLS
targets 150 MHz (`platforms/kv260.json` `clock: 150`) and estimates
4.9–6.7 ns, so:
- 100 → **120 MHz** needs only reset-net treatment (register the reset per
  kernel / `BUFG` on `peripheral_aresetn`) → ~1.2× on everything, low risk;
- **150 MHz** additionally needs the conv MAC tree re-pipelined (HLS
  `set_clock_uncertainty` / target 200 MHz) → deeper sweep pipeline.
This is a separate track (§5) to run after the kernel tracks, because every
kernel change re-opens timing.

**Port budget.**  1.6 GB/s per direction.  After the VectorOP and depthwise
widenings, a binary VectorOP (2 reads) alone can saturate the read channel
(4 elements/cycle effective); concurrent lanes (Conv ‖ Pool ‖ VectorOP) will
then contend.  A second PS slave port (HP0 for VectorOP + Pool, HPC0 for
Conv + Matmul) is the structural answer if the perf runs show it; not in
scope until measured.

**Per-call overhead.**  ≈ 3–5 µs per `run_*` in generated code (9 µs in the
bench, which syncs the output each iteration).  Real gain comes from fewer
calls (an `act` register on VectorOP so Add→Relu is one pass; Relu fused
into the conv/matmul writers) — noted as follow-ups, not in the three tracks.

## 2. Track A — MatmulKernel (files: kernels/matmul/*)

Peak is 16 MACs/cycle (one row lane × kTileM columns per cycle, rotating
over kTileN = 4 lanes), 3.2 GOps/s.  The csynth Bind report shows 32 DSPs
because operands are widened to `AccData_t` before the multiply.

| Step | Change | Expected | Cost / risk |
|---|---|---|---|
| A1 | 16×16 multiply (`a_val * b_tile[kk][m1]` as `Data_t × Data_t`, product type is exactly `AccData_t`) | bit-identical, DSP 32 → 16, shorter MAC latency | none |
| A1 | Issue all `n_valid` A-row requests before draining; B-resident fast path when `m_tiles == k_tiles == 1` (dw-12544×16: 12.2 → ~8 ms) | small | none |
| A2 | **K-split across lanes when `n_valid < kTileN`**: lane = (row, K-segment), `seg_len = ceil(k_valid / lanes_per_row)`, guard the tail, reduce lanes after the k-tile loop | FC 1×256×256 0.268 → ~0.14 ms; 1×1280×1001 5.0 → ~2.6 ms | ~300 LUT; low (tail guard: add C-sim cases 1×261×19, 2×13×5, 3×517×33) |
| A3 | **b_tile ping-pong** (§2.35 recipe): `b_tile[kTileM][2·kTileK]`, one RAM column per m1, flat power-of-two address, explicit column select; prefetch the next (k_tile, m_tile, n_tile, batch) block's words inside the ki loop (blocking `read()` per iteration, ≤ 512 words < 1024 iterations), first block blocking | 256³ packed 18.5 → ~12 ms; with A2 FC 1×256×256 → ~0.10 ms, 1×1280×1001 → ~1.9 ms (then port-bound) | +2–3 k LUT, BRAM unchanged (64); medium — watch "Inferring partial write", BRAM jump (RAM duplication), II=2 |
| A4 | LUT cleanup of the row-major scatter muxes (B 17.9 k, A 9 k of 44 k LUT): 128-bit rotate-by-shift then fixed lane→column wiring | frees ~15 k LUT | prerequisite for A5 |
| A5 | **kTileM 16 → 32** (not kTileN: same peak, halves FC utilisation): peak 6.4 GOps/s, balanced with A3 | 256³ packed → ~6–7 ms (2.8× total) | +16 DSP, +16 BRAM18 (~8 tiles), acc 128×32 FF; medium; platform `tile_m` → scheduler `MATMUL_TILE_M` follows, **all packed models/fixtures regenerated** |
| A6 | Row-major B: sliding 16-request window + one flattened II=1 word loop per block | 256³ row-major 32 → ~14 ms | only if an activation-B model appears (none in the 144-model suite) |
| A7 | 128-bit C writes for `m % 8 == 0` rows (second `c_wide` port; partial-strobe beats are dropped by the PS, §2.36) | C ≈ 12 % after A3+A5 | last |

Sequence A1 → A2 → A3 → (A4 → A5) → A6/A7.  Scheduler impact: none until
A5 (platform constant + regeneration), A7 (register).

## 3. Track B — depthwise ConvKernel (files: kernels/conv/*)

Shares of dw-3×3-64ch-56×56: sweep 44 % (15 cycles per 8-lane pixel-tile
for 9 MACs: `partial_outputs` word load + 5-deep flush + store per pixel),
x loads 28 % (16-bit, one element per cycle, serial with compute), drain +
16-bit y writes 24 %.  Stride-2 dw layers (MobileNet v1 has four) are 60 %
loads.

| Step | Change | Expected | Cost / risk |
|---|---|---|---|
| B1 | **Flatten the dw sweep** (§2.29 pattern): one II=1 loop over (oh, ow, ri) per (mt, owt); init `acc` from a BiasVec at `ri == 0`, store the full 8-lane word at `ri == K−1` (write-only, no RAW); skip Phase 1 for depthwise; `bias_producer` emits (ni, chunk, mt, pixel) order for dw | −20 % on dw (sweep 15 → ~9.3 cycles/tile, Phase 1 gone) | 0 BRAM/DSP; low; store all kTileM lanes (lane-subset store = §2.35 partial-write trap) |
| B2 | **8-lane drain + 128-bit y**: Phase 3 reads one 8-channel word per cycle, an 8×8 register transposer emits one word of 8 consecutive pixels of one channel into a 128-bit `acc_stream` (URAM 16 → ~4); `y` → `burst_maxi<ap_uint<128>>`; per channel run: `write_request(s/8, ceil((s%8+L)/8))`, 2-word barrel re-alignment, head/tail words via `write(word, byte_enable_mask)` (`hls_burst_maxi.h:189`) so neighbouring lanes stay intact | −20 % dw; 1×1 layers −10…−20 % (drain is 11–23 % of their time); all conv layers gain | medium; §2.30 bounds (one burst per request, ≤ 8 outstanding, in words), `max_write_burst_length=64 num_write_outstanding=8`; fixtures with `out_h·out_w % 8 ≠ 0` and multi-chunk runs |
| B3 | **128-bit x + split row loader** (pool §2.13 recipe): new DATAFLOW process `x_row_loader` owning `x`, `read_request` for all `ch_valid` runs of a row, `row_stream` of 128-bit words (depth 512); Phase 1 writes 8 lanes per cycle into `line_buf` partitioned `cyclic factor=8 dim=3` in LUTRAM (~4–5 k LUT; 128 BRAM18 would not fit), lane rotate by `run_off % 8`, channel-bank select by unrolled compare | −25 % dw after B1; stride-2 dw −45 %; standard-conv loads 9–18 % → ~2 % | +5 k LUT, +4 BRAM18; medium |
| B4 | **Depthwise on the idle grid**: ct-tile = 16 channels, `PatchBlock{lane[16][kOwPar]}` per (ow-block, khi, kwi) from B3's column-partitioned `line_buf`, `acc[16][kOwPar]`, `w_buf[16][kMaxKPos]` with power-of-two stride; kOwPar = 4 first (one URAM write port), 8 needs `RAM_T2P` | another ~2.5–3× on dw (then bound at 8 elements/cycle loads and drain) | +64…128 DSP (design has 745 free), +1–2 k LUT; high; extend `TestConvGrid` |
| B5 | dw + pointwise fusion | scheduler-level | note only |

Sequence B1 → B2 → B3 → B4, each logged as a §2.3x entry, the cycle model's
per-phase terms updated and re-validated (≤ 10 % on DW_*) before the next.
Projection: after B1–B3 dw-64ch-56×56 9.0 → ~2.9 ms, MobileNet v1 489 →
~390 ms, v2 400 → ~270 ms; after B4 another ~2.5× on the depthwise layers.

## 4. Track C — VectorOPKernel (files: kernels/vectorop/*)

Model: `T = T_call + outer × (size × 1 cycle + 33)`; every stream is one
16-bit beat per cycle (`Widen Fail` on the runtime-strided access, so
`max_widen_bitwidth` does nothing).  The codegen already guarantees what a
wide port needs: every buffer base is 64-byte aligned, `CHUNK_STRIDE` is a
multiple of 8 elements, `a_inc`/`b_inc` are 0 or `CHUNK_STRIDE`, so **every
run start of a, b, c is 16-byte aligned**; only the run length is arbitrary.

| Step | Change | Expected | Cost / risk |
|---|---|---|---|
| C1 | **128-bit `burst_maxi` ports a, b, c** (matmul §3 reads, conv §2.27/§2.30 writes): word-range requests in ≤ 256-word pieces, lanes outside `[0, size)` masked; `store_c` writes the full last word (tail lanes land in the stride gap / slot pad, which the codegen never reads), sliding response window ≤ 8; unary ops push zero words for b | binary ≥ 16K: 0.59 → ~1.6–2.1 GB/s (2.7–3.5×, reads saturate the port first); unary 0.39 → ~1.6–2.4 GB/s (4–6×) | +6–8 BRAM18, +2–3 k LUT; medium; contract "run starts 16-byte aligned, c may be written to the next 16-byte boundary" documented in VectorOP.h and asserted in C-sim + a scheduler test |
| C2 | 8 lanes per cycle in `compute` (unrolled lane loop with the op switch); DIV: 8 pipelined dividers (~5–7 k LUT) or an II=8 lane loop for DIV only (no shipped model uses Div) | required by C1 | low |
| C3 | **Flatten the outer loop**: one pipelined loop over (o, word) per stage with requests issued 8 runs ahead; fast path when `inc == size` (one contiguous range, e.g. the dw chunk-16 pattern); stride-0 operand ≤ 2048 elements read once into a `VecWord rep_buf[256]` | MUL bcast 12544×16 6.1 → ~0.3 ms | low |
| C4 | `act` register (none / relu / relu6) so the scheduler fuses Add→Relu (every ResNet-18 residual) into one pass; trim redundant register writes in `run_op` | halves that traffic; ~1 µs per call | low, scheduler change |
| C5 | Relu/Clip fused into the conv/matmul writers | MobileNet v2 Clip ≈ 17 % of inference today, ~3–5 % after C1 | note only |

Sequence C1+C2+C3 as one kernel change (they share the loop rewrite),
then C4.  Test-stand and hw block designs: instance widths must equal the
new IP defaults (128 on all three ports); the `vectorop_tb.sv` DDRC shim is
width-generic; add a fixture with a run > 16 × 256 words (§2.30).

## 5. Track D — clock (after A–C land)

D1 register/buffer the PS reset per kernel and rebuild at 120 MHz (PS PL0
clock; kernels already target 150 MHz in HLS).  D2 if wanted: raise the HLS
target so the conv MAC tree gains a pipeline stage, then 150 MHz.  Gate:
WNS ≥ 0 with incremental synthesis OFF, 144 models, demos.

## 6. Parallelisation and integration protocol

The three kernel tracks touch disjoint sources (`kernels/matmul`,
`kernels/conv`, `kernels/vectorop`, each with its own test, test-stand
project, fixtures and log document), so they can be developed by three
agents in parallel, one **git worktree + branch each**, with these rules:

1. **Per-agent verification is local**: C-sim, `make synthesize_<k>_kv260`,
   `make behavior_test_<k>` and the timing comparison run inside the
   agent's own worktree/build tree (xsim runs are CPU-bound; three in
   parallel are fine).  Each agent only edits its own kernel's test-stand
   project (block-design instance widths follow the IP defaults) and its
   own optimisation log.
2. **Shared resources are integrated serially** by the coordinator: the
   `hw/cormorant_hw_128` block design (instance widths), the bitstream
   build (one at a time, incremental synthesis OFF), the board (144 models,
   perf run, demos), the scheduler test suite.  Merge order: C (smallest,
   most demo impact) → B → A, one bitstream per merge, and any new AXI-Lite
   register proven on the board with a write-then-read before trusting
   results (§2.36 lessons).
3. **Resource budget** (design today: BRAM 81/144 tiles, DSP 503/1248,
   LUT 44 %): C +4 tiles/+3 k LUT, B1–B3 +2 tiles/+5 k LUT, A1–A3 +3 k LUT
   and −16 DSP, A5 +8 tiles/+16 DSP, B4 +64…128 DSP.  Everything through
   A3/B3/C3 fits with margin; A5 and B4 are the steps to re-check against
   the post-merge utilisation.
4. **Stop rules**: any II=2, "Inferring partial write", BRAM jump from RAM
   duplication, or a regressed RTL case is a stop-and-report, not a
   workaround (the §2.35 table lists the known traps); no port is ever
   widened in Vivado.

## 7. Expected outcome (model-based)

| Case | Today | After A1–A3 / B1–B3 / C1–C3 |
|---|---|---|
| Matmul 256³ packed | 18.5 ms | ~12 ms (→ ~6–7 ms with A5) |
| Matmul FC 1×1280×1001 | 5.0 ms | ~1.9 ms (port-bound) |
| Conv dw 3×3 64ch 56×56 | 9.0 ms | ~2.9 ms (→ ~1 ms with B4) |
| VectorOP ADD 256K | 2.65 ms | ~0.8–0.95 ms |
| VectorOP RELU 64K | 0.67 ms | ~0.12–0.17 ms |
| MobileNet v2 | 400 ms | ~220–250 ms |
| MobileNet v1 | 489 ms | ~350 ms |
| ResNet-18 | 372 ms | ~330 ms |

Decisions needed before execution: (i) A5's `tile_m` change (regenerates
every packed model and fixture) — include or defer; (ii) B4 (high risk,
largest depthwise gain) — include in this round or after B1–B3 are measured;
(iii) whether to reserve a second PS port now or wait for the perf run.

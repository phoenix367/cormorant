# ResNet-18 at 15 FPS — plan

Date: 2026-09-26.  Target: ResNet-18 (`demo/image_classification`,
`resnet18-simplified-fused.onnx`, 1814 MMAC) at ≤ 66.7 ms per image on the
KV260 at 100 MHz, from 310 ms today (after `doc/THROUGHPUT_PLAN.md`).
Status: **TARGET MET 2026-09-26 — ResNet-18 62.3 ms = 16.0 FPS at 100 MHz** (steps 1–4, 6 and 8; step 5 not needed, kept as an option).

## 0. Where the 310 ms go (board, per-layer profiler, 2026-09-26)

`deploy_and_run.py --profile-layers`, one image, warm-up excluded.  Grid
utilisation = MACs / (time × 128 MAC/cycle × 100 MHz).

| Layer class | time | share | utilisation | note |
|---|---:|---:|---:|---|
| 3×3 convs, 16 layers, 1696 MMAC | 220 ms | 71 % | 59–60 % on every layer (64→64 @56², 128 @28², 256 @14², 512 @7² all alike) | structural, not bandwidth |
| 7×7 s2 stem, 3→64, 112², 118 MMAC | 57 ms | 19 % | 16 % | 3 of 16 IC lanes busy |
| MaxPool 3×3 s2 on 112²×64 | 16 ms | 5 % | 0.13 GB/s | pool kernel 1 element/cycle |
| 1×1 s2 downsamples ×3, 6.4 MMAC each | 11 ms | 4 % | 14 % | 3.6 ms per layer |
| Relu ×9, Add ×8, global pool, FC, other | 6 ms | 2 % | | |

Ceiling of the current array: 1814 MMAC / (128 MAC × 100 MHz) = **142 ms at
100 % utilisation**, so the target needs more MACs and/or a higher clock as
well as the fixes below.  Host launch + poll overhead is < 10 µs per call
(≈ 40 calls, < 0.5 ms total): a hardware sequencer is not a lever here.
DDR traffic per inference ≈ 23 MB weights + ~16 MB activations ≈ 25 ms on
one 128-bit port at 100 MHz — 8 % today, 37 % of the target budget, so a
second PS port becomes necessary once steps 4–5 land (step 6).

## 1. Steps

| # | Step | Files | Expected | Cost / risk |
|---|---|---|---|---|
| 1 | **Stem via space-to-depth** (scheduler transform + host-side reorder in the generated C): Conv 7×7 s2 pad 3 with ic ≤ 4 → block-2 space-to-depth of the input (ic×4 channels, H/2×W/2) + Conv 4×4 s1 pads [top 2, left 2, bottom 1, right 1] with re-indexed weights (`w'[m][(ph·2+pw)·ic+c][R][C] = w[m][c][2R+ph−1][2C+pw−1] (pad 3; general `2R+ph+p−2·ceil(p/2)`)`, zero where out of the 7×7 range).  The kernel already takes `pad_top/pad_left` explicitly and zero-pads bottom/right by bounds check. | `inference-scheduler/src/graph.py`, `nodes.py`, `codegen/*`, tests, `doc/INFERENCE_SCHEDULER.md` | 57 → ~27 ms now (12/16 lanes, 16 taps instead of 49), < 10 ms after 4–5 | scheduler only; reorder runs on the A53 (150 k elements, ~0.3 ms) |
| 2 | **PoolingKernel 8 lanes/cycle** (Track-C-style): 128-bit x is already there; process a full 128-bit word (8 channels) per cycle through window/reduce/write, 128-bit y with byte strobes for tails | `kernels/pool/*`, pool test stand project, `doc/POOL_OPTIMIZATION.md` | 16 → 2–3 ms (MaxPool 3×3 s2 112²×64); all pool cases ×4–6 | pool kernel only |
| 3 | **1×1 stride-2 path**: trace why the 6.4 MMAC downsamples take 3.6 ms (14 %); expected fix is a flat II=1 sweep for kh = kw = 1 (the §2.37 recipe for depthwise) with stride handled in the x loader | `kernels/conv/*` | 11 → 2–3 ms | after step 4 in the same worktree |
| 4 | **Conv grid 16×8 → 16×16** (`tile_m` 8 → 16 in `platforms/kv260.json` + whatever the kernel needs): w_cache moves to URAM (16/64 used) since BRAM is at 119/144; drain/write path must keep up with 16 output channels | `kernels/conv/*`, `platforms/kv260.json`, `gen_conv_models.py` (fixtures re-derive), `doc/CONV_OPTIMISATION.md` | 3×3 convs 220 → ~115–130 ms if sweep efficiency holds | DSP 155 → ~285 (of 1248), LUT +15–20 k (design 62 % → ~77 %), BRAM must not grow |
| 5 | 150 MHz (Track D: reset-net fix, conv DSP chain re-pipelining) | BD, all kernels | ×1.5 | not started |
| 6 | Second PS port for conv w/b (block-design wiring only) | `hw/cormorant_hw_128` | needed after 4–5 | **done 2026-09-26, neutral at 100 MHz (§3.2)** |
| 7 | Sweep efficiency 60 → 80 %+ (cycle-level trace on a 56² fixture) | `kernels/conv/*` | 3×3 convs −25 % | delivered by step 3's flat sweep (§2.41): 89–97 % on the board |
| 8 | **Two output pixels per cycle** on the 16×16 grid (512 MACs/cycle): each sweep iteration multiplies the patch columns of `(ow, ow+1)` against the SAME weight-cache word, so the MAC count doubles without touching w_cache (the width-bound BRAM/URAM resource); patch supply, accumulators, adder trees and the drain rate are what double | `kernels/conv/*`, `doc/CONV_OPTIMISATION.md` §2.42 | 3×3 convs −40…−48 %, depthwise −35 %; 1×1 layers unchanged (drain-bound) | DSP +272 (≈ 930 of 1248 routed), LUT ≤ +15 k routed, BRAM / URAM must not grow |

Model-based outcome: steps 1–6 at 60 % efficiency ≈ 92 ms (11 FPS); with
step 7 at 80 % ≈ 74 ms (13.5 FPS); 15 FPS needs ≥ 85 %.  Int8 (2 MACs per
DSP, 16 lanes per word, half the traffic) is the alternative that makes
the target comfortable and is a separate decision to take before step 4's
grid design is frozen.

## 2. Protocol

Same as `doc/THROUGHPUT_PLAN.md` §6: one worktree + branch per step
(`/home/ivan/projects/axi_demo_wt/{stem,pool,convgrid}`, branches
`perf/stem`, `perf/pool`, `perf/convgrid`); per-agent gates are local
(C-sim bit-exact, `make synthesize_<k>_kv260` II=1 / slack ≥ 0,
`make behavior_test_<k>` all cases, timing comparison); the coordinator
integrates serially (merge → `make synthesize_<k>_kv260` in `build/` →
block design → bitstream with incremental synthesis OFF → 144 models →
perf run → demos with regenerated projects).  Stop rules: II=2,
"Inferring partial write", BRAM duplication, regressed RTL case, any port
widened in Vivado.  Resource budget today (Track A bitstream): LUT 73.2 k
/ 117.1 k (62 %), BRAM 119/144, URAM 16/64, DSP 535/1248, WNS +1.39 ns.

## 3. Measured outcome

- **Step 1 (space-to-depth stem): implemented** on `perf/stem` —
  scheduler transform `OnnxGraph(s2d_stem=True)` / CLI default, host-side
  `SpaceToDepthNode`, ResNet-18 stem becomes `SpaceToDepth(2)` + Conv
  4×4 s1 pad(2,2) 12→64 on 112² (16 taps × 12 of 16 lanes instead of
  49 taps × 3 lanes; MAC-sweep work ≈ ×0.33 per output pixel).
  **Board (merge c4cb9ce):** stem conv **57.5 → 23.9 ms**, logits
  identical — but the host reorder cost **16.3 ms** (profiler layer 0),
  not 0.3 ms: the generated loop did strided 2-byte loads straight from
  the XRT BO, whose CPU mapping is non-cacheable (~100 ns per load), so
  ResNet-18 only went 310 → 292.8 ms and MobileNet v1 / v2 regressed
  (348.7 → 362.3, 221.4 → 235.1: their 3×3 s2 stems save 2.5 ms and paid
  16.3).  Fixed on `perf/stem` by staging the reorder through a cached
  malloc'd block (two sequential memcpys are the only BO traffic);
  **re-measured after the staging fix (merge 58de431): reorder 3.0 ms**
  (memcpy-in from the non-cacheable BO mapping is the remaining cost, ~0.2
  GB/s), **ResNet-18 310 → 279.2 ms**, MobileNet v1 348.7 / v2 221.5 ms
  (unchanged: −2.5 ms stem, +3.0 ms reorder), predictions identical.
  Remaining lever for the 3 ms: allocate the buffer pool cacheable
  (XCL_BO_FLAGS_CACHEABLE, syncs already emitted) — decide separately, it
  changes every buffer's coherency model.  Note the weight index is `2R + ph − 1` for pad 3 (the `−3` in §1 was a
  typo): `t = 2R + ph + p − 2·ceil(p/2)`, see `doc/INFERENCE_SCHEDULER.md`
  §Space-to-depth stem.

- **Step 2: implemented** on `perf/pool` (POOL_OPTIMIZATION.md §2.14) — RTL
  −56.2 % on the 31 common pool cases (1,162,425 → 509,665 ns, every case
  faster, −74/−78 % on the 2×2 s2 wide-W cases), 43/43 RTL and 45/45 C-sim
  bit-exact; the ResNet-stem-shaped fixture (MaxPool 3×3 s2 on 28×112) runs
  in 48.6 µs, ≈ 1.3 ms for the full 112²×64 layer by the cycle model (board
  figure pending integration).  HLS estimate BRAM 44 → 11, LUT +4.9 k.  The
  `PoolingKernel_0` instance needs `C_M_AXI_GMEM1_DATA_WIDTH = 128`.

- **Step 4 — implemented on `perf/convgrid` (CONV_OPTIMISATION.md §2.40),
  RTL −41…−43 % on the 64-channel 3×3 fixtures, −38 % on the 7×7 s2
  stem, −42 % on the depthwise cases, 46/46 RTL PASS; csynth BRAM18
  165 → 127, DSP 262 → 409, LUT 68.4 k → 84.3 k, URAM 16 → 48 (w_cache
  half in URAM).  Board numbers pending integration.**
- **Step 3 — implemented on `perf/convgrid` (CONV_OPTIMISATION.md §2.41),
  as ONE flat II=1 sweep per (ict, ow_tile, M-group) for every kernel
  size (a 1×1-only loop duplicated the DSP grid) plus skipped rows in the
  x loader for stride > window: RTL −75 % on the 1×1 fixtures, a further
  −28 … −54 % on the 3×3 fixtures (this is also step 7's lever), 48/48
  RTL PASS; DSP 407, LUT 87.0 k, BRAM / URAM unchanged.  Model: each
  1×1 s2 downsample 3.6 → 0.4–0.5 ms, `1x1-64to128-56x56` 14.6 → ~2 ms,
  the 3×3 layers 220 → ~75 ms.  Board numbers pending integration.**

- **Step 8 — implemented on `perf/conv2px` (CONV_OPTIMISATION.md
  §2.42), RTL −44.3 % on the sweep-bound 3×3 64→64 28×28 anchor
  (1 274 695 → 710 225 ns), −43 % on the 7×7 stem, −41 % on the 14×14
  multi-tile case, −32 % on the depthwise chunking case, −23 % on the 47
  common cases (the suite is dominated by write-bound 1×1 cases, which
  are unchanged by design); 58/58 RTL PASS, bit-exact.  Each sweep
  iteration multiplies the patch columns of two adjacent output pixels
  against ONE weight-cache word (512 MACs/cycle); the accumulator URAM
  keeps its single read / write port by staggering the two pixels'
  words over the window.  csynth: DSP 407 → 679, LUT 87.0 k → 99.4 k
  (routed-equivalent ≈ +5 k), BRAM 127 / URAM 48 unchanged, II=1,
  slack 0.00.  Model: the sixteen 3×3 layers 70.4 → ~40 ms, ResNet-18
  ≈ 60 ms (~16 FPS) before step 5.  Board numbers pending integration.**

### 3.1. Board results, steps 1–4 together (2026-09-26)

Pool bitstream (steps 1+2): WNS +1.46 ns, LUT 65.9 %, BRAM 111.5/144,
URAM 16; 144/144 models.  Final bitstream (steps 1–4): WNS +1.18 ns,
**LUT 72.4 %, BRAM 111.5/144, URAM 48/64, DSP ≈ 680**; 144/144 models,
predictions identical on every demo.

| Demo | before | after | |
|---|---:|---:|---:|
| ResNet-18 | 310.1 ms | **91.2 ms** | 3.4× (11.0 FPS) |
| MobileNet v1 | 348.7 ms | 87.4 ms | 4.0× |
| MobileNet v2 | 221.4 ms | 71.3 ms | 3.1× |
| MNIST convnet / LeNet | 0.729 / 7.29 ms | 0.388 / 5.83 ms | |

Perf cases (before → after): 3x3-64ch-56x56 15.43 → **4.94 ms (46.8
GOPS, 91 % of the 256-MAC grid)**, 3x3-64ch-28x28 3.88 → 1.27, 1x1-64to128-56x56
14.64 → 1.84, 1x1-128to256-28x28 14.17 → 1.64, 5x5-16ch 0.60 → 0.25,
dw-3x3-64ch-56x56 2.86 → 1.58, dw-3x3-32ch 0.39 → 0.23; MaxPool/AvgPool
2x2/3x3 56x56 3.82 → 0.34, AvgPool-2x2-3x3-32-112 7.57 → 0.68,
GlobalAvgPool-7x7-1024 1.74 → 0.28.

ResNet-18 per layer after steps 1–4 (91.5 ms sum):

| Layer class | time | share | grid utilisation (256 MAC) |
|---|---:|---:|---:|
| 3×3 convs, 16 layers | 70.4 ms | 77 % | 89–97 % |
| stem 4×4 12→64 (s2d) | 9.7 ms | 11 % | 62 % |
| host reorder | 3.0 ms | 3 % | non-cacheable BO memcpy |
| Relu ×9 + Add ×8 | 4.0 ms | 4 % | |
| 1×1 s2 downsamples ×3 | 2.3 ms | 3 % | 21–51 % (drain-bound, tiny) |
| MaxPool | 1.35 ms | 1.5 % | was 15.8 |
| FC, global pool, rest | 0.9 ms | 1 % | |

The 3×3 layers now sit at the 100 MHz compute floor (1696 MMAC / 25.6
GMAC/s = 66 ms at 100 %), so the remaining 25 ms to 15 FPS can only come
from the clock: **step 5 (150 MHz) projects 3×3 ≈ 47 ms, stem ≈ 6.5 ms,
total ≈ 63 ms ≈ 16 FPS**, with step 6 (second PS port for weights)
needed because the 512-channel layers' 4.7 MB of weights then take 2 ms
of a 2 ms layer on one 2.4 GB/s port.  Secondary levers: cacheable buffer
pool (−3 ms), Relu/Add fusion into the conv drain (−4 ms), stem lane
utilisation (12 of 16 lanes).

Incident: the perf case `GlobalMaxPool-14x14` used a 14×14 window (the
platform bound is 7; the scheduler rejects such models).  The old kernel
tolerated it; the new one hangs and wedges the HPC port until a reboot.
Replaced by `GlobalMaxPool-7x7-256`; a kernel-side clamp of the window
to `kMaxPoolH/W` is a robustness follow-up (needs a bitstream).

### 3.2. Step 6: second PS port (2026-09-26)

Block design: `S_AXI_HPC1_FPD` enabled at 128 bits, a second 128-bit
`axi_interconnect` (`axi_mem_intercon`, 3 masters) carries ConvKernel
`gmem1` (w), `gmem2` (b) and MatmulKernel `gmem1` (B); `axi_interconnect_0`
keeps the other nine masters on HPC0.  The loader programs both AFIFM
widths from the HWH (`C_SAXIGP1_DATA_WIDTH`), `read_kernel_regs.sh` shows
HPC0/HPC1 rd/wr = 0 (128-bit).  Vivado trap: the moved masters keep their
HPC0 address segments, which then collide with the HPC1 ones — delete
the stale `SEG_*HPC0*` segments before `assign_bd_address`.  Bitstream
WNS +1.30 ns, LUT 72.7 % (+0.3 k), BRAM/URAM unchanged; 144/144.

**Result: no measurable change at 100 MHz.**  ResNet-18 91.18 → 91.19 ms,
every layer class within ±0.04 ms (stage-4 512-ch 3×3 layers 17.33 →
17.36 ms), MobileNet v1/v2 87.4/71.1, perf cases identical (3x3-64ch
4.941 → 4.941, 1x1-64to128 1.838 → 1.838, FC-1x1280x1001-packed 1.747 →
1.748; row-major 256³ 7.24 → 7.39 ms, +2 %, the only mover).  The
single HPC0 port was therefore not a bottleneck anywhere on this
bitstream — the 3×3 layers are compute-bound at 89–97 %.  Kept in the
design (it costs nothing) because it becomes necessary at 150 MHz, where
the 512-channel layers' weight stream (4.7 MB per layer) would fill one
2.4 GB/s port.

### 3.3. Step 8 on the board: two pixels per cycle (2026-09-26) — target met

Bitstream: WNS +1.18 ns, LUT 85.5 k (73 %), **DSP 1009/1248 (81 %)**,
BRAM 111.5/144, URAM 48/64 (the +5 k routed LUT estimate was right: the
sweep's runtime-g accumulator muxes went away); 144/144 models; demo
predictions identical.

| Demo | before (§3.1) | after | |
|---|---:|---:|---:|
| ResNet-18 | 91.2 ms | **62.3 ms** | **16.0 FPS** |
| MobileNet v1 | 87.4 ms | 83.1 ms | |
| MobileNet v2 | 71.3 ms | 65.9 ms | |
| MNIST convnet / LeNet | 0.388 / 5.83 ms | 0.266 / 5.44 ms | |

Perf: 3x3-64ch-56x56 4.94 → **2.68 ms (86 GOPS = 84 % of the 512-MAC
grid)**, 3x3-64ch-28x28 1.27 → 0.71, 3x3-64ch-56x56-s2 1.27 → 0.71,
5x5-16ch 0.253 → 0.155, 3x3-1ch-28x28-32out 0.223 → 0.149, dw-3x3-64ch
1.58 → 1.01, dw-3x3-32ch 0.224 → 0.154; 1x1 cases unchanged (write-bound
by design).

ResNet-18 per layer (91.5 → 62.7 ms):

| Layer class | before | after | share |
|---|---:|---:|---:|
| 3×3 convs ×16 | 70.41 ms | 45.60 ms | 73 % |
| stem 4×4 12→64 | 9.67 ms | 5.65 ms | 9 % |
| host reorder | 2.96 ms | 2.92 ms | 5 % |
| 1×1 s2 downsamples ×3 | 2.31 ms | 2.31 ms | 4 % |
| Relu ×9 | 2.04 ms | 2.04 ms | 3 % |
| Add ×8 | 1.92 ms | 1.92 ms | 3 % |
| MaxPool | 1.35 ms | 1.35 ms | 2 % |
| FC, global pool, other | 0.87 ms | 0.87 ms | 1 % |

The 3×3 layers run at 73 % of the 512-MAC grid.  What is left: the
stem (5.7 ms at 12 of 16 lanes and 4×4 taps — a 3-channel-aware stem
or 16-lane repack is the next single lever), the host reorder (2.9 ms,
cacheable BO), Relu/Add (4 ms, fusable into the conv drain), the 1×1
downsamples (5 ms, drain-bound).  Step 5 (150 MHz) would now be a
further ×1.4 on the conv layers but needs timing work at DSP 81 %.


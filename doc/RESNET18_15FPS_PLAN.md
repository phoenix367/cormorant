# ResNet-18 at 15 FPS — plan

Date: 2026-09-26.  Target: ResNet-18 (`demo/image_classification`,
`resnet18-simplified-fused.onnx`, 1814 MMAC) at ≤ 66.7 ms per image on the
KV260 at 100 MHz, from 310 ms today (after `doc/THROUGHPUT_PLAN.md`).
Status: **steps 1–2 in progress**, steps 4 and 3 implemented on `perf/convgrid` (§3; step 3 also delivers most of step 7); 5–6 not started.

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
| 1 | **Stem via space-to-depth** (scheduler transform + host-side reorder in the generated C): Conv 7×7 s2 pad 3 with ic ≤ 4 → block-2 space-to-depth of the input (ic×4 channels, H/2×W/2) + Conv 4×4 s1 pads [top 2, left 2, bottom 1, right 1] with re-indexed weights (`w'[m][ic·4+ph·2+pw][r][c] = w[m][ic][2r+ph−3][2c+pw−3]`, zero where out of the 7×7 range).  The kernel already takes `pad_top/pad_left` explicitly and zero-pads bottom/right by bounds check. | `inference-scheduler/src/graph.py`, `nodes.py`, `codegen/*`, tests, `doc/INFERENCE_SCHEDULER.md` | 57 → ~27 ms now (12/16 lanes, 16 taps instead of 49), < 10 ms after 4–5 | scheduler only; reorder runs on the A53 (150 k elements, ~0.3 ms) |
| 2 | **PoolingKernel 8 lanes/cycle** (Track-C-style): 128-bit x is already there; process a full 128-bit word (8 channels) per cycle through window/reduce/write, 128-bit y with byte strobes for tails | `kernels/pool/*`, pool test stand project, `doc/POOL_OPTIMIZATION.md` | 16 → 2–3 ms (MaxPool 3×3 s2 112²×64); all pool cases ×4–6 | pool kernel only |
| 3 | **1×1 stride-2 path**: trace why the 6.4 MMAC downsamples take 3.6 ms (14 %); expected fix is a flat II=1 sweep for kh = kw = 1 (the §2.37 recipe for depthwise) with stride handled in the x loader | `kernels/conv/*` | 11 → 2–3 ms | after step 4 in the same worktree |
| 4 | **Conv grid 16×8 → 16×16** (`tile_m` 8 → 16 in `platforms/kv260.json` + whatever the kernel needs): w_cache moves to URAM (16/64 used) since BRAM is at 119/144; drain/write path must keep up with 16 output channels | `kernels/conv/*`, `platforms/kv260.json`, `gen_conv_models.py` (fixtures re-derive), `doc/CONV_OPTIMISATION.md` | 3×3 convs 220 → ~115–130 ms if sweep efficiency holds | DSP 155 → ~285 (of 1248), LUT +15–20 k (design 62 % → ~77 %), BRAM must not grow |
| 5 | 150 MHz (Track D: reset-net fix, conv DSP chain re-pipelining) | BD, all kernels | ×1.5 | not started |
| 6 | Second PS port for conv w/b (block-design wiring only) | `hw/cormorant_hw_128` | needed after 4–5 | not started |
| 7 | Sweep efficiency 60 → 80 %+ (cycle-level trace on a 56² fixture) | `kernels/conv/*` | 3×3 convs −25 % | not started |

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

(to be filled per step)

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

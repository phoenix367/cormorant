# ConvKernel: 2-D parallel processing plan (v2)

Plan for widening the standard-conv MAC array from the current 1-D grid
(16 input-channel lanes, output channels rotated over time) to a 2-D grid
(input-channel × output-channel), **without losing the correctness that
the current index-heavy loop nest already guarantees**.

Written against the state after §2.21 (see
[CONV_OPTIMISATION.md](CONV_OPTIMISATION.md)); KV260 defaults `kTileIC=16`,
`kTileM=8`, `kMaxMperGroup=4`, `kMaxLineBufRows=16`, clock 150 MHz.  This
supersedes the v1 draft of this file: v1's micro-architecture proposal is
kept (§4), but the measured cycle breakdown (§2) changes the order of work —
**the output write path, not the MAC array, is the first wall**, and the grid
only pays after it is fixed.

---

## 1. Where the multipliers are today

`accumulate_standard` (ConvKernel.cpp) is the only place standard-conv MACs
are instantiated:

```c
for (ri = 0; ri < kh*kw*kTileM; ri++) {        // PIPELINE II=1
    m1 = ri & (kTileM - 1);                     // m rotates over TIME
    lane_sum = Σ_{ic_l<kTileIC} patch[ic_l][khi][kwi] * w[m1][ic_l][khi][kwi];
    acc[m1] += lane_sum;                        // one m lane per cycle
}
```

* **Input-channel axis is spatial** — the `ic_l` loop is fully unrolled:
  16 DSP48 multipliers per cycle feeding a 4-level adder tree.
* **Output-channel axis is temporal** — `m1` cycles through the 8
  accumulator lanes one per clock.  The rotation exists only to give each
  `acc[m1]` a RAW distance of `kTileM` cycles ≥ MAC latency (the §2.4
  II=1 trick, kept when §2.8 added the IC unroll).

So the grid is **16 × 1**: peak 16 MAC/cycle, and one output tile of 8
channels costs `kh·kw·kTileM` cycles per ic-tile.  The depthwise path is
separately **1 × 8** (8 independent lanes, `kh·kw` cycles) and is not
changed by this plan.

The matmul kernel uses the same trick in the other orientation
(`acc[n1][m1]`, `m1` unrolled, `n1` rotating) — it is a 1 × 16 grid for
the same reason.

## 2. Measured cycle breakdown — what actually bounds a layer today

The 39-case RTL run (§2.21, 100 MHz sim clock, so 10 ns = 1 cycle) lets
the ideal-II cycle model be checked against reality.  For the
`M-grouping standard (out_ch=64, 16x16, C=8, 3x3 pad 1)` case:

| Component | Model | How |
|---|---:|---|
| Compute: 256 px × 2 groups × (9 patch + 2 + 4·(8+2 + 72+7 + 8+2)) | 208 k | ideal II=1 + measured pipeline ramp per loop (csynth iteration latencies 2/7/2) |
| Weight fill into `w_cache` (2 groups × 4 × 8 × 8 × 9) | 4.6 k | serial with compute inside the consumer |
| Phase 1 bias init (16 384 elements, 1/cycle) | 16.4 k | |
| Phase 3 drain + **DDR write** (16 384 elements × 11.7 cycles) | **192 k** | see below |
| **Sum / measured** | **421 k / 434 k** | 3 % error |

The same model lands within 3 % on the depthwise chunking case (638 k
model vs 649 k measured) and the standard chunking case (873 k vs 854 k),
so it can be trusted for projections.

**The write path costs 11.7 cycles per output element.**  From the
verbose RTL log of the earlier run: all 133 896 `gmem3` write
transactions have `LEN=0` (single beat — `write_output_tile` strides
`y_addr += out_h·out_w` between consecutive stores, so HLS infers no
burst; the csynth burst table lists `gmem0/1/2` reads only, nothing for
`gmem3`).  AW-to-AW spacing is 3 cycles 62 % of the time but stalls of
24–48 cycles (the VIP's 21-cycle BEST_CASE write-response latency with
`num_write_outstanding=8` exhausted) pull the mean to 11.7.  And because
`process_conv_kernel_tile` runs Phase 1 → 2 → 3 sequentially per chunk
and `acc_stream` is only `kTileM` deep, the writer's stalls serialise
with compute: **~45 % of this layer's runtime is spent writing outputs
one element at a time.**  On the 32×32×32 chunking case it is 50 %.

Implication for the 2-D grid: if compute shrank 7× and nothing else
changed, the 64-ch layer would go 434 k → ~230 k cycles (1.9×), with the
write path then 83 % of the total.  The grid is worth ~7× only after the
write path is fixed.  Hence the order in §7.

## 3. Grid topology: broadcast grid first, systolic only if timing forces it

Two ways to build a 16 × 8 array in HLS:

**(a) Broadcast grid (output-stationary, adder tree per m column).**
Every cycle all 128 products for one `(khi, kwi)` are formed; the 16
products of column `m1` reduce through a tree into `acc[m1]`.  The patch
value of lane `ic_l` is broadcast to the 8 columns; the weight of
`(m1, ic_l)` is private.  This is the v1 proposal and what the
research notes (HLS_CONV_RESEARCH.md §2, the Zhang/Cong PM×PN pattern)
recommend at this scale.  Wiring: fanout 8 on each patch lane, 8
independent 4-level trees.  At 150 MHz (6.67 ns target; today's
estimated clock is 4.87 ns, Fmax 205 MHz) this has ~1.8 ns of slack to
absorb the fanout and the tree.

**(b) True systolic array (weight-stationary, data flows PE to PE).**
Patch values enter at column 0 and shift right one column per cycle;
partial sums flow down; weights sit in PE registers.  Only nearest-
neighbour wiring, so it scales to 32 × 32 and 300 MHz.  Cost: input and
output skew registers (`kTileM` cycles of latency per tile), weight
loading becomes a shift-in, and the per-`(khi,kwi)` accumulate loop
needs a drain of `kTileM` cycles — a real tax when `kh·kw = 1`
(pointwise) since the loop body is only 1 iteration long.  The hlslib
GEMM indexed in the RAG (`MM_PARALLELISM_N × MM_PARALLELISM_M`,
"systolic pass-through" pipes) is the reference shape; it is a
multi-process DATAFLOW design, i.e. a rewrite, not an edit.

**Decision: (a).**  128 DSPs is 10 % of the XCK26; the fanout is 8, not
32; and the design's clock target is 150 MHz, not 300.  A systolic
rewrite buys nothing until the grid grows past ~16 × 16 or the clock
doubles, and it would double the code that has to be re-verified.  If
(a) fails timing after the grid lands (§8 risk 2), the fallback is the
**semi-systolic** variant: register the patch vector once per column
(`patch_reg[m1+1] = patch_reg[m1]`) so each lane drives one register
instead of 8 multipliers — an explicit shift chain inside the same II=1
loop, no DATAFLOW restructuring.

## 4. The 16 × 8 grid (unchanged from v1)

```c
for (ri = 0; ri < kh*kw; ri++) {               // PIPELINE II=1
    const Data_t p[kTileIC] = patch[*][khi][kwi];   // one bank read per lane
    for (m1 = 0; m1 < kTileM; m1++) {          // UNROLL  ← new spatial axis
        AccData_t tree = 0;
        for (ic_l = 0; ic_l < kTileIC; ic_l++) // UNROLL  (as today)
            tree += p[ic_l] * (ic_l < ic_valid ? w_buf[m1][ic_l][khi][kwi] : 0);
        acc[m1] += tree;
    }
    // khi/kwi advance every iteration (drop the & (kTileM-1) gate)
}
```

Loop bound `kh·kw·kTileM → kh·kw` (8× fewer iterations); peak 128
MAC/cycle = 19.2 GMAC/s at 150 MHz.  The only loop-carried recurrence is
`acc[m1] += tree`, distance 1, a lone `ap_fixed<32,16>` add — the
depthwise path already closes exactly this recurrence at II=1.  UG1399
(chunk `…_chunk_12/13`): "Loop carry dependencies prevent pipelining;
use DEPENDENCE pragma to override" — there is no *false* inter-iteration
memory dependence here (acc is a fully partitioned register array), so no
`DEPENDENCE` pragma should be needed; if HLS folds the tree into the
recurrence and reports II=2, register `tree` into a temp first (one
extra latency cycle).

**Why M and not `ow` or kernel position as the second axis** (unchanged
from v1): all `m1` lanes share the same patch read, so the §2.18 patch
banking is untouched; kernel-position parallelism gives zero speed-up on
1×1 layers; `ow` parallelism needs multi-column line_buf access.

**Bit-exactness.**  `AccData_t` is `ap_fixed<32,16>` with wrap
semantics, so addition is associative modulo 2³²: changing the
reduction order (tree per column instead of the serial rotation) gives
**bit-identical** results.  The existing C-sim oracle comparison stays
exact — no tolerance relaxation is needed, which is what makes the
differential testing in §6 possible.

## 5. Memory-system changes the grid forces

In dependency order.  Each is a prerequisite for the grid to be
II=1-useful, not an optional follow-up.

### 5.1 `w_cache`: 128 reads per cycle

Today `w_cache[kMaxMperGroup][kTileM][kTileIC][kMaxKH][kMaxKW]` is
`ARRAY_PARTITION complete dim=3` (16 ic banks, one m row read per
cycle).  The grid needs all `kTileM × kTileIC` values of one
`(mt_in_group, khi, kwi)` per cycle:

```c
#pragma HLS ARRAY_PARTITION variable=w_cache complete dim=2   // m1  → 8 banks
#pragma HLS ARRAY_RESHAPE   variable=w_cache complete dim=3   // ic_l → one 256-bit word
```

UG1399 chunk `…_chunk_2`: *"reshape combines elements into wider words
instead of creating separate arrays"* — 8 RAMs each returning a 256-bit
word (16 ic lanes) = 128 values/cycle from 8 ports, ~24 BRAM36, versus
~6–8 k LUTs if both dims were partitioned complete into 128 LUTRAM
banks.  The fill loop still writes one `Data_t` per cycle into a
reshaped word (weight_stream is ic-serial because DDR layout is
`[m][ic][kh][kw]`); UG1399 chunk `…_chunk_16` notes HLS auto-unrolls
consumers of reshaped arrays but says nothing about partial-word write
II — **measure the fill loop's II after synthesis**; if it is not 1,
fall back to `complete` partition on dim=3 (LUTRAM cost) or pack 16 ic
values per beat in `stream_load_weights` (needs a 16-deep transpose
buffer in the producer).

### 5.2 `partial_outputs`: vector accumulator load/store

The per-mt acc load and writeback loops (`m_valid` cycles each, URAM
RAM_2P, 1 element/cycle) become the largest per-pixel cost once the
reduction is 8× shorter — for a 1×1 they are 16 of 24 cycles.  Widen:

```c
#pragma HLS ARRAY_RESHAPE variable=partial_outputs cyclic factor=kTileM dim=1
```

(URAM binding kept; 8 × 32-bit = 256-bit words.)  A vector access is
single-cycle only when the 8-lane group is word-aligned, i.e.
`idx_base % kTileM == 0`.  Today
`idx_base = (oh_local·out_w + ow)·out_ch + mt·kTileM` is aligned only
when `out_ch % kTileM == 0`.  **Change the internal layout to the padded
channel stride** `m_tiles·kTileM`:

```
idx = (oh_local·out_w + ow) · (m_tiles·kTileM) + mt·kTileM + m1
```

This tightens the capacity rule to
`out_w · ceil(out_ch/kTileM)·kTileM ≤ kMaxAccPersistEntries`;
`compute_oh_chunking` must use the padded row size, and the scheduler
validator (`inference-scheduler/src/nodes.py`, the
`CONV_MAX_ACC_PERSIST_ENTRIES` check) must mirror it — the §2.21 lesson
is that a kernel-internal bound the validator does not know about is a
silent-corruption bug waiting for the right model.

### 5.3 Phase 1 / Phase 3: vector streams

Phases 1 and 3 are serial with Phase 2 within a chunk; after Phase 2
shrinks ~7× they are a fixed tax of `2 × chunk_pixels × out_ch` cycles.
Retype `bias_stream` and `acc_stream` to `kTileM`-lane structs (the
`PatchVec` precedent, §2.12), one beat per `(pixel, mt)`;
`bias_producer` packs 8 lanes per beat (`bias_buf` partitioned
`cyclic factor=kTileM`).

### 5.4 Output write path — **do this first, on its own**

Two independent fixes, both measurable alone against the §2 numbers:

1. **Bursts.**  Reorder the Phase-3 drain and `write_output_tile` to
   channel-major `(mt, m1, oh_local, ow)` so `y` sees contiguous
   `out_w`-long runs.  `partial_outputs` reads become strided, but URAM
   reads are II=1 at any stride.  With `max_write_burst_length=256`
   already on the port, HLS can then infer bursts of `out_w` beats and,
   once `AXI_BUS_WIDTH` > 16, widen them.  (The indexed UG1399 chunks do
   not spell out the burst-inference preconditions; the authoritative
   signal is the csynth burst table — `gmem3` must appear in it with a
   non-trivial length after this step.)
2. **Overlap.**  Give the writer something to do while the next chunk
   computes: make `partial_outputs` a 2-bank ping-pong indexed by
   `chunk & 1` (2 × 16 URAM — fits the 64-URAM device) and split the
   Phase-3 drain out of `process_conv_kernel_tile` into its own
   DATAFLOW process.  UG1399 chunk `…_chunk_5`: DATAFLOW channels
   "default to ping-pong (pipo) buffers; use STREAM pragma to change to
   FIFO", with the limitations "single-producer single-consumer, no
   feedback paths, no conditional execution".  The chunk buffer must
   therefore be written by exactly one process and read by exactly one;
   the accumulate process both reads and writes it, so the clean split
   is: accumulate process owns the buffer and *pushes* the finished
   chunk into a wide `acc_stream` (depth ≥ one chunk row, `type=fifo`)
   while starting the next chunk — the overlap then comes from stream
   depth, not from exposing the array as a pipo.  Sizing the FIFO to a
   full chunk (`kMaxAccPersistEntries / kTileM` vectors) costs another
   16 URAM; sizing it to one row still hides most of the writer's
   latency stalls.

Expected on the 64-ch case: the 192 k write cycles drop to ~16 k
(1 beat/element at bus width 16, less when widened) and mostly overlap
compute.  **This step alone is worth ~1.8× on that layer and is the
single largest lever available before any DSP is added.**

### 5.5 Weight-fill overlap (later)

The `w_cache` fill (`G·m_valid·ic_valid·kh·kw` cycles at 1/cycle) is
serial with compute inside the consumer — 4.6 k of 434 k cycles today
(1 %), but ~10 % after the grid.  Ping-pong `w_cache` (fill group `g+1`
while computing `g`): 2× w_cache storage, no stream-format change.
Defer until the grid lands and the share is re-measured.

## 6. Correctness strategy — containing the index surface

The user-visible risk is the one §2.21 demonstrated: the kernel carries
~15 runtime indices (`ni, chunk, ict, owt, mg, mt_in_group, oh_local,
ow, khi, kwi, ic_l, m1, slot, col_slot, idx_base`) across four DATAFLOW
processes whose loop nests must agree beat-for-beat, and a geometry
corner can be wrong while every test passes.  The plan attacks this on
four fronts; none of them is optional.

### 6.1 Isolate the PE grid from geometry

Factor the compute into a function that sees **no geometry at all**:

```c
// Pure: (patch tile, weight tile, acc-in) -> acc-out.  No ni/oh/ow/chunk/mt.
static void mac_grid(const Data_t   patch[kTileIC][kMaxKH][kMaxKW],
                     const Data_t   w_tile[kTileM][kTileIC][kMaxKH][kMaxKW],
                     AccData_t      acc[kTileM],
                     unsigned kh, unsigned kw, unsigned ic_valid);
```

Its only inputs that vary per call are `kh, kw, ic_valid` (and implicitly
`m_valid` via which acc lanes the caller keeps).  Every index that
selects *which* tile is computed stays in the existing producers and the
consumer's outer nest, which are not changed by the grid.  This is the
boundary at which the 2-D array is developed and unit-tested, so the
grid work cannot introduce a geometry bug and the geometry code cannot
introduce a grid bug.

### 6.2 Three-level test net, all bit-exact

1. **Grid unit test** (new, C-sim, seconds): random `patch`, `w_tile`,
   `acc` for every `(kh, kw) ≤ (7, 7)`, `ic_valid ∈ {1..16}`,
   `m_valid ∈ {1..8}`, compared against a scalar triple loop.  Runs on
   every edit to `mac_grid` alone.
2. **Randomised geometry sweep** (new `--sweep N` mode in
   `TestConvSim.cpp`, C-sim, ~1 min for N=300): random
   `batch ∈ [1,3]`, `in_ch ∈ [1,40]`, `out_ch ∈ [1,72]`,
   `in_h, in_w ∈ [1,70]`, `kh, kw ∈ [1,7]`, stride/dilation ∈ [1,3],
   independent pads ∈ [0,3], depthwise ∈ {0,1}, filtered to the
   scheduler's constraints, seeded, against `ref_conv`.  This is the
   test that would have caught §2.21 on the first run: the failing
   geometries (`out_ch > 32` with a chunk taller than 16 input rows) are
   ~20 % of that space.  Any failure is re-added as a named case.
3. **Differential run**: after each step in §7, the sweep is run against
   both the previous kernel and the new one; because of the §4
   bit-exactness argument, every output must match the oracle *and* the
   previous kernel exactly.  A tolerance is never introduced.

### 6.3 Executable invariants in C-sim

Guard with `#ifndef __SYNTHESIS__` and `assert`:

* `idx_base % kTileM == 0` on every `partial_outputs` vector access
  (§5.2 alignment);
* line_buf residency at every Phase-2 read: the row being read was
  loaded after any row that maps to the same slot (a per-slot "row
  tag" array in C-sim only) — this turns the §2.21 class of bug into an
  assertion instead of a silent wrong answer;
* per-(chunk, ict, owt, mg) token accounting: patch beats emitted by the
  producer == beats consumed, weights emitted == `w_cache` writes.  A
  mismatch shows up as an assert at the boundary instead of as a
  DATAFLOW hang in RTL.

### 6.4 RTL fixtures stay small, one per mechanism

The RTL run is CPU-bound in xsim at ~20 µs of simulated time per
wall-clock second (§2.21), so RTL cannot be the broad net.  Each §7 step
adds at most one fixture, chosen as the *smallest* geometry that
exercises the new mechanism (as the §2.21 stem case was shrunk to
40 ch / 32×32), and the sweep in §6.2 carries the breadth.

## 7. Step order, each gated by `/conv-verify` + the §6.2 sweep

Every step is a separate §2.2x entry in CONV_OPTIMISATION.md with
baseline-first timing.  "Gate" means: C-sim 39/39 + sweep 300/300
bit-exact vs oracle and vs previous kernel, synthesis with no II
violation and slack ≥ 0, RTL 39+/39+ PASS, timing diff recorded.

| # | Step | Touches | Expected on `M-grouping 64ch 16x16` (434 k today) | Gate extra |
|---|---|---|---|---|
| 0 | **Test net first**: `--sweep`, grid unit-test scaffold, C-sim invariants (§6.2–6.3) | test only | — | sweep passes on the current kernel |
| 1 | **Write bursts** (§5.4.1): channel-major drain + writer reorder | consumer Phase 3, `write_output_tile` | ~434 k → ~330 k | `gmem3` appears in csynth burst table |
| 2 | **Write overlap** (§5.4.2): chunk-deep `acc_stream` FIFO, writer runs concurrently | consumer, stream depth | ~330 k → ~240 k | no DATAFLOW warnings; URAM ≤ 32 |
| 3 | **`mac_grid` extraction** (§6.1), *same* 1-D schedule inside | consumer | 0 change (bit-exact, timing-neutral) | refactor-only gate |
| 4 | **Padded `partial_outputs` layout + reshape** (§5.2) + validator update | consumer Phases 1/2/3, `compute_oh_chunking`, `nodes.py`, `_conv_hw_config.py`, fixtures with `out_ch % 8 ≠ 0` | ~240 k → ~215 k (acc load/store 8→1 cycle) | scheduler tests updated for the padded rule |
| 5 | **2-D grid** (§4) + `w_cache` banking (§5.1) | `mac_grid`, `w_cache` pragmas | ~215 k → ~70 k | II=1 on the accumulate loop; DSP ≈ 128+8; fill-loop II checked |
| 6 | **Vector bias / drain streams** (§5.3) | producers, streams | ~70 k → ~45 k | |
| 7 | Re-measure; decide `w_cache` ping-pong (§5.5) and `kTileM` 8→16 from data | | | |

Steps 1–2 are independent of the grid and deliver ~1.8× on their own;
step 3 is zero-risk by construction and is what makes step 5 a local
change.  Steps 4 and 5 are the ones that change indexing and are where
the §6 net earns its keep.

Projected end state for the 64-ch case: ~45 k cycles vs 434 k (≈ 9×),
with compute ≈ 41 k of it — i.e. the layer becomes MAC-bound, which is
the point at which `kTileM` 8→16 (256 DSPs, still 20 % of the device)
becomes a JSON-knob decision rather than an architecture one.

## 7a. Execution record (2026-09-24)

| Step | Landed as | Gate | Result (64-ch 16x16 case, cycles) | Suite Δ |
|---|---|---|---|---|
| 0 | `--sweep`, `TestConvGrid`, C-sim invariants (§2.22) | sweep 300/300 on current kernel; 99/300 FAIL on pre-§2.21 kernel | — | — |
| 1 | §2.22 channel-major drain | 39/39 RTL | 434 k → 267 k | -34.9 % |
| 3+4+5+6a | §2.23 padded layout, §2.24 16×8 grid, §2.25 BiasVec (one gate; each step bit-exact in C-sim alone first) | 39/39 RTL, II=1, DSP 278 | 267 k → 82 k | -63.0 % |
| 2 | §2.26 chunk-deep URAM output FIFO | 39/39 RTL | 82 k → 78 k (multi-chunk layers only) | -1.7 % |
| write path (unplanned) | §2.27 `hls::burst_maxi` y: traced adapter buffering / deferred-tail / serialised-response behaviours (see log) | single-case traces + 39/39 RTL | drain and writer now 1.0 element/cycle | (with §2.28) -13.1 % |
| read path (unplanned) | §2.28 `hls::burst_maxi` x, row-ahead `read_request`s | 39/39 RTL | AR spacing 690 → 30 ns; DW chunking case -22 % | |
| 6b / 7 | §2.29 fused (tile, khi, kwi) loop, all G tiles' accumulators in registers, depthwise straight from the stream | 39/39 RTL, II=1 (lat 6) | 76 k → 63 k | -17.1 % |
| board fix | §2.30 write requests bounded to one burst (MobileNet v2 / ResNet-18 hung on >8-burst runs) | 40/40 RTL, 126/126 on board, demo correct | — | — |
| 7 (weights) | §2.32 128-bit weight/bias ports, tile-major `[M][ict][kH][kW][16]` layout packed by the scheduler, one w_cache word per cycle | 40/40 RTL, scheduler 1301/1301 | 63 k → 52.7 k | -3.1 % (fill-bound cases -12…-31 %) |
| 7 (weights) | §2.34 half tile (8 lanes) for a last ic-tile with ≤ 8 channels | 40/40 RTL, scheduler 1302/1302 | 52.7 k → 51.6 k | -3.5 % (stem -17 %) |
| 7 (w_cache ping-pong) | §2.35 next-slab prefetch into a second bank under the fused sweep; per-m1 RAM columns with a flat power-of-two address (7 synthesis rounds to reach II=1 at BRAM 158) | 40/40 RTL, II=1 depth 7 | 51.6 k → 50.9 k | -2.3 % (stem -14 %) |
| B1 (THROUGHPUT_PLAN §3) | §2.37 flat depthwise sweep — one II=1 loop per (mt, ow_tile), bias-seeded, write-only word store, no Phase 1 | 40/40 RTL, II=1 lat 5 | (standard case unchanged; DW chunking 145.7 k → 114.4 k) | -4.0 % (DW chunking -21.5 %) |
| B2 (THROUGHPUT_PLAN §3) | §2.38 segmented 8×8 bank-rotated LUTRAM transposer in Phase 3, 128-bit `y` with re-aligned runs and byte-strobed run ends | 43/43 RTL (3 new fixtures), II=1 | 63 k-class case: `M-grouping 64ch` 50.8 k → 36.6 k | -20.4 % on the common cases (chunking -28 %, DW chunking -25 %) |

Cumulative after §2.32: **-79.7 %** of the suite's simulated time (42.1 M
→ 8.56 M ns — the suite gained a 40th, 1.6 M-ns case in §2.30); the
64-ch reference case is **8.2×** faster than at §2.21 (434 k → 52.7 k).  What the plan got wrong: §5.4's "1 beat/element at bus
width 16" assumed burst inference would overlap draining and
transmitting — it does not (§2.27); and the read side (§2.28) was not
in the plan at all but was the depthwise layers' actual wall.  The
cycle model (§2) stayed within 6 % of RTL at every step and was what
located both.

§2.35 closed the `w_cache` ping-pong item: measured -2.3 % on the
suite and -14 % on the stem — the fill was already short after §2.32/
§2.34, so the remaining slab-switch cost is small.  Still open from §7: a min-chunks policy on top of §2.26 so single-chunk layers overlap their
write phase, a 2-elements/cycle Phase-3 drain (URAM `RAM_T2P` + 32-bit
`burst_maxi` writes with byte-enables for odd run starts), and the
`kTileM` 8 → 16 scale-up.

## 8. Risks

1. **Recurrence II.**  If HLS schedules the adder tree into the
   `acc[m1] += tree` recurrence, II=2 halves the gain.  Mitigation: a
   registered `tree` temp (§4).  Check the csynth II column before
   proceeding past step 5's synthesis.
2. **Patch fanout / timing.**  Each patch lane drives 8 multipliers.
   Today's slack is 1.8 ns at 150 MHz.  If synthesis shows negative
   slack on the accumulate stage, apply the semi-systolic register chain
   (§3) — same loop, one extra pipeline stage.
3. **Reshaped-word write II** on the `w_cache` fill (§5.1) — measure;
   fall back to LUTRAM partition.
4. **Padded-layout blast radius** (§5.2) touches every phase's indexing
   and the scheduler validator.  The `out_ch % 8 ≠ 0` cases
   (`partial M tile`, `14x14 multi-tile`, `M-grouping in_h=17`) plus
   the sweep are the regression net; step 4 is deliberately separated
   from step 5 so an indexing error cannot be confused with a grid
   error.
5. **Write-overlap deadlock.**  A chunk-deep FIFO between accumulate and
   writer has one producer and one consumer, so it satisfies the
   DATAFLOW rules quoted in §5.4; but if its depth is set below one
   chunk *and* the writer is slower than the accumulate pass, the
   accumulate process blocks on the push — that is back-pressure, not
   deadlock, and only costs the overlap.  Size from the measured write
   rate after step 1.
6. **Pointwise convs may leave ConvKernel.**  The scheduler plans to
   rewrite 1×1 convs into MatMul.  That removes the layer class for
   which the grid's pointwise gains (§4's 1×1 row) matter most; it does
   not change the 3×3 numbers or the write-path conclusion, and the
   matmul kernel has the same 1-D-grid limitation, so the §4/§5 pattern
   is reusable there.
7. **Depthwise path untouched.**  Already 8 MAC/cycle; its bottleneck is
   the patch drain and the same write path (§2), so it gets steps 1–2
   for free.

## 9. Resource budget (XCK26: 1 248 DSP, 144 BRAM36, 64 URAM)

| Resource | today | after step 6 | notes |
|---|---:|---:|---|
| DSP48 | 155 (16 std + 8 dw + geometry/address) | ~270 | 128 in the grid |
| BRAM36 | 79 | ~90–110 | §5.1 layout; +24 if `w_cache` ping-pong |
| URAM | 16 | 32 | chunk-deep output FIFO (step 2) |
| LUT | 37.6 k (32 %) | ~45 k | 8 adder trees ≈ 4 k, vector stream plumbing |

Headroom remains for `kTileM=16` (256 DSPs) after step 7.

## 10. Related files

| File | Role |
|---|---|
| `kernels/conv/kernel/ConvKernel.cpp` | all changes in §5, §4; `mac_grid` extracted in step 3 |
| `kernels/conv/test/TestConvSim.cpp` | `--sweep`, grid unit test (step 0) |
| `inference-scheduler/src/nodes.py`, `src/_conv_hw_config.py` | padded capacity rule (step 4) |
| `hw/test_data/conv_test_data/` | one new small fixture per step |
| `doc/CONV_OPTIMISATION.md` | §2.22+ entries, one per step |
| `doc/HLS_CONV_RESEARCH.md` | background: PM×PN pattern, dataflow taxonomy, DW engine notes |

# PoolingKernel — Optimization Log

This document records the structural and performance optimizations applied to
`kernels/pool/kernel/PoolingKernel.cpp` after the initial scalar implementation
shipped. Each section describes one change, the rationale, and the measured
HW behavior simulation (`make behavior_test_pool`) impact on the kv260 RTL.

For the high-level kernel description see [POOLING_KERNEL.md](POOLING_KERNEL.md);
this file is a complement focused on the optimization arc and the current
final architecture.

---

## 1. Performance progression at a glance

All numbers are total `sim_time_ns` reported by the kv260 behavior testbench
after running the full TestPoolingSim case list.

| Stage | Tests | sim_time_ns | Δ vs prior | Δ vs base |
|---|---:|---:|---:|---:|
| Baseline — sequential loop nest, no caching | 25 | 2,241,775 | — | — |
| + Line buffer, ct hoisted outer of oh | 25 | 1,892,045 | -15.6% | -15.6% |
| + W-tile loop (dormant; kMaxInW=256) | 25 | 1,918,365 | +1.4% | -14.4% |
| + Cache reduced (kMaxLineBufCols=64) | 25 | 1,918,365 | 0% | -14.4% |
| + 3 wide-W tests (W=96, W=128) | 28 | 3,631,625 | (test-set change) | — |
| + 3 batch=2 wide-W tests | 31 | 7,021,605 | (test-set change) | — |
| + Drain/reduce fusion in consumer | 31 | 5,178,425 | -26.3% | — |
| + Vector window_pipe (kTileC lanes/cycle) | 31 | 3,527,035 | -31.9% | — |
| + Producer split (Phase1/Phase2 dataflow) + unrolled valid_count | 31 | 3,416,225 | -3.1% | -51.4% |
| + poly_sqrt (drop FP sqrtf unit on LP-2 path) | 31 | 2,856,995 | -16.4% | -59.3% |
| + Fixed-point AVG reciprocal (drop FP div+mul on AVG path) | 31 | 2,214,605 | -22.5% | -68.5% |
| + kOwParallel=2 reduce (process 2 adjacent ow's per cycle) | 31 | 1,679,945 | -24.1% | -76.1% |
| + Cyclic line_buf banking + shared tile geometry | 31 | 1,485,975 | -11.5% | -78.8% |
| + Closed-form valid_count (window_emitter LUT −2.1k) | 31 | 1,472,005 | -0.9% | -79.0% |
| + 128-bit x port, word reads (§2.13) | 31 | 1,162,425 | -21.0% | -83.4% |
| **+ 8 lanes per cycle end to end: 128-bit y, LUTRAM column-banked line buffer, flattened stages (§2.14)** | **31** (43 with the new tail cases: 696,045) | **509,665** | **-56.2%** | **-92.7%** |

> **kOwParallel — shipped value is 2.** §2.10 *evaluated* `kOwParallel = 4`
> and measured 1,513,475 ns (−9.9 % from §2.9), but that value was reverted;
> `platforms/kv260.json` ships `ow_parallel: 2`. The cyclic line_buf banking
> *code* (§2.10) and the shared tile geometry (§2.11) both ship at
> `kOwParallel = 2` — the −11.5 % row above is their combined effect vs §2.9.
> The `kOwParallel = 4` figure is kept in §2.10 for the record but is not
> part of the shipped progression.

**Net result on the 31-test suite: ~4.77× faster than the post-baseline
(line-buffer-only) implementation; ~79% reduction in total HW sim time.**

For the 25 tests common to every stage the same kernel runs **~7.8× faster**
than the pre-optimization baseline (line-buffer-only equivalent on the same
test list).

---

## 2. Optimization steps

### 2.1. Dataflow restructuring (HLS DATAFLOW)

**Problem.** The original kernel was a single nested loop. Every output's
window load, reduction, and write happened sequentially in the same function,
so the m_axi read latency of `x[]` and the m_axi write latency of `y[]`
serialized end-to-end.

**Change.** Split the body into three sub-functions wired by `hls::stream`:

```mermaid
flowchart LR
    DDR_IN[("x<br/>DDR gmem0")]
    DDR_OUT[("y<br/>DDR gmem1")]
    P["input_window_producer"]
    C["process_pool_kernel_tile"]
    W["write_output_tile"]

    DDR_IN -->|m_axi read| P
    P -->|window_pipe| C
    P -->|denom_pipe| C
    C -->|acc_stream| W
    W -->|m_axi write| DDR_OUT

    classDef ddr fill:#fff7e6,stroke:#d48806,color:#874d00
    classDef stage fill:#e6f7ff,stroke:#1890ff,color:#003a8c
    class DDR_IN,DDR_OUT ddr
    class P,C,W stage
```

Top-level wraps the three calls in `#pragma HLS DATAFLOW`. STABLE pragmas on
all read-only inputs prevent HLS from inserting auto-generated synchronization
stages.

**Result.** The three stages run concurrently in HW. This was the structural
foundation; speedup measured in conjunction with the next change.

### 2.2. Line buffer with row-incremental loading

**Problem.** Adjacent pool windows for `(oh, ow)` and `(oh, ow+stride_w)`
re-read the same `x[]` rows from DDR. With `stride < pool * dilation`, each
input pixel was fetched up to `pool_h × pool_w` times.

**Change.** Hoisted the channel-tile loop `ct` outer of `oh` and added a
`line_buf[kTileC][kMaxLineBufRows][kMaxLineBufCols]` cached across the `oh`
sweep within a `(ni, ct)` chunk. Phase 1 of each oh loads only the *new*
input rows since `last_loaded_row + 1`; Phase 2 streams windows from
`line_buf` (no DDR access). The slot mapping `slot = ih & (kMaxLineBufRows-1)`
gives a circular line buffer when `kMaxLineBufRows ≥ (pool_h-1)*dil_h + 1`.

**Test predictor coupling.** `expected_dup_reads_for(tc)` in
`TestPoolingSim.cpp` was rewritten to mirror the kernel's load schedule, so
the displayed `dup_reads=actual/predicted` shows exact equality — and adapts
automatically when `kMaxLineBufCols`, `kTileC`, or geometry change.

**Result.** **-15.6% sim_time_ns.** Every input pixel is now read from DDR
exactly once per `(ni, c)`. The `MaxPool/AvgPool/LpPool 3x3 stride1 pad1`
group dropped ~32% individually (their per-output cost was dominated by
overlapping window reloads).

### 2.3. W-tile loop (relaxation: in_w > kMaxLineBufCols supported)

**Problem.** The line buffer had a hard upper bound on `in_w` equal to
`kMaxLineBufCols`. Models with wider input feature maps could not run.

**Change.** Added a runtime W-tile dimension `owt` outer of `oh`. For each
W-tile, the producer loads only the input columns the current window range
needs; at tile transitions, overlapping boundary columns are re-read from
DDR. `compute_ow_tile()` solves
`(OW-1)*stride_w + (pool_w-1)*dil_w + 1 ≤ kMaxLineBufCols` for the largest
chunk size; clamps to `out_w` when the input fits in cache.

When `in_w ≤ kMaxLineBufCols` the formula yields `ow_tile = out_w` and
`ow_tiles_w = 1` — single-tile path is bit-identical to the no-W-tile
implementation.

**Renamed `kMaxInW` → `kMaxLineBufCols`** since the constant no longer caps
input width; it sizes the line buffer's column dimension. Default reduced
from 256 to **64** (4× line buffer footprint reduction: 64 KB → 16 KB).
Hard remaining constraint: `(pool_w-1)*dil_w + 1 ≤ kMaxLineBufCols`
(a single window must fit horizontally — trivially satisfied at 64).

**Test additions.** Three `wide W=96 / W=128` cases plus three batch=2
counterparts were added to `TestPoolingSim` to exercise the W-tile boundary
path. With cache=64 the overlap-heavy variants report 16 dup_reads (=
boundary columns × rows × channels), matching the cache-aware predictor.

**Result.** Latent +1.4% on the existing 25 tests (overhead of the dormant
owt loop), unlocked support for `in_w > 64`, and -48 KB of line-buffer URAM
footprint. Per-test cycle counts on the new wide-W tests: 217k–895k ns at
this stage.

### 2.4. Drain/reduce fusion in the consumer

**Problem.** With the line buffer in place, the consumer was now the
bottleneck. C-sim reported max stream depth 55,296 — producer ~2× faster
than consumer. The consumer ran two sequential II=1 loops:

```
drain:    window_pipe → win_buf      pool_h*pool_w*kTileC cycles
reduce:   win_buf → acc[]            pool_h*pool_w*kTileC cycles
```

That's `2 × pool_h × pool_w × kTileC` cycles per output position vs the
producer's `pool_h × pool_w × kTileC`. The two consumer loops disagreed on
read order (drain = `c_l outer, kwi inner`; reduce = `c1 fastest`), so
they couldn't be naively fused.

**Change.** Aligned the orders, then fused.

1. Producer Phase 2 reordered to `(khi, kwi, c_l innermost)` — c_l cycles
   fastest, matching the reducer's natural `ri & (kTileC-1)` lane index.
2. `line_buf` ARRAY_PARTITION switched from `URAM RAM_S2P` to
   `complete dim=1` (kTileC concurrent reads, one BRAM per channel lane).
3. Consumer's two loops collapsed into one II=1 loop that reads
   `window_pipe.read()` directly into the MAX/AVG/LP update on `acc[c1]`.
4. `win_buf[kTileC][kMaxPoolH][kMaxPoolW]` removed — no longer needed.

The fused inner loop preserves the lane-rotation invariant: `acc[c1]` is
written every `kTileC` cycles, so the RAW dependency distance still covers
ap_fixed<32,16> operator latency at 300 MHz.

**Result.** **-26.3% sim_time_ns** (5,178,425 ns total). Overlap-heavy 3x3
cases dropped ~30% individually. C-sim max stream depth drop hidden by the
sequential C-sim execution model — the win was real on RTL.

### 2.5. Vector window_pipe (kTileC lanes per cycle)

**Problem.** After fusion, the producer/consumer were balanced at
`pool_h × pool_w × kTileC` cycles per output. To go faster, both sides
needed higher data rate per cycle.

**Change.** Defined a `WindowLanes` POD struct holding `Data_t lanes[kTileC]`
and changed `window_pipe` from `hls::stream<Data_t>` to
`hls::stream<WindowLanes>`. Both producer Phase 2 and consumer reduce
process one struct per cycle, with the inner `c_l` / `c1` loop fully
unrolled.

```
Producer: pool_h × pool_w cycles/output  (was kTileC× more)
Consumer: pool_h × pool_w cycles/output  (was kTileC× more)
```

`line_buf`'s existing `complete dim=1` partition gives kTileC concurrent
reads; `acc[]`'s existing `complete dim=0` partition gives kTileC parallel
update lanes.

II=1 is achieved for MAX (single-cycle compare). For AVG/LP the
ap_fixed<32,16> add has 2–3 cycle latency with 1-cycle RAW distance, so
HLS schedules the reduce loop at II=2 or II=3 — still ~3× faster than the
prior scalar fused reduce.

**Result.** **-31.9% sim_time_ns** (3,527,035 ns total). The wide-W
tests dropped ~40% (consumer was their dominant work item). Per-output
cost on `MaxPool 3x3 pad1` fell from 110k/256 = 430 ns/output to 71k/256
= 280 ns/output.

C-sim max stream depth dropped from 55,296 → 6,912 (exactly 8× = kTileC,
confirming the vectorization).

### 2.6. Producer split: row_loader + window_emitter (Phase 1 / Phase 2 dataflow)

**Problem.** Within `input_window_producer`, Phase 1 (DDR row loads) and
Phase 2 (line_buf reads + window emit) ran sequentially per oh. For
workloads where Phase 1 ≥ Phase 2 (non-overlapping pool, multi-channel
tile, wide-W with low row span) this was the residual producer-side
bottleneck.

**Change.** Split into two sub-functions wired by `row_data_pipe`:

- **`row_loader`**: pure DDR side. Iterates `(ni, ct, owt, oh, ih, c_l, iw)`
  and emits row pixels onto `row_data_pipe`. No on-chip buffer.
- **`window_emitter`**: owns `line_buf`. Drains `row_data_pipe` into the
  line buffer using the same load schedule, then emits `WindowLanes` and
  denom_pipe entries.

Both functions independently derive `load_start`, `load_end`,
`iw_load_lo`, `iw_load_hi` from the geometry — no metadata stream is
needed because per-oh row counts are deterministic.

Inside the existing top-level DATAFLOW region, the two functions run
concurrently. While `window_emitter` is reducing oh = k, `row_loader`
fetches rows for oh = k+1.

**Plus: unrolled valid_count.** The denominator counter (sequential
`pool_h × pool_w` cycles per output) was replaced with a fully-unrolled
`kMaxPoolH × kMaxPoolW` adder tree (~1 cycle on the 300 MHz clock).
Lives in `window_emitter`.  (Later replaced by the separable closed form —
see §2.12 — which keeps the ~1-cycle latency at a fraction of the LUT.)

**Result.** **-3.1% sim_time_ns** (3,416,225 ns total). Pattern matches
prediction exactly — savings concentrated on tests where Phase 1 was the
bottleneck:

| Test pattern | Δ% |
|---|---:|
| 2x2 stride2 (no overlap) | -9 to -16% |
| Multi-channel-tile (C_16, C_32) | -8 to -10% |
| Wide-W non-overlap | -8 to -10% |
| 3x3 stride1 pad1 (consumer-bound) | ~0% |

### 2.7. poly_sqrt — drop the FP sqrt unit (LP-Pool p=2)

**Problem.** LP-Pool p=2's finalize step called `sqrtf((float)acc)` to apply
the square root. Even though only one of three pool types uses this path, HLS
still has to instantiate a **floating-point square-root unit** as part of the
consumer's compiled hardware. The FP sqrt is a heavy block (multi-cycle
latency, dedicated DSP slices, large LUT footprint), and its presence in the
consumer's pipeline budget tightens timing on every iteration — not just on
LP-2.

**Change.** Replaced the FP round-trip with a fully fixed-point 3rd-order
polynomial approximation. New helper `poly_sqrt(AccData_t)` lives at the top
of `PoolingKernel.cpp`:

1. **Range reduction** — decompose `x = m × 4^k` with `m ∈ [1, 4)`, `k ∈ ℤ`.
   Find the MSB position of the raw 32-bit fixed-point value via an
   unrolled priority encoder (~5 LUT levels), then shift to land
   `m_raw ∈ [2^16, 2^18)`.
2. **Polynomial via Horner's scheme** —
   `√m ≈ 0.4434 + 0.6432·m − 0.0943·m² + 0.0077·m³`, Lagrange-interpolated
   through (1,1), (2,√2), (3,√3), (4,2). Max error ~0.22% on the
   normalized mantissa. All coefficients in `ap_fixed<16,1>` (15 frac bits);
   intermediates in `ap_fixed<24,4>`.
3. **Final scale** — `√x = √m × 2^k` via a single barrel shift on `AccData_t`.

Float fallback under `#ifndef POOL_HAVE_APFIXED` retained for non-Vitis
builds.

**Reference parity.** Both the C++ test reference (`ref_poly_sqrt` in
`TestPoolingSim.cpp`) and the inference-scheduler simulator
(`_pool_poly_sqrt` in `_simulate.py`) now mirror the kernel **bit-exactly**
— same coefficients, same range reduction, same `AP_TRN` intermediate
truncations via a `quantize_trn(v, frac_bits) = floor(v · 2^N) / 2^N` helper.
This keeps the dump-mode hex fixtures aligned with the kernel's RTL output
without any 1-LSB drift.

**Result.** **-16.4% sim_time_ns** (3,416,225 → 2,856,995 ns).

The surprise: gains were **uniform across all pool types**, not just LP-2.
Every 3x3 stride1 pad1 test dropped ~18.7% — including MaxPool and AvgPool
which never call `sqrtf`. The wide-W tests dropped 18–23%.

| Test category | Δ% |
|---|---:|
| 3x3 stride1 pad1 (MaxPool / AvgPool / LpPool) | **-18.7%** |
| Wide-W 3x3 batch=2 (Max / Avg) | **-22.5% to -22.7%** |
| Wide-W 2x2 stride2 (Max) | -17.9% to -18.5% |
| 2x2 stride2 narrow | -1.8% to -3.0% |
| Global pool (small outputs) | -0.3% to -1.5% |
| LpPool subset only | -9.4% (per-LP gain not the whole story) |

**Why the cross-cutting win.** When HLS sees `sqrtf` it reserves area in the
consumer's compiled hardware for the FP unit — even on the MAX/AVG paths
the budget is set by the heaviest operator. Removing the FP block lets HLS:

1. Reclaim the LUTs/DSPs the unit occupied
2. Schedule the consumer's reduce loop more aggressively (likely lower II
   on the AVG/LP path, fewer pipeline registers throughout)
3. Drop the dataflow region's overall resource pressure

The 22% improvement on wide-W tests — which spend the most time in the
consumer reduce — is the smoking gun. The kernel was implicitly paying the
FP sqrt cost on every consumer cycle, regardless of whether LP-2 was active.

### 2.8. Fixed-point AVG reciprocal — drop the FP divider + multiplier

**Problem.** AVG-Pool's finalize step computed the divide-by-`denom` as

```cpp
const float inv_denom = 1.0f / (float)denom_u;
result = AccData_t((float)acc[c1] * inv_denom);
```

Three FP units lived in the consumer's compiled hardware to support this:
an **FP divider** (~28 cyc, ~5 DSPs) computing `1/d` once per output
position; an **FP multiplier** (~3 DSPs) for the per-lane scale; and two
**FP↔fixed converters** for the `(float)acc` cast and the result writeback.
By the same dataflow-budgeting logic as Section 2.7's poly_sqrt: HLS sized
the consumer's pipeline budget around the heaviest operator, so MAX/LP also
paid the cost on every cycle even though they never touch the AVG path.

**Change.** Replaced the FP round-trip with a precomputed fixed-point
reciprocal LUT:

1. New `inv_denom_lookup(d)` returns `1/d` as `ap_ufixed<24, 1>`, indexed
   by `denom_u`.  The table is `constexpr`-built from
   `raw = ((1<<23) + d/2) / d` (integer round-to-nearest of `2^23 / d`)
   and stored as raw 24-bit values; the use site reconstructs the
   `ap_ufixed` by direct `.range()` bit-copy — no FP→fixed converter.
2. Sized to `kMaxLineBufRows × kMaxLineBufCols` (1024 entries with
   defaults), the worst-case denom the line buffers can hold.  HLS
   synthesises a single ~3 KiB ROM (1 BRAM18) with the table contents
   baked in at translation time.
3. The finalize multiply becomes `result = AccData_t(acc[c1] * inv_denom)`
   — a native `ap_fixed<32,16> × ap_ufixed<24,1>` multiply with a single
   fabric multiplier; no DSP-FPU.

**Numerical stability.**  LUT entries hold `1/d` to 23 fractional bits
(LSB ≈ 1.19e-7), well below `Data_t`'s 1/256 LSB.  The new path is in fact
*more* numerically accurate than the prior float path: the old `(float)acc`
cast lost ~8 bits of precision on the 32-bit `AccData_t` whenever
`|acc| > 256`, which the new ap_fixed multiply preserves.

**Reference parity.**  This change shifts the kernel's exact arithmetic on
~2.5% of AVG-Pool inputs (1-LSB drift vs an idealised `acc / denom` divide),
so three reference paths had to be re-aligned bit-for-bit:

- `TestPoolingSim.cpp::ref_avg_pool_fixed` mirrors `inv_denom_lookup` +
  the AccData_t/Data_t truncations exactly.  `ref_pool_elem`'s AVG branch
  routes through it so `--dump-data` writes y.hex fixtures that match the
  RTL output under strict equality.
- `inference-scheduler/_simulate.py::_pool2d_ref` was updated to use the
  same encoded reciprocal — `test_inference.c` compares output bytes for
  strict equality against the generated `expected/*.dat`, so any
  divergence between the kernel and the simulator would surface as a
  legitimate-cell mismatch on-device.
- `hw/test_data/pool_test_data/test_{09,10,26,29}_y.hex` regenerated for
  the four AVG-Pool behavior cases (13/13/90/190 cells changed, all by
  exactly 1 LSB).

A new C-sim subtest (`run_avg_pool_strict_test`) pins this contract from
the kernel side: 5×5 pad=2 over a deterministic LCG-generated input,
compared with strict equality (no tolerance) against `ref_avg_pool_fixed`.
Replay-against-the-old-path shows ~3% of cells would diverge there, so
the test is genuinely sensitive to a regression to the FP reciprocal.

**Result.** **-22.5% sim_time_ns** (2,856,995 → 2,214,605 ns total).

The same cross-cutting pattern Section 2.7 documented for poly_sqrt
repeats here, scaled up — removing the FP div+mul has even more reach
than removing FP sqrt because the AVG path's units sat directly in the
consumer's hot reduce/finalize pipeline rather than guarded behind
`pool_type == kPoolLp`:

| Test category | Δ% |
|---|---:|
| 3x3 stride1 pad1 (Max / Avg / Lp p=1 / Lp p=2) | **-26%** (uniform) |
| Wide-W 3x3 batch=2 (Max / Avg) | **-33% to -34%** |
| Wide-W 3x3 single-batch | **-33%** |
| Dilation=2 pool 2x2 | -20.7% |
| Multi-channel-tile (C_32 2x2 stride2) | -0.4% |
| Global pool (small outputs) | -0.7% |

The five 3x3 stride1 pad1 variants now finish within 240 ns of each other
(42,780–43,080 ns) — the consumer reduce loop runs at the same rate
regardless of pool type, confirming the FP-unit budget tax is fully gone.

**Why even larger than 2.7's win.** Two contributing factors:

1. **FP div+mul is heavier than FP sqrt** in resource and scheduling
   pressure — the divider especially is expensive — so reclaiming both
   frees more for the fabric to retime around.
2. **AVG/LP add latency dominates the consumer's II.** Section 2.5
   noted the consumer reaches II=1 only on MAX (compare is single-cycle);
   AVG/LP run at II=2–3 due to the ap_fixed<32,16> add's RAW dependency
   distance.  Removing FP units from the AVG finalize path lets HLS
   schedule the reduce loop without that combined budget, recovering
   cycles that were previously lost to the worst-case operator latency.

The wide-W cases — which spend the most cycles in the consumer reduce —
again show the largest savings, mirroring 2.7's "FP unit's dataflow tax
compounded over the longer consumer reduce" diagnostic.

### 2.9. kOwParallel reduce — process kOwParallel adjacent ow's per cycle

**Problem.** After §2.8 the consumer reduce ran at one MultiWindow per cycle
(II=1 across all pool types — see §2.5/§2.8 convergence) for a single ow
position, so the per-output-position cost was `pool_h × pool_w` cycles.
On overlap-heavy 3x3 tests and on long-row wide-W tests the consumer was
the dominant fraction of the total wall-clock, with no further headroom
left on the per-position scalar reduce.

**Change.** Duplicated the reduce hardware to process kOwParallel adjacent
ow positions in lockstep:

1. **`MultiWindow` and `MultiDenom` structs** wrap kOwParallel × kTileC
   pixels and kOwParallel denom values per FIFO transaction.  Default
   kOwParallel = 2.
2. **`window_emitter` Phase 2** iterates ow as groups of kOwParallel,
   gathering line_buf reads from kOwParallel adjacent positions per
   (khi, kwi) and emitting one `MultiWindow` per cycle.  Per-position
   denom counts are bundled into one `MultiDenom` write per group.
3. **`process_pool_kernel_tile`** holds `acc[kOwParallel][kTileC]`
   (both dims partitioned `complete dim=0`) and updates ALL kOwParallel ×
   kTileC accumulators in parallel each reduce cycle, fully unrolled.
   The reduce trip count drops to `pool_h × pool_w` per kOwParallel
   outputs — a linear-in-kOwParallel speedup.
4. **`write_output_tile`** drains kOwParallel × c_valid AccData_t per
   group; conditional `if (ow < ow_hi)` gates the m_axi store so the
   read off acc_stream stays unconditional and II=1 holds.
5. **`line_buf` annotated `BIND_STORAGE type=ram_t2p`** so each per-
   channel partitioned bank is a true-dual-port BRAM; that's what gives
   the producer kOwParallel reads per cycle on the same channel without
   structural read-port contention.

**Residual / edge-case handling.**  When `(ow_hi - ow_lo)` is not a
multiple of kOwParallel (most commonly when out_w=1, e.g. global pool
outputs), the producer pads the trailing lanes with the pool-type identity
value; the writer's `ow < ow_hi` mask drops those padded results before
DDR.  The padded cycles still flow through the pipeline so the per-group
cost is exactly `pool_h × pool_w` cycles regardless of group fullness.

**Numerical correctness.**  Per-position math is unchanged — the change
is purely structural (parallel lanes of identical computation), so the
ap_fixed accumulators produce bit-identical results vs the §2.8 single-
position reduce.  C-sim 33/33 PASS, the auto-regenerated y.hex fixtures
under `--dump-data` show **zero diffs** vs the §2.8 goldens, and the HDL
behavior test 31/31 PASS without any fixture refresh.

**Result.** **-24.1% sim_time_ns** (2,214,605 → 1,679,945 ns total).

The savings are concentrated exactly where predicted — long consumer
reduces have the most cycles to halve:

| Test category | Δ% | Δabs (ns) |
|---|---:|---:|
| Wide-W AvgPool 3x3 batch=2 | **-45.2%** | -164,880 |
| Wide-W MaxPool 3x3 batch=2 | **-44.5%** | -109,790 |
| Wide-W AvgPool 3x3 single | **-44.0%** |  -82,270 |
| Wide-W MaxPool 3x3 single | **-42.8%** |  -54,740 |
| Wide-W MaxPool 2x2 stride2 batch=2 | -29.5% |  -45,300 |
| Wide-W MaxPool 2x2 stride2 single | -25.7% |  -20,950 |
| 3x3 stride1 pad1 (Max / Avg×2 / Lp×2) | **-24.7% to -25.0%** (uniform) | -10.6k ea |
| Dilation=2 pool 2x2 | -5.1% | -1,570 |
| 2x2 stride2 narrow | -1 to -2% | ≤ -500 |
| Multi-channel-tile (C_16 / C_32) | ~0% | ≤ +50 |
| out_w=1 (Global pool / 1×1 output) | +0.5 to +1.0% | +50 to +170 |

**Why ~45% on wide-W tests vs the 50% theoretical max for kOwParallel=2.**
The wide-W tests have out_w = 96 or 128 — both even multiples of
kOwParallel — so all groups are fully populated; the reduce truly halves
its trip count.  The remaining 5% gap vs the 50% theoretical comes from
the residual non-reduce cost (Phase 1 row loads, finalize, output drain)
that doesn't speed up.  The 3x3 stride1 pad1 narrow tests (out_w=8) get
~25% because their reduce is a smaller fraction of the test time —
producer Phase 1 dominates more.

**Why out_w=1 cases get slightly slower (+0.5 to +1.0%).**
With kOwParallel=2 every group is two-wide; when the actual ow span is
1 the second lane is padded and its results are discarded.  The pipeline
cost is essentially unchanged but the writer's masked-write loop runs
2× as many drain cycles for the same output count, adding ~50-170 ns
overhead per test.  Negligible vs the wide-W wins.

**Why no change on multi-channel-tile cases (C_16 / C_32).**
These tests are dominated by Phase 1 DDR row loads and the c_tiles outer
sweep; the consumer reduce was already overlapped with the next channel
tile's Phase 1, so halving consumer cycles doesn't reduce the wall-clock
critical path.  Same diagnosis as §2.6's "Multi-channel-tile" row.

### 2.10. Cyclic line_buf banking — and the reverted kOwParallel = 4 evaluation

> **Status — evaluated, partially reverted.** The cyclic line_buf banking
> *code* described below shipped: `line_buf` is parametrised on `kOwParallel`
> and partitioned `cyclic factor=kOwParallel dim=3`. The `kOwParallel = 4`
> *value*, however, was reverted — `platforms/kv260.json` ships
> `ow_parallel: 2`, and the `factor=kOwParallel` pragma is harmless at
> factor 2. The −9.9 % / 1,513,475 ns figures below are the `kOwParallel = 4`
> evaluation, retained as a record of what that configuration achieves; they
> are **not** the shipped numbers. Raising `ow_parallel` back to 4 is a
> one-line JSON change, subject to re-verification.

**Problem.** §2.9 established kOwParallel=2 with `BIND_STORAGE ram_t2p`
on `line_buf`'s per-channel banks (2 ports/cycle/channel — exactly
enough for 2 adjacent ow's).  At kOwParallel=2 every consumer-bound
test still bottlenecked on the consumer's `pool_h × pool_w` cycle
count per ow-group; doubling parallelism is the next obvious lever.
But the dual-port BRAM ceiling caps that lever — going to
kOwParallel=4 needs **4 reads/cycle/channel**, beyond what ram_t2p
alone delivers.

**Change.** Two lines, both touching `window_emitter`:

1. JSON: `platforms/kv260.json` → `kernels.pool.ow_parallel: 2 → 4`.
2. New pragma: `#pragma HLS ARRAY_PARTITION variable=line_buf cyclic
   factor=kOwParallel dim=3` — column-cyclic-splits each per-channel
   bank into kOwParallel sub-banks.  Combined with the existing
   `complete dim=1` (per-channel) and `BIND_STORAGE ram_t2p`, this
   gives `kTileC × kOwParallel = 32` total sub-banks, each dual-port.

The kernel itself needs no other change — `MultiWindow.lanes[N]`,
`MultiDenom.d[N]`, `acc[N][kTileC]`, the producer's per-position
gather loop, and the writer's masked drain are all already
parametrised on `kOwParallel`.

**Banking analysis** — the cyclic+ram_t2p combo handles every stride
the test suite uses:

| stride_w | columns to read | factor=4 banks hit | port pressure | OK? |
|---:|---|---|---|:---:|
| 1 | ow_g+0..3 | 0,1,2,3 (4 distinct) | 1 read/bank | yes |
| 2 | ow_g..ow_g+6 step 2 | 0,2,0,2 (2 distinct) | 2 reads/bank, dual port | yes |
| 3 | ow_g..ow_g+9 step 3 | 0,3,2,1 (4 distinct, gcd(3,4)=1) | 1 read/bank | yes |
| 4 | ow_g..ow_g+12 step 4 | 0,0,0,0 (1 distinct) | 4 reads/bank — exceeds ports | **no** |

`stride_w=4` is the first hard wall.  No test in the current suite uses
it.  If a model ever does, raise `factor` to `kOwParallel × stride_w`
or replicate banks.

**Resource cost.** kTileC × kOwParallel = 32 sub-banks of
[kMaxLineBufRows][kMaxLineBufCols/kOwParallel] = 16×16 × 16 b ≈ 4 Kb
each.  Each sub-bank fits in one BRAM18 (or LUT-RAM if HLS picks),
total ≈ 32 BRAM18 — comfortably inside KV260's 144 BRAM18 budget.
Doubling banks vs §2.9's kOwParallel=2 / 16 banks.

**Numerical correctness.** As with §2.9, a cycle-level shape change —
per-cell math is unchanged — so y.hex fixtures are bit-identical to
§2.9.  C-sim 33/33 PASS, behavior test 31/31 PASS, zero fixture diffs.

**Result.** **-9.9% sim_time_ns** (1,679,945 → 1,513,475 ns total).

The savings concentrate exactly where the §2.9 row predicted — groups
whose `out_w` is divisible by kOwParallel=4 (no padded lanes) and whose
reduce dominates the wall-clock:

| Test category | Δ% (vs §2.9) | Δabs (ns) |
|---|---:|---:|
| Wide-W AvgPool 3x3 batch=2 (out_w=96) | **-31.0%** | -62,040 |
| Wide-W MaxPool 3x3 batch=2 (out_w=128) | **-30.0%** | -41,030 |
| Wide-W AvgPool 3x3 single (out_w=96) | **-29.2%** | -30,610 |
| Wide-W MaxPool 3x3 single (out_w=128) | **-27.1%** | -19,770 |
| Wide-W MaxPool 2x2 stride2 batch=2 (out_w=64) | -11.2% | -12,170 |
| Wide-W MaxPool 2x2 stride2 single (out_w=64) | -8.8% | -5,360 |
| 3x3 stride1 pad1 (Max / Avg×2 / Lp×2) | +0.2% to +0.5% (saturated) | +80 to +170 |
| Multi-channel-tile (C_16 / C_32) — Phase 1-bound | ~0% | ~0 to +100 |
| out_w not divisible by 4 (Global pool, 1×1 output) | +0.3% to +2.4% | +30 to +400 |

**Why the wide-W 3×3 tests get ~30% (not the theoretical 50%).**  The
reduce halves cleanly (`pool_h × pool_w` = 9 cycles per ow-group, but
each group is now 4 outputs instead of 2 — per-output cost goes from
4.5 to 2.25 cycles).  Of the §2.9 wall-clock ~200k ns on AvgPool W=96
batch=2, only ~140k ns was reduce — halving that trims ~70k ns; we
observed -62k ns, roughly matching prediction minus producer/writer
rebalancing overhead.

**Why the narrow 3x3 pad1 tests are unchanged (~+0.3%).**  At
kOwParallel=4 the consumer's per-group pacing is now
`max(reduce, finalize+writer-drain)` = `max(9, kOwParallel × c_valid)`
= `max(9, 4 × 4)` = 16 cycles.  The writer crossed over the reduce
and became the new bottleneck.  Halving the reduce no longer matters
when the writer drain doesn't shrink.  At §2.9 the writer was 8
cycles vs the 9-cycle reduce — perfectly balanced; at kOwParallel=4
the writer doubled to 16, so the reduce halving is wasted on these
narrow `c_valid > 0.5 × pool_h × pool_w` tests.

**Why the wide-W 2×2 stride2 tests get only -10%.**  Two effects
combine.  First, stride_w=2 maps the 4 ow positions to 2 cyclic banks
(0,2,0,2) — the dual-port BRAMs handle it but with less slack than
stride_w=1.  Second, `pool_h × pool_w = 4 ≤ kOwParallel × c_valid = 16`
for c_valid=4, so the writer is again the new bottleneck and the
reduce halving only nibbles the wall-clock.

**Why the out_w=1 cases get slightly slower (+0.3 to +2.4%).** Same
residual-padding cost as §2.9, just larger fraction now: kOwParallel=4
groups always emit 4 lanes, and out_w=1 means 3 of them are padded and
discarded by the writer.  The pipeline runs through the padded cells
anyway, so global pool's per-test cost grows by ~3 × kTileC cycles for
the wider drain.  Negligible vs the wide-W wins.

### 2.11. Shared tile geometry — collapse the per-stage dividers

**Problem.** Every dataflow stage independently recomputed the W-tile count

```cpp
ow_tiles_w = (out_w + ow_tile - 1) / ow_tile;
```

`ow_tile` is a runtime value (it depends on `stride_w` / `pool_w` / `dil_w`
via `compute_ow_tile`), so `÷ ow_tile` is a division by a **runtime
divisor** — HLS synthesises it as a multi-cycle sequential divider.  All
four stages — `row_loader`, `window_emitter`, `process_pool_kernel_tile`,
`write_output_tile` — carried their own copy, so the same divider was
instantiated **four times**.  (`c_tiles = (channels + kTileC - 1) / kTileC`
was also recomputed per stage, but `÷ kTileC` is a compile-time
power-of-two shift, not a divider.)  The geometry is invariant for a whole
kernel invocation — there is no reason to compute it more than once.

`compute_ow_tile` itself (`÷ stride_w`) was already called once in the top
function and passed down as the `ow_tile` scalar, so that divider was
already singular; only `ow_tiles_w` was replicated.

**Change.** Lift the geometry into a struct computed once.  A new
`PoolGeometry { c_tiles, ow_tile, ow_tiles_w }` is filled by
`compute_pool_geometry()` at the top of `PoolingKernel` and passed by value
to all four stages, replacing the previous `ow_tile` scalar parameter.
Each stage reads `geom.c_tiles` / `geom.ow_tile` / `geom.ow_tiles_w`
instead of recomputing them.  The struct crosses the DATAFLOW process
boundaries as one stable scalar channel.

This mirrors `ConvKernel`'s `ConvGeometry` / `compute_conv_geometry` (the
conv optimisation log's §2.19) — the same "compute the geometry once, pass
the struct to every stage" pattern.

**Synthesis.** The csynth report gains one new module —
`compute_pool_geometry`, a 72-cycle one-time block (3 DSP / 1,178 FF /
1,479 LUT) — that holds the single shared divider set in place of the four
stage-local copies.  No new timing-violation rows, no II regressions on any
previously-II=1 loop, and the `m_axi_gmem0/1` data widths are unchanged
(`16 → 16`).

**Numerical correctness.** Purely structural — the geometry values are
bit-identical, just computed once instead of four times.  C-sim 33/33 PASS,
behavior test 31/31 PASS, zero y.hex fixture changes.

**Result.** The shipped kernel (kOwParallel = 2) measures **1,485,975 ns**
total on the 31-test behavior suite. An isolated wall-clock delta for the
geometry struct alone was not separately benchmarked — §2.10's cyclic-banking
code landed in the same window — so the §1 table folds both into one −11.5 %
step vs §2.9. The geometry struct's own contribution is expected to be
small: the dividers were **one-time, per-invocation** costs (~72 cycles),
not per-output, so collapsing four into one removes setup latency and
divider hardware but never touches the steady-state reduce loop that
dominates every test's wall-clock. It is best understood as a **resource
consolidation** (one divider set instead of four) — the same class of change
§2.7 / §2.8 made for FP units, but far smaller because dividers are not in
the hot loop.

### 2.12. Closed-form valid_count — collapse the 98-lane bounds-count

**Problem.** A synthesis-report audit (`csynth.rpt`) flagged `window_emitter`
as the kernel's largest LUT consumer — 11,415 LUT, of which **7,873 LUT** was
combinational "Expression" logic. The dominant term was the MultiDenom
`valid_count`: §2.6 computed it as a fully-unrolled
`kOwParallel × kMaxPoolH × kMaxPoolW` = 2 × 7 × 7 = **98-lane** bounds-check +
popcount adder tree — 98 lanes, each doing two `int` range compares (on
`ih_v` and `iw_v`) plus a conditional increment, all feeding one wide adder
tree.

**Change.** The count is **separable**. A window position `(khi, kwi)` is
in-bounds iff its **row** is in-bounds *and* its **column** is — and the row
test depends only on `khi`, the column test only on `kwi`. The in-bounds set
is therefore the rectangle product `{valid khi} × {valid kwi}`, so

```
valid_count = num_valid_kh * num_valid_kw
```

`num_valid_kh` does not depend on the `ow` position, so it is counted once
per ow-group; `num_valid_kw` is counted per `p`. The 98-lane count becomes
`kMaxPoolH + kOwParallel × kMaxPoolW` = 7 + 2 × 7 = **21 lanes**, each lane a
single 1-D range test (two compares), plus `kOwParallel` multiplies. No
runtime divider is introduced — the per-axis counts are still tallied, not
solved per-axis in closed form, so dilation needs no special handling.

**Numerical correctness.** Exact — `|{valid khi}| × |{valid kwi}|` *is* the
count of in-bounds `(khi, kwi)` pairs, with no approximation. C-sim 33/33
PASS, behavior test 31/31 PASS, zero y.hex fixture diffs.

**Result.** A **resource** optimisation:

| Metric | before | after | Δ |
|---|---:|---:|---:|
| Kernel LUT | 31,854 | 29,713 | −2,141 (−6.7 %) |
| Kernel FF | 16,765 | 16,568 | −197 |
| `window_emitter` total LUT | 11,415 | 9,274 | −2,141 (−18.8 %) |
| `window_emitter` Expression LUT | 7,873 | 5,780 | −2,093 (−26.6 %) |

No new timing-violation rows, no II regressions, BRAM/DSP unchanged. The
shorter combinational count also trims a little wall-clock — **−0.9 %
sim_time_ns** (1,485,975 → 1,472,005), concentrated on the wide-W tests
(AvgPool W=96 batch=2 −3.5 %). The remaining 5,780 LUT of `window_emitter`
Expression is the per-`(khi, kwi)` window-gather index arithmetic, which this
change does not touch.

---

### 2.13. 128-bit x port — word reads with lane extraction

**Change.**  `x` is an `hls::burst_maxi<PoolWord>` port (`ap_uint<128>`,
8 lanes per beat; `PoolingKernel.h`).  The NCHW layout is unchanged.
`row_loader` requests, per input row, all `c_valid ≤ kTileC` channel
runs of the row first (`num_read_outstanding = 16`) so their DDR latency
overlaps, then drains each run's words and pushes the in-range lanes onto
`row_data_pipe` one per cycle — the rate `window_emitter`'s Phase 1
consumes them at, so the pipe and everything downstream are untouched.
A run's first / last word may carry lanes outside the run (row-segment
alignment); they are dropped.  The base address must be 16-byte aligned
(the scheduler aligns every buffer).

`y` stays a 16-bit element port: the writer's `(p, c1)` emit order is
channel-strided, and §6.2 already showed that reordering it costs more
than the packed beats save.

**Result.**  RTL (test stand now with a 128-bit `S_AXI_HPC0_FPD` and
crossbar, like the board): **−21.0 %** over the 31-test suite
(1,472,005 → 1,162,425 ns); the Phase-1-bound multi-channel-tile and
global-pool cases gain most.  31/31 RTL, 33/33 C-sim.  Synthesis: II=1 in
`row_loader`, BRAM 38 → 44 (the adapter FIFO), LUT/FF within 1 %.

**On board** (bitstream WNS +1.43 ns): 126/126 scheduler models PASS.
MNIST LeNet pools 545 / 409 → **388 / 275 µs** (−29 / −33 %), the model
8.28 → 7.99 ms; convnet pools 145 / 88 → 106 / 60 µs, the model
1.06 → **0.91 ms** (with the MatMul widening).  MobileNet v2
435 → **404 ms**, ResNet-18 386 → **374 ms**, MobileNet v1 unchanged
(489 ms; it has no pooling on the critical path).

---

### 2.14. 8 lanes per cycle end to end — 128-bit `y`, column-banked LUTRAM line buffer, flattened stages

**Problem.**  After §2.13 the x *port* was 128-bit but the datapath was
still one 16-bit element per cycle at both ends: `row_loader` pushed one
lane per cycle, `window_emitter` wrote one element per cycle into
`line_buf`, the writer stored one element per cycle through a 16-bit
port, and every stage re-entered a short pipelined loop per ow-group or
per run (a few cycles of ramp each).  On the board ResNet-18's MaxPool
3×3 s2 on 112²×64 took 15.8 ms (0.13 GB/s).

**Lane mapping.**  The tensors are NCHW, so the 8 lanes of a 128-bit
word are 8 consecutive **columns** of one channel row — not 8 channels.
The reduce engine already processes 8 channels × `kOwParallel = 2`
columns per tap-cycle (§2.9); what was one element per cycle was the
traffic into and out of it.  The rework keeps the §2.9 reduce and makes
everything around it 8 lanes wide:

* **`row_loader`** — one flattened `II=1` loop per (ni, ct, owt) chunk
  over every 128-bit word of every (row, channel) run, requests running
  `kReadAhead = 12` runs ahead of the drain (prologue + one request per
  drained run, ≤ 12 × 9 words < the adapter's 256-word buffer), so the DDR
  latency is paid once per chunk.  The loader walks rows 0 .. `rows_per_chunk`
  once, in order, no per-oh bookkeeping; the emitter derives the same run
  sequence (`RunCursor`: adds only, the lane shift tracked in 3-bit
  arithmetic).
* **`window_emitter`** — `line_buf` re-banked as `[channel][column bank]
  [row slot · 8 + column word]`: 8 × 8 = 64 LUTRAMs of 128 × 16 b (1W1R).
  A row word is written whole: lanes rotated by the run's alignment so
  bank *b* takes the lane whose column ≡ *b* (mod 8), per-bank enables drop
  the out-of-run lanes of the first / last word, an unrolled channel
  compare gives every RAM one conditional store.  A window tap reads one
  column of all 8 channels per position — the column picks bank and entry,
  each bank is read once, the value is muxed out of the bank vector.
  `kOwParallel` columns spaced by `stride_w` hit distinct banks unless
  `stride_w` is a multiple of 8, in which case `PoolGeometry::gw` makes the
  groups one column wide (lane 1 padded).  **One flattened `II=1` loop per
  output row** loads the words of the rows output row `oh+1` needs (their
  slots are disjoint from the window in use whenever
  `(pool_h−1)·dil_h + 1 + stride_h ≤ kMaxLineBufRows`; otherwise the same
  loop loads the current row first and emits `kSeqGap = 8` iterations
  later) while emitting the `n_groups × pool_h × pool_w` taps of row `oh`;
  the loop exits when both sides are done.  The denominators are separable
  (§2.12): `kw(ow)` is tallied once per chunk into a 64-entry LUTRAM by a
  serial (position, tap) pre-pass and `kh(oh)` once per row, so the hot
  loop multiplies two 3-bit counts instead of running 28 bound compares.
* **`process_pool_kernel_tile`** — one flattened `II=1` loop per output
  row of `n_groups × slot_len` iterations, `slot_len = max(pool_h·pool_w,
  kTileC)`: iterations `i < pool_h·pool_w` reduce one MultiWindow into
  `acc[2][8]` (the §2.9 lanes), iterations `i < 8` finalise channel *i* of
  the **previous** group's snapshot (`acc_done`, `inv_done`, copied at the
  end of every slot) — AVG multiply / `poly_sqrt` / saturate — two lanes
  (one channel's adjacent positions) per cycle onto a 32-bit `FinBundle`
  stream.  The finalise is therefore 2 lanes wide (not the 16 of §6.2.3),
  never stalls the reduce, and costs no loop re-entry; one extra slot
  after the very last row flushes the last group.
* **`write_output_tile`** — `y` becomes `hls::burst_maxi<PoolWord>`.  Two
  ping-pong row buffers of 8 column banks × 64 LUTRAM entries transpose the
  (group, channel) bundles into channel runs: a bundle's two columns of
  channel `c1` land in banks `col mod 8` at entry `c1·8 + col/8`; a DDR word
  of a channel's run is the 8 banks read at one entry (minus one for the
  lanes that wrap), rotated by the run's alignment.  One flattened `II=1`
  loop per row of `max(n_groups·8, c_valid·nw_max)` iterations fills buffer
  `r&1` while draining row `r−1`: per channel one `write_request` of
  `pool_words_for(start, len) ≤ 9` words, the first / last word written
  with `write(word, byte_enable)` covering only the run's own lanes (masked
  lanes zeroed in WDATA) — the conv §2.38 recipe, so a neighbouring
  channel's lanes and the lanes past the tensor end are never modified and
  **the y buffer needs no tail padding** (the scheduler is unchanged; x
  keeps the §2.13 contract).  `kWriteInFlight = 4 <
  num_write_outstanding = 8`.

Port pragmas: x `max_read_burst_length=16 num_read_outstanding=16`
(unchanged), y `max_write_burst_length=16 num_write_outstanding=8`.  The
wide FIFOs (`row_word_pipe` 128 × 128 b, `window_pipe` 64 × 256 b,
`acc_stream` 64 × 32 b) are `bind_storage fifo impl=lutram` — a 128-bit
FIFO in block RAM costs 4 BRAM18 at any depth.

**Alignment contract** (`PoolingKernel.h`): x and y base addresses
16-byte aligned (the scheduler's 64-byte buffer alignment covers it); the
last word of an x run may extend up to 7 elements past the tensor (bytes
mappable — unchanged since §2.13); y is byte-strobed, no padding needed.
Block-design instance widths must equal the IP defaults: **`PoolingKernel_0`
`C_M_AXI_GMEM1_DATA_WIDTH` = 128** now, next to the existing GMEM0 = 128.

**Traps hit.**

| Form | Result |
|---|---|
| `constexpr unsigned clog2(v) { return … clog2(v/2); }` for the narrow index widths | `HLS 214-139 Recursive function calls are not supported` even for a compile-time-only helper — write it as a `while` loop |
| "last write wins" lane select (`for l: if (l == idx) r = lanes[l]`) | a chain of 7 chained 16-bit 2:1 selects per 8:1 pick — the first synthesis had 3.9 k LUT of `select` in the emitter and 1.7 k in the writer; indexing the partitioned register array (`lanes[idx]`) is estimated as one mux |
| 32-bit `unsigned` for structurally bounded counters (word index ≤ 9, column ≤ 64, tap < 7, channel < 8) | 117 32-bit comparators / 72 adders in the emitter loop; typed as `ap_uint<kWordBits/kColBits/kTapBits/kChBits>` the loop dropped 11.7 k → 6.3 k LUT |
| `pool_words_for(run_off, len)` (two 32-bit divides-by-8 and a subtract) on the loop-carried chain that decides the next run boundary | `HLS 200-887 Cannot meet target clock period … store 'nw' (4.905 ns)`; tracking the run's lane shift in 3-bit arithmetic inside `RunCursor` and computing `(shift + len + 7) >> 3` cleared it (slack 0.00) |
| Per-position `kw_lut[gpos + p]` reads with a runtime index into a `cyclic factor=2` LUTRAM | HLS reads every bank for every position → two loads per 1-read-port bank → `HLS 200-448 … II is 2`; read each bank once at an explicitly computed address and mux afterwards (the same rule as the line-buffer read path) |
| Test geometries with `pool_h/w > kMaxPoolH/W = 7` | the unrolled `num_valid_kh/kw` counts silently cap at 7 taps (the scheduler rejects such models; the two first drafts of the new C-sim cases failed on this, not the kernel) |
| Word 0's wrapped-lane address `k − 1` in the C-sim | out-of-bounds read of the row buffer (segfault) — the lanes are masked in hardware but the address must be clamped for the C model |

**Synthesis** (kv260, 150 MHz target): II=1 on every pipelined loop,
`Pipelined = yes` for all seven, slack 0.00, no `Inferring partial
write`, no `SCHED 204-65`, both ports `128 -> 128`.  Resources
§2.13 → §2.14 (HLS estimate): BRAM18 **44 → 11** (the 16 line_buf BRAM18
and the FIFOs are LUTRAM now; 8 for the x adapter, 3 for the AVG
reciprocal ROM), DSP 120 → 90, FF 16,498 → 19,581, LUT 30,438 →
**35,306** (+4.9 k: 2.0 k of line-buffer LUTRAM, 0.5 k of row buffers,
the lane rotates / muxes and the second finalise lane).

**Test stand.**  `pooling_test`: IP upgraded, `PoolingKernel_0`
`C_M_AXI_GMEM1_DATA_WIDTH` 32 → 128 (the interconnect drops its
up-sizer); `pooling_tb.sv` gains conv_tb's y tail-pad sentinel check and
the DDRC shadow restore of unstrobed bytes (§2.38's VIP race).  Fixtures:
the 31 original cases bit-identical, 12 new word-tail / alignment cases
(43 total): widths 7 / 9 / 11 / 13 / 65, the ResNet stem shape
(3×3 s2 on 28×112, W-tiled), stride 8 (one-column groups), the
no-prefetch mode, 1×1 pooling on a 64-column tile, `C = 9`, LpPool
batch 2 on W = 7, GlobalAvgPool 7×7.  C-sim 45/45 (y handed over as a
sentinel-filled word buffer, pad lanes checked).

**Result (RTL, 43/43 PASS, bit-exact, tail lanes intact on every case).**
The 31 cases common with §2.13: **1,162,425 → 509,665 ns (−56.2 %)**,
every case faster:

| # | Case | §2.13 ns | §2.14 ns | Δ |
|--:|---|---:|---:|---:|
| 0 | MaxPool_2x2_stride2 | 16,775 | 10,685 | -36.3 % |
| 1 | MaxPool_3x3_stride1_pad1 | 28,980 | 13,790 | -52.4 % |
| 2 | MaxPool_2x2_stride2_rect_6x10 | 22,960 | 10,960 | -52.3 % |
| 3 | MaxPool_batch_3_2x2_stride2 | 30,400 | 13,270 | -56.3 % |
| 4 | MaxPool_channels_kTileC__C_16_ | 35,230 | 16,590 | -52.9 % |
| 5 | MaxPool_dilation_2_pool2x2 | 21,100 | 12,030 | -43.0 % |
| 6 | GlobalMaxPool_4x4 | 14,020 | 9,890 | -29.5 % |
| 7 | GlobalMaxPool_batch_2_C_12_6x6 | 45,170 | 17,510 | -61.2 % |
| 8 | AvgPool_2x2_stride2_no_pad | 16,240 | 9,950 | -38.7 % |
| 9 | AvgPool_3x3_stride1_pad1_no_include | 28,970 | 13,790 | -52.4 % |
| 10 | AvgPool_3x3_stride1_pad1_include_pad | 28,630 | 13,470 | -53.0 % |
| 11 | AvgPool_channels_kTileC__C_12_ | 30,070 | 15,330 | -49.0 % |
| 12 | GlobalAvgPool_4x4_C_8 | 13,240 | 9,110 | -31.2 % |
| 13 | GlobalAvgPool_batch_2_C_16_6x6 | 54,630 | 21,850 | -60.0 % |
| 14 | AvgPool_rect_6x10_2x2_stride2 | 16,310 | 9,460 | -42.0 % |
| 15 | LpPool_p_1_2x2_stride2 | 17,000 | 10,710 | -37.0 % |
| 16 | LpPool_p_2_2x2_stride2 | 16,420 | 10,130 | -38.3 % |
| 17 | LpPool_p_1_3x3_pad1 | 28,940 | 13,740 | -52.5 % |
| 18 | LpPool_p_2_3x3_pad1 | 28,670 | 13,490 | -52.9 % |
| 19 | GlobalLpPool_p_1_4x4_C_8 | 13,400 | 9,300 | -30.6 % |
| 20 | GlobalLpPool_p_2_4x4_C_8 | 14,460 | 10,260 | -29.0 % |
| 21 | GlobalLpPool_p_2_batch_2_C_16_6x6 | 54,530 | 21,690 | -60.2 % |
| 22 | MaxPool_1x1_pool_full_5x5 | 13,510 | 9,070 | -32.9 % |
| 23 | AvgPool_corner_padding_3x3_pad1_on_2x2 | 8,850 | 7,940 | -10.3 % |
| 24 | MaxPool_C_32_2x2_stride2 | 60,910 | 24,540 | -59.7 % |
| 25 | MaxPool_wide_W_128_3x3_stride1_pad1 | 51,290 | 24,010 | -53.2 % |
| 26 | AvgPool_wide_W_96_3x3_stride1_pad1 | 71,240 | 28,860 | -59.5 % |
| 27 | MaxPool_wide_W_128_2x2_stride2 | 55,860 | 14,560 | -73.9 % |
| 28 | MaxPool_wide_W_128_3x3_stride1_pad1_batch_2 | 92,220 | 41,030 | -55.5 % |
| 29 | AvgPool_wide_W_96_3x3_stride1_pad1_batch_2 | 131,460 | 50,180 | -61.8 % |
| 30 | MaxPool_wide_W_128_2x2_stride2_batch_2 | 100,940 | 22,470 | -77.7 % |
| 31 | MaxPool_2x2_stride2_W_7___1_word_ | — | 8,560 | new |
| 32 | MaxPool_3x3_stride1_pad1_W_13_out_w_odd | — | 11,860 | new |
| 33 | MaxPool_3x3_stride2_pad1_W_11 | — | 9,680 | new |
| 34 | AvgPool_3x3_stride2_pad1_W_9_C_1_no_include | — | 9,430 | new |
| 35 | MaxPool_3x3_stride2_pad1_28x112_C_2__ResNet_stem_ | — | 48,580 | new |
| 36 | AvgPool_7x7_stride8_16x16__one-column_groups_ | — | 12,460 | new |
| 37 | AvgPool_7x1_dil_h2_stride4_21x8__no_prefetch_ | — | 10,230 | new |
| 38 | MaxPool_1x1_stride1_4x64__64-wide_W-tile_ | — | 18,360 | new |
| 39 | MaxPool_1x1_stride1_3x65__W-tile___1_column_ | — | 16,420 | new |
| 40 | MaxPool_C_9_2x2_stride2_5x6__c_valid_1_tail_tile_ | — | 11,770 | new |
| 41 | LpPool_p_2_batch_2_C_3_3x3_stride2_pad1_W_7 | — | 11,080 | new |
| 42 | GlobalAvgPool_7x7_C_16 | — | 15,750 | new |

The 2×2 s2 wide-W cases gain most (−74 / −78 %: they were loader- and
writer-bound at one element per cycle), the 3×3 groups −52 … −62 %, the
global pools −29 … −61 %, the 8-element corner case −10 % (testbench
fill dominates).  `MaxPool 3x3 s2 pad1 28x112 C=2`, the ResNet stem
shape, runs in 48.6 µs at the 100 MHz sim clock; the reduce is the
bound (9 cycles per group of 16 outputs, ≈ 2.25 cycles per 8 outputs),
so the full 112²×64 layer scales to ≈ 1.3 ms (≈ 125 k cycles by the
§4.9 model) against 15.8 ms on the board today — to be measured by the
coordinator with the integrated bitstream.

**Not done (next levers).**  `kOwParallel = 4` (the §2.10 code path is
gone: the column banks now serve any `gw ≤ 8` with distinct strides, and
the finalise / writer scale with it), which would halve the 3×3 reduce
bound; a 4-lane finalise for 2×2 pools (the consumer's `kTileC = 8`
cycles per group is their bound).

---

#### 2.14.1. On board (2026-09-26)

Bitstream with `PoolingKernel_0` `C_M_AXI_GMEM1_DATA_WIDTH = 128`
(IP default), WNS +1.46 ns, design BRAM 119 → 111.5 tiles; 144/144
models.  Perf (before → after): MaxPool-2x2-56x56 3.82 → **0.335 ms**
(1.50 GB/s), MaxPool-3x3-56x56 3.84 → 0.339, MaxPool-2x2-28x28 0.77 →
0.117, MaxPool-3x3-14x14 0.28 → 0.050, AvgPool-2x2-56x56 3.82 → 0.335,
AvgPool-3x3-28x28 0.77 → 0.116, GlobalAvgPool-7x7-64 0.115 → 0.025,
GlobalAvgPool-7x7-1024 1.74 → 0.283, AvgPool-2x2-7x7-1024 1.74 → 0.283,
AvgPool-2x2-3x3-32-112 7.57 → 0.679 ms.  ResNet-18's MaxPool 3×3 s2 on
112²×64: 15.8 → **1.35 ms** (cycle model said 1.3).

Incident: the perf case `GlobalMaxPool-14x14` (pool 14×14 on a 14×14
input, i.e. `pool_h > kMaxPoolH = 7`, outside the platform contract that
the scheduler enforces) ran on the old kernel but hangs the new one and
wedges the HPC port (all later kernel calls time out; reboot before
reloading the PL).  The case was replaced by `GlobalMaxPool-7x7-256`.
Follow-up: clamp `pool_h/pool_w` to `kMaxPoolH/W` in the kernel so an
out-of-contract call returns instead of hanging.

## 3. Current architecture (post-2.14)

```mermaid
flowchart LR
    DDR_IN[("x<br/>gmem0 · 128 b")]
    DDR_OUT[("y<br/>gmem1 · 128 b")]
    RL["row_loader<br/><i>burst_maxi words, 12 runs in flight</i>"]
    WE["window_emitter<br/><i>owns line_buf: 8 ch × 8 column banks of LUTRAM</i>"]
    PP["process_pool_kernel_tile<br/><i>acc / acc_done [kOwParallel][kTileC]</i><br/>reduce ‖ finalise(prev group)"]
    WO["write_output_tile<br/><i>ping-pong row buffers → byte-strobed bursts</i>"]

    DDR_IN -->|m_axi read| RL
    RL -->|row_word_pipe · 128 b| WE
    WE -->|window_pipe<br/>MultiWindow<br/>= kOwParallel × kTileC pixels| PP
    WE -->|denom_pipe<br/>MultiDenom × kOwParallel| PP
    PP -->|acc_stream<br/>FinBundle = kOwParallel outputs<br/>of one channel| WO
    WO -->|m_axi write| DDR_OUT

    classDef ddr fill:#fff7e6,stroke:#d48806,color:#874d00
    classDef stage fill:#e6f7ff,stroke:#1890ff,color:#003a8c
    class DDR_IN,DDR_OUT ddr
    class RL,WE,PP,WO stage
```

**Four DATAFLOW stages**, all running concurrently, each with ONE
flattened `II=1` loop per (ni, ct, owt) chunk or per output row (§2.14):

1. **`row_loader`** — one 128-bit word per cycle over the chunk's run
   sequence `(ih, c_l)`, requests 12 runs ahead.
2. **`window_emitter`** — owns `line_buf[kTileC][8 banks][16 slots × 8
   words]` (64 LUTRAMs) and the per-chunk `kw_lut`.  Per output row: the
   next row's words in (8 lanes per cycle) while this row's `n_groups ×
   pool_h × pool_w` taps go out (kOwParallel columns × kTileC channels per
   cycle) plus one MultiDenom (`kh(oh) × kw(ow)`, §2.12) per group.
3. **`process_pool_kernel_tile`** — per row `n_groups × max(pool_h·pool_w,
   kTileC)` iterations: reduce one MultiWindow per cycle into
   `acc[kOwParallel][kTileC]`, finalise the previous group's snapshot two
   lanes per cycle (AVG: `inv_denom_lookup` ROM multiply, §2.8; LP-2:
   `poly_sqrt`, §2.7), saturate, push a FinBundle.
4. **`write_output_tile`** — per row, fill one row buffer from the
   FinBundles while draining the other as 128-bit words: one burst per
   (row, channel) run, byte strobes on the run's first / last words.

**Loop nest** (all stages in lockstep): `(ni, ct, owt, oh, ow_group)`.
Each ow_group covers `gw` adjacent ow positions (`gw = kOwParallel`, or 1
when `stride_w` is a multiple of 8); the W-tile dimension `owt` collapses
to a single iteration when `in_w ≤ kMaxLineBufCols`.

**Per-invocation geometry.** `compute_pool_geometry()` runs once at the
top of `PoolingKernel` and fills `PoolGeometry { c_tiles, ow_tile,
ow_tiles_w, gw, rows_per_chunk, red_len, slot_len, prefetch }` passed to
all four stages (§2.11, §2.14); per-chunk values come from `chunk_geom()`,
identical in every stage.

**Cycle counts** per output row of a chunk (`c_valid` channels,
`n_groups` groups):

| Stage | cycles |
|---|---|
| row_loader | `new_rows × c_valid × (run_len/8 + 1)` words |
| window_emitter | `max(loader words, n_groups × pool_h × pool_w)` |
| process_pool_kernel_tile | `n_groups × max(pool_h × pool_w, kTileC)` |
| write_output_tile | `max(n_groups × kTileC, c_valid × (ow_span/8 + 1))` |

3×3 pools are reduce-bound at 9 cycles per group of 16 outputs (the
emitter and consumer are matched); 2×2 s2 pools are bound by the
consumer's 8 cycles per group; global pools by the loader's one word per
cycle.

## 4. Knobs

### 4.1. Single source of truth: `platforms/<name>.json`

All compile-time bounds live in the platform JSON under
`kernels.pool` — same file the C++ build and the Python validator both
read.  Default platform is `kv260`; pick another with
`-DAXI_PLATFORM=<name>` (CMake) or `AXI_PLATFORM=<name>` (Python env).

```jsonc
// platforms/kv260.json
{
  "description": "Xilinx KV260 Starter Kit",
  "part":  "xck26-sfvc784-2LV-c",
  "board": "xilinx.com:kv260_som:part0:1.4",
  "clock": 150,
  "kernels": {
    "pool": {
      "tile_c":            8,
      "max_kh":            7,
      "max_kw":            7,
      "max_line_buf_rows": 16,
      "max_line_buf_cols": 64,
      "ow_parallel":       2
    }
  }
}
```

| JSON field | C++ name (Config.h) | Python name | Default | Hard constraint | Notes |
|---|---|---|---:|---|---|
| `tile_c` | `kTileC` | (not validated) | 8 | power of 2 | Channel tile width; II=1 lane rotation depth.  Any model channels count works (channel-tiled). |
| `max_kh` | `kMaxPoolH` | `POOL_MAX_KH` | 7 | `pool_h ≤ this` | Compile-time pool window height limit. |
| `max_kw` | `kMaxPoolW` | `POOL_MAX_KW` | 7 | `pool_w ≤ this` | Compile-time pool window width limit. |
| `max_line_buf_rows` | `kMaxLineBufRows` | `POOL_MAX_LINE_BUF_ROWS` | 16 | power of 2; `(pool_h-1)*dil_h + 1 ≤ this` | Line-buffer row capacity. |
| `max_line_buf_cols` | `kMaxLineBufCols` | `POOL_MAX_LINE_BUF_COLS` | 64 | `(pool_w-1)*dil_w + 1 ≤ this` | Line-buffer column capacity; W-tiling kicks in for `in_w > this`. |
| `ow_parallel` | `kOwParallel` | (not validated) | 2 | power of 2; line_buf must service kOwParallel reads/cycle for the active stride_w (see §2.10 banking table) | Output-position unroll factor (§2.9, §2.10).  `line_buf` is partitioned `cyclic factor=kOwParallel dim=3` plus `BIND_STORAGE ram_t2p` so each per-channel sub-bank is dual-port; together this delivers kOwParallel reads/cycle for any `stride_w` with `gcd(stride_w, kOwParallel) ≤ 2`.  At the shipped `kOwParallel = 2` this holds for every `stride_w`; §2.10 evaluated 4 (covers stride_w 1–3) but reverted it.  Any out_w works (residual-lane padding). |

`tile_c` and `ow_parallel` are read by the C++ build but **not** validated
by the Python scheduler — both have unconditional run-time fallbacks
(channel tiling for any C; residual-lane padding for any out_w), so
models cannot violate them.  The other four fields gate model
acceptance: `PoolNode.from_onnx_node` raises `SchedulerError` naming the
violated bound.

### 4.2. How CMake reads the JSON

`kernels/pool/CMakeLists.txt::pool_load_constants(platform_json prefix)`
calls `string(JSON … GET … kernels pool <field>)` for each required
key, sets `${prefix}_<UPPER_FIELD>` in the parent scope, and errors
out (`FATAL_ERROR`) on any missing field.  The default-platform
constants drive `Config.h` for the C-sim build; the per-platform
synthesis loop calls the function again per platform JSON so each
synthesised IP gets its own bounds.

`set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS …)`
lists every platform JSON, so editing one auto-triggers a reconfigure
on the next `make` — no manual `cmake` rerun needed.

### 4.3. How the Python scheduler reads the JSON

`inference-scheduler/src/_pool_hw_config.py::resolve(platform_name)`:

- `platform_name=None` (the default) → uses `AXI_PLATFORM` env var,
  falling back to `kv260`.
- Reads `platforms/<name>.json`; raises `PoolHwConfigError` on missing
  file, missing `kernels.pool` object, missing required field, or wrong
  field type.
- No silent defaults — every error names the JSON path and the
  problematic key.

`nodes.py` imports the resolved values; `PoolNode.from_onnx_node`
checks each constraint and raises `SchedulerError` with a message
that quotes both the model's value and the platform's bound, plus the
specific JSON field to bump.

### 4.4. Test predictor and cache-extreme verification

The `expected_dup_reads_for(tc)` helper in `TestPoolingSim.cpp` mirrors
the kernel's load schedule using the same constants, so per-test
`dup_reads=X/Y` always reports cache-aware expectations.  Verified at
three cache extremes by setting `max_line_buf_cols` in the JSON and
rebuilding:

| `max_line_buf_cols` | Wide-W tests | Narrow tests |
|---:|---|---|
| 8 | dup_reads = 168/240 (W-tiling, multi-tile narrow) | 64/64 |
| 64 (default) | dup_reads = 16/32 | 0/0 |
| 256 | dup_reads = 0/0 (single tile) | 0/0 |

---

## 5. Per-test progression highlights

Selected representative tests, all 25-test baseline → final 31-test number
on the kv260 RTL sim. Note: 25-test baseline shown for cases that existed
from the start; wide-W tests added later.

| Test | Baseline (ns) | Final (ns) | Speedup |
|---|---:|---:|---:|
| MaxPool 3x3 stride1 pad1 | 236,600 | 32,500 | **7.28×** |
| AvgPool 3x3 stride1 pad1 (no incl pad) | 236,610 | 32,500 | **7.28×** |
| LpPool p=2 3x3 pad1 | 236,350 | 32,280 | **7.32×** |
| MaxPool C=32 2x2 stride2 | 165,150 | 161,560 | 1.02× |
| MaxPool dilation=2 pool2x2 | 72,270 | 28,970 | **2.49×** |
| GlobalMaxPool batch=2 C=12 6x6 | 76,680 | 53,500 | 1.43× |
| MaxPool wide W=128 3x3 stride1 pad1 | (n/a) | 53,280 | — |
| AvgPool wide W=96 3x3 stride1 pad1 batch=2 | (n/a) | 137,830 | — |
| MaxPool wide W=128 3x3 stride1 pad1 batch=2 | (n/a) | 95,820 | — |

**The 3x3 overlap cases benefited the most** through §2.9 — the line
buffer eliminated duplicate DDR reads (§2.2), consumer fusion +
vectorization halved then quartered the inner reduce (§2.4 + §2.5),
`poly_sqrt` recovered ~19% by removing the implicit FP-sqrt budget tax
(§2.7), the fixed-point AVG reciprocal recovered another ~26% by
retiring FP div+mul from the consumer's pipeline budget (§2.8), and
the kOwParallel=2 reduce trimmed ~25% more by halving the consumer's
per-position cost (§2.9) — leaving all five Max/Avg/Lp variants of the
3x3 pad1 group at ~32k ns, the shipped `kOwParallel = 2` end state.
(§2.10 evaluated `kOwParallel = 4` for these but it would not help: the
writer drain — `kOwParallel × c_valid` = 16 cycles — would cross over the
9-cycle reduce; that value was reverted, see §2.10.)

**Wide-W tests** got the biggest absolute savings from the §2.9
`kOwParallel = 2` reduce — they spend the most cycles in the reduce.
§2.9's kOwParallel=2 trimmed −42–45% off the §2.8 number.  (§2.10's
`kOwParallel = 4` evaluation took a further −27–31% — the wide-W `Final`
figures in the table above reflect that evaluation — but the value was
reverted; the shipped wide-W finals are the §2.9 kOwParallel=2 end state.)

**Multi-channel-tile and non-overlap tests** (C_32, 2x2 stride2 family)
benefited mainly from the producer split — Phase 1 was their dominant
cost; consumer-side optimizations (FP-unit removals, kOwParallel) only
nibble at the residual consumer fraction.

---

## 6. Where the floor is now

> Updated for §2.14.  The loader, line-buffer fill and writer now move 8
> elements per cycle and every stage runs one flattened II=1 loop per
> output row, so the per-group and per-run loop re-entries and the
> one-element-per-cycle DDR paths of the analysis below are gone.  The
> remaining floor is the consumer: `max(pool_h × pool_w, kTileC)` cycles
> per group of `kOwParallel × kTileC` outputs — 9 cycles per 16 outputs on
> a 3×3 pool, 8 per 16 on a 2×2.  Raising `kOwParallel` to 4 (banks serve
> any stride not a multiple of 4; FinBundle, row buffers and the consumer
> scale with it) is the next lever; the §6.2 writer-side conclusions are
> superseded by the 128-bit byte-strobed writer.  The text below is kept
> as the record of the §2.13 state.

The §2.13 kernel ran the consumer reduce at `pool_h × pool_w` cycles
per **kOwParallel = 2** outputs across all pool types.  The wall-clock
critical path split by test geometry:

- **Wide-W consumer-bound tests** (out_w ≥ 64) — reduce-bound:
  per-position cost is `pool_h × pool_w / kOwParallel = 9/2 = 4.5` cycles
  for 3×3.  (§2.10 evaluated `kOwParallel = 4`, which would halve this to
  2.25 cycles, but that value was reverted — see §2.10.)
- **Narrow consumer-bound tests** (3x3 pad1 at out_w=8, 5 variants) —
  reduce-bound: the writer's drain rate `kOwParallel × c_valid` (= 8 for
  c_valid=4) sits just under `pool_h × pool_w` (= 9), so the 9-cycle reduce
  sets the pace.  Raising kOwParallel to 4 would push the writer drain to
  16 cycles and make *it* the bottleneck on these narrow tests — so further
  reduce parallelism helps here only if the writer is widened too.
- **Multi-channel-tile tests** (C_16, C_32) — still Phase 1 DDR-bound
  (§6.1).  The `c_tiles` outer sweep dominates and the consumer reduce
  is fully overlapped behind it.
- **Out_w-odd cases** (Global pool, 1×1 output) — pay a small
  residual-padding overhead (≤ ~1% at the shipped `kOwParallel = 2`: one
  padded lane per group when out_w is odd).  Negligible.

To go further requires more invasive changes:

| Option | Mechanism | Estimated win |
|---|---|---|
| Raise kOwParallel (4, then 8) | `ow_parallel: 4` is a one-line JSON change (§2.10 already validated the cyclic-banking code at factor 4); 8 needs a second cyclic factor or replicated sub-banks, plus wider MultiWindow / acc[][] / writer drain | ~2× (4) / ~4× (8) on wide-W reduce-bound tests; narrow tests gain only if the writer drain is widened in step (see §6 narrow-test note) |
| Writer fanout — emit kOwParallel y[] in parallel | Replicate the m_axi write port or stripe writes into a wide AXI burst | Prerequisite for kOwParallel ≥ 4 to help the narrow 3×3 group — at the shipped kOwParallel = 2 the 8-cycle writer drain is already under the 9-cycle reduce, so this yields nothing on its own |
| Wider window vectors (kwi-fanout) | Emit `pool_w` pixels per cycle along kwi axis; reduce trip drops to `pool_h` cycles per ow-group | 2–4× on consumer (orthogonal to kOwParallel); helpful only when the writer isn't already the bottleneck |

These are deferred until profiling shows pool on a critical path of a real
inference workload.

### 6.1. Tried and rejected: Phase 1 DDR burst (post-§2.9)

Reordering `row_loader` and `window_emitter` Phase 1 from `(ih, c_l, iw)`
to `(c_l, ih, iw)` — so a per-channel sweep is one address-monotonic
range that HLS could fold into a single AXI4 burst — was implemented and
benchmarked.  Result on the kv260 sim: **+0.39% total** (1,677,745 →
1,684,215 ns), within HLS synthesis noise; no individual test moved more
than ±1%.

Two diagnoses combine to explain why:

1. **Phase 1 wasn't on the critical path** for any test post-§2.9.  On
   3x3 stride1 pad1 the consumer's reduce already pipelined at II=1 and
   was the wall-clock bottleneck; halving the producer's per-channel
   issue cost doesn't move it.  On wide-W tests the consumer's
   `pool_h × pool_w / kOwParallel` cycle count likewise sets the floor.
2. **Default HLS burst inference was already adequate** at the
   geometries the test suite exercises — the original `(ih, c_l, iw)`
   nest's per-channel `iw_load_hi - iw_load_lo + 1` bursts (≤128 elems)
   fit inside the default `max_read_burst_length=16` × DDR-controller
   queue pipelining, with `num_read_outstanding` masking AR-channel
   serialisation.

Adding `max_read_burst_length=256`, `num_read_outstanding=4`, and
`max_widen_bitwidth=128` to the `m_axi` pragmas alongside the reorder
made things **worse by 3.4%** — the wider bus widening forced HLS to
generate alignment shifters that add per-burst overhead, hurting the
narrow tests where bursts are short (8–16 elements).  The pragmas and
the loop reorder were both reverted.

Conclusion: the producer/consumer balance after §2.5 + §2.9 is tight
enough that DDR-side coalescing yields no measurable wall-clock benefit
on the existing test suite.  A workload with much larger `in_h × in_w`
(e.g. early-conv-pool stages of a 512×512-input model) would be needed
to put Phase 1 back on the critical path before this is worth retrying.

### 6.2. Tried and rejected: writer-side bus-utilisation fixes (post-§2.10)

A Vivado timing-diagram capture of `write_output_tile`'s `m_axi` activity
revealed that every 32-bit beat carries only one 16-bit `Data_t` —
`WSTRB` is always `0x3` or `0xC`, never `0xF`.  Half the write-bus
bandwidth is unused per beat.  Three attempts were made to flip those
beats to full-`WSTRB` packed writes; **all three regressed**.  Diagnosis
across the trio: the writer was already overlapped behind the producer's
reduce/finalize critical path in the DATAFLOW region, so saving AXI bus
cycles didn't unblock anything — and each attempt's structural change
added cycles that the bus savings couldn't repay.

#### 6.2.1. Writer loop flip `(p, c1) → (c1, p)`

Flipping the finalize/drain loop order so the inner `p` walks adjacent
ow positions (`Δ = sizeof(Data_t) = 2 B`) should have let HLS pack two
16-bit writes into one 32-bit beat with `WSTRB=0xF`.  Result:
**+1.0% total** (1,677,745 → 1,695,345 ns); no individual test moved
more than ±2%.

Diagnosis: HLS doesn't coalesce writes across the inner `p` loop
because of the residual-padding mask `if (ow_ok) y[…] = v` — every beat
could be either a full write or a no-op, so HLS conservatively keeps
them as separate single-element transactions with partial WSTRB.  The
favorable address pattern is wasted on the conditional store.

#### 6.2.2. Row-buffered Phase A → Phase B burst writes

Two-phase rewrite: Phase A drains `acc_stream` into a per-channel row
buffer `out_row[kTileC][kMaxLineBufCols]` (the `ow_ok` mask gates the
buffer write, not the DDR write); Phase B issues a contiguous,
unconditional, monotonic write loop per channel that HLS should widen
into packed beats.  Result: **+3.25% total** (1,677,745 → 1,732,295 ns).
All 31 tests regressed; global pool variants worst-hit at **+10–17%**
(Phase A+B overhead with only 1 element per row is pure waste).

Diagnosis: Phase A and Phase B serialise inside `write_output_tile` —
no inner-function DATAFLOW pulls them apart, so the writer's per-`(oh,
owt)` cycle count grew from `kOwParallel × c_valid × n_groups` (= one
phase, pipelined) to `kOwParallel × c_valid × n_groups + c_valid ×
ow_span` (= two sequential phases).  Even when Phase B's packed beats
halve the AXI traffic, the added Phase B cycles outweigh the savings,
and for `out_w=1` the buffering is pure overhead.

#### 6.2.3. Wider `acc_stream` (`AccBundle`) + parallel finalize

Replaced `hls::stream<AccData_t>` with `hls::stream<AccBundle>` (one
wide transaction per ow-group carrying all kOwParallel × kTileC
AccData_t lanes).  Producer fully unrolls the finalize across both `p`
and `c1` and emits one bundle per group; in theory this collapses the
producer's per-group cost from `pool_h × pool_w + kOwParallel × c_valid`
(reduce + scalar finalize) to `pool_h × pool_w + 1`.

Result: **+19.05% total** (1,677,745 → 1,997,395 ns) — by far the
largest regression of the three.  The slowdown clustered exactly on the
consumer-bound tests this change was meant to help:

| Test category | Δ% vs narrow stream |
|---|---:|
| 3x3 stride1 pad1 narrow (5 variants) | **+40% uniform** |
| Wide-W 3x3 (single + batch=2) | **+33 to +35%** |
| Wide-W 2x2 stride2 | **+36 to +39%** |
| Dilation=2 pool2x2 | +13.6% |
| Phase 1-bound (C_16 / C_32 / GlobalPool) | +0.2 to +1.0% (noise) |

Diagnosis: HLS could not actually fit the fully-unrolled finalize into
~1 cycle.  16 parallel poly_sqrt / AVG-multiply instances feeding a
512-bit wide FIFO write force a multi-cycle pipeline for the bundle
assembly + emit, and the reduce → finalize schedule lengthened beyond
the original narrow-stream `II=1` finalize loop.  The +40% uniform
hit on all five 3x3 pad1 variants is the signature of producer-side
pipeline lengthening (not writer or DDR, which would show selective
patterns).

#### 6.2.4. Conclusion

The half-WSTRB observation is real but **persistently un-improvable
through kernel-side restructuring at the HLS abstraction level the
current code uses**.  Every attempt either keeps the conditional
residual-padding store (HLS won't pack), removes the conditional via
buffering (Phase A/B serialise inside the function), or tries to feed
the writer through a wider FIFO (HLS can't realise the parallel
finalize at II=1 the way the analysis predicts).  The bus-utilisation
inefficiency is harmless because the writer is not on the wall-clock
critical path — saving bus cycles doesn't unblock anything in the
DATAFLOW region.

If the half-WSTRB pattern ever becomes a real problem (e.g. on a
heavily-shared AXI fabric where multiple kernels contend for the same
DDR controller), the right fix is **widening the global
`AXI_BUS_WIDTH`** at the top-level CMake (32 → 64/128).  HLS's m_axi
auto-widening handles the packing without any kernel-side restructure
and benefits all four kernels symmetrically.  It requires re-running
synthesis for all platforms and validating that the Vivado AXI
interconnect matches — outside the scope of pool-only optimisation.

---

## 7. Verification matrix

| Configuration | C-sim (TestPoolingSim) | RTL sim (behavior_test_pool) |
|---|---|---|
| Default (kMaxLineBufCols=64), post-§2.14 | 45/45 PASS (y pad lanes checked) | 43/43 PASS (y tail lanes checked) |
| Default (kMaxLineBufCols=64), §2.13 | 33/33 PASS | 31/31 PASS |
| Reduced cache (kMaxLineBufCols=8) | 33/33 PASS, dup_reads tracks predictor | (not run) |
| Increased cache (kMaxLineBufCols=256) | 33/33 PASS, dup_reads = 0 throughout | (not run) |

The cache-aware predictor in `TestPoolingSim.cpp` ensures the dup_reads
column in test output is meaningful at any cache size.

The C-sim count is 45 (vs 43 RTL): 43 geometry cases against the float64
reference at `kTol = 0.02` (≈ 5 Data_t LSBs) plus 2 strict-equality
sub-cases (`run_avg_pool_strict_test`, `count_include_pad ∈ {0, 1}`)
that pin the AVG path's bit-accurate match to `ref_avg_pool_fixed`.  The
strict cases are sensitive to a regression to the float reciprocal —
~3% of cells in their input set diverge between the two paths.

§2.9's restructure is purely a cycle-level shape change — every
per-position computation is identical to §2.8, so the y.hex fixtures
under `hw/test_data/pool_test_data/` were verified bit-identical after
the kernel change (zero regen needed).  The kOwParallel residual padding
path is exercised by every test with `out_w` not divisible by
kOwParallel (Global pool variants and `MaxPool 1x1 pool_full 5x5`,
`AvgPool corner_padding 3x3 pad1 on 2x2`).

---

## 8. Related files

| File | What changed |
|---|---|
| `kernels/pool/kernel/PoolingKernel.cpp` | Full rewrite into 4 dataflow stages; `poly_sqrt` (§2.7); `inv_denom_lookup` constexpr ROM LUT for AVG-Pool reciprocal divide (§2.8); `MultiWindow` / `MultiDenom` structs and kOwParallel-wide reduce + line_buf `BIND_STORAGE ram_t2p` (§2.9); `cyclic factor=kOwParallel dim=3` partition on line_buf to support kOwParallel ≥ 4 reads/cycle (§2.10); `PoolGeometry` struct + `compute_pool_geometry()` — per-invocation tile geometry (`c_tiles` / `ow_tile` / `ow_tiles_w`) computed once and passed to all four stages, collapsing the per-stage `ow_tiles_w` divider (§2.11); closed-form
`valid_count` — separable `num_valid_kh × num_valid_kw` replacing the
98-lane unrolled bounds-count (§2.12); §2.14: 128-bit `burst_maxi` y, word-wide `row_loader` with `kReadAhead` requests, LUTRAM column-banked `line_buf` written 8 lanes per cycle, per-chunk `kw_lut`, flattened per-row loops in every stage, overlapped 2-lane finalise, ping-pong row-buffer writer with byte-strobed bursts, narrow index types |
| `kernels/pool/include/Config.h.in` | Templates `kTileC`, `kMaxPoolH`, `kMaxPoolW`, `kMaxLineBufRows`, `kMaxLineBufCols`, `kOwParallel` from CMake-side variables (sourced from the platform JSON, §4) |
| `kernels/pool/CMakeLists.txt` | `pool_load_constants(platform_json prefix)` reads `kernels.pool.*` from `platforms/<AXI_PLATFORM>.json` via `string(JSON …)`; default-platform values drive C-sim Config.h, per-platform values drive per-platform synthesis Config.h's; `CMAKE_CONFIGURE_DEPENDS` on every platform JSON so edits auto-trigger reconfigure on next `make` |
| `platforms/<name>.json` | Single source of truth for kernel-side bounds — `kernels.pool` object holds all six values (§4.1).  The C++ build and the Python validator both read from here. |
| `CMakeLists.txt` (top-level) | `AXI_PLATFORM` cache var (default `kv260`) selects which platform JSON drives C-sim builds; `AXI_DEFAULT_PLATFORM_JSON` is the resolved path |
| `inference-scheduler/src/_pool_hw_config.py` | `resolve(platform_name=None)` reads `platforms/<AXI_PLATFORM>.json` (env override → `kv260` default); raises `PoolHwConfigError` on missing file / missing section / missing field / wrong type — no silent fallbacks |
| `inference-scheduler/src/nodes.py` | `PoolNode.from_onnx_node` validates `pool_h ≤ kMaxPoolH`, `pool_w ≤ kMaxPoolW`, dilated vertical span ≤ `kMaxLineBufRows`, dilated horizontal span ≤ `kMaxLineBufCols` against values resolved from the platform JSON; raises `SchedulerError` with an actionable message naming the bound and the JSON field to bump |
| `kernels/pool/test/TestPoolingSim.cpp` | Cache-aware `expected_dup_reads_for()`; 6 wide-W tests added; §2.14: y as a sentinel-filled `PoolWord` buffer with a pad-lane check, 12 word-tail / alignment cases; `quantize_trn` + `ref_poly_sqrt` mirror kernel's fixed-point sqrt bit-exactly; `ref_avg_pool_fixed` mirrors `inv_denom_lookup` bit-exactly; `ref_pool_elem`'s AVG branch routed through it; new `run_avg_pool_strict_test` (2 strict-equality subtests for AVG path) |
| `inference-scheduler/src/codegen/_simulate.py` | `_quantize_trn` + `_pool_poly_sqrt` so generated `expected/*.dat` fixtures match the kernel's RTL output for LP-Pool p=2; `_pool2d_ref` AVG branch updated to use the same encoded reciprocal as the kernel so AVG cells match byte-for-byte under `test_inference.c`'s strict equality check |
| `hw/cormorant_test_stand/kernels/pooling_test/` | §2.14: `PoolingKernel_0` `C_M_AXI_GMEM1_DATA_WIDTH` 128 (IP default), `pooling_tb.sv` y tail-pad check + DDRC shadow restore |
| `hw/test_data/pool_test_data/` | 43-test fixtures (§2.14: 12 word-tail cases added, the 31 originals bit-identical); 31-test fixtures regenerated for kv260 RTL sim; AVG cases (`test_{09,10,26,29}_y.hex`) refreshed for §2.8 reciprocal change (4 files, 306 cells changed total, all by exactly 1 LSB) |

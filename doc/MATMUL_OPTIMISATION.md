# MatmulKernel — Optimization Log

This document records performance investigation and optimization attempts on
`kernels/matmul/kernel/MatmulKernel.cpp`. Each section describes one change or
experiment, the rationale, and the measured HW behavior-simulation
(`make behavior_test_matmul`) impact on the kv260 RTL.

For the high-level kernel description see [MATMUL_KERNEL.md](MATMUL_KERNEL.md).

> **Status (2026-05-19).** §1 is the **current performance baseline** — the
> shipped single-sequential-loop-nest kernel. §2 records a DATAFLOW
> restructuring that was **tried and rejected** (measured regression). §3
> notes where a real speedup would have to come from. No optimization has
> landed yet; the kernel is unchanged from its initial implementation.

---

## 1. Current performance baseline

All numbers are `sim_time_ns` reported by the kv260 behavior testbench
(`matmul_op_test`) over the full 20-test fixture list. The testbench runs at a
fixed 100 MHz sim basis, so `sim_time_ns` / per-test `duration_ns` are
directly comparable between runs regardless of the synthesis-target clock.

> **Baseline.** Original `MatmulKernel.cpp` — a single sequential tiled loop
> nest; each load / reduce / write loop is pipelined at II=1 but the loops
> execute one after another. **sim_time_ns = 7,713,375** (Σ per-test
> `duration_ns` = 7,711,175). 20/20 RTL tests pass. Synthesis: II=1 on every
> loop, Estimated Fmax 205.47 MHz. Captured **2026-05-19** —
> `build/kernels/matmul/kv260/matmul_op_test_report.json`.

| # | Test | n×k×m×batch | duration_ns |
|--:|---|---|---:|
| 0  | 1×1×1 degenerate                       | 1×1×1×1     | 5,855 |
| 1  | TileN × TileK × TileM                  | 4×256×16×1  | 197,580 |
| 2  | 2·TileN × TileK × TileM                | 8×256×16×1  | 391,610 |
| 3  | TileN × 2·TileK × TileM                | 4×512×16×1  | 386,620 |
| 4  | TileN × TileK × 2·TileM                | 4×256×32×1  | 379,010 |
| 5  | partial N                              | 6×256×16×1  | 384,060 |
| 6  | partial K                              | 4×261×16×1  | 201,520 |
| 7  | partial M                              | 4×256×19×1  | 259,790 |
| 8  | partial N, K and M                     | 6×261×19×1  | 519,060 |
| 9  | arbitrary small                        | 7×13×5×1    | 21,180 |
| 10 | outer product (K=1)                    | 5×1×17×1    | 11,490 |
| 11 | row vector (N=1)                       | 1×256×16×1  | 186,510 |
| 12 | column vector (M=1)                    | 4×256×1×1   | 45,850 |
| 13 | multi-tile all dims                    | 12×519×33×1 | 2,449,610 |
| 14 | batch=3, no broadcast                  | 5×64×19×3   | 398,620 |
| 15 | batch=4, A broadcasts                  | 5×64×19×4   | 530,220 |
| 16 | batch=4, B broadcasts                  | 5×64×19×4   | 530,240 |
| 17 | batch=6, both strided                  | 5×64×19×6   | 793,380 |
| 18 | saturation positive                    | 4×3×16×1    | 9,380 |
| 19 | saturation negative                    | 4×3×16×1    | 9,590 |
| | **Σ duration_ns** | | **7,711,175** |
| | **total sim_time_ns** | | **7,713,375** |

---

## 2. Tried and rejected: DATAFLOW pipeline restructuring

**Attempt.** Restructure the kernel as a canonical Vitis HLS `DATAFLOW`
design — four stages running concurrently, linked by `hls::stream` FIFOs:

```
a_producer ──a_stream──► compute ──acc_stream──► writer
b_producer ──b_stream──►
```

`a_producer` reads A from DDR and re-streams it, `b_producer` burst-reads B
tiles, `compute` runs the II=1 K-reduction, `writer` saturates and writes C.
The intent was to hide B's DDR-load latency behind the MAC compute.

**Result — REJECTED.** Functionally correct (20/20 RTL tests pass, C-sim
bit-identical, synthesis II=1, Fmax 205.47 MHz), but a **+46 % performance
regression**:

| Metric | Baseline | DATAFLOW | Δ |
|---|---:|---:|---:|
| total sim_time_ns | 7,713,375 | 11,276,865 | **+46 %** |
| Σ duration_ns | 7,711,175 | 11,274,665 | +46 % |

Per-test (DATAFLOW vs baseline `duration_ns`):

| Test | Baseline | DATAFLOW | Δ |
|---|---:|---:|---:|
| 4×256×16            | 197,580   | 228,030   | +15 % |
| 4×256×19 (partial M)| 259,790   | 414,000   | +59 % |
| 6×261×19 (partial)  | 519,060   | 838,400   | +62 % |
| 4×256×1  (M=1)      | 45,850    | 189,340   | **+313 %** |
| 12×519×33 multi-tile| 2,449,610 | 3,804,850 | +55 % |
| batch=6             | 793,380   | 1,229,970 | +55 % |

Every realistically-sized test regressed 15–62 %; `M=1` is pathological
(+313 %). Only trivially-small degenerate cases (`1×1×1`, `K=1`, saturation)
improved slightly.

**Root cause.** Matmul on this design is **DDR-latency bound**, not
compute-bound. From `csynth.rpt`, the `compute` K-reduction loop is II=1
(~1 k cycles per K-tile) — already cheap. The cost is in the DDR reads:
`b_producer` (and the A-buffer load) issue **one burst per matrix row** —
~256 small bursts for a 256-deep K-tile, each paying a full DDR round-trip.

DATAFLOW overlaps `compute` with the DDR traffic — but `compute` is the
*cheap* stage, so the overlap saves almost nothing, while the dataflow
structure *adds* overhead: A is streamed through a FIFO instead of read
straight from BRAM, the K-reduction does conditional stream reads, and every
tile pays inter-stage FIFO synchronisation plus stage fill/drain. The
overhead exceeds the saving → net loss. `M=1` degrades worst because the
bursts shrink to one beat and the per-tile overhead dominates entirely.

**Disposition.** Reverted. The kernel remains the single sequential loop
nest described in [MATMUL_KERNEL.md](MATMUL_KERNEL.md).

---

## 3. 128-bit A / B ports — implemented (§3 was the plan; 2026-09-24)

**Change.**  `a` and `b` are now `hls::burst_maxi<MatmulWord>` ports
(`ap_uint<128>`, 8 `ap_fixed<16,8>` lanes per beat; `MatmulKernel.h`).
The DDR layout is unchanged — plain row-major, so activations as well as
constants work as B and every batch stride / tile offset is an element
offset handled in-kernel: for each matrix row segment the kernel computes
the aligned word range that covers it (`matmul_words_for`), requests it,
and scatters each word's lanes into bank-explicit buffers:

- A rows: one request per row (≤ 256-word pieces); `a_buf` is
  `[kTileN][kMaxK / 8][8]` with dims 1 and 3 partitioned complete so the 8
  lanes of a word land in 8 banks per cycle.  Lane `l` of word `w` is
  element `(w_lo + w)·8 + l − row_off`; lanes outside `[0, k)` are dropped.
- B tile rows: `m_valid ≤ 16` elements are ≤ 3 words; requests for
  `kBReqAhead = 16` rows (`num_read_outstanding`) are issued before their
  words are drained so consecutive rows' DDR latency overlaps.  Each word's
  lanes go to `b_tile[k1][m1]` with the bank `m1 = w·8 + l − shift`
  explicit in the unrolled loop.
- C stays a 16-bit element port (n × m is small; its writes were never on
  the critical path).

The only requirement on callers is a 16-byte-aligned base address for
`a` and `b` (the scheduler aligns every buffer to `INFERENCE_ALIGN_BYTES`);
a row's last word may over-read up to 7 elements past the matrix end.

**Why not widen in Vivado or with `-m_axi_min_bitwidth`.**  See
CONV_OPTIMISATION.md §2.36: both leave the exported IP inconsistent (4-bit
WSTRB, or single-beat partial-strobe writes the PS drops).  Ports that
must be wide are declared wide in the C++.

**Result.**  RTL (test stand now with a 128-bit `S_AXI_HPC0_FPD` and
crossbar, like the board): **−71.9 %** over the 20-test suite
(7,711,175 → 2,164,935 ns), every tiled case −67…−80 %
(`TileN × TileK × TileM` 197,580 → 43,330 ns; `multi-tile all dims`
2,449,610 → 662,680 ns).  Only the K=1 outer product got slower
(+12 %, one word per 1-element row).  Synthesis: II=1 on every loop,
BRAM 28 → 64 (the 32 `a_buf` banks and the two adapter FIFOs), DSP
unchanged, slack 0.00.  20/20 RTL, 20/20 C-sim.

**On board** (bitstream WNS +1.43 ns, block-design instance widths at the
new IP defaults): 126/126 scheduler models PASS; the MNIST convnet's
256×10 Gemm dropped from 116 µs to below 60 µs (it left the profiler's
top five) and the model from 1.06 → **0.91 ms**; no image demo has a MatMul layer.

---

## 3b. Packed tile-major B for constant weights (`b_packed`)

**Change.**  A new AXI-Lite register `b_packed` (offset 0x6C, appended so
the existing offsets are unchanged) selects a tile-major B layout
(MATMUL_KERNEL.md §1, `matmul_packed_index()`): one contiguous
`[k][kTileM]` block per m-tile, `m` zero-padded to a multiple of kTileM.
With it a `(m_tile, k_tile)` block is one run of `k_valid · kTileM`
elements; the kernel issues it as ≤ 8 back-to-back 64-word requests
(`max_read_burst_length` of `b` raised 16 → 64) and streams the words
straight into `b_tile` (word `w` → row `w / 2`, columns
`(w % 2) · 8 …`), instead of `k_valid` requests of ≤ 3 words each.

The scheduler (`OnnxGraph._pack_matmul_weights`, `MatmulNode.pack_b`)
packs a constant B once per tensor and calls every consumer with
`b_packed = 1`, rescaling `b_batch_stride` / `b_outer_stride` to packed
slices — but only when every reader of the tensor is a MatMul using it
as B with the same `(k, m)`; activations and constants shared with other
readers stay row-major and take the §3 per-row path.  `run_matmul()` /
`run_matmul_at()` gained the `b_packed` argument; `TensorInfo.numel`
follows the packed image (`m` 10 → 16 for the MNIST Gemm).  Kernel C-sim
(29 cases, 9 packed), the RTL fixtures (manifest gained a `b_packed`
column) and 4 new scheduler tests cover both layouts.

**Result (RTL, packed vs row-major twin of the same geometry).**
`TileN × TileK × TileM` 43,470 → 25,760 ns (−41 %), all-partial
154,890 → 81,760 (−47 %), `multi-tile all dims` 663,790 → 331,280
(−50 %), N=1 row vector −47 %, batched −38 %; only the tiny 7×13×5 case
is flat (−6 %).  Combined with §3 the tiled cases are now 7–8× faster
than the §1 baseline (e.g. 197,580 → 25,760 ns).  Synthesis: II=1 on
every loop, BRAM 64 (unchanged), slack 0.00; 29/29 RTL, 29/29 C-sim.

**On board** (bitstream WNS +0.78 ns, full re-synthesis — see the
incremental-synthesis note in hw/cormorant_hw_128/CLAUDE.md): 126/126
scheduler models PASS, plus a 7-size constant-B sweep (k 32…512, m
10…64, 1–8 requests per block).  MNIST convnet 0.910 → **0.898 ms**
(its 256×10 Gemm), MobileNet v2 404 → **400 ms** and ResNet-18
374 → **372 ms** (their Gemm classifiers, 1280×1001 and 512×1000, now
packed); LeNet and MobileNet v1 have no MatMul and are unchanged.
Predictions and logits identical.  Demo projects must be regenerated
after this change: a project generated before the register existed
inherits whatever `b_packed` value the previous run left in the kernel.


The DATAFLOW experiment confirmed the bottleneck is **DDR bandwidth**, not
MAC throughput. The synthesis `M_AXI Burst Information` shows the cause:

- **`Widen Fail` (HLS 214-307)** on all three ports — the AXI buses stay
  16 bits wide (one `ap_fixed<16,8>` element per beat) even though widening
  to 32+ bits is allowed. HLS cannot prove the runtime row strides (`k`, `m`)
  keep rows on a wider alignment boundary, so it refuses to pack.
- One burst is issued **per matrix row**; each pays full DDR latency.

The 128-bit A / B ports (§3) attacked exactly that.  What remains is
latency per B tile row (≤ 3 words per request, 16 in flight) — a
scheduler-packed tile-major B layout for constant weights would turn a
(m_tile, k_tile) block into one 512-word burst — and the 16-bit C port.

---

## 4. 16×16 MAC, batched A-row requests, B-resident fast path (Track A1, 2026-09-25)

Track A of `doc/THROUGHPUT_PLAN.md` starts here.  Every step of the track
is measured against the §3b kernel on the kv260 RTL stand, with the fixture
list extended first (this step) from 29 to 39 cases: the K-split geometries
of §5 (`1×261×19`, `2×13×5`, `3×517×33`, each row-major and packed) and the
prefetch geometries of §7 (`6×517×35` batch 3 with `k_tiles = 3`,
`m_tiles = 3`, `n_tiles = 2`, with and without B broadcast, each layout).
The "before" column of every table in §4–§7 is the **unchanged §3b kernel
run on the 39-case list** (its exported IP was kept and re-run on the new
fixtures), so the new cases have a real baseline too.

**Problem.**  Three small inefficiencies of the §3b kernel, none of them
visible on the shipped models but all in the way of the later steps:

- the K-loop multiplied `AccData_t(a) * AccData_t(b)` — a 32×32 multiply
  per column, 2 DSP48 each (51 DSP in the design, 32 in the K-loop);
- the `kTileN` A rows of an n_tile were requested and drained one row at
  a time, so each row paid the full DDR round trip (`num_read_outstanding
  = 4` allowed all four in flight);
- when B is a single `(m_tile, k_tile)` block (`m ≤ 16`, `k ≤ 256` — every
  depthwise-as-matmul and small-FC shape) it was reloaded for every n_tile
  of every batch slice.

**Change.**

- `acc[n1][m1] += a_val * b_tile[..]` with `Data_t × Data_t` operands.
  For `ap_fixed<16,8>` the product type *is* `ap_fixed<32,16>` — exact,
  no rounding — so accumulating it into `AccData_t` is bit-identical to
  widening first (C-sim bit-exact on all 39 cases) and costs one 16×16
  DSP per column: DSP **51 → 38** (K-loop 32 → 16).
- The A-row loader issues the requests of as many rows as fit in the
  `kAReqOutstanding = 4` window (a row is ≤ 257 words, i.e. one or two
  ≤ 256-word requests; `static_assert` that one row always fits) *before*
  draining any of them, then drains them in order.  Issuing more than the
  window holds would stall the adapter before the first drain — a deadlock
  in this sequential loop — hence the explicit window.
- `b_resident = (m_tiles == 1 && k_tiles == 1)`: the block is loaded at
  the first n_tile of a batch slice only, and with `b_batch_stride == 0`
  once per call.  The B loader moved into `load_b_tile()` (inlined).

**Traps hit.**  Calling the inlined `load_b_tile()` from two sites (the
fast path before the n_tile loop and the normal site in the k_tile loop)
made HLS instantiate its row-major scatter twice: LUT **44.3 k → 63.6 k**.
Folding the fast-path condition into the single call site
(`load_b = !b_resident || (n_tile == 0 && (bi == 0 || b_batch_stride))`)
restored 43.4 k.  Lesson for the rest of the track: the scatter loops are
the kernel's LUT budget; never give them a second call site.

**Test-stand artifact found by the new fixtures.**  In the 39-case
*sequence* the new row-major case `batch=3, 6×517×35` fails — on the §3b
kernel and on every kernel of this track alike, with byte-identical wrong
outputs (6/630, all in column 12 of batch 2) — while each of them passes
the same case run alone, and C-sim is exact.  The wrong outputs are
reproduced exactly by replacing one B element, `B[392][12]` of batch 2
(element 49 922, DDR address `0x1002CEC4`), with **the last C element of
the preceding test** (`C[98] = −388` of `3×517×33 packed`, whose C region
ends at that address): the DDR model of the stand commits the kernel's
final partial-strobe C write late enough that the next test's backdoor
`write_mem` of B at the same address is overwritten by it.  A kernel
cannot cause this (it reads what the model returns).  A 20 µs settle
between tests did not help (the commit is triggered by later activity,
not time); `matmul_tb.sv` now alternates the DDR base address between
consecutive tests so a test's inputs never occupy addresses the previous
kernel wrote (test-stand commit after §7).  The coordinator should know
the stand has this hazard whenever one test's C region overlaps the next
test's inputs.  The case is reported as FAIL in the §4–§7 tables (runs
made before the fix); its timing is valid.

**Result (RTL, 39 cases; before = §3b kernel).**

| # | Test | n×k×m×batch | B | before | after | Δ |
|--:|---|---|---|---:|---:|---:|
| 0 | 1x1x1 | 1×1×1×1 | row-major | 6,065 | 6,095 | +0.5 % |
| 1 | TileN x TileK x TileM | 4×256×16×1 | row-major | 43,470 | 41,830 | -3.8 % |
| 2 | 2 TileN x TileK x TileM | 8×256×16×1 | row-major | 83,300 | 56,450 | -32.2 % |
| 3 | TileN x 2 TileK x TileM | 4×512×16×1 | row-major | 78,050 | 76,410 | -2.1 % |
| 4 | TileN x TileK x 2 TileM | 4×256×32×1 | row-major | 79,340 | 77,700 | -2.1 % |
| 5 | TileN 2 x TileK x TileM partial N | 6×256×16×1 | row-major | 80,470 | 54,760 | -31.9 % |
| 6 | TileN x TileK 5 x TileM partial K | 4×261×16×1 | row-major | 44,740 | 43,040 | -3.8 % |
| 7 | TileN x TileK x TileM 3  partial M | 4×256×19×1 | row-major | 79,050 | 77,410 | -2.1 % |
| 8 | TileN 2 x TileK 5 x TileM 3  all partial | 6×261×19×1 | row-major | 154,890 | 152,630 | -1.5 % |
| 9 | 7 x 13 x 5 arbitrary small | 7×13×5×1 | row-major | 15,280 | 11,520 | -24.6 % |
| 10 | N x 1 x M K 1 outer product | 5×1×17×1 | row-major | 13,200 | 11,760 | -10.9 % |
| 11 | 1 x K x M N 1 row vector | 1×256×16×1 | row-major | 38,920 | 38,920 | +0.0 % |
| 12 | N x K x 1 M 1 column vector | 4×256×1×1 | row-major | 40,830 | 39,160 | -4.1 % |
| 13 | 3 TileN x 2 TileK 7 x 2 TileM 1  multi-tile all | 12×519×33×1 | row-major | 663,790 | 658,530 | -0.8 % |
| 14 | batch 3 no broadcast | 5×64×19×3 | row-major | 129,960 | 125,050 | -3.8 % |
| 15 | batch 4 A broadcasts a stride 0 | 5×64×19×4 | row-major | 172,390 | 165,940 | -3.7 % |
| 16 | batch 4 B broadcasts b stride 0 | 5×64×19×4 | row-major | 172,290 | 165,730 | -3.8 % |
| 17 | batch 6 both strided multi-dim flat | 5×64×19×6 | row-major | 256,650 | 246,850 | -3.8 % |
| 18 | TileN x TileK x TileM B packed | 4×256×16×1 | packed | 25,760 | 24,090 | -6.5 % |
| 19 | TileN 2 x TileK 5 x TileM 3  B packed all partial | 6×261×19×1 | packed | 81,760 | 79,510 | -2.8 % |
| 20 | 7 x 13 x 5 B packed arbitrary small | 7×13×5×1 | packed | 14,310 | 11,060 | -22.7 % |
| 21 | 1 x K x M B packed N 1 row vector | 1×256×16×1 | packed | 20,840 | 20,900 | +0.3 % |
| 22 | N x K x 1 B packed M 1 | 4×256×1×1 | packed | 24,310 | 22,680 | -6.7 % |
| 23 | 3 TileN x 2 TileK 7 x 2 TileM 1  B packed multi-tile | 12×519×33×1 | packed | 331,280 | 326,020 | -1.6 % |
| 24 | 1 x 2 TileK x 4 TileM B packed FC-like | 1×512×64×1 | packed | 135,400 | 135,350 | -0.0 % |
| 25 | batch 3 no broadcast B packed | 5×64×19×3 | packed | 80,690 | 75,780 | -6.1 % |
| 26 | batch 4 B broadcasts B packed b stride 0 | 5×64×19×4 | packed | 106,750 | 100,230 | -6.1 % |
| 27 | 1 x 261 x 19 K-split n 1 | 1×261×19×1 | row-major | 76,670 | 76,660 | -0.0 % |
| 28 | 1 x 261 x 19 B packed K-split n 1 | 1×261×19×1 | packed | 39,910 | 39,920 | +0.0 % |
| 29 | 2 x 13 x 5 K-split n 2 | 2×13×5×1 | row-major | 8,440 | 7,960 | -5.7 % |
| 30 | 2 x 13 x 5 B packed K-split n 2 | 2×13×5×1 | packed | 7,420 | 6,990 | -5.8 % |
| 31 | 3 x 517 x 33 K-split n 3 multi-tile | 3×517×33×1 | row-major | 221,060 | 219,870 | -0.5 % |
| 32 | 3 x 517 x 33 B packed K-split n 3 | 3×517×33×1 | packed | 109,490 | 108,300 | -1.1 % |
| 33 | batch 3 6 x 517 x 35 prefetch crosses tiles | 6×517×35×3 | row-major | 1,312,140 (was FAIL) | 1,305,010 **FAIL** | -0.5 % |
| 34 | batch 3 6 x 517 x 35 B packed prefetch crosses tiles | 6×517×35×3 | packed | 643,760 | 636,710 | -1.1 % |
| 35 | batch 3 6 x 517 x 35 B broadcasts b stride 0 | 6×517×35×3 | row-major | 1,312,195 | 1,304,900 | -0.6 % |
| 36 | batch 3 6 x 517 x 35 B broadcasts B packed b stride 0 | 6×517×35×3 | packed | 643,590 | 636,340 | -1.1 % |
| 37 | sat pos a 100 b 100 K 3  AP MAX | 4×3×16×1 | row-major | 9,670 | 7,910 | -18.2 % |
| 38 | sat neg a 100 b -100 K 3  AP MIN | 4×3×16×1 | row-major | 9,510 | 7,730 | -18.7 % |
| | **Σ duration_ns (common cases)** | | | **7,367,640** | **7,203,705** | **-2.2 %** |


Σ over the 37 common cases **−2.8 %**.  The gain is where the change
applies: every geometry whose B is a single block and has more than one
n_tile no longer reloads it — `2·TileN × TileK × TileM` **−32 %**
(83,300 → 56,450 ns), `partial N` −32 %; the small cases gain the batched
A-row requests (`7×13×5` −25 %, saturation cases −18 %, `K=1` outer
product −11 %); the batch cases −4 …−6 % (the B-resident block is loaded
once per slice instead of per n_tile).  Cases with one n_tile or multi-tile
B are unchanged within noise (±0.5 %); nothing regressed.  DSP 51 → 38 at
the same RTL timing confirms the 16×16 multiply is free.

Synthesis: II=1 on every loop, slack 0.00 ns (K-loop 0.04), BRAM 64,
DSP **51 → 38**, FF 10.8 k → 10.3 k, LUT 44.3 k → 43.4 k.  C-sim 39/39.
`TestMatmulBlas` (the `float` build against `cblas_sgemm`) has not compiled
since §3 changed the ports to `hls::burst_maxi` (it still passes `float*`);
untouched here, noted for the record.

---

## 5. Verification matrix

| Gate | Command | Baseline result |
|---|---|---|
| C-simulation | `ctest -R Matmul` | `TestMatmulRef`, `TestMatmulBlas` pass |
| HLS synthesis | `make synthesize_matmul_kv260` | II=1 all loops; Fmax 205.47 MHz |
| RTL behavior test | `make behavior_test_matmul` | 20/20 pass; sim_time 7,713,375 ns |

---

## 6. Related files

| File | Purpose |
|---|---|
| `kernels/matmul/kernel/MatmulKernel.cpp` | HLS kernel (single sequential loop nest) |
| `doc/MATMUL_KERNEL.md` | Kernel reference (architecture, interface, II=1) |
| `hw/cormorant_test_stand/kernels/matmul_op_test/` | Vivado RTL behavior-test project |
| `build/kernels/matmul/kv260/matmul_op_test_report.json` | Per-test behavior-test report |

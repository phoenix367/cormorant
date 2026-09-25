# MatmulKernel — Optimization Log

This document records performance investigation and optimization attempts on
`kernels/matmul/kernel/MatmulKernel.cpp`. Each section describes one change or
experiment, the rationale, and the measured HW behavior-simulation
(`make behavior_test_matmul`) impact on the kv260 RTL.

For the high-level kernel description see [MATMUL_KERNEL.md](MATMUL_KERNEL.md).

> **Status (2026-09-25).** §1 is the original single-sequential-loop-nest
> kernel; §2 records a DATAFLOW restructuring that was **tried and
> rejected**; §3 / §3b (128-bit ports, packed B) and §4–§8 (Track A of
> `doc/THROUGHPUT_PLAN.md`: 16×16 MAC, K-split, rotate scatter, `b_tile`
> ping-pong prefetch, `kTileM = 32`) have **landed**.  The RTL stand runs
> 39 fixtures; every §4–§8 table compares against the §3b kernel on the
> same 39 cases.

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

## 5. K-split across the row lanes when `n_valid < kTileN` (Track A2, 2026-09-25)

**Problem.**  The K-loop rotates over `kTileN = 4` row lanes so that each
`acc[n1]` register is written every 4 cycles (§5 of MATMUL_KERNEL.md).
With fewer than 4 valid rows in the n_tile the idle lanes still take their
turn: a fully-connected layer (`n = 1`, e.g. the classifier `1×1280×1001`
of MobileNet v2 or the `1×256×256` bench case) spends 75 % of its K-loop
cycles on lanes that compute nothing — 0.49 GOps/s on the board against
a 3.2 GOps/s peak (THROUGHPUT_PLAN.md §0).

**Change.**  Lane `n1` now works on row `n1 / lpr`, K-segment `n1 % lpr`,
with `lpr` (lanes per row) the largest power of two such that
`lpr · n_valid ≤ kTileN` — 4 / 2 / 1 for `n_valid = 1 / 2 / 3–4`.  Per
k_tile, `seg_len = ceil(k_valid / lpr)`, the loop runs `seg_len · kTileN`
iterations (`kk = ki / kTileN` is the offset inside the segment, `kl =
seg · seg_len + kk` the K index inside the tile) and the guard
`kl < k_valid` idles the lanes whose segment overruns the tile when `lpr`
does not divide `k_valid`.  Each iteration still reads exactly one `a_buf`
element and one `b_tile` row — the same ports as before — so II=1 is
untouched.  After the k_tile loop a `log2(kTileN)`-stage tree folds the
segment lanes of each row into lane `row · lpr` (`acc[n1] += acc[n1 +
stride]` for `stride = 1, 2, …` while `lpr > stride`, fully unrolled, one
cycle) and the C writer reads lane `n1 << lpr_log`.  Bit-exact: the
fixed-point accumulate is modular, so the order of the partial sums
cannot change the bits (C-sim 39/39 exact, including the three tail-guard
geometries added for this step).

**Traps hit.**  None in HLS — II=1 first time, no partial writes.  The
cost is the reduction tree and its lane muxes: LUT 43.4 k → 45.9 k, FF
10.3 k → 12.3 k (the plan's "~300 LUT" was optimistic; the tree is 48
32-bit adders plus 2:1 selects on the acc registers).  `seg · seg_len` is
kept out of the pipeline as four precomputed `seg_base[]` registers.

**Result (RTL, 39 cases; before = §3b kernel).**

| # | Test | n×k×m×batch | B | before | after | Δ |
|--:|---|---|---|---:|---:|---:|
| 0 | 1x1x1 | 1×1×1×1 | row-major | 6,065 | 6,095 | +0.5 % |
| 1 | TileN x TileK x TileM | 4×256×16×1 | row-major | 43,470 | 41,830 | -3.8 % |
| 2 | 2 TileN x TileK x TileM | 8×256×16×1 | row-major | 83,300 | 56,450 | -32.2 % |
| 3 | TileN x 2 TileK x TileM | 4×512×16×1 | row-major | 78,050 | 76,410 | -2.1 % |
| 4 | TileN x TileK x 2 TileM | 4×256×32×1 | row-major | 79,340 | 77,700 | -2.1 % |
| 5 | TileN 2 x TileK x TileM partial N | 6×256×16×1 | row-major | 80,470 | 49,640 | -38.3 % |
| 6 | TileN x TileK 5 x TileM partial K | 4×261×16×1 | row-major | 44,740 | 43,050 | -3.8 % |
| 7 | TileN x TileK x TileM 3  partial M | 4×256×19×1 | row-major | 79,050 | 77,410 | -2.1 % |
| 8 | TileN 2 x TileK 5 x TileM 3  all partial | 6×261×19×1 | row-major | 154,890 | 142,240 | -8.2 % |
| 9 | 7 x 13 x 5 arbitrary small | 7×13×5×1 | row-major | 15,280 | 11,480 | -24.9 % |
| 10 | N x 1 x M K 1 outer product | 5×1×17×1 | row-major | 13,200 | 11,760 | -10.9 % |
| 11 | 1 x K x M N 1 row vector | 1×256×16×1 | row-major | 38,920 | 31,250 | -19.7 % |
| 12 | N x K x 1 M 1 column vector | 4×256×1×1 | row-major | 40,830 | 39,200 | -4.0 % |
| 13 | 3 TileN x 2 TileK 7 x 2 TileM 1  multi-tile all | 12×519×33×1 | row-major | 663,790 | 658,530 | -0.8 % |
| 14 | batch 3 no broadcast | 5×64×19×3 | row-major | 129,960 | 113,570 | -12.6 % |
| 15 | batch 4 A broadcasts a stride 0 | 5×64×19×4 | row-major | 172,390 | 150,540 | -12.7 % |
| 16 | batch 4 B broadcasts b stride 0 | 5×64×19×4 | row-major | 172,290 | 150,360 | -12.7 % |
| 17 | batch 6 both strided multi-dim flat | 5×64×19×6 | row-major | 256,650 | 223,790 | -12.8 % |
| 18 | TileN x TileK x TileM B packed | 4×256×16×1 | packed | 25,760 | 24,090 | -6.5 % |
| 19 | TileN 2 x TileK 5 x TileM 3  B packed all partial | 6×261×19×1 | packed | 81,760 | 69,080 | -15.5 % |
| 20 | 7 x 13 x 5 B packed arbitrary small | 7×13×5×1 | packed | 14,310 | 11,060 | -22.7 % |
| 21 | 1 x K x M B packed N 1 row vector | 1×256×16×1 | packed | 20,840 | 13,230 | -36.5 % |
| 22 | N x K x 1 B packed M 1 | 4×256×1×1 | packed | 24,310 | 22,650 | -6.8 % |
| 23 | 3 TileN x 2 TileK 7 x 2 TileM 1  B packed multi-tile | 12×519×33×1 | packed | 331,280 | 326,020 | -1.6 % |
| 24 | 1 x 2 TileK x 4 TileM B packed FC-like | 1×512×64×1 | packed | 135,400 | 73,920 | -45.4 % |
| 25 | batch 3 no broadcast B packed | 5×64×19×3 | packed | 80,690 | 64,280 | -20.3 % |
| 26 | batch 4 B broadcasts B packed b stride 0 | 5×64×19×4 | packed | 106,750 | 84,830 | -20.5 % |
| 27 | 1 x 261 x 19 K-split n 1 | 1×261×19×1 | row-major | 76,670 | 61,040 | -20.4 % |
| 28 | 1 x 261 x 19 B packed K-split n 1 | 1×261×19×1 | packed | 39,910 | 24,320 | -39.1 % |
| 29 | 2 x 13 x 5 K-split n 2 | 2×13×5×1 | row-major | 8,440 | 7,690 | -8.9 % |
| 30 | 2 x 13 x 5 B packed K-split n 2 | 2×13×5×1 | packed | 7,420 | 6,760 | -8.9 % |
| 31 | 3 x 517 x 33 K-split n 3 multi-tile | 3×517×33×1 | row-major | 221,060 | 219,870 | -0.5 % |
| 32 | 3 x 517 x 33 B packed K-split n 3 | 3×517×33×1 | packed | 109,490 | 108,300 | -1.1 % |
| 33 | batch 3 6 x 517 x 35 prefetch crosses tiles | 6×517×35×3 | row-major | 1,312,140 (was FAIL) | 1,212,130 **FAIL** | -7.6 % |
| 34 | batch 3 6 x 517 x 35 B packed prefetch crosses tiles | 6×517×35×3 | packed | 643,760 | 543,810 | -15.5 % |
| 35 | batch 3 6 x 517 x 35 B broadcasts b stride 0 | 6×517×35×3 | row-major | 1,312,195 | 1,212,020 | -7.6 % |
| 36 | batch 3 6 x 517 x 35 B broadcasts B packed b stride 0 | 6×517×35×3 | packed | 643,590 | 543,440 | -15.6 % |
| 37 | sat pos a 100 b 100 K 3  AP MAX | 4×3×16×1 | row-major | 9,670 | 7,910 | -18.2 % |
| 38 | sat neg a 100 b -100 K 3  AP MIN | 4×3×16×1 | row-major | 9,510 | 7,730 | -18.7 % |
| | **Σ duration_ns (common cases)** | | | **7,367,640** | **6,605,485** | **-10.3 %** |


Σ over the 37 common cases **−10.4 %** (−7.8 % on top of §4).  The K-split
does what it was built for: `1 × 2·TileK × 4·TileM packed` (the FC shape,
`n = 1`) **−45 %** (135,400 → 73,920 ns), `1 × K × M packed` −37 %,
`1×261×19 packed` −39 %; the row-major twins gain less (`1×261×19` −20 %,
`1×K×M` −20 %) because their B load, not the K-loop, dominates.  Every
case whose last n_tile is short gains too: `partial N` (6 rows → 4 + 2)
−38 %, `all partial` −8 %, the `5×64×19` batch cases (rows 4 + 1)
−13 …−21 %, `6×517×35 batch 3` −8 % / −16 %.  Full n_tiles are unchanged
(`4×256×16` −3.8 %, unchanged since §4); nothing regressed.  The
`6×517×35` row-major case still shows the §4 stand artifact.

Synthesis: II=1 on every loop, slack 0.00 ns (K-loop 0.04), BRAM 64,
DSP 35, FF 12.3 k, LUT 45.9 k.

---

## 6. Rotate-then-wire lane scatter for the A and row-major B loads (Track A4, 2026-09-25)

Landed *before* the ping-pong of §7 (the plan's order is A3 → A4) because
§7 instantiates the row-major B scatter twice — under the K-loop and in
the first-block drain — and with the §3 scatter that would have meant
+18 k LUT and a meaningless resource delta for §7.

**Problem.**  The two word-to-bank scatter loops of §3 selected, for every
destination bank / column, its lane with a *runtime* part-select
(`word.range(16·(l+1)−1, 16·l)`, `l = (j + shift) % 8` for A, `l = m1 −
c0` for B).  HLS builds a 128-bit barrel shifter per destination: the A
scatter was **9.1 k LUT** (8 banks × 4 row RAMs) and the row-major B
scatter **17.9 k LUT** (16 columns) — 61 % of the kernel's 44 k LUT, all
"Expression" in the loop reports.

**Change.**  `matmul_rotate_lanes(word, shift)` (MatmulKernel.h) rotates
the word right by `shift` lanes once — a chain of the seven constant
rotates selected by `shift`, i.e. one 8-way 128-bit mux — after which
bank / column `j` always takes lane `j % 8` of the rotated word (fixed
wiring, `matmul_word_lane` with a constant index).  Only the *enable*
depends on the geometry:

- A: bank `j` holds element `w·8 + j` of the row when `j + shift < 8`,
  else `(w−1)·8 + j`; write iff that element index is `< k` (and `w > 0`
  for the wrapped half of the first word);
- B: column `m1` (`j = m1 % 8`, `p = m1 / 8`) belongs to word `w` iff
  `p == w && j + shift < 8` or `p == w − 1 && j + shift ≥ 8`; write iff
  that holds and `m1 < m_valid`.

The packed path needs no rotate (`shift = 0`, `part = w % 2` selects the
column half).  C-sim 39/39 bit-exact.

**Result.**  LUT **45.9 k → 21.7 k** (A scatter 9 095 → 1 566, B
scatter 17 916 → 1 125), FF 12.4 k, DSP 35, BRAM 64, II=1 everywhere,
slack 0.00.  On the RTL stand the loops issue the same words per cycle,
so the per-case timings are expected to be within noise of §5:

| # | Test | n×k×m×batch | B | before | after | Δ |
|--:|---|---|---|---:|---:|---:|
| 0 | 1x1x1 | 1×1×1×1 | row-major | 6,065 | 6,095 | +0.5 % |
| 1 | TileN x TileK x TileM | 4×256×16×1 | row-major | 43,470 | 41,830 | -3.8 % |
| 2 | 2 TileN x TileK x TileM | 8×256×16×1 | row-major | 83,300 | 56,380 | -32.3 % |
| 3 | TileN x 2 TileK x TileM | 4×512×16×1 | row-major | 78,050 | 76,370 | -2.2 % |
| 4 | TileN x TileK x 2 TileM | 4×256×32×1 | row-major | 79,340 | 77,700 | -2.1 % |
| 5 | TileN 2 x TileK x TileM partial N | 6×256×16×1 | row-major | 80,470 | 49,600 | -38.4 % |
| 6 | TileN x TileK 5 x TileM partial K | 4×261×16×1 | row-major | 44,740 | 43,050 | -3.8 % |
| 7 | TileN x TileK x TileM 3  partial M | 4×256×19×1 | row-major | 79,050 | 77,350 | -2.2 % |
| 8 | TileN 2 x TileK 5 x TileM 3  all partial | 6×261×19×1 | row-major | 154,890 | 142,200 | -8.2 % |
| 9 | 7 x 13 x 5 arbitrary small | 7×13×5×1 | row-major | 15,280 | 11,520 | -24.6 % |
| 10 | N x 1 x M K 1 outer product | 5×1×17×1 | row-major | 13,200 | 11,760 | -10.9 % |
| 11 | 1 x K x M N 1 row vector | 1×256×16×1 | row-major | 38,920 | 31,250 | -19.7 % |
| 12 | N x K x 1 M 1 column vector | 4×256×1×1 | row-major | 40,830 | 39,190 | -4.0 % |
| 13 | 3 TileN x 2 TileK 7 x 2 TileM 1  multi-tile all | 12×519×33×1 | row-major | 663,790 | 658,470 | -0.8 % |
| 14 | batch 3 no broadcast | 5×64×19×3 | row-major | 129,960 | 113,570 | -12.6 % |
| 15 | batch 4 A broadcasts a stride 0 | 5×64×19×4 | row-major | 172,390 | 150,540 | -12.7 % |
| 16 | batch 4 B broadcasts b stride 0 | 5×64×19×4 | row-major | 172,290 | 150,360 | -12.7 % |
| 17 | batch 6 both strided multi-dim flat | 5×64×19×6 | row-major | 256,650 | 223,790 | -12.8 % |
| 18 | TileN x TileK x TileM B packed | 4×256×16×1 | packed | 25,760 | 24,080 | -6.5 % |
| 19 | TileN 2 x TileK 5 x TileM 3  B packed all partial | 6×261×19×1 | packed | 81,760 | 69,090 | -15.5 % |
| 20 | 7 x 13 x 5 B packed arbitrary small | 7×13×5×1 | packed | 14,310 | 11,060 | -22.7 % |
| 21 | 1 x K x M B packed N 1 row vector | 1×256×16×1 | packed | 20,840 | 13,230 | -36.5 % |
| 22 | N x K x 1 B packed M 1 | 4×256×1×1 | packed | 24,310 | 22,640 | -6.9 % |
| 23 | 3 TileN x 2 TileK 7 x 2 TileM 1  B packed multi-tile | 12×519×33×1 | packed | 331,280 | 325,960 | -1.6 % |
| 24 | 1 x 2 TileK x 4 TileM B packed FC-like | 1×512×64×1 | packed | 135,400 | 73,920 | -45.4 % |
| 25 | batch 3 no broadcast B packed | 5×64×19×3 | packed | 80,690 | 64,280 | -20.3 % |
| 26 | batch 4 B broadcasts B packed b stride 0 | 5×64×19×4 | packed | 106,750 | 84,830 | -20.5 % |
| 27 | 1 x 261 x 19 K-split n 1 | 1×261×19×1 | row-major | 76,670 | 61,040 | -20.4 % |
| 28 | 1 x 261 x 19 B packed K-split n 1 | 1×261×19×1 | packed | 39,910 | 24,320 | -39.1 % |
| 29 | 2 x 13 x 5 K-split n 2 | 2×13×5×1 | row-major | 8,440 | 7,690 | -8.9 % |
| 30 | 2 x 13 x 5 B packed K-split n 2 | 2×13×5×1 | packed | 7,420 | 6,760 | -8.9 % |
| 31 | 3 x 517 x 33 K-split n 3 multi-tile | 3×517×33×1 | row-major | 221,060 | 219,870 | -0.5 % |
| 32 | 3 x 517 x 33 B packed K-split n 3 | 3×517×33×1 | packed | 109,490 | 108,260 | -1.1 % |
| 33 | batch 3 6 x 517 x 35 prefetch crosses tiles | 6×517×35×3 | row-major | 1,312,140 (was FAIL) | 1,212,010 **FAIL** | -7.6 % |
| 34 | batch 3 6 x 517 x 35 B packed prefetch crosses tiles | 6×517×35×3 | packed | 643,760 | 543,690 | -15.5 % |
| 35 | batch 3 6 x 517 x 35 B broadcasts b stride 0 | 6×517×35×3 | row-major | 1,312,195 | 1,211,920 | -7.6 % |
| 36 | batch 3 6 x 517 x 35 B broadcasts B packed b stride 0 | 6×517×35×3 | packed | 643,590 | 543,260 | -15.6 % |
| 37 | sat pos a 100 b 100 K 3  AP MAX | 4×3×16×1 | row-major | 9,670 | 7,910 | -18.2 % |
| 38 | sat neg a 100 b -100 K 3  AP MIN | 4×3×16×1 | row-major | 9,510 | 7,730 | -18.7 % |
| | **Σ duration_ns (common cases)** | | | **7,367,640** | **6,604,575** | **-10.4 %** |


Timing unchanged versus §5 within ±0.1 % on every case (Σ 4,850,025 →
4,849,395 ns): the rewrite is purely structural.  The `6×517×35`
row-major case fails in sequence exactly as in §4 — the new scatter reads
the same stale DDR value — which is what proved the artifact sits in the
stand's memory model and not in the load path.

---

## 7. `b_tile` ping-pong: prefetch the next B block under the K-loop (Track A3, 2026-09-25)

**Problem.**  After §3b a `(m_tile, k_tile)` block of B (≤ 512 words) was
still loaded strictly *before* its K-loop (`k_valid · kTileN` ≥ 1 024
cycles for a full n_tile): on `256³ packed` the block loads were 29 % of
the kernel time and none of it overlapped with the MACs
(THROUGHPUT_PLAN.md §0).  The §2 DATAFLOW attempt had shown that a
generic producer/consumer split costs more than it saves here; the
CONV_OPTIMISATION.md §2.35 recipe — two banks in one RAM column set, the
prefetch cursor advanced inside the compute loop — does not.

**Change.**

- `b_tile` is `[kTileM][2·kTileK]`, partitioned on dim 1, `RAM_2P`: one
  RAM column per `m1`, flat `(bank, k1)` address (`bank·kTileK + k1`,
  `kTileK` a power of two).  The K-loop reads bank `cur_bank`, the
  prefetch writes bank `!cur_bank` — one read and one write port per RAM.
- A fetch cursor (`BFetch`) describes the block the *next* B-loading
  iteration will consume — the iteration order is `k_tile` fastest, then
  `m_tile`, `n_tile`, batch, with the §4 B-resident rule (reload only at
  the next batch slice, never when B broadcasts).  Progress is counted in
  rows for both layouts: a row-major block is `k_valid` rows of ≤ 3 words,
  each its own read request, ≤ `kBReqAhead = 16` rows in flight; a packed
  block issues its ≤ 8 × 64-word requests up front and is drained as
  `k_valid` rows of 2 words.  One `b_fetch_step` per K-loop iteration:
  (row-major) request the next row if the window allows, then drain ONE
  word — `b.read()` blocks only when it has not arrived — and scatter it
  with the §6 rotate + fixed wiring into `b_tile[m1][!cur_bank·kTileK +
  rows_done]` (one conditional store per column RAM).
- The K-loop runs `while (ki < ki_bound || prefetch pending)`: MACs stop
  at `ki_bound`, the loop keeps draining until the next block is complete,
  so every B-loading iteration but the first simply swaps banks (a C-sim
  `assert` checks the cursor is in step).  Only the first block of a call
  is loaded by a blocking drain loop.  There is no separate tail loop and
  no second instance of the MAC.
- The accumulate is unconditional: an idle lane (drain-only iteration, or
  the §5 segment tail) multiplies a zero A operand — bit-identical.

C-sim 39/39 bit-exact, including the four `6×517×35` batch-3 cases added
for this step (the cursor crosses `k_tile`, `m_tile`, `n_tile` and batch
boundaries, with and without broadcast) and the packed FC-like case.

**Traps hit** (three synthesis rounds, all II=1; no `DEPENDENCE` pragma
was needed — HLS proved the two banks disjoint from the flat address):

| Form | Result |
|---|---|
| first version, cursor fields `unsigned`, accumulate under `if (mac)` | II=1, latency 13, but K-loop 3.5 k → **11.9 k LUT**, FF +7 k: 32-bit compares / adds and pipeline registers for every cursor field, and HLS wrapped the predicated `acc +=` in a 33-bit compare / select per column |
| cursor fields as bounded `ap_uint` (`BRowCnt`, `BColCnt`, `BWordCnt`, `BShift`; §2.35 "narrow prefetch cursors") | K-loop 10.5 k LUT, FF −1.8 k |
| **+ unconditional accumulate with a zeroed operand** (indices clamped for the idle lanes) | **K-loop 8.5 k LUT**, design 29.3 k LUT — shipped |

The K-loop's remaining +5 k LUT over §6 is the rotate (0.7 k), the 16
conditional column stores with their enables, the row-request address /
word-count arithmetic and the loop's two exit conditions; iteration
latency 4 → 13 (the blocking read and the rotate sit ahead of the store),
which costs ~9 cycles per K-loop entry — invisible next to the ≥ 1 024
cycles of a block.

**Result (RTL, 39 cases; before = §3b kernel).**

| # | Test | n×k×m×batch | B | before | after | Δ |
|--:|---|---|---|---:|---:|---:|
| 0 | 1x1x1 | 1×1×1×1 | row-major | 6,065 | 6,205 | +2.3 % |
| 1 | TileN x TileK x TileM | 4×256×16×1 | row-major | 43,470 | 34,480 | -20.7 % |
| 2 | 2 TileN x TileK x TileM | 8×256×16×1 | row-major | 83,300 | 49,210 | -40.9 % |
| 3 | TileN x 2 TileK x TileM | 4×512×16×1 | row-major | 78,050 | 56,480 | -27.6 % |
| 4 | TileN x TileK x 2 TileM | 4×256×32×1 | row-major | 79,340 | 57,760 | -27.2 % |
| 5 | TileN 2 x TileK x TileM partial N | 6×256×16×1 | row-major | 80,470 | 42,410 | -47.3 % |
| 6 | TileN x TileK 5 x TileM partial K | 4×261×16×1 | row-major | 44,740 | 35,440 | -20.8 % |
| 7 | TileN x TileK x TileM 3  partial M | 4×256×19×1 | row-major | 79,050 | 58,980 | -25.4 % |
| 8 | TileN 2 x TileK 5 x TileM 3  all partial | 6×261×19×1 | row-major | 154,890 | 110,760 | -28.5 % |
| 9 | 7 x 13 x 5 arbitrary small | 7×13×5×1 | row-major | 15,280 | 11,710 | -23.4 % |
| 10 | N x 1 x M K 1 outer product | 5×1×17×1 | row-major | 13,200 | 11,830 | -10.4 % |
| 11 | 1 x K x M N 1 row vector | 1×256×16×1 | row-major | 38,920 | 23,890 | -38.6 % |
| 12 | N x K x 1 M 1 column vector | 4×256×1×1 | row-major | 40,830 | 33,800 | -17.2 % |
| 13 | 3 TileN x 2 TileK 7 x 2 TileM 1  multi-tile all | 12×519×33×1 | row-major | 663,790 | 467,810 | -29.5 % |
| 14 | batch 3 no broadcast | 5×64×19×3 | row-major | 129,960 | 83,930 | -35.4 % |
| 15 | batch 4 A broadcasts a stride 0 | 5×64×19×4 | row-major | 172,390 | 110,620 | -35.8 % |
| 16 | batch 4 B broadcasts b stride 0 | 5×64×19×4 | row-major | 172,290 | 110,840 | -35.7 % |
| 17 | batch 6 both strided multi-dim flat | 5×64×19×6 | row-major | 256,650 | 164,090 | -36.1 % |
| 18 | TileN x TileK x TileM B packed | 4×256×16×1 | packed | 25,760 | 24,170 | -6.2 % |
| 19 | TileN 2 x TileK 5 x TileM 3  B packed all partial | 6×261×19×1 | packed | 81,760 | 68,380 | -16.4 % |
| 20 | 7 x 13 x 5 B packed arbitrary small | 7×13×5×1 | packed | 14,310 | 11,280 | -21.2 % |
| 21 | 1 x K x M B packed N 1 row vector | 1×256×16×1 | packed | 20,840 | 13,360 | -35.9 % |
| 22 | N x K x 1 B packed M 1 | 4×256×1×1 | packed | 24,310 | 22,800 | -6.2 % |
| 23 | 3 TileN x 2 TileK 7 x 2 TileM 1  B packed multi-tile | 12×519×33×1 | packed | 331,280 | 277,050 | -16.4 % |
| 24 | 1 x 2 TileK x 4 TileM B packed FC-like | 1×512×64×1 | packed | 135,400 | 56,320 | -58.4 % |
| 25 | batch 3 no broadcast B packed | 5×64×19×3 | packed | 80,690 | 53,710 | -33.4 % |
| 26 | batch 4 B broadcasts B packed b stride 0 | 5×64×19×4 | packed | 106,750 | 70,590 | -33.9 % |
| 27 | 1 x 261 x 19 K-split n 1 | 1×261×19×1 | row-major | 76,670 | 45,460 | -40.7 % |
| 28 | 1 x 261 x 19 B packed K-split n 1 | 1×261×19×1 | packed | 39,910 | 24,230 | -39.3 % |
| 29 | 2 x 13 x 5 K-split n 2 | 2×13×5×1 | row-major | 8,440 | 7,860 | -6.9 % |
| 30 | 2 x 13 x 5 B packed K-split n 2 | 2×13×5×1 | packed | 7,420 | 6,810 | -8.2 % |
| 31 | 3 x 517 x 33 K-split n 3 multi-tile | 3×517×33×1 | row-major | 221,060 | 157,070 | -28.9 % |
| 32 | 3 x 517 x 33 B packed K-split n 3 | 3×517×33×1 | packed | 109,490 | 92,520 | -15.5 % |
| 33 | batch 3 6 x 517 x 35 prefetch crosses tiles | 6×517×35×3 | row-major | 1,312,140 (was FAIL) | 828,580 **FAIL** | -36.9 % |
| 34 | batch 3 6 x 517 x 35 B packed prefetch crosses tiles | 6×517×35×3 | packed | 643,760 | 448,130 | -30.4 % |
| 35 | batch 3 6 x 517 x 35 B broadcasts b stride 0 | 6×517×35×3 | row-major | 1,312,195 | 828,640 | -36.9 % |
| 36 | batch 3 6 x 517 x 35 B broadcasts B packed b stride 0 | 6×517×35×3 | packed | 643,590 | 447,740 | -30.4 % |
| 37 | sat pos a 100 b 100 K 3  AP MAX | 4×3×16×1 | row-major | 9,670 | 7,930 | -18.0 % |
| 38 | sat neg a 100 b -100 K 3  AP MIN | 4×3×16×1 | row-major | 9,510 | 7,820 | -17.8 % |
| | **Σ duration_ns (common cases)** | | | **7,367,640** | **4,970,695** | **-32.5 %** |


Σ over the 39 cases **−32.5 %** versus the §3b kernel (−24.7 % on top of
§5/§6).  Every multi-block case now overlaps its B loads with the MACs:
`TileN × TileK × TileM` row-major 43,470 → **34,480 ns** (−21 %),
`TileN × 2·TileK` −28 %, `multi-tile all` −30 % (663,790 → 467,810),
`6×517×35 batch 3` −37 % row-major / −30 % packed, the `5×64×19` batch
cases −33 …−36 %, `1 × 2·TileK × 4·TileM packed` (FC) 135,400 →
**56,320 ns** (−58 %, of which −23 % is this step).  Single-block packed
cases, whose one block cannot be overlapped with anything, are unchanged
(`TileN × TileK × TileM packed` −6 %, all from §4/§5); `1×1×1` +2 %
(140 ns — the longer K-loop pipeline).  Nothing regressed.  The
`6×517×35` row-major case still fails in sequence with the same stale
value as §4–§6 (run before the test-stand fix).

Synthesis: II=1 on every loop, slack 0.00 ns, BRAM 64 (unchanged — the
16 column RAMs simply grow from 256 to 512 entries, still one BRAM18
each), DSP 34, FF 17.4 k, LUT **29.3 k** (baseline 44.3 k).

---

## 8. `kTileM` 16 → 32 (Track A5, 2026-09-25)

**Problem.**  With the B block prefetched under the K-loop (§7) and the
lanes fully used for short n_tiles (§5), the K-loop itself is the bound:
16 MACs per cycle, 3.2 GOps/s peak at 100 MHz.  Doubling `kTileM` doubles
the MACs per cycle at the same II, the same A traffic and the same number
of B words per MAC; doubling `kTileN` instead would keep the peak and halve
the utilisation of FC layers (THROUGHPUT_PLAN.md §2).

**Change.**  `platforms/kv260.json` `kernels.matmul.tile_m` 16 → 32 — the
kernel source is generic in `kTileM` (the §7 cursor types, the §6 column
rule, the packed word count `kMatmulWordsPerTileRow = 4` and the request
count `≤ 16 × 64 words = num_read_outstanding` all follow the constant).
The packed B layout changes with it (`matmul_packed_m(m)` now pads to 32),
so every packed image is regenerated: the C-sim / RTL fixtures here, and —
by the coordinator — every packed constant the scheduler emits
(`MATMUL_TILE_M` is read from the same JSON by
`inference-scheduler/src/_matmul_hw_config.py`, so the generated projects
follow automatically once regenerated).  The test-stand testbench's
packed-broadcast slice size is now a `TILE_M` localparam instead of a
literal 16.

**Result (RTL, 39 cases; before = the §7 kernel at `kTileM = 16`, after
= `kTileM = 32`; the geometry column is the *after* fixture — the
`TileM`-relative cases (`TileN × TileK × TileM` is now `4×256×32`,
`FC-like` `1×512×128`, …) compare different amounts of work, the
literal-size cases compare the same).**

| # | Test | n×k×m×batch | B | before | after | Δ |
|--:|---|---|---|---:|---:|---:|
| 0 | 1x1x1 | 1×1×1×1 | row-major | 6,205 | 6,205 | +0.0 % |
| 1 | TileN x TileK x TileM | 4×256×32×1 | row-major | 34,480 | 35,810 | +3.9 % |
| 2 | 2 TileN x TileK x TileM | 8×256×32×1 | row-major | 49,210 | 51,170 | +4.0 % |
| 3 | TileN x 2 TileK x TileM | 4×512×32×1 | row-major | 56,480 | 53,360 | -5.5 % |
| 4 | TileN x TileK x 2 TileM | 4×256×64×1 | row-major | 57,760 | 55,350 | -4.2 % |
| 5 | TileN 2 x TileK x TileM partial N | 6×256×32×1 | row-major | 42,410 | 44,090 | +4.0 % |
| 6 | TileN x TileK 5 x TileM partial K | 4×261×32×1 | row-major | 35,440 | 36,840 | +4.0 % |
| 7 | TileN x TileK x TileM 3  partial M | 4×256×35×1 | row-major | 58,980 | 60,850 | +3.2 % |
| 8 | TileN 2 x TileK 5 x TileM 3  all partial | 6×261×35×1 | row-major | 110,760 | 114,080 | +3.0 % |
| 9 | 7 x 13 x 5 arbitrary small | 7×13×5×1 | row-major | 11,710 | 11,700 | -0.1 % |
| 10 | N x 1 x M K 1 outer product | 5×1×33×1 | row-major | 11,830 | 12,970 | +9.6 % |
| 11 | 1 x K x M N 1 row vector | 1×256×32×1 | row-major | 23,890 | 24,780 | +3.7 % |
| 12 | N x K x 1 M 1 column vector | 4×256×1×1 | row-major | 33,800 | 33,800 | +0.0 % |
| 13 | 3 TileN x 2 TileK 7 x 2 TileM 1  multi-tile all | 12×519×65×1 | row-major | 467,810 | 466,830 | -0.2 % |
| 14 | batch 3 no broadcast | 5×64×35×3 | row-major | 83,930 | 87,180 | +3.9 % |
| 15 | batch 4 A broadcasts a stride 0 | 5×64×35×4 | row-major | 110,620 | 114,860 | +3.8 % |
| 16 | batch 4 B broadcasts b stride 0 | 5×64×35×4 | row-major | 110,840 | 114,790 | +3.6 % |
| 17 | batch 6 both strided multi-dim flat | 5×64×35×6 | row-major | 164,090 | 170,040 | +3.6 % |
| 18 | TileN x TileK x TileM B packed | 4×256×32×1 | packed | 24,170 | 29,950 | +23.9 % |
| 19 | TileN 2 x TileK 5 x TileM 3  B packed all partial | 6×261×35×1 | packed | 68,380 | 90,390 | +32.2 % |
| 20 | 7 x 13 x 5 B packed arbitrary small | 7×13×5×1 | packed | 11,280 | 11,590 | +2.7 % |
| 21 | 1 x K x M B packed N 1 row vector | 1×256×32×1 | packed | 13,360 | 18,630 | +39.4 % |
| 22 | N x K x 1 B packed M 1 | 4×256×1×1 | packed | 22,800 | 27,900 | +22.4 % |
| 23 | 3 TileN x 2 TileK 7 x 2 TileM 1  B packed multi-tile | 12×519×65×1 | packed | 277,050 | 329,130 | +18.8 % |
| 24 | 1 x 2 TileK x 4 TileM B packed FC-like | 1×512×128×1 | packed | 56,320 | 97,900 | +73.8 % |
| 25 | batch 3 no broadcast B packed | 5×64×19×3 | packed | 53,710 | 32,830 | -38.9 % |
| 26 | batch 4 B broadcasts B packed b stride 0 | 5×64×19×4 | packed | 70,590 | 40,650 | -42.4 % |
| 27 | 1 x 261 x 19 K-split n 1 | 1×261×19×1 | row-major | 45,460 | 25,920 | -43.0 % |
| 28 | 1 x 261 x 19 B packed K-split n 1 | 1×261×19×1 | packed | 24,230 | 19,900 | -17.9 % |
| 29 | 2 x 13 x 5 K-split n 2 | 2×13×5×1 | row-major | 7,860 | 7,860 | +0.0 % |
| 30 | 2 x 13 x 5 B packed K-split n 2 | 2×13×5×1 | packed | 6,810 | 7,080 | +4.0 % |
| 31 | 3 x 517 x 33 K-split n 3 multi-tile | 3×517×33×1 | row-major | 157,070 | 107,940 | -31.3 % |
| 32 | 3 x 517 x 33 B packed K-split n 3 | 3×517×33×1 | packed | 92,520 | 74,240 | -19.8 % |
| 33 | batch 3 6 x 517 x 35 prefetch crosses tiles | 6×517×35×3 | row-major | 828,580 (was FAIL) | 567,070 | -31.6 % |
| 34 | batch 3 6 x 517 x 35 B packed prefetch crosses tiles | 6×517×35×3 | packed | 448,130 | 400,390 | -10.7 % |
| 35 | batch 3 6 x 517 x 35 B broadcasts b stride 0 | 6×517×35×3 | row-major | 828,640 | 567,330 | -31.5 % |
| 36 | batch 3 6 x 517 x 35 B broadcasts B packed b stride 0 | 6×517×35×3 | packed | 447,740 | 399,860 | -10.7 % |
| 37 | sat pos a 100 b 100 K 3  AP MAX | 4×3×32×1 | row-major | 7,930 | 8,570 | +8.1 % |
| 38 | sat neg a 100 b -100 K 3  AP MIN | 4×3×32×1 | row-major | 7,820 | 8,480 | +8.4 % |
| | **Σ duration_ns (common cases)** | | | **4,970,695** | **4,368,315** | **-12.1 %** |


39/39 pass, Σ over the 39 cases **−12.1 %** although 20 of them now carry
twice the M (their "before" is half the work).  Same-geometry cases show
the doubled MAC width directly: `3×517×33` row-major 157,070 →
**107,940 ns** (−31 %), `6×517×35 batch 3` row-major −32 %, `1×261×19`
row-major −43 %, the packed `5×64×19` batch cases −39 …−42 % (m = 19 is
now one m_tile instead of two); their packed twins gain less
(`6×517×35` −11 %, `3×517×33` −20 %) because a packed block is now 1 024
words for 1 024 K-loop iterations — the prefetch just fits and the B
port, not the MACs, is the bound.  The `TileM`-relative rows do twice the
work in +3 …+4 % (row-major, i.e. ~1.9× throughput) or +19 …+32 % (packed,
~1.6×); the packed FC-like `1×512×128` takes +74 % for 2× the work —
at `n = 1` it is purely B-bandwidth bound now (8 192 words in 97.9 µs),
as THROUGHPUT_PLAN.md §2 predicted ("then port-bound").  `1×1×1`,
`2×13×5`, `M = 1` are unchanged; the K = 3 saturation cases +8 % (twice
the C writes).  Integration: every packed weight image and generated
project must be regenerated (`MATMUL_TILE_M` follows the JSON); the one
scheduler test that hard-codes the padding, `test_mixed_kernel.py::
TestSpatialMatmulRelu::test_w_flat` (`224 * 16`), needs `MATMUL_TILE_M`.

Synthesis: II=1 on every loop, slack 0.00 ns, BRAM 64 → **80** (the 16
extra `b_tile` column RAMs; +8 BRAM36 tiles, as budgeted in
THROUGHPUT_PLAN.md §6), DSP 34 → **49**, FF 17.4 k → 24.9 k, LUT
29.3 k → **41.8 k** (K-loop 8.5 k → 14.6 k) — still below the 44.3 k the
kernel had before this track.

---

## 9. Verification matrix

| Gate | Command | Result after §8 |
|---|---|---|
| C-simulation | `ctest -R Matmul` | `TestMatmulRef` 39/39 bit-exact (`TestMatmulBlas` does not compile since §3 — it still passes `float*` to the `burst_maxi` ports) |
| HLS synthesis | `make synthesize_matmul_kv260` | II=1 on every loop, slack 0.00 ns at 150 MHz; BRAM 80, DSP 49, LUT 41.8 k |
| RTL behavior test | `make behavior_test_matmul` | 39/39 pass (test stand with per-test alternating DDR base); per-case timings in the §4–§8 tables |

---

## 10. Related files

| File | Purpose |
|---|---|
| `kernels/matmul/kernel/MatmulKernel.cpp` | HLS kernel (single sequential loop nest) |
| `doc/MATMUL_KERNEL.md` | Kernel reference (architecture, interface, II=1) |
| `hw/cormorant_test_stand/kernels/matmul_op_test/` | Vivado RTL behavior-test project |
| `build/kernels/matmul/kv260/matmul_op_test_report.json` | Per-test behavior-test report |

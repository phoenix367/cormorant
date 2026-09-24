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

## 4. Verification matrix

| Gate | Command | Baseline result |
|---|---|---|
| C-simulation | `ctest -R Matmul` | `TestMatmulRef`, `TestMatmulBlas` pass |
| HLS synthesis | `make synthesize_matmul_kv260` | II=1 all loops; Fmax 205.47 MHz |
| RTL behavior test | `make behavior_test_matmul` | 20/20 pass; sim_time 7,713,375 ns |

---

## 5. Related files

| File | Purpose |
|---|---|
| `kernels/matmul/kernel/MatmulKernel.cpp` | HLS kernel (single sequential loop nest) |
| `doc/MATMUL_KERNEL.md` | Kernel reference (architecture, interface, II=1) |
| `hw/cormorant_test_stand/kernels/matmul_op_test/` | Vivado RTL behavior-test project |
| `build/kernels/matmul/kv260/matmul_op_test_report.json` | Per-test behavior-test report |

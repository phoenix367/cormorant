# PoolingKernel in SystemVerilog

**Status (2026-10-05):** the hardware build's PoolingKernel since
[POOL_RTL_PLAN](../plans/POOL_RTL_PLAN.md) phase 3 (bitstream
`dbb320fb7297`); the Vitis HLS kernel's synthesis is retired, its C++
([POOLING_KERNEL](POOLING_KERNEL.md)) is the reference model and fixture
generator.

A drop-in replacement for the HLS kernel: the same IP (VLNV
`xilinx.com:hls:PoolingKernel:1.0`, the HLS export's 110 ports and 24
parameters with their names and defaults), the same AXI-Lite register map,
the same DDR access pattern and bit-identical results.  Both m_axi ports are
128 bits wide, as on the HLS kernel.

## 1. Contract

**Registers** (`s_axi_ctrl`, 8 address bits): `0x00` ap_ctrl (b0 ap_start,
b1 ap_done clear-on-read, b2 ap_idle, b3 ap_ready clear-on-read, b7
auto_restart, b9 interrupt), `0x04` GIE, `0x08` IER, `0x0C` ISR (toggle on
write), `0x10/14` x, `0x1C/20` y, `0x28` batch, `0x30` channels, `0x38`
in_h, `0x40` in_w, `0x48` out_h, `0x50` out_w, `0x58` pool_h, `0x60`
pool_w, `0x68` stride_h, `0x70` stride_w, `0x78` pad_top, `0x80` pad_left,
`0x88` dil_h, `0x90` dil_w, `0x98` pool_type, `0xA0` lp_order, `0xA8`
count_include_pad.  `interrupt` is level high while GIE and a set ISR bit.

**The contract** (PoolingKernel.cpp's `compute_pool_geometry`; the platform
JSON's `kernels.pool` bounds, checked against `pl_pkg` at configure time):
pool_h, pool_w in 1..7; the dilated window at most 16 rows and 64 columns
(`(pool_h − 1)·dil_h < 16`, `(pool_w − 1)·dil_w < 64`).  Outside it the
kernel reads and writes nothing and returns at once — and so it does when
batch, channels, out_h or out_w is 0 (the HLS kernel's behaviour there is
not defined).

**Arithmetic** per Q8.8 element (raw int16), accumulator `ap_fixed<32,16>`
(raw Q16.16, wrapping); the first tap of a window replaces the accumulator:

| pool_type | per tap | a padded / outside tap | result |
|---|---|---|---|
| 0 MAX | max | −128 (0x8000) | the maximum |
| 1 AVG | sum | 0 | `(sum · inv(d)) >>> 23`, `inv(d) = round(2^23 / d)` (0 for d = 0); d = pool_h·pool_w with count_include_pad, else the in-bounds rows × columns |
| ≥ 2, lp_order 1 | sum of \|x\| | 0 | the sum |
| ≥ 2, other lp_order | sum of x² | 0 | `poly_sqrt(sum)` |

then floored to 8 fraction bits and saturated to Q8.8.  `poly_sqrt` is the
HLS polynomial in integers: for a > 0, P = the top set bit, 2k = (P − 16)
rounded down to even, m = a scaled by 2^−2k to [1, 4) with 16 fraction bits;
three Horner steps `t ← floor(t·m + c)` in `ap_fixed<24,4>` with the
`ap_fixed<16,1>` raw constants c3 = 252, c2 = −3091, c1 = 21076, c0 = 14529;
the result
`(t >>> 4)` scaled by 2^k.  This form was checked against the HLS function
on all 2^31 positive accumulators (0 mismatches).

**Geometry** — PoolingKernel.cpp's, word for word: channel tiles of 8;
W-tiles of `ow_tile = (64 − span_w) / stride_w + 1` output columns (capped at
out_w); per (batch, channel tile, W-tile) chunk, the input rows 0 .. (the
last any output row needs) are read once, each (row, channel) as one run of
words from the first to the last column the tile touches; output rows are
written as one run per (row, channel), the first and last words with byte
strobes for the run's own lanes only.  The prefetch mode (load the rows of
output row oh + 1 while emitting oh) iff `span_h + stride_h ≤ 16`, else
load-then-emit.

## 2. Architecture

```
ctrl ─► config ─► chunk sequencer ─┬► loader: runs ─► AR (gmem0) ─► R FIFO (256) ─┐
                                   ├► emitter: line buffer ◄──────────────────────┘ ─► window beats ─► reducer / finaliser ─┐
                                   └► writer: ping-pong row buffers ◄───────────────────────────────────────────────────────┘ ─► AW / W (gmem1)
```

| module | role |
|---|---|
| `pl_ctrl_s_axi` | the AXI-Lite slave (derived from the RTL VectorOPKernel's) |
| `pl_core` | job FSM: latch the registers; the products the chunks need through one pipelined DSP multiplier and `ow_tile` through a 6-step divider (about 30 cycles per job); the contract; the chunk sequencer (batch, channel tile, W-tile — every offset by increments) feeding a 4-deep chunk FIFO per unit; `ap_done` when all units are idle |
| `pl_loader` | a chunk's (row, channel) runs in the HLS order (rows outer, channels inner); one AR burst per run (at most 9 beats), split at 4 KiB; AR issued only when the 256-beat read FIFO can take every beat (RREADY tied high); ≤ 16 bursts awaiting data; a run descriptor (row slot, channel, lane shift, words) per run |
| `pl_emit` | the line buffer — 8 channels × 8 column banks of 16 rows × 8 words (LUTRAM; one write and one read per bank and cycle), a word written in one cycle with its lanes rotated by the run's alignment; per output row a "step" that loads the rows it needs and emits its window beats: per group of 2 output positions, pool_h × pool_w beats of 2 positions × 8 channels, padded taps carrying the pool type's identity, the AVG denominators with the group's last beat |
| `pl_reduce` | 16 accumulators (2 positions × 8 channels): per beat the contribution (x, \|x\| or x²) and max / add; at a group's end a snapshot for the finaliser, which takes one channel (2 lanes) per cycle through a 9-stage pipeline (AVG product with a 64-entry reciprocal ROM, the LP-2 square root as two-cycle Horner steps) into a 32-entry bundle FIFO |
| `pl_writer` | an output row's bundles into one of two row buffers (8 banks of 2 × 8 × 8 entries, LUTRAM); the other buffer's row drained as one run per channel, lanes rotated to the row's alignment, one AW burst per run (split at 4 KiB), a 32-beat W FIFO, ≤ 8 bursts awaiting B |

Burst sizes and outstanding counts follow the HLS interface, and the IP
declares them on its m_axi interfaces as the HLS export does
(`syn/package_ip.tcl`): gmem0 read-only, 16 / 16 outstanding, gmem1
write-only, 16 / 8, 16-beat maximum bursts (the RTL's are at most 9 beats:
a run of 64 columns at any alignment).  The block design sizes each crossbar
slot's acceptance from these.

Resources (out of context, xck26 −2LV, `make synth_pool_rtl`): 11 956 LUT
(3 446 LUTRAM: the line buffer and the row buffers), 8 483 FF, 2 RAMB36 + 2
RAMB18, 29 DSP; timing met at 300 MHz (WNS +0.077 ns, Fmax ≈ 307 MHz; the PL
runs at 100).  The HLS kernel in the production bitstream (b3309f424562,
routed): 17 208 LUT (3 520 LUTRAM), 14 936 FF, 3 RAMB36 + 1 RAMB18, 94 DSP.

## 3. Performance

The window beats are the bound, as in the HLS kernel: one beat (2 output
positions × 8 channels, one tap) per cycle, at least 8 cycles per group
(the finaliser takes one channel per cycle).  A chunk with a single output
row (the global pools) loads all its rows before its first beat, so there
the loads count too.  Cycles from `ap_start` to `ap_done` with ideal memory
(`make perf_pool_rtl`, Verilator; the HLS kernel's board times at 100 MHz
for comparison):

| job | cycles | window-beat cycles | HLS kernel on the board |
|---|---:|---:|---:|
| MaxPool 3×3 s2 p1, 64 × 112² (ResNet-18; W-tiles of 31 + 25 columns) | 123 548 | 116 928 | 1.35 ms |
| MaxPool 2×2 s2, 64 × 56² | 27 019 | 25 088 | 0.335 ms |
| MaxPool 3×3 s1 p1, 32 × 28² | 15 071 | 14 112 | — |
| GlobalAveragePool 7×7, 512 channels | 10 213 | 3 136 (+ 56 runs loaded per chunk) | — |
| GlobalAveragePool 7×7, 1 024 channels | 20 326 | 6 272 (+ 56 runs loaded per chunk) | 0.283 ms |
| AveragePool 2×2 s2, 16 × 28² (LeNet) | 2 136 | 1 568 | — |

On the test stand (the pooling block design in xsim, the 45 fixtures, the
same day): 584 µs of kernel time against the HLS kernel's 694 µs on the 43
cases both pass (−15.8 %), faster in every case (−7 … −28 %).

On the board (bitstream `dbb320fb7297`, the 11 PoolingKernel benchmarks of
`run_remote_perf.py` against the HLS kernel's bitstream back to back): every
case faster, by 8.6 % (MaxPool 3×3 on 56², bound by the window beats as
before) to 30.2 % (MaxPool 3×3 on 14²); the 2×2 and global pools −15 … −20 %.
The models move little — the pools are a small share of their time
(ResNet-18 59.88 → 59.73 ms, LeNet 2.774 → 2.743 ms).  In the bitstream the
kernel takes 11 869 LUT, 8 406 FF, 2 RAMB36 + 2 RAMB18 and 29 DSP; the design
as a whole 5.4k LUT, 6.5k FF, half a BRAM tile and 65 DSP less than with the
HLS kernel (POOL_RTL_PLAN phases 1–2).

## 4. Verification

| check | where | result |
|---|---|---|
| Verilator lint (`-Wall`) | `lint_pool_rtl` | clean |
| the 45 checked-in fixtures (`hw/test_data/pool_test_data`) | `TestPoolRtl` (ctest) | 45 / 45: y.hex and byte-identical to the HLS oracle on the whole output region |
| random jobs against the HLS kernel's C++, randomised AXI timing, protocol checks, the declared outstanding limits, no stray write | ctest: 200 (seed 1); by hand: 12 × 300 (seeds 2–5, fast / slow / random timing) and 3 × 800 | all bit-exact |
| register table vs RTL and the HLS driver | `PoolRtlDriver` (ctest), `gen_driver.py --check --hls-driver` | 19 arguments + 4 control registers agree; the 56 API prototypes are the HLS driver's |
| the test stand's pooling block design in xsim with the RTL IP (upgraded in place of the HLS IP) | `sysim_pool_rtl` | 45 / 45 (`check_test_report.py` PASS); 584 µs against the HLS IP's 694 µs on the 43 cases both pass |

On the board (POOL_RTL_PLAN phase 2): registers written and read back, the
bank-collision corner (fails on the HLS bitstream, passes on the RTL one),
the 148 models of `run_remote_tests`, the MNIST / image-classification /
BERT demos and the SmolLM2-135M / 360M, SmolVLM and Piper library gates —
all bit-exact.

The random jobs (about a quarter W-tiled, a third with more than 8 channels,
7 % without prefetch, half dilated) cover every pool type (also types 3–4
and lp_order 0, 3), count_include_pad, padding up to the window, global
pools, batches, jobs outside the contract and zero-size jobs (nothing may be
read or written).

**The HLS kernel's bank collision.**  The first random run found an output
the HLS kernel gets wrong: with `stride_w > 8` (not a multiple of 8) and left
padding, a group's first position can be padded while the second reads
column bank 0 — and the HLS emitter let the padded position claim bank 0's
read address (its `ucol` is 0), so the second position read word 0 of the
bank instead of its own.  PoolingKernel.cpp now lets only an in-bounds
position claim a bank (the fix changes no other output), TestPoolingSim has
two cases for it (the old code fails both against the float reference), and
they are fixtures 43–44.  The HLS IP in the production bitstream fails both
on the test stand (8 / 40 and 9 / 72 outputs); no model of the test set or
the demos has such a pool (the largest pool stride of the 218 models is 4).  The RTL never had the collision: a position's
bank is its real column's, and two positions of a group never share one.

## 5. Build targets

`TestPoolRtl`, `lint_pool_rtl`, `perf_pool_rtl`, `pool_rtl_tb_fst`,
`driver_pool_rtl` (`build/kernels/pool_rtl/driver/PoolingKernel_v1_0/src`),
`package_pool_rtl` (`build/rtl_ip/PoolingKernel_ip`), `synth_pool_rtl`,
`xsim_pool_rtl`, `sysim_pool_rtl` — `kernels/pool_rtl/CMakeLists.txt`.

## 6. Invariants for changes

- The read FIFO never fills: every AR reserves its beats (`space`), so
  RREADY can stay high.
- At most `RD_OUTS` / `WR_OUTS` bursts are outstanding — the counts
  `package_ip.tcl` declares (fact `rtl.axi_masters`); change both together.
- The emitter and the loader agree on the run order (rows outer, channels
  inner, per chunk the rows the HLS kernel reads); the emitter writes a run
  only into the row slot of a step that allows it (`lrow < tgt`), and a
  prefetching step's new rows never alias the rows its window reads — the
  prefetch condition `span_h + stride_h ≤ 16` guarantees that.
- The two positions of a group read different column banks (`gw = 2` only
  when `stride_w % 8 ≠ 0`); position 0 wins a bank only by address
  priority, never by being padded.
- The finaliser takes a snapshot only when the previous one is done (8
  cycles); a group shorter than 8 beats waits — the HLS kernel's
  `slot_len = max(red_len, 8)`.
- The config FSM's products are captured 4 cycles after their operands
  (registered operands, three product stages); a deeper multiplier moves
  every capture index.

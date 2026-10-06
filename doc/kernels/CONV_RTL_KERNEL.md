# ConvKernel in SystemVerilog

**Status (2026-10-06):** the hardware build's ConvKernel since
[CONV_RTL_PLAN](../plans/CONV_RTL_PLAN.md) phase 3 (first bitstream
`c2b2a6e5e50e`; bit-exact on the board everywhere, phase 2).  The Vitis HLS
kernel's synthesis is retired; its C++ ([CONV_KERNEL](CONV_KERNEL.md)) is the
reference model, testbench oracle and fixture generator.  The scheduler
prices this kernel with `cost_model.rtl_conv_walk` (§3; platform
`kernels.conv.impl = "rtl"`).

A drop-in replacement for the HLS kernel: the same IP (VLNV
`xilinx.com:hls:ConvKernel:1.0`, the HLS export's 200 ports and 46
parameters with their names and defaults), the same AXI-Lite register map,
the same DDR layouts and bit-identical results.  All four m_axi ports are 128
bits wide, as on the HLS kernel.

## 1. Contract

**Registers** (`s_axi_ctrl`, 8 address bits): `0x00` ap_ctrl (b0 ap_start,
b1 ap_done clear-on-read, b2 ap_idle, b3 ap_ready clear-on-read, b7
auto_restart, b9 interrupt), `0x04` GIE, `0x08` IER, `0x0C` ISR (toggle on
write), `0x10/14` x, `0x1C/20` weight, `0x28/2C` bias, `0x34/38` y, `0x40`
batch, `0x48` in_ch, `0x50` in_h, `0x58` in_w, `0x60` out_ch, `0x68` out_h,
`0x70` out_w, `0x78` kh, `0x80` kw, `0x88` stride_h, `0x90` stride_w, `0x98`
dilation_h, `0xA0` dilation_w, `0xA8` pad_top, `0xB0` pad_left, `0xB8`
has_bias, `0xC0` is_depthwise.  `interrupt` is level high while GIE and a set
ISR bit.

**The contract** (ConvKernel.cpp's `compute_conv_geometry`; the platform
JSON's `kernels.conv` bounds, checked against `cv_pkg` at configure time):
in_ch 1..1024, out_ch 1..1280, kh, kw 1..7, the dilated window at most 16
rows and 64 columns (`(kh − 1)·dilation_h < 16`, `(kw − 1)·dilation_w <
64`), one padded accumulator row within 65 536 entries (`out_w ·
ceil(out_ch / 16) · 16 ≤ 65536`), non-zero strides and dilations.  Outside
it the kernel reads and writes nothing and returns at once — and so it does
when batch, out_h or out_w is 0 (the HLS kernel's behaviour there is not
defined).  Depthwise (`is_depthwise`) takes in_ch = out_ch.

**DDR layouts** (`ConvKernel.h`): x and y NCHW; weights tile-major
`[out_ch][ic_tiles][kh][kw][16 lanes]`, the last tile 8 lanes when at most 8
channels remain; depthwise `[out_ch][roundup(kh·kw, 8)]`; bias
`[roundup(out_ch, 8)]`.  Buffers are 16-byte aligned; y runs are written with
byte strobes on their own lanes, so the neighbouring elements of the first
and last word stay untouched.

**Arithmetic** per output (raw Q8.8 int16 operands, an `ap_fixed<32,16>`
accumulator that wraps): `y = sat16((bias·2⁸ + Σ x·w) >>> 8)` — the sum of
the raw products modulo 2³², floored by 8 bits and saturated to Q8.8.
Standard: the sum over in_ch × kh × kw; depthwise: over kh × kw of the
output channel's own input channel; taps outside the input are 0.  The order
of the terms does not change a wrapping sum, so the RTL adds in its own
order and stays bit-exact.

**Geometry** — ConvKernel.cpp's: oh-chunks of `min(out_h, max(1, 65536 /
(out_w · m_tiles · 16)))` output rows, capped (more than 4 m-tiles) so a
chunk's input rows fit the 16-row line buffer; ow-tiles of `(64 − window_w)
/ stride_w + 1` columns, even when more than one; m-groups of at most 4
m-tiles.  Per (image, chunk, input-channel tile, ow-tile, m-group) one
*sweep*: every output row of the chunk, every pixel pair of the ow-tile,
every m-tile of the group, every window position — `G · max(kh·kw, 2)`
instants per pixel pair.  x is read per (input row, channel) run, the
weights per (output channel) run of the slab, the bias in one burst, y per
(channel, 256-pixel segment) run — the HLS kernel's runs, split at 4 KiB.

## 2. Architecture

```
ctrl ─► config ─► sweep sequencer ─┬► x loader: row runs ─► AR/R (gmem0) ─► row buffer ─► columns ─┐
                                   ├► patch producer: line buffer ◄────────────────────────────────┘ ─► pixel-pair beats ─┐
                                   ├► weight loader: slab runs ─► AR/R (gmem1) ─► weight vectors ─┐                         │
                                   ├► bias: one burst (gmem2) ─► bias RAM ───────────────────────┐│                         │
                                   └► engine: weight cache (2 banks) ◄───────────────────────────┼┘ ◄───────────────────────┘
                                        MAC grid: 32 DSP48E2 chains ◄── seeds (bias or accumulators) ┘
                                        accumulators (2 buffers, URAM) ─► drain / transposer ─► y writer: AW/W (gmem3)
```

| module | role |
|---|---|
| `cv_ctrl_s_axi` | the AXI-Lite slave (the RTL PoolingKernel's, with this register table) |
| `cv_core` | job FSM: latch the registers; the products the job needs through one pipelined DSP multiplier and three dividers (65 536 / row words, the ow-tile and the line-buffer row cap), about 40 cycles per job; the contract; the sweep sequencer (image, chunk, input tile, ow-tile, m-group — offsets by increments) feeding a 4-deep descriptor FIFO per unit; `ap_done` when every unit is idle |
| `cv_xload` | for the first m-group of each (chunk, input tile, ow-tile): the input rows the line buffer lacks, each as one run per channel (≤ 16 beats, split at 4 KiB) into one half of a ping-pong row buffer; AR only into a free half, so RREADY stays high; ≤ 16 bursts outstanding; out comes one column of 16 channels per cycle |
| `cv_patch` | the line buffer — 16 channel banks of 16 rows × 64 columns (block RAM, row ih mod 16, column iw mod 64); per output row the new rows from the loader, then per pixel pair and window position one beat of both pixels' 16-lane columns (two read ports), lanes outside the input, past the tile's channels or of a pixel outside the ow-tile zero; a 64-beat patch FIFO |
| `cv_wload` | per sweep one run per output channel of the slab (standard: n_pos positions × two 8-lane beats, one for a half tile; depthwise: dw_words beats of 8 positions); a run requested only when the 512-vector FIFO can take it all; ≤ 8 bursts outstanding |
| `cv_bias` | at job start the whole bias vector (one burst, two across 4 KiB) into a 16-lane word per m-tile |
| `cv_engine` | the weight cache (16 columns × 512 words of 16 lanes, 8 columns in block RAM and 8 in UltraRAM; two banks, the next sweep's slab fills while the current one is read; depthwise slabs written as diagonal words, column m1 holding only lane m1); the issue sequencer (one instant per cycle: output row, pixel pair, m-tile, window position; m-tiles 1..G−1 replay the pair's patch beats from a 64-entry file); the MAC grid — 2 pixels × 16 columns of `cv_mac_chain` (16 DSP48E2 each); seeds (the bias for the first input tile, else the stored partial sum) and write-back; two accumulator buffers of 4096 × 16 × 32 bits (UltraRAM), one per chunk parity |
| `cv_mac_chain` | one (pixel, column) chain: lane l's product enters DSP l through the PCIN cascade with skewed operands, the seed at DSP 0's C port, DSP 15 accumulates over the window positions (W = P); behavioural model for Verilator, the DSP48E2 primitive for Vivado |
| `cv_drain` | per finished chunk: per (m-tile, 256-pixel segment) the accumulator words, saturated, scattered into a transposer of 8 block-RAM banks (2 buffers × 16 channels × 32 words); then one run per channel, 8 pixels per output word |
| `cv_ywriter` | per run the words shifted onto the y alignment, one AW burst (two across 4 KiB), byte strobes on the run's lanes, a 32-beat W FIFO, ≤ 8 bursts awaiting B |

**Differences from the HLS kernel's structure** (the results are the same):
no bias initialisation pass (the first input tile seeds from the bias RAM);
the drain of chunk n overlaps the computation of chunk n + 1 (two
accumulator buffers); the MAC grid is 512 DSPs in cascades instead of 803
DSPs and 32 LUT adder trees; depthwise, standard and MatMul-on-ConvKernel
share one datapath.

Burst sizes and outstanding counts follow the HLS interface, and the IP
declares them on its m_axi interfaces as the HLS export does
(`syn/package_ip.tcl`): gmem0 x read-only, 16 outstanding, bursts ≤ 16;
gmem1 weight read-only, 8, ≤ 128; gmem2 bias read-only, 2, ≤ 256; gmem3 y
write-only, 8, ≤ 64.  The block design sizes each crossbar slot's acceptance
from these.

Resources (out of context, xck26 −2LV, `make synth_conv_rtl`, routed): 18 377
LUT (3 530 LUTRAM, 2 220 SRL: the operand skew), 21 507 FF, 40 RAMB36 + 24
RAMB18, 48 URAM, 518 DSP; timing met at 300 MHz (WNS +0.025 ns, Fmax ≈ 302
MHz; the PL runs at 100).  The HLS kernel in the production bitstream
(routed): 37 233 LUT, 41 457 FF, 46 RAMB36 + 16 RAMB18, 48 URAM, 803 DSP.

## 3. Performance

The grid takes one instant per cycle: 2 output pixels × 16 output channels ×
16 input channels (512 MACs); a pixel pair costs `G · max(kh·kw, 2)`
instants per m-group and input tile, as in the HLS kernel (a 1×1 kernel
uses half the instants: the accumulator buffer's ports take one seed read
and one write-back per cycle).  The drain runs beside the next chunk, the
weight slab of the next sweep loads beside the current one.  Cycles from
`ap_start` to `ap_done` with ideal memory (`make perf_conv_rtl`, Verilator;
the HLS kernel's board times at 100 MHz, the production bitstream, for
comparison):

| job (`run_remote_perf.py` ConvKernel benchmarks) | RTL cycles | grid bound | HLS kernel on the board |
|---|---:|---:|---:|
| 3×3, 1 → 32 ch, 28² | 11 689 | 7 056 | 0.149 ms |
| the same, batch 16 | 117 546 | 112 896 | 2.074 ms |
| 3×3, 64 → 64 ch, 56² | 228 810 | 225 792 | 2.683 ms |
| the same, stride 2 | 66 101 | 56 448 | 0.710 ms |
| 3×3, 64 → 64 ch, 28² | 66 100 | 56 448 | 0.710 ms |
| 1×1, 64 → 128 ch, 56² | 107 911 | 100 352 | 1.838 ms |
| 1×1, 128 → 256 ch, 28² | 107 425 | 100 352 | 1.642 ms |
| 5×5, 16 → 16 ch, 28² | 13 538 | 9 800 | 0.155 ms |
| depthwise 3×3, 32 ch, 28² | 13 138 | 7 056 | 0.154 ms |
| depthwise 3×3, 64 ch, 56² | 77 994 | 56 448 | 1.013 ms |

(grid bound: instants of the sweeps, `n · out_h · ceil(out_w / 2) · G ·
max(kh·kw, 2)` summed over the input tiles and m-groups.)

**On the board** (`c2b2a6e5e50e`, CONV_RTL_PLAN phase 2, HLS and RTL
bitstream back to back): every ConvKernel benchmark 6–43 % faster (3×3 64 ch
56² 2.683 → 2.295 ms — 229.5 k cycles against 228.8 k with ideal memory, the
grid busy 98.4 % of the time; 1×1 64 → 128 ch 56² 1.838 → 1.089 ms).  Of the
1 042 ConvKernel calls the performance-model campaigns measured on both
bitstreams the median is 28.4 % faster (up to 73.5 %: narrow MatMul-on-ConvKernel
jobs); one is slower: a depthwise 7 × 7, stride 2, 256-channel job on a 4 × 4
output (148.7 → 161.2 µs, +8.4 %; a grid case, no shipped model has it).  Its
sweeps are short (8 pixel pairs × 49 positions) and the depthwise slab fills
one window position per cycle (a diagonal word per position), so the fill is
not hidden behind them.

**Cost model** (`inference-scheduler/src/cost_model.py`, `rtl_conv_walk`;
the conv-cycle-model skill's default): the sweeps as a pipeline recurrence —
the weight loader with its FIFO run-ahead, the two weight-cache banks, the
patch producer and x loader, the sweep, the drain beside the next chunk —
with one parameter set tuned to this testbench (`RTL_CONV_SIM`: median error
0.65 %, p90 5.3 % over the 1 042 calls) and one to the board
(`RTL_CONV_BOARD`: DDR latency, x and weight runs; median 3.1 %, p90 14.6 %).
The worst misses are narrow, tall MatMul-on-ConvKernel jobs whose few short
x runs per row wait on the DDR (up to 2.5× the model).  A change of the
kernel's schedule needs the model re-tuned: `test_matmul_on_conv.py`
(`test_rtl_conv_model`) pins its anchors.

## 4. Verification

| check | where | result |
|---|---|---|
| Verilator lint (`-Wall`) | `lint_conv_rtl` | clean |
| the 63 checked-in fixtures (`hw/test_data/conv_test_data`) | `TestConvRtl` (ctest) | 63 / 63: y.hex and byte-identical to the HLS oracle on the whole output region |
| random jobs against the HLS kernel's C++, randomised AXI timing, protocol checks, the declared outstanding limits, no stray write | ctest: 200 (seed 1); by hand about 9 000 over the development, 2 200 on the final RTL (fast / slow / random timing) | all bit-exact |
| the behavioural MAC chain against Vivado's DSP48E2 simulation model, every cycle | `ConvRtlDsp` (ctest) | 199 899 cycles, 0 mismatches |
| register table vs RTL and the HLS driver | `ConvRtlDriver` (ctest), `gen_driver.py --check --hls-driver` | 21 arguments + 4 control registers agree; the API prototypes are the HLS driver's |
| the test stand's conv block design in xsim with the RTL IP (upgraded in place of the HLS IP) | `sysim_conv_rtl` | 63 / 63 (`check_test_report.py` PASS); 3 768 µs against the HLS IP's 4 903 µs (−23.1 %), every case faster (−4.6 … −52.0 %) |

The random jobs cover standard and depthwise convolutions, MatMul-on-ConvKernel
shapes (1 × kw kernels at stride kw, up to 1024 input and 1280 output
channels), kernels up to 7 × 7 with dilation and padding, strides up to 9,
oh-chunks, m-groups (more than 4 m-tiles), ow-tiles (inputs wider than 64),
batches, jobs outside the contract and zero-size jobs (nothing may be read or
written).

## 5. Build targets

`TestConvRtl`, `ConvRtlDsp` (`conv_rtl_dsp_check`), `lint_conv_rtl`,
`perf_conv_rtl`, `conv_rtl_tb_fst`, `driver_conv_rtl`
(`build/kernels/conv_rtl/driver/ConvKernel_v1_0/src`), `package_conv_rtl`
(`build/rtl_ip/ConvKernel_ip`), `synth_conv_rtl`, `xsim_conv_rtl`,
`sysim_conv_rtl` — `kernels/conv_rtl/CMakeLists.txt`.

## 6. Invariants for changes

- **Units latch the job.**  `j` (the latched registers) changes while the
  next job is configured, before the units' reset; anything a unit still
  compares after its job (the bias loader's word count) is latched at its
  start.
- **Weight-cache banks.**  Slab k lives in bank k mod 2.  The fill counts
  slabs begun (`cB`) and filled (`cF`), the issue sweeps started (`cS`) and
  done reading (`cR`): a fill begins only while `cB − cR < 2`, a sweep
  starts only when its slab is in (`cF ≠ cS`).  The bank comes from `cB`,
  not `cF`: the next fill can begin in the cycle the previous one's done
  pulse is still registered.
- **Accumulator buffers.**  A buffer is free, owned by the engine (from the
  chunk's first sweep) or by the drain (from its drain request until its last
  read); a chunk's first sweep waits for its buffer to be free.  A sweep that
  reads stored partial sums starts only when the previous sweep's
  write-backs are done or that sweep was at least `HAZ` = 80 instants long.
- **The transposer** buffer of a segment is taken from the segment's last
  read on (its scatter takes 3–4 more cycles), so the fill never starts the
  segment after next in it.
- **Read FIFOs never fill**: the x loader requests a row only into a free
  half of its row buffer, the weight loader a run only when its vector FIFO
  can take every beat — RREADY stays high.
- At most `X_OUTS` / `W_OUTS` / `B_OUTS` / `Y_OUTS` bursts are outstanding —
  the counts `package_ip.tcl` declares (fact `rtl.axi_masters`); change both
  together.
- **Chain timing** (`cv_mac_chain`): lane l's operands at E + l, the seed at
  E + 1, `wfb` at E + 16, the sum at E + 18; the issue-to-write-back latency
  `LAT` = 21 and every delay line in `cv_engine` follow from it.  A change of
  the DSP register configuration must keep the behavioural model and the
  primitive equal (`ConvRtlDsp`).

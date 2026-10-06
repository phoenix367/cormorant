# ConvKernel in SystemVerilog: plan

**Status (2026-10-06):** done.  Phase 0: RTL, testbench, IP (bit-exact,
300 MHz out of context, the test stand 23.1 % faster); phase 1: bitstream
`c2b2a6e5e50e`; phase 2: bit-exact on the board everywhere, every conv
benchmark 6–43 % faster (ResNet-18 59.7 → 47.5 ms); phase 3: the build's
ConvKernel, the HLS synthesis retired, the scheduler's RTL cost model (BERT
919 → 839 ms), the performance model and baseline of `c2b2a6e5e50e`.

A SystemVerilog ConvKernel (`kernels/conv_rtl/`) to replace the Vitis HLS
one (`kernels/conv/`), as the MatmulKernel, VectorOPKernel and PoolingKernel
were ([MATMUL_RTL_PLAN](MATMUL_RTL_PLAN.md), [VECTOROP_RTL_PLAN](VECTOROP_RTL_PLAN.md),
[POOL_RTL_PLAN](POOL_RTL_PLAN.md)): a drop-in IP with the same VLNV
(`xilinx.com:hls:ConvKernel:1.0`), ports, register map, m_axi bus parameters,
DDR layouts and bit-identical results; then the block design, the board and
the scheduler; finally the retirement of the HLS kernel's synthesis (its C++
stays as the reference model and fixture generator).  After it no Vitis HLS
kernel is left in the hardware build.

## 1. What the RTL must reproduce

The HLS kernel (`kernels/conv/kernel/ConvKernel.cpp`, post-§2.42 of
[CONV_OPTIMISATION](../kernels/CONV_OPTIMISATION.md); the platform's
`kernels.conv` bounds: tile_m = tile_ic = 16, kernels ≤ 7 × 7, a line buffer
of 16 rows × 64 columns, 65 536 accumulator entries, 4 m-tiles per weight
group, in_ch ≤ 1024, out_ch ≤ 1280).

**Registers** (`xconvkernel_hw.h`, control address width 8): `0x00` ap_ctrl,
`0x04` GIE, `0x08` IER, `0x0C` ISR, `0x10/14` x, `0x1C/20` weight, `0x28/2C`
bias, `0x34/38` y, `0x40` batch, `0x48` in_ch, `0x50` in_h, `0x58` in_w,
`0x60` out_ch, `0x68` out_h, `0x70` out_w, `0x78` kh, `0x80` kw, `0x88`
stride_h, `0x90` stride_w, `0x98` dilation_h, `0xA0` dilation_w, `0xA8`
pad_top, `0xB0` pad_left, `0xB8` has_bias, `0xC0` is_depthwise.

**Ports** (all 128 bits): gmem0 x (read-only; 16-beat bursts, 16
outstanding), gmem1 weight (read-only; 128 / 8), gmem2 bias (read-only;
256 / 2), gmem3 y (write-only; 64 / 8) — the HLS export's bus parameters;
the IP has 200 ports and 46 parameters.

**DDR layouts:** x and y NCHW (16-byte aligned buffers, whole-word padded);
weights tile-major packed (`[out_ch][ic_tiles][kh][kw][16 lanes]`, the last
tile 8 lanes when ≤ 8 channels remain), depthwise `[out_ch][roundup(kh·kw,
8)]`, bias `[roundup(out_ch, 8)]` (`ConvKernel.h`).  y run ends are written
with byte strobes, so neighbouring lanes stay untouched.

**Arithmetic** (Q8.8 in, `ap_fixed<32,16>` accumulators that wrap):
`y = sat16(floor((bias·2⁸ + Σ x·w) / 2⁸))` — the sum of raw int16 products
modulo 2³², floored by 8 bits and saturated.  The order of the terms does
not matter (wrap-around), so the RTL may sum in any order and stay
bit-exact.  Standard: the sum over in_ch × kh × kw; depthwise: over kh × kw of
the output channel's own input channel; out-of-bounds input taps are 0.

**Geometry** (`compute_conv_geometry`): oh-chunks of
`min(out_h, max(1, 65536 / (out_w · out_ch_pad)))` rows (out_ch_pad =
m_tiles · 16), capped so a chunk's input rows fit the 16-row line buffer when
there is more than one m-group; ow-tiles of `(64 − window_w) / stride_w + 1`
columns rounded to even; m-groups of ≤ 4 m-tiles.  The scheduler keeps every
call inside the contract (window ≤ 16 rows / 64 columns, kh, kw ≤ 7,
`out_w · out_ch_pad ≤ 65536`, in_ch ≤ 1024, out_ch ≤ 1280).

**Performance to meet:** the HLS kernel's standard sweep reads one weight
word per column per cycle and computes 2 output pixels × 16 × 16 = 512 MACs
per cycle (a pixel pair costs G · max(kh·kw, 2) cycles per m-group);
depthwise 2 × 16 MACs per cycle.  Its bias initialisation (Phase 1, one
cycle per accumulator word) and drain (Phase 3, 8 outputs per cycle) are
serial with the sweep — on 1×1 layers the drain is as long as the sweep.

## 2. Design

```
ctrl ─► config ─► sweep sequencer ─┬► x loader: row runs ─► AR/R (gmem0) ─► row buffer ─► columns ─┐
                                   ├► patch producer: line buffer ◄─────────────────────────────────┘ ─► pixel-pair beats ─┐
                                   ├► weight loader: slabs ─► AR/R (gmem1) ─► weight vectors ─┐                          │
                                   ├► bias: one burst (gmem2) ─► bias RAM ───────────────────┐│                          │
                                   └► engine: weight cache (ping-pong) ◄────────────────────┼┘ ◄─────────────────────────┘
                                        MAC grid: 32 cascaded DSP chains ◄── seeds (bias or accumulators) ┘
                                        accumulators (ping-pong per chunk, URAM) ─► drain / transpose ─► writer: AW/W (gmem3)
```

- **One work list.**  A sequencer walks the HLS loop nest once and sends
  every unit the same sweep descriptors ((ni, chunk, input tile, ow-tile,
  m-group)); the units run decoupled through FIFOs as the HLS dataflow
  stages do.
- **The MAC grid as DSP cascades.**  Each (output channel column, pixel) is
  a chain of 16 DSP48E2 (one per input-channel lane, products summed
  through the PCIN cascade with skewed operands) whose last DSP also
  accumulates over the kernel window (its P feedback); the seed (bias or the
  stored accumulator) enters at the first DSP's C port.  512 DSPs and no LUT
  adder trees (the HLS kernel: 803 DSPs and 32 trees).
- **Depthwise on the same grid**: a sweep with one m-tile and a diagonal
  weight word (column m1 holds only lane m1), so depthwise, standard and
  MatMul-on-ConvKernel share one datapath.
- **No bias initialisation pass**: the first input tile of every output
  takes its seed from the bias RAM instead of the accumulator buffer.
- **Drain overlapped with compute**: two accumulator buffers (URAM, one per
  chunk parity) — chunk n drains while chunk n + 1 computes.
- DDR reads and writes follow the HLS kernel's runs and bursts (x per (row,
  channel) run, weights per (m, ic-tile) slab, bias in one burst, y per
  (channel, 256-pixel segment) run with byte-strobed ends), split at 4 KiB,
  within the declared outstanding counts.

## 3. Decisions

As for the other three kernels: the RTL lives in this repository; the HLS
kernel's synthesis is retired once the RTL kernel is on the board and
correct; the chat server may be stopped for board work; no 150 MHz attempt
for the bitstream (100 MHz), but 300 MHz out of context as the timing goal.
Outside the contract (a window past the line buffer, kh or kw > 7, zero
sizes, strides or dilations, a padded accumulator row past 65 536 entries,
in_ch > 1024, out_ch > 1280) the RTL reads and writes nothing; the HLS
kernel's behaviour there is not defined.

## 4. Phases

### Phase 0: RTL, testbench, IP

- `kernels/conv_rtl/`: `rtl/`, `tb/verilator/`, `syn/`, `scripts/`,
  `CMakeLists.txt`.
- Verilator testbench: the 63 checked-in fixtures (`hw/test_data/conv_test_data`)
  and random jobs (standard, depthwise, MatMul-on-ConvKernel geometries,
  chunking, m-groups, ow-tiles) checked against the HLS kernel's own C++ on
  the whole output region (every byte), under randomised AXI timing, with
  protocol checks and the declared outstanding limits.  The behavioural DSP
  chain is checked against the DSP48E2 simulation model.
- The C driver from the register table (MatmulKernel's generator); IP
  packaging with the HLS VLNV and bus parameters; out-of-context synthesis;
  the test stand's conv block design in xsim with the RTL IP.
- Done when: lint clean, fixtures + random jobs bit-exact, the test-stand
  run passes and is no slower, timing met at 300 MHz out of context, fewer
  resources than the HLS kernel.

Results (2026-10-05), all met — kernel reference:
[CONV_RTL_KERNEL](../kernels/CONV_RTL_KERNEL.md):

| check | result |
|---|---|
| `lint_conv_rtl` (Verilator `-Wall`) | clean |
| `TestConvRtl` (ctest: 63 fixtures + 200 random, seed 1, random AXI timing) | all pass; by hand about 9 000 more random jobs over the development (fast / slow / random timing), the final RTL 2 200 (seeds 91, 92, 101, 102), all bit-exact |
| `ConvRtlDsp` (the behavioural MAC chain against the DSP48E2 unisim model) | 199 899 cycles, 0 mismatches |
| `perf_conv_rtl` (ideal memory) | every board benchmark below the HLS kernel's board time: 3×3 64 ch 56² 228 810 cycles (HLS 2.683 ms), 1×1 64 → 128 ch 56² 107 911 (1.838 ms), depthwise 64 ch 56² 77 994 (1.013 ms); the 3×3 job at 98.7 % of the grid bound |
| `ConvRtlDriver` / `gen_driver.py --check --hls-driver` | 21 arguments + 4 control registers agree with the RTL and the HLS driver; the API prototypes are the HLS driver's |
| packaged IP against the HLS export (`component.xml`) | the same 200 ports, 46 parameters (values equal) and m_axi bus parameters |
| `synth_conv_rtl` | 18 377 LUT, 21 507 FF, 40 RAMB36 + 24 RAMB18, 48 URAM, 518 DSP (HLS, routed: 37 233 LUT, 41 457 FF, 46 + 16, 48 URAM, 803 DSP); WNS +0.025 ns at 3.333 ns |
| `sysim_conv_rtl` (the test stand's block design, RTL IP) | 63 / 63; 3 768 µs of kernel time against the HLS IP's 4 903 µs on the same flow (−23.1 %), every case faster (−4.6 … −52.0 %) |

Found on the way:

- **Five RTL bugs**, each caught by the testbench before any hardware: the
  transposer's block-RAM read took two cycles but the emitter consumed the
  first (every output word rotated by one); the bias loader ignored its start
  pulse, which came with the unit reset; a sweep's weight slab went to the
  bank of the previous fill when the next fill began in the cycle the
  previous one's done pulse was still registered (fills are now counted as
  they begin); the bias loader kept comparing with the job registers, which
  the next job's configuration overwrites, and issued a stray burst into the
  next job (it latches its word count now); and the drain reserved a
  transposer buffer only when its segment's scatter ended, so with one-pixel
  segments over three or more m-tiles the segment after next overwrote it
  (reserved at the segment's last read now).  The random generator then
  gained MatMul-shaped jobs up to 1024 × 1280 channels and dilations that
  make single-column ow-tiles.
- **300 MHz.**  The first out-of-context run reached 185 MHz: the sweep
  sequencer's tile arithmetic, the drain's saturate-and-scatter after the
  accumulator UltraRAM, the operand skew (one shift register fanning out to
  16 DSP columns) and the weight cache's read address (64 RAMs).  Then 217,
  294 and 302 MHz: the sequencer in three stages with incrementally kept
  tile ends, two drain register stages, the skew as a shift register plus
  four end registers each driving 4 columns, a read-address copy per column,
  the patch producer's next-row window computed during the current row's
  emission, the x loader's and y writer's output registers, the weight mask
  between the cache's two read registers.  Cycle counts within 30 of the
  first version's.
- **The test stand's conv testbench** traced two HLS dataflow processes by
  hierarchical reference (`+VERBOSE` only), which stops the elaboration of
  any other IP; they are under `` `ifdef CONV_TB_HLS_PROBES `` now
  (hw/cormorant_test_stand).  `sysim.tcl` reopens the project once when the
  interconnect's crossbar stays locked after the upgrade (the test stand's
  own flow does the same).

### Phase 1: block design and bitstream — done (2026-10-05)

The hardware build takes the RTL IP (`AXI_CONV_IMPL=rtl`); the behaviour test
and `sim_hw_kv260` with it; a bitstream; utilisation and timing against the
current one.

- `AXI_CONV_IMPL=hls|rtl` (top-level CMake, default `hls` until phase 3):
  with `rtl`, `synthesize_<platform>` depends on `package_conv_rtl` instead
  of `synthesize_conv_<platform>`, `ip_repo_kv260` links
  `build/rtl_ip/ConvKernel_ip`, and `behavior_test_conv` runs on it.
- The bitstream: `build_hw128` configured with `-DAXI_CONV_IMPL=rtl`,
  `make package_conv_rtl ip_repo_kv260`, then `build.sh all -ip-repo
  build_hw128/ip_repo_kv260` (31 min).  The upgrade changed only
  `ConvKernel_0`; its four ports keep 128 bits.

| check | result |
|---|---|
| bitstream **c2b2a6e5e50e** (`/mnt/data/bitstreams/kv260_rtl_c2b2a6e5e50e/`) | timing met at 100 MHz: WNS +1.956 ns (production dbb320fb7297: +0.816 ns, its worst path inside the HLS ConvKernel), WHS +0.010 ns |
| routed utilisation against dbb320fb7297 | 56 516 LUT (−18 956), 48 328 FF (−19 943), 103 BRAM tiles (−2), 48 URAM, 688 DSP (−285) |
| `sim_hw_kv260` (the whole block design) | 68 / 68 (VectorOP 22, Conv 17, Matmul 10, Pool 19) |
| `sysim_conv_rtl` on the final IP (the behaviour test's block design) | see phase 0 |

### Phase 2: on the board — correct everywhere, every conv benchmark faster (2026-10-05/06)

Registers, `run_remote_tests` (148 models), the 10 ConvKernel benchmarks (no
case slower, same-session A/B), the demos and the chat / TTS gates bit-exact.

The chat server was stopped; the production bitstream `dbb320fb7297` ran the
A side, then the local configs moved to the RTL driver
(`build/kernels/conv_rtl/driver/ConvKernel_v1_0/src`; the same API and
register map) and `c2b2a6e5e50e` was loaded with `upload_bitstream.py` (HPC0
/ HPC1 widths 128; UIO `fabric_vecop` / `fabric_matmul` / `fabric_conv` /
`fabric_pool`).  Every board job ran under the board lock; the chat-model
and TTS gates built and installed into a scratch directory (`--remote-dir
/root/cv_gate`, their own weight directories), removed after each gate.
Logs: `/mnt/data/bitstreams/kv260_rtl_c2b2a6e5e50e/` (`phase2.out`, `p2/`).

**Correctness — everything bit-exact:**

| check | result |
|---|---|
| ConvKernel registers: write and read back (GIE, IER, the 25 argument words) | 27 / 27 |
| `run_remote_tests` (`remote_config_all_models.json`, the RTL ConvKernel driver) | 148 / 148 |
| the 60 kernel benchmarks | all pass |
| MNIST, 10 000 images | identical (convnet 98.92 %, LeNet 97.35 %) |
| image classification (ResNet-18, MobileNet v1 / v2) | outputs identical to the HLS bitstream's (logits and top-5) |
| BERT-SQuAD, 50 examples | 50 / 50 bit-exact with the emulation (3 / 3 with the scheduler simulation) |
| `llm_board` SmolLM2-135M, SmolLM2-360M | logits 4 × 33 / 33 bit-exact each; chunked prefill, threads, close → open identical |
| `llm_board` SmolVLM-256M | 2 images, 33 / 33 logits bit-exact each |
| `tts_board` Piper | PCM, text encoder and duration predictor bit-exact |

**Speed.**  The 10 conv benchmarks, HLS and RTL bitstream back to back in
one session (`p2/perf_a.json`, `p2/perf_b.json`):

| case | HLS | RTL | |
|---|---:|---:|---:|
| 3×3 64 ch 56² / its stride-2 / 28² | 2.6831 / 0.7111 / 0.7096 ms | 2.2946 / 0.6671 / 0.6671 ms | −14.5 / −6.2 / −6.0 % |
| 1×1 64 → 128 ch 56², 128 → 256 ch 28² | 1.8381 / 1.6419 ms | 1.0886 / 1.0828 ms | −40.8 / −34.1 % |
| 3×3 1 → 32 ch 28², batch 1 / 16 | 0.1493 / 2.0740 ms | 0.1228 / 1.1817 ms | −17.7 / −43.0 % |
| 5×5 16 ch 28² | 0.1560 ms | 0.1415 ms | −9.3 % |
| depthwise 3×3 32 ch 28², 64 ch 56² | 0.1538 / 1.0127 ms | 0.1380 / 0.7862 ms | −10.3 / −22.4 % |

The 3×3 64-channel job takes 229.5 k cycles on the board against 228.8 k in
the Verilator testbench with ideal memory: the grid is the bound (98.4 %
busy).  The other 50 benchmarks are unchanged (−3.2 … +1.7 %, the small
MatMuls' run-to-run spread).

| workload | `dbb320fb7297` (HLS ConvKernel) | `c2b2a6e5e50e` |
|---|---:|---:|
| ResNet-18 / MobileNet v1 / v2 | 59.73 / 72.83 / 62.69 ms | 47.45 / 41.23 / 38.05 ms (−20.6 / −43.4 / −39.3 %) |
| MNIST convnet / LeNet | 0.256 / 2.743 ms | 0.235 / 2.701 ms |
| BERT mean of 50 | 919.4 ms | 866.1 ms (−5.8 %) |
| SmolLM2-135M decode; prefill 16 / 64 / 256 | 100.3 ms; 251 / 442 / 1279 ms | 100.3 ms; 249 / 385 / 1120 ms |
| SmolLM2-360M decode; prefill 16 / 64 / 256 | 251.7 ms; 568 / 999 / 2897 ms | 251.8 ms; 566 / 831 / 2602 ms |
| SmolVLM `llm_image`, decode | 3.89–3.91 s, 99.9 ms | 3.36 s (−14 %), 99.9 ms |
| Piper RTF (6.9 s utterance); its 128-frame chunk | 0.518; 708 ms | 0.302 (−42 %); 428 ms |

Decode runs on MatmulKernel (GEMV) and does not move; prefill attention,
the projections lowered onto ConvKernel, the vision encoder and Piper's
flow / HiFi-GAN convolutions do.

### Phase 3: the default and the clean-up — done (2026-10-06)

The RTL kernel becomes the build's ConvKernel, the HLS synthesis targets go
(the C++ reference, `TestConvRef` / `TestConvGrid` / `TestConvSweep` stay),
the performance-model campaign for the new bitstream (the cost model's
ConvKernel terms re-checked), perf-regression baseline, facts and docs.

**Build.**  `synthesize_conv_<platform>`, `cosim_conv_<platform>` and
`kernels/conv/scripts/` (`Synthesis.tcl.in`, `Cosim.tcl.in`) are gone;
`synthesize_<platform>` packages the four RTL IPs, `ip_repo_kv260` links
them all from `build/rtl_ip/`, `behavior_test_conv` takes `package_conv_rtl`.
A cache still holding the phase-1 `AXI_CONV_IMPL` other than `rtl` is
refused, and so is a platform JSON whose `kernels.conv.impl` is not `rtl`.
No Vitis HLS synthesis is left: `AXI_BUS_WIDTH` affects no IP any more, the
platform JSON's `clock` is informational.  ctest: 16 tests (`ConvRtlDriver`,
`TestConvRtl`, `ConvRtlDsp` added), all pass.

**Scheduler.**  `platforms/kv260.json` `kernels.conv.impl = "rtl"`
(`_conv_hw_config.CONV_IMPL`, env `AXI_CONV_IMPL=hls` for the older
bitstreams).  `cost_model.conv_cycles` / `conv_board_cycles` follow it: on
`rtl` the new `rtl_conv_walk` — the kernel's sweeps as a pipeline recurrence
(weight loader with its FIFO run-ahead, the two weight-cache banks, patch
producer and x loader, the sweep, the drain beside the next chunk), one
parameter set tuned to the Verilator testbench (`RTL_CONV_SIM`: median
error 0.65 %, p90 5.3 % over the 1 042 ConvKernel calls of the calibration
list) and one to the board (`RTL_CONV_BOARD`: median 3.1 %, p90 14.6 %,
against 18.7 / 37 % for the HLS model on its bitstream).  The perf model's
ConvKernel families gain its terms (`perf_model._rtl_conv_terms`).  The
conv-cycle-model skill calls it by default (`--arch rtl`; `--arch 42` the
HLS walk).  One fix came along for both kernels: the ow-tile width is made
even before it is capped at `out_w` (`_conv_geom`, the skill's `geom`), as
the kernel does.  The scheduler suite has 1651 tests (3 new: the RTL engine
choices, the RTL model's anchors), all pass.

**Engine choices.**  The cheaper ConvKernel moves work onto it: BERT's 12
attention P·V MatMuls (0.21 ms per head on ConvKernel, against 4.45 ms for
the batched MatmulKernel call) — 96 of the 98 MatMuls on ConvKernel, 360
calls; the LLM, VLM and TTS projects keep their engines but change conv
geometries (kw / out_w).  LeNet's fully-connected convs stay on
MatmulKernel; the CNN projects are unchanged.  The five changed projects
were regenerated (135M, 360M, SmolVLM, Piper, BERT) and gated on the board
in a scratch directory:

| check | result |
|---|---|
| `llm_board` SmolLM2-135M / 360M | logits bit-exact (4 × 33 / 33 each); decode 99.8 / 252.0 ms / token; prefill 16 / 64 / 256: 243 / 334 / 1118 ms (phase 2: 249 / 385 / 1120) and 565 / 828 / 2621 ms (566 / 831 / 2602) |
| `llm_board` SmolVLM-256M | 2 images × 33 / 33 logits bit-exact; `llm_image` 3.37 s, decode 99.4 ms |
| `tts_board` Piper | PCM, text encoder and duration predictor bit-exact; RTF 0.303 (6.9 s utterance) |
| BERT-SQuAD demo, 50 examples | 50 / 50 bit-exact, EM / F1 88.0 / 90.3 = float; p50 **839.2 ms** (865.1 in phase 2, 919.4 on the HLS ConvKernel) |
| 60 kernel benchmarks, MNIST, image classification | all pass, the numbers of phase 2 (ResNet-18 47.46 ms, MobileNet v1 / v2 41.23 / 38.10, MNIST 0.235 / 2.701 ms, 98.92 / 97.35 %) |
| perf-regression against the `dbb320fb7297` baseline | 0 regressions, 15 improved (10 conv benchmarks, 4 demos, BERT −8.6 %); recorded as the baseline of `c2b2a6e5e50e` |

Of the 1 042 ConvKernel calls the campaigns measured on both bitstreams the
median is 28.4 % faster; one is slower: a depthwise 7 × 7 stride-2 job on a
4 × 4 output (+8.4 %, a grid case no model has; CONV_RTL_KERNEL §3).

**Performance model** (`perf_models/kv260/c2b2a6e5e50e.json`, for `--plan`):
the campaign started from `dbb320fb7297`'s converged case list (one pass and
a refinement round of 136 calls), then the list was rebuilt with the RTL
cost model's engine choices (1 402 cases) and refined to convergence — 181,
93, 32, 16 and 5 calls — to 2 343 exact calls, repeat spread median
0.017 %; `host.json` unchanged.  The ConvKernel families use the RTL walk's
terms: conv / conv-dw held-out p90 9.4 / 12.1 % (39.4 / 31.6 % with the HLS
terms), conv-mm 42.8 % (as on the HLS bitstream; the planner falls back on
exact calls there).  Against `dbb320fb7297` the 1 042 common ConvKernel
calls are a median 28.4 % faster, the other kernels' calls unchanged (median
0.00 %).  `perf_calibrate.py simulate` against the phase-3 measurements: 23
of 25 phases within ±2.4 % (ResNet-18 / MobileNet v1 / v2 +0.0 / +0.1 /
−0.0 %, BERT +0.3 %, 135M and SmolVLM prefill / decode / `llm_image` within
1.3 %, Piper chunk and encoder buckets within 1.5 %); SmolLM2-360M prefill
16 / 256 −7.2 / +3.8 % — `host.json` has none of its host-op signatures
(the per-kind fits price them), a profile of it would close that.


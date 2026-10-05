# PoolingKernel in SystemVerilog: plan

**Status (2026-10-05):** done.  The SystemVerilog PoolingKernel is the
hardware build's (bitstream `dbb320fb7297`, in production), bit-exact on the
board with every pool benchmark 8.6–30.2 % faster than the HLS kernel; the
HLS synthesis is retired.

A SystemVerilog PoolingKernel (`kernels/pool_rtl/`) to replace the Vitis HLS
one (`kernels/pool/`), as the MatmulKernel ([MATMUL_RTL_PLAN](MATMUL_RTL_PLAN.md))
and the VectorOPKernel ([VECTOROP_RTL_PLAN](VECTOROP_RTL_PLAN.md)) were: a
drop-in IP with the same VLNV (`xilinx.com:hls:PoolingKernel:1.0`), ports,
register map, m_axi bus parameters, DDR behaviour and bit-identical results;
then the block design, the board and the scheduler; finally the retirement
of the HLS kernel's synthesis (its C++ stays as the reference model and
fixture generator).

## 1. What the RTL must reproduce

The HLS kernel (`kernels/pool/kernel/PoolingKernel.cpp`, the platform's
`kernels.pool` bounds: 8 channels per tile, windows ≤ 7 × 7, a line buffer of
16 rows × 64 columns, 2 output positions per cycle).

**Registers** (`xpoolingkernel_hw.h`, control address width 8): `0x00`
ap_ctrl, `0x04` GIE, `0x08` IER, `0x0C` ISR, `0x10/14` x, `0x1C/20` y,
`0x28` batch, `0x30` channels, `0x38` in_h, `0x40` in_w, `0x48` out_h,
`0x50` out_w, `0x58` pool_h, `0x60` pool_w, `0x68` stride_h, `0x70`
stride_w, `0x78` pad_top, `0x80` pad_left, `0x88` dil_h, `0x90` dil_w,
`0x98` pool_type, `0xA0` lp_order, `0xA8` count_include_pad.

**Ports:** gmem0 (x, read-only) and gmem1 (y, write-only), 128 bits; the HLS
export declares 16 / 16 outstanding bursts and 16-beat bursts on gmem0, 16 /
8 and 16-beat bursts on gmem1.

**The contract** (`compute_pool_geometry`): pool_h, pool_w in 1..7; a
dilated window of at most 16 rows and 64 columns (dil_h < 16, dil_w < 64
where the taps are more than one).  Outside it the kernel reads and writes
nothing and returns at once.

**Arithmetic** per Q8.8 element, accumulator `ap_fixed<32,16>` (wraps):

| pool_type | per tap | identity of a padded / outside tap | result |
|---|---|---|---|
| 0 MAX | max | −128 | the maximum |
| 1 AVG | sum | 0 | `sum · inv(d)` with `inv(d) = round(2^23 / d)` (`ap_ufixed<24,1>`, 0 for d = 0), floored to 16 fraction bits; d = pool_h·pool_w with count_include_pad, else the in-bounds taps (rows × columns) |
| 2 LP, lp_order 1 | sum of \|x\| | 0 | the sum |
| ≥ 2 LP, other lp_order | sum of x² | 0 | `poly_sqrt(sum)`: range reduction to m ∈ [1, 4), a cubic in `ap_fixed<24,4>` (Horner, each step floored), scaled back by 2^k |

then floored to 8 fraction bits and saturated to Q8.8.  Pool types ≥ 3 take
the LP branch.

**Geometry:** channel tiles of 8; W-tiles of `ow_tile` output columns whose
input span fits 64 columns; per (batch, channel tile, W-tile) chunk the
input rows 0 .. (the last any output row needs) are read once, as one
(row, channel) run of words each; output rows are written as one
(row, channel) run each, the first / last words with byte strobes for the
run's own lanes only.

## 2. Design

```
ctrl ─► config ─► chunk sequencer ─┬► loader: runs ─► AR (gmem0) ─► R FIFO ─┐
                                   ├► emitter: line buffer ◄────────────────┘ ─► windows ─► reducer / finaliser ─┐
                                   └► writer: row buffers (ping-pong) ◄───────────────────────────────────────────┘ ─► AW / W (gmem1)
```

- The same structure as the HLS dataflow, as one pipeline without per-row
  or per-chunk loop overheads: the line buffer of 8 channels × 8 column
  banks (LUTRAM, one write and one read per bank and cycle), 2 output
  positions × 8 channels per window beat, 2 finalised outputs per cycle, a
  ping-pong row buffer in the writer.
- The DDR reads are the HLS kernel's word ranges (one burst per run, split
  at 4 KiB); at most 16 bursts in flight.  Writes: one burst per run, split
  at 4 KiB, at most 8 awaiting B.
- The AVG reciprocal is a 64-entry ROM (the contract bounds d by 49); the
  LP-2 square root is the HLS polynomial as an integer pipeline.

## 3. Decisions

As for the MatmulKernel and the VectorOPKernel: the RTL lives in this
repository; the HLS kernel's synthesis is retired once the RTL kernel is on
the board and correct; no 150 MHz attempt; the chat server may be stopped
for board work.  Outside the contract with zero sizes (out_h, out_w,
channels or batch 0) the RTL reads and writes nothing; the HLS kernel's
behaviour there is not defined (it can read runs of a wrapped length).

## 4. Phases

### Phase 0: RTL, testbench, IP

- `kernels/pool_rtl/`: `rtl/`, `tb/verilator/`, `syn/`, `scripts/`,
  `CMakeLists.txt`.
- Verilator testbench: the 43 checked-in fixtures (`hw/test_data/pool_test_data`)
  and random jobs checked against the HLS kernel's own C++ on the whole
  output region (every byte), under randomised AXI timing, with protocol
  checks and the declared outstanding limits.
- The C driver from the register table (MatmulKernel's generator); IP
  packaging with the HLS VLNV and bus parameters; out-of-context synthesis;
  the test stand's pooling block design in xsim with the RTL IP.
- Done when: lint clean, fixtures + random jobs bit-exact, the test-stand
  run passes and is no slower, timing met at 300 MHz out of context, fewer
  resources than the HLS kernel.

Results (2026-10-05), all met — kernel reference:
[POOL_RTL_KERNEL](../kernels/POOL_RTL_KERNEL.md):

| check | result |
|---|---|
| `lint_pool_rtl` (Verilator `-Wall`) | clean |
| `TestPoolRtl` (ctest: 45 fixtures + 200 random, seed 1, random AXI timing) | all pass; by hand also 12 × 300 random (seeds 2–5 × fast / slow / random timing) and 4 × 800 (seeds 11–14) |
| `perf_pool_rtl` (ideal memory) | ResNet-18's MaxPool 3×3 s2 on 64 × 112²: 123 548 cycles (the HLS kernel on the board: 1.35 ms); within 6–8 % of the window-beat bound on the strided pools |
| `PoolRtlDriver` / `gen_driver.py --check --hls-driver` | 19 arguments + 4 control registers agree with the RTL and the HLS driver; the 56 API prototypes are the HLS driver's |
| `synth_pool_rtl` | 11 956 LUT, 8 483 FF, 2 RAMB36 + 2 RAMB18, 29 DSP (HLS, routed in b3309f424562: 17 208 LUT, 14 936 FF, 3 + 1, 94 DSP); WNS +0.077 ns at 3.333 ns |
| `sysim_pool_rtl` (the test stand's block design, RTL IP) | 45 / 45; 584 µs against the HLS kernel's 694 µs on the 43 cases both pass (the production HLS IP, the same day), every case 7–28 % faster |

Found on the way:

- **The HLS kernel reads a wrong column when a padded position shares a
  bank.**  With `stride_w > 8` (not a multiple of 8) and left padding, a
  group's padded first position claimed column bank 0's read address (its
  clamped column is 0), so the second position, whose column was in bank 0,
  read word 0 of the bank instead of its own.  The first random run of the
  RTL testbench hit it (LpPool 1×5, stride 4×10, pad_left 2: 9 of 72 outputs
  differ).  PoolingKernel.cpp now lets only an in-bounds position claim a
  bank; two TestPoolingSim cases fail on the old code against the float
  reference and pass now; they are fixtures 43–44 (43 → 45; the old 43
  unchanged).  The production HLS IP fails both on the test stand (8 / 40
  and 9 / 72 outputs) — no model of the test set or the demos has such a
  pool (the largest pool stride of the 218 models is 4).  The RTL never had
  it.
- **`seq_done` of the previous job.**  The second job of a run finished at
  once: the chunk sequencer's done flag was still set during the first
  cycle of the next job (its unit reset comes a cycle after `job_start`).
  Cleared at `job_start` now.
- **300 MHz.**  The first OOC run reached 270 MHz: the config FSM's
  operand mux into the DSP multiplier, the chunk sequencer's input-span
  arithmetic, the emitter's step setup, the LP-2 Horner steps (multiply and
  add in one cycle) and the loader's burst decision.  Each got a register
  (the sequencer one cycle per chunk, the finaliser three of latency);
  window-beat throughput unchanged (cycle counts within 5).
- The shared driver generator lost the space after a 22-character macro
  name (`COUNT_INCLUDE_PAD_DATA`); fixed in `kernels/matmul_rtl/scripts/gen_driver.py`
  (the MatmulKernel and VectorOPKernel drivers are byte-identical).

### Phase 1: block design and bitstream — done (2026-10-05)

The hardware build takes the RTL IP; the behaviour test and `sim_hw_kv260`
with it; a bitstream; utilisation and timing against the current one.

- `AXI_POOL_IMPL=hls|rtl` (top-level CMake, default `hls` until phase 3):
  with `rtl`, `synthesize_<platform>` depends on `package_pool_rtl` instead
  of `synthesize_pool_<platform>`, `ip_repo_kv260` links
  `build/rtl_ip/PoolingKernel_ip`, and `behavior_test_pool` runs on it.
- The bitstream: `build_hw128` configured with `-DAXI_POOL_IMPL=rtl`,
  `make package_pool_rtl ip_repo_kv260`, then `build.sh all -ip-repo
  build_hw128/ip_repo_kv260` (40 min).  The upgrade changed only
  `PoolingKernel_0` (it drops the HLS-only parameters `combinational`,
  `latency`, `II`); its ports keep 128 bits, no instance parameter reset.

| check | result |
|---|---|
| bitstream **dbb320fb7297** (`/mnt/data/bitstreams/kv260_rtl_dbb320fb7297/`) | timing met at 100 MHz: WNS +0.816 ns, WHS +0.010 ns |
| routed utilisation against b3309f424562 | 75 472 LUT (−5 442), 68 271 FF (−6 530), 105 BRAM tiles (−0.5), 48 URAM, 973 DSP (−65) |
| `PoolingKernel_0` (routed, hierarchical) | 11 869 LUT (3 404 LUTRAM), 8 406 FF, 2 RAMB36 + 2 RAMB18, 29 DSP (HLS: 17 208 LUT, 14 936 FF, 3 + 1, 94 DSP) |
| `behavior_test_pool` (the test stand, RTL IP) | 45 / 45 |
| `sim_hw_kv260` (the whole block design) | 68 / 68 (VectorOP 22, Conv 17, Matmul 10, Pool 19) |

### Phase 2: on the board — correct everywhere, every pool benchmark faster (2026-10-05)

`run_remote_tests` (148 models), the 11 pooling benchmarks (no case slower,
same-session A/B), the demos and the chat / TTS gates bit-exact.

The chat server was stopped; the production bitstream `b3309f424562` ran
the A side, then `dbb320fb7297` was loaded with `upload_bitstream.py` (HPC0 /
HPC1 widths 128; UIO `fabric_vecop` / `fabric_matmul` / `fabric_conv` /
`fabric_pool`).  Every board job ran under the board lock; the chat-model and
TTS gates built and installed into a scratch directory (`--remote-dir
/root/pl_gate`, their own weight directories), removed after each gate.
Logs: `/mnt/data/bitstreams/kv260_rtl_dbb320fb7297/` (`phase2.out`, `p2/`).

**Correctness — everything bit-exact:**

| check | result |
|---|---|
| PoolingKernel registers: write and read back (GIE, IER, the 21 argument words) | 23 / 23 |
| the bank-collision corner (one-node AveragePool 3×3 stride 9 pad 1 on 4 × 12 × 40, MaxPool 3×3 stride 9 pad 1 on 16 × 20 × 100) through `run_remote_tests` | HLS bitstream `b3309f424562`: the AveragePool FAILS (y[26], y[31], y[36] … off by one or two LSBs — the bug in the HLS hardware), the MaxPool passes (a wrong tap seldom beats the window maximum); RTL bitstream: both pass |
| `run_remote_tests` (`remote_config_all_models.json`, the PoolingKernel driver from `gen_driver.py`) | 148 / 148 |
| the 60 kernel benchmarks | all pass |
| MNIST, 10 000 images | identical (convnet 98.92 %, LeNet 97.35 %) |
| image classification (ResNet-18, MobileNet v1 / v2) | results identical to the baseline (`compare_perf.py`: 0 result changes) |
| BERT-SQuAD, 50 examples | 50 / 50 bit-exact with the emulation (3 / 3 with the scheduler simulation) |
| `llm_board` SmolLM2-135M, SmolLM2-360M | logits 4 × 33 / 33 bit-exact each; chunked prefill, threads, close → open identical |
| `llm_board` SmolVLM-256M | 2 images, 33 / 33 logits bit-exact each |
| `tts_board` Piper | PCM, text encoder and duration predictor bit-exact |

**Speed.**  The 11 pool benchmarks, HLS and RTL bitstream back to back in
one session (`p2/perf_a.json`, `p2/perf_b.json`):

| case | HLS | RTL | |
|---|---:|---:|---:|
| MaxPool 2×2 56² / 28² | 0.3352 / 0.1164 ms | 0.2758 / 0.0991 ms | −17.7 / −14.9 % |
| MaxPool 3×3 56² / 14² | 0.3388 / 0.0510 ms | 0.3097 / 0.0356 ms | −8.6 / −30.2 % |
| AvgPool 2×2 56², 3×3 28² | 0.3353 / 0.1165 ms | 0.2762 / 0.0996 ms | −17.6 / −14.5 % |
| GlobalMaxPool 7×7 256, GlobalAvgPool 7×7 64 / 1024 | 0.0771 / 0.0256 / 0.2831 ms | 0.0624 / 0.0210 / 0.2275 ms | −19.1 / −18.0 / −19.6 % |
| AvgPool 2×2 on 7×7 × 1024, on 3×3 × 32 × 112 | 0.2833 / 0.6795 ms | 0.2278 / 0.6067 ms | −19.6 / −10.7 % |

The other 49 benchmarks are unchanged (within ±1.7 %, the small MatMuls'
run-to-run spread; `compare_perf.py` against the `b3309f424562` baseline: 0
regressions, 11 improved).

| workload | `b3309f424562` (baseline) | `dbb320fb7297` |
|---|---:|---:|
| ResNet-18 / MobileNet v1 / v2 | 59.88 / 72.91 / 62.76 ms | 59.73 / 72.83 / 62.69 ms |
| MNIST convnet / LeNet | 0.261 / 2.774 ms | 0.256 / 2.743 ms |
| BERT mean of 50 (no pool) | 919.1 ms | 919.4 ms |
| SmolLM2-135M decode; prefill 16 / 64 / 256 | 100.3 ms | 100.3 ms; 251 / 442 / 1279 ms |
| SmolLM2-360M decode; prefill 16 / 64 / 256 | 251.9 ms | 251.7 ms; 568 / 999 / 2897 ms |
| SmolVLM `llm_image`, decode | 3.89–3.91 s | 3.89–3.91 s, 99.9 ms |
| Piper RTF (6.9 s utterance) | 0.519 | 0.518 |

The pools are a small share of the CNNs (ResNet-18 −0.15 ms: its stem
MaxPool), so the models move little.

### Phase 3: the default and the clean-up — done (2026-10-05)

The RTL kernel becomes the build's PoolingKernel, the HLS synthesis targets
go (the C++ reference and `TestPoolingSim` stay), the performance-model
campaign for the new bitstream, perf-regression baseline, facts and docs.

**Build.**  The top-level CMake drops `AXI_POOL_IMPL` and always packages
the RTL IP for `synthesize_<platform>`, `ip_repo_kv260` (`build_hw_kv260`,
`sim_hw_kv260`) and `behavior_test_pool`; a cache that still asks for `hls`
is refused at configure time.  `kernels/pool/` loses `synthesize_pool_<p>`,
`cosim_pool_<p>`, `Synthesis.tcl.in` and `Cosim.tcl.in`; its C++,
`TestPoolingSim` (51 tests) and `gen_pool_test_data` stay.  ConvKernel is
the one Vitis HLS kernel left.  The `pool-verify` skill (HLS synthesis
gates) is gone; `kernel-verify pool` runs the RTL gates (`TestPoolRtl`,
`lint_pool_rtl`, `synth_pool_rtl`, `behavior_test_pool`, the timing diff).

**Driver.**  Projects take the RTL kernel's generated driver from
`build/kernels/pool_rtl/driver/PoolingKernel_v1_0/src` (`make
driver_pool_rtl`; the HLS driver's API and files): the example and local
configs, the scheduler's and the demos' help texts, the docs.  Facts:
`registers.PoolKernel` reads the RTL control block through the driver
generator, `paths.pool_driver` ties the driver directory to its mentions,
`rtl.axi_masters` covers the pool kernel's bus parameters.

**Promotion.**  `dbb320fb7297` is the production bitstream (the board's
`pl.bin`; the chat server restarted on it and answers).  The performance
model `perf_models/kv260/dbb320fb7297.json` started from the converged case
list of `b3309f424562` (the scheduler had not changed): 1581 calls measured
twice (repeat spread median 0.016 %), a refinement round of 0 calls, 1581
exact calls; mm-gemv / mm-tiled held-out p90 0.79 / 3.22 %.  Against
`b3309f424562` the 30 PoolingKernel calls of the shipped models are a median
19.5 % faster (2.7–36.1 %); the Conv, Matmul and VectorOP calls agree to a
median 0.00 % (within ±2.7 %).  Perf-regression baseline
`kv260-dbb320fb7297.json`: the 60 kernel benchmarks of the A/B run and the
phase-2 demos.

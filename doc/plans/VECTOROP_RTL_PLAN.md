# VectorOPKernel in SystemVerilog: plan

**Status (2026-10-05):** done — the SystemVerilog VectorOPKernel is the hardware build's (bitstream `68665fc1833a`, production); the HLS kernel's synthesis is retired.

A SystemVerilog VectorOPKernel (`kernels/vectorop_rtl/`) to replace the
Vitis HLS one (`kernels/vectorop/`), the same way the MatmulKernel was
replaced ([MATMUL_RTL_PLAN](MATMUL_RTL_PLAN.md)): a drop-in IP with the same
VLNV (`xilinx.com:hls:VectorOPKernel:1.0`), ports, register map, DDR
behaviour and bit-identical results; then integration into the block
design, the board and the scheduler, and finally the retirement of the HLS
kernel's synthesis (its C++ stays as the reference model and fixture
generator).

## 1. What the RTL must reproduce

The HLS kernel (`kernels/vectorop/kernel/VectorOP.cpp`, `include/VectorOP.h`):

**Registers** (`xvectoropkernel_hw.h`, control address width 7): `0x00`
ap_ctrl (start, done COR, idle, ready COR, auto_restart, interrupt), `0x04`
GIE, `0x08` IER, `0x0C` ISR (toggle on write), `0x10/14` a, `0x1C/20` b,
`0x28/2C` c, `0x34` size, `0x3C` op, `0x44` outer, `0x4C` a_inc, `0x54`
b_inc, `0x5C` act.

**Arithmetic** on Q8.8 (`ap_fixed<16,8>`, raw int16), per lane:

| op | result (raw) |
|---|---|
| 0 ADD, 1 SUB | `sat16(a ± b)` |
| 2 MUL | `sat16((a · b) >>> 8)` — floor (AP_TRN), then saturate |
| 3 DIV | `b == 0 ? 0 : sat16(trunc((a << 8) / b))` — C division, toward zero |
| 4 RELU | `max(a, 0)` |
| 5 RELU6 | `min(max(a, 0), 0x0600)` |
| ≥ 6 | `a` (b not read) |

then `act` (1 RELU, 2 RELU6, anything else none) on the result.  Ops ≥ 4
are unary: no transaction on gmem1.

**Geometry** (per operand, `inc` = a_inc, b_inc, or c_inc = a_inc + b_inc
for the output): `n_words = ceil(size / 8)`, the tail lanes of every run's
last word (past `size`) read as 0, so the output's tail lanes are
op(0, 0) = 0 and every run's last word is written whole.

- `size == 0` or `outer == 0`: nothing is read or written.
- inputs only: `outer > 1`, `inc == 0`, `n_words ≤ 256` → read once, replay
  `outer` times;
- `outer == 1`, or `inc == size` with `size % 8 == 0` → one contiguous range
  of `outer · n_words` words;
- otherwise `outer` runs of `n_words` words, run `o` at word `o · ⌊inc / 8⌋`.

Base addresses are 16-byte aligned (the contract; the low four bits are
ignored).  The RTL follows the same formulas, so it reads and writes the same
words in the same order even outside the contract (an `inc` that is not a
multiple of 8).

**Speed of the HLS kernel:** one 128-bit word per cycle on every stage, except
DIV (one lane per cycle, 8 cycles per word).

## 2. Design (phase 0)

```
ctrl ─► job config ─┬► a rungen ─► axi_rd (gmem0) ─► tail mask / replay ─┐
                    ├► b rungen ─► axi_rd (gmem1) ─► tail mask / replay ─┼► ALU (8 lanes) ─┐
                    │                                                    └► DIV (1 lane)  ─┼► c FIFO ─► axi_wr (gmem2)
                    └► c rungen ─────────────────────────────────────────────────────────────┘
```

- AXI masters as in the RTL MatmulKernel: reads in INCR bursts of ≤ 64 beats
  that never cross 4 KiB, issued only when the read FIFO can take the whole
  burst (RREADY held high); writes in bursts of ≤ 256 beats, AW issued once
  the burst's beats are buffered, a bounded number awaiting B.
- One word per cycle through the ALU (8 lanes; MUL on DSPs); DIV through one
  pipelined radix-2 divider at one lane per cycle, as HLS.
- `ap_done` once every B response has arrived.

## 3. Decisions

As for the MatmulKernel (2026-10-03): the RTL lives in this repository; the
HLS kernel's synthesis is retired once the RTL kernel is on the board and
correct (after phase 2); no 150 MHz attempt; the chat server may be stopped
for board work.

## 4. Phases

### Phase 0: RTL, testbench, IP

- `kernels/vectorop_rtl/`: `rtl/`, `tb/verilator/`, `syn/`, `scripts/`,
  `CMakeLists.txt`.
- Verilator testbench: the 119 checked-in fixtures (`hw/test_data/vecop_test_data`)
  and random cases checked against the HLS kernel's own C++ (`VectorOP.cpp`
  compiled with the Vitis HLS headers) on the whole output region — every
  byte the HLS model writes and nothing else — under randomised AXI timing,
  with protocol checks.
- The C driver generated from a register table (MatmulKernel's generator,
  made reusable); IP packaging with the HLS VLNV; out-of-context synthesis;
  the test stand's VectorOP block design in xsim with the RTL IP.
- Done when: lint clean, fixtures + random cases bit-exact, the test-stand
  run passes, timing met at 300 MHz out of context (the PL runs at 100), and
  fewer resources than the HLS kernel.

Results (2026-10-05), all met — kernel reference:
[VECTOROP_RTL_KERNEL](../kernels/VECTOROP_RTL_KERNEL.md):

| check | result |
|---|---|
| `lint_vectorop_rtl` (Verilator `-Wall`) | clean |
| `TestVectorOpRtl` (ctest: 119 fixtures + 300 random, seed 1, random AXI timing) | all pass; by hand also 2 × 500 random (seeds 1, 2026) and 200 with slow timing (seed 4711) |
| `perf_vectorop_rtl` (ideal memory) | 0.93–0.97 words per cycle, DIV 1 lane per cycle |
| `VectorOpRtlDriver` / `gen_driver.py --check --hls-driver` | 9 arguments + 4 control registers agree with the RTL and the HLS driver |
| `synth_vectorop_rtl` | 5 243 LUT, 6 453 FF, 10 BRAM36, 11 DSP (HLS: ~22.7k LUT, 13.7k FF, 32 BRAM18, 33 DSP); WNS +0.248 ns at 3.333 ns |
| `sysim_vectorop_rtl` (the test stand's block design, RTL IP) | 119 / 119; 984 µs against the HLS kernel's 1 000 µs (`behavior_test_vectorop` on the HLS IP, the same day) |

Found on the way:

- **The HLS kernel's C simulation and its hardware disagree on one DIV
  input.**  a = 0x8000, b = 0xFFFF: the synthesised divider is 25 bits wide
  and saturates to 0x7FFF, C simulation divides in 24 bits and gives
  0x8000.  The RTL follows the hardware; the testbench's oracle is corrected
  for that pair.  To confirm on the board in phase 2.
- **A packaged RTL IP needs the HLS export's m_axi bus parameters.**  The
  first package declared none, so the block design took every port as
  read-write with 2 outstanding bursts and the crossbar held the reads of
  short-run jobs to about four in flight (the test stand: up to 23 % slower
  than HLS on those).  `package_ip.tcl` now declares the HLS values
  (outstanding 16, burst lengths, read-only / write-only) and the RTL never
  exceeds them.  The RTL MatmulKernel's IP has the same gap: the production
  bitstreams since 1d28630fbfa4 configure its crossbar slots at 2 outstanding
  bursts (the HLS kernel's: A 4, B 16, C 16) — a follow-up of its own,
  outside this plan.
- `vo_burstgen` missed 300 MHz by 0.29 ns (a 60-bit add after the burst
  length); restructured, same cycle counts.

### Phase 1: block design and bitstream — done (2026-10-05)

- `AXI_VECTOROP_IMPL=hls|rtl` (top-level CMake, default `hls` until phase
  3): with `rtl`, `synthesize_<platform>` depends on `package_vectorop_rtl`
  instead of `synthesize_vectorop_<platform>`, `ip_repo_kv260` links
  `build/rtl_ip/VectorOPKernel_ip`, and `behavior_test_vectorop` runs on it.
- The bitstream: `build_hw128` configured with `-DAXI_VECTOROP_IMPL=rtl`,
  `make package_vectorop_rtl ip_repo_kv260`, then `build.sh all -ip-repo
  build_hw128/ip_repo_kv260` (the command `build_hw_kv260` runs, without its
  `synthesize_kv260` step: the Conv and Pool exports in `build_hw128` are the
  current bitstream's).  46 min.  The upgrade changed only `VectorOPKernel_0`
  (IP revision); the crossbar slots of its three ports keep the HLS
  kernel's acceptance (read 16 / 16, write 16).

| check | result |
|---|---|
| bitstream **68665fc1833a** (`/mnt/data/bitstreams/kv260_rtl_68665fc1833a/`) | timing met at 100 MHz: WNS +0.761 ns, WHS +0.010 ns |
| placed utilisation against bbb9a37f73f8 | 80 965 LUT (−3 596), 74 802 FF (−4 777), 105.5 BRAM tiles (−4), 48 URAM, 1 038 DSP (−22) |
| `VectorOPKernel_0` (routed, hierarchical) | 5 160 LUT (230 LUTRAM), 6 469 FF, 10 BRAM36, 11 DSP |
| `behavior_test_vectorop` (the test stand, RTL IP) | 119 / 119, 984 µs (HLS IP: 1 000 µs) |
| `sim_hw_kv260` (the whole block design) | 68 / 68 (VectorOP 22, Conv 17, Matmul 10, Pool 19) |

### Phase 2: on the board — correct everywhere, no case slower (2026-10-05)

The chat server was stopped, bitstream `68665fc1833a` loaded with
`upload_bitstream.py` (HPC0 / HPC1 widths 128; UIO `fabric_vecop` /
`fabric_matmul` / `fabric_conv` / `fabric_pool`).  Every board job ran under
the board lock; the chat-model and TTS gates built and installed into a
scratch directory (`--remote-dir /root/vo_gate`, their own weight
directories), removed after each gate.  Logs:
`/mnt/data/bitstreams/kv260_rtl_68665fc1833a/` (`phase2.out`, `p2/`).

**Correctness — everything bit-exact:**

| check | result |
|---|---|
| VectorOPKernel registers: write and read back (GIE, IER, the 12 argument words) | 14 / 14 |
| DIV of all 65 536 int16 values by −1/256 (a one-node `Div` model through `run_remote_tests`), on the HLS bitstream `bbb9a37f73f8` first, then on the RTL one with its generated driver | both pass: the HLS hardware also gives 0x7FFF for −128 ÷ −1/256 (as the scheduler's simulation; only HLS C simulation gives 0x8000) |
| `run_remote_tests` (`remote_config_all_models.json`, the VectorOPKernel driver from `gen_driver.py`) | 148 / 148 |
| the 60 kernel benchmarks | all pass |
| MNIST, 10 000 images | identical (convnet 98.92 %, LeNet 97.35 %) |
| image classification (ResNet-18, MobileNet v1 / v2) | results identical to the baseline (`compare_perf.py`: 0 result changes) |
| BERT-SQuAD, 50 examples | 50 / 50 bit-exact with the emulation (3 / 3 with the scheduler simulation) |
| `llm_board` SmolLM2-135M, SmolLM2-360M | logits 4 × 33 / 33 bit-exact each; chunked prefill, threads, close → open identical |
| `llm_board` SmolVLM-256M | 2 images, 33 / 33 logits bit-exact each |
| `tts_board` Piper | PCM, text encoder (5 buckets) and duration predictor bit-exact |

**Speed.**  The 15 VectorOP benchmarks, HLS and RTL bitstream back to back
in one session (`p3/perf_hls_ab.json`, `p3/perf_rtl_ab.json`):

| case | HLS | RTL | |
|---|---:|---:|---:|
| ADD 1K / 4K / 16K / 64K / 256K | 8.9 / 17.9 / 48.5 / 171.3 / 662.9 µs | 8.6 / 17.6 / 48.5 / 171.3 / 662.9 µs | −3.4 / −1.7 / 0 / 0 / 0 % |
| MUL 16K / 64K | 49.2 / 171.7 µs | 48.6 / 171.3 µs | −1.2 / −0.2 % |
| DIV 4K | 49.5 µs | 49.3 µs | −0.4 % |
| RELU 16K / 64K, RELU6 16K | 28.3 / 90.3 / 28.3 µs | 28.0 / 89.4 / 28.1 µs | −1.1 / −1.0 / −0.7 % |
| ADD / MUL / RELU broadcast 8 × 16K | 335.3 / 335.8 / 173.8 µs | 335.1 / 335.2 / 171.7 µs | −0.1 / −0.2 / −1.2 % |
| MUL depthwise broadcast 12 544 × 16 | 261.3 µs | 258.4 µs | −1.1 % |

The binary jobs are bound by the shared HPC0 read channel (two operands,
about half a word per cycle each), so the big ones cannot move; the unary
and short jobs gain from the shorter pipeline.  (Against the recorded
baseline of `bbb9a37f73f8`, a day older, four large cases read +0.1–0.2 %
in the first run — session drift: the back-to-back A/B above shows them
equal.)

| workload | `bbb9a37f73f8` (baseline / record) | `68665fc1833a` |
|---|---:|---:|
| ResNet-18 / MobileNet v1 / v2 | 59.9 / 73.0 / 63.0 ms | 59.9 / 72.9 / 62.8 ms |
| BERT mean of 50 (p50) | 920.3 ms | 919.0 ms (917.8) |
| MNIST convnet / LeNet | 0.260 / 2.833 ms | 0.261 / 2.833 ms |
| SmolLM2-135M decode, prefill 16 / 64 / 256 | 100.2 ms; 251 ms | 100.3 ms; 250 / 443 / 1279 ms |
| SmolLM2-360M decode, prefill 16 / 64 / 256 | 254.0 ms | 251.9 ms; 568 / 1000 / 2905 ms |
| SmolVLM `llm_image`, decode | 3.91 s | 3.89–3.91 s, 99.6 ms |
| Piper RTF (6.9 s utterance) | 0.518 | 0.519 |

### Phase 3: the default and the clean-up — done (2026-10-05)

**Build.**  The top-level CMake drops `AXI_VECTOROP_IMPL` and always
packages the RTL IP for `synthesize_<platform>`, `ip_repo_kv260`
(`build_hw_kv260`, `sim_hw_kv260`) and `behavior_test_vectorop`; a cache that
still asks for `hls` is refused at configure time.  `kernels/vectorop/`
loses its synthesis target, `Synthesis.tcl.in` and the unused Vitis-flow
options (`VA_PLATFORM`, `VA_TARGET_CLOCK`, `VA_ENABLE_VITIS_FLOW`); its C++,
`TestSimulation` and `gen_vectorop_test_data` stay (the standalone build of
the directory still configures and passes).  The kernel-verify skill's
VectorOP gates are the RTL ones (`TestVectorOpRtl`, `lint_vectorop_rtl`,
`synth_vectorop_rtl`); its HLS summariser `csynth_check.py` went with the
HLS synthesis.

**Driver.**  Projects take the RTL kernel's generated driver from
`build/kernels/vectorop_rtl/driver/VectorOPKernel_v1_0/src` (`make
driver_vectorop_rtl`; the HLS driver's API and files): the example and local
configs, the scheduler's help texts, the CLI test and the docs.  Facts:
`registers.VectorOPKernel` reads the RTL control block through the driver
generator (the C++ model must have the same ports), `paths.vectorop_driver`
ties the driver directory to its mentions.

**Promotion.**  `68665fc1833a` is the production bitstream (the board's
`pl.bin`; the chat server restarted on it and answers).  The performance
model `perf_models/kv260/68665fc1833a.json` started from the converged case
list of `bbb9a37f73f8` (the scheduler had not changed): 1580 calls measured
twice (repeat spread median 0.017 %), refinement rounds of 1 and 0 calls,
1581 exact calls; mm-gemv / mm-tiled held-out p90 2.91 / 4.20 %.  Against
`bbb9a37f73f8` the Conv, Matmul and Pool calls agree to a median 0.000 %;
the 95 VectorOP calls of the shipped models are a median 1.1 % faster —
69 by more than 0.5 %, up to 38 % for jobs of many 2-word runs (28 672 runs
of 14 elements), the ones the crossbar acceptance of the first package
would have held back — and four 0.5–1.2 % slower (two DIV jobs, two short
ADDs).  Perf-regression baseline `kv260-68665fc1833a.json`: the 60 kernel
benchmarks of the A/B run and the phase-2 demos.  Host suites: scheduler
1647, chat 188, lint, facts, csim 11 / 11, Piper host emulation 18 / 18.

**Follow-up outside this plan: the RTL MatmulKernel's IP declared no m_axi
bus parameters** either, so its crossbar slots ran at 2 outstanding bursts
(the HLS kernel's: A 4, B 16, C 16) — fixed in MATMUL_RTL_PLAN phase 5.

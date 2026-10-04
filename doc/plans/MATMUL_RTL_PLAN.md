# MatmulKernel in SystemVerilog: integration plan

**Status (2026-10-04):** phases 0–3 done.  The RTL kernel, its Verilator
testbench, the C driver and the IP packaging are in `kernels/matmul_rtl/`;
with `AXI_MATMUL_IMPL=rtl` the KV260 bitstream builds (`1d28630fbfa4`,
timing met at 100 MHz, 8.7 k LUT / 17.8 k FF / 6 BRAM / 8 URAM fewer) and
is bit-exact on the board everywhere, no benchmark or workload slower than
with the HLS kernel (phase 2b balanced K between the lanes).  Phase 3: the
bitstream is calibrated, the scheduler models the RTL kernel
(`kernels.matmul.impl`), and its engine choices keep one copy of every
weight — measured on the board, the 16-token LLM prefills run 26–32 %
faster and MobileNet v1 10 %, all bit-exact.  The default is still `hls`;
phase 4 (the switch: platform field, bitstream, chat server, docs) is open.

The RTL kernel ([MATMUL_RTL_KERNEL](../kernels/MATMUL_RTL_KERNEL.md)) is a
drop-in replacement for the Vitis HLS MatmulKernel
([MATMUL_KERNEL](../kernels/MATMUL_KERNEL.md)): same VLNV, register map, DDR
layouts and bit-identical arithmetic; gmem2 (C) is 128-bit instead of 32.  It
does 128 MAC/cycle on GEMM (HLS: 32) in 18.4 k LUT / 9.5 k FF / 38 BRAM / 0
URAM / 128 DSP (HLS: 26.8 k / 27.0 k / 44 / 8 / 128).  GEMV stays port-bound
at 16 MAC/cycle, so LLM decode does not get faster; multi-row MatMuls do
(prefill, BERT, fully-connected layers).

## 1. Readiness check (2026-10-03)

Before the import the kernel was developed in a separate working directory.
Checked there:

| check | result |
|---|---|
| Verilator lint (`-Wall`) | clean |
| HLS-oracle fixtures (39 tiled + 11 GEMV) | 50 / 50 bit-exact |
| random cases, seeds 2026 and 4711, randomised and slow memory timing | 1 600 / 1 600 (AXI protocol checks and "nothing written outside C" included) |
| test-stand block design in xsim (PS VIP, interconnect, DDR model) | 50 / 50 |
| register map vs the bitstream's `xmatmulkernel_hw.h` | identical, 0x00–0x80 |
| `cormorant_hw_128` wiring | gmem0 → HPC0, gmem1 → HPC1, both reach all of DDR (both RTL read ports fetch A and B); gmem2 is 32-bit on the instance and must become 128 |
| OOC synthesis at 300 MHz | Fmax ≈ 297 MHz (3 endpoints miss by ≤ 31 ps); the PL clock is 100 MHz |

Risks carried into the phases:

1. Never in a bitstream or on the board: the PS DDR controller, HPC ports
   shared with the other kernels, the XRT / UIO runtime.
2. The KV260 PS drops beats of single-beat partial-strobe writes
   (2026-09-24); a short or misaligned C run (small m, unaligned C) can be
   exactly such a beat — a targeted board test in phase 2.
3. The speed figures are from ideal memory.  HPC0 is shared with ConvKernel,
   VectorOP and Pool, and the RTL reads B through both ports (the HLS tiled
   path mostly through HPC1).
4. The cost model and the `--plan` performance models were calibrated on the
   HLS kernel; a 4× faster MatmulKernel changes the scheduler's engine
   choices, which stay poor until phase 3.
5. Over-reads: the RTL may read the words around a run — harmless with CMA
   buffers, but it would matter on the unmerged SMMU branch if an over-read
   crossed a mapped page.

## 2. Decisions (2026-10-03)

- The RTL lives in this repository (`kernels/matmul_rtl/`), not in a submodule.
- The HLS MatmulKernel's synthesis is retired after phase 2; its C++
  reference (`ref_matmul_2d`, `TestMatmulRef`) stays as the oracle and the
  fixture generator.
- No 150 MHz attempt in this iteration.
- The chat server may be down for phase 2.

## 3. Phases

### Phase 0: into the repository — done (2026-10-04)

- `kernels/matmul_rtl/`: `rtl/` (17 files), `tb/verilator/`, `syn/` (Vivado
  scripts), `scripts/` (lint waivers, `gen_top_wrapper.py`, `vcdq.py`,
  `gen_driver.py`), `CMakeLists.txt`; no build outputs.  The working
  directory's Makefile became CMake targets: `TestMatmulRtl`,
  `lint_matmul_rtl`, `perf_matmul_rtl`, `matmul_rtl_tb_fst` /
  `matmul_rtl_tb_vcd`, `driver_matmul_rtl`, `package_matmul_rtl`,
  `synth_matmul_rtl`, `xsim_matmul_rtl`, `sysim_matmul_rtl`.  All paths are
  repository-relative; the part comes from the platform JSON.
- ctest: `TestMatmulRtl` (the 50 checked-in fixtures + 200 random cases,
  47 s) and `MatmulRtlDriver` (the driver's register table vs the RTL); both
  in the `run_tests` target and the run-tests agent's `csim` suite.
- **The C driver is generated, not copied.**  The plan was to ship the
  HLS-generated `xmatmulkernel` sources with the RTL; they carry an AMD "All
  Rights Reserved" header, and the repository tracks no AMD-generated file.
  `scripts/gen_driver.py` writes a driver with the same API from a register
  table instead (into the build tree; `package_matmul_rtl` puts it in the IP).
  Checked against the HLS driver of the current bitstream: same `#define`s
  (offsets, widths) and prototypes (`--check --hls-driver`), and the same
  register traffic for every call (a host program compiled against each
  driver over a mock register file printed identical register dumps).  It
  compiles with `-Wall -Wextra -Werror` for x86-64 and aarch64.
- `AXI_MATMUL_IMPL=hls|rtl` (default `hls`): with `rtl`, `synthesize_kv260`
  depends on `package_matmul_rtl` instead of `synthesize_matmul_kv260`, and
  `build_hw_kv260` / `sim_hw_kv260` pass `-ip-repo build/ip_repo_kv260`, a
  directory of symlinks to the three HLS exports and the RTL IP (Vivado
  follows them; checked with `update_ip_catalog`).  The RTL IP is packaged in
  `build/rtl_ip/`, outside `build/kernels/`, so the `hls` build, which scans
  `build/kernels/`, never sees two IPs with the same VLNV.
- The `registers.MatmulKernel` fact also runs `gen_driver.py --check` and
  compares the driver's register names with the HLS kernel's ports.
- Checked in a fresh clone with the changes applied (`cmake ..`,
  `make TestMatmulRtl package_matmul_rtl perf_matmul_rtl`, `ctest -R MatmulRtl`:
  2 / 2, no submodules needed) and in a fresh build directory of the working
  tree (also `lint_matmul_rtl`, `xsim_matmul_rtl`, `synth_matmul_rtl`,
  `sysim_matmul_rtl`); the `csim` suite of run-tests: 9 / 9.

| target | result |
|---|---|
| `TestMatmulRtl` build / ctest | 28 s / 250 passed (50 fixtures + 200 random), 47 s |
| `package_matmul_rtl` | `xilinx.com:hls:MatmulKernel:1.0`, 17 s |
| `perf_matmul_rtl` | 9 shapes, the figures of the kernel reference (GEMV 15.4–15.9, GEMM 98–123 MAC/cycle), 6 s |
| `xsim_matmul_rtl` | snapshot built, 7 s |
| `synth_matmul_rtl` | 18 433 LUT (6 896 LUTRAM), 9 509 FF, 38 BRAM36, 0 URAM, 128 DSP; WNS −0.031 ns at 3.333 ns (Fmax ≈ 297 MHz), as before the import; 13 min |
| `sysim_matmul_rtl` | 50 / 50 (`check_test_report.py` PASS), 4 min; the upgrade changed only gmem2 (32 → 128 bits, WSTRB 4 → 16) and the control address (7 → 8 bits: the test stand's block design predates `a_to_b`) |

### Phase 1: block design and bitstream — done (2026-10-04)

- **Instance widths follow the IP.**  Instead of a hand edit (or a width
  passed per build), the `cormorant_hw_128` scripts (`build.tcl`, `sim.tcl`,
  new `scripts/ip_defaults.tcl`) and the test stand (`ts_prepare_bd`,
  `ts_apply_ip_default_widths`) put every kernel instance's
  `C_M_AXI_*_DATA_WIDTH` back to the default of the IP in the catalogue after
  the IP upgrade — the documented rule "instance widths = IP defaults",
  now enforced.  Vivado has no reset to default (`reset_property` refuses
  `CONFIG.*`; `VALUE_SRC DEFAULT` changes the flag, not the value), so the
  defaults come from a temporary instance of each IP.  HLS → RTL: the upgrade
  already gives gmem2 128 (the RTL IP's parameters are read-only) and nothing
  is reset; RTL → HLS: the upgrade keeps 128 and the reset puts it back to
  32.  Both directions checked on a copy of `cormorant_hw_128` and in the
  test stand itself.
- `AXI_MATMUL_IMPL=rtl` also points `behavior_test_matmul` at the RTL IP
  (`build/rtl_ip/MatmulKernel_ip`, depending on `package_matmul_rtl`), and
  `ip_repo_kv260` depends on `package_matmul_rtl`.
- The bitstream: `build_hw128` configured with `-DAXI_MATMUL_IMPL=rtl`, then
  `make package_matmul_rtl ip_repo_kv260` and `build.sh all -ip-repo
  build_hw128/ip_repo_kv260` — the command `build_hw_kv260` runs, without
  its `synthesize_kv260` step: the VectorOP, Conv and Pool exports in
  `build_hw128` are the ones of the current bitstream (their sources have
  not changed since).  52 min (synthesis 14, implementation 36).  The
  upgrade changed only `MatmulKernel_0` (gmem2 32 → 128 bits, WSTRB 4 → 16)
  and removed the interconnect's gmem2 upsizer (`auto_us_0`).

| | HLS MatmulKernel (`caa67f49a5a3`, deployed) | RTL MatmulKernel (`8b9aee0f54b3`) | change |
|---|---:|---:|---:|
| LUT (placed) | 93 303 (79.7 %) | 84 699 (72.3 %) | −8 604 |
| LUT as memory | — | 20 004 | |
| FF | 97 321 | 79 510 | −17 811 |
| BRAM36 | 115.5 | 109.5 | −6 |
| URAM | 56 | 48 | −8 |
| DSP | 1 058 | 1 058 | 0 |
| WNS / WHS at 100 MHz | +0.671 / +0.010 ns | +1.204 / +0.010 ns | |

| simulation | result |
|---|---|
| `behavior_test_matmul` (test stand, RTL IP) | 50 / 50, 4.6 min |
| the same with the HLS IP again, on the RTL-upgraded test-stand project (the way back) | 50 / 50 (gmem2 reset 128 → 32) |
| `sim_hw_kv260` (the whole block design with the RTL IP) | 68 / 68 (VectorOP 22, Conv 17, Matmul 10, Pool 19), 3.9 min |

- The outputs of both bitstreams are kept outside the repository:
  `/mnt/data/bitstreams/kv260_hls_caa67f49a5a3/` (the deployed one) and
  `/mnt/data/bitstreams/kv260_rtl_8b9aee0f54b3/` (`.bit`, `.hwh`, synthesis
  and implementation reports, build log).  `cormorant_hw_128.runs/impl_1`
  — the path `bitstream_config_kv260.json` names — now holds the RTL one, so
  `facts.py verify hw.utilization` reports the RTL numbers until phase 4
  updates the README; to go back, copy the HLS `.bit` there (or rebuild with
  `AXI_MATMUL_IMPL=hls`).
- After the runs the submodules' tracked project files (`.bd`, `.xci`,
  `.xpr`) were restored; the committed block designs still describe the HLS
  configuration, and every build re-derives the RTL one.  **Restoring them
  alone was a mistake** (found in phase 2b): Vivado's untracked outputs
  (`*.gen`, `*.ip_user_files`, `*.cache`) still described the RTL build, and
  the next build found the PS and the interconnect's internal IPs locked and
  could not generate the block design.  Either leave the tracked files as
  the build left them (the documented convention: do not commit them), or
  restore them AND move those three directories aside, which gives the state
  of a clean checkout that `build.tcl` regenerates from.

### Phase 2: on the board — correct everywhere; one GEMV shape slower (2026-10-04)

The chat server was stopped (`deploy.py --stop`), the RTL bitstream
`8b9aee0f54b3` loaded with `upload_bitstream.py` (HPC0 / HPC1 widths 128,
UIO `fabric_vecop` / `fabric_matmul` / `fabric_conv` / `fabric_pool`), and at
the end the deployed bitstream `caa67f49a5a3` reloaded and the server
restarted (`deploy.py`; a chat request answered).  Every board job ran under
the board lock.  The chat-model and TTS gates built and installed into a
scratch directory (`--remote-dir /root/rtl_gate`, weights copied there),
removed afterwards: the server's libraries and weight directories were not
touched.

**Correctness — everything bit-exact:**

| check | result |
|---|---|
| MatmulKernel registers: write and read back (GIE, IER, the 17 argument words) | 19 / 19 |
| `run_remote_tests` (`remote_config_all_models.json`, the MatmulKernel driver from `gen_driver.py`) | 148 / 148 |
| targeted C writes: C of 5 / 2 / 12 elements (one partial beat), rows of 7, m = 515 (per-row runs, 3-element tails, packed and row-major B), batch slices of 15 / 18 elements at unaligned starts | 8 / 8 |
| image classification (ResNet-18, MobileNet v1 / v2) | top-5 classes and logits identical to the HLS run |
| MNIST, 10 000 images | identical results (convnet 98.92 %, LeNet 97.35 %) |
| BERT-SQuAD, 50 examples | 50 / 50 bit-exact with the emulation (3 / 3 with the scheduler simulation), EM / F1 88.0 / 90.3 = float |
| `llm_board` SmolLM2-135M, SmolLM2-360M | logits 4 × 33 / 33 bit-exact each; chunked prefill, threads, close → open identical |
| `llm_board` SmolVLM-256M | 2 images, 33 / 33 logits bit-exact each |
| `tts_board` Piper | PCM, text encoder and duration predictor bit-exact |
| SmolLM2-135M decode A/B, the same library on both bitstreams | decode checksums identical (FNV 2862720827 / 1879811210 / 2664026437) |

The KV260 PS keeps the RTL kernel's partial-strobe beats (the risk from
2026-09-24 concerned the HLS adapter's single-beat writes): the targeted
cases pass.

**Speed** (`run_remote_perf.py`, compared with the HLS bitstream's
perf-regression baseline; VectorOP, Conv and Pool within ±1.7 %):

| MatmulKernel case | HLS | RTL | |
|---|---:|---:|---:|
| 256×256×256 (row-major / packed) | 7.39 / 7.17 ms | 1.40 / 1.40 ms | 5.3× / 5.1× |
| 64×64×64, batch 4 × 64×64×64 | 0.225, 0.875 ms | 0.033, 0.112 ms | 6.8×, 7.8× |
| depthwise-as-MatMul 12544×16×1 / ×3 | 6.94 / 20.8 ms | 0.99 / 2.97 ms | 7.0× |
| FC 1×256×256 … 1×1280×1001, tiled (row-major / packed) | 0.10–2.44 ms | 0.048–0.95 ms | 2.1–2.9× |
| GEMV 1×512×1000, kw 1 | 0.339 ms | 0.340 ms | +0.5 % |
| GEMV 1×1536×576, kw 4 | 0.568 ms | 0.565 ms | −0.5 % |
| **GEMV 1×576×1536, kw 4** | **0.577 ms** | **0.631 ms** | **+9.2 %** |

| workload | HLS | RTL |
|---|---:|---:|
| SmolLM2-135M decode at 32 / 256 / 1000 (A/B, same library) | 99.3 / 106.4 / 128.6 ms | 103.6 / 110.5 / 134.6 ms (+4 %) |
| SmolLM2-135M prefill 16 / 64 / 256 | 341 / 442 / 1294 ms | 340 / 442 / 1291 ms |
| SmolLM2-360M decode at 32 / 256 / 1000 (HLS: §20 record) | 256 / 270 / 306 ms | 258 / 271 / 312 ms |
| SmolLM2-360M prefill 16 / 64 / 256 | 0.83 / 1.00 / 2.90 s | 0.83 / 1.00 / 2.92 s |
| SmolVLM `llm_image` | 3.885 s | 3.92–3.94 s |
| BERT p50 | 962.3 ms | 964.9 ms |
| ResNet-18 / MobileNet v1 / v2 | 59.9 / 81.0 / 62.9 ms | 59.9 / 81.2 / 63.0 ms |
| LeNet | 2.810 ms | 2.804 ms |
| Piper RTF (6.9 s utterance) | 0.52 | 0.523 |

- **The regression:** first put down to the drain between column chunks;
  phase 2b found the real cause — K-block imbalance.  K = 576 with kw = 4 is
  144 planes = 9 blocks of 16, so read port 0 streamed 5 blocks and port 1
  4: the job takes 5 / 4.5 = 1.11× the balanced time (Verilator with ideal
  memory reproduced it: 62 467 cycles, the board 63 k).  1×1536×576 (24
  blocks) and every kw = 1 shape of the benchmarks (even block counts) were
  unaffected.  It is exactly SmolLM2-135M's k = 576 projections in decode.
- **No end-to-end gain yet:** today's scheduler, calibrated on the HLS
  kernel, sends almost every multi-row MatMul to ConvKernel; the 2–7× of the
  RTL kernel reaches the models only after phase 3 re-prices MatmulKernel.
- **Done criterion not met** ("no case regresses") — fixed in phase 2b.
- Outputs: `/mnt/data/bitstreams/kv260_rtl_8b9aee0f54b3/` (`perf_rtl.json`,
  `perf_rtl.log`); logs of every run in `/mnt/data/tmp/p2_*.log`.

### Phase 2b: K balance between the lanes — fixed (2026-10-04)

**Cause.**  K is split between the two lanes / read ports in blocks of 16
planes, block b to lane b % 2.  With an odd number of blocks one port
streams a whole block more: 9 blocks (k = 576 with kw = 4, k = 144, …) are
5 : 4, so the job takes 5 / 4.5 = 1.11× the balanced time.  Verilator with
ideal memory showed it (1×576×1536 kw 4: 62 467 cycles, 14.2 MAC/cycle,
against 15.8 for 1×1536×576), so it was the RTL, not DDR.

**Fix** (`mm_pkg`, `mm_core`, `mm_awr`, `mm_rungen`; `cfg.split`): with an
odd number of blocks whose last one has more than 8 planes, that block is
split — planes 0–7 to lane 0, planes 8– to lane 1, after its own last block
(or alone when there is one block).  In an A row the halves are whole beats
(beat 2j + h of the block = planes 8h … 8h + 7 of tap j), so the A writer
routes them by beat parity; both lanes store their half at their next
lane-local block, where the x prefetcher's K-index formula already reads.
The run generator's three layouts (packed 16×32 tiles, contiguous image,
per-plane image) emit the halves with the right address, lane-local plane,
rows, element count and accumulator-init flag.  The block constants
(nblk − 1, nblk − 2, the 24-plane byte offset) are computed once per job in
the config, which keeps Fmax: the first version missed 300 MHz by 0.26 ns
(278.5 MHz, a subtract-compare-add chain into lane 1's address adder); the
final one meets it (WNS +0.002 ns, 18 395 LUT, 9 482 FF, 38 BRAM).

| check | result |
|---|---|
| Verilator: fixtures / random (seeds 11, 2026 rand timing; 4711 slow) / directed split cases (nblk = 1, partial and full last blocks, every layout, batches) | 50 / 50, 1 800 / 1 800, 40 / 40 |
| Verilator cycles, before → after | 1×576×1536 kw 4: 62 473 → 56 324 (15.7 MAC/cycle); packed 64×144×576: 51 078 → 46 472; other shapes unchanged |
| `sysim_matmul_rtl` | 50 / 50 (`sysim.tcl` now upgrades every locked IP, as the test stand does — the SmartConnect of the copied project was locked) |
| `behavior_test_matmul`, `sim_hw_kv260` | 50 / 50, 68 / 68 |
| bitstream `1d28630fbfa4` | timing met at 100 MHz (WNS +0.987 ns); 84 603 LUT, 79 544 FF, 109.5 BRAM36, 48 URAM, 1 058 DSP |

**Board** (`1d28630fbfa4`; the chat server stopped, the deployed bitstream
and the server restored afterwards; gates in the scratch directory again):

| check | result |
|---|---|
| registers (write / read back) | 19 / 19 |
| `run_remote_tests` (148 models, generated driver), targeted C writes | 148 / 148, 8 / 8 |
| image classification, MNIST, BERT | identical to the HLS run (top-5 logits; accuracies; BERT 50 / 50 bit-exact, EM / F1 = float) |
| SmolLM2-135M, SmolVLM, Piper gates | bit-exact (135M decode checksums identical to the HLS A/B) |
| SmolLM2-360M gate | logits 4 × 33 / 33 bit-exact; decode checksums identical on both bitstreams (A/B below) |
| kernel benchmarks vs the HLS baseline | 0 regressions, 21 improved (a first run flagged VectorOP ADD-4K +1.1 µs; the re-run: +0.3 µs, jitter) |

| | HLS | RTL before | RTL fixed |
|---|---:|---:|---:|
| GEMV 1×576×1536, kw 4 | 0.577 ms | 0.631 ms | **0.569 ms** |
| depthwise-as-MatMul 12544×16×1 / ×3 (k = 16: one split block) | 6.94 / 20.8 ms | 0.99 / 2.97 ms | **0.62 / 1.84 ms** |
| SmolLM2-135M decode at 32 / 256 / 1000 | 99.3 / 106.4 / 128.6 ms | 103.6 / 110.5 / 134.6 ms | **98.0 / 104.4 / 127.1 ms** |
| SmolLM2-135M prefill 16 / 64 / 256 | 341 / 442 / 1294 ms | 340 / 442 / 1291 ms | 340 / 444 / 1285 ms |
| SmolLM2-360M decode at 32 / 256 / 1000 (A/B, each the first open after a reboot) | 250.4 / 265.6 / 316.3 ms | — | **248.4 / 263.2 / 313.9 ms** |
| SmolLM2-360M prefill 16 / 64 / 256 (the same A/B) | 839 / 1035 / 3087 ms | — | 839 / 1041 / 3092 ms |
| SmolVLM `llm_image` | 3.885 s | 3.92–3.94 s | 3.94–3.95 s |
| BERT p50 | 962.3 ms | 964.9 ms | 963.3 ms |
| Piper RTF (6.9 s utterance) | 0.52 | 0.523 | 0.519 |

(Phase 2's 360M row compared the pre-fix RTL with the §20 record of an
older build: 256 / 270 / 306 ms decode, 0.83 / 1.00 / 2.90 s prefill.  The
same-condition A/B above replaces it; today's project prefills 6 % slower
than that record on both bitstreams.)

**A board observation, not the kernel:** after a day of board jobs the
360M library's 776 MB pool (one contiguous CMA buffer) could not be
allocated although CmaFree was 1010 MB and zocl held no buffer — on the
HLS bitstream too.  After a reboot the first open in a process succeeded,
the following ones failed again: the CMA region fragments (14–24 MB of it
stays in use by other drivers).  The chat server allocates its pools at
start, so it was restarted after a reboot; whether its model swaps
(`--resident auto`) can hit the same limit was not tested.

**Phase 2's criterion is met**: bit-exact everywhere, no benchmark or
workload slower than with the HLS kernel.  Outputs:
`/mnt/data/bitstreams/kv260_rtl_1d28630fbfa4/` (bitstream, reports, build
log, `perf_rtl2*.json`); logs in `/mnt/data/tmp/p2b_*.log`.

### Phase 3: performance models and scheduling — done (2026-10-04)

**3a. Calibration campaign** on the RTL bitstream `1d28630fbfa4`
(`perf_calibrate.py cases`, `run`, `fit`; 2026-10-04): 1 249 calls (754 from
the ten shipped models with today's engine choices, 569 from the grid), both
passes complete, repeat spread median 0.022 %.  Simulator check against the
phase 2b board measurements: every workload within ±2 % (CNNs, LeNet, BERT
−0.7 %, SmolLM2-135M and SmolVLM prefill / decode / `llm_image`, the Piper
encoder) except SmolLM2-360M (−2.6 … −6.0 %) and the Piper chunk (−2.3 %),
which had never been checked: the HLS model misses 360M by the same amounts
on the HLS bitstream, so it is the host-op model, not the kernel.  The
refinement rounds wait for 3b: they rank tactics with the fitted model,
around the shipped choices, and 3b changes both.

**3b. The MatmulKernel models.**
- `platforms/kv260.json` `kernels.matmul.impl` (`hls` / `rtl`) names the
  bitstream's MatmulKernel.  The scheduler's cost model follows it (the
  `AXI_MATMUL_IMPL` environment variable overrides it), and CMake's
  `AXI_MATMUL_IMPL` takes its default from it.
- `cost_model.rtl_matmul_cycles` is a structural model of the RTL job walk
  (the B stream of the busier lane, the drain, the A-panel load, steps,
  read runs), fitted to the 240 measured MatmulKernel calls: median error
  1.3 %, p90 3.3 %.
- The perf models' MatmulKernel families carry those terms (`rtl_*`)
  besides the HLS tile model's, and the fit keeps the set with the lower
  training error.  On the RTL bitstream the tiled family's held-out p90
  drops from 49.9 % to 4.0 % (median 0.8 %), under the planner's 5 % trust
  limit; the HLS bitstream's model refits identically.
- **Finding: the engine choices cannot just follow the new model.**  With
  `impl: rtl`, the RTL model moves 31–450 MatMuls per chat / TTS model from
  ConvKernel or the GEMV path to MatmulKernel's packed tiled path, and the
  multi-entry projects then need a second layout of those weights: the
  pool of SmolLM2-135M grows from 286 to 488 MiB, SmolVLM from 495 to 697,
  Piper from 48 to 55 (360M would exceed the 1000 MB CMA).  On the RTL
  kernel the ConvKernel image layout (`gemv_kw`) is not a one-row path:
  any number of A rows streams B once per 8-row panel, within 1–1.3 % of
  the packed layout's speed (phase 2b benchmarks).  The RTL policy is
  therefore: MatmulKernel reads constant weights in the image layout, for
  any n, unless the shape forbids it (k % 8, m % 8, m · kw ≥ 64), and each
  MatMul's engine choice compares ConvKernel and MatmulKernel on the same
  image, so every weight keeps one copy across the entries.

**3b, the policy** (all of it only with `impl: rtl`; the HLS path is
unchanged, byte for byte):
- `matmul_gemv`: the image ("GEMV") path is eligible for any row count, and
  a weight pinned by another entry is always read in its pinned layout.
- `matmul_lowering.shared_weight_layouts`: the first prefill bucket pins
  every weight's layout — its ConvKernel width, the image width a MatMul
  kept on MatmulKernel reads, or 0 for the tiled path's layout — and both
  entry builders (`llm_entries.entry_graphs`, Piper's
  `generate_tts_project.py`) pass it to the other entries, where
  `OnnxGraph` uses it for the MatmulKernel reads as well.
- `lower_matmuls` prices MatmulKernel on the pinned layout, and a width only
  ConvKernel reads (Piper's encoder: kw 6) keeps the MatMul on ConvKernel.
- Result: every pool unchanged (SmolLM2-135M 285.8, 360M 740.2, SmolVLM
  494.8, Piper 48.1 MiB; no weight in a second layout); the 16-token
  prefill buckets run their linears on MatmulKernel (at 16 rows ConvKernel
  is bound by streaming the weights: 217k cycles measured for a 576×1536
  linear, MatmulKernel 124k estimated), the larger buckets stay on
  ConvKernel; MobileNet v1's last fully-connected Conv becomes a MatMul;
  the LM head is stored packed (it is not shared, so the pool is the same).
- A top-up campaign measured the new shipped calls and one refinement
  round (134 calls; 1 383 exact calls, the GEMV family's held-out p90
  3.2 %).  Predicted with every call priced: SmolLM2-135M / SmolVLM
  16-token prefill 340.5 → 250.6 ms (−26 %), SmolLM2-360M 788.6 → 528.1 ms
  (−33 %), MobileNet v1 81.4 → 73.3 ms (−10 %); everything else unchanged.
- Tests: `test_matmul_gemv.TestRtlImagePolicy` (the rules), and
  `test_llama.test_multi_entry_project_rtl_matmul`: a four-entry tiny
  project with the RTL model keeps one copy of every weight, its 8-row
  bucket reads the ConvKernel image on MatmulKernel, and the generated C on
  the software kernels equals the simulation.
- Not done: the planner (`--plan`) has no MatmulKernel-image tactic for
  several rows yet (`tactics.py` offers GEMV for one row only), so a planned
  project keeps a pinned weight on ConvKernel.

**3c. On the board** (2026-10-04; the RTL bitstream after a reboot; the
projects generated with `AXI_MATMUL_IMPL=rtl` into a scratch directory, the
gates in `/root/rtl_gate` with their own weight directories — the LM head
is stored packed there, so the deployed weight files differ in that one
file; production restored after a reboot):

| workload | today's choices (RTL bitstream) | RTL choices | change |
|---|---:|---:|---:|
| SmolLM2-135M prefill 16 | 340 ms | **251 ms** (predicted 250.6) | −26 % |
| SmolVLM-256M prefill 16 | 339 ms | **250 ms** | −26 % |
| SmolLM2-360M prefill 16 | 839 ms | **568 ms** (predicted 528; the host model misses 360M by 6 %) | −32 % |
| MobileNet v1 | 81.0 ms | **73.0 ms** (predicted 73.3) | −10 % |
| 135M / SmolVLM prefill 64, 256; decode | 443 / 1281 ms; 97.9 ms at 32 | unchanged | — |
| 360M prefill 64 / 256; decode at 32 | 1035 / 3087 ms; 248.4 ms | 999 / 2898 ms; 247.7 ms (same kernel choices: run-to-run) | — |
| ResNet-18, MobileNet v2 | 59.9, 63.0 ms | 59.9, 63.0 ms | — |

Every gate bit-exact: the three chat models' logits (4 × 33 / 33 and the
SmolVLM images), decode checksums identical to the HLS runs, the top-5
classes and logits of the CNNs identical.  Pools unchanged.  BERT, MNIST
and Piper keep today's choices (nothing to gain: their MatMuls stay on
ConvKernel or already run on MatmulKernel).  Outputs:
`/mnt/data/bitstreams/kv260_rtl_1d28630fbfa4/phase3c/`.

**Phase 3 status:** done for the unplanned (default) path; the planner's
multi-row image tactic is open, and the board still runs the HLS bitstream —
phase 4 switches `kernels.matmul.impl` and the bitstream together.

### Phase 4: the default and the clean-up

- `AXI_MATMUL_IMPL=rtl` by default; remove the HLS synthesis targets of
  MatmulKernel (keep its C++ reference).
- Driver paths: the example configs (`perf_config`, `remote_config`, the
  BERT / image-classification / MNIST demo configs), `src/kernels.py`, the
  `registers.MatmulKernel` fact's `driver` glob and REMOTE_TESTING /
  USER_GUIDE name the HLS export's `drivers/MatmulKernel_v1_0/src`; they move
  to `build/rtl_ip/MatmulKernel_ip/drivers/MatmulKernel_v1_0/src`.  (Until
  then either driver works with either bitstream: same API, same register
  traffic.)
- `platforms/kv260.json`: `tile_m` 32 stays (the packed-B DDR layout); decide
  what `tile_n`, `tile_k` and `gemv_max_m` mean for the RTL kernel.
- Facts: `registers.MatmulKernel` reads the RTL control block instead of the
  HLS pragmas; `platform.kv260_bounds`; the architecture-diagram mentions.
- Docs: `MATMUL_KERNEL.md`, the README kernel table, results and diagram,
  both CLAUDE.md files, the kernel-verify and perf-calibrate skills.

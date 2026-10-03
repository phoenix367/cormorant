# MatmulKernel in SystemVerilog: integration plan

**Status (2026-10-04):** phases 0 and 1 done — the RTL kernel, its
Verilator testbench, the C driver and the IP packaging are in
`kernels/matmul_rtl/` and build from a fresh clone; with
`AXI_MATMUL_IMPL=rtl` the KV260 bitstream builds (`8b9aee0f54b3`, timing met
at 100 MHz, 8.6 k LUT / 17.8 k FF / 6 BRAM / 8 URAM fewer) and passes the
matmul behaviour test (50 / 50) and the whole-design simulation (68 / 68).
The default is still `hls`; phases 2–4 (board, performance models, switch)
are open.

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
  configuration, and every build re-derives the RTL one.

### Phase 2: on the board (the chat server may be stopped)

- Upload the bitstream; write and read back every register.
- `run_remote_tests` (148 / 148), plus targeted C-write cases: m < 8,
  unaligned C rows, single-beat runs.
- Demo gates, all bit-exact: BERT (EM / F1 unchanged), the CNNs,
  `llm_board` for SmolLM2-135M / 360M and SmolVLM, `tts_board` for Piper.
- The MatmulKernel benchmarks against the HLS kernel's figures.
- *Done when* every gate is bit-exact and no case regresses; then the HLS
  kernel's synthesis is retired.

### Phase 3: performance models and scheduling

- A `perf_calibrate` campaign for the new bitstream (cases, run, fit,
  refinement rounds), then the simulator check (every workload within ±2 %).
- Refit the MatmulKernel constants of `cost_model.py`; re-evaluate the
  automatic engine choices (`--matmul-on-conv`, `--matmul-gemv`, `--fc-conv`)
  and the planner's ConvKernel ‖ MatmulKernel overlap.
- A new perf-regression baseline; end-to-end before / after: BERT, LLM
  prefill and decode, LeNet, the CNNs.

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

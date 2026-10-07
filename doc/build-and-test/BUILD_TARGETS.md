# Build Targets

Reference for every `make` target produced by the top-level CMake project.

## Setup

The project requires an **out-of-source** build (an in-source `cmake` aborts
with a `FATAL_ERROR`):

```bash
mkdir build && cd build
cmake ..
```

Each kernel also configures on its own, with the same target names
(`cmake -S kernels/<k> -B <dir>` for `vectorop`, `conv`, `pool`, `matmul`,
`matmul_rtl`, `vectorop_rtl`, `pool_rtl`, `conv_rtl`): `cmake/AxiPlatform.cmake`, which the top level includes too,
sets `AXI_BUS_WIDTH`, `AXI_PLATFORM` and the platform list.

Key configure-time cache variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `AXI_BUS_WIDTH` | 32 | Informational since the last HLS synthesis (ConvKernel's) was retired: it was the `-m_axi_max_widen_bitwidth` cap (32/64/128/256/512) of HLS auto-widening.  Every kernel IP is SystemVerilog with fixed 128-bit data ports, so it affects no IP — only the ConvKernel C++ model's `kAxiBusWidth` |
| `AXI_PLATFORM` | `kv260` | Platform whose `platforms/<name>.json` bounds drive the C-sim `Config.h` |
| `VA_DATA_TYPE`, `CONV_DATA_TYPE` / `CONV_ACC_DATA_TYPE`, `MM_DATA_TYPE` / `MM_ACC_DATA_TYPE`, `POOL_DATA_TYPE` / `POOL_ACC_DATA_TYPE` | `ap_fixed<16,8>` / `ap_fixed<32,16>` | Element / accumulator types. Tile sizes and bounds are not cache variables — they come from the platform JSON ([PLATFORM_CONFIGURATION.md](PLATFORM_CONFIGURATION.md)) |

IP-packaging / synthesis / hardware targets require **Vivado 2025.2** — source
`<Xilinx>/2025.2/Vitis/settings64.sh` (it puts `vivado` and `xclbinutil` on
`PATH`) before invoking them. The C-simulation targets need only
the Vitis HLS headers (`ap_fixed.h`, `ap_int.h`, `hls_burst_maxi.h`, found
via `$XILINX_HLS` / `$XILINX_VITIS`, else the built-in fallback
`/mnt/data/xilinx/2025.2/Vitis/include`, the maintainer's install):
every kernel header includes them unconditionally, so the CMake `float`
fallback taken when they are missing no longer compiles.

---

## Kernel libraries

Static libraries — intermediate build products, linked by the test
targets. Rarely built directly.

| Target | Description |
|--------|-------------|
| `vadd_kernel` | VectorOPKernel's C++ model (the retired HLS kernel; the RTL kernel's reference) |
| `conv_kernel` | ConvKernel's C++ model (the retired HLS kernel; the RTL kernel's reference) — configured type (`ap_fixed<16,8>` by default) |
| `matmul_kernel` | MatmulKernel's C++ model (the retired HLS kernel; the RTL kernel's reference) — configured type (`ap_fixed<16,8>` by default) |
| `pool_kernel` / `pool_kernel_float` | PoolingKernel's C++ model (the retired HLS kernel; the RTL kernel's reference) — configured type / forced-`float` builds |

`pool_kernel_float` forces `Data_t=float` regardless of HLS availability;
no test target links it.  ConvKernel and MatmulKernel have no float
variant: their 128-bit ports pack 16-bit elements, so the kernels only
build with a 16-bit `Data_t` (`TestMatmulBlas` drives the configured
`matmul_kernel`).

---

## C-simulation tests

Compile and run with GCC against the Vitis HLS headers — no Vitis tools,
no hardware.

| Target | Kernel | Description |
|--------|--------|-------------|
| `TestSimulation` | VectorOPKernel (C++ model) | All 6 ops across sizes + saturation cases + broadcast / stride-0 / `act` geometry cases |
| `TestConvRef` | ConvKernel (C++ model) | Model vs naive reference oracle; also registered as the CTest test `TestConvSweep` (`TestConvRef --sweep 300`, randomised geometries, ~1 min) |
| `TestConvGrid` | ConvKernel (C++ model) | MAC-grid unit test on `include/ConvMacGrid.h` alone |
| `TestMatmulRef` | MatmulKernel (C++ model) | Model vs the `ref_matmul_2d` / `ref_matmul_batch` oracle, all shape cases; `--dump-data` writes the RTL fixtures |
| `TestMatmulBlas` | MatmulKernel | configured kernel vs `cblas_sgemm`, bit-exact on 2^-8-grid inputs — only if BLAS is found |
| `TestPoolingSim` | PoolingKernel (C++ model) | Max/Average/Lp pooling + global variants; `--dump-data` writes the RTL fixtures |
| `TestMatmulRtl` | MatmulKernel (RTL) | Verilator testbench of the SystemVerilog kernel; the CTest test runs the 50 checked-in fixtures + 200 random cases (~50 s) — only if Verilator 5.x is found ([below](#rtl-matmulkernel-systemverilog)) |
| `TestVectorOpRtl` | VectorOPKernel (RTL) | Verilator testbench of the SystemVerilog kernel; CTest runs it on the 201 checked-in fixtures, 300 random jobs and 8 every-input activation jobs checked against the C++ model — only if Verilator 5.x is found ([below](#rtl-vectoropkernel-systemverilog)) |
| `TestPoolRtl` | PoolingKernel (RTL) | Verilator testbench of the SystemVerilog kernel; CTest runs it on the 45 checked-in fixtures and 200 random jobs checked against the C++ model — only if Verilator 5.x is found ([below](#rtl-poolingkernel-systemverilog)) |
| `TestConvRtl` | ConvKernel (RTL) | Verilator testbench of the SystemVerilog kernel; CTest runs it on the 63 checked-in fixtures and 200 random jobs checked against the C++ model — only if Verilator 5.x is found ([below](#rtl-convkernel-systemverilog)) |

| Aggregate | Description |
|-----------|-------------|
| `run_tests` | Builds `TestSimulation`, `TestConvRef`, `TestMatmulRef`, `TestPoolingSim` (and `TestMatmulBlas`, `TestMatmulRtl`, `TestVectorOpRtl`, `TestPoolRtl`, `TestConvRtl`, `conv_rtl_dsp_check` when present), then runs `ctest --output-on-failure` |
| `test` | Runs the CTest tests without rebuilding (equivalent to `ctest`) |

CTest registers seventeen tests: `TestSimulation`, `TestConvRef`,
`TestConvGrid`, `TestConvSweep`, `TestMatmulRef`, `TestMatmulBlas` (only
with BLAS), `TestPoolingSim`, `MatmulRtlDriver`, `TestMatmulRtl` (only
with Verilator), `VectorOpRtlDriver`, `VectorOpRtlActRom` (the generated
activation table is current), `TestVectorOpRtl` (only with
Verilator), `PoolRtlDriver`, `TestPoolRtl` (only with Verilator),
`ConvRtlDriver`, `TestConvRtl` (only with Verilator) and `ConvRtlDsp` (only
with Verilator and Vivado's `DSP48E2.v` unisim model). `run_tests` does not list `TestConvGrid`
as a dependency — build it with `make` / `make TestConvGrid` first.

---

## IP export

Every kernel of the hardware build is SystemVerilog, packaged as a Vivado
IP under `build/rtl_ip/<Name>_ip` (the targets in the four RTL sections
below; Vivado only).

| Target | Description |
|--------|-------------|
| `synthesize_kv260` | Aggregate — all four kernel IPs: `package_vectorop_rtl`, `package_conv_rtl`, `package_matmul_rtl` and `package_pool_rtl` |

A `synthesize_<platform>` aggregate is generated for every
`platforms/<platform>.json` file; the IPs are packaged for the part of the
`AXI_PLATFORM` platform, whose bounds the RTL kernels hold as constants.

No kernel has a Vitis HLS synthesis or C/RTL co-simulation target any more:
ConvKernel's was the last (CONV_RTL_PLAN phase 3; MatmulKernel's went in
MATMUL_RTL_PLAN phase 4, VectorOPKernel's in VECTOROP_RTL_PLAN phase 3,
PoolingKernel's in POOL_RTL_PLAN phase 3).  `kernels/conv`, `kernels/matmul`,
`kernels/vectorop` and `kernels/pool` keep the HLS kernels' C++ as the
reference models and fixture generators (`TestConvRef`, `TestConvSweep`,
`gen_conv_test_data` — `TestMatmulRef`, `TestMatmulBlas`,
`gen_matmul_test_data` — `TestSimulation`, `gen_vectorop_test_data` —
`TestPoolingSim`, `gen_pool_test_data`); the RTL kernels are simulated with
Verilator and xsim.

---

## RTL MatmulKernel (SystemVerilog)

The MatmulKernel of the hardware build, `kernels/matmul_rtl/`
([MATMUL_RTL_KERNEL](../kernels/MATMUL_RTL_KERNEL.md)): a drop-in for the
retired HLS kernel (same VLNV, registers, layouts and bits).  Each target exists
only when its tool is found: Verilator 5.x (`sudo apt install verilator`) for
the testbench and lint, Vivado for the rest (`driver_matmul_rtl` needs only
Python).

| Target | Description |
|--------|-------------|
| `TestMatmulRtl` | Verilator testbench `build/kernels/matmul_rtl/vl/Vtb` (in `all`); CTest runs it on the fixtures + random cases |
| `lint_matmul_rtl` | `verilator --lint-only -Wall` with `scripts/lint_waivers.vlt` |
| `perf_matmul_rtl` | Cycle counts / MAC per cycle of typical shapes (ideal memory) |
| `matmul_rtl_tb_fst` / `matmul_rtl_tb_vcd` | The testbench with FST / VCD tracing (`vlt/Vtb`, `vcd/Vtb`; `--case "..." --trace FILE`) |
| `driver_matmul_rtl` | The C driver (`scripts/gen_driver.py`, the HLS driver's API) → `build/kernels/matmul_rtl/driver/MatmulKernel_v1_0/`, compiled with `-Werror`; CTest `MatmulRtlDriver` checks its register table against the RTL |
| `package_matmul_rtl` | Vivado IP `xilinx.com:hls:MatmulKernel:1.0` with the driver → `build/rtl_ip/MatmulKernel_ip` (+ `.zip`), ~20 s |
| `synth_matmul_rtl` | Vivado out-of-context synthesis + P&R on the platform's part at `MM_RTL_PERIOD` ns (default 3.333) → `build/kernels/matmul_rtl/synth/*.rpt` |
| `xsim_matmul_rtl` | `xvlog` / `xelab` parse and elaboration |
| `neteq_matmul_rtl` | Vivado's OOC-synthesised netlist of the IP against the RTL, in lockstep in xsim on board-like jobs (`tools/neteq`; exit 1 on a mismatch) → `build/kernels/matmul_rtl/neteq/` |
| `sysim_matmul_rtl` | The test stand's MatmulKernel block design (PS VIP, interconnect, DDR model, `matmul_tb.sv`) with this IP, gmem2 upgraded to 128, on a copy in `build/kernels/matmul_rtl/sysim/` (~4 min) |

The IP is packaged outside `build/kernels/`, in `build/rtl_ip/`.  The
hardware targets scan `build/ip_repo_kv260/` (target `ip_repo_kv260`: links
the four RTL IPs), and `behavior_test_matmul` takes the
RTL IP (depending on `package_matmul_rtl`).  After the IP upgrade, both the
`cormorant_hw_128` scripts and the test stand put every kernel instance's
`C_M_AXI_*_DATA_WIDTH` back to its IP's default, so `MatmulKernel_0`'s gmem2
becomes 128 (the block designs still say 32, the HLS kernel's width).
Projects take the driver from `driver_matmul_rtl`'s output
(`build/kernels/matmul_rtl/driver/MatmulKernel_v1_0/src`; the packaged IP
carries the same files).

---

## RTL VectorOPKernel (SystemVerilog)

The VectorOPKernel of the hardware build, `kernels/vectorop_rtl/`
([VECTOROP_RTL_KERNEL](../kernels/VECTOROP_RTL_KERNEL.md)): a drop-in for the
retired HLS kernel (same VLNV, registers, m_axi bus parameters and bits).
Each target exists only when its tool is found, as for MatmulKernel
(`driver_vectorop_rtl` needs only Python).

| Target | Description |
|--------|-------------|
| `TestVectorOpRtl` | Verilator testbench `build/kernels/vectorop_rtl/vl/Vtb` (in `all`); CTest runs it on the fixtures + random jobs (`VO_RTL_RANDOM_CASES`, checked against the HLS C++ when the Vitis HLS headers are found) |
| `lint_vectorop_rtl` | `verilator --lint-only -Wall` with `scripts/lint_waivers.vlt` |
| `perf_vectorop_rtl` | Cycle counts of typical jobs (ideal memory) |
| `vectorop_rtl_tb_fst` | The testbench with FST tracing (`vlt/Vtb`; `--case "..." --trace FILE`) |
| `driver_vectorop_rtl` | The C driver (`scripts/gen_driver.py`, the HLS driver's API and files) → `build/kernels/vectorop_rtl/driver/VectorOPKernel_v1_0/`, compiled with `-Werror`; CTest `VectorOpRtlDriver` checks its register table against the RTL |
| `package_vectorop_rtl` | Vivado IP `xilinx.com:hls:VectorOPKernel:1.0` with the driver → `build/rtl_ip/VectorOPKernel_ip` (+ `.zip`) |
| `synth_vectorop_rtl` | Vivado out-of-context synthesis + P&R on the platform's part at `VO_RTL_PERIOD` ns (default 3.333) → `build/kernels/vectorop_rtl/synth/*.rpt` |
| `xsim_vectorop_rtl` | `xvlog` / `xelab` parse and elaboration |
| `neteq_vectorop_rtl` | Vivado's OOC-synthesised netlist of the IP against the RTL, in lockstep in xsim on board-like jobs (`tools/neteq`; exit 1 on a mismatch) → `build/kernels/vectorop_rtl/neteq/` |
| `sysim_vectorop_rtl` | The test stand's VectorOPKernel block design (PS VIP, interconnect, DDR model) with this IP, on a copy in `build/kernels/vectorop_rtl/sysim/` |

As for MatmulKernel, the IP is packaged in `build/rtl_ip/`, `ip_repo_kv260`
links it and `behavior_test_vectorop` takes it (depending on
`package_vectorop_rtl`).  Its m_axi interfaces declare the HLS export's bus
parameters (16 outstanding bursts, 64-beat reads, 256-beat writes, gmem0/1
read-only, gmem2 write-only), so the block design sizes the crossbar slots as
for the HLS kernel.  Projects take the driver from `driver_vectorop_rtl`'s
output (`build/kernels/vectorop_rtl/driver/VectorOPKernel_v1_0/src`; the
packaged IP carries the same files).

---

## RTL PoolingKernel (SystemVerilog)

The PoolingKernel of the hardware build, `kernels/pool_rtl/`
([POOL_RTL_KERNEL](../kernels/POOL_RTL_KERNEL.md)): a drop-in for the
retired HLS kernel (same VLNV, ports, registers, m_axi bus parameters and bits).
Each target exists only when its tool is found, as for MatmulKernel
(`driver_pool_rtl` needs only Python).

| Target | Description |
|--------|-------------|
| `TestPoolRtl` | Verilator testbench `build/kernels/pool_rtl/vl/Vtb` (in `all`); CTest runs it on the fixtures + random jobs (`PL_RTL_RANDOM_CASES`, checked against the HLS C++ when the Vitis HLS headers are found) |
| `lint_pool_rtl` | `verilator --lint-only -Wall` with `scripts/lint_waivers.vlt` |
| `perf_pool_rtl` | Cycle counts of typical jobs (ideal memory) |
| `pool_rtl_tb_fst` | The testbench with FST tracing (`vlt/Vtb`; `--case "..." --trace FILE`) |
| `driver_pool_rtl` | The C driver (`scripts/gen_driver.py`, the HLS driver's API and files) → `build/kernels/pool_rtl/driver/PoolingKernel_v1_0/`, compiled with `-Werror`; CTest `PoolRtlDriver` checks its register table against the RTL |
| `package_pool_rtl` | Vivado IP `xilinx.com:hls:PoolingKernel:1.0` with the driver → `build/rtl_ip/PoolingKernel_ip` (+ `.zip`) |
| `synth_pool_rtl` | Vivado out-of-context synthesis + P&R on the platform's part at `PL_RTL_PERIOD` ns (default 3.333) → `build/kernels/pool_rtl/synth/*.rpt` |
| `xsim_pool_rtl` | `xvlog` / `xelab` parse and elaboration |
| `neteq_pool_rtl` | Vivado's OOC-synthesised netlist of the IP against the RTL, in lockstep in xsim on board-like jobs (`tools/neteq`; exit 1 on a mismatch) → `build/kernels/pool_rtl/neteq/` |
| `sysim_pool_rtl` | The test stand's PoolingKernel block design (PS VIP, interconnect, DDR model) with this IP, on a copy in `build/kernels/pool_rtl/sysim/` |

As for the other RTL kernels, the IP is packaged in `build/rtl_ip/`,
`ip_repo_kv260` links it and `behavior_test_pool` takes it (depending on
`package_pool_rtl`).  Its m_axi interfaces declare the HLS export's bus
parameters (16-beat bursts; gmem0 read-only with 16 outstanding bursts, gmem1
write-only with 8), so the block design sizes the crossbar slots as for the
HLS kernel.  `kernels/pool_rtl/rtl/pl_pkg.sv`
holds the platform's `kernels.pool` bounds as constants: configure stops
with a `FATAL_ERROR` when they differ from the platform JSON.  Projects take
the driver from `driver_pool_rtl`'s output
(`build/kernels/pool_rtl/driver/PoolingKernel_v1_0/src`; the packaged IP
carries the same files).

---

## RTL ConvKernel (SystemVerilog)

The ConvKernel of the hardware build, `kernels/conv_rtl/`
([CONV_RTL_KERNEL](../kernels/CONV_RTL_KERNEL.md)): a drop-in for the
retired HLS kernel (same VLNV `xilinx.com:hls:ConvKernel:1.0`, ports,
registers, m_axi bus parameters, DDR layouts and bits).  Each target exists
only when its tool is found, as for the other RTL kernels (`driver_conv_rtl`
needs only Python).

| Target | Description |
|--------|-------------|
| `TestConvRtl` | Verilator testbench `build/kernels/conv_rtl/vl/Vtb` (in `all`); CTest runs it on the fixtures + random jobs (`CV_RTL_RANDOM_CASES`, checked against the HLS C++ when the Vitis HLS headers are found); `--case "…"`, `--only I,J`, `--no-oracle` for one job |
| `conv_rtl_dsp_check` | `cv_mac_chain`'s behavioural model (what the testbench simulates) against Vivado's DSP48E2 unisim model (what the hardware is), every cycle on random stimulus → `dsp/Vdsp`, CTest `ConvRtlDsp` |
| `lint_conv_rtl` | `verilator --lint-only -Wall` with `scripts/lint_waivers.vlt` |
| `perf_conv_rtl` | Cycle counts of the board benchmarks' layers (ideal memory) |
| `conv_rtl_tb_fst` | The testbench with FST tracing (`vlt/Vtb`; `--case "..." --trace FILE`; the conv-rtl-trace skill) |
| `driver_conv_rtl` | The C driver (`scripts/gen_driver.py`, the HLS driver's API and files) → `build/kernels/conv_rtl/driver/ConvKernel_v1_0/`, compiled with `-Werror`; CTest `ConvRtlDriver` checks its register table against the RTL |
| `package_conv_rtl` | Vivado IP `xilinx.com:hls:ConvKernel:1.0` with the driver → `build/rtl_ip/ConvKernel_ip` (+ `.zip`) |
| `synth_conv_rtl` | Vivado out-of-context synthesis + P&R on the platform's part at `CV_RTL_PERIOD` ns (default 3.333) → `build/kernels/conv_rtl/synth/*.rpt` |
| `xsim_conv_rtl` | `xvlog` / `xelab` parse and elaboration |
| `sysim_conv_rtl` | The test stand's ConvKernel block design (PS VIP, interconnect, DDR model, the DSP48E2 models) with this IP, on a copy in `build/kernels/conv_rtl/sysim/` (~75 min for the 63 fixtures) |

As for the other RTL kernels, the IP is packaged in `build/rtl_ip/`,
`ip_repo_kv260` links it and `behavior_test_conv` takes it (depending on
`package_conv_rtl`).  Its m_axi interfaces declare the HLS export's bus
parameters, so the block design sizes the crossbar slots as for the HLS
kernel.  `kernels/conv_rtl/rtl/cv_pkg.sv` holds the platform's
`kernels.conv` bounds as constants: configure stops with a `FATAL_ERROR`
when they differ from the platform JSON.  Projects take the driver from
`driver_conv_rtl`'s output (`build/kernels/conv_rtl/driver/ConvKernel_v1_0/src`;
the packaged IP carries the same files).

---

## HDL test fixtures

Re-run a C-sim test in `--dump-data` mode to emit hex fixtures
(`manifest.txt` + `test_*.hex`) for the RTL behavior testbenches.

| Target | Description |
|--------|-------------|
| `gen_vectorop_test_data` | VectorOPKernel HDL fixtures |
| `gen_conv_test_data` | ConvKernel HDL fixtures |
| `gen_matmul_test_data` | MatmulKernel HDL fixtures |
| `gen_pool_test_data` | PoolingKernel HDL fixtures |

Output goes to `build/<dir>/` (`vectorop_test_data`, `conv_test_data`,
`matmul_test_data`, `pool_test_data`; override with the `VA_` / `CONV_` /
`MATMUL_` / `POOL_TEST_DATA_DIR` cache variables). The targets exist only
when the Vitis HLS headers are found (VectorOPKernel: only for a 16-bit
`ap_fixed` `VA_DATA_TYPE`). The behaviour tests do **not** read these —
they use the checked-in goldens under `hw/test_data/`
(`vecop_test_data`, `conv_test_data`, `matmul_test_data`,
`pool_test_data`), so regenerated fixtures must be copied there.

---

## RTL behavior tests

Drive the per-kernel `cormorant_test_stand` Vivado project through its xsim
flow with the freshly-built IP catalogue and the checked-in golden fixtures
under `hw/test_data/`. Requires Vivado 2025.2 and the
`hw/cormorant_test_stand` submodule. The scoreboard is written to
`build/kernels/<k>/kv260/<test-stand name>_test_report.json` and
`cmake/check_test_report.py` turns its `summary.all_passed` into the
target's exit code.

| Target | Description |
|--------|-------------|
| `behavior_test_conv` | RTL behavior test — ConvKernel |
| `behavior_test_matmul` | RTL behavior test — MatmulKernel |
| `behavior_test_pool` | RTL behavior test — PoolingKernel |
| `behavior_test_vectorop` | RTL behavior test — VectorOPKernel |
| `behavior_test` | Aggregate — all four kernels in sequence |

`behavior_test_<k>` depends on `package_<k>_rtl` (the IP the test stand's
`.xpr` is pointed at, `build/rtl_ip/<Name>_ip`).  All four
pass (201 VectorOP, 63 Conv, 50 Matmul (11 GEMV), 45 Pool cases).  The runs modify
tracked `.bd` / `.xci` / `.xpr` files of `hw/cormorant_test_stand`; do not
commit them.

---

## Hardware / device tree

Require the `hw/cormorant_hw_128` submodule, Vivado, and `dtc`.

| Target | Description |
|--------|-------------|
| `build_hw_kv260` | Vivado synthesis + implementation + bitstream of the 128-bit block design (`hw/cormorant_hw_128/build.sh all`); depends on `synthesize_kv260`, so it re-packages the four kernel IPs first when their sources changed. Modifies tracked `.bd` / `.xci` / `.xpr` files of the submodule (do not commit them); the `File not found as '…/design_cormorant_wrapper.dcp'; using path …` warning (an old incremental-synthesis checkpoint path in the `.xpr`) is harmless |
| `sim_hw_kv260` | Hardware-level simulation of the integrated design (block-design testbench, 73 cases over the four kernels, ~3 min; see [TESTING.md §3](TESTING.md#3-hardware-simulation-vivado-no-board)); `scripts/sim.tcl` exits 1 unless `simulate.log` contains `ALL TESTS PASSED` |
| `dtbo_kv260_cormorant` | Compile the device-tree blob overlay (`.dtbo`) for the KV260; `dtc`'s `reg_format` / `avoid_default_addr_size` warnings are expected |

---

## CMake built-in targets

| Target | Description |
|--------|-------------|
| `all` | Default — builds libraries and C-sim test executables |
| `clean` | Remove build products |
| `test` | Run CTest |
| `edit_cache` / `rebuild_cache` | Edit / regenerate the CMake cache |
| `depend` | Regenerate dependency information |

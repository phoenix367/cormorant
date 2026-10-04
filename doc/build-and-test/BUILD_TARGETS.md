# Build Targets

Reference for every `make` target produced by the top-level CMake project.

## Setup

The project requires an **out-of-source** build (an in-source `cmake` aborts
with a `FATAL_ERROR`):

```bash
mkdir build && cd build
cmake ..
```

Key configure-time cache variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `AXI_BUS_WIDTH` | 32 | `-m_axi_max_widen_bitwidth` cap (32/64/128/256/512) for HLS auto-widening of plain-pointer ports — none is left (the last, the HLS MatmulKernel's `c`, went with its synthesis); the `burst_maxi` data ports are 128-bit by declaration. `build_hw_kv260` expects `-DAXI_BUS_WIDTH=128` |
| `AXI_PLATFORM` | `kv260` | Platform whose `platforms/<name>.json` bounds drive the C-sim `Config.h` |
| `VA_DATA_TYPE`, `CONV_DATA_TYPE` / `CONV_ACC_DATA_TYPE`, `MM_DATA_TYPE` / `MM_ACC_DATA_TYPE`, `POOL_DATA_TYPE` / `POOL_ACC_DATA_TYPE` | `ap_fixed<16,8>` / `ap_fixed<32,16>` | Element / accumulator types. Tile sizes and bounds are not cache variables — they come from the platform JSON ([PLATFORM_CONFIGURATION.md](PLATFORM_CONFIGURATION.md)) |
| `VA_ENABLE_VITIS_FLOW` | `OFF` | VectorOPKernel Vitis `hw` / `hw_emu` xclbin targets (needs an installed Vitis platform, `VA_PLATFORM`) |

Synthesis / cosim / hardware targets require **Vitis 2025.2** — source
`<Xilinx>/2025.2/Vitis/settings64.sh` (it puts `vitis-run`, `vivado` and
`xclbinutil` on `PATH`) before invoking them. The C-simulation targets need only
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
| `vadd_kernel` | VectorOPKernel library |
| `conv_kernel` | ConvKernel — configured type (`ap_fixed<16,8>` by default) |
| `matmul_kernel` | MatmulKernel's C++ model (the retired HLS kernel; the RTL kernel's reference) — configured type (`ap_fixed<16,8>` by default) |
| `pool_kernel` / `pool_kernel_float` | PoolingKernel — configured type / forced-`float` builds |

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
| `TestSimulation` | VectorOPKernel | All 6 ops across sizes + saturation cases + broadcast / stride-0 / `act` geometry cases |
| `TestConvRef` | ConvKernel | Kernel vs naive reference oracle; also registered as the CTest test `TestConvSweep` (`TestConvRef --sweep 300`, randomised geometries, ~1 min) |
| `TestConvGrid` | ConvKernel | MAC-grid unit test on `include/ConvMacGrid.h` alone |
| `TestMatmulRef` | MatmulKernel (C++ model) | Model vs the `ref_matmul_2d` / `ref_matmul_batch` oracle, all shape cases; `--dump-data` writes the RTL fixtures |
| `TestMatmulBlas` | MatmulKernel | configured kernel vs `cblas_sgemm`, bit-exact on 2^-8-grid inputs — only if BLAS is found |
| `TestPoolingSim` | PoolingKernel | Max/Average/Lp pooling + global variants |
| `TestMatmulRtl` | MatmulKernel (RTL) | Verilator testbench of the SystemVerilog kernel; the CTest test runs the 50 checked-in fixtures + 200 random cases (~50 s) — only if Verilator 5.x is found ([below](#rtl-matmulkernel-systemverilog)) |

| Aggregate | Description |
|-----------|-------------|
| `run_tests` | Builds `TestSimulation`, `TestConvRef`, `TestMatmulRef`, `TestPoolingSim` (and `TestMatmulBlas`, `TestMatmulRtl` when present), then runs `ctest --output-on-failure` |
| `test` | Runs the CTest tests without rebuilding (equivalent to `ctest`) |

CTest registers nine tests: `TestSimulation`, `TestConvRef`,
`TestConvGrid`, `TestConvSweep`, `TestMatmulRef`, `TestMatmulBlas` (only
with BLAS), `TestPoolingSim`, `MatmulRtlDriver` and `TestMatmulRtl` (only
with Verilator). `run_tests` does not list `TestConvGrid`
as a dependency — build it with `make` / `make TestConvGrid` first.

---

## HLS synthesis

C synthesis + Vivado IP-catalog export, one component per kernel. Requires
Vitis HLS. Target clock is the platform JSON's `clock` (150 MHz for
`kv260`). The exported archive lands in `build/kernels/<k>/<platform>/ip_catalog.zip`;
the IP directory the test stand uses is
`build/kernels/<k>/<platform>/<k>_<platform>/hls/impl/ip` for conv /
pool (Vitis unified component flow) and
`build/kernels/vectorop/<platform>/vadd_<platform>/solution1/impl/ip` for
VectorOPKernel (legacy `open_project` flow).

| Target | Description |
|--------|-------------|
| `synthesize_vectorop_kv260` | Synthesize VectorOPKernel for the KV260 |
| `synthesize_conv_kv260` | Synthesize ConvKernel for the KV260 |
| `synthesize_pool_kv260` | Synthesize PoolingKernel for the KV260 |
| `synthesize_kv260` | Aggregate — all four kernel IPs: the three HLS kernels and `package_matmul_rtl` |

A `synthesize_<kernel>_<platform>` target is generated for every
`platforms/<platform>.json` file.  The targets always re-run: each starts
by deleting its HLS project, so the kernel's C driver directory
(`…/impl/ip/drivers`) is missing until the synthesis finishes — do not
generate projects that copy the drivers meanwhile.

MatmulKernel has no HLS synthesis: the hardware build packages the
SystemVerilog kernel ([below](#rtl-matmulkernel-systemverilog)).
`kernels/matmul` keeps the HLS kernel's C++ as the reference model
(`TestMatmulRef`, `TestMatmulBlas`, `gen_matmul_test_data`; MATMUL_RTL_PLAN
phase 4).

---

## HLS co-simulation

C synthesis + C/RTL co-simulation in a single component run (`csynth_design`
then `cosim_design`). Slow (minutes); not a dependency of anything. Requires
Vitis HLS.

| Target | Description |
|--------|-------------|
| `cosim_conv_kv260` | Cosim ConvKernel against `TestConvSim.cpp` |
| `cosim_pool_kv260` | Cosim PoolingKernel against `TestPoolingSim.cpp` |

(VectorOPKernel has no cosim target.)

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
| `sysim_matmul_rtl` | The test stand's MatmulKernel block design (PS VIP, interconnect, DDR model, `matmul_tb.sv`) with this IP, gmem2 upgraded to 128, on a copy in `build/kernels/matmul_rtl/sysim/` (~4 min) |

The IP is packaged outside `build/kernels/`, in `build/rtl_ip/`.  The
hardware targets scan `build/ip_repo_kv260/` (target `ip_repo_kv260`: links
to the three HLS exports and the RTL IP), and `behavior_test_matmul` takes the
RTL IP (depending on `package_matmul_rtl`).  After the IP upgrade, both the
`cormorant_hw_128` scripts and the test stand put every kernel instance's
`C_M_AXI_*_DATA_WIDTH` back to its IP's default, so `MatmulKernel_0`'s gmem2
becomes 128 (the block designs still say 32, the HLS kernel's width).
Projects take the driver from `driver_matmul_rtl`'s output
(`build/kernels/matmul_rtl/driver/MatmulKernel_v1_0/src`; the packaged IP
carries the same files).

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

Each `behavior_test_<k>` depends on `synthesize_<k>_kv260` (the IP catalogue
must exist at the revision the test stand's `.xpr` references);
`behavior_test_matmul` on `package_matmul_rtl`.  All four
pass (119 VectorOP, 63 Conv, 50 Matmul (11 GEMV), 43 Pool cases).  The runs modify
tracked `.bd` / `.xci` / `.xpr` files of `hw/cormorant_test_stand`; do not
commit them.

---

## Hardware / device tree

Require the `hw/cormorant_hw_128` submodule, Vivado, and `dtc`.

| Target | Description |
|--------|-------------|
| `build_hw_kv260` | Vivado synthesis + implementation + bitstream of the 128-bit block design (`hw/cormorant_hw_128/build.sh all`); depends on `synthesize_kv260`, so it re-runs all four kernel IPs first; configure with `-DAXI_BUS_WIDTH=128`. Modifies tracked `.bd` / `.xci` / `.xpr` files of the submodule (do not commit them); the `File not found as '…/design_cormorant_wrapper.dcp'; using path …` warning (an old incremental-synthesis checkpoint path in the `.xpr`) is harmless |
| `sim_hw_kv260` | Hardware-level simulation of the integrated design (block-design testbench, 68 cases over the four kernels, ~3 min; see [TESTING.md §3](TESTING.md#3-hardware-simulation-vivado-no-board)); `scripts/sim.tcl` exits 1 unless `simulate.log` contains `ALL TESTS PASSED` |
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

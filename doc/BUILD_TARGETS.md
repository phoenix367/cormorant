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
| `AXI_BUS_WIDTH` | 32 | m_axi master data width (32/64/128/256/512) |
| `AXI_PLATFORM` | `kv260` | Platform whose `platforms/<name>.json` bounds drive the C-sim `Config.h` |
| `<K>_DATA_TYPE` etc. | per kernel | Element / accumulator types and tile sizes (see each kernel's `CMakeLists.txt`) |

Synthesis / cosim / hardware targets require **Vitis 2025.2** — source
`settings64.sh` before invoking them.

---

## Kernel libraries

Static libraries — intermediate build products, linked by the test and
synthesis targets. Rarely built directly.

| Target | Description |
|--------|-------------|
| `vadd_kernel` | VectorOPKernel object library |
| `conv_kernel` / `conv_kernel_float` | ConvKernel — configured type / forced-`float` builds |
| `matmul_kernel` | MatmulKernel — configured type (`ap_fixed<16,8>` by default) |
| `pool_kernel` / `pool_kernel_float` | PoolingKernel — configured type / forced-`float` builds |

The `*_float` variants force `Data_t=float` regardless of HLS availability;
they back the BLAS oracle tests.

---

## C-simulation tests

Compile and run with plain GCC — no Vitis, no hardware.

| Target | Kernel | Description |
|--------|--------|-------------|
| `TestSimulation` | VectorOPKernel | All 6 ops across sizes + saturation cases |
| `TestConvRef` | ConvKernel | Kernel vs naive reference oracle |
| `TestMatmulRef` | MatmulKernel | Kernel vs `ref_matmul` oracle, all shape cases |
| `TestMatmulBlas` | MatmulKernel | configured kernel vs `cblas_sgemm`, bit-exact on 2^-8-grid inputs — only if BLAS is found |
| `TestPoolingSim` | PoolingKernel | Max/Average/Lp pooling + global variants |

| Aggregate | Description |
|-----------|-------------|
| `run_tests` | Builds every C-sim test executable above |
| `test` | Runs them via CTest (equivalent to `ctest`) |

---

## HLS synthesis

C synthesis + Vivado IP-catalog export, one component per kernel. Requires
Vitis HLS. Output IP lands under `build/kernels/<k>/<platform>/<k>_<platform>/hls/impl/ip`.

| Target | Description |
|--------|-------------|
| `synthesize_vectorop_kv260` | Synthesize VectorOPKernel for the KV260 |
| `synthesize_conv_kv260` | Synthesize ConvKernel for the KV260 |
| `synthesize_matmul_kv260` | Synthesize MatmulKernel for the KV260 |
| `synthesize_pool_kv260` | Synthesize PoolingKernel for the KV260 |
| `synthesize_kv260` | Aggregate — all four kernels |

A `synthesize_<kernel>_<platform>` target is generated for every
`platforms/<platform>.json` file.

---

## HLS co-simulation

C synthesis + C/RTL co-simulation in a single component run (`csynth_design`
then `cosim_design`). Slow (minutes); not a dependency of anything. Requires
Vitis HLS.

| Target | Description |
|--------|-------------|
| `cosim_conv_kv260` | Cosim ConvKernel against `TestConvSim.cpp` |
| `cosim_matmul_kv260` | Cosim MatmulKernel against `TestMatmulSim.cpp` |
| `cosim_pool_kv260` | Cosim PoolingKernel against `TestPoolingSim.cpp` |

(VectorOPKernel has no cosim target.)

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

---

## RTL behavior tests

Drive the per-kernel `cormorant_test_stand` Vivado project through its xsim
flow with the freshly-built IP catalogue and the checked-in golden fixtures
under `hw/test_data/`. Requires Vitis and the `hw/cormorant_test_stand`
submodule.

| Target | Description |
|--------|-------------|
| `behavior_test_conv` | RTL behavior test — ConvKernel |
| `behavior_test_matmul` | RTL behavior test — MatmulKernel |
| `behavior_test_pool` | RTL behavior test — PoolingKernel |
| `behavior_test_vectorop` | RTL behavior test — VectorOPKernel |
| `behavior_test` | Aggregate — all four kernels in sequence |

Each `behavior_test_<k>` depends on `synthesize_<k>_kv260` (the IP catalogue
must exist at the revision the test stand's `.xpr` references).

---

## Hardware / device tree

Require the `hw/cormorant_hw_128` submodule, Vivado, and `dtc`.

| Target | Description |
|--------|-------------|
| `synthesize_kv260` → `build_hw_kv260` | `build_hw_kv260` builds the Vivado hardware design; depends on `synthesize_kv260` |
| `sim_hw_kv260` | Hardware-level simulation of the integrated design |
| `dtbo_kv260_cormorant` | Compile the device-tree blob overlay (`.dtbo`) for the KV260 |

---

## CMake built-in targets

| Target | Description |
|--------|-------------|
| `all` | Default — builds libraries and C-sim test executables |
| `clean` | Remove build products |
| `test` | Run CTest |
| `edit_cache` / `rebuild_cache` | Edit / regenerate the CMake cache |
| `depend` | Regenerate dependency information |

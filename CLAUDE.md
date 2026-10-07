# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AXI4 (`m_axi`) vector operation IP core for Xilinx FPGAs, implemented in SystemVerilog (`kernels/vectorop_rtl/`; the original Vitis HLS kernel's C++ in `kernels/vectorop/` is its reference model). The kernel reads up to two equal-length element arrays, applies a runtime-selected element-wise operation, and writes the results to a third array. Six operations are supported (Add, Sub, Mul, Div, Relu, Relu6); the selector is a runtime AXI-Lite register. The vector length and element data type are also configurable at runtime and CMake configure time respectively.

The repository also contains **`inference-scheduler/`**, a Python code-generator that parses an ONNX model and emits a self-contained C project that drives all four hardware IPs (VectorOPKernel, MatmulKernel, ConvKernel, PoolingKernel) using the generated Xilinx driver APIs and the Xil bare-metal library. The codegen overlaps work across different kernel lanes — ops on distinct lanes (e.g. Conv ‖ Pool) run concurrently, with synchronisation funnelled through a single weak-symbol `kernel_wait()` primitive that defaults to polling but can be link-overridden for IRQ/UIO waiting. Pool-slot colouring uses event-stream liveness intervals so two tensors share a slot only when one is fully drained before the other's producer starts.

Documentation index: `doc/README.md` — docs are grouped in `doc/build-and-test/`, `doc/kernels/` (reference + optimisation log per kernel), `doc/scheduler/` and `doc/plans/` (project plans and results); the scheduler's user guides are in `inference-scheduler/doc/`.

**Fact registry:** `facts.yaml` lists the facts that code and docs state in several places — test counts, the bitstream id, CLI flags and script-to-script flags, kernel register maps, VectorOP codes, the supported ops, event-stream kinds, HTTP routes, ctypes bindings, config keys, the platform JSON and kernel bounds, UIO names, clocks, the toolchain release, pool sizes, headline board results and FPGA utilization, the voice samples and the demo video link — with where each is true and every place that repeats it.  `python3 tools/facts/facts.py check` verifies them (`fix` rewrites stale counts; `impact FILE` lists what must change with a file; `changed` checks the facts a branch touches) — `tools/facts/README.md`.  `facts.py install-hook` (once per clone) makes `git commit` refuse a commit that leaves a fact it touches stale; `git commit --no-verify` skips it.  When you find the same fact stale in two places, add it to the registry.

## Build System

All four kernels live under `kernels/` and are built from a single top-level CMake project, all four in SystemVerilog (`kernels/vectorop_rtl/`, `kernels/conv_rtl/`, `kernels/matmul_rtl/`, `kernels/pool_rtl/`); `kernels/vectorop/`, `kernels/conv/`, `kernels/matmul/` and `kernels/pool/` keep the retired Vitis HLS kernels' C++ as their reference models.
See `doc/build-and-test/BUILD_TARGETS.md` for a full reference of every `make` target.

```bash
mkdir build && cd build

# Configure all four kernels at once
cmake ../

# C simulation tests (no hardware needed; they need the Vitis HLS headers)
make TestSimulation    # VectorOPKernel's C++ reference model (the retired HLS kernel)
make TestConvRef       # ConvKernel's C++ reference model (the retired HLS kernel)
make TestConvGrid      # its MAC grid (include/ConvMacGrid.h) alone
make TestMatmulRef     # MatmulKernel's C++ reference model (the retired HLS kernel)
make TestPoolingSim    # PoolingKernel's C++ reference model (the retired HLS kernel)
make TestMatmulRtl     # MatmulKernel in SystemVerilog (kernels/matmul_rtl/, Verilator 5.x)
make TestVectorOpRtl   # VectorOPKernel in SystemVerilog (kernels/vectorop_rtl/, Verilator 5.x)
make TestPoolRtl       # PoolingKernel in SystemVerilog (kernels/pool_rtl/, Verilator 5.x)
make TestConvRtl       # ConvKernel in SystemVerilog (kernels/conv_rtl/, Verilator 5.x)
ctest                  # run all

# Vivado IP export for the KV260 (every kernel's HLS synthesis is retired)
make synthesize_kv260            # all four IPs
make package_conv_rtl            # ConvKernel: the SystemVerilog IP (Vivado)
make driver_conv_rtl             # its C driver, which projects copy (build/kernels/conv_rtl/driver/...)
make package_matmul_rtl          # MatmulKernel: likewise
make driver_matmul_rtl           # its C driver, which projects copy (build/kernels/matmul_rtl/driver/...)
make package_vectorop_rtl        # VectorOPKernel: likewise
make driver_vectorop_rtl         # (build/kernels/vectorop_rtl/driver/...)
make package_pool_rtl            # PoolingKernel: likewise
make driver_pool_rtl             # (build/kernels/pool_rtl/driver/...)
```

**The kernel clock is 250 MHz** (since `doc/plans/FMAX_250_PLAN.md`, bitstream `986cef4866a0`; `6436623029f7` adds VectorOPKernel's activation unit, `doc/plans/ACTIVATIONS_PLAN.md`; production `588d721997cb` adds its softmax unit, `doc/plans/SOFTMAX_PLAN.md`): the block design clocks the four kernels, the interconnects and the PS-PL AXI ports from an MMCM (`clk_wiz_0`) fed by PL0, which stays at 100 MHz (`hw/cormorant_hw_128/scripts/bd_kernel_clock.tcl`; register slices in both data interconnects; impl directives in its `scripts/build.tcl`).  The design closes 4 ns by a small margin (+0.041 ns; +0.061 before the softmax unit, +0.105 before the activation unit), so in-context timing is a property every kernel change must keep: wide fanouts to the URAM / DSP / BRAM columns go through register trees, each kernel registers its own reset (`keep`), and no variable may be written from more than one generate scope — Vivado mis-synthesises it while Verilator and xsim simulate it as written (`Synth 8-4767` in the synthesis log; the PoolingKernel bug of FMAX_250_PLAN).  `make neteq_matmul_rtl` / `neteq_vectorop_rtl` / `neteq_pool_rtl` (`tools/neteq`) runs a kernel's RTL in lockstep with its synthesised netlist in xsim.

Each kernel (every `kernels/<k>`) can also be built standalone — `cmake/AxiPlatform.cmake` gives the ones that read the platform the bus width and platform list the top level sets:

```bash
cd kernels/vectorop && mkdir build && cd build
cmake ../ -DVA_DATA_TYPE=ap_fixed\<16,8\>
make TestSimulation
```

### Key CMake Parameters (VectorOPKernel's C++ model)

| Parameter | Default | Description |
|---|---|---|
| `VA_DATA_TYPE` | `ap_fixed<16,8>` (`float` without the Vitis HLS headers) | Element type of the C++ model and `TestSimulation`: `float`, `double`, `half`, `uint8_t`, `ap_fixed<W,I>` (the RTL kernel is Q8.8 only) |

### Per-platform configuration

Each `platforms/<name>.json` file (top-level, shared across kernels) defines a `synthesize_<name>` target that packages the four RTL IPs for its part (`syn/package_ip.tcl` of each `kernels/<k>_rtl`).  The top-level `AXI_PLATFORM` cache var (default `kv260`) also selects which platform's bounds drive the C-sim Config.h for each kernel.

**Schema reference: [`doc/build-and-test/PLATFORM_CONFIGURATION.md`](doc/build-and-test/PLATFORM_CONFIGURATION.md)** — full field-by-field tables for top-level keys and the `kernels.{conv,matmul,pool}` blocks (VectorOPKernel has none), with constraint formulas, the "add a new platform" workflow, and the "edit existing bounds + regenerate hardware-bound fixtures" workflow.

Key facts to keep in mind when editing platform JSON or any code that touches it:

- The JSON is the **single source of truth**.  The C++ build reads it via `string(JSON …)` in `kernels/<k>/CMakeLists.txt::<k>_load_constants` (no `set(<K>_* CACHE …)` defaults); the Python scheduler reads it via `inference-scheduler/src/_<k>_hw_config.py::resolve()`.  A missing field is a `FATAL_ERROR` during CMake configure and a `<Kernel>HwConfigError` at scheduler load time.
- `CMAKE_CONFIGURE_DEPENDS` is set on every platform JSON, so `make` auto-reruns cmake when the JSON changes.
- The RTL kernels hard-code their bounds: `kernels/pool_rtl/CMakeLists.txt` and `kernels/conv_rtl/CMakeLists.txt` check every `kernels.pool` / `kernels.conv` bound against `pl_pkg.sv` / `cv_pkg.sv` at configure time (`FATAL_ERROR` when they differ), and MatmulKernel's `K_MAX` (`kernels/matmul_rtl/rtl/mm_pkg.sv`) must equal `kernels.matmul.max_k` — changing those bounds means changing the RTL.
- Bumping a `max_*` bound requires re-running `inference-scheduler/test/gen_{conv,matmul,pool}_models.py` so the hardware-bound boundary fixtures re-derive their geometries from the new JSON — otherwise "must raise" / "at limit" tests can start passing on the wrong values.
- `AXI_BUS_WIDTH` is a top-level CMake cache variable, **not** a JSON field.  It steered the HLS kernels' port widening; with every kernel in RTL (fixed 128-bit ports) it no longer affects any IP — only the C++ models' informational `kAxiBusWidth`.

Adding a new platform requires only a JSON file and a re-run of cmake.

## Architecture

### Kernel Interface

```
DDR/PL ──► m_axi_gmem0 (a, read,  128-bit burst_maxi) ─┐
DDR/PL ──► m_axi_gmem1 (b, read,  128-bit burst_maxi) ─┤  VectorOPKernel  ├──► m_axi_gmem2 (c, write, 128-bit burst_maxi) ──► DDR/PL
           s_axi_ctrl: a, b, c, size, op, outer, a_inc, b_inc, act, alpha, smx_cm, smx_cfg, smx_mask, ap_ctrl_hs
```

All three ports are 128 bits wide (8 elements per beat); the kernel reads in bursts of ≤ 64 beats with ≤ 16 outstanding per port, writes bursts of ≤ 256 beats with ≤ 16 awaiting B (the IP declares these counts, as the HLS export did, so the block design sizes the crossbar from them), and processes 8 lanes per cycle (`OP_DIV`: 1 lane per cycle).  Bit-identical to the retired HLS kernel (whose divider, not its C simulation, saturates −128 ÷ −1/256 to 0x7FFF). For unary ops (op ≥ 4: Relu, Relu6 and the activation ops) no AXI transactions are issued on `gmem1`. `act` (0 none / 1 relu / 2 relu6 / 3 leaky relu / 4 silu / 5 gelu / 6 gelu tanh) applies an activation after the op so the scheduler fuses Add→Relu; `alpha` (bits 15:0 / 65536) is LeakyReLU's slope.  The softmax ops 10 (rows) / 11 (keys-major columns, transposed) take `smx_cm` / `smx_cfg` / `smx_mask` and write c at the b stride: an integer softmax within 1 LSB of the exact one (`doc/plans/SOFTMAX_PLAN.md`, production bitstream `588d721997cb`); the scheduler puts ONNX `Softmax` (BERT) and, with the frontends' `vsmx`, the vision / LLM-prefill attention softmax there (platform `kernels.vectorop.softmax`, `AXI_VECTOROP_SOFTMAX=0` for older bitstreams).  LeakyReLU, SiLU and both GELUs (ops 6–9 or acts 3–6, `doc/plans/ACTIVATIONS_PLAN.md`) give the exact function rounded to nearest, ties to even — in the production bitstream `588d721997cb` (and `6436623029f7` before it); the scheduler maps `LeakyRelu`, SiLU and `Gelu` onto them, and SmolVLM's vision GELU (`doc/plans/OFFLOAD_PLAN.md`). **Alignment contract:** every run start of a, b, c is 16-byte aligned (`a_inc`/`b_inc` are 0 or a multiple of 8 elements); `size` is arbitrary; the last word of every output run is written whole (tail lanes = 0), so the caller's buffer / stride gap must cover it (the scheduler's `CHUNK_STRIDE` and 64-byte allocations do). Block-design instance widths must equal the IP defaults (128 on all three ports); see `doc/kernels/VECTOROP_RTL_KERNEL.md` (and `VECTOROP_OPTIMISATION.md` for the HLS kernel's history).

### Supported Operations

| Code | Name | Expression | Arity |
|------|------|------------|-------|
| 0 | `OP_ADD` | `saturate_cast(a[i] + b[i])` | binary |
| 1 | `OP_SUB` | `saturate_cast(a[i] - b[i])` | binary |
| 2 | `OP_MUL` | `saturate_cast(a[i] * b[i])` | binary |
| 3 | `OP_DIV` | `saturate_cast(a[i] / b[i])` | binary |
| 4 | `OP_RELU` | `max(a[i], 0)` | unary |
| 5 | `OP_RELU6` | `min(max(a[i], 0), 6)` | unary |
| 6 | `OP_LEAKY_RELU` | `a[i] >= 0 ? a[i] : alpha * a[i]` (alpha register) | unary |
| 7 | `OP_SILU` | `a[i] * sigmoid(a[i])` | unary |
| 8 | `OP_GELU` | `a[i] * Phi(a[i])` (erf form) | unary |
| 9 | `OP_GELU_TANH` | GELU, tanh approximation | unary |
| 10 | `OP_SOFTMAX` | softmax over each row (`smx_*` registers; c at the b stride) | unary |
| 11 | `OP_SOFTMAX_T` | softmax over the keys of each query column, transposed | unary |

### Key Files

- **`kernels/vectorop_rtl/`** — VectorOPKernel in SystemVerilog, the hardware build's since `doc/plans/VECTOROP_RTL_PLAN.md` phase 3 (bitstream `68665fc1833a`): a drop-in for the HLS one (same VLNV, registers, m_axi bus parameters, DDR access pattern, bit-exact).  `rtl/` (`vo_core` job FSM and geometry, `vo_rd_port` ×2 with tail mask and stride-0 replay RAM, `vo_compute` 8-lane ALU + `vo_div`, `vo_act` activation unit with the generated table ROM `vo_act_rom.sv` (`scripts/gen_act_rom.py`), `vo_smx` softmax unit with the generated exponential table `vo_smx_rom.sv` (`scripts/gen_smx_rom.py`; `doc/plans/SOFTMAX_PLAN.md`), `vo_wr_port`), Verilator testbench (`TestVectorOpRtl`: the 212 fixtures + random cases + every-input activation and softmax jobs against the HLS C++), `scripts/gen_driver.py` (the HLS driver's API, MatmulKernel's generator), Vivado packaging (`syn/package_ip.tcl`).  Read `doc/kernels/VECTOROP_RTL_KERNEL.md` (contract, architecture, invariants) before changing the RTL.
- **`kernels/vectorop/kernel/VectorOP.cpp`** — the retired HLS kernel: the RTL kernel's reference model, testbench oracle and fixture generator. DATAFLOW of `load_words` (a, b) → `compute_words` → `store_words` on 128-bit `VecWord` streams; each stage is one flattened II=1 loop over all `outer × ceil(size/8)` words (contiguous fast path when `inc == size`, on-chip replay of a stride-0 operand ≤ 2048 elements). Three `m_axi` `burst_maxi` ports (`gmem0/1/2`) with `offset=slave`; all registers are in `s_axi_ctrl`.
- **`kernels/vectorop/include/VectorOP.h`** — Kernel declaration, `Op` enum (OP_ADD … OP_GELU_TANH), `Act` enum and `job_act`, `VecWord` / `kVecLanes` lane helpers, the alignment contract, and the `saturate_cast<T>` / `round_cast<T>` templates.
- **`kernels/vectorop/include/Config.h.in`** — CMake template that produces `Config.h` with `Data_t`, `kDataWidthBits`, and `kSeed`.
- **`kernels/vectorop/test/TestSimulation.cpp`** — Tests all 10 element-wise operations across sizes 1…4097 (tail words), saturation and activation boundary cases, every Q8.8 input of each activation (exactly), the softmax in both modes against the integer specification (and within 1 LSB of the exact softmax), and geometry / `act` cases (broadcast chunk 12 stride 16, outer 1000 × 16, stride-0 operands at and past the replay bound, runs > 16 × 256 words); asserts the alignment contract (`inc % 8 == 0`, tail lanes 0, gaps untouched). `--dump-data` writes the RTL fixtures (manifest with `act`, `alpha` and the three softmax register columns) into the build tree (`make gen_vectorop_test_data`); the behaviour tests read the checked-in copies in `hw/test_data/vecop_test_data`.
- **`kernels/matmul_rtl/`** — MatmulKernel in SystemVerilog, a drop-in for the HLS one (same VLNV, registers, layouts, bit-exact; gmem2 128-bit; 128 MAC/cycle GEMM).  Verilator testbench (`TestMatmulRtl`: the HLS fixtures + random cases), `scripts/gen_driver.py` (the C driver with the HLS driver's API, checked against `rtl/mm_ctrl_s_axi.sv`), Vivado packaging.  The hardware build's MatmulKernel since `doc/plans/MATMUL_RTL_PLAN.md` phase 4 (bitstream `1d28630fbfa4`; the hw_128 and test-stand scripts reset instance m_axi widths to the IP defaults after the upgrade, so gmem2 becomes 128); `kernels/matmul/` keeps the HLS kernel's C++ as the reference model and fixture generator.  `platforms/kv260.json` `kernels.matmul.impl = "rtl"` tells the scheduler (`AXI_MATMUL_IMPL=hls` models older bitstreams).  Since phase 5 (bitstream `b3309f424562`) the IP declares its m_axi bus parameters (16 outstanding bursts per port, read- / write-only), as the RTL VectorOPKernel's does — without them the block design gives each crossbar slot 2 (fact `rtl.axi_masters`).  Read `doc/kernels/MATMUL_RTL_KERNEL.md` (contract, architecture, invariants) before changing the RTL.
- **`kernels/pool_rtl/`** — PoolingKernel in SystemVerilog, the hardware build's since `doc/plans/POOL_RTL_PLAN.md` phase 3 (bitstream `dbb320fb7297`): a drop-in for the HLS one (same VLNV `xilinx.com:hls:PoolingKernel:1.0`, ports, register map, m_axi bus parameters — gmem0 read-only, gmem1 write-only, 16-beat bursts — bit-exact).  `rtl/` (`pl_core` job FSM and chunk sequencer, `pl_loader`, `pl_emit` line buffer, `pl_reduce` + finaliser, `pl_writer`; `pl_pkg.sv` holds the `kernels.pool` bounds), Verilator testbench (`TestPoolRtl`: the 45 fixtures + random jobs against the HLS C++), `scripts/gen_driver.py` (the HLS driver's API, MatmulKernel's generator), Vivado packaging (`syn/package_ip.tcl`); `kernels/pool/` keeps the HLS kernel's C++ (`kernel/PoolingKernel.cpp`, `TestPoolingSim`, `gen_pool_test_data`) as the reference model and fixture generator.  Read `doc/kernels/POOL_RTL_KERNEL.md` (contract, architecture, invariants) before changing the RTL.
- **`kernels/conv_rtl/`** — ConvKernel in SystemVerilog, the hardware build's since `doc/plans/CONV_RTL_PLAN.md` phase 3 (bitstream `c2b2a6e5e50e`): a drop-in for the HLS one (same VLNV `xilinx.com:hls:ConvKernel:1.0`, 200 ports, register map, m_axi bus parameters — gmem0 x 16 / 16, gmem1 weight 8 / 128, gmem2 bias 2 / 256 (outstanding / burst), gmem3 y write-only 8 / 64 — DDR layouts, bit-exact).  `rtl/` (`cv_core` job FSM and sweep sequencer, `cv_xload`, `cv_patch` line buffer, `cv_wload`, `cv_bias`, `cv_engine` — weight cache, the MAC grid as 32 chains of 16 DSP48E2 (`cv_mac_chain`), two accumulator buffers —, `cv_drain` transposer, `cv_ywriter`; `cv_pkg.sv` holds the `kernels.conv` bounds), Verilator testbench (`TestConvRtl`: the 63 fixtures + random jobs against the HLS C++; `ConvRtlDsp`: the behavioural MAC chain against the DSP48E2 unisim model), `scripts/gen_driver.py`, Vivado packaging; `kernels/conv/` keeps the HLS kernel's C++ (`kernel/ConvKernel.cpp`, `TestConvRef` / `TestConvGrid` / `TestConvSweep`, `gen_conv_test_data`) as the reference model and fixture generator.  `platforms/kv260.json` `kernels.conv.impl = "rtl"` selects the scheduler's RTL cost model (`AXI_CONV_IMPL=hls` models older bitstreams).  Read `doc/kernels/CONV_RTL_KERNEL.md` (contract, architecture, invariants) before changing the RTL.
- **`platforms/kv260.json`** — KV260 Starter Kit platform config (shared by all kernels).  Holds FPGA `part`/`board`/`clock` plus `kernels.conv` / `kernels.matmul` / `kernels.pool` compile-time bounds (see §"Per-platform configuration" above).

### Inference Scheduler

**`inference-scheduler/`** — Python code-generator. Parses an ONNX model and
emits a complete C project that drives up to four hardware kernels:

| Kernel | ONNX ops |
|--------|----------|
| VectorOPKernel | `Add`, `Sub`, `Mul`, `Div`, `Relu`, `Clip(0,6)`; `LeakyRelu`, SiLU (`Mul(x, Sigmoid(x))`, fused by `src/fusion.py`) and `Gelu` (native or fused) on its activation unit — platform `kernels.vectorop.activations`, `AXI_VECTOROP_ACTIVATIONS=0` for older bitstreams (`Gelu` then stays a host op), `src/vectorop_act.py`; fused into a producing Add / Sub / Mul / Div through `act` |
| MatmulKernel | `MatMul` (and fully-connected `Conv`s — the kernel covers the whole input — rewritten to MatMul where faster, `--fc-conv`, `src/fc_conv.py`) — the ones not lowered onto ConvKernel (batch-1 FC layers, `K % 16 ≠ 0`, `M % 8 ≠ 0`, < 16 rows, 4D×3D outer loops, or not estimated faster); B in ConvKernel's image (its GEMV / image path, `--matmul-gemv`, through both read ports) where that is faster or another entry pinned the weight's layout — one copy of every weight (`doc/scheduler/INFERENCE_SCHEDULER.md` §MatMul GEMV streaming) |
| ConvKernel | `Conv`; `MatMul` with swapped operand roles (A = conv weight, B = conv input, 1×kw kernel, stride (1, kw)) wherever the engine cost model says it is faster — `--matmul-on-conv auto` (default) / `always` / `off` (`--no-matmul-on-conv`), bit-identical either way (`doc/plans/BERT_PLAN.md` §2 2A) |
| PoolingKernel | `MaxPool`, `AveragePool`, `LpPool`, `GlobalMaxPool`, `GlobalAveragePool`, `GlobalLpPool` |
| (zero-cost) | `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`, `Dropout`, `Identity` and same-kind `Cast` (buffer aliases), contiguous 64-byte-aligned `Split` / `Slice` pieces (sub-buffer views), `Gemm` (decomposed → MatMul + Add), `Constant` (→ initializer) |
| (host CPU) | `SpaceToDepth` — also produced by the opt-in stride-2 stem rewrite (`OnnxGraph(s2d_stem=True)`, CLI default): Conv 7×7 s2 on ≤ 4 channels → SpaceToDepth(2) + Conv 4×4 s1 on 4·C channels |
| (host CPU) | `Softmax` (last axis; opset < 13 coerce-to-2-D), `LayerNormalization`, `Gelu` (tanh / erf), `Transpose`, `Slice` / `Split` copies, `Gather` (axis 0, int ids), `OneHot`, `Cast` (int ↔ Data_t) — `src/host_nodes.py`: double math, round-half-even + saturate write-back, staged through cached memory.  TF-style LayerNorm and GELU tanh / erf subgraphs are fused into single host nodes by `src/fusion.py` (`OnnxGraph(fuse_patterns=True)`, library + CLI default); an unmatched `ReduceMean` / `Pow` / `Sqrt` / `Reciprocal` / `Tanh` / `Erf` is rejected.  Integer tensors (ids, masks) are raw int16 in `inference_buf_t`.  BERT-base (bertsquad-12) generates — see `doc/plans/BERT_PLAN.md` |
| (host CPU, Llama decoders) | domain `axi.llm`: `LlmEmbed`, `LlmRMSNorm`, `LlmResAdd` (float32 residual), `LlmAttention` (RoPE + KV-cache write + causal GQA attention, one float region; decode with `decode_attn="host"`), `LlmSiluMul`, `LlmSelectRow`, `LlmDequant`, and attention on the FPGA — prefill, and decode in the chat libraries (`decode_attn="fpga"`, `doc/plans/KV_DECODE_PLAN.md`): `LlmAttnPrep`, `LlmAttnScores` / `LlmAttnPV` (q·Kᵀ / P·V on ConvKernel over the cached keys, `LlmAttnConvNode`), `LlmAttnSoftmax`, `LlmAttnMerge` — `src/llm_nodes.py`; the graphs come from the Llama frontend `src/llama.py` (config.json + safetensors + calibrated formats → decode / prefill_<T> / head entries) with power-of-two exponents, host tensors and states in the model's `axi.numeric` metadata (`src/numeric.py`).  SmolLM2-135M → `libsmollm2.so`, SmolLM2-360M → `libsmollm2_360m.so` — see `doc/plans/CHAT_PLAN.md` §13, §16, §20 |
| (host CPU, vision encoder) | domain `axi.llm`: `VitEmbedAdd`, `VitLayerNorm`, `VitResAdd`, `VitAttnPrep`, `VitAttnSoftmax`, `VitGelu`, `VitPixelShuffle`, `VitSumDequant` — `src/vit_nodes.py`; q·Kᵀ / P·V on ConvKernel (`LlmAttnConvNode`, static keys); `VitGelu` on VectorOPKernel's activation unit where its output is at 2^-8 (`VitGeluVopNode`: ADD bias, MUL + GELU_TANH; 11 of 12 layers, `doc/plans/OFFLOAD_PLAN.md` §2.1); the `vision` entry from `src/vit.py` (SigLIP-style ViT + Idefics3 connector → image-feature state that the text model's prefill `LlmEmbed` reads for ids V + k).  SmolVLM-256M → `libsmolvlm_256m.so` (`llm_image()`) — see `doc/plans/CHAT_PLAN.md` §22–§24 |
| (host CPU, text to speech) | domain `axi.llm`: `TtsPrep`, `TtsGate`, `TtsSum`, `TtsFlowOut`, `TtsInterleave`, `TtsPcm` — `src/tts_nodes.py` (the decoder's residual sums are VectorOP ADDs, `TtsAddVopNode`: one exponent per decoder stage, `doc/plans/OFFLOAD_PLAN.md` §2.2); the flow and HiFi-GAN convs are ConvKernel `Conv`s with power-of-two exponents (`numeric.encode_conv_weights`); the `chunk` entry from `src/piper.py` (Piper / VITS: 128 frames = 1.49 s of audio per pass); the text encoder as `encode_<T>` entries (`TtsEmbed`, `TtsRowPrep`, `TtsAttnSoftmax`, `TtsAttnMerge`, `TtsResNorm`, `TtsEncOut` + MatMuls and attention on ConvKernel).  Piper lessac-medium → `libpiper_tts.so`, the chat server's `/v1/audio/speech` — see `doc/plans/TTS_PLAN.md` §4–§5 |

```bash
cd inference-scheduler
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Generate test models (all generators required before running tests)
.venv/bin/python test/gen_test_models.py
.venv/bin/python test/gen_matmul_models.py
.venv/bin/python test/gen_mixed_kernel_models.py
.venv/bin/python test/gen_conv_models.py
.venv/bin/python test/gen_pool_models.py
.venv/bin/python test/gen_reshape_gemm_models.py
.venv/bin/python test/gen_mixed_all_kernels_models.py
.venv/bin/python test/gen_parallel_models.py
.venv/bin/python test/gen_bert_models.py      # tiny BERT-like fixtures (host ops, fusion)
.venv/bin/python test/gen_llama_models.py     # tiny random Llama (decoder ops, exponents; imports demo/chat/scripts/llm_study.py)
# (or all of the above at once: .venv/bin/python test/gen_all_models.py)

# Run the scheduler on a model
.venv/bin/python inference_scheduler.py test/models/mixed_ops.onnx --out-dir /tmp/out

# A multi-entry project: one library, inference_run_<name>() per graph, one weight pool
.venv/bin/python inference_scheduler.py --entry decode=test/models/llama_tiny_decode.onnx \
    --entry head=test/models/llama_tiny_head.onnx --out-dir /tmp/multi

# Run all tests (1691 tests, none skipped; test_bert_base.py downloads bertsquad-12 — 435 MB — into
# demo/bert_squad/assets/ on its first run (demo/bert_squad/scripts/fetch_assets.py); BERT_SQUAD_DOWNLOAD=0 skips it instead)
.venv/bin/python -m pytest test/ -v

# Every host suite with one structured report (failures, unexpected skips, warnings, short runs;
# baselines in .claude/agents/run-tests/baselines.json) — the `run-tests` subagent drives the same helper
python3 ../.claude/agents/run-tests/run_tests.py --suite all      # scheduler, chat, lint, facts, csim, tts-host
```

Key source files:
- **`inference-scheduler/inference_scheduler.py`** — CLI entry point
- **`inference-scheduler/src/graph.py`** — ONNX loading, shape inference, Gemm preprocessing, tensor registry
- **`inference-scheduler/src/nodes.py`** — `ScheduledNode`, `MatmulNode`, `ConvNode`, `MatmulConvNode` (a MatMul on ConvKernel), `PoolNode`, `ReshapeNode`, `SpaceToDepthNode`; every kernel node lists its calls with `kernel_calls()` (`src/perf_calls.py`)
- **`inference-scheduler/src/matmul_lowering.py`** / **`cost_model.py`** — MatMul → ConvKernel engine choice and geometry (`conv_plans`, incl. the row split of accumulator-limited MatMuls); ConvKernel (conv-cycle-model port; `conv_board_cycles` adds the board's weight-request cost for the engine choice) and MatmulKernel (the RTL kernel's structural model fitted to its calibration; the HLS kernel's board-calibrated block model with `impl = "hls"`) cycle estimates; `shared_weight_layouts` (one layout per weight across entries)
- **`inference-scheduler/src/fc_conv.py`** — fully-connected Convs (one output pixel) → Flatten + MatMul + Reshape (+ bias Add) where the cost model says MatmulKernel is faster (LeNet 5.44 → 2.81 ms, `doc/plans/LENET_PLAN.md`)
- **`inference-scheduler/src/matmul_gemv.py`** / **`src/llm_entries.py`** — MatmulKernel GEMV / image pass (`MatmulNode.gemv_kw`); the Llama entry graphs that let decode read the prefill weight images (one copy per weight)
- **`inference-scheduler/src/host_nodes.py`** — host-CPU nodes (Softmax, LayerNorm, Gelu, Transpose, Slice, Gather, OneHot, Cast): numpy reference + C helper library side by side
- **`inference-scheduler/src/fusion.py`** — Constant folding, Split → Slice lowering, LayerNorm / GELU pattern fusion, VectorOP constant-broadcast normalisation
- **`inference-scheduler/src/tensor.py`** — Weight encoding (float → ap_fixed<16,8>), buffer declarations
- **`inference-scheduler/src/schedule.py`** — `Dag`: data-flow DAG (tensor edges plus RAW / WAR / WAW edges of persistent states; predecessors, topological order, independent pairs)
- **`inference-scheduler/src/numeric.py`** — `axi.numeric` metadata: per-tensor / per-channel power-of-two exponents (constant MatMul weights at the rank-1 exponent `f_w = f_out + 8 − f_in`), host-memory tensors (f32 / i32 / i16), persistent states
- **`inference-scheduler/src/llama.py`** / **`src/llm_nodes.py`** — Llama frontend and the `axi.llm` host ops (numpy reference + C helpers)
- **`inference-scheduler/src/vit.py`** / **`src/vit_nodes.py`** — vision-encoder frontend (`VitFrontend`, the `vision` entry) and its `Vit*` host ops
- **`inference-scheduler/src/piper.py`** / **`src/tts_nodes.py`** — Piper (VITS) text-to-speech frontend (`PiperChunkFrontend`, the `chunk` entry) and its `Tts*` host ops; `demo/tts/` holds the specification, the library (`libpiper_tts.so`) and the board gate
- **`inference-scheduler/src/codegen/`** — Multi-mixin code generator (header, source, buf_impl, test, cmake); `multi.py` — multi-entry projects (CLI `--entry NAME=MODEL.onnx`: weights deduplicated by name + image, states shared, intermediates overlapping)
- **Planning (`--plan`, optional)** — `src/planning.py`, `tactics.py`, `perf_calls.py`, `perf_model.py` / `perf_fit.py` / `host_model.py`, `order_search.py`, `codegen/timing.py`, `timeline_html.py` (the predicted execution as an interactive timeline, `timeline.html`): MatMul tactics and the issue order chosen from per-bitstream performance models measured once on the board (`perf_calibrate.py`, `perf_models/kv260/`); bit-identical results; without `--plan` every choice is unchanged — see `doc/plans/TACTICS_PLAN.md` §9

See `doc/scheduler/INFERENCE_SCHEDULER.md` for the full technical reference and `inference-scheduler/doc/SCHEDULER_DAG.md` for the DAG / event-stream / liveness / slot-coloring algorithm specifics.

### HLS Pragmas Used (the VectorOP C++ model)

`#pragma HLS INTERFACE m_axi offset=slave` (three 128-bit `burst_maxi` data ports, one bundle each, explicit burst length / outstanding counts), `#pragma HLS INTERFACE s_axilite` (pointer addresses + scalars + return, all in `bundle=ctrl`), `#pragma HLS dataflow`, `#pragma HLS PIPELINE II=1` (one flattened loop per stage), `#pragma HLS UNROLL` (8 lanes), `#pragma HLS bind_storage … fifo impl=lutram` (word FIFOs).

## Dependencies

- **`cmake/FindVitis.cmake`** — bundled in this repo; locates `vitis_hls`/`vitis-run` and sets `Vitis_HLS` / `Vitis_HLS_TCL_FLAG` (no CMakeLists includes it any more — every HLS synthesis target is retired; the C++ models need only the Vitis HLS headers). No external hlslib dependency.
- **Verilator 5.x** (`sudo apt install verilator`; 5.020 here) for the four RTL kernels' testbenches and lint; without it those targets are skipped.
- **Xilinx Vitis 2025.2** at `/mnt/data/xilinx/2025.2`. Source `settings64.sh` before building. From 2024.x, `vitis-run --tcl` replaces the older `vitis_hls -f` invocation; `FindVitis.cmake` handles this automatically via `${Vitis_HLS_TCL_FLAG}`.
- **KV260 board**: PL0 must be at 100 MHz (the MMCM's input; `upload_bitstream.py` checks and sets `/sys/devices/platform/fclk0/set_rate` from the HWH).  Ubuntu 22.04 (kernel 5.15.0-xilinx-zynqmp), XRT 2.13, `cma=1000M` on the kernel command line (BERT + SmolLM2 pools), 3.9 GB RAM and no swap (build generated projects `-j1`). Keep `board/kv260/kv260-no-cpu-powerdown.conf` installed in `/etc/tmpfiles.d/` (`demo/chat/deploy.py` does it): the PSCI core power-down idle state can park a core forever and hang the board (`doc/plans/CHAT_PLAN.md` §18).
- **One job per board**: every board tool (`run_remote_tests.py`, `run_remote_perf.py`, `perf_calibrate.py run`, `upload_bitstream.py`, the demos' `deploy_and_run.py`, the chat `deploy.py` / `llm_board.py`, the TTS `tts_board.py`) holds the per-board lock `/tmp/kv260-board-<host>.lock` (`inference-scheduler/src/remote/lock.py`; a config's `board_lock` overrides it, `false` disables it) and waits while another job holds it; tools started by a holder re-enter it.  Never wrap them in a shell `flock` on that file (they would wait forever) — use `python -m src.remote.locked -- CMD` from `inference-scheduler/` for other commands.

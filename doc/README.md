# Documentation

Start with the project [README](../README.md) (what the project is, results,
quick start).  This folder holds the reference documentation and the project
logs, grouped by topic:

```
doc/
├── build-and-test/   building, configuring and testing the hardware and software
├── kernels/          the kernels (Conv in HLS, MatMul, VectorOP and Pool in RTL, with their retired HLS versions): reference + optimisation log
├── scheduler/        the ONNX-to-C inference scheduler
├── plans/            project plans and their results (chronological logs)
└── images/
```

The scheduler's user-level guides live next to its code in
[`inference-scheduler/doc/`](../inference-scheduler/doc/), and each demo has
its own README under [`demo/`](../demo/README.md); the chat server's guides
are in [`demo/chat/doc/`](../demo/chat/README.md#documentation).

---

## Where do I find…

| I want to… | Read |
|---|---|
| build the C simulation, synthesise the kernels, make a bitstream or an overlay | [BUILD_TARGETS](build-and-test/BUILD_TARGETS.md) |
| run the tests (host, RTL, board) | [TESTING](build-and-test/TESTING.md); on the board: [REMOTE_TESTING](../inference-scheduler/doc/REMOTE_TESTING.md) |
| keep facts that code and docs repeat in step (counts, register maps, flags, bindings, pool sizes), or find what else a change must touch | [`tools/facts`](../tools/facts/README.md) (`facts.yaml`) |
| change a kernel bound or add a board | [PLATFORM_CONFIGURATION](build-and-test/PLATFORM_CONFIGURATION.md) |
| compile an ONNX model and use the generated C library | [USER_GUIDE](../inference-scheduler/doc/USER_GUIDE.md) |
| know which ops are supported and how a model maps to the kernels | [INFERENCE_SCHEDULER](scheduler/INFERENCE_SCHEDULER.md) |
| prepare a downloaded ONNX model | [MODEL_PREPARATION](../inference-scheduler/doc/MODEL_PREPARATION.md) |
| understand a kernel's interface, dataflow and limits | the kernel's reference in [kernels/](#kernels) |
| know why a kernel is built the way it is, and how fast each step made it | the kernel's optimisation log in [kernels/](#kernels) |
| profile a model on the board | [PROFILER](scheduler/PROFILER.md) |
| plan tactics and the issue order from measured performance (`--plan`), or calibrate a new bitstream | [INFERENCE_SCHEDULER "Planning"](scheduler/INFERENCE_SCHEDULER.md#planning---plan), [TACTICS_PLAN §9](plans/TACTICS_PLAN.md); `perf_calibrate.py` and its data: [`perf_models/`](../inference-scheduler/perf_models/README.md) |
| see the latest board results | the project [README](../README.md#results-on-the-board) |
| run the chat server on the board, or use it from a client | the chat server's [README](../demo/chat/README.md), [DEPLOY](../demo/chat/doc/DEPLOY.md) and [CLIENTS](../demo/chat/doc/CLIENTS.md) |
| fix a board that stops responding | [CHAT_PLAN §18](plans/CHAT_PLAN.md) and [`board/kv260/`](../board/kv260/) |
| debug a strange RTL simulation failure | [SIMULATION_ISSUES](build-and-test/SIMULATION_ISSUES.md) |

---

## build-and-test

| Document | Contents |
|---|---|
| [BUILD_TARGETS](build-and-test/BUILD_TARGETS.md) | Every CMake / make target: C-simulation tests, HLS synthesis and cosim, the RTL MatmulKernel, VectorOPKernel and PoolingKernel (Verilator, packaging), test fixtures, RTL behaviour tests, Vivado bitstream, device-tree overlay |
| [TESTING](build-and-test/TESTING.md) | The five test layers (scheduler unit tests, kernel C simulation, RTL simulation, on-board correctness and performance) and how to run each |
| [PLATFORM_CONFIGURATION](build-and-test/PLATFORM_CONFIGURATION.md) | `platforms/<name>.json`: FPGA part, clock and each kernel's compile-time bounds; adding a platform, changing a bound |
| [SIMULATION_ISSUES](build-and-test/SIMULATION_ISSUES.md) | Traps in the Vivado PS VIP simulation (partial write strobes, DDR aliasing, backdoor loads) and their workarounds |

## kernels

One reference (how the kernel works now) and one optimisation log (how it got
there, with measured numbers) per kernel:

| Kernel | Reference | Optimisation log |
|---|---|---|
| VectorOPKernel — SystemVerilog, element-wise ops | [VECTOROP_RTL_KERNEL](kernels/VECTOROP_RTL_KERNEL.md) | — ([VECTOROP_RTL_PLAN](plans/VECTOROP_RTL_PLAN.md)) |
| MatmulKernel — SystemVerilog, 128 MAC/cycle GEMM, GEMV / image path | [MATMUL_RTL_KERNEL](kernels/MATMUL_RTL_KERNEL.md) | — ([MATMUL_RTL_PLAN](plans/MATMUL_RTL_PLAN.md)) |
| PoolingKernel — SystemVerilog, max / average / Lp pooling | [POOL_RTL_KERNEL](kernels/POOL_RTL_KERNEL.md) | — ([POOL_RTL_PLAN](plans/POOL_RTL_PLAN.md)) |
| ConvKernel — SystemVerilog, 2-D and depthwise convolution, 512 MAC/cycle | [CONV_RTL_KERNEL](kernels/CONV_RTL_KERNEL.md) | — ([CONV_RTL_PLAN](plans/CONV_RTL_PLAN.md)) |
| ConvKernel in Vitis HLS — 2-D and depthwise convolution (retired from the hardware build in CONV_RTL_PLAN phase 3; its C++ is the RTL kernel's reference model) | [CONV_KERNEL](kernels/CONV_KERNEL.md) | [CONV_OPTIMISATION](kernels/CONV_OPTIMISATION.md) |
| MatmulKernel in Vitis HLS — tiled GEMM (retired from the hardware build in MATMUL_RTL_PLAN phase 4; its C++ is the RTL kernel's reference model) | [MATMUL_KERNEL](kernels/MATMUL_KERNEL.md) | [MATMUL_OPTIMISATION](kernels/MATMUL_OPTIMISATION.md) |
| VectorOPKernel in Vitis HLS — element-wise ops (retired from the hardware build in VECTOROP_RTL_PLAN phase 3; its C++ is the RTL kernel's reference model) | [VECTOROP_KERNEL](kernels/VECTOROP_KERNEL.md) | [VECTOROP_OPTIMISATION](kernels/VECTOROP_OPTIMISATION.md) |
| PoolingKernel in Vitis HLS — max / average / Lp pooling (retired from the hardware build in POOL_RTL_PLAN phase 3; its C++ is the RTL kernel's reference model) | [POOLING_KERNEL](kernels/POOLING_KERNEL.md) | [POOL_OPTIMISATION](kernels/POOL_OPTIMISATION.md) |

[HLS_CONV_RESEARCH](kernels/HLS_CONV_RESEARCH.md) is a background literature
survey of HLS convolution techniques (not a plan).

## scheduler

| Document | Contents |
|---|---|
| [INFERENCE_SCHEDULER](scheduler/INFERENCE_SCHEDULER.md) | Technical reference: supported ops, kernel mapping, MatMul on ConvKernel, host-CPU ops, numerics, the Llama frontend, the vision encoder, text to speech (Piper), planning (`--plan`), cache coherency, the generated code |
| [PROFILER](scheduler/PROFILER.md) | Per-layer wall-clock and DDR-bandwidth profiling of generated projects on the board |

In [`inference-scheduler/doc/`](../inference-scheduler/doc/):

| Document | Contents |
|---|---|
| [USER_GUIDE](../inference-scheduler/doc/USER_GUIDE.md) | CLI, generated project layout, C API, building and running the generated project |
| [ARCHITECTURE](../inference-scheduler/doc/ARCHITECTURE.md) | Code-generator internals: passes, node classes, layout engine, mixins |
| [SCHEDULER_DAG](../inference-scheduler/doc/SCHEDULER_DAG.md) | Data-flow DAG, event stream, tensor liveness and buffer-slot colouring |
| [BUFFER_REUSE](../inference-scheduler/doc/BUFFER_REUSE.md) | Buffer-pool slot reuse with worked examples (the algorithm is in SCHEDULER_DAG) |
| [MODEL_PREPARATION](../inference-scheduler/doc/MODEL_PREPARATION.md) | `simplify_onnx.py` and handling ops that survive simplification |
| [REMOTE_TESTING](../inference-scheduler/doc/REMOTE_TESTING.md) | On-board correctness and performance runners over SSH, bitstream upload |

## plans

Each plan records its goal, the design, and the measured outcome; the status
line at the top says what is done.

| Plan | Outcome |
|---|---|
| [THROUGHPUT_PLAN](plans/THROUGHPUT_PLAN.md) | MatMul, depthwise and VectorOP throughput tracks (2026-09-25/26): executed |
| [CONV_2D_GRID_PLAN](plans/CONV_2D_GRID_PLAN.md) | The 2-D MAC grid for ConvKernel: executed, then grown further by RESNET18_15FPS_PLAN |
| [RESNET18_15FPS_PLAN](plans/RESNET18_15FPS_PLAN.md) | ResNet-18 at 15 FPS: met (47.4 ms = 21.1 FPS) |
| [BERT_PLAN](plans/BERT_PLAN.md) | BERT-base SQuAD on the board: 971 ms per inference (839 ms p50 on 2026-10-06, the RTL ConvKernel bitstream), accuracy equal to float32 |
| [CHAT_PLAN](plans/CHAT_PLAN.md) | Chat app: OpenAI-compatible server, SmolLM2-135M on the FPGA (~10 tokens/s), sampling, attention, the board-hang workaround (§18), the one-copy GEMV decode (§19), SmolLM2-360M (§20), reproducible calibration (§21), SmolVLM-256M image chat (§22–§24, 3.9 s per image) and a generator that needs 4× less host RAM (§25) |
| [LENET_PLAN](plans/LENET_PLAN.md) | LeNet study (the `model-study` skill): numerics equal to float; fully-connected Convs run as MatMul (`--fc-conv`), 5.44 → 2.81 ms per image |
| [TTS_PLAN](plans/TTS_PLAN.md) | Text to speech: Audio8 TTS Preview 0.1B — NO-GO for real time (18 GB/s of weights needed, 2.9 GB/s available); candidate screen — Piper, TinyTTS, Kitten nano and Supertonic-3 fit; Piper lessac-medium study — GO (int16 within 0.19 dB log-mel of float); implementation — `libpiper_tts.so` bit-exact on the board, RTF 0.30; `/v1/audio/speech` in the chat server; the text encoder on the FPGA (int16, 0.37 dB log-mel from float, 3.7–4.9× faster) and the duration predictor in C (3×); first audio 1.0–1.5 s |
| [MATMUL_RTL_PLAN](plans/MATMUL_RTL_PLAN.md) | Replacing the HLS MatmulKernel with the SystemVerilog one: done — in the repository, bitstream `1d28630fbfa4` (timing met, 8.7 k LUT / 17.8 k FF fewer), bit-exact on the board with no workload slower, the scheduler's RTL cost model and one-copy engine choices (16-token LLM prefill −26…−32 %, MobileNet v1 −10 %), the default of the build, the scheduler and the board since phase 4; phase 5 gives the IP its m_axi bus parameters (crossbar slots 2 → 16 outstanding bursts, FC / GEMV up to 13 % faster, bitstream `b3309f424562`) |
| [VECTOROP_RTL_PLAN](plans/VECTOROP_RTL_PLAN.md) | Replacing the HLS VectorOPKernel with the SystemVerilog one, the way MATMUL_RTL_PLAN replaced MatmulKernel: a drop-in IP (same VLNV, registers and m_axi bus parameters, bit-exact, fewer resources); the hardware build's VectorOPKernel since phase 3, the HLS synthesis retired |
| [POOL_RTL_PLAN](plans/POOL_RTL_PLAN.md) | Replacing the HLS PoolingKernel with the SystemVerilog one, likewise: a drop-in IP (same VLNV, ports, registers and m_axi bus parameters, bit-exact; 11 956 LUT / 29 DSP against 17 208 / 94, 300 MHz out of context, the test stand 15.8 % faster); found an HLS line-buffer bank bug (stride_w > 8 with left padding, no shipped model affected); on the board bit-exact with every pool benchmark 8.6–30.2 % faster; the hardware build's PoolingKernel since phase 3 (bitstream `dbb320fb7297`), the HLS synthesis retired |
| [CONV_RTL_PLAN](plans/CONV_RTL_PLAN.md) | Replacing the HLS ConvKernel with the SystemVerilog one, the last HLS kernel: a drop-in IP (same VLNV, 200 ports, registers and m_axi bus parameters, bit-exact; 18 377 LUT / 518 DSP against 37 233 / 803, 300 MHz out of context, the test stand 23.1 % faster); on the board bit-exact everywhere, every conv benchmark 6–43 % faster (ResNet-18 59.7 → 47.4 ms, MobileNet v1 / v2 −43 / −39 %, Piper RTF 0.52 → 0.30); the hardware build's ConvKernel since phase 3 (bitstream `c2b2a6e5e50e`), the HLS synthesis retired |
| [TACTICS_PLAN](plans/TACTICS_PLAN.md) | Optional planning (`--plan`) from performance models calibrated once per bitstream: T0–T4 done (§9), simulator within 2 % of the board (except SmolLM2-360M and the Piper chunk), BERT and SmolVLM vision −1.1 % |

## Claude Code skills

Packaged workflows in [`.claude/skills/`](../.claude/skills/) (each
`SKILL.md` has the procedure; Claude Code loads them by name, e.g.
`/perf-regression`):

| Skill | Use it to |
|---|---|
| `model-study` | decide GO / NO-GO for a new model before porting it: operators, memory, fixed-point numerics against float, projected latency |
| `llm-onboard` | take a Llama-family model that passed `model-study` to a served chat model: calibrate, generate, bit-exact gates, board install, backend, deploy |
| `add-host-op` | add an operator that runs on the board's CPU (numpy reference + C helper, bit-exact, A53-fast, timing model, tests) |
| `kernel-verify` | verify a kernel change end to end (`matmul` / `vectorop` / `pool` / `conv`, all SystemVerilog): C-sim of the reference model, the Verilator testbench and lint, Vivado out-of-context synthesis, the test-stand behaviour test, timing diff |
| `conv-cycle-model`, `conv-rtl-trace` | predict ConvKernel cycles per layer; trace one RTL case's AXI / FIFO activity |
| `hls-rag` | ground HLS pragma / TCL edits in the indexed Vitis HLS user guide |
| `board-deploy` | build and load a bitstream, then run the on-board tests and demos |
| `perf-regression` | run the kernel benchmarks (and demo latencies) and compare them with the recorded baseline of the loaded bitstream |
| `perf-calibrate` | measure the per-bitstream performance model that `--plan` needs (after every new bitstream) |
| `docs-audit` | audit the docs against the code, or reproduce every documented path from a fresh clone |

Subagents in [`.claude/agents/`](../.claude/agents/):

| Agent | Use it to |
|---|---|
| `run-tests` | run all or selected test suites and get a structured JSON report. Suites: scheduler and chat pytest, ruff, the fact registry, kernel C-sim, Piper host emulation; RTL only on request. The report lists every failure (with a diagnosis and a flaky rerun), unexpected skip, warning and short run, measured against `run-tests/baselines.json`. The helper also runs on its own: `python3 .claude/agents/run-tests/run_tests.py --suite default`. |
| `code-audit` | audit code within a scope: the paths or subsystem named, by default the changes not yet on origin/main. It fixes stale facts in comments, docstrings and help texts, and removes provably dead code, keeping public APIs and reporting them. It reports interface and protocol inconsistencies with evidence from both sides: ctypes vs C headers, CLI flags between scripts, config keys, binary file writer vs reader, register maps, HTTP API, board paths. It checks its edits with the tests. Its checkers (`ctypes_check.py`, `unused.py`, `flags_check.py`, `config_keys.py` in `.claude/agents/code-audit/`) also run on their own. |

# Documentation

Start with the project [README](../README.md) (what the project is, results,
quick start).  This folder holds the reference documentation and the project
logs, grouped by topic:

```
doc/
├── build-and-test/   building, configuring and testing the hardware and software
├── kernels/          the four HLS kernels: reference + optimisation log each
├── scheduler/        the ONNX-to-C inference scheduler
├── plans/            project plans and their results (chronological logs)
└── images/
```

The scheduler's user-level guides live next to its code in
[`inference-scheduler/doc/`](../inference-scheduler/doc/), and each demo has
its own README under [`demo/`](../demo/README.md).

---

## Where do I find…

| I want to… | Read |
|---|---|
| build the C simulation, synthesise the kernels, make a bitstream or an overlay | [BUILD_TARGETS](build-and-test/BUILD_TARGETS.md) |
| run the tests (host, RTL, board) | [TESTING](build-and-test/TESTING.md); on the board: [REMOTE_TESTING](../inference-scheduler/doc/REMOTE_TESTING.md) |
| change a kernel bound or add a board | [PLATFORM_CONFIGURATION](build-and-test/PLATFORM_CONFIGURATION.md) |
| compile an ONNX model and use the generated C library | [USER_GUIDE](../inference-scheduler/doc/USER_GUIDE.md) |
| know which ops are supported and how a model maps to the kernels | [INFERENCE_SCHEDULER](scheduler/INFERENCE_SCHEDULER.md) |
| prepare a downloaded ONNX model | [MODEL_PREPARATION](../inference-scheduler/doc/MODEL_PREPARATION.md) |
| understand a kernel's interface, dataflow and limits | the kernel's reference in [kernels/](#kernels) |
| know why a kernel is built the way it is, and how fast each step made it | the kernel's optimisation log in [kernels/](#kernels) |
| profile a model on the board | [PROFILER](scheduler/PROFILER.md) |
| see the latest board results | the project [README](../README.md#results-on-the-board) |
| fix a board that stops responding | [CHAT_PLAN §18](plans/CHAT_PLAN.md) and [`board/kv260/`](../board/kv260/) |
| debug a strange RTL simulation failure | [SIMULATION_ISSUES](build-and-test/SIMULATION_ISSUES.md) |

---

## build-and-test

| Document | Contents |
|---|---|
| [BUILD_TARGETS](build-and-test/BUILD_TARGETS.md) | Every CMake / make target: C-simulation tests, HLS synthesis and cosim, test fixtures, RTL behaviour tests, Vivado bitstream, device-tree overlay |
| [TESTING](build-and-test/TESTING.md) | The five test layers (scheduler unit tests, kernel C simulation, RTL simulation, on-board correctness and performance) and how to run each |
| [PLATFORM_CONFIGURATION](build-and-test/PLATFORM_CONFIGURATION.md) | `platforms/<name>.json`: FPGA part, clock and each kernel's compile-time bounds; adding a platform, changing a bound |
| [SIMULATION_ISSUES](build-and-test/SIMULATION_ISSUES.md) | Traps in the Vivado PS VIP simulation (partial write strobes, DDR aliasing, backdoor loads) and their workarounds |

## kernels

One reference (how the kernel works now) and one optimisation log (how it got
there, with measured numbers) per kernel:

| Kernel | Reference | Optimisation log |
|---|---|---|
| VectorOPKernel — element-wise ops | [VECTOROP_KERNEL](kernels/VECTOROP_KERNEL.md) | [VECTOROP_OPTIMISATION](kernels/VECTOROP_OPTIMISATION.md) |
| MatmulKernel — tiled GEMM | [MATMUL_KERNEL](kernels/MATMUL_KERNEL.md) | [MATMUL_OPTIMISATION](kernels/MATMUL_OPTIMISATION.md) |
| ConvKernel — 2-D and depthwise convolution | [CONV_KERNEL](kernels/CONV_KERNEL.md) | [CONV_OPTIMISATION](kernels/CONV_OPTIMISATION.md) |
| PoolingKernel — max / average / Lp pooling | [POOLING_KERNEL](kernels/POOLING_KERNEL.md) | [POOL_OPTIMISATION](kernels/POOL_OPTIMISATION.md) |

[HLS_CONV_RESEARCH](kernels/HLS_CONV_RESEARCH.md) is a background literature
survey of HLS convolution techniques (not a plan).

## scheduler

| Document | Contents |
|---|---|
| [INFERENCE_SCHEDULER](scheduler/INFERENCE_SCHEDULER.md) | Technical reference: supported ops, kernel mapping, MatMul on ConvKernel, host-CPU ops, numerics, the Llama frontend, cache coherency, the generated code |
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
| [RESNET18_15FPS_PLAN](plans/RESNET18_15FPS_PLAN.md) | ResNet-18 at 15 FPS: met (60.3 ms = 16.6 FPS) |
| [BERT_PLAN](plans/BERT_PLAN.md) | BERT-base SQuAD on the board: 971 ms per inference, accuracy equal to float32 |
| [CHAT_PLAN](plans/CHAT_PLAN.md) | Chat app: OpenAI-compatible server, SmolLM2-135M on the FPGA (~10 tokens/s), sampling, attention, the board-hang workaround (§18) and the one-copy GEMV decode (§19) |

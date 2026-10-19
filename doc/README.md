# Documentation

Start with the project [README](../README.md) (what the project is, results,
quick start).  This folder holds the reference documentation and the project
logs, grouped by topic:

```
doc/
├── ARCHITECTURE.md   hardware architecture: requirements, interconnect, clocks / reset, memory, datapath, interfaces, verification, timing
├── build-and-test/   building, configuring and testing the hardware and software
├── kernels/          the five SystemVerilog kernels and the C++ reference models of four (the retired HLS kernels): reference + optimisation log
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
| understand the hardware: the A53 ↔ R5 ↔ PL interconnect, clocks and reset, memory, datapath, interfaces, verification, timing and utilization | [ARCHITECTURE](ARCHITECTURE.md) |
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
| compare the FPGA with the board's CPU (onnxruntime on the four A53s: BERT, SmolLM2, SmolVLM) | [`tools/cpu_baseline`](../tools/cpu_baseline/README.md); the results: [BERT_PLAN §4](plans/BERT_PLAN.md), [CHAT_PLAN §26](plans/CHAT_PLAN.md) |
| run the chat server on the board, or use it from a client | the chat server's [README](../demo/chat/README.md), [DEPLOY](../demo/chat/doc/DEPLOY.md) and [CLIENTS](../demo/chat/doc/CLIENTS.md) |
| talk to the board ("Hello Miranda") | [`demo/miranda`](../demo/miranda/README.md) |
| fix a board that stops responding | [CHAT_PLAN §18](plans/CHAT_PLAN.md) and [`board/kv260/`](../board/kv260/) |
| debug a strange RTL simulation failure | [SIMULATION_ISSUES](build-and-test/SIMULATION_ISSUES.md) |

---

## build-and-test

| Document | Contents |
|---|---|
| [BUILD_TARGETS](build-and-test/BUILD_TARGETS.md) | Every CMake / make target: C-simulation tests, IP export, the RTL MatmulKernel, VectorOPKernel, PoolingKernel, ConvKernel and RowOpsKernel (Verilator, packaging), test fixtures, RTL behaviour tests, Vivado bitstream, device-tree overlay |
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
| RowOpsKernel — SystemVerilog, the host row ops in fixed point (residual add, RMSNorm / LayerNorm, SiLU · up, dequant, RoPE, transpose, copy); the fifth kernel of the hardware build since bitstream `1cedaf2ed816` (`S_AXI_HP0_FPD`; the scheduler lane behind `kernels.rowops.unit`); op 9, FOLD (Piper's `TtsPrep`), since `de968ccc45da` (`kernels.rowops.fold`); descriptor chains since `33a799febc56` (`kernels.rowops.chains`); op 10, GATE, and the SCALE flag since `04259a7af917` (`kernels.rowops.gate` / `scale`) | [ROWOPS_RTL_KERNEL](kernels/ROWOPS_RTL_KERNEL.md) | — ([HOSTOPS_PLAN](plans/HOSTOPS_PLAN.md)) |
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
| [RESNET18_15FPS_PLAN](plans/RESNET18_15FPS_PLAN.md) | ResNet-18 at 15 FPS: met (20.2 ms = 49.6 FPS at 250 MHz) |
| [BERT_PLAN](plans/BERT_PLAN.md) | BERT-base SQuAD on the board: 971 ms per inference (266 ms p50 on 2026-10-15, at 250 MHz with its GELUs, softmaxes, LayerNorms and transposes on the FPGA and ConvKernel's x port past the CCI), accuracy equal to float32 |
| [CHAT_PLAN](plans/CHAT_PLAN.md) | Chat app: OpenAI-compatible server, SmolLM2-135M on the FPGA (~20 tokens/s), sampling, attention, the board-hang workaround (§18), the one-copy GEMV decode (§19), SmolLM2-360M (§20), reproducible calibration (§21), SmolVLM-256M image chat (§22–§24, 3.9 s per image) and a generator that needs 4× less host RAM (§25) |
| [LENET_PLAN](plans/LENET_PLAN.md) | LeNet study (the `model-study` skill): numerics equal to float; fully-connected Convs run as MatMul (`--fc-conv`), 5.44 → 2.81 ms per image |
| [TTS_PLAN](plans/TTS_PLAN.md) | Text to speech: Audio8 TTS Preview 0.1B — NO-GO for real time (18 GB/s of weights needed, 2.9 GB/s available); candidate screen — Piper, TinyTTS, Kitten nano and Supertonic-3 fit; Piper lessac-medium study — GO (int16 within 0.19 dB log-mel of float); implementation — `libpiper_tts.so` bit-exact on the board, RTF 0.11; `/v1/audio/speech` in the chat server; the text encoder on the FPGA (int16, 0.37 dB log-mel from float, 3.7–4.9× faster) and the duration predictor in C (3×); first audio 1.0–1.5 s |
| [ASR_PLAN](plans/ASR_PLAN.md) | Speech to text (the `model-study` skill): candidate screen on 200 LibriSpeech test-clean utterances — Moonshine beats Whisper (Whisper's 30 s window cannot be shortened: WER 5.05 → 67 %); Moonshine base study — GO with policy `opt` (int16 WER 2.89 % against float's 2.98 %; tiny 4.56 % against 4.47 %), a projected 0.87 s per utterance (RTF 0.10); the plan: frontend with length buckets, library, board gate, `/v1/audio/transcriptions`; phase 1 done (§4): `src/moonshine.py` + `src/asr_nodes.py`, both sizes bit-exact with the specification `moonshine_q.py`; phase 2 done (§5): `libmoonshine_<size>.so` (`generate_asr_project.py`, `asr_api.c`), its token ids bit-exact with the specification on the host emulation; phase 3 done (§6): on the board all 200 utterances bit-exact (`asr_board.py`), WER = the specification's (base 2.93 %, tiny 4.58 %), RTF 0.13 / 0.074, the performance model topped up; phase 4 done (§7): the chat server's `POST /v1/audio/transcriptions` (`moonshine_backend.py`, json / text / srt / vtt / verbose_json, streaming), deployed — 1.3 s for 10 s of audio |
| [MATMUL_RTL_PLAN](plans/MATMUL_RTL_PLAN.md) | Replacing the HLS MatmulKernel with the SystemVerilog one: done — in the repository, bitstream `1d28630fbfa4` (timing met, 8.7 k LUT / 17.8 k FF fewer), bit-exact on the board with no workload slower, the scheduler's RTL cost model and one-copy engine choices (16-token LLM prefill −26…−32 %, MobileNet v1 −10 %), the default of the build, the scheduler and the board since phase 4; phase 5 gives the IP its m_axi bus parameters (crossbar slots 2 → 16 outstanding bursts, FC / GEMV up to 13 % faster, bitstream `b3309f424562`) |
| [VECTOROP_RTL_PLAN](plans/VECTOROP_RTL_PLAN.md) | Replacing the HLS VectorOPKernel with the SystemVerilog one, the way MATMUL_RTL_PLAN replaced MatmulKernel: a drop-in IP (same VLNV, registers and m_axi bus parameters, bit-exact, fewer resources); the hardware build's VectorOPKernel since phase 3, the HLS synthesis retired |
| [POOL_RTL_PLAN](plans/POOL_RTL_PLAN.md) | Replacing the HLS PoolingKernel with the SystemVerilog one, likewise: a drop-in IP (same VLNV, ports, registers and m_axi bus parameters, bit-exact; 11 956 LUT / 29 DSP against 17 208 / 94, 300 MHz out of context, the test stand 15.8 % faster); found an HLS line-buffer bank bug (stride_w > 8 with left padding, no shipped model affected); on the board bit-exact with every pool benchmark 8.6–30.2 % faster; the hardware build's PoolingKernel since phase 3 (bitstream `dbb320fb7297`), the HLS synthesis retired |
| [CONV_RTL_PLAN](plans/CONV_RTL_PLAN.md) | Replacing the HLS ConvKernel with the SystemVerilog one, the last HLS kernel: a drop-in IP (same VLNV, 200 ports, registers and m_axi bus parameters, bit-exact; 18 377 LUT / 518 DSP against 37 233 / 803, 300 MHz out of context, the test stand 23.1 % faster); on the board bit-exact everywhere, every conv benchmark 6–43 % faster (ResNet-18 59.7 → 47.4 ms, MobileNet v1 / v2 −43 / −39 %, Piper RTF 0.52 → 0.30); the hardware build's ConvKernel since phase 3 (bitstream `c2b2a6e5e50e`), the HLS synthesis retired |
| [FMAX_250_PLAN](plans/FMAX_250_PLAN.md) | The whole design at 250 MHz: why the 100 MHz build read as "124 MHz" (a 10 ns constraint never pushed the placer; the worst paths were one LUT and 7 ns of wire), four hypotheses in parallel, an MMCM kernel clock, interconnect register slices, register trees in all four kernels, a PoolingKernel synthesis / simulation mismatch found on the board (and the netlist lockstep check it led to); bitstream `986cef4866a0` closes 4 ns (+0.105 ns) and is bit-exact everywhere: ResNet-18 47.5 → 20.5 ms, BERT 839 → 541 ms, SmolLM2-135M 10 → 18 tokens/s |
| [ACTIVATIONS_PLAN](plans/ACTIVATIONS_PLAN.md) | LeakyReLU, SiLU, GELU and GELU tanh on the FPGA: step 1 in VectorOPKernel — ops 6–9 and acts 3–6, the `alpha` register; the exact function rounded to nearest-even (double is exact for every Q8.8 input; BERT's float32 GELU constants give the same values), SiLU / GELU from a 4 096 × 7 table of `f(−|x|)`, LeakyReLU in a DSP; bit-exact on every input, 300 MHz out of context (+0.277 ns); the scheduler maps ONNX `Gelu` / `LeakyRelu` / `x · Sigmoid(x)` onto it behind a platform capability; bitstream `6436623029f7` (250 MHz, +0.061 ns; production 2026-10-06 to 2026-10-07): 156 / 156 models, the library gates bit-exact, BERT 541 → 525 ms with its GELUs on the FPGA |
| [KV_DECODE_PLAN](plans/KV_DECODE_PLAN.md) | Decode attention on the FPGA: the chat models' decode steps read the KV cache with ConvKernel (the prefill path with one row, policy `pow2+sink+p12`); bit-exact on the board, from ~130 positions faster than the host attention — at 1000: 135M 81.5 → 72.5, 360M 187.7 → 170.9, SmolVLM 81.3 → 72.2 ms per token |
| [OFFLOAD_PLAN](plans/OFFLOAD_PLAN.md) | SmolVLM's GELU onto VectorOPKernel's activation unit (two exponent-aware passes; image 2.33 → 2.19 s, bit-exact, quality unchanged); Piper's LeakyReLU stays in its host prep, its decoder's residual sums become VectorOP ADDs (one exponent per stage; chunk 327 → 272 ms, 0.2 dB SNR); the engine cost model refitted at 250 MHz (`tools/fit_cost_model.py`: per-call error halved, no engine choice moved) |
| [SOFTMAX_PLAN](plans/SOFTMAX_PLAN.md) | Softmax on the FPGA: VectorOPKernel's softmax unit (ops 10 / 11 — rows, and keys-major columns transposed; an integer softmax within 1 LSB, `src/vectorop_smx.py`), bitstream `588d721997cb`; BERT 525 → 427 ms, SmolVLM image 2.20 → 1.93 s, SmolLM2 prefill-256 −9 % (135M) / −6 % (360M), every library bit-exact, quality within each model's yardstick |
| [PS_PORTS_PLAN](plans/PS_PORTS_PLAN.md) | PS port layout: the kernels' outputs on `S_AXI_HP0_FPD` (bitstream `4a820240ba9c`: bit-exact, performance-neutral, dropped), then VectorOPKernel's b read port on `S_AXI_HPC1_FPD` (`hw/cormorant_hw_128/scripts/bd_vop_b_hpc1.tcl`): bitstream `8599aa7a5f12` (production 2026-10-08 to 2026-10-10), binary VectorOP 24–38 % faster, BERT 427 → 418 ms, ResNet-18 20.5 → 20.2 ms, every result bit-exact; §6: the outputs on `S_AXI_HP0_FPD` again on top of that design (a third interconnect, `axi_out_intercon`; bitstream `5de9b4f80908`: bit-exact, a 0.5–1 % loss, dropped); §7: the same outputs on `S_AXI_HP2_FPD` (DDRC port 4; `4a61fc965037`: bit-exact, the same small loss); §8: how the DDRC takes the PS ports — a CCI port is spread over two DDRC channels, an HP port has one; §9: static read QoS 15 on the CCI ports (AFIFM `RDQOS`, no bitstream change): memory-bound VectorOP calls −4…6 %, end to end −0.1…−0.2 % |
| [SCRATCHPAD_PLAN](plans/SCRATCHPAD_PLAN.md) | Study: an on-chip URAM scratchpad between the kernels (576 KB free) — the ideal-memory bound is 8–20 % on the depthwise CNNs and ~1 % elsewhere, most of it needing tiled producer → consumer scheduling; NO-GO: the x path reached that bound the same day without it (DEPTHWISE_PLAN §8.5) |
| [STEREO_PLAN](plans/STEREO_PLAN.md) | Stereo depth (the `model-study` skill): candidate screen (3D aggregation and warping rule most out), LightStereo-S GO with power-of-two exponents and per-channel weights; implemented — the LightStereo frontend (`src/stereo.py`), the `Stereo*` ops, per-channel Conv exponents (`chexp`), the `demo/stereo_depth` demo: 466 ms per pair at 640 × 480 on the board, EPE 0.667 px (float 0.650 on the same input), bit-exact with the simulation; a RealSense D435 on the board (§4.4; recaptured 2026-10-18): depth within 10 % of the camera's own on 91 % of the pixels |
| [DEPTHWISE_PLAN](plans/DEPTHWISE_PLAN.md) | Depthwise convolutions on ConvKernel: the analysis (board probes and Verilator ideal- / slow-memory runs: depthwise and few-channel 1 × 1 jobs are x-path bound, 1.4–3.3× their ideal), the ranked options, the transposed grid mode (built, 2–4× slower, reverted), and the fix (§8): overlapped line-buffer loads (`cv_patch`) plus ConvKernel's x port on `S_AXI_HP2_FPD` past the CCI — bitstream `f70ce632a97a` (production 2026-10-10 to 2026-10-12): the x-bound jobs −27…−67 %, LightStereo-S 599 → 469 ms, MobileNet v2 −14 %, BERT 418 → 364 ms, bit-exact; the UltraRAM row store designed and found unnecessary |
| [AREA_PLAN](plans/AREA_PLAN.md) | The kernels' LUTs: an area pass over the five kernels after the FOLD design stopped routing (97.7 k LUTs, 99.9 % CLBs) — block-RAM FIFO skid stages as LUTRAM rings, per-byte control writes, 48-bit address arithmetic, Matmul's packed accumulators, Conv's weight skew in the RAM addresses and the DSPs' A registers; done: bitstream `de968ccc45da` routes at 89.5 k LUTs (76.5 %, −4.4 k against `1cedaf2ed816` without FOLD), WNS +0.029 ns, every kernel call as fast as before |
| [HOSTOPS_PLAN](plans/HOSTOPS_PLAN.md) | Study: the host row ops (RMSNorm / LayerNorm, SiLU·up, residual adds, attention prep / merge, the TTS row prep, the transposes) on a streaming row-ops unit on `S_AXI_HP0_FPD` — the data movers and sigmoid the user asked for are ≤ 8 % and 0 % of any model; the float row ops are 30–80 % of the transformer entries; by the timed replay SmolLM2 prefill-256 −48 / −41 %, BERT −28 %, the Piper encoder −69 %, decode −23 / −14 %; fixed point throughout — phase 0 done: the residual stream as int32 at 2^−15 and the integer specs of every op (`src/rowops_spec.py`) pass the gates of all five models; phase 1 done: the RTL (`kernels/rowops_rtl`, [ROWOPS_RTL_KERNEL](kernels/ROWOPS_RTL_KERNEL.md)) is bit-exact, four lanes per op, the tables in UltraRAM, WNS +0.025 ns at 300 MHz, 31 k LUT / 6 BRAM / 7 URAM / 121 DSP; phases 2–4b done (the scheduler lane, the block design — bitstream `1cedaf2ed816` —, the prefill / vision / Piper attention prep on the unit: SmolLM2-135M prefill-256 776 → 398 ms, BERT 364 → 288 ms, SmolVLM's image 2.07 → 1.64 s); phase 5a (§11): Piper's `TtsPrep` as op 9, FOLD, in bitstream `de968ccc45da` (production 2026-10-15 to 2026-10-16, after AREA_PLAN): the chunk ≈ 260 → 183 ms, RTF 0.13; the replay is `perf_calibrate.py rowops` |
| [RPU_PLAN](plans/RPU_PLAN.md) | Plan: every workload onto the FPGA and the kernels driven from a Cortex-R5, OS-free, with interrupt-driven waits — phase 0 (the study) done, track A1 (the remaining host ops onto the row-ops unit, §8: BERT 265 ms, Piper encoder 81 ms) and track B1 (§9: the kernels driven from the Cortex-R5 — a model-independent firmware replays the command stream the Linux library records, interrupt-driven, bit-exact on the CNN demos; `inference-scheduler/runtime/rpu/`, `rpu_board.py`, `tools/rpu/`, `board/kv260/rpu/`) done, B2 done (§10), B3 started (§11: the R5 asleep between commands), the A53's interrupt waits (§12); track A2 done (§13: a two-level ConvKernel y pattern that writes the attention P·V in head order, RowOps descriptor chains, the decode attention prep on the unit; bitstream `33a799febc56`, production 2026-10-16) and A3a done (§14: the unit's GATE op and SCALE flag; the production bitstream `04259a7af917` since 2026-10-17, the Piper chunk 177 → 148 ms); the study: what is left on the CPU after the row-ops unit (1–8 % of the Llama / vision / BERT / CNN paths, half of the Piper encoder and chunk, 11 % of stereo) and its mapping onto existing ops or new kernel features (descriptor chains, a ConvKernel output stride, an unaligned row read, a windowed copy, tanh / sigmoid acts); the runtime model on the R5 (the never-compiled bare-metal target and its three blockers, the five PL interrupts the software never enables, the loader split, the RPMsg bridge, the sampler); the board's remoteproc / RPMsg support and its power (4.56 W idle, 4.59 W during a chat request) |
| [YOLO_PLAN](plans/YOLO_PLAN.md) | YOLOv5n object detection (the `model-study` skill): Concat and Resize as host ops, plain Q8.8 (COCO128 mAP@0.5:0.95 0.343 against float's 0.349), the `demo/object_detection` demo — 47 ms per image on the board, bit-exact with the simulation |
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
| `rowops-offload` | move a host op onto the row-ops unit: the node, its factory and frontend flag, the specification and study policy, the tests, the libraries and the board gate — with the contract's traps (aligned streams, raw images, per-column exponents) |
| `kernel-verify` | verify a kernel change end to end (`matmul` / `vectorop` / `pool` / `conv`, all SystemVerilog): C-sim of the reference model, the Verilator testbench and lint, Vivado out-of-context synthesis, the test-stand behaviour test, timing diff |
| `conv-cycle-model`, `conv-rtl-trace` | predict ConvKernel cycles per layer; trace one RTL case's AXI / FIFO activity |
| `hls-rag` | ground HLS pragma / TCL edits in the indexed Vitis HLS user guide |
| `board-deploy` | build and load a bitstream, then run the on-board tests and demos |
| `perf-regression` | run the kernel benchmarks (and demo latencies) and compare them with the recorded baseline of the loaded bitstream |
| `perf-calibrate` | measure the per-bitstream performance model that `--plan` needs (after every new bitstream) |
| `board-window` | one board session after a scheduler change: regenerate the libraries, gate / install / profile each on the KV260, top up the perf and host models, benchmark + baseline, the chat server back, the simulator check (`scripts/board_window.sh`) |
| `docs-audit` | audit the docs against the code, or reproduce every documented path from a fresh clone |

Subagents in [`.claude/agents/`](../.claude/agents/):

| Agent | Use it to |
|---|---|
| `run-tests` | run all or selected test suites and get a structured JSON report. Suites: scheduler and chat pytest, ruff, the fact registry, kernel C-sim, Piper host emulation; RTL only on request. The report lists every failure (with a diagnosis and a flaky rerun), unexpected skip, warning and short run, measured against `run-tests/baselines.json`. The helper also runs on its own: `python3 .claude/agents/run-tests/run_tests.py --suite default`. |
| `code-audit` | audit code within a scope: the paths or subsystem named, by default the changes not yet on origin/main. It fixes stale facts in comments, docstrings and help texts, and removes provably dead code, keeping public APIs and reporting them. It reports interface and protocol inconsistencies with evidence from both sides: ctypes vs C headers, CLI flags between scripts, config keys, binary file writer vs reader, register maps, HTTP API, board paths. It checks its edits with the tests. Its checkers (`ctypes_check.py`, `unused.py`, `flags_check.py`, `config_keys.py` in `.claude/agents/code-audit/`) also run on their own. |

# The kernels driven from the R5 cores — an OS-free, interrupt-driven runtime (plan)

**Status (2026-10-12): phase 0 DONE — the study (§1–§6).  Verdict: GO in two
tracks.  Track A (§3) moves what is left of the host ops onto the kernels,
in the order of their cost; the Llama / vision / BERT / CNN paths are within
reach with existing ops, the Piper chunk and the stereo demo need new unit
features (§3.3).  Track B (§4) makes the generated runtime run on a Cortex-R5
with interrupt-driven waits — the bare-metal target exists but has never
been compiled, targets the A53 BSP and has three known blockers (§4.1) — and
puts the chat path on it first (§5).  Power (§6): the board draws 4.56 W idle
and 4.59 W during a chat request; the saving is the APU's share, which the
R5 move only realises together with an APU frequency / suspend policy.**

The goal (the user, 2026-10-12): move as much as possible of every workload
onto the FPGA, reduce the CPU to driving the kernels, then move that driver
onto the low-power R cores, OS-free, with interrupts instead of polling.

## 1. Where the CPU time is today

After the row-ops unit's phases 0–4b (`HOSTOPS_PLAN.md`) the host ops' share
of every shipped workload, from the board profiles of 2026-10-12
(`inference-scheduler/perf_models/kv260/host.json` where measured; the
stereo numbers from its board profile of 2026-10-09):

| workload | wall | host ops | share | the ops |
|---|---:|---:|---:|---|
| SmolLM2-135M prefill 256 | 398 ms | 7 ms | 2 % | `LlmAttnMerge` 5.5, `LlmEmbed` 1.0, `LlmDequant` 0.8, `LlmSelectRow` |
| SmolLM2-135M decode | 49.4 ms / token | 4 ms (+ 2 ms sampler, Python) | 8 % | `LlmAttnPrep` × 30 2.9, `LlmDequant` 0.8, `LlmAttnMerge` 0.3, the 90 p12 softmaxes ~0.4 |
| SmolLM2-360M prefill 256 | 1 050 ms | 12 ms | 1 % | merge 9.7, embed 1.7, dequant 0.8 |
| SmolVLM vision | 1.64 s | 30 ms | 2 % | merge 12, `VitEmbedAdd` 7, patch embed 5, `VitSumDequant` 4, pixel shuffle 2 |
| Piper encoder, 400 ids | 96 ms | 47 ms | 49 % | `TtsAttnMerge` 23, `TtsAttnRelAdd` 17, `TtsEncOut` 4, `TtsEmbed` 2 |
| Piper chunk | 260 ms | 139 ms | 53 % | `TtsPrep` 77, the 19 host `TtsSum`s 30, `TtsGate` 20, `TtsPcm` 5, `TtsInterleave` 4.7, `TtsFlowOut` 2.3 |
| BERT | 288 ms | 15 ms | 5 % | 49 `Transpose`s 15 (29 in the host model's older profile), `Gather` 0.2 |
| LightStereo-S 640 × 480 | 466 ms | 53 ms | 11 % | `StereoCorrelation` 21, `StereoUpsample` 18, `StereoInstanceNorm` 12, `StereoPadEdge` 1.2 |
| ResNet-18 / MobileNet | 20 / 21 / 17 ms | 0.6 ms | 3 % | `SpaceToDepth` |

Besides the ops, the CPU issues and waits for every kernel call: the 135M
decode step makes 542 kernel starts and as many polled waits per token
(211 MatMul, 180 Conv, 151 RowOps) and 271 cache syncs; the chat path adds
per token the sampler (`libsampler.so`: penalties, DRY, top-k / top-p over
49 152 float logits — 0.8 ms greedy, 1.9 ms with the defaults, ~2.5 with
DRY) and the Python tokenizer's incremental decode (6 µs).

## 2. The primitives the kernels offer today

What a host op can be mapped onto without a bitstream change:

- **VectorOPKernel**: element-wise ADD / SUB / MUL / DIV / RELU / RELU6 on
  int16 runs (16-byte-aligned starts, `a_inc` / `b_inc` multiples of 8), the
  activation unit (LeakyReLU with alpha at 2^−16, SiLU, GELU, GELU-tanh;
  fused after a binary op), the softmax unit in row mode and in the
  keys-major column mode (16-column blocks).
- **RowOpsKernel**: COPY / ADDX with per-column shifts, an int bias, ReLU and
  a row window; DEQUANT to float32; MULACT (SiLU · up); NORM (rows ≤ 4096);
  ROPE; TRANSPOSE of int16 tiles into granules with byte strobes — a 3-level
  address pattern per side, 16-byte-aligned rows and strides.
- **ConvKernel / MatmulKernel**: the GEMMs, incl. the batched q·Kᵀ / P·V over
  cached keys with runtime key counts.

What none of them has: a runtime-indexed gather, an unaligned row start
(reading a diagonal of a matrix), tanh / sigmoid as standalone activations,
a multiply by a non-power-of-two constant with round-to-nearest, float32
arithmetic on inputs, rows longer than 4096 columns in a norm.

## 3. Track A — the host ops onto the kernels

### 3.1 With existing ops (scheduler work only)

| op | workload | ms per run | mapping |
|---|---|---:|---|
| `LlmDequant` | 135M / 360M / VLM decode + head | 0.8 | RO DEQUANT (per-column `sh_c` 8 / 9 / 10, exact) into a float32 DMA buffer; the sampler then reads DMA memory (one invalidate of 192 KB) |
| `LlmSelectRow` | prefill | 0.002 | RO COPY of one int32 row (the base from the runtime n) |
| `LlmAttnMerge` | prefill / vision | 5.5 / 9.7 / 12 | RO COPY per head (128-byte rows): 147 µs against the host's 180 at prefill 256 — worth it only once the CPU is an R5 (§4.3); the real fix is a ConvKernel output stride (§3.3) |
| `VitSumDequant` | vision | 1.2 | two RO ADDX (int32, exact) + DEQUANT; the image state becomes a DMA tensor |
| `VitPixelShuffle` | vision | 1.2 | RO COPY as several jobs (a 4-index shuffle over the 3-level pattern) |
| `VitEmbedAdd` | vision | 5.6 | RO ADDX with the position table as an int32 DMA stream (today a float32 host table) |
| BERT `Transpose`s | BERT | 15 | the head split / merge as RO COPY (128-byte rows), Kᵀ as RO TRANSPOSE (R 8, L 64) |
| `TtsSum` of the flow h (12 calls) | Piper chunk | 3 | RO ADDX with per-tensor shifts (the C's integer path, exact) |
| `TtsInterleave` (3) | Piper chunk | 4.7 | RO TRANSPOSE (R = the phase count 8 / 8 / 4, L ≤ 1024 tiles) |
| `TtsEncOut` x (1) | Piper encoder | 1 | RO DEQUANT into a DMA float32 buffer |
| `TtsAttnRelAdd` (12) | Piper encoder | 17 | q·ekᵀ as a 16-output-channel MatMul on ConvKernel (`[T][96] × [96][16]`, the 9 relative keys zero-padded), the 9 diagonals written into a zeroed `[C][T]` band buffer by 9 scattered TRANSPOSE jobs (R 1, granule stride 2(T + 1): the write side has byte strobes), one VectorOP ADD onto the scores — ≈ 11 jobs, ~100 µs per head-layer against 1.4 ms |
| decode `LlmAttnSoftmax` (90 / token) | decode | 0.4 | the softmax unit's row mode after a TRANSPOSE of the keys-major `[keys][3]` scores into 3 query rows; the numerics become the unit's integer softmax (`vsmx` for decode: a study policy, gated as the prefill's was) |

These remove every host op of the Llama prefills, the vision pass and
BERT, and 26 of the Piper encoder's 47 ms.  The decode step's `LlmAttnPrep`
(2.9 ms per token) stays a host op until the descriptor chains of §3.3 make
nine unit jobs one issue; the embeddings (`LlmEmbed`, `TtsEmbed`, BERT's
`Gather`: a runtime-indexed row gather, 1–2 ms) stay on the CPU — a few KB
of copies per token that the R5 does as well as the A53.

### 3.2 Numerics that change with the move

The decode softmax (p12 double table → the unit's integer softmax), the
decode RoPE (`irope` for decode as for prefill, §3.3), `TtsGate`'s tanh ·
sigmoid and `TtsPcm`'s tanh (a table act against libm), `TtsPrep`'s
LeakyReLU (double alpha 0.1 against 2^−16 steps), the stage averages (÷ 3).
Each is a study policy and a gate, as every move so far: `llm_study.py`
(`vsmx="decode"`, `irope=True`), `piper_study.py study` (log-mel distance
against float).

### 3.3 New kernel features (a bitstream each)

| feature | kernel | frees |
|---|---|---|
| **Descriptor chains**: one Start runs a job list from DDR | RowOpsKernel | the decode step's 30 one-row preps (2.9 ms per token), the issue cost of every multi-job node (10–27 jobs per layer today) |
| **An output channel stride** (`y` rows written at a programmable stride) | ConvKernel | every attention merge: P·V writes head order directly (−5.5 / −9.7 / −12 ms) |
| **Unaligned row start** (a lane offset on the A stream) | RowOpsKernel | reading diagonals: `TtsAttnMerge`'s banded MAC as COPY + a tiny MatMul (23 ms of the Piper encoder) |
| **A 2-D windowed copy with element strides and LeakyReLU** | RowOpsKernel | `TtsPrep` (77 ms per chunk: halo rows at unaligned starts, row strides of W·2 bytes, a runtime column mask, alpha 0.1 / 0.01, float32 sources) |
| **tanh and sigmoid acts** (acts 7 / 8 from 4 096-entry tables, as HOSTOPS §4.3 proposed for sigmoid) | VectorOPKernel | `TtsGate` (two acts + a MUL), `TtsPcm` (20 + 5 ms per chunk) |
| **A scaled add / average** (MUL by an int16 constant with round-to-nearest, or a DIV that rounds) | VectorOPKernel | the three ÷ 3 stage averages (25 ms per chunk) |
| **Long-row norms** (a two-pass norm over several 4 096-column rows) | RowOpsKernel | `StereoInstanceNorm` (12 ms) |
| **The correlation** (per-row banded `[160 × 24] · [24 × 160]`) | ConvKernel per-row q·Kᵀ (STEREO_PLAN names it) | 21 ms of the stereo pass |

The first two are the ones that matter for the chat path and the vision
pass; the next three are what the Piper chunk needs to leave the CPU at
all (139 ms of host time per 260 ms chunk today — on an R5 that would grow,
§4.3); the last two are the stereo demo's.

## 4. Track B — the runtime on the R5

### 4.1 What exists and what does not

- The generated project has an `INFERENCE_TARGET=BARE_METAL` build (the
  default of its CMake): Xil cache maintenance, `malloc` with VA = PA for the
  DMA buffers, the drivers' `_sinit.c`.  **It has never been compiled**: no
  CI step, no script, `host_emu` builds the Linux path against emulated
  drivers; every deploy uses LINUX.  It is written for the A53 BSP
  (`psu_cortexa53_0`, `aarch64-none-elf` in its comments).
- Three blockers found by inspection: `inference_init()` passes the UIO
  instance names (`const char *`) where the bare-metal `X<K>_Initialize`
  takes a `u16` device id or a `UINTPTR` base address; `inference_prof.c`
  calls `clock_gettime` without a guard; the chat library and `llm_bench`
  targets are LINUX-only in the generated CMake.
- OS dependencies of the generated code, by category: DMA allocation (XRT
  BOs on Linux; `malloc` + Xil cache ops on bare metal, unaligned); file I/O
  (`fopen` / `fread` of `weights/*.dat` into the pool and of host tables
  over 64 KiB — the 135M's bf16 embedding table is 56.6 MB; the weights dir
  is a compile-time path); threads (a pthread pool on Linux, serial on bare
  metal — the host ops then run on one core); timers (`clock_gettime`,
  profiling only); libm in double (`exp`, `tanh`, `erf`, `sqrt`,
  `nearbyint`, …); `getenv`, `fprintf(stderr)`.
- **`kernel_wait`** is one weak function that polls `IsDone`; nothing tracks
  the "most recent Start" at run time (the generator emits waits only where
  a lane is in flight).  Every kernel's control block has GIE / IER / ISR
  and a level interrupt output (the ap_ctrl_hs block of each `*_ctrl_s_axi.sv`),
  and every generated driver has `InterruptGlobalEnable`, `InterruptEnable`,
  `InterruptClear`, `InterruptGetStatus` — **no software enables them today**;
  `/proc/interrupts` shows 0 for the five `fabric_*` lines.
- **The interrupt lines**: `ilconcat_0` In0 … In4 (VectorOP, Matmul, Conv,
  Pool, RowOps) → `pl_ps_irq0[0..4]` = SPI 89 … 93 = GIC IDs 121 … 125.
  The device tree's UIO nodes say 0x59 (VectorOP) and 0x5d (RowOps)
  correctly but 0x60 / 0x61 / 0x62 for Matmul / Conv / Pool (GIC 128 / 129 /
  130) — wrong, unnoticed because nobody waits on them; the R5 build fixes
  them to 0x5a / 0x5b / 0x5c.
- **The board** (Ubuntu 22.04, kernel 5.15.0-1080-xilinx-zynqmp): the R5
  remoteproc driver is a module (`zynqmp_r5_remoteproc.ko`), `rpmsg_char` /
  `rpmsg_virtio` are modules, `CONFIG_ZYNQMP_IPI_MBOX=y`, the `zynqmp-ipi`
  node is in the device tree; **no R5 cluster node, no TCM, no reserved
  memory yet** — a device-tree overlay (the `rpu` node in split or lockstep
  mode, the TCM banks, a reserved DDR region for the firmware and the
  vrings) is the first thing to write.
- **The host tools**: Vitis 2025.2 has `armr5-none-eabi-gcc` 13.3, the
  standalone BSP v9.4 for the R5, `xilpm`, `xilmailbox`, OpenAMP 1.12 and
  libmetal 2.9 — everything a bare-metal R5 firmware with RPMsg needs.

### 4.2 The runtime model

- **Memory.**  The weight pool stays where it is: one DMA allocation Linux
  makes (CMA, `cma=1000M`: 294 / 758 / 511 / 53 MiB for the four libraries)
  and fills from the `.dat` files once; the R5 receives its physical address
  and the DMA states' and intermediates' offsets.  The R5 runs with VA = PA,
  its code in DDR (the 135M's `inference.c` is 6 MB of source; the straight-
  line entries do not fit the 128 KB TCMs), its stack and the small host
  arenas in TCM / OCM.  The DMA buffers are mapped non-cacheable in the R5's
  MPU, or cacheable with the `Xil_DCache*Range` calls the bare-metal target
  already emits — the first is simpler and costs little once the host ops
  are gone (§3), which is why Track A comes first.
- **Interrupts.**  `kernel_wait(k)` becomes: enable IER / GIE of kernel k at
  init; on wait, sleep in WFI until the GIC delivers SPI 121 + k, clear ISR,
  return.  The five SPIs route to the R5's GIC interface like any PL
  interrupt; `ap_done` is a clear-on-read latch, so the ISR must be cleared
  by the wait that consumed it.  The `run_matmul_at` loop's direct poll
  becomes a `kernel_wait` too.
- **The bridge.**  OpenAMP / RPMsg over the IPI mailbox: the A53 sends
  requests (`prefill ids[n]`, `decode id`, `image rgb` as a DMA buffer
  address) and receives results (the sampled token, or the logits' DMA
  address when the sampler stays on the A53); the ring and the vrings live
  in the reserved DDR region.  Per token the A53 wakes once, detokenises
  (6 µs) and writes the HTTP stream.
- **The sampler.**  Either on the A53 (today's `libsampler.so`, the A53
  wakes per token anyway) or on the R5 in fixed point from the int16
  logits: a 49 152-entry top-k selection (~0.3 ms at 600 MHz), the
  penalties over the recent ids, DRY's history scan, a 50-entry softmax —
  about 1 ms, which removes the logits from the per-token traffic.  The
  tokenizer stays Python on the A53 (encode once per prompt, decode per
  token).
- **The loader split.**  `inference_init()` splits into the A53 part (pool
  allocation, `.dat` loading, the DMA states' init images, the table load
  of the row-ops unit) and the R5 part (driver bases, interrupt enables,
  the pool's addresses); the generator emits both from the same layout.

### 4.3 What the R5 costs

- Issue: a kernel call is ~19 AXI-Lite register writes through the LPD →
  FPD path plus the Start; the decode step's 542 calls were ~3 ms of issue
  on the A53 and will be of the same order on the R5 — the descriptor
  chains of §3.3 are what shrinks it.
- The host ops left after Track A are copies and gathers (embeddings,
  the select row) of a few KB per token: fine.  Anything still computed in
  double would be 3–5 × slower than on one A53 core (600 MHz, VFPv3 double,
  no NEON, no threads): the Piper chunk's 139 ms of host ops would become
  ~0.5 s — that workload stays on Linux until §3.3's features exist.
- Latency: an IRQ wait adds the GIC delivery and the WFI wake (a few µs)
  over a tight poll; with 542 waits per token that is ≤ 2 ms, inside the
  margin the issue cost dominates anyway.

## 5. Phases and gates

| phase | what | gate | effort |
|---|---|---|---|
| **A1 — existing ops** | §3.1 as scheduler nodes: DEQUANT of the logits into DMA, the merges as COPY (behind the R5 flag), `VitEmbedAdd` / `VitSumDequant` / pixel shuffle, BERT's transposes, the Piper h sums and interleaves, the relative-key add through the diagonal transposes; the decode softmax on the unit (`vsmx="decode"` gate) | bit-exact sched-checks and board gates; BERT ≤ 275 ms (its 0.75 × gate of HOSTOPS §7), the Piper encoder ≤ 75 ms at 400 ids, nothing slower | 3 days |
| **A2 — descriptor chains + the ConvKernel output stride** | one bitstream: the unit's job list from DDR; `y` rows at a stride; the decode prep on the unit (`irope=True` gate) | WNS ≥ 0 at 4 ns; 135M decode ≤ 45 ms per token, the merges gone (prefill 256 ≤ 390 ms); every library bit-exact | 4 days incl. the build and campaign |
| **B1 — the R5 firmware for one model** | the device-tree overlay (rpu, TCM, reserved memory), the BARE_METAL fixes of §4.1 for the R5 BSP, the IRQ `kernel_wait`, the loader split, an A53 test harness over shared memory; MNIST convnet, then ResNet-18 | bit-exact with the Linux build; the five IRQs seen; latency within 5 % of the Linux build — **DONE 2026-10-13 (§9), as a model-independent firmware replaying the library's recorded kernel calls: the six CNN demos bit-exact, interrupt-driven, +2–5 % on the real models (YOLO +4 %, MobileNet v2 +5 %)** | 5 days |
| **B2 — the chat path on the R5** | the RPMsg bridge, the sampler on the R5 (fixed point, gated against `libsampler`), `kv260_chat_server.py`'s backend over RPMsg; SmolLM2-135M, then 360M and SmolVLM | the same tokens as the Linux library on the gate prompts; tokens / s not slower; the A53 idle between tokens — **DONE 2026-10-13 in the B1 design (§10): no RPMsg or sampler on the R5 (the A53 keeps the host ops and the output, by the user's direction); protocol v4 + the doorbell; every chat library, BERT and Piper gated bit-exact with the R5 driving; the chat server runs with it (`deploy.py --rpu`); decode +2 % (135M), the bench's A53 CPU 3 s instead of 10 s** | 5 days |
| **B3 — power** | the APU at 333 MHz (`userspace` governor) or suspended between requests with wake on RPMsg; the measurements of §6 before / after | ≥ 0.5 W off the 4.6 W idle, no latency loss | 2 days |
| **A3 — the Piper and stereo features** | §3.3's unit features (unaligned row start, the windowed copy, the acts, the scaled add, long-row norms) and the correlation on ConvKernel; Piper's chunk and the stereo pass on the R5 | the chunk's host time ≤ 10 ms; stereo ≤ 450 ms; bit-exact | 2 weeks |

Order: A1, then B1 and A2 in parallel (A2 is a bitstream; B1 needs none),
B2, B3, then A3.

## 6. Power on the board today (2026-10-12, read-only)

The KV260's INA260 (`hwmon2`, the carrier's 5 V rail): **4.56 W** idle with
the chat server loaded (median of 10 samples, 4.53–5.39), **4.59 W** median
during a SmolLM2-135M chat request (25 samples at 100 ms, max 5.15).  The
APU runs at 1.33 GHz under the `userspace` governor (333 / 444 / 666 /
1 333 MHz available), idle state WFI only (`cpu-sleep-0` disabled since
CHAT_PLAN §18's hang).  The PL at 250 MHz with 833 DSPs is the larger
static share.  So the dynamic cost of inference is small; the saving of the
R5 move is the APU cluster's idle power, which needs B3's frequency /
suspend policy to show — to be measured, not assumed.

## 7. Open questions for phase 1

- Lockstep or split R5 (one core suffices; lockstep halves the TCM per
  core but doubles the fault coverage — split, with the second core idle).
- Whether the R5 maps the DMA buffers non-cacheable (simplest, slower copies)
  or cacheable with range maintenance (today's bare-metal code path).
- The sampler's home (§4.2) — decide after B1 measures the R5's copy and
  double throughput on the board.

## 8. Track A1 — the host ops onto existing kernel ops (2026-10-12/13)

Status: the scheduler work is in; the board gates run in the evening of
2026-10-12 (numbers below as they come in).  What moved, each a bit copy or
the host op's value bit for bit (the specification unchanged unless said):

| op | node | jobs |
|---|---:|---|
| BERT's `Transpose`s (48 of 49) | `TransposeRoNode` | a head split / merge: one COPY (rows D1, blocks D2); Kᵀ: one TRANSPOSE of 8-row tiles; the odd one (`[2, 0, 1]`-like perms, a last dim not a multiple of 8) stays the host op |
| Piper chunk `TtsSum` (12 of the 19 host ones: the two-input flow sums with differing exponents) | `TtsSumRoNode` | one ADDX with per-tensor shifts at the max exponent; the four masked post sums and the three ÷ 3 stage averages stay host ops (§3.3) |
| Piper chunk `TtsInterleave` (3) | `TtsInterleaveRoNode` | one TRANSPOSE, R = the phase count, the columns in ≤ 1 024-wide chunks as blocks |
| Piper encoder `TtsAttnRelAdd` (12) — **numerics**: `piper_vits.encoder_forward(rel_kernel=True)` | `TtsRelBandRoNode` + 16 extra K cache rows (`VitAttnPrepRoNode` copies the 9 relative keys in, raw at f_s + 8 − f_q, so the q·Kᵀ call's rows T .. T + 8 are q·ekᵀ at 2^-f_s — `piper.raw_rel_keys`) | 9 TRANSPOSEs of R = 1 (granules 2(T + 1) bytes apart: byte-strobed) onto the diagonals of a zero band state `enc{T}.ro.band{g}`, one ADDX of the band onto the scores; the softmax masks keys / queries ≥ n |
| `LlmDequant` (the Llama logits; the prefill / head / decode entries) | `DequantRoNode` (A i16 / i32 → C f32; a 49 152-wide row as 12 rows of 4 096 at its one exponent) + `LlmToHostNode` (invalidate + memcpy into the caller's `float *`) | the new `dma: f32` kind: a float32 DMA tensor / state (4 bytes per element in the pool) |
| `VitSumDequant` (the connector's three parts) | `ResAddBiasRoNode` (zero bias) + `ResAddRoNode` into an int32 row at the finest part exponent, `DequantRoNode` into the image-feature **state** (`vlm.img`, `dma: f32`); the text model's `LlmEmbed` reads it from DMA memory after an invalidate | 3 jobs |
| `LlmSelectRow` (the prefills' last row) | `SelectRowRoNode` | one COPY (i32) with the base from the runtime n |
| `VitPixelShuffle` (3) | `PixelShuffleRoNode` | one COPY per source block the column window touches (the output rows as rows j × blocks i): 5–7 jobs per part |

Numerics gate (the only change of specification): `piper_study.py
encoder-vsmx`, the "VectorOP softmax + unit + kernel rel" policy — 5 / 1 138
durations changed against float (max 1 frame), log-mel distance 0.408 dB
(the host: 0.388, the unit's LN 0.362; yardstick 2.64 dB), max |x − float|
0.0265 — GO.

Studied and declined: `VitEmbedAdd` as an ADDX with the position table as
an int32 DMA stream (the table rounded once at 2^-15 — `vlm_study.py`
policy `pow2+p12+vgelu+vsmx+rowops+iembed`, 40 min): image features rel.
error 0.0547 (0.0548 shipped), answer top-1 0.969 (0.971), KL 0.0038
(0.0033), greedy answers identical to float 5 / 24 (8 / 24), ppl 12.208
(same) — a small but real step down for 5.6 ms of a 1.64 s image, so it
stays an opt-in (`VitFrontend(iembed=True)`, `EmbedAddRoNode`,
`src/vit.py IEMBED_DEFAULT = False`; `test_vit.TestVisionRowopsIembed`
keeps it bit-exact against the study).  Left on the host, by design: the attention merges (the plan's §3.1: a wash on the A53, A2's
ConvKernel output stride removes them), the decode step's prep and
softmax (A2: descriptor chains), `TtsEncOut` (its stats add a float32 bias
after the dequant: a double rounding the unit cannot reproduce; the x half
alone is 1 ms) and the embeddings.

Verification: every suite (`run_tests.py --suite all`), the generated
projects on the host emulation (`test_piper`, `test_vit`,
`test_rowops_unit`: every new node's C against the simulation, incl. the
byte-strobed diagonals and the float32 DMA rows), the sched-checks and the
board gates bit for bit.

One board lesson: a stream A of the unit must start 16-byte aligned (the
contract's §1.4), which the host emulation did not check — the first band
node read a negative key offset's row from element |d| and passed every
host test while the board disagreed on every sequence.  The band state now
carries W guard rows above the matrix so each diagonal job reads its row
from the start (the granules of i < |d| land in the guard rows), and
`test/host_emu.py` aborts on an unaligned base or stride.

Board results (bitstream `1cedaf2ed816`, 2026-10-12 evening):

| workload | before | after | gate |
|---|---:|---:|---|
| BERT-base, 20 SQuAD windows of 256 (`run_demo.py --regenerate --n 20 --profile-layers`) | 288 ms | **265.4 ms** (min 265.2, max 265.6); bit-exact 3 / 3 vs the simulation, 20 / 20 vs the emulation | ≤ 275 ms ✓ |
| Piper encoder (`tts_board.py --profile`), 20 / 60 / 88 / 162 / 268 / 400 ids | 9.5 / 14.4 / 28.3 / 56.8 / 92.1 / 95.8 ms | **8.2 / 12.1 / 23.8 / 47.6 / 78.5 / 81.1 ms**; every encode and duration bit-exact (12 / 12 sequences) | ≤ 75 ms at 400 ids ✗ (−15 %; the band 17 → 2.65 ms, the merge's 23 ms is A2's) |
| Piper chunk (the sums and interleaves on the unit) | 260 ms, RTF 0.221 / 0.192 | 257 ms (217.9 … 278.0), RTF 0.224 / 0.194; PCM bit-exact 2 / 2 | bit-exact ✓, no faster (the chunk's CPU time is `TtsPrep` / `TtsGate` / the ÷ 3 sums: A3) |
| SmolLM2-135M (`llm_board.py --profile`) | prefill 16 / 64 / 256: other host 0.77 ms (the dequant) | the same prefills with "other host" 0.12 ms (the copy out); logits bit-exact 4 × 33 / 33 | bit-exact ✓ |
| SmolLM2-360M | other host 0.77 ms | 0.10 ms; logits bit-exact 4 × 33 / 33 | bit-exact ✓ |
| SmolVLM-256M `llm_image` (the connector sum + DEQUANT into the DMA image state, the pixel shuffles) | 1695 / 1701 ms | **1641 / 1639 ms**; logits bit-exact 2 × 33 / 33 (COCO 39769, 1268) | bit-exact ✓ |

The performance model was topped up with the new calls (`perf_calibrate.py
cases` / `run --resume` / `fit`, one refinement round: 2 131 → 2 187 exact
calls, the `rowops` family at 182), the host model merged from the five
profiles (170 exact signatures, 41 kinds; `LlmToHostNode` priced), and the
simulator agrees with the board within −4.3 … +1.4 % on every phase (BERT
+0.8 %, the Piper encoder −0.4 … −4.0 %, the chunk −2.7 %, no unpriced
node).  The kernel benchmark (72 cases) passed twice; between the two runs,
ten minutes apart with no kernel change, four MatmulKernel cases moved by
+2 … +7 % and one pool case by ±8 % — run-to-run noise of this bitstream's
small cases (the 4a window saw the same); the 22:00 run is the recorded
baseline.  The chat server is back up with every library regenerated
(SmolLM2-135M / 360M, SmolVLM, Piper, the BERT project).

## 9. Track B1 — the kernels driven from the Cortex-R5 (2026-10-12/13)

**Result: every CNN demo runs with its kernels driven by a bare-metal
firmware on RPU core 0, bit-exact with the same library driving them from
the A53 (20 / 20 inferences each), every wait woken by the kernel's
interrupt (0 missed), within 2–5 % of the A53's latency on the real
models.**  The firmware knows no model.

| project | calls / inference | batches / inference | A53 drives | R5 drives | Δ |
|---|---:|---:|---:|---:|---:|
| MNIST convnet | 8 | 1 | 0.111 ms | 0.249 ms | +124 % |
| LeNet | 10 | 1 | 1.259 ms | 1.418 ms | +13 % |
| ResNet-18 | 41 | 1 | 20.00 ms | 20.41 ms | +2.0 % |
| MobileNet v1 | 57 | 1 | 20.47 ms | 21.00 ms | +2.6 % |
| MobileNet v2 | 100 | 1 | 17.48 ms | 18.38 ms | +5.2 % |
| YOLOv5n | 127 | 14 | 47.00 ms | 48.91 ms | +4.1 % |

(20 runs each, ramp inputs offset per run, bitstream `1cedaf2ed816`; a
batch ends before each host op — YOLO's Concat / Resize — and at the output
sync.)  The cost is per call and per batch: about 4 µs a call (the R5's
40-odd AXI-Lite writes across the LPD–FPD bridge and the interrupt wake)
and some 50 µs a batch (the A53 polls the control block through /dev/mem).
Both have obvious cuts — sending only the registers that changed, waking
the A53 through the IPI (track B2) — and neither matters once the inference
is tens of milliseconds.

### 9.1 The design

The first cut (2026-10-12) put the generated library itself on the R5: a
bare-metal `inference_init` / `inference_run` with the weights embedded in
the ELF, the DMA buffers carved from a region the A53 handed over, the
I/O through a control block — the MNIST convnet ran that way, bit-exact
(0.141 vs 0.110 ms).  It does not scale: the TCMs hold 64 + 64 KiB, the
model's code and weights would need a reserved DDR region in the boot
device tree (a reboot), the loader a split, and every host op would have
had to run on the R5.  The user's direction — *the R5 needs no weights:
transfer the list of calls and their parameters, the R5 does its job, the
A53 processes the output* — is what shipped:

- **The firmware** (`inference-scheduler/runtime/rpu/`: `rpu_main.c`,
  `rpu_irq.c`, the protocol `rpu_ctrl.h`, `lscript.ld`, CMake + the R5
  toolchain; 10 KiB of code, TCM-resident; one build per bitstream against
  the BSP of `tools/rpu/`) replays a **command stream**: `REGS lane n`
  followed by the kernel's argument registers from offset 0x10, `START
  lane`, `WAIT lane`.  Its lanes are the bitstream's five kernels (bases and
  PL lines from `xparameters.h`; a control-block mask says which exist).  A
  batch is `ring[0 .. batch_words)` in a DDR buffer the A53 names in the
  control block (the BTCM, polled); the firmware invalidates the batch, arms
  the kernels' GIE for its duration, writes the registers, starts, and waits
  in WFI for the lane's ap_done interrupt (the GIC setup and its traps of
  §9.2 unchanged).
- **The recorder** (`runtime/inference_rpu.[ch]`, compiled into every
  Linux library): `inference_rpu_attach()` swaps each driver instance's
  `Ctrl_BaseAddress` to a shadow page (seeded from the kernel's registers),
  so the generated code's register writes go to memory; the generated
  `X*_Start()` calls are macros that, while attached, copy the shadow
  page's argument block (the register map's exact length,
  `KernelDesc.rpu_reg_words` — the control blocks decode 7–8 address bits,
  a write past the map aliases) into the ring with a START; the default
  `kernel_wait()` records a WAIT.  `INFERENCE_RPU_FLUSH()` — emitted
  before every host op and before the output sync, and called when the
  ring fills — submits the batch and waits for `done_seq`.  Nothing else
  changes: the weights, the pool, the host ops, the cache syncs and the
  outputs are the library's as before, so multi-entry chat libraries get it
  for free (`inference_rpu_attach()` once after `inference_init()`).
- **The gate** (`rpu/rpu_host.c` in every project, `rpu_board.py` on the
  board, under the board lock): N inferences, each first on the A53 (the
  reference), then attached, compared bit for bit, both timed, plus the
  firmware's batch / call / wait / interrupt counters.

### 9.2 Traps (each cost a round trip on the board)

1. The xlnx 5.15 remoteproc reads the singular `power-domain` on the TCM
   nodes and translates the banks' global addresses to the R5's 0x0 /
   0x20000 itself: the ELF uses the R5's own addresses.
2. The RPU GIC keeps its state across remoteproc stop / start: an interrupt
   a previous firmware acknowledged and never completed stays *active* with
   the running priority at the kernels' level, masking everything below —
   even a software interrupt.  The firmware completes and clears every
   active interrupt and drains the pending ones before the driver's init
   (which also skips its distributor setup when it finds the distributor
   enabled — so it is disabled first).
3. The kernels' level interrupts can fire a second time right after the ISR
   clear while the line is still seen high: the handler counts a completion
   only when it saw ap_done in the ISR register, else the next wait on that
   lane would return early.
4. The lines are shared with the APU: a firmware with the kernels' GIE on
   consumes the completions of jobs the Linux library issues (the ISR's
   ap_ctrl read clears ap_done, the Linux poll hangs).  The firmware arms
   GIE only around a batch and never reads ap_ctrl.
5. The R5's PMU cycle counter did not count from the firmware; the A53
   times the runs.
6. A write to a register offset past a kernel's map lands on a register:
   the AXI-Lite blocks decode `awaddr[6:2]` / `[7:2]` only.  The recorder
   copies exactly the map (`test_kernels` checks the counts against the
   driver generators).

### 9.3 Who drives: small models on the A53, large ones on the R5 (2026-10-13)

The R5's cost is per call and per batch, so it matters for small models and
vanishes into large ones.  `inference-scheduler/src/rpu_policy.py` decides
per entry at generation time:

- **The overhead model.**  8.4 µs a kernel call, 61 µs a batch — the
  least-squares fit of the six measurements above, within 2 % on the four
  large models and within 10 % on the two small ones.  A batch ends before
  every host op that follows kernel work and at the output sync.
- **The prediction.**  The event stream replayed with the bitstream's
  performance model (`perf_models/kv260/`, as `--plan` uses it) and the
  host-op model; a call the performance model cannot price falls back on
  the analytic cost model.  With nothing priced, ≥ 32 calls per batch
  counts as large.
- **The rule.**  The R5 drives an entry when its overhead is at most 5 % of
  the predicted inference time.  The header carries `INFERENCE_RPU_CALLS /
  _BATCHES / _PREDICTED_US / _OVERHEAD_US / _RECOMMENDED` (with an
  `_<ENTRY>` suffix per entry of a multi-entry project), `report.md` a
  "Cortex-R5" paragraph.
- **The runtime.**  The generated `inference_init()` calls
  `inference_rpu_select()`: when the firmware runs it attaches in auto
  mode, and the top of every `inference_run*()` picks that run's driver
  from the entry's recommendation (`INFERENCE_RPU_ENTER`).  Without the
  firmware nothing changes.  `INFERENCE_RPU=0` never attaches,
  `INFERENCE_RPU=1` puts every run on the R5; `inference_rpu_set_mode()`
  does the same from code.  `inference_deinit()` detaches.  The ring is
  named in the control block at every flush, so several libraries in one
  process (the chat server) share the firmware.

On the board (20 runs each, bitstream `1cedaf2ed816`; `rpu_board.py
--mode both`: one pass with the R5 forced, one with the policy):

| project | predicted | overhead | policy | forced R5 | policy run | A53 |
|---|---:|---:|---|---:|---:|---:|
| MNIST convnet | 0.10 ms | 0.13 ms (125 %) | A53 | 0.248 ms | 0.112 ms | 0.110 ms |
| LeNet | 1.22 ms | 0.15 ms (11.9 %) | A53 | 1.405 ms | 1.254 ms | 1.253 ms |
| ResNet-18 | 20.08 ms | 0.41 ms (2.0 %) | R5 | 20.38 ms | 20.40 ms | 19.99 ms |
| MobileNet v1 | 20.54 ms | 0.54 ms (2.6 %) | R5 | 20.97 ms | 21.00 ms | 20.47 ms |
| MobileNet v2 | 17.56 ms | 0.90 ms (5.1 %) | A53 | 18.34 ms | 17.43 ms | 17.46 ms |
| YOLOv5n | 42.10 ms | 1.92 ms (4.6 %) | R5 | 49.07 ms | 48.98 ms | 46.97 ms |

Every pass bit-exact; in the policy pass each project ran all 20 runs where
its header said (0 or 20 on the R5).  The predictions are within 1 % of the
A53 measurements except YOLO's (−10 %: its host ops, Concat and Resize,
are priced low), which only makes its overhead fraction look larger.
MobileNet v2 sits just over the limit with 100 calls: the cheaper protocol
of B2 (only the changed registers, the IPI instead of polling) moves it
and the 5 % limit is the knob (`MAX_OVERHEAD`).

Next: the chat libraries regenerated (the policy attaches them by itself
once the firmware runs at boot), the cheaper protocol (changed registers
only, the IPI instead of polling: B2), then B3 power with the A53 cores
idle during a batch.

## 10. Track B2 — the chat path on the R5 (2026-10-13)

**Result: the chat server runs with the Cortex-R5 driving the kernels.**  The
regenerated SmolLM2-135M, SmolLM2-360M and SmolVLM libraries, BERT and
Piper all passed their board gates bit-exact with the firmware running (the
chat libraries also with the R5 forced for every entry), and
`demo/chat/deploy.py --rpu BSP_DIR` (or `rpu.enabled` in `chat_config.json`)
starts the firmware with the server.  A chat request moves the firmware's
counters (four requests: 17 304 batches, 65 472 kernel calls, 0 missed
interrupts, the A53 asleep on 3 072 of the batches).

| library (R5 driving, 2026-10-13) | gate | decode | prefill 16 / 64 / 256 | A53 CPU of the bench |
|---|---|---:|---:|---:|
| SmolLM2-135M | logits bit-exact, 4 prompts | 51.1 ms/token (A53: 49.9) | 83 / 100 / 400 ms (A53: 80 / 98 / 397) | 3.0 s (A53: 10.1 s) |
| SmolLM2-360M | logits bit-exact | 132.2 ms/token | 194 / 238 / 1054 ms | 5.2 s |
| SmolVLM-256M | logits bit-exact, `llm_image` 1617 ms | 51.1 ms/token | 83 / 100 / 401 ms | 2.9 s |
| BERT (SQuAD) | 20 / 20 bit-exact | 266.1 ms / inference | | |
| Piper | PCM bit-exact | | | |

What the plan's B2 had and what shipped: the plan put the sampler and an
RPMsg bridge on the R5 so the whole token ran there; the design that
shipped (§9) keeps the host ops and the output on the A53 and has the R5
replay the kernel calls — so B2 became making that cheap enough for decode
(543 calls and 151 batches a token) and letting the A53 sleep.

### 10.1 Protocol v4: the overhead an order of magnitude down

| change | why |
|---|---|
| the kernels' register window (0xA000_0000, 1 MiB) mapped as device memory in the R5's MPU | the BSP maps the PL strongly ordered: every register write waited for its completion |
| `REGSO lane n` + first index: only the registers that changed since the lane's last call (the whole block on a lane's first call of a run) | 28–45 register words a call; the control blocks change none themselves |
| the command ring an uncached buffer (`inference_buf_alloc_uncached`) | the A53 cleaned the whole 1 MiB ring from its cache every batch |
| a lane's GIE armed at its first Start of a batch, dropped at the end only when it is not in flight | arming all five lanes per batch; and a lane in flight across a host op lost its completion |

Measured (R5 minus A53): chains of 1 / 16 / 128 tiny VectorOP calls +77 / +176 /
+917 µs → +3 / +21 / +146 µs; ResNet-18 +410 → +36 µs, MobileNet v2 +900 →
+111 µs, YOLOv5n +1910 → +276 µs.  The policy's fit (§9.3) became 1.0 µs a call
and 10.6 µs a batch: every chat entry now goes to the R5, and only
sub-millisecond models (the MNIST convnet, chains of tiny calls) stay on the A53.

**The doorbell.**  While the R5 runs a batch the A53 spins for
`INFERENCE_RPU_SPIN_US` (50 µs), then sleeps in `poll()` on the `rpu_ipi` UIO
device (`board/kv260/rpu/rpu-ipi.dts`: IPI channel 7, the PL0 agent at
0xFF34_0000, SPI 29 — Linux's own mailbox is channel 0, SPI 35); the R5 rings it
(its channel's TRIG, channel 7's bit) after a batch when the A53 asked
(`wake` in the control block, with the done / wake check fenced on both sides).
ResNet-18 costs the A53 0.65 ms of CPU per 20 ms inference instead of 20 ms;
YOLOv5n 20.6 ms instead of 54.5 ms (its host ops).  The wake costs 20–70 µs per
sleeping batch, which the policy does not model yet: LeNet (10 calls, one
1.3 ms batch) runs +3.7 % on the R5 against the 1.6 % predicted.

### 10.2 The interrupt handling, third design

The first two designs counted completions in the interrupt handler and hung
under load (the 135M and 360M 16-token prefill, SmolVLM): a VectorOP job
finished — ISR set, line high — and the R5 waited forever.

1. *Toggle-clear in the handler.*  With posted writes the ISR clear's
   read-back can overtake the write (AXI reads and writes are separate
   channels); a re-taken level interrupt then read the bit still set, toggled
   it back and counted one completion twice — a wait returned early, the next
   Start went to a busy kernel and was wiped by its ap_ready.
2. *Count, mask at the GIC, clear and unmask at the next Start.*  The unmask
   (a write to the GIC) could overtake the posted ISR clear (a write to the
   PL): the GIC delivered the previous job's still high line, the handler
   counted it and masked the lane, the real completion was never seen.  And
   the driver's `XScuGic_Enable` / `_Disable` rewrite `GICD_ITARGETSR`, one
   word for the ids of VectorOP, MatmulKernel and ConvKernel.
3. *Shipped: the ISR is the record.*  `rpu_irq_start()` clears ISR bit 0 and
   reads it back until the clear has landed, before the ap_start (same write
   channel); after a Start the bit means "this job is done".  `rpu_irq_wait()`
   reads it; between reads it unmasks the lane (`GICD_ISENABLER`, a set
   register: no read-modify-write) and sleeps in WFI with the CPU's IRQs held
   off (`cpsid i` … `wfi` … `cpsie i`: no wake-up lost between the read and the
   WFI).  The handler only masks the line and returns.  Nothing is counted,
   so nothing can be counted twice or missed.  On the board: the six demos and
   the three chat libraries with the R5 forced, all bit-exact, no hang; the
   demos' latency unchanged.

### 10.3 Traps (B2)

7. Every multi-entry project (the chat and TTS libraries) is written by
   `multi.write_project`, not the CLI: the runtime files the generated CMake
   names were missing there (`src/codegen/runtime_files.py` is now the one list).
8. The TCMs are powered only while remoteproc runs the firmware: a library
   reading the control block with the R5 stopped died of SIGBUS in
   `inference_init()`.  The library checks `/sys/class/remoteproc/*/state` first.
9. Large pools (the 360M's 795 MB) failed to allocate with 880–990 MB of CMA
   free after uploads, builds and gates (page cache in the CMA area,
   fragmentation).  `src/remote/cma.py` (`python -m src.remote.cma --config
   CFG`) drops the page cache, compacts memory and names the processes holding
   FPGA buffers; `deploy.py`, `llm_board.py` and `rpu_board.py` call it before
   the allocation.  No reboot needed.
10. A tool edited while a background session was about to run it started the
    firmware from a half-written file: no edits to the board tools while a
    board session runs.

Next: B3 power — the A53 now sleeps during long batches, so the APU's
frequency / idle policy can be measured against §6; the wake cost in the
policy; and fewer, longer decode batches (the decode host ops onto the
kernels, A2) so the A53 can sleep through a token.

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
| **A2 — descriptor chains + the ConvKernel output stride** | one bitstream: the unit's job list from DDR; `y` rows at a stride; the decode prep on the unit (`irope=True` gate) — **planned in §13 (2026-10-15)**: a two-level y pattern (the Llama's grouped heads), chains of ≤ 16 descriptors, the decode merge without a bitstream | WNS ≥ 0 at 4 ns; every library bit-exact; the gates of §13.6 (135M decode ≤ 48.5 ms, prefill 256 ≤ 402 ms, SmolVLM image ≤ 1.63 s), replacing the first estimate (≤ 45 / ≤ 390 ms) | about 8 days (§13.6) |
| **B1 — the R5 firmware for one model** | the device-tree overlay (rpu, TCM, reserved memory), the BARE_METAL fixes of §4.1 for the R5 BSP, the IRQ `kernel_wait`, the loader split, an A53 test harness over shared memory; MNIST convnet, then ResNet-18 | bit-exact with the Linux build; the five IRQs seen; latency within 5 % of the Linux build — **DONE 2026-10-13 (§9), as a model-independent firmware replaying the library's recorded kernel calls: the six CNN demos bit-exact, interrupt-driven, +2–5 % on the real models (YOLO +4 %, MobileNet v2 +5 %)** | 5 days |
| **B2 — the chat path on the R5** | the RPMsg bridge, the sampler on the R5 (fixed point, gated against `libsampler`), `kv260_chat_server.py`'s backend over RPMsg; SmolLM2-135M, then 360M and SmolVLM | the same tokens as the Linux library on the gate prompts; tokens / s not slower; the A53 idle between tokens — **DONE 2026-10-13 in the B1 design (§10): no RPMsg or sampler on the R5 (the A53 keeps the host ops and the output, by the user's direction); protocol v4 + the doorbell; every chat library, BERT and Piper gated bit-exact with the R5 driving; the chat server runs with it (`deploy.py --rpu`); decode +2 % (135M), the bench's A53 CPU 3 s instead of 10 s** | 5 days |
| **B3 — power** | the APU at 333 MHz (`userspace` governor) or suspended between requests with wake on RPMsg; the measurements of §6 before / after | ≥ 0.5 W off the 4.6 W idle, no latency loss — **started 2026-10-13 (§11): the A53 cores' power-down idle −9 mW (on, as a soak), the R5 asleep between commands (protocol v5) −50 mW idle; the APU clock, the PL and DDR next** | 2 days |
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
| Piper encoder (`tts_board.py --profile`), 20 / 60 / 88 / 162 / 268 / 400 ids | 9.5 / 14.4 / 28.3 / 56.8 / 92.1 / 95.8 ms | **8.2 / 12.1 / 23.8 / 47.6 / 78.5 / 81.1 ms**; every encode and duration bit-exact (12 / 12 sequences) | ≤ 75 ms at 400 ids ✗ (−15 %; the band 17 → 2.65 ms; the merge's 23 ms is a banded MAC, not a copy: A3's, §13) |
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

## 11. Track B3 — power, first measurements (2026-10-13)

The INA260 (`hwmon2`, the carrier's 5 V rail) at 10 Hz, 60–90 s windows,
the two settings of each comparison interleaved (A B A B) so drift cancels;
the fan PWM and the SoC temperature logged with each window.

### 11.1 The A53 cores' power-down idle state (`cpu-sleep-0`)

Disabled since CHAT_PLAN §18 (a core parked in `psci_power_down_wfi`).  The
hang was not reproduced in 4.6 h and ~420 000 power-downs: cpu1–3 quiet
(57 k entries, 15 min), cpu1–3 under the chat load plus a threaded burst
load that wakes cores early (186 k entries, 3 228 rejected, ~95 % woken
before the 10 ms residency, 2 h), all four cores through the three chat gates
(174 k, 49 min); the PMU firmware's node states only ever 1 / 2 / 3 in
passing.  Under the R5-driven chat load the A53 is ~92 % idle, unlike the
2026-09-26 load that hung (polled waits, host ops in four threads).

Trap: with this kernel's hrtimer tick broadcast the CPU that arms the
broadcast timer is refused deep idle (`tick_broadcast_enter` → busy, a silent
fall-back to WFI), so a single CPU allowed into `cpu-sleep-0` never enters
it — a test needs at least two.

| | deep idle off | on | |
|---|---:|---:|---|
| idle chat server | 4.768 W | 4.760 W | −9 mW, fan constant |
| back-to-back SmolLM2-135M chat | 6.206 W | 6.170 W | −36 mW, the fan 34 → 68 PWM across the windows |
| decode, the three chat gates (R5 forced) | 51.1 / 132.2 / 51.1 ms/token | 52.2 / 135.0 / 52.1 | +2 %, bit-exact |

Worth a few mW: not the lever.  Left on at runtime as a soak since
2026-10-13 (the tmpfiles rule and `deploy.py` turn it off again).

### 11.2 The R5 asleep between commands (protocol v5)

The firmware polled the control block whenever no batch ran.  Protocol v5
(`runtime/rpu/rpu_ctrl.h`): a library that has the `rpu_ipi` doorbell says
so at attach (`a53_rings`); the firmware polls for `RPU_IDLE_SPIN_US` (1 ms)
after a command, then sets `r5_sleeping`, fences, rechecks `cmd_seq` and
sleeps in WFI with its IPI line (channel 1, GIC 65) unmasked and IRQs held
off; after a submission the library fences and, when it finds
`r5_sleeping`, rings — channel 7's TRIG with RPU0's bit (`rpu_irq_idle`, the
third design's pattern of §10.2).  A v4 library refuses the v5 firmware and
drives its kernels from the A53, so an old library never meets a sleeping
R5.

Trap: the BSP's `cpu_init.S` sets `PMCR.D` — the cycle counter ticks once
per 64 cycles — so the first build's 20 µs spin was 1.3 ms.  1 ms is kept on
purpose: the R5 sleeps between requests and once per decode token (the
sampler and the server between tokens), not in the ~0.1 ms host-op gaps
inside an inference, where a ring per gap would buy ~3 mW.

| | board |
|---|---:|
| no server: the R5 off / polling / asleep | 4.555 / 4.672 / 4.624 W (asleep −48 mW; running at all costs 69 mW) |
| idle chat server: R5 polling / asleep | 4.765 / 4.715 W (−50 mW) |
| back-to-back 135M chat: polling / asleep | 6.177 / 6.177 W (the R5 sleeps ~20 times a second, once per token) |

Every chat library (SmolLM2-135M / 360M, SmolVLM), Piper and the BERT
project rebuilt with the v5 recorder and gated with the R5 forced:
bit-exact; decode 52.2 / 135.1 / 52.1 ms/token and the prefills as with v4
(no cost); 2 862 sleeps under the chat load, every one woken by its ring.

The demo projects too (the six CNN demos given the v5 runtime files, the
stereo and camera projects regenerated): `rpu_board.py --mode r5`, 50 runs
each against the A53 — MNIST convnet / LeNet, ResNet-18, MobileNet v1 / v2,
YOLOv5n and the camera's MobileNet v1 bit-exact; LightStereo-S (host-memory
I/O: no `rpu_host`) through its own runner with the firmware up: the R5 drove
it (1 035 batches, 22 590 calls), 468.8 ms per pair, EPE 0.667 px, the maps
bit-exact against the simulation.  The cost of the sleep in those gates:
R5 over A53 +0.6 / +0.7 / +1.4 / +1.7 % for ResNet-18 / MobileNet v1 / v2 /
YOLO (+0.4 / +0.4 / +1.1 / +1.1 % with v4) and LeNet +6.5 % (+3.7 %) — the
gate runs the A53 reference between the R5 runs, so the R5 sleeps before
every run, and YOLO's host ops leave gaps over 1 ms; a wake seems to cost tens
of µs, more than the interrupt itself should (open).

The policy (§9.3) gained the v5 costs: `RUN_US` 61 µs once per inference
(the R5 woken for the first batch, the A53 asleep on the doorbell through the
last — the convnet, whose runs come back to back, shows the A53's ~23 µs, LeNet
the R5's wake on top) and `WAKE_US` 38 µs for every further wake (a host
interval over 1 ms in the timed replay).  LeNet: 82 µs = 6.7 % of 1.22 ms →
the A53 again (it went to the R5 with v4's 21 µs); every other decision as
before (decode 3.6 %, prefill ≤ 1.6 %, Piper's smallest encode bucket 4.8 %).
A per-long-batch term was tried and dropped: the replay calls all 151 decode
batches long, the board's doorbell count about a fifth.

Where B3's 0.5 W has to come from: the idle board is ~4.7 W with the PL at
250 MHz, the DDR, the carrier and the fan; the A53 cores' power-down and the
R5's sleep together give ~60 mW.  Next: the APU clock (333 MHz between
requests, `userspace` governor), the R5 stopped while no library is attached
(69 mW), the PL clocks and DDR between requests.

## 12. The A53's waits on the kernels' interrupts (2026-10-13)

With the overlay's interrupt numbers fixed (MatmulKernel, ConvKernel and
PoolingKernel had SPIs 96–98 in `dts/kv260/cormorant.dts`, the bitstream
raises them on 90–92), a library that drives the kernels from the A53 can
sleep on them too.  `runtime/inference_irq.[ch]` (every generated Linux
project) does what the R5's `rpu_irq.c` does, through UIO:

- **record**: the kernel's ISR bit 0 (IER bit 0 set);
- **Start** (`inference_irq_start()`, called by the generated
  `X<Kernel>_Start` macro while the A53 drives): clear the previous job's bit
  and confirm the clear landed, raise GIE;
- **wait** (`kernel_wait()` → `inference_irq_wait()`): read the ISR for
  `INFERENCE_IRQ_SPIN_US`, then unmask the line (a write of 1 to the kernel's
  `/dev/uioN`; `uio_pdrv_genirq` masks it when it fires), read again, sleep in
  `poll()`; drop GIE after the wait.  The level line is high from ap_done to
  the next Start, so an unmask after the job ended fires at once.

`INFERENCE_WAIT=poll` (or `-DINFERENCE_IRQ_MODE_DEFAULT=0`) polls as before; a
lane without a UIO interrupt polls.  The gate is `rpu_board.py --mode irq`
(`rpu_host --irq`): each run first polling, then on the interrupts, bit for
bit.

**Board** (bitstream `1cedaf2ed816`, deep idle on, the firmware stopped; 20
runs, BERT 10; projects from the CLI): every run bit-exact, 0 timeouts, 0
missed interrupts in 30 000 waits.  Latency and A53 CPU against polling, by
the spin before a sleep:

| spin | LeNet (1.29 ms) | MobileNet v2 (17.4 ms) | ResNet-18 (20.0 ms) | BERT (265 ms) |
|---|---|---|---|---|
| 0 µs | +13 % / −83 % | +14 % / −84 % | +8 % / −93 % | +7 % / −95 % |
| 100 µs | +3 % / −64 % | +11 % / −46 % | +7 % / −78 % | +4 % / −76 % |
| **200 µs** | **+2 % / −57 %** | **+7 % / −26 %** | **+2 % / −68 %** | **+2 % / −70 %** |
| 500 µs | +3 % / −34 % | +1.5 % / −7 % | +2.5 % / −40 % | +2 % / −57 % |
| 1000 µs | ±0 / ±0 | ±0 / ±0 | +1 % / −6 % | +1.5 % / −42 % |

A sleep costs 20–35 µs of wake-up (spin 0: MobileNet v2 +2.4 ms over 100
sleeps an inference, BERT +18 ms over 548).  200 µs, the first default:
most short calls end within the spin, and the long ones sleep; MobileNet v2
loses most, because many of its waits run just past the spin.

250 and 300 µs, run beside 200 µs in one session (two rounds each): the
latency is the same within the noise — the two rounds of one setting differ
by up to 3 points (ResNet-18 at 250 µs: +5.2 / +1.8 %) — while the CPU saving
shrinks:

| spin | LeNet | MobileNet v2 | ResNet-18 | BERT |
|---|---|---|---|---|
| 200 µs | +2.5 % / −56 % | +3.8 % / −26 % | +3.9 % / −68 % | +2.2 % / −70 % |
| 250 µs | +3.6 % / −52 % | +3.4 % / −21 % | +3.5 % / −63 % | +2.7 % / −68 % |
| 300 µs | +1.9 % / −48 % | +2.8 % / −17 % | +3.9 % / −58 % | +2.5 % / −66 % |

Only MobileNet v2 and BERT sleep less as the spin grows (600 → 360 → 260
and 1 450 → 1 330 → 1 210 sleeps per 20 / 10 runs); LeNet and ResNet-18 sleep
as often.  The default is 300 µs: in this session the lowest latency for
LeNet and MobileNet v2, the fewest sleeps, for a smaller CPU saving — A53 CPU
−48 / −17 / −58 / −66 % at +1.9 / +2.8 / +3.9 / +2.5 % latency.

The R5 gate (`--mode both`) with the A53 reference on interrupts — the same
kernels handed between the A53's and the R5's waits every run: LeNet,
ResNet-18 and BERT bit-exact; the R5 path unchanged (ResNet-18 20.1 ms, BERT
266 ms, A53 CPU 0.66 / 1.1 ms an inference).

### 12.1 The early wake (2026-10-14)

A sleep on the line costs its 20–35 µs wake-up after the job ended; a wait
that knows its call's length can sleep on a timer until just before the end
and read the ISR from there.  The generated code puts
`INFERENCE_KERNEL_EXPECT(k, µs)` before every node's Start: the predicted
length of the node's shortest call (`_source.py _kernel_expect_us`: the
bitstream's performance model where it prices the call, else the analytic
cost model — as the R5 policy prices them; every call of the four projects
below priced).  `inference_irq_start()` notes the Start's time; the wait, when
more than 50 µs of sleep is left, sleeps in `ppoll()` with the line armed
until `INFERENCE_IRQ_EARLY_US` before the predicted end — a job that ends
sooner fires the line, as before — then spins up to `INFERENCE_IRQ_SPIN_US`
and falls back to the line.

The margin, swept beside the spin alone (300 µs) in one session, two rounds,
latency / A53 CPU against polling:

| margin | LeNet | MobileNet v2 | ResNet-18 | BERT |
|---|---|---|---|---|
| off (spin 300 µs) | +3.0 % / −49 % | +3.7 % / −17 % | +4.3 % / −58 % | +2.3 % / −66 % |
| 50 µs | +3.4 % / −81 % | +6.9 % / −77 % | +2.8 % / −91 % | +3.4 % / −91 % |
| 75 µs | +0.7 % / −66 % | +0.8 % / −69 % | +0.5 % / −88 % | +0.6 % / −83 % |
| **100 µs** | **+0.4 % / −64 %** | **+0.7 % / −58 %** | **+0.1 % / −85 %** | **+0.4 % / −81 %** |
| 150 µs | +1.0 % / −61 % | +0.4 % / −46 % | +0.3 % / −79 % | +0.3 % / −78 % |

The timer's wake-up lands about 65 µs after the asked time (the default
50 µs timer slack and the wake-up itself): at 100 µs the waits woke about
33 µs before the job's end (ResNet-18 420 of 420 timed sleeps, MobileNet v2
740 of 740, BERT 1 580 of 1 700 — 120 woken by the line, calls shorter than
predicted); at 50 µs most woke after it, on the line (ResNet-18 539 of 540 in
one round), and paid the wake-up again.  100 µs is the default.  Every run
bit-exact, 0 missed interrupts.

## 13. Track A2 plan — the ConvKernel y pattern and RowOps descriptor chains (2026-10-15)

One bitstream after `de968ccc45da` with two kernel features, and the
scheduler work that uses them.  The goal is the A53's (or the R5's) share of
the chat path: the attention merges, the decode attention prep and the issue
cost of multi-job row-ops nodes.  What it does **not** do: the Piper
encoder's `TtsAttnMerge` (23 ms at 400 ids) is a banded MAC of P against the
relative value embeddings plus a re-rounding, not a copy — a head-ordered
P·V write removes only its ~0.1 ms gather; the rest needs A3's unaligned row
start (§3.3).  §8's "the merge's 23 ms is A2's" was wrong.

### 13.1 What it removes (board profiles, 2026-10-12 … 15)

| host work | where | cost | A2 mechanism |
|---|---|---:|---|
| `LlmAttnMerge` at decode (`[G][HD]` per group, already in head order) | 135M / 360M / SmolVLM decode | 0.3 / 0.55 ms per token | none: P·V writes at `y_off = g·G·HD` (the existing `run_conv_at` argument) — **no bitstream** |
| `LlmAttnMerge`, prefill (G = 3: channel p = h'·T + t → `(t·H + g·G + h')·HD`) | 135M / 360M prefill 256 | 5.5 / 9.7 ms | the ConvKernel **y pattern** (two levels, §13.2) |
| `LlmAttnMerge`, vision (G = 1, one call per head) | SmolVLM `llm_image` | 12 ms | the y pattern's first level alone |
| decode `LlmAttnPrep` (k / q dequant, RoPE in double, the cache writes, the q·Kᵀ image) | 135M / 360M decode | 95 / 126 µs per layer = 2.9 / 4.0 ms per token | the unit's ops (as prefill's `AttnPrepRoNode`) in one **chain** per layer (§13.3) |
| the prefill prep's 10–15 jobs per layer, each issued after the previous one's wait | prefill (135M / 360M / VLM text), vision prep | ≈ 40 µs per job beyond the calibrated back-to-back time (HOSTOPS §10.8) | one chain per layer |
| `LlmDequant`'s 12 jobs (a 49 152-wide row as 12 × 4 096) | every decode / head | 12 issues per token | one chain |
| `TtsRelBandRoNode`'s 9 diagonal TRANSPOSEs + ADDX | Piper encoder, 6 layers | 10 issues per layer | one chain per layer |

### 13.2 ConvKernel: the y channel pattern

Today a channel's plane starts `out_h · out_w` elements after the previous
one (`cv_drain.sv:132` per m-tile of 16 channels, `:299` per channel; the
writer already places a run at any element offset with byte strobes,
`cv_ywriter.sv:77-103`, and splits runs at 4 KiB).  Three registers after
`is_depthwise` (0xC0; the 8-bit control address leaves 0xC4–0xFC free):

| offset | register | meaning |
|---|---|---|
| 0xC8 | `y_cs` | elements from channel c to channel c + 1 inside a group; 0 = `out_h · out_w` (dense: every library today) |
| 0xD0 | `y_cn` | channels per group; 0 = one group (no wrap) |
| 0xD8 | `y_gs` | elements from group k's first channel to group k + 1's |

Channel c lands at `y + (c mod y_cn) · y_cs + (c div y_cn) · y_gs`; a
channel's own `out_h · out_w` pixels stay contiguous (the drain's 256-pixel
segments are linear inside a channel).  For a Llama P·V call of group g:
`y_off = g·G·HD`, `y_cs = H·HD`, `y_cn = T`, `y_gs = HD`; for vision
`y_cs = H·HD`, `y_cn = 0`.  **Contract:** `batch = 1` when `y_cs ≠ 0`, and
`y_cn` a multiple of 16 (an m-tile of 16 channels never straddles a group:
the prefill buckets are 16 / 64 / 256, vision has no wrap); every run is
then HD elements — 128 B, 16-byte aligned for HD 64 — so no partial strobes
on the shipped calls (the general case keeps them).

RTL: three fields in `job_t` (`cv_pkg.sv`), latched beside `q_y`
(`cv_core.sv:416`); in `cv_drain` the m-tile base becomes a group walker
(channel-in-group += 16; at `y_cn`, base = group base + `y_gs`), the
per-channel step `y_cs` instead of `out_hw`, both selected at configuration
time (no mux on the 250 MHz drain path: the dense case loads `y_cs = out_hw`
into the same register).  Estimate: ≈ 300 LUT, no RAM.

Everything that describes ConvKernel's y moves with it:
- the C++ model: `ConvKernel()` gains the three scalars; `write_output_tile` uses the pattern;
- the Verilator testbench: the `Case` fields, the poisoned region and the stray-write check over the strided footprint, `--case` with 20 numbers, random strided jobs;
- the fixture manifest (3 columns) and its parsers: the testbench, the test stand's `conv_tb.sv`, the block-design `conv_classes.svh`;
- the drivers (`gen_driver.py` REGS, ctest `ConvRtlDriver`, fact `registers.ConvKernel`);
- `host_emu`'s ConvKernel, `perf_calls.FIELDS` (the pattern does not change the cycles; it joins the key so a strided call is measured as one), `calib_runner` / `bench_conv`;
- the R5 recorder (`rpu_reg_words` 45 → 51).

**Every** generated `run_conv*` writes the three registers (0 when dense): a
register keeps its value across runs, and a library generated before them
would inherit a stride (the hazard board-deploy §3 records).

### 13.3 RowOpsKernel: descriptor chains

| offset | register | meaning |
|---|---|---|
| 0x70 / 0x74 | `chain` | base of the descriptor list (16-byte aligned) |
| 0x78 | `chain_n` | jobs in the list; 0 = the single job of the registers (every library today) |

A descriptor is the 24 argument words 0x10 … 0x6C in register order, padded
to 32 words (128 B, 8 beats).  With `chain_n` = N (1 … 16) the unit reads
the N descriptors on port B (ARID 1, as the P-block load does — a new
`T_DFETCH` phase before `T_CFG0`) into a 16 × 32-word LUTRAM, then for each
loads the job registers from it instead of from the AXI-Lite copies and
runs the job exactly as today (P load, units, every B response back before
the next), and raises `ap_done` once after the last.  A job of a chain sees
the previous one's writes, as consecutive calls do now.  Estimate: ≈ 1.2 k
LUT (the RAM, the fetch FSM, the 24-word mux in front of the `j_*`
registers — on paths *into* `j_aux`, not out of it, where the worst path
starts).  A list longer than 16 is split by the software.

Software: one DMA buffer per chained node in the pool (N × 128 B), its
static words written by `inference_init()`, the position-dependent words
(the RoPE B row `pos · 8 · HD`, the K-cache offset, the boundary V granules,
whose number varies with `pos % 64`) patched per call and cleaned to DDR
before the Start; `run_rowops_chain(ptr, n)`.  With the R5 driving, the A53
patches and cleans before it records the Start, so the replay sees them; a
chain is one register block (+2 words: `rpu_reg_words` 24 → 27) instead of
N.  `host_emu` loops over the descriptors.  The performance model prices a
chain as its jobs' exact entries plus one chain overhead (a `rowops-chain`
term fitted from chained cases of `calib_runner`); `rpu_policy` and the
timing model count one issue per chain.

### 13.4 Scheduler

- **P·V into the merged tensor.**  A new *output view*: the P·V node's
  output `o_g` is declared a patterned view of the merge's output `pv`
  (offset `g·G·HD`, the pattern of §13.2) instead of its own buffer; the
  merge node stays in the graph — its `reference()` (a pure permutation) is
  the simulation's, so the numbers cannot move — but emits no code, like the
  zero-cost aliases (`_is_alias_node`).  The planner gives `pv` one
  allocation whose life starts at the first P·V call; the one-producer rule
  holds for the views.  Decode (T = 1) needs only the offset (dense
  pattern) and works on today's bitstream; prefill and vision need
  `kernels.conv.y_pattern` (platform flag, `AXI_CONV_YPATTERN`, as `fold`).
- **Decode prep on the unit.**  `AttnPrepRoNode` for T = 1 (today it requires
  `T % 8 == 0` for the 8-row q tiles): the q image as granules of R = G, the
  K-cache COPY and the V granules at the runtime `pos`, all in one chain per
  layer.  Numerics: the decode RoPE becomes the unit's integer one —
  `irope="all"` in `llm_study.py` (135M, 360M, SmolVLM text: top-1, KL and ppl
  against the shipped `irope="prefill"`), the gate HOSTOPS §5.2 used for
  prefill.  The decode step then writes no cache line from the CPU: the
  `llm_cache_inval_rows` / flush pair goes, and the coherency audit's
  "states start clean" covers decode too.
- **Chains for the existing multi-job nodes**: `AttnPrepRoNode` (prefill),
  `VitAttnPrepRoNode`, `TtsRowPrepRoNode`, `TtsRelBandRoNode`,
  `DequantRoNode` (≥ 2 jobs) — one chain where every job of the node is
  static or patchable; behind `kernels.rowops.chains`.
- **Optional, its own gate:** the decode softmax (90 calls per token, 0.4 ms)
  on VectorOP's row mode after a TRANSPOSE (§3.1, `vsmx="decode"`).

### 13.5 Expected effect (estimates, against the 2026-10-15 board)

| workload | today | A2 | how |
|---|---:|---:|---|
| SmolLM2-135M decode | 50.5 ms / token | ≈ 48.3 | merges −0.3; prep 95 → ≈ 33 µs per layer (≈ 10 jobs at the 2.5 µs chain rate + one issue): −1.9 |
| SmolLM2-360M decode | 131.8 | ≈ 128.4 | −0.55; prep 126 → ≈ 35 µs × 32: −2.9 |
| SmolLM2-135M prefill 16 / 256 | 81.5 / 408 ms | ≈ 79 / 400 | the prep's issues (≈ −2 / −3 ms), the merge (−0.4 / −5.5) |
| SmolLM2-360M prefill 256 | 1 063 ms | ≈ 1 048 | merge −9.7, issues ≈ −4 |
| SmolVLM `llm_image` | 1 642 ms | ≈ 1 625 | merge −12, the vision prep's issues |
| Piper encoder, 400 ids | 81.7 ms | ≈ 80 | the band's chains |
| A53 / R5 | 30 host ops per decode token | 0 | each was a batch break on the R5: −30 batches per token (§10: 10.6 µs + a wake each) |

The §5 gates of A2 (135M decode ≤ 45 ms, prefill 256 ≤ 390 ms) were set on
the 2026-10-12 numbers before the A53's interrupt waits and the deep-idle
soak and assumed the merges and the prep were worth more than measured;
they are replaced by §13.6's.

### 13.6 Phases and gates

| phase | what | gate |
|---|---|---|
| **A2.0 — no bitstream** | output views for P·V; the decode merge through `y_off`; the `irope="all"` study — **DONE 2026-10-15 (§13.8)** | the libraries bit-exact (sched-checks, board); decode −0.3 / −0.55 ms per token; `irope="all"` within the prefill policy's yardstick (top-1, KL, ppl) — ✓ (−0.26 / −0.54 ms; the study unchanged) |
| **A2.1 — ConvKernel y pattern** | §13.2 in RTL, the model, the testbench (random strided jobs: G 1 / 3, T 16 / 64 / 256, HD 64 / 96 — a 4 KiB split), fixtures (+3 columns, 3 strided cases), driver — **DONE 2026-10-15 (§13.9)** | `kernel-verify conv`: fixtures + 300 random × seeds at every memory timing, `ConvRtlDsp`, lint; OOC WNS ≥ 0 at 300 MHz; the perf shapes' cycles unchanged; the behaviour test 100 % |
| **A2.2 — RowOps chains** | §13.3 in RTL, the C++ oracle (`rowops_ref.h`) running a list, random chains in the testbench (mixed ops, N 1 … 16, P loads and P_KEEP across jobs, windows), 2 chain fixtures — **DONE 2026-10-15 (§13.9)** | fixtures + 300 random chains × 4 seeds at every timing; a chain's cycles = its jobs' + the fetch (≈ 0.4 µs); OOC WNS ≥ 0 |
| **A2.3 — bitstream** | `build_hw_kv260`, `sim_hw` with a strided conv case and a chain — **DONE 2026-10-16 (§13.10): `33a799febc56`, WNS +0.022 ns, 90 607 LUT, `sim_hw` 116 / 116** | WNS ≥ 0 at 4 ns (today +0.029); LUT ≤ 92 k; `sim_hw` 100 % |
| **A2.4 — scheduler** | the pattern behind `kernels.conv.y_pattern`, chains behind `kernels.rowops.chains`, the decode prep on the unit, host_emu, the chain term in the perf model and `calib_runner` — **DONE 2026-10-16 (§13.11)** | every suite; sched-checks bit-exact (135M / 360M / VLM, BERT, Piper); the generated projects = the simulation on the host emulation |
| **A2.5 — board** | gates, the 72 benchmarks, the perf campaign, the libraries, the chat server (R5) | bit-exact everywhere; 135M decode ≤ 48.5 ms, 360M ≤ 129, 135M prefill 256 ≤ 402, SmolVLM image ≤ 1.63 s; nothing slower; the R5 path's batches per token −30 |

Order: A2.0 first (no bitstream; it also builds the output-view machinery
the pattern needs), A2.1 and A2.2 in parallel, then one build.  Effort: A2.0
1 day, A2.1 1½, A2.2 2, the build and `sim_hw` ½, A2.4 2, the board 1 — about
eight days.

### 13.7 Risks

- **Timing.**  +0.029 ns today; both features add logic next to paths in the
  20 worst (the unit's `j_aux` fan-out, ConvKernel's job configuration).  The
  pattern's step is chosen at configuration time and the descriptor mux
  feeds registers, so neither adds a level to an existing path; the build is
  still the gate, with AREA_PLAN's open items (the replay file, Matmul's
  handshake counters: ≈ 1.5 k LUT) as the reserve.
- **Resources.**  ≈ 1.5 k LUT in all, no block RAM or URAM (94 % / 86 % used).
- **Stale registers.**  Every library is regenerated with the new bitstream
  (the run helpers write the new registers); a library generated before it
  must not run on it — the platform flags make the regeneration necessary
  anyway.
- **Descriptor coherence.**  The buffers are cleaned after every patch; the
  host emulation checks that a chain's buffer was cleaned since its last
  write (as it checks the 16-byte stream alignment since A1).
- **The planner.**  A patterned view is the first non-contiguous view in the
  memory planner (`_demote_strided_views` drops them today): its liveness and
  the slot colouring are what A2.0 tests first, on decode, where the pattern
  is dense.

### 13.8 A2.0 results (2026-10-15)

**The decode merge, in place.**  No `y_off` was needed: at T = 1 the P·V
output o_g `[G][HD]` is the contiguous slice `pv[g·G·HD …]` — 384-byte
offsets for HD 64, G 3 — so the existing view machinery serves it.
`LlmAttnMergeNode.view_offsets()` gives the slice offsets (None where rows
interleave), `CodeGenerator._merge_in_place` marks a merge `in_place` where
every o_g is a flat pool intermediate written by one `LlmAttnConvNode` and
read by the merge alone, pv a flat pool intermediate and every offset
64-byte aligned; the in-place merge is an alias node (no event, no code — a
comment in the generated C), its inputs views of pv (`_view_aliases`,
`inference_buf_init_view`), pv live from the first P·V's start
(`_compute_live_intervals` resolves a writer's output through the alias),
and a host reader of pv invalidates it as kernel-written
(`_written_by_kernel`).  The node keeps its `reference()`, so the simulation
is the same function of the same inputs.  `AXI_INPLACE_MERGE=0` restores the
copy.  Test: `test_llama.TestSimVsStudyFpgaDecode.test_decode_merge_in_place`
(the views, the liveness, the opt-out, prefill untouched); the generated C on
the host emulation (`TestHostEmulation`, coherent and incoherent) and the
scheduler suite (1 841) pass.

The three chat libraries regenerated: 30 / 32 / 30 decode merges in place
(SmolLM2-135M / 360M / SmolVLM); the prefill entries keep their 90 / 96 / 90
host merges until the y pattern.  On the board (`de968ccc45da`): every
library **bit-exact** (`llm_board.py`: SmolLM2 4 prompts × 33 / 33 logits,
SmolVLM 2 images × 33 / 33).  A / B in one session (the previous libraries
built beside, `--remote-dir /root/a2_old`; deep idle off for both):

| library | decode before | decode after | Δ | prefill 16 / 64 / 256 |
|---|---:|---:|---:|---|
| SmolLM2-135M | 49.75 ms / token | 49.49 | **−0.26** | 81 / 100 / 405 → 81 / 100 / 404 |
| SmolLM2-360M | 130.84 | 130.30 | **−0.54** | 192 / 239 / 1 062 → unchanged |

SmolVLM decode 49.38 ms per token, `llm_image` 1.64 s (unchanged).  The chat
server is redeployed with the three libraries (the R5 driving).

**The decode RoPE on the unit (numerics).**  The study policy
`pow2+sink+p12+vsmx+rowops+drope` (`llm_study.py`: the shipped policy with
`irope=True`, every step's RoPE the unit's integer one) against the shipped
`pow2+sink+p12+vsmx+rowops`, the same prompts, windows and yardsticks as
HOSTOPS §5.2 / §10.8:

| model | shipped: top-1 all / resp / held · KL tf / held · ppl · identical | `+drope` | time |
|---|---|---|---|
| SmolLM2-135M | 0.9750 / 0.9818 / 0.9684 · 0.0023 / 0.0025 · 15.600 · 5 / 12 | the same to the digit · 4 / 12 | 25 min |
| SmolLM2-360M | 0.9554 / 0.9949 / 0.9762 · 0.0016 / 0.0018 · 12.387 · 8 / 12 | the same · 9 / 12 | 45 min |
| SmolVLM-256M (`vlm_study.py`, vision `pow2+p12+vgelu+vsmx+rowops`) | features 0.0548, answer top-1 0.9712, KL 0.0033, ppl 12.208, 8 / 24 | the same · 8 / 24 | 65 min |

The teacher-forced and held-out metrics run through the prefill path, where
the policies agree; the greedy generations (the decode steps) diverge from
float at other points — 135M 5 → 4 identical, 360M 8 → 9, SmolVLM 8 → 8 —
inside the spread the earlier gates accepted (HOSTOPS §5.2: 2–6 of 12 across
the integer specs).  **GO**: A2.4's decode prep on the unit may use the
unit's RoPE.

Commands (the venv of the studies, `.venv-export`):

```bash
P=pow2+sink+p12+vsmx+rowops
.venv-export/bin/python demo/chat/scripts/llm_study.py --assets demo/chat/assets/smollm2-135m-instruct study \
    --policies $P,$P+drope --out OUT            # the same for smollm2-360m-instruct
.venv-export/bin/python demo/chat/scripts/vlm_study.py study --combos \
    "pow2+p12+vgelu+vsmx+rowops/$P,pow2+p12+vgelu+vsmx+rowops/$P+drope" --out OUT
```

### 13.9 A2.1 and A2.2 results (2026-10-15)

**A2.1 — ConvKernel's y channel pattern.**  As §13.2: `y_cs` / `y_cn` /
`y_gs` at 0xC8 / 0xD0 / 0xD8 (`cv_ctrl_s_axi`), latched with the job, the
contract (`y_cn` a multiple of 16, ≤ 1280; batch 1 with a pattern) in
`cv_core`; `job_t` carries `y_cs` (out_h · out_w when 0), `y_cs16`, `y_tpg`
(tiles per group) and `y_gs`; `cv_drain`'s fill walks the groups (the
m-tile base `+ y_cs16`, or the next group's base `+ y_gs` every `y_tpg`
tiles) and its emitter steps `y_cs` per channel.  The C++ model
(`ConvKernel()` three defaulted arguments, `write_output_tile`'s channel
base), the Verilator testbench (the registers on every case, the y region /
poison / stray check over the strided footprint, `--case` and the manifest
with the three numbers, random strided jobs — attention-like interleaved
groups, one group with gaps, blocks of groups, odd strides — and patterns
outside the contract that must write nothing), the fixtures (66: the 63
unchanged, their manifest rows `0 0 0`; an attention P·V G 3 T 16 HD 64, a
vision one-group P·V, odd strides at HD 96 with 4 KiB crossings), the
driver (`gen_driver.py`, ctest `ConvRtlDriver`), the test stand's
`conv_tb.sv` (the columns, the registers, a strided scoreboard), the host
emulation's ConvKernel, and every writer of the registers (the generated
`run_conv*`, `calib_runner.c`, `bench_conv.c`: 0 = dense, behind
`XCONVKERNEL_CTRL_ADDR_Y_CS_DATA`; the R5 recorder's window 45 → 51 words).
Gates: the 66 fixtures and 12 × 300 random jobs (seeds 1–4 × fast / rand /
slow) pass; `ctest -R Conv` 6 / 6; lint clean; the perf shapes' cycles equal
HEAD's when its testbench makes the same register traffic (the count moves
by ±5 cycles with the start's phase alone); out of context at 3.333 ns WNS
+0.026 ns, 14 973 LUT, 37 793 FF, 60 BRAM tiles, 48 URAM, 518 DSP — the 20
worst paths in the weight and x loaders, the drain's worst +0.110 ns (its
transposer, unchanged).  The test stand: the first run 65 / 66 — the vision
case's gaps held stale bytes the PS VIP's racing write commits past a burst
(the previous transfer's byte count at the new address, the row-ops unit's
bring-up finding); `conv_tb.sv`'s write fix now restores that range from its
shadow as `hw_128/tb_infra.svh` does; the rerun (from a clean simulation
directory: the incremental `xvlog` hung once, as in `sim_hw` this session):
**66 / 66**, the three strided cases included (56 min).

**A2.2 — RowOps descriptor chains.**  As §13.3: `chain` (0x70 / 0x74) and
`chain_n` (0x78); a chain's descriptors (N ≤ 16, larger clamped) are read on
port B in one run (bursts ≤ 64 beats inside 4 KiB) into six 16 × 128-bit
LUTRAMs (beats 0–5; 6–7 discarded), each job loads `j_*` from the RAM's
registered read through a 2:1 mux and runs the unchanged single-job path,
`ap_done` once after the last; TLOAD is allowed in a chain.  The C++
oracle's `run_chain` models the parameter RAM across jobs (P_KEEP); the
testbench's random chains (every op, N 1 … 16, ~30 % of the jobs reading an
earlier job's output, write-after-write / -read, P_KEEP within and across
chains with a decoy block, `chain_n` > 16, junk in words 24–31, the argument
registers unchanged) and two chain fixtures (a decode attention prep of 7
jobs, a residual / norm chain of 7: `hw/test_data/rowops_test_data/chains/`).
Gates: fixtures + 300 random jobs + 300 chains × seeds 1–4 × three timings
(12 × 638) and a 1 000-chain soak (24 × 1 338) pass; mutants (no wait for the
B responses, P_KEEP ignored, no clamp) caught; lint clean; the driver check;
every existing perf shape's cycles unchanged; a 10-job decode prep chain 3 006
cycles against 2 928 for its jobs one by one, 12 DEQUANTs 14 556 against 14 460
(+0.3–0.4 µs: the fetch and the job starts — against 6–11 µs per job issued
by the CPU).  Out of context at 3.333 ns: WNS −0.053 ns (the de968ccc45da
RTL's −0.067 ns: NORM and sigmoid-URAM paths), +723 LUT (34 189), +1 084 FF,
no RAM or DSP added, no chain path among the 20 worst (its own +0.19 / +0.27
ns).  Watch in A2.3: the paths out of `j_aux` (+0.025 … +0.074 ns OOC
depending on placement) are the design's worst in-context class; per-unit
copies of `j_aux` are the remedy if the build misses.  The scheduler writes
`chain` / `chain_n` = 0 on every single job (`run_rowops`, behind
`XROWOPSKERNEL_CTRL_ADDR_CHAIN_N_DATA`), the host emulation runs a chain's
descriptors, the R5 window 24 → 27 words; the strict `xvlog` (`xsim_rowops_rtl`)
needs `--relax` (use before declaration in four files at HEAD).


### 13.10 A2.3 results (2026-10-15/16)

**`sim_hw`.**  The block-design testbench (`hw/cormorant_hw_128` sim_1)
gains both features.  ConvKernel: the three registers in `conv_regmap.svh`,
`conv_item.set_pattern()`, the driver writes them on every case (0 = dense)
and the scoreboard poisons the pattern's whole span, then checks every
channel plane and that every gap kept the poison; two cases — an attention
P·V (G 3, T 16, HD 64: `y_cs` 576, `y_cn` 16, `y_gs` 64) and odd strides
(23 / 16 / 11, 20 channels).  RowOpsKernel: the chain registers and a
segment / descriptor buffer in `ro_regmap.svh` (the descriptors straddle a
4 KiB boundary), `ro_driver.run_chain`, `ro_scoreboard.run_bytes` (the whole
segment) and `ro_test.run_chains`, which builds the descriptors from
`chains/manifest.txt` (the segment base + the byte offsets) and runs each
chain with one Start.  Two testbench traps on the way: `scripts/sim.tcl`
read a run that stopped on an xsim `FATAL_ERROR` (no overall summary) as a
pass — it now passes only on the testbench's `##  ALL TESTS PASSED`; and
under xsim a 22-conversion `$sscanf` crashed the simulator kernel and a
tokenizer built by `{cur, string'(line[i])}` miscounted the fields (24 of
22 on some lines, so the first run fed the RTL wrong descriptors) — the
manifest is now split with `substr`.  Result: **116 / 116** (VectorOP 29,
Conv 19, Matmul 10, Pool 19, RowOps 39 = 36 fixtures + the table load + the
2 chains), 8 min.

**The bitstream `33a799febc56`** (hw_128 at 51920b6 + the testbench; the
block design unchanged, so the HWH is the same size and address map as
`de968ccc45da`'s): `build_hw_kv260` in 65 min, **WNS +0.022 ns**, WHS
+0.010 ns at 4 ns, every net routed.  The router converged faster than
de968's (rip-up −0.189 → −0.006 → +0.022 against −0.883 … +0.029).

| | `de968ccc45da` | `33a799febc56` | Δ |
|---|---:|---:|---:|
| LUT (placed) | 89 547 | 90 607 | +1 060 |
| FF | 108 713 | 109 798 | +1 085 |
| ConvKernel LUT / FF | 14 827 / 37 497 | 15 096 / 37 788 | +269 / +291 |
| RowOpsKernel LUT / FF | 33 340 / 26 712 | 34 083 / 27 427 | +743 / +715 |
| the other three kernels | | | ±20 LUT (placement) |
| BRAM / URAM / DSP | 136 / 55 / 841 | 136 / 55 / 841 | 0 |

The new logic is far from the critical paths (post-route physopt
checkpoint): out of `j_aux` +0.470 ns (into FOLD's `k_half` resets), the
other `j_*` fields +0.286 (`j_n` into the stream unit), the chain state
+0.393 (`ch_on`'s replica into NORM's DSP B inputs), ConvKernel's pattern
registers +1.141 (the AXI-Lite read mux).  The worst path is unchanged code:
VectorOPKernel's softmax load (`ld_j` → a DSP's CEA1) at +0.022, then the
PoolingKernel tile counter, NORM and Matmul's reset tree at +0.095 … +0.107
— the classes of de968's 20 worst.  Per-unit copies of `j_aux` are not
needed.  Archived in `/mnt/data/bitstreams/kv260_250_33a799febc56` (bit,
HWH, the post-route timing and placed utilization reports, the build log,
the path query).  Not on the board yet: the overlay, the board gates and the
promotion are A2.5's, after the scheduler (A2.4).

### 13.11 A2.4 results — the scheduler (2026-10-16)

**Platform flags.**  `kernels.conv.y_pattern` (`AXI_CONV_YPATTERN`,
`_conv_hw_config.CONV_Y_PATTERN`) and `kernels.rowops.chains`
(`AXI_ROWOPS_CHAINS`, `_rowops_hw_config.ROWOPS_CHAINS`, needs `unit`) —
required keys, **false in `platforms/kv260.json` until A2.5** promotes
`33a799febc56`, so every library generated today is unchanged.

**P·V into the merged tensor (prefill, vision).**
`LlmAttnMergeNode.pattern_views()`: o_g at `g·G·HD` with `y_cs = H·HD`,
`y_cn = T`, `y_gs = HD` (T a multiple of 16, ≤ 1280); one head per group (the
vision encoder, also split by rows: split r at `r·(T/R)·H·HD + g·HD`) without
the wrap.  `CodeGenerator._merge_in_place` takes them where the platform has
the pattern, marks the P·V node (`LlmAttnConvNode.y_pat`) and the merge
emits nothing; the P·V call is `run_conv_at_y()` (an `#error` without the
driver's registers), the dense `run_conv_at()` byte-identical to before.
The views' offset rule is ConvKernel's 16 bytes, not a Slice view's 64: the
CPU never reads or syncs them (the tiny ViT's HD 16 needs it).  A mutant
`y_gs` fails the host emulation.

**Descriptor chains.**  `RowOpsNode.chain_jobs()` (the most jobs a node
issues at run time: AttnPrep — its jobs + 7 per KV group for the V image's
runtime ones —, VitAttnPrep, TtsRowPrep, TtsRelBand, a re-rowed DEQUANT,
the pixel shuffle) and `CodeGenerator._ro_chains` (nodes of ≥ 2 jobs).  The
generated C: the node between `ro_chain_begin()` / `ro_chain_end()`;
`run_rowops()` appends a descriptor to the node's region (a view at the end
of the pool, 128 B per job, `_ro_desc_bytes` in the advertised pool size);
`kernel_wait(KERNEL_ROWOPS)` returns at once inside the node; a chain starts
at 16 jobs and at the end (the node's previous chain waited for first); the
region is cleaned before each Start.  Every node's emission is unchanged —
the collect mode turns its calls into descriptors.  The libraries generated
with the flags: SmolLM2-135M 122 chained nodes, Piper 185.  Mutants (a descriptor's
row count, the missing clean) fail the host emulation in its incoherent mode.
Found on the way: `TestLlamaOnUnit`'s prefill host emulation compared
nothing (a prefill entry has no outputs) — it runs with the head now.

**The decode prep on the unit (`drope`).**  `AttnPrepRoNode` with one row and
no `n` input: ROPE of q / k, the q image as one TRANSPOSE (R = G — a group's
heads as the tile's rows —, the groups as blocks, contiguous granules
x_g[d][h']), the K row and the V granules at the runtime pos, no zeroing.
`LlamaFrontend(drope=True)` (needs `rowops`); the chat generators
(`generate_llm_project.py`, `llm_project`, `vlm_project`, the sched-checks)
take `--drope auto|on|off`, auto = `rowops` and the unit's chains; policy
`…+rowops+drope` (and `pow2+sink+p12+rowops+drope` without vsmx, for the
tiny tests).  The tiny Llama's simulation equals the study under both
policies; on it the integer RoPE happens to round as the double one did.

**Pricing.**  `calib_runner` case `Q`: one row-ops call as a chain of k
copies (the descriptors written once and cleaned; `R` cases now write
`chain_n = 0`); `perf_calibrate.py cases` adds the six shortest shipped
row-ops calls at k = 1, 2, 4, 8, 16 (keys `<call>#chain<k>`, set `chain`);
`perf_fit.fit_chain` fits t_k = k·(t₁ − per_job_us) + per_chain_us, stored as
the model's `rowops_chain`; `PerfModel.chain_adjust_us` prices a chained node
(`kernel_duration_fn(..., chains=cg._ro_chains)` in the timing replay, the
order search, the report, the timeline and `perf_calibrate simulate`); the
R5 policy counts one call per chain; the early wake predicts a chained
node's shortest chain.  Without a measured term a chained node costs what
its jobs did — A2.5's campaign measures it.

**Gates** (A2 flags on: `AXI_CONV_YPATTERN=1 AXI_ROWOPS_CHAINS=1`).
- Suites: scheduler 1856 / 1856, chat 188, lint, C simulation 20 / 20.
- Sched-checks against the studies, under `pow2+sink+p12+vsmx+rowops+drope`:
  SmolLM2-135M 3 / 3 prompts + the second turn bit-exact (1 063 s),
  SmolLM2-360M 3 / 3 + the second turn (2 000 s), SmolVLM-256M both images'
  features (the vision P·V through the pattern) and both answers' prefill +
  16 decode steps (375 s); BERT 3 / 3 examples (every DDR tensor identical).
- Host emulation of the generated libraries: SmolLM2-135M (`llm_host_emu`,
  4 runs × 17 logits, the re-open identical) bit-exact with the simulation;
  Piper (`tts_host_emu --incoherent`: both utterances' PCM, every encode
  bucket, the duration predictor) bit-exact with the specification.

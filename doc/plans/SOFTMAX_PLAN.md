# Softmax on the FPGA: VectorOPKernel softmax modes for BERT, the attention paths and Piper

**Status (2026-10-08):** phases 0–5 done — production bitstream
`588d721997cb` (VectorOPKernel's softmax unit), the platform gate
`kernels.vectorop.softmax` on, the chat server's libraries regenerated with
the softmax on the FPGA and gated bit-exact on the board: BERT 525 → 427 ms,
SmolVLM image 2.20 → 1.93 s, SmolLM2 prefill-256 811 → 737 ms (135M) /
2059 → 1926 ms (360M).  Phase 6 (Piper's encoder softmax) not started.
Uncommitted — §4.

The user asked for softmax on the FPGA for every workload, with an
approximation allowed (re-validated per model, as SmolVLM's GELU was in
OFFLOAD_PLAN).

## 1. Where it stands

Every softmax runs on the host today (double precision, one rounding):

| workload | op | per run (host model) | share |
|---|---|---:|---:|
| SmolVLM image | `VitAttnSoftmax`, 12 layers × 12 heads × [1024 keys][1024 queries] | ~848 ms | ~39 % of 2.19 s |
| BERT inference | ONNX `Softmax`, 12 × [12 heads][256][256] | ~124 ms | ~24 % of 525 ms |
| LLM 256-token prefill (135M) | `LlmAttnSoftmax`, 30 × 3 groups × [keys][768] | ~136 ms | ~17 % of 811 ms |
| Piper encoder (400 ids) | `TtsAttnSoftmax` (scores + relative-key terms) | ~66 ms | — |

Layouts: BERT's scores are query-major rows (the softmax axis contiguous);
the attention paths' q·Kᵀ on ConvKernel writes **keys-major** scores
`s[key][query]` and their softmax writes query-major `P[query][key]` (the P·V
call's weight) — a transpose.

## 2. Design

### 2.1 The numeric specification (all modes)

Per softmax vector (valid length v ≤ n; raw int16 inputs x at exponent f_s,
logit scale σ — 1/√HD, or 1 where the graph already scaled):

```
m   = max_{j<v} x_j                                   (raw)
d_j = m − x_j                                          (0 … 65535)
y_j = (d_j · Cm) >> Cs        Cm · 2^−Cs ≈ log2(e) · σ · 2^−f_s · 2^F   (Cm 24-bit, 2^23 ≤ Cm < 2^24)
e_j = TAB[y_j mod 2^F] >> (y_j div 2^F)               TAB[k] = round(2^E · 2^(−k/2^F)); e_j = 0 past 2^−E
S   = Σ_{j<v} e_j                                     (exact integer)
R   = floor(2^RB / S)
P_j = sat16((e_j · R + 2^(RB−f_p−1)) >> (RB − f_p))   (j ≥ v: 0)
```

with F = 12 (a 4096-entry table), E = 16, RB = 40 as the starting point
(phase 0 checks them).  Everything is integer: the scheduler's simulation, the
C++ reference model and the RTL compute the same bits; per element two DSP
multiplies (d·Cm: 17 × 25 bits; e·R: 17 × 25 bits) and one table read; per
vector one division.  Error budget: the table step (2^−12 · ln 2 ≈ 1.7e−4
relative), R's truncation (≤ 2^−13 relative), e's step (2^−16 of the maximum).

### 2.2 Two modes in VectorOPKernel

- **Row mode** (`OP_SOFTMAX` = 10): `outer` rows of `size` elements, input row
  stride `a_inc`, output row stride `b_inc` (softmax modes write c at the b
  stride; there is no b operand).  A row (≤ 2048 elements) is buffered on
  chip; pass 1 the maximum (while reading), pass 2 e and S, pass 3 P.  BERT.
- **Column mode** (`OP_SOFTMAX_T` = 11): input `s[size keys][outer queries]`
  (row stride `a_inc`), output `P[outer][size]` (row stride `b_inc`) — a
  transposing softmax over the keys of each query column.  Blocks of 16 query
  columns (a 2-beat run per key row; `outer` a multiple of 16 — the kernel
  processes `outer & ~15`) are buffered in 8 skewed banks (element (k, 8w + i)
  in bank (k + i) mod 8 at address 2k + w: a key row is one write, an output
  word of 8 keys one read); ≤ 1024 keys.  The attention paths
  (`VitAttnSoftmax`, `LlmAttnSoftmax` in prefill) without touching q·Kᵀ or the
  caches.
- **One pipeline, e twice.**  Per unit (a row or a block): load (maxima on
  the way in), EXP (read the buffer, e, the sums), the divisions (restoring,
  25 cycles per vector), OUT (read the buffer again, e recomputed, e·R): the
  buffer holds the inputs only (8 BRAM36) and EXP / OUT share one 21-stage
  pipeline (2 DSPs per lane, the 4096 × 17 table in 8 BRAM36).
- **Masks.**  Valid keys per vector: `v(q) = min(size, valid0 + (q mod period))`
  (`period` 0: `valid0`) — the causal prefill rows (valid0 = pos + 1, period =
  T within one head's columns) and the padded runtime key count; keys ≥ v give
  P = 0.
- **Registers** (inside today's 7-bit control space): `smx_cm` 0x6C (Cm),
  `smx_cfg` 0x74 (Cs [5:0], f_p [12:8]), `smx_mask` 0x7C (valid0 [15:0],
  period [31:16]).  One call per head where heads have their own score
  exponents.

### 2.3 Scheduler

- ONNX `Softmax` (Q8.8, last axis, rows ≤ 2048) → a ScheduledNode with
  `OP_SOFTMAX`; the host op where the platform lacks the unit
  (`kernels.vectorop.softmax`, `AXI_VECTOROP_SOFTMAX=0`).
- `VitAttnSoftmax` / `LlmAttnSoftmax` (prefill) → an `LlmKernelNode` issuing
  `OP_SOFTMAX_T` per head; decode keeps the host softmax (3 query columns per
  group: below the block, and a few µs).
- `TtsAttnSoftmax` adds relative-key terms before its softmax: phase 6 decides
  whether a split (host terms → VectorOP softmax) pays.
- Each model's specification follows the hardware (`llm_study.py`,
  `vlm_study.py`, the BERT quality check, `piper_vits.py`), and each library
  is gated bit-exact on the board again.

## 3. Phases

| phase | work | done when |
|---|---|---|
| 0 study | the spec in numpy (`src/vectorop_smx.py`); its error against the exact softmax over the shipped models' score ranges; quality per model — BERT EM / F1 (60 SQuAD examples, simulation), SmolLM2-135M / 360M (`llm_study.py` policy with the FPGA prefill softmax), SmolVLM (vision + text); a DDR-bound cost estimate | parameters fixed; GO / NO-GO per workload (quality within each model's yardstick) |
| 1 C++ model | `VectorOP.cpp` / `VectorOP.h`: both modes and the registers; `TestSimulation` against the numpy spec; fixtures | C sim exact |
| 2 RTL | `vo_smx` (row + column modes), the registers, the job sequencer; Verilator (fixtures, random, every mask shape), lint, OOC synthesis at 300 MHz | `TestVectorOpRtl` passes, WNS ≥ 0 at 300 MHz |
| 3 bitstream | behaviour test, neteq, `sim_hw_kv260`, the 250 MHz design (WNS ≥ 0), board: model suite + benchmarks with softmax cases | board bit-exact, timing met |
| 4 scheduler | the two node kinds, the platform gate, the simulators, C emission, `kernel_calls`, tests; the studies' policies; host gates (`llm_sched_check`, `vlm_sched_check`, BERT) | every host gate bit-exact |
| 5 production | board gates per library / demo, timings, the chat server, the performance-model campaign of the new bitstream, perf-regression baseline, docs, facts | every gate bit-exact, the server healthy |
| 6 Piper (optional) | the encoder softmax split, if phase 0 prices it worth it | — |

Rules: every library change is proven on the board (its gate bit-exact against
the scheduler's simulation) before it reaches the chat server; scratch
`--remote-dir` for gates; one board job at a time; large generated projects on
`/mnt/data` (the root disk is nearly full); commit only when asked.

## 4. Results

### 4.0 Phase 0: quality (2026-10-07)

**BERT** (`bert_study.py --n 60 --policies sched,sched+vsmx`, 60 single-window
SQuAD dev examples, 57 min): **GO**.

| policy | EM | F1 | same span as float | mean max |logit error| |
|---|---:|---:|---:|---:|
| float | 93.3 | 96.5 | — | — |
| `sched` (the host softmax) | 93.3 | 97.1 | 59 / 60 | 1.298 |
| `sched+vsmx` (VectorOP's softmax) | 93.3 | 96.5 | 60 / 60 | 1.289 |

The scheduler's simulation equals `sched+vsmx` bit for bit
(`AXI_VECTOROP_SOFTMAX=1 bert_sched_check.py --n 3`: 3 / 3, all 301 DDR
tensors identical).

**SmolLM2-135M** (`llm_study.py study --policies pow2+sink+p12,pow2+sink+p12+vsmx`:
12 prompts, 3 × 1024 held-out WikiText tokens, 128 new tokens, 77 min): **GO**
— the prefill softmax on the unit (decode keeps the host's):

| policy | top-1 all / responses / held-out | top-5 held-out | KL held-out | ppl (float 15.598) | identical gens |
|---|---|---:|---:|---:|---:|
| `pow2+sink+p12` (today) | 0.9756 / 0.9798 / 0.9690 | 0.9997 | 0.0024 | 15.616 | 2 / 12 |
| `pow2+sink+p12+vsmx` | 0.9732 / 0.9788 / 0.9677 | 0.9997 | 0.0025 | 15.613 | 2 / 12 |

Top-1 within 0.005, KL within 10 % (OFFLOAD_PLAN's yardstick); no weight
saturates, no accumulator wraps.

**SmolLM2-360M** (the same study, 79 min): **GO**, with one caveat.

| policy | top-1 all / responses / held-out | prompt positions only | KL held-out | ppl (float 12.373) | identical gens |
|---|---|---:|---:|---:|---:|
| bf16 (yardstick) | — / 0.9873 / 0.9782 | 0.9985 (648 / 649) | — | 12.395 | — |
| `pow2+sink+p12` (today) | 0.9777 / 0.9936 / 0.9759 | 0.9584 (622 / 649) | 0.0018 | 12.394 | 8 / 12 |
| `pow2+sink+p12+vsmx` | 0.9638 / 0.9949 / 0.9749 | 0.9260 (601 / 649) | 0.0019 | 12.389 | 7 / 12 |

The answer positions (782 / 786 against 781), the held-out windows (−3 of
3069), KL and perplexity are within the yardstick; the drop of "top-1 all"
(−0.0139) is the prompt positions' — 21 more of 649 low-margin predictions of
the user's own text flip, where the int16 policy already lost 26 against
bf16's 1; chat never uses those logits.  Not a bias of the specification: on
sink-dominated score rows of 8 … 1024 keys its mean difference from the exact
softmax is 0.000 LSB (±1 LSB on 0.02–2.4 % of the elements, most with few
keys), and rounding e instead of truncating it changes nothing.

**SmolVLM-256M** (`vlm_study.py study --combos "pow2+p12+vgelu/pow2+sink+p12,pow2+p12+vgelu+vsmx/pow2+sink+p12+vsmx"`:
24 COCO images, 96 new tokens; the vision softmax and the text prefill softmax
on the unit): **GO**.

| combo (vision / text) | features rel. error (max) | cos min | answer top-1 | KL | identical | ppl (float 12.221) |
|---|---:|---:|---:|---:|---:|---:|
| `pow2+p12+vgelu` / `pow2+sink+p12` (today) | 0.0544 (0.0674) | 0.99797 | 0.9667 | 0.0034 | 7 / 24 | 12.223 |
| `pow2+p12+vgelu+vsmx` / `pow2+sink+p12+vsmx` | 0.0548 (0.0687) | 0.99788 | 0.9704 | 0.0034 | 6 / 24 | 12.221 |

### 4.1 Phase 1: the C++ model (2026-10-07)

`VectorOP.h` `smx_vector` / `smx_valid` and `VectorOP.cpp` `softmax_job` (the
registers appended as `smx_cm` / `smx_cfg` / `smx_mask`; Cm is `smx_cm[23:0]`).
`TestSimulation`: 224 tests (11 softmax cases — row mode 1 … 2048 elements,
tails, causal masks, column mode 16 … 1024 keys, strides) exact against the
numpy specification on 978 vectors, every result within 1 LSB of the exact
softmax.  The RTL fixtures grew to 212 cases (the first 201 unchanged; the
manifest has the three register columns).

### 4.2 Phase 2: the RTL (2026-10-07)

`rtl/vo_smx.sv` (the unit), `rtl/vo_smx_rom.sv` (generated by
`scripts/gen_smx_rom.py`, checked against `vectorop_smx.TAB`: ctest
`VectorOpRtlSmxRom`), the burst generator's block loop (`geom_t.n_blk` /
`blk_w`), the three registers, `vo_core`'s routing (a softmax job's words to
`vo_smx`, its P words to the write port; c advances by `b_inc`).

| check | result |
|---|---|
| Verilator lint | clean |
| `TestVectorOpRtl` (ctest: 212 fixtures, 300 random jobs — about 40 softmax —, every Q8.8 input through a 2048-row softmax at two scales) | all bit-exact |
| random jobs by hand: seeds 1–3 × rand / slow / fast timing × 300 | 2 700 / 2 700 |
| `synth_vectorop_rtl` (out of context, 300 MHz, 29 min) | WNS +0.175 ns (Fmax ≈ 317 MHz); 11 672 LUT, 13 751 FF, 30 BRAM36, 35 DSP — +5.5k LUT, +5.1k FF, +16 BRAM36, +16 DSP for the unit; the worst path is the argument latch's enable (routing only); no `Synth 8-4767` |

Cycles with ideal memory (`Vtb --timing fast`), at the kernel clock:

| job | cycles | model |
|---|---:|---|
| row mode, rows of n | 75 + rows · (58 + 3 ⌈n/8⌉) | BERT 256 × 256: 39 056 (0.21 words / cycle) |
| column mode, K keys | 60 + (outer / 16) · (455 + 6 K) | 1024 keys × 64 queries: 26 579; 256 × 256: 31 947 |

The phases run one after the other (load, EXP, the divisions, OUT): about a
third of the memory-bound rate.  At 250 MHz a BERT head takes ~0.16 ms (the
host ~0.9 ms), a SmolVLM head ~1.7 ms (host ~5.9 ms), a 135M prefill-256 head
~0.13 ms.

### 4.2b Phase 3: the bitstream (2026-10-07)

| check | result |
|---|---|
| `behavior_test_vectorop` (the test stand, 212 fixtures) | 212 / 212 (33 min); the element-wise cases' kernel cycles unchanged (Verilator perf within ±4 cycles of the table; the stand's per-case time +1.1–1.5 µs = the three register writes the testbench adds) |
| `neteq_vectorop_rtl` (42 jobs) | 0 mismatches (after the masked-lane fix above) |
| `build_hw_kv260` (250 MHz) | bitstream `588d721997cb`: WNS +0.041 ns, WHS +0.010 ns (73 min); 62 347 LUT (53.2 %), 123 BRAM tiles (85.4 %), 48 URAM, 712 DSP; no `Synth 8-4767` |
| `sim_hw_kv260` (block-design testbench) | 75 / 75 (VectorOPKernel 29, the two softmax cases among them) |
| board: registers | `smx_cm` / `smx_cfg` / `smx_mask` (and `alpha`) written and read back; PS port widths 128-bit |
| board: `run_remote_tests` (`AXI_VECTOROP_SOFTMAX=1`) | 159 / 159 — the four tiny BERTs and the three `smx_*` models on the unit, every output equal to the simulation (12 min) |
| board: `run_remote_perf` (60 cases + 3 softmax) | 63 / 63; against `6436623029f7`'s baseline: 0 regressions on a re-run (a first run flagged `batch4-A-bcast` +2.15 %, Δ 1 µs; the re-run +1.72 %) |
| board: MNIST, image classification | 98.92 % / 97.35 %; MobileNet v1 / v2 21.98 / 20.44 ms, ResNet-18 20.50 ms — unchanged (no Softmax) |
| board: BERT (regenerated, softmax on the unit) | **425.8 ms** (525 before, −19 %); EM / F1 88.0 / 90.3 = float; bit-exact against the simulation (3 / 3) and the `sched+vsmx` emulation (50 / 50) |
| board: SmolLM2-135M (`pow2+sink+p12+vsmx`) | logits bit-exact (4 × 33); prefill 16 / 64 / 256: **152 / 276 / 737 ms** (156 / 297 / 811 before); decode 54.6 ms / token (unchanged) |
| board: SmolLM2-360M (`pow2+sink+p12+vsmx`) | logits bit-exact (4 × 33); prefill 16 / 64 / 256: **330 / 661 / 1926 ms** (337 / 695 / 2059); decode 137.5 ms / token (unchanged) |
| board: SmolVLM (`pow2+p12+vgelu+vsmx` / `pow2+sink+p12+vsmx`) | logits bit-exact (2 images × 33); `llm_image` **1941 / 1927 ms** (2204 / 2177, −12 %); text prefill 152 / 275 / 736 ms, decode 54.3 ms / token |

Softmax calls on the board (`run_remote_perf`): BERT head (rows 256 × 256)
0.160 ms (Verilator 0.156), SmolVLM head (1024 keys × 1024 queries) 3.77 ms
(1.69 — the column mode's two-beat reads per key row cost half the
bandwidth), a 135M prefill-256 head 0.182 ms (0.128).

### 4.3 Phase 4: the scheduler (2026-10-07)

`src/smx_nodes.py`: `SoftmaxVopNode` (ONNX Softmax, row mode, where the
platform has the unit), `VitAttnSoftmaxVopNode` and `LlmAttnSoftmaxVopNode`
(the attribute `vsmx = 1`, set by the frontends' `vsmx` option: one
column-mode call per head; the prefill's over the runtime key count with
valid0 = pos + 1, period = T — padded rows t ≥ n are computed too, they feed
padded rows only).  The gate `kernels.vectorop.softmax` (false until the
bitstream is in production; `AXI_VECTOROP_SOFTMAX`), `run_softmax()` and the
`inference_init()` probe in the generated C, the emulator's software unit,
the performance model's family `vecop-smx`.  The chat scripts take
`--vsmx auto|on|off` (`generate_llm_project.py`, `llm_sched_check.py`,
`vlm_sched_check.py`; `project.json` records it); the policies are
`pow2+sink+p12+vsmx` (text) and `pow2+p12+vgelu+vsmx` (vision); BERT's check
follows the gate (`sched+vsmx`).  `test/test_softmax_unit.py` (13 tests): the
simulation equals the specification, the tiny ViT and the tiny Llama equal
their studies' vsmx policies bit for bit, and the generated C does in the
emulator (also `test/gen_softmax_models.py`'s three board-suite models).

Host gates with the unit (`AXI_VECTOROP_SOFTMAX=1`):

| gate | result |
|---|---|
| `bert_sched_check.py --n 3` (policy `sched+vsmx`) | 3 / 3 bit-exact, all 301 DDR tensors |
| `llm_sched_check.py --vsmx on`, SmolLM2-135M (`pow2+sink+p12+vsmx`) | 3 / 3 prompts and a second turn bit-exact (30 min) |
| `vlm_sched_check.py --vsmx on --text`, SmolVLM (`pow2+p12+vgelu+vsmx` / `pow2+sink+p12+vsmx`) | image features 0 of 36 864 differ (COCO 39769, 1268); text 2 / 2 bit-exact (12 min) |

The scheduler suite passes with the gate off (1691) and with it on — the
tests that exercise the host Softmax pin the gate off, the others follow it
(`test_bert_tiny`, `test_bert_base`, `test_host_ops`, `test_cache_coherency`
— whose checker learnt `run_softmax`).  The emulator's software unit had one
bug of its own (an `int32_t` table entry shifted by up to 62 bits), found by
the board suite's `smx_rows_64`.

`neteq_vectorop_rtl` (30 element-wise + 12 softmax jobs) first reported 23
mismatches: column mode's last output word of a query reads buffer words of
keys past `size` that the unit never wrote, and on those masked lanes
`e = 0 >> sh` carried the unknown input's X into P in the RTL simulation
(the netlist and the hardware give 0).  With the shift zeroed on masked lanes
too: 0 mismatches.

### 4.4 Phase 5: production (2026-10-07 / 08)

- **Gate on:** `platforms/kv260.json` `kernels.vectorop.softmax: true`; the
  local `bitstream_config_kv260.json` names `588d721997cb`
  (`/mnt/data/bitstreams/kv260_250_588d721997cb/`).  The scheduler suite
  passes both ways (1 691 with the gate on and with `AXI_VECTOROP_SOFTMAX=0`).
- **Chat server:** `libsmollm2.so`, `libsmollm2_360m.so`, `libsmolvlm_256m.so`
  generated with `--vsmx on` and installed (`llm_board.py --install-only`;
  their weights byte-identical), the BERT project regenerated
  (`deploy.py --regenerate`); Piper's library unchanged (no softmax on the
  unit — phase 6) and gated on the new bitstream (PCM, encoder and duration
  predictor bit-exact, RTF 0.236 / 0.204).  All five backends answer.
- **Performance model** `kv260/588d721997cb`: 1 858 exact calls (a fresh case
  list of 1 443, three refinement rounds 139 + 126 + 78, then the prefill
  softmax at every runtime key count — `LlmAttnSoftmaxVopNode` is priced at
  `keys_of(sn)` keys like the attention convs); the simulator within ±2 % of
  the board for the CNNs, BERT, SmolLM2-135M and SmolVLM (SmolLM2-360M's
  prefill 16 −12 %, its host ops priced by the kinds' fits).  The first pass
  wrote the softmax registers on every VectorOP call (~80 ns each, 4–8 % on
  the shortest benchmarks); `calib_runner.c` and `bench_vectorop.c` now write
  them for ops 10 / 11 only, as `run_softmax()`, and the 99 element-wise
  calls were re-measured.
- **Engine cost model:** the call floor moved 448 → 434 cycles; ONE / one
  +14 so the sums every decision sees are unchanged (the regenerated 135M
  library is identical); the constants re-scored on the new campaign (ConvKernel
  18.7 / 48.2 %, MatmulKernel 4.6 / 17.7 %) and kept.
- **perf-regression baseline** `kv260-588d721997cb.json`: the 63 benchmarks
  (three softmax cases added) and the three demos.  Against `6436623029f7`:
  no regression beyond `MatmulKernel/batch4-A-bcast` at +1.7–2.4 % (1 µs, two
  of three runs; MatmulKernel unchanged — the new place and route).
- **Docs and facts:** the kernel reference, the scheduler reference and user
  guide, the platform schema, the chat docs, README; 60 / 60 facts.

Open: phase 6 (Piper's `TtsAttnSoftmax`, relative-key terms); the column
mode's DDR rate (two-beat runs per key row: SmolVLM's head 3.77 ms against
1.69 with ideal memory — wider blocks or a query-major score layout would
help); the phases of a unit could overlap (a second buffer) for about 3× the
throughput.

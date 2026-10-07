# SmolVLM and Piper on VectorOP's activation unit; the engine cost model at 250 MHz

**Status (2026-10-07):** phase 1 (SmolVLM) done — gate bit-exact on the board, image 2.33 → 2.19 s; phase 2 (Piper) done — PCM bit-exact on the board, chunk 327 → 272 ms; both libraries in the chat server; phase 3 done — the cost model refitted (no shipped engine choice moved, per-call error halved), the performance model topped up to 1852 exact calls; phase 4 done (libraries deployed, docs and the fact registry updated; uncommitted).

Follows [ACTIVATIONS_PLAN](ACTIVATIONS_PLAN.md) (the activation unit, in the
production bitstream `6436623029f7`) and [FMAX_250_PLAN](FMAX_250_PLAN.md)
(the open cost-model refit).  No new bitstream: everything below runs on
VectorOPKernel's existing ops (`ADD`, `MUL` with a per-channel operand) and
acts (`GELU`, `LEAKY_RELU`).

## 1. Where the time goes

Every node of the shipped graphs priced with the board-measured models (the
host-op model `host.json`, the kernel model `kv260/6436623029f7`; kernel
calls serialised — the real time overlaps some of them):

| graph | host ops | kernel calls | the activations today |
|---|---:|---:|---|
| SmolVLM `vision` (one image, 2.33 s on the board) | 1 402 ms | 1 111 ms (ConvKernel) | `VitGelu` × 12: **242 ms** (bias + GELU, 20 ms per layer); the attention softmax (848 ms) is not an activation of the unit |
| Piper `chunk` (1.49 s of audio, ~0.35 s) | 215 ms | 124 ms (ConvKernel) | LeakyReLU inside 25 of the 46 `TtsPrep` (79 ms in all); the residual sums they read, `TtsSum` × 37: **104 ms** |

## 2. Design

**The exponents.**  Both models keep their tensors at calibrated power-of-two
exponents (`src/numeric.py`); the unit's tables are Q8.8's (exponent 8).
VectorOP nodes may not touch exponent tensors today (`numeric.check`), so the
offloaded ops stay LLM-domain nodes (as `LlmAttnConvNode` runs ConvKernel) that
issue VectorOP calls with exponent-aware constants.  Neither move can keep
today's numbers bit for bit — an exact table at exponents 9–13 would need
4–32× the entries per lane (beyond the free BRAM) — so each changes its
model's specification (the study scripts the scheduler matches bit for bit)
and must pass that model's quality check again.

### 2.1 SmolVLM: `VitGelu` → two VectorOP passes

Today `a = gelu_{f,8}[sat16(raw + braw)]`: the fc1 output `f` at per-channel
exponents 9–13 (mostly 11–12), the bias added at those exponents, the GELU
evaluated exactly for that input and written at exponent 8 (one 65 536-entry
table per input exponent).  New:

1. `ADD`: `s = sat16(raw + braw + h)`, h = 2^(f−9) for f > 8 (half an LSB of
   Q8.8 at the input's exponent; 0 for f ≤ 8);
2. `MUL` by the per-channel scale 2^(16−f) (the value 2^(8−f)) with act
   `GELU_TANH` (SigLIP's tanh form): the kernel floors `s · 2^(8−f)` — with h,
   a round-half-up to Q8.8 — saturates and applies the exact GELU of that
   Q8.8 value.

Both passes broadcast a per-channel row ([3072]) over the 1 024 patch rows.
Eligible: the output at exponent 8 for every channel (11 of the 12 layers; the
last one keeps nine channels at 2^-7 and stays on the host).  The numeric
change is one rounding of the GELU input to 2^-8.

As built: `src/vit.py` gives `VitGelu` two more inputs, the rows `ba` and `sc`
(`_gelu_vop_rows`, raw int16 at 2^-8) where the layer is eligible;
`VitGeluNode.from_onnx_node` returns `VitGeluVopNode` (an `LlmKernelNode` on
VectorOPKernel: `run_op` ADD, `kernel_wait`, `run_op_act` MUL + `GELU_TANH` in
place) when the platform has the activation unit, else the host op (the
extra inputs ignored).  The study policy is `pow2+p12+vgelu`
(`vlm_study.VISION_VOP`); `vlm_sched_check.py` checks the policy the gate
selects (`vision_policy`).

### 2.2 Piper: the LeakyReLUs and the sums they read

The LeakyReLUs live in `TtsPrep`, a host copy that folds a conv input into rows
with halos, masks outside the utterance and requantises it (every one of the 25
changes exponent).  Moving only the LeakyReLU adds an FPGA pass and removes no
host pass; fusing it into the sum before it is impossible too, because the
un-activated residual `y` feeds the next sum.  What can move is the residual sums
(`TtsSum`).

The decoder's calibrated exponents already give most resblock convs their
residual's exponent; the upsampler outputs and a few blocks differ by 1–3 bits.
With **one exponent per decoder stage** — the upsampler's output, every resblock
conv's output and every residual `y` at the smallest calibrated one (8 / 10 / 11:
`vop_exponents`) — every `y = rq(t + y)` adds two raw tensors at one exponent:
an exact sum, whose round-half-even is the identity and whose saturation is
VectorOP `ADD`'s.  The three k7 dilation-12 convs' tap groups are summed (and
saturated) first, so each sum has two inputs.  21 `TtsSum` (76.9 ms on the
host) become 21 `ADD` calls (18.5 ms predicted; the chunk ≈ 339 ms of serial
work → −17 % before the cache invalidations the following preps now need).
No activation unit is involved: any VectorOP bitstream runs it.  The three stage
averages (÷ 3) and the flows' sums (different exponents, small) stay on the
host.

As built: `piper_vits.chunk_forward(..., vop_sums=True)` is the library's
specification (`vop_sums=False`: the one before); `PiperChunkFrontend(...,
vop_sums=True)` applies `vop_exponents` and emits the tap-group sums;
`TtsSumNode.from_onnx_node` returns `TtsAddVopNode` (an `LlmKernelNode`: one
`run_op` ADD) for every two-input, whole-tensor, unmasked sum whose inputs share
the output's exponent — bit-identical to the host op wherever it applies.

**Gate:** the phase-2 study prices the new chunk with the performance model
and measures its quality (waveform SNR and log-mel distance against float,
TTS_PLAN §3); implement only if the chunk gets ≥ 10 % faster and the quality
stays within the yardstick (log-mel distance ≤ 0.22 dB against today's 0.18,
SNR mean within 1 dB of 23.3).

### 2.3 The engine cost model at 250 MHz

`cost_model.py`'s board terms — `RTL_CONV_BOARD` (ConvKernel's recurrence
parameters), `RTL_COEF` (MatmulKernel's term weights), `CALL_OVERHEAD` — were
fitted at 100 MHz in kernel cycles; at 250 MHz a fixed DDR or host cost is
2.5× the cycles, so the MatMul engine choice (ConvKernel vs MatmulKernel, GEMV
or tiled, the lowered conv's kw / row split) and `--fc-conv` use stale
crossovers.  Refit them to the `6436623029f7` campaign (1 729 exact calls) with
a script that can be re-run (`RTL_COEF` by non-negative least squares as
before; `RTL_CONV_BOARD` by a coordinate search on its integer parameters;
`CALL_OVERHEAD` from the measured call floor), then diff the engine choices of
the shipped models; libraries whose choices change are regenerated (results
bit-identical either way) and re-gated.

## 3. Phases

| phase | work | done when |
|---|---|---|
| 1a SmolVLM study | the new GELU in `vlm_study.py`'s emulation (a policy beside today's); image-feature error, answer top-1 / KL against float on the study's prompts | indistinguishable from today's policy (top-1 within 0.005, KL within 10 %) |
| 1b SmolVLM implementation | `vit_nodes.py` (the VectorOP form of `VitGelu`), `vit.py`, the simulator / C emission / `kernel_calls`, `vlm_sched_check.py`; tests | `vlm_sched_check` bit-exact; scheduler tests; the C emulation |
| 1c SmolVLM board | regenerate `libsmolvlm_256m.so`, `llm_board.py` gate (bit-exact against the new spec), image time | gate bit-exact, image time measured |
| 2a Piper study | `TtsSum` → VectorOP (+ fused LeakyReLU) in `piper_vits.py` as a variant; quality on the study's sentences; predicted chunk time | the gate of §2.2 |
| 2b Piper implementation (if GO) | frontend, nodes, `tts_host_emu.py`, tests | `tts_host_emu` PCM bit-exact against the new spec |
| 2c Piper board (if GO) | `tts_board.py` gate, RTF | PCM bit-exact, RTF measured |
| 3 cost model | fit script; refit; engine-choice diff; tests' anchors; regenerate and re-gate what changed; perf-model top-up | the model's error at 250 MHz (median / p90) reported against today's; every shipped library re-gated |
| 4 production | the chat server's libraries, perf-regression baselines, docs, facts | the server healthy, every gate bit-exact |

Rules: every library change is proven on the board (its gate bit-exact against
the scheduler's simulation) before it reaches the chat server; one board job at
a time; commit only when asked.

## 4. Results

### 4.1 SmolVLM (phase 1)

**Study** (`vlm_study.py study --combos "pow2+p12/pow2+sink+p12+mix,pow2+p12+vgelu/pow2+sink+p12+mix"`,
24 COCO images, 96 new tokens, 64 min on the host; `/mnt/data/act/vlm_study/`):

| combo (vision / text) | image features rel. error (max) | cos min | answer top-1 | KL | identical answers | ppl (float 12.221) |
|---|---:|---:|---:|---:|---:|---:|
| `pow2+p12` / `pow2+sink+p12+mix` (before) | 0.0545 (0.0676) | 0.99794 | 0.9697 | 0.0034 | 5 / 24 | 12.223 |
| `pow2+p12+vgelu` / `pow2+sink+p12+mix` | 0.0544 (0.0674) | 0.99797 | 0.9667 | 0.0034 | 6 / 24 | 12.223 |

GO: top-1 within 0.005 (−0.0030), KL equal; no weight saturates, no
accumulator wraps (the saturation counts move by tens of the residual's 45 557).

**Implementation:** the scheduler's simulation equals the study's emulation
bit for bit — the tiny ViT (`test/test_vit.py`: simulation, the generated C
in the coherent / incoherent emulator, the host form without the unit) and
the real model (`vlm_sched_check.py`: COCO 39769 and 1268, 0 of 36 864 image
features differ).  The emulator's flush now writes its whole range
(`test/host_emu.py`): a kernel-written slot that only kernels read is never
invalidated, so a host op reusing it stored bytes equal to its stale copy,
which the byte-compare model skipped (a dirty line on the board).

**Board** (`llm_board.py`, scratch `--remote-dir`; the 519 MB pool needed a
`drop_caches` + `compact_memory` after the weight upload):

| | before | after |
|---|---:|---:|
| `llm_image` (COCO 39769 / 1268) | 2.33 s | 2204 / 2177 ms |
| logits (33 vectors per image) | bit-exact | bit-exact |
| decode, prefill 16 / 64 / 256 | 54.2 ms / token | 54.2 ms / token; 156 / 295 / 809 ms |

11 of the 12 layers' GELU (242 ms on the host) became 22 VectorOP calls; the
gain, ~0.14 s, is a little below the performance model's 0.18 s (the
`[1024][3072]` calls with a stride-0 row longer than the 2048-element replay
re-read it per row; no exact entry yet — phase 3's top-up).

### 4.2 Piper (phase 2)

**Quality** (the study's 11 evaluation sentences against float with the same
noise; `synthesize_chunked`, `/mnt/data/act/piper_vop/`):

| specification | SNR mean / min (dB) | log-mel distance (dB) |
|---|---:|---:|
| the library before (`vop_sums=False`) | 22.89 / 12.19 | 0.182 |
| stage exponents + VectorOP sums | 22.68 / 12.16 | 0.208 |
| — against the one before | 35.65 / 34.08 | 0.160 |

GO: SNR within 1 dB of the yardstick's 23.3, log-mel distance ≤ 0.22.  The
exponents that moved: `dec.up0` 10 → 8, the stage-0 resblocks 1–2 9 → 8,
`dec.up1` 11 → 10, `dec.rb11` 11 → 10, `dec.up2` 13 → 11, `dec.rb20` 12–13 →
11, `dec.rb21.c1` 12 → 11.

**Implementation:** the chunk graph has 21 `TtsAddVopNode` (18 residual sums,
3 tap-group sums) and 19 host `TtsSum`; `test/test_piper.py` (simulation =
specification, both variants; the generated C in the emulator) and
`tts_host_emu.py --incoherent` (PCM, encoder, duration predictor bit-exact).
`generate_tts_project.py` writes `tts_glue_init()` — `inference_init()` with
every kernel the project drives (now VectorOPKernel too), which
`tts_api.c` calls.

**Board** (`tts_board.py`, scratch `--remote-dir`):

| | before (`6436623029f7` gate) | after |
|---|---:|---:|
| chunk (median of 7) | 327.2 ms | 272.3 ms (−17 %) |
| utterance 2.15 s / 6.88 s of audio | 617.6 / 1676.8 ms (RTF 0.288 / 0.244) | 508.4 / 1405.4 ms (RTF 0.237 / 0.204) |
| PCM, encoder, duration predictor | bit-exact | bit-exact |

Deployed: both libraries installed with `llm_board.py --install-only` /
`tts_board.py --install-only`, the chat server restarted (`deploy.py`);
`/v1/audio/speech` and an image question answer through it.

### 4.3 The engine cost model at 250 MHz (phase 3)

`tools/fit_cost_model.py` (new): `fit` (the campaign's calls against the
constants, NNLS for `RTL_COEF`, a coordinate search over `RTL_CONV_BOARD` on a
vectorised copy of `rtl_conv_walk` checked against it), `decisions` / `diff`
(every shipped model's unplanned engine choices under two constant sets,
priced by the performance model).

**What a measured call covers.** `calib_runner.c` times the register writes,
Start and the IsDone poll: a call is the kernel model + `CALL_OVERHEAD`, as
the callers add them.  The old constants were fitted net of the ~315-cycle
call floor but charged 1 500 per call (≈ 1 200 cycles too much: up to 2× on
small MatmulKernel calls).  Now `CALL_OVERHEAD` = the campaign's call floor
(448 cycles = 1.79 µs at 250 MHz) and ONE / one are fitted net of it.

**Two fits.**  A plain refit (all calls, relative error) fits slightly
better but moves BERT's 12 per-head P·V MatMuls to MatmulKernel tiled — +5.1 ms
per inference by the performance model (their ratios, 0.55 / 0.78, sit near
`LOWER_MARGIN` 0.9).  The applied set weights the shipped calls 0.75 and
penalises engine choices the performance model calls wrong: **no engine choice
of any shipped model moves** (10 models, every entry; 2 265 MatMuls priced
both ways — today's choices are all right except Piper `encode_32`'s six
32×576×768 MatMuls, 0.59 ms per call of that bucket under either set).

| per-call error, median / p90 (6436623029f7) | before (as used, +1 500) | after (+448) |
|---|---:|---:|
| ConvKernel (1 190 calls) | 33.9 / 64.5 % (bias −0.41) | 18.3 / 45.8 % (bias −0.10) |
| MatmulKernel (414 calls) | 8.8 / 29.6 % | 4.5 / 17.7 % (leave-one-out 4.6 / 18.1 %) |
| the other 250 MHz campaign, 986cef4866a0: ConvKernel / MatmulKernel | 35.6 / 64.9, 9.0 / 29.7 % | 19.3 / 46.5, 4.5 / 17.6 % |

The script reproduces the 100 MHz set on c2b2a6e5e50e (MatmulKernel 0.36 /
1.70 %, ConvKernel 3.2 / 17.0 %), so the method is the old one.  At 250 MHz
MatMul-on-ConvKernel calls take 1.66× (p90 2.47×) their 100 MHz cycles, CNN
convs 1.03×: the recurrence's DDR terms are effective, not physical (equally
good fits spread ONE 0–66, FL 231–650), and narrow input-heavy calls (16–64
output channels, ≥ 200 input channels) stay 2.5–3× under-predicted — a better
fit needs the model's structure (DDR bandwidth shared by the x, weight and y
ports), not constants.

**Coupling removed.**  The performance models use the board walk as ConvKernel
features; they now take a frozen copy, `RTL_CONV_FEATURES` (the 100 MHz set),
so the stored models of 6436623029f7, 986cef4866a0 and c2b2a6e5e50e predict
exactly as before without a refit.  `cost_model.KERNEL_MHZ` (250) converts
the estimates in `report.md`.  Test anchor (`test_rtl_conv_model`): the 3×3
64-channel 56² job 230 546 cycles on the board (model 230 092), BERT's
per-head P·V 30 576 (model 34 907, +14 %).

**Performance-model top-up** (`perf_calibrate.py`: a fresh case list, 1406 cases; `run --resume`;
three refinement rounds, 125 + 29 + 5 calls): `kv260/6436623029f7` 1729 → 1852 exact calls,
repeat spread median 0.076 %, the families unchanged.  The new VectorOP calls measure: SmolVLM's
GELU passes ([1024][3072], ADD and MUL + GELU_TANH) 3.20 ms each — DDR-bound at the contiguous
ADD's ~1 ns per element (the stride-0 row past the 2048-element replay costs nothing extra);
Piper's ADDs 0.21 / 0.81 / 1.62 ms (196 608 / 786 432 / 1 572 864 elements).  The perf-regression
baseline is unchanged (same bitstream; no CNN or BERT choice moved).

**Fact registry:** `cost_model.call_overhead` (`CALL_OVERHEAD` = the model's call floor),
`cost_model.board_fit`, `cost_model.conv_anchors`, `vlm.vision_gelu`, `piper.vop_sums`; 60 facts
pass.

## 5. Next steps

1. **SmolVLM's GELU in one pass.**  The ADD writes `s` only for the MUL to read it back: 22 of the
   image's VectorOP passes move 3 × 6 MB each.  A fused bias (an `ADD`-then-`MUL` op, or the bias
   folded into fc1 at the output exponent) would halve that — ~35 ms per image.
2. **The engine cost model's structure.**  At 250 MHz ConvKernel's recurrence misses narrow,
   input-heavy MatMuls by 2.5–3× (p90 45 %): a DDR bandwidth shared by the x, weight and y ports
   would fix the shape, constants cannot (§4.3).
3. **Piper's `encode_32`.**  Its six 32×576×768 MatMuls run on ConvKernel at 0.60 ms where
   MatmulKernel's image path takes 0.50 ms (both constant sets choose ConvKernel).

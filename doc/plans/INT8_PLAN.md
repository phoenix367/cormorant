# INT8_PLAN — an 8-bit configuration of Cormorant

Goal: the next version of Cormorant gets an 8-bit configuration: 8-bit weights and
8-bit arithmetic in kernels of their own, with the models fine-tuned for it.
This document is the investigation that picks the quantization scheme (§0–§6) and
the development plan that follows from it (§7).

**Status (2026-11-22):** phase 0 (the study) done.  The scheme is chosen (§0, §6):
- int8 weights with one power-of-two exponent per output channel;
- int8 activations with a power-of-two exponent per block of 32 values, set at run
  time by the kernel that writes the tensor.

Screened on SmolLM2-135M / 360M, ResNet-18 and MobileNet v2, with a fine-tuning
trial on 135M.  The open decisions are in §8.

Scripts (scratch, not in git; they become study scripts in phase 1, §7):
`/mnt/data/cormorant_repro/tmp/claude_scratch/int8/` —
- `llm_screen.py`: fake quantization of a SmolLM2 checkpoint in torch, every scheme
  of §3.2;
- `cnn_screen.py`: the same for torchvision's ResNet-18 and MobileNet v2, BN folded;
- `qat.py`: the fine-tuning trial (§4).

## 0. Summary

| question | answer |
|---|---|
| can the 8-bit kernels sit beside the int16 ones? | no.  BRAM is 142 / 144, URAM 58 / 64, CLBs 98 % (§1).  The 8-bit kernels go into a bitstream of their own and replace the int16 kernels there; production stays int16 |
| weights | **int8, symmetric [−127, 127], one power-of-two exponent per output channel.** A single exponent per tensor is unusable: SmolLM2-135M ppl 18.16 → 74.8, MobileNet v2 78.8 → 72.8 %.  Per channel costs +0.26 % ppl, and the CNNs stay within 0.2 points.  Real-valued scales gain nothing measurable (§3) |
| activations | **int8 with a power-of-two exponent per block of 32 values along the reduction axis.** The kernel that writes the tensor sets each block's exponent from the block maximum, as a 3-bit shift below a calibrated per-tensor exponent.  With no training: SmolLM2-135M +0.5 % ppl and 360M +0.7 %.  One static exponent per tensor gives +9.9 % on 135M and collapses 360M (ppl 2747); one exponent per token gives +3.8 %.  CNNs at float: ResNet-18 78.2 %, MobileNet v2 78.7 % (float 78.2 / 78.8) |
| attention, KV cache | K, V and q at 8 bits with static per-head exponents; P at 8 bits in blocks of 32 keys (a static 2^−8 grid costs +7 %).  **The complete 8-bit configuration costs +1.2 % ppl on 135M and on 360M before any fine-tuning**, against +0.1 % for int16 |
| what stays wider | the residual stream (int32 on the row-ops unit), the norms, the softmax arithmetic, the accumulators |
| fine-tuning | a polish, not a requirement, for the chosen scheme.  600 distillation steps (23 min on the 4 GB GTX 1050) take its 135M linears from +0.6 to +0.3 %.  The same steps leave the static scheme at +5.3 %.  The complete configuration ends at +0.8 to +1.2 %, with its LM head frozen for lack of GPU memory (§4) |
| why not real-valued scales (TFLite) | no accuracy gain at 8 bits in any screen here or in the literature after fine-tuning, while every kernel output would need a 32-bit multiplier and the bias / residual paths a rescale |
| why not FP8 / MXFP8 | worse than int8 at equal bits on CNNs, more logic per MAC, and no DSP48E2 packing (§2) |
| hardware gain | two int8 MACs per DSP48E2 (§5.1): the MAC grids double at the same DSP count, and every weight and activation byte halves.  LLM decode ≈ +50 % tokens/s, every weight pool halves (§5.4) |

## 1. Constraints

**Area.**  The production bitstream `b4e63457057a` uses 845 / 1248 DSP, 87.0 k /
117.1 k LUT (74 %), 142 / 144 BRAM tiles, 58 / 64 URAM and 98 % of the CLBs
(`facts.yaml` `hw.utilization`).  A second set of five kernels does not fit
beside the first, and neither does any enlarged copy of one of them.  So "separate
kernels" means:
- an 8-bit kernel set in a bitstream of its own (an 8-bit hardware configuration,
  `platforms/kv260_i8.json`);
- the int16 bitstream and its libraries unchanged.

The int8 kernels halve every on-chip buffer counted in elements (weight cache, line
buffers, FIFOs, A panel).  That frees the block RAM and UltraRAM that their wider
accumulators need (§5.3).

**Today's numerics** (doc/ARCHITECTURE.md §5.1–5.2, `inference-scheduler/src/numeric.py`):
- every DMA tensor is int16 with a power-of-two exponent (Q8.8 or per tensor / per
  channel);
- the kernels multiply raw int16, sum in a wrapping int32 and write
  `sat16(floor(acc / 2^8))`;
- a constant weight is encoded at the rank-1 exponent `f_w = f_out + 8 − f_in`, so
  every rescale is a shift;
- the transformer residual is int32 at 2^−15 on the row-ops unit.

The 8-bit scheme should keep the shift-only requantization and the scheduler's
exponent bookkeeping.

**What limits each workload today:**

| workload | limit | what 8 bits change |
|---|---|---|
| LLM decode (SmolLM2-135M 49–51 ms/token, 360M 132 ms) | weight bandwidth: MatmulKernel's GEMV reads 16 int16 weights a cycle through two ports, about 34 of the 50 ms on 135M | 32 weights per cycle on the same ports |
| CNNs, prefill, ViT encoders | ConvKernel's MAC grid (512 MAC/cycle) | 1024 MAC/cycle on the same 512 DSPs (§5.1) |
| depthwise CNNs, stereo | ConvKernel's x path (DEPTHWISE_PLAN) | half the bytes per pixel |
| CMA | the chat server swaps models: 299 + 758 + 495 + 216 + 186 + 54 MiB of pools against `cma=1400M` | weights at half size |

**Fine-tuning resources:**
- a GTX 1050 with 4 GB.  Pascal (sm_61): CUDA 13 dropped it, but torch 2.14.0+cu126
  runs on it at 1.37 TFLOPS fp32, from a scratch venv
  (`/mnt/data/cormorant_repro/tmp/claude_scratch/int8/.venv-cuda`; the repository's
  `.venv-export` is unchanged);
- 8 CPU cores and 46 GB of RAM;
- no ImageNet, COCO or SQuAD training sets locally.  Fine-tuning is therefore
  distillation from the float model on unlabeled inputs: text, images, audio.

## 2. Candidate schemes (literature)

A survey of the primary sources (list at the end):

| scheme | 8-bit accuracy, as published | hardware on the KV260 |
|---|---|---|
| per-tensor weights and activations, power-of-two scales (Xilinx DPU, TQT [14], AMD Quark VINT8 [36]) | PTQ collapses depthwise CNNs: TQT's MobileNet v1 / v2 drop to 0.6 / 0.3 %.  About 2 epochs of retraining recover float or better: TQT MobileNet v2 71.8 % against float 70.1, ResNet-50 75.4 against 75.2 | shift-only requantization; the simplest |
| per-channel weights, per-tensor activations, real-valued scales with an integer multiplier and a shift (gemmlowp / TFLite [10][11]) | PTQ MobileNet v2 69.8 against 71.9; QAT 71.1 (Krishnamoorthi [12]).  Nagel [13]: per-channel brings no significant gain at W8A8 | one multiplier per output lane; residual adds need a rescale |
| per-token dynamic activations (ZeroQuant [20]) | GPT-3-350M Wikitext-2: 21.5 float, 26.2 static PTQ, 21.7 per token | a row maximum before every linear |
| SmoothQuant [19] | LLaMA-7B 11.51 → 11.56 at W8A8 | free: folded into the norms and weights |
| MXINT8 (32 values share an 8-bit power-of-two scale [16]) | direct cast, no calibration: ResNet-18 70.80 against 70.79, MobileNet v2 71.61 against 72.14, LLaMA-7B ppl 9.504 against 9.488 | a shift per block in the accumulation |
| FP8 E4M3 / MXFP8 [16][17] | below int8 on CNNs after QAT: ResNet-18 69.35 against 70.43, MobileNet v2 70.89 against 71.82 | 53–183 % more gates per MAC than int8; no DSP packing |
| W8A16 weight-only | lossless at 8 bits on LLMs [28] | halves weight bytes, but one MAC per DSP |

On the FPGA, one DSP48E2 does two int8 MACs when the two products share an
operand (WP486 [1][2]).  That packing is what doubles the MAC grids (§5.1).

## 3. Accuracy without fine-tuning

### 3.1 Method

**LLMs (`llm_screen.py`)** — the deployed protocol:
- **Position 0.** `<|im_start|>` is position 0, and its K / V come from a float run
  (the sink); positions 1… run the quantized model.
- **Data.** 12 windows of 1023 tokens of the WikiText-2 held-out text
  (`heldout_wikitext2_test.txt`); calibration on the 6 windows of
  `calib_wikitext2_valid.txt`.
- **Metrics.** ppl, and top-1 agreement with float.
- **What is quantized.** q / k / v / o / gate / up / down weights and inputs;
  options for the LM head, the KV cache, q and P.  Norms, residual and SiLU stay
  float.
- **Calibration.** Weights round half away from zero, saturate, and take the
  candidate scale with the lowest error (MSE).  Static activation scales are MSE-best
  over 7–16 candidates on the calibration windows.

For reference, today's int16 policy `pow2+sink+p12` is +0.1 % ppl and 0.969 top-1
on its own held-out set (CHAT_PLAN §10.2).

**CNNs (`cnn_screen.py`):**
- torchvision ResNet-18 / MobileNet v2 (IMAGENET1K_V1), BN folded;
- top-1 over all 1000 classes on Imagenette's 3925 validation images (10 ImageNet
  classes);
- calibration on 260 training images;
- every Conv / Linear input quantized.

Rules: `pt` = one exponent per tensor, `pc` = per output channel, `p2` =
power-of-two, `f` = real-valued, `mx<N>` = a power-of-two exponent per block of
N values along the reduction axis, `bmx<N>w<W>` = the same within W steps below a
calibrated per-tensor exponent (the hardware's form, §5.2), `tok` = per token.

### 3.2 SmolLM2-135M (float ppl 18.156)

| weights | activations (linear inputs) | other | ppl | Δ | top-1 vs float |
|---|---|---|---:|---:|---:|
| pt-p2 | float | | 74.752 | +312 % | 0.453 |
| pc-p2 | float | | 18.203 | +0.26 % | 0.957 |
| pc-f / pc-f MSE-clipped / mx32 | float | | 18.19 | +0.2 % | 0.971–0.973 |
| pc-p2 | float | LM head 8-bit | 18.284 | +0.71 % | 0.941 |
| pc-p2 | pt-p2 static | | 19.948 | +9.87 % | 0.818 |
| pc-f | pt-f static | | 19.709 | +8.55 % | 0.828 |
| pc-f | pt-f static, down_proj input 16-bit | | 18.701 | +3.00 % | 0.895 |
| pc-f | pt-f static + SmoothQuant α 0.5 in the norms | | 19.388 | +6.78 % | 0.852 |
| pc-f | same, down_proj input 16-bit | | 18.432 | +1.52 % | 0.931 |
| pc-f / pc-p2 | per token, real / power-of-two | | 18.612 / 18.845 | +2.5 / +3.8 % | 0.907 / 0.872 |
| pc-p2 | **mx16 / mx32 / mx64** | | **18.249 / 18.264 / 18.290** | **+0.5 / +0.6 / +0.7 %** | 0.948 / 0.946 / 0.936 |
| mx32 | mx32 | | 18.217 | +0.33 % | 0.953 |
| mx32 | mx32 | K, V 8-bit (static per head) | 18.248 | +0.51 % | 0.950 |
| mx32 | mx32 | K, V, q 8-bit | 18.286 | +0.71 % | 0.947 |
| pc-p2 | mx32 | K, V, q, LM head 8-bit | 18.372 | +1.19 % | 0.925 |
| pc-p2 | pt-p2 static | K, V, q 8-bit | 20.060 | +10.5 % | 0.818 |

(The first SmoothQuant runs that also folded into v_proj / up_proj were void: the
float sink's V row came from the unfolded model.  Fixed; the rerun is in §3.3.)

Findings:
- Per-channel weights are necessary and sufficient for the weights.
- With 8-bit activations, one static exponent per tensor costs ~10 % ppl.  The
  outlier channels of the down_proj input and the norm outputs set the range for
  everyone.
- Exponents per block of 32 confine an outlier to its own block: +0.6 % with no
  training and no smoothing.  The block size matters little (16 → 64: +0.5 → +0.7 %).
- The KV cache and q go to 8 bits with static per-head exponents almost free.

### 3.3 The hardware's form, P, SmoothQuant, SmolLM2-360M

SmolLM2-135M (float 18.156):

| weights | activations | other | ppl | Δ | top-1 |
|---|---|---|---:|---:|---:|
| pc-p2 | pt-p2 static + SmoothQuant α 0.5 (all four folds) | | 19.023 | +4.77 % | 0.875 |
| pc-p2 | same, down_proj input 16-bit | | 18.417 | +1.44 % | 0.917 |
| pc-p2 | **bmx32w8** (3-bit block shifts) | | **18.252** | **+0.53 %** | 0.943 |
| pc-p2 | bmx32w16 (4-bit) | | 18.273 | +0.64 % | 0.944 |
| pc-p2 | bmx32w8, o_proj input static | | 18.291 | +0.74 % | 0.940 |
| pc-p2 | bmx32w8 | K, V, q, LM head 8-bit; P 2^−12 (16-bit) | 18.373 | +1.19 % | 0.925 |
| pc-p2 | bmx32w8 | same, P uint8 at 2^−8 | 19.427 | +7.00 % | 0.860 |
| pc-p2 | bmx32w8 | same, P in blocks of 32 keys | 18.390 | +1.29 % | 0.924 |
| pc-p2 | bmx32w8 | same, P in blocks of 16 keys, unsigned | 18.374 | +1.20 % | 0.921 |
| pc-p2 | **bmx32w8, o_proj input static** | **K, V, q, LM head 8-bit; P blocks of 32 keys** | **18.376** | **+1.21 %** | **0.922** |

SmolLM2-360M (float 13.798):

| weights | activations | other | ppl | Δ | top-1 |
|---|---|---|---:|---:|---:|
| pc-p2 | float | | 13.870 | +0.52 % | 0.968 |
| pc-p2 | float | LM head 8-bit | 13.926 | +0.92 % | 0.944 |
| pc-p2 | **pt-p2 static** | | **2747** | collapse | 0.104 |
| pc-f | pt-f static + SmoothQuant in the norms, down_proj input 16-bit | | 14.041 | +1.76 % | 0.937 |
| pc-p2 | mx32 / bmx32w8 | | 13.891 / 13.893 | +0.67 / +0.69 % | 0.956 |
| pc-p2 | bmx32w8 | K, V, q, LM head 8-bit; P 2^−12 | 13.963 | +1.19 % | 0.929 |
| pc-p2 | **bmx32w8** | **K, V, q, LM head 8-bit; P blocks of 32 keys** | **13.971** | **+1.25 %** | **0.927** |

Findings:
- **The hardware's form loses nothing.**  Block exponents bounded to 3-bit shifts below a
  calibrated per-tensor exponent are as good as unbounded MX blocks (18.252 against
  18.264), and 4-bit shifts gain nothing.
- **A static exponent per tensor is not an option for LLMs.**  On 360M one static
  exponent per tensor collapses (ppl 2747).  The 360M model's activation outliers are
  larger, and static per tensor is not even a fine-tuning starting point there.
- **SmoothQuant is not enough on its own.**  Fixed, it brings static 135M from +9.9
  to +4.8 %, and to +1.4 % only with the down_proj input kept at 16 bits.  That would
  mean a 16 × 8 datapath beside the 8 × 8 one.
- **P needs block exponents.**  At 8 bits it costs +7 % on a static 2^−8 grid.  In
  blocks of 32 keys (the reduction axis of P·V) it costs nothing.
- **The o_proj input can stay static** (+0.04): ConvKernel's P·V need not produce
  block exponents.
- **The complete 8-bit configuration costs +1.2 % ppl on both sizes with no training:**
  every linear's weights and inputs, K, V, q, P and the LM head.  For comparison,
  int16 costs +0.1 %.

### 3.4 CNNs (Imagenette, 1000-way top-1)

| weights | activations | ResNet-18 (float 78.17) | MobileNet v2 (float 78.80) |
|---|---|---:|---:|
| pt-p2 | float | 77.50 | 72.84 |
| pc-p2 | float | 78.06 | 78.60 |
| pc-f | float | 78.78 | 78.42 |
| pt-p2 | pt-p2, unsigned after ReLU (the DPU's scheme) | 77.43 | 72.28 |
| pc-p2 | pt-p2, unsigned after ReLU | 78.19 | 78.62 |
| pc-p2 | pt-p2, signed only | 77.76 | 76.92 |
| pc-f | pt-f, unsigned after ReLU (TFLite-like, symmetric) | 78.75 | 78.14 |
| pc-p2 | **mx32 over channels, signed** | **78.17** | **78.70** |

Findings:
- MobileNet v2 needs per-channel weights, as the literature says.
- A static exponent per tensor is then fine for CNNs if post-ReLU tensors use the
  unsigned range.  Signed-only costs MobileNet v2 1.7 points.
- Block exponents recover that bit with signed elements.
- Power-of-two scales lose nothing against real-valued ones.

### 3.5 Attention operands

- K, V (the KV cache) and q at 8 bits with static per-head power-of-two exponents:
  +0.03 / +0.07 ppl (§3.2).  The KV cache halves, and so do the decode attention's
  bytes.
- P (the softmax output) is 2^−12 in today's datapath.  At 8 bits: §3.3.

## 4. Fine-tuning trial

| SmolLM2-135M, 600 steps (615 k tokens, 23 min) | step 0 | 150 | 300 | 450 | 600 |
|---|---:|---:|---:|---:|---:|
| pc-p2 weights, pt-p2 static inputs | 20.017 | 19.515 | 19.394 | 19.092 | **19.121 (+5.3 %)** |
| pc-p2 weights, mx32 inputs | 18.264 | 18.233 | 18.193 | 18.217 | **18.214 (+0.3 %)** |
| the complete 8-bit configuration of §3.3 (LM head frozen at its calibration, q / k / v per-head steps learned, P straight through) | 18.437 | 18.367 | 18.307 | 18.341 | **18.370 (+1.2 %)** |

(ppl on the 12 held-out windows; top-1 agreement 0.817 → 0.851, 0.945 → 0.949 and
0.915 → 0.927.)

**The complete configuration.**
- Its step 0 is worse than §3.3's 18.376 because `qat.py` sets the q / k / v steps
  from the calibration maximum, not by MSE.
- Fine-tuning moves it to 18.31–18.37 (+0.8 to +1.2 %).
- The LM head is frozen: its weight-sized gradients do not fit in 4 GB.
- 600 steps of distillation therefore buy about what a better calibration does.
- Phase 1 has to show whether a longer run with the head trained (teacher logits on
  disk, §8) closes the remaining +1 %.  It is within the ≤ +1.5 % gate either way.

Findings:
- **Static scales.**  Fine-tuning halves the static scheme's loss, but after 600 steps
  it is still ten times the block scheme's untrained loss, and on 360M there is nothing
  to start from (§3.3).
- **Block scheme.**  Fine-tuning takes the remaining 0.6 % down to 0.3 %.

So fine-tuning is a polish step for the block scheme, not a requirement.

`qat.py` fine-tunes SmolLM2-135M by distillation:
- **Student.** Every linear fake-quantized: int8 weights with a learnable log2 step
  per output channel, inputs static per tensor (learnable log2 step) or mx32.  With
  `--p2` the steps round to powers of two (straight-through).
- **Loss.** KL(float ‖ student) over positions ≥ 1.
- **Data.** Random 256-token windows of WikiText-2 train, 4 per step; AdamW, lr
  2e-5 cosine, the steps at 2e-3.
- **Initialization.** The PTQ calibration.
- **Memory.** A custom autograd function recomputes the fake-quantized weight in the
  backward pass rather than storing it, so student, teacher and Adam fit in 4 GB.
- **Speed.** 2.2 s a step on the GTX 1050 (17 s on the CPU).

## 5. Hardware

### 5.1 Two int8 MACs per DSP48E2

The DSP48E2 multiplies 27 × 18 bits.  Two int8 weights of two output channels
share one activation x:

    A:D = w0 · 2^19 + w1        (27 bits; the packed word is a constant, stored packed)
    B   = x                     (int8, sign-extended)
    P  += (w0 · 2^19 + w1) · x  =  S0 · 2^19 + S1

- **The low field.** Weights in [−127, 127] keep every product within ±16 256, so
  16 products sum to |S1| ≤ 260 096 < 2^18.  The low field of 19 bits holds S1 for
  a whole 16-deep cascade, exactly ConvKernel's chain length.
- **Extraction.** `S1 = sext(P[18:0])`, `S0 = (P − S1) >> 19`: exact, one adder per
  field at the chain's end.
- **The weight range.** With −128 allowed, `A:D` could overflow 27 bits and the
  16-term bound would fail.  So weights are symmetric, which costs nothing (§3).
- **Comparison with WP486 [2].** WP486 spaces the fields 18 bits apart, allows
  −128, and gets 7 terms.
- **Unsigned activations** (post-ReLU) are stored as signed with a fixed offset:
  `x_s = x_u − 128`, a flip of the sign bit.
  - The constant `128 · Σ_k w[k][j]` folds into the output channel's bias.
  - Padding takes the value −128.
  - So the full 8 bits survive and the chains stay 16 deep; uint8 × int8 would allow
    only 8.
  - With block exponents (§3.4) this is optional.

Effect:
- ConvKernel: 2 pixels × 32 output channels × 16 input channels = 1024 MAC/cycle on
  its 512 DSPs.
- MatmulKernel's GEMM: 256 MAC/cycle on its 128.

**The DSP's own accumulation.**  ConvKernel's DSP 15 accumulates over the window
positions in P today.  Packed, it cannot, because the low field would overflow.
The chain's output is split every instant, and the two sums are accumulated in the
fabric or the accumulator RAM.

### 5.2 Block exponents in the datapath

A blocked int8 tensor carries:
- one calibrated per-tensor exponent E, as today;
- per block of 32 values along the reduction axis (channels of a pixel for convs,
  features of a row for linears), a 3-bit shift `δ ∈ [0, 7]`.

The block's elements are at `2^(E − δ)`.

**The producer** (ConvKernel's drain, the row-ops unit's NORM / MULACT / COPY,
MatmulKernel's drain):
1. take the block's maximum from the accumulators;
2. set δ from its leading zeros;
3. round and saturate every element at `E − δ`;
4. write the δ bytes beside the block (1 / 32 more bytes, 0.25 bit per element).

**The consumer** (ConvKernel, MatmulKernel):
- shift each 16-term block sum by its δ before accumulating;
- the shift sits where §5.1 already splits the packed fields, so its cost is a
  3-level barrel shift on ~20 bits per field;
- with E bounded and δ ≤ 7, the accumulator needs ~20 + 7 + log2(K / 16) bits,
  within 40.

Static tensors are blocked tensors with δ = 0.  CNN tensors read by VectorOP
(residual adds) or PoolingKernel can stay static per tensor (§3.4), and those
kernels then only align two per-tensor exponents by a shift.

### 5.3 Per kernel

| kernel | int8 change | area notes |
|---|---|---|
| ConvKernel | packed MAC grid (2 px × 32 oc × 16 ic); split + δ shift + 2 × 32 accumulators in fabric per pixel pair; weight cache and line buffer hold twice the elements at the same width; drain: block max → δ, int8 out, bias with the zero-point fold | accumulator buffers carry 64 sums per instant instead of 32: +8–16 URAM at 24–32 bits, paid by the halved weight cache / line buffers; or a first step at 512 MAC/cycle on 256 DSPs (bandwidth gains only) |
| MatmulKernel | GEMV 32 MAC per beat pair (int8 B), GEMM 256 MAC/cycle packed; drain → δ | the A panel halves |
| VectorOPKernel | 16 lanes of int8 per beat; operand pre-shifts for per-tensor exponents; activation / softmax units read int8, softmax P out per §3.3 | tables unchanged |
| PoolingKernel | 16 int8 lanes per beat | — |
| RowOpsKernel | int32 residual and norms unchanged; int8 / blocked outputs (NORM, MULACT, COPY), int8 inputs | small |

### 5.4 Projected gains

These are estimates from the limits of §1, not measurements.  Phase 2 prices them
with the cost models.

| workload | today | int8 estimate |
|---|---|---|
| SmolLM2-135M decode | 20.4 tok/s (49 ms; ~34 ms GEMV) | ~30 tok/s (GEMV ~17 ms) |
| SmolLM2-360M decode | 7.8 tok/s (132 ms; ~90 ms GEMV) | ~11.5 tok/s |
| chat pools (135M / 360M / SmolVLM) | 299 / 758 / 495 MiB | roughly half: all chat models resident in `cma=1400M` at once |
| ResNet-18 | 20.0 ms (grid-bound) | ~11–12 ms with the 1024 MAC/cycle grid |
| depthwise CNNs, stereo, prefill, ViT | x-path / grid-bound | 1.3–1.8× |

## 6. The choice: int8 with power-of-two blocks

1. **Weights.**
   - int8, symmetric [−127, 127], one power-of-two exponent per output channel; the
     kernel's output shift is per channel.
   - This replaces the rank-1 rule's coupling `f_w = f_out + 8 − f_in`, which at 8
     bits would let the activations' exponents dictate the weights' grid.
   - Calibration: the MSE-best of three candidate exponents.
2. **Activations read by ConvKernel / MatmulKernel.**
   - int8 elements, a calibrated per-tensor exponent E (as today's formats), and per
     block of 32 values along the reduction axis a 3-bit shift δ ∈ [0, 7] (§5.2).
   - Elements are at `2^(E − δ)`.
   - The writer of the tensor sets δ from the block's maximum and rounds to nearest,
     ties to even.
   - A tensor may be static (δ = 0 throughout).  That covers the KV cache, q, and the
     CNN tensors that VectorOP / PoolingKernel also read.
   - The o_proj input is blocked: ConvKernel's P·V writes it as blocks.  A static o
     input cost 0.2 points on 12 windows (§9.4).
3. **Attention.**
   - K, V (the KV cache) and q: int8, static per head.
   - P: int8 in blocks of 32 keys, δ set by the softmax unit.
4. **Unsigned (post-ReLU) static tensors.**  Stored with an offset of −128.  The bias
   absorbs `128 · Σ w`, and padding takes −128 (§5.1).
5. **Arithmetic.**
   - Two MACs per DSP48E2 (symmetric weights, 19-bit fields, 16-term chains).
   - Exact block sums, shifted by δ and accumulated in ≥ 40 bits.
   - Outputs: a shift, round to nearest-even and saturate.  int8 for the next MatMul /
     Conv; int16 / int32 where a row op or the residual reads them.
6. **Unchanged.**  The int32 residual stream, RMSNorm / LayerNorm, the softmax's
   arithmetic, RoPE, SiLU / GELU tables (int8 in, int8 or int16 out), and the host
   ops.
7. **Fine-tuning.**
   - Calibration first (PTQ).
   - Then, where a model's yardstick needs it, distillation from the float model:
     learnable power-of-two steps, straight-through rounding, unlabeled inputs, `qat.py`
     generalized.
   - On the GTX 1050 this covers models up to ~135M parameters.  Larger ones (360M,
     SmolVLM, Moonshine base) need teacher logits precomputed to disk or block-wise
     distillation (§8).

Against the alternatives:
- **Static per-tensor power-of-two (Xilinx DPU).**  Simplest hardware, but it collapses
  360M and leaves 135M at +5.3 % after fine-tuning.
- **Real-valued scales (TFLite).**  No measurable gain anywhere here, and a multiplier
  per output lane.
- **Per-token scales.**  Worse than blocks (+3.8 %) and needs a row maximum before
  every linear.
- **W8A16.**  Halves the weight bytes but keeps one MAC per DSP, and needs a 16 × 8
  datapath.
- **FP8.**  More logic, no packing, worse on CNNs.

The block scheme costs the hardware two things:
- a 3-bit shift where §5.1 already splits the packed fields;
- a block-maximum unit in the writers that need one: the row-ops unit, the softmax
  unit and ConvKernel's drain (the P·V output, §9.4); MatmulKernel's drain if a model
  needs it.

## 7. Development plan

| phase | content | gate |
|---|---|---|
| 0 study | this document | the scheme chosen (§6) |
| 1 numerics | `inference-scheduler/src/int8_spec.py` — the integer specification (block format, δ selection, packed products, accumulation, rounding); the study scripts' `i8` policies emulating it bit for bit (`llm_study.py`, `vlm_study.py`, `bert_study.py`, `asr_study.py`, the Piper / stereo / YOLO / V-JEPA 2 / SmolVLA studies); formats JSONs with per-channel weight exponents and block flags; the distillation tool (`qat.py` → a study script); a screen of every shipped model | each model within its yardstick: LLM ppl ≤ +1.5 %, CNN top-1 / mAP within 0.5 point, WER, log-mel, EPE, EM / F1 as their plans define |
| 2 kernels | separate 8-bit kernels with their C++ reference models and Verilator benches — MatmulKernel first (the GEMV: decode), then ConvKernel (packed grid, δ shift, fabric accumulators, block-max drain), VectorOP / Pool (16 int8 lanes, exponent pre-shifts), the row-ops unit's and the softmax unit's int8 / blocked outputs; OOC synthesis at 300 MHz | bit-exact with the spec; area within the 8-bit build's budget |
| 3 scheduler | an int8 `DataType`, the `axi.numeric` extensions (per-channel output shifts, block flags, the −128 offset), weight encoding and packing, the simulator and `host_emu` software kernels bit-exact with the spec, the cost models | every library generated for the 8-bit platform, bit-exact on `host_emu` |
| 4 bitstream | `platforms/kv260_i8.json`, the 8-bit block design (a hw project of its own), timing at 250 MHz, the performance model, the board gates | every library bit-exact on the board; the §5.4 projections measured |
| 5 models | calibrate, then fine-tune where needed, every shipped model; the chat server runs the 8-bit bundle | quality gates of phase 1 on the board |

## 8. Open decisions and risks

**Decisions for the maintainer:**
1. **One 8-bit bitstream replacing the int16 kernels** (recommended), against a
   bitstream swap per workload.
   - A library is built for one configuration.
   - The chat server would load the 8-bit bundle, since all its models fit in CMA at
     once at 8 bits.
2. **ConvKernel's grid in the 8-bit build.**
   - 1024 MAC/cycle on the 512 DSPs needs 2× accumulator throughput: +8–16 URAM,
     paid by the halved weight cache.
   - Alternatively, a first step at 512 MAC/cycle on 256 DSPs: the bandwidth gains
     only, and room left for the δ logic.
3. **Where the 8-bit block design lives.**  A hw project of its own beside
   `hw/cormorant_hw_128` (a submodule change).
4. **Fine-tuning larger models** (360M, SmolVLM, Moonshine base).  The 4 GB GPU holds
   135M with Adam.  Beyond that:
   - teacher logits precomputed to disk;
   - block-wise distillation (EfficientQAT [27]);
   - or the CPU at ~17 s per 1 k tokens.

**Not screened yet (phase 1):**
- BERT-base: the literature finds its FFN residual sensitive, but here the residual
  stays int32;
- Moonshine, Piper (log-mel / audio quality), LightStereo-S (a regression output),
  YOLOv5n (mAP), the SigLIP / V-JEPA 2 ViTs, SmolVLA, Gemma 3.

**Not measured yet:** the numbers of §5.4 are estimates.  Phase 2 prices them with
the cost models.

## 9. Phase 1 log

### 9.1 The integer specification (2026-11-23)

`inference-scheduler/src/int8_spec.py` (test: `test/test_int8_spec.py`, 19 cases
against exact arithmetic) fixes the details §6 left open:
- **Weights.**  `encode_weights`: round half to even, [−127, 127], per output
  channel the least-error exponent of three (the finest that does not clip, and two
  finer).
- **Block rule.**  `block_round`: d = the largest shift in [0, 7] at which the
  block's rne values fit [−128, 127]; one rounding per element; none fits → d = 0,
  saturated.  For the hardware this means a compare of the block's maximum and
  minimum per candidate shift.
- **MatMul.**  `matmul`: block sums shifted by 7 − d into an accumulator at
  `f_x + 7 + f_w[n]`, two's complement over 40 bits.  The spec evaluates it exactly
  in float64 (|x · 2^(7−d)| < 2^14, partial sums < 2^53).  A static input takes any
  K; a blocked one needs whole blocks.
- **Requantization.**  `requant` / `requant_blocked`: a per-channel shift, rne,
  saturation (int8, or int16 where a row op reads the output).
- **Packed pair and offset.**  `pack_pair` / `unpack`: the 27-bit word and the
  extraction, checked over random and extreme 16-term chains; −128 weights break
  them.  `to_offset` / `offset_bias`: post-ReLU tensors stored at −128.

### 9.2 The LLM study's `i8` policy

`demo/chat/scripts/llm_study.py` gets:
- the policy modifier `i8` (ablations `i8lin`: 16-bit attention; `i8h16`: 16-bit LM
  head);
- `Model.lin8` and `Model.attn8_head` on `int8_spec`;
- a second calibration pass, `I8Cal`.  It picks the static operands' int8 exponents
  (q per head, K / V per KV head, the o input per tensor) by least squared error over
  the maximum's exponent and 7 finer ones, and the blocked inputs' int16 exponent
  E₈ + 8.

Tested on the tiny Llama: `inference-scheduler/test/test_int8_llm.py`, 4 cases, a
linear and an attention head bit-exact with the spec.

Two findings from bringing it up, both now requirements of the spec:
1. **Static operands need least-error exponents.**  The int16 exponent − 8 gave
   some heads an int8 step of 8, and ppl 1009.
2. **A blocked tensor's producer must round from a grid finer than its finest
   block.**  The int16 formats keep the norm and SiLU outputs at Q8.8, which protects
   the int16 weights; at 8 bits the weights no longer need it.  Taken as the base,
   Q8.8 left the blocks 4–6 bits short.  The row-ops unit's NORM / MULACT must round
   to blocks from their internal precision, not from the int16 they write today.

SmolLM2-135M, the study's 3 held-out windows (float 15.598), 12 greedy answers:

| policy | ppl | Δ | top-1 held-out | KL | identical answers |
|---|---:|---:|---:|---:|---:|
| `pow2+sink+p12+vsmx+rowops+drope` (int16, deployed) | 15.600 | +0.01 % | 0.968 | 0.0025 | 4/12 |
| `…+i8` | 15.893 | +1.9 % | 0.922 | 0.018 | 1/12 |
| `…+i8h16` (LM head 16-bit) | 15.824 | +1.45 % | 0.924 | 0.015 | 1/12 |
| `…+i8lin` (attention 16-bit; before the base fix) | 15.904 | +2.0 % | 0.934 | 0.017 | – |
| torch screen, same windows: everything 8-bit / linears only | 15.752 / 15.615 | +1.0 / +0.1 % | | | |

Bisection of the gap to the screen (a scratch ppl, float 15.656, LM head 16-bit):

| | o input static | o input blocked |
|---|---:|---:|
| the screen's linears in the study's float model | — | 15.688 (+0.21 %) |
| the study's 8-bit linears | 15.783 (+0.81 %) | 15.739 (+0.53 %) |
| + the study's 8-bit attention | 15.875 (+1.40 %) | 15.841 (+1.18 %) |

What the bisection shows:
- **Row ops.**  The integer row ops add nothing: on the float-row-ops base the same
  `i8h16` gives 15.817.
- **The static o input** costs ~0.25 %.  ConvKernel's P·V should write blocks: its
  drain sees all of a token's channels.
- **The 8-bit attention** costs ~0.6 %, against ~0.2 % in the screen.  The screen
  quantized q / K before RoPE and never rounded the sink row.  The sink is not the
  cause: 2 of 180 sink rows clip, and protecting them is worse (15.953).
- **The operand ablation** (one of q, K, V, P back at 16 bits) is §9.3.

### 9.3 The attention's operands, and the evaluation's resolution

With one operand at a time back at 16 bits (the rest as in `i8h16`, o input static):

| | q16 | k16 | v16 | p16 | all 8-bit |
|---|---:|---:|---:|---:|---:|
| ppl (scratch definition, float 15.656) | 15.886 | 15.912 | 15.860 | 15.899 | 15.875 |

No single operand accounts for the attention's ~0.6 %.  The spread between these
runs (±0.04, ~0.3 %) is the resolution of 3 windows of 1024 tokens, so the
attention's cost and the study-vs-screen gap below ~0.3 % need the full 12
held-out windows.

The study gets the modifier `i8ob`: P·V writes the o input as blocks of 32 straight
from its accumulator (`requant_blocked` per head; base `pvb`, the int8 exponent of
the calibration maximum).  The o linear re-derives the same values: the block rule is
idempotent on values, `test_rule_is_idempotent_on_values`.  On the 12 windows: §9.4.

### 9.4 SmolLM2-135M on all 12 held-out windows

`llm_study.py study --windows 12`: `--windows` now asks `build_data` for that many
windows, up to what the held-out text holds; before, it could only shrink the
profile's 3.  Float 18.156 is the torch screen's float on the same windows.

| policy | ppl | Δ | top-1 held-out | KL held-out |
|---|---:|---:|---:|---:|
| `pow2+sink+p12+vsmx+rowops+drope` (int16, deployed) | 18.192 | +0.20 % | 0.965 | 0.0028 |
| `…+i8` (o input static) | 18.495 | +1.87 % | 0.911 | 0.019 |
| **`…+i8ob`** (o input blocked) | **18.456** | **+1.65 %** | **0.915** | **0.018** |
| torch screen: everything 8-bit, o static (§3.3) | 18.376 | +1.21 % | 0.922 | |

What these show:
- **The o input.**  A blocked o input is worth 0.2 points: P·V writes blocks (§6
  item 2 changes accordingly: no static o input).
- **The integer spec costs ~0.4 points more than the screen.**  About 0.2 of that is
  the int16 base's own cost on these windows.  The rest is the bounded P window,
  the int16 producers' double rounding, and the screen's pre-RoPE q / K, each below
  this evaluation's resolution on its own (§9.3).
- **The gate.**  +1.65 % sits above the phase's ≤ +1.5 % gate.  The levers, in order
  of cost:
  1. the LM head (~0.5 points, §9.2) at 16 bits or W8A16;
  2. fine-tuning against the integer spec (`qat.py` against the spec, not the
     screen);
  3. per-channel smoothing of q / K before RoPE (the screen's form).

## Sources

[1] https://docs.amd.com/v/u/en-US/wp486-deep-learning-int8 ·
[2] https://www.edge-ai-vision.com/2016/11/deep-learning-with-int8-optimization-on-xilinx-devices/ ·
[10] Jacob et al., https://arxiv.org/pdf/1712.05877 ·
[11] https://developers.google.com/edge/litert/performance/quantization_spec ·
[12] Krishnamoorthi, https://arxiv.org/pdf/1806.08342 ·
[13] Nagel et al., https://arxiv.org/html/2106.08295v1 ·
[14] TQT, https://arxiv.org/html/1903.08066v3 ·
[16] OCP MX, https://arxiv.org/html/2310.10537 ·
[17] FP8 vs INT8, https://arxiv.org/html/2303.17951 ·
[19] SmoothQuant, https://arxiv.org/html/2211.10438 ·
[20] ZeroQuant, https://arxiv.org/html/2206.01861v1 ·
[27] EfficientQAT, https://arxiv.org/html/2407.11062 ·
[28] LLM-QAT, https://arxiv.org/html/2305.17888 ·
[36] https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_quantization/model_quantization.html

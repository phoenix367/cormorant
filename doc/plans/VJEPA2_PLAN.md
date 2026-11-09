# VJEPA2_PLAN — V-JEPA 2 video classification on the KV260

Goal: one of Meta's V-JEPA 2 video models on the FPGA.  §0 checks which of
them can fit; §1 is the study of the one that fits best, the ViT-L
Something-Something v2 classifier (the `model-study` skill, route C); §2 its
implementation (the frontend `src/vjepa2.py`, its ops, `libvjepa2.so`); §3 the
board: 13.8 s per video, bit-exact with the specification.

Scripts: `demo/video_classification/scripts/` —
- `vjepa2_study.py`: fetch, validate, study, ablate, costs, formats;
- `vjepa2_q.py`: the numpy float64 reference and the bit-level emulation of the
  partition: the specification the frontend and the library match;
- `vjepa2_study_inputs.json`: the pinned checkpoint and dataset, the sample
  selection, and the hashes of every decoded clip;
- `requirements-study.txt`: the chat studies' `.venv-export` plus pyarrow;
- `generate_vjepa2_project.py`, `vjepa2_sched_check.py`, `vjepa2_host_emu.py`,
  `vjepa2_board.py`: the library and its gates (§2, §3; usage:
  `demo/video_classification/README.md`).

Assets and results: `demo/video_classification/assets/` (not in git; on the
maintainer's machine a link to /mnt/data): the checkpoint (1.4 GB), the dataset
(463 MB), the selected videos and their clips (1.6 GB), and `study/`.

## 0. Which V-JEPA 2 model (2026-11-08)

The family was checked from Meta's source (facebookresearch/vjepa2, MIT), the
paper (arXiv 2506.09985) and the Hugging Face configs, with the board's measured
rates: ~80–100 GMAC/s for transformer layers (SmolVLA's vision encoder,
SmolLM2-360M's prefill) and a DMA pool limit of ~1335 MiB (`cma=1400M`; at most
1600 MB, the low 2 GB of DDR).

| model | int16 weights | per input | verdict |
|---|---:|---|---|
| **V-JEPA 2-AC** (action-conditioned world model: ViT-g encoder 1012 M + predictor 305 M) | 2513 MiB | encoder 266 GMAC = ~3.3 s per frame; predictor ~1 s per planning sample; the paper's planner (800 samples × 10 iterations, 16 s per action on an RTX 4090) ~2.3 h per action | **NO-GO**: the weights are above the CMA limit, and planning is far too slow |
| ViT-g / g-384, 2.1 g / G backbones and their heads | ≥ 1931 MiB | — | NO-GO (memory) |
| ViT-H backbone | ~1.2 GiB | ~20 s per clip | fits only with nothing else loaded, and Meta released no head for it |
| 2.1 ViT-B/16-384, ViT-L/16-384 backbones | 162 / 577 MiB | 0.7 / 2.4 s per image | fit, but have no released heads |
| 32-frame classifiers (Diving48, EPIC-KITCHENS-100) | — | 4096 tokens | the softmax unit's row mode stops at 2048 keys, and raising that limit costs block RAM the build doesn't have (142 / 144 used) |
| **ViT-L, 16 frames at 256, SSv2 classifier** | 671 MiB | 2048 tokens, 935 GMAC | **the best fit**: §1 |

The 3-D RoPE that every one of these uses maps onto the row-ops unit's ROPE op
(§1.6).

## 1. Study: the ViT-L SSv2 classifier (2026-11-09)

**Verdict: GO with policy `pow2+p12+vgelu+vsmx+rowops`.**  This is SmolVLM's and
SmolVLA's recipe:
- per-channel power-of-two exponents;
- P at 2⁻¹²;
- GELU and the softmax on VectorOPKernel;
- the row-ops unit's int32 residual, LayerNorms, bias adds and RoPE.

On 100 Something-Something v2 clips the int16 datapath does as well as float:
- it agrees with the float64 reference on 99 / 100 top-1 classes;
- its accuracy equals float's (71.0 %);
- its KL to float's class distribution is 0.0023, against bf16's 0.0017.
  Measured per clip, that difference is not statistically significant.

The weights take 671 MiB as int16.  The projected time is 9–12 s per clip.

### 1.1 Model

`facebook/vjepa2-vitl-fpc16-256-ssv2` @ `4aa02df8` (MIT), loaded as transformers
5.17's `VJEPA2ForVideoClassification`.  The checkpoint is float32: 353.4 M
parameters used, plus a 22.1 M predictor that the classifier doesn't use.

| part | shape | parameters |
|---|---|---:|
| patch embedding | Conv3d 3 → 1024, kernel = stride 2 × 16 × 16: 16 frames at 256 × 256 → 8 × 16 × 16 = **2048 tokens** | 1.6 M |
| encoder | 24 pre-LN layers, 1024 wide, 16 heads of 64, GELU (erf) MLP 4096, LayerNorm ε 1e-6, biases on q / k / v / proj, 3-D RoPE on q and k; a final LayerNorm | 302.3 M |
| attentive pooler | 3 self-attention layers (the same block without RoPE), then a cross-attention layer: one learned query attends LN1(h) through q / k / v projections, with no output projection; residual = query + attention, then LN2 and the MLP | 49.3 M |
| classifier | 1024 → 174 (the SSv2 classes) | 0.18 M |

How the 3-D RoPE works:
- Each head of 64 dims is split by position axis: dims 0–19 rotate by the
  token's frame, 20–39 by its row and 40–59 by its column.  Dims 60–63 are not
  rotated.
- The rotations act on interleaved pairs (2i, 2i + 1).
- Element j of a 20-dim segment turns by pos · 10000^−((j mod 10)/10).  This
  comes from HF / Meta's `repeat(…, 2)` expansion, and it means the two elements
  of a pair turn by different angles.

Preprocessing (the checkpoint's video processor, reproduced in
`vjepa2_q.make_clip`):
- **Frame sampling:** Meta's one-segment sampling, 16 frames with a step of 4.
  On short videos the indices repeat.
- **Resize:** bilinear, PIL-style (a horizontal pass, then a vertical one, each
  rounded to uint8), to a shortest edge of 292.
- **Crop:** a 256 × 256 center crop at the floor offset.
- **Normalisation:** ImageNet mean and std.

### 1.2 Method

- **Reference:** `vjepa2_q.py` in numpy float64, validated against transformers
  in torch float32 on 3 clips (`vjepa2_study.py validate`):
  - encoder output within 1.0e-4 relative;
  - logits within 1.2e-5;
  - top-1 equal, and correct, on all three;
  - `make_clip`'s pixels against the checkpoint's own video processor: at most
    1 level apart, on ≤ 0.01 % of the pixels.
- **Data:** MTEB's Something-Something v2 subset `mteb/SomethingSomethingV2` @
  `13bbc49a`: 5444 labelled validation videos in one parquet file.  The videos
  are Qualcomm's (20BN) SSv2, for research use under its license.
  - Evaluation: 200 rows spread evenly over the file, of which the study scores
    the first 100 (70 distinct classes).
  - Calibration: 16 other rows.
  - The videos are decoded with ffmpeg (rgb24, every frame), mostly at 240 × 427
    and 19–72 frames long.  Every clip is pinned by SHA-256.
- **Metrics,** each against the float64 reference:
  - top-1 agreement and top-5 overlap;
  - KL of the class distributions;
  - the relative error of the logits, of the encoder output (2048 × 1024) and
    of the pooled row;
  - accuracy against the labels.
- **Yardstick:** bf16 at every boundary.
- **The emulated partition** (`vjepa2_q.py`'s docstring; llm_study and
  vlm_study conventions):
  - Kernels take raw int16, sum exactly, and write floor(acc / 2⁸) saturated.
    Host regions compute in double and round half to even.
  - Pixels enter as their uint8 values at exponent 0.  The normalisation is
    folded into the patch MatMul: W / (255 · std_c) and b − Σ W · mean_c / std_c.
  - The layers are `vlm_study.VisionModel`'s.
  - GELU (erf form) is VectorOPKernel's ADD + MUL + act GELU.
  - The attention softmax runs in the softmax unit's row mode, 2048 keys per row.
  - The residual stream is int32 at 2⁻¹⁵, with the LayerNorms and bias adds on
    the row-ops unit.
  - RoPE is the unit's ROPE on the head dims permuted into rotate-half order,
    with int24 cos / sin tables per token.
  - The cross layer's query projection is a constant (precomputed in float64).
    Its one-row tensor classes take per-tensor exponents: per-channel exponents
    calibrated on one row per clip over-fit the calibration clips.

### 1.3 Results

**100 evaluation clips** (`study -n 100`; the rerun after §2.1's resize fix):

| policy | top-1 agree | top-5 overlap | KL (max) | KL median | logits rel | encoder rel | pooled rel | accuracy (top-5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float64 reference | – | – | – | – | – | – | – | 71.0 % (92.0 %) |
| bf16 (yardstick) | 1.000 | 0.978 | 0.0017 (0.022) | 0.0003 | 0.030 | 0.189 | 0.044 | 71.0 % (91.0 %) |
| **pow2+p12+vgelu+vsmx+rowops** (planned) | **0.990** | **0.980** | **0.0023 (0.036)** | **0.0006** | **0.036** | **0.192** | **0.050** | **71.0 % (92.0 %)** |
| pow2+p14+vgelu+vsmx+rowops (P at 2⁻¹⁴) | 1.000 | 0.982 | 0.0028 (0.041) | 0.0004 | 0.036 | 0.195 | 0.051 | 71.0 % (93.0 %) |

**16 clips** (`study --quick`, the first 16 evaluation clips; all the policies;
run before §2.1's resize fix, when 4 of these clips — rows 55, 191, 219, 383 — and
one calibration clip had 1–4 pixels one level off):

| policy | top-1 agree | top-5 overlap | KL (max) | logits rel | encoder rel | pooled rel | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1.000 | 0.988 | 0.0013 (0.007) | 0.027 | 0.182 | 0.040 | 75.0 % |
| pow2+p12+vgelu+vsmx+rowops | 1.000 | 0.962 | 0.0017 (0.005) | 0.035 | 0.181 | 0.046 | 75.0 % |
| … + ch_floor 6 (per-channel exponents at most 6 below the tensor's) | 1.000 | 0.962 | 0.0019 (0.006) | 0.035 | 0.180 | 0.047 | 75.0 % |
| pow2+p12 (GELU, softmax, RoPE and residual on the host) | 1.000 | 0.962 | 0.0019 (0.007) | 0.035 | 0.178 | 0.049 | 75.0 % |
| pow2+p14+vgelu+vsmx+rowops | 1.000 | 0.988 | 0.0012 (0.006) | 0.032 | 0.188 | 0.046 | 75.0 % |
| pow2+p12 with float weights (diagnostic) | 1.000 | 0.975 | 0.0016 (0.007) | 0.028 | 0.170 | 0.039 | 75.0 % |
| q88 (everything at Q8.8) | 0.062 | 0.050 | 5.18 (8.56) | 1.08 | 1.38 | 1.16 | 6.2 % |

How to read it:

- **The task result doesn't move.** Accuracy is 71.0 % for float, for bf16 and
  for both int16 variants.  Only one clip (row 656) changes its top-1 class:
  float says class 151, int16 says 155, and the label is 135, so both are wrong.
  These numbers are for one clip per video.  Meta's published 73.7 % averages 6
  clips per video, and on the board each extra clip costs a full pass.
- **The int16 KL is at bf16's level.** The planned policy's mean KL is 1.3 × bf16's.
  The paired bootstrap 95 % interval of the per-clip difference is
  [−0.0002, +0.0014], which includes zero.  For comparison, SmolVLM passed at
  2.6 × bf16 (CHAT_PLAN §22).
- **A few uncertain clips make up most of the KL.** The median clip's KL is
  0.0006, against 0.0003 for bf16.  The largest are clips where float itself is
  unsure: row 2653 (float's top class at p = 0.65) and row 2352 (p = 0.33).
- **P at 2⁻¹⁴ buys nothing measurable.** The 16-clip run favoured it (KL 0.0012
  against 0.0017).  At 100 clips the order reversed (0.0028 against 0.0023), and
  the paired interval of the difference, [−0.0003, +0.0015], includes zero.
  Meanwhile it saturates 14 times as many P·V values (1429 against 99).  The
  plan keeps 2⁻¹², the setting of every shipped library.  With 2048 keys an
  average P is 2⁻¹¹, two steps of 2⁻¹², and that is still enough.
- **The FPGA units cost nothing.** The VectorOP GELU and softmax, and the
  row-ops residual, LayerNorm, bias and RoPE, score the same as the all-host
  `pow2+p12`.  The channel floor makes no difference either.
- **Weight rounding is a small share.** Float weights move KL only from 0.0019
  to 0.0016.
- **Plain Q8.8 is broken.** The residual reaches 358 and the scores 501, both
  far outside Q8.8's ±128.
- **The encoder output's 19 % error comes from the model, not the int16
  datapath.**
  - bf16 has it too (18.9 %), and so do float weights (17.0 %).
  - In a per-token breakdown of one clip, the worst 100 of the 2048 tokens carry
    63 % of the error.  The median token is 6.6 % off, and no channel stands
    out.
  - The pooler averages it out: the pooled row is within 4.4–5.1 %, and the
    logits within 3.0–3.6 %.

Value ranges (float, max |x| over the 100 clips):

| class | max | class | max |
|---|---:|---|---:|
| patch embedding | 110.3 | P · V | 13.0 |
| residual | 357.6 | out projection | 124.0 |
| LayerNorm out | 14.3 (LN2: 20.9) | fc1 / GELU | 28.3 / 25.8 |
| q / k before bias, after bias, after RoPE | 13.6 / 13.9 / 14.3 | fc2 | 309.8 |
| scores | 500.9 | encoder out / pooled / logits | 54.4 / 4.5 / 23.6 |

Saturation in the planned policy, over the 100 clips (about 10¹¹ written
activations, not counting scores and P):
- no weight saturates;
- no accumulator wraps, and the largest is 195 of ap_fixed<32,16>'s 32 768;
- these values saturate: out projection 794, fc1 238, P · V 99, fc2 11, v 1.

### 1.4 Memory

- **DMA pool:** 671 MiB of int16 weight matrices (the patch embedding, 27
  self-attention blocks, the cross layer's k / v and MLP, the classifier), plus
  the intermediates.
  - The intermediates at 2048 tokens: the int32 residual 8 MiB, the MLP
    hidden 16 MiB, one head's scores and P 8 MiB each, q / k / v 4 MiB each.
    Estimated at 60–100 MiB after slot colouring.
  - The total, about **750 MiB** (estimated), fits `cma=1400M` alone or beside
    BERT or SmolLM2-135M.  It does not fit beside SmolLM2-360M (758 MiB) or
    SmolVLA, so in the chat server it would swap with them.
- **Host:**
  - a clip is 16 × 256 × 256 × 3 bytes (3 MiB);
  - the RoPE tables are 2048 × 128 int32 (1 MiB) in the pool;
  - decoding the video on the A53 is the only large host job.

### 1.5 Cost

`vjepa2_study.py costs` gives 935 GMAC per clip:

| part | GMAC |
|---|---:|
| encoder | 828 (of it attention 206) |
| pooler self-attention layers | 103 |
| cross layer and classifier | 4.3 |

The projection scales from SmolVLA's vision encoder on the board (107 GMAC in
1313 ms, of which the softmax took 261 ms; VLA_PLAN §4).  Its kernels run at
102 GMAC/s, which gives **9.2 s** of MACs per clip.  The softmax unit adds
432 rows of 2048 × 2048 at 1.81 ms per 1024², or 3.1 s.  So the clip takes
**9.2 s** when the softmax runs beside ConvKernel (as in the query-major pieces
of VLA_PLAN §4) and **12.3 s** when it runs after it.  Earlier projections came
in about 10 % low on the board (BERT 10.8 → 12.1 s, SmolVLA 4.2 → 4.65 s).

This is not real time.  It suits a demo that classifies a recorded clip, or the
last few seconds of camera video (64 source frames) every ~10 s.

### 1.6 What the implementation needs

(The plan as written before the implementation; §2 is what was built.)

Every operation maps onto the five kernels of the production bitstream
`b4e63457057a` as they are: no new FPGA block and no bitstream change.  The
work is in the scheduler, the frontend and the library.  The one hardware
bound the model reaches is the softmax unit's 2048-key row; raising it would
need block RAM the build lacks (§0).  The A53 decodes the video, forms the
patch rows and at most runs the pooler's one-row tail; none of the kernel
work moves to it.

1. **A frontend** `src/vjepa2.py` built on `GraphBuilder`: `src/vit.py` for the
   blocks and `src/smolvla.py` / `src/moonshine.py` for the entry layout.  One
   entry, `clip`, takes 2048 patch rows and returns 174 logits.  Each step
   below is checked against `vjepa2_q.py`:
   - **The patch rows:** the host forms the [2048, 1536] int16 rows from the
     uint8 clip (channel, frame, row, column).  The normalisation is already in
     the MatMul's weights and bias.
   - **3-D RoPE on the row-ops unit:**
     - `VitAttnPrepRoNode` (the bias COPYs, `iqkv`) gains a ROPE job per head,
       as the Llama prefill's `AttnPrepRoNode` has.
     - The q / k weight columns are permuted per head into rotate-half order.
       q · k is unchanged because both sides get the same permutation.
     - The cos / sin tables are 2048 rows × 64 dims.  Dims 60–63 get cos 1 and
       sin 0.
     - The unit's ROPE reads one cos / sin row per token
       (ROWOPS_RTL_KERNEL op 6).  The three position axes and the per-element
       frequencies are therefore only table content; no RTL change is needed.
   - **Attention at 2048 tokens:**
     - The softmax unit's row mode runs exactly at its 2048-key cap.
     - The scores are 2048 × 2048 per head.
     - The queries go to ConvKernel in two row pieces of 1024 (`max_out_ch`
       1280; VLA_PLAN §4's pipelined pieces).
     - No board run has used 2048 tokens yet (SmolVLM and SmolVLA use 1024).
       The first board gate must cover it.
   - **GELU, erf form,** on the activation unit (`VitGelu approximate="none"`,
     as Moonshine's encoder).
   - **The pooler:**
     - Its 3 self-attention layers are encoder blocks without RoPE.
     - The cross layer is one row: k / v MatMuls over the 2048 tokens, a
       constant query per head (an initializer), 16 one-row score / softmax /
       P · V passes, residual = query + attention, then LN2 and the MLP on one
       row, then the classifier.
     - Its 4.3 GMAC are the k / v projections over the 2048 tokens: kernel
       MatMuls, as in the encoder.  The one-row rest (the 16 score / softmax /
       P · V passes, LN2, the MLP and the classifier, about 13 MMAC) can run
       as host ops on the A53, as SmolVLA's state row does, or as one-row
       kernel calls.
2. **A `formats` step** in `vjepa2_study.py`.  It writes the calibrated
   exponents (today they sit in `results.json`, `vjepa2_q.formats_json`) to a
   tracked `demo/video_classification/vjepa2_vitl_ssv2_<policy>_formats.json`
   for the frontend's `ExponentTable`.
3. **A library and a demo:**
   - `generate_project.py` builds `libvjepa2.so`, with a C API (`vjepa2_api.c`)
     that does the frame sampling, the two-pass resize, the crop and the patch
     rows.  `vjepa2_q.make_clip` is the specification.
   - Video decoding (ffmpeg on the A53) or a camera ring buffer.
   - A ctypes wrapper.
4. **Gates,** as SmolVLA's:
   - the scheduler's simulation equals `vjepa2_q`'s emulation bit for bit;
   - the host emulation of the generated C;
   - the board on the pinned clips (`vla_board.py`'s pattern).
5. **Performance model:** a `perf_calibrate.py` top-up for the 2048-row softmax
   and the new ConvKernel geometries (`--plan` only).

Effort is comparable to SmolVLA's frontend and library (VLA_PLAN §2–§3): the
blocks exist, and the new parts are the RoPE job in the vision prep, the
2048-token attention and the one-row pooler.

### 1.7 Commands and runtimes

```bash
PY=.venv-export/bin/python
PIP_CONFIG_FILE=/dev/null $PY -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r demo/video_classification/scripts/requirements-study.txt
$PY demo/video_classification/scripts/vjepa2_study.py fetch      # checkpoint, parquet, 216 clips: 3.5 min
$PY demo/video_classification/scripts/vjepa2_study.py validate   # 3 clips vs transformers: 6.8 min
$PY demo/video_classification/scripts/vjepa2_study.py study --quick --jobs 8 \
    --out demo/video_classification/assets/study/quick \
    --policies bf16,pow2+p12+vgelu+vsmx+rowops,pow2+p12+vgelu+vsmx+rowops+cf6,pow2+p12,pow2+p14+vgelu+vsmx+rowops,pow2+p12+fw,q88
                                                                 # 16 clips, 7 policies: 2.4 h
$PY demo/video_classification/scripts/vjepa2_study.py study -n 100 --jobs 8 \
    --out demo/video_classification/assets/study/full \
    --policies bf16,pow2+p14+vgelu+vsmx+rowops,pow2+p12+vgelu+vsmx+rowops
                                                                 # 100 clips, 3 policies: 6.3–7.6 h
$PY demo/video_classification/scripts/vjepa2_study.py costs
$PY demo/video_classification/scripts/vjepa2_study.py formats    # the planned policy's exponents: 14 min
```

- `fetch --record` once chose the samples and wrote their hashes; a plain
  `fetch` reproduces them.
- Runtimes on the dev PC, with 8 workers of one BLAS thread each:
  - calibration (16 clips): 765–814 s;
  - per clip in a worker: about 270–360 s for the float64 reference, 370–530 s
    for bf16 and 550–630 s for an int16 policy (the upper ends: the rerun of
    §2.1, beside the test suite and the host emulation).
- Deterministic: fixed data, no sampling.  The runs' `results.json` (every
  logit, the formats) are in `assets/study/{quick,full}/`; the 100-clip run on
  the clips before §2.1's resize fix in `assets/study/full_blas_resize/`.

## 2. Implementation (2026-11-09)

**Result:** the `clip` entry, its ops and `libvjepa2.so` equal the
specification bit for bit — in the scheduler's simulation of the real model, in
the library on the host emulation, and on the board (§3).  The FPGA is the
production bitstream `b4e63457057a` as it is: every operation maps onto the
five kernels (§1.6).

### 2.1 Two fixes to the specification

Bit-exactness needs a portable specification.  Two of its float operations went
through BLAS, whose summation order (and use of FMA) depends on the library
build:

- **The cross layer's query projection** (a constant, query · Wqᵀ + bq) is now,
  per output, a correctly rounded sum of the exact products (`math.fsum`), in
  `vjepa2_q.cross_query` and `src/vjepa2.py cross_query` alike.  Its int16
  values did not change (the nearest rounding tie is 5 · 10⁻⁴ LSB away).
- **The resize** (`vjepa2_q.resample_taps` / `_resample`) sums its taps left to
  right instead of a matrix product, the weights normalised by a left-to-right
  sum: the operations `vjepa2_api.c` repeats.  This one mattered.  Where the two
  filter weights are 0.5 ± 10⁻¹⁶, a pixel pair with an odd sum lands next to a
  .5 tie (155.49999999999997 against 155.5), and the order of the sum decides
  the rounding.  41 of the 216 pinned clips changed, each in 1–4 of its 3.1 M
  pixels, by one level.  The clips were re-pinned (`fetch --record`), the
  formats recalibrated (one exponent moved: channel 66 of layer 8's out
  projection, 2⁻¹¹ → 2⁻¹²) and the 100-clip study rerun; §1.3's 100-clip table
  is the rerun's, the 16-clip table the earlier clips'.

### 2.2 The scheduler: the frontend and its ops

`inference-scheduler/src/vjepa2.py` (`VJepa2Frontend`) writes the `clip` entry
straight from the checkpoint and the formats of `vjepa2_study.py formats`
(`demo/video_classification/vjepa2_vitl_ssv2_pow2+p12+vgelu+vsmx+rowops_formats.json`).
The layers reuse the vision encoder's nodes (`src/vit.py`'s row pieces and
pipelines): 3400 nodes.

| part | nodes |
|---|---|
| input | the patch rows in two pieces of 1024, one graph input each (`clip.patches.p0` / `.p1`): a Slice of a graph input would be a host copy, and a 2048-row MatMul exceeds ConvKernel's out_ch bound |
| patch embedding | a MatMul per piece (the pixel normalisation folded into W and b), then `VjResCopy` with the bias into the int32 residual at 2⁻¹⁵ |
| 24 encoder layers | LayerNorm, q / k / v MatMuls (each head's q / k columns in rotate-half order, the biases too), `VitAttnPrep rope = 1`, per head and query part q·Kᵀ / softmax / P·V, the merge, the out projection and the MLP — the linears and row ops in two pipelined row pieces |
| pooler | the final LayerNorm, `VjResCopy` into a new residual, 3 self-attention layers (the same block, `rope = 0`) |
| cross layer | LayerNorm, k / v MatMuls (pieces), `VjCrossAttn`, LN2 and the MLP on one row |
| classifier | `VjResCopy` out of the residual (int16 at 2⁻⁸), a MatMul, `VjLogits` (+ the bias, float32) |

What is new:

- **3-D RoPE on the row-ops unit** (`VitAttnPrepRoNode`, attribute `rope`): q0
  + bq rounded at the per-head bias exponent into a staging state, a ROPE job
  into the q staging rows at f_q, the TRANSPOSE into the image; k likewise, a
  ROPE job, a COPY into the K cache with the heads as blocks; then v as before.
  The ROPE jobs read row t of a per-token, per-element cos / sin table
  (`rope_ints_tensor`): int24 values rounded once from the double angles.  The
  float32 tables of the Llama path could not carry them exactly.
- **Query parts** (`LlmAttnScores` attribute `rows`): with the scores
  query-major, a static q.K^T reads a range of the "K cache" rows (the queries).
  So each head runs as two parts of 1024 queries — q·Kᵀ, the softmax unit's row
  mode over 2048 keys (`VitAttnSoftmax` query-major pieces, s [1024][2048]), and
  P·V — each within ConvKernel's out_ch bound of 1280.
- **`src/vjepa2_nodes.py`:**
  - `VjResCopy`: one COPY with a per-column bias into or out of the int32
    residual (the patch embedding, the pooler's input, the pooled row).
  - `VjCrossAttn`: a host op.  The pooler's one query row over the 2048 keys:
    the k / v bias adds (`rowops_spec.bias_round`), q·kᵀ and P·v as ConvKernel
    sums (ap_fixed<32,16> wrap, floor, saturate), the softmax unit's integer
    specification, + the query into the int32 residual.
  - `VjLogits`: + the classifier's bias, float32.

The project (`generate_vjepa2_project.py`): a 742 MiB DMA pool (weights 678,
states 20, intermediates 44), weight files 711 MB, 14 min and 6.5 GB to
generate.

### 2.3 The library

`demo/video_classification/scripts/generate_vjepa2_project.py` builds
`libvjepa2.so` and `vjepa2_bench` from `demo/video_classification/src/`
(`vjepa2_api.{c,h}`, `vjepa2_bench.c`; the ctypes wrapper
`demo/video_classification/vjepa2_lib.py`):

- `vjepa2_classify(frames, n, h, w, logits)`: a decoded video.  It samples the
  clip's 16 frames, resizes and crops each (`vjepa2_frame`), writes the patch
  rows and runs the entry.
- `vjepa2_classify_clip`, `vjepa2_frame`, `vjepa2_sample_frames`,
  `vjepa2_label(c)` (the 174 class names) give the steps one at a time.
- The resize computes only the cropped outputs.  Every operation is the
  specification's, with FMA contraction off.

### 2.4 Tests and gates

- **`inference-scheduler/test/test_vjepa2.py`** (6 tests), on a small random
  V-JEPA 2 (2 layers 64 wide, 2 heads of 32, a one-layer pooler, 4 frames of
  128 × 128 = 128 tokens in two row pieces and two query parts) with formats
  the specification calibrates:
  - the simulation equals the specification bit for bit (the encoder output and
    the logits of two clips);
  - the generated C equals the simulation, coherent and incoherent;
  - the library on the host emulation equals `VJEPA2.forward(make_clip(…))` on
    three videos: both sampler branches, upscaled and antialiased downscaled
    frames, the repetitions and the reopen identical; its sampler and resize
    through ctypes equal `vjepa2_q`'s;
  - the RoPE tables and the node census.
- **`vjepa2_sched_check.py`**: the real model's simulation against the study's
  logits — clips 0 and 27 bit-exact (7 min, 6.5 GB).
- **`vjepa2_host_emu.py`**: the real project on the host emulation (software
  kernels): video 0, decoded whole, sampled, resized and cropped by the library,
  bit-exact with the specification run on the same video (61 min, `--spec`).

## 3. On the board (2026-11-09, bitstream `b4e63457057a`)

`vjepa2_board.py` uploads the project and the weights (711 MB, 270 s), builds
`vjepa2_bench` and `libvjepa2.so` on the board (190 s with the profiled build;
cc1 at most 317 MB), and runs 8 evaluation videos twice each, then closes and
reopens.  The videos are decoded on the host and shipped whole, at most 65
frames (the sampler reads frames < 64), 110 MiB.  The chat server was stopped.

- **Bit-exact.**  All 8 videos' logits equal the specification's (the 100-clip
  study's `pow2+p12+vgelu+vsmx+rowops` logits of the same videos;
  `vjepa2_board.py --check-only`), video 0's the host emulation's as well, and
  every repetition and the reopen are identical.
- **Time per video** (medians; every video within 7 ms of them):

  | step | time |
  |---|---:|
  | frame sampling, resize and crop (16 frames, A53) | 0.26 s |
  | the network | 13.53 s |
  | **a video** | **13.79 s** |

  `vjepa2_open` takes 6.8 s right after the upload.  §1.5 projected 9.2 s with
  the softmax beside ConvKernel and 12.3 s after it; earlier projections came in
  about 10 % low, which puts 13.5 s at the serial end.
- **Where the time goes** (the profiled build, per kind; the brackets overlap):

  | calls | ms |
  |---|---:|
  | the linears (MatMuls on ConvKernel) | 5923 |
  | P·V on ConvKernel | 3419 |
  | the softmax, row mode | 3396 |
  | q·Kᵀ on ConvKernel | 3275 |
  | LayerNorms / GELU / residual adds | 1500 / 1175 / 764 |
  | the attention prep (bias adds, RoPE, images) | 510 |
  | `VjCrossAttn` (host) | 317 |

  The attention is about two thirds of the clip.  P·V costs as much as q·Kᵀ: its
  calls read the V image with 512 input channels for 64 output pixels.
- **Memory:** 742 MiB of CMA.  It does not fit beside the chat server's models
  (the server must be stopped), only alone or beside BERT / SmolLM2-135M (§1.4).

### What next

- **The performance model.**  The bitstream's model predicts 8.1 s, but 947 of
  the 3400 nodes are unpriced: all 864 softmax calls (it has never measured the
  2048-key rows), the 27 RoPE preps, 54 of the MatMuls and the two host ops.  A `perf_calibrate.py` top-up would let `--plan`
  price them and the A53's early wake predict them.
- **`VjCrossAttn` on four cores** (heads in parallel: about 317 → 80 ms), or its
  k / v bias adds on the unit.
- **The attention's balance.**  The softmax and P·V each take as long as q·Kᵀ;
  a third query part, or P·V with a wider V interleave, may hide more of the
  softmax behind ConvKernel.

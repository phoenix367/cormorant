# Stereo depth (disparity) on the KV260

**Status (2026-10-08):** study done (the `model-study` skill, host only); nothing implemented.
- **Verdict: GO with power-of-two exponents and per-channel weights.**
  - The model is LightStereo-S with OpenStereo's StereoAnything weights.
  - It is the only candidate with 2D layers throughout.
  - The emulated FPGA datapath reaches float's quality on 42 Middlebury and
    ETH3D pairs with ground truth: EPE 0.628 px against float's 0.633, and
    bad-2 4.40 % against 4.44 %.
  - Its distance from float equals bfloat16's.
  - Plain Q8.8 is a NO-GO (EPE 1.60 px, bad-2 13.6 %).
- **Speed (projected, ±30 %).**  About 0.5 s per pair at 640 × 480 (2 FPS),
  0.3 s at 512 × 384, and 0.14 s at 320 × 256 (7 FPS).  The depthwise and
  1 × 1 layers of its MobileNetV2 blocks run at 5 and 35 GMAC/s on
  ConvKernel, against 105 for a standard 3 × 3 conv.
- **What it needs.**
  - A frontend (route C, like Piper's) that writes the exponents.
  - Four host ops: the correlation volume, the instance norm, the
    pixel shuffle of the polyphase deconvs, and the context upsampling.
  - A per-channel rescale node.
  - Stripe convolutions longer than 7 taps split into pieces.
  - No bitstream change.
- **License caveat.**  OpenStereo's code is for academic use only, and its
  weights carry no license of their own.

The user asked to "find a model that can convert a stereo pair into a depth
map" and to see whether it can be supported.  A rectified pair gives a
disparity map d(x, y), and depth is f · B / d with the camera's focal length f
and baseline B.  The study compares disparity, the networks' output, against
the datasets' ground truth.

## 1. Candidates

Stereo networks build a cost volume (left features against right features
shifted by each candidate disparity), aggregate it, and regress a disparity
per pixel.  The fit to this design depends on the aggregation.  ConvKernel is
2D-only (kernels ≤ 7 × 7), and recurrent refinement with `grid_sample`
lookups has no kernel.  Screened in PINTO's model zoo (ONNX) and the
literature:

| model | aggregation / refinement | fit |
|---|---|---|
| RealtimeStereo (Chang, ACCV 2020; PINTO 165, 0.21 M parameters) | 15 `Conv3d` 3 × 3 × 3; `grid_sample` exported as 144 `Gather`s | no: 3D convs, warping |
| CoEx (ICRA 2022; PINTO 135) | 16 `Conv3d` + 3 `ConvTranspose3d` | no: 3D |
| MobileStereoNet-2D (WACV 2022) | a `Conv3d` stack run once per disparity (48 ×) before 2D hourglasses | no: 3D, run 48 times |
| Fast-ACVNet, CGI-Stereo, StereoNet, NVIDIA stereoDNN, PSMNet / FADNet (the Vitis AI zoo) | 3D cost aggregation (FADNet: a correlation layer, 154 G ops) | no: 3D |
| CREStereo, RAFT-Stereo, IGEV | recurrent GRU updates with correlation lookups (`grid_sample`) | no: iterative, warping |
| HITNet (CVPR 2021; PINTO 142) | 2D; tile hypotheses, per-iteration warping by gathers, argmin | host-bound warping (not studied further) |
| **LightStereo-S** (2024; OpenStereo) | **a correlation volume aggregated by 2D MobileNetV2 blocks** ("channel boost", disparity as channels) | **studied** |

## 2. Model

**LightStereo-S** (Guo et al., "LightStereo: Channel Boost Is All You Need for
Efficient 2D Cost Aggregation", arXiv 2406.19833).
- **Code:** [OpenStereo](https://github.com/XiandaGuo/OpenStereo) branch v2
  @ `23d71c92e33a`.
- **Weights:** Hugging Face `XiandaGuo/OpenStereo` @ `cac6f81baeb0`.
- **Checkpoint:** `StereoAnything-LightStereo_S.pt`, SHA-256 `fe62602f…`.  It
  is LightStereo-S trained on the Stereo Anything data mix.
- **Size:** 3.44 M parameters.
- **License:** OpenStereo says "This code is only used for academic purposes;
  people cannot use this code for anything that might be considered
  commercial use".  The weights have no model card or license.

| part | layers | at 640 × 480 |
|---|---|---:|
| backbone (both images) | timm `mobilenetv2_100` blocks 0–5 (ReLU6) | 3.14 GMAC |
| FPN (both images) | `ConvTranspose` 4 × 4 s2 + concat + 3 × 3 conv (LeakyReLU 0.2) × 3; a replicate-padded 3 × 3 conv + InstanceNorm → 24 channels at 1/4 | 1.56 GMAC |
| correlation volume | mean over 24 channels of left · right shifted by d = 0 … 47 (192 / 4) | 18.9 MMAC |
| aggregation | MobileNetV2 blocks (expansion 4) on 48 / 96 / 192 channels at 1/4, 1/8, 1/16; three "stripe attention" modules from the left features (depthwise 1 × 7, 1 × 11, 1 × 21 and transposed, 1 × 1 convs, a product with the cost); two `ConvTranspose` 3 × 3 s2 back to 1/4 | 3.86 GMAC |
| regression | softmax over the 48 disparities, Σ P · d | 0.9 MMAC |
| refinement | a 1/2-resolution stem of the left image, two convs with InstanceNorm, an FPN step, `ConvTranspose` 4 × 4 s2 → 9 weights per full-resolution pixel | 1.06 GMAC |
| context upsampling | softmax over the 9 weights, Σ w_k · 4 · d(3 × 3 neighbourhood at 1/4) | 2.8 MMAC |
| **total** | | **9.64 GMAC** (512 × 384: 6.17; 320 × 256: 2.57) |

The checkpoint choice, in float on the 42 pairs:

| checkpoint | EPE (px) | bad-1 | bad-2 |
|---|---:|---:|---:|
| **StereoAnything-LightStereo_S** | **0.633** | **10.70 %** | **4.44 %** |
| LightStereo-S-SceneFlow-General | 5.79 | 29.5 % | 17.4 % |
| LightStereo-S-KITTI | 7.39 | 47.0 % | 29.9 % |

The SceneFlow and KITTI checkpoints break down on ETH3D's grayscale indoor
and outdoor scenes (EPE 8.2 / 10.3 px there).

## 3. Results

### 3.1 Operators and kernel bounds

LightStereo has no 3D convolution and no warping.  Each part maps onto a
kernel or a host op of this repo; the new ones are in **bold**:

| part | where | how |
|---|---|---|
| convs, depthwise convs, 1 × 1 convs, BatchNorm | ConvKernel | BN folded into the weights |
| ReLU6, LeakyReLU, ReLU, residual adds, the attention product | VectorOPKernel | RELU6 (clamps at raw 6.0 in Q8.8: the output stays at 2^-8), the activation unit, ADD, MUL |
| `ConvTranspose` 4 × 4 s2 / 3 × 3 s2 | ConvKernel | **polyphase**: the four output phases as one conv (3 × 3 → 4C channels, or 2 × 2), then a **pixel shuffle** (a host copy; Piper's 1-D deconvs already run polyphase) |
| depthwise 1 × 11, 1 × 21 (and transposed) | ConvKernel | **pieces of ≤ 7 taps** (kernels ≤ 7 × 7) on shifted inputs, summed by VectorOP adds |
| InstanceNorm (4 ×) | host | **instance norm** (the LayerNormalization host op over H·W does the same arithmetic) |
| replicate pad | host | a copy |
| correlation volume | host | **a correlation host op**: 18.9 MMAC at 640 × 480, row-major features; the FPGA alternative is a per-row q·Kᵀ like the attention scores (band of 48 out of W/4) |
| softmax over the disparities | VectorOPKernel | the softmax unit's column mode (`OP_SOFTMAX_T`: keys-major in, transposed out, P at 2^-15) |
| Σ P · d | MatmulKernel / ConvKernel | a GEMV (48 → 1) |
| 9-way softmax + upsampling | VectorOPKernel + host | the softmax unit (9 keys per pixel), then **the weighted 3 × 3 sum** on the host (2.8 MMAC) |

Nothing exceeds a kernel bound.  The widest conv has 384 → 96 channels.  The
largest tensors are the refinement's 32 × 240 × 320 at 1/2 resolution and
the 9 upsampling weights per pixel at full resolution (9 × 480 × 640).

### 3.2 Memory

The latency proxy (§3.4) at 640 × 480 needs a DMA pool of 86 MiB: weights
12 MiB (the polyphase deconvs included) and intermediates 74 MiB.  That is
9 % of `cma=1000M`, so it fits beside the chat server's models.

### 3.3 Numerics

**Method.**
- **Reference.**  `stereo_study.py` writes LightStereo-S as explicit tensor
  operations.  In float64 it equals OpenStereo's module to 2.7e-6 px
  (`validate`).  OpenStereo's own float32 differs from its float64 by
  1.9e-4 px.
- **Emulation.**  The same code under a numeric policy emulates the planned
  partition of §3.1:
  - Every ConvKernel call takes int16 operands, sums exactly and writes
    floor(acc / 2^8), saturated.
  - VectorOP ADD is exact and saturating; MUL is truncated.
  - LeakyReLU rounds to nearest, ties to even.
  - The softmax unit writes P at 2^-15.
  - Host ops compute in double and round half to even.
  - The 1 × 21 stripes run as floored 7-tap pieces.
- **Data.**  Evaluated at the pairs' native resolution, padded to a multiple
  of 32 as OpenStereo's evaluation does:
  - Middlebury MiddEval3 training at quarter resolution: 15 pairs, 347 × 277
    to 741 × 497.
  - ETH3D low-res two-view training: 27 pairs, 707 × 425 to 942 × 491.
  - Pixels: the non-occluded ones with ground truth.
- **Metrics.**
  - EPE: mean |d − d_gt| in px.
  - bad-1 / bad-2: the percentage of pixels off by more than 1 / 2 px.
  - Δfloat: the mean |d − d_float|.
- **Calibration.**  Exponents come from the 15 MiddEval3 *test* pairs.  These
  have no ground truth and are disjoint from the evaluation.
  - Each site gets the largest |value| seen there, with one bit of headroom.
  - Conv sites also get the largest value per output channel.

**Policies** (all 42 pairs):

| policy | EPE (px) | bad-1 | bad-2 | Δfloat (px) | Δfloat > 1 px |
|---|---:|---:|---:|---:|---:|
| float | 0.633 | 10.70 % | 4.44 % | — | — |
| `q88`: every tensor and weight at 2^-8 | 1.599 | 21.44 % | 13.62 % | 1.708 | 20.4 % |
| `pow2`: per-tensor exponents | 0.713 | 12.62 % | 5.29 % | 0.377 | 5.7 % |
| **`pow2+pc`: + per-output-channel weight exponents** | **0.628** | **10.58 %** | **4.40 %** | **0.118** | **1.4 %** |
| `pow2+pc5`: per-channel weights only where per-tensor rounding loses > 5 % (17 conv calls) | 0.627 | 10.80 % | 4.36 % | 0.254 | 3.4 % |
| bfloat16 everywhere (the yardstick) | 0.636 | 10.81 % | 4.47 % | 0.116 | 1.2 % |

By set, `pow2+pc` against float:
- Middlebury: EPE 0.957 against 0.941, bad-2 7.46 % against 7.36 %.
- ETH3D: EPE 0.446 against 0.462, bad-2 2.70 % against 2.82 %.

At half resolution (the pairs resized to 0.5, about 360 × 250 and 470 × 245,
the 320 × 256 class): float EPE 0.944, `pow2+pc` 0.950; bad-2 8.34 % and
8.44 %.  The datapath stays at float's level; the halved resolution itself
costs 0.3 px.

**What breaks and what fixes it** (an ablation on every third pair, 14
pairs, float EPE 0.509):
- **`q88` loses all of it in the aggregation.**
  - Rounding only the aggregation's MobileNetV2 blocks gives EPE 1.459.
  - Every other stage rounded alone stays within 0.2 px of float: input
    0.016, correlation 0.002, regression 0.023, refinement 0.003, backbone
    0.060, FPN 0.151, attention 0.193.
  - Float weights with Q8.8 activations give 0.557.
- **The causes are weights and range.**
  - `cost_agg.conv5`'s weights are all below 1/512: rounding loses 99 % of
    them.
  - The last attention product reaches 2929, far outside Q8.8's ±128.
- **`pow2` gives each tensor its range.**  The calibrated exponents run
  from 2^-2 (that product) to 2^-14.  Most sites are at 2^-9 … 2^-13.  The
  rest of its error is weights again (`pow2+fw`: Δfloat 0.035):
  - A conv's weight exponent is f_y + 8 − f_x.
  - So a 1 × 1 conv with small inputs and large outputs (the attention
    `conv3`s: f_x 12, f_y 7 → f_w 3) rounds most of its weights away.
  - The same holds for an expansion conv whose ReLU6 output is pinned at
    2^-8.
- **Per-output-channel weight exponents fix it.**
  - Channel c gets the finest f_w,c that fits its weights and its
    calibrated output range.
  - The kernel writes the channel at f_w,c + f_x − 8.
  - One rescale per channel floors it to the tensor's exponent.  A floor of
    a floor is one floor, so this is exact.
  - The worst weight error drops from 41 % to 4.6 %.
- **Two things did not help.**  Freeing ReLU6 outputs from 2^-8 and doing
  the attention product on the host at a free exponent gain nothing on top
  of `pc` (0.506 → 0.504 / 0.506 on the subset).
- **Saturation stays negligible.**  At most 3e-9 of the values in
  `pow2+pc`, and no weight.
- **Exponent mismatches** at adds and concats cost 41 rescales in the
  emulation (the finer operand floored).  A frontend's exponent solver gives
  the producing convs the coarser exponent directly, with the same result.

### 3.4 Speed

**Method.**
- `stereo_proxy.py` writes LightStereo-S's layers at a resolution as an ONNX
  graph of ops the scheduler takes today, with random weights.  The
  stand-ins are priced as the implementation would run them:
  - polyphase convs and a pixel-shuffle transpose for the deconvs;
  - 7-tap pieces for the stripes;
  - LayerNormalization over H·W for the instance norms.
- `--price` replays the generated event stream with bitstream
  `8599aa7a5f12`'s performance model, as `onnx_study.py` does.
  - Calls outside a family's fitted range are extrapolated by the family
    model, reported apart.
  - The two proxy-only transposes in front of the softmaxes are dropped.
    The real column-mode softmax needs neither.
- The correlation volume, the upsampling, the 9-way softmax and the
  per-channel rescales are estimated by hand.

| resolution | GMAC | proxy (kernels + host) | of which extrapolated | + correlation, upsampling, 9-way softmax, concats | + rescales `pc5` / `pc` | **per pair** |
|---|---:|---:|---:|---:|---:|---:|
| 640 × 480 | 9.64 | 455 ms (ConvKernel 324, VectorOP 81, host 57) | 134 ms | ~25 ms | ~15 / ~50 ms | **~0.50 s (2 FPS)** |
| 512 × 384 | 6.17 | 286 ms | 69 ms | ~15 ms | ~10 / ~30 ms | **~0.31 s (3 FPS)** |
| 320 × 256 | 2.57 | 124 ms | 13 ms | ~6 ms | ~3 / ~12 ms | **~0.14 s (7 FPS)** |

The family models carry their held-out error (ConvKernel families up to
±36 % p90; depthwise calls this long were never measured), so the totals
are ±30 %.

**Where the time goes** (640 × 480, ConvKernel by kernel shape):

| layer | calls | ms | GMAC | GMAC/s |
|---|---:|---:|---:|---:|
| 1 × 1 convs (MobileNetV2 expand / project) | 86 | 171.8 | 5.96 | 35 |
| depthwise 3 × 3 | 41 | 84.3 | 0.39 | 5 |
| standard 3 × 3 (FPN, refinement, polyphase deconvs) | 20 | 32.6 | 3.43 | 105 |
| depthwise stripes (1 × 7, 7 × 1, 1 × 4, 4 × 1) | 36 | 25.6 | 0.14 | 5 |
| polyphase 2 × 2 | 5 | 9.6 | 1.00 | 104 |

The network is MobileNet-shaped, and those layers use the MAC grid poorly.
ResNet-18 runs about 90 GMAC/s on the same kernel.  The levers, none of
which the verdict needs:
- A per-channel output shift in ConvKernel would remove the rescale calls.
- A faster depthwise path would cut the 110 ms of depthwise and stripe
  convs.
- Fusing ReLU6 into ConvKernel would remove 82 VectorOP calls.
- The two images' backbones are independent: no lane overlap is lost, but
  nothing overlaps them either.

### 3.5 What an implementation needs

1. **A frontend** (`src/stereo.py`, on `src/piper.py`'s pattern):
   - The checkpoint becomes the graph: BN folded, deconvs polyphase, long
     stripes split.
   - The `axi.numeric` exponents come from a calibration step (this
     study's `calibrate`).
   - An exponent solver: ReLU6 outputs at 2^-8, adds and concats at one
     exponent, and the attention MUL's f_a + f_b − 8.
   - Per-channel conv weights with their rescale node: a depthwise 1 × 1
     ConvKernel call with weights 2^(8 − k_c) (an exact floor), or a
     VectorOP MUL.
   - The emulation in `stereo_study.py` is the specification it must match
     bit for bit.
2. **Host ops** (`src/host_nodes.py` or a `stereo_nodes.py`): correlation
   volume, instance norm (or the LayerNorm op on a reshaped view), pixel
   shuffle, context upsampling (from the softmax unit's 9 weights).  The
   correlation runs over row-major [H][W][C] features, threaded, under
   20 ms at 640 × 480.
3. **The softmax unit's column mode** on the 48-disparity volume (as the
   attention paths use it), and a GEMV for Σ P · d.
4. **A demo** (`demo/stereo_depth`):
   - Middlebury / ETH3D pairs, or the user's own rectified pairs.
   - Colourised disparity, EPE against the ground truth, the board's
     disparity bit-exact against the simulation.
   - Optionally depth from the calibration (f · B / d).
   - A live stereo camera would need rectification on the host; the
     RealSense of `demo/camera` gives IR pairs.

Estimated size: the frontend and its tests about 1 500 lines, the host ops
about 400 lines of C and Python, the demo on `demo/object_detection`'s
pattern.

## 4. Commands and runtimes

Assets are untracked: `demo/stereo_depth/assets` → `/mnt/data/cormorant_repro/stereo_study`
(about 500 MB of checkpoints, datasets and OpenStereo).

```bash
PY=.venv-export/bin/python      # torch 2.14 CPU + timm 1.0.30 (+ the CPU torchvision 0.29.1)
$PY demo/stereo_depth/scripts/stereo_study.py fetch        # checkpoints, MiddEval3-Q, ETH3D, OpenStereo (SHA-256 checked)
$PY demo/stereo_depth/scripts/stereo_study.py validate     # functional model vs OpenStereo: 2.7e-6 px (5 s)
$PY demo/stereo_depth/scripts/stereo_study.py costs        # MACs per stage (640 x 480)
$PY demo/stereo_depth/scripts/stereo_study.py calibrate    # exponents from MiddEval3 testQ (2 min)
$PY demo/stereo_depth/scripts/stereo_study.py study --tag final \
    --policies float,q88,pow2,pow2+pc,pow2+pc5,bf16        # 42 pairs x 6 policies: 17 min (8 threads)
$PY demo/stereo_depth/scripts/stereo_study.py study --stride 3 --tag ablate \
    --policies float,q88,q88+fw,q88+oaggregation,...       # the per-stage ablation (o<stage>, x<stage>, fw)
$PY demo/stereo_depth/scripts/stereo_study.py study --scale 0.5 --tag half --policies float,pow2+pc
inference-scheduler/.venv/bin/python demo/stereo_depth/scripts/stereo_proxy.py --height 480 --width 640 --price   # 10 s
```

Results: `demo/stereo_depth/assets/study/` (`study_anything-s_*.json`, the
logs, `formats_anything-s.json`, `proxy_*.onnx` / `*.price.json`).  Two
torch processes with 8 threads each on the 8-core host slow each other
down by an order of magnitude: run one at a time.

# VLA_PLAN — SmolVLA (vision-language-action) on the KV260

Goal: SmolVLA, Hugging Face's 450 M-parameter vision-language-action policy
(`lerobot/smolvla_base`), on the FPGA: camera frames, a task in words and the
robot's joint state in, a chunk of 50 future actions out.  The study (the
`model-study` skill, route C) is host only; this document is its write-up and
the implementation.

Scripts: `demo/vla/scripts/` — `vla_study.py` (select, fetch, validate,
study, ablate, formats, costs), `smolvla_q.py` (the numpy float64 reference
and the bit-level emulation of the partition: the specification the frontend
matches), `vla_study_inputs.json` (the pinned checkpoint, SmolVLM2's config
and tokenizer, the two SO100 datasets and the samples' frame hashes),
`requirements-study.txt` (the study's venv: lerobot 0.6.1, CPU torch).
Assets and results: `demo/vla/assets/` (not in git; a link to /mnt/data on the
maintainer's machine).

## 1. Study (2026-11-01)

**Verdict: GO with policy `pow2+p12+vgelu+vsmx+rowops`** — SmolVLM's recipe
(per-channel power-of-two exponents, P at 2⁻¹², the VectorOP GELU and softmax,
the row-ops unit's int32 residuals) keeps the action chunks within 4.0 % of
float with the same noise.  That is 2.9 × lerobot's own default precision
(bf16, 1.4 %) and 6 % of the spread the policy's own noise gives (66 %), and
the error against the recorded actions is unchanged.  4.2 s per chunk was
projected for two cameras; the board measures 4.65 s (§3).  656 MiB of int16
weights.

### Model

`lerobot/smolvla_base` @ `5e8d12a6` (Apache-2.0), 450 M parameters (bf16, the
cross-attention k / v projections float32), built on
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `7b375e1b` (Apache-2.0: the
config and the tokenizer; the checkpoint holds the weights).

| part | shape | parameters |
|---|---|---|
| vision encoder | SmolVLM2's SigLIP: 12 layers, 768 wide, 12 heads of 64, FFN 3072, GELU tanh, LayerNorm; 512 × 512 in 16 × 16 patches = 1024 tokens | 86.4 M |
| connector | pixel shuffle × 4 → 64 tokens of 12 288, one linear → 960, × √960 | 11.8 M |
| VLM | SmolVLM2's text model cut to its first 16 of 32 layers: 960 wide, 15 / 5 heads of 64, SwiGLU 2560, RMSNorm; RoPE rotate-half with θ 10 000 (lerobot's `apply_rope`, not the checkpoint's 100 000); only its K / V caches are used | 157.3 M (+ 47.3 M embedding, + 47.3 M unused LM head) |
| action expert | 16 layers, 720 wide (q 720 → 960, 15 / 5 heads of 64), SwiGLU 2048; even layers self-attention over [the VLM layer's K / V; their own], odd layers cross-attention over the VLM layer's K / V through 320 × 320 linears | 98.3 M |
| suffix / output | action_in_proj 32 → 720, time MLP 1440 → 720 → 720 (sinusoidal time, SiLU), action_out_proj 720 → 32; state_proj 32 → 960 | 1.6 M |

The inference (`SmolVLAPolicy.predict_action_chunk`):

- **Images:** each camera (the config lists three; the SO100 datasets have
  two) is resized with padding to 512 × 512 (bilinear, the aspect ratio kept,
  black on the left and the top), scaled to [−1, 1], encoded to 64 tokens.
- **Prefix:** [image tokens per camera, the task's tokens (task + "\n", at
  most 48, the padding masked), one state token].  Image and language tokens
  attend each other in both directions; the state token attends everything;
  nothing attends the state token.  The VLM runs once over the prefix and
  keeps every layer's K / V.
- **Flow matching:** x = N(0, 1) noise [50][32]; ten Euler steps
  x ← x − 0.1 · v(x, t), t = 1, 0.9, …, 0.1.  Each v is one pass of the expert
  over 50 action tokens (causal among themselves, positions after the
  prefix) that reads the prefix caches.  The actions are normalised by the
  dataset's mean / std; the robot's are the first 6 of 32.

### Method

- **Reference:** `smolvla_q.py` in numpy float64, validated against lerobot's
  `SmolVLAPolicy` in torch float32 (`vla_study.py validate`, 2 samples, the
  same noise): token ids equal, resized pixels within 2.7e-7, every prefix
  cache within 2.2e-6 relative, the first velocity within 5.6e-7, the action
  chunk within 1.4e-6.
- **Data:** the SmolVLA paper's real-world SO100 sets `lerobot/svla_so100_pickplace`
  @ `728583b5` and `lerobot/svla_so100_stacking` @ `933bcc61` (Apache-2.0;
  two 480 × 640 cameras, top → camera1, wrist → camera2).  Calibration: 8
  frames of 8 episodes; evaluation: 24 frames of 24 other episodes (12 per
  task), each at 10 / 35 / 60 % of its episode.  States and actions
  normalised by each dataset's stats.  The frames are decoded with ffmpeg
  (AV1, rgb24) and pinned by SHA-256.
- **Metric:** the action chunk (50 × the robot's 6) against float with the
  same noise, in normalised units: mean |d|, its first 10 actions (what a robot
  executes before replanning), relative L2.
- **Yardsticks:**
  - float with other noise: the spread of the policy itself, since one chunk
    is one sample of a distribution;
  - bf16 at every boundary;
  - lerobot's default precision, from `from_pretrained`: the VLM and the
    expert in bf16, the cross k / v and the projections in float32;
  - each one's error against the recorded actions.
- **The emulated partition** (`smolvla_q.py`'s docstring; llm_study's
  conventions):
  - the vision encoder is SmolVLM's emulation (`vlm_study.VisionModel`), with
    √960 folded into the connector;
  - the VLM and the expert are llm_study's primitives with the prefix-LM mask;
  - the K / V caches are shared: the expert's self layers write their 50 rows
    at the VLM's exponents;
  - the suffix embedding is fused: action_in_proj and the action half of
    time_mlp_in become one 32 → 720 MatMul, and the time half becomes a bias
    per step;
  - SiLU of the time MLP comes from a table, with the bias an integer at the
    input's exponent;
  - x and v are float32 between the steps, as in torch.

### Results (24 evaluation samples)

Policies against float with the same noise (`vla_study.py study --torch`).
Normalised units: one unit is one standard deviation of the joint over the
dataset. "gt" is the mean |d| against the recorded actions.

| policy | mean \|d\| | first 10 | rel. L2 (max) | vs gt | image features | K layer 0 / 15 | first v |
|---|---:|---:|---:|---:|---:|---:|---:|
| float (reference) | – | – | – | 0.655 | – | – | – |
| float, other noise (the policy's spread) | 0.405 | 0.228 | 66.0 % (113 %) | 0.676 | – | – | 112 % |
| lerobot's default precision (bf16 VLM and expert) | 0.0088 | 0.0043 | 1.39 % (5.1 %) | 0.657 | | | |
| bf16 at every boundary | 0.0073 | 0.0042 | 1.14 % (4.0 %) | 0.654 | 1.5 % | 0.6 / 1.3 % | 0.9 % |
| q88 (the BERT partition) | 0.416 | 0.243 | 68.6 % (110 %) | 0.699 | 76 % | 53 / 68 % | 40 % |
| pow2+p12 | 0.0286 | 0.0130 | 4.13 % (14.1 %) | 0.647 | 5.2 % | 1.8 / 3.5 % | 2.8 % |
| pow2+p12+vgelu+vsmx | 0.0266 | 0.0129 | 3.95 % (12.2 %) | 0.649 | 5.2 % | 1.8 / 3.5 % | 2.7 % |
| **pow2+p12+vgelu+vsmx+rowops** (shipped) | **0.0270** | **0.0129** | **3.93 % (14.3 %)** | **0.648** | 5.3 % | 1.8 / 3.5 % | 2.6 % |
| pow2+p12 with float weights (diagnostic) | 0.0207 | 0.0096 | 3.13 % (8.3 %) | 0.650 | 4.7 % | 1.6 / 2.8 % | 2.0 % |

How to read it:

- **One chunk is one sample.** Drawing other noise moves the chunk by 66 %
  (mean |d| 0.40). The int16 datapath moves it by 3.9 % (0.027), 6 % of that
  spread.
- **The task error does not move.** Against the recorded actions, float's
  mean |d| is 0.655 and the shipped policy's 0.648. The base checkpoint was
  not fine-tuned on these tasks, which is why both are far from the
  recordings.
- **lerobot's own default precision** (bf16) is 1.4 % from float. The int16
  datapath is 2.8× that, much as SmolVLM's was (KL 2.6× bf16's, CHAT_PLAN
  §22).
- **The first 10 actions** (what a robot executes before it replans) are
  closer still: 0.013.
- **Plain Q8.8 is broken**, at the level of other noise. The VLM's residual
  reaches 3773 (massive activations, as SmolLM2's 15 606), the image features
  1403 (×√960), the vision scores 435.
- **Weight rounding is a quarter of the error.** Float weights give 3.1 %
  against 3.9 %; the rest is the activations, mostly the image features'
  5.3 % (as SmolVLM's 5.4 %).

Value ranges (float, max |x|):

| part | values |
|---|---|
| vision | residual 631, fc2 630, image features 1403 |
| VLM | residual 3773, down_proj 1659, scores 211 |
| expert | residual 13.7, scores 184, x_t 4.8, v 5.6 |

Saturation, in the shipped policy: no weight saturates, and the accumulators
stay below 167 of ap_fixed<32,16>'s 32 768.  A few values saturate:

- vision fc1: 2;
- VLM down_proj: 2;
- the expert's P·V: 308 of about 2·10⁸ over the 24 samples.

**A lesson from the spec's revision.** The first draft fed V' (the cross
layers' 320 × 320 value projection) from the v projection at its own output
exponent, up to 2⁻¹⁵. With f_w = f_out + 8 − f_in that took up to 7 bits
from its weights, and the action error tripled (12.4 %; the first velocity
5.9 % against 2.6 %). Re-rounding v at a capped input exponent (2⁻⁸, like
every host-written kernel input; one row-ops COPY) restores it. The table
above was measured before that revision (the spec's x / v were float64 then,
V' read the V cache); the final specification's numbers follow.

**The final specification** (`smolvla_q.py` as the library computes it:
x / v float32, V' from v re-rounded at 2⁻⁸, the state row's double sum left
to right, the expert's first bias rounded at 2⁻¹⁵, the image features
float32; `study --torch --policies pow2+p12+vgelu+vsmx+rowops`, 24 samples):

| policy | mean \|d\| | first 10 | rel. L2 (max) | vs gt | first v |
|---|---:|---:|---:|---:|---:|
| float | – | – | – | 0.655 | – |
| float, other noise | 0.405 | 0.228 | 66.0 % (113 %) | 0.676 | 112 % |
| lerobot's default precision | 0.0088 | 0.0043 | 1.39 % (5.1 %) | 0.657 | |
| **pow2+p12+vgelu+vsmx+rowops** | **0.0272** | **0.0125** | **4.03 % (12.4 %)** | **0.647** | 2.7 % |

Where the error comes from (`vla_study.py ablate`, 8 samples, one part
quantised at a time, the others float):

| quantised | mean \|d\| | rel. L2 | first v |
|---|---:|---:|---:|
| all three | 0.041 | 5.6 % | 3.8 % |
| the vision encoder only | 0.033 | 4.3 % | 2.5 % |
| the VLM only | 0.010 | 1.4 % | 0.8 % |
| the expert only | 0.010 | 1.8 % | 1.1 % |
| the VLM and the expert | 0.013 | 2.1 % | 1.5 % |

The vision encoder is most of it: its image features carry SmolVLM's 5.3 %
error (CHAT_PLAN §22: mostly fc1 / out_proj weight rounding and P at 2⁻¹²).
Whatever improves SmolVLM's encoder improves this too.

These are the formats the library ships with
(`demo/vla/smolvla_base_pow2+p12+vgelu+vsmx+rowops_formats.json`) and the
numbers the board computes bit for bit (§3).

### Cost

`vla_study.py costs` (MACs per chunk; projected from the board's SmolVLM
vision encoder, 1614 ms per image, and SmolLM2-360M's prefill, CHAT_PLAN §26):

| cameras | prefix rows | vision | VLM prefix | expert (10 steps) | projected |
|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 107.7 GMAC (1.61 s) | 11.6 GMAC (0.14 s) | 50.1 GMAC (10 × 68 ms) | ~2.4 s |
| 2 | 141 | 215.3 GMAC (3.23 s) | 21.5 GMAC (0.26 s) | 51.1 GMAC | ~4.2 s |
| 3 | 205 | 323.0 GMAC (4.84 s) | 31.6 GMAC (0.39 s) | 52.1 GMAC | ~5.9 s |

Weights (int16): vision + connector 186 MiB, VLM 282 MiB, expert 188 MiB =
656 MiB; the embedding table 90 MiB on the host (bf16).  The vision encoder
is three quarters of the time: each camera costs as much as a SmolVLM image.

## 2. Implementation (2026-11-01)

### Scheduler: the frontend and its ops

`inference-scheduler/src/smolvla.py` writes the entries straight from the
checkpoint, the way `src/llama.py` and `src/moonshine.py` do.  Every exponent
comes from the formats of `vla_study.py formats`
(`demo/vla/smolvla_base_pow2+p12+vgelu+vsmx+rowops_formats.json`).  It needs
the row-ops unit and VectorOPKernel's softmax and activation units.

| entry | inputs | does |
|---|---|---|
| `vision` | 1024 patches of the resized frame (raw uint8) | SmolVLM's vision entry (`src/vit.py`, rowops + vsmx), √960 folded into the connector → state `vla.img` (float32, DMA) |
| `image_<c>` | — | copies `vla.img` into camera c's state `vla.img<c>` (host float32) |
| `prefix_<C>` | ids [48], n (64 C + the ids + 1), state [32], 0 | `VlaPrefix` assembles the rows (the cameras' features, the embeddings × √960, the state row) into the int32 residual; 16 VLM layers on T = roundup(64 C + 49, 64) rows → the K / V caches of layers 0–14; for the odd layers K' = RoPE(k)·Wk' and V' = v·Wv' (v re-rounded at 2⁻⁸) → the cross caches |
| `expert` | x [50][32] (f32), step, pos = n, 50, xpos = n − 50, 0 | `VlaActIn` → the fused input MatMul → `VlaTimeSilu` (the step's bias) → mlp_out + bias into the int32 residual; 16 expert layers on 64 rows (even: self-attention into the VLM caches' rows pos …; odd: cross-attention over K' / V'); RMSNorm → action_out_proj → `VlaVelocity` → v [50][32] (f32) |

How the prefix-LM mask maps onto the kernels:

- **Attention:** q·Kᵀ and P·V run on ConvKernel.
- **Softmax:** VectorOPKernel's softmax unit runs it with the new
  `LlmAttnSoftmax causal = 2` mode: every row sees the keys < n − 1.
- **The state row:** it alone sees its own key, so the host op
  `VlaStateAttn` computes it per KV group from the same scores. It uses the
  unit's integer softmax specification over the n keys, then P·V the way
  ConvKernel computes it. `VlaAttnMerge` puts that row in place.
- **Why a host op:** a kernel node may not do host work. With the Cortex-R5
  driving, a node's kernel calls are only recorded until a host op flushes
  the batch.

The expert's even layers write their 50 K / V rows into the VLM's caches
right after the prefix (`LlmAttnPrep` at pos = n). The two decoders share
those caches' exponents by construction (`smolvla_q.make_formats`).

New ops (`src/vla_nodes.py`):

- **Host ops:** `VlaPrefix`, `VlaActIn`, `VlaTimeSilu`, `VlaVelocity`,
  `VlaStateAttn`, `VlaAttnMerge`.
- **Row-ops node:** `VlaAttnPrep`, `LlmAttnPrep`'s jobs with parts left out.
  - `q`: the expert's cross layers, RoPE(q) at positions 0 … 49 and the q
    image, no cache.
  - `k`: the prefix's odd layers, RoPE(k) of every row for the K' MatMul.
  - `v`: the prefix's odd layers, v re-rounded at V''s capped input exponent
    (one COPY).
- **Reused:**
  - `AsrCrossPrep` writes the cross caches;
  - the Llama ops do the rest.

### The library

`demo/vla/scripts/generate_vla_project.py` → `libsmolvla.so` + `vla_bench`
(`demo/vla/src/vla_api.{c,h}`, `vla_bench.c`; a ctypes wrapper
`demo/vla/vla_lib.py`):

- **`vla_set_image(c, rgb, h, w)`:** resizes with padding exactly as the
  specification does (the same double operations, round half to even), then
  runs the vision entry and camera c's slot.
- **`vla_act(C, ids, n, state, noise, actions)`:** runs the prefix, then ten
  expert steps. The Euler update is in float32.
- **`vla_noise(seed)`:** reproducible standard normal noise.

The generated project (3 cameras):

- 2721 nodes in 8 entries;
- a 703 MiB DMA pool;
- the weights 761 MB on disk, including the 94 MB embedding table on the host;
- `inference.c` 7.8 MB;
- 7 min to generate, 8.8 GB peak.

The bitstream's performance model predicts:

- the vision entry: 1.59 s (SmolVLM's measures 1.61 s on the board);
- `prefix_1` / `_2` / `_3`: 0.23 / 0.36 / 0.47 s;
- one expert step: 89 ms.

That makes **4.4 s for a chunk from two cameras** and 2.7 s from one. The R5
is recommended for every entry but the image slots.

### Tests (`inference-scheduler/test/test_smolvla.py`)

These run a small random SmolVLA (a 2-layer 64-wide vision encoder on
128 × 128 frames, a 4-layer VLM and expert, 10 actions, 3 steps) with formats
the specification calibrates:

- **The simulation equals the specification bit for bit** at every point
  checked: the image features, the K caches, K', and every step's velocity
  over two cameras.
- **The generated C equals the simulation**, coherent and incoherent (where
  the syncs move the data).
- **The library** equals `SmolVLA.sample` on the host emulation: one camera
  with no resize, two cameras with resize and padding, the repetitions and
  the reopen identical. The ctypes wrapper drives it too.
- **The node census.**

The tiny model also made one constraint explicit. The cross attention starts
at xpos = n − chunk, so one camera's image tokens + 1 must be at least the
chunk. SmolVLA: 65 ≥ 50.

## 3. On the board (2026-11-01, bitstream `b4e63457057a`)

`vla_board.py` uploads the project and the weights (794 MB), builds
`vla_bench` and `libsmolvla.so` on the board (106 s, cc1 at most 335 MB), and
runs the 24 evaluation samples twice each, then closes and reopens. The A53
drives the kernels; the R5's firmware was not running.

- **Every action of the 24 samples is bit-exact with the specification**, and
  the repetitions and the re-open are identical. Two things the gates caught
  on the way:
  - **FMA contraction on the A53.** The first board run matched 1 of 4
    samples, the others differing by at most 6e-8, with the network identical.
    The cause: GCC fuses `x + dt·v` into an FMA on the A53 by default, the
    host emulation on x86 does not, and `inference.c` turns it off with a
    pragma but `vla_api.c` did not yet.
  - **The V' input exponent** (§1). It needed a spec change, new formats and
    a regenerated project.
- **The host emulation** (`vla_host_emu.py`) matches the specification on
  the real model too: one sample of the final project, 24 min of software
  kernels, the repetition and the re-open identical.

Timing (`vla_bench`, medians; the image includes the resize):

Before §4's faster vision encoder:

| cameras | each image | prefix | ten steps | a chunk of 50 actions |
|---:|---:|---:|---:|---:|
| 1 | 1.66 s | 0.30 s | 0.82 s | 2.78 s |
| 2 | 1.66 s | 0.47 s | 0.86 s | 4.65 s |
| 3 | 1.66 s | 0.66 s | 0.89 s | 6.53 s |

The bitstream's performance model predicted 1.59 s, 0.23 / 0.36 / 0.47 s and
0.89 s. The prefix runs 30–40 % over its prediction because the model does
not price the host ops: the prefix assembly, the state rows, the merges. The
expert step comes in under its prediction (86 ms against 89 ms).

`vla_open` takes 1.4 s from the page cache and 17 s from the SD card. The pool
is 703 MiB of CMA, so the chat server cannot run beside it. Miranda can: it
holds no FPGA buffer.

### What next

- **The vision encoder is 70 % of a two-camera chunk.** SmolVLM's encoder
  measures the same 1.6 s per image, so making it faster serves both models
  (done: 1.66 → 1.31 s, §4).
  Lerobot's asynchronous inference would also hide some of it: start the next
  chunk while the robot executes this one.
- **One camera is enough for many tasks.** `vla_act(1, …)` gives a chunk in
  2.8 s.
- **A fine-tuned SmolVLA** (the base checkpoint is meant for fine-tuning)
  needs:
  - its formats (`vla_study.py formats` on its own data);
  - the project regenerated;
  - the gates rerun.
  
  `vla_study_inputs.json` pins the base checkpoint and the SO100 samples.
- **Live cameras.** The camera demo's AR1335 path (`demo/camera`) can feed
  `vla_set_image` directly. A robot needs its state and the dataset's
  normalisation stats.
- **The performance model.** SmolVLA's kernel calls are not in
  `perf_calibrate.SHIPPED` yet. A top-up would let `--plan` price them, and
  the prefix's predictions include its host ops (the kernel calls done in §4;
  the host ops still need a profile merged into `host.json`).

## 4. A faster vision encoder (2026-11-01)

**Result: 1.66 → 1.31 s per camera image (−21 %), a two-camera chunk 4.65 →
3.96 s (−15 %), every action of the 24 samples still bit-exact** (production
bitstream `b4e63457057a`, no spec change). The encoder is SmolVLM2's SigLIP
(12 layers, 1024 patches, D 768, 12 heads of 64), the same shape as SmolVLM's,
so SmolVLM's library gains the same once regenerated.

### Where the time went

The bitstream's performance model priced the old entry's calls at 1.59 s, and
the board measured 1.66 s with the resize. Per kind:

| calls | ms | |
|---|---:|---|
| the linears (76 MatMuls on ConvKernel, ~96 % of its MAC peak) | 706 | |
| the softmax, column mode (`OP_SOFTMAX_T`) | 542 | 3.76 ms a head |
| q·Kᵀ / P·V on ConvKernel | 270 / 154 | 1.87 / 1.07 ms a head |
| the attention prep, GELU, residual adds, LayerNorms | 57 / 44 / 29 / 20 | |

The attention was softmax-bound: 45 ms a layer on VectorOPKernel against
ConvKernel's 35. The column mode reads its keys-major scores two beats per
key row; the row mode measured 1.81 ms for the same 1024 × 1024
(`run_remote_perf.py`).

### What changed (`src/vit.py`, all bit-identical)

- **Query-major scores.** `VitAttnPrep` `query_major = 1`
  (`rowops_nodes.from_vit_attn_prep`) swaps q and k: k goes through the image
  path into the q·Kᵀ conv's x image and q into the "K cache" rows, its
  weight. So the conv writes s [queries][keys] (the same integer dot
  products), and `VitAttnSoftmax` `query_major = 1` runs one `OP_SOFTMAX`
  (row mode) call (`smx_nodes.VitAttnSoftmaxVopNode`). Default with `vsmx` and
  `rowops`. **1372 ms** an image.
- **Row pieces, pipelined.** The linears, LayerNorms, GELUs and residual
  adds run in two row halves, pinned to ConvKernel by the new `axi.numeric`
  key `engine`. The pieces of a weight share one image
  (`matmul_lowering.pinned_weight_images`): before, a weight read by two
  MatMuls fell back to kw 1, half the speed. The residual stream stays in
  pieces from the patch embedding (`Slice` views of `pe`) to the post-norm.
  q / k / v and the post-norm output are `Concat`s the kernels do in place
  (`CodeGenerator._merge_in_place`), and the attention output is read by row
  `Slice` views. A row `Concat` / `Slice` of per-channel exponent tensors is
  now a valid raw copy (`numeric.check`). The frontend issues the nodes in
  the order of a greedy list schedule over estimated lane times
  (`vit._Pipe`), so one piece's row ops run under the other's linears. The
  heads' q·Kᵀ / softmax / P·V go the same way. **1314 ms**, with two pieces
  or four.

### What did not work: MatmulKernel beside ConvKernel

MatmulKernel sits idle through the whole encoder. Two levers gave it work,
priced on the performance model's standalone measurements:

- a fifth row piece of every linear on MatmulKernel, reading ConvKernel's
  weight image through its image path (balanced at 176 of 1024 rows);
- q·Kᵀ of 8 of 12 heads on MatmulKernel (the cache rows times the image,
  row-major).

Simulated, they gave 1164 ms with the piece and 1038 ms with the heads too.
**On the board they lost**: 1826 ms
with both, and 1354 ms with the heads alone against 1314 without. Beside
ConvKernel, MatmulKernel's calls ran ~1.9× slower than alone (an fc1 piece's
bracket 34 ms against 19.7 measured alone; q·Kᵀ 2.6 → ~5 ms). The two kernels
share DDR through the coherent PS ports, and a linear on MatmulKernel re-reads
its weight for every row group. The performance model measures each call
alone and cannot see this.

The code was removed. Its patch, with the q·Kᵀ node `LlmAttnScoresMmNode`
and `run_matmul_off()`, is kept outside the repo on the maintainer's machine.
The engine pin keeps both kernels (`test_matmul_on_conv.TestEnginePin`
splits a linear between them, bit-exact).

### Board runs (`vla_board.py`, the A53 driving, a camera image with its resize)

| vision entry | per image | two-camera chunk | simulated (kernels) |
|---|---:|---:|---:|
| before: keys-major, column-mode softmax | 1660 ms | 4650 ms | 1590 ms |
| query-major, row-mode softmax | 1372 ms | 4074 ms | 1317 ms |
| + 4 row pieces (256), pipelined | 1314 ms | 3963 ms | 1211 ms |
| + 2 row pieces (512), pipelined — **shipped** | **1313 ms** | **3956 ms** | 1214 ms |
| 4 pieces + q·Kᵀ of 8 heads on MatmulKernel | 1354 ms | 4037 ms | — |
| 4 ConvKernel + 1 MatmulKernel piece + the heads | 1826 ms | 4985 ms | 1038 ms |

Every variant was bit-exact (2 samples each). The shipped one passed the
full gate: 24 / 24 samples, the repetitions and the re-open identical. The
prefix and the ten steps are unchanged; per camera count (`vla_bench`,
medians; the 1- and 3-camera runs on the gate's first 4 samples twice, the
third camera the top view again):

| cameras | each image | prefix | ten steps | **a chunk of 50 actions** |
|---:|---:|---:|---:|---:|
| 1 | 1.31 s | 0.30 s | 0.83 s | **2.44 s** |
| 2 | 1.31 s | 0.47 s | 0.86 s | **3.96 s** |
| 3 | 1.31 s | 0.67 s | 0.90 s | **5.51 s** |

The pool is 701 MiB (703 before: q / k / v and the post-norm output are
written in place).

The kernel model predicts the shipped entry at 1214 ms against the board's
1313 ms with the resize. About 55 ms of that is the resize and the image
copy, as before; the other ~45 ms are waits the model does not see. In the
simulation, ConvKernel's only remaining idle time is the attention prep:
4.8 ms a layer, 57 ms an image. A prep per row piece would hide most of it
(the V image's 64-row blocks fit pieces of 256 or 512).

### The performance model

`perf_calibrate.SHIPPED` now includes SmolVLA (`vla_graphs`: the entries of
`generate_vla_project.py`). Three `cases` / `cases --refine` rounds and two
`run --resume` sessions measured 441 + 79 new calls → 3533 exact calls in
`perf_models/kv260/b4e63457057a.json`, every call of the vision entry among
them. The A53's early wake depends on these predictions. The row-mode softmax
at its family fit (2.96 ms against 1.81) would have woken every wait about
1 ms late.

### Tests

- `test_vit.TestVisionRowops`: `test_query_major` (the same image features as
  keys-major) and `test_pieces` (pieces on both kernels: engines, one image,
  in-place Concats, views; == the study's spec; host emulation with separate
  CPU / DDR copies).
- `test_host_ops.TestDataMovement.test_concat_in_place`.
- `test_matmul_on_conv.TestEnginePin`.
- `test_perf_calls`: the pieced entry's calls == the emitted C.
- The full library on the host emulation: `vla_host_emu.py`, bit-exact.

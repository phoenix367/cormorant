---
description: Go / no-go feasibility study of a new neural network for the KV260 stack BEFORE porting it — operator coverage and kernel bounds, memory (DMA pool vs cma=1000M, board and host RAM), numeric accuracy of the int16 power-of-two datapath against float, projected latency — ending in a written verdict in a plan doc. Use when asked whether a model can run on the board, to "study" a model, or before adding a model to a demo or the chat server. Llama-family checkpoints go through llama_fit.py + llm_calibrate.py / llm_study.py; any ONNX model through onnx_study.py; other architectures get a study script modelled on vlm_study.py.
allowed-tools: Bash Read Write Edit
---

# model-study

A study answers, in this order, and stops at the first hard NO-GO (but
lists every blocker it saw): **operators → memory → numerics → speed →
what the implementation needs**.  It is host-only — no board time, no
bitstream.  It is how BERT-base (BERT_PLAN §0), SmolLM2-135M (CHAT_PLAN
§3.2 B0, results §9–§10), SmolLM2-360M (§20) and SmolVLM-256M (§22) were
decided; read the matching section before starting a similar model.

## 0. Intake

- **Source and pin.**  ONNX file, or a Hugging Face repo pinned to a
  revision (commit hash — `resolve/main` moves), its license, the SHA-256 of
  every file.  Assets are untracked (`demo/chat/assets/`, `.gitignore`d):
  check disk first (`.venv-export` is 1.2 GB, the 360M checkpoint 724 MB).
- **Architecture table**: parameters, layers, width, heads / KV heads,
  head_dim, FFN, vocabulary, context, norm, activation, positional encoding,
  tied head.  Note anything unusual (biases, RoPE scaling, a new vision
  tower, dims above kernel bounds).

## 1. Pick the route

| route | the model | tools |
|---|---|---|
| **A** | Llama-family decoder: RMSNorm, rotate-half RoPE (no scaling), GQA, SwiGLU / SiLU, no biases, one `model.safetensors` | `scripts/llama_fit.py`, then `demo/chat/scripts/llm_calibrate.py` + `llm_study.py` (the frontend `src/llama.py` exists) |
| **B** | an ONNX graph the scheduler takes as is — CNNs, BERT-like encoders | `scripts/onnx_study.py` (+ the task metric on real data) |
| **C** | anything else — another decoder family, a new vision tower, ops the census lists MISSING | a new `<model>_study.py` modelled on `vlm_study.py` (§5) |

Scripts are in this skill's `scripts/` directory; paths below are from the repo root.

## 2. The gates (every route)

**2.1 Operators and kernel bounds.**  Route B: the op census of
`onnx_study.py` (MISSING = no kernel or host op; pattern-only =
`ReduceMean` / `Pow` / `Sqrt` / `Reciprocal` / `Tanh` / `Erf`, fine only
inside a fused LayerNorm / GELU).  Kernel bounds are in
`platforms/kv260.json` `kernels.{conv,matmul,pool}`; a K beyond every
kernel is split into chunks with a host sum (SmolVLM's connector:
K 12 288 = 3 × 4096).  For each missing op name the fix and its cost: a host
op (`src/host_nodes.py`: numpy reference + C helper), a pattern fusion
(`src/fusion.py`), or a graph rewrite.

**2.2 Memory.**
- DMA pool (int16 weights, one copy — decode reads the prefill image
  through GEMV; KV caches; intermediates) against `cma=1000M` (~954 MiB
  usable, idle CmaFree ~1011 MB).  Pools served today: BERT 216, SmolLM2-135M
  286, SmolLM2-360M 740, SmolVLM 495 MiB.  Above ~920 MiB: NO-GO without a
  CMA change.  Above what is left beside the others: it swaps under
  `--resident auto` (a warm swap costs 1–2 s, a cold load 20–55 s).
- Board: 3.9 GB RAM, no swap — host tensors (float32 residual, embedding
  table, image buffers) plus the server; generated projects build `-j1`.
- Dev PC (46 GB): generating a Llama library peaks at 3.3 GB (135M) /
  8.2 GB (360M), about 23 bytes per parameter (CHAT_PLAN §25);
  `llm_sched_check.py` more (135M: 5.1 GB) — one at a time.

**2.3 Numerics — the core.**
1. A float reference, validated first: numpy vs onnxruntime (≤ 1e-5) or vs
   torch float32 (logits ~7e-5, greedy tokens and chat-template ids equal).
2. A bit-level emulation of the planned partition: kernels take raw int16,
   sum exactly in ap_fixed<32,16>, write floor(acc / 2^8) saturated; host
   regions compute in double and write back round-half-even + saturate.
   The scheduler must later match this BIT FOR BIT (`llm_sched_check.py`,
   `vlm_sched_check.py`, `bert_sched_check.py`), so the emulation is the
   specification.
3. Policies in cost order — `q88` → float residual → attention sink
   (precomputed position-0 K / V) → per-channel power-of-two exponents
   (weights at f_w = f_out + 8 − f_in) → softmax P at 2^-12 (`p12`) → mixed
   prefill / decode attention; an ablation per tensor class
   (`llm_study.py ablate --base fit+sink`) finds what breaks, and `+fw`
   (float weights) bounds what a bitstream change could buy.
4. Yardstick: bf16 against float.  The criterion is answer quality
   indistinguishable from float, not exact match (bf16 matched only 5 / 12
   greedy answers of 135M).

What passed before (GO):

| model | metrics (policy vs float; bf16 in brackets) |
|---|---|
| BERT-base | EM / F1 93.3 / 97.1 vs 93.3 / 96.5 on 60 SQuAD dev, same span 59 / 60 (one example = 1.7 points) |
| SmolLM2-135M | top-1 0.976 prompts / 0.969 held-out (0.984 / 0.980), KL 0.0024, ppl 15.616 vs 15.598 (15.607); 433 of 4.0 G values saturate, accumulator peak 295 of 32 768 |
| SmolLM2-360M | top-1 0.978 / 0.976, KL 0.0018, ppl 12.394 vs 12.373 (12.395), 8 / 12 identical, no weight saturates |
| SmolVLM-256M | image features rel. error 0.054 (0.017), answer top-1 0.975, KL 0.0031 (2.6 × bf16), accumulators < 261 |
| LeNet (MNIST) | top-1 97.35 % vs float 97.37 % on 10 000 images, 99.91 % agreement; 20 % of logits saturate at ±128 (the 9 disagreements) — LENET_PLAN |

What failed: plain `q88` everywhere on decoders (135M top-1 0.100, ppl
22 856; SmolVLM features rel. error 0.879, 0 / 24 identical) — the
position-0 attention sink (residual 25 982), softmax P at 1/256 and
saturating LayerNorm / GELU intermediates (x², x³) run op by op.

**2.4 Speed.**  Route A: `llama_fit.py` (fitted on both SmolLM2 libraries)
and `llm_study.py costs` (bytes and GMAC per token / per prefill).  Route
B: `onnx_study.py` replays the event stream with the bitstream's
performance model (`perf_models/kv260/`, within ±2 % where the calls were
measured; the family-model share carries its error band — ConvKernel
families up to ±37 % p90).  ConvKernel per-layer detail: the
`conv-cycle-model` skill.  Older studies scaled GMAC at ~40 GMAC/s
(ConvKernel) / ~2.3 (MatmulKernel); the board came in slower (BERT 10.8 s
projected → 12.1 s, SmolVLM 5–7 s → 7.7 s) — prefer the simulator.

**2.5 What the implementation needs**: host ops, frontend, exponent
metadata (`src/numeric.py`), entries (decode / prefill_<T> / head /
vision), server backend, CMA — each with a size estimate.

## 3. Route A — Llama-family decoder

```bash
python3 .claude/skills/model-study/scripts/llama_fit.py <assets dir or config.json> [--files ...]
# eligibility, params, pool vs CMA, decode / prefill estimate, host RAM; exit 1 = not eligible as is

PY=.venv-export/bin/python      # numpy, tokenizers, torch CPU, transformers; if missing:
#   python3 -m venv .venv-export && PIP_CONFIG_FILE=/dev/null .venv-export/bin/pip install \
#     --extra-index-url https://download.pytorch.org/whl/cpu -r demo/chat/scripts/requirements-study.txt
python3 demo/chat/scripts/llm_calibrate.py add <name> --repo <org/Repo> [--revision REV]  # pin in llm_models.json
python3 demo/chat/scripts/llm_calibrate.py fetch <name>                  # checkpoint + WikiText-2 texts, SHA-256 checked
python3 demo/chat/scripts/llm_calibrate.py validate <name>               # float64 numpy vs torch float32 (~2 min)
$PY demo/chat/scripts/llm_study.py --assets demo/chat/assets/<name> study [--quick]       # policies (135M: 42 min / 12 policies)
$PY demo/chat/scripts/llm_study.py --assets demo/chat/assets/<name> ablate --base fit+sink  # per tensor class (~12 min)
$PY demo/chat/scripts/llm_study.py --assets demo/chat/assets/<name> costs
python3 demo/chat/scripts/llm_calibrate.py calibrate <name> --record    # formats + provenance, hash stored
```

Start with `study --quick` (4 prompts, 32 tokens, one window) to rank the
policies, then the full run on the candidates (`--policies bf16,pow2+sink+p12,pow2+sink+p12+mix`).
Results: `demo/chat/assets/study/<name>/` (results.json, generations.txt).
The calibration is deterministic per machine; another CPU may round a sink
value differently (OpenBLAS) — `llm_calibrate.py check` names it.

## 4. Route B — ONNX model

```bash
inference-scheduler/.venv/bin/python .claude/skills/model-study/scripts/onnx_study.py MODEL.onnx \
    [--inputs X.npz] [--samples N] [--threshold 0.05] [--json OUT]
```

- **Census and coverage**: ops by status, then the scheduler's partition
  (nodes per kernel / host / zero-cost) or its first error.
- **Memory**: DMA pool = weights + intermediates, share of CMA.
- **Numerics**: the scheduler's simulation (what the generated C computes)
  against onnxruntime on the same grid-quantised inputs — per output rel.
  L2, cosine, top-1 agreement; per tensor the drift, the float max, the
  saturated and below-resolution fractions; the first tensor over the
  threshold.  **Use real inputs** (`--inputs`: `{input name: array}`,
  `[N, *shape]`, or N images stacked on a batch-1 input's axis) for the verdict — random data only measures drift (ResNet-18
  logits drift 20 % on noise, 10 % on a real image, top-1 identical).  The
  demos' preprocessed files are raw int16 at Q8.8
  (`np.fromfile(..., '<i2') / 256`).  Then the task metric on tens of real
  samples (top-1 / EM-F1 vs float), as BERT did with 60 SQuAD examples.
- **Latency**: predicted total, kernel lanes, CPU waits, the measured vs
  modelled share and every unpriced node.

Measure the model's own float accuracy on the same encoded inputs before
blaming the datapath: LeNet's 97.35 % on the board looked like a loss next
to the convnet's 98.92 %, but float scores 97.37 % — and check the
encoding itself (scale, centring, orientation) in float first.

Reading it: saturation of a masking constant is benign (BERT's
(1 − mask) · −10000 clamps to −128: exp still ~0); saturation or large drift
in real activations means the model needs exponent metadata
(`src/numeric.py`) through a frontend — route C, as the Llama and ViT
graphs did.  Saturation before a ReLU is harmless when it is on the
negative side (LeNet conv1 / conv2).  A slow layer in the latency line is
the next lever: a Conv whose kernel covers its whole input is a
fully-connected layer — `--fc-conv auto` runs it as a GEMV MatMul (LeNet
conv3: 4.63 → 2.03 ms); check `report.md`'s transformations and the
per-node prices (`kernel_duration_fn`) for similar mismatches.  Checked:
MNIST 0.26 ms predicted vs 0.268 on the board, ResNet-18 60.0 vs 59.9 ms,
LeNet 5.44 / 2.81 ms before / after the rewrite, both equal to the board.

## 5. Route C — new architecture

Write `demo/<app>/scripts/<model>_study.py` on `vlm_study.py`'s pattern:
`fetch` (pinned revision, SHA-256), `validate` (numpy float64 vs torch /
onnxruntime), `study` (policies, the metrics of 2.3, greedy generations
side by side), `ablate`, `formats` (calibrated exponents, one bit of
headroom, separate calibration data), `costs`.  Emulate the partition the
scheduler will generate — the study script becomes the spec its frontend
must match bit for bit.  Deterministic: greedy only, fixed data.

## 6. The write-up

A new section in the plan the model belongs to (`doc/plans/CHAT_PLAN.md` for
chat models, else `doc/plans/<MODEL>_PLAN.md`, linked from `doc/README.md`):

```
## N. <Model> study (<date>)
**Verdict: GO | GO with <policy> | NO-GO (<blocker>)** — one sentence, first.
Model: repo @ revision, license; architecture table; pool MiB (fits / swaps with …)
Method: reference and its validation, data, emulated partition, policies, yardstick
Results: policy table (top-1 prompts / held-out, KL, ppl, identical answers — or the task metric)
What breaks and what fixes it: ablation per tensor class; value ranges / saturation
Cost: projected latency / tokens per s, and how it was projected
What the implementation needs: host ops, frontend, entries, backend, CMA
Commands and runtimes (reproducible)
```

## Rules

- Host only: a study never needs the board or the chat server stopped.
- Real data for the verdict; seeded, deterministic runs; record the
  commands and runtimes in the write-up.
- Keep study outputs in the untracked assets tree; only provenance / pins
  (`llm_models.json`, `*.provenance.json`) and the write-up go to git.
- Do not commit unless the user asks.

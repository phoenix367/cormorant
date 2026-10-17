# ASR_PLAN — speech to text on the KV260

Goal: an English speech-recognition model on the FPGA for the chat server
(`POST /v1/audio/transcriptions`, OpenAI-compatible), next to text to
speech (TTS_PLAN).  The study (the `model-study` skill, route C) is host
only; this document is its write-up and the implementation plan.

Scripts: `demo/asr/scripts/` — `asr_study.py` (fetch, utterance selection,
validation, the float screen, Whisper's shortened window, the int16 study
and its ablation), `whisper_np.py` / `moonshine_np.py` (numpy float64
references), `moonshine_q.py` (the bit-level emulation of the planned
partition: the specification a frontend must match), `asr_costs.py` (proxy
graphs priced by the bitstream's performance model), `asr_study_inputs.json`
(the pinned checkpoints, archives and utterances).  Assets and results:
`demo/asr/assets/` (not in git; a link to /mnt/data on the maintainer's
machine).

## 1. Candidate screen (2026-10-17)

**Result: Moonshine, not Whisper.**  Moonshine's encoder runs on the audio's
length; Whisper's always on a 30 s window, which it cannot shorten without
breaking (WER 5.05 → 67.4 %).  On the same 200 utterances Moonshine tiny is
more accurate than Whisper tiny.en at a quarter of the compute, and Moonshine
base is the most accurate of the four at about half Whisper tiny's.

**Data.**  LibriSpeech test-clean (CC BY 4.0, openslr.org/12; SHA-256 in
`asr_study_inputs.json`): 200 utterances, 5 per speaker evenly spaced over
each speaker's sorted list, ≤ 29 s (one Whisper window) — 28.0 min of audio,
mean 8.4 s, 1.9–28.4 s.  Calibration (§2) uses 40 utterances of dev-clean
(one per speaker, other speakers).  WER after Whisper's English normaliser on
both sides (the Open ASR leaderboard's).  Greedy decoding throughout:
Whisper with its checkpoint's suppress lists and `<|notimestamps|>`,
Moonshine up to 6.5 tokens per second of audio (its model card's limit).

| model (pinned revision) | params | license | WER | GMAC per utterance | encoder | per token |
|---|---|---|---|---|---|---|
| openai/whisper-tiny.en (`87c71024`) | 37.8 M | Apache-2.0 | 5.05 % | 21.1 | 20.2 (30 s window) | 32.8 MMAC |
| openai/whisper-base.en (`911407f4`) | 72.6 M | Apache-2.0 | 4.15 % | 49.9 | 48.4 (30 s window) | 57.8 MMAC |
| UsefulSensors/moonshine-tiny (`390624ed`) | 27.1 M | MIT | 4.47 % | 4.8 | 4.1 (8.4 s mean) | 19.1 MMAC |
| UsefulSensors/moonshine-base (`7a73d8d5`) | 61.5 M | MIT | **2.98 %** | 11.5 | 10.1 | 39.9 MMAC |

Whisper pads every utterance to 30 s; its encoder (1500 positions) costs the
same for a 2 s question as for a 30 s one.  Running it on the audio's length
(whisper.cpp's `audio_ctx`: the first 2·T mel frames, positions 0..T−1,
rounded up to 5 s buckets) left 112 of 200 transcripts unchanged and turned
most of the rest into repetition loops — WER 67.4 % (`asr_study.py window`).
Without fine-tuning, Whisper on the KV260 costs its 30 s window every time.

Priced on the current bitstream (`04259a7af917`) with proxy graphs
(`asr_costs.py`, §2 Cost): the Whisper tiny.en encoder ≈ 490 ms per utterance
whatever its length (Whisper base.en ≈ 1.14 s), against 15 / 36 / 138 /
303 ms for 2 / 5 / 10 / 20 s of audio on Moonshine tiny.  A decoder step:
Whisper tiny.en 10.4 ms, base.en 18.5 ms, Moonshine tiny 8.1 ms, base
15.5 ms (every one bandwidth-bound: the weights and the tied vocabulary head
are read once per token).

Not screened further: Whisper small.en (922 MB, 12 + 12 layers: ≥ 4× tiny's
cost), distil-small.en (its encoder is Whisper small's), wav2vec2-base-960h
(CTC, 95 M params, 12 layers × 768 over 50 Hz frames: more encoder compute
than Moonshine base for every second of audio; not measured).

## 2. Moonshine study (2026-10-17)

**Verdict: GO with Moonshine base, policy `opt`.**  On the planned int16
partition its WER is float's (2.89 % against 2.98 %, 179 of 200 transcripts
identical, teacher-forced top-1 0.996), at a projected 0.87 s per utterance
on average (RTF 0.10) and a ~215 MiB DMA pool.  Moonshine tiny is GO with
the same policy too (4.56 % against 4.47 %; 0.47 s, RTF 0.05, ~140 MiB): the
fallback if latency matters more than the third fewer errors.

The policy beyond the scheduler's defaults: power-of-two exponents per
channel (per head for the q·Kᵀ operands) — one exponent per tensor gives
11.4 % — and the inputs of every constant-weight MatMul written at ≤ 2^−7
(the stem's conv3 input at 2^−6), which makes the weights (at f_out + 8 −
f_in) one or two bits finer; the encoder's GELUs stay on VectorOPKernel at
Q8.8.

### Model

UsefulSensors/moonshine-tiny @ `390624ed33d594443aa4aa221f5b9f283b545b5a`
and moonshine-base @ `7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf`, MIT, one
`model.safetensors` each (SHA-256 in `asr_study_inputs.json`).

| | tiny | base |
|---|---|---|
| parameters | 27.1 M | 61.5 M |
| input | raw 16 kHz audio (no spectrogram) | same |
| conv stem | conv1 1 → 288, k 127, stride 64, no bias; tanh; GroupNorm (1 group); conv2 288 → 576, k 7, s 3, GELU; conv3 576 → 288, k 3, s 2, GELU — one frame per 384 samples (41.7 Hz) | 416 / 832 channels |
| encoder | 6 layers, width 288, 8 heads of 36, FFN 1152 (GELU, erf), LayerNorm without bias, pre-norm | 8 layers, 416, 8 × 52, FFN 1664 |
| positions | RoPE, interleaved pairs, on the first 32 of each head's 36 dims (`partial_rotary_factor` 0.9), θ 10 000; encoder and decoder self-attention | first 32 of 52 (0.62) |
| decoder | 6 layers: causal self-attention (RoPE), cross-attention over the encoder frames (no positions), SwiGLU MLP (fc1 288 → 2 × 1152 with bias, SiLU(gate) · x, fc2 with bias) | 8 layers, 416, 2 × 1664 |
| head | tied to the 32 768 × 288 embedding | 32 768 × 416 |
| tokens | BPE with byte fallback, `<s>` = 1, `</s>` = 2; at most 194 positions (≈ 30 s of speech) | same |

Dims the kernels need padded: the head dim to a multiple of 16 (36 → 48,
52 → 64; zero rows in q / k / v, exact).  Nothing exceeds a kernel bound:
the largest K is conv2's im2col 7 × 288 = 2016 (base 2912) and fc2's 1152
(1664), the rows per call at most 1248 (30 s).

### Method

- **Float reference** (`moonshine_np.py`, numpy float64) against
  transformers 5.17 float32 (`asr_study.py validate`, the shortest, a median
  and the longest utterance): encoder states within 2.2e-6 (tiny) /
  3.7e-6 (base) relative, first logits within 6.5e-7, greedy transcripts
  equal.  `whisper_np.py` likewise (log-mel within 1.6e-5, encoder 1.5e-6).
- **Emulated partition** (`moonshine_q.py`, the specification): conv1 on
  the polyphase image of the int16 PCM (exponent 15, exact), every MatMul
  and conv a kernel call (raw int16, exact sums, floor(acc / 2^8),
  saturate; weights at f_out + 8 − f_in), tanh and GroupNorm on the host
  (double, round half even), the residual stream int32 at 2^−15 with every
  add exact, LayerNorm / RoPE / SiLU · x / the bias COPYs as the row-ops
  unit's integer specifications (`rowops_spec.py`), the softmax as
  VectorOPKernel's (`vectorop_smx.py`, P at 2^−12), GELU on VectorOPKernel
  (ADD bias, MUL to Q8.8, act GELU) where its output is at 2^−8 on every
  channel, else a host table, the logits at one exponent (argmax on raw
  values).  Exponents per channel (per head for the q·Kᵀ operands) from the
  float maxima of 40 dev-clean utterances with one bit of headroom,
  then lowered where a weight would leave int16 (`make_formats`).
- **Yardstick**: bf16 at the same boundaries (float math).  The criterion:
  WER indistinguishable from float's on the 200 utterances; also the
  transcripts equal to float's, the teacher-forced top-1 on float's tokens
  (every decoder step, `</s>` included) and the encoder output's relative
  error.

### Results (200 test-clean utterances, 4 432 words)

Moonshine tiny — float 198 word errors (4.47 %):

| policy | WER | errors vs float | utterances better / worse | transcripts = float | teacher-forced top-1 | encoder rel. error |
|---|---|---|---|---|---|---|
| float | 4.47 % | — | — | 200 | — | — |
| bf16 (yardstick) | 4.44 % | −1 | 2 / 1 | 189 | 0.998 | 0.005 |
| `unit`: every kernel input at ≤ 2^−8 | 4.65 % | +8 | 10 / 18 | 130 | 0.983 | 0.073 |
| **`opt`**: kernel inputs at ≤ 2^−7, the stem's conv3 input at 2^−6, the encoder's GELUs on VectorOPKernel (Q8.8) | **4.56 %** | **+4** | 7 / 14 | **155** | **0.991** | 0.063 |
| `unit+in6`: every kernel input at ≤ 2^−6 (GELUs on the host) | 4.35 % | −5 | 9 / 9 | 157 | 0.992 | 0.064 |
| `unit+in7` | 4.58 % | +5 | 9 / 14 | 143 | 0.988 | 0.066 |
| `unit+hostgelu` (unit, GELUs on the host) | 4.51 % | +2 | 11 / 16 | 129 | 0.982 | 0.073 |
| `unit+p15` (P at 2^−15) | 4.74 % | +12 | 10 / 18 | 125 | 0.981 | 0.074 |
| `unit+fw` (diagnostic: float weights) | 4.54 % | +3 | 7 / 9 | 163 | 0.992 | 0.051 |
| `unit+in9` | 7.47 % | +133 | 16 / 65 | 71 | 0.950 | 0.171 |
| `unit+tensor` (one exponent per tensor) | 11.37 % | +306 | 16 / 115 | 34 | 0.916 | 0.282 |

Moonshine base — float 132 word errors (2.98 %):

| policy | WER | errors vs float | utterances better / worse | transcripts = float | teacher-forced top-1 | encoder rel. error |
|---|---|---|---|---|---|---|
| float | 2.98 % | — | — | 200 | — | — |
| bf16 (yardstick) | 2.96 % | −1 | 1 / 0 | 194 | 0.999 | 0.006 |
| `unit` | 3.63 % | +29 | 11 / 14 | 121 | 0.981 | 0.200 |
| **`opt`** | **2.89 %** | **−4** | 4 / 5 | **179** | **0.996** | 0.087 |
| `unit+in7` | 2.87 % | −5 | 4 / 4 | 179 | 0.996 | 0.075 |

On both models `opt` and `in6` / `in7` are equivalent within the sample, and
`opt` is indistinguishable from float (tiny's 7 better / 14 worse utterances:
p ≈ 0.19 by a sign test; base's 4 / 5); `opt` keeps the encoder's GELUs on
the FPGA.  Saturation under `opt`: ~120 values (tiny) / ~990 (base) of over
10^9 — tanh's ±1, and base's fc2 outputs in its last encoder layers.

### What breaks and what fixes it

- **One exponent per tensor** (`unit+tensor`): 11.4 % — per-channel
  exponents are required (the conv stem's outputs span 2^−2 … 2^−13).
- **Inputs too fine** (`unit+in9`): 7.5 %.  The kernels' output shift is 8,
  so a weight is stored at f_out + 8 − f_in: every bit of input precision is
  a bit of weight precision lost.  2^−6 … 2^−7 is the balance (`in6`, `in7`,
  `opt`); a per-channel analytic balance of the two rounding errors
  (`moonshine_q.balanced`) was worse (encoder error 0.050–0.081 against
  0.045) and is not used.
- **Base's outlier channels.**  Its last encoder layers produce a few
  channels of |fc2| ≈ 2000 and |out_proj| ≈ 430 (the transformer's "massive
  activations"); at ≤ 2^−8 inputs (`unit`) they saturate (11 483 and 3 374
  values) and their weights are W × 8 — three empty transcripts and a
  repetition loop among the +29 errors.  At 2^−7 inputs (`opt`) the saturation is
  747 / 48 values and the WER float's.
- **Where the rest of the error is** (the ablation below, and the encoder
  output alone on five utterances): the conv stem — conv2's output and
  conv3's input, written at 2^−6 so that conv3's weights stay fine (its
  outputs reach ±2500, its large channels at 2^−2).  The stem in float halves
  the encoder error (tiny 0.039 → 0.029, base 0.063 → 0.035).  Splitting
  conv2 / conv3's weights into a rounded part and its remainder at 2^6 finer
  (two kernel calls, summed by the host's GELU op) with the stem's inputs at
  2^−8 … 2^−10 takes base's encoder error to 0.046 (tiny's stays 0.04) — a
  lever for the implementation, not needed for the WER.
- **Softmax P at 2^−15** instead of 2^−12: no better (+12 errors against
  +8).  **GELU on the host** instead of VectorOPKernel: the same (+2 against
  +8, within the sample).  **Float weights** (diagnostic): +3 — the weights
  are half of `unit`'s error, the activations the other half.

Ablation on Moonshine tiny (`asr_study.py ablate --ablate-base opt`): one
group of tensor classes at a time computed exactly — a kernel output
unfloored, a host or row-ops op unrounded; the consuming kernel still reads
its input at the format's exponent.  `opt`: 155 transcripts = float,
teacher-forced top-1 0.9910, encoder error 0.063.

| left exact | transcripts = float | teacher-forced top-1 | encoder rel. error |
|---|---|---|---|
| conv1, tanh, GroupNorm | 154 | 0.9907 | 0.063 |
| **conv2, GELU, conv3, GELU (the stem)** | **170** | **0.9944** | **0.040** |
| the encoder's LayerNorm outputs | 159 | 0.9913 | 0.063 |
| q0, k0, RoPE (encoder and decoder) | 158 | 0.9913 | 0.063 |
| scores, softmax | 159 | 0.9913 | 0.063 |
| v, P·V, out_proj | 157 | 0.9910 | 0.063 |
| fc1, GELU, fc2 | 157 | 0.9912 | 0.063 |
| the decoder's LayerNorm outputs | 155 | 0.9910 | 0.063 |
| cross-attention (q, K, V, scores, P·V, out_proj) | 152 | 0.9904 | 0.063 |
| SwiGLU (fc1, the bias COPYs) | 156 | 0.9910 | 0.063 |
| logits | 155 | 0.9910 | 0.063 |

Only the stem stands out, and within it conv2's output and conv3's input
(`c2`, `g2`): exact, they take the encoder error to 0.031 (tiny) / 0.035
(base) on five utterances, while conv3's output or the last GELU alone do
not help.  The implementation's lever is there: a searched exponent
for conv3's input (Piper's per-conv input search, TTS_PLAN §3) or the split
weights above.

### Cost

`asr_costs.py` builds proxy ONNX graphs with the real shapes and standard ops
(MatMul, LayerNormalization, Softmax, Gelu, Transpose), schedules them as
BERT is scheduled (linears and q·Kᵀ on ConvKernel in row parts of ≤ 1280
rows, softmax on VectorOPKernel's unit, LayerNorm and the head transposes on
the row-ops unit, GELU on VectorOPKernel's activation unit) and replays the
event stream with the performance model of bitstream `04259a7af917`.  Calls
the model has no entry or family for are priced by the analytic cost model
(`rpu_policy.estimate_us`); their share is given.  The conv stem is priced
as its im2col MatMuls, the head dims padded (48 / 64).

| encoder per bucket | 4 s (165 frames) | 8 s (332) | 16 s (665) | 30 s (1248) |
|---|---|---|---|---|
| Moonshine tiny | 29 ms | 100 ms | 196 ms | 668 ms (524 analytic) |
| + cross K / V of 6 layers | 2 ms | 5 ms | 9 ms | 17 ms |
| Moonshine base | 63 ms | 190 ms | 369 ms | 1092 ms (767 analytic) |
| + cross K / V of 8 layers | 6 ms | 11 ms | 22 ms | 44 ms |

The 30 s column is an upper bound: the proxy leaves P·V over more than 1024
keys on MatmulKernel (ConvKernel's `max_in_ch` at kw 1), priced
analytically; the frontend reads V at kw 2 on ConvKernel.  A decoder step
(self-attention over 32 tokens, cross-attention over 1248 frames): tiny
8.8 ms, base 16.8 ms — bandwidth-bound (37 / 78 MiB of weights
read per token, the tied head alone 19 / 27 MB).

Per utterance (the bucket's encoder + cross K / V + one decoder step per
token, the tokens of the float transcripts) over the 200 evaluation
utterances:

| | mean | p50 | p90 | max | mean RTF | 0–4 s (51) | 4–8 s (66) | 8–16 s (58) | 16–30 s (25) |
|---|---|---|---|---|---|---|---|---|---|
| Moonshine tiny | 466 ms | 325 ms | 1.17 s | 1.69 s | 0.054 | 144 ms | 300 ms | 581 ms | 1.30 s |
| Moonshine base | 869 ms | 620 ms | 2.07 s | 3.00 s | 0.103 | 283 ms | 573 ms | 1.10 s | 2.30 s |

The decoder is half or more of it (30 tokens × 9 / 17 ms).  Whisper
tiny.en, by the same method: ≈ 490 ms of encoder for every utterance, plus
27 tokens × 10.4 ms ≈ 0.8 s on average, 0.6 s even for a 2 s question.

### Memory

| | tiny | base |
|---|---|---|
| int16 weights (one copy: the head is the embedding) | 52 MiB | 117 MiB |
| cross K / V (30 s) + self KV cache (194 positions) | 11 + 2 MiB | 20 + 3 MiB |
| encoder intermediates, 30 s bucket (proxy pool minus weights) | ~75 MiB | ~75 MiB |
| **DMA pool** | **~140 MiB** | **~215 MiB** |
| host: the float32 embedding table (the decoder's input rows) | 36 MiB | 52 MiB |

Both fit beside the chat server's other models under `--resident auto`
(CMA ~954 MiB usable; the pools served on 2026-10-17: SmolLM2-135M 299, BERT 216,
SmolVLM 495 MiB).

### Commands and runtimes

On the maintainer's 8-core host, `.venv-export` (torch 2.14 CPU,
transformers 5.17, numpy 2.5; `ffmpeg` decodes the FLAC), the study's assets
on /mnt/data (`HF_HOME` there too: `/` is nearly full).

```bash
PY=.venv-export/bin/python
$PY demo/asr/scripts/asr_study.py fetch          # 4 checkpoints (0.8 GB) + test-clean / dev-clean (0.66 GB), SHA-256 checked
$PY demo/asr/scripts/asr_study.py select         # once: the 200 / 40 utterance ids -> asr_study_inputs.json (12 s)
$PY demo/asr/scripts/asr_study.py validate       # numpy vs transformers, 3 utterances per model (~1-3 min per model)
$PY demo/asr/scripts/asr_study.py screen         # transformers float32, 200 utterances x 4 models (16 min)
$PY demo/asr/scripts/asr_study.py window --models whisper-tiny.en   # 30 s vs 5 s buckets (24 min)
$PY demo/asr/scripts/asr_study.py study --model moonshine-tiny \
    --policies bf16,unit,opt,unit+in6,unit+in7      # calibration (~20 s) + ~4.5 min per policy, 7 workers
$PY demo/asr/scripts/asr_study.py study --model moonshine-base --policies bf16,unit,opt,unit+in7   # ~9 min per policy
$PY demo/asr/scripts/asr_study.py ablate --model moonshine-tiny --ablate-base opt   # 11 groups, ~50 min
$PY demo/asr/scripts/asr_study.py formats --model moonshine-base   # -> demo/asr/moonshine-base_opt_formats.json (§4)
inference-scheduler/.venv/bin/python demo/asr/scripts/asr_costs.py --models moonshine-tiny moonshine-base \
    --seconds 4 8 16 30 --ctx 32 128               # proxy graphs on the perf model (~1 min)
```

Results: `demo/asr/assets/study/` — `screen.json`, `window.json`,
`costs*.json`, `<model>/results.json` (every policy's transcripts, WER,
saturation), `<model>/transcripts.txt` (the ones that differ from float),
`<model>/calib.pkl` and `formats_<policy>.pkl` (the exponents).  The whole
study took about 2.5 h of compute.

## 3. Implementation plan

The pieces exist: the vision encoder's attention on ConvKernel
(`src/vit.py`, static keys), the Llama decode step with KV caches and
runtime key counts (`src/llama.py`, KV_DECODE_PLAN), the row-ops unit's
NORM / ROPE / MULACT / ADDX / COPY / TRANSPOSE / FOLD, VectorOPKernel's GELU
and softmax with a valid length (`smx_mask`), Piper's 1-D convs and length
buckets (`src/piper.py`), the chat server's speech backend
(`piper_backend.py`).  What is new is their combination and three small host
ops.

**Model: Moonshine base** (§2), the frontend written for both sizes (tiny
the same code, a smaller config).

**Phase 1 — the frontend and its specification** (done: §4; the design changed where §4 says).
- `inference-scheduler/src/moonshine.py` (route C, like `vit.py` /
  `piper.py`): config.json + safetensors + the calibrated formats →
  - `encode_<T>` entries, one per length bucket (4 / 8 / 16 / 30 s: 165 /
    332 / 665 / 1248 frames, rounded up to 16).  The audio's valid length is
    a runtime argument: the GroupNorm statistics cover the valid samples,
    the attention masks the padded keys (the softmax's `valid0`, the
    runtime key count of the attention nodes), so a bucket computes what the
    unpadded utterance computes, bit for bit.  Each entry ends with the
    cross-attention K / V of every decoder layer (persistent DMA states, in
    the q·Kᵀ / P·V image layouts);
  - a `decode` entry: one token at a runtime position, the self-attention
    KV caches as states, the cross states over the runtime frame count.
- Conv stem: the polyphase frame image of conv1 (127 taps, stride 64 →
  [t1][2 × 64]) and the im2col images of conv2 / conv3 as row-ops FOLD /
  COPY jobs, the convs as MatMuls on ConvKernel; the encoder's linears in
  row parts of ≤ 1280 rows (ConvKernel's `max_out_ch`; the 30 s bucket has
  1248), P·V over > 1024 keys at kw 2.
- Head dims padded to a multiple of 16 for the kernels and the ROPE head
  segment (tiny 36 → 48, base 52 → 64: zero rows in q / k / v, exact);
  RoPE's interleaved partial rotation as the unit's rotate-half on permuted
  head dims, the pass-through pairs with cos = 1, sin = 0
  (`moonshine_q.rope_perm`, `rope_raw_tables`).
- Host ops (`src/asr_nodes.py`): `AsrTanhNorm` (tanh table + the
  utterance's GroupNorm over the valid frames, int16 → int16), `AsrEmbed`
  (the float32 embedding row → the int32 residual), the argmax.  Everything
  else maps onto existing nodes.
- Exponents: policy `opt` — per channel from the calibration maxima, the
  kernel inputs capped at 2^−7 (conv3's input at 2^−6), the encoder's GELU
  outputs at 2^−8 for VectorOPKernel; `asr_study.py study` writes them
  (`formats_opt.pkl`), the frontend reads a formats file with provenance as
  `llm_calibrate.py` records one (checkpoint and data hashes).  Optional
  lever: conv2 / conv3 with split weights (`wsplit`, §2).
- Gate: `test_moonshine.py` — every entry's simulation equals
  `moonshine_q.QM` bit for bit on real utterances (both sizes), the bucket
  masking equals the unpadded run, host_emu builds the project.

**Phase 2 — the library.**  `demo/asr/scripts/generate_asr_project.py` →
`libmoonshine_asr.so` with a C API `asr_transcribe(pcm, n, ids, max_ids)`:
choose the bucket, run `encode_<T>`, the greedy loop over `decode` (argmax
on the host, stop at `</s>` or 6.5 tokens per second of audio).  Gates:
`asr_host_emu.py` (the library on the host emulation equals the
specification on N utterances), the R5 policy per entry (`rpu_policy.py`).

**Phase 3 — the board.**  `asr_board.py` (board lock, the chat server
stopped and restored): bit-exact against the specification on 20
utterances, the WER of all 200 on the board, latency per bucket; the
performance model topped up with the new shapes (`perf_calibrate.py`
`SHIPPED` + `cases --refine` + `run --resume` + `fit`), the host model
profiled (`host --merge`), `simulate` within ±2 %.

**Phase 4 — the chat server.**  `moonshine_backend.py` (a backend with
`transcription = True`) and `POST /v1/audio/transcriptions` in
`kv260_chat_server.py`: multipart/form-data (`file`, `model`,
`response_format` json / text, `language` en), WAV read natively (PCM16, any
rate: mono mix and a windowed-sinc resampler to 16 kHz in numpy) and the other
formats through ffmpeg where the board has it (as `/v1/audio/speech` encodes
mp3 / opus / aac / flac), the transcript decoded
by a pure-Python reader of `tokenizer.json` (pieces, `▁` → space, byte
fallback); `deploy.py` installs the library and the weights; the chat
client gets `--transcribe FILE` / `/listen FILE`.  Gates: the server tests
(`demo/chat/tests`), an end-to-end request on the board.

## 4. Phase 1: the frontend (2026-10-17)

**Done.**  `inference-scheduler/src/moonshine.py` builds the `encode_<T>` and
`decode` entries of either Moonshine size; the scheduler's simulation of them
equals the specification `demo/asr/scripts/moonshine_q.py` bit for bit, and
the generated C equals the simulation (`test/test_moonshine.py`).

### What changed against §3's design

- **Buckets: 128 / 256 / 512 / 1024 frames (3.1 / 6.2 / 12.3 / 24.6 s), not
  up to 30 s.**  ConvKernel writes q·Kᵀ keys-major, so the encoder's softmax is
  the unit's column mode, which takes at most 1024 keys (`vectorop_smx.MAX_KEYS`);
  the V caches' 4-way interleave (P·V 2.3–2.6× faster, CHAT_PLAN §16) wants
  multiples of 64 frames.  Longer audio is split (`moonshine_q.chunks`): into
  the fewest parts of ≤ 24.6 s, each boundary at the centre of the quietest
  0.1 s window within ±2 s of the equal split.  An equal split mid-word made
  greedy decoding loop ("the principles of" until the token limit, +47
  errors on one utterance); with quiet points the 4 long utterances of the
  evaluation set cost float 1 error (tiny) / 2 (base).
- **The stem: ConvKernel convs on folded rows, host ops between them.**  conv1
  is a 1 × 2 conv over the 64-phase image of the PCM (`AsrFrames` writes it),
  conv2 / conv3 1 × 7 / 1 × 3 convs over inputs folded into rows of ≤ 65 536
  accumulators (Piper's fold, `moonshine.fold_dims`), per-channel exponents
  (`chexp`).  The biases are in ConvKernel's accumulator (one saturation);
  `AsrStemNorm` (tanh from an int16 table + the GroupNorm from exact integer
  sums, in double) and `AsrStemGelu` (GELU from an int32 table at 2^−16, then
  round half even to the next conv's folded input, or for conv3 straight into
  the int32 residual — no int16 step) write the next conv's layout.  The
  tables are computed by the generator (libm's `erf` / `tanh` in Python) and
  shipped as data: nothing on the board calls libm for them.
- **A decode step's softmaxes run on the host** (LlmAttnSoftmax's p12, as
  Llama's decode): one query per head, and the unit's column blocks are 16
  wide.  The encoder's stay on the unit.
- **Scheduler extensions:** `LlmAttnSoftmax` `causal = 0` (every row over the
  keys < pos + n; the unit only, `smx_mask` period 0) and `logit_scale` (the
  true head dim's scale: Moonshine's heads are padded 36 → 48 / 52 → 64);
  `VitGelu` `approximate = "none"` (the unit's GELU erf act; the erf form has
  no host path); `HostTable` kind `i32`.  New nodes (`src/asr_nodes.py`): the
  host ops `AsrFrames`, `AsrStemNorm`, `AsrStemGelu`, `AsrFrameCount`,
  `AsrArgmax`; the row-ops nodes `AsrCrossPrep` (the cross K / V into the
  decoder's caches: one COPY for K, the V image's TRANSPOSEs), `AsrCrossQ` (a
  decode step's cross q into the q image) and `AsrSwiGLU` (fc1's halves plus
  the bias — two COPYs with a bias — then MULACT).
- **Inputs:** `encode_<T>` takes `pcm`, `n` (the valid samples) and `pos`
  (always 0: the attention's start, an input because an int32 constant
  initializer does not type-check); `decode` takes `ids`, `pos` and `xpos`
  (the valid frames − 1: the cross attention's keys = xpos + 1).

### The graph per entry (moonshine-tiny, bucket 128)

`encode_128`: 3 ConvNodes and 5 host ops for the stem; per layer 1 row-ops
attention prep (RoPE, the caches, the q image), 16 ConvKernel attention calls,
8 softmaxes on VectorOPKernel, 6 MatMuls on ConvKernel, 2 LayerNorms and 2
residual adds on the row-ops unit, 1 VectorOP GELU (the merge is in place, the
y pattern); then 6 × 2 cross K / V MatMuls and 6 `AsrCrossPrep`.  `decode`:
49 GEMVs on MatmulKernel, 192 ConvKernel attention calls (8 heads × 2
attentions × q·Kᵀ and P·V × 6 layers), 96 host softmaxes, 55 row-ops nodes
(19 LayerNorms, 18 residual adds, 6 attention preps, 6 cross q images, 6
SwiGLUs), the embedding and the argmax on the host.

### Gates

- `test_moonshine.py` (6 tests): a random Moonshine (width 80, heads 20 → 32,
  2 + 2 layers, formats calibrated by the specification on synthetic audio)
  — the encoder output over a bucket longer than the audio and the logits of
  4 decode steps equal the specification, two utterances; the graph census;
  the bucket bounds; the generated C of both entries equals the simulation
  (host_emu); and, where the assets are, the real moonshine-tiny.
- The real checkpoints on LibriSpeech (`/mnt/data/.../asr_dev/compare.py`,
  not in the repo): moonshine-tiny on a 1.9 s and a 7.0 s utterance — the
  encoder output bit-exact, all 8 / 32 greedy tokens and every step's
  logits equal ("And often has my mother said while on her lap I laid my
  head…"); moonshine-base on the same two — bit-exact, 8 / 31 tokens equal.
- The specification's WER after these changes (the 200 utterances, long ones
  split at quiet points): moonshine-tiny float 4.49 %, `opt` 4.58 % (203 errors
  against 199, 8 utterances better / 16 worse, 154 transcripts equal,
  teacher-forced top-1 0.991); moonshine-base float 3.02 %, `opt` 2.93 % (130
  against 134, 4 / 5, 180 equal, 0.996) — §2's verdict stands.  The formats:
  `demo/asr/moonshine-{tiny,base}_opt_formats.json` (`asr_study.py formats`,
  the calibration's provenance inside).

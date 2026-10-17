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

**Phase 2 — the library** (done: §5; one library per size,
`libmoonshine_tiny.so` / `libmoonshine_base.so`).  `demo/asr/scripts/generate_asr_project.py` →
`libmoonshine_asr.so` with a C API `asr_transcribe(pcm, n, ids, max_ids)`:
choose the bucket, run `encode_<T>`, the greedy loop over `decode` (argmax
on the host, stop at `</s>` or 6.5 tokens per second of audio).  Gates:
`asr_host_emu.py` (the library on the host emulation equals the
specification on N utterances), the R5 policy per entry (`rpu_policy.py`).

**Phase 3 — the board** (done: §6).  `asr_board.py` (board lock, the chat server
stopped and restored): bit-exact against the specification on 20
utterances, the WER of all 200 on the board, latency per bucket; the
performance model topped up with the new shapes (`perf_calibrate.py`
`SHIPPED` + `cases --refine` + `run --resume` + `fit`), the host model
profiled (`host --merge`), `simulate` within ±2 %.

**Phase 4 — the chat server** (done: §7).  `moonshine_backend.py` (a backend with
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

## 5. Phase 2: the library (2026-10-17)

**Done.**  `demo/asr/scripts/generate_asr_project.py` generates the project
behind `libmoonshine_<size>.so` (`libmoonshine_tiny.so`,
`libmoonshine_base.so`: one per size, so the chat server can serve both);
on the host emulation its transcripts equal the specification token for
token (moonshine-tiny and moonshine-base, four LibriSpeech utterances each —
every bucket, the longest split in two — and a random model in the
scheduler suite).

### The project

- **Entries**: `encode_128` / `_256` / `_512` / `_1024` and `decode` in one
  multi-entry project (`MultiEntryGenerator`).  The largest bucket plans the
  encoder's MatMul weight layouts (ConvKernel's kernel widths) and the
  others reuse them (`OnnxGraph(matmul_conv_kw=…)`), so the project keeps
  one copy of every weight; the decoder's weights are its own.
- **Glue**: `test/asr_glue.h` (generated): the buckets, `ASR_MAX_SAMPLES`,
  the model's `<s>` / `</s>` / positions / vocabulary, `inference_init()`
  with the four kernels and the dispatch to `inference_run_encode_<T>()`.
- **Targets**: `asr_bench` and the shared `moonshine_<size>` (only the
  `asr_*` symbols exported) over `test/asr_api.c`; `weights/tokenizer.json`
  (the checkpoint's) for the caller's detokenizer; `project.json` (buckets,
  nodes per entry, the formats and their SHA-256) and `layers.json`.

| | moonshine-tiny | moonshine-base |
|---|---:|---:|
| nodes: each `encode_<T>` / `decode` | 249 / 406 | 329 / 540 |
| DMA pool | 94.8 MiB | 185.5 MiB |
| generation | 31 s, 1.5 GB | 52 s, 2.9 GB |
| `encode_<T>`, T = 128 / 256 / 512 / 1024, predicted | 56 / 120 / 224 / 477 ms | 101 / 208 / 402 / 836 ms |
| a `decode` step, predicted | 9.8 ms | 18.6 ms |
| who drives (`rpu_policy.py`) | encode: the R5 (0.1–0.6 % overhead); decode: the A53 (97 batches, 14 %) | the same (decode 9.8 %) |

The predictions are the bitstream's performance model (`04259a7af917`):
most of the new ConvKernel shapes are priced by its family models (p90
error up to ±36 %); phase 3 measures them and tops the model up.  A
decode step's 97 batches come from its host softmaxes (each flushes the
R5's batch), so the A53 keeps the decoder.

### The C API (`demo/asr/src/asr_api.h`)

`asr_open(weights_dir)` / `asr_close()` / `asr_last_error()`;
`asr_transcribe(pcm, n, ids, max_ids)` (16 kHz int16 in, token ids out,
without `<s>` / `</s>`); the steps `asr_chunks()`, `asr_encode()`,
`asr_decode_step()` for tests and diagnostics; `asr_last_stats()` (encode /
decode / total ms, parts, tokens, steps); the constants (sample rate, the
buckets, `asr_max_samples()`, `<s>`, `</s>`, the vocabulary).

- **Parts**: `asr_chunks` is `moonshine_q.chunks` integer for integer — the
  window energies exact in int64 (a sliding sum), the first minimum, the
  same clamps.
- **Bucket**: the smallest with 384 T + 511 ≥ the part's samples; the
  samples copied into the entry's buffer, zero past n.
- **Greedy loop**: from `<s>` at position 0 until `</s>` or the limit of
  min(⌊n / 16000 × 6.5⌋ + 1, the decoder's positions) tokens, `<s>`
  included; the cross attention's `xpos` = the part's encoder frames − 1.
  Audio shorter than 0.154 s has no token and is not encoded.

**The specification** changed with it, without changing a transcript of the
study: `chunks(pcm, frames)` / `transcribe_long(…, frames)` take the
library's largest bucket (the hermetic test's library has buckets 64 / 128),
`transcribe` caps the token limit at the decoder's positions (194 for both
checkpoints, the model card's cap) and does not encode audio too short for
a token.

### Gates

- `test_moonshine.py` `TestMoonshineLibrary` (7 tests in the file now): the
  random Moonshine of §4 generated with buckets 64 / 128, the library and
  `asr_bench` compiled against the host emulation, a 1.1 s and a 7.0 s
  utterance (three parts at quiet points) — every token id equals
  `transcribe_long`, both repetitions and the re-open identical (5 s).
- `asr_host_emu.py` (the board gate without the board): `asr_bench`
  compiled with `-Werror` against the host emulation, two repetitions and a
  re-open, then `transcribe_long` with the project's formats:

  | utterance | audio | bucket | parts | tiny tokens | base tokens |
  |---|---:|---:|---:|---:|---:|
  | 5683-32879-0025 | 1.91 s | 128 | 1 | 8 | 8 |
  | 260-123440-0020 | 5.00 s | 256 | 1 | 19 | 19 |
  | 8463-294825-0003 | 9.94 s | 512 | 1 | 37 | 37 |
  | 5639-40744-0031 | 28.42 s | 1024 | 2 | 111 | 112 |

  Every id bit-exact, both repetitions and the re-open identical (tiny
  144 s, 0.7 GB; base 300 s, 1.4 GB).
- `asr_chunks` against `moonshine_q.chunks` (a one-off check through
  ctypes, not in the repo): 25 inputs — silence, noise, tone bursts and a
  full-scale square wave of one sample over the limit up to 10 min (25
  parts) — every boundary equal.
- The scheduler suite: 1880 passed; lint clean; the fact registry updated.

### Commands and runtimes

```bash
PY=inference-scheduler/.venv/bin/python
$PY demo/asr/scripts/generate_asr_project.py --model moonshine-base      # 52 s -> demo/asr/build/moonshine-base_project
$PY demo/asr/scripts/asr_host_emu.py --project demo/asr/build/moonshine-base_project   # 5 min (tiny 2.4 min)
```

## 6. Phase 3: the board (2026-10-17)

**Done.**  Both libraries on the KV260 (production bitstream `04259a7af917`):
every token id of the 200 evaluation utterances bit-exact with the
specification, so the board's WER is the specification's — moonshine-base
**2.93 %** (float 3.02 %), moonshine-tiny **4.58 %** (float 4.49 %) — at an
RTF of **0.131** (base, 1.10 s per utterance) and **0.074** (tiny, 0.63 s).

### The gate (`demo/asr/scripts/asr_board.py`)

Under the board lock, with the chat server stopped: the project uploaded
to `/root/kv260_asr/<model>_project`, the weights to
`/root/moonshine_<size>_weights`, `asr_bench` and `libmoonshine_<size>.so`
built on the board (and a profiling build), the library copied to
`/root/kv260_asr/lib/`; `asr_bench` on the 200 utterances, two repetitions
and a re-open; the specification's ids computed on the host in parallel and
cached (`<project>/reference/`, keyed by the formats and the specification's
sources); with `--profile` a profiled run over the fullest one-part
utterance of each bucket.  `asr_study.py score` (in `.venv-export`) decodes
the board's ids and computes the WER as the study does.

| | moonshine-tiny | moonshine-base |
|---|---:|---:|
| ids bit-exact with the specification | 200 / 200 | 200 / 200 |
| repetitions / re-open identical | yes / yes | yes / yes |
| WER (board = specification) | 4.58 % (203 errors) | 2.93 % (130 errors) |
| `encode_128` / `_256` / `_512` / `_1024`, median of the one-part utterances | 60 / 120 / 242 / 578 ms | 108 / 206 / 439 / 961 ms |
| a decode step, median | 10.0 ms | 18.3 ms |
| per utterance (mean of 200, 8.4 s of audio), RTF | 626 ms, 0.074 | 1098 ms, 0.131 |
| `asr_open` warm / cold | 0.22 s / — | 0.40 s / 12.5 s |
| CMA | — | 184.9 MiB (the pool: 185.5 MiB) |
| board build (`-j1`), cc1 peak | 69 s, 250 MB | 89 s, 327 MB |
| `asr_bench` (200 × 2 + re-open) | 252 s | 447 s |

- **The A53 drove every kernel**: `deploy.py --stop` stops the R5
  firmware with the server.  The libraries' policy puts the encode entries
  on the R5 when its firmware runs (phase 4, the chat server).
- **`asr_open` cold** is the weights read from the SD card (190 MB at
  ~15 MB/s, the page cache dropped); 6.0 s on the first open after the
  upload, 0.40 s warm.  The CMA readings of the other runs (220–240 MiB
  base, 128 MiB tiny) include page-cache pages in the CMA area; the one
  after a dropped cache equals the pool.
- **Against §2's projection** (base 0.87 s, RTF 0.10, from proxy graphs
  priced by the performance model): the board is 26 % slower; against the
  generated project's own predictions (§5, the model before this phase's
  top-up, every bucket full) the encode medians are −18…+1 % off (the
  largest bucket the most) and the decode step within 2 %.
- **The study's transcripts**: tiny's board transcripts equal the
  study's `opt` texts on 196 / 200 utterances (four differ by a word or a
  comma, the error count is the same), base's on 200 / 200: the study's
  tiny run predates the specification's last edit of phase 1, which the
  board and the cached references follow.

### The performance model and the simulator

Moonshine tiny / base are shipped models of `perf_calibrate.py` (`SHIPPED`,
their entries built as `generate_asr_project.py` builds them), so the
bitstream's model `kv260/04259a7af917` prices their calls exactly:

- **Cases**: `cases` 2599 (tiny 454 and base 390 new calls — the attention's
  at every runtime key count, 64 … 1024 —, the other models' calls since the
  last campaign), two refinement rounds (+223, +82) → 2904 cases, the 1059
  not measured yet run in two board sessions (`run --resume`: 2 × 38 s and
  2 × 3 s); `fit` → **2995 exact calls** (1936 before), repeat spread median
  0.085 %.  The other models' predictions are unchanged (ResNet-18,
  MobileNet v1 / v2, BERT within ±0.3 % of their board runs, as before).
- **Host ops**: `host --merge` with both libraries' profiles adds the
  kinds `AsrFrames`, `AsrStemNorm`, `AsrStemGelu`, `AsrFrameCount` and
  `AsrArgmax` (208 exact signatures, 46 kinds; 170 / 41 before).  The
  stem's host ops scale with the valid frames (`AsrStemNorm` 8.3 ms at 78
  frames, 12.3 ms at 126 in the 128 bucket), so the profiled run takes each
  bucket's fullest one-part utterance — a first profile over shorter ones
  priced the full buckets 5–13 % low.
- **`simulate`** (`asr_board.py --out` results, the new rows of
  `perf_calibrate.py`: per bucket its fullest one-part utterance, the
  attention at its frames; a decode step of the utterance the profile
  decodes, at half its steps):

  | | tiny predicted | measured | error | base predicted | measured | error |
  |---|---:|---:|---:|---:|---:|---:|
  | `encode_128` (3.1 s) | 59.5 ms | 62.7 ms | −5.2 % | 106.7 ms | 112.0 ms | −4.7 % |
  | `encode_256` (6.1 s) | 122.2 ms | 129.9 ms | −6.0 % | 209.5 ms | 220.7 ms | −5.0 % |
  | `encode_512` (12.2 s) | 268.2 ms | 273.0 ms | −1.7 % | 473.3 ms | 483.9 ms | −2.2 % |
  | `encode_1024` (24.5 s) | 724.4 ms | 726.5 ms | −0.3 % | 1169.6 ms | 1174.2 ms | −0.4 % |
  | decode step (pos ~24) | 11.9 ms | 12.1 ms | −1.8 % | 21.2 ms | 21.6 ms | −1.9 % |

  The small buckets miss the ±2 % target: their per-layer attention —
  q·Kᵀ, the softmax and P·V per head, short dependent calls on two kernels,
  and the RoPE / cache prep chains — costs 14–26 µs a call more in the
  graph than the calibration's back-to-back measurement (issue and
  wake-up), a share that shrinks with the bucket.  The profile charges the
  cross-cache prep with the ConvKernel calls it overlaps (its end is
  recorded after the next layer's MatMuls); the simulator's lanes do not.
- **The decode step depends on the audio**: 10.0 ms median over the
  200 utterances, 12.1 ms over the 48 steps of a 12.2 s one (tiny) — the
  cross attention reads the utterance's frames, the self attention its
  positions.

### Commands and runtimes

```bash
PY=inference-scheduler/.venv/bin/python
cd demo/chat && $PY deploy.py --stop && cd ../..                    # the server owns the FPGA
$PY demo/asr/scripts/asr_board.py --project demo/asr/build/moonshine-base_project --profile \
    --out /tmp/board_base.json                    # spec 5 min (cached after), build 3 min, bench 7.5 min
.venv-export/bin/python demo/asr/scripts/asr_study.py score --model moonshine-base --ids /tmp/board_base.json
cd inference-scheduler
.venv/bin/python perf_calibrate.py cases && .venv/bin/python perf_calibrate.py cases --refine   # 7 + 17-25 min
.venv/bin/python perf_calibrate.py run --config perf_config.json --resume                      # 2 x 38 s
.venv/bin/python perf_calibrate.py fit
.venv/bin/python perf_calibrate.py host --merge --profile moonshine-base=/tmp/board_base.json ...
.venv/bin/python perf_calibrate.py simulate --profile moonshine-base=/tmp/board_base.json ...
cd ../demo/chat && $PY deploy.py                  # the server back
```

## 7. Phase 4: the chat server (2026-10-17)

**Done.**  The chat server serves Moonshine base as `moonshine-base` on
OpenAI's transcription API, `POST /v1/audio/transcriptions`; deployed on the
board beside the five other models.  Through the server a 9.9 s recording
takes 1.30 s (1.24 s on the FPGA, RTF 0.124), and the transcript equals the
specification's.  The user guide is `demo/chat/doc/SPEECH_TO_TEXT.md`.

### What was built

- **`demo/chat/moonshine_backend.py`**:
  - `LibAsrEngine`, the ctypes bindings of `asr_api.h` (fact `ctypes.asr`);
  - `Detokenizer`, a pure-Python reader of Moonshine's `tokenizer.json`
    (BPE pieces, `▁` → space, byte fallback, specials skipped, one leading
    space stripped);
  - `MoonshineBackend`, which transcribes part by part (`asr_chunks`, then
    `asr_transcribe` per part, cancellable between parts) and yields a
    `Segment` per part.
- **`demo/chat/speech_input.py`**, the upload → 16 kHz mono int16:
  - WAV is read here: PCM 8–32-bit, IEEE float, `WAVE_FORMAT_EXTENSIBLE`,
    any rate and channel count;
  - channels are averaged and resampled by a polyphase Kaiser-windowed sinc
    (32 zero crossings, cut-off 0.94 of the lower Nyquist rate);
  - 16 kHz mono 16-bit passes through sample for sample;
  - other formats go through ffmpeg.
- **`kv260_chat_server.py`**, the route:
  - `multipart/form-data`, split at the boundary directly (a 25 MB upload
    in 17 ms on the host; RFC 2046 delimiter lines);
  - `response_format` `json` / `text` / `srt` / `vtt` / `verbose_json`,
    with a segment per part;
  - `stream` → `transcript.text.delta` / `transcript.text.done` events;
  - the aliases `whisper-1`, `gpt-4o-transcribe` and
    `gpt-4o-mini-transcribe`;
  - `--asr-lib`, `--asr-weights`, `--asr-tokenizer`, `--asr-model-id`,
    `--asr-cma-mb` (200) and `--asr-max-seconds` (600);
  - a 32 MB body limit when a speech-to-text model is served.
  - `chat_backend.py` gains `TranscriptionRequest`, `Segment` and the
    backend's `transcription` contract.
- **`deploy.py` and `chat_config.json.example`**: a `moonshine` block
  (`lib`, `weights_dir`, `tokenizer`, `model_id`, `cma_mb`, `max_seconds`),
  its preflight checks, `moonshine_backend.py` / `speech_input.py` and the
  tokenizer uploaded, and the server's arguments.
- **`asr_board.py --install-only`** builds and installs the library, as
  `tts_board.py --install-only` does for Piper.  The default remote
  directory is now the chat install directory, so the library lands in
  `<dir>/lib/`; phase 3 ran in `/root/kv260_asr`.
- **`chat.py`**: `--transcribe FILE` prints a transcript, `/listen FILE`
  prints one in a session; only the text is recognized, nothing goes to
  the chat model.

### What changed against §3's design

- **The library is installed by `asr_board.py`, not `deploy.py`.**  This
  matches Piper (`tts_board.py --install-only`) and the LLM libraries
  (`llm_board.py`); `deploy.py` checks for the library and uploads the
  tokenizer.
- **More of OpenAI's API than planned.**  The plan listed `json` / `text`;
  `srt`, `vtt`, `verbose_json` and streaming came almost free from the
  per-part segments.
- **The R5 drives the encoder.**  The server starts the R5 firmware, and
  the libraries' policy (§5) puts the encode entries on it.  Everything
  stays bit-exact.  The encode call returns while the R5's batch still
  runs, because the entry has no output to wait for, so `encode_ms` /
  `decode_ms` split the same total differently than with the A53 driving.

### Gates

- **`demo/chat/tests/test_transcription.py`** (22 tests; the chat suite has
  215 with the client tests below):
  - `speech_input.py` covers every WAV form, the resampler (a 1 kHz tone
    at 44.1 kHz stereo within 2·10⁻³, a 12 kHz tone at 48 kHz removed, 8 kHz
    upsampled) and ffmpeg's formats;
  - the detokenizer is checked against the `tokenizers` package on
    Moonshine's real tokenizer.json (encoded Unicode text, 30 random id
    sequences; in `.venv-export`, which has the package);
  - `MoonshineBackend` runs over a fake library;
  - the route is tested in every format, with SSE, the aliases and every
    validation error, plus a client that leaves between parts.
- **`asr_host_emu.py --lib-check`**: the chat server's backend over the
  host-built library (ctypes, the upload as WAV, the detokenizer).  On
  moonshine-tiny all four utterances are bit-exact with the specification,
  including the 28.4 s one in two parts.
- **On the board**, the server deployed with `moonshine` in
  `server.backends`, and all of these ran against it:

  | request | audio | the request | FPGA | |
  |---|---:|---:|---:|---|
  | 16 kHz WAV, the first (the model loads: 12.7 s, SmolLM2-360M evicted) | 9.9 s | 14.05 s | 1244 ms | = the specification |
  | 16 kHz WAV | 9.9 s | 1.30 s | 1235 ms | = the specification |
  | 44.1 kHz stereo WAV (resampled on the board: 1094 ms) | 9.9 s | 2.66 s | 1233 ms | the same text |
  | FLAC (ffmpeg on the board: 3091 ms) | 9.9 s | 4.39 s | 1239 ms | the same text |
  | 16 kHz WAV, `srt` / `json` / `stream` (two parts) | 28.4 s | 4.55 s | 4417 ms | = the specification; streaming: the first part after 2.1 s |

  - `chat.py --transcribe` and `/listen` worked.  (`/listen` then still
    sent the transcript to the chat model, and SmolLM2-135M answered it;
    since the fix below it only prints the text.)
  - The `openai` SDK 3.33.0, unchanged, returned `json`, a typed
    `TranscriptionVerbose` (`verbose_json`, two segments), `srt` and typed
    streaming events.
- **The fact registry** gains `ctypes.asr`, Moonshine's CMA in
  `chat.cma_mb`, and `SPEECH_TO_TEXT.md` as a consumer of
  `chat.http_routes`.

### A fix after the deploy: which model answers

`/listen` sent its transcript to the session's model, and a session started
with `--model moonshine-base` (or `piper-lessac-medium`) got a 400 from
`/v1/chat/completions`: the client could not tell chat models from the
speech ones.  And in a chat session `/listen` should only recognize the
text; it no longer sends anything to the chat model.

- **The server:** `/v1/models` now carries the non-standard
  `"kv260": {"type": "chat" | "speech" | "transcription"}`.
- **The client:** `chat.py` picks the first chat model by default.
  `--model` / `/model` naming a speech-to-text model makes it the one
  `/listen` uses, and a text-to-speech model the voice of `/say`; the
  answers to typed messages keep coming from a chat model.  `/listen` prints
  the transcript and sends nothing to the chat model.
- **Gates:** `test_chat_client.py` `ChatClientModels` /
  `ChatClientNoChatModel`, and the session replayed against the board.

### Open

- **Two parts of ~14 s each run in the 24.6 s bucket** (RTF 0.155 for the
  28.4 s recording, against 0.124 for 9.9 s).  The split is the
  specification's (`chunks`: the fewest parts); splitting at most into
  12.3 s parts would use the 512 bucket.
- **Uploads other than 16 kHz WAV cost host time before the queue**: the
  resampler 0.11 s per second of 44.1 kHz stereo audio, and ffmpeg 3 s
  (its start-up on the board).

## 8. The phrase check (2026-10-17)

**Why.**  Miranda (`demo/miranda`) wakes on "Hello Miranda" in the
transcript of each speech segment.  For one speaker's voice Moonshine wrote
the phrase as "How old, Miranda", "Oh, Mjolanda", "How old, Muganda", "A
warm Yuranda" — no fuzzy match on the text can follow it everywhere without
waking on everything.  Moonshine has no prompt: its tokenizer has `<s>`,
`</s>`, `<unk>` and 768 unused placeholders, and the decoder starts from
`<s>` alone.  But the library hands the decoder's tokens to the caller
(`asr_decode_step`), so the phrase can be fed as the first tokens and
scored: teacher forcing, how the decoder was trained.

### What was built

- **The library** (`demo/asr/src/asr_api.h`): `asr_last_logits(tokens, n,
  out)` — the last step's raw int16 logits of the tokens asked for —,
  `asr_logit_frac()` (their exponent, 8 for both models;
  `generate_asr_project.py` writes it into `asr_glue.h` as
  `ASR_LOGIT_FRAC`, from the argmax node) and `asr_max_steps(n)` (the
  decode steps `asr_transcribe` allows a part).  The generated argmax
  helper (`src/asr_nodes.py` `ASR_C`) keeps a pointer to the logits it read;
  an `asr_encode` invalidates it (the encoder's buffers may share the
  logits' pool slot).  The token ids are unchanged.
- **The check** (`demo/chat/moonshine_backend.py` `phrase_decode`):
  - the phrase's tokens from `Tokenizer.encode` (a BPE encoder for
    `tokenizer.json`, equal to the `tokenizers` library on 200+ phrases;
    the class was `Detokenizer`);
  - the margin = Σ (logit[phrase token] − logit[the step's argmax]) / 2^8
    over the steps where they differ — the log-softmax's normaliser
    cancels, so no exponential is needed;
  - punctuation the decoder picks before one of the phrase's words is
    taken as it is, at no cost (at most two in a row);
  - once the sum is below the limit it stops (`complete: false`), and the
    plain transcript continues from the first step's token: the decoder
    rewrites its self-attention cache from position 1 on, and its
    attention reads rows 0 .. pos only;
  - at or above the limit the decoder goes on greedily from the phrase.
- **The API**: the non-standard form fields `kv260_wake_phrase` (≤ 64
  characters) and `kv260_wake_margin` (≤ 0, default −10); the response's
  `kv260.wake` (`phrase`, `margin`, `complete`, `passed`, `min_margin`, or
  `error` for audio longer than one encoder pass or a library without
  `asr_last_logits`); the request log's `wake=-2.37+`
  ([Speech to text](../../demo/chat/doc/SPEECH_TO_TEXT.md#the-phrase-check)).
- **Miranda**: every segment is checked (`--wake-margin`, default −10;
  `--no-wake-check`) — while awake or in a dialogue too, so the phrase said
  again is not taken for the request; its log line shows the margin;
  `--save-segments DIR` keeps the segments with their transcripts and
  margins, to tune the limit to a voice.

### Gates

- **Host emulation** (`asr_host_emu.py --lib-check`, moonshine-tiny, 4
  utterances): the transcripts bit-exact as before, and on the three
  one-part utterances the check through the backend equal to
  `phrase_decode` over the specification (`SpecEngine`: a step at an
  earlier position drops the cache rows from it on) — kept at −1000
  (margins −18.6 / −25.1 / −27.9, the decoding from the forced phrase) and
  dropped at the default (the plain transcript after the phrase wrote the
  cache): ids and margins equal.
- **Board** (`asr_board.py`, moonshine-base regenerated): 200 / 200
  utterances bit-exact, re-open identical; installed in the chat server.
- **Tests**: `test_transcription.py` (a scripted decoder: kept, dropped at
  the first token with no extra step, dropped later with the cache
  rewritten, the phrase written by the decoder itself, the step limit,
  punctuation only between words; the fields; the response) — the chat
  suite 225; `test_miranda.py` `test_phrase_check` — 22.

### Calibration on the board

Through the chat server's API, `kv260_wake_margin=-1000` (the whole margin);
Piper's voice at speed 1.0, 0.85 with pink noise at −45 dBFS, and 1.2 at
−40 dBFS; LibriSpeech test-clean (the study's evaluation set, ≤ 24.5 s):

| Speech | Takes | Margin |
|---|---:|---|
| "Hello Miranda …" (17 sentences) | 51 | 0.00; "Hello Miranda." alone −0.44 / −0.24 in two takes |
| "Haloma Randa …", "Hallo Maranda …" | 6 | 0.00 … −3.95 |
| "Okay, hello Miranda …" | 3 | −5.2 … −6.2 |
| "Hello Veranda / Amanda / Melinda / Rwanda / Fernanda" | 15 | −3.3 … −10.6 |
| "Hi Miranda", "Hey Miranda", "Miranda, …" | 9 | −10.3 … −18.1 |
| "Hello there / everyone / my friend / Linda / Rhonda", "Hello, Mr. Anderson", "Hello, Mirabel", "Yellow veranda" | 24 | −10.8 … −21.0 |
| "How old is Miranda?", "Miranda is at home.", "I saw Miranda yesterday." | 9 | −12.5 … −29.6 |
| Piper saying the mishearings ("How old, Miranda", "Oh, Mjolanda", "How old, Muganda", "A warm Yuranda", "Harew. Hello") | 15 | −15.9 … −22.7 |
| LibriSpeech utterances | 196 | −15.5 … −41.9 (median −28.6) |

**The limit, −10**, keeps every LibriSpeech utterance out by 5.5 nats and
the greetings to someone else out, and takes the phrase in any accent the
decoder still finds within 10 nats (a factor of e^10 below its own choice,
summed over the phrase).  Names that rhyme with Miranda after "Hello" wake
her — as the transcript's name path (`wake.py`) already did.

**Cost** (20 LibriSpeech utterances, and 4 "Hello Miranda" ones, each
transcribed with and without the check): +2.7 ms on average, +33 ms at
most, for speech without the phrase (the sum usually falls below the limit
at the first token, and the plain transcript reuses that step); +0 ms for
the phrase.  The transcripts were the same (24 / 24).

### The speaker's voice

Piper saying the mishearings scores −16 … −23, so it was open whether the
speaker's "Hello Miranda" would come out near 0 or as low as ordinary
speech (LibriSpeech reaches −15.5).  The speaker said it ten times into
Miranda (Miranda's log, 2026-10-17 15:38–15:40):
- **Checked (listening):** margins −7.67, 0.00, −5.75, −2.68, −2.15 — five
  of five kept, each transcribed "Hello, Miranda."; the one segment
  without the phrase −12.38.
- **Unchecked (awake: Miranda checked only while listening then):**
  "Hello, Meganda.", "Homey granda", "How old me, Randa?", "Hello,
  Miranda." — "Homey granda" was answered as the request.  Since then every
  segment is checked, and the phrase said again while awake waits for the
  request anew (`test_miranda.py` `test_phrase_check`).

### Commands and runtimes

```bash
PY=inference-scheduler/.venv/bin/python
$PY demo/asr/scripts/generate_asr_project.py --model moonshine-base           # ~1 min (also moonshine-tiny)
$PY demo/asr/scripts/asr_host_emu.py --project demo/asr/build/moonshine-tiny_project --lib-check   # ~3 min
$PY demo/chat/deploy.py --stop
$PY demo/asr/scripts/asr_board.py --project demo/asr/build/moonshine-base_project   # 15 min (the specification
                                                                               # recomputed: its cache key changed)
$PY demo/chat/deploy.py && $PY demo/miranda/deploy.py
```

# TTS_PLAN — text to speech on the KV260

Studies of speech-synthesis models for the board: the `model-study`
skill's gates (operators → memory → numerics → speed → what the
implementation needs), host only.

## 1. Audio8 TTS Preview 0.1B study (2026-09-29)

**Verdict: NO-GO for real-time speech.**  One second of audio needs about
10–13 s on the board, and never less than 6.3 s even if the codec ran
fully in parallel with the AR model.  The floor is weight bandwidth, not
software:

- The fast AR model streams its 34.6 M weights 10 times per 46 ms frame.
- The whole AR model streams **841 MB of int16 weights per frame** —
  18.1 GB/s for real time (9.0 GB/s at int8).
- The kernels' ports deliver 2.94 GB/s, and the SOM's DDR4 peaks at
  19.2 GB/s.

Offline synthesis at ~10× below real time would be possible, but it
needs a large implementation (below).  The numerics gate was not run: the
study stops at the hard NO-GO.

### Model

[`Edge0/Audio8-TTS-Preview-0.1b`](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.1b)
@ `b476f0208438dfa791abee44d11029f055aeae04` (2026-08-25).
- **License:** Audio8 Community License v1.0.  Non-commercial use is free;
  commercial use is free below US$2M annual revenue; redistribution keeps
  the license.
- **Checkpoints:** `model.safetensors` 340 MB (bf16) and `codec.pth`
  1.35 GB (codec encoder + decoder).  Neither is needed for this study:
  only the config, the modeling code and the safetensors header are
  downloaded, with SHA-256 pinned in `demo/tts/scripts/audio8_study.py`.
- **Languages:** zh / en primary; de, es, fr, it, ja, ko experimental.
- **Voice cloning:** zero-shot, from a reference clip and its transcript.

| part | architecture | parameters |
|---|---|---|
| embeddings | text / semantic 69 633 × 512, 10 codebooks × 4096 × 512 (summed per frame), fast 4096 × 512 | 58.7 M (host tables, one row per token) |
| slow AR (1 step per frame) | Falcon-H1 hybrid: 24 layers, width 512.  Each layer runs attention (8 heads / 2 KV heads, head dim 64, RoPE θ 1e11) **in parallel with a Mamba-2 SSM** (24 heads × 32, state 64, conv 4), then a SwiGLU FFN 768; μP multipliers; head 512 → 4097 (4096 semantic tokens + EOS) | 76.5 M |
| fast AR (10 passes per frame) | 4 layers, width 512, 8 / 2 heads, SwiGLU FFN 4864, RoPE over the codebook index, KV cache of 10.  Pass 0 reads the slow hidden state; passes 1–9 emit codebooks 1–9 (codebook 0 = the semantic token) | 34.6 M |
| codec decoder | codebook lookups (8-dim) + 1×1 projections to 1024; post transformer (8 layers, dim 1024, 16 / 8 heads, FFN 1216, window 128) at 21.5 Hz; 2 × (transposed conv ×2 + ConvNeXt) to 86 Hz; DAC decoder (conv 1024 → 1536 k7, then 4 blocks of transposed conv s = 8, 8, 4, 2 and 3 dilated residual units k7 d = 1 / 3 / 9, Snake activations), tanh at 44.1 kHz | 130.1 M |

Sampling is stochastic by design: temperature 0.7, top-p 0.9, and
repetition-aware resampling of the semantic token (window 10).  At most
512 frames (23.8 s) are generated per call.

### Operators (gate 1): no blocker, many new pieces

| piece | on the board | status |
|---|---|---|
| all linear layers | GEMV on MatmulKernel (AR); MatMul on ConvKernel (prefill, codec transformer) | supported.  Fast FFN M = K = 4864 is above `gemv_max_m` / `max_k` = 4096: split in two, with a host sum for the K split |
| Mamba-2 (step and chunked prefill scan), Falcon-H1 layer wiring | host op | **new** — small per step (~50 k MACs per layer) |
| slow attention (GQA, RoPE), fast attention over ≤ 10 positions, codec window attention | host ops (`LlmAttention`-like) | variants of existing ops |
| 10-codebook embedding sum, repetition-aware sampling | host / server | **new**, small |
| 1-D convs | ConvKernel as H = 1 convs | **rewrite**: one padded output row must hold out_w · out_ch ≤ 65 536 accumulators, so time is folded into rows of w0 samples (each with its (k − 1) · dil samples of left context — a host re-layout between convs); channels split at in ≤ 1024 / out ≤ 1280 (the first conv's 1536 outputs, the first transposed conv's 1536 inputs) |
| transposed convs (kernel 2s, stride s) | ConvKernel | **rewrite**: polyphase — a kernel-2 conv with s · C_out outputs + an interleave, the same MACs |
| Snake `x + sin²(αx) / α` | host op | **new**: 78 M evaluations per second of audio |
| ConvNeXt (depthwise k7, LayerNorm, exact GELU) | ConvKernel depthwise + host ops | existing ops |
| codec encoder (voice cloning) | — | or encode the reference voices once on the PC |

### Memory (gate 2): fits, swaps with the others

- **DMA pool (int16):** about 515 MiB — slow AR 146, fast AR 66, codec
  decoder 248, slow KV cache 25 (2048 positions), codec activations
  ~30 (1 s chunks with slot reuse).  That fits `cma=1000M`, but not
  beside SmolLM2-360M (740 MiB), so under `--resident auto` it swaps.
- **Host:** the embedding tables in bf16, 112 MiB.
- **Dev PC:** generating the library takes about 7 GB (CHAT_PLAN §25:
  ~23 bytes per parameter).

### Speed (gate 4): the blocker

One slow step and one fast pass were priced from proxy graphs with the
real MatMul shapes, split at the kernel bounds.  One second of the codec's
DAC decoder was priced the same way, as H = 1 convs with time folded into
rows and polyphase transposed convs.  Both used the production bitstream's
performance model, `perf_models/kv260/caa67f49a5a3.json`.

| per 46.4 ms frame | time |
|---|---|
| slow step (76.3 M weights, 217 GEMV calls) | 51.95 ms (±1 %) |
| fast AR: 10 passes × 21.04 ms + 9 heads × 1.36 ms (343.9 M weights) | 222.7 ms |
| host ops (24 slow + 40 fast layer-steps, ~10 samplings; SmolLM2's 0.2–0.45 ms per layer-step) | 18–30 ms (estimate) |
| **AR per second of audio** | **6.30–6.56 s** |

| per second of audio: codec | time |
|---|---|
| 1×1 convs and one polyphase conv, priced (10.6 GMAC) | 0.79 s (±36 %) |
| other polyphase convs (6.9 GMAC) at the priced 14.0 GMAC/s | 0.49 s |
| 1×7 dilated convs (52.5 GMAC) — no geometry in the model's families; bounded by the priced 1×1 rate (13.4 GMAC/s) and ResNet-18's 3×3 rate (30) | 1.75–3.92 s |
| Snake, 78 M evaluations on the host (5–13 ns each, 4 threads) | 0.39–1.01 s |
| post transformer + upsampler (75 M weights once per chunk) | ~0.05 s |
| **codec per second of audio** | **3.4–6.2 s** |

- **In sequence:** 9.7–12.8 s per second of audio.
- **AR and codec overlapped** (MatmulKernel and ConvKernel in parallel):
  still ≥ 6.3 s, and they share the DDR ports.
- **Time to first audio:** a prefill of the prompt (text + reference
  transcript + ~100 reference frames).  After that each 46 ms frame
  takes ~0.3 s, so audio cannot stream.

**Why no lever reaches real time.**  The AR weight traffic is fixed by the
architecture: 81 % of it is the fast AR, re-read 10 times per frame.

- It cannot stay on chip: 34.6 M weights, against ~3 MB of BRAM + URAM.
- Frames cannot be batched: each frame's codes feed the next slow step.
- int8 weights (a bitstream change) halve the AR to ~3.2 s per second of
  audio.
- Real time would then still need 9 GB/s, 3× the current ports and half
  the DDR peak — a different memory system.  CPU-only inference (the
  vendor's ONNX INT8 build) meets the same DDR wall on the A53s.  That
  path was not measured.

### Numerics (gate 3): not run

The study stopped at speed.  If offline synthesis were wanted, these are
the risks to study:
- the Mamba-2 state recurrence — on the host in double, so low risk;
- the codec's int16 activations: the audio SNR of the DAC decoder with
  power-of-two exponents per layer, and Snake / tanh on the host;
- the sampled codebooks' distributions under int16 logits.  Temperature
  0.7 / top-p 0.9 is less sensitive than greedy decoding.

The study script would need `validate` (numpy against the torch reference)
and an emulation of the partition above.

### What an offline implementation would need

- **Frontend** (a new `src/audio8.py`): slow / fast / codec entries.
- **Host ops:** Mamba-2 step + scan, Snake, the 10-codebook embedding sum.
- **Codec rewrites:** the time fold with halos and the polyphase
  transposed convs.
- **Sampler and server:** repetition-aware sampling, and an audio
  endpoint on the chat server.
- **Voice registration:** encode the reference voices on the PC.

That is several weeks for about 10× below real time.  A TTS model whose
weight traffic per second of audio fits 2.9 GB/s — non-autoregressive, or
convolutional at tens of M parameters — would be the thing to study
instead.

### Commands (reproducible, ~12 s)

```bash
PY=inference-scheduler/.venv/bin/python
$PY demo/tts/scripts/audio8_study.py all     # fetch (SHA-256 checked) + params + costs
```

The files land in `demo/tts/assets/audio8-tts-preview-0.1b/` (untracked,
~150 KB; no weights).  `costs` prices the proxies with the local
bitstream's performance model (`--perf-model FILE` overrides).

## 2. Candidate screen: which TTS fits (2026-09-29)

**Result:** four open models fit real time on the current bitstream
(compute, not bandwidth).
- **Piper** (VITS, medium voices) is the lowest-risk first port.
- **Supertonic-3** gives the best quality and language coverage that fits.
- **TinyTTS** and **Kitten TTS nano** also fit.
- Kokoro-82M does not fit; Piper high and Kitten mini are borderline.

**Method** (`demo/tts/scripts/tts_screen.py`, ~2 min after the download):
each model's ONNX export runs once in onnxruntime on a sentence with
profiling on.  From every node's real shapes, the script counts Conv /
ConvTranspose / MatMul / Gemm MACs per second of generated audio.  None
of these models is autoregressive, so unlike Audio8 (§1) weight traffic
is not the limit — compute is.  The board estimate divides by the
ConvKernel rates of §1: 13.4 GMAC/s (priced 1-D 1×1 convs) to 30 GMAC/s
(ResNet-18).  It does not include host ops.

| model (pinned export) | params | audio | GMAC per s of audio | board, kernels | host ops beyond the scheduler's | license, languages |
|---|---|---|---|---|---|---|
| **Piper** lessac-medium (VITS + HiFi-GAN) | 15.7 M | 22.05 kHz | **2.5** (decoder 1.7, flow 0.6) | **0.08–0.19 s/s** | alignment expansion, stochastic duration flow, noise, LeakyReLU / gated tanh·sigmoid; k11 convs split (max_kw 7) | MIT; ~40 languages, many voices |
| **Kitten TTS nano** v0.8 (StyleTTS2-style) | 14.0 M | 24 kHz | 3.9 (decoder 3.2) | 0.13–0.29 s/s | LSTM, AdaIN / InstanceNorm, harmonic source (sin), iSTFT | Apache-2.0; English |
| **TinyTTS** (VITS-style, zero BERT features) | 8.1 M in 4 graphs | 44.1 kHz | 4.7 (decoder 3.4, flow 1.3) | 0.16–0.35 s/s | as Piper | Apache-2.0; English |
| **Supertonic-3** (flow matching, 5 steps) | 99 M in 4 graphs | 44.1 kHz | 11.1 (estimator 8.9 on 14 Hz latents, vocoder 2.2) | 0.37–0.83 s/s | few: LayerNorm / GELU / attention at 14–86 Hz; no host op at the audio rate (the vocoder emits 512 samples per frame) | OpenRAIL-M; 31 languages |
| Piper lessac-high | 28.3 M | 22.05 kHz | 13.6 | 0.45–1.0 s/s | as Piper | MIT |
| Kitten TTS mini v0.8 | 73.2 M | 24 kHz | 14.2 | 0.47–1.06 s/s | as Kitten nano | Apache-2.0 |
| Kokoro-82M v1.0 | 81.2 M | 24 kHz | 26.8 (decoder 25.4) | 0.9–2.0 s/s | LSTM, STFT / iSTFT, harmonic source | Apache-2.0 |

**Risks common to all:**
- **Output precision.** A waveform at Q8.8 has 1/256 resolution, about
  48 dB — too coarse.  The output and the last decoder layers need
  power-of-two exponents (`src/numeric.py`) through a frontend, as the
  Llama and ViT graphs did.  That means route C, with a numerics study
  against float (SNR / mel distance, then listening).
- **Dynamic lengths.** Durations set the audio length, so the graphs must
  split into fixed-size entries: text-length buckets, then fixed-length
  audio chunks with overlap, as the LLM prefill buckets do.
- **1-D convs** need the §1 fold (out_w · out_ch ≤ 65 536 per row) and
  polyphase transposed convs.  The 1×k geometries are not in the
  performance model yet: calibrate them before trusting the kernel times.

**Next:** a full `model-study` of Piper medium (the fewest parts, the
widest language coverage) — done, §3: GO — then Supertonic-3 if the
quality is wanted.

```bash
PY=inference-scheduler/.venv/bin/python
$PY demo/tts/scripts/tts_screen.py --dir /mnt/data/tts_screen all   # ~1.2 GB download, then the table's GMAC column
```

## 3. Piper en_US-lessac-medium study (2026-09-30)

**Verdict: GO with policy `opt`.**
- **Quality:** the int16 datapath is spectrally indistinguishable from
  float — log-mel distance 0.19 dB, against 2.64 dB between two float
  samples of the model itself.
- **Speed:** about 2–5× faster than real time.

The fixed-point policy needs two things beyond the scheduler's defaults:
calibrated power-of-two exponents per tensor, and each conv's input
written at a searched exponent.  Every tensor stays int16: no float state
on the host between ops.  The one audible risk left is a −67.8 dBFS noise
floor in pauses, about 65 dB below the speech; listening (2026-09-30)
found the quality good.

### Model

[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices)
@ `c10ece1aade47bb51c153c893d14e5bf8e5b7117`, `en/en_US/lessac/medium`.
- **Files:** ONNX 63 MB + JSON, SHA-256 pinned in
  `demo/tts/scripts/piper_study.py`.
- **License:** the voice is MIT, but its training data is the Blizzard 2013
  Lessac corpus under its own license.  Check that before shipping; Piper
  voices trained on public-domain data (LJSpeech) exist.
- **Input:** 22.05 kHz audio from espeak-ng IPA, en-us.

VITS, Piper's medium configuration:

| part | structure | params | where |
|---|---|---|---|
| phoneme embedding + text encoder | 256 × 192; 6 layers, hidden 192, 2 heads with relative attention (window 4), FFN 768 (k3), projection to mean / log-scale | 6.34 M | host, double (once per sentence) |
| stochastic duration predictor | DDS convs (depthwise k3, dilations 1 / 3 / 9, LayerNorm, GELU) + 3 inverse rational-quadratic spline flows (10 bins) | 0.56 M | host, double |
| alignment + noise | durations = ceil(exp(logw)); mean / log-scale repeated per frame; z_p = m + ε · e^logs · 0.667 | — | host |
| flow (reverse) | 4 coupling layers; each a WaveNet of 4 layers (k5 conv 192 → 384, tanh · sigmoid, 1×1 res / skip), mean-only | 7.09 M | ConvKernel + host gates |
| HiFi-GAN decoder | conv 192 → 256 k7; transposed convs ×8, ×8, ×4 (hop 256) to 128 / 64 / 32 channels; after each, 3 two-conv residual blocks (k3 d 1 / 2, k5 d 2 / 6, k7 d 3 / 12) averaged; conv 32 → 1 k7; tanh | 1.66 M | ConvKernel + host LeakyReLU / adds |

### Method

- **Float reference.** `piper_vits.py` re-implements the inference in
  float64 numpy with the ONNX export's weights (the VITS module names
  survive the export).  Checked against onnxruntime with noise off on 11
  sentences: encoder statistics within 5e-6, durations identical, waveform
  within 1e-4 (88–101 dB SNR — onnxruntime's float32).
- **Data.** Harvard sentences: list 2 (10) calibrates, list 1 (10)
  evaluates, plus one 6.9 s chat reply.  Phonemized with espeak-ng
  (`espeakng-loader` + `phonemizer-fork`) into Piper's phoneme ids.
- **Emulated partition.**
  - The text encoder, the duration predictor and the alignment run on
    the host in double.  So every policy gets the same durations and
    noise, and the waveforms align sample by sample.
  - Every flow and decoder conv runs on ConvKernel: raw int16 operands,
    the sum exact in ap_fixed<32,16>, floor(acc / 2^8) saturated.  The
    weight sits at the rank-1 exponent f_w = f_y + 8 − f_x and the bias
    at f_y.  Transposed convs compute the same MACs; the frontend makes
    them polyphase.
  - Every other op runs on the host in double, with round-half-even +
    saturate where it writes an int16 tensor.
- **Metrics** (against float with the same noise):
  - waveform SNR;
  - log-mel distance (80 mels, both floored at the reference's peak
    − 60 dB);
  - the error's level in the reference's pauses (frames below −50 dBFS);
  - saturation counts and the accumulator peak.
- **Yardsticks:** bf16 at the same boundaries, and a second float sample
  that differs only in the flow's noise — the model's own variability.

### Results (11 sentences, 29 s of speech)

| policy | SNR mean / min | log-mel distance | pause floor | saturated | accumulator peak |
|---|---|---|---|---|---|
| float, other flow noise (the model's variability) | −2.6 / −3.0 dB | 2.64 dB | — | — | — |
| bf16 | 28.2 / 20.4 dB | 0.09 dB | −84.2 dBFS | — | — |
| `q88` (the scheduler's default: every tensor at 2^-8) | 10.1 / 3.5 dB | 1.70 dB | −54.6 dBFS | 0 | 0.35 % of 2^31 |
| `pow2` (calibrated exponents, 1 bit headroom) | −0.4 / −1.5 dB | 1.29 dB | −56.8 dBFS | 0 | 0.20 % |
| **`opt`** (pow2 + searched conv-input exponents + the flow's chain in float) | **22.5 / 11.3 dB** | **0.19 dB** | **−67.5 dBFS** | 0 | 0.21 % |
| `opt`, float weights (the bound of a finer weight grid) | 27.4 / 17.1 dB | 0.16 dB | −67.9 dBFS | 0 | 0.20 % |
| `opt`, flow only / decoder only | 22.7 / 37.8 dB | 0.10 / 0.14 dB | −80.3 / −67.6 dBFS | 0 | 0.20 % |
| `opt` + the decoder's chain in float too | 22.6 / 11.3 dB | 0.18 dB | −68.4 dBFS | 0 | 0.21 % |
| **`opt`, every tensor int16 (the recommended policy)** | **23.3 / 13.3 dB** | **0.18 dB** | **−67.8 dBFS** | 0 | 0.20 % |

WAVs to listen to (float, bf16, q88, pow2, opt, opt + chain, opt with
int16 chains) are in
`demo/tts/assets/piper-lessac-medium/study/` for `eval00` (2 s) and
`eval10` (6.9 s).

### What breaks and what fixes it

- **Weights: ConvKernel's fixed output shift.**
  - Because the shift is fixed at 8, a conv's weights sit at
    f_w = f_y + 8 − f_x.
  - In the flow, inputs are finer than outputs: tanh · sigmoid gates at
    2^-14 feed a 1×1 conv whose output is at 2^-12.  With calibrated
    exponents that leaves weights at 2^-4 to 2^-7: 5–22 % RMS weight
    error, and `pow2`'s waveform decorrelates.
  - The host writes every conv input anyway, so it can write it at a
    coarser exponent: a finer weight grid for a little activation
    precision.  A per-conv search on the calibration data (the output
    MSE over input exponents) moves the flow's inputs 3–6 bits coarser,
    and the SNR goes from −0.4 to 22.5 dB.  The remaining gap to float
    weights (27.4 dB) is small.
  - An output-shift register on ConvKernel (a bitstream change) would
    close it, but is not needed.
  - Keeping the residual chains (the flow's WaveNet sums, the decoder's
    residual blocks) in float on the host is **not** needed either: with
    the searched conv-input exponents, all-int16 chains score the same
    (23.3 dB, 0.18 dB).
- **The flow's SNR undersells it.**  The flow's output is a sample
  (z_p carries noise with scale 0.667 · e^logs), and its int16 error is
  far smaller than that noise.  Flow-only `opt` is 0.10 dB from float in
  log-mel, 26× closer than another float sample.
- **Pause noise floor.**  Fixed-point steps don't scale with the signal:
  pauses get −67.5 dBFS of noise (bf16: −84).  It comes from the
  decoder's conv inputs / outputs, not its adds — keeping the residual
  chain in float moves it by 1 dB only.
- **No saturation** anywhere with one bit of headroom.  Accumulators
  peak at 0.2 % of the int32 range.

### Cost

The flow and the decoder per second of audio: proxy graphs with the real
conv shapes, split as the frontend would.  Time is folded into rows
(out_w · out_ch ≤ 65 536 per row, which over-counts ~1.4× by rounding the
rows up); transposed convs are polyphase kernel-2 convs; the k7 dilation-12
conv is split into taps spanning ≤ 64 columns.  None of these 1×k
geometries is in the performance model's families, so they are priced at
13.4–30 GMAC/s (§1).

| per second of audio | time |
|---|---|
| flow (0.91 GMAC as proxied, 40 calls) | 0.04–0.07 s |
| decoder (2.26 GMAC, 26 calls) | 0.08–0.17 s |
| host element ops (LeakyReLU, adds, averages, gates: ~21 M) | 0.04–0.11 s |
| text encoder + duration predictor (host, once per sentence) | ~0.2 GMAC in double |
| **total** | **~0.2–0.45 s (2–5× real time)** |

Time to first audio is the host front end plus the first decoder chunk.
The flow runs on the whole utterance (it is cheap at 86 Hz); the decoder
runs in chunks of a few hundred ms with a few frames of halo.  That puts
the first audio at a few hundred ms.

### Memory

- **DMA pool:** the flow + decoder weights at int16, 17.5 MB, plus a
  chunk's activations (the largest, 32 channels × 22 050 samples, is
  1.4 MB) — about 25 MiB.  It stays resident next to every other model.
- **Host:** the text encoder and the duration predictor, 28 MB float32.

### What the implementation needs

- **Frontend** `src/piper.py` (route C): the flow and a decoder chunk as
  entries (fixed frame counts, like the LLM prefill buckets).  It emits
  the exponent metadata of `opt` with int16 chains: every tensor int16,
  each conv's input exponent from the search (the host op that writes it
  uses that exponent).
- **Host code** (the chat server or C host ops): text encoder, duration
  predictor, alignment, noise; espeak-ng on the board (Ubuntu's
  `espeak-ng` package).
- **Scheduler:**
  - Conv weights encoded at the rank-1 exponent (`encode_matmul_weights`
    covers MatMul only today).
  - Conv tensors with power-of-two exponents.
  - 1-D convs as time-folded images: the host op that writes a conv input
    writes the folded layout with its halo, at no extra pass.
  - Polyphase transposed convs with an interleave; the k7 d12 conv split
    in taps.
  - LeakyReLU as a host op (or a VectorOP `act`).
- **Board time:** calibrate the 1×k conv families of the performance
  model before trusting the kernel range above.
- **Server:** an OpenAI-style `/v1/audio/speech` endpoint returning WAV.

### Commands and runtimes

```bash
PY=inference-scheduler/.venv/bin/python
python3 -m venv /mnt/data/tts_venv && PIP_CONFIG_FILE=/dev/null /mnt/data/tts_venv/bin/pip install \
    -r demo/tts/scripts/requirements-phonemize.txt
$PY demo/tts/scripts/piper_study.py fetch                         # 63 MB, SHA-256 checked
/mnt/data/tts_venv/bin/python demo/tts/scripts/piper_study.py phonemize   # 21 sentences, 2 s
$PY demo/tts/scripts/piper_study.py validate                      # vs onnxruntime, 17 s
$PY demo/tts/scripts/piper_study.py calibrate                     # exponents + conv-input search, 60 s
$PY demo/tts/scripts/piper_study.py study                         # 10 policies x 11 sentences, 3.5 min
                                                                  # (--policies "opt;opt, int16 chains": 1 min)
$PY demo/tts/scripts/piper_study.py costs                         # < 1 s
```

## 4. Piper on the KV260: implementation and board gate (2026-09-30)

**Result: done, bit-exact.**
- **Output:** the library's PCM on the board equals the specification
  sample for sample.
- **Speed:** 0.52 s per second of audio (RTF) on a 6.9 s sentence, about
  0.7 s per 1.49 s chunk.
- **Pool:** 35.5 MiB, resident next to every other model.

Policy: `opt` with every tensor int16 (§3), exponents from
`piper_study.py calibrate` (`demo/tts/assets/piper-lessac-medium/exponents.json`,
untracked like every study output; the board-gated file has SHA-256
`23ef510cfbcc90f3…`).

### The pieces

- **Scheduler: Conv with power-of-two exponents.**
  - `numeric.encode_conv_weights`: a Conv whose x / y carry scalar
    exponents gets its weight at f_w = f_y + 8 − f_x and its bias at f_y,
    both raw int16 (`wexp`).  Weight / bias exponents must not be
    annotated; per-channel exponents are rejected.
  - The simulator's `_conv_exp` computes it in integers: the exact sum,
    the int32 wrap, floor(acc / 2^8), saturation.
  - Tests: `test/test_conv_exp.py` against an independent integer loop
    (plain, dilated, depthwise, saturation) and the generated C (host_emu).
- **Host ops: `src/tts_nodes.py`**, domain `axi.llm`, numpy reference + C
  helper each.
  - `TtsPrep`: a conv input — channels, reversal, LeakyReLU, the
    utterance mask, the folded layout with halo.
  - `TtsGate`: tanh · sigmoid.
  - `TtsSum`: 2–4 inputs, / div, masked.
  - `TtsFlowOut`: the coupling update on the float32 flow state.
  - `TtsInterleave`: the polyphase output.
  - `TtsPcm`: tanh → int16 samples.
  - tanh / exp come from libm on both sides (Python `math` through
    `libm_map`).
  - Tests: `test/test_tts_ops.py` checks every branch of the C helpers
    against a numpy model.
- **Frontend: `src/piper.py`, the `chunk` entry** — one fixed-size pass
  of the flow and the decoder.
  - **Input:** z_p [192][256] frames (float32, host) plus lo / hi, the
    utterance's frames in chunk coordinates.  The mask is applied at
    every conv input and at the flow state.
  - **Output:** the decoder window, flow frames [32, 224) — 192 × 256
    int16 samples.  The frames [64, 192) are valid: 128 frames, 32 768
    samples, 1.49 s.
  - **Context:** 64 frames on each side cover the receptive field of the
    flow and the decoder.  So chunk k covers utterance frames
    [128k − 64, 128k + 192), and the stitched chunks equal one long pass
    bit for bit.
  - **Deviation from §3:** the flow runs per chunk rather than once per
    utterance.  The entry stays fixed-size, at the price of computing
    the flow twice.
  - **1-D convs as folded images:** `TtsPrep` writes
    [C][rows][w0 + halo], and the conv is valid along the row.  w0 is the
    largest power of two dividing L with w0 · roundup16(O) ≤ 65 536 (the
    accumulator).
  - **Transposed convs:** polyphase, as a kernel-3 'same' conv with s · O
    outputs plus `TtsInterleave`.
  - **The k7 dilation-12 conv** (span 73 > 64 line-buffer columns): split
    into tap groups [0, 4) and [4, 7), each floored, summed by `TtsSum`.
  - **Graph:** 173 nodes — 66 Conv, 46 TtsPrep, 37 TtsSum, 16 TtsGate,
    4 TtsFlowOut, 3 TtsInterleave, 1 TtsPcm.  9.09 M weights.
  - **Pool:** 35.5 MiB (weights 17.3 MiB + intermediates 18.2 MiB).
- **Specification: `demo/tts/scripts/piper_vits.py`**
  - `chunk_forward`: the bit-level chunk (float32 weights, int32 wrap,
    floor).
  - `synthesize_chunked`: the stitching.
  - `front_end`: the host front end.
  - `test/test_piper.py` (random weights at Piper's shapes, random
    exponents) checks:
    - the simulation equals `chunk_forward` for four mask cases;
    - the stitched chunks equal one pass;
    - the node census and the kernel bounds;
    - the generated C (host_emu) equals the simulation.
- **Library: `libpiper_tts.so`** (`demo/tts/src/tts_api.{c,h}`,
  `demo/tts/scripts/generate_tts_project.py`).
  - `tts_open` / `tts_close` / `tts_synthesize_chunk(zp, frames, k, pcm)` /
    `tts_synthesize`, over the generated `inference_run_chunk`.
  - z_p is channel-major [192][frames]; chunk k returns its
    min(128, frames − 128k) · 256 samples.
  - `tts_bench.c` is the board runner.
  - The project also carries `weights/frontend.npz` (text encoder +
    duration predictor, float32) and `weights/voice.json` (phoneme ids,
    espeak voice, noise / length scales) for the server.

### Gates

| gate | what | result |
|---|---|---|
| host emulation (`tts_host_emu.py`) | the generated inference.c + tts_api.c + tts_bench.c against the software ConvKernel, on the real voice (eval00 2.15 s, eval10 6.88 s), stitched by the library | PCM bit-exact with `synthesize_chunked`; `--incoherent` (separate CPU / DDR copies) too; re-open identical |
| library through the server's backend (`tts_host_emu.py --lib-check`) | `piper_backend.PiperBackend` over a host-built `libpiper_tts.so` (ctypes), host espeak-ng, `frontend.npz` | bit-exact on 2 texts |
| **board** (`tts_board.py --profile --reopen`, bitstream `caa67f49a5a3`) | tts_bench on eval00 + eval10, 3 repetitions | **PCM bit-exact**; repetitions identical; close / re-open identical; `tts_open` 52 ms; CMA 35.5 MB |

Board timing per chunk (1.49 s of audio):

| | first build | after the host-op rewrite |
|---|---|---|
| chunk | 985–1009 ms | 670–726 ms |
| RTF eval00 / eval10 | 0.90 / 0.72 | **0.64 / 0.52** |
| ConvKernel (66 calls) | 503 ms | 503 ms |
| TtsSum | 213 ms | 99 ms |
| TtsPrep | 188 ms | 77 ms |
| TtsGate | 47 ms | 20 ms |
| TtsInterleave | 36 ms | 4.8 ms |
| TtsPcm + TtsFlowOut | 7 ms | 7 ms |

**The host-op rewrite** (A53 in order, gcc 11 at -O2; bit-identical to
the first helpers on random data, and to the numpy models in
`test_tts_ops.py`):
- Valid ranges are hoisted: the masked and out-of-window columns are
  memset runs, with no test per element.
- Struct fields live in locals and the pointers are `restrict`, so the
  compiler stops reloading them.
- **Two-input sums** with power-of-two scales run in int64:
  Σ x_i << a_i, then round half to even by 2^m.  That is exact, so it
  equals nearbyint on the exact double.
- **The resblock average (three inputs / 3)** runs in int32: floor-divide
  by 3, then round half to even by 2^j.  A tie is exact, and any other
  value is at least 1 / (6 · 2^j) from a boundary, far beyond the
  rounding of v / 3.
- LeakyReLU is branch-free, in four independent chains.
- **Gates** read per-exponent tables of tanh and sigmoid over all 65 536
  int16 values, filled with the same libm calls.
- The interleave is parallel over channels, reading rows in order.

Single-thread A53 micro-benchmark, per op of 32 × 49 152:

| op | before | after |
|---|---|---|
| sum | 59 ms | 22 ms |
| masked sum | 68 ms | 10 ms |
| average of three | 109 ms | 47 ms |
| prep (leaky) | 55 ms | 23 ms |
| gate | 10.5 ms | 3.0 ms |

**Against the §3 projection** (0.2–0.45 s per second of audio), the
ConvKernel alone costs 0.34 s/s.  Two reasons:
- The chunking overlaps: the flow runs 256 frames and the decoder 192 for
  128 valid frames — 2× and 1.5×.
- The 1×k conv families were priced without calibration.

Levers:
- **Larger chunks:** for example 384 flow / 256 out gives overlaps of
  1.5× and 1.25×, at the price of first-audio latency and pool.
- **Calibrate the 1×k families** in the performance model.
- **The remaining ~200 ms of host ops.**

## 5. Text to speech in the chat server (2026-09-30)

**`POST /v1/audio/speech`** (OpenAI's speech API) on the KV260 chat
server, backend `piper`, model id `piper-lessac-medium`.
- **End to end on the board** (phonemes, front end, FPGA): first audio
  after 1.8–3.5 s, RTF 0.78–0.94.
- **Bit-exact** with the host running the same pipeline.

### The request

| field | |
|---|---|
| `model` | `piper-lessac-medium`, or the aliases `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`; default: the first speech backend |
| `input` | text, up to 4096 characters |
| `voice` | any name (one voice); `{"id": ...}` accepted |
| `response_format` | `wav` (default — OpenAI's default is mp3), `pcm` (s16le mono at 22 050 Hz, header `X-Sample-Rate`; OpenAI's pcm is 24 kHz), `mp3` / `opus` / `aac` / `flac` (ffmpeg on the board) |
| `speed` | 0.25–4: length_scale / speed |
| `stream_format` | `audio` (default) or `sse`: `speech.audio.delta` events (base64) and `speech.audio.done` with usage (input tokens = phoneme ids, output tokens = frames) |
| `seed` | extension: the noise seed (default 0: the same text gives the same audio) |

The audio streams chunk by chunk as the FPGA produces it.
- **wav / pcm** go out with a Content-Length: the length is known after
  the front end.
- **Encoded formats** are chunked through an ffmpeg pipe.
- **Cancellation:** a client that disconnects stops the synthesis at the
  next chunk.
- **Wrong endpoint:** chat requests to the speech model, and speech
  requests to chat models, get a 400.

### The pipeline

1. **Phonemes: `demo/chat/piper_phonemize.py`**, espeak-ng through ctypes
   (the board's libespeak-ng 1.50; no Python packages).
   - `espeak_TextToPhonemes` returns one clause at a time.  The clause's
     terminator is read from the text it consumed; espeak reads a
     character or a word ahead, which is dropped.
   - Phonemes plus terminators match the study's phonemizer-fork output
     on all 21 sentences.
   - Ids follow Piper: ^ _ (phoneme _)* $.
   - Sentences without a phoneme letter are dropped.
2. **Packing:** whole sentences into utterances of ≤ 400 ids, one
   front-end pass each; a longer sentence is split at words.
3. **Front end:** `piper_vits.front_end` in float64 numpy, outside the
   FPGA lock, with `erf_fast`.  GELU's `math.erf` element by element cost
   seconds on the A53; Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7) gives
   identical frame counts on all 21 sentences.
4. **`libpiper_tts.so`** per chunk, under the FPGA lock; the samples are
   sent as soon as each chunk is done.

**A numpy pitfall on the A53.**
- The front end first took 12.4 s for a 143-id sentence (0.12 s on the
  PC).
- `w[:, :, t] @ x` with a strided tap slice falls back to numpy's naive
  matmul loop, 145× slower than OpenBLAS there.
- Contiguous taps (`np.ascontiguousarray`) bring it to 0.78 s.

### Results

`demo/tts/scripts/tts_speech_check.py`, 3 texts.  "Bit-exact" means the
board's samples equal the host running the same backend front end
(espeak-ng on the PC) and the specification.

| audio | first audio | total | RTF | bit-exact |
|---|---|---|---|---|
| 3.38 s | 1.76 s | 3.18 s | 0.94 | yes |
| 5.89 s | 2.45 s | 4.64 s | 0.79 | yes |
| 12.48 s (2 utterances) | 3.45 s | 9.75 s | 0.78 | yes |

Before the front-end fix, first audio took 13–66 s.  Before the host-op
rewrite it took 2.0–3.6 s, at RTF 1.0–1.2.

### Deployment

- `tts_board.py --install-only` builds and installs `libpiper_tts.so` to
  `<chat dir>/lib/`, the weights and voice to `/root/piper_weights`.
- `deploy.py` does the rest when `piper` is in `server.backends`:
  - uploads `piper_backend.py`, `piper_phonemize.py` and `piper_vits.py`;
  - passes `--tts-lib` / `--tts-weights` / `--tts-cma-mb 40`;
  - preflight checks the library, `frontend.npz`, `voice.json`, numpy,
    libespeak-ng and ffmpeg.
- **Residency:** 40 MB of CMA.  With `--resident auto` Piper loads at
  startup next to SmolLM2-360M (CmaFree 250 → 215 MB).
- **CMA pitfall.**
  - After uploads and board builds, SmolLM2-360M's 776 MB pool failed to
    allocate at 834 MB CmaFree: page-cache pages sat in the CMA area.
  - `drop_caches` brought CmaFree to 1009 MB and the allocation
    succeeded.
  - `deploy.py` now drops the clean page cache before it starts the
    server.
- **Client:** `chat.py --audio` (or `/audio` in the REPL) reads the
  answers aloud.
  - It fetches raw PCM and pipes it into pw-play / paplay / aplay /
    ffplay / play as it streams.
  - `/say TEXT` speaks a text, and `/audio save FILE.wav` keeps the last
    audio.
  - A SmolLM2-360M answer of 12.7 s of speech started sounding 3.3 s after
    its text.
- **Tests:** `demo/chat/tests/test_speech.py` (the endpoint with a fake
  speech backend: formats, SSE, errors, aliases, HTTP/1.0,
  disconnects, the shared queue) and `test_piper_backend.py` (the backend
  with a fake library; the phonemizer with espeak-ng) and `test_chat_client.py` (the client's audio).  179 chat tests.
- **License:** the lessac voice's training data (Blizzard 2013) has its
  own license (§3).

### Commands

```bash
PY=inference-scheduler/.venv/bin/python
$PY demo/tts/scripts/generate_tts_project.py            # -> demo/tts/build/piper_project, 4 s
$PY demo/tts/scripts/tts_host_emu.py                    # generated C vs the spec on the host, 60 s
$PY demo/tts/scripts/tts_host_emu.py --lib-check        # the chat backend over a host-built library, 45 s
$PY demo/chat/deploy.py --stop                          # the server owns the FPGA
$PY demo/tts/scripts/tts_board.py --profile --reopen    # board gate, ~2 min (installs the library)
$PY demo/tts/scripts/tts_board.py --install-only        # or: install only
$PY demo/chat/deploy.py                                 # server with "piper" in server.backends
$PY demo/tts/scripts/tts_speech_check.py                # end to end vs the host, ~1 min
curl http://192.168.100.8:8000/v1/audio/speech -H 'Content-Type: application/json' \
     -d '{"model": "tts-1", "input": "Hello from the FPGA."}' -o hello.wav
```

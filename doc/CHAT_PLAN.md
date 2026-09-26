# Chat app on the KV260 — plan (for review)

Date: 2026-09-26.  Status: **approved 2026-09-26 — decisions: A then B,
SmolLM2-135M-Instruct, server on the board, existing CLIs + `chat.py`, context
1024.**  Phase 1 (server + CLI + backend A) **done** (branch `feat/chatsrv`, §9).
**Phase 2 done: GO for SmolLM2-135M on today's bitstream with the numeric
policy `pow2+sink+p12` (§10)** — a float residual, a precomputed position-0
sink, per-channel power-of-two exponents, and softmax P at 2^-12.
**Phase 4 server side done (branch `feat/llmsrv`, §12)** — tokenizer, chat
template, sampling, the `smollm2` backend with prefix-cache reuse and
two-model residency, tested against fakes of the §11 library; the board gate
waits for phase 3's `libsmollm2.so`.  Builds on doc/BERT_PLAN.md (BERT-base SQuAD at 971 ms per
inference on the board, bit-exact with the scheduler simulation).

## 0. The constraint that shapes everything

**BERT cannot generate text.**  bertsquad-12 is an encoder: it scores every
input token as a possible answer start / end and we pick the best span out
of the given context.  It has no vocabulary head and no next-token output, so
"implement the generation part" means one of two different things:

| | A. BERT-QA chat | B. generative chat |
|---|---|---|
| What answers | the existing BERT-SQuAD, extracting a span from a document the user supplies | a new small decoder LLM (SmolLM2 / Qwen2.5 class) generating tokens |
| Chat feel | Q&A over a document; answers are copied phrases, no free text | real chat: free-form answers, multi-turn, streaming tokens |
| New hardware | none | none required (a second weight port helps 2×) |
| New scheduler work | none (sliding windows are host-side) | decoder ops (RMSNorm, SiLU, RoPE, GQA), KV cache, prefill / decode step graphs sharing one weight pool |
| Risk | low | numerics of Q8.8 on a decoder (outlier activations) — needs a study like BERT's §0 before committing |
| Speed on the board | ~1 s per 256-token window | ~5 tokens/s (135M model), ~2 tokens/s (360M) — estimates |

**Recommendation: both, in that order.**  A is small, gives a working
OpenAI-compatible server and CLI flow within the first phase, and stays
useful as a "chat with a document" model.  B is the real generation work
and starts with a go/no-go numeric study.  The server and CLI are shared.

## 1. Decisions for you

1. **Backends:** A then B (recommended) / A only / B only.
2. **Generative model for B** (all Apache-2.0, instruction-tuned, Llama
   architecture; estimates at today's 1.5 GB/s batch-1 weight stream):

   | model | params | weights (16-bit) | est. decode | fits 1 GB CMA | notes |
   |---|---:|---:|---:|---|---|
   | **SmolLM2-135M-Instruct** (recommended start) | 135 M | 269 MB | ~5.5 tok/s | yes, with BERT loaded too | 30 layers, hidden 576, 9 heads / 3 KV, FFN 1536, vocab 49152; weak but coherent |
   | SmolLM2-360M-Instruct | 362 M | 723 MB | ~2 tok/s | yes, alone | noticeably better answers; same code as 135M |
   | Qwen2.5-0.5B-Instruct | 494 M | 988 MB | ~1.5 tok/s | no — needs a larger CMA (`cma=` boot arg) | best quality; FFN-down K = 4864 > MatmulKernel `max_k` 4096 (needs a K-split or a bitstream) |

   The code is model-agnostic within the Llama family, so switching between
   the SmolLM2 sizes is a regeneration, not new work.
3. **Where the server runs:** on the board (recommended — self-contained,
   the chat endpoint is the board's IP) / on the host PC driving the board.
4. **Clients (the CLI question, §5):** reuse existing OpenAI-compatible CLIs
   plus a tiny dependency-free script of our own (recommended) / also add
   an Ollama-API shim so `ollama run` works.
5. **Context length for B:** 1024 tokens (recommended; KV cache 24 MB for
   135M) or 2048.

## 2. Architecture

```
 laptop / board shell                               KV260 (Ubuntu 22.04, Python 3.10)
 ┌──────────────────────┐   HTTP, OpenAI protocol   ┌──────────────────────────────────────────┐
 │ llm / aichat / curl  │ ────────────────────────▶ │ kv260_chat_server.py  (stdlib only)       │
 │ openai SDK / chat.py │ ◀──── SSE token stream ── │  /v1/models  /v1/chat/completions         │
 └──────────────────────┘                           │  one request at a time (FPGA lock)        │
                                                    │  tokenizer · chat template · sampling     │
                                                    │        │ ctypes                           │
                                                    │  libbert_squad.so   libdecoder.so          │
                                                    │  (generated inference projects, shared    │
                                                    │   weight pool, cacheable BOs, host ops)   │
                                                    │        │ XRT / UIO                        │
                                                    │  ConvKernel · MatmulKernel · VectorOP      │
                                                    └──────────────────────────────────────────┘
```

The generated projects are built as shared libraries next to the existing
executables (a CMake target, no API change); the server loads them with
`ctypes`.  The board has numpy and pip, but the server itself stays stdlib
only so it runs on a fresh image.

## 3. Backends

### 3.1 A — `bert-squad` (extractive QA chat)

- **Protocol mapping.**  The document comes from the system message (or a
  user message starting `Context:`); the question is the last user message;
  the reply is the extracted span.  Later questions in the same conversation
  reuse the document.  An empty document gets a short usage hint back.
- **Long documents:** sliding 256-token windows with stride 128 (standard
  SQuAD practice), best span across windows by start + end logit;
  ~1 s per window.  A cap (default 8 windows) bounds latency.
- **Streaming:** the span arrives as one delta chunk; `finish_reason: stop`.
- **Reuse:** tokenizer, feature builder and `best_span` from
  `demo/bert_squad/scripts/bert_study.py`, the runner logic of `squad_bench.c`.

### 3.2 B — generative decoder (`smollm2-135m-instruct` first)

**B0. Numeric study — go / no-go (before any scheduler work).**  Same method
as BERT_PLAN §0: numpy interpreter of the exported graph, validated against
onnxruntime; Q8.8 emulation of the planned partition; metrics: top-1
next-token agreement with float over a prompt set, greedy continuation
match length, and perplexity on a held-out text.  Llama-family models are
known for a few very large residual activations (hundreds or more) that
would saturate Q8.8's ±128.  Candidate fixes, in cost order, all without a
bitstream: keep the residual stream in float on the host (RMSNorm and the
residual adds are host ops anyway), fold power-of-two scales into the
weights that produce the outliers and undo them in the float residual add,
and per-weight power-of-two scaling if an accumulator-shift register is
ever added.  Exit criterion: greedy answers indistinguishable in quality
from float on the prompt set (I will show you examples side by side).
**Done 2026-09-26: GO (§10).**  The reference is a numpy Llama forward from the
safetensors weights, validated against transformers / torch rather than
onnxruntime.  Plain Q8.8 fails: a 25 982 attention-sink activation and
softmax P at 1/256.  The fix is host-side only (§10.5).

**B1. Export.**  Two fixed-shape ONNX graphs from the Hugging Face model
(host-side export with torch / transformers in a separate venv):
- `prefill(ids[1,P], mask[1,P]) → logits of the last position, K/V for P positions`
  with P bucketed (64 / 128 / 256 / 512), padded and masked;
- `decode(id[1,1], pos, K_cache[L][C], V_cache[L][C], mask[1,C]) → logits, k_new[L], v_new[L]`
  with the cache as ordinary graph inputs (C = context length).  The host
  writes `k_new / v_new` into the cache at `pos` (23 KB per token for 135M),
  so the cache never moves.  The host re-rounds V to the cache exponent as it
  writes it (§10.5).
- Cache row 0 (`<|im_start|>`, the attention sink) is a precomputed constant
  and prefill starts at position 1 (§10.3).

**B2. Scheduler.**
- Host ops / fusion patterns: RMSNorm (fused float region, like LayerNorm),
  SiLU·gate (SwiGLU; 65 536-entry SiLU table, bit-identical), RoPE
  (rotate-half with float cos / sin tables), GQA key / value head repeat
  (zero-cost view or host copy), causal / padding mask construction.
- **Multi-entry projects:** one generated library with `inference_run_prefill_P()`
  and `inference_run_decode()` sharing one weight pool (weights deduplicated
  by initializer); without it every graph would carry its own 269 MB copy.
- Engine choice: decode is N = 1 everywhere → MatmulKernel (packed B,
  weight-bandwidth bound); prefill (N = P) → ConvKernel via the §2A
  lowering.  The existing cost model already decides this.
- The tied embedding is used twice: a Gather table (row-major) and the LM
  head (packed B) — two copies, +57 MB.
- **Numerics from B0 (§10.5):**
  - per-tensor / per-channel power-of-two exponents (host `ld` / `st` scale
    vectors, rank-1 weight exponents);
  - a float32 residual host tensor, with the adds fused into the RMSNorm
    regions;
  - the position-0 sink;
  - softmax P at 2^-12, with the V cache re-rounded by the host;
  - calibrated exponents from `llm_study.py formats`.

**B3. Generation loop (in the library, C):** prefill the prompt, then per
token decode → logits → sampling (greedy, temperature, top-k, top-p,
repetition penalty, seed) → stop on EOS, a stop string or `max_tokens`;
emit tokens through a callback so the server can stream them.

**B4. Tokenizer and chat template (server, Python):** SmolLM2's byte-level
BPE from `tokenizer.json` in pure Python (a few hundred lines, fast enough
for prompts of a few hundred tokens; `tokenizers` from pip is an optional
accelerator), and its ChatML template (`<|im_start|>role … <|im_end|>`).
History is trimmed from the oldest turn to fit the context.

## 4. Server

- `kv260_chat_server.py`, Python stdlib (`http.server.ThreadingHTTPServer`),
  binds `0.0.0.0:8000` (configurable), optional API key.
- **Endpoints:** `GET /v1/models`; `POST /v1/chat/completions` with
  `stream: false` (one JSON) and `stream: true` (SSE `data: {chunk}` lines
  and `data: [DONE]`), `usage` token counts, `finish_reason` stop / length;
  `GET /health`.  Optional `POST /v1/completions` for raw-prompt clients.
- **Parameters honoured:** `model`, `messages`, `stream`, `max_tokens`,
  `temperature`, `top_p`, `stop`, `seed`, `n = 1`.  Unknown fields are
  ignored (clients send many); unsupported values → OpenAI-style 400 JSON
  errors.
- **Concurrency:** one FPGA → one request at a time; others wait in a FIFO
  up to a timeout, then 503.  A disconnecting streaming client cancels
  generation at the next token.
- Started by hand or as a systemd unit; logs one line per request with
  prompt / completion tokens, time-to-first-token and tokens/s.

## 5. CLI — can we reuse an existing one?

**Yes, for OpenAI-compatible clients; not directly for Ollama's.**
- `ollama run` talks to an Ollama server's *native* API (`/api/chat`,
  `/api/show`, `/api/tags`, NDJSON streaming), not the OpenAI one; pointing
  it at our server (`OLLAMA_HOST`) works only if we also implement that
  API.  Possible as an optional ~200-line shim over the same backend; not
  recommended for the first version.
- Work unchanged against `http://<board>:8000/v1`:
  - **`llm`** (Simon Willison; `pip install llm`) — an `extra-openai-models.yaml`
    entry with `api_base`; interactive `llm chat -m kv260`.
  - **`aichat`** (single Rust binary, also for aarch64) — an
    `openai-compatible` client in its config; interactive REPL.
  - the official `openai` Python SDK (`OPENAI_BASE_URL`), `curl` — scripting
    and tests.
- **Ours:** `demo/chat/chat.py`, ~150 lines of stdlib Python — REPL with
  history, streaming, `/reset`, `/model`, `/system`, `/doc <file>` (for
  bert-squad).  Zero install, runs on the board or the laptop, doubles as the
  test client.  Recommended in addition to the above, not instead.

## 6. Phases and gates

| # | Phase | Deliverable | Gate |
|---|---|---|---|
| 1 | Server + CLI + backend A | shared-library build of the BERT project, `kv260_chat_server.py`, `chat.py`, sliding-window QA, docs with `llm` / `aichat` configs | OpenAI SDK + `llm` + `aichat` + `chat.py` against the board; streaming and non-streaming; answers identical to the demo's spans (bit-exact path); long-document QA |
| 2 | B0 numeric study | study script + report | go / no-go with side-by-side float vs Q8.8 generations — **done, GO (§10)** |
| 3 | B1–B3 decoder | export, scheduler ops, multi-entry project, generation loop; tiny random Llama fixtures in the 148-model board suite | scheduler simulation bit-exact with the study emulation; board logits bit-exact on prefill and 32 decode steps; greedy text identical to the emulation |
| 4 | B4 + server integration | tokenizer, chat template, sampling, streaming | multi-turn chat through `llm`, `aichat`, `chat.py`; tokens/s and TTFT measured |
| 5 | Performance (optional) | split packed-B streaming over HPC0 + HPC1 (~2× decode, bitstream), overlap sampling with the next step, int8 weights (large) | measured tokens/s |

Each phase lands as its own branch and board run, like BERT phases 1–2.

## 7. Expected performance (estimates, 100 MHz, today's bitstream)

| | 135M | 360M |
|---|---:|---:|
| decode, per token (weights + KV / 1.5 GB/s + calls and host ops; 135M: 270–293 MB → 180–195 ms + 391 calls, §10.6) | ~195–205 ms (~5 tok/s) | ~490 ms (2 tok/s) |
| prefill, 256-token prompt (ConvKernel, ~40 GMAC/s) | ~0.7 s | ~2 s |
| with weights split over two HPC ports (phase 5) | ~10 tok/s | ~4 tok/s |
| BERT-QA answer, one 256-token window | ~1.0 s | |

## 8. Risks

- **Q8.8 numerics on the decoder** — resolved by B0 (§10) without a bitstream.
  Residual risks:
  - calibration coverage: 433 of 4.0 G values saturate on the evaluation data;
    widen the margin or the calibration set if chats show more;
  - the size of the exponent machinery in the scheduler (§10.5).
- **Small-model quality** — 135M chats plausibly but gets facts wrong; that
  is the model, not the port.  360M is the same code.
- **Decode speed** is bound by weight bandwidth, not compute; the conv grid
  does not help it (it does help prefill).
- **Static shapes** — prefill buckets and a fixed context length; longer
  conversations are trimmed.
- **One FPGA** — requests are serialised; fine for a demo, not a service.

## 9. Phase 1 outcome (2026-09-26, branch `feat/chatsrv`)

Delivered as [`demo/chat/`](../demo/chat/) (README there: deploy, the
client recipes with transcripts, the API and backend reference).

* **Shared library.**  `demo/bert_squad/src/bert_api.{h,c}` — `bert_open(weights_dir)`,
  `bert_run(ids, seg, mask, start_logits, end_logits)` (raw int16 in, raw
  Q8.8 bits out), `bert_run_ex` (+ the pass-through unique id), `bert_close`,
  `bert_seq_len`, `bert_model_name`, `bert_weights_dir`, `bert_last_error`:
  the init / buffer / UIO-instance code `squad_bench.c` had inline, now used
  by `squad_bench` and by `libbert_squad.so`.  `generate_project.py` adds
  the `bert_squad` shared-library target (the `inference` static library
  built position-independent; `-Wl,--exclude-libs,ALL` exports only the 8
  `bert_*` symbols, so a second generated library — the decoder — can live
  in the same process without `inference_*` clashes).  No change to the
  generated inference API.  `weights_dir` must be the build's
  `INFERENCE_WEIGHTS_DIR` (the generated `_load_weight()` resolves it at
  compile time; a relative build directory is `chdir`'d into instead).
* **Text module.**  `demo/bert_squad/scripts/squad_text.py` (stdlib only):
  WordPiece tokenizer, `build_feature` (one window), `build_features`
  (sliding 256-token windows, question ≤ 64 tokens, stride 128, per-token
  max-context flags), `best_span` and `best_span_windows` (cross-window,
  n-best with softmax probabilities, answer ≤ 30 tokens), SQuAD EM / F1;
  `bert_study.py` re-imports it.  Only change in behaviour: equal logits are
  ranked by position (stable) instead of by numpy's unstable argsort —
  unchanged results on all real data checked (demo inputs byte-identical,
  60 / 60 spans over the demo's board / float / emulation logits, study
  `--policies q88,sched --n 5` and `bert_sched_check.py --n 2` identical).
* **Server** `kv260_chat_server.py` (stdlib `ThreadingHTTPServer`, HTTP/1.1
  keep-alive, chunked SSE): `/health`, `/v1/models[/{id}]`,
  `/v1/chat/completions` (stream / non-stream, `stream_options.include_usage`,
  `usage`, `finish_reason`), OpenAI-style 400 / 401 / 404 / 413 / 503
  errors, optional API key, FIFO FPGA queue with timeout and length limit,
  cancellation on client disconnect (socket peek between steps), one log
  line per request.  Backend interface `chat_backend.py`: `load / prepare
  (no FPGA) / generate (yields Delta…, Finish; checks cancel between steps)
  / health / close`.  Backend A as §3.1 with an 8-window cap; an `echo`
  backend for protocol testing without the FPGA.
* **Client** `chat.py` (stdlib, streaming, history, `/doc /system /model
  /reset`, one-shot `-q`).  **Deploy** `deploy.py`: generate if needed,
  weights by checksum, cached build, transient systemd unit `kv260-chat`,
  `/health` wait, `--status`, `--stop`, `--hold` (keeps the board lock and
  stops the server on exit).

**Gates.**
1. Host: 48 unit tests (stdlib `unittest`): chat.completion and chunk
   schema field by field, errors, queue order / timeout / overflow,
   disconnect while streaming, while computing and while queued; windows,
   max-context and cross-window spans against independent references; the
   backend's mapping, windows, cap, `max_tokens` / `stop`, cancellation, and
   a replay of the demo's board logits giving the demo's spans.
2. Board (under the lock; same bitstream as BERT phase 2): the demo with
   the refactored `squad_bench` (`deploy_and_run.py --n 12`) bit-exact with
   the simulation 3 / 3 and the emulation 12 / 12, 966.7 ms per inference,
   logits byte-identical to main's run; the server gives **the demo's spans
   for 12 / 12 questions**; **971 ms FPGA time per window**, 1027 ms per
   single-window request end to end, startup 1.6 s; long document (1048
   tokens, 8 windows) EM 90.0 / F1 94.2 over 10 questions at 7.8–8.0 s each;
   a 19-window document truncated to 8 and reported; a dropped 8-window
   request frees the FPGA after the current window.
3. Clients against the board: `chat.py` (one-shot, scripted REPL),
   `curl` (stream / non-stream / errors), `openai` 3.19.2 (models, create,
   stream with usage, 404 / 400 exceptions), `llm` 0.36 (`-m`, `--no-stream`,
   `-c`, `llm chat`), `aichat` 0.30.0 (one-shot, `-S`, `Context:`, REPL with
   `.prompt`) — all work unchanged; transcripts in the README.

**For phases 3–4.**
* *Memory.*  The BERT server process holds ~221 MB of CMA (CmaFree 811 →
  590 MB); idle CmaFree on the board was 626–813 MB of 1000 MB today (1013 MB
  after a fresh boot in BERT phase 1; the display stack and earlier jobs
  take some).  A 135M decoder pool (~270 MB of
  weights + the embedding's second copy + KV cache + activations) fits
  beside BERT only when CmaFree is at the high end — plan to load one
  backend at a time or check CmaFree at `load()`; 360M needs BERT unloaded.
* *Two libraries in one process* are safe symbol-wise (only `bert_*` /
  the decoder's own exports are visible); each opens its own XRT device
  handle and pool BO and maps the same UIO devices; the server's single FIFO
  lock serialises them — keep all kernel work under it.
* *Streaming* works token by token already (`Delta` per yield; `StopStream`
  for stop strings); TTFT and tok/s are in the log line.  The decoder's
  `prepare()` should apply the chat template and tokenize outside the lock.
* *Lock etiquette.*  The running server owns the FPGA; `deploy.py --hold`
  ties the host board lock to the server's lifetime, otherwise stop it
  before other board jobs.

**Open issues.**  Answers keep the punctuation of the document's
whitespace words (SQuAD convention, same as the demo: "February 7, 2016,");
`FIFO` means order of reaching the queue after host preparation (a request
with a long, not yet cached document can be overtaken by a short one);
`temperature` / `top_p` / `seed` are ignored by the extractive backend; no
`/v1/completions` and no Ollama shim.

## 10. B0 results — numeric study (2026-09-26)

**Verdict: GO, with policy `pow2+sink+p12`, on today's bitstream.**  Plain
Q8.8 (the BERT partition) is unusable for this model: top-1 agreement is
10 % and the output is word salad.  Three host-side changes fix it, with no
kernel or bitstream change:

1. a float residual stream;
2. a precomputed position-0 attention sink;
3. power-of-two exponents per tensor and per channel, which put softmax P at 2^-12.

With those, the Q8.8 datapath agrees with float about as closely as bf16
inference does.  Top-1 is 97.6 % on the prompt set and 96.9 % on held-out
text; bf16 gets 98.4 % / 98.0 %.  Perplexity is 15.616, against 15.607 for
bf16 and 15.598 for float.  The greedy answers read like the float model's;
see §10.4 and the 12 full side-by-sides in `generations.txt`.

Script: `demo/chat/scripts/llm_study.py` (`.venv-export`; `validate`, `study`,
`ablate`, `formats`, `costs`; semantics in its docstring).  Assets and results
stay untracked under `demo/chat/assets/` (`study/results.json`,
`study/generations.txt`, `study/formats_pow2+sink+p12.json`).

### 10.1 Method

- **Float reference.**  A numpy float64 Llama forward, written from the
  safetensors weights (bf16 upcast exactly): RMSNorm, RoPE θ = 100 000 with
  rotate-half, GQA with 9 / 3 heads, SwiGLU, tied LM head, KV cache.  Checked
  against transformers 5.17 / torch float32 (eager attention):
  - logits max |diff| 7e-5 on the prompts and 1.15e-4 on a 1024-token window;
  - greedy 128-token generations identical on 12/12 prompts;
  - our ChatML formatter + `tokenizers` gives the same ids as HF
    `apply_chat_template` on 15/15 conversations.
- **Data.**
  - *Chat prompts.*  12 prompts through SmolLM2's ChatML template (default
    system prompt unless given): 2 short factual, instruction, poem,
    summarise-a-paragraph, arithmetic, word problem, code, explanation,
    translation, custom system prompt, 3-turn chat.
  - *Prompt set.*  Each prompt followed by float's greedy answer,
    teacher-forced: 1 639 positions, 990 of them in answers.
  - *Held-out text.*  WikiText-2 test, 3 windows of `<|im_start|>` + 1 023
    tokens (the context length): 3 069 positions.
  - *Calibration (separate data).*  WikiText-2 validation (1 023 tokens) plus 3
    other prompts.
  - Position 0 is excluded from every metric.
- **Emulation.**  Every DDR tensor is int16 with a power-of-two exponent f
  (value = raw · 2^-f; Q8.8 is f = 8).  f is either a scalar or a vector
  over the last-axis channels.
  - *Kernels are unchanged.*  Raw operands, exact products summed into
    ap_fixed<32,16> (the int32 wrap is emulated), `floor(acc / 2^8)` + saturate.
    So f_out = f_in + f_w − 8, and every weight is encoded, round-half-even +
    saturate, at `f_w[i][j] = f_out[j] + 8 − f_in[i]`.
  - *Host regions.*  RMSNorm (+ residual add), RoPE, softmax (+ 1/8 scale,
    causal mask), SiLU·up and the V-cache write compute in double.  They read
    `raw · 2^-f[c]` and write `round_half_even(v · 2^f[c])`, saturated.
    Softmax exp and SiLU are 65 536-entry libm tables per exponent (built from
    the same double arithmetic, so bit-identical).
- **Exponents.**  Chosen from the float calibration maxima with one bit of
  headroom (`make_formats`).
- **Yardstick.**  `bf16` rounds every tensor at the same boundaries to bf16
  and keeps the residual in bf16 (as torch bf16 inference does); the math is
  float.  It shows how much disagreement a numerically benign format already
  causes.  Even bf16 reproduces float's greedy text exactly on only 5 of 12
  prompts, because near-tie tokens flip and the text then diverges.  So the
  criterion is answer quality, not exact match.

### 10.2 Results

Top-1 = argmax equal to float's; KL = mean KL(float ‖ policy) per position;
match = leading tokens identical to float's greedy answer (of ≤ 128).

| policy | top-1 prompt set (all / answers) | top-1 held-out | top-5 held-out | KL held-out | KL prompt set | ppl held-out | identical answers | mean match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float (numpy f64) | – | – | – | – | – | 15.598 | – | – |
| bf16 (yardstick) | 0.984 / 0.981 | 0.980 | 1.000 | 0.0007 | 0.0010 | 15.607 | 5/12 | 28.6 |
| `q88` (BERT partition: Q8.8, VectorOP residual adds) | 0.100 / 0.108 | 0.062 | 0.143 | 7.26 | 7.74 | 22 856 | 0/12 | 0 |
| `res_float` (float residual, all Q8.8) | 0.170 / 0.208 | 0.073 | 0.198 | 8.09 | 6.66 | 61 014 | 0/12 | 0 |
| `res_float+sink` | 0.734 / 0.747 | 0.666 | 0.913 | 0.757 | 0.617 | 32.90 | 0/12 | 1.1 |
| `fit+sink` (Q8.8 unless the range needs coarser, per channel) | 0.962 / 0.964 | 0.885 | 0.995 | 0.061 | 0.0071 | 16.47 | 3/12 | 20.8 |
| `pow2+sink` | 0.966 / 0.968 | 0.890 | 0.995 | 0.056 | 0.0043 | 16.29 | 1/12 | 25.1 |
| **`pow2+sink+p12` (recommended)** | **0.976 / 0.980** | **0.969** | **1.000** | **0.0024** | **0.0021** | **15.616** | **2/12** | **15.9** |
| `pow2+sink+hattn` (float host attention) | 0.977 / 0.982 | 0.970 | 1.000 | 0.0022 | 0.0025 | 15.648 | 4/12 | 32.6 |
| `pow2+p12` (no sink) | 0.964 / 0.964 | 0.956 | 1.000 | 0.0065 | 0.0063 | 15.72 | 2/12 | 8.4 |
| `pow2_tensor+sink+p12` (per-tensor exponents) | 0.929 / 0.940 | 0.908 | 0.998 | 0.025 | 0.027 | 16.04 | 2/12 | 15.9 |
| `pow2+sink+p12+emb_q88` (Gather from a Q8.8 table) | 0.974 / 0.979 | 0.967 | 1.000 | 0.0024 | 0.0024 | 15.621 | 4/12 | 26.1 |
| `pow2+sink+p12+fw` (diagnostic: float weights) | 0.983 / 0.988 | 0.987 | 1.000 | 0.0007 | 0.0007 | 15.595 | 3/12 | 34.1 |

`fit` / `pow2` differ only in where the kernel outputs' exponents go.
`fit` caps them at 8.  `pow2` gives every kernel output that the host
reads (q, k, v, o, gate, up, down, logits) the finest exponent its range
allows, and the weights inherit those bits.  In both, host-written inputs
(RMSNorm outputs, silu·up, K / V cache) stay ≤ 8.  `p12` writes P at 2^-12.
To keep P·V inside int16, the host re-rounds the V cache when it writes it
(2^-8 on 91 % of the channels, 2^-7 / 2^-6 on the rest).  Matmul accumulators peak at 295 in
Q16.16 units (the wrap is at 32 768).  Under the recommended policy 433 of
4.0 G quantised values saturate: o / up / gate / down / q / P·V channels that exceeded
their calibration maximum by more than the one-bit headroom.

### 10.3 What breaks Q8.8, and what fixes it

- **Massive activation at the attention sink.**  Position 0 (`<|im_start|>`,
  the first token of every chat) builds a residual value of **25 982**
  (channel 507) in layer 11's MLP.  There, silu·up reaches 3 164 and
  down_proj 25 937; the residual stays above 10 000 through layer 29.
  Saturating these kernel outputs at ±128 destroys the sink.  Every later
  attention layer depends on it, which is why even a float residual (`res_float`) fails.  Two
  observations lead to the fix:
  - position 0's K / V rows depend only on that one token, so they are
    constants of the application;
  - K / V are all that later positions see of position 0.
  **Sink:** precompute those rows in float, write them once into the cache
  (23 KB), and start prefill at position 1.  The massive activation then
  never exists on the datapath.  Without the sink, per-channel exponents
  cope, but at 2.7× the KL (`pow2+p12`).
- **Other outliers, excluding position 0.**  The residual reaches 1 442.
  down_proj reaches 994 (layer 29, and 692 in layer 2); o_proj 204; silu·up
  185; raw q·k 723.  Per-channel exponents handle all of them: down_proj's
  outlier channels get 2^-3 … 2^-7, and q is lowered per head (2^-5 … 2^-7)
  so that q·k fits.  Per-tensor exponents cost 10× the KL.  The residual
  stream itself must be float.
- **Softmax P is the dominant error once the ranges fit.**  At 1/256
  resolution, a 1024-key row loses every probability below 0.002.  Per-class
  ablation of `fit+sink` (one class quantised at a time, float weights;
  first held-out window + prompt set; `llm_study.py ablate --base fit+sink`):

  | quantised (everything else float) | KL held-out | KL prompt set |
  |---|---:|---:|
  | everything | 0.0579 | 0.0071 |
  | weights only | 0.0030 | 0.0028 |
  | activations only | 0.0567 | 0.0045 |
  | **only P** | **0.0558** | 0.0018 |
  | only o / P·V / V | 0.0004 / 0.0003 / 0.0002 | 0.0005 / 0.0004 / 0.0002 |
  | only x2 / silu·up / down / gate+up | 0.0003 / 0.0001 / 0.0001 / 0.0001 | 0.0004 / 0.0001 / 0.0001 / 0.0001 |
  | only x / q0,k0 / RoPE q,k / scores / final norm / logits | ≤ 0.0001 | ≤ 0.0001 |

- **The fixed `>> 8` couples exponents.**  f_P + f_Vcache − 8 = f_PV, and
  f_PV + f_Wo − 8 = f_o.  So bits given to P come out of V or of W_o unless
  o's exponent can rise.  Under `fit` (outputs capped at 8), P at 2^-11 /
  2^-12 / 2^-13 improved the held-out KL only to 0.011 / 0.025 / 0.084.  The
  prompt-set KL got *worse* (0.014 / 0.047 / 0.12, first-window run),
  because W_o lost the bits.  `pow2` lets the outputs use
  their range (o at 2^-10 … 2^-12), so P can have 12 bits.  Host float
  attention (`hattn`) avoids the question and scores the same.
- **What remains is weight precision.**  With float weights KL falls from
  0.0024 to 0.0007, the bf16 level.  That is the most an accumulator-shift
  register (a bitstream change) could buy.  It is not needed.
- **No effect (first-window / quick runs):**
  - adding ½ LSB to compensate the kernels' floor bias (`+ff`: KL
    0.0579 → 0.0560 under `fit+sink`, 0.0033 → 0.0034 under
    `pow2+sink+p12`);
  - a Q8.8 Gather table instead of bf16 (`+emb_q88`, full run above);
  - lowering the cap on host-written exponents to 7 or 6 (better on one set,
    worse on the other).

Value ranges (max |x| before quantisation, all data; the policy column
excludes position 0 through the sink; "≥ 128" is the fraction that would
saturate plain Q8.8):

| tensor | float max (layer) | ≥ 128 | `pow2+sink+p12` max | exponents used (share of channels) |
|---|---:|---:|---:|---|
| residual h | 25 982 (L11) | 0.107 % | 1 442 | float32 host tensor |
| RMSNorm out x / x2 | 11.8 / 32.0 | 0 | 11.8 / 8.6 | 8 |
| q = x·Wq, k = x·Wk | 24.6 / 25.6 | 0 | 24.7 / 25.7 | 9–15 (mostly 11–12) |
| RoPE q / k (K cache) | 24.5 / 25.5 | 0 | 24.6 / 25.6 | q 5–8 per head, k 8 |
| scores q·k (before 1/8) | 717 (L18) | 1.4 % | 723 | f_q + f_k − 8 = 5–8 per head |
| P | 1.0 | 0 | 1.0 | 12 |
| v (V cache) | 14.8 | 0 | 14.8 | v out 10–15; cache 8 (7 / 6 on 9 %) |
| P·V | 12.6 | 0 | 12.5 | 12 (11 / 10 on 9 %) |
| o_proj out | 202 (L24) | 0.006 % | 204 | 8–15 (mostly 10–12) |
| gate / up | 50.6 / 78.5 | 0 | 37.9 / 40.5 | 10–13 |
| silu·up | 3 164 (L11) | 0.0001 % | 185 | 8 (7 on 9 channels) |
| down_proj out | 25 937 (L11) | 0.016 % | 981 | 3–13 (mostly 9–11) |
| final RMSNorm out | 52.0 | 0 | 51.8 | 8 |
| logits | 46.3 | 0 | 45.6 | 9 |

### 10.4 Side by side (float → `pow2+sink+p12`; all 12 in `generations.txt`)

*Code* — same function, different wording after it:
```
float:  Here's a Python function that checks whether a number is prime:
        def is_prime(n): if n < 2: return False / for i in range(2, int(n**0.5) + 1): if n % i == 0: return False / return True
        This function works by checking divisibility from 2 to the square root of the number. If the number is
        divisible by any of these values, it's not prime. If it's not divisible by any of these values, it's prime.
q8.8:   Here is a Python function that checks whether a number is prime:
        (identical code)
        This function works by checking if the input number `n` is divisible by any number from 2 to the square
        root of `n`. If `n` is divisible by any of these numbers, it is not prime. If `n` is not divisible by any
```
*Factual* — same answer, then keeps going (float stops):
```
float:  The capital of France is Paris.
q8.8:   The capital of France is Paris. It is a city known for its historical landmarks, cultural institutions, and
        cultural attractions. Paris is a major global city, and it is the political, economic, and cultural center of France.
```
*Summarise* (169-token prompt) — a different, equally valid summary:
```
float:  The honeybee colony is a complex, interconnected system of workers, drones, and queens that work together to
        maintain the colony's food source.
q8.8:   The paragraph summarizes the characteristics of a honeybee colony, including the diverse workforce of workers,
        sterile female bees, and the unique dance that communicates the location of food sources.
```
*Instruction* — the same three tips, reworded:
```
float:  1. Create a Dedicated Study Space: Choose a quiet and comfortable place where you can focus without distractions. …
        2. Set Clear Goals: Before each study session, set specific, measurable, achievable, relevant, and time-bound (SMART) goals …
        3. Take Regular Breaks: Regular breaks are crucial for maintaining focus and preventing burnout. Try to take short breaks …
q8.8:   1. Create a Dedicated Study Space: Find a quiet and comfortable place to study where you can focus without distractions. …
        2. Set Clear Goals: Before starting your study sessions, set specific, measurable, achievable, relevant, and time-bound (SMART) goals …
        3. Take Regular Breaks: Regular breaks are crucial for maintaining focus and preventing burnout. Try to take short breaks …
```
The arithmetic ("17 + 25 = 42.") and custom-system-prompt answers are
identical.  The word problem is wrong in both, as in float: the 135M model
cannot do it.  For contrast, the failing policies on "What is the capital of
France?":

- `q88`: " even at us / even at them / the none them one them one …"
- `res_float`: " I. No. no. no. Rep. No. other other …"
- `res_float+sink`: "Paris, the City of Light, is the capital of the Kingdom of the Netherlands."

### 10.5 What phase 3 (B1–B3) must implement for `pow2+sink+p12`

The emulation is the spec.  The phase-3 gate "scheduler simulation bit-exact
with the study emulation" refers to this policy.
`llm_study.py formats` writes its exponents and the sink rows to
`formats_pow2+sink+p12.json` (422 entries `class@layer`, per-channel lists;
sink K / V as raw int16).

1. **Exponents per tensor.**  An int, or an int vector over the last-axis
   channels (per head for q, k, P), in place of the implicit Q8.8.
   - Host ops read `raw · 2^-f[c]` and write
     `round_half_even(v · 2^f[c])`, saturated: a per-channel scale vector in
     `host_ld` / `host_st`.
   - The kernels and their `>> 8` are unchanged.
   - Kernel-to-kernel tensors (P·V → o_proj) carry the derived exponent
     `f_P + f_Vcache − 8`.
2. **Weight encoding with a rank-1 exponent**,
   `f_w[i][j] = f_out[j] + 8 − f_in[i]`: round-half-even, saturate (none
   saturate).  The tied embedding is encoded twice: a Q8.8 Gather table
   (row-major; no measurable loss, §10.2) and an LM head at 2^-9 (packed B).
3. **Float residual stream.**  A new float32 host tensor type (not in the
   CMA pool).  Every residual Add fuses into the next RMSNorm region
   (`h = float32(h + delta)`, then RMSNorm), and the last into the final norm.
   RMSNorm is `ss` left to right, `r = 1 / sqrt(ss/576 + 1e-5)`,
   `y = (h·r)·γ`, with γ as float32.
4. **Position-0 sink.**  Cache row 0 of every layer is a constant (float run
   of `<|im_start|>`, rounded at the cache exponents), written at init.
   Prefill buckets start at position 1, and the chat template must always
   begin with `<|im_start|>` (it does).
5. **Attention.**
   - q·Kᵀ per head, or per KV group if the 3 heads share a q exponent (taking
     the group minimum costs nothing measurable: quantising q, k has KL
     0.0000).
   - Softmax host region: `k = raw_max − raw`, `e = exp(−k·2^-f_s·0.125)`
     from a 65 536-entry table per score exponent, sum left to right, P
     written at 2^-12.
   - P·V on the kernel.
   - The host writes K (after RoPE) and V into the cache, re-rounding V to
     the cache exponent.  In decode the host copies `k_new` / `v_new` anyway;
     in prefill this is a host op over P × 192 values per layer.
   - Alternative: decode attention as one float host region (`hattn`,
     numerically equivalent; see §10.6).
6. **Other host regions.**  RoPE with float32 cos / sin tables (formulas in
   the script docstring).  SiLU·up with a 65 536-entry silu table per gate
   exponent (gate exponents 8–14: at most 7 tables, 3.5 MiB of doubles, or
   compute with libm).
7. **Calibration.**  The exponents come from a float run over calibration data
   with one bit of headroom.  Phase 3 can consume the JSON, or port
   `make_formats` (70 lines).  Its map is `class@layer` to ONNX tensors
   and weights.
8. **Export (B1).**  Two constraints on the graphs:
   - they must expose RMSNorm, RoPE, SiLU·gate, the residual adds and the KV
     cache writes as fusable patterns;
   - the Gemms must be separable per layer and class, so that exponents can
     be attached.

   The study's `Model` class (float forward + emulation, with KV cache) is a
   complete numeric spec of the Llama block.  A direct safetensors + config.json
   frontend for the scheduler may be less work than ONNX export plus pattern
   fusion; decide at the start of phase 3.  The HF repo also ships
   `onnx/model.onnx` (transformers.js, dynamic `past_key_values`), which is
   not usable as-is.  `.venv-export` (torch 2.14 CPU, transformers 5.17,
   1.2 GB) is in place.  torch's default dynamo exporter would also need
   `onnxscript`.

### 10.6 Cost refresh (`llm_study.py costs`; feeds §7)

| | value |
|---|---|
| parameters | 134.5 M: layers 106.2 M (3.54 M / layer) + tied embedding 28.3 M |
| weights | 269 MB, + 57 MB second copy of the embedding |
| KV cache | 22.5 KiB per token → 22.5 MiB at 1024 |
| decode MACs per token | 137 M (ctx 64) … 170 M (ctx 1024); attention is 1.2 M per layer at 1024 |
| decode bytes per token | 270 … 293 MB → **180 … 195 ms at 1.5 GB/s** before call and host-op overhead |
| kernel calls per token | 391: per layer q, k, v, 3 × q·Kᵀ, 3 × P·V (per KV group), o, gate, up, down; + LM head |
| host ops per token | 152 |
| prefill, P = 64 / 128 / 256 / 512 | 7.0 / 14.2 / 29.5 / 63.5 GMAC → 0.17 / 0.35 / 0.74 / 1.6 s at 40 GMAC/s (ConvKernel) |

**Per-call overhead.**  §7's "~5 ms of calls and host ops" assumed far fewer
calls than 391.  At an assumed 10–20 µs per call (not measured), 391 calls
alone take 4–8 ms.  Fusing
q / k / v and gate / up (same input, concatenated weights, per-column
exponents already supported) saves 90 calls per token.

**Decode attention on the host (`hattn`).**  Numerically equivalent, as
§10.2 shows, and it removes the 180 attention calls.  The cost is about
35 M MAC per token on the A53s at full context (~9 ms on 4 cores at an
assumed 4 GMAC/s), plus reading the 22.5 MiB cache from cacheable memory.  The kernel path streams
the same cache at 1.5 GB/s (15 ms).  Measure both in phase 3.

**Runtimes.**  On 8 host cores the study takes 42 min for the full set
(12 policies), `ablate` 12.5 min, `validate` 2 min and `formats` 22 s.
Disk: `.venv-export` 1.2 GB and assets 272 MB, both untracked.

## 11. Phases 3 and 4 — split and the library contract (2026-09-26)

Phase 3 (decoder on the FPGA: frontend, exponent machinery, host ops, KV
cache, multi-entry project, `libsmollm2.so`) and phase 4 (server side:
tokenizer, chat template, sampling, `smollm2` backend) run in parallel
against this C API.  Phase 3 owns the library; phase 4 owns everything above
it and tests against a fake with the same interface.

```c
/* libsmollm2.so — every int function returns >= 0 on success, < 0 on error */
int         llm_open(const char *weights_dir);   /* weights, sink rows, KV cache, pool BO */
void        llm_close(void);
const char *llm_last_error(void);
int         llm_vocab_size(void);                /* 49152 */
int         llm_context_size(void);              /* 1024 (cache positions incl. the sink) */
int         llm_position(void);                  /* positions filled, incl. the sink at 0 */
int         llm_truncate(int n);                 /* keep positions [0, n), n >= 1 (1 = sink only) */
int         llm_prefill(const int32_t *tokens, int n, float *logits);
                  /* append n >= 1 tokens (the library splits over its prefill buckets);
                     writes the next-token logits after the last one (vocab floats) */
int         llm_decode(int32_t token, float *logits);
                  /* append one token; next-token logits */
```

Position 0 is always `<|im_start|>` (the precomputed sink, §10.5 item 4):
callers pass the conversation's token ids **after** that leading token, and
reuse the cache across turns by `llm_truncate` to the common prefix and
prefilling only the new tokens.  Logits are the LM head's output converted
from its fixed-point exponent to float.  Sampling is not in this library.

## 12. Phase 4 outcome — server side (2026-09-26, branch `feat/llmsrv`)

Everything above `libsmollm2.so`, built and tested on the host against fakes
of the §11 contract; no FPGA used (the board only for CPU timing).  Files in
[`demo/chat/`](../demo/chat/) (README section "Generative chat"):

* **`smollm2_tokenizer.py`** (stdlib, ~300 lines + Unicode tables): the
  `tokenizer.json` pipeline — special tokens matched first (never split),
  `Digits(individual_digits)`, GPT-2's ByteLevel regex with Oniguruma's
  `\s` (White_Space) and embedded Unicode 15.0 `\p{L}` / `\p{N}` tables (the
  board's Python 3.10 has Unicode 13; the tables make it split like the
  host), byte-level alphabet (21 byte symbols missing from the vocabulary are
  dropped, as `tokenizers` does without an unk token), BPE by merge rank with
  a word cache; `decode` = UTF-8 with U+FFFD like `from_utf8_lossy`;
  `IncrementalDecoder` holds back incomplete UTF-8 for streaming.
* **`chatml.py`**: the template exactly as `apply_chat_template` renders it
  (default system prompt when the first message is not `system`); prompts are
  tokenized block by block with a cache (every block starts with the special
  `<|im_start|>`, so ids concatenate) — a follow-up turn tokenizes only its
  new blocks; `fit()` trims the history (§3.2 B4): drop the oldest messages,
  an assistant message left at the front goes with them, the system message
  and the last message always stay, `ContextTooLong` if they alone do not
  fit.
* **`src/sampler.{h,c}` → `libsampler.so`, `sampler.py`**: penalties
  (repetition HF-style, presence / frequency OpenAI-style, over the last
  `repeat_last_n` tokens of prompt + answer) → greedy (temperature 0 or
  top-k 1: first argmax) → temperature → top-k (heap) → softmax → top-p (exact
  nucleus via a provably safe prefilter `e ≥ (1 − p)·Z / V` and a sort of the
  survivors) → draw with splitmix64.  All in double in a fixed order, so the
  pure-Python fallback returns the same token for the same seed.
* **`smollm2_backend.py`**: `LibLlmEngine` (ctypes, exactly §11) and
  `Smollm2Backend`.  `prepare()` (no FPGA): `developer` → `system`, template,
  trim to `context − min(max_tokens, --llm-reserve 256)`, sampling parameters
  (request, extras `top_k` / `repetition_penalty` / `repeat_last_n`, server
  defaults temperature 0.2 / top-p 0.9 / top-k 50), a random seed if none.
  `generate()` (FPGA lock): **prefix-cache reuse** — the token list in the KV
  cache is kept; `llm_truncate(1 + common prefix)` and `llm_prefill` of the
  rest only (at least the last token, for its logits; the `<|im_start|>` sink
  is never passed); then sample → detokenize → stop strings → `Delta`, and
  `llm_decode` of the token unless the answer is done.  Ends at `<|im_end|>` /
  `<|endoftext|>` / `<|im_start|>`, a stop string, `max_tokens` or a full
  context (`length`).  `cancel.check()` before every library call; the cache
  list is updated only after a call succeeds, cleared (full prefill next
  time) if one fails.  `Finish.info`: cached / prefilled tokens, prefill ms,
  TTFT, decode tok/s, library and sampler ms, finish detail, seed, settings.
* **Server**: `--backend smollm2` and `--llm-*` options; **residency**
  (`--resident auto` default / `one` / `all`) for two FPGA models in the
  tight CMA pool: backends declare `cma_mb` (BERT 224, SmolLM2 360 —
  estimate); `auto` loads the first backend at startup and the others when
  CmaFree allows or on first request, evicting least-recently-used models
  while CmaFree < need + 32 MB and once more if a load still fails (not on a
  dlopen failure); `one` swaps.  Loads / evictions happen under the FIFO FPGA
  lock.  Backend interface: `load_host()` (tokenizer; at startup, so
  `prepare()` works for unloaded models), `load()`, `unload()`.  `/health`
  reports `loaded`, `resident`, `cma_free_mb`, `loads` / `unloads`.
  A backend that fails to load lazily answers 503 `model_not_loaded` and is
  retried on the next request.
* **deploy.py**: uploads the new modules, `src/sampler.[ch]` and
  `tokenizer.json`; builds `lib/libsampler.so` on the board (skipped when
  unchanged); passes `server.resident` and the `smollm2` block (`lib`,
  `weights_dir`, sampling defaults, `cma_mb`); the preflight checks for
  `libsmollm2.so` and a C compiler.
* **Fakes**: `tests/fake_llm.py` — `ScriptedEngine` (fast, scripted logits,
  call log, error / delay injection) and `FloatModelEngine` (`llm_study.py`'s
  float64 model with its KV cache — exact, ~10 tok/s on the host);
  `tests/fake_libsmollm2.c` — the §11 C API with scripted logits, driven
  through the real ctypes binding.  `kv260_chat_server.py --llm-fake
  float|scripted` serves them.

**Gates.**
1. *Bit-identical text side.*  `scripts/validate_text.py` against
   transformers 5.17 `TokenizersBackend` (the `tokenizer.json` pipeline):
   **2960 / 2960** diverse strings (held-out text, 25 scripts, emoji / ZWJ,
   combining marks, number forms, code, whitespace and control characters,
   special tokens in text, random code points of all planes) identical in
   ids, decode with and without special tokens and incremental decode;
   2000 / 2000 random id sequences decode identically; **68 / 68** template
   renderings (34 conversations × generation prompt on / off) identical in
   text and ids.  Finding: transformers **5.x `AutoTokenizer`** returns a
   `GPT2Tokenizer` that **drops the `Digits` pre-tokenizer** of
   `tokenizer.json` — it disagrees on 249 strings, every one explained by
   that (a numeral after ≥ 2 whitespace characters, non-ASCII numerals).
   The model was trained with the `tokenizer.json` pipeline (transformers
   4.x gives it), so that is the reference; the B0 study used `tokenizers`
   directly and is unaffected.
2. *Unit tests* (stdlib `unittest`, 59 new, **107 total**, ~40 s):
   tokenizer / template against a 500-string + 34-conversation fixture,
   trimming rules; sampler (greedy = argmax, penalties, top-k / top-p masks
   against independent references, frequencies over 20 000 seeded draws
   within 5σ, determinism, C == Python on 54 cases); backend against both
   fakes — streaming schema, multi-turn reuse (**exactly the new tokens
   prefilled**, the sink never), repeat of the same prompt (1 token
   prefilled), edited history (truncated to the common prefix), stop strings
   across tokens, `max_tokens` → `length`, cancellation mid-generation (the
   cache stays consistent and is reused), context full, trimming, UTF-8
   across tokens, seeds, library errors (500 / SSE error, cache cleared),
   client disconnect; residency `one` / `auto` (fits both / evicts by CmaFree
   / retries after a failed load) / `all` between `bert-squad` and `smollm2`
   with fake engines and a fake CMA pool, 503 and recovery.  The 48 phase-1
   tests pass unchanged.
3. *End to end* (`scripts/e2e_check.py`): `chat.py` → server
   (`--llm-fake float`) → **9 / 9 answers identical to transformers
   `generate(do_sample=False)`** (6 one-shot incl. a system prompt, code and
   an emoji prompt; a 3-turn REPL chat whose turns 2 and 3 prefilled 17 and
   23 tokens, reusing 135 / 152 and 247 / 270 positions — the re-tokenized
   answers matched the generated ids).

**Measured on the board's A53** (CPU only, Python 3.10): tokenizer load
1.45 s; 996-token prompt 39 ms cold / 13 ms warm; template + trim of a
14-message chat 69 ms, next turn 2.7 ms; detokenizer 6 µs / token;
`libsampler.so` 0.8 ms greedy, 1.9 ms with the defaults (top-k 50, top-p 0.9),
6.7 ms top-p alone, 5.1 ms plain temperature (the pure-Python fallback
25–205 ms).  So ~2 ms of host work per ~200 ms decode step; no tokenizer
accelerator needed.

**What integration with phase 3's `libsmollm2.so` needs.**
* The library at `<dir>/lib/libsmollm2.so` on the board (or `smollm2.lib`
  in `chat_config.json`), built by the decoder project; `weights_dir` in the
  config if `llm_open(NULL)` does not find the weights by itself.  Export only
  the `llm_*` symbols (like `bert_*`, §9) — both libraries live in one
  process.
* `llm_open` must work again after `llm_close` (eviction under `--resident
  auto` / `one`), and `llm_close` must return the CMA (pool BO, KV cache).
* Calls come from the server's handler threads — one at a time (FIFO lock)
  but not always the same OS thread.
* `llm_vocab_size()` must be 49 152 (checked at load) and
  `llm_context_size()` is used for trimming once loaded (`--llm-context` only
  until then).  Logits as float in token-id order; `llm_position()` must stay
  consistent after an error (or the server just re-truncates to 1).
* Measure the loaded model's CMA and set `smollm2.cma_mb` (360 is the §10.6
  estimate); with BERT at ~224 MB both fit only at the high end of idle
  CmaFree, so `auto` will usually swap.
* Board gate of phase 4 then: `deploy.py` with `["bert-squad", "smollm2"]`,
  multi-turn chats through `llm`, `aichat`, `chat.py`; measure tokens/s and
  TTFT; compare greedy board answers with the emulation's
  (`llm_study.py`'s `pow2+sink+p12` policy) — they should be identical if
  the library is bit-exact with it.

**Open issues.**
* Nothing measured on the FPGA yet (TTFT, tok/s, load time, CMA).
* `--llm-prefill-chunk` (split long prefills so a disconnect cancels between
  chunks) is off by default: every extra call costs an LM head (~40 ms).
* Special-token text inside user messages is parsed as special tokens (as
  transformers does); a user can inject `<|im_end|>` / `<|im_start|>`.
* The server's default sampling (temperature 0.2) is not OpenAI's 1.0; clients
  that send their own temperature are unaffected.
* The pure-Python sampler fallback is too slow for sampling on the A53
  (25–205 ms per token); deploy builds the C one.

## 13. Phase 3 outcome — SmolLM2-135M on the FPGA kernels (2026-09-26, branch `feat/llmdec`)

**Result.**  `libsmollm2.so` implements the §11 C API.  The scheduler
simulation of SmolLM2-135M-Instruct equals the study emulation of the
shipped policy **bit for bit**, and so does the generated C on the host
emulation.  On the board:
- the logits of 3 chat prompts (prefill + 32 greedy decode steps) are
  bit-exact with the simulation, and the greedy text equals the study
  emulation's;
- decode takes 203 ms / token (4.9 tokens / s), bound by weight bandwidth;
- prefilling 16 / 64 / 256 tokens takes 0.37 / 0.60 / 3.6 s;
- `llm_open` takes 1.0 s warm and 36 s cold;
- the model occupies 461 MiB of CMA.

The library closes and reopens cleanly.  The full board suite passes
151 / 151.

### 13.1 Decisions

* **Frontend: direct safetensors + config.json → fixed-shape ONNX
  (`inference-scheduler/src/llama.py`), not a torch export.**  An exported
  graph would have to be pattern-matched back into the structure the study
  already specifies: RMSNorm as `Pow / ReduceMean / Add / Sqrt / Div / Mul`,
  RoPE as `Slice / Neg / Concat / Mul / Add` over cos / sin, GQA as
  `Expand / Reshape`, masks as `Where`, and the KV cache as dynamic
  `past_key_values` concats.  It would also need `onnxscript`, a
  fixed-shape re-export per bucket, and exponents attached to fused nodes
  by name matching.  The frontend writes what the scheduler runs instead:
  one standard `MatMul` per linear (per layer and class, weights shared by
  name across entries, so the existing engine choice, packing and
  MatMul-on-ConvKernel lowering apply unchanged) and host ops of a custom
  domain `axi.llm` for the float regions.  Exponents, host tensors and
  states ride in the model's `axi.numeric` metadata.  Nothing beyond
  config.json is model specific; SmolLM2-360M needs `llm_study.py formats`
  and a regeneration.
* **Shipped numeric policy: `pow2+sink+p12+xattn`** (new in
  `llm_study.py`).  These are the p12 formats of §10.5 with attention as
  one float host region (the `hattn` idea) whose operation order is fixed
  so that C reproduces it bit for bit:
  - RoPE(q) stays in double;
  - scores are `dot8` (8 lane sums over d ascending, combined
    ((0+1)+(2+3))+((4+5)+(6+7))) × 1/√HD;
  - `exp` comes from libm; sums run left to right;
  - P·V accumulates from the first product;
  - P is never quantised, and pv is written at f_p + f_vc − 8.

  One region per layer replaces 6 kernel calls, 2 host ops and their cache
  hand-offs.  §13.4 measures the FPGA alternative.
* **Residual stream, ids and KV cache in host memory.**  h is a float32 host
  tensor, ids / pos / n are int32 host inputs, and the KV cache is an int16
  host state `[C][KV·HD]` per layer (22.5 MiB, not CMA).  No syncs are
  needed, since only host ops touch them.  Row 0 is the sink.
* **Embedding.**  A bf16 host table (57 MB of normal memory, exact — the
  policy's `emb = float`); the LM head is the second copy, packed for
  MatmulKernel at the logits exponents.
* **Entries.**  `decode` (1 token → logits), `prefill_16 / 64 / 256`
  (padded rows, n valid; leaves the last valid row in the `h_last` state)
  and `head` (h_last → logits).  llm_prefill() splits n tokens into the
  least-cost sequence of bucket calls, using the board cost per call (§13.4:
  a padded 64-row call costs 364 ms and a 16-row call 304 ms, because the
  MatMuls stream the weights once per call).  The last call is padded, and
  the logits do not depend on the split.  llm_prefill() runs the head once,
  so several calls may precede a decode.  Context 1024 including the sink.
* **Engines.**  decode and head are N = 1 → MatmulKernel with packed B
  (211 calls per token).  prefill → ConvKernel through the §2A lowering
  (cost model: 210 of 210 MatMuls, at every bucket).  **The two need
  different weight layouts** (MatmulKernel's packed tile-major B, ConvKernel's
  kw image), and no single layout serves both.  A conv batch over the packed
  32-column tiles is weight-request-latency bound, like BERT's P·V; conv
  decode in the standard orientation costs 1.5× by the cost model and
  fetches weight slabs row by row.  So the multi-entry project keeps **two
  copies of the layer weights**: the decode copy (212 MB) and one conv copy
  shared by all three buckets (kernel widths pinned by the largest bucket,
  212 MB), plus the LM head (57 MB).  `--prefill-engine matmul` keeps one
  copy, but the cost model puts the prefill MatMuls on MatmulKernel at
  0.63 / 2.5 / 9.9 s for 16 / 64 / 256 rows.  On ConvKernel it predicts
  0.24 / 0.26 / 0.70 s, and the board measures 0.24 / 0.26 / 0.67 s.

### 13.2 Scheduler features (doc/INFERENCE_SCHEDULER.md)

- **Numerics beyond the element type** (`src/numeric.py`):
  - per-tensor / per-channel power-of-two exponents;
  - rank-1 weight encoding f_w = f_out + 8 − f_in;
  - the simulator's MatMul exponent path (exact sums, int32 wrap, floor at f_out);
  - host tensors (f32 / i32 / i16) in a liveness-coloured host arena;
  - states: persistent, shared across entries, initialised from their non-zero prefix.
- **Host ops** (`src/llm_nodes.py`):
  - LlmEmbed, LlmResAdd, LlmRMSNorm, LlmAttention (RoPE + KV write with V
    re-rounding + causal GQA xattn, threaded over rows × heads);
  - LlmSiluMul (silu tables per gate exponent, filled at init by the same
    libm code), LlmSelectRow, LlmDequant.
- **Multi-entry projects** (`src/codegen/multi.py`, CLI `--entry`):
  - weights deduplicated by name + image; states shared;
  - intermediates / host arena / staging overlapping;
  - global layer indices for the profiler;
  - driver release in deinit, so the library can be closed and reopened.
- **Compact codegen** for LLM / multi-entry projects: pool views and
  runtime items come from descriptor tables with loops, and run functions
  are split into `noinline` parts.  The straight-line form made gcc's
  register allocator need 2.6 GB for SmolLM2's inference.c and hung the
  board (§13.4); the compact form needs 0.18 GB.
- **Byte-identical** for every existing model: 179 / 179 projects equal
  main's, file by file (the 168 test models and the 11 demo models:
  BERT-SQuAD, ResNet-18, MobileNet v1 / v2 and 7 MNIST models).

### 13.3 Gates

1. **Scheduler tests.**  1474 pass (5 opt-in / HLS skips).  New:
   - `test_numeric.py`;
   - `test_llm_ops.py`, incl. every op on the host emulation and the silu
     tables C vs Python on all 65 536 inputs;
   - `test_llama.py`: a tiny random Llama (hidden 64, 2 layers, GQA 4/2,
     vocab 256, context 32) whose simulation equals the study emulation bit
     for bit over one-call / split / padded prefills, decode, truncate and
     re-prefill, plus every entry and a 4-entry project on the host
     emulation (-Werror, cached / staged, 1 / 3 / 4 threads, re-open).
2. **Bit-exactness, SmolLM2-135M** (`demo/chat/scripts/llm_sched_check.py`):
   the scheduler simulation equals the study emulation of
   pow2+sink+p12+xattn on the logits of every step:

   | prompt | tokens (incl. sink) | prefill calls: rows / bucket | decode steps | result |
   |---|---:|---|---:|---|
   | factual | 37 | 36 / 64 | 32 | bit-exact |
   | summarise | 169 | 168 / 256 | 32 | bit-exact |
   | multi-turn | 97 | 64 / 64 + 32 / 64 | 32 | bit-exact |

   An earlier split (16 + 16 + 4 / 16; 64 + 64 + 16 + 16 + 8 / 16;
   64 + 16 + 16) was bit-exact too: the logits do not depend on the split.
   The generated project on the host emulation (`llm_host_emu.py`:
   inference.c + llm_api.c + llm_bench.c against the software kernels)
   equals the simulation on all 3 × 33 logits vectors, its greedy tokens
   equal the study emulation's, and a close / re-open reproduces the
   logits.  Weights saturated: 0.
3. **Board** (`demo/chat/scripts/llm_board.py --profile --reopen`; KV260,
   hw_128 bitstream, 100 MHz):
   - the 3 tiny Llama fixtures pass in the board suite, and the full suite
     passes 151 / 151 (the 148 existing models + 3);
   - SmolLM2's logits are bit-exact with the simulation on all 3 × 33
     vectors (prefill + 32 greedy decode steps per prompt), and the greedy
     tokens equal the study emulation's;
   - the library through ctypes (`llm_lib_check.py`) behaves as required:
     - it exports only `llm_*`;
     - `llm_vocab_size()` is 49152 and `llm_context_size()` is 1024;
     - chunked prefill calls give the same logits as one call;
     - calls from different threads give the same logits as one thread;
     - close → open reproduces the logits.

### 13.4 Board numbers

**Decode: 203 ms / token (4.9 tokens / s)** at positions 37–201.  It is
bound by weight bandwidth:

| decode, per token | ms | |
|---|---:|---|
| MatMul linear (210 calls, MatmulKernel, packed B) | 145.3 | 212 MB of weights at 1.46 GB/s, 91 % of the 128-bit port |
| LM head (1 call) | 38.8 | 57 MB, also 1.46 GB/s |
| attention (host xattn, 30 layers) | 6.7 | 37–200 cached keys |
| SiLU·up | 4.7 | |
| RMSNorm | 1.4 | |
| residual add | 0.9 | |
| other host (embedding, row select, logits dequant) | 0.8 | |

**Prefill**: one call with n = bucket tokens, in ms:

| | 16 | 64 | 256 |
|---|---:|---:|---:|
| **total** | **366** | **604** | **3588** |
| MatMul linear (ConvKernel) | 237 | 257 | 670 |
| attention (host) | 27 | 201 | 2600 |
| SiLU·up | 39 | 45 | 163 |
| RMSNorm | 19 | 41 | 87 |
| residual add | 10 | 24 | 54 |
| LM head (once) | 39 | 39 | 39 |

The llm_prefill() cost table (§13.1) is these totals minus the attention
(which covers only the valid rows) and the head: 304 / 364 / 930 ms, from
the previous profile run (within 2 % of this one).  With the least-cost
split, the chat prompts of 36 / 168 / 96 tokens reach their first logits
in **497 / 2164 / 1196 ms**.  The first split (the largest bucket the tokens
fill, the rest padded into the smallest) took 1042 / 2883 / 1442 ms.

**llm_open**: 1.0 s with the weights in the page cache, 35.6 s cold (538 MB
from the SD card).  close → open: 0.95 s, identical logits.

**Memory**:
- CMA: one pool BO of 461.3 MiB — weights 459.0 MiB (the decode copy, the
  conv copy and the LM head) and intermediates 2.3 MiB.  CmaFree drops by
  440–486 MiB on open.  The spread is page cache, which Linux keeps in
  movable CMA pages and migrates out when the BO is allocated.  After
  `llm_close`, CmaFree comes back within a few MiB of its value before
  `llm_open`.  **`cma_mb` for the server: 480.**
- Normal memory: host tables 54.2 MiB (the bf16 embedding, RoPE), KV cache
  states 22.5 MiB, host arena 1.1 MiB.

**Board build and the hang.**  The first board session built the
straight-line inference.c (3.7 MB) with `make -j4`.  cc1 grew to 3.0 GB,
MemAvailable fell to 232 MB (the board has no swap), the board hung, and
it had to be power-cycled.  On the host, aarch64 gcc 13 at -O1 peaked at
2.59 GB, 2.19 GB of it in the register allocator.  The cause was
straight-line init code that took the addresses of the statics the run
functions read.  The compact codegen (§13.2) cuts the peak to 124 MB (-O1)
/ 196 MB (-O2) on the host.  On the board, inference.c is now 3.1 MB and
the -O2 build takes 42 s with a cc1 peak of 178 MB.
`llm_board.py` builds with `-j1` and compiles inference.c once for both
targets.  It starts only with ≥ 1.5 GB MemAvailable, and a guard kills
make if MemAvailable drops below 400 MB or the memory PSI (some avg10)
exceeds 50.

**Decode attention: FPGA p12 vs host xattn, measured.**  The alternative,
`pow2+sink+p12`, works per layer and KV group:
- q·Kᵀ on MatmulKernel against a tile-packed transposed K cache;
- a host softmax quantised to p12;
- P·V on the row-major V cache.

`llm_attn_kernel_bench.py` timed its kernel calls on the board:

| cached keys | FPGA kernel calls / token | FPGA incl. host softmax + ~360 cache syncs (est.) | host xattn / token |
|---:|---:|---:|---:|
| 64 | 3.1 ms | ~9 ms | ~4 ms |
| 256 | 8.1 ms | ~14 ms | ~16 ms |
| 512 | 14.1 ms | ~20 ms | ~32 ms |
| 1024 | 25.9 ms | ~32 ms | ~65 ms |

Host xattn was measured at 1.69 ms / layer with 800 keys (51 ms / token)
and at 6.7 ms / token over 37–200 keys; the other rows scale linearly with
the keys.  The crossover is about 250 keys.  **Host xattn ships.**  It is
the faster path below ~250 keys, where every conversation starts.  At full
context the FPGA path would save ~33 ms of ~260 ms per token (13 %).  In
return it needs:
- a second numeric policy (P quantised to p12);
- K / V caches in CMA, including a transposed K written at every step;
- an emulation that switches policy at the same length.

§13.6 keeps it as an option.  The host attention itself was made 3× faster
without changing its output (items balanced over the threads, prefill K / V
rows converted once, lane-wise vectorisation).

### 13.5 Building and deploying libsmollm2.so (for phase 4)

```bash
# assets (not in git, see llm_study.py): demo/chat/assets/smollm2-135m-instruct
# and demo/chat/assets/study/formats_pow2+sink+p12.json (llm_study.py formats)
inference-scheduler/.venv/bin/python demo/chat/scripts/generate_llm_project.py
#   -> demo/chat/build/llm_project (~100 s; 538 MB weights/*.dat, 3.1 MB inference.c)
inference-scheduler/.venv/bin/python demo/chat/deploy.py --stop   # the server owns the FPGA
inference-scheduler/.venv/bin/python demo/chat/scripts/llm_board.py --install-only
#   the full gate instead: llm_sched_check.py --save gate2.json, then
#   llm_board.py --profile --reopen --study-json gate2.json
```

`llm_board.py` takes the SSH settings and driver directories from
`demo/bert_squad/bert_squad_config.json` and holds the board lock.  It
uploads the sources, syncs only the changed weight files, builds on the
board under the memory guard and installs the library.  On the board:

| path | contents |
|---|---|
| `/root/kv260_chat/lib/libsmollm2.so` | the library (the server's default path); exports `llm_*` only |
| `/root/smollm2_weights/weights/*.dat` | 424 files, 538 MB; `INFERENCE_WEIGHTS_DIR` of the build, so `llm_open(NULL)` uses `/root/smollm2_weights` (`llm_weights_dir()`) |
| `/root/kv260_chat/llm_project` | the generated sources; `build/` (library + `llm_bench`), `build_prof/` (profiler) |

`llm_open(dir)` takes the directory that holds `weights/`.  The library
keeps no thread-local state, so calls may come from any thread as long as
they are serialised.  The standalone benchmark is `llm_bench [-w dir]
[-i prompts.bin] [-o logits.bin] [-k 32] [-P 16,64,256] [-R reps] [-r]`,
where `-r` adds a close / re-open.  The `build_prof/` build also prints
`PROFILE_PHASE` / `LAYERS_JSON` lines.

Notes for the server (§12):
- `smollm2.cma_mb`: 480.  With BERT (~224 MB) the two need ~705 MB of the
  1000 MiB CMA; the idle CmaFree was 905–1014 MiB in this session.
- The greedy reference is the study emulation of **`pow2+sink+p12+xattn`**
  (`llm_study.py`), not `pow2+sink+p12`: the attention is the C-exact
  float region (§13.1).
- Every `llm_prefill` runs the LM head once (39 ms), so `--llm-prefill-chunk`
  costs that per extra chunk.  Otherwise a chunk costs what the least-cost
  split of its tokens costs.
- Errors (not open, context full, token out of range, bad arguments) are
  detected before any state changes, so `llm_position()` is unchanged after
  a failed call.

### 13.6 Open issues

- **Prefill attention** is 72 % of a 256-token prefill (2.6 of 3.6 s).
  Running q·Kᵀ / P·V for the prefill rows on the FPGA under a mixed policy
  (p12 in prefill, xattn in decode, the emulation following suit) should
  bring the 256-token prefill to ~1.1 s.  The host loop is also well below
  the A53's peak (~1.2 µs per head and query-key pair per thread).
- **Decode** is at the 128-bit port's weight bandwidth (1.46 GB/s).  The
  way forward is phase 5's dual-port / HPC1 weight streaming.  Fusing
  q / k / v and gate / up into single MatMuls would remove 90 of the 211
  calls per token and their fixed overhead.
- **Decode attention at long context**: the FPGA path above ~250 keys
  (§13.4), up to −13 % per token at 1024.
- **Two weight copies** (459 MiB of CMA).  A layout that both engines can
  read, or `--prefill-engine matmul` (one copy, prefill MatMuls 0.63 / 2.5 /
  9.9 s by the cost model), would free ~210 MiB.
- **Cold `llm_open`** reads 538 MB from the SD card (35.6 s).  The server
  should open once at start-up.
- CmaFree after close sits a few MiB below its value before open.  This is
  page cache in CMA pageblocks, not a leak: repeated cycles do not
  accumulate beyond that noise.

## 14. Integration on the board (2026-09-26, main dba302e)

`demo/chat/deploy.py` with `server.backends = ["smollm2", "bert-squad"]`,
`resident: auto`, `smollm2.cma_mb = 480`: both models load at startup
(SmolLM2 1.0 s from the page cache → CmaFree 411 MB; BERT 14.3 s cold →
CmaFree 199 MB), so no swapping was needed on this boot.  Measured through
the OpenAI API (`chat.py`, stdlib `urllib`):

| request | result |
|---|---|
| "What is the capital of France?", greedy | "The capital of France is Paris. It is a city known for its historical landmarks, …" — the study's Q8.8 greedy text word for word (§10.4); first token 0.5 s, **5.0 tok/s** |
| "Three tips for learning a new language", default sampling (T 0.2, top-p 0.9) | coherent 3-item list, first token 0.4 s, 4.9 tok/s, 99 tokens in 20.7 s |
| 3-turn conversation | prefix reuse 24/45 → 124/141 → 220/238 cached positions; only 21 / 17 / 18 new tokens prefilled; TTFT 0.48 / 0.60 / 0.72 s; 4.6–4.95 tok/s |
| switch to `bert-squad` ("How much memory does the K26 have?" over a 2-sentence document) | "4 GB", 1.0 s |
| back to `smollm2` | TTFT 0.38 s (cache reused 24/35) |

**Quality is the model's.**  The float model (transformers, float32, greedy,
same template) gives the same weak answers on the PC: it forgets the user's
name in turn 3 ("My name is Alex Chen …" in both), and "Say hello in French"
degenerates the same way ("Bonjour, merci pour notre aide. Je suis de
nombreux, …" in both).  SmolLM2-360M is the same code path (regenerate with
its safetensors; ~2 tok/s, needs BERT unloaded or `resident: one`).

**Next levers (phase 5):** prefill attention on the FPGA (256-token prefill
3.6 → ~1.1 s), q/k/v and gate/up fusion (−90 of 211 calls per token), and
dual-port weight streaming for decode (~2×).

## 15. Repetition: DRY sampling (2026-09-26)

The 135M model loops verbatim at the default temperature 0.2 (e.g. a
story repeating "My toys are all in the big room." until `max_tokens`).
The sampler (`src/sampler.{h,c}`, `sampler.py`) now implements DRY
(p-e-w's "Don't Repeat Yourself", as in text-generation-webui and
llama.cpp): after the repetition penalties, a token that would extend a run
already in the context by L matched tokens loses
`dry_multiplier · dry_base^(L − dry_allowed_length)`; sequence breakers cut
runs.  Breakers for byte-level BPE are *every token whose bytes contain*
"\n", ":", "\"" or "*", plus all special tokens.  Defaults on the server:
0.8 / 1.75 / 2 over the whole context; every parameter is a request field
(`dry_*`).  C and Python are identical token for token and match a
transcription of the reference algorithm on 200 random histories.  Board:
the cat-story loop is gone (repeated trigrams 20 % → 5 %), answers without
repeats (facts, lists, code blocks) are unchanged, ~2.5 ms per token.


**Follow-up (same day): loops of short lines.**  DRY with the stock breaker
set still let a line-level loop through ("The cat loves her adventures /
This is a short story." repeated to `max_tokens`): "\n" cut every run at the
line end, so the strongest penalty any loop token met was 4.3 (mean 1.1).
New defaults: breakers `: " *` (newline dropped), repetition_penalty 1.1
over the last 64 tokens, and a loop guard (stop when the answer ends in a
block repeated verbatim 3 times, >= 24 tokens; `kv260.finish` "loop").
Board, the conversation that looped, seeds 1–3: no loops (0–3 % repeated
trigrams; two end at <|im_end|>, one runs to the 500-token cap with 1 %);
factual / list / code answers still correct (the code block is identical,
the prose is reworded).  Chat tests 121 pass.

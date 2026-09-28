# Chat app on the KV260 — plan and log

Status (2026-09-28): **all planned phases done, on main.**  Phase 1 BERT-QA
server (§9); phase 2 numeric study GO (§10); phase 3 SmolLM2-135M
`libsmollm2.so`, bit-exact (§13); phase 4 server integration with both
backends on the board (§12, §14), DRY sampling (§15); phase 5 prefill attention
on the FPGA, 256-token prefill 3.6 → 1.3 s (§16); decode attention on all host
threads, 5.06 tok/s at position 32 and 4.39 at 1000 (§17).  Board-hang
workaround in §18.  Dual-port weight streaming (MatmulKernel's GEMV mode):
one weight copy, decode **10.07 tok/s** at position 32 and 7.67 at 1000,
CMA pool 488 → 286 MiB (§19).  SmolLM2-360M-Instruct: bit-exact on the
board, 3.9 tok/s, 740 MiB pool, served by the same server (§20).  The study
stage is reproducible from pinned checkpoints and texts, with the formats
hashes recorded (`llm_calibrate.py`, §21).  SmolVLM-256M (image input): numeric
study GO (§22), then implemented — bit-exact on the board, 7.7 s per image +
~9.5 tok/s, served by the chat server with OpenAI image_url parts (§23).  Not done:
q/k/v + gate/up fusion, int8 weights.  §7 is the pre-implementation estimate; measured numbers are in
§13.4, §16.3, §17 and §19.  Builds on doc/plans/BERT_PLAN.md (BERT-base SQuAD at
971 ms per inference on the board, bit-exact with the scheduler simulation).

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

Delivered as [`demo/chat/`](../../demo/chat/) (README there: deploy, the
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
[`demo/chat/`](../../demo/chat/) (README section "Generative chat"):

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

### 13.2 Scheduler features (doc/scheduler/INFERENCE_SCHEDULER.md)

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

## 16. Phase 5 — prefill attention on the FPGA (2026-09-26, branch `feat/prefill-attn`)

**Result.**  The prefill's attention runs on ConvKernel with the p12 softmax
on the host; decode keeps the xattn host region.  On the board the 256-token
prefill takes **1317 ms instead of 3575 ms** (2.7×; attention 310 ms
instead of 2600 ms), the 64-token prefill 466 ms instead of 604 ms, and the
first logits of the three chat prompts arrive after 461 / 1259 / 901 ms
instead of 484 / 2176 / 1198 ms.  Decode is unchanged (203.7 ms per token).
The logits are bit-exact with the scheduler simulation, which equals the
study emulation of the new mixed policy `pow2+sink+p12+mix` bit for bit —
over first prefills, 32 greedy decode steps, a multi-bucket split and a
second chat turn prefilled at position 69.  No kernel or bitstream change;
the §11 C API and `llm_prefill` semantics are unchanged.

### 16.1 Design

* **Numerics: `pow2+sink+p12+mix`** (`llm_study.py`): prefill calls use the
  pow2+sink+p12 attention of §10.5 item 5 — RoPE(q) rounded at the per-head
  q exponent, q·Kᵀ and P·V as kernel MatMuls (exact sums, floor, saturate),
  the softmax on the host with P at 2⁻¹²; decode steps use the xattn region
  of §13.1.  Both read and write the same K / V caches at the same
  exponents (the p12 formats JSON, unchanged).  `Model.forward(...,
  phase=)` selects the path per call; every teacher-forced run is a prefill.
* **Runtime key count, not key-length buckets.**  A prefill call at
  position `pos` with `n` valid rows attends to `pos + n` keys.  The two
  ConvKernel calls per KV head run over `keys = roundup(pos + n, 64)` keys:
  the C computes it from the entry's `pos` / `n` inputs (`llm_keys`) and
  writes it into the AXI-Lite registers (`out_ch` of q·Kᵀ, `in_ch` of P·V);
  the rest of the geometry is fixed at codegen by the cost model.  The
  scheduler's simulation evaluates the same integer semantics over the
  actual count.  Key-length buckets (e.g. 64 … 1024) would have multiplied
  the prefill entries by five — inference.c is 4.2 MB with three and cc1
  already peaks at 300 MB — and padded the keys by up to 2×; runtime
  registers cost nothing.
* **Calls per KV head, heads of a group batched.**  GQA 9 / 3: the three q
  heads sharing a KV head form one call.  q·Kᵀ: `s[j][p] = Σ_d K[j][d] q[d][p]`
  as a MatMul on ConvKernel (BERT_PLAN §2 2A) with the K cache rows as the
  weight (`out_ch = keys`) and the three heads' queries as the x image
  (`p = (h', t)`, 3T pixels, 1×2 kernel — the host writes the image, so the
  cost model's kw is free).  P·V: P (3T rows, written by the softmax in the
  weight layout) × the V cache (`in_ch = keys / 4`, 1×4).  Per layer: 3 q·Kᵀ
  + 3 P·V calls, one prep, three softmaxes and a merge on the host.  Cost
  model per group (T 256, 256 keys, 100 MHz): q·Kᵀ 0.68 ms, P·V 0.49 ms,
  against 9.9 and 5.5 ms on MatmulKernel.
* **The V cache is the P·V conv's 1×4 x image.**  P·V streams P as the conv
  weight, one 2-beat request per P row and input-channel tile: 64 requests
  per slab, latency-bound (BERT's P·V was 2.5× its cycle model).  Storing
  each KV head's V rows as the x image of a 1×4 lowered MatMul
  (`numeric` layout `[KV, HD, 4]`) makes each P row of a slab one 8-beat
  request.  Measured alone on the board: 768 × 320 · 320 × 64 takes 2.26 ms
  with kw 1, 1.40 with kw 2, **0.97 ms with kw 4**; 768 × 1024 · 1024 × 64
  6.93 / 4.24 / 2.68 ms (the cycle model, blind to request latency, predicts
  0.78 / 0.61 / 0.60 ms).  The 256-token prefill gained 44 ms from it.  The
  xattn decode reads the interleaved rows with stride 4: its attention went
  from 6.6 to 7.3 ms per token (+0.3 % of a decode step).
* **KV caches in the CMA pool.**  The caches are DMA states (a new scheduler
  notion: persistent pool buffers after the weights, never shared,
  initialised with the sink row), stored group-major `[KV][C][HD]` so each
  KV head's first `keys` rows are one dense conv weight / input.  Host ops
  write them in place through the cacheable BO mapping; `LlmAttnPrep`
  flushes rows `[0, keys)` of both caches before the kernels read them
  (~6 syncs of ≤ 128 KiB per layer).  Decode steps write their row without a
  sync — it stays dirty until the next prefill flushes it — so decode pays
  nothing.  Kernels never write the caches, so no invalidation is needed.
* **Host work overlaps the kernel.**  The node order q·Kᵀ 0, q·Kᵀ 1,
  softmax 0, P·V 0, then softmax g, q·Kᵀ g+1, P·V g issues every softmax right
  after a ConvKernel call it can hide behind.  The softmax reads each
  32-column block of the scores once (transposed into a stack buffer),
  multiplies by `2^12 / sum` instead of dividing (the exact quotient within
  1e-7 of a rounding tie: identical integers) and balances causal rows over
  the threads: 1.5 ms per group call at T 256 (2.5 ms at first).  The prep
  writes the q image in whole cache lines.
* **Least-cost split.**  The bucket costs now include the attention:
  326 / 427 / 1275 ms per 16 / 64 / 256-row call (at position 1, head
  excluded).  The chat prompts split as before (36 → 64; 168 → 256;
  96 → 64 + 32/64).

### 16.2 Numerics (`llm_study.py study`, same data as §10.1)

| policy | top-1 prompt set (all / answers) | top-1 held-out | top-5 held-out | KL held-out | KL prompt set | ppl held-out | identical answers | mean match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float (numpy f64) | – | – | – | – | – | 15.598 | – | – |
| `pow2+sink+p12` (§10) | 0.976 / 0.980 | 0.969 | 1.000 | 0.0024 | 0.0021 | 15.616 | 2/12 | 15.9 |
| `pow2+sink+p12+xattn` (phase 3, shipped until now) | 0.976 / 0.982 | 0.968 | 1.000 | 0.0023 | 0.0022 | 15.612 | 3/12 | 27.9 |
| **`pow2+sink+p12+mix` (ships now)** | **0.976 / 0.980** | **0.969** | **1.000** | **0.0024** | **0.0021** | **15.616** | **3/12** | **21.8** |

Teacher-forced metrics are prefill-only, so the mixed policy's equal p12's
by construction; its greedy answers mix both attentions.  The answers read
like float's — for example (all 12 in the study output's
`generations.txt`):

```
summarise  float: The honeybee colony is a complex, interconnected system of workers, drones, and queens that
                  work together to maintain the colony's food source.
           mix:   The honeybee colony is a complex, interconnected system of workers, drones, and queens that
                  work together to maintain the colony's food source and survival.
           xattn: The paragraph summarizes the characteristics of a honeybee colony, including the diverse
                  workforce of workers, sterile female bees, and the unique dance of the waggle dance ...
factual    float: The capital of France is Paris.
           mix / xattn: The capital of France is Paris. It is a city known for its historical landmarks, ...
code       float / xattn: Here's a Python function that checks whether a number is prime: (identical code)
           mix:   Here is a Python function that checks whether a number is prime: (identical code)
```

**Bit-exactness** (`llm_sched_check.py`, SmolLM2-135M): the scheduler
simulation equals the study emulation of the mixed policy on the logits of
every step:

| prompt | prefill | calls (rows / bucket) | decode steps | result |
|---|---|---|---:|---|
| factual | 36 tokens at position 1 | 36 / 64 | 32 | bit-exact |
| factual, second turn | 19 tokens at position 69 (after the 32 decoded tokens) | 19 / 64 | 32 | bit-exact |
| summarise | 168 at 1 | 168 / 256 | 32 | bit-exact |
| multi-turn | 96 at 1 | 64 / 64 + 32 / 64 | 32 | bit-exact |

The generated project on the host emulation (`llm_host_emu.py
--incoherent`: separate CPU and DDR copies of every buffer, so a missing or
too-narrow cache sync changes the result) equals the simulation on all
4 × 33 logits vectors; its greedy tokens equal the study emulation's; close
/ re-open reproduces the logits.

### 16.3 Board (KV260, hw_128 bitstream, 100 MHz; `llm_board.py --profile --reopen`)

Before = the phase-3 library, measured in the same session.

| | before | after |
|---|---:|---:|
| prefill 16 tokens | 365.5 ms | 366.7 ms |
| prefill 64 tokens | 603.6 ms | 466.3 ms |
| prefill 256 tokens | 3575.2 ms | **1317.0 ms** |
| first logits: factual (36 tokens) | 484.1 ms | 460.5 ms |
| first logits: summarise (168) | 2176.5 ms | 1258.9 ms |
| first logits: multi-turn (96) | 1197.7 ms | 901.0 ms |
| first logits: factual, second turn (19 at position 69) | – | 457.1 ms |
| decode | 203.0 ms / token | 203.7 ms / token |
| `llm_open` (weights cached) | 1107 ms | 1041 ms (close → open 975 ms) |
| pool BO | 461.3 MiB | 488.2 MiB (+22.5 KV caches, intermediates 2.3 → 6.75) |
| CmaFree drop on open | 519.6 MB | 522.6 MB (page-cache noise ±30 MB) |

Prefill per call after (profile; attention = the wall time of its nodes —
the softmax windows overlap the ConvKernel windows):

| ms | 16 | 64 | 256 |
|---|---:|---:|---:|
| total | 367 | 466 | 1317 |
| MatMul linear (ConvKernel) | 237 | 257 | 670 |
| **attention** (before: host) | **24** (27) | **61** (201) | **310** (2600) |
| · prep (host) | 12 | 25 | 84 |
| · q·Kᵀ / P·V windows (ConvKernel, softmax inside) | 11 | 37 | 217 |
| SiLU·up | 38 | 44 | 161 |
| RMSNorm | 19 | 41 | 84 |
| residual add | 10 | 23 | 52 |
| LM head (once) | 39 | 39 | 39 |

The 16-row bucket gains nothing (its host attention was 27 ms); the 256-row
prefill is now 51 % linears and 24 % attention.

**Board gates.**  Logits bit-exact with the simulation on all 4 × 33 vectors
(the 3 prompts + the second turn), greedy tokens equal to the study
emulation's; the library through ctypes (`llm_lib_check.py`): only `llm_*`
exported, chunked prefill = one call, threads identical, close / re-open
identical.  Board build: cc1 peak 284 MB (-O2, gcc 11, -j1, 77 s; aarch64
gcc 13 on the host: 312 MB), MemAvailable ≥ 3.2 GB throughout.

**Late positions** (the key count follows the position: a 672-token
conversation, then new tokens; `llm_bench` continuation prompts):

| new tokens at position 673 | FPGA prefill attention | host attention (phase 3, estimated*) |
|---:|---:|---:|
| 16 | 482 ms | ~1.2 s |
| 64 | 732 ms | ~4 s |
| 256 | 2085 ms | ~17 s |

\* the phase-3 host attention scaled linearly in rows × keys (2.6 s for
256 rows over 1–257 keys; §13.4).  The 672-token prompt itself (256 + 256 +
160/256 calls) takes 4.6 s.  At these positions the host softmax dominates
the attention (see §16.5).

**Board suite** (`run_remote_tests.py`, the 148 models + the three tiny
Llama fixtures, which now run the FPGA prefill attention and DMA-state
caches): **151 / 151 pass**.

**Chat server** (library installed with `llm_board.py --install-only`,
server restarted with `deploy.py` from the main checkout; both models
resident, CmaFree 266 MB after both loaded — after `drop_caches`; with a
warm page cache in CMA, `auto` left BERT for its first request).  A 4-turn
conversation through the OpenAI API, temperature 0 (the server's default
repetition penalty and DRY apply):

| turn | cached / prefilled tokens | TTFT | decode |
|---|---|---:|---:|
| "What is the capital of France?" | 1 / 36 | 457 ms | 4.9 tok/s |
| "What is a famous museum there?" | 116 / 19 | 462 ms | 4.8 tok/s |
| "Tell me one more fact about that city." | 214 / 21 | 480 ms | 4.6 tok/s |
| "Thanks! Now summarise our conversation in one sentence." | 302 / 22 | 502 ms | 4.5 tok/s |

("… The Louvre is home to thousands of works of art including the Mona
Lisa, Venus de Milo …"; "Paris is a city renowned for its art,
architecture, culture, history, and cuisine, …".)

### 16.4 Scheduler and tests

* New scheduler notions (doc/scheduler/INFERENCE_SCHEDULER.md): **DMA states** (pool
  buffers that persist across calls and entries; host ops write them in
  place, kernels read them after `llm_cache_flush`), the **group-major /
  interleaved state layout** (`numeric` `layout`), and a **runtime
  dimension** of a kernel call (`LlmAttnConvNode`: the key count from the
  entry's `pos` / `n`).  New ops `LlmAttnPrep`, `LlmAttnScores`,
  `LlmAttnSoftmax`, `LlmAttnPV`, `LlmAttnMerge`; `LlmAttention` reads the
  new cache layout.  `LlamaFrontend(prefill_attn="host")` keeps phase 3's
  graphs (bit-exact with `pow2+sink+p12+xattn`).
* Tests: **1487 pass** (5 skips).  New: the prep → q·Kᵀ → softmax → P·V →
  merge chain on the host emulation (kernel widths 1 / 2 / 4, V interleave
  1 / 2 / 4, context clamp, no valid row, MHA, 4 groups), against an
  independent p12 computation; the key count and the softmax exp tables C
  == Python; the V image equals `conv_lowered_b_image`; the tiny Llama
  simulation == the study for both policies incl. a second turn after
  decode steps and a truncate; the SmolLM2-360M layer shape on the host
  emulation; a library-style call sequence of the multi-entry project run
  in C; the **incoherent host emulation** (separate CPU / DDR copies of every
  buffer: dropping the cache flush, a host op's output flush or an
  invalidate each makes the outputs differ); the coherency audit per entry
  of the multi-entry project with its `noinline` parts inlined (it saw only
  the part calls before) and DMA states starting dirty.
* Every existing model's generated project is byte-identical to main's
  (timestamp line excluded): 166 / 166 — the 155 test models that generate,
  BERT-SQuAD, ResNet-18, MobileNet v1 / v2 and 7 MNIST / LeNet models; the
  13 models that exist to fail fail identically; only the three tiny Llama
  fixtures change.

### 16.5 Open issues

* **The host softmax at late positions.**  It scales with rows × keys: 1.5
  ms per group call for 256 rows at position 1, ~10 ms at position ~700
  (board microbenchmark), where it outweighs the ConvKernel calls it hides
  behind — 2.1 s for 256 new tokens at position 673.  Options: keep it in
  float32 (a numeric change for the study), or have the q·Kᵀ call write the
  scores query-major (a transposed K cache, which decode would read with a
  stride).
* **The prep is serial** (84 ms of the 256-token prefill): its q part could
  run while the k / v projections are on ConvKernel (split the op,
  ~ −35 ms).
* **P·V is still ~1.6× its cycle model** with kw 4 (request latency);
  deeper weight-request pipelining in `stream_load_weights` is a kernel
  change.
* **The 16-row bucket gains nothing** (host attention was 27 of its 366 ms);
  short follow-up turns still take ~0.45 s, bound by the linears (237 ms)
  and the LM head.
* **CMA** grew by 26.9 MiB (pool 488.2 MiB).  The server's `smollm2.cma_mb`
  should be ~510 (the example config now says so; the board's local config
  still has 480).  `auto` residency judges by CmaFree, which page cache in
  CMA depresses: after many `llm_open`s it left BERT unloaded until its
  first request; after `drop_caches` both fit.  One `llm_open` failed
  transiently (`xclAllocBO` of 488 MiB with 825 MB CmaFree — page migration)
  and succeeded on retry.
* The whole-buffer syncs of the softmax input / output (1.5 MiB each at
  T 256) could be limited to the run-time extent (~0.4 ms per layer).

## 17. Decode attention on all host threads (2026-09-27)

**Symptom.**  In a long chat every answer came out slower: 5.0 tokens/s at
~40 cached positions, 3.7 at ~710 (server log `decode=`).  The decode step's
attention (`LlmAttention`, policy xattn: exact double on the host) reads every
cached key for every token, so its cost grows linearly with the position,
while the rest of the step (weight streaming, ~195 ms) does not.

**Where the time went** (board microbenchmark of the generated C, 30 layers,
H 9 / KV 3 / HD 64, VK 4, realistic score ranges):

| position | per-head code, 4 threads | single thread |
|---|---|---|
| 32 | 4.4 ms / token | |
| 512 | 39 ms | |
| 1000 | 75 ms | 215 ms (~8 cycles per multiply-add) |

* 9 heads over 4 threads → the busiest thread runs 3 heads (1.33× the mean).
* Each head re-reads and re-converts its group's K / V rows (3 heads per KV
  group read the same rows).
* gcc 11 at `-O2` emitted scalar code with the query pointers spilled (the
  `tree-vectorize` pragma did not vectorise these loops); per key the V row
  address took two integer divisions by the run-time interleave.
* The A53 does about one double operation per cycle, so a multiply-add
  (separate multiply and add — no FMA, for bit-exactness) costs ≥ 2 cycles.

**Change (`src/llm_nodes.py`, `llm_attn_decode`, used when n = 1).**  One
pool dispatch per call (a pool wake-up is ~80 µs, too much per phase); inside
it, phases separated by spin barriers:

1. scores per (KV group, key quarter): the G = 3 heads of a group share each
   K row load and int16 → double conversion; partial max per quarter;
2. `exp(s − max)` per (group, key quarter);
3. `Σe` per head, left to right (the reference order);
4. `p = e / Σe` per (group, key quarter);
5. P·V per (group, lane quarter): 8 lanes × 3 heads of accumulators stay in
   registers across all keys, the keys walked block / lane / row (no
   divisions).

The power-of-two cache scales are folded out exactly: `q · 2^-f_k[d]` is
exact, so the score terms are the same products; P·V sums `p · raw v` and
multiplies by `2^-f_v` once — power-of-two scaling commutes with rounding
while every term is normal, which holds for `p ≥ 2^-900` (a head with a
smaller non-zero p keeps the per-term scale; a `q · 2^-f` below 2^-1000
falls back to the per-head code).  On aarch64 the score and P·V kernels are
NEON (`vmulq_f64` + `vaddq_f64`, int16 → f32 → f64 conversions, `vld4q_s16`
for the ×4-interleaved V rows).  Every value is the per-head code's bit for
bit for any thread count; prefill (n > 1) is unchanged.

Env `INFERENCE_LLM_DECODE_PAR` (read once): `0` = the per-head code, `1` = the
phases on all host threads (default), `2` = the phases on the calling thread
only (no pool dispatch, no spin barriers) — the same bits in every mode
(board, 950 decode steps each: 238 / 216 / 249 ms per token on average).

Microbenchmark (same data, 4 threads; bit-exact against the per-head code on
240 random cases incl. extreme score ranges, at 1–4 threads):

| position | before | after | |
|---|---|---|---|
| 32 | 4.4 ms | 3.8 ms | 1.2× |
| 256 | 20.5 ms | 10.4 ms | 2.0× |
| 512 | 39.3 ms | 17.7 ms | 2.2× |
| 768 | 57.9 ms | 25.5 ms | 2.3× |
| 1000 | 74.6 ms | 32.7 ms | 2.3× |

**Board** (`llm_board.py --decode-at 32,128,256,512,768,1000 --decode-at-steps
8`: prefill to the position, then 8 greedy decode steps; the old library's
bench rebuilt with the same `-D` for the baseline):

| position | before | after | tokens / s |
|---|---|---|---|
| 32 | 197.3 ms | 197.7 ms | 5.07 → 5.06 |
| 128 | 204.7 ms | 199.9 ms | 4.89 → 5.00 |
| 256 | 214.4 ms | 204.2 ms | 4.66 → 4.90 |
| 512 | 234.8 ms | 212.3 ms | 4.26 → 4.71 |
| 768 | 253.8 ms | 220.2 ms | 3.94 → 4.54 |
| 1000 | 270.9 ms | 227.8 ms | 3.69 → 4.39 |

The growth over the context fell from ~74 to ~30 ms per token.  The FNV of
every position's decode logits is identical before and after (bit-exact on
the real model up to position 1000); the gate's logits are bit-exact with the
simulation (4 prompts × 33), `threads_identical`, re-open identical;
prefill 364 / 466 / 1314 ms and CMA unchanged.  The scheduler tests gained
decode cases (G 1 / 2 / 3, VK 1 / 2 / 4, first / last row; a perturbed
decode output fails 11 of them).

**What is left of the slope** (~30 ms at 1000): the P·V and score kernels at
~3 cycles per multiply-add (FP64 on the A53), 270 k `exp` + divisions per
token.  Float32 attention would halve it but is a numeric-policy change (the
study and the emulator define xattn in double).

## 18. Board "hangs": a core parked in the PSCI power-down idle state (2026-09-26)

**Symptom.**  During chat generation the token stream stopped, and some time
later (0 – 3 minutes) the board stopped answering ping and SSH; three times in
one evening (and likely the BERT-phase "userspace starvation" incident, §BERT
phase 2).  Memory, CMA, power (INA260, 4.2 W) and temperature (≤ 35 °C) were
normal up to the moment; the local journal showed other userspace still
running for minutes after the network died.  The decode attention change of
§17 was suspected first: it is not the cause (950-step `llm_bench` runs in all
three `INFERENCE_LLM_DECODE_PAR` modes passed; the hangs are random in time).

**Root cause (serial console + JTAG, the user's report of 2026-09-27).**  CPU1
entered cpuidle state 1, `cpu-sleep-0` (PSCI core power-down).  TF-A v2.8
(`xlnx_rebase_v2.8_2023.2`) set `APU.PWRCTL.CPUPWRDWNREQ` and parked the core
on the final `wfi` of `psci_power_down_wfi` (EL3, GIC CPU interface off); the
PMU firmware never powered it down (`PMU_GLOBAL.PWR_STATE` still shows it on),
so it never wakes.  Its timer and IPI counters freeze; every task queued on it
(a chat-server thread, `netwatch`, kthreads) never runs — the stream stops;
later any all-CPU cross-call (`kick_all_cpus_sync`, e.g. `sshd`'s seccomp BPF
JIT) spins forever on the caller's CPU (soft lockups, RCU stalls), each new
SSH connection taking another core.  Each core powers down ~10 times a second
when idle, so the lost handshake is a matter of time.

**Workaround (applied on the board).**  Keep the cores out of the power-down
state; WFI idle stays:

```
# /etc/tmpfiles.d/kv260-no-cpu-powerdown.conf   (applied at boot)
w /sys/devices/system/cpu/cpu*/cpuidle/state1/disable - - - - 1
```

The rule is `board/kv260/kv260-no-cpu-powerdown.conf`; `demo/chat/deploy.py`'s
preflight installs it when missing, applies it at once and says so.
The rule applies a few seconds into boot (a few hundred power-downs happen
before it); `cpuidle.off=1` on the kernel command line closes that window.
The fix proper is newer boot firmware (`xmutil bootfw_update`: TF-A + PMU
firmware) — then re-test with the state enabled.

**Diagnosing the next one.**  Do not open SSH sessions to a board in this
state (each one locks a core); use the serial console (FT4232H channel B,
115200) — `/proc/interrupts` twice (a frozen `arch_timer` column), SysRq-l (a
core that prints no backtrace).  Do not read PMU RAM over JTAG on a live
system: it wedged the PMU and then CPU0.

## 19. One weight copy and dual-port decode: MatmulKernel GEMV (2026-09-27)

**What.**  MatmulKernel gained a GEMV streaming mode (`gemv_kw`, `a_to_b`;
MATMUL_KERNEL.md §1, MATMUL_OPTIMISATION.md §8b) for one-row MatMuls: B is
streamed once per A row through **both** read ports, in the image ConvKernel
reads for the prefill MatMuls (kernel width `kw`).  The scheduler's
`src/llm_entries.py` builds the prefill buckets first (power-of-two kernel
widths, `kw = 4` for every SmolLM2 linear), then the decode and head graphs
reading the same images (`OnnxGraph(matmul_gemv_kw=...)`); multi.py keeps one
buffer per weight.  This is the "dual-port / HPC1 weight streaming" of §13.6
and removes its "Two weight copies" item.

**Pool.**  421 → 211 weight buffers (210 linears + the LM head, none
renamed), weights 459.0 → 256.5 MiB, CMA pool 488.2 → 285.8 MiB, weight
files 538 → 326 MB.  `smollm2.cma_mb` 510 → 330 (example config, deploy
default, `--llm-cma-mb`).  SmolLM2-360M by the same arithmetic: layer
weights 600 MiB + LM head 90 + KV caches 40 + ~12 ≈ 742 MiB, inside
`cma=1000M`.

**Board** (new bitstream, WNS +0.671 ns; `llm_bench -D 32,256,1000 -S 8`
and `llm_board.py --skip-build --reopen`):

| | before (§17) | GEMV |
|---|---:|---:|
| decode at position 32 | 197.3 ms (5.07 tok/s) | **99.3 ms (10.07 tok/s)** |
| decode at 256 | 203.2 ms | 106.5 ms |
| decode at 1000 | 228.3 ms (4.38 tok/s) | 130.4 ms (7.67 tok/s) |
| prefill 16 / 64 / 256 | 367 / 466 / 1317 ms | 340 / 444 / 1279 ms |
| `llm_open` (weights cached) | 1.0 s | 0.6–0.7 s |

The decode checksums are identical to the pre-GEMV build (FNV 186711104 /
3694798025 / 1483111907), the logits bit-exact with the simulation on all
4 × 33 vectors, chunked prefill / threads / close → open identical.  Decode
is still weight-bandwidth bound: 256 MiB per token at ~3.1 GB/s (the two
read ports at 100 MHz) ≈ 87 ms of the 99; the prefill gain is the LM head
of the head entry (39 → 18 ms).  `llm_lib_check.py`'s `abs(CmaFree after
close − before open) < 4 MB` rule failed on 2 of 3 runs by 10 MB: the value
after close stays at 872–875 MB over four cycles while the one before open
moves with the page cache (the newly uploaded weight files) — no drift, no
leak.

Through the chat API (temperature 0, both models resident; the first turn
had 24 tokens in the prefix cache from an earlier request):

| turn | cached / prefilled tokens | TTFT | decode |
|---|---|---:|---:|
| "What is the capital of France?" | 24 / 13 | 348 ms | 9.8 tok/s |
| "What is a famous museum there?" | 100 / 19 | 434 ms | 9.6 tok/s |
| "Tell me one more fact about that city." | 182 / 21 | 456 ms | 9.3 tok/s |
| "Thanks! Now summarise our conversation in one sentence." | 266 / 23 | 468 ms | 9.1 tok/s |

BERT (`libbert_squad.so`) had to be regenerated with the new code generator:
its `run_matmul()` now writes `gemv_kw` / `a_to_b` on every call — a library
generated before the registers existed would inherit the `gemv_kw` a
SmolLM2 decode left in the kernel (`deploy.py --regenerate --rebuild`).

## 20. SmolLM2-360M-Instruct (2026-09-28)

**Model.**  `HuggingFaceTB/SmolLM2-360M-Instruct` (Apache-2.0, revision
a10cc15): 32 layers, hidden 960, 15 / 5 heads of 64, FFN 2560, vocab 49152,
tied embedding, RoPE theta 100000 — the same Llama graph as 135M, and a
byte-identical tokenizer (tokenizer, template and prompt sets carry over).
Assets in `demo/chat/assets/smollm2-360m-instruct/` (not in git, 724 MB).
The study and the formats of a model other than 135M live in
`assets/study/<model dir>/` (`llm_study.study_dir`).

**Numerics** (`llm_study.py study --assets assets/smollm2-360m-instruct`,
the §10 data, 38 min):

| policy | top-1 all | top-1 resp | top-1 held | top-5 held | KL held | ppl (float 12.373) | greedy identical |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 0.9923 | 0.9873 | 0.9782 | 1.0000 | 0.0006 | 12.395 | 6 / 12 |
| pow2+sink+p12 | 0.9777 | 0.9936 | 0.9759 | 0.9997 | 0.0018 | 12.394 | 8 / 12 |
| **pow2+sink+p12+mix** (shipped) | 0.9777 | 0.9936 | 0.9759 | 0.9997 | 0.0018 | **12.394** | 8 / 12 |

360M keeps the float perplexity as well as bf16 does (135M: 15.616 vs
15.598).  `validate`: numpy float64 = torch float32 to 1.5e-4 in the
logits, greedy identical 12 / 12.  Formats: 450 exponent entries, no weight
saturates.

**Project** (`generate_llm_project.py --assets assets/smollm2-360m-instruct
--model-name smollm2-360m-instruct` → `build/llm_project_smollm2_360m`):
prefill 224 / 224 MatMuls on ConvKernel (kw = 4), decode 225 on the GEMV
path in the same images (§19): one copy of every weight.  Pool **740.2
MiB** (weights 690.0, KV caches 40.0, intermediates 10.25) — inside
`cma=1000M` — host tables 90.2 MiB, weight files 818 MB.

Generating it first ran out of the host's 46 GB (the kernel killed it at
40 GB): every scheduled node held its NodeProto, which pins the whole
shape-inferred ModelProto with its initializers, and each of the five
entries kept its own copy of every weight array.  Fixed in the scheduler —
`OnnxGraph` keeps detached NodeProto copies, `src/llm_entries.entry_graphs`
consumes the entry models one at a time and shares equal weight arrays
between the entries: SmolLM2-135M's generation peaks at 12.6 GB instead of
17.9, 360M's at 31.9 GB.  `llm_project.make_codegens` builds the checks'
simulations the same way.

**Gates.**  `llm_sched_check.py --assets ...`: the scheduler simulation
equals the study emulation bit for bit (3 prompts × 32 decode steps and a
second turn at position 69; 23 min, 27 GB).  Board (`llm_board.py --project
build/llm_project_smollm2_360m --study-json .../gate2.json --reopen`): logits
bit-exact on all 4 × 33 vectors, greedy tokens = the study emulation's,
chunked prefill / threads / re-open identical.  `llm_lib_check.py`'s
4 MB CMA rule missed by 0.4 MB (page cache, as in §19).

**Board** (same bitstream as §19):

| | SmolLM2-360M | SmolLM2-135M (§19) |
|---|---:|---:|
| decode at position 32 / 256 / 1000 | 256 / 270 / 306 ms (3.9 / 3.7 / 3.3 tok/s) | 99 / 107 / 130 ms |
| prefill 16 / 64 / 256 tokens | 0.83 / 1.00 / 2.90 s | 0.34 / 0.44 / 1.28 s |
| `llm_open` cached / cold (SD card) | 1.5–1.7 s / 58.8 s | 0.6–0.7 s |
| CMA used | 736–740 MB | ~300 MB |

Decode is weight-bandwidth bound as for 135M: 690 MiB per token at ~3.1 GB/s
≈ 233 ms of the 256.  Through the chat API (the §19 conversation,
temperature 0): TTFT 0.97–1.00 s, decode 3.87–3.97 tok/s; the answers are
short and on topic (the Louvre, Notre-Dame, and a one-sentence summary
naming three landmarks) where 135M's ran to the 64-token limit.
With both models resident the 740 MiB pool and BERT's ~224 MB do not fit
together: `--resident auto` swaps them (a BERT question after a SmolLM2
turn took 15.1 s, the next SmolLM2 turn 10.3 s, both including the swap).

**Deploying it.**  `llm_board.py` installs a model other than 135M next to
it: `lib/libsmollm2_360m.so`, `/root/smollm2_360m_weights`,
`llm_project_smollm2_360m` (`board_paths`).  The server serves the model
the library names (`llm_model_name()`; `smollm2.model_id` /
`--llm-model-id` override): set `smollm2.lib` to
`/root/kv260_chat/lib/libsmollm2_360m.so` and `smollm2.cma_mb` to 760 in
`chat_config.json`.  The board runs 135M by default.


## 21. The study stage, reproducible (2026-09-28)

The libraries' numerics are fixed by one file per model: the calibrated
exponents and the position-0 sink K / V, `formats_pow2+sink+p12.json`, which
`llm_study.py formats` computes from the checkpoint and the WikiText-2
calibration text (§10).  Until now both inputs were fetched by hand from
moving targets (`resolve/main`, the datasets server) and nothing recorded
which bytes the shipped formats came from.

**Pins** — `demo/chat/scripts/llm_models.json` (tracked): per model the
Hugging Face repo at a commit (135M 12fd25f, 360M a10cc15), the SHA-256 of
its eight checkpoint files, the calibration policy and the SHA-256 of the
formats JSON, plus the shipped policy's study metrics (`pow2+sink+p12+mix`);
the SHA-256 of both texts, which are model independent.
`requirements-study.txt` is the environment that produced them (numpy 2.5.3
with its OpenBLAS 0.3.34, tokenizers 0.23.2; torch / transformers for
`validate` only).

**`llm_calibrate.py`** — `fetch` (download at the pinned revision, every
hash verified, matching files kept, texts copied from another model's
assets or fetched by `llm_study.py fetch`), `calibrate` (formats into a
scratch file, installed only when the hash reproduces or with `--record` /
`--force`, otherwise left as `formats_*.new.json`; the provenance —
input hashes, commit, `llm_study.py`'s hash, package / BLAS versions, CPU —
goes to `assets/study/<model>/formats_*.provenance.json`, tracked), `check` (the same without installing, plus a diff of
the exponents / sink values against the installed file), `study` (bf16 and
the shipped policies → `study[/<model>]/shipped/`, metrics compared with the
manifest), `validate`, `all`, `add` (pin a new checkpoint).  Stdlib only; the
study steps run in the study environment.  17 tests in
`tests/test_llm_calibrate.py` (file:// URLs for Hugging Face, a fake
`llm_study.py`).

**Verified:**

| | 135M | 360M |
|---|---|---|
| `fetch` into an empty directory | 58 s, 10 / 10 hashes (texts from the datasets server) | 135 s, 10 / 10 (texts copied) |
| formats SHA-256 (reruns, clean directory, `check`) | b870edfe… reproduced | 7196e674… reproduced |
| `study` (shipped policy) | ppl 15.616, top-1 0.976 / 0.980 / 0.969, KL 0.0024, 3 / 12 identical (= §16.2); 15 min | recorded from §20's run (the same code path) |

The calibration is deterministic on this machine: reruns are byte
identical.  numpy's OpenBLAS picks CPU-specific kernels at run time, so on
another CPU a calibration maximum near a power of two, or a sink value near
a rounding boundary, can come out differently; `check` names such
differences, and a differing formats file is a different (not a wrong)
library — its board gate is the bit-exactness against its own study
emulation (§13, §20).

## 22. SmolVLM-256M-Instruct — numeric study (2026-09-28)

**Verdict: GO on today's bitstream.**  The vision encoder runs on the
existing kernels with the text model's recipe: per-channel power-of-two
exponents, a float residual, and LayerNorm / GELU / softmax / bias adds on
the host.  End to end with the shipped text policy, answers agree with
float about as closely as bf16 inference does.  Plain Q8.8 (the BERT
partition) is broken for this encoder.

**Model** (`HuggingFaceTB/SmolVLM-256M-Instruct`, Apache-2.0, revision
7e3e67e; Idefics3):
- **Vision encoder:** SigLIP-style, 12 layers, 768 wide, FFN 3072, 12 heads
  of 64, GELU tanh, LayerNorm.  A 512 × 512 image in 16 × 16 patches gives
  1024 tokens.
- **Connector:** pixel shuffle ×4 (64 tokens of 12288), then one linear
  12288 → 576.
- **Text model:** a Llama shaped like SmolLM2-135M (30 layers, 576 wide,
  9 / 3 heads, RoPE θ 100000) with an untied LM head and vocab 49280.
- **Prompt:** `<|im_start|>User:<fake_token_around_image><global-img>`,
  64 × `<image>`, `<fake_token_around_image>`, the text, then
  `<end_of_utterance>\nAssistant:` (81 tokens for a short question).
- **One tile per image:** image splitting is off, and the image is squared
  to 512 × 512 by two LANCZOS resizes (longest edge to 2048, then to 512).
  The default 4 × 4 + 1 split would need 1088 image tokens, more than the
  1024-token context.

**Script:** `demo/chat/scripts/vlm_study.py`.  Commands:
- `fetch`: the pinned checkpoint and COCO val2017 images, checked against
  `vlm_study_inputs.json`.
- `validate`: against transformers.
- `ablate`: the vision error by tensor class, by weight, or by whole policy.
- `study`: the policy comparison.

It reuses `llm_study.py`'s emulation primitives.  Two changes to
`llm_study.py`, neither of which changes the SmolLM2 formats hashes:
an `embed()` hook, and weight fitting against `lm_head.weight` for untied
heads.

Calibration uses 10 COCO images with float's own answers teacher-forced,
plus the WikiText-2 calibration text.  Evaluation uses 24 other images with
three generic prompts in rotation and 96 new tokens.

**Validation** (numpy float64 against torch float32, 4 images):
- the prompt ids and the pixel values equal the Idefics3 processor's
  (PIL backend) on 24 / 24 images;
- image features agree to 2.4e-4 (values up to 120), logits to 5.9e-4;
- greedy answers are identical on 4 / 4 images.

**Emulated vision datapath:**
- **Pixels:** the uint8 values enter the patch-embedding MatMul exactly, at
  exponent 0.  The (x − 0.5) / 0.5 normalisation is folded into its
  weights (× 2/255) and bias.
- **Kernel MatMuls:** patch embedding, q / k / v, q·kᵀ, P·V, out_proj, fc1,
  fc2 and the connector, all with per-channel pow2 exponents.
- **Host passes:**
  - add the q / k / v biases and write q, k per head and V per channel;
  - softmax over all 1024 keys, P at 2⁻¹²;
  - add the fc1 bias, apply GELU;
  - add the out_proj / fc2 biases inside the float32 residual adds;
  - LayerNorm with left-to-right sums;
  - pixel shuffle, which only moves data: channel c′ keeps the exponent of
    c′ mod 768.
- **Into the text model:** the image features are dequantised into its
  float residual at the `<image>` positions.  The text model is
  `llm_study.Model` under the shipped `pow2+sink+p12+mix`.

| vision / text policy | features rel. err (max) | answer top-1 | top-5 | KL | identical | WikiText ppl (float 12.221) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 / bf16 | 0.017 (0.022) | 0.978 | 1.000 | 0.0012 | 7 / 24 | 12.196 |
| q88 (BERT partition) / float | 0.879 (1.150) | 0.813 | 0.973 | 0.1891 | 0 / 24 | |
| pow2+p12 / float | 0.054 (0.067) | 0.985 | 1.000 | 0.0018 | 11 / 24 | |
| pow2+p14+in7 / float | 0.048 (0.057) | 0.985 | 1.000 | 0.0014 | 11 / 24 | |
| pow2+hattn (float attention) / float | 0.037 (0.047) | 0.985 | 1.000 | 0.0010 | 11 / 24 | |
| float / pow2+sink+p12+mix | 0 | 0.977 | 1.000 | 0.0017 | 7 / 24 | 12.223 |
| **pow2+p12 / pow2+sink+p12+mix** | 0.054 (0.067) | **0.975** | 1.000 | 0.0031 | 6 / 24 | 12.223 |
| pow2+p14+in7 / pow2+sink+p12+mix | 0.048 (0.057) | 0.975 | 1.000 | 0.0033 | 8 / 24 | 12.223 |

How to read the table:
- **Answer columns** are teacher-forced on float's greedy answers, over the
  answer positions only.  "Identical" counts greedy answers equal to
  float's.
- **The vision emulation costs less than the text emulation.**  With float
  text it stays at top-1 0.985.  The end-to-end numbers match the text
  study's: for SmolLM2-135M, pow2+sink+p12 had KL 3.4× bf16's; here the
  ratio is 2.6×.
- **Answers read like float's**, and diverge where bf16's do
  (`generations.txt`).  For example, "There are three jet planes flying in
  the sky." is word for word, and the surfing photo's description matches
  in substance.

**Why Q8.8 fails.**  Attention scores reach 483, far past Q8.8's ±128.
With 1024 keys a typical probability is about 2⁻¹⁰, which rounds to
nothing at 2⁻⁸.  The residual reaches 753 and fc2's output 749.

**Where the remaining error comes from.**  `ablate` on 3 evaluation images,
starting from 5.2 % feature error:
- **Weight rounding: 3.7 %**, mostly fc1 (2.6 %) and out_proj (2.0 %).
  fc2 adds 1.0 % and the connector 0.9 %; q / k / v and the patch
  embedding each add under 0.7 %.
- **P at 2⁻¹²: most of the 3.4 %** that all activations cause with exact
  weights.  The two sources add in quadrature.
- **Every other tensor class: at most 0.1 %.**

Both weight groups read finely scaled inputs: out_proj reads P·V at about
2⁻¹¹, fc1 reads LayerNorm outputs at the 2⁻⁸ cap.  With the kernel's fixed
>> 8, f_w = f_out + 8 − f_in, so input bits come out of the weights.

Finer P does not help: 2⁻¹³…2⁻¹⁵ stays at 5.3–5.6 %, because P·V must fit
int16 and V loses the bits.  Capping host-written inputs at 2⁻⁷ does help
(4.5 %); 2⁻⁶ gives 4.8 % and 2⁻⁵ 9.2 %.  Float attention on the host
(hattn) reaches the weight-rounding floor, 3.3–3.7 %, but q·kᵀ and P·V in
double on the A53s would take roughly 10 s per image.  A runtime output shift in the kernels (sh > 8)
would give every weight more bits; that is a bitstream change, not needed.

**Other findings:**
- **Attention sink:** the text model has SmolLM2's residual, 15,606 from
  layer 11 on (SmolLM2-135M: 25,982), so the precomputed position-0 sink
  carries over.
- **Image features reach 125:** SmolLM2's embedding rows are below 1.  They
  go straight into the float residual; the connector writes them at
  per-channel exponents (`img`).
- **Weights:** no vision weight saturates; the accumulators stay below 261
  of ap_fixed<32,16>'s 32,768 range.
- **Out-of-range values:** values beyond the calibration range saturate
  146 fc1, 29 out_proj and 1 patch-embedding elements in the evaluation run
  (pow2+p14+in7 also 26 P·V), out of about 10⁹.

**What an implementation needs** (costs scaled from BERT's measured phase-2
breakdown in doc/plans/BERT_PLAN.md; per 512 × 512 image):
- **Linears:** 87.0 GMAC, about 2.0 s at 44 GMAC/s.
- **Attention MatMuls:** 19.3 GMAC.  BERT's ran at 9 GMAC/s at 256 tokens;
  larger matrices should do better, so 0.5–2 s.
- **Softmax:** 151 M exponentials on the host, about 2 s at BERT's
  13 ns / element.
- **LayerNorm, GELU, bias and residual passes:** about 0.5–1 s.
- **Total:** about 5–7 s per image before any optimisation.
- **Then the text side:** prefill of about 80 tokens (0.5 s) and decode at
  the 135M rate (about 10 tok/s).  The vision weights (93 M, 185 MB int16)
  bring the pool to roughly 510 MiB.

The pieces:
1. **The vision graph:** a SigLIP frontend next to `src/llama.py`, as a
   `vision` entry of the multi-entry project, with host passes as above.
   The GELU could also take its bias from VectorOP so that a table on the
   raw input applies.
2. **The text model:** `llama.py` for the nested config and the untied
   head.  `LlmEmbed` takes image-feature rows at `<image>` ids.  The prefix
   cache must key on the image content.
3. **The server:** OpenAI `image_url` content parts, PIL on the board (the
   same two LANCZOS resizes), the Idefics3 template, and
   `<end_of_utterance>` as the stop token.
4. **The gates:** scheduler simulation and board bit-exact against this
   emulation.

## 23. SmolVLM-256M on the FPGA (2026-09-28)

**Result.**  `libsmolvlm_256m.so` implements the §11 C API plus images, and
the chat server serves `smolvlm-256m-instruct` next to SmolLM2-360M and
BERT.
- **Board gate:** the logits of 2 image prompts × (prefill + 16 greedy
  steps) are bit-exact with the scheduler simulation, which equals the
  study emulation.
- **Chat API:** an image question's greedy answer through the server
  (JPEG decoded and resized on the board) equals the study emulation's.
- **Speed:** the vision encoder takes 7.7 s per image; the text runs at
  135M speed (101 ms / token).

**Numerics, final (vlm_study.py; the specification the C reproduces):**
- **GELU:** a 65 536-entry table per fc1 exponent, filled with libm exp in
  GELU's tanh form (y = x − x / (exp(2u) + 1)).  fc1's bias is added as an
  integer at fc1's exponent first.  libm tanh is avoided because its
  aarch64 and x86 results need not agree; exp agrees, as the SiLU tables
  showed.
- **Connector:** K = 12 288 exceeds every kernel (MatmulKernel 4096,
  ConvKernel 1024 input channels × kw ≤ 7).  It runs as three
  4096-row MatMuls at a shared output exponent, summed exactly on the host.
- **Patch embedding:** the bias and position-embedding table is float32,
  and the folded patch weights are float32 as the ONNX initializer holds
  them.
- **Formats:** `vlm_study.py formats` writes the text formats (llm_study
  format, with the sink rows) and the vision formats (`pow2+p12`) into
  `assets/study/smolvlm-256m-instruct/`.  Reruns are byte-identical.
- **The final spec, rerun on the §22 data** (24 images, 96 new tokens;
  `study/smolvlm-256m-instruct/final/`):

  | vision / text | features rel. err | answer top-1 | KL | identical |
  |---|---:|---:|---:|---:|
  | bf16 / bf16 | 0.017 | 0.983 | 0.0011 | 7 / 24 |
  | pow2+p12 / float | 0.055 | 0.981 | 0.0019 | 9 / 24 |
  | float / pow2+sink+p12+mix | 0 | 0.977 | 0.0016 | 6 / 24 |
  | **pow2+p12 / pow2+sink+p12+mix (built)** | 0.055 | **0.970** | 0.0034 | 5 / 24 |
  | pow2+p14+in7 / pow2+sink+p12+mix | 0.047 | 0.979 | 0.0028 | 5 / 24 |

  The same as §22's within noise (the float reference moved slightly too:
  its b0 table is float32 now).  pow2+p14+in7 did better here and equal in
  §22.  Switching needs only its vision formats and a regeneration: every
  exponent is data.

**Scheduler** (`inference-scheduler/src/vit.py`, `src/vit_nodes.py`):
- **The `vision` entry:** patches [1024][768] (raw uint8 values at exponent
  0) → state `vlm.img` [64][576] float32.
- **Graph:** 598 nodes, and the cost model puts all 76 MatMuls on
  ConvKernel.
- **Host ops:**
  - VitEmbedAdd, VitLayerNorm, VitResAdd (the bias inside the float32 add);
  - VitAttnPrep: biases, q / k at per-head exponents, the q.Kᵀ input image,
    and the K / V caches;
  - VitAttnSoftmax: every key, P at 2⁻¹²;
  - VitGelu, VitPixelShuffle (column chunks), VitSumDequant.
  Their C helpers (`VIT_C`) are emitted only when used, so every other
  project stays byte-identical; SmolLM2-135M's project differs only in its
  timestamp line.
- **Attention:** q.Kᵀ and P.V reuse `LlmAttnConvNode` with a static key
  count (no pos / n inputs; keys = 1024).  The K / V caches are one pair of
  raw DMA states (exponent 0, the exponents in the prep op's attributes),
  shared by all 12 layers.
- **The text model:** `LlamaFrontend(image_rows=64)`.  `LlmEmbed` gets an
  optional image-state input: ids V .. V + 63 read its rows.
- **Project:** `src/llm_entries.entry_graphs` builds extra entries (vision)
  with the default lowering.
- **Tests:** `test/test_vit.py` covers a tiny random ViT:
  - simulation == study emulation, both engine choices;
  - generated C == simulation in 3 build configurations;
  - image rows against the study text model;
  - a combined vision + text project.
  The full suite: 1510 pass, 5 skipped.

**Library** (`generate_llm_project.py --model-name smolvlm-256m-instruct` →
`build/llm_project_smolvlm_256m`, 100 s):
- **Pool:** 495 MiB (weights 433, KV caches 26, intermediates 36).  Host
  tables 57 MiB, weight files 514 MB.  One copy of every weight: decode's
  GEMV path reads the prefill image.
- **C API:** `llm_api.c` adds `llm_image_tokens()` (64), `llm_image_size()`
  (512) and `llm_image(rgb)`.  `llm_image(rgb)` patches the 512 × 512 × 3
  image into a DMA buffer and runs the vision entry.  `llm_prefill()`
  accepts ids up to V + 63.  Text-only libraries report 0 and refuse
  images.
- **Bench:** `llm_bench -I images.bin` encodes image p before prompt p.
- **Board paths:** `llm_board.py` installs `lib/libsmolvlm_256m.so`,
  `/root/smolvlm_256m_weights` and `llm_project_smolvlm_256m`.

**Gates:**

| gate | result |
|---|---|
| `vlm_sched_check.py --text` (simulation vs study) | vision 2 / 2 images bit-exact; text 2 / 2 prompts (80 / 83 tokens, split 64 + 16 / 64 + 19) + 16 decode steps bit-exact |
| `vlm_host_emu.py` (generated C on the host) | 2 × 9 / 9 logits bit-exact, re-open identical |
| `llm_board.py` (KV260) | 2 × 17 / 17 logits bit-exact, re-open identical; `llm_lib_check`: exports, chunked prefill, threads and re-open all fine; the 4 MB CMA-return rule missed by 7.6 MB (page cache, as in §19 / §20) |

**Board** (same bitstream as §19; profile per kind, with the host softmax
overlapping the ConvKernel calls):

| | |
|---|---:|
| `llm_image` (vision encoder + connector) | **7.73–7.76 s** |
| … MatMuls (76, ConvKernel) | 3.6 s |
| … softmax (host, 144 × 1024²) | 2.3 s |
| … P.V (ConvKernel) | 2.1 s |
| … q.Kᵀ (ConvKernel) | 0.85 s |
| … GELU 0.43, LayerNorm 0.32, attention prep 0.24, residual adds 0.12 | 1.1 s |
| decode | 101 ms / token |
| prefill 16 / 64 / 256 tokens | 0.34 / 0.44 / 1.28 s |
| `llm_open` cached / cold (SD card) | 1.0 / 21.6 s |
| CMA | 520–530 MB |

**Server** (`smolvlm_backend.py`, `idefics3.py`, `vlm_image.py`;
`kv260_chat_server.py --backend smolvlm`):
- **Images in requests:** OpenAI `image_url` parts with base64 data URLs.
  The server fetches no remote URLs.  Text models answer an image with 400
  `images_not_supported`, and the body limit is 32 MB when an image model
  is served.
- **Preprocessing:** Pillow, the processor's two LANCZOS resizes.  The
  pixels are identical under Pillow 9.0.1 (board), 10.2 and 12.3 for the
  test JPEGs, and an LRU cache keyed by URL keeps a resent history from
  being decoded again.
- **Template:** `idefics3.py` produces transformers' Idefics3Processor ids
  for 7 conversation shapes (`tests/data/smolvlm_cases.json`).
- **Prefix cache:** keyed by content, with image rows keyed as (SHA-256,
  row).  New positions are prefilled in segments with `llm_image()` before
  each image block.  The answer's first leading space is dropped, so a
  resent answer tokenizes exactly as generated.
- **Tests:** 11 in `tests/test_smolvlm_backend.py`; the chat suite passes
  149 tests.
- **Clients:** `deploy.py` (a `smolvlm` config block, Pillow in the
  preflight) and `chat.py --image FILE` / `/image FILE`.

Through the API on the board, greedy:

| request | cached / prefilled tokens | TTFT | decode |
|---|---|---:|---:|
| first image (model load: 360M evicted, cold `llm_open`) | 1 / 80 | 9.7 s (57 s total) | 9.1 tok/s |
| the same image, another question | 70 / 12 | 0.36 s | 9.0 tok/s |
| a follow-up turn | 104 / 15 | 0.36 s | 9.6 tok/s |
| a new image | 5 / 79 | 9.1 s | 9.5 tok/s |

The answers read like the float model's.  For the cats photo, "The image
depicts two cats lying on a pink surface. The cat on the left is smaller
and appears to be a mutt …" is word for word the study emulation's.

**Open items:**
- **Vision speed:** 7.7 s per image.
  - The host softmax (2.3 s) could use more threads or a narrower exp
    table.
  - P.V (2.1 s) is weight-request-latency bound, like BERT's.
  - The MatMuls reach only 24 GMAC/s against BERT's 44: K = 768 is short.
- **Image splitting:** image splitting (more tiles) needs a larger context.

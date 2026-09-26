# Chat app on the KV260 — plan (for review)

Date: 2026-09-26.  Status: **approved 2026-09-26 — decisions: A then B,
SmolLM2-135M-Instruct, server on the board, existing CLIs + `chat.py`, context
1024.**  Phase 1 (server + CLI + backend A) **done** (branch `feat/chatsrv`,
§9); phase 2 (B0 numeric study) started in parallel.  Builds on doc/BERT_PLAN.md (BERT-base SQuAD at 971 ms per
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

**B1. Export.**  Two fixed-shape ONNX graphs from the Hugging Face model
(host-side export with torch / transformers in a separate venv):
- `prefill(ids[1,P], mask[1,P]) → logits of the last position, K/V for P positions`
  with P bucketed (64 / 128 / 256 / 512), padded and masked;
- `decode(id[1,1], pos, K_cache[L][C], V_cache[L][C], mask[1,C]) → logits, k_new[L], v_new[L]`
  with the cache as ordinary graph inputs (C = context length).  The host
  writes `k_new / v_new` into the cache at `pos` (23 KB per token for 135M),
  so the cache never moves.

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
| 2 | B0 numeric study | study script + report | go / no-go with side-by-side float vs Q8.8 generations |
| 3 | B1–B3 decoder | export, scheduler ops, multi-entry project, generation loop; tiny random Llama fixtures in the 148-model board suite | scheduler simulation bit-exact with the study emulation; board logits bit-exact on prefill and 32 decode steps; greedy text identical to the emulation |
| 4 | B4 + server integration | tokenizer, chat template, sampling, streaming | multi-turn chat through `llm`, `aichat`, `chat.py`; tokens/s and TTFT measured |
| 5 | Performance (optional) | split packed-B streaming over HPC0 + HPC1 (~2× decode, bitstream), overlap sampling with the next step, int8 weights (large) | measured tokens/s |

Each phase lands as its own branch and board run, like BERT phases 1–2.

## 7. Expected performance (estimates, 100 MHz, today's bitstream)

| | 135M | 360M |
|---|---:|---:|
| decode, per token (weights / 1.5 GB/s + ~5 ms of calls and host ops) | ~185 ms (5.4 tok/s) | ~490 ms (2 tok/s) |
| prefill, 256-token prompt (ConvKernel, ~40 GMAC/s) | ~0.7 s | ~2 s |
| with weights split over two HPC ports (phase 5) | ~10 tok/s | ~4 tok/s |
| BERT-QA answer, one 256-token window | ~1.0 s | |

## 8. Risks

- **Q8.8 numerics on the decoder** — the reason phase 2 is a go / no-go; if
  the study fails, the fallback is wider activations on the host path or an
  accumulator-shift register (bitstream).
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


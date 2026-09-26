# Chat app on the KV260 — plan (for review)

Date: 2026-09-26.  Status: **proposal — nothing implemented; decisions in §1
are open.**  Builds on doc/BERT_PLAN.md (BERT-base SQuAD at 971 ms per
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

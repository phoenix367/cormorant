# KV260 chat server (OpenAI-compatible)

An OpenAI-compatible chat endpoint served **by the KV260 itself**: a
standard-library Python server on the board (`kv260_chat_server.py`) that
answers `POST /v1/chat/completions` with a model running on this repo's FPGA
kernels, so existing clients — `curl`, the `openai` SDK, `llm`, `aichat` —
and our own zero-install `chat.py` talk to the board directly.  Plan and
decisions: [`doc/CHAT_PLAN.md`](../../doc/CHAT_PLAN.md).

Phase 1 ships one backend, **`bert-squad`**: BERT-base fine-tuned on SQuAD
(the [`bert_squad/`](../bert_squad/) demo's model, 971 ms per 256-token
window on the board, logits bit-exact with the scheduler simulation).  BERT
is an extractive model — it cannot write free text — so the chat is
**question answering over a document you supply**: the reply is the passage
of the document that answers the question.  A generative decoder
(SmolLM2-135M-Instruct, CHAT_PLAN phases 3–4) plugs into the same server as
a second backend.

```
 laptop / board shell                          KV260 (Ubuntu 22.04, Python 3.10 stdlib)
 ┌──────────────────────┐  HTTP, OpenAI API   ┌─────────────────────────────────────────────┐
 │ chat.py · curl       │ ──────────────────▶ │ kv260_chat_server.py  /v1/models            │
 │ openai SDK · llm     │ ◀── JSON / SSE ──── │   /v1/chat/completions  /health             │
 │ aichat               │                     │   FIFO: one request on the FPGA at a time   │
 └──────────────────────┘                     │ bert_squad_backend.py  (squad_text.py:      │
                                              │   WordPiece, sliding windows, best span)    │
                                              │        │ ctypes                              │
                                              │ lib/libbert_squad.so  (bert_api.c + the     │
                                              │   generated BERT project, weights in CMA)   │
                                              │        │ XRT / UIO                           │
                                              │ ConvKernel · MatmulKernel · VectorOPKernel  │
                                              └─────────────────────────────────────────────┘
```

## Files

```
demo/chat/
├── kv260_chat_server.py     — HTTP server, OpenAI protocol, validation, FPGA queue, echo backend
├── chat_backend.py          — the backend interface (Backend, ChatRequest, Delta, Finish, ...)
├── bert_squad_backend.py    — backend A: document / question mapping, windows, libbert_squad.so
├── chat.py                  — stdlib REPL / one-shot client
├── deploy.py                — generate, upload, build, start / stop on the board (host side)
├── chat_config.json.example
└── tests/                   — unittest: protocol, text module, backend; board_gate.py
```

Reused, not copied: `demo/bert_squad/scripts/squad_text.py` (tokenizer,
SQuAD features incl. sliding windows, span decoding — stdlib only, also used
by the study and the demo), `demo/bert_squad/scripts/generate_project.py`
(which now adds the `bert_squad` shared-library target), and
`demo/bert_squad/src/bert_api.{h,c}` (the C API that `squad_bench` also
uses).

## Deploy

Prerequisites are those of the [`bert_squad/`](../bert_squad/) demo (its
config, the model, the HLS driver sources; the weights go to the board once).

```bash
cd demo/chat
cp chat_config.json.example chat_config.json   # defaults: port 8000, no API key
PY=../../inference-scheduler/.venv/bin/python

$PY deploy.py                  # generate (if needed), weights, build, start, wait for /health
$PY deploy.py --status         # unit state + /health
$PY deploy.py --stop           # stop the server (frees the FPGA)
$PY deploy.py --hold           # start, keep the board lock, follow the log; Ctrl-C stops it
```

`deploy.py` takes everything board-specific (ssh, UIO names, weights
directory, board lock) from `../bert_squad/bert_squad_config.json`.  On the
board it installs to `/root/kv260_chat/` (`project/` sources and build,
`lib/libbert_squad.so`, the server files, `vocab.txt`); the build is skipped
when the generated sources and cmake options are unchanged, and
`weights/*.dat` are uploaded only when their size or SHA-1 changed.  The
server runs as a **transient systemd unit** `kv260-chat` (`systemd-run`):
it survives the ssh session, is not started at boot, logs to
`journalctl -u kv260-chat`, and stops on SIGTERM after the running request
(`server.launcher: "nohup"` uses a detached process instead).  First
deploy: 16 s build; startup 1.6 s (`bert_open`: pool BO + 217 MB of weights
from the page cache).

**The FPGA and the board lock.**  The running server owns the FPGA (its
process holds the 216 MiB pool BO and the UIO mappings); nothing else may
run kernels until it is stopped.  `deploy.py` holds the shared board lock
(`board_lock`, `flock`) only while deploying — use `--hold` to keep it for
as long as the server runs, or `deploy.py --stop` before handing the board
to another job.

By hand on the board (for development):

```bash
python3 /root/kv260_chat/kv260_chat_server.py --bert-lib /root/kv260_chat/lib/libbert_squad.so \
    --bert-weights /root/bert_squad_weights --vocab /root/kv260_chat/vocab.txt [--api-key KEY]
python3 kv260_chat_server.py --backend echo --port 8001     # anywhere, no FPGA: protocol only
```

## Talking to it — `bert-squad` conventions

* **The document** is the system message (a leading `Context:` /
  `Document:` label is dropped), or a user message that starts with
  `Context:` — optionally followed by a line `Question: ...` in the same
  message.  Follow-up questions in the same conversation reuse it (clients
  resend the history).
* **The question** is the last user message; a pure `Context:` message is
  acknowledged ("Got the document (… words, … tokens) …"); no document at
  all gets a short usage hint — neither touches the FPGA.
* **The reply** is the best span of the document, as SQuAD reports it: the
  document's whitespace-separated words, punctuation kept ("February 7,
  2016,").
* **Long documents** are split into 256-token windows (question ≤ 64
  tokens, stride 128); each window is one FPGA inference, the best span is
  taken across windows.  At most `--max-windows` (default 8, ≈ 8 s) are run
  per question; beyond that the document is truncated and the response says
  so (`kv260.truncated`, `windows 8/19` in `chat.py`).
* `max_tokens` caps the answer in WordPiece tokens (`finish_reason:
  "length"`), `stop` cuts it; `temperature`, `top_p`, `seed` are accepted and
  ignored (extractive and deterministic).  `usage.prompt_tokens` = real
  tokens over the windows run, `completion_tokens` = answer tokens.
* Every response carries a non-standard `kv260` object (on the last chunk
  when streaming): `windows`, `windows_total`, `truncated`, `window`,
  `span` (token positions in that window), `score` (start + end logit),
  `confidence` (softmax over the n-best), `fpga_ms`, `window_ms`, `n_best`
  (top 5).  SDKs keep it (`response.model_extra["kv260"]` in `openai`).

## Clients

All transcripts below are real, against the board at `192.168.100.8`
(2026-09-26); `sb50.txt` is the first paragraph of SQuAD's "Super Bowl 50"
article.  Any OpenAI-compatible client works with base URL
`http://<board>:8000/v1` and any (or, with `api_key` set, that) API key.

### `chat.py` (ours, stdlib only — laptop or board)

```
$ python3 demo/chat/chat.py --url http://192.168.100.8:8000/v1 --doc sb50.txt -q "Which NFL team represented the AFC at Super Bowl 50?"
Denver Broncos
[1.0 s · 1/1 window(s) · confidence 0.75 · 171 + 2 tokens]

$ printf 'Who won Super Bowl 50?\n/doc sb50.txt\nWho won Super Bowl 50?\nWhat was the final score?\nWhere was the game played?\n/model\n/reset\nWhat color was emphasized for the 50th anniversary?\n/quit\n' \
    | python3 demo/chat/chat.py --url http://192.168.100.8:8000/v1 -v      # input echoed when piped
http://192.168.100.8:8000/v1 · model bert-squad · /help for commands
> Who won Super Bowl 50?
I answer questions about a document by quoting the passage of it that answers them (BERT-base
fine-tuned on SQuAD, running on the KV260 FPGA). Send the document first - as the system message,
or as a user message starting with "Context:" - and then ask questions about it.
[0.0 s · 6 + 65 tokens]
> /doc sb50.txt
(document: sb50.txt, 124 words; conversation cleared)
> Who won Super Bowl 50?
Denver Broncos
[1.0 s · 1/1 window(s) · confidence 0.50 · 166 + 2 tokens]
> What was the final score?
24–10
[1.0 s · 1/1 window(s) · confidence 0.97 · 166 + 3 tokens]
> Where was the game played?
Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
[1.1 s · 1/1 window(s) · confidence 0.34 · 166 + 15 tokens]
> /model
models: bert-squad; using bert-squad
> /reset
(conversation cleared)
> What color was emphasized for the 50th anniversary?
"golden
[1.0 s · 1/1 window(s) · confidence 0.39 · 169 + 1 tokens]
> /quit
```

(Interactively the same, with line editing and history via `readline`.)

Options: `--url` (or `KV260_CHAT_URL`), `--api-key` (or `KV260_CHAT_API_KEY`
/ `OPENAI_API_KEY`), `--model`, `--doc FILE`, `--system TEXT`, `-q QUESTION`
(one-shot; exit code 1 on errors), `--no-stream`, `--max-tokens`,
`--temperature`, `-v`.  Commands: `/doc FILE`, `/system [TEXT]`,
`/model [M]`, `/reset`, `/help`, `/quit`.  On the board:
`python3 /root/kv260_chat/chat.py --doc sb50.txt`.

### `curl`

```
$ curl -s http://192.168.100.8:8000/v1/models
{"object": "list", "data": [{"id": "bert-squad", "object": "model", "created": 1790403749, "owned_by": "kv260"}]}

$ cat req.json
{"model": "bert-squad", "messages": [{"role": "system", "content": "Super Bowl 50 was an American football game ..."},
                                     {"role": "user", "content": "Who was the AFC champion?"}]}
$ curl -s http://192.168.100.8:8000/v1/chat/completions -H "Content-Type: application/json" -d @req.json
{"id": "chatcmpl-02793ebde1dd44a5ab12aa32", "object": "chat.completion", "created": 1790403879,
 "model": "bert-squad", "system_fingerprint": "kv260-bertsquad12-q8.8",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "Denver Broncos", "refusal": null},
              "logprobs": null, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 166, "completion_tokens": 2, "total_tokens": 168},
 "kv260": {"backend": "bert-squad", "windows": 1, "windows_total": 1, "truncated": false,
           "fpga_ms": 973.1, "window_ms": [973.1], "window": 0, "span": [41, 42], "score": 14.0508,
           "confidence": 0.8557, "n_best": [{"text": "Denver Broncos", "score": 14.0508, "prob": 0.8557},
           {"text": "Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers",
            "score": 12.1445, "prob": 0.1272}, ...]}}

$ curl -sN http://192.168.100.8:8000/v1/chat/completions -H "Content-Type: application/json" \
       -d @req_stream.json      # same messages, "What was the final score?", "stream": true,
                                # "stream_options": {"include_usage": true}
data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", "created": 1790403880, "model": "bert-squad", "system_fingerprint": "kv260-bertsquad12-q8.8", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}], "usage": null}

data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", ..., "choices": [{"index": 0, "delta": {"content": "24–10"}, "logprobs": null, "finish_reason": null}], "usage": null}

data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", ..., "choices": [{"index": 0, "delta": {}, "logprobs": null, "finish_reason": "stop"}], "usage": null, "kv260": {"windows": 1, "fpga_ms": 969.7, "span": [54, 56], "confidence": 0.9735, ...}}

data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", ..., "choices": [], "usage": {"prompt_tokens": 166, "completion_tokens": 3, "total_tokens": 169}}

data: [DONE]

$ curl -s http://192.168.100.8:8000/v1/chat/completions -H "Content-Type: application/json" \
       -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}' -w ' HTTP %{http_code}\n'
{"error": {"message": "The model 'gpt-4o' does not exist or you do not have access to it.", "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}
 HTTP 404
$ curl -s ... -d '{"model":"bert-squad","messages":[{"role":"user","content":"hi"}],"temperature":5}' -w ' HTTP %{http_code}\n'
{"error": {"message": "Invalid 'temperature': expected a value <= 2, but got 5 instead.", "type": "invalid_request_error", "param": "temperature", "code": "invalid_value"}}
 HTTP 400
```

### The official `openai` Python SDK

```python
# sdk_example.py — OPENAI_BASE_URL / OPENAI_API_KEY come from the environment
from openai import OpenAI, NotFoundError, BadRequestError

client = OpenAI()
doc = open("sb50.txt").read()
print("models:", [m.id for m in client.models.list()])

r = client.chat.completions.create(
    model="bert-squad",
    messages=[{"role": "system", "content": doc},
              {"role": "user", "content": "Which team did the Broncos defeat?"}])
print("answer:", r.choices[0].message.content, "| finish:", r.choices[0].finish_reason,
      "| usage:", r.usage.prompt_tokens, "+", r.usage.completion_tokens,
      "| confidence:", r.model_extra["kv260"]["confidence"])

stream = client.chat.completions.create(
    model="bert-squad", stream=True, stream_options={"include_usage": True},
    messages=[{"role": "user", "content": "Context: " + doc + "\nQuestion: Where was the game played?"}])
parts = []
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        parts.append(chunk.choices[0].delta.content)
    if chunk.usage:
        print("stream usage:", chunk.usage.total_tokens)
print("streamed:", "".join(parts))

try:
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
except NotFoundError as e:
    print("404:", e.body["message"])
try:
    client.chat.completions.create(model="bert-squad", n=2, messages=[{"role": "user", "content": "hi"}])
except BadRequestError as e:
    print("400:", e.body["message"])
```

```
$ OPENAI_BASE_URL=http://192.168.100.8:8000/v1 OPENAI_API_KEY=none python sdk_example.py
models: ['bert-squad']
answer: Carolina Panthers | finish: stop | usage: 167 + 2 | confidence: 0.9962
stream usage: 181
streamed: Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
404: The model 'gpt-4o' does not exist or you do not have access to it.
400: Invalid 'n': this server generates one choice per request (n = 1).
```

(`openai` 3.19.2.)

### `llm` (Simon Willison, `pip install llm`)

`extra-openai-models.yaml` in llm's config directory (`dirname "$(llm logs path)"`,
e.g. `~/.config/io.datasette.llm/`):

```yaml
- model_id: kv260
  model_name: bert-squad
  api_base: "http://192.168.100.8:8000/v1"
```

```
$ llm models | grep kv260
OpenAI Chat: kv260
$ llm -m kv260 -s "$(cat sb50.txt)" "Who won Super Bowl 50?"
Denver Broncos
$ llm -m kv260 --no-stream -s "$(cat sb50.txt)" "What was the final score?"
24–10
$ llm -c "Where was the game played?"          # continues the last conversation (same document)
Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.

$ printf 'Who won Super Bowl 50?\nWho did they beat?\nWhen was the game played?\nexit\n' \
    | llm chat -m kv260 -s "$(cat sb50.txt)"            # llm does not echo piped input
Chatting with kv260
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
Type '!edit' to open your default editor and modify the prompt
Type '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments
> Denver Broncos
> Carolina Panthers
> February 7, 2016,
> 
```

(`llm` 0.36.)  Scripts: `llm` reads a piped / redirected stdin and prepends
it to the prompt, so in non-interactive shells run it with `</dev/null` —
otherwise it waits for stdin.  Add `api_key_name: kv260` to the YAML and
`llm keys set kv260` when the server has an API key.

### `aichat` (single Rust binary; x86_64 and aarch64 releases)

`~/.config/aichat/config.yaml` (or `$AICHAT_CONFIG_DIR/config.yaml`):

```yaml
model: kv260:bert-squad
stream: true
save: false
clients:
  - type: openai-compatible
    name: kv260
    api_base: http://192.168.100.8:8000/v1
    # api_key: <key>            # when the server has one
    models:
      - name: bert-squad
        max_input_tokens: 100000
```

```
$ aichat --list-models
kv260:bert-squad
$ aichat --prompt "$(cat sb50.txt)" "Who won Super Bowl 50?"
Denver Broncos
$ aichat -S --prompt "$(cat sb50.txt)" "What was the final score?"      # -S: no streaming
24–10
$ aichat "Context: $(cat sb50.txt)
Question: Where was the game played?"
Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.

$ aichat                          # REPL (driven through a pseudo-terminal; escape codes removed)
Welcome to aichat 0.30.0
Type ".help" for additional help.
> .prompt Super Bowl 50 was an American football game to determine the champion of ...
%%> Who won Super Bowl 50?
Denver Broncos
%%> Who did they beat?
Carolina Panthers
%%> When was the game played?
February 7, 2016,
%%> .exit
```

`.prompt TEXT` starts a temporary role (`%%`) with TEXT as the system
message — the document.

(`aichat` 0.30.0, `aichat-v0.30.0-x86_64-unknown-linux-musl`.)  `aichat -f
FILE` puts the file into the *user* message without a `Context:` label, so
use `--prompt` / `.prompt` for the document.

### Not `ollama run`

`ollama run` speaks **Ollama's native API** (`/api/chat`, `/api/show`,
`/api/tags`, NDJSON streaming), not the OpenAI one, so pointing it at this
server with `OLLAMA_HOST` does not work.  It would need a separate ~200-line
shim over the same backends (CHAT_PLAN §5; not implemented).  Use any
OpenAI-compatible client instead.

## API reference

| | |
|---|---|
| `GET /health` | `{"status": "ok"\|"error", "version", "models": [{id, ready, library, model_name, weights_dir, seq_len, max_windows, windows_run}], "busy", "waiting", "requests", "uptime_s"}`; 503 when a model failed to load; no API key needed |
| `GET /v1/models`, `GET /v1/models/{id}` | model list / object (`id`, `object`, `created`, `owned_by`) |
| `POST /v1/chat/completions` | `stream: false` → `chat.completion`; `stream: true` → SSE `chat.completion.chunk` lines (role chunk, content chunks, a final chunk with `finish_reason`, a usage chunk with `stream_options.include_usage`), then `data: [DONE]`; chunked transfer (HTTP/1.1, keep-alive) or connection close (HTTP/1.0) |

The paths also work without `/v1`.  Honoured: `model` (default: the first
backend), `messages` (string content or text parts), `stream`,
`stream_options.include_usage`, `max_tokens` / `max_completion_tokens`,
`temperature` (0–2), `top_p` (0–1), `stop` (≤ 4), `seed`, `n` (1 only);
anything else is ignored.  Errors are OpenAI-shaped
`{"error": {"message", "type", "param", "code"}}`: 400 invalid request
(`param` names the field), 401 missing / wrong API key (`invalid_api_key`),
404 unknown model (`model_not_found`) or URL (`unknown_url`), 411 / 413 body
without length / over 4 MB, 500 backend failure (`backend_error`; an SSE
`data: {"error": ...}` event once a stream has started), 503 busy
(`server_busy`, `Retry-After: 5`) or model not loaded.

**Concurrency.**  One request at a time runs on the FPGA; others wait in a
FIFO (in the order they reach it, after host-side preparation such as
tokenizing) for up to `--queue-timeout` (120 s) and at most `--max-queue`
(16) at once — beyond that 503.  A client that disconnects is noticed
without writing to it (a non-blocking peek at its socket): while queued it
leaves the queue, while running it is cancelled at the next step (the next
256-token window here).  One log line per request:

```
2026-09-26 06:29:58 192.168.100.7 POST /v1/chat/completions 200 model=bert-squad stream=1 prompt=2048 completion=3 windows=8/19 fpga_ms=7751 conf=0.93 queue=297ms ttft=8062ms tok/s=1805.6 total=8063ms finish=stop
2026-09-26 06:29:25 192.168.100.7 POST /v1/chat/completions 499 model=bert-squad stream=0 queue=31ms total=2938ms [cancelled (client disconnected)]
```

## Backend interface (for the decoder backend)

`chat_backend.py` holds the contract; a backend serves one model id:

```python
class MyBackend(Backend):
    model_id = "smollm2-135m-instruct"
    def load(self): ...                  # once, before the server listens (dlopen, init)
    def prepare(self, req: ChatRequest): # outside the FPGA lock: chat template, tokenize,
        return job                       # validate (raise BackendError -> 4xx)
    def generate(self, job, cancel):     # under the FPGA lock
        for ...:
            cancel.check()               # raises Cancelled when the client left
            yield Delta(text)            # streamed as a chunk as soon as it is yielded
        yield Finish("stop" | "length", prompt_tokens, completion_tokens, info={...})
    def health(self): return {...}      # extra /health fields
    def close(self): ...
```

`ChatRequest` carries the validated `messages`, `max_tokens`,
`temperature`, `top_p`, `stop`, `seed`, `stream` and the raw JSON;
`StopStream` does incremental stop-string matching for token streams;
`Finish.info` is returned under `kv260` (its `log` dict goes to the log
line).  Register the backend in `build_backends()` in
`kv260_chat_server.py`.  The server does streaming, usage, errors, the
queue and disconnects.

## Measured (2026-09-26, KV260 at 100 MHz, bitstream of BERT phase 2)

* **Same spans as the demo:** the 12 questions of a `bert_squad`
  `deploy_and_run.py --n 12` run sent through the server (context as the
  system message) — **12 / 12 identical spans** (text and token positions;
  `tests/board_gate.py`).  The demo run itself, with `squad_bench` now on
  `bert_api.c`: bit-exact with the simulation 3 / 3 and the emulation
  12 / 12, 966.7 ms per inference, logits byte-identical to main's earlier
  run.
* **Latency:** 971 ms of FPGA time per window (mean over the 12), request
  latency 1027 ms (tokenizing, JSON, network); startup 1.6 s; the server
  process holds ~221 MB of CMA (CmaFree 811 → 590 MB).
* **Long document:** "Super Bowl 50" paragraphs 0–9 (848 words, 1048
  tokens → 8 windows), 10 questions whose answers sit in different
  paragraphs: **EM 90.0 / F1 94.2**, 7.8–8.0 s per question (970 ms per
  window); paragraphs 0–15 need 19 windows → truncated to 8 and reported
  (`windows 8/19`).
* **Disconnect:** a client that drops an 8-window request after 2.5 s
  frees the FPGA after the current window (request logged at 2.9 s,
  cancelled); the three requests queued behind it ran next.

## Tests

```bash
cd demo/chat/tests
python3 -m unittest -v                  # 48 tests, ~20 s, stdlib only (numpy optional)
python3 board_gate.py --url http://<board>:8000/v1     # against a running server
```

`test_protocol.py` checks the `chat.completion` / `chat.completion.chunk`
schema field by field, errors, the FIFO queue and its timeout, and
disconnects (streaming, non-streaming, while queued) with a fake backend;
`test_squad_text.py` checks the sliding windows, max-context flags and
cross-window span decoding against independent reference implementations;
`test_bert_backend.py` the document / question mapping, windows and their
cap, `max_tokens` / `stop`, cancellation between windows and — when
`demo/bert_squad/build/{logits.bin,results.json}` exist — that replaying the
demo's board logits gives the demo's spans.

**Span decoding change (study / demo).**  `best_span` moved to
`squad_text.py` and ranks equal logits by position (the order of a stable
argsort) instead of numpy's unstable `argsort`, whose tie order depends on
the numpy build.  On every real logit set checked — the demo's board,
float and emulation logits (60 / 60) — and in the study (`bert_study.py
--policies q88,sched --n 5`), `prepare_inputs.py` (byte-identical inputs)
and `bert_sched_check.py --n 2` the results are unchanged; on synthetic
tie-heavy logits 2844 / 3000 spans are unchanged and the rest differ only
by that tie order (3000 / 3000 against the old code with a stable sort).

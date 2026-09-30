# Development

The source files, how to add a backend, the tests, and the host-side
validation against transformers.

- [Source files](#source-files)
- [Adding a backend](#adding-a-backend)
- [Tests](#tests)
- [Host validation](#host-validation)

## Source files

```
demo/chat/
├── kv260_chat_server.py     — HTTP server, OpenAI protocol, validation, FPGA queue, residency, echo backend
├── chat_backend.py          — the backend interface (Backend, ChatRequest, Delta, Finish, ...)
├── bert_squad_backend.py    — document / question mapping, windows, libbert_squad.so
├── smollm2_backend.py       — SmolLM2: prompt, prefix cache, decode loop, libsmollm2.so (CHAT_PLAN §11 API)
├── smolvlm_backend.py       — SmolVLM: images (llm_image), prefix cache keyed by image content
├── idefics3.py, vlm_image.py — SmolVLM's chat template with images; data URL -> 512 x 512 pixels (Pillow)
├── piper_backend.py         — text to speech: front end, libpiper_tts.so per chunk
├── piper_phonemize.py       — espeak-ng through ctypes -> Piper phoneme ids (+ ../tts/scripts/piper_vits.py)
├── smollm2_tokenizer.py     — SmolLM2's byte-level BPE (tokenizer.json) + incremental detokenizer
├── chatml.py                — SmolLM2's ChatML template, block-wise tokenizing, history trimming
├── sampler.py, src/sampler.{c,h} — sampling (libsampler.so; pure-Python fallback, same tokens)
├── chat.py                  — stdlib REPL / one-shot client
├── deploy.py                — generate, upload, build, start / stop on the board (host side)
├── chat_config.json.example
├── src/llm_api.{c,h}, src/llm_bench.c — the LLM libraries' C API (CHAT_PLAN §11) and a standalone benchmark
├── scripts/
│   ├── generate_llm_project.py — schedule SmolLM2 / SmolVLM into build/llm_project[_<model>]
│   │                          (libsmollm2.so, ...); takes the planning options (--plan, ...)
│   ├── llm_board.py         — build / install the library on the board, board gate, timings,
│   │                          per-layer profiles (--profile; --out saves them as profile_layers)
│   ├── llm_project.py       — shared model / formats loading, library simulation (SimSession)
│   ├── llm_sched_check.py   — scheduler simulation == study emulation, bit for bit (gate 2)
│   ├── llm_host_emu.py      — the generated C on the host against software kernels
│   ├── llm_lib_check.py     — libsmollm2.so through ctypes (runs on the board)
│   ├── llm_attn_kernel_bench.py — decode attention on the FPGA vs host (CHAT_PLAN §13.4)
│   ├── llm_study.py         — numeric study, calibrated formats JSON (.venv-export)
│   ├── llm_calibrate.py     — the study stage from pinned inputs: fetch, calibrate, check, study
│   ├── llm_models.json      — pinned checkpoints / texts (SHA-256) and the expected formats hashes
│   ├── requirements-study.txt — the .venv-export versions that reproduce those hashes
│   ├── vlm_study.py         — SmolVLM-256M numeric study (vision encoder + image prompts, CHAT_PLAN §22)
│   ├── vlm_project.py, vlm_sched_check.py, vlm_host_emu.py — SmolVLM entries, gate (sim == study), C on the host
│   ├── vlm_template_fixture.py — tests/data/smolvlm_cases.json (template ids, image pixels; .venv-export)
│   ├── vlm_study_inputs.json — its pinned checkpoint revision, file and COCO image hashes
│   └── validate_text.py, e2e_check.py — tokenizer / template / greedy answers vs transformers (.venv-export)
├── doc/                     — these guides
├── assets/                  — not in git: <model>/ (Hugging Face checkpoint, texts), study/
└── tests/                   — unittest: protocol, text modules, sampler, backends, fakes; board_gate.py
```

**Reused from other demos, not copied:**
- `demo/bert_squad/scripts/squad_text.py`: the WordPiece tokenizer, SQuAD
  features (incl. sliding windows), span decoding.  It is stdlib-only and
  also used by the BERT study and demo.  Its span decoding ranks equal
  logits by position; the
  [bert_squad README](../../bert_squad/README.md#how-it-works) records why
  and what that changed.
- `demo/bert_squad/scripts/generate_project.py`: generates the BERT
  project, including the `bert_squad` shared-library target.
- `demo/bert_squad/src/bert_api.{h,c}`: the C API that `squad_bench` also
  uses.
- `demo/tts/`: the Piper library (`libpiper_tts.so`) and
  `scripts/piper_vits.py`, the Piper front end.

## Adding a backend

A backend serves one model id.  `chat_backend.py` holds the contract:

```python
class MyBackend(Backend):
    model_id = "my-model"
    cma_mb = 300.0                       # CMA held while loaded (--resident auto)
    def load_host(self): ...             # at startup: tokenizer etc. (no FPGA)
    def load(self): ...                  # at startup or before its first request (dlopen, init)
    def unload(self): ...                # evicted for another model: free the FPGA / CMA
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

- **`ChatRequest`** carries the validated `messages`, `max_tokens`,
  `temperature`, `top_p`, `stop`, `seed`, `stream` and the raw JSON.
- **`StopStream`** does incremental stop-string matching for token
  streams.
- **`Finish.info`** is returned under `kv260`; its `log` dict goes to the
  log line.

Register the backend in `build_backends()` in `kv260_chat_server.py`.  The
server does the rest: streaming, usage, errors, the queue and disconnects.
A text-to-speech backend sets `speech = True` and implements
`prepare_speech(req)` (outside the FPGA lock) and `synthesize(job, cancel)`
(under the lock, yielding audio bytes and a final `Finish`) instead; see
`piper_backend.py`.

## Tests

```bash
cd demo/chat/tests
python3 -m unittest -v                  # 185 tests, ~45 s
python3 board_gate.py --url http://<board>:8000/v1     # against a running server
```

The suite also runs under pytest from the repo root:
`inference-scheduler/.venv/bin/python -m pytest demo/chat/tests -q`.

**What the tests need.**  The core runs on the standard library alone.
Some parts need more:
- a C compiler for the C parts;
- numpy and paramiko for the deploy tests;
- Pillow and the SmolVLM tokenizer for the image tests;
- numpy, ffmpeg and libespeak-ng for the speech tests.

About 60 tests skip until their assets are present:
- the SmolLM2 tokenizer: `scripts/llm_calibrate.py fetch`;
- the SmolVLM tokenizer: `scripts/vlm_study.py fetch`;
- BERT's `vocab.txt`: `../bert_squad/scripts/fetch_assets.py vocab`.

**What each test module covers.**

Server and protocol:
- `test_protocol.py` — the `chat.completion` / `chat.completion.chunk`
  schema field by field, errors, the FIFO queue and its timeout, and
  disconnects (streaming, non-streaming, while queued), with a fake
  backend.

Document Q&A:
- `test_squad_text.py` — sliding windows, max-context flags and
  cross-window span decoding, against independent reference
  implementations.
- `test_bert_backend.py` — the document / question mapping, windows and
  their cap, `max_tokens` / `stop`, cancellation between windows.  When
  `demo/bert_squad/build/{logits.bin,results.json}` exist, it also checks
  that replaying the demo's board logits gives the demo's spans.

Generative chat:
- `test_smollm2_text.py` — the tokenizer, the detokenizer and trimming.
  - Tokenizer ids and decoding against 500 transformers-tokenized strings
    and 34 template conversations (`tests/data/smollm2_text_cases.json`,
    written by `validate_text.py --write-fixture`).
  - The incremental detokenizer under every split.
  - The trimming rules.
- `test_sampler.py` — the sampler.
  - Greedy equals argmax.
  - Penalties, top-k and top-p masks against independent references.
  - Sampled frequencies with fixed seeds, and determinism.
  - C equals Python, token for token.
- `test_smollm2_backend.py` — the backend against `tests/fake_llm.py`
  (scripted logits) and `tests/fake_libsmollm2.c` (the same library
  contract as a C library, through the real ctypes binding).  It covers:
  - the streaming schema;
  - multi-turn prefix reuse: exactly the new tokens are prefilled, the
    sink never;
  - stop strings across tokens, `max_tokens`, cancellation, a full
    context and trimming;
  - UTF-8 across tokens, seeds and parameters, library errors;
  - residency: `one` / `auto` / `all` switching between `bert-squad` and
    `smollm2`, with fake engines and a fake CMA pool.
- `test_llm_calibrate.py` — the calibration tool.
  - `llm_models.json`'s schema.
  - Verified downloads and fetch, with file:// standing in for Hugging
    Face: kept, replaced, copied and rejected files.
  - The formats diff.
  - `calibrate` / `check` against a fake `llm_study.py`: installed only on
    a reproduced or recorded hash, the provenance, `check` writing
    nothing.

Images:
- `test_smolvlm_backend.py` — the image side.
  - The Idefics3 template against transformers' processor (7
    conversation shapes).
  - The image pixels against the study's (Pillow's LANCZOS).
  - The backend over a fake engine: `llm_image()` before each new image
    block, image rows as vocab + k, prefix reuse keyed by image content,
    trimming with images.
  - Image parts over HTTP, and a text model's 400.

Speech:
- `test_speech.py` — `/v1/audio/speech` with a fake speech backend:
  - wav (a real WAV header and Content-Length) and pcm;
  - mp3 / flac / opus / aac through ffmpeg;
  - SSE deltas and usage;
  - aliases and the default model;
  - every validation error, the unknown-length WAV, HTTP/1.0;
  - a client leaving mid-audio;
  - speech and chat sharing the FPGA queue.
- `test_piper_backend.py` — the Piper backend over a fake library and
  front end (sentence packing, the chunk loop, seeds, cancellation, the log
  line), and `piper_phonemize.py` against espeak-ng when it is installed
  (clause terminators, Piper's ids).

Client and deploy:
- `test_chat_client.py` — `chat.py`'s audio: streaming into a recording
  player; `--say`, `-q --audio`, `--audio-out`; `/audio`, `/audio save`,
  `/say`; the WAV fallback; the spoken-text cleanup.
- `test_deploy_vocab.py` — `deploy.py` fetches BERT's `vocab.txt` before
  anything else.
  - It downloads to the BERT config's path, before the project is
    generated and before the board connection.
  - It keeps an existing file.
  - A failed download stops the deploy.
  - `--check-only` downloads nothing.

  It needs numpy and paramiko, and is skipped without them.

**The board gate.**  `tests/board_gate.py` runs against a live server
(stdlib only, from the host or the board):
1. It sends the questions of the `bert_squad` demo's board run
   (`demo/bert_squad/build/results.json`) through the server.  The spans
   must equal the demo's exactly: text and token positions.
2. It asks questions about a long document, the first `--paragraphs`
   paragraphs of the "Super Bowl 50" article, whose answers sit in
   different paragraphs.  It reports the answer, the gold answer, the
   windows and the latency.

## Host validation

These scripts check the server's text handling and answers against
transformers.  They run in `.venv-export`, a virtualenv with torch,
transformers, safetensors and numpy ([Generative chat](SMOLLM2.md#install-smollm2-135m)),
from the repo root:

```bash
PY=.venv-export/bin/python
$PY demo/chat/scripts/validate_text.py  # tokenizer on 2960 strings, decode, 34 chat templates
$PY demo/chat/scripts/e2e_check.py      # chat.py -> server (--llm-fake float) == HF greedy, 9 answers
```

**Tokenizer reference.**  `validate_text.py` compares against transformers'
`TokenizersBackend.from_pretrained`.  That is the `tokenizer.json`
pipeline: what SmolLM2 was trained with, and what transformers 4.x
`AutoTokenizer` returns.  The results: 2960 / 2960 strings, 2000 / 2000
random id sequences and 68 / 68 template renderings are identical.

transformers **5.x** `AutoTokenizer` instead builds a `GPT2Tokenizer` class
that **drops tokenizer.json's `Digits` pre-tokenizer**.  It differs on 249
of the strings, all of them explained by that: a numeral after two or more
whitespace characters (e.g. `"  1"` → `Ġ Ġ 1` instead of `ĠĠ 1`), or
non-ASCII numerals.  `e2e_check.py` uses the faithful pipeline.

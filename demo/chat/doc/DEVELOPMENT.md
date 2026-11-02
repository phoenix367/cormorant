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
├── stories_backend.py       — stories15m: a base model's text completion (/v1/completions) and story chat
├── idefics3.py, vlm_image.py — SmolVLM's chat template with images; data URL -> 512 x 512 pixels (Pillow)
├── piper_backend.py         — text to speech: front end, libpiper_tts.so per chunk
├── piper_phonemize.py       — espeak-ng through ctypes -> Piper phoneme ids (+ ../tts/scripts/piper_vits.py)
├── moonshine_backend.py     — speech to text: libmoonshine_<size>.so per part, detokenizing with hf_tokenizer.py
├── speech_input.py          — uploads -> 16 kHz mono int16 (WAV here, a windowed-sinc resampler; ffmpeg otherwise)
├── hf_tokenizer.py          — every model's tokenizer: the tokenizer.json pipeline as `tokenizers` runs it
│                              (BPE; normalizers, pre-tokenizers, added tokens, decoders, post-processor;
│                              Oniguruma regexes as Python re with Unicode 16 tables) + incremental decoder
├── smollm2_tokenizer.py, llama_tokenizer.py — hf_tokenizer.py under the names the SmolLM2 / stories backends
│                              import (BOS / EOS fallbacks <|im_start|> / <|im_end|>, <s> / </s>)
├── chatml.py                — SmolLM2's ChatML template, block-wise tokenizing, history trimming
├── sampler.py, src/sampler.{c,h} — sampling (libsampler.so; pure-Python fallback, same tokens)
├── chat.py                  — stdlib REPL / one-shot client
├── deploy.py                — generate, upload, build, start / stop on the board (host side)
├── chat_config.json.example
├── src/llm_api.{c,h}, src/llm_bench.c — the LLM libraries' C API (CHAT_PLAN §11) and a standalone benchmark
├── scripts/
│   ├── generate_llm_project.py — schedule SmolLM2 / SmolVLM / stories15m into build/llm_project[_<model>]
│   │                          (libsmollm2.so, ...); takes the planning options (--plan, ...)
│   ├── llm_board.py         — build / install the library on the board, board gate, timings,
│   │                          per-layer profiles (--profile; --out saves them as profile_layers)
│   ├── llm_project.py       — shared model / formats loading, library simulation (SimSession)
│   ├── llm_model.py         — a checkpoint's names (tag, project, board paths, study dir) and
│   │                          profile (chat / text, sink, end ids, context) from its own files
│   ├── llm_sched_check.py   — scheduler simulation == study emulation, bit for bit (gate 2)
│   ├── llm_host_emu.py      — the generated C on the host against software kernels
│   ├── llm_lib_check.py     — libsmollm2.so through ctypes (runs on the board)
│   ├── llm_attn_kernel_bench.py — decode attention on the FPGA vs host (CHAT_PLAN §13.4)
│   ├── llm_study.py         — numeric study, calibrated formats JSON (.venv-export); a chat or a
│   │                          text profile (a base model without a chat template: story prompts)
│   ├── llm_calibrate.py     — the study stage from pinned inputs: fetch, calibrate, check, study
│   ├── llm_models.json      — pinned checkpoints / texts (SHA-256) and the expected formats hashes
│   ├── requirements-study.txt — the .venv-export versions that reproduce those hashes
│   ├── vlm_study.py         — SmolVLM-256M numeric study (vision encoder + image prompts, CHAT_PLAN §22)
│   ├── vlm_project.py, vlm_sched_check.py, vlm_host_emu.py — SmolVLM entries, gate (sim == study), C on the host
│   ├── vlm_template_fixture.py — tests/data/smolvlm_cases.json (template ids, image pixels; .venv-export)
│   ├── vlm_study_inputs.json — its pinned checkpoint revision, file and COCO image hashes
│   ├── validate_text.py, e2e_check.py — tokenizer / template / greedy answers vs transformers (.venv-export;
│   │                          --model stories15m / --assets DIR / --all: hf_tokenizer.py vs tokenizers)
│   └── tokenizer_assets.json — the tokenizers of other families validate_text.py --all checks (revision + SHA-256)
├── doc/                     — these guides
├── assets/                  — not in git: <model>/ (Hugging Face checkpoint, texts), study/, tokenizers/<name>/
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
- `demo/asr/`: the Moonshine libraries (`libmoonshine_<size>.so`,
  `scripts/generate_asr_project.py`, `src/asr_api.{c,h}`) and their gates
  (`asr_host_emu.py`, `asr_board.py`).

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
- **A backend that cannot start** raises from `load_host()` or `load()`
  with a message that says why: at startup the server drops that backend
  alone and serves the others
  ([A model that fails to start](DEPLOY.md#choosing-the-models)).

Register the backend in `build_backends()` in `kv260_chat_server.py`.  The
server does the rest: streaming, usage, errors, the queue and disconnects.
A text-to-speech backend sets `speech = True` and implements
`prepare_speech(req)` (outside the FPGA lock) and `synthesize(job, cancel)`
(under the lock, yielding audio bytes and a final `Finish`) instead; see
`piper_backend.py`.  A speech-to-text backend sets `transcription = True`
and implements `prepare_transcription(req)` (outside the lock: decode the
upload) and `transcribe(job, cancel)` (under the lock, yielding a
`Segment` per part and a final `Finish` whose `info["text"]` is the
transcript); see `moonshine_backend.py`.

## Tests

```bash
cd demo/chat/tests
python3 -m unittest -v                  # 334 tests, ~45 s
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

About 70 tests skip until their assets are present:
- the SmolLM2 tokenizer: `scripts/llm_calibrate.py fetch`;
- the SmolVLM tokenizer: `scripts/vlm_study.py fetch`;
- the stories15m tokenizer: `scripts/llm_calibrate.py fetch stories15m`;
- the tokenizers of other model families (Qwen2.5, TinyLlama, Phi-3,
  Llama-3.2, Gemma-3, Mistral): `scripts/validate_text.py --fetch`;
- BERT's `vocab.txt`: `../bert_squad/scripts/fetch_assets.py vocab`.

**What each test module covers.**

Server and protocol:
- `test_protocol.py` — the `chat.completion` / `chat.completion.chunk`
  schema field by field, errors, the FIFO queue and its timeout, and
  disconnects (streaming, non-streaming, while queued), with a fake
  backend.  A backend whose host side or startup load fails is dropped:
  left out of `/v1/models`, listed under `/health`'s `unavailable`, 503
  with the reason, never loaded or evicted again (`--resident one` /
  `auto` / `all`), exit 1 when none is left.

Document Q&A:
- `test_squad_text.py` — sliding windows, max-context flags and
  cross-window span decoding, against independent reference
  implementations.
- `test_bert_backend.py` — the document / question mapping, windows and
  their cap, `max_tokens` / `stop`, cancellation between windows.  When
  `demo/bert_squad/build/{logits.bin,results.json}` exist, it also checks
  that replaying the demo's board logits gives the demo's spans.

Generative chat:
- `test_hf_tokenizer.py` — `hf_tokenizer.py`, the tokenizer of every model.
  - Each pipeline part on small synthetic `tokenizer.json` dicts, with the
    values the `tokenizers` library gives for them: the Oniguruma → `re`
    translation (Qwen2 / Llama-3 split patterns, case folding, Unicode 16
    classes, empty matches), the normalizers (NFC / NFKC with the
    reference's Unicode 9 data, Lowercase), every Split behaviour,
    ByteLevel, Digits, Metaspace, BPE options (byte fallback, unk fusing,
    `ignore_merges`, subword prefix / suffix), added-token flags and ids,
    the decoders and the incremental decoder under every split, the
    post-processors.
  - Every unsupported part raises `ValueError` naming it; the Unicode
    tables agree with `unicodedata`; the compatibility shims.
  - The tokenizers of other families (`validate_text.py --fetch`) against
    the reference ids in `tests/data/hf_tokenizer_cases.json`
    (`validate_text.py --all --write-fixture`).
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
  contract as a C library, its sizes compile-time constants as the real
  libraries', through the real ctypes binding).  It covers:
  - the streaming schema;
  - multi-turn prefix reuse: exactly the new tokens are prefilled, the
    sink never;
  - stop strings across tokens, `max_tokens`, cancellation, a full
    context and trimming;
  - UTF-8 across tokens, seeds and parameters, library errors;
  - the library and the tokenizer: a padded vocabulary (more rows than
    tokens) never sampled or reported, a smaller one refused, the context
    known before the model is opened, the end-of-generation ids (none of
    them found: refused); every backend's `cma_mb` fallback is the
    server's default;
  - residency: `one` / `auto` / `all` switching between `bert-squad` and
    `smollm2`, with fake engines and a fake CMA pool.
- `test_stories_backend.py` — stories15m: its tokenizer against 520
  `tokenizers`-tokenized strings (`tests/data/stories15m_text_cases.json`,
  `validate_text.py --model stories15m --write-fixture`), special tokens,
  the continuation detokenizer; the story a conversation tells; the backend
  (prompt trimming, `<s>` / `</s>` ending a story, the leading space kept)
  over a scripted engine and over `fake_libsmollm2.c` through
  `build_backends`; `POST /v1/completions` (schema, `echo`, streaming,
  OpenAI's default `max_tokens`, refused fields, a chat model's 400).
- `test_llm_model.py` — a checkpoint's names and profile (`scripts/llm_model.py`)
  from SmolLM2-, stories15m-, Llama-3- and template-file-shaped checkpoints;
  `llm_study.Profile` refusing a template other than ChatML.
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
  line).
  - The real front end over a stand-in for the library's encoder and
    duration predictor, from a voice directory holding only `voice.json`.
  - A library without `tts_encode` / `tts_duration`, or with fewer ids per
    utterance than the server packs, is refused.
  - `piper_phonemize.py` against espeak-ng when it is installed (clause
    terminators, Piper's ids).

Speech to text:
- `test_transcription.py` — `speech_input.py` (16 kHz PCM passed through
  sample for sample, stereo mixing, 8 / 24 / 32-bit and float WAV, the
  extensible forms, the resampler's tone and stop band, ffmpeg's formats);
  the detokenizer (bytes, specials, the leading space; against the
  `tokenizers` package on Moonshine's tokenizer.json where both exist);
  `MoonshineBackend` over a fake library (parts, cancellation between
  them, the refusals); `/v1/audio/transcriptions` (json / text / srt /
  vtt / verbose_json, SSE, the aliases, every validation error, a client
  leaving between parts).
- The real library through the same backend:
  `demo/asr/scripts/asr_host_emu.py --lib-check` (host emulation, ids
  bit-exact with the specification).

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
$PY demo/chat/scripts/validate_text.py --fetch   # the pinned tokenizers of other families
$PY demo/chat/scripts/validate_text.py --all     # hf_tokenizer.py vs tokenizers: every tokenizer, regexes, parts
$PY demo/chat/scripts/validate_text.py --assets demo/chat/assets/<model>   # a new checkpoint's tokenizer
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

**Every tokenizer.**  `validate_text.py --all` checks `hf_tokenizer.py`
against the `tokenizers` library (0.23) for SmolLM2-135M, SmolVLM-256M,
stories15m, Moonshine base / tiny and the tokenizers pinned in
`scripts/tokenizer_assets.json` (Qwen2.5-0.5B, TinyLlama-1.1B, Phi-3-mini,
Llama-3.2-1B, Gemma-3-270M, Mistral-7B v0.1; `--fetch` downloads only their
tokenizer files into `assets/tokenizers/`): on ~4 000 strings each (the
corpus above, a Unicode corpus — combining sequences, NFC / NFKC and
case-sensitive characters, every script's digits, Unicode 16 / 17
letters, invisible and bidi characters, emoji, CJK — and each tokenizer's
added tokens in context) the ids with and without special tokens and with
special-token strings as text, decode with and without special tokens, the
incremental decoder token by token and split in two, and decoding as a
continuation are identical; so are 2000 random id sequences each.  `--regex`
(also in `--all`) splits the corpus and one string of every code point with
the GPT-2, Qwen2, Llama-3 and GPT-4o patterns and patterns using every
other class: identical to `tokenizers`' Split.  `--synthetic` runs each
supported pipeline part on small `tokenizer.json` dicts.

One difference is deliberate: `decode(skip_special_tokens=True)` skips the
special tokens by id.  `tokenizers` compares their decoded string, so a
special token the normalizer changes (stories15m's `"▁<s>"`) stays in its
output; the check counts those decodes separately.

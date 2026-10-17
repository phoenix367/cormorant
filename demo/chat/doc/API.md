# API reference

The server implements the parts of the OpenAI API that its models need.
This page lists the endpoints, the request fields it honours, its errors,
the `kv260` extension, and how requests share the FPGA.  How each model
interprets a request is in its guide ([Document Q&A](BERT_QA.md),
[Generative chat](SMOLLM2.md), [Chat about images](SMOLVLM.md),
[Text to speech](TEXT_TO_SPEECH.md), [Speech to text](SPEECH_TO_TEXT.md)).

## Endpoints

The paths also work without the `/v1` prefix.

| Endpoint | What it returns |
|---|---|
| `GET /health` | the server's state (below); 503 when a model failed to load; no API key needed |
| `GET /v1/models`, `GET /v1/models/{id}` | the model list / one model object (`id`, `object`, `created`, `owned_by`, and the non-standard `kv260: {"type"}` — `chat`, `speech` or `transcription`: the endpoint the model serves) |
| `POST /v1/chat/completions` | a chat answer: `chat.completion`, or with `stream: true` server-sent events (below) |
| `POST /v1/audio/speech` | speech from the speech backends ([Text to speech](TEXT_TO_SPEECH.md)) |
| `POST /v1/audio/transcriptions` | the transcript of an uploaded recording from the speech-to-text backends ([Speech to text](SPEECH_TO_TEXT.md)) |

**Streaming chat.**  With `stream: true`, the response is a series of
`data: {...}` lines holding `chat.completion.chunk` objects:
1. a chunk with the role;
2. the content chunks;
3. a final chunk with `finish_reason` (and the `kv260` details);
4. with `stream_options.include_usage`, a chunk with the usage;
5. `data: [DONE]`.

The transfer is chunked (HTTP/1.1, keep-alive), or ends by closing the
connection (HTTP/1.0).

**Speech responses.**  The audio arrives as it is synthesized:
- **wav / pcm:** with an exact Content-Length;
- **mp3 / opus / aac / flac:** chunked, through ffmpeg;
- **`stream_format: "sse"`:** `speech.audio.delta` events and a final
  `speech.audio.done` with the usage.

**Transcription responses.**  The request is `multipart/form-data` with the
audio as `file`; the answer is the transcript as `json` (`{"text",
"usage": {"type": "duration", "seconds"}, "kv260"}`), `text`, `srt`, `vtt`
or `verbose_json` (a segment per part of at most 24.6 s); with `stream:
true`, `transcript.text.delta` events (one per part) and a final
`transcript.text.done` with the whole text.

**`/health`** returns:
- `status`: `"ok"` or `"error"`;
- `version`;
- `models`: one entry per model — `id`, `ready`, `loaded`, and backend
  fields:
  - every model: `library`, `weights_dir` (Piper: `voice_dir`);
  - BERT: `model_name`, `seq_len`, `max_windows`, `windows_run`;
  - SmolLM2: `vocab_size`, `context_size`, `reserve`, `sampler`,
    `cached_tokens`, `defaults`, `requests`,
    `prompt/reused/prefilled/generated_tokens`;
  - SmolVLM: `image_cache`;
  - Piper: `type`, `sample_rate`, `espeak_voice`, `requests`, `audio_s`,
    `chunks`, `rtf`, `frontend_s`;
  - Moonshine: `type` (`transcription`), `tokenizer`, `sample_rate`,
    `max_seconds`, `requests`, `audio_s`, `tokens`, `parts`, `rtf`;
- `resident`, `cma_free_mb`, `loads`, `unloads`: see
  [residency](DEPLOY.md#several-models-on-one-fpga);
- `busy`, `waiting`: the FPGA queue;
- `requests`, `uptime_s`.

## Request fields

| Field | Notes |
|---|---|
| `model` | default: the first backend (the first speech backend for `/audio/speech`, the first speech-to-text backend for `/audio/transcriptions`) |
| `messages` | string content or text parts; `image_url` parts (base64 data URLs) for `smolvlm` |
| `stream`, `stream_options.include_usage` | streaming, as above |
| `max_tokens` / `max_completion_tokens` | the answer's length cap |
| `temperature` (0–2), `top_p` (0–1) | sampling (ignored by `bert-squad`) |
| `stop` | up to 4 stop strings |
| `seed` | sampling seed |
| `logprobs`, `top_logprobs` (0–20) | SmolLM2 only, not with `stream`: `choices[0].logprobs.content`, one entry per generated token (`token`, `logprob`, `bytes`, `top_logprobs`) — the model's distribution before the sampler's penalties, as OpenAI returns it |
| `n` | 1 only |
| `top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `repeat_last_n`, `dry_*`, `loop_guard` | the generative models' extra sampling fields ([Sampling](SMOLLM2.md#sampling)) |

Anything else is ignored.  The speech and transcription fields are in
their guides ([Text to speech](TEXT_TO_SPEECH.md#requests),
[Speech to text](SPEECH_TO_TEXT.md#requests): also the non-standard
`kv260_wake_phrase` / `kv260_wake_margin`, which check a phrase at the start
of a recording against the audio).

## Errors

Errors have OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`,
so OpenAI clients raise their usual exceptions.

| Status | When |
|---|---|
| 400 | invalid request; `param` names the field.  `images_not_supported`: an image sent to a text model.  `context_length_exceeded`: a generative prompt that cannot be trimmed to fit |
| 400 `unsupported_format` | a speech format that needs ffmpeg, without ffmpeg |
| 400 `model_not_supported` | a chat model at `/audio/speech` or `/audio/transcriptions`, a speech or speech-to-text model at `/chat/completions` |
| 400 `invalid_audio`, `audio_too_long`, `unsupported_language` | an upload that cannot be decoded (not WAV, without ffmpeg) or holds no samples; longer than `--asr-max-seconds`; a language other than English |
| 401 `invalid_api_key` | missing or wrong API key |
| 404 `model_not_found` / `unknown_url` | unknown model or URL |
| 411 / 413 | a body without a length / over 4 MB (32 MB when an image or speech-to-text model is served; `--max-body-mb`) |
| 500 `backend_error` | the backend failed; once a stream has started, an SSE `data: {"error": ...}` event instead |
| 503 `server_busy` | the queue is full, or the request waited longer than `--queue-timeout` (`Retry-After: 5`) |
| 503 `model_not_loaded` | the model's library could not be loaded |

## The `kv260` object

Every chat response carries a non-standard `kv260` object (on the last
chunk when streaming) with the backend's details: timings, token counts,
why generation stopped.  SDKs keep unknown fields:
`response.model_extra["kv260"]` in `openai`.  The fields are listed per
model:
- [`bert-squad`](BERT_QA.md#the-kv260-details): windows, answer span,
  confidence, n-best;
- [SmolLM2 and SmolVLM](SMOLLM2.md#the-kv260-details-and-the-log-line):
  cache reuse, prefill and decode timing, sampler settings.

A transcription's JSON body has one too: timings, parts, tokens, and with
`kv260_wake_phrase` the phrase check's result `wake`
([Speech to text](SPEECH_TO_TEXT.md#the-phrase-check)).

## Sharing the FPGA

**One request at a time runs on the FPGA.**  The others wait in a FIFO, in
the order they reach it, after the host-side preparation (such as
tokenizing).
- **A chat answer lends the FPGA.**  Between two of its tokens it hands
  the FPGA to a waiting `/audio/speech` or `/audio/transcriptions` request
  for a model that is already loaded.  That request runs at once and gives
  the FPGA back to the answer, ahead of the queue.  So a voice client can
  speak an answer's first sentence while the rest is written: Miranda's
  speech starts 0.5–1.9 s after the request instead of 2.7–3.8 s.  The
  answer's text arrives later by the speech's FPGA time.
  - Chat requests are not lent to: one answer would stall for another.
  - Neither are models that must load first, since a load may evict the
    answering model.
  - The request log shows the lends (`lent=3/1212ms`).  The server's
    `tok/s` leaves them out; the backend's `decode=` rate does not.
  - On the board an answer with three lends was token for token the same
    as alone (temperature 0), and the speech sample for sample.
  - `--no-interleave` keeps the FPGA to the answer's end.
- **Queue limits.**  A request waits at most `--queue-timeout` (120 s),
  and at most `--max-queue` (16) requests wait at once.  Beyond either
  limit the answer is 503.
- **Disconnects.**  A client that disconnects is noticed without writing
  to it, by a non-blocking peek at its socket:
  - while queued, its request leaves the queue;
  - while running, it is cancelled at the next step.  For `bert-squad`
    that is the next 256-token window.  For `smollm2` it is the next
    token, or the next prefill chunk with `--llm-prefill-chunk`.  For
    `piper` it is the next audio chunk, for `moonshine` the next part of
    the recording (at most 24.6 s).

`server.queue_timeout` and `server.max_queue` in `chat_config.json` set
the two limits.

## The log

The server writes one log line per request (`journalctl -u kv260-chat` on
the board).  The first line below is a streamed BERT request over 8 of 19
windows; the second is a request whose client disconnected (status 499):

```
2026-09-26 06:29:58 192.168.100.7 POST /v1/chat/completions 200 model=bert-squad stream=1 prompt=2048 completion=3 windows=8/19 fpga_ms=7751 conf=0.93 queue=297ms ttft=8062ms tok/s=1805.6 total=8063ms finish=stop
2026-09-26 06:29:25 192.168.100.7 POST /v1/chat/completions 499 model=bert-squad stream=0 queue=31ms total=2938ms [cancelled (client disconnected)]
```

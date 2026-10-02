# Clients

The server speaks the OpenAI API, so you do not need a special client:
anything that can talk to OpenAI can talk to the board.  This page shows
five clients and what each one is good for:

| Client | What it is | Use it when |
|---|---|---|
| [`chat.py`](#chatpy) | this repo's client: a REPL and a one-shot command, standard library only | you want to chat with the board from a terminal, on a laptop or on the board itself, with nothing to install; it can also read answers aloud |
| [`curl`](#curl) | plain HTTP from the shell | you want to see the raw requests and responses, script the server from the shell, or debug the protocol |
| [`openai` SDK](#the-openai-python-sdk) | OpenAI's official Python library | you are writing your own Python program that uses the board's models |
| [`llm`](#llm) | Simon Willison's command-line tool for LLMs | you already use `llm`, or want one-line prompts with conversations stored and continued |
| [`aichat`](#aichat) | a single-binary chat client written in Rust | you want a full-featured terminal chat client; its aarch64 build also runs on the board |

**Connection settings for every client:**
- **Base URL:** `http://<board>:8000/v1`.
- **API key:** any value.  If `server.api_key` is set in `chat_config.json`,
  send that key.
- **Model:** one of the ids in `GET /v1/models`, e.g. `bert-squad` or
  `smollm2-135m-instruct` (the [model table](../README.md#what-it-serves)).

**About the examples.**  All transcripts below are real, recorded against
the board at `192.168.100.8` on 2026-09-26.  Most use `bert-squad`, which
answers questions about a document.  The document, `sb50.txt`, is the first
paragraph of SQuAD's "Super Bowl 50" article (124 words), taken from the
BERT demo's copy of the SQuAD dev set:

```bash
python3 -c "import json; d = json.load(open('demo/bert_squad/assets/dev-v1.1.json')); print(next(a for a in d['data'] if a['title'] == 'Super_Bowl_50')['paragraphs'][0]['context'])" > sb50.txt
```

With a generative model (`smollm2-135m-instruct`, ...), leave the document
out and just chat.

## `chat.py`

`demo/chat/chat.py` is the client that comes with the server.  It is a
single Python file that uses only the standard library, so it runs
anywhere Python does — your laptop or the board itself
(`python3 /root/kv260_chat/chat.py`).  It knows the server's conventions:
- `--doc` / `/doc` loads a document for `bert-squad`;
- `--image` / `/image` attaches a picture for `smolvlm-256m-instruct`;
- after each answer it prints a line of statistics (time, windows or
  tokens/s, confidence, token counts);
- it can read the answers aloud when the server has a speech model;
- an interactive session in a terminal opens with the CORMORANT banner
  and a box naming the server and the model (below).  `--no-banner`, a
  terminal narrower than 60 columns, or piped input give the plain
  one-line header instead; colors follow `NO_COLOR`.

![The CORMORANT banner of chat.py](../../../doc/images/chat_banner.png)

**One question, then exit** (`-q`); the exit code is 1 on errors, so it
also works in scripts:

```
$ python3 demo/chat/chat.py --url http://192.168.100.8:8000/v1 --doc sb50.txt -v -q "Which NFL team represented the AFC at Super Bowl 50?"
Denver Broncos
[1.0 s · 1/1 window(s) · confidence 0.75 · 171 + 2 tokens]
```

**A conversation.**  Without `-q` it starts a REPL with line editing and
history (`readline`).  The session below was piped in (so the input is echoed).  It
shows the hint the server gives when no document has been sent yet, then
`/doc`, three questions about the document, `/model` to list the models,
and `/reset` to clear the conversation:

```
$ printf 'Who won Super Bowl 50?\n/doc sb50.txt\nWho won Super Bowl 50?\nWhat was the final score?\nWhere was the game played?\n/model\n/reset\nWhat color was emphasized for the 50th anniversary?\n/quit\n' \
    | python3 demo/chat/chat.py --url http://192.168.100.8:8000/v1 -v
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

The statistics line is printed on a terminal, and with `-v` when the
output is piped or redirected.

**Options:**

| Option | Meaning |
|---|---|
| `--url URL` | the server (or `KV260_CHAT_URL`) |
| `--api-key KEY` | the API key (or `KV260_CHAT_API_KEY` / `OPENAI_API_KEY`) |
| `--model M` | the model to start with (default: the server's first) |
| `--doc FILE` | a document for `bert-squad` |
| `--system TEXT` | a system message |
| `--image FILE` | an image for the first question (image models) |
| `-q QUESTION` | ask one question and exit (exit code 1 on errors) |
| `--no-stream`, `--max-tokens N`, `--temperature T`, `--timeout S` | request settings |
| `-v` | print the statistics line even when not on a terminal |
| `--audio`, `--say TEXT`, `--audio-out FILE.wav`, `--tts-model`, `--voice`, `--speed`, `--player CMD` | speech; see [Reading answers aloud](#reading-answers-aloud) |

**Commands in the REPL:**

| Command | Meaning |
|---|---|
| `/doc FILE` | load a document (clears the conversation) |
| `/system [TEXT]` | set or clear the system message |
| `/image FILE` | attach an image to the next message (or end a message with it) |
| `/model [M]` | list the models, or switch to M (the history is kept) |
| `/audio [on\|off]`, `/audio save FILE.wav`, `/say [TEXT]` | speech |
| `/reset` | clear the conversation |
| `/help`, `/quit` | |

**Other models.**  With a generative model the REPL is an ordinary chat.
`/model smollm2-135m-instruct` switches to it and `/model bert-squad` back;
`--model` picks it at start:

```
$ python3 demo/chat/chat.py --url http://<board>:8000/v1 --model smollm2-135m-instruct --temperature 0
> What is the capital of France?
The capital of France is Paris.
```

For images, see [Chat about images](SMOLVLM.md#sending-images).

### Reading answers aloud

When the server also serves a speech model ([Text to speech](TEXT_TO_SPEECH.md)),
`chat.py` can speak the answers:

```
$ python3 demo/chat/chat.py --url http://192.168.100.8:8000/v1 --model smollm2-360m-instruct --audio
> In two sentences, what is an FPGA?
An FPGA (Field-Programmable Gate Array) is a type of semiconductor device ...
[13.3 s · first token 1.0 s, 3.9 tok/s · 40 + 48 tokens]
[audio 12.7 s · first sound 3.3 s · 9.1 s]
> /say The birch canoe slid on the smooth planks.
[audio 2.3 s · first sound 1.4 s · 2.0 s]
> /audio save canoe.wav
(saved 2.3 s to canoe.wav)
```

- **Commands:**
  - `/audio [on|off]` toggles reading each answer aloud (`--audio` at
    start).
  - `/say TEXT` speaks a text; `/say` alone speaks the last answer again.
  - `/audio save FILE.wav` keeps the last audio.
  - Ctrl-C stops the audio.
- **Without the REPL:** `-q QUESTION --audio` speaks one answer;
  `--say TEXT` speaks a text without a chat; `--audio-out FILE.wav` also
  saves it.
- **What is sent:** `POST /v1/audio/speech` for raw PCM, with
  `--tts-model` (default `tts-1`, which the server maps to Piper),
  `--voice` and `--speed`.  Markdown marks and code blocks are left out of
  the spoken text, and a heading or list item ends with a pause.
- **Playback:** the sound starts with the first chunk (~1.5 s), piped into
  the first player found: `pw-play`, `paplay`, `aplay`, `ffplay` or sox's
  `play`.
  - `--player "CMD"` (or `KV260_CHAT_PLAYER`) names another player.  It
    reads raw s16le mono from stdin; `{rate}` in the command is replaced by
    the sample rate.
  - `--player none` never plays.
  - Without a streaming player, the audio plays once it is complete
    (`afplay` on macOS, `winsound` on Windows), or goes to a WAV file whose
    path is printed.

## `curl`

`curl` shows the protocol as it is: the JSON you send and the JSON (or
server-sent events) you get back.  Use it to script the server from a shell,
to check what a client should receive, or to debug.  The four examples
below list the models, ask a question, stream an answer, and show two
errors.

**List the models the server serves** (`GET /v1/models`):

```
$ curl -s http://192.168.100.8:8000/v1/models
{"object": "list", "data": [{"id": "bert-squad", "object": "model", "created": 1790403749, "owned_by": "kv260"}]}
```

**Ask a question** (`POST /v1/chat/completions`).  The request puts the
document in the system message and the question in the user message; the
answer is in `choices[0].message.content`.  Besides the standard fields the
response carries a `kv260` object with the backend's details: here the
windows run, the FPGA time, the answer's position and confidence, and the
five best candidate answers ([API reference](API.md#the-kv260-object)):

```
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
```

**Stream the answer.**  With `"stream": true` the answer arrives as
server-sent events, one `data:` line per chunk:
1. a chunk with the assistant role;
2. the content;
3. a chunk with `finish_reason` and the `kv260` details;
4. with `stream_options.include_usage`, a chunk with the token counts;
5. `data: [DONE]`.

A generative model sends one content chunk per token.  `-N` stops curl from
buffering.  The request is the same document with "What was the final
score?", plus `"stream": true, "stream_options": {"include_usage": true}`:

```
$ curl -sN http://192.168.100.8:8000/v1/chat/completions -H "Content-Type: application/json" -d @req_stream.json
data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", "created": 1790403880, "model": "bert-squad", "system_fingerprint": "kv260-bertsquad12-q8.8", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}], "usage": null}

data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", ..., "choices": [{"index": 0, "delta": {"content": "24–10"}, "logprobs": null, "finish_reason": null}], "usage": null}

data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", ..., "choices": [{"index": 0, "delta": {}, "logprobs": null, "finish_reason": "stop"}], "usage": null, "kv260": {"windows": 1, "fpga_ms": 969.7, "span": [54, 56], "confidence": 0.9735, ...}}

data: {"id": "chatcmpl-bb99c95da8cc42feb05a9c5f", "object": "chat.completion.chunk", ..., "choices": [], "usage": {"prompt_tokens": 166, "completion_tokens": 3, "total_tokens": 169}}

data: [DONE]
```

**Errors** come back in OpenAI's shape, with the HTTP status an OpenAI
client expects.  The first request names a model the server does not have
(404); the second sends a temperature outside the allowed range (400):

```
$ curl -s http://192.168.100.8:8000/v1/chat/completions -H "Content-Type: application/json" \
       -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}' -w ' HTTP %{http_code}\n'
{"error": {"message": "The model 'gpt-4o' does not exist or you do not have access to it.", "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}
 HTTP 404
$ curl -s ... -d '{"model":"bert-squad","messages":[{"role":"user","content":"hi"}],"temperature":5}' -w ' HTTP %{http_code}\n'
{"error": {"message": "Invalid 'temperature': expected a value <= 2, but got 5 instead.", "type": "invalid_request_error", "param": "temperature", "code": "invalid_value"}}
 HTTP 400
```

Speech with curl: [Text to speech](TEXT_TO_SPEECH.md#requests).

## The `openai` Python SDK

To use the board from your own Python code, use OpenAI's official
library, `openai`, unchanged: point it at the board with a base URL.  The
script below:
1. lists the models;
2. asks a question with the document as the system message, and reads the
   answer, the finish reason, the token usage and the board's `confidence`
   (the SDK keeps the extra `kv260` object in `model_extra`);
3. streams a second answer, with the document and the question in one user
   message (the `Context:` / `Question:` form);
4. catches the errors for an unknown model (404) and for an unsupported
   parameter (400, `n=2`).

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

Run it with the board's URL in the environment:

```
$ OPENAI_BASE_URL=http://192.168.100.8:8000/v1 OPENAI_API_KEY=none python sdk_example.py
models: ['bert-squad']
answer: Carolina Panthers | finish: stop | usage: 167 + 2 | confidence: 0.9962
stream usage: 181
streamed: Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
404: The model 'gpt-4o' does not exist or you do not have access to it.
400: Invalid 'n': this server generates one choice per request (n = 1).
```

Tested with `openai` 3.19.2.  For a generative model, change `model=` and
drop the document; the extra sampling fields (`top_k`, `dry_multiplier`,
...) go in `extra_body={...}`.  Speech with the SDK:
[Text to speech](TEXT_TO_SPEECH.md#requests).

## `llm`

[`llm`](https://llm.datasette.io/) (`pip install llm`) is a command-line
tool for language models.  It sends one prompt per command and stores
every conversation, so `llm -c` continues the last one; `llm chat` opens
an interactive chat.  It talks to OpenAI-compatible servers once they are
declared in `extra-openai-models.yaml`, in llm's config directory
(`dirname "$(llm logs path)"`, e.g. `~/.config/io.datasette.llm/`):

```yaml
- model_id: kv260
  model_name: bert-squad
  api_base: "http://192.168.100.8:8000/v1"
- model_id: kv260-smol                  # the generative backend
  model_name: smollm2-135m-instruct
  api_base: "http://192.168.100.8:8000/v1"
```

`model_id` is the name you type after `-m`; `model_name` is the server's
model.  When the server has an API key, add `api_key_name: kv260` to each
entry and store the key with `llm keys set kv260`.

The examples pass the document as the system prompt (`-s`), ask a
follow-up with `-c`, and run a piped `llm chat`:

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

- **Generative chat:** `llm chat -m kv260-smol`; add `-o temperature 0`
  for greedy, repeatable answers.
- **In scripts:** `llm` reads a piped or redirected stdin and prepends it
  to the prompt, so in a non-interactive shell run it with `</dev/null`;
  otherwise it waits for stdin.

Tested with `llm` 0.36.

## `aichat`

[`aichat`](https://github.com/sigoden/aichat) is a terminal chat client
shipped as a single Rust binary, with x86_64 and aarch64 releases, so it
runs on the host or on the board itself.  It has a REPL with roles and
sessions, and one-shot commands.  Declare the board as an
OpenAI-compatible client in `~/.config/aichat/config.yaml` (or
`$AICHAT_CONFIG_DIR/config.yaml`):

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
      - name: smollm2-135m-instruct     # generative; the server trims long histories itself
        max_input_tokens: 100000
```

For `bert-squad`, pass the document as the system prompt: `--prompt` on
the command line, `.prompt` in the REPL.  (`aichat -f FILE` puts a file
into the *user* message without a `Context:` label, so the server would
not treat it as the document.)

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
```

In the REPL, `.prompt TEXT` starts a temporary role (the `%%` prompt) with
TEXT as the system message, i.e. the document.  (This session was driven
through a pseudo-terminal; escape codes removed.)

```
$ aichat
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

Tested with `aichat` 0.30.0 (`aichat-v0.30.0-x86_64-unknown-linux-musl`).

## Not `ollama run`

`ollama run` does not work with this server, even with `OLLAMA_HOST`
pointing at it: it speaks **Ollama's own API** (`/api/chat`, `/api/show`,
`/api/tags`, NDJSON streaming), not OpenAI's.  Supporting it would take a
separate ~200-line shim over the same backends (CHAT_PLAN §5; not
implemented).  Use any OpenAI-compatible client instead.

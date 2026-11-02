# Stories — stories15m

`stories15m` writes children's stories.  It is Andrej Karpathy's llama2.c
`stories15M`, a 15 M-parameter Llama trained on the TinyStories dataset,
run on the FPGA from its own library, `libstories15m.so`, bit-exact with the
scheduler simulation (CHAT_PLAN §27):

| Model | Backend name | Decode speed | First token | Context |
|---|---|---|---|---|
| `stories15m` | `stories15m` | ~166 tokens/s on the board (6.0 ms per step); 130–150 greedy, ~95 sampled through the API | 12–23 ms | 256 tokens |

It is a **base model**: it follows no instructions and has no chat
template.  It continues the text it is given — give it the beginning of a
story.  When a story ends it writes `<s>` (in training every story started
with it); the server stops there.

## Try it

Its natural API is OpenAI's legacy text completion, `POST /v1/completions`:

```bash
curl http://<board>:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "stories15m", "prompt": "Once upon a time, there was a little dragon named Sparky.",
  "max_tokens": 250}'
```

```json
{"id": "cmpl-…", "object": "text_completion", "model": "stories15m",
 "choices": [{"text": " Sparky was very frightened because he was lost in the big forest. …",
              "index": 0, "logprobs": null, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 17, "completion_tokens": 194, "total_tokens": 211},
 "kv260": {"finish": "eos", "decode_tok_s": 94.88, …}}
```

- `max_tokens` defaults to 16, as OpenAI's; a whole story is ~150–250 tokens.
- `echo: true` puts the prompt in front of the text; `stream: true` sends
  `text_completion` chunks ([API](API.md#endpoints)).
- The `openai` SDK: `client.completions.create(model="stories15m",
  prompt="...", max_tokens=200)`.

**As a chat.**  `/v1/chat/completions` (and `chat.py --model stories15m`)
reads each message the way a chat user means it:

| Your message | What the model does | `kv260.story` |
|---|---|---|
| how a story begins: "One day, a small kitten got stuck in a tall tree." | starts a new story with it and continues it | `new` |
| a request: "Write a short story about a cat", "Tell me a bedtime story about two dragons, please" | starts a new story from an opening made of the request — "Once upon a time, there was a cat" — and the answer begins with that opening (`kv260.opening`) | `new` |
| "continue", "go on", "more", "what happened next?" … | goes on with the last story (the earlier messages, read the same way) | `continue` |
| "...and then a big dog came." | goes on with the last story, your text first | `continue` |

An answer is never empty: a new story writes at least one token and
"continue" at least 16 — after a finished story the model would start the
next one at once (the request's `min_tokens` overrides this; on
`/v1/completions` it is 0 unless set).  System messages are ignored, and
anything else ("Hi!") is taken as the
start of a story too: the model follows no instructions.  The request
patterns are a best effort — "about friendship" becomes "there was a
friendship".  A chat answer has no leading space; a `/v1/completions`
continuation keeps it.  For the message "One day, a small kitten got stuck
in a tall tree." the board answered (seed 7, the default sampling): "The
kitten was very scared and started to cry. It tried to get down, but it
could not. A kind man saw the kitten and wanted to help. …"

## Requests

- **Sampling.**  The defaults are temperature 0.8 and top_p 0.9 (llama2.c's
  run.c uses 1.0 / 0.9), with no top-k, no repetition penalty and no DRY:
  these stories repeat names and phrases by design.  Every field of
  [Sampling](SMOLLM2.md#sampling) can be set per request; temperature 0 is
  greedy.  The loop guard stays on.
- **Length.**  The context is 256 tokens (the model's own; position 0 is
  `<s>`).  A story longer than the context less the reserve (128 tokens,
  `max_tokens` if smaller) keeps its most recent tokens; `kv260.trimmed_tokens`
  says how many were cut.  A continuation that fills the context stops with
  `finish_reason: "length"` (`kv260.finish: "context_full"`).
- **An empty prompt** is refused (400): the logits after `<s>` alone are
  never computed (position 0 is a precomputed constant).
- **Not supported** on `/v1/completions`: `suffix`, `logprobs`, `best_of` > 1,
  token-id prompts, several prompts at once.

## Install

From the repo root (the study environment `.venv-export` and the
scheduler's `PY=inference-scheduler/.venv/bin/python`, as in
[SmolLM2](SMOLLM2.md#install-smollm2-135m)):

```bash
python3 demo/chat/scripts/llm_calibrate.py all stories15m          # checkpoint (MIT, 30 MB), TinyStories
                                          # texts, formats -> demo/chat/assets/study/stories15m/ (~10 s)
$PY demo/chat/scripts/generate_llm_project.py --assets demo/chat/assets/stories15m
                                          # -> demo/chat/build/llm_project_stories15m (24 s, 0.6 GB RAM)
$PY demo/chat/deploy.py --stop
$PY demo/chat/scripts/llm_board.py --install-only --project demo/chat/build/llm_project_stories15m
                                          # -> <dir>/lib/libstories15m.so, /root/stories15m_weights
```

Then add `"stories15m"` to `server.backends` in `chat_config.json` and run
`deploy.py`.  The `stories15m` block's defaults fit: `lib`
`<dir>/lib/libstories15m.so`, `tokenizer` `assets/stories15m/tokenizer.json`,
`cma_mb` 50 (the pool is 34.8 MiB), `temperature` 0.8, `top_p` 0.9,
`reserve` 128.

**The tokenizer** on the board is `llama_tokenizer.py`, the Llama-2
SentencePiece pipeline of the checkpoint's `tokenizer.json` in pure Python;
`scripts/validate_text.py --model stories15m` checks it against
`tokenizers` (3044 strings and 2000 id sequences identical) and writes the
fixture of the unit tests.

**Gates** before serving, as for any generative model ([Development](DEVELOPMENT.md)):
`scripts/llm_sched_check.py --assets assets/stories15m --save
assets/study/stories15m/gate2.json`, `scripts/llm_host_emu.py --project
build/llm_project_stories15m --study-json …`, then `scripts/llm_board.py
--project build/llm_project_stories15m --study-json … --reopen`: logits
bit-exact with the simulation on the three story prompts and a second turn.

## Speed and quality

- **Quality** (`llm_study.py`, CHAT_PLAN §27): against the float model, the
  library's numerics agree on the next token 99.0 % of the time on held-out
  TinyStories text (bf16 99.5 %), perplexity 3.322 against 3.314 — the
  stories read like float's, worded a little differently.
- **Board** (production bitstream): decode 5.7 / 6.3 / 6.8 ms per token at
  positions 32 / 128 / 240; prefill 8 / 12 / 32 ms for 16 / 64 / 255 tokens;
  `llm_open` 4.5 s cold, 0.15 s warm; 33 MB of CMA.  The LM head (32000 × 288)
  is 2.5 ms of a decode step.
- **Memory**: 33 MB of CMA (the pool: weights 30.7 MiB, KV caches 2.3,
  intermediates 1.7), and in normal memory the embedding table, 17.6 MiB —
  int16 at the residual's 2^-15, exact for this checkpoint's float16 values
  (CHAT_PLAN §27) — and the tokenizer, ~17 MiB of Python objects.

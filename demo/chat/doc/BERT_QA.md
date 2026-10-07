# Document Q&A — `bert-squad`

The `bert-squad` model answers questions about a document you send.  It is
BERT-base fine-tuned on SQuAD, the model of the [`bert_squad/`](../../bert_squad/README.md)
demo, running on the FPGA: 525 ms per 256-token window, with logits
bit-exact with the scheduler simulation.

BERT is an **extractive** model: it cannot write free text.  The answer is
always a passage copied from the document, the one that best answers the
question.  So a chat with it is question answering over a document: send
the document once, then ask as many questions as you like.

- [Setting it up](#setting-it-up)
- [Sending the document and the questions](#sending-the-document-and-the-questions)
- [Long documents](#long-documents)
- [Request fields](#request-fields)
- [The `kv260` details](#the-kv260-details)
- [Measured](#measured)

## Setting it up

`bert-squad` is the default backend: `deploy.py` generates, uploads and
builds everything it needs ([Deploying](DEPLOY.md)).  The model (435 MB)
and its vocabulary are downloaded on the first run.  On the board it holds
~224 MB of CMA.

## Sending the document and the questions

**The document** can be sent in either of two ways:
- as the **system message**.  A leading `Context:` or `Document:` label is
  dropped.
- as a **user message that starts with `Context:`**, optionally followed by
  a line `Question: ...` in the same message.

Follow-up questions in the same conversation reuse the document: clients
resend the whole history with every request.

**The question** is the last user message.
- **A message that is only a document** (`Context:` with no question) is
  acknowledged: "Got the document (… words, … tokens) …".
- **A question without any document** gets a short usage hint.

Neither of these touches the FPGA.

**The answer** is the best span of the document, as SQuAD reports it: the
document's own whitespace-separated words, punctuation kept ("February 7,
2016,").

```
> /doc sb50.txt
(document: sb50.txt, 124 words; conversation cleared)
> What was the final score?
24–10
[1.0 s · 1/1 window(s) · confidence 0.97 · 166 + 3 tokens]
```

More examples with every client: [Clients](CLIENTS.md).

## Long documents

BERT reads 256 tokens at a time.  A longer document is split into
overlapping 256-token windows: the question takes up to 64 tokens, and
consecutive windows start 128 tokens apart.  Each window is one FPGA
inference (~1 s), and the best span across all windows is the answer.

At most `--max-windows` windows (default 8, ≈ 8 s) run per question.  A
document beyond that is truncated, and the response says so:
`kv260.truncated` is true, and `chat.py` shows `windows 8/19`.
`server.max_windows` and `server.doc_stride` in `chat_config.json` set the
window cap and the stride.

## Request fields

| Field | Effect |
|---|---|
| `max_tokens` | caps the answer in WordPiece tokens (`finish_reason: "length"`) |
| `stop` | cuts the answer at a stop string |
| `temperature`, `top_p`, `seed` | accepted and ignored: the answer is extractive and deterministic |

`usage.prompt_tokens` counts the real tokens over the windows run;
`completion_tokens` counts the answer's tokens.

## The `kv260` details

Every response carries a non-standard `kv260` object, on the last chunk
when streaming.  SDKs keep it: `response.model_extra["kv260"]` in `openai`.

| Field | Meaning |
|---|---|
| `windows`, `windows_total`, `truncated` | windows run, windows the document needs, whether it was cut |
| `window`, `span` | the window of the answer and its token positions in it |
| `score` | start + end logit of the answer |
| `confidence` | softmax over the n-best candidates |
| `n_best` | the top 5 candidate answers with score and probability |
| `fpga_ms`, `window_ms` | FPGA time, total and per window |

## Measured

On the KV260 at 100 MHz, 2026-09-26, with the bitstream of BERT phase 2.

**In short:** the server gives exactly the answers of the BERT demo,
about 1 s per 256 tokens of document; a long document answers in ~8 s at
EM 90 / F1 94.

- **Same answers as the demo.**  The 12 questions of a `bert_squad`
  `deploy_and_run.py --n 12` run were sent through the server, with the
  context as the system message.  All **12 / 12 spans were identical**,
  text and token positions (`tests/board_gate.py`).
- **The demo itself**, rerun with `squad_bench` built on the server's C API
  (`bert_api.c`):
  - its board logits equal the scheduler simulation bit for bit (3 / 3
    examples checked) and the study's emulation (12 / 12);
  - 966.7 ms per inference;
  - logits byte-identical to an earlier run on main.
- **Latency.**
  - FPGA time: 971 ms per window (mean over the 12).
  - Request latency: 1027 ms (tokenizing, JSON, network).
  - Startup: 1.6 s.
  - Memory: the server process holds ~221 MB of CMA (CmaFree 811 → 590 MB).
- **Long document.**
  - "Super Bowl 50" paragraphs 0–9: 848 words, 1048 tokens, 8 windows.
  - 10 questions whose answers sit in different paragraphs:
    **EM 90.0 / F1 94.2**, 7.8–8.0 s per question (970 ms per window).
  - Paragraphs 0–15 need 19 windows: truncated to 8 and reported as
    `windows 8/19`.
- **Disconnect.**  A client dropped an 8-window request after 2.5 s.  The
  FPGA was free after the current window (the request was logged at 2.9 s
  as cancelled), and the three requests queued behind it ran next.

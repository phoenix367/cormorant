# Text to speech — Piper

The `piper` backend turns text into speech through OpenAI's speech API,
`POST /v1/audio/speech`.  The voice is Piper (VITS) en_US-lessac-medium;
details of the port are in [`demo/tts/`](../../tts/README.md) and
TTS_PLAN §4–§7.

**Listen** — samples spoken by the board through this endpoint:
▶ [hello](../../tts/samples/hello.mp3) ·
▶ [a paragraph](../../tts/samples/paragraph.mp3) ·
▶ [a question](../../tts/samples/question.mp3) ·
▶ [`speed` 1.5](../../tts/samples/fast.mp3)
(texts: [demo/tts](../../tts/README.md#samples)).

- **Speed.**  The first sound comes 1.0–1.5 s after the request, and the
  speech is made faster than it plays (real-time factor 0.58–0.79 end to
  end).  The audio streams while it is being made.
- **Exactness.**  The samples are bit-exact with the same pipeline run on
  the host.
- **Memory.**  It needs 55 MB of CMA (a 48 MiB pool).  Under
  `--resident auto` it stays loaded even next to SmolLM2-360M, the largest
  model.

## Install

The commands run from the repo root.

1. **Fetch the voice and calibrate it**: step 1 of the
   [`demo/tts` commands](../../tts/README.md#commands) (`piper_study.py
   fetch`, `phonemize`, `calibrate` and `encoder`).  It downloads the voice
   and computes its fixed-point exponents into `demo/tts/assets/`.
2. **Generate the library project** (4 s):

   ```bash
   PY=inference-scheduler/.venv/bin/python
   $PY demo/tts/scripts/generate_tts_project.py         # -> demo/tts/build/piper_project
   ```

3. **Build and install it on the board.**  Stop the server first, since it
   owns the FPGA:

   ```bash
   $PY demo/chat/deploy.py --stop
   $PY demo/tts/scripts/tts_board.py --install-only     # <dir>/lib/libpiper_tts.so, /root/piper_weights
   ```

**Enable it:**
1. Add `"piper"` to `server.backends` in `chat_config.json`.  The `piper`
   block sets the library, the voice directory and `cma_mb` (55).
2. Run `deploy.py`.

**On the board it needs:**
- `libespeak-ng1`, `espeak-ng-data` and numpy (Ubuntu 22.04 on the KV260
  has them);
- ffmpeg, for mp3 / opus / aac / flac.

## Requests

**With curl**, the request is JSON and the response is the audio file:

```
$ curl http://192.168.100.8:8000/v1/audio/speech -H 'Content-Type: application/json' \
       -d '{"model": "tts-1", "input": "Hello! I am running on an FPGA board."}' -o hello.wav
```

**With the `openai` SDK**, the same call as for OpenAI.  The streaming
response writes the audio to the file as it arrives:

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.100.8:8000/v1", api_key="none")
with client.audio.speech.with_streaming_response.create(
        model="tts-1", voice="alloy", input="The birch canoe slid on the smooth planks.",
        response_format="wav") as r:
    r.stream_to_file("canoe.wav")
```

**With `chat.py`**, `--say TEXT` speaks a text, and `--audio` reads every
chat answer aloud ([Clients](CLIENTS.md#reading-answers-aloud)).

**Request fields:**

| Field | Values |
|---|---|
| `model` | `piper-lessac-medium`, or the aliases `tts-1` / `tts-1-hd` / `gpt-4o-mini-tts` |
| `input` | the text, up to 4096 characters |
| `voice` | any name is accepted; there is one voice |
| `response_format` | `wav` (the default here; OpenAI's default is mp3), `pcm`, or `mp3` / `opus` / `aac` / `flac` through ffmpeg |
| `speed` | 0.25–4 |
| `stream_format` | `sse` gives `speech.audio.delta` / `speech.audio.done` events instead of the audio file |
| `seed` | an extension; the default 0 makes the same text always give the same audio |

`pcm` is raw s16le mono at **22 050 Hz** (OpenAI's pcm is 24 kHz); the
`X-Sample-Rate` header says so.

## How a request runs

1. **Outside the FPGA lock**, the server turns the text into phonemes
   (espeak-ng, then Piper's ids).  It packs whole sentences into
   utterances of at most 400 ids.
2. **Under the lock**, for each utterance:
   - the text encoder runs on the FPGA (`tts_encode`, 26–260 ms);
   - the duration predictor runs as C code in the library
     (`tts_duration`, 18–200 ms);
   - the alignment and the noise are computed in numpy;
   - then the audio is made in chunks of 128 frames (1.49 s of speech):
     the flow and the HiFi-GAN decoder run on ConvKernel, ~0.7 s per
     chunk.
3. **Delivery.**  Each chunk's samples go out as soon as they exist.  wav
   and pcm responses carry an exact Content-Length.
4. **Disconnects.**  A client that disconnects stops the synthesis at the
   next chunk.

## Measured

2026-09-30:
- **First audio.**  1.0–1.3 s for 3.4 and 5.9 s of speech; 1.5 s for
  12.5 s of speech.
- **RTF** (time to make the audio ÷ its length): 0.58–0.79 end to end.
- **Samples.**  Bit-exact with the same pipeline on the host
  (`demo/tts/scripts/tts_speech_check.py`).
- **Residency.**  55 MB of CMA (a 48 MiB pool).  With `--resident auto` it
  stays loaded next to SmolLM2-360M.

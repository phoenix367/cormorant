# Speech to text — Moonshine

The `moonshine` backend transcribes recordings through OpenAI's
transcription API, `POST /v1/audio/transcriptions`.  The model is Moonshine
base (UsefulSensors, MIT; Moonshine tiny with its own library), English
only; how it was chosen and ported is in
[ASR_PLAN](../../../doc/plans/ASR_PLAN.md) §1–§7 and
[`demo/asr/`](../../asr/README.md).

- **Accuracy.**  The board's transcripts equal the fixed-point
  specification token for token (200 / 200 LibriSpeech test-clean
  utterances): word error rate 2.93 % for base, 4.58 % for tiny, against
  3.02 % / 4.49 % for the float models.
- **Speed.**  The FPGA work is 0.12–0.16 s per second of audio for base
  (0.074 for tiny); a 10 s recording is transcribed in 1.3 s.
- **Memory.**  It needs 200 MB of CMA (a 185.8 MiB pool); tiny 95 MiB.

## Install

The commands run from the repo root.

1. **Fetch the checkpoint**: `demo/asr/scripts/asr_study.py fetch` (it
   needs only numpy: the scheduler's venv runs it; the study's other steps,
   ASR_PLAN §2, need `.venv-export`, created as in
   [Generative chat](SMOLLM2.md#install-smollm2-135m) step 1) downloads
   Moonshine and the LibriSpeech audio into `demo/asr/assets/`.  The calibrated exponents are
   in the repository (`demo/asr/moonshine-base_opt_formats.json`).
2. **Generate the library project** (52 s, 2.9 GB):

   ```bash
   PY=inference-scheduler/.venv/bin/python
   $PY demo/asr/scripts/generate_asr_project.py --model moonshine-base   # -> demo/asr/build/moonshine-base_project
   ```

3. **Build and install it on the board.**  Stop the server first, since it
   owns the FPGA:

   ```bash
   $PY demo/chat/deploy.py --stop
   $PY demo/asr/scripts/asr_board.py --project demo/asr/build/moonshine-base_project --install-only
   #   -> <dir>/lib/libmoonshine_base.so, /root/moonshine_base_weights
   ```

   `--remote-dir DIR` / `--weights-dir DIR` install elsewhere — for a second
   install beside an existing one; set the same paths in `chat_config.json`
   (`remote.dir`, `moonshine.weights_dir`).

**Enable it:**
1. Add `"moonshine"` to `server.backends` in `chat_config.json`.  The
   `moonshine` block sets the library, its weights directory, the
   tokenizer (`demo/asr/assets/moonshine-base/tokenizer.json`, uploaded by
   `deploy.py`), `cma_mb` (200) and `max_seconds` (600).
2. Run `deploy.py`.

**On the board it needs** numpy, and ffmpeg for uploads other than WAV.

## Requests

**With curl**, the request is a multipart form with the recording as
`file`:

```
$ curl http://192.168.100.8:8000/v1/audio/transcriptions -F model=whisper-1 -F file=@nemo.wav
{"text": "Nemo builds a fabulous futuristic submarine, the Nautilus, then conducts an underwater campaign of vengeance against his imperialist oppressor.", "usage": {"type": "duration", "seconds": 10}, "kv260": {"audio_s": 9.935, "parts": 1, "tokens": 37, "encode_ms": 240.4, "decode_ms": 995.5, "fpga_ms": 1237.1, "input_ms": 1.3, "rtf": 0.125, "source": "wav"}}
$ curl http://192.168.100.8:8000/v1/audio/transcriptions -F model=whisper-1 -F file=@nemo.wav -F response_format=srt
1
00:00:00,000 --> 00:00:09,935
Nemo builds a fabulous futuristic submarine, the Nautilus, then conducts an underwater campaign of vengeance against his imperialist oppressor.
```

(`nemo.wav` is LibriSpeech test-clean 8463-294825-0003, under
`demo/asr/assets/librispeech/` after `asr_study.py fetch`:
`ffmpeg -i demo/asr/assets/librispeech/LibriSpeech/test-clean/8463/294825/8463-294825-0003.flac nemo.wav`;
`-F stream=true` gives server-sent events.)  In `kv260`, `fpga_ms` is the library's time;
`encode_ms` / `decode_ms` split it as the library sees it — with the
Cortex-R5 driving the kernels the encoder's batch is still running when
the call returns (it has no output to wait for), so its time shows up in
the first decode step (the A53 driving: 459 + 784 ms for the same audio,
1243 ms in all).

**With the `openai` SDK**, the same call as for OpenAI:

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.100.8:8000/v1", api_key="none")
with open("speech.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

**With `chat.py`**, `--transcribe FILE` prints a transcript, and
`/listen FILE` prints one in a session; neither sends the text to the chat
model ([Clients](CLIENTS.md#speech-to-text)).

**Form fields:**

| Field | Values |
|---|---|
| `file` | the recording: WAV (PCM 8–32-bit or float, any rate, any channels; mixed to mono and resampled to 16 kHz here), or mp3 / flac / ogg / opus / m4a / webm through ffmpeg; at most `--asr-max-seconds` (600) of audio and a 32 MB upload |
| `model` | `moonshine-base`, or the aliases `whisper-1` / `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` |
| `response_format` | `json` (the default), `text`, `srt`, `vtt`, `verbose_json` |
| `stream` | `true`: `transcript.text.delta` events, one per part, then `transcript.text.done` |
| `language` | `en` (or absent); other languages are refused |
| `prompt`, `temperature` | accepted and ignored: Moonshine decodes greedily and takes no prompt |
| `kv260_wake_phrase`, `kv260_wake_margin` | non-standard: check a phrase expected at the start of the recording against the audio ([below](#the-phrase-check)) |

**Segments.**  `srt`, `vtt` and `verbose_json` have a segment per part of
the recording: the library splits audio longer than 24.6 s at quiet points
(the quietest 0.1 s within ±2 s of an equal split), so a segment spans up
to 24.6 s.

## The phrase check

Moonshine takes no prompt, so a recognizer that mishears a wake phrase
cannot be told to expect it.  The non-standard field `kv260_wake_phrase`
instead checks the phrase against the audio, at the start of the
recording:

```
$ curl http://192.168.100.8:8000/v1/audio/transcriptions -F model=whisper-1 -F file=@haloma.wav
{"text": "Helumor Renda, what is the capital of France?", ...}
$ curl http://192.168.100.8:8000/v1/audio/transcriptions -F model=whisper-1 -F file=@haloma.wav -F "kv260_wake_phrase=Hello Miranda"
{"text": "Hello, Miranda. What is the capital of France?", "usage": {"type": "duration", "seconds": 4}, "kv260": {"audio_s": 3.386, "parts": 1, "tokens": 12, "encode_ms": 110.7, "decode_ms": 302.7, "fpga_ms": 414.0, "input_ms": 1.0, "rtf": 0.122, "source": "wav", "wake": {"phrase": "Hello Miranda", "margin": -2.367, "complete": true, "passed": true, "min_margin": -10.0}}}
```

(Piper saying "Haloma Randa, what is the capital of France?" — how
Moonshine once heard Piper's "Hello Miranda" in
[Miranda](../../miranda/README.md)'s test recording.)

- **How.**  The decoder is fed the phrase's tokens as its first ones
  (`Hello` `Mir` `anda`).  Where the decoder would have written another
  token, the margin adds log P(the phrase's token) − log P(its own choice),
  in nats.  Punctuation it writes between the phrase's words ("Hello,
  Miranda") is kept and costs nothing.  So the margin is 0 when the decoder
  writes the phrase itself, and it falls the less the audio sounds like it.
- **The result.**  At a margin of at least `kv260_wake_margin` (default
  −10), the decoder goes on from the phrase and the transcript starts with
  it.  Below, the transcript is the plain one, the same ids as without the
  check.  `kv260.wake` holds `phrase`, `margin`, `complete` (false: the
  sum stopped once it was below the limit), `passed` and `min_margin`.
- **Cost.**  Usually nothing: the sum mostly falls below the limit at the
  first token, and the plain transcript reuses that step.  On the board,
  20 LibriSpeech utterances took 2.7 ms longer on average (33 ms at most),
  and the phrase itself took no extra time.
- **Margins on the board** (2026-10-17, Piper's voice with noise and
  speed changes, three takes each; `doc/plans/ASR_PLAN.md` §8):

  | Speech | Margin |
  |---|---|
  | "Hello Miranda …" (17 sentences, 51 takes) | 0.00, but "Hello Miranda." alone −0.44 / −0.24 in two of its takes |
  | "Haloma Randa …", "Hallo Maranda …" | 0.00 to −3.95 |
  | "Okay, hello Miranda …" | −5.2 to −6.2 |
  | "Hello Veranda", "Hello Amanda", "Hello Melinda", "Hello Rwanda", "Hello Fernanda" | −3.3 to −10.6 |
  | "Hi Miranda …", "Hey Miranda …", "Miranda, …" | −10.3 to −18.1 |
  | other greetings ("Hello there", "Hello everyone", "Hello, Mr. Anderson", "Hello Linda") | −11.3 to −21.0 |
  | 196 LibriSpeech utterances | −15.5 at most (median −28.6) |

- **Limits.**  Only at the start of the recording, and only for audio that
  fits one encoder pass (24.6 s; longer audio is transcribed as usual and
  `kv260.wake` says why it was not checked).  The phrase is cased as the
  model writes it ("Hello Miranda", not "hello miranda").  A library built
  before the check (without `asr_last_logits`) transcribes as usual, and
  `kv260.wake.error` says so.

## How a request runs

1. **Outside the FPGA lock**, the server reads the upload: a WAV file in
   `speech_input.py` (16 kHz mono 16-bit passes through unchanged; other
   rates go through a Kaiser-windowed sinc resampler), anything else
   through ffmpeg.
2. **Under the lock**, `libmoonshine_base.so` splits the audio into parts
   (`asr_chunks`) and transcribes them one by one (`asr_transcribe`):
   - the encoder runs in the smallest of four buckets that holds the part
     (128 / 256 / 512 / 1024 frames: 3.1 / 6.2 / 12.3 / 24.6 s) — the
     conv stem's host ops, the attention on ConvKernel with VectorOPKernel's
     softmax unit, the LayerNorms, residuals and RoPE on the row-ops unit;
   - the decoder runs greedily, one token per step (~18 ms), until `</s>`
     or 6.5 tokens per second of audio.
3. **Delivery.**  The token ids become text on the host (the checkpoint's
   `tokenizer.json`, `moonshine_backend.Tokenizer`).  Streaming
   requests get each part's text as soon as it exists.
4. **Disconnects.**  A client that disconnects stops the transcription at
   the next part.

## Measured

2026-10-17, through the server (ASR_PLAN §7; the R5 driving the encoder,
as the libraries' policy chooses when its firmware runs):
- **A 9.9 s recording**: 1.30 s for the request, 1.24 s of it on the FPGA
  (RTF 0.124); the transcript equals the specification's.  As a 44.1 kHz
  stereo WAV the resampler adds 1.1 s; as FLAC, ffmpeg adds 3.1 s.
- **A 28.4 s recording** (two parts of ~14 s, each in the 24.6 s bucket):
  4.55 s (RTF 0.155); streaming, the first part's text after 2.1 s.
- **Loading**: 12.7 s the first time (190 MB of weights from the SD card);
  under `--resident auto` it evicted SmolLM2-360M to fit.

The library alone (ASR_PLAN §6, the A53 driving the kernels):
- **Per bucket** (the encoder, median): 108 / 206 / 439 / 961 ms for
  recordings up to 3.1 / 6.2 / 12.3 / 24.6 s; a decode step 18.3 ms.
- **Per utterance** (200 LibriSpeech utterances, 8.4 s on average): 1.10 s,
  RTF 0.131.  Tiny: 0.63 s, RTF 0.074.
- **Exactness.**  Every token id equals the specification's
  (`demo/asr/scripts/asr_board.py`); the chat backend over the same
  library equals it too (`asr_host_emu.py --lib-check`).

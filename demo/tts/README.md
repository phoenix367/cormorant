# Text to speech on the KV260 (Piper)

This directory runs Piper (VITS) text to speech with the
[`en_US-lessac-medium`](https://huggingface.co/rhasspy/piper-voices) voice
on this repo's FPGA kernels, and serves it as OpenAI's
`POST /v1/audio/speech` in the [chat server](../chat/README.md#text-to-speech--piper-lessac-medium).

Plan, study and results: [`doc/plans/TTS_PLAN.md`](../../doc/plans/TTS_PLAN.md).
- §3: the numeric study.
- §4: the library and its board gate.
- §5: the server.

```
 text ──espeak-ng──▶ phoneme ids ──numpy (float64)──────────────────▶ z_p [192][frames]
                     (piper_phonemize)   text encoder, duration predictor,       │
                                         alignment, noise (piper_vits.front_end) │ chunks of 128 frames
                                                                                 ▼
                     libpiper_tts.so: reverse flow + HiFi-GAN on ConvKernel (66 convs per chunk)
                     + host ops (LeakyReLU / masks / folding, gates, sums, interleave)  ──▶ int16 PCM, 22 050 Hz
```

## Results (2026-09-30)

- **Output:** bit-exact with the specification on the board.
- **Library speed:** about 0.7 s per 1.49 s chunk, an RTF of 0.52 on a
  6.9 s sentence.
- **Through the chat server:** first audio after 1.8–3.5 s, RTF 0.78–0.94
  end to end.
- **Memory:** 35.5 MiB of CMA.
- **Quality:** the int16 datapath is within 0.19 dB log-mel of float (§3).

## Files

```
demo/tts/
├── src/tts_api.{c,h}          — libpiper_tts.so: tts_open / tts_synthesize_chunk / tts_synthesize / tts_close
├── src/tts_bench.c            — the board gate's runner (timing, repetitions, re-open, profile)
├── scripts/
│   ├── piper_vits.py          — float reference, the bit-level chunk specification (chunk_forward,
│   │                            synthesize_chunked) and the host front end (front_end); also runs in the server
│   ├── piper_study.py         — fetch / phonemize / validate / calibrate / study / costs (TTS_PLAN §3)
│   ├── generate_tts_project.py — src/piper.py's chunk entry -> build/piper_project (+ frontend.npz, voice.json)
│   ├── tts_host_emu.py        — the generated C on the host vs the spec; --lib-check: the chat backend over it
│   ├── tts_board.py           — build / install on the board, board gate (bit-exact, timing, profile)
│   ├── tts_speech_check.py    — /v1/audio/speech on the board vs the same pipeline on the host
│   ├── audio8_study.py, tts_screen.py — the Audio8 study (NO-GO) and the candidate screen (TTS_PLAN §1–§2)
│   └── requirements-phonemize.txt — phonemizer-fork + espeakng-loader for the study's phonemize step
└── assets/                    — not in git: the voice, texts.json, exponents.json, study WAVs
```

The scheduler parts are in `inference-scheduler/`:
- `src/piper.py`: the frontend.
- `src/tts_nodes.py`: the host ops.
- `src/numeric.py`: Conv weight encoding.
- Tests: `test/test_piper.py`, `test_tts_ops.py`, `test_conv_exp.py`.

## Commands

```bash
PY=inference-scheduler/.venv/bin/python
# 1. the voice and its calibrated exponents (TTS_PLAN §3 "Commands")
$PY demo/tts/scripts/piper_study.py fetch
/mnt/data/tts_venv/bin/python demo/tts/scripts/piper_study.py phonemize    # venv from requirements-phonemize.txt
$PY demo/tts/scripts/piper_study.py calibrate
# 2. the library project and its host checks
$PY demo/tts/scripts/generate_tts_project.py         # -> demo/tts/build/piper_project, 4 s
$PY demo/tts/scripts/tts_host_emu.py                 # the generated C == the spec on real sentences, 60 s
$PY demo/tts/scripts/tts_host_emu.py --lib-check     # the chat server's backend over a host-built library
# 3. the board (the chat server owns the FPGA: stop it first)
$PY demo/chat/deploy.py --stop
$PY demo/tts/scripts/tts_board.py --profile --reopen # gate + install, ~2 min
$PY demo/chat/deploy.py                              # with "piper" in chat_config.json server.backends
$PY demo/tts/scripts/tts_speech_check.py             # end to end, ~1 min
```

## Notes

- **Chunks.**
  - Each pass computes 256 flow frames and a 192-frame decoder window for
    128 output frames; the 64 frames of context on each side cover the
    receptive field.
  - That context is also the main cost: the ConvKernel spends 0.5 s of
    the 0.7 s per chunk.
  - Larger chunks would cut the overlap, at the price of first-audio
    latency (TTS_PLAN §4, "Levers").
- **The front end on the A53.**
  - numpy falls back to a naive loop for a matmul with a strided operand,
    145× slower there, so `conv1d` makes its taps contiguous.
  - GELU uses `erf_fast` instead of `math.erf` element by element.
- **License:** the voice is MIT, but its training data (Blizzard 2013
  Lessac) has its own license.  Check it before shipping.

#!/usr/bin/env python3
"""
tts_speech_check.py — the chat server's text to speech end to end
(doc/plans/TTS_PLAN.md §5): POST /v1/audio/speech on the board for a few
texts, timed (time to first audio, total, real-time factor), and every sample
compared with the host's own run of the same pipeline — the chat server's
PiperBackend front end (espeak-ng on this machine, the library's text
encoder as piper_vits.encoder_forward, the C duration predictor as
duration_predictor_seq, the request's seed) and the specification
(piper_vits.synthesize_chunked) — so the board's phonemes, front end and
library are checked together.  WAVs of both are written next to the report.

usage: inference-scheduler/.venv/bin/python demo/tts/scripts/tts_speech_check.py
           [--url http://192.168.100.8:8000/v1] [--project demo/tts/build/piper_project]
           [--format pcm] [--out demo/tts/build/speech_check] [--texts "..."] [--no-reference]
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from urllib.parse import urlsplit

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tts_board as tb                                              # noqa: E402

sys.path.insert(0, os.path.join(tb.REPO, "demo", "chat"))

TEXTS = ("Hello! I am a small assistant running on an FPGA board.",
         "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
         "It's easy to tell the depth of a well.",
         "Text to speech on the KV260: the flow and the HiFi-GAN decoder run on the ConvKernel, "
         "in chunks of a hundred and twenty-eight frames, while the text encoder and the duration "
         "predictor run on the ARM cores.")


def speak(url: str, body: dict, api_key=None):
    """(status, headers, audio bytes, seconds to the first body byte, total seconds)."""
    u = urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    t0 = time.monotonic()
    c.request("POST", u.path.rstrip("/") + "/audio/speech", body=json.dumps(body), headers=h)
    r = c.getresponse()
    first = r.read(1)
    t1 = time.monotonic()
    rest = r.read()
    t2 = time.monotonic()
    hd = {k.lower(): v for k, v in r.getheaders()}
    c.close()
    return r.status, hd, first + rest, t1 - t0, t2 - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default="http://192.168.100.8:8000/v1")
    ap.add_argument("--project", default=tb.DEFAULT_PROJECT)
    ap.add_argument("--model", default="tts-1")
    ap.add_argument("--format", default="pcm", choices=("pcm", "wav"))
    ap.add_argument("--texts", action="append", default=None)
    ap.add_argument("--out", default=os.path.join(tb.REPO, "demo", "tts", "build", "speech_check"))
    ap.add_argument("--api-key", default=os.environ.get("KV260_CHAT_API_KEY"))
    ap.add_argument("--no-reference", action="store_true", help="timing only")
    args = ap.parse_args(argv)
    texts = args.texts or TEXTS
    os.makedirs(args.out, exist_ok=True)
    ref = None
    if not args.no_reference:
        from chat_backend import SpeechRequest
        from piper_backend import PiperBackend
        project = os.path.abspath(args.project)
        summary, W, E = tb.load(project)
        E_enc = json.load(open(os.path.join(summary["assets"], "exponents.json")))["encoder"]
        # the host's twin of the board: the library's encoder is encoder_forward
        ref = PiperBackend(None, os.path.join(project, "weights"),
                           encoder=tb.pv.library_encoder(tb.pv.encoder_weights(W), E_enc),
                           duration=lambda x, z: tb.pv.duration_predictor_seq(W, x, z))
        ref.load_host()
    report, ok = [], True
    for i, text in enumerate(texts):
        st, hd, audio, ttfa, total = speak(args.url, {"model": args.model, "input": text, "seed": i,
                                                      "response_format": args.format}, args.api_key)
        if st != 200:
            print(f"[{i}] HTTP {st}: {audio[:300]!r}")
            return 1
        pcm = np.frombuffer(audio[44:] if args.format == "wav" else audio, "<i2")
        audio_s = pcm.size / tb.SR
        rec = {"text": text, "samples": int(pcm.size), "audio_s": round(audio_s, 3),
               "ttfa_s": round(ttfa, 3), "total_s": round(total, 3), "rtf": round(total / audio_s, 3)}
        tb.write_wav(os.path.join(args.out, f"board_{i}.wav"), pcm)
        if ref is not None:
            job = ref.prepare_speech(SpeechRequest(model=ref.model_id, input=text, seed=i))
            ref.front_end(job)
            want = np.concatenate([tb.pv.synthesize_chunked(W, E, zp) for zp in job.utterances])
            tb.write_wav(os.path.join(args.out, f"host_{i}.wav"), want)
            same = want.shape == pcm.shape and np.array_equal(want, pcm)
            rec["bit_exact"] = bool(same)
            if not same:
                rec["host_samples"] = int(want.size)
                if want.shape == pcm.shape:
                    rec["mismatches"] = int((want != pcm).sum())
            ok &= same
        report.append(rec)
        print(f"[{i}] {audio_s:.2f} s audio: first audio {ttfa:.2f} s, total {total:.2f} s "
              f"(RTF {total / audio_s:.2f})"
              + ("" if ref is None else "; " + ("bit-exact with the host" if rec["bit_exact"] else
                                                f"DIFFERS from the host ({rec})")), flush=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    if ref is not None:
        print("SPEECH CHECK: " + ("board == host pipeline, bit for bit" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

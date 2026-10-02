#!/usr/bin/env python3
"""
make_samples.py — the voice samples linked from the docs (demo/tts/samples/).

Each text of SAMPLES is spoken by the chat server's Piper backend on the
board — POST /v1/audio/speech, the deployed libpiper_tts.so — and saved as
the server's mp3 (its ffmpeg path, 64 kbps mono at 22 050 Hz).  The
manifest.json beside them records the text, the speed, the length and the
server that made each one.  The server must be running with the piper
backend (demo/chat/deploy.py); the samples are deterministic (seed 0).

usage: python3 demo/tts/scripts/make_samples.py --url http://<board>:8000 [--api-key KEY] [--only ID ...]
       (KV260_CHAT_API_KEY is used when --api-key is not given)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "samples"
MODEL = "piper-lessac-medium"

SAMPLES = [
    {"id": "hello", "speed": 1.0,
     "text": "Hello! This voice comes from a Kria KV260 board. The neural network that speaks runs on its "
             "FPGA and its Arm cores, in sixteen-bit fixed point."},
    {"id": "paragraph", "speed": 1.0,
     "text": "Cormorant turns neural networks into C programs that drive four hardware kernels. On this one "
             "board they answer questions about a document, chat with small language models, talk about "
             "pictures, and read their answers aloud, like this one."},
    {"id": "question", "speed": 1.0,
     "text": "Can you tell that this sentence was spoken by a machine? Questions rise at the end. "
             "Statements fall."},
    {"id": "fast", "speed": 1.5,
     "text": "The same voice at one and a half times the normal speed, for when you are in a hurry."},
]


def _request(url: str, path: str, body: dict | None, key: str | None):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read(), dict(r.headers)


def _seconds(path: Path):
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True)
    try:
        return round(float(out.stdout.strip()), 2)
    except ValueError:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--url", required=True, help="the chat server, e.g. http://192.168.1.10:8000")
    ap.add_argument("--api-key", default=os.environ.get("KV260_CHAT_API_KEY"))
    ap.add_argument("--only", nargs="+", metavar="ID", help="only these samples")
    a = ap.parse_args(argv)
    health = json.loads(_request(a.url, "/health", None, None)[0])
    tts = next((m for m in health.get("models", []) if m.get("id") == MODEL), None)
    if tts is None:
        print(f"error: the server does not serve {MODEL} (backends: piper)", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"samples": {}}
    for s in SAMPLES:
        if a.only and s["id"] not in a.only:
            continue
        audio, headers = _request(a.url, "/v1/audio/speech", {"model": MODEL, "input": s["text"], "speed": s["speed"],
                                                               "response_format": "mp3"}, a.api_key)
        f = OUT / f"{s['id']}.mp3"
        f.write_bytes(audio)
        manifest["samples"][s["id"]] = {"file": f.name, "text": s["text"], "speed": s["speed"], "bytes": len(audio),
                                        "seconds": _seconds(f), "sample_rate": int(headers.get("X-Sample-Rate", 0))}
        print(f"{f.relative_to(OUT.parent)}: {len(audio)} bytes, {manifest['samples'][s['id']]['seconds']} s")
    manifest.update({"model": MODEL, "server": f"kv260-chat {health.get('version', '?')} (demo/chat), "
                                                "POST /v1/audio/speech, response_format mp3, seed 0",
                     "library": tts.get("library"), "date": time.strftime("%Y-%m-%d"),
                     "made_by": "demo/tts/scripts/make_samples.py"})
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

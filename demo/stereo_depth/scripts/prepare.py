#!/usr/bin/env python3
"""prepare.py — the stereo_depth demo's input pairs (doc/plans/STEREO_PLAN.md).

Rectified pairs with ground truth from Middlebury MiddEval3-Q and ETH3D
(``run.pairs`` of them, alternating between the two sets) and every pair in
assets/pairs/<name>/{left,right}.png (yours; no ground truth), each fitted to
the network's H x W as OpenStereo evaluates: scaled down (bilinear) when it
is larger, padded at the top and the right with its edge pixels, divided by
255, ImageNet-normalised and encoded as raw int16 at the input's exponent
(the frontend's ``input`` site, 2^-12):

  build/data/pairs.bin     per pair: left [3][H][W] then right [3][H][W], int16
  build/data/manifest.txt  "<name>\\t<byte offset>" per pair (stereo_pairs.c)
  build/data/meta.json     per pair: set, directory, original size, scale, the
                           scaled size (postprocess.py), the input exponent

usage: inference-scheduler/.venv/bin/python demo/stereo_depth/scripts/prepare.py [--pairs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(HERE))
from stereo_assets import FORMATS, gt_pairs  # noqa: E402

DATA = DEMO / "build" / "data"
USER = DEMO / "assets" / "pairs"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def input_exponent() -> int:
    return int(json.loads(FORMATS.read_text())["exponents"]["input"])


def fit(img: Image.Image, height: int, width: int):
    """(the [H][W][3] padded image as float64 0..255, scale, scaled (w, h))."""
    w, h = img.size
    s = min(1.0, width / w, height / h)
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    sh, sw = a.shape[:2]
    a = np.pad(a, ((height - sh, 0), (0, width - sw), (0, 0)), "edge")
    return a, s, (sw, sh)


def encode(a: np.ndarray, f: int) -> np.ndarray:
    """[H][W][3] 0..255 -> [3][H][W] raw int16 at 2^-f (round half to even, saturate)."""
    x = (a / 255.0 - MEAN) / STD
    return np.clip(np.rint(x * 2.0 ** f), -32768, 32767).astype("<i2").transpose(2, 0, 1).copy()


def select(n: int) -> list:
    """n pairs with ground truth, alternating Middlebury / ETH3D (all: n == 0;
    none: n < 0, your pairs only)."""
    if n < 0:
        return []
    mb = [p for p in gt_pairs() if p[0] == "middlebury"]
    eth = [p for p in gt_pairs() if p[0] == "eth3d"]
    out = []
    for i in range(max(len(mb), len(eth))):
        out += ([mb[i]] if i < len(mb) else []) + ([eth[i]] if i < len(eth) else [])
    return out if n <= 0 else out[:n]


def prepare(n: int, height: int, width: int, out_dir: Path = DATA) -> list:
    items = [(s, d, d / "im0.png", d / "im1.png") for s, d in select(n)]
    if USER.is_dir():
        items += [("user", d, d / "left.png", d / "right.png") for d in sorted(USER.iterdir())
                  if (d / "left.png").exists() and (d / "right.png").exists()]
    if not items:
        raise SystemExit("no pairs: run scripts/stereo_assets.py, or put pairs in assets/pairs/<name>/")
    f = input_exponent()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, off = [], 0
    with open(out_dir / "pairs.bin", "wb") as fb, open(out_dir / "manifest.txt", "w") as fm:
        for s, d, lp, rp in items:
            li, ri = Image.open(lp), Image.open(rp)
            if li.size != ri.size:
                raise SystemExit(f"{d}: left {li.size} and right {ri.size} differ")
            la, scale, ssz = fit(li, height, width)
            ra, _, _ = fit(ri, height, width)
            name = f"{s}_{d.name}" if s != "user" else d.name
            fm.write(f"{name}\t{off}\n")
            for a in (la, ra):
                raw = encode(a, f)
                fb.write(raw.tobytes())
                off += raw.nbytes
            meta.append({"name": name, "set": s, "dir": str(d), "size": list(li.size), "scale": scale,
                         "scaled": list(ssz), "height": height, "width": width, "input_exp": f})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--pairs", type=int, default=12, help="pairs with ground truth (0: all 42; -1: none)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args(argv)
    meta = prepare(a.pairs, a.height, a.width)
    print(f"{len(meta)} pairs ({sum(m['set'] != 'user' for m in meta)} with ground truth) -> {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

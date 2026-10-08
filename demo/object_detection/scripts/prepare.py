#!/usr/bin/env python3
"""prepare.py — the object_detection demo's input images (doc/plans/YOLO_PLAN.md).

The first N COCO128 images (with their labels, for the mAP) and every picture
in assets/images/ (yours; no labels), letterboxed to 640 x 640 as YOLOv5 does
(yolo_post.letterbox) and encoded as raw int16 Q8.8 (round(x * 256), x in
[0, 1]) in NCHW:

  build/data/images.bin    one [3][640][640] int16 image after another
  build/data/manifest.txt  "<name>\\t<byte offset>" per image (detect_images.c)
  build/data/meta.json     per image: source path, size, letterbox ratio and
                           padding, labels path or null (scripts/postprocess.py)

usage: inference-scheduler/.venv/bin/python demo/object_detection/scripts/prepare.py [--images N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import yolo_post as yp            # noqa: E402
from yolo_study import coco128    # noqa: E402

DATA = os.path.join(DEMO, "build", "data")
USER = os.path.join(DEMO, "assets", "images")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def encode(x: np.ndarray) -> np.ndarray:
    """[3][640][640] in [0, 1] -> raw Q8.8 int16 (the FPGA graph's input)."""
    return np.round(x * 256.0).astype("<i2")


def prepare(n_coco: int, out_dir: str = DATA) -> list:
    items = [(p, lab) for p, lab in coco128()[:n_coco]]
    if os.path.isdir(USER):
        items += [(os.path.join(USER, f), None) for f in sorted(os.listdir(USER)) if f.lower().endswith(EXTS)]
    if not items:
        raise SystemExit("no images: run scripts/yolo_study.py fetch, or put pictures in assets/images/")
    os.makedirs(out_dir, exist_ok=True)
    meta, off = [], 0
    with open(os.path.join(out_dir, "images.bin"), "wb") as fb, \
            open(os.path.join(out_dir, "manifest.txt"), "w") as fm:
        for path, lab in items:
            img = Image.open(path)
            x, r, pad = yp.letterbox(img)
            raw = encode(x)
            name = os.path.splitext(os.path.basename(path))[0]
            fm.write(f"{name}\t{off}\n")
            fb.write(raw.tobytes())
            off += raw.nbytes
            meta.append({"name": name, "path": path, "size": list(img.size), "ratio": r, "pad": list(pad),
                         "labels": lab if lab and os.path.exists(lab) else None})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--images", type=int, default=16, help="COCO128 images (all: 128)")
    a = ap.parse_args(argv)
    meta = prepare(a.images)
    print(f"{len(meta)} images ({sum(m['labels'] is not None for m in meta)} with labels) -> {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

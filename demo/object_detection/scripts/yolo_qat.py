#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""yolo_qat.py — fine-tune YOLOv5n for the 8-bit configuration: distillation from the float
model through a forward that follows the integer specification of an onnx_i8 policy
(doc/plans/INT8_PLAN.md §9.8; demo/image_classification/scripts/onnx_qat.py: the cut graph
in torch, int8 weights per output channel and each tensor at its calibrated format, straight
through).

Loss: the mean squared difference of the sigmoids of the three head maps (everything YOLO's
decode reads goes through a sigmoid), student against the float teacher, on unlabeled
images letterboxed to 640 (COCO val2017 in INT8_PLAN: none of COCO128's images), random
flips; the formats from onnx_i8.calibrate on --calib-images of them.  Writes the fine-tuned
weights into a copy of the cut graph (float32): `yolo_study.py i8 --onnx OUT.onnx`
recalibrates and judges it bit for bit.

usage: <python with torch and onnx> demo/object_detection/scripts/yolo_qat.py OUT.onnx --images DIR
           [--policy i8ba+a16+in16] [--steps 3000] [--batch 4] [--lr 2e-5]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "demo", "image_classification", "scripts"))
import onnx_i8 as oi  # noqa: E402
import onnx_qat as oq  # noqa: E402
import yolo_post as yp  # noqa: E402
import yolo_study as ys  # noqa: E402


def images(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def load(path, flip=False):
    img = Image.open(path).convert("RGB")
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    x, _, _ = yp.letterbox(img)
    return np.asarray(x, np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--images", required=True, help="a directory of unlabeled training images")
    ap.add_argument("--policy", default="i8ba+a16+in16")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--calib-images", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    files = images(a.images)
    rng.shuffle(files)
    t0 = time.time()
    g = oi.Graph(ys.model_path(True))
    cal = [{g.inputs[0]: load(p)[None].astype(np.float64)} for p in files[:a.calib_images]]
    fm = oi.calibrate(g, cal, **oi.policy_options(a.policy))
    print(f"{len(files)} images; formats of {a.policy} from {len(cal)} ({time.time() - t0:.0f} s)", flush=True)
    train = files[a.calib_images:]

    def batches():
        while True:
            yield np.stack([load(p, flip=rng.random() < 0.5) for p in rng.sample(train, a.batch)])
    student, log = oq.distill(g, fm, batches(), a.steps, a.lr, loss="sigmoid")
    oq.export(student, ys.model_path(True), a.out)
    json.dump(dict(policy=a.policy, steps=a.steps, batch=a.batch, lr=a.lr, images=len(files), loss=log),
              open(os.path.splitext(a.out)[0] + "_qat_log.json", "w"))
    print(f"-> {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

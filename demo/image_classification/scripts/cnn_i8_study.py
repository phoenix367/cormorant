#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""cnn_i8_study.py — the 8-bit configuration (doc/plans/INT8_PLAN.md §9.7) of the shipped CNN
classifiers: float64 (onnx_i8's reference) against the 8-bit emulation (onnx_i8, the integer
specification src/int8_spec.py) — top-1 and the agreement with float.

  ImageNet  MobileNet v1 / v2, ResNet-18 (the image-classification demo's ONNX models) on
            Imagenette's validation images: ten ImageNet classes, 1000-way top-1; Resize(256) +
            CenterCrop(224), each model's normalisation; calibration on Imagenette's training
            images (--calib per class)
  MNIST     LeNet and mnist-8 (the MNIST demo's) on the MNIST test set, pixels p / 256 as the demo
            encodes them; calibration on the last --calib test images, evaluation on the others

Policies (onnx_i8.policy_options): i8 — every tensor static int8 per tensor (unsigned after ReLU),
int8 weights per output channel; i8b — and the Conv outputs that only Convs read in blocks of 32
channels; i8ba — any writer's; +a16 / +in16 modifiers.

  fetch   Imagenette (imagenette2-320.tgz, pinned SHA-256) into assets/imagenette
  study   --models resnet18,mobilenet_v2,mobilenet_v1,lenet,mnist8 [--policies i8,i8b]
          [--per-class N] [--calib N] [--onnx FILE] [--out DIR]  -> DIR/cnn_i8_<model>.json
          (--onnx: a fine-tuned graph of the one model, judged against its float and the
          original's float)
  qat     --models M --policies P --qat-out FILE.onnx [--steps N] [--batch N] [--lr X]:
          fine-tune an ImageNet classifier for the policy (onnx_qat: distillation of the
          logits on Imagenette's training images outside the calibration ones, random
          flips; a python with torch and onnx)

The ONNX models come from the demos' download_assets.py (assets/models/).

usage: inference-scheduler/.venv/bin/python demo/image_classification/scripts/cnn_i8_study.py CMD
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tarfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(DEMO, "..", ".."))
sys.path.insert(0, HERE)
import onnx_i8 as oi  # noqa: E402

ASSETS = os.path.join(DEMO, "assets")
MNIST = os.path.join(REPO, "demo", "mnist", "assets")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_SHA = "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c"
IMAGENETTE = os.path.join(ASSETS, "imagenette", "imagenette2-320")
# the ten Imagenette classes: WordNet id -> ImageNet-1k index
WNID = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
        "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
MODELS = {
    # name: ONNX file, data set, normalisation, logit index of ImageNet class 0
    "mobilenet_v1": dict(path=os.path.join(ASSETS, "models", "mobilenet_v1_1.0_224_no_softmax.onnx"),
                         data="imagenet", norm="tf", offset=1),
    "mobilenet_v2": dict(path=os.path.join(ASSETS, "models", "mobilenetv2-12_simplified.onnx"),
                         data="imagenet", norm="imagenet", offset=0),
    "resnet18": dict(path=os.path.join(ASSETS, "models", "resnet18-simplified-fused.onnx"),
                     data="imagenet", norm="imagenet", offset=0),
    "lenet": dict(path=os.path.join(MNIST, "models", "lenet_simplified.onnx"), data="mnist"),
    "mnist8": dict(path=os.path.join(MNIST, "models", "mnist-simplified.onnx"), data="mnist"),
}
BATCH = 16


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_fetch(_args):
    d = os.path.dirname(IMAGENETTE)
    os.makedirs(d, exist_ok=True)
    tgz = os.path.join(d, "imagenette2-320.tgz")
    if not (os.path.exists(tgz) and sha256(tgz) == IMAGENETTE_SHA):
        print(f"downloading {IMAGENETTE_URL}", flush=True)
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
    got = sha256(tgz)
    if got != IMAGENETTE_SHA:
        raise SystemExit(f"{tgz}: sha256 {got}, expected {IMAGENETTE_SHA}")
    if not os.path.isdir(IMAGENETTE):
        with tarfile.open(tgz) as t:
            t.extractall(d, filter="data")
    print(f"Imagenette in {IMAGENETTE}")


# ------------------------------------------------------------------ data
def imagenet_image(path, norm):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    a = np.asarray(im.crop((l, t, l + 224, t + 224)), np.float64)
    a = a / 127.5 - 1.0 if norm == "tf" else (a / 255.0 - MEAN) / STD
    return a.transpose(2, 0, 1)


def imagenette(split, per_class, norm):
    if not os.path.isdir(IMAGENETTE):
        raise SystemExit(f"no Imagenette in {IMAGENETTE}: run `cnn_i8_study.py fetch`")
    xs, ys = [], []
    for wn, c in WNID.items():
        d = os.path.join(IMAGENETTE, split, wn)
        for f in sorted(os.listdir(d))[:per_class]:
            xs.append(imagenet_image(os.path.join(d, f), norm))
            ys.append(c)
    return np.stack(xs), np.array(ys)


def mnist_test():
    d = os.path.join(MNIST, "data")
    with open(os.path.join(d, "t10k-images-idx3-ubyte"), "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), np.uint8).reshape(n, 1, r, c)
    with open(os.path.join(d, "t10k-labels-idx1-ubyte"), "rb") as f:
        f.read(8)
        y = np.frombuffer(f.read(), np.uint8).astype(np.int64)
    return x.astype(np.float64) / 256.0, y


def data(spec, per_class, calib):
    if spec["data"] == "imagenet":
        xv, yv = imagenette("val", per_class, spec["norm"])
        xc, _ = imagenette("train", calib, spec["norm"])
        return xv, yv + spec["offset"], xc
    x, y = mnist_test()
    nc = calib * 20
    n = per_class * 10 if per_class else len(x) - nc
    return x[:n], y[:n], x[len(x) - nc:]


# ------------------------------------------------------------------ study
def predict(fn, x):
    out = []
    for i in range(0, len(x), BATCH):
        out.append(np.asarray(fn(x[i:i + BATCH])).reshape(min(BATCH, len(x) - i), -1))
    return np.concatenate(out)


def study_model(name, policies, per_class, calib, log, onnx=None):
    spec = MODELS[name]
    if not os.path.exists(spec["path"]):
        raise SystemExit(f"{spec['path']} missing: run the demo's download_assets.py")
    t0 = time.time()
    g = oi.Graph(spec["path"])
    xv, yv, xc = data(spec, per_class, calib)
    inp = g.inputs[0]
    log(f"[{name}] {len(xv)} evaluation, {len(xc)} calibration inputs  ({time.time() - t0:.0f}s)")
    lf = predict(lambda b: g.run_float({inp: b})[0], xv)
    pf = lf.argmax(1)
    res = {"float": dict(top1=float((pf == yv).mean()))}
    log(f"[{name}] float top-1 {res['float']['top1'] * 100:.2f} %  ({time.time() - t0:.0f}s)")
    if onnx:                                         # a fine-tuned graph: its own float, then its policies
        g = oi.Graph(onnx)
        lo = predict(lambda b: g.run_float({inp: b})[0], xv)
        res["float (fine-tuned)"] = dict(top1=float((lo.argmax(1) == yv).mean()),
                                         agree=float((lo.argmax(1) == pf).mean()))
        log(f"[{name}] fine-tuned float top-1 {res['float (fine-tuned)']['top1'] * 100:.2f} %")
    cal = [{inp: xc[i:i + BATCH]} for i in range(0, len(xc), BATCH)]
    for pol in policies:
        fm = oi.calibrate(g, cal, **oi.policy_options(pol))
        li = predict(lambda b, fm=fm: g.run_i8({inp: b}, fm)[0], xv)
        pi = li.argmax(1)
        rel = float(np.linalg.norm(li - lf) / np.linalg.norm(lf))
        r = dict(top1=float((pi == yv).mean()), agree=float((pi == pf).mean()), logits_rel=rel,
                 formats={k: repr(v) for k, v in fm.items()})
        res[pol] = r
        kinds = {}
        for v in fm.values():
            kinds[v.kind + ("u" if v.unsigned else "")] = kinds.get(v.kind + ("u" if v.unsigned else ""), 0) + 1
        log(f"[{name}] {pol:4s} top-1 {r['top1'] * 100:.2f} % ({(r['top1'] - res['float']['top1']) * 100:+.2f}), "
            f"agreement {r['agree'] * 100:.2f} %, logits rel {rel:.4f}, tensors {kinds}  ({time.time() - t0:.0f}s)")
    return dict(model=name, n=len(xv), n_calib=len(xc), onnx=onnx, results=res)


def cmd_study(args):
    out = args.out or os.path.join(ASSETS, "study")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "cnn_i8_study.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()
    for name in args.models.split(","):
        r = study_model(name, args.policies.split(","), args.per_class, args.calib, log, args.onnx)
        with open(os.path.join(out, f"cnn_i8_{name}{'_ft' if args.onnx else ''}.json"), "w") as f:
            json.dump(r, f, indent=1)


def cmd_qat(args):
    import random
    import torch
    import onnx_qat as oq
    name, pol = args.models, args.policies
    spec = MODELS[name]
    assert spec["data"] == "imagenet", "qat: the ImageNet classifiers"
    rng = random.Random(0)
    torch.manual_seed(0)
    g = oi.Graph(spec["path"])
    _, _, xc = data(spec, 1, args.calib)
    fm = oi.calibrate(g, [{g.inputs[0]: xc[i:i + BATCH]} for i in range(0, len(xc), BATCH)],
                      **oi.policy_options(pol))
    files = [os.path.join(IMAGENETTE, "train", wn, f) for wn in WNID
             for f in sorted(os.listdir(os.path.join(IMAGENETTE, "train", wn)))[args.calib:]]
    print(f"[{name}] {pol}: {len(files)} training images", flush=True)

    def load(p):
        a = imagenet_image(p, spec["norm"])
        return a[:, :, ::-1] if rng.random() < 0.5 else a

    def batches():
        while True:
            yield np.stack([load(p) for p in rng.sample(files, args.batch)]).astype(np.float32)
    student, log = oq.distill(g, fm, batches(), args.steps, args.lr, loss="softmax")
    oq.export(student, spec["path"], args.qat_out)
    with open(os.path.splitext(args.qat_out)[0] + "_qat_log.json", "w") as f:
        json.dump(dict(model=name, policy=pol, steps=args.steps, batch=args.batch, lr=args.lr, loss=log), f)
    print(f"-> {args.qat_out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "study", "qat"])
    ap.add_argument("--models", default="mobilenet_v1,mobilenet_v2,resnet18,lenet,mnist8")
    ap.add_argument("--policies", default="i8,i8b")
    ap.add_argument("--per-class", type=int, default=50,
                    help="evaluation images per class (Imagenette; MNIST: x 10, 0 = every test image)")
    ap.add_argument("--calib", type=int, default=10,
                    help="calibration images per class (Imagenette) / last test images (MNIST: x 20)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--onnx", default=None, help="study: a fine-tuned graph of the one model")
    ap.add_argument("--qat-out", default=None, help="qat: the fine-tuned graph to write")
    ap.add_argument("--steps", type=int, default=2000, help="qat")
    ap.add_argument("--batch", type=int, default=16, help="qat")
    ap.add_argument("--lr", type=float, default=1e-5, help="qat")
    args = ap.parse_args(argv)
    {"fetch": cmd_fetch, "study": cmd_study, "qat": cmd_qat}[args.cmd](args)


if __name__ == "__main__":
    main()

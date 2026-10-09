#!/usr/bin/env python3
"""sim_quality.py — the implemented graph's quality (doc/plans/STEREO_PLAN.md §4).

The frontend (inference-scheduler/src/stereo.py) at each pair's resolution
(padded to a multiple of 32, as the study), simulated by the scheduler
(CodeGenerator._forward_pass: what the generated C computes, bit for bit),
against the ground truth of the study's 42 pairs (Middlebury MiddEval3-Q,
ETH3D two-view).  -> assets/study/sim_quality[_TAG].json

usage: inference-scheduler/.venv/bin/python demo/stereo_depth/scripts/sim_quality.py [--stride N] [--pc T]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

DEMO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO.parent.parent / "inference-scheduler"))
from src.codegen import CodeGenerator                                       # noqa: E402
from src.graph import OnnxGraph                                             # noqa: E402
from src.stereo import LightStereoFrontend, load_checkpoint, load_formats   # noqa: E402

A = DEMO / "assets"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def pfm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        sc = float(f.readline())
        d = np.fromfile(f, "<f4" if sc < 0 else ">f4").reshape(h, w)
    return np.flipud(d).astype(np.float64)


def prep(img, th, tw, f):
    """RightTopPad (edge), /255, ImageNet normalisation, on the input's 2^-f grid."""
    h, w = img.shape[:2]
    img = np.pad(img, ((th - h, 0), (0, tw - w), (0, 0)), "edge")
    x = (img / 255.0 - MEAN) / STD
    x = np.clip(np.round(x * 2.0 ** f), -32768, 32767) * 2.0 ** -f
    return x.transpose(2, 0, 1)[None]


def pairs():
    out = []
    for s, root in (("middlebury", A / "data" / "MiddEval3" / "trainingQ"), ("eth3d", A / "data" / "eth3d")):
        out += [(s, d) for d in sorted(root.iterdir()) if (d / "disp0GT.pfm").exists()]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pc", type=float, default=None,
                    help="per-channel weight exponents where per-tensor rounding loses more than this (0: all)")
    a = ap.parse_args(argv)
    sd = load_checkpoint(A / "ckpt" / "StereoAnything-LightStereo_S.pt")
    fm = load_formats(DEMO / "lightstereo_s_formats.json")
    rows = []
    t_all = time.time()
    for s, d in pairs()[::a.stride]:
        im_l, im_r = rgb(d / "im0.png"), rgb(d / "im1.png")
        gt = pfm(d / "disp0GT.pfm")
        mask = np.asarray(Image.open(d / "mask0nocc.png")) == 255
        valid = np.isfinite(gt) & (gt > 0) & (gt < 192) & mask
        h, w = im_l.shape[:2]
        th, tw = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
        fe = LightStereoFrontend(sd, fm, th, tw, per_channel=a.pc)
        cg = CodeGenerator(OnnxGraph(fe.build()), model_path="lightstereo.onnx")
        f = fe.exponents["left"]
        t0 = time.time()
        disp = cg._forward_pass({"left": prep(im_l, th, tw, f), "right": prep(im_r, th, tw, f)})["disparity"]
        disp = disp[th - h:, :w]
        e = np.abs(disp - gt)[valid]
        row = {"set": s, "name": d.name, "epe": float(e.mean()), "bad1": float((e > 1).mean() * 100),
               "bad2": float((e > 2).mean() * 100), "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"{s:10s} {d.name:18s} {w}x{h} EPE {row['epe']:.3f} bad2 {row['bad2']:5.2f}", flush=True)
    summary = {}
    for s in ("middlebury", "eth3d", "all"):
        rs = [r for r in rows if s in ("all", r["set"])]
        if rs:
            summary[s] = {k: float(np.mean([r[k] for r in rs])) for k in ("epe", "bad1", "bad2")}
            print(f"{s:10s} EPE {summary[s]['epe']:.3f} bad1 {summary[s]['bad1']:.2f} bad2 {summary[s]['bad2']:.2f}")
    out = A / "study" / f"sim_quality{'_' + a.tag if a.tag else ''}.json"
    out.write_text(json.dumps({"pairs": rows, "summary": summary, "seconds": round(time.time() - t_all, 1)},
                              indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

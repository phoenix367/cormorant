#!/usr/bin/env python3
"""postprocess.py — the host half of the object_detection demo (doc/plans/YOLO_PLAN.md).

From the board's heads.bin (detect_images.c: the three raw int16 head maps of
every image) and build/data/meta.json (prepare.py):

  detections   decode + class-aware NMS (yolo_post; conf / iou of the
               config's run section) in the original images' pixels
               -> build/results/detections.json, annotated JPEGs in
               build/results/annotated/
  mAP          the labelled (COCO128) images at val.py's settings, the board
               against float (onnxruntime on the same letterboxed images)
  bit-exact    the first K images' head maps against the scheduler's
               simulation of the generated C (the board must equal it)

usage: inference-scheduler/.venv/bin/python demo/object_detection/scripts/postprocess.py
           [--heads build/results/heads.bin] [--check K]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import yolo_post as yp                                    # noqa: E402
from yolo_study import HEADS, VAL, model_path            # noqa: E402

DATA = os.path.join(DEMO, "build", "data")
RESULTS = os.path.join(DEMO, "build", "results")
SHAPES = [(255, 80, 80), (255, 40, 40), (255, 20, 20)]
PER_IMAGE = sum(int(np.prod(s)) for s in SHAPES)


def class_names() -> list:
    """COCO's 80 names from the export's metadata (names: {0: 'person', ...})."""
    import onnx
    m = onnx.load(model_path(), load_external_data=False)
    names = ast.literal_eval(next(p.value for p in m.metadata_props if p.key == "names"))
    return [names[i] for i in range(len(names))]


def read_heads(path: str, n: int) -> list:
    """heads.bin -> per image [P3, P4, P5] raw int16 arrays [255][ny][nx]."""
    raw = np.fromfile(path, "<i2")
    if raw.size != n * PER_IMAGE:
        raise SystemExit(f"{path}: {raw.size} values, expected {n} x {PER_IMAGE}")
    out = []
    for i in range(n):
        o, hs = i * PER_IMAGE, []
        for s in SHAPES:
            k = int(np.prod(s))
            hs.append(raw[o:o + k].reshape(s))
            o += k
        out.append(hs)
    return out


def detect(heads_raw, m, conf=0.25, iou=0.45, **kw) -> np.ndarray:
    """Raw int16 head maps (Q8.8) -> detections [K][6] in the image's pixels."""
    hs = [h.astype(np.float64) / 256.0 for h in heads_raw]
    det = yp.nms(yp.decode(hs), conf=conf, iou=iou, **kw)
    return yp.to_image(det, m["ratio"], tuple(m["pad"]), tuple(m["size"]))


def annotate(m, det, names, path) -> None:
    img = Image.open(m["path"]).convert("RGB")
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, s, c in det:
        col = tuple(int(v) for v in np.random.default_rng(int(c)).integers(64, 256, 3))
        d.rectangle([x1, y1, x2, y2], outline=col, width=max(2, round(min(img.size) / 250)))
        txt = f"{names[int(c)]} {s:.2f}"
        tw = d.textlength(txt)
        d.rectangle([x1, max(0, y1 - 12), x1 + tw + 4, max(12, y1)], fill=col)
        d.text((x1 + 2, max(0, y1 - 12)), txt, fill=(0, 0, 0))
    img.save(path, quality=90)


def simulate(meta_items, data_dir) -> list:
    """The scheduler's simulation of the first images' head maps (raw int16)."""
    from generate_project import graph_and_generator
    with open(os.devnull, "w") as dn:
        so, sys.stdout = sys.stdout, dn
        try:
            _, cg = graph_and_generator()
        finally:
            sys.stdout = so
    raw = np.fromfile(os.path.join(data_dir, "images.bin"), "<i2").reshape(-1, 3, 640, 640)
    out = []
    for i, _m in enumerate(meta_items):
        r = cg.simulate({"images": raw[i:i + 1].astype(np.float64) / 256.0})
        out.append([np.round(np.asarray(r[h]) * 256.0).astype(np.int64).reshape(s)
                    for h, s in zip(HEADS, SHAPES, strict=True)])
    return out


def postprocess(heads_path: str, check: int = 2, conf: float = 0.25, iou: float = 0.45,
                data_dir: str = DATA, out_dir: str = RESULTS) -> dict:
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    heads = read_heads(heads_path, len(meta))
    names = class_names()
    ann = os.path.join(out_dir, "annotated")
    os.makedirs(ann, exist_ok=True)
    dets, stats_b, stats_f = {}, [], []
    sess = None
    for m, hs in zip(meta, heads, strict=True):
        det = detect(hs, m, conf, iou)
        dets[m["name"]] = [{"class": names[int(c)], "class_id": int(c), "score": round(float(s), 4),
                            "box": [round(float(v), 1) for v in (x1, y1, x2, y2)]}
                           for x1, y1, x2, y2, s, c in det]
        annotate(m, det, names, os.path.join(ann, m["name"] + ".jpg"))
        if m["labels"]:
            lab = yp.labels_xyxy(open(m["labels"]).read().splitlines(), tuple(m["size"]))
            d = detect(hs, m, **VAL)
            stats_b.append((yp.match(d, lab), d[:, 4], d[:, 5], lab[:, 0]))
            if sess is None:
                import onnxruntime as ort
                sess = ort.InferenceSession(model_path(True), providers=["CPUExecutionProvider"])
            x, _, _ = yp.letterbox(Image.open(m["path"]))
            hf = sess.run(HEADS, {"images": x[None].astype(np.float32)})
            df = yp.to_image(yp.nms(yp.decode(hf), **VAL), m["ratio"], tuple(m["pad"]), tuple(m["size"]))
            stats_f.append((yp.match(df, lab), df[:, 4], df[:, 5], lab[:, 0]))
    json.dump(dets, open(os.path.join(out_dir, "detections.json"), "w"), indent=1)
    res = {"images": len(meta), "detections": int(sum(len(v) for v in dets.values())),
           "annotated": ann, "labelled": len(stats_b)}
    if stats_b:
        b, f = yp.evaluate(stats_b), yp.evaluate(stats_f)
        res.update({"map50": round(b["map50"], 4), "map": round(b["map"], 4),
                    "float_map50": round(f["map50"], 4), "float_map": round(f["map"], 4)})
    if check:
        sims = simulate(meta[:check], data_dir)
        bad = [m["name"] for m, hs, sm in zip(meta, heads, sims, strict=False)
               if any(not np.array_equal(h.astype(np.int64), s) for h, s in zip(hs, sm, strict=True))]
        res["bit_exact"] = {"checked": len(sims), "differ": bad}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--heads", default=os.path.join(RESULTS, "heads.bin"))
    ap.add_argument("--check", type=int, default=2, help="images compared with the simulation")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args(argv)
    r = postprocess(a.heads, a.check, a.conf, a.iou)
    print(json.dumps(r, indent=1))
    return 0 if not r.get("bit_exact", {}).get("differ") else 1


if __name__ == "__main__":
    sys.exit(main())

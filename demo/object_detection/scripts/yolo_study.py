#!/usr/bin/env python3
"""yolo_study.py — model study of YOLOv5n for the KV260 (model-study skill,
route B; doc/plans/YOLO_PLAN.md §3).  Host only.

  fetch     yolov5n.onnx (Ultralytics v7.0) and COCO128 at pinned SHA-256s,
            and the FPGA graph: the network cut at the three Detect convs
            (yolov5n_raw.onnx)
  validate  yolo_post.decode() on the cut graph's head maps against the
            export's own decode (output0 of yolov5n.onnx), onnxruntime
  study     COCO128 under three policies, decode + NMS + val.py's mAP:
              float     onnxruntime on the float input
              q88-in    onnxruntime on the input rounded to Q8.8
              fpga      the scheduler's simulation of the generated C
                        (what the board computes, bit for bit)
            -> assets/study/yolo_study.json

usage: inference-scheduler/.venv/bin/python demo/object_detection/scripts/yolo_study.py CMD
           [--images N] [--workers K]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCHED = os.path.join(REPO, "inference-scheduler")
ASSETS = os.path.join(REPO, "demo", "object_detection", "assets")
sys.path.insert(0, HERE)
import yolo_post as yp  # noqa: E402

MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
MODEL_SHA = "04f0e55c26f58d17145b36045780fe1250d5bd2187543e11568e5141d05b3262"
COCO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_SHA = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
HEADS = [f"/model.24/m.{i}/Conv_output_0" for i in range(3)]
VAL = dict(conf=0.001, iou=0.6, max_det=300, multi_label=True)        # val.py's NMS


def model_path(raw: bool = False) -> str:
    return os.path.join(ASSETS, "models", "yolov5n_raw.onnx" if raw else "yolov5n.onnx")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _get(url: str, path: str, sha: str) -> None:
    if os.path.exists(path) and _sha(path) == sha:
        print(f"  {os.path.basename(path)}: present, SHA-256 ok")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  {os.path.basename(path)} <- {url}")
    urllib.request.urlretrieve(url, path + ".part")
    got = _sha(path + ".part")
    if got != sha:
        raise SystemExit(f"{path}: SHA-256 {got}, expected {sha}")
    os.replace(path + ".part", path)


def to_float32(m):
    """The release's float16 graph in float32 (exact: every float16 is a
    float32): initializers, Constant values and every tensor type."""
    import onnx
    from onnx import TensorProto, numpy_helper
    g = m.graph
    for i, t in enumerate(g.initializer):
        if t.data_type == TensorProto.FLOAT16:
            g.initializer[i].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    for n in g.node:
        for at in n.attribute:
            if at.type == onnx.AttributeProto.TENSOR and at.t.data_type == TensorProto.FLOAT16:
                at.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(at.t).astype(np.float32), at.t.name))
    for v in list(g.input) + list(g.output) + list(g.value_info):
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = TensorProto.FLOAT
    return m


def cut_model() -> str:
    """The network up to the three Detect convs (their outputs are the FPGA
    graph's outputs; the decode runs on the host), in float32."""
    import onnx
    from onnx import utils
    utils.extract_model(model_path(), model_path(True), ["images"], HEADS)
    m = to_float32(onnx.load(model_path(True)))
    m.metadata_props.add(key="axi.yolo.source",
                         value=f"{MODEL_URL} sha256 {MODEL_SHA}, cut at {HEADS}, float16 -> float32")
    onnx.checker.check_model(m)
    onnx.save(m, model_path(True))
    return _sha(model_path(True))


def coco128():
    """[(image path, label path)] of COCO128, sorted by name."""
    d = os.path.join(ASSETS, "coco128")
    ims = sorted(os.listdir(os.path.join(d, "images", "train2017")))
    return [(os.path.join(d, "images", "train2017", f),
             os.path.join(d, "labels", "train2017", os.path.splitext(f)[0] + ".txt")) for f in ims]


def cmd_fetch(a=None) -> int:  # noqa: ARG001
    _get(MODEL_URL, model_path(), MODEL_SHA)
    z = os.path.join(ASSETS, "coco128.zip")
    _get(COCO_URL, z, COCO_SHA)
    if not os.path.isdir(os.path.join(ASSETS, "coco128", "images")):
        zipfile.ZipFile(z).extractall(ASSETS)
    print(f"  coco128: {len(coco128())} images")
    print(f"  yolov5n_raw.onnx (cut at the Detect convs): SHA-256 {cut_model()}")
    return 0


def _ort(raw: bool):
    import onnxruntime as ort
    return ort.InferenceSession(model_path(raw), providers=["CPUExecutionProvider"])


def cmd_validate(a) -> int:
    full, cut = _ort(False), _ort(True)
    worst = 0.0
    for im, _ in coco128()[:a.images or 4]:
        x, _, _ = yp.letterbox(Image.open(im))
        ref = full.run(["output0"], {"images": x[None].astype(np.float16)})[0][0].astype(np.float64)
        got = yp.decode(cut.run(HEADS, {"images": x[None].astype(np.float16).astype(np.float32)}))
        d = float(np.abs(got - ref).max() / max(1.0, np.abs(ref).max()))
        worst = max(worst, d)
        print(f"  {os.path.basename(im)}: decode vs output0 max |diff| / max {d:.2e}")
    # the export computes in float16 (the release model), the cut graph in float32
    print(f"validate: {'PASS' if worst < 1e-2 else 'FAIL'} (worst {worst:.2e}; float16 export vs float32)")
    return 0 if worst < 1e-2 else 1


_CG = None


def _sim_init():
    global _CG
    sys.path.insert(0, SCHED)
    from src.codegen import CodeGenerator
    from src.graph import OnnxGraph
    with open(os.devnull, "w") as dn:
        so, sys.stdout = sys.stdout, dn
        try:
            _CG = CodeGenerator(OnnxGraph(model_path(True), fuse_act=True, s2d_stem=True),
                                model_path=model_path(True))
        finally:
            sys.stdout = so


def _sim(xq: np.ndarray):
    out = _CG.simulate({"images": xq[None]})
    return [np.asarray(out[h], np.float64) for h in HEADS]


def cmd_study(a) -> int:
    items = coco128()[:a.images or None]
    cut = _ort(True)
    pre = []
    for im, lab in items:
        img = Image.open(im)
        x, r, pad = yp.letterbox(img)
        xq = np.round(x * 256.0) / 256.0
        pre.append((im, lab, img.size, r, pad, x, xq))
    t0 = time.time()
    with Pool(a.workers, initializer=_sim_init) as pool:
        sims = pool.map(_sim, [p[6] for p in pre], chunksize=1)
    t_sim = time.time() - t0
    pols = {"float": [], "q88-in": [], "fpga": []}
    dets = {k: [] for k in pols}
    rel = []
    for (_im, lab, shape, r, pad, x, xq), sim in zip(pre, sims, strict=True):
        lines = open(lab).read().splitlines() if os.path.exists(lab) else []
        labels = yp.labels_xyxy(lines, shape)
        heads = {"float": cut.run(HEADS, {"images": x[None].astype(np.float32)}),
                 "q88-in": cut.run(HEADS, {"images": xq[None].astype(np.float32)}),
                 "fpga": sim}
        for i, h in enumerate(heads["fpga"]):
            f = np.asarray(heads["float"][i], np.float64).reshape(h.shape)
            rel.append(float(np.linalg.norm(h - f) / max(np.linalg.norm(f), 1e-12)))
        for k, hs in heads.items():
            d = yp.to_image(yp.nms(yp.decode(hs), **VAL), r, pad, shape)
            pols[k].append((yp.match(d, labels), d[:, 4], d[:, 5], labels[:, 0]))
            dets[k].append(yp.to_image(yp.nms(yp.decode(hs)), r, pad, shape))       # conf 0.25 (display)
    res = {k: yp.evaluate(v) for k, v in pols.items()}
    # the displayed detections (conf 0.25): FPGA against float, same class and IoU >= 0.5
    agree = n_f = n_q = 0
    for df, dq in zip(dets["float"], dets["fpga"], strict=True):
        n_f, n_q = n_f + len(df), n_q + len(dq)
        if len(df) and len(dq):
            iou = yp.box_iou(df[:, :4], dq[:, :4]) * (df[:, 5:6] == dq[:, 5])
            agree += int((iou.max(1) >= 0.5).sum())
    out = {"images": len(items), "policies": res, "head_rel_l2_mean": float(np.mean(rel)),
           "head_rel_l2_max": float(np.max(rel)), "displayed": {"float": n_f, "fpga": n_q, "matched": agree},
           "sim_s": round(t_sim, 1)}
    print(f"COCO128, {len(items)} images (simulation {t_sim:.0f} s with {a.workers} workers):")
    for k, v in res.items():
        print(f"  {k:7s} mAP@0.5 {v['map50']:.4f}  mAP@0.5:0.95 {v['map']:.4f}  "
              f"({v['detections']} detections, {v['labels']} labels, {v['classes']} classes)")
    print(f"  head maps, FPGA vs float: rel L2 mean {np.mean(rel) * 100:.2f} %, max {np.max(rel) * 100:.2f} %")
    print(f"  displayed (conf 0.25): float {n_f}, FPGA {n_q}, FPGA matching a float box {agree}")
    os.makedirs(os.path.join(ASSETS, "study"), exist_ok=True)
    path = os.path.join(ASSETS, "study", "yolo_study.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("cmd", choices=("fetch", "validate", "study"))
    ap.add_argument("--images", type=int, default=0, help="the first N COCO128 images (default: all)")
    ap.add_argument("--workers", type=int, default=6, help="study: simulation processes")
    a = ap.parse_args(argv)
    return {"fetch": cmd_fetch, "validate": cmd_validate, "study": cmd_study}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

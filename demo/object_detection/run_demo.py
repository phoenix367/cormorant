#!/usr/bin/env python3
"""run_demo.py — YOLOv5n object detection on the KV260 (doc/plans/YOLO_PLAN.md).

  1. fetch     yolov5n.onnx (Ultralytics v7.0) and COCO128, SHA-256 checked;
               the FPGA graph: the network cut at its Detect convs, float32
  2. prepare   the first run.images COCO128 images + your pictures in
               assets/images/, letterboxed to 640 x 640, raw Q8.8
  3. generate  the inference project (inference-scheduler) + detect_images
  4. board     upload, build, run on the FPGA, fetch the head maps
  5. results   decode + NMS on the host: annotated images, detections.json,
               mAP against the COCO128 labels (board and float), the first
               run.check_images head maps bit for bit against the scheduler's
               simulation -> build/results.json

usage: inference-scheduler/.venv/bin/python demo/object_detection/run_demo.py
           [--config object_detection_config.json] [--skip-download] [--stop-server] [--images N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO / "scripts"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--config", default=str(DEMO / "object_detection_config.json"))
    ap.add_argument("--skip-download", action="store_true", help="use the assets already fetched")
    ap.add_argument("--stop-server", action="store_true", help="stop the chat server for the run, restart it after")
    ap.add_argument("--images", type=int, default=None, help="COCO128 images (default: run.images)")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        print(f"{cfg_path} not found: copy object_detection_config.json.example to it and edit ssh / driver_dirs")
        return 2
    cfg = json.loads(cfg_path.read_text())
    run = cfg.get("run", {})
    steps, metrics, ok = [], {"model": "yolov5n"}, True

    def step(name, fn):
        print(f"== {name}", flush=True)
        t0 = time.monotonic()
        r = fn()
        steps.append({"name": name, "ok": True, "seconds": round(time.monotonic() - t0, 1)})
        return r

    import deploy_and_run
    import generate_project
    import postprocess
    import prepare
    import yolo_study

    if not a.skip_download:
        step("fetch", lambda: yolo_study.cmd_fetch(None))
    meta = step("prepare", lambda: prepare.prepare(a.images if a.images is not None else int(run.get("images", 16))))
    print(f"  {len(meta)} images ({sum(m['labels'] is not None for m in meta)} with labels)")
    gen = step("generate", lambda: generate_project.generate(DEMO / "build" / "project",
                                                             cfg.get("local", {}).get("driver_dirs", {})))
    if gen["missing_drivers"]:
        print(f"missing driver files {gen['missing_drivers']}: build them (make driver_vectorop_rtl "
              f"driver_conv_rtl driver_pool_rtl) or fix local.driver_dirs")
        return 1
    s = step("board", lambda: deploy_and_run.run(cfg, DEMO / "build" / "project", DEMO / "build" / "data",
                                                 DEMO / "build" / "results", a.stop_server))
    metrics.update({k: s[k] for k in ("images", "mean_ms", "p50_ms", "min_ms", "max_ms", "fps", "bitstream")})
    metrics["throughput_ips"] = s["fps"]
    r = step("results", lambda: postprocess.postprocess(str(DEMO / "build" / "results" / "heads.bin"),
                                                        int(run.get("check_images", 2)),
                                                        float(run.get("conf", 0.25)), float(run.get("iou", 0.45))))
    metrics.update({k: v for k, v in r.items() if k != "annotated"})
    be = r.get("bit_exact", {})
    if be.get("differ"):
        ok = False
        steps[-1]["ok"] = False
    out = DEMO / "build" / "results.json"
    out.write_text(json.dumps([{"name": "yolov5n", "ok": ok, "metrics": metrics, "steps": steps}], indent=1))

    print(f"\nYOLOv5n on the KV260 (bitstream {metrics['bitstream']}): {metrics['images']} images, "
          f"{metrics['mean_ms']:.1f} ms mean / {metrics['p50_ms']:.1f} ms p50 per image "
          f"({metrics['fps']:.1f} FPS, FPGA graph only)")
    print(f"  detections at conf {run.get('conf', 0.25)}: {r['detections']}; annotated images in {r['annotated']}")
    if "map50" in r:
        print(f"  COCO128 ({r['labelled']} labelled images): mAP@0.5 {r['map50']:.3f} "
              f"(float {r['float_map50']:.3f}), mAP@0.5:0.95 {r['map']:.3f} (float {r['float_map']:.3f})")
    if be:
        print("  head maps vs the scheduler's simulation: "
              + ("bit-exact" if not be["differ"] else f"DIFFER {be['differ']}") + f" ({be['checked']} images)")
    print(f"  -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

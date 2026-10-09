#!/usr/bin/env python3
"""run_demo.py — stereo depth (LightStereo-S) on the KV260 (doc/plans/STEREO_PLAN.md).

  1. fetch     the LightStereo-S checkpoint (StereoAnything weights; Hugging
               Face, pinned) and the Middlebury MiddEval3-Q / ETH3D pairs,
               SHA-256 checked; --capture: stereo pairs from the board's
               RealSense camera (scripts/capture.py; then those pairs only)
  2. prepare   run.pairs pairs with ground truth + your pairs in
               assets/pairs/<name>/{left,right}.png, fitted to the network's
               run.height x run.width, raw int16
  3. generate  the frontend graph + the inference project (inference-scheduler)
               + stereo_pairs
  4. board     upload, build, run on the FPGA, fetch the disparity maps
  5. results   the maps at each pair's resolution (PNG, PFM), EPE / bad-2
               against the ground truth, the first run.check_pairs maps bit
               for bit against the scheduler's simulation -> build/results.json

usage: inference-scheduler/.venv/bin/python demo/stereo_depth/run_demo.py
           [--config stereo_depth_config.json] [--skip-download] [--stop-server] [--pairs N] [--profile]
           [--capture]
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
    ap.add_argument("--config", default=str(DEMO / "stereo_depth_config.json"))
    ap.add_argument("--skip-download", action="store_true", help="use the assets already fetched")
    ap.add_argument("--stop-server", action="store_true", help="stop the chat server for the run, restart it after")
    ap.add_argument("--pairs", type=int, default=None,
                    help="pairs with ground truth (default: run.pairs, -1 with --capture; 0: all; -1: none)")
    ap.add_argument("--capture", action="store_true",
                    help="first capture stereo pairs from the board's RealSense camera (scripts/capture.py)")
    ap.add_argument("--profile", action="store_true", help="per-layer times (INFERENCE_PROFILING) in results.json")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        print(f"{cfg_path} not found: copy stereo_depth_config.json.example to it and edit ssh / driver_dirs")
        return 2
    cfg = json.loads(cfg_path.read_text())
    run = cfg.get("run", {})
    H, W = int(run.get("height", 480)), int(run.get("width", 640))
    steps, metrics, ok = [], {"model": "lightstereo_s", "height": H, "width": W}, True

    def step(name, fn):
        print(f"== {name}", flush=True)
        t0 = time.monotonic()
        r = fn()
        steps.append({"name": name, "ok": True, "seconds": round(time.monotonic() - t0, 1)})
        return r

    import capture
    import deploy_and_run
    import generate_project
    import postprocess
    import prepare
    import stereo_assets

    if not a.skip_download:
        step("fetch", lambda: stereo_assets.fetch())
    if a.capture:
        step("capture", lambda: capture.capture(cfg))
    n = a.pairs if a.pairs is not None else (-1 if a.capture else int(run.get("pairs", 12)))
    meta = step("prepare", lambda: prepare.prepare(n, H, W))
    print(f"  {len(meta)} pairs ({sum(m['set'] != 'user' for m in meta)} with ground truth) at {W} x {H}")
    gen = step("generate", lambda: generate_project.generate(DEMO / "build" / "project",
                                                             cfg.get("local", {}).get("driver_dirs", {}), H, W,
                                                             run.get("per_channel")))
    metrics["per_channel_convs"] = gen["per_channel_convs"]
    if gen["missing_drivers"]:
        print(f"missing driver files {gen['missing_drivers']}: build them (make driver_vectorop_rtl "
              f"driver_matmul_rtl driver_conv_rtl) or fix local.driver_dirs")
        return 1
    s = step("board", lambda: deploy_and_run.run(cfg, DEMO / "build" / "project", DEMO / "build" / "data",
                                                 DEMO / "build" / "results", a.stop_server, a.profile))
    metrics.update({k: s[k] for k in ("pairs", "mean_ms", "p50_ms", "min_ms", "max_ms", "fps", "bitstream")})
    metrics["throughput_ips"] = s["fps"]
    if "layer_stats" in s:
        metrics["layer_stats"] = s["layer_stats"]
    r = step("results", lambda: postprocess.postprocess(int(run.get("check_pairs", 2))))
    metrics.update({k: v for k, v in r.items() if k not in ("annotated", "per_pair")})
    be = r.get("bit_exact", {})
    if be.get("differ"):
        ok = False
        steps[-1]["ok"] = False
    out = DEMO / "build" / "results.json"
    out.write_text(json.dumps([{"name": "lightstereo_s", "ok": ok, "metrics": metrics, "steps": steps,
                                "per_pair": r["per_pair"]}], indent=1))

    print(f"\nLightStereo-S on the KV260 (bitstream {metrics['bitstream']}), {W} x {H}: {metrics['pairs']} pairs, "
          f"{metrics['mean_ms']:.1f} ms mean / {metrics['p50_ms']:.1f} ms p50 per pair ({metrics['fps']:.2f} FPS)")
    if "epe" in r:
        print(f"  {r['labelled']} pairs with ground truth: EPE {r['epe']:.3f} px, bad-1 {r['bad1']:.2f} %, "
              f"bad-2 {r['bad2']:.2f} %; disparity maps in {r['annotated']}")
    if be:
        print("  disparity vs the scheduler's simulation: "
              + ("bit-exact" if not be["differ"] else f"DIFFER {be['differ']}") + f" ({be['checked']} pairs)")
    print(f"  -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

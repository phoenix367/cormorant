#!/usr/bin/env python3
# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
camera_loop.py — board-side capture → inference → annotate loop for the
KV260 camera demo.  Runs ON the KV260.

Per frame:
  1. grab a colour frame from the RealSense camera (pyrealsense2),
  2. preprocess it to the NCHW ap_fixed<16,8> blob the FPGA expects
     (preprocessing.preprocess_frame),
  3. hand the blob to the persistent classify_stream process over a pipe
     and read back the top-K JSON,
  4. annotate the original colour frame with the prediction overlay
     (visualization.annotate_frame),
  5. write it atomically to --out so the host display loop can SFTP-pull it.

classify_stream is started once and kept alive for the whole session, so the
FPGA inference pipeline is initialised exactly once.  A blocking write/read
pair against its stdin/stdout keeps exactly one frame in flight.

Shutdown: the loop ends when --stop-file appears (the host touches it on
quit), when --duration elapses, or on an unrecoverable pipe/camera error.
All diagnostics go to stderr; stdout is unused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# preprocessing.py / visualization.py sit next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_args(argv):
    p = argparse.ArgumentParser(description="KV260 camera demo board loop")
    p.add_argument("--binary", required=True,
                   help="path to the compiled classify_stream executable")
    p.add_argument("--labels", required=True,
                   help="imagenet_1001_labels.txt (one label per line)")
    p.add_argument("--out", required=True,
                   help="path the annotated JPEG is written to (atomically)")
    p.add_argument("--stop-file", required=True,
                   help="loop exits once this path exists")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--normalize", default="tf",
                   choices=["tf", "unit", "none"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--power-poll", type=float, default=2.0,
                   help="SOM power-sensor poll interval in seconds")
    p.add_argument("--target-fps", type=float, default=0.0,
                   help="throttle the loop to at most this rate (0 = uncapped)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="auto-stop after this many seconds (0 = until stop file)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # Board-side dependencies are imported lazily so a missing package
    # produces a clear message instead of an import traceback at load time.
    try:
        import numpy as np
        import cv2
        import pyrealsense2 as rs
    except ImportError as exc:
        _log(f"error: missing board dependency — {exc}")
        _log("  the KV260 needs: pyrealsense2 (librealsense), numpy, "
             "opencv-python")
        return 1

    from preprocessing import preprocess_frame
    from visualization import annotate_frame
    from power_monitor import PowerMonitor

    labels = [ln for ln in Path(args.labels).read_text().splitlines()]
    _log(f"camera_loop: loaded {len(labels)} class labels")

    # SOM power sampling runs on its own thread so a (slow) sensor read never
    # stalls the capture/inference loop.
    power = PowerMonitor(poll_interval=args.power_poll)
    power.start()

    # ── start the persistent inference process ───────────────────────────────
    _log(f"camera_loop: launching {args.binary}")
    proc = subprocess.Popen(
        [args.binary, str(args.warmup), str(args.top_k)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)

    ready_line = proc.stdout.readline()
    if not ready_line:
        _log("error: classify_stream exited before the ready handshake")
        return 1
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError:
        _log(f"error: unparseable handshake line: {ready_line!r}")
        return 1
    if ready.get("status") != "ready":
        _log(f"error: classify_stream init failed: {ready}")
        return 1
    expect_numel = ready.get("input_numel")
    _log(f"camera_loop: classify_stream ready (input_numel={expect_numel})")

    # ── start the RealSense colour stream ─────────────────────────────────────
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height,
                      rs.format.bgr8, args.fps)
    try:
        profile = pipeline.start(cfg)
    except Exception as exc:                            # noqa: BLE001
        _log(f"error: could not start the RealSense camera — {exc}")
        _log("  check the USB 3.0 connection and `rs-enumerate-devices`")
        proc.stdin.close()
        proc.wait(timeout=5)
        return 1
    dev_name = profile.get_device().get_info(rs.camera_info.name)
    _log(f"camera_loop: RealSense '{dev_name}' streaming "
         f"{args.width}x{args.height}@{args.fps}")

    out_path  = Path(args.out)
    tmp_path  = out_path.with_suffix(out_path.suffix + ".tmp")
    stop_path = Path(args.stop_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_idx   = 0
    t_start     = time.monotonic()
    fps_window  = t_start
    fps_count   = 0
    disp_fps    = 0.0
    min_period  = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
    exit_reason = "eof"

    try:
        while True:
            if stop_path.exists():
                exit_reason = "stop file"
                break
            if args.duration > 0.0 and \
                    (time.monotonic() - t_start) >= args.duration:
                exit_reason = "duration elapsed"
                break

            loop_t0 = time.monotonic()

            frames = pipeline.wait_for_frames()
            color  = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())

            blob = preprocess_frame(bgr, args.input_size, args.normalize)
            if expect_numel and (len(blob) // 2) != expect_numel:
                _log(f"error: preprocessed {len(blob) // 2} elements but "
                     f"classify_stream expects {expect_numel} — check "
                     f"--input-size")
                exit_reason = "input size mismatch"
                break

            # One frame in flight: write then block for its result.
            try:
                proc.stdin.write(blob)
                proc.stdin.flush()
            except BrokenPipeError:
                _log("error: classify_stream closed its input pipe")
                exit_reason = "inference process died"
                break
            line = proc.stdout.readline()
            if not line:
                _log("error: classify_stream produced no result (crashed?)")
                exit_reason = "inference process died"
                break
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                _log(f"warning: dropping unparseable result line: {line!r}")
                continue

            infer_ms = float(result.get("latency_ms", 0.0))
            preds = []
            for entry in result.get("top", []):
                cid = int(entry.get("class_id", 0))
                label = labels[cid] if 0 <= cid < len(labels) else f"#{cid}"
                preds.append((label, float(entry.get("prob", 0.0))))

            # Display-FPS estimate over a rolling 1-second window.
            fps_count += 1
            now = time.monotonic()
            if now - fps_window >= 1.0:
                disp_fps   = fps_count / (now - fps_window)
                fps_window = now
                fps_count  = 0

            watts = power.watts()
            annotated = annotate_frame(bgr, preds, infer_ms, disp_fps,
                                       frame_idx, power_w=watts)
            ok, enc = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if ok:
                tmp_path.write_bytes(enc.tobytes())
                os.replace(tmp_path, out_path)      # atomic for the puller

            if frame_idx % 10 == 0:
                top1 = preds[0] if preds else ("?", 0.0)
                power_str = f"{watts:.1f}W" if watts is not None else "n/a"
                _log(f"  frame {frame_idx}: {top1[0]} "
                     f"({top1[1] * 100.0:.1f}%)  infer={infer_ms:.1f}ms  "
                     f"display={disp_fps:.1f}fps  power={power_str}")
            frame_idx += 1

            if min_period > 0.0:
                slack = min_period - (time.monotonic() - loop_t0)
                if slack > 0.0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        exit_reason = "interrupted"
    finally:
        _log(f"camera_loop: stopping ({exit_reason}) after "
             f"{frame_idx} frame(s)")
        power.stop()
        try:
            proc.stdin.close()          # EOF → classify_stream shuts down
        except Exception:               # noqa: BLE001
            pass
        try:
            proc.wait(timeout=10)
        except Exception:               # noqa: BLE001
            proc.kill()
        try:
            pipeline.stop()
        except Exception:               # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

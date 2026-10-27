#!/usr/bin/env python3
# Copyright 2025 GradeBuilder SL
# Modifications copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
camera_loop.py — board-side capture → inference → annotate loop for the
KV260 camera demo.  Runs ON the KV260.

Per frame:
  1. grab a colour frame from the camera: a RealSense on USB (pyrealsense2,
     --source realsense) or the KV260's own AR1335 behind the carrier card's
     AP1302 ISP (--source mipi: the capture path of a bitstream built with
     hw/cormorant_hw_128/scripts/bd_mipi_capture.tcl and its overlay
     dts/kv260/cormorant_capture.dts — NV12 frames from the V4L2 node of the
     ISP's media graph, read through v4l2-ctl),
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
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# preprocessing.py / visualization.py sit next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class RealSenseCamera:
    """The colour stream of a RealSense camera on USB."""

    def __init__(self, width: int, height: int, fps: int):
        import pyrealsense2 as rs
        self._rs = rs
        self.width, self.height, self.fps = width, height, fps
        self.pipeline = None
        self.name = "RealSense"

    def start(self) -> None:
        rs = self._rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        try:
            profile = self.pipeline.start(cfg)
        except Exception as exc:                        # noqa: BLE001
            raise RuntimeError(f"could not start the RealSense camera — {exc}; "
                               "check the USB 3.0 connection and "
                               "`rs-enumerate-devices`") from exc
        self.name = "RealSense '" + profile.get_device().get_info(rs.camera_info.name) + "'"

    def read(self):
        """The next BGR frame, or None."""
        import numpy as np
        color = self.pipeline.wait_for_frames().get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


def _media_entities(dev: str) -> dict:
    """{entity name: video / subdev node or ''} of one media device."""
    out = subprocess.run(["media-ctl", "-d", dev, "-p"], capture_output=True,
                         text=True).stdout
    ents, name = {}, None
    for line in out.splitlines():
        m = re.match(r"- entity \d+: (.+?) \(\d+ pads?", line)
        if m:
            name = m.group(1)
            ents[name] = ""
            continue
        m = re.search(r"device node name (\S+)", line)
        if m and name is not None:
            ents[name] = m.group(1)
    return ents


class MipiCamera:
    """The KV260's AR1335 sensor behind the carrier card's AP1302 ISP: the ISP
    scales the 4208 x 3120 sensor to width x height and sends YUV 4:2:0 over
    MIPI CSI-2; the bitstream's frame buffer writer puts NV12 in DDR.  The ISP
    and the receiver are set up with media-ctl, the frames streamed by
    v4l2-ctl (the node is multiplanar-only, which OpenCV's V4L2 backend cannot
    read); a thread keeps only the newest one, so the one-frame-in-flight loop
    never classifies a stale image."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3 // 2
        self.name = "MIPI"
        self.proc = None
        self._cond = threading.Condition()
        self._latest = None
        self._seq = self._taken = 0
        self._done = False

    def _pipeline(self):
        for dev in sorted(glob.glob("/dev/media*")):
            ents = _media_entities(dev)
            isp = next((e for e in ents if e.startswith("ap1302")), None)
            if isp is None:
                continue
            csi = next((e for e in ents if e.endswith(".csiss")), None)
            video = next((n for e, n in ents.items() if n.startswith("/dev/video")), None)
            if csi is None or video is None:
                raise RuntimeError(f"{dev}: the ISP '{isp}' without a CSI-2 receiver "
                                   f"or a video node ({sorted(ents)})")
            return dev, isp, csi, video
        raise RuntimeError("no media device with an AP1302 ISP: load a bitstream "
                           "with the capture path and its overlay "
                           "(dts/kv260/cormorant_capture.dts)")

    def start(self) -> None:
        dev, isp, csi, video = self._pipeline()
        fmt = f"[fmt:VYYUYY8_1X24/{self.width}x{self.height} field:none]"
        r = subprocess.run(["media-ctl", "-d", dev, "-V", f'"{isp}":2 {fmt}'],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"media-ctl: the ISP refused {fmt}: {r.stderr.strip()}")
        for pad in (0, 1):           # the receiver may refuse; it streams what it gets
            subprocess.run(["media-ctl", "-d", dev, "-V", f'"{csi}":{pad} {fmt}'],
                           capture_output=True)
        self.proc = subprocess.Popen(
            ["v4l2-ctl", "-d", video,
             f"--set-fmt-video=width={self.width},height={self.height},pixelformat=NV12",
             "--stream-mmap=4", "--stream-to=-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        threading.Thread(target=self._reader, daemon=True).start()
        self.name = f"MIPI '{isp}' -> {video}"

    def _reader(self) -> None:
        f = self.proc.stdout
        while True:
            buf = bytearray(self.frame_bytes)
            view, got = memoryview(buf), 0
            while got < self.frame_bytes:
                n = f.readinto(view[got:])
                if not n:
                    with self._cond:
                        self._done = True
                        self._cond.notify_all()
                    return
                got += n
            with self._cond:
                self._latest = buf
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 5.0):
        """The newest BGR frame not read before, or None (the stream ended or
        no frame for `timeout` s)."""
        import numpy as np
        import cv2
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._taken or self._done,
                                       timeout=timeout) or self._seq == self._taken:
                return None
            buf, self._taken = self._latest, self._seq
        nv12 = np.frombuffer(buf, np.uint8).reshape(self.height * 3 // 2, self.width)
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


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
    p.add_argument("--source", default="realsense", choices=["realsense", "mipi"],
                   help="realsense: a RealSense on USB; mipi: the KV260's "
                        "AR1335 / AP1302 through the bitstream's capture path")
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
        import numpy  # noqa: F401
        import cv2
        camera = (MipiCamera(args.width, args.height) if args.source == "mipi" else
                  RealSenseCamera(args.width, args.height, args.fps))
    except ImportError as exc:
        _log(f"error: missing board dependency — {exc}")
        _log("  the KV260 needs: numpy, opencv-python, and for --source "
             "realsense pyrealsense2 (librealsense)")
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

    # ── start the camera ──────────────────────────────────────────────────────
    try:
        camera.start()
    except RuntimeError as exc:
        _log(f"error: {exc}")
        proc.stdin.close()
        proc.wait(timeout=5)
        return 1
    _log(f"camera_loop: {camera.name} streaming "
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

            bgr = camera.read()
            if bgr is None:
                if args.source == "mipi":
                    _log("error: no frame from the camera for 5 s")
                    exit_reason = "camera stopped"
                    break
                continue

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
            camera.stop()
        except Exception:               # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

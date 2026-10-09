#!/usr/bin/env python3
"""realsense_capture.py — runs ON the KV260: a rectified stereo pair from an
Intel RealSense D4xx (doc/plans/STEREO_PLAN.md; uploaded by
scripts/capture.py).

The D4xx's two infrared cameras are a rectified, undistorted stereo pair
(the Y8 infrared streams 1 and 2).  For each emitter setting this saves, into
OUT/<name>[_emitter-on|off]/:

  left.png, right.png   infrared 1 / 2, 8-bit grey
  depth_rs.png          the camera's own depth (uint16, units of depth_scale m)
  color.png             the colour image when --color (and the link allows it)
  meta.json             fx, fy, cx, cy of infrared 1 (px), the baseline (m,
                        infrared 1 -> 2), depth_scale, the emitter, the
                        device and its USB link

The stereo pair needs both infrared streams: a D4xx offers them on USB 3,
and on USB 2 with firmware 5.17.3.10 (5.12.3 streams only the left camera
there); without infrared 2 the script says so and stops.

usage (on the board): python3 realsense_capture.py --out DIR [--name NAME] [--emitter on|off|both]
                          [--width 640 --height 480 --fps 15] [--warmup 30] [--color]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="realsense")
    ap.add_argument("--emitter", choices=("on", "off", "both"), default="both",
                    help="the IR dot projector: on (texture), off (passive), both (two pairs)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=30, help="frames to skip (auto exposure)")
    ap.add_argument("--color", action="store_true", help="also save the colour image")
    a = ap.parse_args(argv)

    import cv2
    import numpy as np
    import pyrealsense2 as rs

    devs = rs.context().query_devices()
    if len(devs) == 0:
        print("error: no RealSense device", file=sys.stderr)
        return 2
    dev = devs[0]
    usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
    name = dev.get_info(rs.camera_info.name)
    ir_idx = {p.stream_index() for p in dev.first_depth_sensor().get_stream_profiles()
              if p.stream_type() == rs.stream.infrared}
    if 2 not in ir_idx:
        fw = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else "?"
        print(f"error: {name} (firmware {fw}) on USB {usb} offers infrared stream(s) {sorted(ir_idx)} only: the right "
              f"camera (infrared 2) needs firmware 5.17.3.10 or a USB 3 link",
              file=sys.stderr)
        return 3

    modes = {"on": [1], "off": [0], "both": [1, 0]}[a.emitter]
    out = []
    for em in modes:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.infrared, 1, a.width, a.height, rs.format.y8, a.fps)
        cfg.enable_stream(rs.stream.infrared, 2, a.width, a.height, rs.format.y8, a.fps)
        cfg.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
        if a.color:
            cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, a.fps)
        prof = pipe.start(cfg)
        try:
            ds = prof.get_device().first_depth_sensor()
            if ds.supports(rs.option.emitter_enabled):
                ds.set_option(rs.option.emitter_enabled, float(em))
            for _ in range(a.warmup):
                pipe.wait_for_frames()
            fs = pipe.wait_for_frames()
            ir1, ir2, dep = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_depth_frame()
            p1 = ir1.get_profile().as_video_stream_profile()
            p2 = ir2.get_profile().as_video_stream_profile()
            intr = p1.get_intrinsics()
            ext = p1.get_extrinsics_to(p2)
            sub = a.name + ("" if a.emitter != "both" else ("_emitter-on" if em else "_emitter-off"))
            d = os.path.join(a.out, sub)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "left.png"), np.asanyarray(ir1.get_data()))
            cv2.imwrite(os.path.join(d, "right.png"), np.asanyarray(ir2.get_data()))
            cv2.imwrite(os.path.join(d, "depth_rs.png"), np.asanyarray(dep.get_data()))
            if a.color:
                c = fs.get_color_frame()
                if c:
                    cv2.imwrite(os.path.join(d, "color.png"), np.asanyarray(c.get_data()))
            meta = {"device": name, "serial": dev.get_info(rs.camera_info.serial_number), "usb": usb,
                    "width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                    "cx": intr.ppx, "cy": intr.ppy, "baseline_m": abs(float(ext.translation[0])),
                    "depth_scale": ds.get_depth_scale(), "emitter": bool(em)}
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            out.append(sub)
            print(f"{sub}: {intr.width} x {intr.height}, fx {intr.fx:.1f} px, baseline "
                  f"{meta['baseline_m'] * 1000:.1f} mm, emitter {'on' if em else 'off'}")
        finally:
            pipe.stop()
    print(json.dumps({"pairs": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

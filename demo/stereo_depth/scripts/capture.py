#!/usr/bin/env python3
"""capture.py — stereo pairs from a RealSense camera on the KV260
(doc/plans/STEREO_PLAN.md).

Uploads src/board/realsense_capture.py, runs it on the board (pyrealsense2:
the D4xx's two rectified infrared cameras, the camera's own depth and the
calibration; the emitter on and off by default) and downloads every pair to
assets/pairs/<name>[_emitter-on|off]/ {left,right}.png, depth_rs.png,
meta.json — the user pairs prepare.py picks up.  The right infrared camera
needs firmware 5.17.3.10 on a USB 2 link (or USB 3).  Holds the per-board
lock (the FPGA is not used).

usage: inference-scheduler/.venv/bin/python demo/stereo_depth/scripts/capture.py
           [--config stereo_depth_config.json] [--name realsense] [--emitter both|on|off]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
REPO = DEMO.parent.parent
sys.path.insert(0, str(REPO / "inference-scheduler"))
from src.remote import RemoteSession, hold_board_lock  # noqa: E402

PAIRS = DEMO / "assets" / "pairs"
BOARD_SCRIPT = DEMO / "src" / "board" / "realsense_capture.py"
FILES = ("left.png", "right.png", "depth_rs.png", "color.png", "meta.json")


def capture(cfg: dict, name: str = "realsense", emitter: str = "both", color: bool = False) -> list:
    """Capture on the board; returns the local pair directories."""
    hold_board_lock(cfg)
    session = RemoteSession(cfg["ssh"])
    session.connect()
    work = f"{cfg['remote']['work_dir']}/capture"
    try:
        session.exec(f"rm -rf {shlex.quote(work)} && mkdir -p {shlex.quote(work)}", timeout=60)
        sftp = session._client.open_sftp()                                        # noqa: SLF001
        try:
            sftp.put(str(BOARD_SCRIPT), f"{work}/realsense_capture.py")
            t0 = time.monotonic()
            out, err, rc = session.exec(f"python3 {work}/realsense_capture.py --out {work}/pairs "
                                        f"--name {shlex.quote(name)} --emitter {emitter}"
                                        f"{' --color' if color else ''}", timeout=300)
            if rc:
                raise SystemExit(f"capture failed (rc {rc}):\n{(out + err)[-2000:]}")
            print(out.rstrip())
            pairs = json.loads(out.strip().splitlines()[-1])["pairs"]
            dirs = []
            for p in pairs:
                dst = PAIRS / p
                dst.mkdir(parents=True, exist_ok=True)
                for f in FILES:
                    try:
                        sftp.get(f"{work}/pairs/{p}/{f}", str(dst / f))
                    except OSError:
                        if f != "color.png":
                            raise
                dirs.append(dst)
            print(f"  {len(dirs)} pair(s) -> {PAIRS} ({time.monotonic() - t0:.0f} s)")
            return dirs
        finally:
            sftp.close()
    finally:
        session.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--config", default=str(DEMO / "stereo_depth_config.json"))
    ap.add_argument("--name", default="realsense")
    ap.add_argument("--emitter", choices=("on", "off", "both"), default="both")
    ap.add_argument("--color", action="store_true", help="also the colour image (bandwidth permitting)")
    a = ap.parse_args(argv)
    capture(json.loads(Path(a.config).read_text()), a.name, a.emitter, a.color)
    return 0


if __name__ == "__main__":
    sys.exit(main())

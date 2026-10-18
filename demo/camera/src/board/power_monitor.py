# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
power_monitor.py — board-side SOM power sampling for the KV260 camera demo.

Reads whole-board (System-on-Module) power draw in watts on a background
thread so the camera loop can overlay it without ever blocking on a sensor
read.

Power sources, probed in priority order ONCE at start():

  1. XRT  — pyxrt device "electrical" info.  The XRT power API
            (xrt::info::device::electrical) is an Alveo datacenter-card
            feature; on the KV260's embedded zocl stack it is normally NOT
            populated, so this is attempted first (the demo is asked to use
            XRT if it can) but usually falls through to source 2.
  2. xlnx_platformstats — `xmutil xlnx_platformstats` exposes the SOM INA260
            sensor as "SOM total power : N mW".  This is the reliable KV260
            source and what the camera loop ends up using in practice.
  3. hwmon — /sys/class/hwmon/*/power1_input (microwatts).  Last resort; the
            INA260 hwmon node name varies across board images, so this is
            best-effort only.

If none work, watts() returns None and the overlay shows "power n/a".
The whole module runs ON the KV260 (imported by camera_loop.py).
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Source probes — each returns whole-board watts (float) or None.
# ──────────────────────────────────────────────────────────────────────────────

def _read_xrt() -> Optional[float]:
    """pyxrt electrical info.  Returns None unless XRT actually reports a
    non-zero power_consumption_watts (true on Alveo, rarely on embedded)."""
    try:
        import pyxrt   # noqa: F401
    except ImportError:
        return None
    try:
        dev = pyxrt.device(0)
        raw = dev.get_info(pyxrt.xrt_info_device.electrical)
    except Exception:                       # noqa: BLE001 — pyxrt API varies
        return None
    try:
        info = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(info, dict):
        return None
    val = info.get("power_consumption_watts")
    try:
        w = float(val)
    except (TypeError, ValueError):
        return None
    return w if w > 0.0 else None


def _read_xmutil() -> Optional[float]:
    """`xmutil xlnx_platformstats` — the KV260 SOM INA260 total-power line."""
    try:
        out = subprocess.run(["xmutil", "xlnx_platformstats"],
                             capture_output=True, text=True, timeout=8.0)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"SOM total power\s*:\s*([\d.]+)\s*mW", out.stdout)
    return (float(m.group(1)) / 1000.0) if m else None


_HWMON_FILES: list = []


def _read_hwmon() -> Optional[float]:
    """First /sys/class/hwmon/*/power1_input reading (microwatts → watts)."""
    global _HWMON_FILES
    if not _HWMON_FILES:
        _HWMON_FILES = [Path(f) for f in
                        sorted(glob.glob("/sys/class/hwmon/hwmon*/power1_input"))]
    for f in _HWMON_FILES:
        try:
            return int(f.read_text().strip()) / 1.0e6
        except (OSError, ValueError):
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Background monitor
# ──────────────────────────────────────────────────────────────────────────────

class PowerMonitor:
    """Samples SOM power on a daemon thread; watts() returns the latest value."""

    _SOURCES: list = [
        ("xrt",                _read_xrt),
        ("xlnx_platformstats", _read_xmutil),
        ("hwmon",              _read_hwmon),
    ]

    def __init__(self, poll_interval: float = 2.0) -> None:
        self.poll_interval = max(0.5, float(poll_interval))
        self._watts:  Optional[float] = None
        self._source: Optional[str]   = None
        self._reader: Optional[Callable[[], Optional[float]]] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _select_source(self) -> Tuple[Optional[str],
                                      Optional[Callable], Optional[float]]:
        for name, fn in self._SOURCES:
            try:
                w = fn()
            except Exception:               # noqa: BLE001
                w = None
            if w is not None:
                return name, fn, w
        return None, None, None

    def start(self) -> None:
        name, fn, w = self._select_source()
        self._source = name
        self._reader = fn
        with self._lock:
            self._watts = w
        if name is None:
            _log("power_monitor: no power source available "
                 "(tried XRT, xlnx_platformstats, hwmon) — overlay shows n/a")
            return
        _log(f"power_monitor: source='{name}'  initial={w:.2f} W")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                w = self._reader()          # type: ignore[misc]
            except Exception:               # noqa: BLE001
                w = None
            if w is not None:
                with self._lock:
                    self._watts = w

    def watts(self) -> Optional[float]:
        with self._lock:
            return self._watts

    def source(self) -> Optional[str]:
        return self._source

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

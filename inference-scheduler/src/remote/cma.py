# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""Clean up the board's CMA area before a large pool is allocated.

Every FPGA library takes its DMA pool as one contiguous CMA buffer (an XRT
buffer object): SmolLM2-360M 795 MB, SmolVLM 550 MB, SmolLM2-135M 311 MB
out of the cma= area (1000M then).  The allocation can fail with enough CmaFree when the page
cache of uploads and builds sits in the CMA area or has fragmented it — seen
2026-09-30 (776 MB with 834 MB free) and three times on 2026-10-13 (795 MB
with 990 MB free, after builds and gates).  Dropping the clean page cache
and compacting memory restores the contiguous space; a reboot is not needed.

    .venv/bin/python -m src.remote.cma --config remote_config.json             # clean + report
    .venv/bin/python -m src.remote.cma --config remote_config.json --check     # report only
    .venv/bin/python -m src.remote.cma --config remote_config.json --min-mb 800
                                       # exit 1 unless 800 MB are free after the clean-up

The report: CmaTotal / CmaFree before and after, the largest free CMA chunk
where debugfs shows it (/sys/kernel/debug/cma/*/maxchunk), and the processes
holding the FPGA's render device (their buffers stay in CMA until they exit:
a leftover benchmark or server is the other common cause).  The board lock
is held (src/remote/lock.py) unless --no-lock.  From Python:
clean_cma(session) / cma_state(session) on a connected RemoteSession.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

CLEAN_CMD = "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"

# key=value lines (sizes in kB), then "holder=<pid> <cmdline>" per process
STATE_SCRIPT = r"""
awk '/^CmaTotal/{print "total_kb=" $2} /^CmaFree/{print "free_kb=" $2}' /proc/meminfo
for d in /sys/kernel/debug/cma/*/; do
  [ -r "$d/maxchunk" ] || continue
  echo "maxchunk_kb=$(( $(cat $d/maxchunk) * $(getconf PAGESIZE) / 1024 ))"
done 2>/dev/null
for p in /proc/[0-9]*; do
  ls -l $p/fd 2>/dev/null | grep -q 'renderD' && \
    echo "holder=${p#/proc/} $(tr '\0' ' ' < $p/cmdline 2>/dev/null | cut -c1-100)"
done
true
"""


def parse_state(text: str) -> Dict:
    """{total_kb, free_kb, maxchunk_kb (None: no debugfs), holders: [str]}."""
    st: Dict = {"total_kb": 0, "free_kb": 0, "maxchunk_kb": None, "holders": []}
    for line in text.splitlines():
        key, _, val = line.strip().partition("=")
        if key in ("total_kb", "free_kb"):
            st[key] = int(val)
        elif key == "maxchunk_kb":
            st["maxchunk_kb"] = max(st["maxchunk_kb"] or 0, int(val))
        elif key == "holder":
            st["holders"].append(val)
    return st


def _sudo_wrap(cmd: str, sudo: str) -> str:
    return f"{sudo}sh -c {_quote(cmd)}" if sudo else cmd


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def cma_state(session, sudo: str = "") -> Dict:
    out, _err, _rc = session.exec(_sudo_wrap(STATE_SCRIPT, sudo), timeout=60)
    return parse_state(out)


def clean_cma(session, sudo: str = "") -> Dict:
    """Drop the page cache and compact memory; {before, after} states."""
    before = cma_state(session, sudo)
    session.exec(_sudo_wrap(CLEAN_CMD, sudo), timeout=120)
    return {"before": before, "after": cma_state(session, sudo)}


def describe(st: Dict) -> str:
    mb = lambda kb: f"{kb / 1024:.0f} MB"                       # noqa: E731
    s = f"CmaFree {mb(st['free_kb'])} of {mb(st['total_kb'])}"
    if st["maxchunk_kb"] is not None:
        s += f", largest free chunk {mb(st['maxchunk_kb'])}"
    return s


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Clean up the KV260's CMA area (drop the page cache, "
                                             "compact memory) and report it.")
    ap.add_argument("--config", required=True, help="a board config JSON (its ssh block, board_lock)")
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--min-mb", type=int, default=0,
                    help="exit 1 unless this many MB are free afterwards (the largest free chunk "
                         "where debugfs shows it, else CmaFree)")
    ap.add_argument("--no-lock", action="store_true", help="do not take the board lock")
    args = ap.parse_args(argv)

    from .config import load_config
    from .lock import hold_board_lock
    from .session import RemoteSession
    cfg = load_config(args.config)
    if not args.no_lock:
        hold_board_lock(cfg)
    sudo = "sudo " if cfg["ssh"].get("user", "root") != "root" else ""
    session = RemoteSession(cfg["ssh"])
    session.connect()
    try:
        if args.check:
            after = cma_state(session, sudo)
            print(f"cma: {describe(after)}")
        else:
            r = clean_cma(session, sudo)
            after = r["after"]
            print(f"cma: before {describe(r['before'])}")
            print(f"cma: after  {describe(after)}")
        for h in after["holders"]:
            print(f"cma: held by pid {h}")
    finally:
        session.close()
    free_kb = after["maxchunk_kb"] if after["maxchunk_kb"] is not None else after["free_kb"]
    if args.min_mb and free_kb < args.min_mb * 1024:
        print(f"cma: only {free_kb / 1024:.0f} MB free, {args.min_mb} MB needed"
              + (" (stop the processes above)" if after["holders"] else ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("CLEAN_CMD", "STATE_SCRIPT", "parse_state", "cma_state", "clean_cma", "describe", "main")

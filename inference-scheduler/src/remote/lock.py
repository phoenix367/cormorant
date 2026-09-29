"""
Board lock — one flock() per board, shared by every tool that drives it.

Two jobs on one KV260 at the same time corrupt each other (the kernels, the
CMA pool and the DDR buffers are shared), so every board tool takes an
exclusive flock on a per-board lock file for its whole session:

  * the path: the config's ``board_lock`` if set, else
    ``/tmp/kv260-board-<ssh host>.lock`` — the same file for every tool and
    checkout that drives that board (/tmp explicitly, not $TMPDIR, so that
    processes with different TMPDIRs still agree); ``board_lock: false``
    (or "none") disables it;
  * re-entrant across processes: while a process holds the lock it sets
    ``KV260_BOARD_LOCK_HELD=<path>`` in its environment, and a tool started
    underneath it (perf_calibrate.py run --stop-server -> deploy.py,
    deploy.py -> generate_project.py …) sees the variable and does not wait
    for a lock its own parent holds;
  * a second job prints "waiting for board lock …" and blocks until the
    first one exits.

Do not wrap these tools in a shell ``flock`` on the same file: the shell's
lock is not marked in the environment, so the tool inside would wait for it
forever.  To run any other command under the lock (an ssh maintenance
command, a script that does not lock) use this module, which also marks the
environment for the tools the command starts:

    cd inference-scheduler
    .venv/bin/python -m src.remote.locked --config CFG.json -- CMD ARG ...
    .venv/bin/python -m src.remote.locked --host 192.168.100.8 --print   # the path
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, List, Optional

from .colors import _dim

LOCK_ENV = "KV260_BOARD_LOCK_HELD"
_HELD: List[IO] = []          # hold_board_lock(): files kept open until the process exits


def default_lock_path(host: Optional[str]) -> str:
    """``/tmp/kv260-board-<host>.lock`` (the host sanitised for a file name)."""
    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in str(host or "board"))
    return f"/tmp/kv260-board-{safe}.lock"


def lock_path(cfg: dict, override: Optional[str] = None) -> Optional[str]:
    """The lock file of the board ``cfg`` drives: ``override`` (a --board-lock
    option), else ``cfg["board_lock"]``, else the per-board default from
    ``cfg["ssh"]["host"]``; None when the config disables locking
    (``board_lock: false`` / "none")."""
    v = override if override else cfg.get("board_lock")
    if v is False or (isinstance(v, str) and v.strip().lower() in ("none", "off", "false")):
        return None
    if v:
        return str(Path(str(v)).expanduser())
    return default_lock_path((cfg.get("ssh") or {}).get("host"))


def _held_by_parent(p: Path) -> bool:
    return os.environ.get(LOCK_ENV) == str(p)


def _acquire(p: Path) -> IO:
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a")
    t0 = time.monotonic()
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(_dim(f"waiting for board lock {p} (another job is using the board) ..."), flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
        print(_dim(f"board lock acquired after {time.monotonic() - t0:.0f} s"), flush=True)
    return f


@contextmanager
def board_lock(path: Optional[str]):
    """Hold the lock at ``path`` for the ``with`` block (no-op for None, or
    when a parent process already holds it)."""
    if not path:
        yield
        return
    p = Path(path).expanduser()
    if _held_by_parent(p):
        yield
        return
    f = _acquire(p)
    prev = os.environ.get(LOCK_ENV)
    os.environ[LOCK_ENV] = str(p)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(LOCK_ENV, None)
        else:
            os.environ[LOCK_ENV] = prev
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def hold_board_lock(cfg: dict, override: Optional[str] = None) -> Optional[str]:
    """Take the board lock of ``cfg`` for the rest of this process — the OS
    releases it when the process exits.  Returns the path (None: disabled)."""
    path = lock_path(cfg, override)
    if not path:
        return None
    p = Path(path)
    if not _held_by_parent(p):
        _HELD.append(_acquire(p))
        os.environ[LOCK_ENV] = str(p)
    return str(p)


def main(argv=None) -> int:
    import argparse
    import json
    import subprocess
    ap = argparse.ArgumentParser(description="Run a command under the KV260 board lock, or print "
                                             "the lock path.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="a board config JSON (its ssh.host / board_lock)")
    g.add_argument("--host", help="the board's ssh host")
    g.add_argument("--path", help="an explicit lock file")
    ap.add_argument("--print", action="store_true", help="print the lock path and exit")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- CMD ARG ...")
    a = ap.parse_args(argv)
    if a.config:
        with open(a.config) as f:
            cfg = json.load(f)
    else:
        cfg = {"board_lock": a.path, "ssh": {"host": a.host}}
    path = lock_path(cfg)
    if a.print:
        print(path or "")
        return 0
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command (-- CMD ARG ...)")
    with board_lock(path):
        return subprocess.call(cmd)


__all__ = ("LOCK_ENV", "default_lock_path", "lock_path", "board_lock", "hold_board_lock")


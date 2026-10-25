#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
deploy.py — install and start Miranda's server on the KV260 (run on the host
with inference-scheduler/.venv/bin/python).

  check   the KV260 chat server answers on the board and serves a chat, a
          text-to-speech and a speech-to-text model (GET /v1/models: their
          kv260.type) — Miranda's FPGA work goes through it
  upload  miranda_server.py, miranda_proto.py, vad.py, wake.py, weather.py -> <dir>/miranda
          (<dir>: the chat server's install directory, remote.dir of
          demo/chat/chat_config.json)
  unit    /etc/systemd/system/kv260-miranda.service (journalctl -u kv260-miranda),
          not enabled: Miranda never starts at boot, only by hand —
          `systemctl start kv260-miranda` on the board or `deploy.py --start`;
          ordered after the chat server (`systemctl start kv260-chat
          kv260-miranda` starts both, Miranda once the chat server's models are
          loaded) but it does not start it; restarted when it fails
  start   systemctl start kv260-miranda on --port (TCP control + UDP RTP,
          8770), talking to the chat server on the board's loopback (its API
          key file when it has one); then waits until the port answers

The board, ssh and the chat server's port and API key come from
demo/chat/chat_config.json (deploy.py there installs the chat server).
Miranda's server does not touch the FPGA itself, so it takes no board lock
and may keep running while the chat server is stopped for a board job (its
requests then fail until the chat server is back).  --no-start installs
everything but leaves Miranda stopped (the chat server need not run then);
--uninstall stops it and removes the unit (the files stay).

--fetch-segments [DIR] downloads the segments the server recorded
(miranda_server.py --save-segments: a WAV file per segment and
segments.jsonl) into DIR (default demo/miranda/build/segments), skipping the
files already there with the same size; it reads the running server's
--save-segments directory, else <dir>/miranda/segments.  --status says
whether the server records.

usage:  deploy.py [--config ../chat/chat_config.json] [--port 8770] [--no-start]
                  [--chat-model smollm2-360m-instruct] [-- SERVER ARGS...]
        deploy.py --start | --stop | --status | --uninstall | --fetch-segments [DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import socket
import stat
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.join(os.path.dirname(HERE), "chat")
sys.path.insert(0, CHAT)

import deploy as chat_deploy                                           # noqa: E402  (demo/chat/deploy.py)
from src.remote import RemoteSession, _green, _red, _yellow            # noqa: E402

UNIT = "kv260-miranda"
FILES = ("miranda_server.py", "miranda_proto.py", "vad.py", "wake.py", "weather.py")
SEGMENTS = os.path.join(HERE, "build", "segments")      # --fetch-segments' default


def unit_text(cfg: dict, argv) -> str:
    """kv260-miranda.service: after the chat server, never enabled by deploy.py."""
    d = cfg["remote"]["dir"]
    return "\n".join([
        "# Installed by demo/miranda/deploy.py (demo/miranda/README.md).  Not enabled: Miranda starts",
        f"# only by hand (systemctl start {UNIT}, or deploy.py --start); after the chat server",
        f"# when both start (systemctl start {chat_deploy.UNIT} {UNIT}), but it does not start it.",
        "[Unit]",
        "Description=Miranda voice assistant (KV260)",
        f"After=network.target {chat_deploy.UNIT}.service",
        "StartLimitIntervalSec=300",
        "StartLimitBurst=5",
        "",
        "[Service]",
        f"WorkingDirectory={d}/miranda",
        f"ExecStart={chat_deploy.exec_line(argv)}",
        "KillSignal=SIGTERM",
        "TimeoutStopSec=30",
        "Restart=on-failure",
        "RestartSec=5",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        ""])


def unit_port(session, default: int) -> int:
    """The --port of the installed unit's server (default without one)."""
    out, _, _ = session.exec(f"grep -o -- '--port [0-9]*' /etc/systemd/system/{UNIT}.service 2>/dev/null",
                             timeout=15)
    return int(out.split()[1]) if len(out.split()) >= 2 else default


def wait_port(session, su: str, host: str, port: int) -> bool:
    """Until host:port answers (20 s); prints the unit's log when it does not."""
    for _ in range(40):
        if port_open(host, port):
            return True
        time.sleep(0.5)
    out, _, _ = session.exec(f"{su}journalctl -u {UNIT} -n 30 --no-pager -o cat", timeout=30)
    print(_red(f"  start    -> FAIL  {host}:{port} does not answer\n{out}"))
    return False


def start(session, cfg: dict, su: str, port: int, t0: float) -> int:
    """systemctl start kv260-miranda, then wait for its port."""
    host = cfg["ssh"]["host"]
    out, err, rc = session.exec(f"{su}systemctl start {UNIT}", timeout=cfg["health_timeout"] + 240)
    if rc != 0:
        print(_red(f"  start    -> FAIL  {(out + err).strip()}"))
        return 1
    if not wait_port(session, su, host, port):
        return 1
    print(f"  start    -> {_green('OK')}  systemd: {UNIT} on {host}:{port} ({time.monotonic() - t0:.1f} s)")
    out, _, _ = session.exec(f"systemctl is-active {chat_deploy.UNIT} 2>/dev/null", timeout=15)
    if out.strip() != "active":
        print(_yellow(f"  note: the chat server is {out.strip() or 'inactive'}: Miranda's requests fail until "
                      f"it runs (demo/chat/deploy.py --start)"))
    print(f"\n  python3 demo/miranda/miranda_client.py --server {host}\n")
    return 0


def kinds(cfg: dict):
    """{model id: type} the chat server serves (None: it does not answer)."""
    url = f"http://{cfg['ssh']['host']}:{cfg['server']['port']}/v1/models"
    req = urllib.request.Request(url)
    if cfg["server"].get("api_key"):
        req.add_header("Authorization", f"Bearer {cfg['server']['api_key']}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return {m["id"]: (m.get("kv260") or {}).get("type", "chat") for m in json.load(r)["data"]}
    except (OSError, ValueError):
        return None


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def recording(session, d: str):
    """The running server's --save-segments directory (None: it does not record;
    False: it does not run)."""
    out, _, _ = session.exec(f"pid=$(systemctl show -p MainPID --value {UNIT} 2>/dev/null); "
                             "[ -n \"$pid\" ] && [ \"$pid\" != 0 ] && tr '\\0' '\\n' < /proc/$pid/cmdline",
                             timeout=15)
    argv = out.splitlines()
    if not argv:
        return False
    for i, a in enumerate(argv):
        if a == "--save-segments" and i + 1 < len(argv):
            return posixpath.join(f"{d}/miranda", argv[i + 1])
        if a.startswith("--save-segments="):
            return posixpath.join(f"{d}/miranda", a.split("=", 1)[1])
    return None


def fetch_segments(session, src: str, dest: str) -> int:
    """Download src's files (WAVs + segments.jsonl) that dest lacks or holds at another size."""
    sftp = session._client.open_sftp()                     # noqa: SLF001
    try:
        try:
            files = [a for a in sftp.listdir_attr(src) if stat.S_ISREG(a.st_mode or 0)]
        except FileNotFoundError:
            print(_red(f"  fetch    -> FAIL  no {src} on the board (record with: deploy.py -- --save-segments {src})"))
            return 1
        try:
            os.makedirs(dest, exist_ok=True)
        except OSError as e:
            print(_red(f"  fetch    -> FAIL  {e}"))
            return 1
        new = [a for a in files if not (os.path.isfile(p := os.path.join(dest, a.filename))
                                        and os.path.getsize(p) == a.st_size)]
        for a in sorted(new, key=lambda a: a.filename):
            sftp.get(posixpath.join(src, a.filename), os.path.join(dest, a.filename))
    finally:
        sftp.close()
    wavs = sum(a.filename.endswith(".wav") for a in files)
    print(f"  fetch    -> {_green('OK')}  {src}: {wavs} segments, {len(new)} file{'s' * (len(new) != 1)} new or grown "
          f"({sum(a.st_size for a in new) / 1e6:.1f} MB) -> {dest}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="the chat server's config (default demo/chat/chat_config.json)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--chat-model", default="smollm2-360m-instruct",
                    help="the chat model (smollm2-135m-instruct: faster, weaker answers)")
    ap.add_argument("--start", action="store_true", help="start the installed unit (no upload)")
    ap.add_argument("--no-start", action="store_true", help="upload and install the unit, leave Miranda stopped")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--uninstall", action="store_true",
                    help="stop Miranda and remove its unit (the files stay)")
    ap.add_argument("--fetch-segments", nargs="?", const=SEGMENTS, default=None, metavar="DIR",
                    help="download the segments the server recorded (--save-segments) into DIR "
                         "(default demo/miranda/build/segments)")
    ap.add_argument("server_args", nargs="*", help="more miranda_server.py arguments (after --)")
    args = ap.parse_args(argv)
    cfg = chat_deploy.load_config(args.config)
    host = cfg["ssh"]["host"]
    d = cfg["remote"]["dir"]
    su = chat_deploy.sudo(cfg)
    session = RemoteSession(cfg["ssh"])
    session.connect()
    try:
        if args.fetch_segments is not None:
            src = recording(session, d) or f"{d}/miranda/segments"
            return fetch_segments(session, src, args.fetch_segments)
        if args.start or args.status:
            args.port = unit_port(session, args.port)
        if args.status or args.stop or args.uninstall:
            if args.stop or args.uninstall:
                session.exec(f"{su}systemctl stop {UNIT} 2>/dev/null; {su}systemctl reset-failed {UNIT} 2>/dev/null",
                             timeout=60)
            if args.uninstall:
                chat_deploy.uninstall_unit(session, cfg, UNIT)
            out, _, _ = session.exec(f"systemctl is-active {UNIT} 2>/dev/null || true", timeout=15)
            print(f"  {UNIT}: {out.strip() or 'inactive'}")
            if chat_deploy.unit_installed(session, UNIT):
                out, _, _ = session.exec(f"systemctl is-enabled {UNIT} 2>/dev/null", timeout=15)
                state = out.strip() or "?"
                print(f"  unit: /etc/systemd/system/{UNIT}.service, {state}"
                      + (" (not started at boot)" if state == "disabled" else ""))
            else:
                print("  unit: not installed")
            print(f"  {host}:{args.port}: {'answers' if port_open(host, args.port) else 'no answer'}")
            rec = recording(session, d)
            if rec is not False:
                print(f"  recording: {rec or 'off'}")
            return 0
        t0 = time.monotonic()
        if args.start:
            if not chat_deploy.unit_installed(session, UNIT):
                print(_red(f"  start    -> FAIL  no {UNIT} unit on the board: deploy first (deploy.py)"))
                return 1
            return start(session, cfg, su, args.port, t0)
        k = kinds(cfg)
        if k is None and args.no_start:
            print(_yellow(f"  check    -> SKIPPED  the chat server at {host}:{cfg['server']['port']} does not "
                          f"answer: its models are not checked"))
        elif k is None:
            print(_red(f"  check    -> FAIL  the chat server at {host}:{cfg['server']['port']} does not answer "
                       f"(demo/chat/deploy.py --start; --no-start installs Miranda without it)"))
            return 1
        else:
            missing = [t for t in ("chat", "speech", "transcription") if t not in k.values()]
            if missing:
                print(_red(f"  check    -> FAIL  the chat server serves no {' / '.join(missing)} model "
                           f"(server.backends: smollm2-360m, piper, moonshine)"))
                return 1
            if args.chat_model not in k or k[args.chat_model] != "chat":
                print(_red(f"  check    -> FAIL  --chat-model {args.chat_model} is not a chat model of the server "
                           f"({', '.join(m for m, t in k.items() if t == 'chat')})"))
                return 1
            print(f"  check    -> {_green('OK')}  {', '.join(f'{m} ({t})' for m, t in k.items())}")
        session.exec_checked(f"mkdir -p {d}/miranda", timeout=15)
        sftp = session._client.open_sftp()                     # noqa: SLF001
        try:
            for f in FILES:
                sftp.put(os.path.join(HERE, f), f"{d}/miranda/{f}")
        finally:
            sftp.close()
        print(f"  upload   -> {_green('OK')}  {len(FILES)} files -> {d}/miranda")
        session.exec(f"{su}systemctl stop {UNIT} 2>/dev/null; {su}systemctl reset-failed {UNIT} 2>/dev/null",
                     timeout=60)
        argv = ["/usr/bin/python3", "-u", f"{d}/miranda/miranda_server.py", "--port", str(args.port),
                "--chat-url", f"http://127.0.0.1:{cfg['server']['port']}/v1", "--chat-model", args.chat_model]
        if cfg["server"].get("api_key"):
            argv += ["--api-key-file", f"{d}/api_key"]
        argv += args.server_args
        if not chat_deploy.install_unit(session, cfg, UNIT, unit_text(cfg, argv)):
            return 1
        if args.no_start:
            print(f"\n  Miranda installed, not started: systemctl start {UNIT} (on the board), "
                  f"or demo/miranda/deploy.py --start\n")
            return 0
        return start(session, cfg, su, args.port, t0)
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())

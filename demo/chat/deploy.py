#!/usr/bin/env python3
"""
deploy.py — install and start the KV260 chat server on the board (run on the
host with inference-scheduler/.venv/bin/python).

  project  -> demo/bert_squad/scripts/generate_project.py (only when the
              generated project is missing or predates libbert_squad.so;
              --regenerate forces it)
  weights  -> remote weights_dir: only files whose size or SHA-1 changed
  build    -> project sources to <dir>/project, cmake + make bert_squad,
              <dir>/lib/libbert_squad.so (skipped when the sources and the
              cmake options are unchanged since the last build)
  server   -> kv260_chat_server.py, chat_backend.py, bert_squad_backend.py,
              squad_text.py, chat.py, vocab.txt, and the smollm2 side
              (smollm2_backend.py, smollm2_tokenizer.py, chatml.py, sampler.py,
              src/sampler.{c,h}; tokenizer.json -> <dir>/smollm2/) to <dir>
  sampler  -> cc -O2 -shared -fPIC src/sampler.c -> <dir>/lib/libsampler.so
              (skipped when unchanged; without it the server samples in Python)
  start    -> transient systemd unit 'kv260-chat' (systemd-run: survives the
              ssh session, not started at boot, logs in journalctl -u
              kv260-chat) or, with server.launcher = "nohup", a detached
              process; then waits for GET /health from the host

The smollm2 backend (server.backends containing "smollm2") needs
libsmollm2.so on the board at smollm2.lib (default <dir>/lib/libsmollm2.so;
built by the decoder project of CHAT_PLAN phase 3, not by this script);
preflight reports whether it is there.

The board lock (flock on board_lock) is held while deploying.  The running
server owns the FPGA: other board jobs must wait until `deploy.py --stop`.
With --hold, deploy.py keeps the lock after starting, follows the server log
and stops the server (releasing the lock) on Ctrl-C / SIGTERM.

usage:  deploy.py [--config chat_config.json] [--regenerate] [--rebuild]
                  [--hold] [--no-lock] [--board-lock FILE] [--check-only] [-v]
        deploy.py --stop | --status
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

CHAT = Path(__file__).resolve().parent
DEMO = CHAT.parent
BERT_SCRIPTS = DEMO / "bert_squad" / "scripts"
if str(BERT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BERT_SCRIPTS))

import _common as bert_common                                   # noqa: E402  (+ scheduler path)
from deploy_and_run import board_lock, sync_weights             # noqa: E402
from src.remote import (RemoteSession, _bold, _dim, _green, _red,  # noqa: E402
                        _yellow, check_prerequisites)

CONFIG = CHAT / "chat_config.json"
EXAMPLE = CHAT / "chat_config.json.example"
UNIT = "kv260-chat"
SERVER_FILES = [CHAT / "kv260_chat_server.py", CHAT / "chat_backend.py",
                CHAT / "bert_squad_backend.py", CHAT / "chat.py", BERT_SCRIPTS / "squad_text.py",
                CHAT / "smollm2_backend.py", CHAT / "smollm2_tokenizer.py", CHAT / "chatml.py",
                CHAT / "sampler.py"]
SAMPLER_SRC = [CHAT / "src" / "sampler.c", CHAT / "src" / "sampler.h"]


def step(name: str, ok: bool, t0: float, msg: str = "") -> bool:
    tag = _green("OK") if ok else _red("FAIL")
    print(f"  {name:<8} -> {tag:<13} {time.monotonic() - t0:6.1f}s" + (f"  {msg}" if msg else ""),
          flush=True)
    return ok


def tail(text: str, n: int = 25) -> None:
    lines = (text or "").rstrip().splitlines()
    for ln in (["..."] + lines[-n:]) if len(lines) > n else lines:
        print(_dim(f"      {ln}"))


# ── configuration ────────────────────────────────────────────────────────────

def chat_path(value: str) -> Path:
    p = Path(os.path.expanduser(value))
    return p if p.is_absolute() else CHAT / p


def load_config(path: Optional[str]) -> dict:
    p = Path(path) if path else CONFIG
    if not p.exists():
        print(f"\nERROR: {p} not found.\n\n  cp '{EXAMPLE}' '{p}'\n  $EDITOR '{p}'\n\n"
              f"The BERT-SQuAD demo config it points at (bert_squad_config) supplies ssh, the\n"
              f"UIO devices, the weights directory and the board lock; see {CHAT / 'README.md'}.\n",
              file=sys.stderr)
        sys.exit(2)
    cfg = json.loads(p.read_text())
    bpath = chat_path(cfg.get("bert_squad_config", "../bert_squad/bert_squad_config.json"))
    cfg["_bert_path"] = str(bpath)
    bcfg = bert_common.load_config(str(bpath))
    cfg["_bert"] = bcfg
    cfg["ssh"] = cfg.get("ssh") or bcfg["ssh"]
    rem = cfg.setdefault("remote", {})
    rem.setdefault("dir", "/root/kv260_chat")
    rem["weights_dir"] = rem.get("weights_dir") or bcfg["remote"]["weights_dir"]
    srv = cfg.setdefault("server", {})
    for k, v in (("host", "0.0.0.0"), ("port", 8000), ("api_key", None), ("backends", ["bert-squad"]),
                 ("max_windows", 8), ("doc_stride", 128), ("queue_timeout", 120), ("max_queue", 16),
                 ("launcher", "systemd"), ("env", {}), ("resident", "auto")):
        srv.setdefault(k, v)
    llm = cfg.setdefault("smollm2", {})
    for k, v in (("lib", None), ("weights_dir", None),
                 ("tokenizer", "assets/smollm2-135m-instruct/tokenizer.json"), ("context", 1024),
                 ("reserve", 256), ("temperature", 0.2), ("top_p", 0.9), ("top_k", 50),
                 ("repetition_penalty", 1.1), ("cma_mb", 360), ("dry_multiplier", 0.8),
                 ("dry_base", 1.75), ("dry_allowed_length", 2), ("dry_penalty_last_n", -1),
                 ("dry_sequence_breakers", [":", "\"", "*"]), ("loop_guard", True)):
        if llm.get(k) is None:
            llm[k] = v
    if not llm["lib"]:
        llm["lib"] = f"{rem['dir']}/lib/libsmollm2.so"
    cfg.setdefault("build", {}).setdefault("jobs", 4)
    cfg["build"].setdefault("timeout", 1800)
    cfg["board_lock"] = cfg.get("board_lock") or bcfg.get("board_lock")
    cfg.setdefault("health_timeout", 180)
    return cfg


def uses_llm(cfg: dict) -> bool:
    return any(b in ("smollm2", "smollm2-135m-instruct") for b in cfg["server"]["backends"])


def sudo(cfg: dict) -> str:
    return "sudo -n " if cfg["_bert"].get("run", {}).get("use_sudo", True) else ""


# ── host side: the generated project ─────────────────────────────────────────

def ensure_project(cfg: dict, regenerate: bool) -> dict:
    summary = bert_common.PROJECT_SUMMARY
    fresh = False
    if summary.exists():
        s = json.loads(summary.read_text())
        proj = Path(s["project_dir"])
        fresh = ("bert_squad" in s.get("targets", []) and (proj / "test" / "bert_api.c").exists()
                 and (proj / "CMakeLists.txt").exists())
    if regenerate or not fresh:
        print(_bold("\nGenerating the BERT project") + _dim(" (demo/bert_squad/scripts/generate_project.py)"))
        import generate_project
        rc = generate_project.main(["--config", cfg["_bert_path"]])
        if rc != 0:
            sys.exit(rc)
    s = json.loads(summary.read_text())
    if s.get("missing"):
        print(_red(f"driver files missing in the project: {s['missing']}"))
        sys.exit(1)
    return s


def project_files(proj: Path) -> List[Path]:
    return [p for p in sorted(proj.rglob("*"))
            if p.is_file() and p.relative_to(proj).parts[0] not in ("weights", "build")]


def build_defs(cfg: dict, active: List[str]) -> str:
    uio = cfg["_bert"]["remote"].get("uio_devices", {})
    defs = [f"-DINFERENCE_{k.upper()}_INSTANCE=\\\"{uio[k]}\\\"" for k in active if k in uio]
    return (f"-DCMAKE_BUILD_TYPE=Release -DINFERENCE_TARGET=LINUX "
            f"-DINFERENCE_WEIGHTS_DIR={shlex.quote(cfg['remote']['weights_dir'])} "
            f"-DINFERENCE_PROFILING=OFF -DINFERENCE_BUILD_TEST=OFF "
            f"{' '.join(defs)} {' '.join(cfg['_bert']['remote'].get('cmake_args', []))}").strip()


def build_stamp(files: List[Path], proj: Path, defs: str) -> str:
    h = hashlib.sha1(defs.encode())
    for p in files:
        h.update(str(p.relative_to(proj)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ── board side ───────────────────────────────────────────────────────────────

def preflight(session: RemoteSession, cfg: dict) -> bool:
    print(_bold("\nPreflight (remote)"))
    ok = check_prerequisites(session, cfg["_bert"], label_width=36)
    out, _, rc = session.exec("python3 -c 'import sys; print(sys.version.split()[0]); "
                              "sys.exit(sys.version_info < (3, 8))'", timeout=15)
    good = rc == 0
    print(f"    {_green('OK     ') if good else _red('MISSING')} {'python3 >= 3.8':<36} {_dim(out.strip())}")
    ok &= good
    out, _, _ = session.exec("command -v systemd-run || echo none", timeout=15)
    print(f"    {_dim('info   ')} {'systemd-run':<36} {_dim(out.strip())}")
    out, _, _ = session.exec("grep -E 'CmaFree|CmaTotal' /proc/meminfo | tr -s ' ' | tr '\\n' ' '",
                             timeout=15)
    print(f"    {_dim('info   ')} {'CMA':<36} {_dim(out.strip())}  (the BERT pool BO needs ~216 MiB)")
    out, _, rc = session.exec("command -v cc || command -v gcc", timeout=15)
    print(f"    {_dim('info   ') if rc == 0 else _yellow('MISSING')} {'C compiler (libsampler.so)':<36} "
          f"{_dim(out.strip() or 'none: the server samples in Python (slower)')}")
    if uses_llm(cfg):
        lib = cfg["smollm2"]["lib"]
        _, _, rc = session.exec(f"test -f {shlex.quote(lib)}", timeout=15)
        good = rc == 0
        print(f"    {_green('OK     ') if good else _red('MISSING')} {'libsmollm2.so (smollm2 backend)':<36} "
              f"{_dim(lib if good else lib + ' - build it with the decoder project (CHAT_PLAN phase 3)')}")
        ok &= good
        tok = chat_path(cfg["smollm2"]["tokenizer"])
        good = tok.exists()
        print(f"    {_green('OK     ') if good else _red('MISSING')} {'tokenizer.json (host)':<36} {_dim(str(tok))}")
        ok &= good
    return ok


def stop_server(session: RemoteSession, cfg: dict, quiet: bool = False) -> str:
    d = cfg["remote"]["dir"]
    su = sudo(cfg)
    session.exec(f"{su}systemctl stop {UNIT} 2>/dev/null; {su}systemctl reset-failed {UNIT} 2>/dev/null; "
                 f"if [ -f {d}/server.pid ]; then {su}kill $(cat {d}/server.pid) 2>/dev/null; "
                 f"rm -f {d}/server.pid; fi", timeout=120)
    # anything else still running the server file (the [k] keeps pkill off this shell)
    session.exec(f"{su}pkill -f '[k]v260_chat_server.py' 2>/dev/null; sleep 0.5", timeout=30)
    out, _, _ = session.exec("pgrep -fa '[k]v260_chat_server.py' || echo none", timeout=15)
    state = out.strip()
    if not quiet:
        print(f"  server stopped" if state == "none" else _yellow(f"  still running: {state}"))
    return state


def upload_and_build(session: RemoteSession, cfg: dict, summary: dict, force: bool,
                     verbose: bool) -> bool:
    proj = Path(summary["project_dir"])
    d = cfg["remote"]["dir"]
    rproj = f"{d}/project"
    files = project_files(proj)
    defs = build_defs(cfg, summary["active"])
    stamp = build_stamp(files, proj, defs)
    out, _, _ = session.exec(f"cat {d}/lib/BUILD_STAMP 2>/dev/null; test -f {d}/lib/libbert_squad.so "
                             f"&& echo have-lib", timeout=15)
    if not force and stamp in out and "have-lib" in out:
        t0 = time.monotonic()
        return step("build", True, t0, "unchanged since the last build (lib/libbert_squad.so kept)")
    t0 = time.monotonic()
    session.exec_checked(f"rm -rf {rproj} && mkdir -p {rproj} {d}/lib", timeout=60)
    sftp = session._client.open_sftp()                          # noqa: SLF001
    try:
        dirs = set()
        for p in files:
            rel = p.relative_to(proj)
            for parent in list(rel.parents)[::-1][1:]:
                if parent not in dirs:
                    dirs.add(parent)
                    with contextlib.suppress(OSError):
                        sftp.mkdir(f"{rproj}/{parent.as_posix()}")
            sftp.put(str(p), f"{rproj}/{rel.as_posix()}")
    finally:
        sftp.close()
    step("upload", True, t0, f"{len(files)} project files -> {rproj}")
    t0 = time.monotonic()
    cmd = f"cmake -S {rproj} -B {rproj}/build {defs} 2>&1"
    out, _, rc = session.exec(cmd, timeout=cfg["build"]["timeout"])
    if not step("cmake", rc == 0, t0):
        tail(f"$ {cmd}\n{out}")
        return False
    if verbose:
        tail(out, 10)
    t0 = time.monotonic()
    cmd = (f"make -C {rproj}/build -j{cfg['build']['jobs']} bert_squad 2>&1 && "
           f"install -m 755 {rproj}/build/libbert_squad.so {d}/lib/libbert_squad.so && "
           f"echo {stamp} > {d}/lib/BUILD_STAMP && "
           f"nm -D --defined-only {d}/lib/libbert_squad.so | awk '{{print $3}}' | grep -c '^bert_'")
    out, _, rc = session.exec(cmd, timeout=cfg["build"]["timeout"])
    ok = rc == 0
    step("make", ok, t0, f"libbert_squad.so ({out.strip().splitlines()[-1]} bert_* symbols exported)"
         if ok else "")
    if not ok or verbose:
        tail(out)
    return ok


def upload_server(session: RemoteSession, cfg: dict) -> bool:
    t0 = time.monotonic()
    d = cfg["remote"]["dir"]
    vocab = bert_common.demo_path(cfg["_bert"].get("inputs", {}).get("vocab", "assets/vocab.txt"))
    session.exec_checked(f"mkdir -p {d}", timeout=15)
    sftp = session._client.open_sftp()                          # noqa: SLF001
    try:
        for p in SERVER_FILES + [vocab]:
            sftp.put(str(p), f"{d}/{p.name}")
        key = cfg["server"].get("api_key")
        if key:
            with sftp.open(f"{d}/api_key", "w") as f:
                f.write(key + "\n")
            sftp.chmod(f"{d}/api_key", 0o600)
        else:
            with contextlib.suppress(OSError):
                sftp.remove(f"{d}/api_key")
    finally:
        sftp.close()
    session.exec(f"chmod +x {d}/kv260_chat_server.py {d}/chat.py", timeout=15)
    extra = ""
    session.exec_checked(f"mkdir -p {d}/src {d}/lib", timeout=15)
    sftp = session._client.open_sftp()                          # noqa: SLF001
    try:
        for p in SAMPLER_SRC:
            sftp.put(str(p), f"{d}/src/{p.name}")
        if uses_llm(cfg):
            session.exec_checked(f"mkdir -p {d}/smollm2", timeout=15)
            sftp.put(str(chat_path(cfg["smollm2"]["tokenizer"])), f"{d}/smollm2/tokenizer.json")
            extra = " + smollm2/tokenizer.json"
    finally:
        sftp.close()
    return step("server", True, t0, f"{len(SERVER_FILES)} files + vocab.txt + src/sampler.[ch]{extra} -> {d}")


def build_sampler(session: RemoteSession, cfg: dict) -> bool:
    """lib/libsampler.so from src/sampler.c (a second; skipped when unchanged).
    Not fatal: without it the server samples in pure Python."""
    t0 = time.monotonic()
    d = cfg["remote"]["dir"]
    stamp = hashlib.sha1(b"".join(p.read_bytes() for p in SAMPLER_SRC)).hexdigest()
    out, _, _ = session.exec(f"cat {d}/lib/SAMPLER_STAMP 2>/dev/null; test -f {d}/lib/libsampler.so "
                             f"&& echo have-lib", timeout=15)
    if stamp in out and "have-lib" in out:
        return step("sampler", True, t0, "unchanged (lib/libsampler.so kept)")
    cmd = (f"cc -O2 -shared -fPIC -o {d}/lib/libsampler.so {d}/src/sampler.c -lm 2>&1 && "
           f"echo {stamp} > {d}/lib/SAMPLER_STAMP")
    out, _, rc = session.exec(cmd, timeout=120)
    if rc != 0:
        step("sampler", True, t0, _yellow("cc failed - the server will sample in Python"))
        tail(out)
        return False
    return step("sampler", True, t0, "lib/libsampler.so")


def server_argv(cfg: dict) -> List[str]:
    d, s = cfg["remote"]["dir"], cfg["server"]
    argv = ["/usr/bin/python3", "-u", f"{d}/kv260_chat_server.py", "--host", str(s["host"]),
            "--port", str(s["port"]), "--bert-lib", f"{d}/lib/libbert_squad.so",
            "--bert-weights", cfg["remote"]["weights_dir"], "--vocab", f"{d}/vocab.txt",
            "--max-windows", str(s["max_windows"]), "--doc-stride", str(s["doc_stride"]),
            "--queue-timeout", str(s["queue_timeout"]), "--max-queue", str(s["max_queue"])]
    for b in s["backends"]:
        argv += ["--backend", b]
    argv += ["--resident", str(s["resident"])]
    if uses_llm(cfg):
        llm = cfg["smollm2"]
        argv += ["--llm-lib", llm["lib"], "--llm-tokenizer", f"{d}/smollm2/tokenizer.json",
                 "--llm-sampler-lib", f"{d}/lib/libsampler.so", "--llm-context", str(llm["context"]),
                 "--llm-reserve", str(llm["reserve"]), "--llm-temperature", str(llm["temperature"]),
                 "--llm-top-p", str(llm["top_p"]), "--llm-top-k", str(llm["top_k"]),
                 "--llm-repetition-penalty", str(llm["repetition_penalty"]),
                 "--llm-dry-multiplier", str(llm["dry_multiplier"]),
                 "--llm-dry-base", str(llm["dry_base"]),
                 "--llm-dry-allowed-length", str(llm["dry_allowed_length"]),
                 "--llm-dry-penalty-last-n", str(llm["dry_penalty_last_n"]),
                 "--llm-dry-sequence-breakers", json.dumps(llm["dry_sequence_breakers"]),
                 *([] if llm["loop_guard"] else ["--llm-no-loop-guard"]),
                 "--llm-cma-mb", str(llm["cma_mb"])]
        if llm.get("weights_dir"):
            argv += ["--llm-weights", llm["weights_dir"]]
    if s.get("api_key"):
        argv += ["--api-key-file", f"{d}/api_key"]
    return argv


def start_server(session: RemoteSession, cfg: dict) -> bool:
    t0 = time.monotonic()
    d, s = cfg["remote"]["dir"], cfg["server"]
    argv = " ".join(shlex.quote(a) for a in server_argv(cfg))
    env = {k: str(v) for k, v in s.get("env", {}).items() if not k.startswith("_")}
    launcher = s.get("launcher", "systemd")
    if launcher == "systemd":
        out, _, rc = session.exec("command -v systemd-run", timeout=15)
        if rc != 0:
            print(_yellow("  systemd-run not found; using nohup"))
            launcher = "nohup"
    if launcher == "systemd":
        setenv = " ".join(f"--setenv={shlex.quote(k + '=' + v)}" for k, v in env.items())
        cmd = (f"{sudo(cfg)}systemd-run --unit={UNIT} --collect "
               f"--description='KV260 OpenAI-compatible chat server' "
               f"--working-directory={d} -p KillSignal=SIGTERM -p TimeoutStopSec=90 {setenv} {argv}")
    else:
        envs = " ".join(shlex.quote(f"{k}={v}") for k, v in env.items())
        cmd = (f"cd {d} && {sudo(cfg)}env {envs} setsid nohup {argv} > {d}/server.log 2>&1 "
               f"< /dev/null & echo $! > {d}/server.pid")
    out, err, rc = session.exec(cmd, timeout=60)
    return step("start", rc == 0, t0, f"{launcher}: {UNIT}" if rc == 0 else (out + err).strip())


def server_log(session: RemoteSession, cfg: dict, n: int = 40) -> str:
    d = cfg["remote"]["dir"]
    out, _, _ = session.exec(f"{sudo(cfg)}journalctl -u {UNIT} -n {n} --no-pager -o cat 2>/dev/null; "
                             f"[ -f {d}/server.log ] && tail -n {n} {d}/server.log", timeout=30)
    return out


def health_url(cfg: dict) -> str:
    return f"http://{cfg['ssh']['host']}:{cfg['server']['port']}"


def get_health(cfg: dict, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(health_url(cfg) + "/health", timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        with contextlib.suppress(Exception):
            return json.loads(e.read())
        return None
    except (OSError, ValueError):
        return None


def wait_health(session: RemoteSession, cfg: dict) -> Optional[dict]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < cfg["health_timeout"]:
        h = get_health(cfg)
        if h and h.get("status") == "ok":
            step("health", True, t0, f"{health_url(cfg)}/health: models "
                 f"{', '.join(m['id'] for m in h['models'])}")
            return h
        out, _, _ = session.exec(f"systemctl is-active {UNIT} 2>/dev/null || "
                                 f"(test -f {cfg['remote']['dir']}/server.pid && "
                                 f"kill -0 $(cat {cfg['remote']['dir']}/server.pid) && echo active)",
                                 timeout=15)
        if "active" not in out.split() and "activating" not in out.split():
            break
        time.sleep(1.0)
    step("health", False, t0, "the server did not come up")
    tail(server_log(session, cfg))
    return None


def follow(session: RemoteSession, cfg: dict) -> None:
    """Print the server log until Ctrl-C / SIGTERM."""
    cmd = (f"{sudo(cfg)}journalctl -fu {UNIT} -o cat -n 0" if cfg["server"]["launcher"] == "systemd"
           else f"tail -n 0 -f {cfg['remote']['dir']}/server.log")
    chan = session._client.get_transport().open_session()      # noqa: SLF001
    chan.exec_command(cmd)
    try:
        while True:
            if chan.recv_ready():
                sys.stdout.write(chan.recv(65536).decode("utf-8", "replace"))
                sys.stdout.flush()
            elif chan.exit_status_ready():
                break
            else:
                time.sleep(0.2)
    finally:
        chan.close()


def connect(cfg: dict) -> RemoteSession:
    s = RemoteSession(cfg["ssh"])
    print(f"{_bold('Connecting')} to {cfg['ssh']['user']}@{cfg['ssh']['host']}:{cfg['ssh'].get('port', 22)} ...")
    s.connect()
    return s


def print_ready(cfg: dict, h: dict) -> None:
    url = health_url(cfg) + "/v1"
    key = cfg["server"].get("api_key") or "none"
    print(_bold("\n  KV260 chat server is up"))
    print(f"    OpenAI base URL   {url}")
    print(f"    models            {', '.join(m['id'] for m in h['models'])}")
    print(f"    API key           {key}")
    print(f"    try               python3 demo/chat/chat.py --url {url} --doc some.txt")
    if any(m["id"] == "smollm2-135m-instruct" for m in h["models"]):
        print(f"                      python3 demo/chat/chat.py --url {url} --model smollm2-135m-instruct")
    print(f"    logs              ssh {cfg['ssh']['user']}@{cfg['ssh']['host']} journalctl -fu {UNIT}")
    print(f"    stop              demo/chat/deploy.py --stop   (the server owns the FPGA until then)\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--regenerate", action="store_true", help="regenerate the BERT project first")
    ap.add_argument("--rebuild", action="store_true", help="rebuild libbert_squad.so even if unchanged")
    ap.add_argument("--hold", action="store_true",
                    help="keep the board lock, follow the log, stop the server on Ctrl-C / SIGTERM")
    ap.add_argument("--no-lock", action="store_true", help="do not take the board lock (caller holds it)")
    ap.add_argument("--board-lock", default=None, help="override the board lock file")
    ap.add_argument("--stop", action="store_true", help="stop the server")
    ap.add_argument("--status", action="store_true", help="show whether the server runs and its /health")
    ap.add_argument("--check-only", action="store_true", help="remote preflight only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.board_lock:
        cfg["board_lock"] = args.board_lock

    if args.stop or args.status:
        session = connect(cfg)
        try:
            if args.stop:
                state = stop_server(session, cfg)
                tail(server_log(session, cfg, 6), 6)
                return 0 if state == "none" else 1
            out, _, _ = session.exec(f"systemctl is-active {UNIT} 2>/dev/null; "
                                     f"pgrep -fa '[k]v260_chat_server.py' || true", timeout=15)
            print(f"  {UNIT}: {out.strip() or 'not running'}")
            h = get_health(cfg)
            print(f"  {health_url(cfg)}/health: {json.dumps(h) if h else 'no answer'}")
            return 0 if h and h.get("status") == "ok" else 1
        finally:
            session.close()

    summary = ensure_project(cfg, args.regenerate) if not args.check_only else None
    lock = None if args.no_lock else cfg["board_lock"]
    stop_on_exit = False
    with board_lock(lock):
        session = connect(cfg)
        try:
            if not preflight(session, cfg):
                print(_red("\npreflight: remote prerequisites missing"))
                return 1
            if args.check_only:
                print(_green("\npreflight: all checks passed"))
                return 0
            print(_bold("\nDeploy"))
            stop_server(session, cfg, quiet=True)
            t0 = time.monotonic()
            w = sync_weights(session, Path(summary["project_dir"]) / "weights",
                             cfg["remote"]["weights_dir"])
            if not step("weights", w.ok, t0, w.output):
                return 1
            if not upload_and_build(session, cfg, summary, args.rebuild, args.verbose):
                return 1
            upload_server(session, cfg)
            build_sampler(session, cfg)
            if not start_server(session, cfg):
                return 1
            h = wait_health(session, cfg)
            if h is None:
                stop_server(session, cfg)
                return 1
            print_ready(cfg, h)
            if args.hold:
                stop_on_exit = True

                def _term(signum, _frame):
                    raise KeyboardInterrupt
                signal.signal(signal.SIGTERM, _term)
                print(_dim(f"holding the board lock ({lock or 'none'}); Ctrl-C or SIGTERM stops "
                           f"the server\n"), flush=True)
                try:
                    follow(session, cfg)
                except KeyboardInterrupt:
                    pass
        finally:
            if stop_on_exit:
                print(_bold("\nStopping the server"))
                with contextlib.suppress(Exception):
                    stop_server(session, cfg)
            session.close()
    if not args.hold:
        print(_yellow("note: the server keeps the FPGA busy until `deploy.py --stop`"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

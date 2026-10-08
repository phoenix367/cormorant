#!/usr/bin/env python3
"""deploy_and_run.py — the object_detection demo on the KV260 (doc/plans/YOLO_PLAN.md).

Uploads build/project (generate_project.py) and build/data (prepare.py) to
remote.work_dir, configures and builds detect_images on the board (the
loaded overlay's UIO names), runs it (live per-image latency), and fetches
the head maps to build/results/heads.bin and the run summary to
build/results/run.json.  Holds the per-board lock; the chat server owns the
FPGA, so it refuses while the server runs (--stop-server stops it and
restarts it afterwards).

usage: inference-scheduler/.venv/bin/python demo/object_detection/scripts/deploy_and_run.py
           [--config object_detection_config.json] [--stop-server]
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
REPO = DEMO.parent.parent
SCHED = REPO / "inference-scheduler"
sys.path.insert(0, str(SCHED))
from src.remote import RemoteSession, check_prerequisites, hold_board_lock  # noqa: E402

UIO_DEFINE = {"VectorOPKernel": "INFERENCE_VECTOROPKERNEL_INSTANCE",
              "MatmulKernel": "INFERENCE_MATMULKERNEL_INSTANCE",
              "ConvKernel": "INFERENCE_CONVKERNEL_INSTANCE",
              "PoolKernel": "INFERENCE_POOLKERNEL_INSTANCE"}
CHAT_DEPLOY = REPO / "demo" / "chat" / "deploy.py"


def _stream(session: RemoteSession, cmd: str, timeout: int):
    """exec with stderr streamed line by line (the per-image latencies)."""
    _, stdout, _ = session._client.exec_command(cmd, timeout=float(timeout))   # noqa: SLF001
    ch, out, err, part = stdout.channel, [], [], ""
    deadline = time.monotonic() + timeout
    while True:
        busy = False
        while ch.recv_ready():
            out.append(ch.recv(4096).decode(errors="replace"))
            busy = True
        while ch.recv_stderr_ready():
            c = ch.recv_stderr(4096).decode(errors="replace")
            err.append(c)
            part += c
            while "\n" in part:
                line, part = part.split("\n", 1)
                print("    " + line, file=sys.stderr, flush=True)
            busy = True
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
        if time.monotonic() > deadline:
            ch.close()
            raise TimeoutError(f"{cmd!r}: no exit after {timeout} s")
        if not busy:
            time.sleep(0.05)
    return "".join(out), "".join(err), ch.recv_exit_status()


def chat_server_running() -> bool:
    r = subprocess.run([sys.executable, str(CHAT_DEPLOY), "--status"], capture_output=True, text=True)
    return r.returncode == 0


def run(cfg: dict, project: Path, data: Path, results: Path, stop_server: bool = False,
        profile: bool = False) -> dict:
    stopped = False
    if chat_server_running():
        if not stop_server:
            raise SystemExit("the chat server is running (it owns the FPGA): stop it "
                             "(demo/chat/deploy.py --stop) or pass --stop-server")
        subprocess.run([sys.executable, str(CHAT_DEPLOY), "--stop"], check=True)
        stopped = True
    hold_board_lock(cfg)
    session = RemoteSession(cfg["ssh"])
    session.connect()
    try:
        if not check_prerequisites(session, cfg):
            raise SystemExit("board prerequisites missing (see above)")
        work = cfg["remote"]["work_dir"]
        proj, rdata = f"{work}/project", f"{work}/data"
        t0 = time.monotonic()
        session.exec(f"rm -rf {shlex.quote(proj)} {shlex.quote(rdata)}", timeout=60)
        n = session.upload_dir(project, proj)
        session.upload_dir(data, rdata)
        print(f"  upload: {n} project files + the images ({time.monotonic() - t0:.0f} s)", flush=True)
        active = json.loads((project / "demo.json").read_text())["active_kernels"]
        uio = cfg["remote"].get("uio_devices", {})
        defs = " ".join(f"-D{UIO_DEFINE[k]}={shlex.quote(uio[k])}" for k in active if k in uio)
        b = f"{proj}/build"
        t0 = time.monotonic()
        out, _, rc = session.exec(f"cmake -S {proj} -B {b} -DCMAKE_BUILD_TYPE=Release -DINFERENCE_TARGET=LINUX "
                                  f"-DBENCH_DATA_DIR={shlex.quote(rdata)} {defs} "
                                  f"{'-DINFERENCE_PROFILING=ON ' if profile else ''}"
                                  f"{' '.join(cfg['remote'].get('cmake_args', []))} 2>&1",
                                  timeout=cfg["build"]["timeout"])
        if rc:
            raise SystemExit(f"cmake failed:\n{out[-3000:]}")
        out, _, rc = session.exec(f"make -C {b} -j{cfg['build']['jobs']} detect_images 2>&1",
                                  timeout=cfg["build"]["timeout"])
        if rc:
            raise SystemExit(f"make failed:\n{out[-3000:]}")
        print(f"  build: detect_images ({time.monotonic() - t0:.0f} s)", flush=True)
        session.exec("sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory", timeout=60)
        sudo = "sudo -n " if cfg["run"].get("use_sudo", True) else ""
        heads = f"{rdata}/heads.bin"
        out, err, rc = _stream(session, f"{sudo}env BENCH_DATA_DIR={shlex.quote(rdata)} {b}/detect_images "
                                        f"{int(cfg['run'].get('warmup', 1))} {heads}", cfg["run"]["timeout"])
        if rc:
            raise SystemExit(f"detect_images failed (rc {rc}):\n{(out + err)[-3000:]}")
        summary = next((json.loads(ln) for ln in reversed(out.splitlines()) if ln.strip().startswith("{")), None)
        if summary is None:
            raise SystemExit(f"detect_images printed no summary:\n{out[-2000:]}")
        for ln in out.splitlines():
            if ln.strip().startswith("LAYERS_JSON:"):
                summary["layer_stats"] = json.loads(ln.strip()[len("LAYERS_JSON:"):])
        results.mkdir(parents=True, exist_ok=True)
        sftp = session._client.open_sftp()                                       # noqa: SLF001
        try:
            sftp.get(heads, str(results / "heads.bin"))
        finally:
            sftp.close()
        bid, _, _ = session.exec("sha256sum /lib/firmware/pl.bin | cut -c1-12", timeout=30)
        summary["bitstream"] = bid.strip()
        (results / "run.json").write_text(json.dumps(summary, indent=1))
        return summary
    finally:
        session.close()
        if stopped:
            subprocess.run([sys.executable, str(CHAT_DEPLOY)], check=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--config", default=str(DEMO / "object_detection_config.json"))
    ap.add_argument("--stop-server", action="store_true", help="stop the chat server for the run, restart it after")
    ap.add_argument("--profile", action="store_true", help="per-layer times (INFERENCE_PROFILING) in run.json")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    s = run(cfg, DEMO / "build" / "project", DEMO / "build" / "data", DEMO / "build" / "results", a.stop_server,
            a.profile)
    print(f"  {s['images']} images: mean {s['mean_ms']:.2f} ms, p50 {s['p50_ms']:.2f} ms ({s['fps']:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

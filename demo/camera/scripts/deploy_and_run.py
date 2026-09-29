#!/usr/bin/env python3
"""
deploy_and_run.py — camera demo deployment + live display.

Pipeline (single model):
  upload  → cmake -DINFERENCE_TARGET=LINUX → make classify_stream
          → launch board/camera_loop.py on the KV260
          → SFTP-pull the annotated frame it writes and display it locally

The board runs a persistent capture→inference→annotate loop; this host
script just builds the project, starts that loop, and shows the frames it
streams back.  Quit with 'q' (or ESC) in the display window, with Ctrl-C, or
let it stop after run.duration_s seconds.

Reuses the SSH/SFTP helpers from inference-scheduler/src/remote/.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

DEMO_DIR  = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent.parent
SCHED_DIR = REPO_ROOT / "inference-scheduler"

sys.path.insert(0, str(SCHED_DIR))
from src.remote import (   # noqa: E402
    _green, _red, _yellow, _bold, _dim,
    RemoteSession, check_prerequisites, hold_board_lock,
)


# ──────────────────────────────────────────────────────────────────────────────
# Preflight
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_DRIVER_HEADERS = {
    "VectorOPKernel": ["xvectoropkernel.h", "xvectoropkernel_hw.h"],
    "MatmulKernel":   ["xmatmulkernel.h",   "xmatmulkernel_hw.h"],
    "ConvKernel":     ["xconvkernel.h",     "xconvkernel_hw.h"],
    "PoolKernel":     ["xpoolingkernel.h",  "xpoolingkernel_hw.h"],
}

_BOARD_FILES = ["camera_loop.py", "preprocessing.py", "visualization.py",
                "power_monitor.py"]

_UIO_CMAKE_DEFINE = {
    "VectorOPKernel": "INFERENCE_VECTOROPKERNEL_INSTANCE",
    "MatmulKernel":   "INFERENCE_MATMULKERNEL_INSTANCE",
    "ConvKernel":     "INFERENCE_CONVKERNEL_INSTANCE",
    "PoolKernel":     "INFERENCE_POOLKERNEL_INSTANCE",
}


def _check_label(label: str, ok: bool, detail: str = "") -> bool:
    tag = _green("OK     ") if ok else _red("MISSING")
    line = f"    {tag} {label}"
    if detail:
        line += f"  {_dim(detail)}"
    print(line)
    return ok


def preflight_local(cfg: dict, project: dict, assets_dir: Path) -> bool:
    print(_bold("\nPreflight (local)"))
    ok = True

    labels = assets_dir / "labels" / "imagenet_1001_labels.txt"
    ok &= _check_label("assets/labels/imagenet_1001_labels.txt", labels.exists(),
                       f"{labels.stat().st_size:,} B" if labels.exists()
                       else str(labels))

    ssh = cfg.get("ssh", {})
    ok &= _check_label("ssh.host configured", bool(ssh.get("host")),
                       ssh.get("host", "(missing)"))

    uio      = cfg.get("remote", {}).get("uio_devices", {})
    proj_dir = Path(project["project_dir"])
    active   = project.get("active", [])

    ok &= _check_label(f"project '{project['model_name']}' on disk",
                       proj_dir.is_dir(), str(proj_dir))
    if proj_dir.is_dir():
        driver_dir = proj_dir / "driver"
        for kernel in active:
            for h in _REQUIRED_DRIVER_HEADERS.get(kernel, []):
                hp = driver_dir / h
                ok &= _check_label(f"  driver/{h}  ({kernel})",
                                   hp.exists(), "" if hp.exists() else str(hp))
            ok &= _check_label(f"  uio_devices.{kernel}", kernel in uio,
                               uio.get(kernel, f"(missing — falls back to "
                                                f"{kernel}_0)"))
        for f in _BOARD_FILES:
            bp = proj_dir / "board" / f
            ok &= _check_label(f"  board/{f}", bp.exists(),
                               "" if bp.exists() else str(bp))
    return ok


def preflight_remote(session: RemoteSession, cfg: dict) -> bool:
    print(_bold("\nPreflight (remote)"))
    ok = check_prerequisites(session, cfg, label_width=36)

    work_dir = cfg["remote"]["work_dir"].rstrip("/")
    parent   = "/".join(work_dir.split("/")[:-1]) or "/"
    out, _, rc = session.exec(
        f"test -w {shlex.quote(parent)} && echo writable || echo NO", timeout=10)
    ok &= _check_label(f"work_dir parent writable ({parent})",
                       rc == 0 and "writable" in out, work_dir)

    # Board-side Python dependencies for the camera loop.
    board_py = cfg["run"].get("board_python", "python3")
    probe = (f"{shlex.quote(board_py)} -c "
             "'import numpy, cv2, pyrealsense2 as rs; "
             "print(len(rs.context().query_devices()))'")
    out, err, rc = session.exec(probe, timeout=30)
    deps_ok = (rc == 0)
    ok &= _check_label("board python deps (pyrealsense2, numpy, cv2)", deps_ok,
                       "" if deps_ok else (err.strip().splitlines()[-1:]
                                           or ["import failed"])[0])
    if deps_ok:
        n_dev = (out.strip() or "0")
        ok &= _check_label("RealSense camera detected", n_dev not in ("0", ""),
                           f"{n_dev} device(s)")
    else:
        print(_yellow("    → install on the board: see the demo README "
                      "'Board setup' section"))
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Build helpers
# ──────────────────────────────────────────────────────────────────────────────

def upload_labels(session: RemoteSession, assets_dir: Path,
                  remote_assets: str) -> int:
    remote = f"{remote_assets}/labels"
    session.exec(f"mkdir -p {shlex.quote(remote)}", timeout=15)
    return session.upload_dir(assets_dir / "labels", remote)


def configure_and_build(session: RemoteSession, cfg: dict, remote_proj: str,
                        active_kernels: List[str]) -> bool:
    build_dir = f"{remote_proj}/build"
    extra     = " ".join(cfg["remote"].get("cmake_args", []))

    uio_defs = []
    uio_devs = cfg["remote"].get("uio_devices", {})
    for k in active_kernels:
        name = uio_devs.get(k)
        if name:
            uio_defs.append(f'-D{_UIO_CMAKE_DEFINE[k]}={shlex.quote(name)}')

    top_k_def = ""
    top_k = int(cfg["run"].get("top_k", 5))
    if top_k:
        top_k_def = f"-DBENCH_TOP_K={top_k}"

    cmake_cmd = (
        f"cmake -S {shlex.quote(remote_proj)} -B {shlex.quote(build_dir)} "
        f"-DCMAKE_BUILD_TYPE=Release -DINFERENCE_TARGET=LINUX "
        f"{top_k_def} {' '.join(uio_defs)} {extra} 2>&1"
    )
    print(f"  {_dim('cmake')} …")
    out, _, rc = session.exec(cmake_cmd, timeout=cfg["build"]["timeout"])
    if rc != 0:
        print(_red("  cmake failed:"))
        for ln in out.strip().splitlines()[-25:]:
            print(_dim(f"      {ln}"))
        return False

    make_cmd = (f"make -C {shlex.quote(build_dir)} "
                f"-j{cfg['build']['jobs']} classify_stream 2>&1")
    print(f"  {_dim('make classify_stream')} …")
    out, _, rc = session.exec(make_cmd, timeout=cfg["build"]["timeout"])
    if rc != 0:
        print(_red("  make failed:"))
        for ln in out.strip().splitlines()[-25:]:
            print(_dim(f"      {ln}"))
        return False
    print(_green("  build OK"))
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Board camera loop + host display
# ──────────────────────────────────────────────────────────────────────────────

def _build_loop_cmd(cfg: dict, remote_proj: str, remote_assets: str,
                    work_dir: str) -> str:
    """Assemble the sudo'd camera_loop.py invocation that runs on the board."""
    run   = cfg["run"]
    cam   = cfg.get("camera", {})
    prep  = cfg.get("preprocess", {})
    disp  = cfg.get("display", {})

    board_py = run.get("board_python", "python3")
    sudo     = "sudo -n " if run.get("use_sudo", True) else ""

    env_pairs = [f"{shlex.quote(str(k))}={shlex.quote(str(v))}"
                 for k, v in run.get("env", {}).items()
                 if not str(k).startswith("_")]
    env_prefix = f"env {' '.join(env_pairs)} " if env_pairs else ""

    args = {
        "--binary":      f"{remote_proj}/build/classify_stream",
        "--labels":      f"{remote_assets}/labels/imagenet_1001_labels.txt",
        "--out":         f"{work_dir}/stream/latest.jpg",
        "--stop-file":   f"{work_dir}/STOP",
        "--input-size":  int(prep.get("input_size", 224)),
        "--normalize":   prep.get("normalize", "tf"),
        "--top-k":       int(run.get("top_k", 5)),
        "--width":       int(cam.get("width", 640)),
        "--height":      int(cam.get("height", 480)),
        "--fps":         int(cam.get("fps", 30)),
        "--warmup":      int(run.get("warmup", 1)),
        "--jpeg-quality":int(disp.get("jpeg_quality", 80)),
        "--power-poll":  float(run.get("power_poll_s", 2.0)),
        "--target-fps":  float(run.get("target_fps", 0.0)),
        "--duration":    float(run.get("duration_s", 0.0)),
    }
    arg_str = " ".join(f"{k} {shlex.quote(str(v))}" for k, v in args.items())
    script  = f"{remote_proj}/board/camera_loop.py"
    return (f"{sudo}{env_prefix}{shlex.quote(board_py)} "
            f"{shlex.quote(script)} {arg_str}")


def _stderr_pump(chan, stop_evt: threading.Event) -> None:
    """Echo the board camera loop's stderr (its only diagnostic channel)."""
    buf = ""
    while not stop_evt.is_set():
        if chan.recv_stderr_ready():
            buf += chan.recv_stderr(4096).decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                sys.stderr.write("    " + line + "\n")
                sys.stderr.flush()
        elif chan.exit_status_ready() and not chan.recv_stderr_ready():
            break
        else:
            time.sleep(0.05)
    if buf:
        sys.stderr.write("    " + buf + "\n")


def run_camera_demo(session: RemoteSession, cfg: dict, project: dict,
                    remote_proj: str, remote_assets: str, work_dir: str,
                    *, save_only: bool, stream_dir: Path) -> bool:
    """Start the board loop and display/save the frames it streams back."""
    remote_jpg = f"{work_dir}/stream/latest.jpg"
    stop_file  = f"{work_dir}/STOP"
    session.exec(f"mkdir -p {shlex.quote(work_dir)}/stream && "
                 f"rm -f {shlex.quote(stop_file)} {shlex.quote(remote_jpg)}",
                 timeout=15)

    disp        = cfg.get("display", {})
    poll_ms     = int(disp.get("poll_interval_ms", 40))
    window      = disp.get("window_title", "Cormorant Camera Demo")
    duration_s  = float(cfg["run"].get("duration_s", 0.0))

    cv2 = None
    if not save_only:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            print(_yellow("  opencv-python not installed on the host — "
                          "falling back to --save-only"))
            save_only = True

    stream_dir.mkdir(parents=True, exist_ok=True)

    # Start the board loop on its own SSH channel so the SFTP display pulls
    # run independently of it.
    cmd  = _build_loop_cmd(cfg, remote_proj, remote_assets, work_dir)
    chan = session._client.get_transport().open_session()   # noqa: SLF001
    chan.exec_command(cmd)

    stop_evt = threading.Event()
    pump = threading.Thread(target=_stderr_pump, args=(chan, stop_evt),
                            daemon=True)
    pump.start()

    print(_bold(f"\n  ── streaming from the KV260 — "
                f"{'press q to quit' if not save_only else 'Ctrl-C to quit'}"
                f" ──"))
    if duration_s > 0:
        print(_dim(f"  (auto-stops after {duration_s:.0f}s)"))

    sftp       = session._client.open_sftp()                # noqa: SLF001
    local_jpg  = stream_dir / "latest.jpg"
    last_mtime = None
    saved      = 0
    t_start    = time.monotonic()
    interrupted = False

    try:
        while True:
            if chan.exit_status_ready():
                print(_yellow("\n  board loop ended"))
                break
            if duration_s > 0 and (time.monotonic() - t_start) >= duration_s:
                break

            try:
                mtime = sftp.stat(remote_jpg).st_mtime
            except (FileNotFoundError, IOError):
                mtime = None

            if mtime is not None and mtime != last_mtime:
                try:
                    sftp.get(remote_jpg, str(local_jpg))
                    last_mtime = mtime
                except (FileNotFoundError, IOError):
                    pass
                else:
                    if save_only:
                        dst = stream_dir / f"frame_{saved:05d}.jpg"
                        dst.write_bytes(local_jpg.read_bytes())
                        saved += 1
                    elif cv2 is not None:
                        img = cv2.imread(str(local_jpg))
                        if img is not None:
                            cv2.imshow(window, img)

            if save_only:
                time.sleep(poll_ms / 1000.0)
            else:
                key = cv2.waitKey(poll_ms) & 0xFF
                if key in (ord("q"), 27):       # q or ESC
                    break
    except KeyboardInterrupt:
        interrupted = True
        print(_yellow("\n  interrupted"))
    finally:
        # Signal the board loop to stop, then wait for it to drain.
        session.exec(f"touch {shlex.quote(stop_file)}", timeout=10)
        deadline = time.monotonic() + 15.0
        while not chan.exit_status_ready() and time.monotonic() < deadline:
            time.sleep(0.1)
        stop_evt.set()
        pump.join(timeout=3)
        try:
            sftp.close()
        except Exception:               # noqa: BLE001
            pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:           # noqa: BLE001
                pass

    exit_status = chan.recv_exit_status() if chan.exit_status_ready() else -1
    if save_only:
        print(_dim(f"  saved {saved} frame(s) to {stream_dir}"))
    return interrupted or exit_status in (0, -1)


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if not path.exists():
        from _config_help import missing_config_die
        missing_config_die(path)
    with open(path) as f:
        return json.load(f)


def deploy(cfg: dict, project: dict, assets_dir: Path, *,
           check_only: bool, save_only: bool, stream_dir: Path) -> bool:
    if not preflight_local(cfg, project, assets_dir):
        print(_red("\npreflight: one or more local prerequisites missing"))
        if not check_only:
            return False

    hold_board_lock(cfg)                  # one job per board (src/remote/lock.py)
    session = RemoteSession(cfg["ssh"])
    print(f"\n{_bold('Connecting')} to "
          f"{cfg['ssh']['user']}@{cfg['ssh']['host']}:{cfg['ssh']['port']} …")
    try:
        session.connect()
    except Exception as exc:            # noqa: BLE001
        print(_red(f"  connection failed: {exc}"))
        return False
    print(_green("  connected"))

    try:
        remote_ok = preflight_remote(session, cfg)
        if not remote_ok:
            print(_red("\npreflight: one or more remote prerequisites missing"))
            if not check_only:
                return False
        if check_only:
            print(_green("\npreflight: checks complete"))
            return remote_ok

        work_dir      = cfg["remote"]["work_dir"].rstrip("/")
        remote_assets = f"{work_dir}/assets"
        remote_proj   = f"{work_dir}/projects/{project['model_name']}"
        local_proj    = Path(project["project_dir"])

        print(f"\n{_bold('Uploading')} → {work_dir}")
        session.exec(f"rm -rf {shlex.quote(remote_proj)}", timeout=30)
        n_lbl  = upload_labels(session, assets_dir, remote_assets)
        n_proj = session.upload_dir(local_proj, remote_proj)
        print(_green(f"  uploaded {n_proj} project + {n_lbl} label file(s)"))

        if not configure_and_build(session, cfg, remote_proj,
                                   project.get("active", [])):
            return False

        ok = run_camera_demo(session, cfg, project, remote_proj,
                             remote_assets, work_dir,
                             save_only=save_only, stream_dir=stream_dir)

        if cfg.get("cleanup", True):
            print(f"\n{_dim('cleanup')} {work_dir}")
            session.exec(f"rm -rf {shlex.quote(work_dir)}", timeout=30)
        return ok
    finally:
        session.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEMO_DIR / "camera_config.json"))
    p.add_argument("--projects-dir", default=str(DEMO_DIR / "build" / "projects"),
                   help="root of generated projects (must contain projects.json)")
    p.add_argument("--assets-dir", default=str(DEMO_DIR / "assets"),
                   help="local directory containing labels/")
    p.add_argument("--check-only", action="store_true",
                   help="run the local + remote preflight checks and exit")
    p.add_argument("--save-only", action="store_true",
                   help="save streamed frames to build/stream/ instead of "
                        "opening a display window (for headless hosts)")
    p.add_argument("--no-cleanup", action="store_true",
                   help="leave the remote work_dir in place after the run")
    args = p.parse_args(argv)

    cfg = _load_json(Path(args.config))
    if args.no_cleanup:
        cfg["cleanup"] = False
    if cfg.get("display", {}).get("save_only"):
        args.save_only = True

    projects_summary = Path(args.projects_dir) / "projects.json"
    if not projects_summary.exists():
        print(f"error: {projects_summary} not found — "
              f"run scripts/generate_project.py first", file=sys.stderr)
        return 1
    projects = json.loads(projects_summary.read_text())
    if not projects:
        print("error: projects.json is empty", file=sys.stderr)
        return 1
    project = projects[0]      # the camera demo drives a single model

    assets_dir = Path(args.assets_dir)
    labels = assets_dir / "labels" / "imagenet_1001_labels.txt"
    if not labels.exists() and not args.check_only:
        print(f"error: {labels} missing — run scripts/download_assets.py first",
              file=sys.stderr)
        return 1

    stream_dir = Path(args.projects_dir).parent / "stream"
    ok = deploy(cfg, project, assets_dir,
                check_only=args.check_only,
                save_only=args.save_only,
                stream_dir=stream_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

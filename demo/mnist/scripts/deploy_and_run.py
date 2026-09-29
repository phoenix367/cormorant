#!/usr/bin/env python3
"""
deploy_and_run.py — upload generated MNIST projects to the KV260, build
them on the board, push the MNIST test set, and run bench_mnist for each
model.  Prints a summary table and writes results to results.json.

Pipeline per model:
  upload  → cmake -DINFERENCE_TARGET=LINUX  → make → run bench_mnist
            (the dataset is uploaded once and shared across runs)

The benchmark binary prints a single-line JSON object that we parse.

Reuses the SSH/SFTP helpers from inference-scheduler/src/remote/.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEMO_DIR  = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent.parent
SCHED_DIR = REPO_ROOT / "inference-scheduler"

sys.path.insert(0, str(SCHED_DIR))
from src.remote import (   # noqa: E402
    _green, _red, _yellow, _bold, _dim,
    RemoteSession, check_prerequisites, hold_board_lock,
)


# ──────────────────────────────────────────────────────────────────────────────
# Preflight checks
# ──────────────────────────────────────────────────────────────────────────────

# Header files we expect under each project's driver/ directory once the
# generate stage has run.  If any are missing, the on-board cmake will fail
# with a fatal error, so we surface the problem here instead.
_REQUIRED_DRIVER_HEADERS = {
    "VectorOPKernel": ["xvectoropkernel.h", "xvectoropkernel_hw.h"],
    "MatmulKernel":   ["xmatmulkernel.h",   "xmatmulkernel_hw.h"],
    "ConvKernel":     ["xconvkernel.h",     "xconvkernel_hw.h"],
    "PoolKernel":     ["xpoolingkernel.h",  "xpoolingkernel_hw.h"],
}


def _check_label(label: str, ok: bool, detail: str = "") -> bool:
    tag = _green("OK     ") if ok else _red("MISSING")
    line = f"    {tag} {label}"
    if detail:
        line += f"  {_dim(detail)}"
    print(line)
    return ok


def preflight_local(cfg: dict, projects: List[dict],
                     data_dir: Path) -> bool:
    """Verify the local working state needed to deploy and run.  No SSH yet."""
    print(_bold("\nPreflight (local)"))
    ok = True

    # MNIST IDX files present?
    for f in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        p = data_dir / f
        ok &= _check_label(f"data/{f}", p.exists(),
                           f"{p.stat().st_size:,} B" if p.exists() else str(p))

    # SSH config minimally populated?
    ssh = cfg.get("ssh", {})
    ok &= _check_label("ssh.host configured",
                       bool(ssh.get("host")),
                       ssh.get("host", "(missing)"))

    # uio_devices map covers everything declared by `local.driver_dirs` so we
    # can compile in the right runtime instance names later.
    uio = cfg.get("remote", {}).get("uio_devices", {})

    # Each generated project must have its driver headers in place.
    for proj in projects:
        name        = proj["model_name"]
        proj_dir    = Path(proj["project_dir"])
        active      = proj.get("active", [])
        driver_dir  = proj_dir / "driver"

        ok &= _check_label(f"project '{name}' on disk", proj_dir.is_dir(),
                           str(proj_dir))
        if not proj_dir.is_dir():
            continue

        for kernel in active:
            for h in _REQUIRED_DRIVER_HEADERS.get(kernel, []):
                hp = driver_dir / h
                ok &= _check_label(f"  driver/{h}  ({kernel})",
                                   hp.exists(), "" if hp.exists() else str(hp))
            # UIO mapping for kernels actually used by this model.
            ok &= _check_label(f"  uio_devices.{kernel}",
                               kernel in uio,
                               uio.get(kernel, "(missing — will fall back to "
                                                f"{kernel}_0)"))

    return ok


def preflight_remote(session: RemoteSession, cfg: dict) -> bool:
    """Verify the on-board environment is ready (cmake, gcc, XRT, UIO, sudo)."""
    print(_bold("\nPreflight (remote)"))
    ok = check_prerequisites(session, cfg, label_width=36)

    # Additionally verify the work_dir parent is writable.
    work_dir = cfg["remote"]["work_dir"].rstrip("/")
    parent   = "/".join(work_dir.split("/")[:-1]) or "/"
    out, _, rc = session.exec(
        f"test -w {shlex.quote(parent)} && echo writable || echo NO",
        timeout=10)
    ok &= _check_label(f"work_dir parent writable ({parent})",
                        rc == 0 and "writable" in out, work_dir)
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepLog:
    name:     str
    ok:       bool
    duration: float
    output:   str = ""


@dataclass
class ModelResult:
    name:    str
    steps:   List[StepLog]      = field(default_factory=list)
    metrics: Optional[dict]     = None

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)


# ──────────────────────────────────────────────────────────────────────────────
# Upload + build helpers
# ──────────────────────────────────────────────────────────────────────────────

def upload_dataset(session: RemoteSession, local_data: Path,
                    remote_data: str) -> StepLog:
    t0 = time.monotonic()
    out = ""
    try:
        session.exec(f"mkdir -p {shlex.quote(remote_data)}", timeout=15)
        n   = session.upload_dir(local_data, remote_data)
        ok  = True
        out = f"{n} files"
    except Exception as exc:
        ok  = False
        out = str(exc)
    return StepLog("dataset", ok, time.monotonic() - t0, out)


def upload_project(session: RemoteSession, local_proj: Path,
                    remote_proj: str) -> StepLog:
    t0 = time.monotonic()
    try:
        session.exec(f"rm -rf {shlex.quote(remote_proj)}", timeout=30)
        n  = session.upload_dir(local_proj, remote_proj)
        return StepLog("upload", True, time.monotonic() - t0, f"{n} files")
    except Exception as exc:
        return StepLog("upload", False, time.monotonic() - t0, str(exc))


_UIO_CMAKE_DEFINE = {
    "VectorOPKernel": "INFERENCE_VECTOROPKERNEL_INSTANCE",
    "MatmulKernel":   "INFERENCE_MATMULKERNEL_INSTANCE",
    "ConvKernel":     "INFERENCE_CONVKERNEL_INSTANCE",
    "PoolKernel":     "INFERENCE_POOLKERNEL_INSTANCE",
}


def configure_and_build(session: RemoteSession, cfg: dict,
                         remote_proj: str, remote_data: str,
                         active_kernels: List[str]) -> Tuple[StepLog,
                                                              StepLog]:
    build_dir = f"{remote_proj}/build"
    extra     = " ".join(cfg["remote"].get("cmake_args", []))

    # The bench_glue.h header defaults each UIO instance name to "<Kernel>_0".
    # If the loaded overlay uses different node labels (commonly fabric_vecop,
    # fabric_matmul, fabric_conv, fabric_pool), forward them as compile-time
    # defines so inference_init() opens the right /dev/uioN.
    uio_defs = []
    uio_devs = cfg["remote"].get("uio_devices", {})
    for k in active_kernels:
        name = uio_devs.get(k)
        if name:
            macro = _UIO_CMAKE_DEFINE[k]
            uio_defs.append(f'-D{macro}={shlex.quote(name)}')

    # Per-layer profiling: when enabled, the inference target compiles
    # inference_prof.c and bench_mnist's #if INFERENCE_PROFILING blocks.
    profile_def = ""
    if cfg["run"].get("profile_layers", False):
        profile_def = "-DINFERENCE_PROFILING=ON"

    cmake_cmd = (
        f"cmake -S {shlex.quote(remote_proj)} -B {shlex.quote(build_dir)} "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DINFERENCE_TARGET=LINUX "
        f"-DBENCH_DATA_DIR={shlex.quote(remote_data)} "
        f"{profile_def} {' '.join(uio_defs)} {extra} 2>&1"
    )

    t0 = time.monotonic()
    out, _, rc = session.exec(cmake_cmd, timeout=cfg["build"]["timeout"])
    cmake_step = StepLog("cmake", rc == 0, time.monotonic() - t0, out)
    if rc != 0:
        return cmake_step, StepLog("make", False, 0.0, "skipped (cmake failed)")

    t0 = time.monotonic()
    make_cmd = (f"make -C {shlex.quote(build_dir)} "
                f"-j{cfg['build']['jobs']} bench_mnist 2>&1")
    out, _, rc = session.exec(make_cmd, timeout=cfg["build"]["timeout"])
    make_step = StepLog("make", rc == 0, time.monotonic() - t0, out)
    return cmake_step, make_step


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark execution
# ──────────────────────────────────────────────────────────────────────────────

def _stream_exec(session: RemoteSession, cmd: str, *,
                  on_stderr_line, timeout: int) -> Tuple[str, str, int]:
    """
    Like RemoteSession.exec but streams stderr line-by-line so the caller can
    render progress messages live.  Returns (stdout, stderr, exit_code) once
    the command completes.  Reuses the existing paramiko transport.
    """
    client = session._client          # noqa: SLF001 — internal field reuse
    if client is None:
        raise RuntimeError("RemoteSession: not connected")

    _, stdout, _ = client.exec_command(cmd, timeout=float(timeout),
                                       get_pty=False)
    chan = stdout.channel
    out_chunks: List[str] = []
    err_chunks: List[str] = []
    err_partial = ""

    deadline = time.monotonic() + float(timeout)
    while True:
        progressed = False
        while chan.recv_ready():
            out_chunks.append(chan.recv(4096).decode("utf-8", errors="replace"))
            progressed = True
        while chan.recv_stderr_ready():
            chunk = chan.recv_stderr(4096).decode("utf-8", errors="replace")
            err_chunks.append(chunk)
            err_partial += chunk
            while "\n" in err_partial:
                line, err_partial = err_partial.split("\n", 1)
                on_stderr_line(line)
            progressed = True
        if chan.exit_status_ready() and not chan.recv_ready() \
                and not chan.recv_stderr_ready():
            break
        if time.monotonic() > deadline:
            chan.close()
            raise TimeoutError(f"streaming exec timed out after {timeout}s")
        if not progressed:
            time.sleep(0.05)

    if err_partial:
        on_stderr_line(err_partial)
    return ("".join(out_chunks), "".join(err_chunks), chan.recv_exit_status())


def run_benchmark(session: RemoteSession, cfg: dict,
                   remote_proj: str) -> Tuple[StepLog, Optional[dict]]:
    binary = f"{remote_proj}/build/bench_mnist"

    sudo   = "sudo -n " if cfg["run"].get("use_sudo", True) else ""
    iters  = int(cfg["run"].get("iters", 0))
    warmup = int(cfg["run"].get("warmup", 50))

    # Optional run-time environment variables.  We launch via `env K=V …`
    # rather than `K=V sudo …` because sudo wipes its own environment by
    # default; the leading `env(1)` runs after sudo and applies the
    # variables to the bench_mnist process.
    env_pairs = []
    for k, v in cfg["run"].get("env", {}).items():
        env_pairs.append(f"{shlex.quote(str(k))}={shlex.quote(str(v))}")
    env_prefix = f"env {' '.join(env_pairs)} " if env_pairs else ""

    cmd = f"{sudo}{env_prefix}{shlex.quote(binary)} {iters} {warmup}"

    # Render `progress:` lines in place (CR-overwrite); pass other stderr
    # output through verbatim so any error / setup message is still visible.
    state = {"progress_active": False}
    is_tty = sys.stderr.isatty()

    def _on_line(line: str) -> None:
        line = line.rstrip("\r")
        if line.startswith("progress:"):
            if is_tty:
                # Pad to clear leftovers from a longer previous line.
                sys.stderr.write("\r    " + line.ljust(78))
                sys.stderr.flush()
                state["progress_active"] = True
            else:
                sys.stderr.write("    " + line + "\n")
        else:
            if state["progress_active"]:
                sys.stderr.write("\n")
                state["progress_active"] = False
            if line:
                sys.stderr.write("    " + line + "\n")

    t0 = time.monotonic()
    try:
        out, err, rc = _stream_exec(session, cmd, on_stderr_line=_on_line,
                                     timeout=cfg["run"]["timeout"])
    except TimeoutError as exc:
        if state["progress_active"]:
            sys.stderr.write("\n"); state["progress_active"] = False
        return StepLog("run", False, time.monotonic() - t0, str(exc)), None
    finally:
        if state["progress_active"]:
            sys.stderr.write("\n")
    duration = time.monotonic() - t0

    if rc != 0:
        return StepLog("run", False, duration, (out + err).strip()), None

    # Find the summary JSON line emitted by bench_mnist (allow stderr noise
    # on stdout).  Skip the per-layer profile line ("LAYERS_JSON: {…}") here
    # — it's parsed separately below.
    metrics = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("LAYERS_JSON:"):
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                metrics = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if metrics is None:
        return StepLog("run", False, duration,
                       f"could not parse JSON output\n{out}{err}"), None

    # Optional per-layer stats and whole-run DDR counters: bench_mnist
    # prints exactly one "<MARKER>: {...}" line per source when
    # INFERENCE_PROFILING is on.
    for marker, key in (("LAYERS_JSON:", "layer_stats"),
                        ("DDR_JSON:",    "ddr_stats")):
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(marker):
                payload = line[len(marker):].strip()
                try:
                    metrics[key] = json.loads(payload)
                except json.JSONDecodeError:
                    metrics[key] = {"_parse_error": payload}
                break

    full_log = "STDOUT:\n" + out.rstrip() + "\nSTDERR:\n" + err.rstrip()
    return StepLog("run", True, duration, full_log), metrics


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def _format_step(s: StepLog) -> str:
    tag = _green("OK") if s.ok else _red("FAIL")
    return f"  {s.name:<8} → {tag:<14}  {s.duration:5.1f}s"


def _print_failure_tail(s: StepLog, *, max_lines: int = 30) -> None:
    """Echo the captured stdout/stderr of a failed step (capped)."""
    text = (s.output or "").rstrip()
    if not text:
        return
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = ["…"] + lines[-max_lines:]
    for ln in lines:
        print(_dim(f"      {ln}"))


def _save_log(log_dir: Path, model: str, step: StepLog) -> None:
    """Persist a step's full output for post-mortem inspection."""
    if not step.output:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{model}.{step.name}.log"
    path.write_text(step.output)


def _print_top_layers(layer_stats: Optional[dict], top_n: int = 5) -> None:
    """Echo the slowest-by-mean layers to stderr.  No-op when profiling
    was disabled (layer_stats is None or empty)."""
    if not layer_stats:
        return
    layers = layer_stats.get("layers") or []
    if not layers:
        return
    ranked = sorted(layers, key=lambda L: L.get("mean_us", 0.0), reverse=True)
    print(f"    per-layer (top {min(top_n, len(ranked))} by mean):")
    for L in ranked[:top_n]:
        print(f"      [{L['i']:>3}] {L['name'][:32]:<32} "
              f"calls={L['calls']:<6} "
              f"mean={L['mean_us']:>9.2f}us  "
              f"min={L['min_us']:>9.2f}us  "
              f"max={L['max_us']:>9.2f}us")


def _print_ddr(ddr_stats: Optional[dict]) -> None:
    """Echo whole-run DDR bandwidth to stderr, or report why it's missing.

    Schema matches inference_ddr.c::dump_json (post-vtable refactor):
      {available, backend, read_bytes, write_bytes, duration_ns,
       read_gbs, write_gbs, <backend-specific fields>}.
    For the zuplus_apm backend, slots[] carries a per-slot breakdown
    that we render under the aggregate total.
    """
    if not ddr_stats:
        return
    if not ddr_stats.get("available"):
        reason = ddr_stats.get("reason") or "(unspecified)"
        print(f"    ddr: {_yellow('unavailable')} — {reason}")
        return

    backend = ddr_stats.get("backend", "?")
    rgbs    = ddr_stats.get("read_gbs",  0.0)
    wgbs    = ddr_stats.get("write_gbs", 0.0)
    rbytes  = ddr_stats.get("read_bytes",  0)
    wbytes  = ddr_stats.get("write_bytes", 0)
    dur     = ddr_stats.get("duration_ns", 0) / 1e9

    extra = _format_ddr_backend_fields(ddr_stats)
    print(f"    ddr ({backend}{extra}): total "
          f"read={rgbs:.2f} GB/s ({_human_bytes(rbytes)})  "
          f"write={wgbs:.2f} GB/s ({_human_bytes(wbytes)})  "
          f"over {dur:.1f}s")

    # Per-slot breakdown (zuplus_apm backend).  Lets the user see at a
    # glance which DDRC ports actually carry FPGA traffic — the whole
    # point of asking for multiple slots in INFERENCE_DDR_APM_SLOTS.
    slots = ddr_stats.get("slots") or []
    if len(slots) > 1:
        print("      per-slot:")
        for s in slots:
            sn = s.get("slot")
            sr = s.get("read_bytes",  0)
            sw = s.get("write_bytes", 0)
            if sr == 0 and sw == 0:
                print(f"        slot {sn}: {_dim('idle')}")
            else:
                sr_gbs = sr / dur / 1e9 if dur > 0 else 0.0
                sw_gbs = sw / dur / 1e9 if dur > 0 else 0.0
                print(f"        slot {sn}: "
                      f"read={sr_gbs:.2f} GB/s ({_human_bytes(sr)})  "
                      f"write={sw_gbs:.2f} GB/s ({_human_bytes(sw)})")

    if rbytes == 0 and wbytes == 0:
        # Surface backend diagnostics so it's clear *why* the counters
        # are zero — the APM may be live but watching the wrong slots,
        # OR the block may be entirely dead (clock-gated, wrong addr).
        ctl_wb = ddr_stats.get("ctl_writeback")
        diag = f" — ctl_writeback={ctl_wb}" if ctl_wb else ""
        print(f"      {_yellow('warning')}: counters did not increment{diag}")
        print(f"      DDR APM (0xFD490000) has 6 slots — one per DDRC port. "
              f"Try INFERENCE_DDR_APM_SLOTS=0,1,2,3,4 (max 5 per run).")


def _format_ddr_backend_fields(ddr: dict) -> str:
    """Render any backend-specific fields (base_addr, slot, …) inline."""
    parts = []
    if "base_addr" in ddr: parts.append(f"@{ddr['base_addr']}")
    if "slot"      in ddr: parts.append(f"slot={ddr['slot']}")
    return f" {' '.join(parts)}" if parts else ""


def _human_bytes(n: int) -> str:
    if n <  1024:                return f"{n} B"
    if n <  1024 * 1024:         return f"{n / 1024:.1f} KiB"
    if n <  1024 * 1024 * 1024:  return f"{n / 1024 / 1024:.1f} MiB"
    return f"{n / 1024 / 1024 / 1024:.2f} GiB"


def print_report(results: List[ModelResult]) -> None:
    print()
    print(_bold("  ── MNIST KV260 BENCHMARK ──"))
    print()
    if not results:
        print(_red("  no models were run — see preflight output above"))
        print()
        return
    rows = []
    for r in results:
        if r.metrics:
            m = r.metrics
            rows.append((r.name, "OK",
                         f"{m['accuracy_pct']:.2f}%",
                         f"{m['mean_ms']:.3f}",
                         f"{m['p50_ms']:.3f}",
                         f"{m['p99_ms']:.3f}",
                         f"{m['throughput_ips']:.1f}"))
        else:
            rows.append((r.name, "FAIL", "-", "-", "-", "-", "-"))

    name_w = max(len(r[0]) for r in rows)
    hdr = f"  {'Model':<{name_w}}  {'Status':<6}  {'Acc':>8}  " \
          f"{'mean(ms)':>9}  {'p50(ms)':>9}  {'p99(ms)':>9}  {'IPS':>9}"
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr); print(sep)
    for r in rows:
        status = _green(r[1]) if r[1] == "OK" else _red(r[1])
        print(f"  {r[0]:<{name_w}}  {status:<14}  {r[2]:>8}  "
              f"{r[3]:>9}  {r[4]:>9}  {r[5]:>9}  {r[6]:>9}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def deploy_models(cfg: dict, projects: List[dict],
                   data_dir: Path, *, log_dir: Path,
                   check_only: bool = False,
                   verbose: bool = False) -> List[ModelResult]:
    if not preflight_local(cfg, projects, data_dir):
        print(_red("\npreflight: one or more local prerequisites missing"))
        if not check_only:
            return []

    hold_board_lock(cfg)                  # one job per board (src/remote/lock.py)
    session = RemoteSession(cfg["ssh"])
    print(f"\n{_bold('Connecting')} to "
          f"{cfg['ssh']['user']}@{cfg['ssh']['host']}:{cfg['ssh']['port']} …")
    try:
        session.connect()
    except Exception as exc:
        print(_red(f"  connection failed: {exc}"))
        return []
    print(_green("  connected"))

    if not preflight_remote(session, cfg):
        print(_red("\npreflight: one or more remote prerequisites missing"))
        if not check_only:
            session.close()
            return []
    if check_only:
        print(_green("\npreflight: all checks passed"))
        session.close()
        return []

    work_dir    = cfg["remote"]["work_dir"].rstrip("/")
    remote_data = f"{work_dir}/data"
    results: List[ModelResult] = []

    def _record(res: ModelResult, step: StepLog) -> None:
        res.steps.append(step)
        print(_format_step(step))
        _save_log(log_dir, res.name, step)
        if not step.ok:
            _print_failure_tail(step)
        # `run` output was already streamed live; don't echo it again on success.
        elif verbose and step.output and step.name != "run":
            _print_failure_tail(step)

    try:
        # Upload dataset once.
        print(f"\n{_bold('Uploading dataset')} → {remote_data}")
        ds_step = upload_dataset(session, data_dir, remote_data)
        print(_format_step(ds_step))
        if not ds_step.ok:
            _print_failure_tail(ds_step)
            print(_red("dataset upload failed; aborting"))
            return results

        for proj in projects:
            name        = proj["model_name"]
            local_proj  = Path(proj["project_dir"])
            remote_proj = f"{work_dir}/projects/{name}"
            print(f"\n{_bold(name)}")
            res = ModelResult(name=name)

            up = upload_project(session, local_proj, remote_proj)
            _record(res, up)
            if not up.ok:
                results.append(res); continue

            cm, mk = configure_and_build(session, cfg, remote_proj, remote_data,
                                          proj.get("active", []))
            _record(res, cm)
            if not cm.ok:
                results.append(res); continue
            _record(res, mk)
            if not mk.ok:
                results.append(res); continue

            run, metrics = run_benchmark(session, cfg, remote_proj)
            _record(res, run)
            if metrics:
                res.metrics = metrics
                print(f"    accuracy = {metrics['accuracy_pct']:.2f}%   "
                      f"mean = {metrics['mean_ms']:.3f} ms   "
                      f"throughput = {metrics['throughput_ips']:.1f} img/s")
                _print_top_layers(metrics.get("layer_stats"))
                _print_ddr      (metrics.get("ddr_stats"))
            results.append(res)

        if cfg.get("cleanup", True):
            print(f"\n{_dim('cleanup')} {work_dir}")
            session.exec(f"rm -rf {shlex.quote(work_dir)}", timeout=30)
    finally:
        session.close()

    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEMO_DIR / "mnist_config.json"))
    p.add_argument("--projects-dir",
                   default=str(DEMO_DIR / "build" / "projects"),
                   help="root of generated projects (must contain projects.json)")
    p.add_argument("--data-dir",
                   default=str(DEMO_DIR / "assets" / "data"),
                   help="local directory with the unpacked MNIST IDX files")
    p.add_argument("--results",
                   default=str(DEMO_DIR / "build" / "results.json"),
                   help="path to write the JSON results summary")
    p.add_argument("--no-cleanup", action="store_true",
                   help="leave the remote work_dir in place after the run")
    p.add_argument("--check-only", action="store_true",
                   help="run the local + remote preflight checks and exit")
    p.add_argument("--profile-layers", action="store_true",
                   help="enable per-layer wall-clock profiling "
                        "(overrides run.profile_layers in the config)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    cfg = _load_json(Path(args.config))
    if args.no_cleanup:
        cfg["cleanup"] = False
    if args.profile_layers:
        cfg.setdefault("run", {})["profile_layers"] = True

    projects_summary = Path(args.projects_dir) / "projects.json"
    if not projects_summary.exists():
        print(f"error: {projects_summary} not found — "
              f"run scripts/generate_project.py first", file=sys.stderr)
        return 1
    projects = json.loads(projects_summary.read_text())

    data_dir = Path(args.data_dir)
    for required in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        if not (data_dir / required).exists():
            print(f"error: {data_dir / required} missing — "
                  f"run scripts/download_assets.py first", file=sys.stderr)
            return 1

    log_dir = Path(args.results).parent / "logs"
    results = deploy_models(cfg, projects, data_dir,
                             log_dir=log_dir,
                             check_only=args.check_only,
                             verbose=args.verbose)
    if args.check_only:
        return 0
    if log_dir.exists():
        print(_dim(f"per-step logs written to {log_dir}"))

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    summary = [{"name":   r.name,
                "ok":     r.ok,
                "metrics": r.metrics,
                "steps":   [{"name": s.name, "ok": s.ok,
                             "duration_s": s.duration} for s in r.steps]}
               for r in results]
    Path(args.results).write_text(json.dumps(summary, indent=2))

    print_report(results)
    n_ok = sum(1 for r in results if r.ok)
    return 0 if n_ok == len(results) and results else 1


if __name__ == "__main__":
    sys.exit(main())

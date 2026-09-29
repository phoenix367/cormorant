#!/usr/bin/env python3
"""
run_remote_tests.py — Hardware inference test runner for KV260.

For each ONNX model in the configured list:
  1. Generate a C inference project locally (inference_scheduler.py)
  2. Upload the project to the remote board via SSH/SFTP
  3. Build on the remote machine  (cmake -DINFERENCE_TARGET=LINUX + make)
  4. Run the test binary as root  (sudo ./build/test_inference)
  5. Collect pass/fail results and print a summary report

Prerequisites on the remote machine:
  - gcc/g++, cmake ≥ 3.19, make
  - XRT runtime installed (xrt.h reachable via pkg-config or /opt/xilinx/xrt)
  - UIO devices whose /sys/class/uio/uio*/name values match the configured
    per-kernel instance names (see remote.uio_devices in the config); verify:
      cat /sys/class/uio/uio*/name
  - Passwordless sudo for the SSH user, OR log in directly as root

Usage:
  # Run all models listed in the config file
  python run_remote_tests.py --config remote_config.json

  # Override model list on the command line
  python run_remote_tests.py --config remote_config.json \\
      --models test/models/single_add.onnx test/models/relu_chain.onnx

  # Verify SSH connectivity and remote prerequisites without running tests
  python run_remote_tests.py --config remote_config.json --check-only

  # Keep remote directories after the run (for debugging build/run failures)
  python run_remote_tests.py --config remote_config.json --no-cleanup

  # Show full build + test output for every model, not only failures
  python run_remote_tests.py --config remote_config.json --verbose
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.remote import (  # noqa: E402
    _green, _red, _yellow, _bold, _dim,
    load_config, uio_devices_from_cfg,
    RemoteSession, check_prerequisites,
)
try:
    from src.kernels import KERNEL_REGISTRY as _KERNEL_REGISTRY
except ImportError:
    _KERNEL_REGISTRY = {}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration — tests-specific additions on top of SHARED_DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

_EXTRA_DEFAULTS: dict = {
    "models": [],
}


# ──────────────────────────────────────────────────────────────────────────────
# Test result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    name:     str
    ok:       bool
    output:   str   = ""
    duration: float = 0.0


@dataclass
class TestResult:
    model_name: str
    steps:      List[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def status(self) -> str:
        if not self.steps:
            return "SKIPPED"
        for s in self.steps:
            if not s.ok:
                return f"{s.name.upper()}_ERROR"
        return "PASSED"

    @property
    def total_time(self) -> float:
        return sum(s.duration for s in self.steps)

    @property
    def failed_step(self) -> Optional[StepResult]:
        for s in self.steps:
            if not s.ok:
                return s
        return None



# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cmake_instance_flags(uio_devices: dict) -> List[str]:
    """Convert per-kernel UIO name map to cmake -D compile-definition flags."""
    flags = []
    for kernel_name, uio_name in uio_devices.items():
        macro = f"INFERENCE_{kernel_name.upper()}_INSTANCE"
        flags.append(f'-D{macro}=\\"{uio_name}\\"')
    return flags


def _resolve_local_driver(cfg: dict, merge_dir: Path) -> Optional[Path]:
    """
    Return a single local driver directory for inference_scheduler.py --driver-dir.

    With local.driver_dir set: returns it directly.
    With local.driver_dirs dict set: merges all source directories into *merge_dir*
      and returns that.  Useful for multi-kernel models where each kernel's HLS
      output lives in a separate directory.

    Raises FileNotFoundError if any configured path does not exist on disk.
    """
    single = cfg["local"].get("driver_dir")
    per_kernel = cfg["local"].get("driver_dirs", {})

    if single:
        p = Path(single)
        if not p.is_dir():
            raise FileNotFoundError(
                f"local.driver_dir not found: {p}\n"
                f"Run HLS synthesis or update the path in your remote_config."
            )
        return p

    if not per_kernel:
        return None

    for kernel_name, kdir in per_kernel.items():
        src = Path(kdir)
        if not src.is_dir():
            raise FileNotFoundError(
                f"local.driver_dirs[{kernel_name!r}] not found: {src}\n"
                f"Run HLS synthesis or update the path in your remote_config."
            )

    merge_dir.mkdir(parents=True, exist_ok=True)
    for kdir in per_kernel.values():
        src = Path(kdir)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, merge_dir / f.name)
    return merge_dir


# ──────────────────────────────────────────────────────────────────────────────
# Project generation (local)
# ──────────────────────────────────────────────────────────────────────────────

def generate_project(model_path: Path, out_dir: Path,
                     driver_dir: Optional[Path]) -> Tuple[bool, str]:
    """
    Invoke inference_scheduler.py to produce the C project under *out_dir*.

    Returns (success, log_text).  The scheduler writes progress to stderr;
    both stdout and stderr are captured and returned as the log.
    """
    import subprocess

    scheduler = Path(__file__).parent / "inference_scheduler.py"
    cmd: List[str] = [
        sys.executable, str(scheduler),
        str(model_path),
        "--out-dir", str(out_dir),
    ]
    if driver_dir:
        cmd += ["--driver-dir", str(driver_dir)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log  = proc.stderr + proc.stdout          # scheduler logs to stderr
    return proc.returncode == 0, log


# ──────────────────────────────────────────────────────────────────────────────
# Remote build
# ──────────────────────────────────────────────────────────────────────────────

def build_remote(session: RemoteSession,
                 project_dir: str,
                 build_dir: str,
                 uio_devices: dict,
                 extra_cmake: List[str],
                 jobs: int,
                 timeout: int) -> Tuple[bool, str]:
    """
    Configure and compile the inference project on the remote machine.

    Two commands are run sequentially:
      cmake -S <project_dir> -B <build_dir> -DINFERENCE_TARGET=LINUX ...
      make -C <build_dir> -j<jobs> test_inference

    *uio_devices* maps kernel name → UIO sysfs name and is forwarded as per-kernel
    cmake -D flags (e.g. -DINFERENCE_VECTOROPKERNEL_INSTANCE=\"VectorOPKernel_0\").
    An empty dict means the defaults compiled into the test binary are used.

    Returns (success, combined_log).
    """
    cmake_flags = [
        "-DINFERENCE_TARGET=LINUX",
        # weights/ and expected/ subdirs live under the project source tree
        f"-DINFERENCE_WEIGHTS_DIR={project_dir}",
    ] + _cmake_instance_flags(uio_devices) + extra_cmake

    cmake_cmd = (
        f"cmake -S {project_dir} -B {build_dir} "
        + " ".join(cmake_flags)
    )
    make_cmd = f"make -C {build_dir} -j{jobs} test_inference 2>&1"

    log_parts: List[str] = []

    for cmd in (cmake_cmd, make_cmd):
        out, err, rc = session.exec(cmd, timeout=timeout)
        log_parts.append(f"$ {cmd}\n{out}")
        if err.strip():
            log_parts.append(err)
        if rc != 0:
            return False, "\n".join(log_parts)

    return True, "\n".join(log_parts)


# ──────────────────────────────────────────────────────────────────────────────
# Remote test execution
# ──────────────────────────────────────────────────────────────────────────────

def run_test(session: RemoteSession,
             project_dir: str,
             build_dir: str,
             use_sudo: bool,
             timeout: int) -> Tuple[bool, str]:
    """
    Execute test_inference on the remote machine.

    The binary is run from *project_dir* so that relative paths like
    ./weights/*.dat and ./expected/*.dat resolve correctly.

    Root privileges are obtained via "sudo" when *use_sudo* is True and the
    SSH user is not root; if the user is already root, sudo is skipped.
    The test passes when the binary exits with code 0 and prints "PASSED".

    UIO device names are baked into the binary via CMake compile definitions at
    build time (INFERENCE_*_INSTANCE macros); no runtime argument is needed.

    Returns (passed, full_output).
    """
    binary = f"{build_dir}/test_inference"

    # Use sudo -n (non-interactive) so the command fails clearly if
    # passwordless sudo is not configured, rather than hanging for a password.
    prefix = "sudo -n " if use_sudo else ""

    cmd = f"cd {project_dir} && {prefix}{binary}"

    try:
        out, err, rc = session.exec(cmd, timeout=timeout)
    except TimeoutError as exc:
        return False, str(exc)

    combined = out + err
    passed   = (rc == 0) and ("PASSED" in combined)
    return passed, combined


# ──────────────────────────────────────────────────────────────────────────────
# Per-model orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_model_test(model_path: Path,
                   cfg: dict,
                   session: RemoteSession,
                   tmp_root: Path) -> TestResult:
    """
    Execute the full pipeline (generate → upload → build → run) for one model.

    Each phase records a StepResult; the pipeline aborts on first failure.
    """
    name   = model_path.stem
    result = TestResult(model_name=name)

    local_project = tmp_root / name
    merge_dir     = tmp_root / f"{name}_drv_merge"
    try:
        local_driver = _resolve_local_driver(cfg, merge_dir)
    except FileNotFoundError as exc:
        result.steps.append(StepResult("drivers", False, str(exc), 0.0))
        return result
    uio_devices   = uio_devices_from_cfg(cfg)

    work_dir        = cfg["remote"]["work_dir"].rstrip("/")
    remote_project  = f"{work_dir}/{name}"
    remote_build    = f"{remote_project}/build"

    # ── 1. Generate ─────────────────────────────────────────────────────── #
    _step_print(name, "generate", "generating project...")
    t0    = time.monotonic()
    ok, gen_log = generate_project(model_path, local_project, local_driver)
    dur   = time.monotonic() - t0
    result.steps.append(StepResult("generate", ok, gen_log, dur))
    _step_done(ok, dur)
    if not ok:
        return result

    # ── 2. Upload ────────────────────────────────────────────────────────── #
    _step_print(name, "upload", "uploading...")
    t0 = time.monotonic()
    try:
        n = session.upload_dir(local_project, remote_project)
        dur = time.monotonic() - t0
        _step_done(True, dur, extra=f"{n} files")
        result.steps.append(StepResult("upload", True,
                                       f"{n} files uploaded in {dur:.1f}s", dur))
    except Exception as exc:
        dur = time.monotonic() - t0
        _step_done(False, dur)
        result.steps.append(StepResult("upload", False, str(exc), dur))
        return result

    # ── 2b. Copy drivers from remote (if configured and no local bundle) ── #
    remote_driver_dirs = cfg["remote"].get("driver_dirs", {})
    remote_driver_dir  = cfg["remote"].get("driver_dir")
    if not local_driver:
        if remote_driver_dirs:
            # Per-kernel remote dirs — copy each kernel's files separately
            all_ok_drv, drv_parts = True, []
            for kname, kdir in remote_driver_dirs.items():
                _step_print(name, "drivers", f"copying {kname} drivers...")
                t0 = time.monotonic()
                ok, drv_log = _copy_remote_drivers(
                    session, kdir, remote_project, kernel_names=[kname]
                )
                dur = time.monotonic() - t0
                _step_done(ok, dur)
                drv_parts.append(drv_log)
                if not ok:
                    all_ok_drv = False
            result.steps.append(StepResult("drivers", all_ok_drv,
                                           "\n".join(drv_parts)))
            if not all_ok_drv:
                return result

        elif remote_driver_dir:
            # Single shared remote directory for all kernel files
            _step_print(name, "drivers", "copying remote drivers...")
            t0 = time.monotonic()
            ok, drv_log = _copy_remote_drivers(
                session, remote_driver_dir, remote_project
            )
            dur = time.monotonic() - t0
            _step_done(ok, dur)
            result.steps.append(StepResult("drivers", ok, drv_log, dur))
            if not ok:
                return result

    # ── 3. Build ─────────────────────────────────────────────────────────── #
    _step_print(name, "build", "cmake + make...")
    t0 = time.monotonic()
    ok, build_log = build_remote(
        session,
        project_dir = remote_project,
        build_dir   = remote_build,
        uio_devices = uio_devices,
        extra_cmake = cfg["remote"].get("cmake_args", []),
        jobs        = cfg["build"]["jobs"],
        timeout     = cfg["build"]["timeout"],
    )
    dur = time.monotonic() - t0
    _step_done(ok, dur)
    result.steps.append(StepResult("build", ok, build_log, dur))
    if not ok:
        return result

    # ── 4. Run ───────────────────────────────────────────────────────────── #
    _step_print(name, "run", "running test...")
    t0 = time.monotonic()
    ok, run_output = run_test(
        session,
        project_dir = remote_project,
        build_dir   = remote_build,
        use_sudo    = cfg["run"]["use_sudo"],
        timeout     = cfg["run"]["timeout"],
    )
    dur = time.monotonic() - t0
    _step_done(ok, dur)
    result.steps.append(StepResult("run", ok, run_output, dur))

    return result


def _copy_remote_drivers(session: RemoteSession,
                         src_dir: str, project_dir: str,
                         kernel_names: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Copy kernel driver files from an existing directory on the remote machine
    into <project_dir>/driver/.  Used when drivers are pre-deployed on the board.

    *kernel_names* limits copying to the given kernel entries in KERNEL_REGISTRY.
    When None (or registry unavailable), falls back to the VectorOPKernel file list.
    """
    if _KERNEL_REGISTRY and kernel_names:
        driver_files: List[str] = []
        seen: set = set()
        for kname in kernel_names:
            kd = _KERNEL_REGISTRY.get(kname)
            if kd:
                for f in kd.driver_files:
                    if f not in seen:
                        seen.add(f)
                        driver_files.append(f)
    elif _KERNEL_REGISTRY:
        # copy all registered kernel files
        driver_files = []
        seen = set()
        for kd in _KERNEL_REGISTRY.values():
            for f in kd.driver_files:
                if f not in seen:
                    seen.add(f)
                    driver_files.append(f)
    else:
        driver_files = [
            "xvectoropkernel.h",
            "xvectoropkernel_hw.h",
            "xvectoropkernel.c",
            "xvectoropkernel_sinit.c",
            "xvectoropkernel_linux.c",
        ]

    dst = f"{project_dir}/driver"
    missing = []
    for fname in driver_files:
        src  = f"{src_dir}/{fname}"
        cmd  = f"cp {src} {dst}/{fname} 2>&1"
        _, _, rc = session.exec(cmd, timeout=15)
        if rc != 0:
            missing.append(fname)

    if missing:
        return False, f"Missing driver files in {src_dir}: {', '.join(missing)}"
    return True, f"Copied {len(driver_files)} driver files from {src_dir}"


# ──────────────────────────────────────────────────────────────────────────────
# Remote cleanup
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_remote(session: RemoteSession, cfg: dict,
                   model_names: List[str]) -> None:
    work_dir = cfg["remote"]["work_dir"].rstrip("/")
    for name in model_names:
        session.exec(f"rm -rf {work_dir}/{name}", timeout=30)
    # the work dir itself only if nothing else is in it (it may be shared)
    session.exec(f"rmdir {work_dir} 2>/dev/null || true", timeout=15)



# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _step_print(model: str, step: str, msg: str) -> None:
    print(f"  [{_bold(model)}] {step:<9} {msg}", end="", flush=True)


def _step_done(ok: bool, dur: float, extra: str = "") -> None:
    tag   = _green("OK")  if ok else _red("FAIL")
    extra = f"  {_dim(extra)}" if extra else ""
    print("\r  \033[K  " + " " * 40 +  # clear line
          f"  → {tag}  {dur:.1f}s{extra}")


def print_report(results: List[TestResult], verbose: bool = False) -> None:
    """Print a summary table and, when verbose or failed, per-step output."""
    col = max((len(r.model_name) for r in results), default=20) + 2
    hdr = f"  {'Model':<{col}}  {'Status':<18}  {'Time':>7}  Steps"

    print()
    print(_bold(hdr))
    print("  " + "─" * (len(hdr) - 2))

    for r in results:
        status_col = _green("PASSED") if r.ok else _red(r.status)
        steps_str  = "  ".join(
            (_green(s.name) if s.ok else _red(s.name)) for s in r.steps
        )
        print(f"  {r.model_name:<{col}}  {status_col:<27}  "
              f"{r.total_time:>6.1f}s  {steps_str}")

        if verbose or not r.ok:
            failed = r.failed_step
            if failed:
                print(f"\n    ─── {failed.name} output ───")
                for line in failed.output.strip().splitlines():
                    print(f"    {_dim(line)}")
                print()

    print("  " + "─" * (len(hdr) - 2))
    passed  = sum(1 for r in results if r.ok)
    total   = len(results)
    summary = f"  {passed}/{total} passed"
    print(_bold(_green(summary) if passed == total else _red(summary)))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference hardware tests on a remote KV260 board.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Config file: see remote_config.json.example

            Remote prerequisites:
              cmake ≥ 3.19, make, gcc
              XRT runtime (xrt.h + libxrt_core.so)
              UIO devices listed in remote.uio_devices (e.g. "VectorOPKernel_0")
                verify: cat /sys/class/uio/uio*/name
              Passwordless sudo  -OR-  SSH as root (user: "root")
        """),
    )
    p.add_argument("--config",    metavar="FILE", required=True,
                   help="JSON config file (see remote_config.json.example)")
    p.add_argument("--models",    metavar="ONNX", nargs="+",
                   help="Override model list; paths relative to this script's dir")
    p.add_argument("--check-only", action="store_true",
                   help="Check SSH connectivity and remote prerequisites, then exit")
    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep remote build directories (for debugging)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print build/run output for every test (not only failures)")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop after the first test failure")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ── Load config ─────────────────────────────────────────────────────── #
    try:
        cfg = load_config(args.config, _EXTRA_DEFAULTS)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # ── Resolve model list ───────────────────────────────────────────────── #
    base = Path(__file__).parent
    raw_models = args.models if args.models else cfg["models"]
    if not raw_models:
        print("error: no models specified in config or --models", file=sys.stderr)
        return 1

    models: List[Path] = []
    for m in raw_models:
        p = Path(m) if Path(m).is_absolute() else base / m
        models.append(p)

    missing = [m for m in models if not m.exists()]
    if missing:
        for m in missing:
            print(f"error: model not found: {m}", file=sys.stderr)
        return 1

    # ── Connect ─────────────────────────────────────────────────────────── #
    ssh_cfg = cfg["ssh"]
    print(f"\n{_bold('Connecting')} to "
          f"{ssh_cfg['user']}@{ssh_cfg['host']}:{ssh_cfg['port']} …")
    session = RemoteSession(ssh_cfg)
    try:
        session.connect()
        print(f"  {_green('Connected')}")
    except Exception as exc:
        print(f"  {_red('FAILED')}: {exc}", file=sys.stderr)
        return 1

    try:
        # ── Prerequisite check ───────────────────────────────────────────── #
        print(f"\n{_bold('Remote prerequisites')}")
        prereqs_ok = check_prerequisites(session, cfg)

        if args.check_only:
            print(f"\n{_green('All OK') if prereqs_ok else _red('Issues found')} "
                  f"— connectivity check complete.")
            return 0 if prereqs_ok else 1

        if not prereqs_ok:
            print(
                f"\n{_yellow('Warning')}: some prerequisites are missing. "
                "Tests may fail at build or run time.",
                file=sys.stderr,
            )

        # ── Create remote working directory ──────────────────────────────── #
        work_dir = cfg["remote"]["work_dir"]
        session.exec(f"mkdir -p {work_dir}", timeout=10)

        # ── Run tests ────────────────────────────────────────────────────── #
        print(f"\n{_bold('Running tests')} "
              f"({len(models)} model{'s' if len(models) != 1 else ''})\n")

        results: List[TestResult] = []
        tmp_root = Path(tempfile.mkdtemp(prefix="inference_hw_"))

        try:
            for model_path in models:
                result = run_model_test(model_path, cfg, session, tmp_root)
                results.append(result)

                if args.verbose and result.ok:
                    # Show test output even on success
                    run_step = next(
                        (s for s in result.steps if s.name == "run"), None
                    )
                    if run_step and run_step.output.strip():
                        print("    ─── run output ───")
                        for line in run_step.output.strip().splitlines():
                            print(f"    {_dim(line)}")
                        print()

                if not result.ok and not args.verbose:
                    # Show failure output immediately (also in the final report)
                    failed = result.failed_step
                    if failed:
                        print(f"\n    ─── {failed.name} output ───")
                        for line in failed.output.strip().splitlines():
                            print(f"    {_dim(line)}")
                        print()

                if args.fail_fast and not result.ok:
                    print(_yellow("  Stopping after first failure (--fail-fast)."))
                    break

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        # ── Cleanup remote ────────────────────────────────────────────────── #
        if not args.no_cleanup:
            print(f"\n{_bold('Cleaning up')} remote directories…")
            cleanup_remote(session, cfg, [r.model_name for r in results])
            print(f"  {_green('Done')}")
        else:
            print(f"\n{_dim('Remote directories kept (--no-cleanup).')}")
            print(f"  {_dim(work_dir)}/")
            for r in results:
                print(f"    {_dim(r.model_name)}/")

        # ── Report ───────────────────────────────────────────────────────── #
        print_report(results, verbose=args.verbose)

        return 0 if all(r.ok for r in results) else 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())

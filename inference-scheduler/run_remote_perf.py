#!/usr/bin/env python3
"""
run_remote_perf.py — On-device performance benchmark for KV260 hardware kernels.

Generates a self-contained C benchmark project, uploads it to the remote KV260,
builds all four kernel benchmarks once, then runs each test case and reports
latency (ms) and throughput (GB/s for memory-bound kernels, GOps/s for compute).

Test cases cover:
  VectorOPKernel — all 6 ops, sizes 1K–256K, broadcasting variants
  MatmulKernel   — square and non-square matrices, batch, A-broadcast
  ConvKernel     — 1×1 and 3×3 convolutions typical of CNN architectures
  PoolingKernel  — max/avg pool, global pool variants

Usage:
  python run_remote_perf.py --config remote_config.json
  python run_remote_perf.py --config remote_config.json --iters 200 --warmup 20
  python run_remote_perf.py --config remote_config.json --no-cleanup --verbose
  python run_remote_perf.py --config remote_config.json --check-only
  python run_remote_perf.py --config remote_config.json \\
      --kernels VectorOPKernel MatmulKernel
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.remote import (  # noqa: E402
    _green, _red, _yellow, _bold, _dim, _cyan,
    load_config, uio_devices_from_cfg,
    RemoteSession, check_prerequisites,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration — perf-specific additions on top of SHARED_DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

_EXTRA_DEFAULTS: dict = {
    "build": {"timeout": 300},
    "run":   {"timeout": 60},
    "benchmarks": {
        "VectorOPKernel": {"enabled": True, "warmup": 10},
        "MatmulKernel":   {"enabled": True, "warmup": 10},
        "ConvKernel":     {"enabled": True, "warmup": 10},
        "PoolingKernel":  {"enabled": True, "warmup": 10},
    },
}

# Required driver files per kernel (linux variants).
_DRIVER_FILES: Dict[str, List[str]] = {
    "VectorOPKernel": [
        "xvectoropkernel.h", "xvectoropkernel_hw.h",
        "xvectoropkernel.c", "xvectoropkernel_linux.c",
    ],
    "MatmulKernel": [
        "xmatmulkernel.h", "xmatmulkernel_hw.h",
        "xmatmulkernel.c", "xmatmulkernel_linux.c",
    ],
    "ConvKernel": [
        "xconvkernel.h", "xconvkernel_hw.h",
        "xconvkernel.c", "xconvkernel_linux.c",
    ],
    "PoolingKernel": [
        "xpoolingkernel.h", "xpoolingkernel_hw.h",
        "xpoolingkernel.c", "xpoolingkernel_linux.c",
    ],
}

_BENCH_BINARY = {
    "VectorOPKernel": "bench_vectorop",
    "MatmulKernel":   "bench_matmul",
    "ConvKernel":     "bench_conv",
    "PoolingKernel":  "bench_pool",
}


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark case definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchCase:
    kernel: str        # "VectorOPKernel", "MatmulKernel", etc.
    label:  str        # human-readable tag shown in the report
    args:   List[str]  # positional args after instance_name and label
    warmup: int = 10


@dataclass
class BenchResult:
    case:        BenchCase
    ok:          bool
    lat_ms:      float = 0.0
    metric:      float = 0.0   # GB/s or GOps/s
    metric_key:  str   = ""    # "gbs" or "gops"
    raw:         str   = ""


# Field name → positional arg order for each kernel (matches binary's main() parsing)
_CASE_FIELDS: Dict[str, List[str]] = {
    "VectorOPKernel": ["op", "size", "outer", "a_inc", "b_inc", "iters"],
    "MatmulKernel":   ["n", "k", "m", "batch", "a_stride", "b_stride", "b_packed", "iters"],
    "ConvKernel":     ["batch", "in_ch", "in_h", "in_w", "out_ch",
                       "kh", "kw", "stride_h", "stride_w",
                       "dilation_h", "dilation_w", "pad_top", "pad_left",
                       "has_bias", "is_dw", "iters"],
    "PoolingKernel":  ["batch", "channels", "in_h", "in_w",
                       "pool_h", "pool_w", "stride_h", "stride_w",
                       "pad_top", "pad_left", "dil_h", "dil_w",
                       "pool_type", "lp_order", "count_include_pad", "iters"],
}


def _case_from_dict(kernel: str, d: dict, default_warmup: int = 10) -> "BenchCase":
    """Build a BenchCase from a named-param dict using _CASE_FIELDS ordering."""
    fields = _CASE_FIELDS[kernel]
    if kernel == "VectorOPKernel":
        op = int(d["op"])
        if op < 0 or op >= len(_OP_NAMES):
            valid = ", ".join(f"{i}={n}" for i, n in enumerate(_OP_NAMES))
            raise ValueError(
                f"VectorOPKernel case {d.get('label', '?')!r}: "
                f"unsupported op={op} (valid: {valid})"
            )
    # Optional fields default to 0 (e.g. MatmulKernel b_packed, MATMUL_OPTIMISATION §3b).
    args = [str(d.get(f, 0)) for f in fields]
    warmup = int(d.get("warmup", default_warmup))
    return BenchCase(kernel=kernel, label=d["label"], args=args, warmup=warmup)


def _load_cases(cfg: dict, cli_kernels) -> List["BenchCase"]:
    """Load benchmark cases from config, respecting enabled flags and cli_kernels filter.

    Args:
        cfg: merged config dict
        cli_kernels: set of kernel names from --kernels CLI arg, or None for config-driven
    Returns:
        list of BenchCase instances in kernel declaration order
    """
    cases: List[BenchCase] = []
    bench_cfg = cfg.get("benchmarks", {})
    for kernel in _BENCH_BINARY:
        k_cfg = bench_cfg.get(kernel, {})
        if cli_kernels is not None:
            if kernel not in cli_kernels:
                continue
        else:
            if not k_cfg.get("enabled", True):
                continue
        warmup = int(k_cfg.get("warmup", 10))
        raw_cases = k_cfg.get("cases")
        if not raw_cases:
            continue
        for d in raw_cases:
            cases.append(_case_from_dict(kernel, d, warmup))
    return cases


_BENCH_SRC_DIR = Path(__file__).parent / "bench_src"


# ──────────────────────────────────────────────────────────────────────────────
# Project generation and build
# ──────────────────────────────────────────────────────────────────────────────

def _populate_local_drivers(cfg: dict, driver_dir: Path) -> List[str]:
    """Copy local driver files into *driver_dir*. Returns list of missing files."""
    local_cfg  = cfg.get("local", {})
    per_kernel = local_cfg.get("driver_dirs", {})
    single     = local_cfg.get("driver_dir")
    missing: List[str] = []

    if per_kernel:
        for kernel_name, src_dir in per_kernel.items():
            # Normalise key: accept both "PoolKernel" and "PoolingKernel"
            if kernel_name not in _DRIVER_FILES and kernel_name + "ing" in _DRIVER_FILES:
                kernel_name = kernel_name + "ing"
            files = _DRIVER_FILES.get(kernel_name, [])
            for fname in files:
                src = Path(src_dir) / fname
                if src.exists():
                    shutil.copy2(src, driver_dir / fname)
                else:
                    missing.append(f"{kernel_name}/{fname}")
    elif single:
        src_path = Path(single)
        for files in _DRIVER_FILES.values():
            for fname in files:
                src = src_path / fname
                if src.exists():
                    shutil.copy2(src, driver_dir / fname)
                else:
                    missing.append(fname)

    return missing


def _generate_local_project(tmp_root: Path, cfg: dict) -> Path:
    proj = tmp_root / "kv260_perf"
    proj.mkdir(exist_ok=True)
    drv_dir = proj / "driver"
    drv_dir.mkdir(exist_ok=True)
    for src in _BENCH_SRC_DIR.iterdir():
        if src.is_file():
            shutil.copy2(src, proj / src.name)
    missing = _populate_local_drivers(cfg, drv_dir)
    if missing:
        print(f"  {'drivers':<10} → {_yellow('WARN')}  local files missing: "
              f"{', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}")
    return proj


def _copy_remote_drivers(session: RemoteSession, cfg: dict,
                          remote_proj: str) -> Tuple[bool, str]:
    """Copy per-kernel driver files from remote driver dirs into <remote_proj>/driver/."""
    remote_driver_dirs = cfg["remote"].get("driver_dirs", {})
    remote_driver_dir  = cfg["remote"].get("driver_dir")

    dst = f"{remote_proj}/driver"
    missing_all: List[str] = []

    if remote_driver_dirs:
        for kernel_name, kdir in remote_driver_dirs.items():
            files = _DRIVER_FILES.get(kernel_name, [])
            for fname in files:
                _, _, rc = session.exec(
                    f"cp {kdir}/{fname} {dst}/{fname} 2>/dev/null", timeout=10)
                if rc != 0:
                    missing_all.append(f"{kernel_name}/{fname}")
    elif remote_driver_dir:
        for files in _DRIVER_FILES.values():
            for fname in files:
                _, _, rc = session.exec(
                    f"cp {remote_driver_dir}/{fname} {dst}/{fname} 2>/dev/null",
                    timeout=10)
                if rc != 0:
                    missing_all.append(fname)

    if missing_all:
        return False, f"Missing driver files: {', '.join(missing_all)}"
    return True, "Drivers copied."


def build_benchmark_project(session: RemoteSession, cfg: dict,
                             local_proj: Path) -> Tuple[bool, str, str]:
    """Upload and build the benchmark project. Returns (ok, build_dir, log)."""
    work_dir    = cfg["remote"]["work_dir"].rstrip("/")
    remote_proj = f"{work_dir}/kv260_perf"
    build_dir   = f"{remote_proj}/build"

    print(f"  {'upload':<10}", end="", flush=True)
    t0 = time.monotonic()
    session.exec(f"rm -rf {remote_proj}", timeout=30)
    n = session.upload_dir(local_proj, remote_proj)
    dur = time.monotonic() - t0
    print(f"\r  {'upload':<10} → {_green('OK')}  {dur:.1f}s  ({n} files)")

    ok, drv_log = _copy_remote_drivers(session, cfg, remote_proj)
    if not ok:
        print(f"  {'drivers':<10} → {_yellow('WARN')}  {drv_log}")

    print(f"  {'cmake':<10}", end="", flush=True)
    t0 = time.monotonic()
    extra = " ".join(cfg["remote"].get("cmake_args", []))
    cmake_cmd = (f"cmake -S {remote_proj} -B {build_dir} "
                 f"-DCMAKE_BUILD_TYPE=Release {extra} 2>&1")
    out, _, rc = session.exec(cmake_cmd, timeout=cfg["build"]["timeout"])
    dur = time.monotonic() - t0
    if rc != 0:
        print(f"\r  {'cmake':<10} → {_red('FAIL')}  {dur:.1f}s")
        return False, build_dir, out
    print(f"\r  {'cmake':<10} → {_green('OK')}  {dur:.1f}s")

    print(f"  {'make':<10}", end="", flush=True)
    t0 = time.monotonic()
    make_cmd = f"make -C {build_dir} -j{cfg['build']['jobs']} 2>&1"
    out, _, rc = session.exec(make_cmd, timeout=cfg["build"]["timeout"])
    dur = time.monotonic() - t0
    if rc != 0:
        print(f"\r  {'make':<10} → {_red('FAIL')}  {dur:.1f}s")
        return False, build_dir, out
    print(f"\r  {'make':<10} → {_green('OK')}  {dur:.1f}s")

    return True, build_dir, out


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark execution
# ──────────────────────────────────────────────────────────────────────────────

def run_bench_case(session: RemoteSession, build_dir: str,
                   case: BenchCase, cfg: dict,
                   iters_override: Optional[int],
                   warmup_override: Optional[int]) -> BenchResult:
    uio_devices = uio_devices_from_cfg(cfg)
    instance    = uio_devices.get(case.kernel, f"{case.kernel}_0")
    binary      = f"{build_dir}/{_BENCH_BINARY[case.kernel]}"

    args = list(case.args)
    if iters_override is not None:
        args[-1] = str(iters_override)         # last positional arg is iters
    warmup_arg = str(warmup_override) if warmup_override is not None else str(case.warmup)

    prefix = "sudo -n " if cfg["run"]["use_sudo"] else ""
    cmd = f"{prefix}{binary} {instance} {case.label} {' '.join(args)} {warmup_arg}"

    try:
        out, err, rc = session.exec(cmd, timeout=cfg["run"]["timeout"])
    except TimeoutError as exc:
        return BenchResult(case=case, ok=False, raw=str(exc))

    if rc != 0:
        return BenchResult(case=case, ok=False, raw=(out + err).strip())

    try:
        data      = json.loads(out.strip())
        lat_ms    = float(data["lat_ms"])
        mkey      = "gbs" if "gbs" in data else "gops"
        metric    = float(data[mkey])
        return BenchResult(case=case, ok=True,
                           lat_ms=lat_ms, metric=metric,
                           metric_key=mkey, raw=out.strip())
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return BenchResult(case=case, ok=False,
                           raw=f"parse error: {exc}\n{out}{err}")



# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

_OP_NAMES = ["ADD", "SUB", "MUL", "DIV", "RELU", "RELU6"]
_POOL_TYPES = ["MaxPool", "AvgPool", "LpPool"]


def _vop_detail(args: List[str]) -> str:
    op   = int(args[0]) if args else 0
    size = int(args[1]) if len(args) > 1 else 0
    outr = int(args[2]) if len(args) > 2 else 1
    op_s = _OP_NAMES[op] if op < len(_OP_NAMES) else str(op)
    return f"{op_s:<6} size={size:<7} outer={outr}"


def _mm_detail(args: List[str]) -> str:
    n, k, m = (int(args[i]) if len(args) > i else 0 for i in range(3))
    bat     = int(args[3]) if len(args) > 3 else 1
    return f"N={n:<4} K={k:<4} M={m:<4} batch={bat}"


def _conv_detail(args: List[str]) -> str:
    ic = int(args[1]) if len(args) > 1 else 0
    ih = int(args[2]) if len(args) > 2 else 0
    iw = int(args[3]) if len(args) > 3 else 0
    oc = int(args[4]) if len(args) > 4 else 0
    kh = int(args[5]) if len(args) > 5 else 0
    kw = int(args[6]) if len(args) > 6 else 0
    return f"{ic}ch {ih}x{iw}→{oc}ch {kh}x{kw}k"


def _pool_detail(args: List[str]) -> str:
    ch = int(args[1]) if len(args) > 1 else 0
    ih = int(args[2]) if len(args) > 2 else 0
    iw = int(args[3]) if len(args) > 3 else 0
    ph = int(args[4]) if len(args) > 4 else 0
    pw = int(args[5]) if len(args) > 5 else 0
    pt = int(args[12]) if len(args) > 12 else 0
    pt_s = _POOL_TYPES[pt] if pt < len(_POOL_TYPES) else str(pt)
    return f"{pt_s} {ph}x{pw} {ch}ch {ih}x{iw}"


_DETAIL_FN = {
    "VectorOPKernel": _vop_detail,
    "MatmulKernel":   _mm_detail,
    "ConvKernel":     _conv_detail,
    "PoolingKernel":  _pool_detail,
}


def print_report(results: List[BenchResult], verbose: bool = False) -> None:
    # Group by kernel
    by_kernel: Dict[str, List[BenchResult]] = {}
    for r in results:
        by_kernel.setdefault(r.case.kernel, []).append(r)

    print()
    for kernel, rows in by_kernel.items():
        passed = [r for r in rows if r.ok]
        metric_key = passed[0].metric_key if passed else "gbs"
        metric_hdr = "GB/s" if metric_key == "gbs" else "GOps/s"

        lbl_w = max((len(r.case.label) for r in rows), default=12) + 2
        det_w = 30
        print(_bold(f"  {kernel}"))
        print("  " + "─" * (lbl_w + det_w + 30))
        print(f"  {'Label':<{lbl_w}}  {'Parameters':<{det_w}}  {'Lat(ms)':>9}  {metric_hdr:>8}")
        print("  " + "─" * (lbl_w + det_w + 30))

        detail_fn = _DETAIL_FN.get(kernel, lambda a: "")
        for r in rows:
            detail = detail_fn(r.case.args)
            if r.ok:
                lat_s    = f"{r.lat_ms:9.4f}"
                metric_s = f"{r.metric:8.3f}"
                line = (f"  {r.case.label:<{lbl_w}}  {detail:<{det_w}}"
                        f"  {lat_s}  {metric_s}")
            else:
                line = (f"  {_red(r.case.label):<{lbl_w+9}}  {detail:<{det_w}}"
                        f"  {'ERR':>9}  {'---':>8}")
            print(line)
            if verbose and not r.ok:
                for ln in r.raw.splitlines():
                    print(f"      {_dim(ln)}")

        if passed:
            best_lat = min(r.lat_ms  for r in passed)
            peak     = max(r.metric  for r in passed)
            print("  " + "─" * (lbl_w + det_w + 30))
            print(f"  {'peak ' + metric_hdr:>{lbl_w + det_w + 3}}  "
                  f"{'':>9}  {peak:8.3f}")
            print(f"  {'min latency':>{lbl_w + det_w + 3}}  "
                  f"{best_lat:9.4f}  {'':>8}")

        n_fail = sum(1 for r in rows if not r.ok)
        summary = (f"  {_green(str(len(passed))+'/' + str(len(rows))+' OK')}"
                   if n_fail == 0
                   else f"  {_red(str(n_fail)+' FAILED')}")
        print(summary)
        print()

    total   = len(results)
    n_ok    = sum(1 for r in results if r.ok)
    n_fail  = total - n_ok
    overall = _green(f"All {total} cases passed") if n_fail == 0 \
              else _red(f"{n_fail}/{total} cases FAILED")
    print(_bold(f"  ── OVERALL: {overall} ──"))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="On-device performance benchmark for KV260 hardware kernels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            The benchmark project is built once; each test case is a separate
            binary invocation with different parameters.

            Config file: same format as remote_config.json for run_remote_tests.py.
            Required fields: ssh.host.
            Optional: remote.driver_dirs (per-kernel driver paths on remote board),
                      remote.uio_devices (per-kernel UIO sysfs names).

            Example:
              python run_remote_perf.py --config remote_config_matmul.json \\
                  --kernels MatmulKernel --iters 200
        """),
    )
    p.add_argument("--config", metavar="FILE", required=True,
                   help="JSON config file")
    p.add_argument("--kernels", metavar="K", nargs="+",
                   choices=list(_BENCH_BINARY.keys()),
                   help="Kernels to benchmark (default: all four)")
    p.add_argument("--iters", metavar="N", type=int,
                   help="Override iteration count for all test cases")
    p.add_argument("--warmup", metavar="N", type=int,
                   help="Override warmup count for all test cases")
    p.add_argument("--check-only", action="store_true",
                   help="Check SSH + prerequisites then exit")
    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep remote build directory after run")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print error details for failed cases")
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        cfg = load_config(args.config, _EXTRA_DEFAULTS)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cli_kernels = set(args.kernels) if args.kernels else None

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
        print(f"\n{_bold('Remote prerequisites')}")
        prereqs_ok = check_prerequisites(session, cfg)

        if args.check_only:
            print(f"\n{_green('All OK') if prereqs_ok else _red('Issues found')} "
                  f"— check complete.")
            return 0 if prereqs_ok else 1

        if not prereqs_ok:
            print(f"\n{_yellow('Warning')}: some prerequisites are missing — "
                  "build may fail.", file=sys.stderr)

        # Create remote work directory
        session.exec(f"mkdir -p {cfg['remote']['work_dir']}", timeout=10)

        # Generate local benchmark project
        print(f"\n{_bold('Building benchmark project')}")
        tmp_root = Path(tempfile.mkdtemp(prefix="kv260_perf_"))
        try:
            local_proj = _generate_local_project(tmp_root, cfg)
            ok, build_dir, build_log = build_benchmark_project(
                session, cfg, local_proj)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        if not ok:
            print(f"\n{_bold('Build output:')}")
            for line in build_log.splitlines()[-30:]:
                print(f"  {_dim(line)}")
            return 1

        # Select test cases
        try:
            cases = _load_cases(cfg, cli_kernels)
        except (KeyError, ValueError) as exc:
            print(f"\n{_red('config error')}: {exc}", file=sys.stderr)
            return 1
        kernels = {c.kernel for c in cases}
        print(f"\n{_bold('Running benchmarks')} "
              f"({len(cases)} cases across {len(kernels)} kernel(s))\n")

        results: List[BenchResult] = []

        for case in cases:
            print(f"  {_cyan(case.kernel):<30} {case.label:<28}", end="", flush=True)
            t0 = time.monotonic()
            r  = run_bench_case(session, build_dir, case, cfg,
                                args.iters, args.warmup)
            dur = time.monotonic() - t0
            results.append(r)

            if r.ok:
                mkey = r.metric_key.upper()
                print(f" → {_green('OK')}  {r.lat_ms:7.3f} ms  "
                      f"{r.metric:7.3f} {mkey}  "
                      f"{_dim(f'{dur:.1f}s')}")
            else:
                print(f" → {_red('FAIL')}  {_dim(f'{dur:.1f}s')}")
                if args.verbose:
                    for ln in r.raw.splitlines()[:5]:
                        print(f"    {_dim(ln)}")

        # Cleanup
        if not args.no_cleanup:
            work_dir = cfg["remote"]["work_dir"].rstrip("/")
            session.exec(f"rm -rf {work_dir}/kv260_perf", timeout=30)
        else:
            print(f"\n{_dim('Remote build kept at: '+ build_dir)}")

        print_report(results, verbose=args.verbose)
        return 0 if all(r.ok for r in results) else 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())

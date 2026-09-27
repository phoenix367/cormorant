#!/usr/bin/env python3
"""
run_demo.py — single-command MNIST demo: download → generate → deploy.

Stages (each can be skipped via --skip-<stage>):

  1. download   fetch MNIST test split + ONNX models  (scripts/download_assets.py)
  2. generate   run inference-scheduler per model     (scripts/generate_project.py)
  3. deploy     upload, build, and benchmark on KV260 (scripts/deploy_and_run.py)

Usage:
  python3 run_demo.py
  python3 run_demo.py --config my_config.json
  python3 run_demo.py --skip-download           # use cached data + models
  python3 run_demo.py --skip-deploy             # only generate locally
  python3 run_demo.py --models mnist_lenet      # restrict to one model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
SCRIPTS  = DEMO_DIR / "scripts"


def _run(label: str, cmd: list) -> int:
    print(f"\n=== {label} ===")
    print(" ".join(map(str, cmd)))
    return subprocess.call(cmd)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEMO_DIR / "mnist_config.json"))
    p.add_argument("--models", nargs="+",
                   help="restrict generate/deploy stages to these model names")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-deploy",   action="store_true")
    p.add_argument("--force-download", action="store_true",
                   help="re-download even if cached files exist")
    p.add_argument("--check-only", action="store_true",
                   help="run preflight checks for every stage and exit")
    p.add_argument("--profile-layers", action="store_true",
                   help="forward --profile-layers to deploy_and_run.py "
                        "(per-layer wall-clock stats)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    py = sys.executable

    if not args.skip_download:
        cmd = [py, str(SCRIPTS / "download_assets.py"), "--config", args.config]
        if args.force_download:
            cmd.append("--force")
        if args.check_only:
            cmd.append("--check-only")
        rc = _run("download_assets", cmd)
        if rc != 0:
            return rc

    if not args.skip_generate:
        cmd = [py, str(SCRIPTS / "generate_project.py"),
               "--config", args.config]
        if args.models:
            cmd += ["--models", *args.models]
        if args.check_only:
            cmd.append("--check-only")
        rc = _run("generate_project", cmd)
        if rc != 0:
            return rc

    if not args.skip_deploy:
        cmd = [py, str(SCRIPTS / "deploy_and_run.py"), "--config", args.config]
        if args.verbose:
            cmd.append("--verbose")
        if args.check_only:
            cmd.append("--check-only")
        if args.profile_layers:
            cmd.append("--profile-layers")
        rc = _run("deploy_and_run", cmd)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
run_demo.py — single-command camera demo: download → generate → deploy.

Stages (each can be skipped via --skip-<stage>):

  1. download   fetch the ONNX model + ImageNet labels
                (scripts/download_assets.py)
  2. generate   run inference-scheduler on the model
                (scripts/generate_project.py)
  3. deploy     upload + build on the KV260, start the board camera loop,
                and display the annotated frames it streams back
                (scripts/deploy_and_run.py)

Usage:
  python3 run_demo.py
  python3 run_demo.py --config my_config.json
  python3 run_demo.py --skip-download             # reuse cached assets
  python3 run_demo.py --skip-deploy               # only generate locally
  python3 run_demo.py --save-only                 # headless host (no window)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
SCRIPTS  = DEMO_DIR / "scripts"

sys.path.insert(0, str(SCRIPTS))
from _config_help import format_missing_config  # noqa: E402


def _run(label: str, cmd: list) -> int:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(map(str, cmd)))
    return subprocess.call(cmd)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEMO_DIR / "camera_config.json"))
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-deploy",   action="store_true")
    p.add_argument("--force-download", action="store_true",
                   help="re-fetch the ONNX model + labels")
    p.add_argument("--check-only", action="store_true",
                   help="run preflight checks for every stage and exit")
    p.add_argument("--save-only", action="store_true",
                   help="forward --save-only to deploy_and_run.py "
                        "(headless host: save frames instead of a window)")
    args = p.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(format_missing_config(config_path), file=sys.stderr)
        return 2

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
        cmd = [py, str(SCRIPTS / "generate_project.py"), "--config", args.config]
        if args.check_only:
            cmd.append("--check-only")
        rc = _run("generate_project", cmd)
        if rc != 0:
            return rc

    if not args.skip_deploy:
        cmd = [py, str(SCRIPTS / "deploy_and_run.py"), "--config", args.config]
        if args.check_only:
            cmd.append("--check-only")
        if args.save_only:
            cmd.append("--save-only")
        rc = _run("deploy_and_run", cmd)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())

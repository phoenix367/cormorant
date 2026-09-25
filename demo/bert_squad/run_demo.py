#!/usr/bin/env python3
"""
run_demo.py — one-shot BERT-SQuAD demo: prepare_inputs -> generate_project
-> deploy_and_run.

The first two stages are skipped when their outputs already exist (pass
--regenerate to redo them); every other argument is forwarded to
scripts/deploy_and_run.py (e.g. --n 20 --profile-layers, --check-only).

usage:  run_demo.py [--config CFG] [--regenerate] [deploy_and_run options]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
SCRIPTS = DEMO / "scripts"


def stage(name: str, argv) -> None:
    print(f"\n=== {name} ===", flush=True)
    rc = subprocess.call([sys.executable, str(SCRIPTS / f"{name}.py"), *argv])
    if rc != 0:
        sys.exit(rc)


def main() -> int:
    args = sys.argv[1:]
    regen = "--regenerate" in args
    args = [a for a in args if a != "--regenerate"]
    cfg = []
    if "--config" in args:
        i = args.index("--config")
        cfg = args[i:i + 2]
    sys.path.insert(0, str(SCRIPTS))
    from _common import PROJECT_SUMMARY, load_config, preprocessed_dir
    c = load_config(cfg[1] if cfg else None)
    if regen or not (preprocessed_dir(c) / "inputs.bin").exists():
        stage("prepare_inputs", cfg)
    if regen or not PROJECT_SUMMARY.exists():
        stage("generate_project", cfg)
    stage("deploy_and_run", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

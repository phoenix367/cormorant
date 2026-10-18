#!/usr/bin/env python3
# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
download_assets.py — fetch the ONNX model + ImageNet labels for the KV260
camera demo.

Unlike the image_classification demo there is NO image-preprocessing step:
frames come live from the RealSense camera on the board, so this script only
fetches the two static assets, each idempotent and individually skippable:

  1. Download ONNX models from the Google Drive folder via `gdown --folder`.
  2. Download the ImageNet 1000-class index JSON and derive the 1001-class
     label list MobileNetV1 expects (class 0 = "background").

Usage:
  python3 scripts/download_assets.py
  python3 scripts/download_assets.py --force          # re-fetch everything
  python3 scripts/download_assets.py --skip-models    # only labels
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List

DEMO_DIR    = Path(__file__).resolve().parent.parent
ASSETS_DIR  = DEMO_DIR / "assets"
LABELS_DIR  = ASSETS_DIR / "labels"
MODELS_DIR  = ASSETS_DIR / "models"

LABELS_FILE = LABELS_DIR / "imagenet_1001_labels.txt"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _http_download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    _log(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "axi-demo/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=64 * 1024)
    tmp.rename(dst)


def _load_config(path: Path) -> dict:
    if not path.exists():
        from _config_help import missing_config_die
        missing_config_die(path)
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX models (Google Drive)
# ──────────────────────────────────────────────────────────────────────────────

def _gdown_folder(url: str, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"  gdown --folder {url}")
    cmd = [sys.executable, "-m", "gdown", "--folder", url,
           "--output", str(out_dir), "--quiet"]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(
            f"gdown failed (rc={rc}). Install with: pip install gdown\n"
            f"If the folder is rate-limited, run gdown manually and place "
            f"the .onnx files into {out_dir}/"
        )
    return sorted(p for p in out_dir.rglob("*.onnx") if p.is_file())


def download_models(folder_url: str, want: Iterable[str],
                    force: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"ONNX models → {MODELS_DIR}")

    want    = list(want)
    have    = {p.name: p for p in MODELS_DIR.glob("*.onnx")}
    missing = [n for n in want if n not in have]

    if missing or force:
        staging = MODELS_DIR / "_drive"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            for f in _gdown_folder(folder_url, staging):
                dst = MODELS_DIR / f.name
                if dst.exists() and not force:
                    continue
                shutil.copy2(f, dst)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    have = {p.name: p for p in MODELS_DIR.glob("*.onnx")}
    out  = {}
    for n in want:
        if n not in have:
            raise RuntimeError(
                f"model '{n}' not found in {MODELS_DIR} after download. "
                f"Check the Drive folder contents or override the file name "
                f"in camera_config.json → models[].drive_filename."
            )
        out[n] = have[n]
        _log(f"  ✓ {n}  ({have[n].stat().st_size:,} B)")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet 1001-class labels
# ──────────────────────────────────────────────────────────────────────────────

def download_labels(url: str, force: bool = False) -> Path:
    """
    Fetch the standard 1000-class index and emit the 1001-line label file
    MobileNetV1 expects.  Class 0 is the synthetic 'background' category TF
    Slim prepended; classes 1..1000 are the canonical ImageNet classes.
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if LABELS_FILE.exists() and not force:
        n = sum(1 for _ in LABELS_FILE.open())
        _log(f"  cached {LABELS_FILE.name} ({n} lines)")
        return LABELS_FILE

    raw = LABELS_DIR / "imagenet_class_index.json"
    if not raw.exists() or force:
        _http_download(url, raw)
    idx = json.loads(raw.read_text())

    labels = ["background"]
    for i in range(1000):
        entry = idx.get(str(i))
        if not entry or len(entry) < 2:
            raise RuntimeError(f"imagenet_class_index.json: missing entry {i}")
        labels.append(entry[1])
    if len(labels) != 1001:
        raise RuntimeError(f"derived {len(labels)} labels, expected 1001")

    LABELS_FILE.write_text("\n".join(labels) + "\n")
    _log(f"  ✓ {LABELS_FILE.name} (1001 labels)")
    return LABELS_FILE


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _preflight(cfg: dict, *, need_models: bool, need_labels: bool) -> bool:
    ok = True
    if need_models:
        try:
            import gdown    # noqa: F401
        except ImportError:
            _log("error: 'gdown' not installed — pip install -r requirements.txt")
            ok = False
        if not cfg.get("drive", {}).get("folder_url"):
            _log("error: drive.folder_url missing in config")
            ok = False
        if not cfg.get("models"):
            _log("warning: no models listed in config; nothing to download")
    if need_labels and not cfg.get("labels", {}).get("url"):
        _log("error: labels.url missing in config")
        ok = False
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",
                   default=str(DEMO_DIR / "camera_config.json"))
    p.add_argument("--force", action="store_true",
                   help="re-download cached ONNX + labels")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--skip-labels", action="store_true")
    p.add_argument("--check-only", action="store_true",
                   help="validate config + tools and exit without downloading")
    args = p.parse_args(argv)

    cfg = _load_config(Path(args.config))
    if not _preflight(cfg,
                      need_models=not args.skip_models,
                      need_labels=not args.skip_labels):
        return 1
    if args.check_only:
        _log("preflight: ok")
        return 0

    if not args.skip_models:
        models = cfg.get("models", [])
        want   = [m["drive_filename"] for m in models]
        url    = cfg.get("drive", {}).get("folder_url")
        if want and url:
            download_models(url, want, force=args.force)

    if not args.skip_labels:
        url = cfg.get("labels", {}).get("url")
        if url:
            download_labels(url, force=args.force)

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

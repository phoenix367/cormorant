#!/usr/bin/env python3
# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
download_assets.py — fetch MNIST test set + ONNX models for the demo.

Downloads two kinds of assets into demo/mnist/assets/:
  1. The MNIST test split (10 000 images + labels) in IDX format.
     Source: https://ossci-datasets.s3.amazonaws.com/mnist/  (CC mirror of
     LeCun's original files).
  2. ONNX models from a Google Drive folder using `gdown --folder`.

Both are skipped if the target file already exists and has the expected size.

Usage:
  python3 scripts/download_assets.py                     # use mnist_config.json
  python3 scripts/download_assets.py --config my.json
  python3 scripts/download_assets.py --force             # re-download everything
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List

DEMO_DIR   = Path(__file__).resolve().parent.parent
ASSETS_DIR = DEMO_DIR / "assets"
DATA_DIR   = ASSETS_DIR / "data"
MODELS_DIR = ASSETS_DIR / "models"

MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "t10k-images-idx3-ubyte.gz": 1648877,
    "t10k-labels-idx1-ubyte.gz": 4542,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _http_download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    _log(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=64 * 1024)
    tmp.rename(dst)


def _gunzip(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as r, open(dst, "wb") as w:
        shutil.copyfileobj(r, w, length=64 * 1024)


def _verify_idx_images(path: Path) -> int:
    """Return the image count if the IDX file looks valid, else raise."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != 0x00000803:
        raise RuntimeError(f"{path}: bad IDX-3 magic 0x{magic:08x}")
    if (rows, cols) != (28, 28):
        raise RuntimeError(f"{path}: expected 28x28, got {rows}x{cols}")
    return n


def _verify_idx_labels(path: Path) -> int:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
    if magic != 0x00000801:
        raise RuntimeError(f"{path}: bad IDX-1 magic 0x{magic:08x}")
    return n


# ──────────────────────────────────────────────────────────────────────────────
# MNIST test split
# ──────────────────────────────────────────────────────────────────────────────

def download_mnist(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _log("MNIST test split → " + str(DATA_DIR))

    for fname, _ in MNIST_FILES.items():
        gz = DATA_DIR / fname
        if gz.exists() and not force:
            _log(f"  cached {fname} ({gz.stat().st_size} B)")
        else:
            _http_download(f"{MNIST_BASE}/{fname}", gz)

        out = DATA_DIR / fname[: -len(".gz")]
        if out.exists() and not force:
            continue
        _gunzip(gz, out)

    n_img = _verify_idx_images(DATA_DIR / "t10k-images-idx3-ubyte")
    n_lbl = _verify_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte")
    if n_img != n_lbl:
        raise RuntimeError(f"image/label count mismatch: {n_img} vs {n_lbl}")
    _log(f"  ✓ {n_img} test images")


# ──────────────────────────────────────────────────────────────────────────────
# ONNX models from Google Drive
# ──────────────────────────────────────────────────────────────────────────────

def _gdown_folder(url: str, out_dir: Path) -> List[Path]:
    """Download a Drive folder via gdown and return the list of new files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"  gdown --folder {url}")
    cmd = [sys.executable, "-m", "gdown", "--folder", url,
           "--output", str(out_dir), "--quiet"]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(
            f"gdown failed (rc={rc}). Install with: pip install gdown\n"
            f"If the folder is large or rate-limited, run gdown manually and "
            f"place the .onnx files into {out_dir}/"
        )
    return sorted(p for p in out_dir.rglob("*.onnx") if p.is_file())


def download_models(folder_url: str, want: Iterable[str],
                    force: bool = False) -> dict:
    """
    Download ONNX models from the configured Drive folder and return a mapping
    {drive_filename: absolute_path}.  Files already present are kept unless
    force=True.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"ONNX models → {MODELS_DIR}")

    want   = list(want)
    have   = {p.name: p for p in MODELS_DIR.glob("*.onnx")}
    missing = [n for n in want if n not in have]

    if missing or force:
        # Drop the folder into a staging dir to keep gdown's per-folder structure
        # tidy, then move .onnx files up to MODELS_DIR.
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
                f"in mnist_config.json → models[].drive_filename."
            )
        out[n] = have[n]
        _log(f"  ✓ {n}  ({have[n].stat().st_size:,} B)")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — copy mnist_config.json.example "
            f"and edit it."
        )
    with open(path) as f:
        return json.load(f)


def _preflight(cfg: dict, *, need_models: bool) -> bool:
    """Validate inputs needed for the requested download stages."""
    ok = True
    # gdown is only required for the model-download path.
    if need_models:
        try:
            import gdown  # noqa: F401
        except ImportError:
            _log("error: 'gdown' not installed — pip install -r requirements.txt")
            ok = False
        if not cfg.get("drive", {}).get("folder_url"):
            _log("error: drive.folder_url missing in config")
            ok = False
        if not cfg.get("models"):
            _log("warning: no models listed in config; nothing to download")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEMO_DIR / "mnist_config.json"),
                   help="path to mnist_config.json")
    p.add_argument("--force", action="store_true",
                   help="re-download even if files are cached")
    p.add_argument("--skip-models", action="store_true",
                   help="only fetch the MNIST dataset")
    p.add_argument("--skip-data", action="store_true",
                   help="only fetch the ONNX models")
    p.add_argument("--check-only", action="store_true",
                   help="validate config + tools and exit without downloading")
    args = p.parse_args(argv)

    cfg = _load_config(Path(args.config))
    if not _preflight(cfg, need_models=not args.skip_models):
        return 1
    if args.check_only:
        _log("preflight: ok")
        return 0

    if not args.skip_data:
        download_mnist(force=args.force)

    if not args.skip_models:
        models = cfg.get("models", [])
        want   = [m["drive_filename"] for m in models]
        url    = cfg.get("drive", {}).get("folder_url")
        if not want:
            _log("warning: no models listed in config; nothing to download")
        elif not url:
            _log("error: drive.folder_url missing in config")
            return 1
        else:
            download_models(url, want, force=args.force)

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

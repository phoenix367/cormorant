#!/usr/bin/env python3
"""stereo_assets.py — the stereo depth demo's pinned assets (doc/plans/STEREO_PLAN.md).

Standard library only (the demo runs in inference-scheduler/.venv; the study,
stereo_study.py, imports these pins too).

  * the LightStereo-S checkpoint with the StereoAnything weights (Hugging
    Face XiandaGuo/OpenStereo at a pinned revision; the study also fetches the
    SceneFlow and KITTI checkpoints and OpenStereo's code at a pinned commit);
  * Middlebury MiddEval3 at quarter resolution (training pairs with ground
    truth; the test pairs calibrate the exponents) and ETH3D's low-res
    two-view training pairs;
  * the compact formats file the frontend reads (FORMATS, checked in): the
    exponent of every site and of every conv's output channels, from the
    study's calibration (``stereo_study.py calibrate``).

usage: python3 demo/stereo_depth/scripts/stereo_assets.py [--study]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
ASSETS = DEMO / "assets"
CKPT_DIR = ASSETS / "ckpt"
DATA_DIR = ASSETS / "data"
STUDY_DIR = ASSETS / "study"
OPENSTEREO = ASSETS / "OpenStereo"
FORMATS = DEMO / "lightstereo_s_formats.json"

HF_REPO = "XiandaGuo/OpenStereo"
HF_REV = "cac6f81baeb0099522adc72859b1bda46bfcf6e7"
OPENSTEREO_COMMIT = "23d71c92e33ad1f80dfc42bf29f5c6a914d38769"     # branch v2
CKPTS = {   # name: (path in the HF repo, SHA-256)
    "anything-s": ("checkpoint/StereoAnything/StereoAnything-LightStereo_S.pt",
                   "fe62602ffc17cef1a7971d0237bb1ba801c5711bc70485523363ddc9f4b30abc"),
    "sceneflow-s": ("checkpoint/LightStereo/LightStereo-S-SceneFlow-General.pth",
                    "695c91639de3647db293fa48228c085a6ac70916b5081d4eb02536a90ea6c687"),
    "kitti-s": ("checkpoint/LightStereo/LightStereo-S-KITTI.ckpt",
                "3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a"),
}
DEMO_CKPT = "anything-s"
DATASETS = {   # file: (url, SHA-256)
    "MiddEval3-data-Q.zip": ("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip",
                             "a1411f283b523541e0d2e8b3a1dd6618974a0e539aea0122490791a041fffc09"),
    "MiddEval3-GT0-Q.zip": ("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip",
                            "489b6a4015951a3c865aa54c0f37dcb648b5d0efad8019264e22499cef35ca45"),
    "two_view_training.7z": ("https://www.eth3d.net/data/two_view_training.7z",
                             "cb7683a01ba037759f30b5a4f561d302453e20f1f7095cbdc51c8564b218224b"),
    "two_view_training_gt.7z": ("https://www.eth3d.net/data/two_view_training_gt.7z",
                                "e440f7ec4444dfd2af33a44495f325a3aed06762432e2419277596ba7dcb9b40"),
}


def ckpt_path(name: str = DEMO_CKPT) -> Path:
    return CKPT_DIR / Path(CKPTS[name][0]).name


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, sha: str) -> None:
    if not (dst.exists() and sha256(dst) == sha):
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(dst), url], check=True)
    got = sha256(dst)
    if got != sha:
        raise SystemExit(f"{dst}: SHA-256 {got}, expected {sha}")


def fetch(study: bool = False) -> None:
    """The demo's checkpoint and the datasets (SHA-256 checked); ``study``: every
    checkpoint and OpenStereo's code too."""
    for name, (path, sha) in CKPTS.items():
        if study or name == DEMO_CKPT:
            download(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{path}", ckpt_path(name), sha)
            print(f"  {name}: {Path(path).name}")
    for f, (url, sha) in DATASETS.items():
        download(url, DATA_DIR / f, sha)
    if not (DATA_DIR / "MiddEval3" / "trainingQ").exists():
        for f in ("MiddEval3-data-Q.zip", "MiddEval3-GT0-Q.zip"):
            subprocess.run(["unzip", "-q", "-o", f], cwd=DATA_DIR, check=True)
    if not (DATA_DIR / "eth3d" / "delivery_area_1l" / "disp0GT.pfm").exists():
        (DATA_DIR / "eth3d").mkdir(exist_ok=True)
        for f in ("two_view_training.7z", "two_view_training_gt.7z"):
            subprocess.run(["7z", "x", "-y", f"../{f}"], cwd=DATA_DIR / "eth3d", check=True,
                           stdout=subprocess.DEVNULL)
    if study:
        if not OPENSTEREO.exists():
            subprocess.run(["git", "clone", "-q", "-b", "v2", "https://github.com/XiandaGuo/OpenStereo.git",
                            str(OPENSTEREO)], check=True)
        subprocess.run(["git", "-C", str(OPENSTEREO), "checkout", "-q", OPENSTEREO_COMMIT], check=True)
    print(f"  datasets: {len(gt_pairs())} pairs with ground truth")


def gt_pairs():
    """[(set, directory)] — Middlebury MiddEval3 trainingQ, then ETH3D two-view training."""
    out = []
    for s, root in (("middlebury", DATA_DIR / "MiddEval3" / "trainingQ"), ("eth3d", DATA_DIR / "eth3d")):
        if root.exists():
            out += [(s, d) for d in sorted(root.iterdir()) if (d / "disp0GT.pfm").exists()]
    return out


def site_exponent(m: float) -> int:
    """A calibrated range -> its exponent with one bit of headroom, within [0, 15]."""
    return 15 if m <= 0 else max(0, min(15, math.floor(math.log2(16383.0 / m))))


def write_formats(maxes: dict, ckpt: str, calibration: str, path: Path = FORMATS) -> dict:
    """The compact formats file from a calibration's largest |values|."""
    out = {"model": {"repo": HF_REPO, "revision": HF_REV, "file": CKPTS[ckpt][0], "sha256": CKPTS[ckpt][1]},
           "calibration": calibration, "headroom_bits": 1,
           "exponents": {k: site_exponent(v) for k, v in sorted(maxes.items()) if not k.endswith(":c")},
           "channel_exponents": {k[:-2]: [site_exponent(x) for x in v] for k, v in sorted(maxes.items())
                                 if k.endswith(":c")}}
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--study", action="store_true", help="every checkpoint and OpenStereo's code too")
    a = ap.parse_args(argv)
    fetch(a.study)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
fetch_assets.py — download the BERT-SQuAD demo's assets into
demo/bert_squad/assets/ (not in git) and check them:

  model      assets/models/bertsquad-12-simplified.onnx  434 997 241 bytes, md5 f6818d48…
             (bertsquad-12 from the ONNX model zoo, inputs pinned to [1, 256],
             simplified — the file every result in README.md was measured with;
             a copy on Google Drive)
  vocab      assets/vocab.txt       bert-base-uncased WordPiece vocabulary (Hugging Face)
  squad_dev  assets/dev-v1.1.json   SQuAD 1.1 dev set

A file that already exists is kept as it is (another onnxsim version writes a
slightly different model that gives the same results; `--verify` checks the
hashes anyway).  Downloads go to <file>.part, resume after an interruption,
and are renamed into place only when size and md5 match.  `--from-zoo`
builds the model from the ONNX model zoo instead (README.md "Assets", needs
onnxsim) when the Google Drive copy is unavailable.

The demo scripts (run_demo.py, prepare_inputs.py, generate_project.py) and
inference-scheduler/test/test_bert_base.py call ensure() themselves; run this
script to fetch everything ahead of time.  Standard library only.

usage: fetch_assets.py [model|vocab|squad_dev ...] [--dest-dir DIR] [--verify] [--force] [--from-zoo]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent.parent
ASSETS_DIR = DEMO_DIR / "assets"

DRIVE_ID = "1hdlHoD0VaAumbCcbL0y2iOEL9GLdYPY2"
DRIVE_VIEW = f"https://drive.google.com/file/d/{DRIVE_ID}/view"
ZOO_URL = ("https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/"
           "bert-squad/model/bertsquad-12.onnx")
ZOO_SHAPES = ("input_ids:0=1,256", "input_mask:0=1,256", "segment_ids:0=1,256",
              "unique_ids_raw_output___9:0=1")


@dataclass(frozen=True)
class Asset:
    name: str
    rel: str              # path under assets/
    url: str
    size: int
    md5: str
    manual: str           # where a person gets it by hand


ASSETS = {a.name: a for a in (
    Asset("model", "models/bertsquad-12-simplified.onnx",
          f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t",
          434_997_241, "f6818d482d18e703fbd8d7c3b609a98a", DRIVE_VIEW),
    Asset("vocab", "vocab.txt", "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
          231_508, "64800d5d8528ce344256daf115d4965e",
          "https://huggingface.co/bert-base-uncased/blob/main/vocab.txt"),
    Asset("squad_dev", "dev-v1.1.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
          4_854_279, "3e85deb501d4e538b6bc56f786231552", "https://rajpurkar.github.io/SQuAD-explorer/"),
)}


class FetchError(RuntimeError):
    pass


LEGACY_MODEL = REPO_ROOT / "inference-scheduler" / "bertsquad-12-simplified.onnx"


def default_model_path() -> Path:
    """$BERT_SQUAD_MODEL, else assets/models/…, else an older checkout's
    inference-scheduler/bertsquad-12-simplified.onnx if that exists, else
    assets/models/… (where ensure() downloads it)."""
    env = os.environ.get("BERT_SQUAD_MODEL")
    if env:
        return Path(env)
    demo = ASSETS_DIR / ASSETS["model"].rel
    return LEGACY_MODEL if not demo.exists() and LEGACY_MODEL.exists() else demo


def default_assets_dir() -> Path:
    """$BERT_SQUAD_ASSETS (vocab.txt, dev-v1.1.json), else assets/."""
    return Path(os.environ.get("BERT_SQUAD_ASSETS") or ASSETS_DIR)


def ensure_all(model=None, assets_dir=None, quiet: bool = False) -> tuple:
    """(model, assets dir) with the model, vocab.txt and dev-v1.1.json present."""
    model = ensure("model", model or default_model_path(), quiet)
    d = Path(assets_dir) if assets_dir else default_assets_dir()
    for name in ("vocab", "squad_dev"):
        ensure(name, d / ASSETS[name].rel, quiet)
    return model, d


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, part: Path, size: int, quiet: bool, retries: int = 3) -> None:
    """url -> part, resuming what part already holds; raises FetchError."""
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            return
        req = urllib.request.Request(url, headers={"User-Agent": "axi_demo-fetch_assets/1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if "text/html" in (r.headers.get("Content-Type") or ""):
                    raise FetchError(f"{url} returned a web page instead of the file (Google Drive quota, "
                                     f"a changed link or a login wall)")
                if have and r.status != 206:              # no resume support: start over
                    have = 0
                t0, done = time.monotonic(), have
                last = t0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        now = time.monotonic()
                        if not quiet and now - last >= 5:
                            last = now
                            rate = (done - have) / max(now - t0, 1e-6)
                            _log(f"  {part.name}: {done / 2**20:.0f} / {size / 2**20:.0f} MiB "
                                 f"({rate / 2**20:.1f} MiB/s)")
            if part.stat().st_size >= size:
                return
            raise FetchError(f"connection closed at {part.stat().st_size} of {size} bytes")
        except FetchError as e:
            if "web page" in str(e) or attempt == retries:
                raise
            _log(f"  {e}; retrying ({attempt}/{retries})", quiet)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == retries:
                raise FetchError(f"download of {url} failed: {e}") from e
            _log(f"  {e}; retrying ({attempt}/{retries})", quiet)
            time.sleep(2 * attempt)


def download(asset: Asset, dest: Path, quiet: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    _log(f"downloading {asset.name} ({asset.size / 2**20:.1f} MiB) -> {dest}", quiet)
    _download(asset.url, part, asset.size, quiet)
    got_size, got_md5 = part.stat().st_size, md5_of(part)
    if (got_size, got_md5) != (asset.size, asset.md5):
        part.unlink()
        raise FetchError(f"{asset.name}: downloaded {got_size} bytes, md5 {got_md5}; expected "
                         f"{asset.size} bytes, md5 {asset.md5} (removed; get it from {asset.manual})")
    os.replace(part, dest)
    _log(f"  {dest.name}: {got_size} bytes, md5 ok", quiet)
    return dest


def model_from_zoo(dest: Path, quiet: bool = False) -> Path:
    """bertsquad-12 from the ONNX model zoo, simplified with its inputs pinned
    (README.md "Assets"); the md5 depends on the onnx / onnxsim versions."""
    py = REPO_ROOT / "inference-scheduler" / ".venv" / "bin" / "python"
    py = py if py.exists() else Path(sys.executable)
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = dest.with_name("bertsquad-12.onnx")
    if not raw.exists():
        _log(f"downloading bertsquad-12 from the ONNX model zoo -> {raw}", quiet)
        _download(ZOO_URL, raw.with_name(raw.name + ".part"), 435_852_736, quiet)
        os.replace(raw.with_name(raw.name + ".part"), raw)
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".onnx", delete=False) as t:
        tmp = Path(t.name)
    cmd = [str(py), str(REPO_ROOT / "inference-scheduler" / "simplify_onnx.py"), str(raw), "-o", str(tmp)]
    for s in ZOO_SHAPES:
        cmd += ["--input-shape", s]
    _log(f"simplifying: {' '.join(cmd)}", quiet)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"simplify_onnx.py failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)                     # NamedTemporaryFile made it 0600
    os.replace(tmp, dest)
    m = md5_of(dest)
    _log(f"  {dest.name}: {dest.stat().st_size} bytes, md5 {m}"
         + ("" if m == ASSETS["model"].md5 else " (not the reference file: another onnxsim version; "
            "the results are the same)"), quiet)
    return dest


def verify(name: str, path: Path) -> str:
    """'' if path holds the reference file, else what differs."""
    a = ASSETS[name]
    size = path.stat().st_size
    if size != a.size:
        return f"{size} bytes, reference {a.size}"
    m = md5_of(path)
    return "" if m == a.md5 else f"md5 {m}, reference {a.md5}"


def ensure(name: str, dest=None, quiet: bool = False, from_zoo: bool = False) -> Path:
    """The asset's path, downloading it first when missing (FetchError on failure)."""
    a = ASSETS[name]
    dest = Path(dest) if dest else ASSETS_DIR / a.rel
    if dest.exists():
        return dest
    if dest.is_symlink():
        raise FetchError(f"{dest} is a dangling symlink to {os.readlink(dest)}: remove it or fix its target")
    if name == "model" and from_zoo:
        return model_from_zoo(dest, quiet)
    try:
        return download(a, dest, quiet)
    except FetchError as e:
        hint = (f"; or build it from the ONNX model zoo: {Path(__file__).name} model --from-zoo"
                if name == "model" else "")
        raise FetchError(f"{e}\n  get {a.name} by hand from {a.manual} into {dest}{hint}") from e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", metavar="ASSET", help=f"{', '.join(ASSETS)} (default: all)")
    ap.add_argument("--dest-dir", default=None, help=f"instead of {ASSETS_DIR}")
    ap.add_argument("--verify", action="store_true", help="md5 of existing files against the reference")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    ap.add_argument("--from-zoo", action="store_true", help="build the model from the ONNX model zoo")
    args = ap.parse_args(argv)
    bad = [n for n in args.names if n not in ASSETS]
    if bad:
        ap.error(f"unknown asset(s) {bad}; known: {', '.join(ASSETS)}")
    base = Path(args.dest_dir) if args.dest_dir else ASSETS_DIR
    rc = 0
    for name in args.names or list(ASSETS):
        dest = base / ASSETS[name].rel
        try:
            if args.force and (dest.exists() or dest.is_symlink()):
                dest.unlink()
            existed = dest.exists()
            ensure(name, dest, from_zoo=args.from_zoo)
            if existed:
                diff = verify(name, dest) if args.verify else None
                print(f"{name}: {dest} present" + ("" if diff is None else
                                                    ", reference file" if not diff else f", DIFFERS: {diff}"))
                rc |= bool(diff)
            else:
                print(f"{name}: {dest} downloaded")
        except FetchError as e:
            print(f"{name}: FAILED — {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

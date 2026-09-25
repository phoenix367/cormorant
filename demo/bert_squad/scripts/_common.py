"""Shared helpers for the BERT-SQuAD demo scripts: paths, config loading
(with a copy-pasteable bootstrap message when the config is missing), the
study module import, and the preprocessed-input readers.

The tokenizer, SQuAD feature builder, span decoding, SQuAD EM / F1 and the
numpy reference interpreter all live in bert_study.py; the demo imports
them from there rather than keeping a second copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
DEMO_DIR    = SCRIPTS_DIR.parent
REPO_ROOT   = DEMO_DIR.parent.parent
SCHED_DIR   = REPO_ROOT / "inference-scheduler"
CONFIG      = DEMO_DIR / "bert_squad_config.json"
EXAMPLE     = DEMO_DIR / "bert_squad_config.json.example"
BUILD_DIR   = DEMO_DIR / "build"
PROJECT_DIR = BUILD_DIR / "project"
PROJECT_SUMMARY = BUILD_DIR / "project.json"

for _p in (str(SCRIPTS_DIR), str(SCHED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Role -> ONNX tensor name for bertsquad-12 (overridable under "io" in the
# config).  Roles, not C names: the C names come from OnnxGraph.
DEFAULT_IO = {
    "input_ids":      "input_ids:0",
    "segment_ids":    "segment_ids:0",
    "input_mask":     "input_mask:0",
    "unique_ids":     "unique_ids_raw_output___9:0",
    "start_logits":   "unstack:0",
    "end_logits":     "unstack:1",
    "unique_ids_out": "unique_ids:0",
}
INPUT_ROLES  = ("input_ids", "segment_ids", "input_mask", "unique_ids")
OUTPUT_ROLES = ("start_logits", "end_logits", "unique_ids_out")
OPTIONAL_ROLES = ("unique_ids", "unique_ids_out")


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# ── config ───────────────────────────────────────────────────────────────────

def format_missing_config(path: Path) -> str:
    lines = ["", "ERROR: BERT-SQuAD demo config not found.", "",
             f"  Expected at: {path}"]
    if EXAMPLE.exists():
        lines += [
            f"  Example   : {EXAMPLE}", "",
            "To bootstrap the demo:", "",
            f"  cp '{EXAMPLE}' '{path}'",
            f"  $EDITOR '{path}'", "",
            "Required edits before the first run:",
            "  - ssh.host / ssh.user / ssh.key_file  (the KV260)",
            "  - local.driver_dirs.{VectorOPKernel,MatmulKernel} (HLS driver sources:",
            "      make synthesize_vectorop_kv260 synthesize_matmul_kv260 from build/)",
            "  - model  (bertsquad-12-simplified.onnx, see README.md)",
            "", f"See {DEMO_DIR / 'README.md'} for every field.", ""]
    return "\n".join(lines)


def load_config(path: Optional[str] = None) -> dict:
    p = Path(path) if path else CONFIG
    if not p.exists():
        print(format_missing_config(p), file=sys.stderr)
        sys.exit(2)
    with open(p) as f:
        cfg = json.load(f)
    io = dict(DEFAULT_IO)
    io.update({k: v for k, v in (cfg.get("io") or {}).items() if not k.startswith("_")})
    cfg["io"] = io
    return cfg


def demo_path(value: str) -> Path:
    """Resolve a config path: absolute, or relative to demo/bert_squad/."""
    p = Path(os.path.expanduser(value))
    return p if p.is_absolute() else (DEMO_DIR / p)


def model_path(cfg: dict) -> Path:
    return demo_path(cfg.get("model", "assets/models/bertsquad-12-simplified.onnx"))


def preprocessed_dir(cfg: dict) -> Path:
    return demo_path(cfg.get("inputs", {}).get("out_dir", "assets/preprocessed"))


# ── study module ─────────────────────────────────────────────────────────────

def import_study(cfg: dict):
    """Import bert_study with this config's model / asset locations (it
    reads BERT_SQUAD_MODEL / BERT_SQUAD_ASSETS at import time)."""
    inp = cfg.get("inputs", {})
    vocab = demo_path(inp.get("vocab", "assets/vocab.txt"))
    os.environ["BERT_SQUAD_ASSETS"] = str(vocab.parent)
    os.environ["BERT_SQUAD_MODEL"] = str(model_path(cfg))
    import bert_study  # noqa: E402
    bert_study.HERE = str(vocab.parent)
    bert_study.MODEL = str(model_path(cfg))
    return bert_study


# ── preprocessed inputs ──────────────────────────────────────────────────────

def load_features(cfg: dict) -> dict:
    p = preprocessed_dir(cfg) / "features.json"
    if not p.exists():
        log(f"error: {p} not found — run scripts/prepare_inputs.py first")
        sys.exit(1)
    return json.loads(p.read_text())


def load_inputs(cfg: dict, seq: int) -> np.ndarray:
    """inputs.bin -> int16 array [N, 3, seq] (input_ids, segment_ids, input_mask)."""
    p = preprocessed_dir(cfg) / "inputs.bin"
    raw = np.fromfile(p, dtype="<i2")
    if raw.size % (3 * seq):
        raise ValueError(f"{p}: {raw.size} int16 values is not a multiple of 3 x {seq}")
    return raw.reshape(-1, 3, seq)


def feeds_for(io: Dict[str, str], rec: np.ndarray, uid: int) -> Dict[str, np.ndarray]:
    """ONNX feeds (int64, [1, seq]) for one inputs.bin record."""
    f = {io["input_ids"]:   rec[0].astype(np.int64)[None, :],
         io["segment_ids"]: rec[1].astype(np.int64)[None, :],
         io["input_mask"]:  rec[2].astype(np.int64)[None, :]}
    if io.get("unique_ids"):
        f[io["unique_ids"]] = np.array([uid], np.int64)
    return f


def sha1_of_files(paths: List[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(str(p.relative_to(REPO_ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()

#!/usr/bin/env python3
"""
reference.py — host-side reference logits for the BERT-SQuAD demo.

For the examples in assets/preprocessed/inputs.bin computes

  float  bert_study.py's float32 numpy interpreter (matches onnxruntime to
         4e-6) — every example                          [N, 2, seq] float32
  emu    bert_study.py policy "sched": the op-by-op Q8.8 emulation of the
         partition the scheduler generates — every example
                                                        [N, 2, seq] int16 bits
  sim    the scheduler's own simulation (OnnxGraph + CodeGenerator
         ._forward_pass, what test_inference.c's expected values come from)
         — the first K examples, the board's bit-exact reference
                                                        [K, 2, seq] int16 bits

(emu == sim bit for bit is BERT_PLAN gate (b); bert_sched_check.py.)  Each
example's result is cached under build/reference/cache/ keyed by the model
file, the code that computes it and the example's int16 record, so re-runs
are free.  Examples are spread over a process pool (reference.workers).

Writes build/reference/reference.npz.  deploy_and_run.py starts this script
in the background while the board runs.

usage:  reference.py [--config CFG] [--n N] [--k K] [--workers W] [--no-emulation]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from _common import (BUILD_DIR, SCHED_DIR, SCRIPTS_DIR, feeds_for, import_study,
                     load_config, load_features, load_inputs, log, model_path,
                     sha1_of_files)

REF_DIR = BUILD_DIR / "reference"
CACHE = REF_DIR / "cache"

_W: dict = {}          # per-worker state (models loaded lazily)


def _to_bits(v) -> np.ndarray:
    """Q8.8 values (float64 multiples of 1/256) -> raw int16 bits."""
    return np.round(np.asarray(v, np.float64).reshape(-1) * 256.0).astype(np.int16)


def _worker_init(cfg: dict) -> None:
    _W["cfg"] = cfg


def _study():
    if "bs" not in _W:
        bs = import_study(_W["cfg"])
        _W["bs"] = bs
        _W["float"] = bs.Bert(bs.MODEL)
        _W["sched"] = bs.Bert(bs.MODEL, bs.POLS["sched"], base=_W["float"])
    return _W["bs"], _W["float"], _W["sched"]


def _scheduler():
    if "cg" not in _W:
        from src.codegen import CodeGenerator
        from src.graph import OnnxGraph
        m = str(model_path(_W["cfg"]))
        g = OnnxGraph(m, fuse_act=True, s2d_stem=True, fuse_patterns=True)
        _W["cg"] = CodeGenerator(g, model_path=m)
    return _W["cg"]


def _task(kind: str, i: int, rec: np.ndarray, cache_file: str):
    io = _W["cfg"]["io"]
    feeds = feeds_for(io, rec, i)
    so, eo = io["start_logits"], io["end_logits"]
    if kind == "study":
        _bs, fl, sc = _study()
        ef = fl.run(feeds, "float")
        eq = sc.run(feeds, "q88")
        res = {"float": np.stack([np.asarray(ef[so], np.float32).reshape(-1),
                                  np.asarray(ef[eo], np.float32).reshape(-1)]),
               "emu": np.stack([_to_bits(eq[so]), _to_bits(eq[eo])])}
    else:
        cg = _scheduler()
        sim = cg._forward_pass({k: np.asarray(v, np.float64) for k, v in feeds.items()})
        res = {"sim": np.stack([_to_bits(sim[so]), _to_bits(sim[eo])])}
    np.savez(cache_file, **res)
    return kind, i, res


def code_keys(cfg: dict) -> dict:
    m = model_path(cfg)
    st = m.stat()
    model_id = f"{m.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    study = sha1_of_files([SCRIPTS_DIR / "bert_study.py"])
    sched = sha1_of_files(sorted((SCHED_DIR / "src").rglob("*.py")))
    return {"study": hashlib.sha1(f"{model_id}|{study}".encode()).hexdigest()[:16],
            "sim": hashlib.sha1(f"{model_id}|{sched}".encode()).hexdigest()[:16]}


def compute(cfg: dict, n: int, k: int, workers: int, emulation: bool = True) -> dict:
    feats = load_features(cfg)
    seq = feats["seq_len"]
    recs = load_inputs(cfg, seq)[:n]
    n = len(recs)
    k = min(k, n)
    keys = code_keys(cfg)
    CACHE.mkdir(parents=True, exist_ok=True)

    out = {"float": np.zeros((n, 2, seq), np.float32) if emulation else None,
           "emu": np.zeros((n, 2, seq), np.int16) if emulation else None,
           "sim": np.zeros((k, 2, seq), np.int16)}
    todo = []
    for kind, count in (("sim", k), ("study", n if emulation else 0)):
        for i in range(count):
            h = hashlib.sha1(recs[i].tobytes() + str(i).encode()).hexdigest()[:16]
            cf_path = CACHE / f"{kind}_{keys[kind]}_{h}.npz"
            if cf_path.exists():
                with np.load(cf_path) as z:
                    for key in z.files:
                        out[key][i] = z[key]
            else:
                todo.append((kind, i, recs[i], str(cf_path)))
    t0 = time.time()
    if todo:
        log(f"reference: {len(todo)} to compute ({sum(t[0] == 'sim' for t in todo)} sim, "
            f"{sum(t[0] == 'study' for t in todo)} float+emu), {workers} worker(s)")
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx,
                                    initializer=_worker_init, initargs=(cfg,)) as ex:
            futs = [ex.submit(_task, *t) for t in todo]
            for done, f in enumerate(cf.as_completed(futs), 1):
                kind, i, res = f.result()
                for key, v in res.items():
                    out[key][i] = v
                log(f"reference: [{done}/{len(todo)}] {kind} #{i}  {time.time() - t0:.0f} s")
    else:
        log("reference: all results cached")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(REF_DIR / "reference.npz",
             **{key: v for key, v in out.items() if v is not None},
             n=n, k=k, inputs_sha1=feats["inputs_sha1"])
    (REF_DIR / "reference.json").write_text(json.dumps(
        {"n": n, "k": k, "emulation": emulation, "keys": keys,
         "inputs_sha1": feats["inputs_sha1"], "seconds": round(time.time() - t0, 1)}))
    return out


def load_reference(expect_sha1: str):
    p = REF_DIR / "reference.npz"
    if not p.exists():
        return None
    z = np.load(p)
    if str(z["inputs_sha1"]) != expect_sha1:
        return None
    return {key: z[key] for key in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=0, help="examples (0 = all in inputs.bin)")
    ap.add_argument("--k", type=int, default=None, help="scheduler-simulation examples")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-emulation", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rc = cfg.get("reference", {})
    n = args.n or 10 ** 9
    k = args.k if args.k is not None else int(rc.get("bitexact_examples", 3))
    workers = args.workers or int(rc.get("workers", 2))
    emulation = bool(rc.get("emulation", True)) and not args.no_emulation
    compute(cfg, n, k, workers, emulation)
    return 0


if __name__ == "__main__":
    sys.exit(main())

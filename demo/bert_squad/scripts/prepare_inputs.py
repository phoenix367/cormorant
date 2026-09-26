#!/usr/bin/env python3
"""
prepare_inputs.py — tokenize SQuAD 1.1 dev questions for the KV260 BERT demo.

Takes the first N questions (config inputs.num_examples, default 50) of
SQuAD 1.1 dev whose question + context fit one 256-token window, tokenizes
them with the WordPiece tokenizer and feature builder of squad_text.py
(through bert_study.py; imported, not copied), and writes

  assets/preprocessed/inputs.bin     int16 little-endian, per example:
                                     input_ids[256], segment_ids[256],
                                     input_mask[256]   (1536 bytes)
  assets/preprocessed/features.json  per example: qid, title, question,
                                     gold answers, ctx_off, n_ctx,
                                     sub_to_orig, doc tokens — everything
                                     the span decoder needs.

inputs.selection = "study" instead picks the study's example set
(bert_study.pick_examples: N drawn with seed 0 from the first 4N that fit).

usage:  prepare_inputs.py [--config CFG] [--n N] [--selection first|study]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

from _common import import_study, load_config, log, preprocessed_dir, demo_path


def select_first(bs, tok, squad_path, n):
    data = json.load(open(squad_path, encoding="utf-8"))["data"]
    out = []
    for art in data:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                f = bs.build_feature(tok, qa["question"], para["context"])
                if f is not None:
                    out.append((art["title"], qa, f))
                    if len(out) == n:
                        return out
    return out


def select_study(bs, n):
    _tok, pick = bs.pick_examples(n)
    return [("", qa, f) for qa, f in pick]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None, help="override inputs.num_examples")
    ap.add_argument("--selection", choices=("first", "study"), default=None,
                    help="override inputs.selection")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    inp = cfg.get("inputs", {})
    n = args.n or int(inp.get("num_examples", 50))
    selection = args.selection or inp.get("selection", "first")
    vocab = demo_path(inp.get("vocab", "assets/vocab.txt"))
    squad = demo_path(inp.get("squad_dev", "assets/dev-v1.1.json"))
    for p in (vocab, squad):
        if not p.exists():
            log(f"error: {p} not found — see README.md 'Assets' for the download commands")
            return 1
    if selection == "study" and squad.parent != vocab.parent:
        log("error: selection 'study' needs vocab.txt and dev-v1.1.json in one directory")
        return 1

    bs = import_study(cfg)
    tok = bs.Tokenizer(str(vocab))
    picked = select_first(bs, tok, squad, n) if selection == "first" else select_study(bs, n)
    if len(picked) < n:
        log(f"warning: only {len(picked)} single-window questions available")
    seq = bs.SEQ

    records, feats = [], []
    for i, (title, qa, f) in enumerate(picked):
        rec = np.stack([f["ids"][0], f["seg"][0], f["mask"][0]])
        if rec.min() < -32768 or rec.max() > 32767:
            raise ValueError(f"example {i}: token id does not fit int16")
        records.append(rec.astype("<i2"))
        feats.append({
            "index":       i,
            "qid":         qa["id"],
            "title":       title,
            "question":    qa["question"],
            "answers":     [a["text"] for a in qa["answers"]],
            "n_tokens":    int(f["mask"].sum()),
            "ctx_off":     int(f["ctx_off"]),
            "n_ctx":       int(f["n_ctx"]),
            "sub_to_orig": [int(v) for v in f["sub_to_orig"]],
            "doc":         f["doc"],
        })

    out_dir = preprocessed_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = np.stack(records).tobytes()
    (out_dir / "inputs.bin").write_bytes(blob)
    meta = {
        "seq_len":     seq,
        "n":           len(feats),
        "selection":   selection,
        "record":      ["input_ids", "segment_ids", "input_mask"],
        "dtype":       "int16 little-endian",
        "inputs_sha1": hashlib.sha1(blob).hexdigest(),
        "vocab_sha1":  hashlib.sha1(vocab.read_bytes()).hexdigest(),
        "examples":    feats,
    }
    (out_dir / "features.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    ntok = [x["n_tokens"] for x in feats]
    log(f"wrote {len(feats)} examples ({selection}) -> {out_dir}/inputs.bin "
        f"({len(blob):,} B) + features.json; tokens per window "
        f"min {min(ntok)} / mean {np.mean(ntok):.0f} / max {max(ntok)} of {seq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

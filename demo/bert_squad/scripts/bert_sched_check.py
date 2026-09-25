#!/usr/bin/env python3
"""BERT_PLAN phase 1, gate (b): the inference scheduler's fixed-point
simulation of bertsquad-12 must equal the study's independent emulation of
the same partition (bert_study.py policy "sched") BIT FOR BIT.

The scheduler side is OnnxGraph (CLI defaults: pattern fusion, activation
fusion) + CodeGenerator.simulate(), i.e. exactly what the generated
test_inference.c expectations come from; its host ops mirror the generated C
(double arithmetic, round half to even).  The study side interprets the
original ONNX graph op by op in numpy (LayerNorm and GELU as their 12 / 8
separate nodes) with its own region / quantisation rules.

usage:  inference-scheduler/.venv/bin/python demo/bert_squad/scripts/bert_sched_check.py [--n 3]
        (BERT_SQUAD_MODEL / BERT_SQUAD_ASSETS override the model / asset paths)
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))

import bert_study as bs                                  # noqa: E402
from src.graph import OnnxGraph                          # noqa: E402
from src.codegen import CodeGenerator                    # noqa: E402

OUTS = ("unstack:0", "unstack:1")


def scheduler_sim(model_path):
    g = OnnxGraph(model_path, fuse_act=True, s2d_stem=True, fuse_patterns=True)
    return g, CodeGenerator(g, model_path=model_path)


def compare(cg, bert, feeds):
    """Returns ({out: (n_diff, max_lsb)}, sched logits, study logits,
    (tensors compared, tensors differing)) — the last over every node output
    both sides materialise (every DDR tensor of the schedule)."""
    sim = cg._forward_pass({k: np.asarray(v, np.float64) for k, v in feeds.items()})
    env = bert.run(feeds, "q88")
    produced = {sn.output.onnx_name for sn in cg._graph.nodes}
    common = [n for n in produced if n in env and n in sim]
    n_bad = sum(int(not np.array_equal(np.asarray(sim[n], np.float64).reshape(-1),
                                       np.asarray(env[n], np.float64).reshape(-1)))
                for n in common)
    res, a_out, b_out = {}, {}, {}
    for o in OUTS:
        a = np.asarray(sim[o], np.float64).reshape(-1)
        b = np.asarray(env[o], np.float64).reshape(-1)
        res[o] = (int(np.count_nonzero(a != b)), float(np.abs(a - b).max() * 256.0))
        a_out[o], b_out[o] = a, b
    return res, a_out, b_out, (len(common), n_bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="SQuAD examples (study selection)")
    args = ap.parse_args()
    t0 = time.time()
    tok, pick = bs.pick_examples(args.n)
    g, cg = scheduler_sim(bs.MODEL)
    bert = bs.Bert(bs.MODEL, bs.POLS["sched"])
    print(f"setup {time.time() - t0:.0f} s  (scheduler: {len(g.nodes)} nodes, "
          f"fusion {g.fusion_counts})", flush=True)
    bad = 0
    for k, (qa, f) in enumerate(pick):
        res, a, b, (n_cmp, n_bad) = compare(cg, bert, bs.feeds_of(f))
        span_a, txt_a = bs.best_span(f, a["unstack:0"], a["unstack:1"])
        span_b, _ = bs.best_span(f, b["unstack:0"], b["unstack:1"])
        span_a, span_b = tuple(int(v) for v in span_a), tuple(int(v) for v in span_b)
        ok = all(n == 0 for n, _ in res.values()) and n_bad == 0
        bad += not ok
        print(f"[{k}] {'BIT-EXACT' if ok else 'MISMATCH'}  "
              + "  ".join(f"{o}: {n} diff (max {m:.0f} LSB)" for o, (n, m) in res.items())
              + f"  DDR tensors {n_cmp - n_bad}/{n_cmp} identical"
              + f"  span {span_a}{'==' if span_a == span_b else '!='}{span_b} '{txt_a}'"
              + f"  gold {[x['text'] for x in qa['answers']][:2]}  {time.time() - t0:.0f} s",
              flush=True)
    print(f"{args.n - bad}/{args.n} examples bit-exact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

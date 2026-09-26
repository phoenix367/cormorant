#!/usr/bin/env python3
"""Phase-3 gate 2 (doc/CHAT_PLAN.md): the inference scheduler's fixed-point
simulation of SmolLM2-135M must equal the study's emulation of the shipped
policy (llm_study.py pow2+sink+p12+xattn) BIT FOR BIT on the logits — for a
prefill of real chat prompts and greedy decode steps.

The scheduler side is exactly what the generated library computes: the
frontend's entry graphs (src/llama.py) through OnnxGraph + CodeGenerator's
simulation (the source of test_inference.c's expectations), driven like
llm_api.c drives the entries (SimSession: prefill split over the buckets,
head entry, decode entry, one shared KV-cache state).  The study side is
llm_study.Model — its own forward with KV cache — prefilling the whole prompt
at once and decoding one token per call.  Every step's logits must be equal
(both are raw int16 at the logits exponents, the scheduler's converted to
float32 — exact); the greedy tokens follow.

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/llm_sched_check.py
           [--prompts factual,code,multi-turn] [--decode 32] [--buckets 16,64,256]
           [--save out.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_project as lp                                   # noqa: E402
import llm_study as study                                  # noqa: E402

DEFAULT_PROMPTS = "factual,summarise,multi-turn"


def study_model(W, fd, cfg):
    """llm_study.Model of the shipped policy with the formats JSON."""
    sc = study.Cfg.__new__(study.Cfg)
    sc.L, sc.D, sc.H, sc.KV, sc.F, sc.V, sc.HD = cfg.L, cfg.D, cfg.H, cfg.KV, cfg.FF, cfg.V, cfg.HD
    sc.eps, sc.theta, sc.tied = cfg.eps, cfg.theta, cfg.tied
    fmt = {}
    for k, v in fd["exponents"].items():
        c, l = k.split("@")
        fmt[(c, int(l))] = v if isinstance(v, int) else np.asarray(v, np.int64)
    fm = study.Model(W, sc, None)
    return study.Model(W, sc, study.POLICIES[lp.POLICY], fmt, sink_model=fm), sc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--buckets", default=",".join(map(str, lp.BUCKETS)))
    ap.add_argument("--assets", default=None)
    ap.add_argument("--formats", default=None)
    ap.add_argument("--save", default=None, help="write tokens / logits checksums as JSON")
    args = ap.parse_args()
    t0 = time.time()
    buckets = tuple(int(b) for b in args.buckets.split(","))
    ids = lp.tokenize_prompts(args.prompts.split(","))
    cfg, W, fmt, fd = lp.load_model(args.assets, args.formats)
    fe = lp.frontend(cfg, W, fmt)
    cgs = {name: lp.make_codegen(m, name) for name, m in lp.entry_models(fe, buckets).items()}
    sm, sc = study_model(W, fd, cfg)
    print(f"setup {time.time() - t0:.0f} s: entries {sorted(cgs)}, policy {lp.POLICY}, "
          f"weights saturated {sum(sum(cg._graph.weights_saturated.values()) for cg in cgs.values())}",
          flush=True)
    bad, report = 0, {}
    for name, toks in ids.items():
        t1 = time.time()
        assert toks[0] == 1, "chat prompts start with <|im_start|> (the sink)"
        sess = lp.SimSession(cgs, buckets=buckets)
        seq = study.Seq(sc, len(toks) + args.decode + 1)
        a = sess.prefill(toks[1:])
        b = sm.forward([seq], [np.asarray(toks, np.int64)])[0][-1]
        steps, gen = [], []
        for step in range(args.decode + 1):
            eq = np.array_equal(a, b)
            steps.append((eq, float(np.abs(a - b).max())))
            if not eq:
                break
            nxt = int(np.argmax(a))
            gen.append(nxt)
            if step == args.decode:
                break
            a = sess.decode(nxt)
            b = sm.forward([seq], [np.array([nxt])])[0][-1]
        ok = all(e for e, _ in steps) and len(steps) == args.decode + 1
        bad += not ok
        first_bad = next((i for i, (e, _) in enumerate(steps) if not e), None)
        print(f"[{name}] prompt {len(toks)} tokens, {len(steps) - 1} decode steps: "
              f"{'BIT-EXACT' if ok else f'MISMATCH at step {first_bad} (max |diff| {steps[first_bad][1]})'}"
              f"  ({time.time() - t1:.0f} s)  greedy ids {gen[:12]}{'...' if len(gen) > 12 else ''}",
              flush=True)
        report[name] = {"prompt_tokens": len(toks), "generated": gen, "bit_exact": ok}
    print(f"{len(ids) - bad}/{len(ids)} prompts bit-exact ({time.time() - t0:.0f} s)")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

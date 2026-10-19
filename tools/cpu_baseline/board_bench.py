#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
board_bench.py — runs on the KV260 (in the venv cpu_baseline.py makes there):
a model with onnxruntime on the CPU, timed as the FPGA results are.

  bert MODEL inputs.bin OUT.npz [--threads T] [--n N] [--warmup W]
      bertsquad-12 on the BERT demo's inputs.bin records ([N, 3, 256] int16:
      input_ids, segment_ids, input_mask): one warm-up run, then each
      example's run timed; OUT.npz holds the start / end logits [N, 2, 256]
      and the latencies.

  llm MODEL prompts.json OUT.json [--threads T] [-k 32] [-P 16,64,256] [-R 3]
      a SmolLM2 export (KV cache as inputs / outputs), timed as
      demo/chat/src/llm_bench.c times the FPGA library: the <|im_start|>
      sink cached at position 0; per prompt (llm_board.py's ids) a prefill of
      the ids after the sink — a "<name>/turn2" continuation: all its ids,
      after the previous prompt's decode steps — and K greedy steps (argmax,
      first maximum), each timed; then prefills of 16 / 64 / 256 tokens after
      the sink (the first prompt's ids repeated), best of R.  The KV cache
      stays in onnxruntime's OrtValues.

Both print one line "RESULT: {...}" (the summary) on stdout and their
progress on stderr.  CPU execution provider, ORT_ENABLE_ALL, sequential
execution, --threads intra-op threads.
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import onnxruntime as ort


def session(model: str, threads: int):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    return s, (time.perf_counter() - t0) * 1e3


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def bert(a) -> dict:
    seq = 256
    recs = np.fromfile(a.inputs, dtype="<i2").reshape(-1, 3, seq)[:a.n]
    sess, open_ms = session(a.model, a.threads)
    names = {i.name for i in sess.get_inputs()}

    def feeds(rec, uid):
        f = {"input_ids:0": rec[0].astype(np.int64)[None, :],
             "segment_ids:0": rec[1].astype(np.int64)[None, :],
             "input_mask:0": rec[2].astype(np.int64)[None, :]}
        if "unique_ids_raw_output___9:0" in names:
            f["unique_ids_raw_output___9:0"] = np.array([uid], np.int64)
        return f

    outs = ["unstack:0", "unstack:1"]
    for _ in range(a.warmup):
        sess.run(outs, feeds(recs[0], 0))
    lat, logits = [], []
    for i, rec in enumerate(recs):
        t0 = time.perf_counter()
        st, en = sess.run(outs, feeds(rec, i))
        lat.append((time.perf_counter() - t0) * 1e3)
        logits.append(np.stack([st.reshape(-1), en.reshape(-1)]))
        print(f"example {i + 1}/{len(recs)}: {lat[-1]:.1f} ms", file=sys.stderr, flush=True)
    lat = np.asarray(lat)
    np.savez(a.out, logits=np.asarray(logits, np.float32), lat_ms=lat)
    return {"n": len(lat), "open_ms": round(open_ms, 1), "mean_ms": round(float(lat.mean()), 2),
            "p50_ms": round(float(np.median(lat)), 2), "min_ms": round(float(lat.min()), 2),
            "max_ms": round(float(lat.max()), 2)}


def llm(a) -> dict:
    sess, open_ms = session(a.model, a.threads)
    kv_in = [i for i in sess.get_inputs() if i.name.startswith("past_key_values.")]
    outs = ["logits"] + [i.name.replace("past_key_values.", "present.") for i in kv_in]
    ov = ort.OrtValue.ortvalue_from_numpy

    def step(ids, past, pos):
        """ids at positions pos.. on top of past -> (last position's logits, new past)."""
        n = len(ids)
        feed = {"input_ids": ov(np.asarray(ids, np.int64)[None, :]),
                "attention_mask": ov(np.ones((1, pos + n), np.int64)),
                "position_ids": ov(np.arange(pos, pos + n, dtype=np.int64)[None, :])}
        feed.update({i.name: v for i, v in zip(kv_in, past, strict=True)})
        r = sess.run_with_ort_values(outs, feed)
        return r[0].numpy()[0, -1], r[1:]

    prompts = json.load(open(a.prompts))
    names = list(prompts)
    empty = [ov(np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)) for i in kv_in]
    _, sink = step(prompts[names[0]][:1], empty, 0)
    res = {"open_ms": round(open_ms, 1), "prompts": [], "prefill": []}
    dec = []
    past, pos = sink, 1
    for name in names:
        cont = name.endswith("/turn2")
        ids = prompts[name] if cont else prompts[name][1:]
        if not cont:
            past, pos = sink, 1
        pos0 = pos
        t0 = time.perf_counter()
        lg, past = step(ids, past, pos)
        pre_ms = (time.perf_counter() - t0) * 1e3
        pos += len(ids)
        tok = int(np.argmax(lg))
        toks, ms = [tok], []
        for _ in range(a.k):
            t0 = time.perf_counter()
            lg, past = step([tok], past, pos)
            ms.append((time.perf_counter() - t0) * 1e3)
            pos += 1
            tok = int(np.argmax(lg))
            toks.append(tok)
        dec += ms
        res["prompts"].append({"name": name, "n": len(ids), "pos0": pos0, "prefill_ms": round(pre_ms, 2),
                               "toks": toks, "decode_ms_mean": round(float(np.mean(ms)), 2)})
        print(f"{name}: {len(ids)} tokens at {pos0}, prefill {pre_ms:.0f} ms, "
              f"decode {np.mean(ms):.1f} ms / token", file=sys.stderr, flush=True)
    first = prompts[names[0]][1:]
    for n in (int(x) for x in a.P.split(",") if x):
        ids = [first[i % len(first)] for i in range(n)]
        ts = []
        for _ in range(a.R):
            t0 = time.perf_counter()
            step(ids, sink, 1)
            ts.append((time.perf_counter() - t0) * 1e3)
        res["prefill"].append({"n": n, "best_ms": round(min(ts), 2), "mean_ms": round(float(np.mean(ts)), 2)})
        print(f"prefill {n}: best {min(ts):.0f} ms", file=sys.stderr, flush=True)
    res["decode_ms_mean"] = round(float(np.mean(dec)), 2)
    res["decode_steps"] = len(dec)
    with open(a.out, "w") as f:
        json.dump(dict(res, peak_rss_mb=peak_rss_mb()), f)
    return {k: res[k] for k in ("open_ms", "decode_ms_mean", "decode_steps", "prefill")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bert")
    b.add_argument("model")
    b.add_argument("inputs")
    b.add_argument("out")
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--warmup", type=int, default=1)
    m = sub.add_parser("llm")
    m.add_argument("model")
    m.add_argument("prompts")
    m.add_argument("out")
    m.add_argument("-k", type=int, default=32)
    m.add_argument("-P", default="16,64,256")
    m.add_argument("-R", type=int, default=3)
    for p in (b, m):
        p.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    r = bert(a) if a.cmd == "bert" else llm(a)
    r.update(model=os.path.basename(a.model), ort=ort.__version__, threads=a.threads, peak_rss_mb=peak_rss_mb())
    print("RESULT: " + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

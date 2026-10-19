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

  vlm VISION EMBED DECODER prompts.json images.bin OUT.json [...]
      SmolVLM's three exports (the vision encoder with its connector, the
      token embedding, the decoder on inputs_embeds), timed as llm_bench.c
      times the FPGA library with images: per prompt its image (one 512 x 512
      RGB, images.bin as llm_board.py writes it) through the vision encoder
      (llm_image), then the prompt as in llm — its image tokens' embeddings
      replaced by the image features, the embedding lookup inside the timed
      prefill and decode steps.

All print one line "RESULT: {...}" (the summary) on stdout and their
progress on stderr.  CPU execution provider, ORT_ENABLE_ALL, sequential
execution, --threads intra-op threads.
"""

import argparse
import json
import os
import resource
import struct
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


class Decoder:
    """A decoder export with its KV cache as inputs / outputs, on input_ids or
    (with an embedding export) inputs_embeds; the cache in OrtValues."""

    def __init__(self, model: str, threads: int, embed: str = None):
        self.sess, self.open_ms = session(model, threads)
        self.emb = None
        if embed:
            self.emb, ms = session(embed, threads)
            self.open_ms += ms
        self.kv = [i for i in self.sess.get_inputs() if i.name.startswith("past_key_values.")]
        self.outs = ["logits"] + [i.name.replace("past_key_values.", "present.") for i in self.kv]

    def embeds(self, ids, rows=None):
        """inputs_embeds [1, n, D] of ids; rows {position: feature row} replace theirs."""
        e = self.emb.run(None, {"input_ids": np.asarray(ids, np.int64)[None, :]})[0]
        for p, r in (rows or {}).items():
            e[0, p] = r
        return e

    def step(self, ids, past, pos, rows=None):
        """ids at positions pos.. on top of past -> (last position's logits, new past)."""
        ov = ort.OrtValue.ortvalue_from_numpy
        n = len(ids)
        x = ({"inputs_embeds": ov(self.embeds(ids, rows))} if self.emb
             else {"input_ids": ov(np.asarray(ids, np.int64)[None, :])})
        x.update({"attention_mask": ov(np.ones((1, pos + n), np.int64)),
                  "position_ids": ov(np.arange(pos, pos + n, dtype=np.int64)[None, :])})
        x.update({i.name: v for i, v in zip(self.kv, past, strict=True)})
        r = self.sess.run_with_ort_values(self.outs, x)
        return r[0].numpy()[0, -1], r[1:]

    def empty(self):
        return [ort.OrtValue.ortvalue_from_numpy(np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32))
                for i in self.kv]


def generate(dec: Decoder, res: dict, name: str, ids, past, pos, k: int, rows=None):
    """Prefill ids at pos, then k greedy steps -> (past, pos, decode step times)."""
    pos0 = pos
    t0 = time.perf_counter()
    lg, past = dec.step(ids, past, pos, rows)
    pre_ms = (time.perf_counter() - t0) * 1e3
    pos += len(ids)
    tok = int(np.argmax(lg))
    toks, ms = [tok], []
    for _ in range(k):
        t0 = time.perf_counter()
        lg, past = dec.step([tok], past, pos)
        ms.append((time.perf_counter() - t0) * 1e3)
        pos += 1
        tok = int(np.argmax(lg))
        toks.append(tok)
    res["prompts"].append({"name": name, "n": len(ids), "pos0": pos0, "prefill_ms": round(pre_ms, 2),
                           "toks": toks, "decode_ms_mean": round(float(np.mean(ms)), 2)})
    print(f"{name}: {len(ids)} tokens at {pos0}, prefill {pre_ms:.0f} ms, "
          f"decode {np.mean(ms):.1f} ms / token", file=sys.stderr, flush=True)
    return past, pos, ms


def prefill_timing(dec: Decoder, res: dict, sink, first, lens: str, reps: int, rows_of=None):
    """Prefills of n tokens after the sink (first repeated), best of reps."""
    for n in (int(x) for x in lens.split(",") if x):
        ids = [first[i % len(first)] for i in range(n)]
        rows = rows_of(n) if rows_of else None
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            dec.step(ids, sink, 1, rows)
            ts.append((time.perf_counter() - t0) * 1e3)
        res["prefill"].append({"n": n, "best_ms": round(min(ts), 2), "mean_ms": round(float(np.mean(ts)), 2)})
        print(f"prefill {n}: best {min(ts):.0f} ms", file=sys.stderr, flush=True)


def finish(a, res: dict, dec_ms: list) -> dict:
    res["decode_ms_mean"] = round(float(np.mean(dec_ms)), 2)
    res["decode_steps"] = len(dec_ms)
    with open(a.out, "w") as f:
        json.dump(dict(res, peak_rss_mb=peak_rss_mb()), f)
    return {k: res[k] for k in res if k not in ("prompts",)}


def llm(a) -> dict:
    dec = Decoder(a.model, a.threads)
    prompts = json.load(open(a.prompts))
    names = list(prompts)
    _, sink = dec.step(prompts[names[0]][:1], dec.empty(), 0)
    res = {"open_ms": round(dec.open_ms, 1), "prompts": [], "prefill": []}
    dec_ms = []
    past, pos = sink, 1
    for name in names:
        cont = name.endswith("/turn2")
        if not cont:
            past, pos = sink, 1
        past, pos, ms = generate(dec, res, name, prompts[name] if cont else prompts[name][1:], past, pos, a.k)
        dec_ms += ms
    prefill_timing(dec, res, sink, prompts[names[0]][1:], a.P, a.R)
    return finish(a, res, dec_ms)


def vlm(a) -> dict:
    vis, vis_ms = session(a.vision, a.threads)
    dec = Decoder(a.decoder, a.threads, embed=a.embed)
    spec = json.load(open(a.prompts))                      # {"image_token", "prompts": [{"name", "ids"}]}
    with open(a.images, "rb") as f:
        count, side = struct.unpack("<ii", f.read(8))
        pix = np.frombuffer(f.read(), np.uint8).reshape(count, side, side, 3)
    image_token = spec["image_token"]
    res = {"open_ms": round(vis_ms + dec.open_ms, 1), "prompts": [], "prefill": [], "images": []}
    _, sink = dec.step(spec["prompts"][0]["ids"][:1], dec.empty(), 0)
    dec_ms, feats = [], None
    for i, p in enumerate(spec["prompts"]):
        # transformers' rescale + normalize: float32((float32(p / 255) - 0.5) / 0.5), [3, H, W]
        x = ((pix[i].astype(np.float64) * (1 / 255)).astype(np.float32).transpose(2, 0, 1)
             - np.float32(0.5)) / np.float32(0.5)
        t0 = time.perf_counter()
        out = vis.run(None, {"pixel_values": x[None, None],
                             "pixel_attention_mask": np.ones((1, 1, side, side), bool)})[0]
        img_ms = (time.perf_counter() - t0) * 1e3
        feats = out.reshape(-1, out.shape[-1])               # [64, D]: one row per image token
        res["images"].append({"i": i, "ms": round(img_ms, 2)})
        print(f"image {i}: {img_ms:.0f} ms", file=sys.stderr, flush=True)
        ids = p["ids"][1:]
        rows = dict(zip([j for j, t in enumerate(ids) if t == image_token], feats, strict=True))
        _, _, ms = generate(dec, res, p["name"], ids, sink, 1, a.k, rows)
        dec_ms += ms
    first = spec["prompts"][0]["ids"][1:]
    img_pos = [j for j, t in enumerate(first) if t == image_token]

    def rows_of(n):            # the first prompt's ids repeated: its image rows from the last image's features
        return {j: feats[img_pos.index(j % len(first))] for j in range(n) if (j % len(first)) in img_pos}
    prefill_timing(dec, res, sink, first, a.P, a.R, rows_of)
    return finish(a, res, dec_ms)


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
    v = sub.add_parser("vlm")
    v.add_argument("vision")
    v.add_argument("embed")
    v.add_argument("decoder")
    v.add_argument("prompts")
    v.add_argument("images")
    v.add_argument("out")
    for p in (m, v):
        p.add_argument("-k", type=int, default=32)
        p.add_argument("-P", default="16,64,256")
        p.add_argument("-R", type=int, default=3)
    for p in (b, m, v):
        p.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    r = {"bert": bert, "llm": llm, "vlm": vlm}[a.cmd](a)
    r.update(model=os.path.basename(getattr(a, "model", None) or a.decoder), ort=ort.__version__,
             threads=a.threads, peak_rss_mb=peak_rss_mb())
    print("RESULT: " + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

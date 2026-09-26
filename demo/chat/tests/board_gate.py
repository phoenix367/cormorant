#!/usr/bin/env python3
"""
board_gate.py — phase-1 board gate of doc/CHAT_PLAN.md against a running
server (stdlib only; run from the host or the board):

  1. the demo's SQuAD questions (demo/bert_squad/build/results.json, the
     board run of demo/bert_squad) sent through the server — context as
     the system message, question as the user message — must give exactly
     the demo's spans (text and token positions);
  2. a long document (the first --paragraphs paragraphs of the "Super Bowl
     50" article, several 256-token windows) with questions whose answers
     sit in different paragraphs: answer, gold, windows, latency.

usage:  board_gate.py --url http://kv260:8000/v1 [--api-key K] [--n 12] [--paragraphs 10]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(os.path.dirname(HERE))
BERT = os.path.join(DEMO, "bert_squad")
sys.path.insert(0, os.path.join(BERT, "scripts"))
import squad_text  # noqa: E402


def chat(args, messages, stream=False):
    body = {"model": "bert-squad", "messages": messages, "stream": stream}
    req = urllib.request.Request(args.url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if args.api_key:
        req.add_header("Authorization", f"Bearer {args.api_key}")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.load(r)
    return obj, (time.monotonic() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--n", type=int, default=0, help="demo questions (0 = all in results.json)")
    ap.add_argument("--paragraphs", type=int, default=10)
    ap.add_argument("--results", default=os.path.join(BERT, "build", "results.json"))
    ap.add_argument("--squad", default=os.path.join(BERT, "assets", "dev-v1.1.json"))
    args = ap.parse_args()

    data = json.load(open(args.squad, encoding="utf-8"))["data"]
    ctx = {qa["id"]: p["context"] for a in data for p in a["paragraphs"] for qa in p["qas"]}

    # ── 1. same spans as the demo ──
    preds = json.load(open(args.results, encoding="utf-8"))["predictions"]
    preds = preds[:args.n] if args.n else preds
    same, lat, fpga = 0, [], []
    print(f"1. demo questions through the server ({len(preds)}):")
    for i, p in enumerate(preds):
        obj, ms = chat(args, [{"role": "system", "content": ctx[p["qid"]]},
                              {"role": "user", "content": p["question"]}])
        text = obj["choices"][0]["message"]["content"]
        k = obj["kv260"]
        ok = text == p["text"] and k["span"] == p["span"] and k["windows"] == 1
        same += ok
        lat.append(ms)
        fpga.append(k["fpga_ms"])
        print(f"  [{i:2d}] {'SAME' if ok else 'DIFF'}  {text!r:40.40s} demo {p['text']!r:32.32s} "
              f"span {k['span']} demo {p['span']}  {ms:6.0f} ms (FPGA {k['fpga_ms']:.0f})")
    print(f"  => {same}/{len(preds)} identical spans; request latency mean {sum(lat) / len(lat):.0f} ms, "
          f"FPGA per window mean {sum(fpga) / len(fpga):.0f} ms")

    # ── 2. long document ──
    art = data[0]
    paras = art["paragraphs"][:args.paragraphs]
    doc = "\n\n".join(p["context"] for p in paras)
    tok = squad_text.Tokenizer(os.path.join(BERT, "assets", "vocab.txt"))
    n_tok = len(squad_text.tokenize_context(tok, doc)[2])
    picks = [(i, paras[i]["qas"][0]) for i in range(0, len(paras), 2)] + \
            [(i, paras[i]["qas"][1]) for i in range(1, len(paras), 2) if len(paras[i]["qas"]) > 1]
    print(f"\n2. long document: '{art['title']}' paragraphs 0..{len(paras) - 1}, "
          f"{len(doc.split())} words, {n_tok} tokens")
    em = f1 = 0.0
    for i, qa in picks:
        obj, ms = chat(args, [{"role": "system", "content": doc},
                              {"role": "user", "content": qa["question"]}])
        text = obj["choices"][0]["message"]["content"]
        k = obj["kv260"]
        golds = [a["text"] for a in qa["answers"]]
        e = max(squad_text.em(text, g) for g in golds)
        f = max(squad_text.f1(text, g) for g in golds)
        em, f1 = em + e, f1 + f
        per = k["fpga_ms"] / max(k["windows"], 1)
        print(f"  para {i}: {qa['question'][:58]!r}\n"
              f"     -> {text!r}  gold {golds[0]!r}  EM {e:.0f} F1 {f:.2f}  windows {k['windows']}/"
              f"{k['windows_total']} (best in #{k['window']})  conf {k['confidence']:.2f}  "
              f"{ms:.0f} ms, {per:.0f} ms per window")
    print(f"  => EM {100 * em / len(picks):.1f}  F1 {100 * f1 / len(picks):.1f} over {len(picks)} questions")
    return 0 if same == len(preds) else 1


if __name__ == "__main__":
    sys.exit(main())

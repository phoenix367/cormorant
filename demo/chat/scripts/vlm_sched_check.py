#!/usr/bin/env python3
"""
vlm_sched_check.py — gate: the scheduler simulation of SmolVLM-256M's
entries equals the numeric study's emulation bit for bit
(doc/plans/CHAT_PLAN.md §23; vlm_study.py is the specification).

  vision   the image features of each image: the scheduler's simulation of
           the `vision` entry (src/vit.py) against vlm_study.VisionModel under
           the scheduler's vision policy — pow2+p12+vgelu (GELU on VectorOPKernel's
           activation unit, doc/plans/OFFLOAD_PLAN.md §2.1), pow2+p12 without the unit
           (AXI_VECTOROP_ACTIVATIONS=0) — and the shipped formats.
  text     (--text) the logits of each image prompt and 16 greedy decode
           steps: SimSession over the text entries (src/llama.py with the
           image rows) against vlm_study.TextModel under pow2+sink+p12 (decode
           attention on the FPGA, doc/plans/KV_DECODE_PLAN.md; --decode-attn
           host: pow2+sink+p12+mix).

Runs in the scheduler's venv (inference-scheduler/.venv); the pixels come
from .venv-export (Pillow), cached as .npy.

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/vlm_sched_check.py [--images 39769,1268]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_project as lp                                          # noqa: E402  (paths, scheduler imports)
import vlm_study as vs                                            # noqa: E402
from src.codegen import CodeGenerator                             # noqa: E402
from src.graph import OnnxGraph                                   # noqa: E402
from src.llama import Formats, LlamaConfig, LlamaFrontend        # noqa: E402
from src.vectorop_act import enabled as vectorop_activations    # noqa: E402
from src.vit import VisionFormats, VitConfig, VitFrontend, patches  # noqa: E402



def study_formats(path):
    """vlm_study formats JSON -> the emulation's fmt dict {(cls, layer): exps}."""
    d = json.load(open(path))
    out = {}
    for k, v in d["exponents"].items():
        cls, layer = k.rsplit("@", 1)
        out[(cls, int(layer))] = v if isinstance(v, int) else np.asarray(v, np.int64)
    return out


def pixels(ids, cache_dir):
    """uint8 [512][512][3] per image (vlm_study.load_pixels in .venv-export)."""
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for i in ids:
        p = os.path.join(cache_dir, f"pixels_{i:012d}.npy")
        if not os.path.exists(p):
            code = ("import sys, numpy as np; sys.path.insert(0, %r); import vlm_study as v; "
                    "np.save(%r, v.load_pixels(v.image_path(%d)))" % (HERE, p, i))
            subprocess.run([lp.EXPORT_PY, "-c", code], check=True)
        out[i] = np.load(p)
    return out


def prompt_tokens(items, assets, cache):
    """{image id: prompt ids (vlm_study.prompt_ids, incl. <|im_start|>)} —
    tokenized once with .venv-export's tokenizers, cached."""
    if os.path.exists(cache):
        got = json.load(open(cache))
        if all(str(i) in got for i, _ in items):
            return {int(k): v for k, v in got.items()}
    code = ("import json, sys; sys.path.insert(0, %r); import vlm_study as v, llm_study as ls; "
            "t = ls.Tok(%r); json.dump({str(i): [int(x) for x in v.prompt_ids(t, q)] for i, q in %r}, "
            "open(%r, 'w'))" % (HERE, assets, items, cache))
    subprocess.run([lp.EXPORT_PY, "-c", code], check=True)
    return {int(k): v for k, v in json.load(open(cache)).items()}


def sim_ids(toks, vocab):
    """The library's ids for a prompt: after the leading <|im_start|> (the
    sink), the k-th image token -> vocab + k (an image-feature row)."""
    out, k = [], 0
    for t in toks[1:]:
        if t == vs.IMAGE_TOK:
            out.append(vocab + k)
            k += 1
        else:
            out.append(int(t))
    return out


def text_check(args, ids, pix, vm, vcg, fe, tc, Wt, sd, t0):
    import llm_study as ls
    buckets = tuple(int(b) for b in args.buckets.split(","))
    cj = json.load(open(os.path.join(args.assets, "config.json")))
    tcfg = LlamaConfig.from_dict({**cj["text_config"], "tie_word_embeddings": False})
    tpath = os.path.join(sd, f"formats_{vs.TEXT_FORMATS}.json")
    fte = LlamaFrontend(tcfg, Wt, Formats.from_file(tpath, tcfg), ctx=ls.CTX, name="smolvlm",
                        prefill_attn="fpga", decode_attn=args.decode_attn, image_rows=fe.cfg.n_img,
                        image_state=fe.image_state)
    tpol = lp.policy("fpga", args.decode_attn)
    cgs = lp.make_codegens(fte, buckets)
    cgs["vision"] = vcg
    cs = ls.rope_tables(tc, 2 * ls.CTX)
    sm = vs.TextModel(Wt, tc, ls.POLICIES[tpol], study_formats(tpath), cos_sin=cs,
                      sink_model=vs.TextModel(Wt, tc, None, cos_sin=cs))
    items = [(i, vs.EVAL_PROMPTS[vs.EVAL_IDS.index(i) % len(vs.EVAL_PROMPTS)]
              if i in vs.EVAL_IDS else vs.EVAL_PROMPTS[0]) for i in ids]
    toks = prompt_tokens(items, args.assets, os.path.join(sd, "sched_check", "prompt_ids.json"))
    print(f"text entries {sorted(cgs)} ({time.time() - t0:.0f} s), policy {tpol}", flush=True)
    bad = 0
    for i, q in items:
        t1 = time.time()
        sess = lp.SimSession(cgs, buckets=buckets)
        feat = vm.forward(pix[i])
        sess._run("vision", {"vision.patches": patches(pix[i], fe.cfg.P).astype(np.float64)})
        tk = toks[i]
        a = sess.prefill(sim_ids(tk, tcfg.V))
        seq = ls.Seq(tc, len(tk) + args.decode + 2)
        sm.img = feat
        b = sm.forward([seq], [np.asarray(tk, np.int64)])[0][-1]
        sm.img = None
        steps, gen = [], []
        for step in range(args.decode + 1):
            eq = np.array_equal(a, b)
            steps.append((eq, float(np.abs(a - b).max())))
            if not eq:
                break
            nxt = int(np.argmax(a))
            gen.append(nxt)
            if step == args.decode or nxt == vs.EOU:
                break
            a = sess.decode(nxt)
            b = sm.forward([seq], [np.array([nxt])], phase="decode")[0][-1]
        ok = all(e for e, _ in steps)
        bad += not ok
        fb = next((k for k, (e, _) in enumerate(steps) if not e), None)
        print(f"COCO {i} '{q}': prefill {len(tk) - 1} tokens "
              f"({' + '.join(f'{k}/{B}' for k, B in lp.split_prefill(len(tk) - 1, buckets))}), "
              f"{len(steps) - 1} decode steps: "
              + ("BIT-EXACT" if ok else f"MISMATCH at step {fb} (max |diff| {steps[fb][1]})")
              + f"  ({time.time() - t1:.0f} s)  ids {gen[:10]}...", flush=True)
    print(f"text: {len(items) - bad}/{len(items)} prompts bit-exact ({time.time() - t0:.0f} s)")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--assets", default=vs.default_assets())
    ap.add_argument("--images", default="39769,1268")
    ap.add_argument("--matmul-on-conv", default="auto", choices=("auto", "always", "off"))
    ap.add_argument("--text", action="store_true", help="also the text entries' logits")
    ap.add_argument("--decode-attn", choices=("fpga", "host"), default=lp.DECODE_ATTN,
                    help="the text entries' decode attention (fpga: policy pow2+sink+p12)")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--buckets", default=",".join(map(str, lp.BUCKETS)))
    args = ap.parse_args(argv)
    t0 = time.time()
    sd = lp.study.study_dir(args.assets)
    vpath = os.path.join(sd, f"vision_formats_{vs.VISION_SHIPPED}.json")
    tc, vc, scale, Wt, Wv = vs.load_all(args.assets)
    KW = vs.kernel_weights(Wv, vc)
    vpol = vs.vision_policy(vectorop_activations())
    vm = vs.VisionModel(Wv, KW, vc, scale, vs.VPOLICIES[vpol], study_formats(vpath))
    Wh = {"model." + k: v for k, v in Wv.items()}                  # HF names
    cfg = VitConfig.from_file(os.path.join(args.assets, "config.json"))
    fe = VitFrontend(cfg, Wh, VisionFormats.from_file(vpath, cfg))
    model = fe.entry()
    g = OnnxGraph(model, fuse_act=True, s2d_stem=True, matmul_on_conv=args.matmul_on_conv)
    cg = CodeGenerator(g, model_path="vision.onnx")
    st = g.matmul_conv_stats
    print(f"vision entry: {len(g.nodes)} nodes, MatMul on ConvKernel {st['lowered']} / kept {st['kept']}"
          f", policy {vpol} ({time.time() - t0:.0f} s)", flush=True)
    ids = [int(x) for x in args.images.split(",")]
    pix = pixels(ids, os.path.join(sd, "sched_check"))
    ok = 0
    for i in ids:
        t1 = time.time()
        ref = vm.forward(pix[i])
        states = cg.initial_states()
        cg._forward_pass({"vision.patches": patches(pix[i], cfg.P).astype(np.float64)}, states=states)
        got = np.asarray(states[fe.image_state], np.float64)
        same = np.array_equal(got, ref)
        ok += int(same)
        d = np.abs(got - ref)
        print(f"COCO {i}: image features {'BIT-EXACT' if same else 'DIFFER'} "
              f"(max |diff| {d.max():.3g}, {int((d > 0).sum())} of {d.size} differ; "
              f"max |f| {np.abs(ref).max():.1f})  {time.time() - t1:.0f} s", flush=True)
    print(f"vision: {ok}/{len(ids)} bit-exact ({time.time() - t0:.0f} s)")
    bad = len(ids) - ok
    if args.text:
        bad += text_check(args, ids, pix, vm, cg, fe, tc, Wt, sd, t0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

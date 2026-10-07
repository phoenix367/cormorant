#!/usr/bin/env python3
"""
vlm_host_emu.py — the generated SmolVLM project on the HOST (doc/plans/CHAT_PLAN.md
§23): inference.c, llm_api.c and llm_bench.c compiled unchanged against the
software kernels of inference-scheduler/test/host_emu.py (llm_host_emu.build),
image prompts run through the library's C API — llm_image() then
llm_prefill() with the image tokens as vocab + k, greedy llm_decode() — and
every logits vector compared bit for bit with the scheduler simulation of the
same calls (vlm_project.VlmSession), plus a close / re-open.

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/vlm_host_emu.py
           [--project demo/chat/build/llm_project_smolvlm_256m] [--images 39769,1268]
           [--decode 8] [--incoherent]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_board                                                    # noqa: E402
import llm_host_emu                                                 # noqa: E402
import llm_project as lp                                            # noqa: E402
import vlm_project as vp                                            # noqa: E402
import vlm_sched_check as vc                                        # noqa: E402
import vlm_study as vs                                              # noqa: E402

DEFAULT_PROJECT = os.path.join(os.path.dirname(HERE), "build", "llm_project_smolvlm_256m")


def write_inputs(work, lib_ids, pix):
    """prompts.bin (the library ids) and images.bin (one image per prompt)."""
    with open(os.path.join(work, "prompts.bin"), "wb") as f:
        f.write(struct.pack("<i", len(lib_ids)))
        for t in lib_ids:
            f.write(struct.pack("<i", len(t)))
            f.write(np.asarray(t, "<i4").tobytes())
    side = pix[0].shape[0]
    with open(os.path.join(work, "images.bin"), "wb") as f:
        f.write(struct.pack("<ii", len(pix), side))
        for p in pix:
            f.write(np.ascontiguousarray(p, np.uint8).tobytes())


def check(project, items, lib_ids, pix, res, logits_path, decode, log=print):
    """Replay the run's calls on VlmSession; compare every logits vector."""
    summary = json.load(open(os.path.join(project, "project.json")))
    tp, vpth = summary["formats"]
    m = vp.load(summary["assets"], tp, vpth)
    fe_t, fe_v = vp.frontends(m, ctx=summary["context"], name=summary["model"],
                              decode_attn=summary.get("decode_attn", "host"),
                              vsmx=summary.get("vsmx", False))
    cgs = vp.make_codegens(fe_t, fe_v, summary["buckets"])
    raw = np.fromfile(logits_path, "<f4").reshape(-1, m.tcfg.V)
    rep, k = {}, 0
    for pi, (img_id, q) in enumerate(items):
        sess = vp.VlmSession(cgs, m.vcfg.P, ctx=summary["context"], buckets=summary["buckets"])
        sess.image(pix[pi])
        a = sess.prefill(lib_ids[pi])
        toks = [s["tok"] for s in res["prompts"][pi]["steps"]]
        exact, first_bad = 0, None
        for s in range(decode + 1):
            b = raw[k]
            k += 1
            same = np.array_equal(a.astype(np.float32).view(np.uint32), b.view(np.uint32))
            exact += same
            if not same and first_bad is None:
                first_bad = s
            if s < decode:
                a = sess.decode(toks[s])
        rep[f"{img_id}"] = {"prompt": q, "steps": decode + 1, "bit_exact_steps": exact,
                            "first_mismatch": first_bad, "tokens": toks}
        log(f"COCO {img_id}: {exact}/{decode + 1} logits vectors bit-exact"
            + ("" if first_bad is None else f" (first mismatch at step {first_bad})"))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--images", default="39769,1268")
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--work", default=None)
    ap.add_argument("--incoherent", action="store_true")
    args = ap.parse_args(argv)
    project = os.path.abspath(args.project)
    for name in lp_c_sources():                      # the current C sources (llm_image, -I)
        shutil.copy2(os.path.join(os.path.dirname(HERE), "src", name),
                     os.path.join(project, "test", name))
    work = args.work or os.path.join(project, "host_emu")
    os.makedirs(work, exist_ok=True)
    exe = llm_host_emu.build(project, work, args.incoherent)
    summary = json.load(open(os.path.join(project, "project.json")))
    ids = [int(x) for x in args.images.split(",")]
    items = [(i, vs.EVAL_PROMPTS[vs.EVAL_IDS.index(i) % len(vs.EVAL_PROMPTS)]
              if i in vs.EVAL_IDS else vs.EVAL_PROMPTS[0]) for i in ids]
    sd = lp.study.study_dir(summary["assets"])
    toks = vc.prompt_tokens(items, summary["assets"], os.path.join(sd, "sched_check", "prompt_ids.json"))
    pixd = vc.pixels(ids, os.path.join(sd, "sched_check"))
    V = summary["config"]["vocab"]
    image_token = json.load(open(os.path.join(summary["assets"], "config.json")))["image_token_id"]
    lib_ids = [vp.library_ids(toks[i], V, image_token) for i, _ in items]
    pix = [pixd[i] for i, _ in items]
    write_inputs(work, lib_ids, pix)
    t0 = time.time()
    logits = os.path.join(work, "logits_host.bin")
    r = subprocess.run([exe, "-i", "prompts.bin", "-I", "images.bin", "-o", logits,
                        "-k", str(args.decode), "-r"], capture_output=True, text=True, cwd=work,
                       env=dict(os.environ, INFERENCE_HOST_THREADS="4"))
    print(r.stderr[-3000:])
    if r.returncode:
        print(r.stdout[-3000:])
        return 1
    res = llm_board.parse(r.stdout)
    print(f"llm_bench on the host emulation: {time.time() - t0:.0f} s, re-open {res.get('llm_reopen')}",
          flush=True)
    rep = check(project, items, lib_ids, pix, res, logits, args.decode)
    ok = all(v["bit_exact_steps"] == v["steps"] for v in rep.values()) and \
        res.get("llm_reopen", {}).get("identical", False)
    with open(os.path.join(work, "result.json"), "w") as f:
        json.dump({"check": rep, "bench": res}, f, indent=1)
    print("HOST EMULATION: " + ("logits bit-exact with the simulation" if ok else "MISMATCH"))
    return 0 if ok else 1


def lp_c_sources():
    import generate_llm_project as g
    return g.C_SOURCES


if __name__ == "__main__":
    sys.exit(main())

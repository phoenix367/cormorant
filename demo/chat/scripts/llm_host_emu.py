#!/usr/bin/env python3
"""
llm_host_emu.py — run the generated SmolLM2 project on the HOST: the
generated inference.c, llm_api.c and llm_bench.c compiled unchanged against
the software models of MatmulKernel / ConvKernel and the malloc-backed
buffers of inference-scheduler/test/host_emu.py, then every logits vector of
llm_bench (prefill + greedy decode of the chat prompts) compared bit for bit
with the scheduler simulation (llm_project.SimSession, replaying the same
tokens) — phase-3 gate 2's "the generated C on host emulation equals
_simulate", through the library's own prefill split and entry calls.

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/llm_host_emu.py
           [--project demo/chat/build/llm_project] [--prompts factual,summarise,multi-turn]
           [--decode 32]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_board                                                    # noqa: E402
import llm_project as lp                                            # noqa: E402

sys.path.insert(0, os.path.join(lp.SCHED, "test"))
import host_emu                                                     # noqa: E402


def build(project: str, work: str) -> str:
    from src._conv_hw_config import CONV_TILE_IC
    from src._matmul_hw_config import MATMUL_TILE_M
    emu = os.path.join(work, "emu")
    os.makedirs(emu, exist_ok=True)
    for name, text in (("inference_buf_emu.c", host_emu._BUF_EMU), ("emu_common.h", host_emu._COMMON),
                       ("xvectoropkernel.h", host_emu._VOP), ("xmatmulkernel.h", host_emu._MM),
                       ("xconvkernel.h", host_emu._CONV)):
        with open(os.path.join(emu, name), "w") as f:
            f.write(text)
    exe = os.path.join(work, "llm_bench_host")
    cmd = [host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
           "-Wno-unused-function", "-pthread", f"-DEMU_TILE_M={MATMUL_TILE_M}",
           f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}", f'-DINFERENCE_WEIGHTS_DIR="{project}"',
           f'-DLLM_API_WEIGHTS_DIR="{project}"', "-I", os.path.join(project, "include"),
           "-I", os.path.join(project, "test"), "-I", emu,
           os.path.join(project, "src", "inference.c"), os.path.join(emu, "inference_buf_emu.c"),
           os.path.join(project, "test", "llm_api.c"), os.path.join(project, "test", "llm_bench.c"),
           "-lm", "-o", exe]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("compile failed:\n" + r.stdout + r.stderr[-5000:])
    print(f"compiled with -Werror ({time.time() - t0:.0f} s)", flush=True)
    return exe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--project", default=llm_board.DEFAULT_PROJECT)
    ap.add_argument("--prompts", default="factual,summarise,multi-turn")
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--work", default=None)
    ap.add_argument("--study-json", default=None)
    args = ap.parse_args(argv)
    project = os.path.abspath(args.project)
    work = args.work or os.path.join(project, "host_emu")
    os.makedirs(work, exist_ok=True)
    exe = build(project, work)
    ids = lp.tokenize_prompts(args.prompts.split(","))
    prompts = os.path.join(work, "prompts.bin")
    names = llm_board.write_prompts(prompts, ids)
    logits = os.path.join(work, "logits_host.bin")
    t0 = time.time()
    r = subprocess.run([exe, "-i", prompts, "-o", logits, "-k", str(args.decode), "-r"],
                       capture_output=True, text=True, cwd=work,
                       env=dict(os.environ, INFERENCE_HOST_THREADS="4"))
    print(r.stderr[-3000:])
    if r.returncode:
        print(r.stdout[-3000:])
        return 1
    res = llm_board.parse(r.stdout)
    print(f"llm_bench on the host emulation: {time.time() - t0:.0f} s, re-open "
          f"{res.get('llm_reopen')}", flush=True)
    rep = llm_board.check_logits(project, names, ids, res, logits, args.decode, args.study_json)
    ok = all(v["bit_exact_steps"] == v["steps"] for v in rep.values()) and \
        res.get("llm_reopen", {}).get("identical", False)
    with open(os.path.join(work, "result.json"), "w") as f:
        json.dump({"check": rep, "bench": res}, f, indent=1)
    print("HOST EMULATION: " + ("logits bit-exact with the simulation" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

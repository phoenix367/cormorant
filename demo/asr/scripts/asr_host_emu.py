#!/usr/bin/env python3
"""
asr_host_emu.py — run the generated Moonshine project on the HOST: the
generated inference.c, asr_api.c and asr_bench.c compiled unchanged against the
software kernels and the malloc-backed buffers of
inference-scheduler/test/host_emu.py, then every utterance's token ids
compared with the specification (moonshine_q.transcribe_long with the
project's formats, the long ones split at the same quiet points) — the board
gate without the board.

The utterances are LibriSpeech test-clean ones of the study's evaluation set
(asr_study.py fetch): by default the shortest and the ones closest to 5 s and
10 s and the longest (split in two: longer than one encoder pass) — every bucket
of the default ones.

usage: inference-scheduler/.venv/bin/python demo/asr/scripts/asr_host_emu.py
           [--project demo/asr/build/moonshine-tiny_project] [--utts auto | N | UID,UID,...]
           [--incoherent]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))
sys.path.insert(0, os.path.join(REPO, "inference-scheduler", "test"))

import host_emu                                                     # noqa: E402

DEFAULT_PROJECT = os.path.join(REPO, "demo", "asr", "build", "moonshine-tiny_project")


def _emu_sources(work: str) -> str:
    emu = os.path.join(work, "emu")
    os.makedirs(emu, exist_ok=True)
    for name, text in (("inference_buf_emu.c", host_emu.buf_emu_source()), ("emu_common.h", host_emu._COMMON),
                       ("xvectoropkernel.h", host_emu.vop_source()), ("xmatmulkernel.h", host_emu._MM),
                       ("xconvkernel.h", host_emu._CONV), ("xrowopskernel.h", host_emu._ROWOPS)):
        with open(os.path.join(emu, name), "w") as f:
            f.write(text)
    return emu


def build(project: str, work: str, incoherent: bool = False, shared: bool = False) -> str:
    """asr_bench (or with ``shared`` the library) over the host emulation."""
    from src._conv_hw_config import CONV_TILE_IC
    from src._matmul_hw_config import MATMUL_TILE_M
    summary = json.load(open(os.path.join(project, "project.json")))
    emu = _emu_sources(work)
    exe = os.path.join(work, "libasr_host.so" if shared else "asr_bench_host")
    srcs = [os.path.join(project, "src", f) for f in ("inference.c", "inference_rpu.c", "inference_irq.c")]
    cmd = [host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
           "-pthread", f"-DEMU_TILE_M={MATMUL_TILE_M}", *(["-DEMU_INCOHERENT"] if incoherent else []),
           f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}", f'-DINFERENCE_WEIGHTS_DIR="{project}"',
           f'-DASR_API_WEIGHTS_DIR="{project}"', f'-DASR_MODEL_NAME="{summary["model"]}"',
           "-I", os.path.join(project, "include"), "-I", os.path.join(project, "test"), "-I", emu,
           *[p for p in srcs if os.path.exists(p)], os.path.join(emu, "inference_buf_emu.c"),
           os.path.join(project, "test", "asr_api.c"),
           *(["-shared", "-fPIC"] if shared else [os.path.join(project, "test", "asr_bench.c")]),
           "-lm", "-o", exe]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("compile failed:\n" + r.stdout + r.stderr[-5000:])
    print(f"compiled with -Werror ({time.time() - t0:.0f} s)", flush=True)
    return exe


def write_utts(path: str, pcms) -> None:
    """utts.bin of asr_bench: int32 count, then per utterance int32 n and n int16 samples."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(pcms)))
        for x in pcms:
            x = np.asarray(x, "<i2")
            f.write(struct.pack("<i", x.size))
            f.write(x.tobytes())


def read_ids(path: str, count: int) -> list:
    """ids.bin of asr_bench: per utterance int32 count, then the ids."""
    out = []
    with open(path, "rb") as f:
        for _ in range(count):
            (n,) = struct.unpack("<i", f.read(4))
            out.append(np.frombuffer(f.read(4 * n), "<i4").tolist())
    return out


def parse(stdout: str) -> dict:
    """asr_bench's result lines -> {"open", "utts": [...], "reopen", "summary"}."""
    res = {"utts": []}
    for line in stdout.splitlines():
        for key, name in (("ASR_OPEN: ", "open"), ("ASR_UTT: ", "utts"), ("ASR_REOPEN: ", "reopen"),
                          ("ASR_SUMMARY: ", "summary")):
            if line.startswith(key):
                v = json.loads(line[len(key):])
                if name == "utts":
                    res["utts"].append(v)
                else:
                    res[name] = v
    return res


def run_bench(exe: str, work: str, pcms, reps: int = 2, reopen: bool = True, threads: int = 4):
    """(parsed results, ids per utterance) of asr_bench on ``pcms``."""
    utts, ids = os.path.join(work, "utts.bin"), os.path.join(work, "ids_host.bin")
    write_utts(utts, pcms)
    r = subprocess.run([exe, "-i", utts, "-o", ids, "-R", str(reps), *(["-r"] if reopen else [])],
                       capture_output=True, text=True, cwd=work,
                       env=dict(os.environ, INFERENCE_HOST_THREADS=str(threads)))
    if r.returncode:
        raise RuntimeError(f"asr_bench failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return parse(r.stdout), read_ids(ids, len(pcms))


def spec(project: str):
    """(QM with the project's formats, the config dict, the largest bucket) of a
    generated project (project.json: assets, formats)."""
    import asr_study
    import moonshine_np as mn
    import moonshine_q as mq
    summary = json.load(open(os.path.join(project, "project.json")))
    fdoc = json.load(open(summary["formats"]))
    pol = asr_study.POLICIES[fdoc["policy"]]
    cfgd, W64 = mn.load(summary["assets"])
    return mq.QM(cfgd, W64, pol=pol, fmt=mq.load_formats_dict(fdoc)), cfgd, max(summary["buckets"])


def select(which: str):
    """[(uid, reference, int16 samples)] of the evaluation set: ``auto`` the shortest,
    the ones closest to 5 s and 10 s and the longest; N the N shortest; or ids."""
    import asr_study
    utts = asr_study.utterances(asr_study.ASSETS, "evaluation")
    if which == "auto":
        by_len = sorted(utts, key=lambda u: len(u[2]))
        near = [min(by_len, key=lambda u: abs(len(u[2]) - s * 16000)) for s in (5, 10)]
        return [by_len[0], *near, by_len[-1]]
    if which.isdigit():
        return sorted(utts, key=lambda u: len(u[2]))[:int(which)]
    want = which.split(",")
    by_id = {u[0]: u for u in utts}
    return [by_id[w] for w in want]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--utts", default="auto", help="auto, N (the N shortest) or utterance ids")
    ap.add_argument("--work", default=None)
    ap.add_argument("--incoherent", action="store_true",
                    help="separate CPU / DDR buffer copies: every cache sync must be right")
    args = ap.parse_args(argv)
    import moonshine_q as mq
    project = os.path.abspath(args.project)
    work = args.work or os.path.join(project, "host_emu")
    os.makedirs(work, exist_ok=True)
    exe = build(project, work, args.incoherent)
    utts = select(args.utts)
    t0 = time.time()
    res, got = run_bench(exe, work, [u[2] for u in utts])
    print(f"asr_bench on the host emulation: {time.time() - t0:.0f} s, re-open {res.get('reopen')}", flush=True)
    m, cfgd, frames = spec(project)
    ok = bool(res.get("reopen", {}).get("identical", False))
    rep = []
    for (uid, _ref, pcm), ids, r in zip(utts, got, res["utts"], strict=True):
        want = mq.transcribe_long(m, cfgd, pcm, frames)
        same = ids == want
        ok &= same and r["reps_identical"]
        rep.append({"uid": uid, "seconds": len(pcm) / 16000, "parts": r["parts"], "tokens": len(ids),
                    "equal": same, "reps_identical": r["reps_identical"]})
        print(f"  {uid}: {len(pcm) / 16000:.2f} s, {r['parts']} part(s), {len(ids)} tokens: "
              + ("bit-exact" if same else f"MISMATCH (spec {len(want)} tokens)"), flush=True)
    with open(os.path.join(work, "result.json"), "w") as f:
        json.dump({"check": rep, "bench": res}, f, indent=1)
    print("HOST EMULATION: " + ("token ids bit-exact with the specification" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

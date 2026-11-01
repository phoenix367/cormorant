#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
vla_host_emu.py — run the generated SmolVLA project on the HOST: the generated
inference.c, vla_api.c and vla_bench.c compiled unchanged against the software
kernels and the malloc-backed buffers of inference-scheduler/test/host_emu.py,
then every sample's action chunk compared with the specification
(smolvla_q.SmolVLA.sample with the project's formats) — the board gate without
the board.

The samples are the study's evaluation samples (vla_study.py fetch): two
cameras (top, wrist) at 480 x 640, the task's ids, the normalised state, the
seeded noise of vla_study.noise.

usage: inference-scheduler/.venv/bin/python demo/vla/scripts/vla_host_emu.py
           [--project demo/vla/build/smolvla_base_project] [-n 2] [--incoherent]
"""

from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_PROJECT = os.path.join(REPO, "demo", "vla", "build", "smolvla_base_project")
SPEC_PYTHON = os.path.join(REPO, "demo", "vla", "assets", ".venv-lerobot", "bin", "python")
SPEC_SOURCES = ("smolvla_q.py", "vla_study.py")

# The specification runs in the study's venv (numpy, tokenizers: demo/vla/scripts/requirements-study.txt):
# ``ids`` writes every evaluation sample's token ids, ``chunks`` the chunks of the named samples.
SPEC_WORKER = r"""
import json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import smolvla_q as sq, vla_study as st
what, project, out = sys.argv[2], sys.argv[3], sys.argv[4]
proj = json.load(open(os.path.join(project, "project.json")))
assets = proj["assets"]
cfg = sq.VLACfg(st.ckpt_dir(assets), st.vlm_dir(assets))
tok = sq.ls.Tok(st.vlm_dir(assets))
samples = st.load_samples(assets, "evaluation")
if what == "ids":
    json.dump({s.name: [int(i) for i in st.task_ids(tok, s.task, cfg)] for s in samples}, open(out, "w"))
    sys.exit(0)
names = sys.argv[5].split(",")
W = sq.ls.load_safetensors(os.path.join(st.ckpt_dir(assets), "model.safetensors"))
policy, fmt = sq.load_formats(proj["formats"])
spec = sq.SmolVLA(cfg, W, sq.POLICIES[policy], fmt)
for k, s in enumerate(samples):
    if s.name in names:
        a = spec.sample(s.frames, st.task_ids(tok, s.task, cfg), np.asarray(s.state, np.float32),
                        st.noise(cfg, k).astype(np.float32))
        np.save(os.path.join(out, s.name + ".npy"), a.astype(np.float32))
"""


def _spec_dir(project: str) -> str:
    """<project>/reference/<key>: the key covers the formats and the specification's sources."""
    summary = json.load(open(os.path.join(project, "project.json")))
    h = hashlib.sha256(summary.get("formats_sha256", "").encode())
    for f in SPEC_SOURCES:
        h.update(open(os.path.join(HERE, f), "rb").read())
    for f in ("llm_study.py", "vlm_study.py"):
        h.update(open(os.path.join(REPO, "demo", "chat", "scripts", f), "rb").read())
    d = os.path.join(project, "reference", h.hexdigest()[:16])
    os.makedirs(d, exist_ok=True)
    return d


def sample_inputs(project: str, n: int, python: str = SPEC_PYTHON):
    """(the samples, [(frames, ids, state f32, noise f32)]) of the first n evaluation samples."""
    import smolvla_q as sq
    import vla_study as st
    summary = json.load(open(os.path.join(project, "project.json")))
    assets = summary["assets"]
    path = os.path.join(_spec_dir(project), "ids.json")
    if not os.path.exists(path):
        subprocess.run([python, "-c", SPEC_WORKER, HERE, "ids", project, path], check=True)
    ids = json.load(open(path))
    cfg = sq.VLACfg(st.ckpt_dir(assets), st.vlm_dir(assets))
    samples = st.load_samples(assets, "evaluation", n)
    return samples, [(s.frames, ids[s.name], np.asarray(s.state, np.float32), st.noise(cfg, k).astype(np.float32))
                     for k, s in enumerate(samples)]


def reference(project: str, names, python: str = SPEC_PYTHON, jobs: int = 6) -> dict:
    """{sample name: the specification's chunk (float32)}, cached in the spec directory."""
    out = _spec_dir(project)
    todo = [n for n in names if not os.path.exists(os.path.join(out, n + ".npy"))]
    if todo:
        parts = [todo[i::jobs] for i in range(min(jobs, len(todo)))]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        procs = [subprocess.Popen([python, "-c", SPEC_WORKER, HERE, "chunks", project, out, ",".join(p)], env=env)
                 for p in parts]
        if any(p.wait() for p in procs):
            raise SystemExit("the specification failed")
    return {n: np.load(os.path.join(out, n + ".npy")) for n in names}


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
    """vla_bench (or with ``shared`` the library, demo/vla/vla_lib.py's) over the host emulation."""
    from src._conv_hw_config import CONV_TILE_IC
    from src._matmul_hw_config import MATMUL_TILE_M
    emu = _emu_sources(work)
    exe = os.path.join(work, "libsmolvla_host.so" if shared else "vla_bench_host")
    srcs = [os.path.join(project, "src", f) for f in ("inference.c", "inference_rpu.c", "inference_irq.c")]
    cmd = [host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
           "-pthread", f"-DEMU_TILE_M={MATMUL_TILE_M}", *(["-DEMU_INCOHERENT"] if incoherent else []),
           f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}", f'-DINFERENCE_WEIGHTS_DIR="{project}"',
           f'-DVLA_API_WEIGHTS_DIR="{project}"',
           "-I", os.path.join(project, "include"), "-I", os.path.join(project, "test"), "-I", emu,
           *[p for p in srcs if os.path.exists(p)], os.path.join(emu, "inference_buf_emu.c"),
           os.path.join(project, "test", "vla_api.c"),
           *(["-shared", "-fPIC"] if shared else [os.path.join(project, "test", "vla_bench.c")]),
           "-lm", "-o", exe]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("compile failed:\n" + r.stdout + r.stderr[-5000:])
    print(f"compiled with -Werror ({time.time() - t0:.0f} s)", flush=True)
    return exe


def write_samples(path: str, samples) -> None:
    """samples.bin of vla_bench: [(frames (uint8 [h][w][3] each), ids, state (f32), noise (f32))]."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(samples)))
        for frames, ids, state, noise in samples:
            f.write(struct.pack("<i", len(frames)))
            for fr in frames:
                fr = np.ascontiguousarray(fr, np.uint8)
                f.write(struct.pack("<ii", fr.shape[0], fr.shape[1]))
                f.write(fr.tobytes())
            ids = np.asarray(ids, "<i4")
            f.write(struct.pack("<i", ids.size))
            f.write(ids.tobytes())
            st = np.asarray(state, "<f4")
            f.write(struct.pack("<i", st.size))
            f.write(st.tobytes())
            f.write(np.asarray(noise, "<f4").tobytes())


def parse(stdout: str) -> dict:
    res = {"samples": []}
    for line in stdout.splitlines():
        for key, name in (("VLA_OPEN: ", "open"), ("VLA_SAMPLE: ", "samples"), ("VLA_REOPEN: ", "reopen"),
                          ("VLA_SUMMARY: ", "summary")):
            if line.startswith(key):
                v = json.loads(line[len(key):])
                if name == "samples":
                    res["samples"].append(v)
                else:
                    res[name] = v
    return res


def run_bench(exe: str, work: str, samples, shape, reps: int = 2, reopen: bool = True, threads: int = 4):
    """(parsed results, actions [count][chunk][action] float32) of vla_bench."""
    sp, ap = os.path.join(work, "samples.bin"), os.path.join(work, "actions_host.bin")
    write_samples(sp, samples)
    r = subprocess.run([exe, "-i", sp, "-o", ap, "-R", str(reps), *(["-r"] if reopen else [])],
                       capture_output=True, text=True, cwd=work,
                       env=dict(os.environ, INFERENCE_HOST_THREADS=str(threads)))
    if r.returncode:
        raise RuntimeError(f"vla_bench failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    acts = np.fromfile(ap, "<f4").reshape(len(samples), *shape)
    return parse(r.stdout), acts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("-n", type=int, default=2, help="evaluation samples")
    ap.add_argument("--incoherent", action="store_true", help="separate CPU / DDR copies: the syncs move the data")
    ap.add_argument("--work", default=None)
    ap.add_argument("--spec-python", default=SPEC_PYTHON)
    a = ap.parse_args(argv)
    import tempfile
    project = os.path.abspath(a.project)
    proj = json.load(open(os.path.join(project, "project.json")))
    samples, ins = sample_inputs(project, a.n, a.spec_python)
    refs = reference(project, [s.name for s in samples], a.spec_python)
    work = a.work or tempfile.mkdtemp(prefix="vla_host_emu_")
    exe = build(project, work, a.incoherent)
    t0 = time.time()
    res, acts = run_bench(exe, work, ins, (proj["chunk"], proj["max_action"]))
    print(f"host emulation {time.time() - t0:.0f} s; reopen identical {res.get('reopen', {}).get('identical')}")
    bad = 0
    for k, s in enumerate(samples):
        want = refs[s.name]
        same = np.array_equal(acts[k], want)
        bad += not same
        print(f"  {s.name}: actions {'equal to the specification' if same else 'DIFFER'} "
              f"(max |d| {np.abs(acts[k] - want).max():.3g}); reps identical {res['samples'][k]['reps_identical']}")
    print(f"{len(samples) - bad} / {len(samples)} samples bit-exact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
vjepa2_host_emu.py — run the generated V-JEPA 2 project on the HOST: the
generated inference.c, vjepa2_api.c and vjepa2_bench.c compiled unchanged
against the software kernels and the malloc-backed buffers of
inference-scheduler/test/host_emu.py, then every video's logits compared with
the specification's (vjepa2_q.VJEPA2.forward of make_clip with the project's
formats) — the board gate without the board.

The videos are the study's evaluation samples (vjepa2_study.py fetch), decoded
whole (ffmpeg, rgb24) — the library samples the frames, resizes and crops
them itself.  The reference: the study's logits of the planned policy
(assets/study/full/results.json, the same specification and formats; the
project's formats must equal the study's), else the specification run here
(--spec).  The software kernels are slow: one clip of the real model takes
hours.

usage: inference-scheduler/.venv/bin/python demo/video_classification/scripts/vjepa2_host_emu.py
           [--project demo/video_classification/build/vjepa2_project] [-n 1] [--incoherent] [--spec]
           [--reps R] [--reopen]
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

DEFAULT_PROJECT = os.path.join(REPO, "demo", "video_classification", "build", "vjepa2_project")
POLICY = "pow2+p12+vgelu+vsmx+rowops"


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
    """vjepa2_bench (or with ``shared`` the library, vjepa2_lib.py's) over the host emulation."""
    from src._conv_hw_config import CONV_TILE_IC
    from src._matmul_hw_config import MATMUL_TILE_M
    emu = _emu_sources(work)
    exe = os.path.join(work, "libvjepa2_host.so" if shared else "vjepa2_bench_host")
    srcs = [os.path.join(project, "src", f) for f in ("inference.c", "inference_rpu.c", "inference_irq.c")]
    cmd = [host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
           "-pthread", f"-DEMU_TILE_M={MATMUL_TILE_M}", *(["-DEMU_INCOHERENT"] if incoherent else []),
           f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}", f'-DINFERENCE_WEIGHTS_DIR="{project}"',
           f'-DVJEPA2_API_WEIGHTS_DIR="{project}"',
           "-I", os.path.join(project, "include"), "-I", os.path.join(project, "test"), "-I", emu,
           *[p for p in srcs if os.path.exists(p)], os.path.join(emu, "inference_buf_emu.c"),
           os.path.join(project, "test", "vjepa2_api.c"),
           *(["-shared", "-fPIC"] if shared else [os.path.join(project, "test", "vjepa2_bench.c")]),
           "-lm", "-o", exe]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("compile failed:\n" + r.stdout + r.stderr[-5000:])
    print(f"compiled with -Werror ({time.time() - t0:.0f} s)", flush=True)
    return exe


def write_videos(path: str, videos) -> None:
    """videos.bin of vjepa2_bench: [frames uint8 [n][h][w][3]]."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(videos)))
        for v in videos:
            v = np.ascontiguousarray(v, np.uint8)
            f.write(struct.pack("<iii", *v.shape[:3]))
            f.write(v.tobytes())


def parse(stdout: str) -> dict:
    res = {"videos": []}
    for line in stdout.splitlines():
        for key, name in (("VJ_OPEN: ", "open"), ("VJ_VIDEO: ", "videos"), ("VJ_REOPEN: ", "reopen"),
                          ("VJ_SUMMARY: ", "summary")):
            if line.startswith(key):
                v = json.loads(line[len(key):])
                if name == "videos":
                    res["videos"].append(v)
                else:
                    res[name] = v
    return res


def run_bench(exe: str, work: str, videos, classes: int, reps: int = 2, reopen: bool = True, threads: int = 4):
    """(parsed results, logits [count][classes] float32) of vjepa2_bench."""
    vp, lp = os.path.join(work, "videos.bin"), os.path.join(work, "logits_host.bin")
    write_videos(vp, videos)
    r = subprocess.run([exe, "-i", vp, "-o", lp, "-R", str(reps), *(["-r"] if reopen else [])],
                       capture_output=True, text=True, cwd=work,
                       env=dict(os.environ, INFERENCE_HOST_THREADS=str(threads)))
    if r.returncode:
        raise RuntimeError(f"vjepa2_bench failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return parse(r.stdout), np.fromfile(lp, "<f4").reshape(len(videos), classes)


def study_reference(assets: str, formats: str, n: int):
    """[(sample index, the study's logits of the planned policy)] for the first n
    evaluation samples — after checking that the project's formats are the study's."""
    res = json.load(open(os.path.join(assets, "study", "full", "results.json")))
    fp = json.load(open(formats))
    if res["formats"][POLICY]["exponents"] != fp["exponents"]:
        raise SystemExit(f"{formats}: not the study's formats (rerun with --spec)")
    return [(int(i), np.asarray(res["logits"][POLICY][k], np.float64)) for k, i in enumerate(res["samples"][:n])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("-n", type=int, default=1, help="evaluation videos")
    ap.add_argument("--incoherent", action="store_true", help="separate CPU / DDR copies: the syncs move the data")
    ap.add_argument("--spec", action="store_true", help="run the specification here instead of the study's logits")
    ap.add_argument("--reps", type=int, default=1, help="runs per video (identical logits)")
    ap.add_argument("--reopen", action="store_true", help="close, reopen and run the first video again")
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    import tempfile
    import vjepa2_study as st
    project = os.path.abspath(a.project)
    proj = json.load(open(os.path.join(project, "project.json")))
    assets = proj["assets"]
    if a.spec:
        import vjepa2_q as vq
        idx = st.inputs()["samples"]["evaluation"][:a.n]
        cfg, W, KW = st.load_model_inputs(assets)
        fd = json.load(open(proj["formats"]))
        fmt = {(k.split("@")[0], int(k.split("@")[1])): vq.ls._vec_or_scalar(v) for k, v in fd["exponents"].items()}
        spec = vq.VJEPA2(cfg, W, vq.POLICIES[fd["policy"]], fmt, KW=KW)
        refs = [(i, spec.forward(st.Sample(assets, i).clip)) for i in idx]
    else:
        refs = study_reference(assets, proj["formats"], a.n)
    videos = [st.decode_video(os.path.join(assets, "videos", f"{i:04d}.webm")) for i, _ in refs]
    work = a.work or tempfile.mkdtemp(prefix="vjepa2_host_emu_")
    exe = build(project, work, a.incoherent)
    t0 = time.time()
    res, lg = run_bench(exe, work, videos, proj["classes"], reps=a.reps, reopen=a.reopen)
    print(f"host emulation {time.time() - t0:.0f} s; reopen identical {res.get('reopen', {}).get('identical')}")
    bad = 0
    for k, (i, want) in enumerate(refs):
        want32 = np.asarray(want, np.float64).astype(np.float32)
        same = np.array_equal(lg[k], want32)
        bad += not same
        print(f"  ssv2_{i:04d}: logits {'equal to the specification' if same else 'DIFFER'} "
              f"(max |d| {np.abs(lg[k] - want32).max():.3g}, top-1 {int(np.argmax(lg[k]))} / "
              f"{int(np.argmax(want32))}); reps identical {res['videos'][k]['reps_identical']}")
    print(f"{len(refs) - bad} / {len(refs)} videos bit-exact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

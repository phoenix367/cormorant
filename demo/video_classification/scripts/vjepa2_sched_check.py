#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
vjepa2_sched_check.py — the scheduler's simulation of the real V-JEPA 2 ``clip``
entry (src/vjepa2.py over the checkpoint and the formats of vjepa2_study.py
formats) against the specification, bit for bit: the logits of the study's first
n evaluation clips equal (as float32) the study's logits of the planned policy
(assets/study/full/results.json — vjepa2_q.VJEPA2.forward with the same formats;
the formats are checked to be the study's).  Host only; one clip simulates in
a few minutes.  First the entry's predicted time: the event stream replayed
with the bitstream's performance model (inference-scheduler/perf_models/kv260/
<bitstream-id>.json, the host-op model host.json; codegen/timing.simulate).

usage: inference-scheduler/.venv/bin/python demo/video_classification/scripts/vjepa2_sched_check.py
           [--assets demo/video_classification/assets] [--formats JSON] [-n 2] [--perf-model JSON]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))

import generate_vjepa2_project as gvp                               # noqa: E402
import vjepa2_host_emu as he                                        # noqa: E402
import vjepa2_study as st                                           # noqa: E402
from src.codegen import CodeGenerator                               # noqa: E402
from src.graph import OnnxGraph                                     # noqa: E402
from src.vjepa2 import VJepa2Frontend, patches                      # noqa: E402


def predict(cg, path=None) -> None:
    """The entry's time by the performance model (kernel calls, host ops, waits)."""
    from src.codegen.timing import kernel_duration_fn, simulate
    from src.host_model import HostModel, default_host_model_path
    from src.perf_calls import local_bitstream_id
    from src.perf_model import PerfModel
    if path is None:
        path = os.path.join(REPO, "inference-scheduler", "perf_models", "kv260", f"{local_bitstream_id()}.json")
    pm = PerfModel.load(path)
    tl = simulate(cg, kernel_duration_fn(pm, cg._layouts, chains=cg._ro_chains),
                  HostModel.load(default_host_model_path()).us)
    unpriced = {}
    for i in sorted(set(tl.unpriced)):
        k = type(cg._graph.nodes[i]).__name__
        unpriced[k] = unpriced.get(k, 0) + 1
    print(f"predicted: {tl.total_us / 1e3:.0f} ms a clip ({os.path.basename(path)}; CPU waits "
          f"{tl.wait_us / 1e3:.0f} ms; unpriced nodes: {unpriced or 'none'})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--assets", default=gvp.ASSETS)
    ap.add_argument("--formats", default=None)
    ap.add_argument("-n", type=int, default=2, help="evaluation clips (the study's order; 0: the prediction only)")
    ap.add_argument("--perf-model", default=None, help="the performance model (default: the local bitstream's)")
    a = ap.parse_args(argv)
    formats = a.formats or gvp.default_formats()
    t0 = time.time()
    cfg, W, fmt = gvp.load(a.assets, formats)
    fe = VJepa2Frontend(cfg, W, fmt)
    cg = CodeGenerator(OnnxGraph(fe.entry()), model_path="clip.onnx")
    print(f"graph: {len(cg._graph.nodes)} nodes, weights saturated {sum(cg._graph.weights_saturated.values())} "
          f"({time.time() - t0:.0f} s)", flush=True)
    predict(cg, a.perf_model)
    refs = he.study_reference(a.assets, formats, a.n) if a.n else []
    bad = 0
    for i, want in refs:
        t0 = time.time()
        pt = patches(st.Sample(a.assets, i).clip, cfg).astype(np.float64)
        ins, r0 = {}, 0
        for p, (rows, _k) in enumerate(fe.pieces):
            ins[f"clip.patches.p{p}"] = pt[r0:r0 + rows]
            r0 += rows
        o = cg._forward_pass(ins, states=cg.initial_states(), keep=["clip.logits"])
        got = np.asarray(o["clip.logits"], np.float64).reshape(-1)
        want32 = np.asarray(want, np.float64).astype(np.float32).astype(np.float64)
        same = np.array_equal(got, want32)
        bad += not same
        print(f"  ssv2_{i:04d}: logits {'equal to the specification' if same else 'DIFFER'} (max |d| "
              f"{np.abs(got - want32).max():.3g}, top-1 {int(np.argmax(got))} / {int(np.argmax(want32))}; "
              f"{time.time() - t0:.0f} s)", flush=True)
    print(f"{len(refs) - bad}/{len(refs)} clips bit-exact" + ("" if bad else "  SCHED CHECK: bit-exact"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
vla_board.py — build libsmolvla.so + vla_bench on the KV260 and run the VLA
board gate (doc/plans/VLA_PLAN.md): every action of the study's evaluation
samples bit-exact with the specification (smolvla_q.SmolVLA.sample with the
project's formats), the time of each camera's image, of the prefix and of the
ten denoising steps, vla_open time, the repetitions and the close / re-open
cycle identical.

  upload  project sources (demo/vla/build/smolvla_base_project,
          generate_vla_project.py) -> <remote dir>/smolvla_base_project
          (weights/ and build/ excluded; default remote dir /root/smolvla)
  weights weights/*.dat -> <weights dir>/weights (only changed files)
  build   cmake -DINFERENCE_WEIGHTS_DIR=<weights dir> + make vla_bench smolvla
          (the library copied to <remote dir>/lib/)
  run     vla_bench on samples.bin (the selected samples, two cameras each),
          -R repetitions, the re-open cycle
  check   every chunk against the specification, computed on the host in
          parallel and cached in <project>/reference/ (keyed by the formats and
          the specification's sources)

The board lock is held for the whole session.  The chat server must not hold
the FPGA (demo/chat/deploy.py --stop).

usage: inference-scheduler/.venv/bin/python demo/vla/scripts/vla_board.py
           [--project demo/vla/build/smolvla_base_project] [-n 24] [--reps 2] [--skip-build]
           [--remote-dir /root/smolvla] [--weights-dir /root/smolvla_weights] [--out results.json]
           [--spec-only]
(the specification needs the lerobot venv's numpy: --spec-python demo/vla/assets/.venv-lerobot/bin/python)
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per spec worker
    os.environ.setdefault(_v, "1")

import argparse                                                     # noqa: E402
import json                                                         # noqa: E402
import statistics                                                   # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402
from pathlib import Path                                            # noqa: E402

import numpy as np                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))
sys.path.insert(0, os.path.join(REPO, "demo", "chat", "scripts"))
sys.path.insert(0, os.path.join(REPO, "demo", "bert_squad", "scripts"))

import vla_host_emu as he                                           # noqa: E402

DEFAULT_PROJECT = os.path.join(REPO, "demo", "vla", "build", "smolvla_base_project")
RUN_DIR = "/tmp/vla_bench"


def main(argv=None) -> int:
    import llm_board
    from deploy_and_run import _stream_exec, board_lock, lock_path, sync_weights
    from src.remote import RemoteSession, _green, _red

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bert-config", default=None)
    ap.add_argument("-n", type=int, default=24, help="evaluation samples (the study's order)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--board-lock", default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="make -j on the board (default 1)")
    ap.add_argument("--jobs-spec", type=int, default=6, help="host processes for the specification")
    ap.add_argument("--spec-python", default=he.SPEC_PYTHON)
    ap.add_argument("--remote-dir", default="/root/smolvla")
    ap.add_argument("--weights-dir", default="/root/smolvla_weights")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--spec-only", action="store_true", help="only compute (and cache) the specification's chunks")
    ap.add_argument("--out", default=None, help="write the results JSON here")
    args = ap.parse_args(argv)
    project = os.path.abspath(args.project)
    cfg = llm_board.bert_config(args.bert_config)
    summary = json.load(open(os.path.join(project, "project.json")))
    lib = summary["library"]
    tag = lib.removeprefix("lib").removesuffix(".so")
    remote_proj = f"{args.remote_dir}/{summary['model']}_project"
    remote_lib = f"{args.remote_dir}/lib/{lib}"
    samples, ins = he.sample_inputs(project, args.n, args.spec_python)
    he.write_samples(os.path.join(project, "samples.bin"), ins)
    chunk, steps = summary["chunk"], summary["steps"]
    refs = None
    if not args.no_check:
        t0 = time.time()
        refs = he.reference(project, [s.name for s in samples], args.spec_python, args.jobs_spec)
        print(f"specification: {len(refs)} samples ({time.time() - t0:.0f} s)", flush=True)
    if args.spec_only:
        return 0
    results = {"project": summary, "samples": [s.name for s in samples], "weights_dir": args.weights_dir}
    with board_lock(lock_path(cfg, args.board_lock)):
        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            out, _, _ = session.exec("grep -E 'CmaFree|CmaTotal' /proc/meminfo | tr -s ' '", timeout=15)
            print(f"board: {out.strip()}", flush=True)
            results["cma_before"] = out.strip()
            if not args.skip_build:
                t0 = time.monotonic()
                n = llm_board.upload_project(session, project, remote_proj, skip=("reference", "board_gate.json"))
                print(f"  upload: {n} files ({time.monotonic() - t0:.0f} s)", flush=True)
                s = sync_weights(session, Path(project) / "weights", args.weights_dir)
                print(f"  weights: {'OK' if s.ok else 'FAIL'} {s.output} ({s.duration:.0f} s)", flush=True)
                if not s.ok:
                    return 1
                t0 = time.monotonic()
                llm_board.build(session, remote_proj, cfg, summary["active_kernels"], False, args.jobs,
                                args.weights_dir, targets=f"vla_bench {tag}")
                print(f"  build ({time.monotonic() - t0:.0f} s)", flush=True)
                session.exec_checked(f"mkdir -p {os.path.dirname(remote_lib)} && "
                                     f"cp {remote_proj}/build/{lib} {remote_lib}", timeout=30)
            sftp = session._client.open_sftp()                   # noqa: SLF001
            session.exec_checked(f"mkdir -p {RUN_DIR}", timeout=15)
            sftp.put(os.path.join(project, "samples.bin"), f"{RUN_DIR}/samples.bin")
            cmd = (f"cd {RUN_DIR} && {remote_proj}/build/vla_bench -i samples.bin -o actions.bin "
                   f"-R {args.reps} -r")
            t0 = time.monotonic()
            out, err, rc = _stream_exec(session, cmd, timeout=7200,
                                        on_stderr_line=lambda ln: print("    " + ln, flush=True))
            print(f"  vla_bench rc={rc} ({time.monotonic() - t0:.0f} s)", flush=True)
            if rc not in (0, 3):
                print(out[-2000:], err[-2000:])
                return 1
            results["bench"] = he.parse(out)
            local = os.path.join(project, "actions_board.bin")
            sftp.get(f"{RUN_DIR}/actions.bin", local)
            sftp.close()
        finally:
            session.exec(f"rm -rf {RUN_DIR}", timeout=30)
            session.close()
    acts = np.fromfile(local, "<f4").reshape(len(samples), chunk, summary["max_action"])
    bench = results["bench"]
    ss = bench["samples"]
    img = [t for x in ss for t in x["image_ms"]]
    lat = {"image_ms_median": statistics.median(img), "prefix_ms_median": statistics.median(x["prefix_ms"] for x in ss),
           "expert_ms_median": statistics.median(x["expert_ms"] for x in ss),
           "act_ms_median": statistics.median(x["act_ms"] for x in ss),
           "chunk_ms_median": statistics.median(sum(x["image_ms"]) + x["act_ms"] for x in ss)}
    results["latency"] = lat
    print(f"vla_open {bench.get('open', {}).get('ms')} ms; per camera image {lat['image_ms_median']:.0f} ms, "
          f"prefix {lat['prefix_ms_median']:.0f} ms, {steps} steps {lat['expert_ms_median']:.0f} ms; "
          f"a chunk of {chunk} actions from {ss[0]['cameras']} cameras {lat['chunk_ms_median']:.0f} ms")
    ok = all(x["reps_identical"] for x in ss) and bench.get("reopen", {}).get("identical", True)
    if refs is not None:
        bad = [s.name for k, s in enumerate(samples) if not np.array_equal(acts[k], refs[s.name])]
        results["check"] = {"samples": len(samples), "mismatches": len(bad), "differ": bad}
        print(f"check: {len(samples) - len(bad)}/{len(samples)} samples bit-exact with the specification"
              + (f"; differ: {bad[:10]}" if bad else ""))
        ok = ok and not bad
    results["actions"] = {s.name: acts[k].tolist() for k, s in enumerate(samples)}
    out_path = args.out or os.path.join(project, "board_gate.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path}")
    print(_green("BOARD GATE: actions bit-exact") if ok else _red("BOARD GATE: MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

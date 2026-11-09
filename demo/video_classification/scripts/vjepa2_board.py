#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
vjepa2_board.py — build libvjepa2.so + vjepa2_bench on the KV260 and run the
V-JEPA 2 board gate (doc/plans/VJEPA2_PLAN.md): the logits of the study's
evaluation videos bit-exact with the specification (vjepa2_q.VJEPA2.forward of
make_clip with the project's formats: the study's logits of the planned
policy, assets/study/full/results.json), the time of the frames' resize and of
the network, vjepa2_open time, the repetitions and the close / re-open cycle
identical.

  upload  project sources (demo/video_classification/build/vjepa2_project,
          generate_vjepa2_project.py) -> <remote dir>/<model>_project
          (weights/ and build/ excluded; default remote dir /root/vjepa2)
  weights weights/*.dat -> <weights dir>/weights (only changed files)
  build   cmake -DINFERENCE_WEIGHTS_DIR=<weights dir> + make vjepa2_bench vjepa2
          (the library copied to <remote dir>/lib/)
  run     vjepa2_bench on videos.bin (the decoded videos, at most their first 65
          frames: the sampler reads frames < 64 only, and 65 keeps its branch),
          -R repetitions, the re-open cycle; with --profile also the profiled
          build (build_prof/, -DINFERENCE_PROFILING=ON) on the first video, grouped
          by kind in the results JSON
  check   every video's logits against the specification's (--no-check: later, with
          --check-only on the saved results, e.g. once the study's logits exist)

The board lock is held for the whole session.  The chat server must not hold
the FPGA (demo/chat/deploy.py --stop).

usage: inference-scheduler/.venv/bin/python demo/video_classification/scripts/vjepa2_board.py
           [--project demo/video_classification/build/vjepa2_project] [-n 8] [--reps 2] [--skip-build]
           [--remote-dir /root/vjepa2] [--weights-dir /root/vjepa2_weights] [--profile] [--out results.json]
           [--no-check | --check-only results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))
sys.path.insert(0, os.path.join(REPO, "demo", "chat", "scripts"))
sys.path.insert(0, os.path.join(REPO, "demo", "bert_squad", "scripts"))
sys.path.insert(0, os.path.join(REPO, "demo", "vla", "scripts"))

import vjepa2_host_emu as he                                        # noqa: E402

DEFAULT_PROJECT = os.path.join(REPO, "demo", "video_classification", "build", "vjepa2_project")
RUN_DIR = "/tmp/vjepa2_bench"
MAX_FRAMES = 65                         # the sampler reads frames < 64; 65 keeps the n > 64 branch


def compare(logits: dict, refs) -> list:
    """The videos whose board logits differ from the specification's (float32)."""
    bad = [f"ssv2_{i:04d}" for i, want in refs
           if not np.array_equal(np.asarray(logits[f"ssv2_{i:04d}"], np.float32),
                                 np.asarray(want, np.float64).astype(np.float32))]
    print(f"check: {len(refs) - len(bad)}/{len(refs)} videos bit-exact with the specification"
          + (f"; differ: {bad[:10]}" if bad else ""))
    return bad


def check_only(path: str) -> int:
    """--check-only: a saved board run's logits against the study's."""
    from src.remote import _green, _red
    res = json.load(open(path))
    summary = res["project"]
    refs = he.study_reference(summary["assets"], summary["formats"], len(res["videos"]))
    if [f"ssv2_{i:04d}" for i, _ in refs] != res["videos"]:
        raise SystemExit(f"{path}: its videos are not the study's first {len(refs)}")
    bad = compare(res["logits"], refs)
    res["check"] = {"videos": len(refs), "mismatches": len(bad), "differ": bad}
    ok = not bad and all(x["reps_identical"] for x in res["bench"]["videos"]) \
        and res["bench"].get("reopen", {}).get("identical", True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(_green("BOARD GATE: logits bit-exact") if ok else _red("BOARD GATE: MISMATCH"))
    return 0 if ok else 1


def main(argv=None) -> int:
    import llm_board
    import vjepa2_study as st
    from deploy_and_run import _stream_exec, board_lock, lock_path, sync_weights
    from src.remote import RemoteSession, _green, _red
    from vla_board import profile_summary, profiles

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bert-config", default=None)
    ap.add_argument("-n", type=int, default=8, help="evaluation videos (the study's order, at most 100)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--board-lock", default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--profile", action="store_true", help="also build with the profiler and profile video 0")
    ap.add_argument("--jobs", type=int, default=1, help="make -j on the board (default 1)")
    ap.add_argument("--remote-dir", default="/root/vjepa2")
    ap.add_argument("--weights-dir", default="/root/vjepa2_weights")
    ap.add_argument("--out", default=None, help="write the results JSON here")
    ap.add_argument("--no-check", action="store_true", help="no reference yet: keep the logits, check later")
    ap.add_argument("--check-only", default=None, metavar="RESULTS",
                    help="compare a board run's saved logits (its results JSON) with the specification's")
    args = ap.parse_args(argv)
    project = os.path.abspath(args.project)
    if args.check_only:
        return check_only(args.check_only)
    cfg = llm_board.bert_config(args.bert_config)
    summary = json.load(open(os.path.join(project, "project.json")))
    lib = summary["library"]
    tag = lib.removeprefix("lib").removesuffix(".so")
    remote_proj = f"{args.remote_dir}/{summary['model']}_project"
    remote_lib = f"{args.remote_dir}/lib/{lib}"
    if args.no_check:
        rows = st.inputs()["samples"]["evaluation"][:args.n]
        refs = [(int(i), None) for i in rows]
    else:
        refs = he.study_reference(summary["assets"], summary["formats"], args.n)
    t0 = time.time()
    videos = [st.decode_video(os.path.join(summary["assets"], "videos", f"{i:04d}.webm"))[:MAX_FRAMES]
              for i, _ in refs]
    vpath = os.path.join(project, "videos.bin")
    he.write_videos(vpath, videos)
    print(f"{len(videos)} videos decoded ({os.path.getsize(vpath) / 2**20:.0f} MiB, {time.time() - t0:.0f} s)",
          flush=True)
    results = {"project": summary, "videos": [f"ssv2_{i:04d}" for i, _ in refs], "weights_dir": args.weights_dir}
    with board_lock(lock_path(cfg, args.board_lock)):
        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            out, _, _ = session.exec("grep -E 'CmaFree|CmaTotal|MemAvailable' /proc/meminfo | tr -s ' '; "
                                     "df -h /root /tmp | tail -2", timeout=15)
            print(f"board: {out.strip()}", flush=True)
            results["board_before"] = out.strip()
            if not args.skip_build:
                t0 = time.monotonic()
                n = llm_board.upload_project(session, project, remote_proj,
                                             skip=("reference", "board_gate.json", "videos.bin"))
                print(f"  upload: {n} files ({time.monotonic() - t0:.0f} s)", flush=True)
                s = sync_weights(session, Path(project) / "weights", args.weights_dir)
                print(f"  weights: {'OK' if s.ok else 'FAIL'} {s.output} ({s.duration:.0f} s)", flush=True)
                if not s.ok:
                    return 1
                t0 = time.monotonic()
                llm_board.build(session, remote_proj, cfg, summary["active_kernels"], args.profile, args.jobs,
                                args.weights_dir, targets=f"vjepa2_bench {tag}")
                print(f"  build ({time.monotonic() - t0:.0f} s)", flush=True)
                session.exec_checked(f"mkdir -p {os.path.dirname(remote_lib)} && "
                                     f"cp {remote_proj}/build/{lib} {remote_lib}", timeout=30)
            sftp = session._client.open_sftp()                   # noqa: SLF001
            session.exec_checked(f"mkdir -p {RUN_DIR}", timeout=15)
            t0 = time.monotonic()
            sftp.put(vpath, f"{RUN_DIR}/videos.bin")
            print(f"  videos uploaded ({time.monotonic() - t0:.0f} s)", flush=True)
            cmd = (f"cd {RUN_DIR} && {remote_proj}/build/vjepa2_bench -i videos.bin -o logits.bin "
                   f"-R {args.reps} -r")
            t0 = time.monotonic()
            out, err, rc = _stream_exec(session, cmd, timeout=7200,
                                        on_stderr_line=lambda ln: print("    " + ln, flush=True))
            print(f"  vjepa2_bench rc={rc} ({time.monotonic() - t0:.0f} s)", flush=True)
            if rc not in (0, 3):
                print(out[-2000:], err[-2000:])
                return 1
            results["bench"] = he.parse(out)
            local = os.path.join(project, "logits_board.bin")
            sftp.get(f"{RUN_DIR}/logits.bin", local)
            if args.profile:
                ppath = os.path.join(project, "videos_prof.bin")
                he.write_videos(ppath, videos[:1])
                sftp.put(ppath, f"{RUN_DIR}/videos_prof.bin")
                out, err, rc = _stream_exec(session, f"cd {RUN_DIR} && {remote_proj}/build_prof/vjepa2_bench "
                                            f"-i videos_prof.bin -o logits_prof.bin -R 1 > prof.txt", timeout=3600,
                                            on_stderr_line=lambda ln: print("    " + ln, flush=True))
                print(f"  vjepa2_bench [profile] rc={rc}", flush=True)
                local_prof = os.path.join(project, "profile_board.txt")
                sftp.get(f"{RUN_DIR}/prof.txt", local_prof)
                results["profile"] = profile_summary(project, profiles(open(local_prof).read()))
            sftp.close()
        finally:
            session.exec(f"rm -rf {RUN_DIR}", timeout=30)
            session.close()
    lg = np.fromfile(local, "<f4").reshape(len(refs), summary["classes"])
    bench = results["bench"]
    vs = bench["videos"]
    lat = {k + "_median": statistics.median(x[k] for x in vs) for k in ("prep_ms", "run_ms", "total_ms", "best_ms")}
    results["latency"] = lat
    print(f"vjepa2_open {bench.get('open', {}).get('ms')} ms; per video: resize + crop {lat['prep_ms_median']:.0f} ms, "
          f"network {lat['run_ms_median']:.0f} ms, total {lat['total_ms_median']:.0f} ms (median)")
    ok = all(x["reps_identical"] for x in vs) and bench.get("reopen", {}).get("identical", True)
    results["logits"] = {f"ssv2_{i:04d}": lg[k].tolist() for k, (i, _) in enumerate(refs)}
    if not args.no_check:
        bad = compare(results["logits"], refs)
        results["check"] = {"videos": len(refs), "mismatches": len(bad), "differ": bad}
        ok = ok and not bad
    else:
        print("check: skipped (--no-check); later: --check-only <this results JSON>")
    out_path = args.out or os.path.join(project, "board_gate.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path}")
    if args.no_check:
        print(_green("BOARD RUN: repetitions and re-open identical (not yet checked against the specification)")
              if ok else _red("BOARD RUN: repetitions or re-open DIFFER"))
    else:
        print(_green("BOARD GATE: logits bit-exact") if ok else _red("BOARD GATE: MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
asr_board.py — build libmoonshine_<size>.so + asr_bench on the KV260 and run
the ASR board gate (doc/plans/ASR_PLAN.md §6): every token id of the
LibriSpeech evaluation utterances bit-exact with the specification
(moonshine_q.transcribe_long with the project's formats), the encode time per
bucket and the time per decode step, the real-time factor, asr_open time,
CMA and the close / re-open cycle; the board's ids for the WER
(asr_study.py score).

  upload  project sources (demo/asr/build/<model>_project,
          generate_asr_project.py) -> <remote dir>/<model>_project (weights/
          and build/ excluded; remote dir: the chat install directory,
          demo/chat/chat_config.json remote.dir — where the chat server's
          moonshine backend loads the library)
  weights weights/*.dat -> <weights dir>/weights (only changed files)
  build   cmake -DINFERENCE_WEIGHTS_DIR=<weights dir> + make asr_bench
          moonshine_<size> (and build_prof/ with -DINFERENCE_PROFILING=ON when
          --profile); the library copied to <remote dir>/lib/ (--install-only:
          these steps only)
  run     asr_bench on utts.bin (the selected utterances), -R repetitions,
          the re-open cycle; with --profile a profiled run over the fullest
          one-part utterance of each bucket (phases encode_<T>, and decode on
          the second largest bucket's)
  check   every utterance's ids against the specification, computed on the
          host in parallel and cached in <project>/reference/ (keyed by the
          formats, the specification's sources and the largest bucket)

The board lock is held for the whole session.  Stop the chat server first
(demo/chat/deploy.py --stop) — it owns the FPGA.

usage: inference-scheduler/.venv/bin/python demo/asr/scripts/asr_board.py
           [--project demo/asr/build/moonshine-tiny_project] [--utts all | auto | N | UID,...]
           [--reps 2] [--profile] [--skip-build] [--install-only] [--remote-dir /root/kv260_chat]
           [--weights-dir /root/<lib>_weights] [--jobs-spec 8] [--out results.json] [--spec-only]
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per spec worker
    os.environ.setdefault(_v, "1")

import argparse                                                     # noqa: E402
import hashlib                                                      # noqa: E402
import json                                                         # noqa: E402
import statistics                                                   # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402
from pathlib import Path                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))
sys.path.insert(0, os.path.join(REPO, "demo", "chat", "scripts"))
sys.path.insert(0, os.path.join(REPO, "demo", "bert_squad", "scripts"))

import asr_host_emu as he                                           # noqa: E402

DEFAULT_PROJECT = os.path.join(REPO, "demo", "asr", "build", "moonshine-tiny_project")
RUN_DIR = "/tmp/asr_bench"
SR = 16000


def bucket_of(n: int, buckets) -> int:
    """The bucket asr_encode() picks for n samples: the smallest with 384 T + 511 >= n."""
    return next((T for T in sorted(buckets) if 384 * T + 511 >= n), max(buckets))


def select(which: str):
    """[(uid, reference, int16 samples)]: ``all`` the 200 evaluation utterances,
    ``auto`` one per bucket (asr_host_emu.select), N evenly spaced by length, or ids."""
    import asr_study
    if which == "all":
        return asr_study.utterances(asr_study.ASSETS, "evaluation")
    if which.isdigit():
        utts = sorted(asr_study.utterances(asr_study.ASSETS, "evaluation"), key=lambda u: len(u[2]))
        k = int(which)
        return [utts[round(i * (len(utts) - 1) / max(k - 1, 1))] for i in range(k)]
    return he.select(which)


def profile_set(utts, buckets):
    """The profiled run's utterances: per bucket the fullest one-part utterance (the
    stem's host ops scale with the valid frames, and simulate compares a bucket's
    fullest utterance), the second largest bucket's first — asr_bench profiles the
    decode steps of utterance 0."""
    full = {}
    for u in utts:
        n = len(u[2])
        if n <= 384 * max(buckets) + 511:
            T = bucket_of(n, buckets)
            if n > len(full.get(T, (None, None, ()))[2]):
                full[T] = u
    order = sorted(full)
    first = order[-2] if len(order) > 1 else order[0]
    return [full[first]] + [full[T] for T in order if T != first]


def _spec_key(project: str) -> str:
    """The reference cache's key: the formats, the specification's sources and the
    largest bucket (a regenerated project with other exponents misses the cache)."""
    summary = json.load(open(os.path.join(project, "project.json")))
    h = hashlib.sha256()
    for p in (summary["formats"], os.path.join(HERE, "moonshine_q.py"), os.path.join(HERE, "moonshine_np.py")):
        h.update(open(p, "rb").read())
    h.update(str(max(summary["buckets"])).encode())
    return h.hexdigest()[:16]


_SPEC = None


def _spec_one(item):
    import moonshine_q as mq
    uid, pcm = item
    m, cfgd, frames = _SPEC
    return uid, mq.transcribe_long(m, cfgd, pcm, frames)


def reference(project: str, utts, jobs: int = 8) -> dict:
    """{uid: ids} of the specification for ``utts``, from the cache where it has them,
    the rest computed in ``jobs`` processes and added to the cache."""
    global _SPEC
    path = os.path.join(project, "reference", f"spec_{_spec_key(project)}.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    todo = [(u, pcm) for u, _, pcm in utts if u not in cache]
    if todo:
        import multiprocessing as mp
        t0 = time.time()
        _SPEC = he.spec(project)
        todo.sort(key=lambda x: -len(x[1]))                 # the longest first: an even finish
        with mp.get_context("fork").Pool(jobs) as p:
            for k, (u, ids) in enumerate(p.imap_unordered(_spec_one, todo), 1):
                cache[u] = ids
                if k % 20 == 0 or k == len(todo):
                    print(f"  spec: {k}/{len(todo)} utterances ({time.time() - t0:.0f} s)", flush=True)
        _SPEC = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f)
    return {u: cache[u] for u, _, _ in utts}


def latency(bench: dict, utts, buckets) -> dict:
    """Encode ms per bucket (one-part utterances: median, min, count), ms per decode
    step (median over utterances), mean ms per utterance and the real-time factor."""
    enc, step = {}, []
    for (_, _, pcm), r in zip(utts, bench["utts"], strict=True):
        if r["parts"] == 1 and r["steps"]:
            enc.setdefault(bucket_of(len(pcm), buckets), []).append(r["encode_ms"])
        if r["steps"]:
            step.append(r["decode_ms"] / r["steps"])
    s = bench.get("summary", {})
    return {"encode_ms": {T: {"median": round(statistics.median(v), 2), "min": round(min(v), 2), "n": len(v)}
                          for T, v in sorted(enc.items())},
            "decode_step_ms": round(statistics.median(step), 3) if step else None,
            "utterance_ms_mean": round(s.get("total_ms", 0.0) / max(len(utts), 1), 1),
            "rtf": s.get("rtf")}


def main(argv=None) -> int:
    import llm_board
    from deploy_and_run import _stream_exec, board_lock, lock_path, sync_weights
    from src.remote import RemoteSession, _green, _red

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bert-config", default=None)
    ap.add_argument("--utts", default="all", help="all (200), auto (one per bucket), N or utterance ids")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--profile", action="store_true",
                    help="also build with the profiler and profile the fullest utterance of each bucket")
    ap.add_argument("--board-lock", default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--install-only", action="store_true",
                    help="upload, build and install the library (and its weights); no bench or checks")
    ap.add_argument("--jobs", type=int, default=1, help="make -j on the board (default 1)")
    ap.add_argument("--jobs-spec", type=int, default=8, help="host processes for the specification")
    ap.add_argument("--remote-dir", default=None,
                    help="<dir>/<model>_project and <dir>/lib/lib<model>.so on the board (default: "
                         "remote.dir of demo/chat/chat_config.json, else /root/kv260_chat)")
    ap.add_argument("--weights-dir", default=None, help="default /root/<lib>_weights")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--spec-only", action="store_true",
                    help="only compute (and cache) the specification's ids; no board")
    ap.add_argument("--out", default=None, help="write the results JSON here")
    args = ap.parse_args(argv)
    if args.install_only and args.skip_build:
        ap.error("--install-only builds; it cannot be combined with --skip-build")
    if args.install_only:
        args.no_check = True
    project = os.path.abspath(args.project)
    cfg = llm_board.bert_config(args.bert_config)
    summary = json.load(open(os.path.join(project, "project.json")))
    lib = summary["library"]                                     # libmoonshine_<size>.so
    tag = lib.removeprefix("lib").removesuffix(".so")
    weights_dir = args.weights_dir or f"/root/{tag}_weights"
    remote_dir = args.remote_dir or llm_board.chat_remote_dir()
    remote_proj = f"{remote_dir}/{summary['model']}_project"
    remote_lib = f"{remote_dir}/lib/{lib}"
    utts = [] if args.install_only else select(args.utts)
    if utts:
        he.write_utts(os.path.join(project, "utts.bin"), [u[2] for u in utts])
    prof_utts = []
    if args.profile and not args.install_only:
        prof_utts = profile_set(utts if args.utts == "all" else select("all"), summary["buckets"])
        he.write_utts(os.path.join(project, "utts_prof.bin"), [u[2] for u in prof_utts])
    refs = None
    if not args.no_check:
        t0 = time.time()
        refs = reference(project, utts, args.jobs_spec)
        print(f"specification: {len(refs)} utterances ({time.time() - t0:.0f} s)", flush=True)
    if args.spec_only:
        return 0
    results = {"project": summary, "utts": [u[0] for u in utts], "weights_dir": weights_dir}
    with board_lock(lock_path(cfg, args.board_lock)):
        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            out, _, _ = session.exec("grep -E 'CmaFree|CmaTotal' /proc/meminfo | tr -s ' '", timeout=15)
            print(f"board: {out.strip()}", flush=True)
            if not args.skip_build:
                t0 = time.monotonic()
                n = llm_board.upload_project(session, project, remote_proj,
                                             skip=("reference", "board_gate.json"))
                print(f"  upload: {n} files ({time.monotonic() - t0:.0f} s)", flush=True)
                s = sync_weights(session, Path(project) / "weights", weights_dir)
                print(f"  weights: {'OK' if s.ok else 'FAIL'} {s.output} ({s.duration:.0f} s)", flush=True)
                if not s.ok:
                    return 1
                llm_board.build(session, remote_proj, cfg, summary["active_kernels"], args.profile,
                                args.jobs, weights_dir, targets=f"asr_bench {tag}")
                session.exec_checked(f"mkdir -p {os.path.dirname(remote_lib)} && "
                                     f"cp {remote_proj}/build/{lib} {remote_lib}", timeout=30)
                out, _, _ = session.exec(f"nm -D --defined-only {remote_lib} | awk '{{print $3}}' | sort",
                                         timeout=30)
                results["exported"] = [s for s in out.split() if s]
                print(f"  {lib} exports: {results['exported']}", flush=True)
                if args.install_only:
                    print(f"installed {remote_lib} (weights {weights_dir})")
                    return 0
            sftp = session._client.open_sftp()                   # noqa: SLF001
            session.exec_checked(f"mkdir -p {RUN_DIR}", timeout=15)
            sftp.put(os.path.join(project, "utts.bin"), f"{RUN_DIR}/utts.bin")
            runs = [("build", "main", "utts.bin", args.reps, True)]
            if args.profile:
                sftp.put(os.path.join(project, "utts_prof.bin"), f"{RUN_DIR}/utts_prof.bin")
                runs.append(("build_prof", "prof", "utts_prof.bin", 1, False))
            for bdir, rtag, utts_bin, reps, reopen in runs:
                cmd = (f"cd {RUN_DIR} && {remote_proj}/{bdir}/asr_bench -i {utts_bin} -o ids_{rtag}.bin "
                       f"-R {reps} {'-r' if reopen else ''}")
                t0 = time.monotonic()
                out, err, rc = _stream_exec(session, cmd, timeout=7200,
                                            on_stderr_line=lambda ln: print("    " + ln, flush=True))
                print(f"  asr_bench [{rtag}] rc={rc} ({time.monotonic() - t0:.0f} s)", flush=True)
                if rc != 0:
                    print(out[-2000:], err[-2000:])
                    return 1
                res = he.parse(out)
                res["profiles"] = profiles(out)
                local_ids = os.path.join(project, f"ids_board_{rtag}.bin")
                sftp.get(f"{RUN_DIR}/ids_{rtag}.bin", local_ids)
                if rtag == "main":
                    results["bench"] = res
                    results["ids"] = dict(zip(results["utts"], he.read_ids(local_ids, len(utts)), strict=True))
                else:
                    results["profile_utts"] = [{"uid": u, "samples": len(pcm)} for u, _, pcm in prof_utts]
                    results["profile_bench"] = {k: v for k, v in res.items() if k != "profiles"}
                    results["profile_layers"] = {
                        ph: [{k: ly[k] for k in ("i", "name", "calls", "mean_us", "min_us", "total_us")
                              if k in ly} for ly in p.get("layers", []) if ly.get("calls")]
                        for ph, p in res["profiles"].items()}
            sftp.close()
        finally:
            session.close()
    bench = results["bench"]
    o = bench.get("open", {})
    print(f"asr_open {o.get('open_ms')} ms, CmaFree {o.get('cma_free_kb_before')} -> "
          f"{o.get('cma_free_kb_after')} kB "
          f"({(o.get('cma_free_kb_before', 0) - o.get('cma_free_kb_after', 0)) / 1024:.1f} MB)")
    results["latency"] = lat = latency(bench, utts, summary["buckets"])
    for T, v in lat["encode_ms"].items():
        print(f"encode_{T}: median {v['median']} ms, min {v['min']} ms ({v['n']} utterances)")
    print(f"decode step: median {lat['decode_step_ms']} ms; {lat['utterance_ms_mean']} ms per utterance, "
          f"RTF {lat['rtf']}")
    if "reopen" in bench:
        print(f"re-open: {bench['reopen']}")
    for ph, ly in results.get("profile_layers", {}).items():
        print(f"profile {ph}: {sum(x['total_us'] for x in ly) / 1000:.1f} ms in {len(ly)} layers")
    ok = all(u["reps_identical"] for u in bench["utts"]) and bench.get("reopen", {}).get("identical", True)
    if refs is not None:
        bad = [u for u in results["utts"] if results["ids"][u] != refs[u]]
        results["check"] = {"utterances": len(refs), "mismatches": len(bad), "differ": bad}
        print(f"check: {len(refs) - len(bad)}/{len(refs)} utterances bit-exact with the specification"
              + (f"; differ: {bad[:10]}" if bad else ""))
        ok = ok and not bad
    out_path = args.out or os.path.join(project, "board_gate.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path} (WER: .venv-export/bin/python demo/asr/scripts/asr_study.py score "
          f"--model {summary['model']} --ids {out_path})")
    print(_green("BOARD GATE: token ids bit-exact") if ok else _red("BOARD GATE: MISMATCH"))
    return 0 if ok else 1


def profiles(out: str) -> dict:
    """{phase: the profiler's LAYERS_JSON} of asr_bench's "PROFILE_PHASE: <entry>" lines."""
    res, phase = {}, None
    for line in out.splitlines():
        if line.startswith("PROFILE_PHASE:"):
            phase = line.split(":", 1)[1].strip()
        elif line.startswith("LAYERS_JSON:") and phase:
            res[phase] = json.loads(line[len("LAYERS_JSON:"):])
            phase = None
    return res


if __name__ == "__main__":
    sys.exit(main())

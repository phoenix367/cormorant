#!/usr/bin/env python3
"""
llm_board.py — build libsmollm2.so + llm_bench on the KV260 and run the
phase-3 board gate (doc/CHAT_PLAN.md): logits bit-exact with the scheduler
simulation on prefill + greedy decode for real chat prompts, the greedy text
against the study emulation, decode ms / token (per-kind profile), prefill
times, llm_open time, CMA, and the close / re-open cycle.

  upload  project sources (demo/chat/build/llm_project, generate_llm_project.py)
          -> <remote dir>/llm_project (weights/ and build/ excluded)
  weights weights/*.dat -> <weights_dir>/weights (only changed files)
  build   cmake -DINFERENCE_WEIGHTS_DIR=<weights_dir> + make llm_bench smollm2
          (and build_prof/ with -DINFERENCE_PROFILING=ON when --profile)
  install <remote dir>/lib/libsmollm2.so (the chat server's default path)
  run     llm_bench on prompts.bin (the chat prompts' token ids after the
          leading <|im_start|>); logits.bin downloaded
  check   every logits vector against SimSession (llm_project.py), replaying
          the board's greedy tokens; the tokens against the study emulation's
          greedy ids (llm_sched_check.py --save JSON, optional)

The board lock (flock) is held for the whole session.  Stop the chat server
first (demo/chat/deploy.py --stop) — it owns the FPGA.

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/llm_board.py
           [--project demo/chat/build/llm_project] [--prompts factual,summarise,multi-turn]
           [--decode 32] [--prefill-lens 16,64,256] [--profile] [--reopen]
           [--study-json gate2.json] [--board-lock FILE] [--skip-build]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_project as lp                                           # noqa: E402

BERT_SCRIPTS = os.path.join(lp.REPO, "demo", "bert_squad", "scripts")
sys.path.insert(0, BERT_SCRIPTS)
from deploy_and_run import _stream_exec, board_lock, sync_weights  # noqa: E402
from src.remote import RemoteSession, _dim, _green, _red           # noqa: E402

DEFAULT_PROJECT = os.path.join(os.path.dirname(HERE), "build", "llm_project")
REMOTE_DIR = "/root/kv260_chat"
WEIGHTS_DIR = "/root/smollm2_weights"
RUN_DIR = "/tmp/llm_bench"


def bert_config(path=None) -> dict:
    """ssh / uio_devices / run settings of the BERT-SQuAD demo config (the
    same board)."""
    for p in ([path] if path else []) + [
            os.path.join(lp.REPO, "demo", "bert_squad", "bert_squad_config.json"),
            os.path.join(lp._main_checkout(), "demo", "bert_squad", "bert_squad_config.json")]:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise SystemExit("bert_squad_config.json not found (--bert-config)")


def write_prompts(path: str, ids: dict) -> list:
    names = list(ids)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(names)))
        for n in names:
            t = ids[n][1:]                       # after the sink <|im_start|>
            f.write(struct.pack("<i", len(t)))
            f.write(np.asarray(t, "<i4").tobytes())
    return names


def upload_project(session, local: str, remote: str) -> int:
    session.exec_checked(f"mkdir -p {shlex.quote(remote)}", timeout=30)
    session.exec_checked(f"find {shlex.quote(remote)} -mindepth 1 -maxdepth 1 "
                         f"! -name build ! -name build_prof -exec rm -rf {{}} +", timeout=60)
    sftp = session._client.open_sftp()                           # noqa: SLF001
    n = 0
    try:
        for p in sorted(Path(local).rglob("*")):
            rel = p.relative_to(local)
            if rel.parts[0] in ("weights", "build", "build_prof", "expected", "host_emu") or rel.suffix == ".bin":
                continue
            dst = f"{remote}/{rel.as_posix()}"
            if p.is_dir():
                try:
                    sftp.mkdir(dst)
                except OSError:
                    pass
            else:
                sftp.put(str(p), dst)
                n += 1
    finally:
        sftp.close()
    return n


GUARD = r"""
set -u
B="$1"; shift
LOG="$B.guard.log"; : > "$LOG"
( "$@" ; echo $? > "$B.rc" ) &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  psi=$(awk '/^some/{split($2,a,"="); print a[2]}' /proc/pressure/memory 2>/dev/null || echo 0)
  rss=$(ps -C cc1 -o rss= 2>/dev/null | sort -n | tail -1); rss=${rss:-0}
  [ "$rss" -gt "$peak" ] && peak=$rss
  echo "$(date +%T) MemAvailable=${avail}kB psi_some_avg10=$psi cc1_rss=${rss}kB" >> "$LOG"
  if [ "$avail" -lt LIMIT_KB ] || awk "BEGIN{exit !($psi > PSI_MAX)}"; then
    echo "GUARD: killing the build (MemAvailable=${avail}kB psi=$psi)" | tee -a "$LOG" >&2
    pkill -P $pid; pkill -x cc1; pkill -x make; pkill -x cmake; echo 99 > "$B.rc"
    break
  fi
  sleep 2
done
wait $pid 2>/dev/null
echo "GUARD_PEAK_CC1_KB=$peak" >&2
exit $(cat "$B.rc")
"""


def build(session, remote_proj: str, cfg: dict, active, profile: bool, jobs: int,
          min_avail_mb: int = 1500, limit_mb: int = 400, psi_max: float = 50.0) -> None:
    """cmake + make llm_bench smollm2 on the board under a memory guard: the
    build starts only with >= min_avail_mb MemAvailable, and is killed when
    MemAvailable drops below limit_mb or /proc/pressure/memory some avg10
    exceeds psi_max (the 3.7 MB generated inference.c once took cc1 to 3 GB
    and starved the board).  jobs defaults to 1: inference.c is compiled once
    (the PIC static library is linked into both targets)."""
    uio = cfg["remote"].get("uio_devices", {})
    defs = " ".join(f"-DINFERENCE_{k.upper()}_INSTANCE=\\\"{uio[k]}\\\"" for k in active if k in uio)
    guard = GUARD.replace("LIMIT_KB", str(limit_mb * 1024)).replace("PSI_MAX", str(psi_max))
    sftp = session._client.open_sftp()                           # noqa: SLF001
    with sftp.open(f"{remote_proj}/build_guard.sh", "w") as f:
        f.write(guard)
    sftp.close()
    for bdir, prof in (("build", False),) + ((("build_prof", True),) if profile else ()):
        out, _, _ = session.exec("awk '/MemAvailable/{print $2}' /proc/meminfo", timeout=15)
        avail = int(out.strip() or 0) // 1024
        if avail < min_avail_mb:
            raise SystemExit(f"build {bdir}: MemAvailable {avail} MB < {min_avail_mb} MB, not building")
        b = f"{remote_proj}/{bdir}"
        cmd = (f"mkdir -p {b} && cmake -S {remote_proj} -B {b} -DCMAKE_BUILD_TYPE=Release "
               f"-DINFERENCE_TARGET=LINUX -DINFERENCE_WEIGHTS_DIR={WEIGHTS_DIR} "
               f"-DINFERENCE_PROFILING={'ON' if prof else 'OFF'} {defs} > {b}.cmake.log 2>&1 && "
               f"make -C {b} -j{jobs} llm_bench smollm2 > {b}.make.log 2>&1")
        t0 = time.monotonic()
        lines = []
        _o, _e, rc = _stream_exec(session, f"bash {remote_proj}/build_guard.sh {b} bash -c "
                                  f"{shlex.quote(cmd)}", timeout=5400,
                                  on_stderr_line=lambda ln: (lines.append(ln),
                                                             print("    " + ln, flush=True)))
        guard_log, _, _ = session.exec(f"tail -3 {b}.guard.log; tail -20 {b}.make.log", timeout=30)
        peak = next((ln.split("=")[1] for ln in lines if ln.startswith("GUARD_PEAK_CC1_KB=")), "?")
        print(f"  build {bdir}: {'OK' if rc == 0 else 'FAIL'} ({time.monotonic() - t0:.0f} s, "
              f"cc1 peak {peak} kB)\n{_dim(guard_log)}", flush=True)
        if rc != 0:
            raise SystemExit(f"build {bdir} failed (rc={rc})")


def run_bench(session, remote_proj: str, bdir: str, args, tag: str) -> str:
    extra = f"-P {args.prefill_lens} -R {args.reps}" if args.prefill_lens else ""
    cmd = (f"mkdir -p {RUN_DIR} && cd {RUN_DIR} && "
           f"{remote_proj}/{bdir}/llm_bench -i prompts.bin -o logits_{tag}.bin -k {args.decode} "
           f"{extra} {'-r' if args.reopen and tag == 'main' else ''}")
    t0 = time.monotonic()
    out, err, rc = _stream_exec(session, cmd, timeout=7200,
                                on_stderr_line=lambda ln: print("    " + ln, flush=True))
    print(f"  llm_bench [{tag}] rc={rc} ({time.monotonic() - t0:.0f} s)", flush=True)
    if rc != 0:
        print(out[-2000:], err[-2000:])
        raise SystemExit(1)
    return out


def parse(out: str) -> dict:
    res = {"prompts": [], "prefill": [], "profiles": {}}
    phase = None
    for line in out.splitlines():
        for key in ("LLM_OPEN", "LLM_PROMPT", "LLM_PREFILL", "LLM_REOPEN", "LLM_SUMMARY"):
            if line.startswith(key + ":"):
                d = json.loads(line[len(key) + 1:])
                if key == "LLM_PROMPT":
                    res["prompts"].append(d)
                elif key == "LLM_PREFILL":
                    res["prefill"].append(d)
                else:
                    res[key.lower()] = d
        if line.startswith("PROFILE_PHASE:"):
            phase = line.split(":", 1)[1].strip()
        elif line.startswith("LAYERS_JSON:") and phase:
            res["profiles"][phase] = json.loads(line[len("LAYERS_JSON:"):])
            phase = None
    return res


def breakdown(profile: dict, layers: list, per: int = 1) -> dict:
    """{kind: ms per call of the phase} from the profiler's per-layer totals."""
    kind = {L["i"]: L["kind"] for L in layers}
    out = {}
    for L in profile["layers"]:
        if L["calls"]:
            k = kind.get(L["i"], "other")
            out[k] = out.get(k, 0.0) + L["total_us"] / 1000.0 / per
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def check_logits(project: str, names: list, ids: dict, res: dict, logits_path: str,
                 decode: int, study_json: str = None) -> dict:
    """Replay the board's tokens on SimSession; compare every logits vector."""
    cfg, W, fmt, fd = lp.load_model()
    summary = json.load(open(os.path.join(project, "project.json")))
    fe = lp.frontend(cfg, W, fmt, ctx=summary["context"])
    cgs = {n: lp.make_codegen(m, n, "off") for n, m in lp.entry_models(fe, summary["buckets"]).items()}
    raw = np.fromfile(logits_path, "<f4").reshape(-1, cfg.V)
    study = json.load(open(study_json)) if study_json and os.path.exists(study_json) else {}
    rep, k = {}, 0
    for pi, name in enumerate(names):
        pr = res["prompts"][pi]
        toks = [s["tok"] for s in pr["steps"]]
        sess = lp.SimSession(cgs, ctx=summary["context"], buckets=summary["buckets"])
        a = sess.prefill(ids[name][1:])
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
        st = study.get(name, {}).get("generated")
        rep[name] = {"steps": decode + 1, "bit_exact_steps": exact, "first_mismatch": first_bad,
                     "tokens": toks,
                     "study_tokens_equal": (toks == st[:len(toks)]) if st else None}
        print(f"  [{name}] logits bit-exact {exact}/{decode + 1}"
              + (f" (first mismatch at step {first_bad})" if first_bad is not None else "")
              + (f", greedy tokens {'==' if rep[name]['study_tokens_equal'] else '!='} study emulation"
                 if st else ""), flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bert-config", default=None)
    ap.add_argument("--prompts", default="factual,summarise,multi-turn")
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--prefill-lens", default="16,64,256")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="also build and run with the profiler")
    ap.add_argument("--reopen", action="store_true", help="llm_close + llm_open cycle")
    ap.add_argument("--study-json", default=None, help="llm_sched_check.py --save output")
    ap.add_argument("--board-lock", default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="make -j on the board (default 1)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-lib-check", action="store_true",
                    help="skip llm_lib_check.py (ctypes: exports, threads, chunks, re-open)")
    ap.add_argument("--out", default=None, help="write the results JSON here")
    args = ap.parse_args(argv)
    cfg = bert_config(args.bert_config)
    summary = json.load(open(os.path.join(args.project, "project.json")))
    layers = json.load(open(os.path.join(args.project, "layers.json")))
    ids = lp.tokenize_prompts(args.prompts.split(","))
    local_prompts = os.path.join(args.project, "prompts.bin")
    names = write_prompts(local_prompts, ids)
    results = {"project": summary, "prompts": names}
    with board_lock(args.board_lock or cfg.get("board_lock")):
        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            out, _, _ = session.exec("grep -E 'CmaFree|CmaTotal' /proc/meminfo | tr -s ' '", timeout=15)
            print(f"board: {out.strip()}", flush=True)
            remote_proj = f"{REMOTE_DIR}/llm_project"
            if not args.skip_build:
                t0 = time.monotonic()
                n = upload_project(session, args.project, remote_proj)
                print(f"  upload: {n} files ({time.monotonic() - t0:.0f} s)", flush=True)
                s = sync_weights(session, Path(args.project) / "weights", WEIGHTS_DIR)
                print(f"  weights: {'OK' if s.ok else 'FAIL'} {s.output} ({s.duration:.0f} s)",
                      flush=True)
                if not s.ok:
                    return 1
                build(session, remote_proj, cfg, summary["active_kernels"], args.profile,
                      args.jobs)
                session.exec_checked(f"mkdir -p {REMOTE_DIR}/lib && cp {remote_proj}/build/libsmollm2.so "
                                     f"{REMOTE_DIR}/lib/libsmollm2.so", timeout=30)
                out, _, _ = session.exec(f"nm -D --defined-only {REMOTE_DIR}/lib/libsmollm2.so "
                                         f"| awk '{{print $3}}' | sort", timeout=30)
                syms = [s for s in out.split() if s]
                results["exported"] = syms
                print(f"  libsmollm2.so exports: {syms}", flush=True)
            sftp = session._client.open_sftp()                   # noqa: SLF001
            session.exec_checked(f"mkdir -p {RUN_DIR}", timeout=15)
            sftp.put(local_prompts, f"{RUN_DIR}/prompts.bin")
            text = run_bench(session, remote_proj, "build", args, "main")
            res = parse(text)
            local_logits = os.path.join(args.project, "logits_board.bin")
            sftp.get(f"{RUN_DIR}/logits_main.bin", local_logits)
            results["bench"] = res
            if args.profile:
                pargs = argparse.Namespace(**vars(args))
                pargs.reopen = False
                ptext = run_bench(session, remote_proj, "build_prof", pargs, "prof")
                pres = parse(ptext)
                # decode: the profile covers the first prompt's decode steps;
                # prefill_<n>: llm_bench resets it before each repetition, so
                # it holds exactly one call
                results["profile"] = {
                    ph: breakdown(p, layers, per=(args.decode if ph == "decode" else 1))
                    for ph, p in pres["profiles"].items()}
                results["profile_runs"] = {"prompts": pres["prompts"], "prefill": pres["prefill"],
                                           "summary": pres.get("llm_summary")}
            if not args.no_lib_check:
                sftp.put(os.path.join(HERE, "llm_lib_check.py"), f"{RUN_DIR}/llm_lib_check.py")
                out, err, rc = session.exec(f"cd {RUN_DIR} && python3 llm_lib_check.py "
                                            f"{REMOTE_DIR}/lib/libsmollm2.so", timeout=1800)
                line = next((ln for ln in out.splitlines() if ln.startswith("LLM_LIB_CHECK:")), None)
                results["lib_check"] = json.loads(line.split(":", 1)[1]) if line else {
                    "error": (out + err)[-2000:]}
                print(f"  lib check (ctypes): {results['lib_check']}", flush=True)
            sftp.close()
        finally:
            session.close()
    o = res.get("llm_open", {})
    print(f"llm_open {o.get('open_ms')} ms, CmaFree {o.get('cma_free_kb_before')} -> "
          f"{o.get('cma_free_kb_after')} kB "
          f"({(o.get('cma_free_kb_before', 0) - o.get('cma_free_kb_after', 0)) / 1024:.1f} MB)")
    print(f"decode {res.get('llm_summary', {}).get('decode_ms_mean')} ms/token; prefill "
          + ", ".join(f"{p['n']}: {p['best_ms']:.0f} ms" for p in res["prefill"]))
    if "llm_reopen" in res:
        print(f"re-open: {res['llm_reopen']}")
    for ph, bd in results.get("profile", {}).items():
        print(f"profile {ph}: " + ", ".join(f"{k} {v:.2f} ms" for k, v in bd.items()))
    if not args.no_check:
        results["check"] = check_logits(args.project, names, ids, res, local_logits,
                                        args.decode, args.study_json)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    ok = all(r["bit_exact_steps"] == r["steps"] for r in results.get("check", {}).values())
    print(_green("BOARD GATE: logits bit-exact") if ok else _red("BOARD GATE: MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

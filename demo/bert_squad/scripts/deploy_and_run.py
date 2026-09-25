#!/usr/bin/env python3
"""
deploy_and_run.py — run the generated BERT-SQuAD project on the KV260 and
score it.

  weights  -> remote.weights_dir (persistent; only files whose size or
              checksum changed are uploaded — 208 MB the first time)
  upload   -> project sources (no weights) + inputs.bin to remote.work_dir
  cmake    -> -DINFERENCE_TARGET=LINUX -DINFERENCE_WEIGHTS_DIR=<weights_dir>
  make     -> squad_bench (+ test_inference with run.smoke_test)
  smoke    -> the generated test_inference (expects PASSED)
  run      -> squad_bench over the examples; logits.bin downloaded

Meanwhile scripts/reference.py computes, on the host, the float model and
the Q8.8 emulation for every example and the scheduler's simulation for the
first K (reference.bitexact_examples).  The report then gives

  * SQuAD EM / F1 against the gold answers — float, emulation, board —
    and agreement with the float model's spans;
  * the board logits compared BIT FOR BIT with the scheduler simulation
    (first K examples) and with the emulation (every example);
  * latency per inference (mean / min / max) and, with --profile-layers, the
    per-layer profile aggregated by kind (MatMul linears, attention MatMuls,
    VectorOP, LayerNorm, GELU, Softmax, Transpose, other) plus the top layers.

Results: build/results.json, build/logits.bin, logs under build/logs/.

usage:  deploy_and_run.py [--config CFG] [--n N] [--profile-layers] [--no-smoke]
                          [--smoke-only] [--no-reference] [--board-lock FILE]
                          [--no-cleanup] [--check-only] [-v]
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from _common import (BUILD_DIR, PROJECT_SUMMARY, SCRIPTS_DIR, demo_path,
                     import_study, load_config, load_features, load_inputs,
                     preprocessed_dir)
from src.remote import (_bold, _dim, _green, _red, _yellow,  # noqa: E402
                        RemoteSession, check_prerequisites)

LOG_DIR = BUILD_DIR / "logs"
KINDS = ("MatMul linear", "MatMul attention", "VectorOP", "LayerNorm", "GELU",
         "Softmax", "Transpose", "other")
_REQUIRED_DRIVER_HEADERS = {
    "VectorOPKernel": ["xvectoropkernel.h", "xvectoropkernel_hw.h"],
    "MatmulKernel":   ["xmatmulkernel.h", "xmatmulkernel_hw.h"],
    "ConvKernel":     ["xconvkernel.h", "xconvkernel_hw.h"],
}


@dataclass
class StepLog:
    name: str
    ok: bool
    duration: float
    output: str = ""


def _check(label: str, ok: bool, detail: str = "") -> bool:
    tag = _green("OK     ") if ok else _red("MISSING")
    print(f"    {tag} {label}" + (f"  {_dim(detail)}" if detail else ""))
    return ok


def _fmt_step(s: StepLog, extra: str = "") -> str:
    tag = _green("OK") if s.ok else _red("FAIL")
    return f"  {s.name:<8} -> {tag:<13} {s.duration:6.1f}s" + (f"  {extra}" if extra else "")


def _tail(text: str, n: int = 30) -> None:
    lines = (text or "").rstrip().splitlines()
    if len(lines) > n:
        lines = ["..."] + lines[-n:]
    for ln in lines:
        print(_dim(f"      {ln}"))


def _save_log(step: StepLog) -> None:
    if step.output:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{step.name}.log").write_text(step.output)


# ── preflight ────────────────────────────────────────────────────────────────

def preflight_local(cfg: dict, summary: dict, feats: dict) -> bool:
    print(_bold("\nPreflight (local)"))
    ok = True
    proj = Path(summary["project_dir"])
    ok &= _check("project", proj.is_dir(), str(proj))
    for kernel in summary["active"]:
        for h in _REQUIRED_DRIVER_HEADERS.get(kernel, []):
            ok &= _check(f"driver/{h}", (proj / "driver" / h).exists(), kernel)
        ok &= _check(f"remote.uio_devices.{kernel}",
                     kernel in cfg["remote"].get("uio_devices", {}),
                     cfg["remote"].get("uio_devices", {}).get(kernel, ""))
    w = sorted((proj / "weights").glob("*.dat"))
    ok &= _check("weights/*.dat", len(w) == summary["weights"]["files"],
                 f"{len(w)} files, {sum(p.stat().st_size for p in w) / 1e6:.0f} MB")
    inp = preprocessed_dir(cfg) / "inputs.bin"
    ok &= _check("inputs.bin", inp.exists(),
                 f"{feats['n']} examples ({feats['selection']}), seq {feats['seq_len']}")
    ok &= _check("seq_len matches the model", feats["seq_len"] == summary["seq_len"],
                 f"inputs {feats['seq_len']} / model {summary['seq_len']}")
    ok &= _check("ssh.host", bool(cfg.get("ssh", {}).get("host")), cfg.get("ssh", {}).get("host", ""))
    return ok


def preflight_remote(session: RemoteSession, cfg: dict) -> bool:
    print(_bold("\nPreflight (remote)"))
    ok = check_prerequisites(session, cfg, label_width=36)
    for d in (cfg["remote"]["work_dir"], cfg["remote"]["weights_dir"]):
        parent = str(Path(d).parent)
        out, _, rc = session.exec(f"test -w {shlex.quote(parent)} && df -m {shlex.quote(parent)} "
                                  f"| tail -1 | awk '{{print $4\" MB free\"}}'", timeout=15)
        ok &= _check(f"writable {parent}", rc == 0, out.strip())
    out, _, _ = session.exec("grep -E 'CmaFree|CmaTotal' /proc/meminfo | tr -s ' ' | tr '\\n' ' '",
                             timeout=15)
    print(f"    {_dim('info   ')} {out.strip()}")
    return ok


# ── board lock ───────────────────────────────────────────────────────────────

@contextlib.contextmanager
def board_lock(path: Optional[str]):
    if not path:
        yield
        return
    p = Path(path).expanduser()
    with open(p, "a") as f:
        t0 = time.monotonic()
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(_dim(f"waiting for board lock {p} ..."), flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
            print(_dim(f"board lock acquired after {time.monotonic() - t0:.0f} s"), flush=True)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── transfers ────────────────────────────────────────────────────────────────

def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_weights(session: RemoteSession, local_dir: Path, remote_root: str) -> StepLog:
    """Upload weights/*.dat to <remote_root>/weights, skipping files whose
    name and size match the board copy (and, when the board has a manifest
    from an earlier upload, whose checksum matches too)."""
    t0 = time.monotonic()
    remote = f"{remote_root.rstrip('/')}/weights"
    manifest_path = f"{remote}/MANIFEST.json"
    try:
        local = {p.name: (p.stat().st_size, _sha1(p)) for p in sorted(local_dir.glob("*.dat"))}
        session.exec_checked(f"mkdir -p {shlex.quote(remote)}", timeout=15)
        sftp = session._client.open_sftp()          # noqa: SLF001
        try:
            sizes = {a.filename: a.st_size for a in sftp.listdir_attr(remote)}
            try:
                with sftp.open(manifest_path) as f:
                    manifest = json.loads(f.read().decode())
            except (IOError, ValueError):
                manifest = {}
            todo = [n for n, (sz, sh) in local.items()
                    if sizes.get(n) != sz or (n in manifest and manifest[n][1] != sh)]
            nbytes = sum(local[n][0] for n in todo)
            if todo:
                print(f"    uploading {len(todo)}/{len(local)} weight files "
                      f"({nbytes / 1e6:.0f} MB) -> {remote}", flush=True)
            for k, n in enumerate(todo, 1):
                sftp.put(str(local_dir / n), f"{remote}/{n}")
                if k % 10 == 0 or k == len(todo):
                    print(_dim(f"      {k}/{len(todo)}  {time.monotonic() - t0:.0f} s"), flush=True)
            manifest.update({n: list(v) for n, v in local.items()})
            with sftp.open(manifest_path, "w") as f:
                f.write(json.dumps(manifest, indent=0))
        finally:
            sftp.close()
        msg = (f"{len(todo)} uploaded ({nbytes / 1e6:.0f} MB), "
               f"{len(local) - len(todo)} already on the board")
        return StepLog("weights", True, time.monotonic() - t0, msg)
    except Exception as exc:                                  # noqa: BLE001
        return StepLog("weights", False, time.monotonic() - t0, str(exc))


def upload_project(session: RemoteSession, local_proj: Path, remote_proj: str,
                   inputs: Path, remote_data: str) -> StepLog:
    t0 = time.monotonic()
    try:
        session.exec_checked(f"rm -rf {shlex.quote(remote_proj)} {shlex.quote(remote_data)} && "
                             f"mkdir -p {shlex.quote(remote_proj)} {shlex.quote(remote_data)}",
                             timeout=60)
        sftp = session._client.open_sftp()          # noqa: SLF001
        n = 0
        try:
            for p in sorted(local_proj.rglob("*")):
                rel = p.relative_to(local_proj)
                if rel.parts[0] in ("weights", "build"):
                    continue
                dst = f"{remote_proj}/{rel.as_posix()}"
                if p.is_dir():
                    try:
                        sftp.mkdir(dst)
                    except OSError:
                        pass
                else:
                    sftp.put(str(p), dst)
                    n += 1
            sftp.put(str(inputs), f"{remote_data}/inputs.bin")
        finally:
            sftp.close()
        return StepLog("upload", True, time.monotonic() - t0, f"{n} project files + inputs.bin")
    except Exception as exc:                                  # noqa: BLE001
        return StepLog("upload", False, time.monotonic() - t0, str(exc))


# ── build / run ──────────────────────────────────────────────────────────────

def configure_and_build(session: RemoteSession, cfg: dict, remote_proj: str,
                        active: List[str], *, profile: bool, smoke: bool
                        ) -> Tuple[StepLog, StepLog]:
    build = f"{remote_proj}/build"
    uio = cfg["remote"].get("uio_devices", {})
    defs = [f"-DINFERENCE_{k.upper()}_INSTANCE=\\\"{uio[k]}\\\"" for k in active if k in uio]
    cmd = (f"cmake -S {shlex.quote(remote_proj)} -B {shlex.quote(build)} "
           f"-DCMAKE_BUILD_TYPE=Release -DINFERENCE_TARGET=LINUX "
           f"-DINFERENCE_WEIGHTS_DIR={shlex.quote(cfg['remote']['weights_dir'])} "
           f"-DINFERENCE_PROFILING={'ON' if profile else 'OFF'} "
           f"-DINFERENCE_BUILD_TEST={'ON' if smoke else 'OFF'} "
           f"{' '.join(defs)} {' '.join(cfg['remote'].get('cmake_args', []))} 2>&1")
    t0 = time.monotonic()
    out, _, rc = session.exec(cmd, timeout=cfg["build"]["timeout"])
    cm = StepLog("cmake", rc == 0, time.monotonic() - t0, f"$ {cmd}\n{out}")
    if rc != 0:
        return cm, StepLog("make", False, 0.0, "skipped (cmake failed)")
    targets = "squad_bench" + (" test_inference" if smoke else "")
    cmd = f"make -C {shlex.quote(build)} -j{cfg['build']['jobs']} {targets} 2>&1"
    t0 = time.monotonic()
    out, _, rc = session.exec(cmd, timeout=cfg["build"]["timeout"])
    return cm, StepLog("make", rc == 0, time.monotonic() - t0, f"$ {cmd}\n{out}")


def _stream_exec(session: RemoteSession, cmd: str, *, timeout: int,
                 on_stderr_line) -> Tuple[str, str, int]:
    chan = session._client.get_transport().open_session()   # noqa: SLF001
    chan.exec_command(cmd)
    out, err, partial = [], [], ""
    deadline = time.monotonic() + timeout
    while True:
        busy = False
        while chan.recv_ready():
            out.append(chan.recv(65536).decode("utf-8", "replace"))
            busy = True
        while chan.recv_stderr_ready():
            chunk = chan.recv_stderr(65536).decode("utf-8", "replace")
            err.append(chunk)
            partial += chunk
            while "\n" in partial:
                line, partial = partial.split("\n", 1)
                on_stderr_line(line)
            busy = True
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if time.monotonic() > deadline:
            chan.close()
            raise TimeoutError(f"timed out after {timeout} s: {cmd}")
        if not busy:
            time.sleep(0.05)
    if partial:
        on_stderr_line(partial)
    return "".join(out), "".join(err), chan.recv_exit_status()


def _echo(line: str) -> None:
    line = line.rstrip("\r")
    if line:
        sys.stdout.write("    " + line + "\n")
        sys.stdout.flush()


def run_smoke(session: RemoteSession, cfg: dict, remote_proj: str) -> StepLog:
    sudo = "sudo -n " if cfg["run"].get("use_sudo", True) else ""
    cmd = f"cd {shlex.quote(remote_proj)} && {sudo}./build/test_inference"
    t0 = time.monotonic()
    try:
        out, err, rc = session.exec(cmd, timeout=cfg["run"]["timeout"])
    except TimeoutError as exc:
        return StepLog("smoke", False, time.monotonic() - t0, str(exc))
    text = out + err
    return StepLog("smoke", rc == 0 and "PASSED" in text, time.monotonic() - t0, text)


def run_bench(session: RemoteSession, cfg: dict, remote_proj: str, remote_data: str,
              n: int, warmup: int) -> Tuple[StepLog, Optional[dict]]:
    sudo = "sudo -n " if cfg["run"].get("use_sudo", True) else ""
    env = " ".join(f"{shlex.quote(str(k))}={shlex.quote(str(v))}"
                   for k, v in cfg["run"].get("env", {}).items() if not str(k).startswith("_"))
    cmd = (f"cd {shlex.quote(remote_data)} && {sudo}{'env ' + env + ' ' if env else ''}"
           f"{shlex.quote(remote_proj + '/build/squad_bench')} -i inputs.bin -o logits.bin "
           f"-n {n} -w {warmup}")
    t0 = time.monotonic()
    try:
        out, err, rc = _stream_exec(session, cmd, timeout=cfg["run"]["timeout"],
                                    on_stderr_line=_echo)
    except TimeoutError as exc:
        return StepLog("run", False, time.monotonic() - t0, str(exc)), None
    dur = time.monotonic() - t0
    full = f"$ {cmd}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    if rc != 0:
        return StepLog("run", False, dur, full), None
    metrics = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            metrics = json.loads(line)
        for marker, key in (("LAYERS_JSON:", "layer_stats"), ("DDR_JSON:", "ddr_stats")):
            if line.startswith(marker) and metrics is not None:
                try:
                    metrics[key] = json.loads(line[len(marker):].strip())
                except json.JSONDecodeError:
                    metrics[key] = None
    if metrics is None:
        return StepLog("run", False, dur, "no JSON summary\n" + full), None
    return StepLog("run", True, dur, full), metrics


def download(session: RemoteSession, remote: str, local: Path) -> None:
    sftp = session._client.open_sftp()          # noqa: SLF001
    try:
        sftp.get(remote, str(local))
    finally:
        sftp.close()


# ── scoring ──────────────────────────────────────────────────────────────────

def decode_all(bs, feats: dict, logits: np.ndarray) -> List[dict]:
    """logits [N, 2, seq] (float values) -> best spans with SQuAD scores."""
    out = []
    for ex, (st, en) in zip(feats["examples"], logits):
        f = {"ctx_off": ex["ctx_off"], "n_ctx": ex["n_ctx"],
             "sub_to_orig": ex["sub_to_orig"], "doc": ex["doc"]}
        (s, e), text = bs.best_span(f, np.asarray(st, np.float64), np.asarray(en, np.float64))
        golds = ex["answers"]
        out.append({"span": [int(s), int(e)], "text": text,
                    "em": max(bs.em(text, g) for g in golds),
                    "f1": max(bs.f1(text, g) for g in golds)})
    return out


def score(preds: List[dict], ref: Optional[List[dict]] = None, bs=None) -> dict:
    n = len(preds)
    r = {"em": 100.0 * sum(p["em"] for p in preds) / n,
         "f1": 100.0 * sum(p["f1"] for p in preds) / n}
    if ref is not None:
        r["same_span"] = sum(p["span"] == q["span"] for p, q in zip(preds, ref))
        r["em_vs"] = 100.0 * sum(bs.em(p["text"], q["text"]) for p, q in zip(preds, ref)) / n
        r["f1_vs"] = 100.0 * sum(bs.f1(p["text"], q["text"]) for p, q in zip(preds, ref)) / n
    return r


def bit_compare(board: np.ndarray, ref: np.ndarray) -> List[dict]:
    res = []
    for b, r in zip(board, ref):
        d = np.abs(b.astype(np.int32) - r.astype(np.int32))
        res.append({"start_diff": int(np.count_nonzero(d[0])), "end_diff": int(np.count_nonzero(d[1])),
                    "max_lsb": int(d.max()), "exact": bool(not d.any())})
    return res


def kind_breakdown(layer_stats: dict, layers: List[dict], n_runs: int) -> Tuple[dict, List[dict]]:
    """Per-kind per-inference time (ms) from the profiler's total_us."""
    by_i = {L["i"]: L for L in layers}
    kinds = {k: {"ms": 0.0, "layers": 0} for k in KINDS}
    rows = []
    for L in layer_stats.get("layers", []):
        meta = by_i.get(L["i"], {"kind": "other", "op": "?", "engine": "?"})
        ms = L.get("total_us", 0.0) / 1000.0 / max(n_runs, 1)
        k = kinds[meta["kind"]]
        k["ms"] += ms
        k["layers"] += 1 if L.get("calls", 0) else 0
        rows.append({"i": L["i"], "name": L["name"], "kind": meta["kind"], "op": meta["op"],
                     "engine": meta["engine"], "ms": ms, "nkm": meta.get("nkm")})
    rows.sort(key=lambda x: -x["ms"])
    return kinds, rows


# ── report ───────────────────────────────────────────────────────────────────

def print_report(res: dict, top: int) -> None:
    print(_bold("\n  ── BERT-SQuAD KV260 ──\n"))
    m = res.get("metrics") or {}
    if m:
        print(f"  examples {m['examples']} (warm-up {m['warmup']}), seq {m['seq_len']}:  "
              f"latency mean {m['mean_ms']:.1f} ms  min {m['min_ms']:.1f}  max {m['max_ms']:.1f}  "
              f"({1000.0 / m['mean_ms']:.3f} inferences/s);  inference_init {m['init_ms'] / 1000:.1f} s;  "
              f"unique_ids passed through {m['uid_ok']}/{m['examples']}")
    acc = res.get("accuracy") or {}
    if acc:
        print(f"\n  {'':12s} {'EM':>6s} {'F1':>6s}   {'same span as float':>18s}   "
              f"{'EM / F1 vs float answer':>23s}")
        for name in ("float", "emulation", "board"):
            a = acc.get(name)
            if not a:
                continue
            same = f"{a['same_span']}/{acc['n']}" if "same_span" in a else "-"
            vs = f"{a['em_vs']:.1f} / {a['f1_vs']:.1f}" if "em_vs" in a else "-"
            print(f"  {name:12s} {a['em']:6.1f} {a['f1']:6.1f}   {same:>18s}   {vs:>23s}")
    be = res.get("bitexact") or {}
    if be.get("sim"):
        s = be["sim"]
        n_ok = sum(x["exact"] for x in s)
        tag = _green("BIT-EXACT") if n_ok == len(s) else _red("MISMATCH")
        print(f"\n  board vs scheduler simulation (first {len(s)}): {tag}  {n_ok}/{len(s)} examples")
        for i, x in enumerate(s):
            if not x["exact"]:
                print(f"    #{i}: start {x['start_diff']} / end {x['end_diff']} values differ, "
                      f"max {x['max_lsb']} LSB")
    if be.get("emu"):
        e = be["emu"]
        n_ok = sum(x["exact"] for x in e)
        print(f"  board vs emulation (all {len(e)}): {n_ok}/{len(e)} examples bit-exact"
              + ("" if n_ok == len(e) else
                 f"; {sum(x['start_diff'] + x['end_diff'] for x in e)} values differ, "
                 f"max {max(x['max_lsb'] for x in e)} LSB"))
    kb = res.get("profile")
    if kb:
        tot = sum(v["ms"] for v in kb["kinds"].values())
        print(f"\n  per-layer profile (per inference, warm-up excluded; sum {tot:.0f} ms vs "
              f"wall {m.get('mean_ms', 0):.0f} ms — kernel windows are issue -> wait and can "
              f"overlap host work):")
        for k in KINDS:
            v = kb["kinds"][k]
            if v["layers"] == 0 and v["ms"] == 0:
                continue
            print(f"    {k:18s} {v['ms']:9.1f} ms  {100 * v['ms'] / tot if tot else 0:5.1f} %"
                  f"   ({v['layers']} layers)")
        print(f"  top {top} layers:")
        for r in kb["top"][:top]:
            nkm = f"  n,k,m,b={r['nkm']}" if r.get("nkm") else ""
            print(f"    [{r['i']:3d}] {r['kind']:17s} {r['name'][:48]:48s} {r['ms']:8.1f} ms{nkm}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def start_reference(cfg_path: Optional[str], n: int) -> Optional[subprocess.Popen]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPTS_DIR / "reference.py"), "--n", str(n)]
    if cfg_path:
        cmd += ["--config", cfg_path]
    log = open(LOG_DIR / "reference.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=0, help="examples to run (0 = all in inputs.bin)")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--profile-layers", action="store_true",
                    help="build with INFERENCE_PROFILING and report the per-layer breakdown")
    ap.add_argument("--top", type=int, default=15, help="layers listed with --profile-layers")
    ap.add_argument("--no-smoke", action="store_true", help="skip test_inference")
    ap.add_argument("--smoke-only", action="store_true", help="run test_inference only")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the host reference (no bit-exact check / float comparison)")
    ap.add_argument("--board-lock", default=None, help="flock() this file for the board session")
    ap.add_argument("--no-cleanup", action="store_true", help="keep remote.work_dir")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run = cfg.setdefault("run", {})
    profile = args.profile_layers or bool(run.get("profile_layers", False))
    smoke = (bool(run.get("smoke_test", True)) and not args.no_smoke) or args.smoke_only
    warmup = args.warmup if args.warmup is not None else int(run.get("warmup", 1))
    if not PROJECT_SUMMARY.exists():
        print(f"error: {PROJECT_SUMMARY} not found — run scripts/generate_project.py first",
              file=sys.stderr)
        return 1
    summary = json.loads(PROJECT_SUMMARY.read_text())
    feats = load_features(cfg)
    n = min(args.n or feats["n"], feats["n"])
    if not preflight_local(cfg, summary, feats) and not args.check_only:
        print(_red("\npreflight: local prerequisites missing"))
        return 1

    ref_proc = None
    if not (args.no_reference or args.check_only or args.smoke_only):
        ref_proc = start_reference(args.config, n)
        print(_dim(f"\nhost reference started in the background (log: {LOG_DIR / 'reference.log'})"))

    work = cfg["remote"]["work_dir"].rstrip("/")
    remote_proj, remote_data = f"{work}/project", f"{work}/data"
    local_logits = BUILD_DIR / "logits.bin"
    steps: List[StepLog] = []
    metrics = None
    ok = True

    def record(step: StepLog, extra: str = "") -> bool:
        steps.append(step)
        _save_log(step)
        print(_fmt_step(step, extra or (step.output if step.name in ("weights", "upload") else "")))
        if not step.ok or (args.verbose and step.name in ("cmake", "make")):
            _tail(step.output)
        return step.ok

    with board_lock(args.board_lock or cfg.get("board_lock")):
        session = RemoteSession(cfg["ssh"])
        print(f"\n{_bold('Connecting')} to {cfg['ssh']['user']}@{cfg['ssh']['host']}:"
              f"{cfg['ssh'].get('port', 22)} ...")
        try:
            session.connect()
        except Exception as exc:                              # noqa: BLE001
            print(_red(f"  connection failed: {exc}"))
            if ref_proc:
                ref_proc.terminate()
            return 1
        try:
            if not preflight_remote(session, cfg):
                print(_red("\npreflight: remote prerequisites missing"))
                return 1
            if args.check_only:
                print(_green("\npreflight: all checks passed"))
                return 0
            print(_bold(f"\nDeploy (profile={'on' if profile else 'off'}, "
                        f"smoke test={'on' if smoke else 'off'})"))
            ok = record(sync_weights(session, Path(summary["project_dir"]) / "weights",
                                     cfg["remote"]["weights_dir"]))
            ok = ok and record(upload_project(session, Path(summary["project_dir"]), remote_proj,
                                              preprocessed_dir(cfg) / "inputs.bin", remote_data))
            if ok:
                cm, mk = configure_and_build(session, cfg, remote_proj, summary["active"],
                                             profile=profile, smoke=smoke)
                ok = record(cm) and record(mk)
            if ok and smoke:
                st = run_smoke(session, cfg, remote_proj)
                verdict = "PASSED" if st.ok else "FAILED"
                ok = record(st, f"test_inference {verdict}")
                if args.verbose or not st.ok:
                    _tail(st.output, 20)
            if ok and not args.smoke_only:
                print(_bold(f"\nsquad_bench: {n} examples, warm-up {warmup}"))
                st, metrics = run_bench(session, cfg, remote_proj, remote_data, n, warmup)
                ok = record(st)
                if ok:
                    download(session, f"{remote_data}/logits.bin", local_logits)
            if cfg.get("cleanup", True) and not args.no_cleanup:
                session.exec(f"rm -rf {shlex.quote(work)}", timeout=60)
                print(_dim(f"cleanup {work} (weights kept in {cfg['remote']['weights_dir']})"))
        finally:
            session.close()

    res = {"config": {"n": n, "warmup": warmup, "profile": profile,
                      "selection": feats["selection"]},
           "steps": [{"name": s.name, "ok": s.ok, "duration_s": round(s.duration, 1)} for s in steps],
           "metrics": metrics}
    if not ok or metrics is None:
        if ref_proc:
            ref_proc.terminate()
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        (BUILD_DIR / "results.json").write_text(json.dumps(res, indent=1))
        if args.smoke_only and ok:
            print(_green("\nsmoke test passed"))
            return 0
        print(_red("\nfailed — see the step logs under " + str(LOG_DIR)))
        return 1

    # ── score ──
    seq = feats["seq_len"]
    board = np.fromfile(local_logits, dtype="<i2").reshape(-1, 2, seq)[:n]
    bs = import_study(cfg)
    ex = feats["examples"][:n]
    fsub = dict(feats, examples=ex)
    pred_board = decode_all(bs, fsub, board / 256.0)
    acc = {"n": n, "board": score(pred_board)}
    bitexact = {}
    ref = None
    if ref_proc is not None:
        print(_dim("waiting for the host reference ..."), flush=True)
        rc = ref_proc.wait()
        from reference import load_reference
        ref = load_reference(feats["inputs_sha1"]) if rc == 0 else None
        if ref is None:
            print(_yellow(f"host reference failed (rc={rc}) — see {LOG_DIR / 'reference.log'}"))
    if ref is not None:
        if "float" in ref and len(ref["float"]) >= n:
            pred_float = decode_all(bs, fsub, ref["float"][:n])
            pred_emu = decode_all(bs, fsub, ref["emu"][:n] / 256.0)
            acc["float"] = score(pred_float)
            acc["emulation"] = score(pred_emu, pred_float, bs)
            acc["board"] = score(pred_board, pred_float, bs)
            bitexact["emu"] = bit_compare(board, ref["emu"][:n])
            for p, q, r in zip(pred_board, pred_float, pred_emu):
                p["float_text"], p["emu_text"] = q["text"], r["text"]
        k = min(int(ref["k"]), n)
        if k:
            bitexact["sim"] = bit_compare(board[:k], ref["sim"][:k])
    res.update(accuracy=acc, bitexact=bitexact,
               predictions=[dict(p, qid=e["qid"], question=e["question"], answers=e["answers"])
                            for p, e in zip(pred_board, ex)])
    if metrics.get("layer_stats"):
        layers = json.loads((Path(summary["project_dir"]) / "layers.json").read_text())
        kinds, rows = kind_breakdown(metrics["layer_stats"], layers, metrics["examples"])
        res["profile"] = {"kinds": kinds, "top": rows}
    (BUILD_DIR / "results.json").write_text(json.dumps(res, indent=1))
    print_report(res, args.top)
    print(_dim(f"results: {BUILD_DIR / 'results.json'}   logits: {local_logits}   logs: {LOG_DIR}"))
    sim_ok = all(x["exact"] for x in bitexact.get("sim", []))
    return 0 if sim_ok else 3


if __name__ == "__main__":
    sys.exit(main())

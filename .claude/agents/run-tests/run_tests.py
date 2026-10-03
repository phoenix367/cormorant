#!/usr/bin/env python3
"""
run_tests.py — run the project's test suites and report a structured status
(JSON on stdout) for the run-tests agent (.claude/agents/run-tests.md).

Suites (--suite, comma-separated; default: scheduler,chat,lint,facts):
  scheduler  inference-scheduler pytest (test/)                     ~3.5 min
  chat       demo/chat/tests with pytest                            ~1 min
  lint       ruff check of inference-scheduler/ (the CI lint)        seconds
  facts      tools/facts: facts.yaml checked against code and docs (stale counts,
             register maps, supported ops, CLI and script flags, HTTP routes,
             ctypes, config keys, pool sizes, board results) + the tool's own
             unittest                                                ~15 s
  csim       kernel C simulation: make + ctest in build/ (Vitis HLS headers;
             Verilator for the RTL MatmulKernel's TestMatmulRtl)  ~3-6 min
  tts-host   demo/tts/scripts/tts_host_emu.py (+ --lib-check): the Piper library's
             generated C on the host vs the specifications (needs the voice assets
             and demo/tts/build/piper_project)                       ~2 min
  rtl        make behavior_test in build/ (Vivado xsim; re-synthesises; LONG, and
             it modifies tracked files of the hw/ submodules)       ~1 h
  all        scheduler,chat,lint,facts,csim,tts-host (not rtl)
Board suites (run_remote_tests.py, run_remote_perf.py, tts_board.py, llm_board.py)
are never run here: the board is shared and the chat server owns the FPGA.

Selections: --tests PATH[::NODE] ... (pytest suites: the given files / node ids
instead of the whole suite), -k EXPR (pytest -k), --pytest-args "...".

Per suite the JSON holds the command, exit code, duration, counts, every failure
(id, file:line, message, E lines, traceback excerpt, failed subtests), skips
(reason, expected or not), xfails, warnings grouped by category + message (source
project / third_party, known = in the baseline), anomalies (fewer tests than the
baseline, unexpected skips, a run much shorter than usual, no tests, odd exit
codes, XPASS) and a status: fail (failures, errors, timeouts, broken runs) >
warn (anomalies, new project warnings) > not_run (prerequisite missing) > pass.
The overall status is the worst one.  Full logs and JUnit XML stay in --out-dir.

  The report is always saved as <out-dir>/report.json (its path is the
  report's "report" key); --json FILE writes a copy.
  --detach           run in the background: prints {"pid", "json", "log"} and
                     writes the report to "json" when done (wait with
                     `timeout 590 tail --pid=PID -f /dev/null`, then read it)
  --record-baseline  store this run's counts / skips / duration as the suite's
                     baseline (baselines.json) — only for a whole-suite run
                     without failures
  --list             print the suites and baselines

usage: python3 .claude/agents/run-tests/run_tests.py [--suite S,...] [--tests ...] [-k EXPR]
                                                     [--out-dir DIR] [--json FILE] [--detach]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCHED = REPO / "inference-scheduler"
VENV_PY = SCHED / ".venv" / "bin" / "python"
RUFF = SCHED / ".venv" / "bin" / "ruff"
BUILD = REPO / "build"
VITIS = os.environ.get("VITIS_SETTINGS", "/mnt/data/xilinx/2025.2/Vitis/settings64.sh")
BASELINES = HERE / "baselines.json"
FACTS = REPO / "tools" / "facts"

DEFAULT = ("scheduler", "chat", "lint", "facts")
ALL = ("scheduler", "chat", "lint", "facts", "csim", "tts-host")
KNOWN = ALL + ("rtl",)
RANK = {"pass": 0, "not_run": 1, "warn": 2, "fail": 3}
TB_LINES = 40


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_baselines() -> dict:
    try:
        return json.loads(BASELINES.read_text())
    except (OSError, ValueError):
        return {}


def run(cmd, cwd: Path, logfile: Path, timeout: int, shell: bool = False, env=None):
    """(exit code, seconds, output); the output also goes to logfile."""
    t0 = time.monotonic()
    env = {k: v for k, v in dict(os.environ, **(env or {})).items() if k not in ("FORCE_COLOR", "PY_COLORS")}
    env.setdefault("NO_COLOR", "1")
    with open(logfile, "w") as f:
        f.write(f"$ {cmd if shell else shlex.join(cmd)}\n# cwd {cwd}\n")
        f.flush()
        try:
            p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               shell=shell, executable="/bin/bash" if shell else None,
                               env=env)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    dt = time.monotonic() - t0
    return rc, dt, plain(logfile.read_text(errors="replace"))


ANSI = re.compile(r"(?:\x1b|#x1B)\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    return ANSI.sub("", text or "")


def tail(text: str, n: int) -> str:
    return "\n".join(text.rstrip().splitlines()[-n:])


def base_result(name: str, cmd, cwd: Path) -> dict:
    return {"suite": name, "command": cmd if isinstance(cmd, str) else shlex.join(str(c) for c in cmd),
            "cwd": str(cwd), "exit_code": None, "duration_s": None,
            "counts": {}, "failures": [], "skips": [], "warnings": [], "anomalies": [],
            "notes": [], "log": None, "status": "pass"}


def not_run(r: dict, why: str, fix: str = "") -> dict:
    r["status"] = "not_run"
    r["anomalies"].append(why + (f" (fix: {fix})" if fix else ""))
    return r


# ---- pytest ---------------------------------------------------------------- #

def _skip_expected(reason: str, allow) -> bool:
    return any(re.search(p, reason) for p in allow)


def node_id(tc) -> str:
    """The pytest node id (relative to the suite's cwd) from an xunit1 testcase."""
    f, cls, name = tc.get("file"), tc.get("classname", ""), tc.get("name", "")
    if not f:
        return f"{cls}::{name}" if cls else name
    mod = f[:-3].replace("/", ".") if f.endswith(".py") else f
    rest = cls[len(mod):].lstrip(".") if cls.startswith(mod) else ""
    return "::".join([f, *(rest.split(".") if rest else []), name])


def parse_junit(xml_path: Path, r: dict, allow) -> None:
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as e:
        r["anomalies"].append(f"no JUnit report ({e}): the run did not finish")
        return
    c = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for tc in root.iter("testcase"):
        c["tests"] += 1
        tid = node_id(tc)
        loc = f"{tc.get('file')}:{int(tc.get('line')) + 1}" if tc.get("file") and tc.get("line") else tc.get("file")
        kid = next((k for k in tc if k.tag in ("failure", "error", "skipped")), None)
        if kid is None:
            c["passed"] += 1
            continue
        msg = plain(kid.get("message") or "").strip()
        text = plain(kid.text or "").strip()
        if kid.tag == "skipped":
            if "xfail" in (kid.get("type") or ""):
                c["xfailed"] = c.get("xfailed", 0) + 1
                r.setdefault("xfails", []).append({"id": tid, "reason": msg})
                continue
            c["skipped"] += 1
            reason = re.sub(r"^\S+:\d+: ", "", msg)
            r["skips"].append({"id": tid, "reason": reason, "expected": _skip_expected(reason, allow)})
            continue
        c["failed" if kid.tag == "failure" else "errors"] += 1
        kind = kid.tag if kid.tag == "failure" else "error (setup/teardown)"
        if msg == "collection failure":
            kind, tid = "collection_error", tc.get("file") or tid
        e_lines = [ln for ln in (text + "\n" + msg).splitlines() if re.match(r"\s*E\s", ln)]
        r["failures"].append({"id": tid, "kind": kind,
                              "location": loc, "message": msg[:1500],
                              "assertion": "\n".join(dict.fromkeys(ln.strip() for ln in e_lines[:30])),
                              "traceback": tail(text, TB_LINES), "time_s": float(tc.get("time") or 0)})
    r["counts"].update(c)


def parse_pytest_output(out: str, r: dict) -> None:
    """The summary line, subtest failures and the warnings summary (JUnit has neither)."""
    found = re.findall(r"^(?:=+ )?(\d+ (?:passed|failed|skipped|errors?|deselected|xfailed|xpassed|warnings?|"
                       r"no tests ran).*?) in ([\d.]+)s(?: \([\d:]+\))?(?: =+)?$", out, re.M)
    if found:
        r["counts"]["summary"] = found[-1][0]
        for num, what in re.findall(r"(\d+) (subtests passed|subtests failed|warnings?|deselected|xpassed|rerun)",
                                    found[-1][0]):
            r["counts"]["warnings" if what.startswith("warning") else what.replace(" ", "_")] = int(num)
    by_name = {f["id"].split("::")[-1]: f for f in r["failures"]}
    for kind, params, tid, msg in re.findall(r"^(SUBFAILED|FAILED|ERROR)(\(.*?\))? (\S+)(?: - (.*))?$", out, re.M):
        f = by_name.get(tid.split("::")[-1])
        if "::" not in tid and any(x["kind"] == "collection_error" for x in r["failures"]):
            continue                                          # the file's collection error, from JUnit
        if kind == "SUBFAILED" and f is not None:
            f.setdefault("failed_subtests", []).append(params.strip("()"))
        elif f is None:
            r["failures"].append({"id": tid + (f" [{params.strip('()')}]" if params else ""), "kind": kind.lower(),
                                  "location": None, "message": (msg or "")[:1000], "assertion": "",
                                  "traceback": "", "time_s": None})
    sec = re.search(r"^=+ warnings summary =+$\n(.*?)(?=^=+ |\Z)", out, re.M | re.S)
    if not sec:
        return
    groups, where = {}, None
    for ln in sec.group(1).splitlines():
        if not ln.strip() or ln.startswith("-- Docs"):
            continue
        if not ln.startswith(" "):
            where = ln.strip()
            continue
        w = re.match(r"\s+(\S+?):(\d+): (\w+(?:Warning|Error)?): (.*)", ln)
        if w:
            key = (w.group(3), w.group(4)[:300])
            g = groups.setdefault(key, {"category": w.group(3), "message": w.group(4)[:300],
                                        "location": f"{w.group(1)}:{w.group(2)}", "tests": [], "count": 0})
            g["count"] += 1
            if where and len(g["tests"]) < 5:
                g["tests"].append(where)
    r["warnings"] += list(groups.values())


def check_baseline(r: dict, bl: dict, whole: bool) -> None:
    c = r["counts"]
    if not whole or not bl:
        return
    n = c.get("tests", c.get("checks", 0))
    if n < bl.get("min_tests", 0):
        r["anomalies"].append(f"fewer tests than the baseline: {n} < {bl['min_tests']} "
                              f"(tests not collected or skipped by collection?)")
    if bl.get("max_skipped") is not None and c.get("skipped", 0) > bl["max_skipped"]:
        r["anomalies"].append(f"more skips than the baseline: {c.get('skipped', 0)} > {bl['max_skipped']}")
    d, typ = r["duration_s"] or 0.0, bl.get("typical_duration_s")
    if typ and d < 0.4 * typ:
        r["anomalies"].append(f"run much shorter than usual: {d:.0f} s vs typically {typ:.0f} s "
                              f"(tests skipped, not collected or cut short?)")
    elif typ and d > 2.5 * typ:
        r["notes"].append(f"run much slower than usual: {d:.0f} s vs typically {typ:.0f} s (machine load?)")


def run_pytest(name: str, a, out_dir: Path, bls: dict) -> dict:
    cfg = {"scheduler": (SCHED, ["test/"]), "chat": (REPO, ["demo/chat/tests"])}[name]
    cwd, targets = cfg
    if a.tests:
        targets = a.tests
    xml = out_dir / f"{name}.junit.xml"
    cmd = [str(VENV_PY), "-m", "pytest", *targets, "-q", "-rfEs", "--color=no", "-p", "no:cacheprovider",
           "-o", "junit_family=xunit1", f"--junitxml={xml}", "--durations=10"]
    if a.k:
        cmd += ["-k", a.k]
    if a.pytest_args:
        cmd += shlex.split(a.pytest_args)
    r = base_result(name, cmd, cwd)
    if not VENV_PY.exists():
        return not_run(r, f"no scheduler virtualenv at {VENV_PY}",
                       "cd inference-scheduler && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    bl = bls.get(name, {})
    log(f"[{name}] {r['command']}")
    rc, dt, out = run(cmd, cwd, out_dir / f"{name}.log", a.timeout or 3000)
    r.update(exit_code=rc, duration_s=round(dt, 1), log=str(out_dir / f"{name}.log"))
    parse_junit(xml, r, bl.get("skip_allow", []))
    parse_pytest_output(out, r)
    whole = not (a.tests or a.k or a.pytest_args)
    check_baseline(r, bl, whole)
    if r["counts"].get("xpassed"):
        r["anomalies"].append(f"{r['counts']['xpassed']} xfail-marked test(s) passed unexpectedly (XPASS)")
    for s in r["skips"]:
        if not s["expected"]:
            r["anomalies"].append(f"unexpected skip: {s['id']} — {s['reason']}")
    if "models" in " ".join(s["reason"] for s in r["skips"]).lower() and \
            any("gen_" in s["reason"] for s in r["skips"] if not s["expected"]):
        r["notes"].append("skips name missing test models: run .venv/bin/python test/gen_all_models.py "
                          "in inference-scheduler/")
    if rc == "timeout":
        r["anomalies"].append(f"timed out after {a.timeout or 3000} s")
    elif rc not in (0, 1):
        why = {2: "interrupted / collection errors", 3: "internal error", 4: "usage error",
               5: "no tests collected"}.get(rc, "unexpected exit code")
        r["anomalies"].append(f"pytest exit code {rc}: {why}; log tail:\n{tail(out, 15)}")
    elif rc == 1 and not r["failures"]:
        r["anomalies"].append(f"pytest exit code 1 without a parsed failure; log tail:\n{tail(out, 15)}")
    if r["counts"].get("tests", 0) == 0 and rc != "timeout":
        r["anomalies"].append("no tests ran")
    return r


# ---- ruff ------------------------------------------------------------------ #

def run_lint(a, out_dir: Path, bls: dict) -> dict:
    cmd = [str(RUFF), "check", ".", "--output-format=json"]
    r = base_result("lint", cmd, SCHED)
    if not RUFF.exists():
        return not_run(r, "ruff is not installed in the scheduler venv",
                       "inference-scheduler/.venv/bin/pip install 'ruff>=0.15,<0.16'")
    log(f"[lint] {r['command']}")
    rc, dt, out = run(cmd, SCHED, out_dir / "lint.log", 300)
    r.update(exit_code=rc, duration_s=round(dt, 1), log=str(out_dir / "lint.log"))
    body = out.split("\n", 2)[2] if out.count("\n") >= 2 else ""
    try:
        items = json.loads(body) if body.strip() else []
    except ValueError:
        items = None
    if items is None:
        r["anomalies"].append(f"ruff output not JSON; log tail:\n{tail(out, 15)}")
        items = []
    for v in items:
        loc = v.get("location") or {}
        r["failures"].append({"id": v.get("code"), "kind": "lint",
                              "location": f"{os.path.relpath(v.get('filename', ''), REPO)}:{loc.get('row')}",
                              "message": v.get("message", ""), "assertion": "", "traceback": "",
                              "time_s": None})
    r["counts"] = {"violations": len(items)}
    if rc not in (0, 1):
        r["anomalies"].append(f"ruff exit code {rc}")
    return r


# ---- facts ------------------------------------------------------------------ #

def run_facts(a, out_dir: Path, bls: dict) -> dict:
    """facts.py check --json: a failing fact's errors are failures, its warnings
    anomalies; then the tool's own tests."""
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    cmd = [py, str(FACTS / "facts.py"), "check", "--json"]
    r = base_result("facts", cmd, REPO)
    if subprocess.run([py, "-c", "import yaml"], capture_output=True).returncode:
        return not_run(r, f"{py} has no PyYAML", "inference-scheduler/.venv/bin/pip install -r "
                                                 "inference-scheduler/requirements.txt")
    log(f"[facts] {r['command']}")
    rc, dt, out = run(cmd, REPO, out_dir / "facts.log", a.timeout or 900)
    body = out.split("\n", 2)[2] if out.count("\n") >= 2 else ""
    try:
        facts = json.loads(body[body.find("["):]) if "[" in body else None
    except ValueError:
        facts = None
    if facts is None:
        r["anomalies"].append(f"exit code {rc}: facts.py output not JSON; log tail:\n{tail(out, 15)}")
        facts = []
    for f in facts:
        for x in f["findings"]:
            if x["level"] == "error":
                r["failures"].append({"id": f["id"], "kind": "fact", "location": x["where"] or None,
                                      "message": x["msg"], "assertion": "", "traceback": "", "time_s": None})
            elif x["level"] == "warn":
                r["anomalies"].append(f"{f['id']}: {x['where'] + ': ' if x['where'] else ''}{x['msg']}")
    st = [f["status"] for f in facts]
    r["counts"] = {"facts": len(facts), "ok": st.count("ok"), "failed": st.count("fail"),
                   "warned": st.count("warn"), "skipped": st.count("skipped")}
    ucmd = [py, "-m", "unittest", str(FACTS / "test_facts.py")]
    rc2, dt2, out2 = run(ucmd, REPO, out_dir / "facts_selftest.log", 300)
    m = re.search(r"Ran (\d+) tests?", out2)
    r["counts"]["self_tests"] = int(m.group(1)) if m else 0
    if rc2 != 0:
        r["failures"].append({"id": "test_facts.py", "kind": "self_test", "location": "tools/facts/test_facts.py",
                              "message": f"exit {rc2}", "assertion": "", "traceback": tail(out2, 30),
                              "time_s": None})
    r.update(exit_code=rc if rc2 == 0 else rc2, duration_s=round(dt + dt2, 1), log=str(out_dir / "facts.log"))
    return r


# ---- ctest ------------------------------------------------------------------ #

def run_csim(a, out_dir: Path, bls: dict) -> dict:
    xml = out_dir / "ctest.junit.xml"
    cmd = (f"source {shlex.quote(VITIS)} >/dev/null 2>&1 && make -j8 2>&1 && "
           f"ctest --output-on-failure --output-junit {shlex.quote(str(xml))}")
    r = base_result("csim", cmd, BUILD)
    if not (BUILD / "CMakeCache.txt").exists():
        return not_run(r, "no configured build/ directory", "mkdir build && cd build && cmake ..")
    if not os.path.exists(VITIS):
        return not_run(r, f"Vitis settings not found at {VITIS}", "set VITIS_SETTINGS=<Vitis>/settings64.sh")
    log(f"[csim] {cmd}")
    rc, dt, out = run(cmd, BUILD, out_dir / "csim.log", a.timeout or 3600, shell=True)
    r.update(exit_code=rc, duration_s=round(dt, 1), log=str(out_dir / "csim.log"))
    errs = re.findall(r"^(\S+:\d+:\d+: (?:fatal )?error: .*)$", out, re.M)
    warns = re.findall(r"^(\S+:\d+:\d+: warning: .*)$", out, re.M)
    for e in errs[:30]:
        r["failures"].append({"id": "build", "kind": "compile_error", "location": e.split(": ")[0],
                              "message": e, "assertion": "", "traceback": "", "time_s": None})
    groups = {}
    for w in warns:
        loc, _, msg = w.partition(": warning: ")
        g = groups.setdefault(msg, {"category": "compiler warning", "message": msg[:300], "location": loc,
                                    "tests": [], "count": 0})
        g["count"] += 1
    r["warnings"] += list(groups.values())
    try:
        root = ET.parse(xml).getroot()
        c = {"tests": 0, "passed": 0, "failed": 0, "not_run": 0}
        for tc in root.iter("testcase"):
            c["tests"] += 1
            st = tc.get("status", "")
            kid = next((k for k in tc if k.tag in ("failure", "skipped", "error")), None)
            if kid is None and st in ("run", ""):
                c["passed"] += 1
                continue
            key = "not_run" if (st == "notrun" or (kid is not None and kid.tag == "skipped")) else "failed"
            c[key] += 1
            so = tc.find("system-out")
            r["failures"].append({"id": tc.get("name"), "kind": key, "location": None,
                                  "message": (kid.get("message") if kid is not None else st) or "",
                                  "assertion": "", "traceback": tail(so.text or "", TB_LINES) if so is not None else "",
                                  "time_s": float(tc.get("time") or 0)})
        r["counts"] = c
    except (OSError, ET.ParseError):
        r["anomalies"].append("no ctest JUnit report: the build failed or ctest did not run; log tail:\n"
                              + tail(out, 20))
    check_baseline(r, bls.get("csim", {}), True)
    if rc not in (0,) and not r["failures"] and not r["anomalies"]:
        r["anomalies"].append(f"exit code {rc}; log tail:\n{tail(out, 15)}")
    return r


# ---- tts-host ----------------------------------------------------------------- #

def run_tts_host(a, out_dir: Path, bls: dict) -> dict:
    project = REPO / "demo" / "tts" / "build" / "piper_project"
    script = REPO / "demo" / "tts" / "scripts" / "tts_host_emu.py"
    r = base_result("tts-host", f"{VENV_PY} {script}  &&  ... --lib-check", SCHED)
    if not (project / "project.json").exists():
        return not_run(r, "no generated Piper project",
                       "inference-scheduler/.venv/bin/python demo/tts/scripts/generate_tts_project.py")
    result = project / "host_emu" / "result.json"
    c, rcs, dt_all, logs = {"checks": 0, "bit_exact": 0, "mismatches": 0}, [], 0.0, []

    def check(cid: str, bad: int, detail: str = "") -> None:
        c["checks"] += 1
        if bad == 0:
            c["bit_exact"] += 1
            return
        c["mismatches"] += 1
        r["failures"].append({"id": cid, "kind": "mismatch", "location": None,
                              "message": f"{bad} mismatches {detail}".strip(), "assertion": "",
                              "traceback": "", "time_s": None})

    for tag, extra in (("emu", []), ("lib", ["--lib-check"])):
        cmd = [str(VENV_PY), str(script), *extra]
        lf = out_dir / f"tts-host-{tag}.log"
        logs.append(str(lf))
        if tag == "emu" and result.exists():
            result.unlink()                                   # never read a stale report
        log(f"[tts-host] {shlex.join(cmd)}")
        rc, dt, out = run(cmd, SCHED, lf, a.timeout or 1800)
        dt_all += dt
        rcs.append(rc)
        if tag == "emu" and result.exists():
            rep = json.loads(result.read_text())["check"]
            for utt, v in rep.items():
                if utt in ("encoder", "duration"):
                    for case, w in v.items():
                        check(f"emu:{utt}:{case}", w["mismatches"], f"(ids {w.get('ids')})")
                elif utt == "_size":
                    check("emu:pcm_size", 1, json.dumps(v))
                else:
                    check(f"emu:pcm:{utt}", v["mismatches"],
                          f"of {v.get('samples')} samples (first at {v.get('first_mismatch')})")
        for text, zp, pcm in re.findall(r"^\s+\[(.*?)\.\.\.\].*z_p (bit-exact|MISMATCH), samples (bit-exact|MISMATCH)",
                                        out, re.M):
            check(f"lib:z_p:{text}", 0 if zp == "bit-exact" else 1)
            check(f"lib:samples:{text}", 0 if pcm == "bit-exact" else 1)
        verdict = re.search(r"^(HOST EMULATION|LIB CHECK): (.*)$", out, re.M)
        r["notes"].append(f"{tag}: " + (verdict.group(0) if verdict else "no verdict line"))
        if rc != 0 and not any(f["id"].startswith(tag) for f in r["failures"]):
            r["failures"].append({"id": tag, "kind": "error", "location": None, "message": f"exit code {rc}",
                                  "assertion": "", "traceback": tail(out, TB_LINES), "time_s": None})
        for w in re.findall(r"^.*\b(?:\w*Warning|warning:).*$", out, re.M):
            r["warnings"].append({"category": "output", "message": w.strip()[:300], "location": tag,
                                  "tests": [], "count": 1})
    r.update(exit_code=next((x for x in rcs if x != 0), 0), duration_s=round(dt_all, 1), counts=c,
             log=" ".join(logs))
    if c["checks"] == 0:
        r["anomalies"].append("no checks parsed (no result.json, no lib-check lines)")
    check_baseline(r, bls.get("tts-host", {}), True)
    return r


# ---- rtl ------------------------------------------------------------------------ #

def run_rtl(a, out_dir: Path, bls: dict) -> dict:
    cmd = f"source {shlex.quote(VITIS)} >/dev/null 2>&1 && make behavior_test 2>&1"
    r = base_result("rtl", cmd, BUILD)
    if not os.path.exists(VITIS):
        return not_run(r, f"Vitis settings not found at {VITIS}")
    log(f"[rtl] {cmd}")
    rc, dt, out = run(cmd, BUILD, out_dir / "rtl.log", a.timeout or 14400, shell=True)
    r.update(exit_code=rc, duration_s=round(dt, 1), log=str(out_dir / "rtl.log"))
    c = {}
    for kern, ok, tot in re.findall(r"(\w+) Test Summary:\s*(\d+)\s*/\s*(\d+) passed", out):
        c[kern] = f"{ok}/{tot}"
        if ok != tot:
            r["failures"].append({"id": kern, "kind": "rtl_mismatch", "location": None,
                                  "message": f"{ok} / {tot} passed", "assertion": "", "traceback": "",
                                  "time_s": None})
    r["counts"] = c
    r["notes"].append("behavior_test modifies tracked .bd / .xci / .xpr files of the hw/ submodules: "
                      "do not commit them")
    if rc != 0 and not r["failures"]:
        r["anomalies"].append(f"exit code {rc}; log tail:\n{tail(out, 20)}")
    return r


RUNNERS = {"scheduler": lambda a, o, b: run_pytest("scheduler", a, o, b),
           "chat": lambda a, o, b: run_pytest("chat", a, o, b),
           "lint": run_lint, "facts": run_facts, "csim": run_csim, "tts-host": run_tts_host, "rtl": run_rtl}


def classify_warnings(r: dict, bl: dict) -> None:
    allow = bl.get("warning_allow", [])
    for w in r["warnings"]:
        loc = (w.get("location") or "").split(":")[0]
        path = Path(loc) if os.path.isabs(loc) else REPO / loc
        inside = str(path.resolve()).startswith(str(REPO)) and not any(
            x in loc for x in (".venv", "site-packages", "dist-packages"))
        w["source"] = "project" if inside else "third_party"
        w["known"] = any(re.search(p, f"{w['category']}: {w['message']}") for p in allow)


def finish(r: dict, bls: dict) -> dict:
    if r["status"] == "not_run":
        return r
    classify_warnings(r, bls.get(r["suite"], {}))
    if r["failures"] or r["exit_code"] == "timeout":
        r["status"] = "fail"
    elif r["anomalies"] and any(a.startswith(("pytest exit code", "no JUnit", "no ctest", "exit code",
                                              "timed out", "no tests ran")) for a in r["anomalies"]):
        r["status"] = "fail"
    elif r["anomalies"] or any(not w["known"] and w["source"] == "project" for w in r["warnings"]):
        r["status"] = "warn"
    return r


def record_baseline(r: dict, bls: dict) -> None:
    c = r["counts"]
    bls[r["suite"]] = {"min_tests": c.get("tests", c.get("checks", 0)), "max_skipped": c.get("skipped", 0),
                       "skip_allow": sorted({re.sub(r"([.^$*+?{}\[\]\\|()])", r"\\\1", s["reason"])
                                             for s in r["skips"]}),
                       "warning_allow": sorted({re.sub(r"([.^$*+?{}\[\]\\|()])", r"\\\1",
                                                       f"{w['category']}: {w['message']}")
                                                for w in r["warnings"]}),
                       "typical_duration_s": r["duration_s"],
                       "recorded": time.strftime("%Y-%m-%d %H:%M")}
    BASELINES.write_text(json.dumps(bls, indent=1) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default=",".join(DEFAULT))
    ap.add_argument("--tests", nargs="+", default=None, help="pytest paths / node ids (scheduler or chat)")
    ap.add_argument("-k", default=None)
    ap.add_argument("--pytest-args", default=None)
    ap.add_argument("--timeout", type=int, default=None, help="seconds per suite command")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--record-baseline", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    bls = load_baselines()
    if a.list:
        print(json.dumps({"suites": {"default": DEFAULT, "all": ALL, "opt-in": ["rtl"]}, "baselines": bls},
                         indent=1))
        return 0
    suites = []
    for s in a.suite.split(","):
        s = s.strip()
        suites += list(ALL) if s == "all" else list(DEFAULT) if s == "default" else [s]
    bad = [s for s in suites if s not in KNOWN]
    if bad:
        ap.error(f"unknown suite(s) {bad}; known: {', '.join(KNOWN)}, all, default "
                 f"(board suites are not run by this tool)")
    if a.tests and any(s not in ("scheduler", "chat") for s in suites):
        ap.error("--tests applies to one pytest suite: --suite scheduler or --suite chat")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.environ.get("CLAUDE_JOB_DIR")
    out_dir = Path(a.out_dir or (Path(base) / "tmp" / f"run-tests-{stamp}" if base
                                 else Path("/tmp") / f"run-tests-{os.getuid()}-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.detach:
        json_path = Path(a.json or out_dir / "report.json")
        args, skip = [], False
        for x in (argv if argv is not None else sys.argv[1:]):
            if skip:
                skip = False
            elif x in ("--out-dir", "--json"):
                skip = True
            elif x != "--detach" and not x.startswith(("--out-dir=", "--json=")):
                args.append(x)
        cmd = [sys.executable, str(Path(__file__).resolve()), *args, "--out-dir", str(out_dir),
               "--json", str(json_path)]
        lf = open(out_dir / "runner.log", "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": p.pid, "json": str(json_path), "log": str(out_dir / "runner.log"),
                          "wait": f"timeout 590 tail --pid={p.pid} -f /dev/null"}))
        return 0
    results = []
    t0 = time.monotonic()
    for s in dict.fromkeys(suites):
        r = finish(RUNNERS[s](a, out_dir, bls), bls)
        results.append(r)
        log(f"[{s}] {r['status']}: {r['counts']} ({r['duration_s']} s)")
        if a.record_baseline:
            whole = not (a.tests or a.k or a.pytest_args)
            if whole and r["status"] in ("pass", "warn") and not r["failures"]:
                record_baseline(r, bls)
                r["notes"].append("baseline recorded")
            else:
                r["notes"].append("baseline NOT recorded (partial run or failures)")
    overall = max((r["status"] for r in results), key=lambda s: RANK[s]) if results else "not_run"
    report = {"status": overall, "duration_s": round(time.monotonic() - t0, 1), "repo": str(REPO),
              "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                    text=True).stdout.strip(),
              "out_dir": str(out_dir), "suites": results}
    report["report"] = str(out_dir / "report.json")
    text = json.dumps(report, indent=1)
    (out_dir / "report.json").write_text(text + "\n")
    log(f"status {overall}; report {out_dir / 'report.json'}")
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)
    return {"pass": 0, "warn": 0}.get(overall, 1)


if __name__ == "__main__":
    sys.exit(main())

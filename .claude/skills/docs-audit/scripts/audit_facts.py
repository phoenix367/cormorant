#!/usr/bin/env python3
"""audit_facts.py — helpers for docs-audit mode A (static audit, docs vs code).

  audit_facts.py areas  [--repo R]
      every tracked .md file (and the hw/ submodules' docs) grouped into the
      audit areas, one agent each, with line counts; files no area claims
      are listed as UNASSIGNED (give them to an agent or extend AREAS).

  audit_facts.py counts [--repo R] [--run] [--build DIR] [--all] [--briefs OUT]
      the countable facts measured from the repo — pytest collection of the
      scheduler and chat suites (--run: run them and take passed / skipped /
      failed), ctest -N (--build), RTL fixture manifests, the on-board model
      set, the perf cases — then every count-like claim in the current-state
      docs whose number matches none of them (--all: every claim).  History
      docs (doc/plans, *_OPTIMISATION.md, HLS_CONV_RESEARCH) are not scanned.
      --briefs OUT also writes the filled agent briefs: OUT/area_<name>.md per
      non-empty area (templates/static_area_brief.md) and OUT/review.md
      (templates/static_review_brief.md), with the measured counts and the doc
      files already modified before the audit.

Standard library only; pytest runs in <repo>/inference-scheduler/.venv.
"""
import argparse
import datetime
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# (area, what the agent checks, patterns) — first match wins; '*' spans '/'
AREAS = [
    ("root", "README.md, CLAUDE.md, doc/README.md", ["README.md", "CLAUDE.md", "doc/README.md"]),
    ("scheduler-ref", "scheduler reference", ["doc/scheduler/*", "inference-scheduler/CLAUDE.md",
                                              "inference-scheduler/perf_models/*"]),
    ("scheduler-guides", "scheduler user guides", ["inference-scheduler/doc/*", "inference-scheduler/*"]),
    ("kernels", "kernel references; optimisation logs: current-state parts only", ["doc/kernels/*"]),
    ("build-test", "build & test docs, skills, hw/ submodule READMEs",
     ["doc/build-and-test/*", ".claude/*", "hw/*", "board/*", "dts/*", "platforms/*"]),
    ("demos", "demo READMEs", ["demo/*"]),
    ("plans", "status lines at the top + doc/README plans table only; history untouched", ["doc/plans/*"]),
]
HISTORY = ["doc/plans/*", "*_OPTIMISATION.md", "doc/kernels/HLS_CONV_RESEARCH.md"]


def sh(cmd, cwd, timeout=3600):
    env = dict(os.environ, PY_COLORS="0", NO_COLOR="1")
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout + p.stderr


def md_files(repo):
    # tracked + untracked-not-ignored (new docs about to be committed)
    _, out = sh(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"], repo)
    files = [f for f in out.splitlines() if f]
    _, subs = sh(["git", "config", "-f", ".gitmodules", "--get-regexp", r"\.path$"], repo)
    for line in subs.splitlines():
        sub = line.split()[1]
        if (repo / sub / ".git").exists():
            _, out = sh(["git", "ls-files", "*.md"], repo / sub)
            files += [f"{sub}/{f}" for f in out.splitlines() if f]
    return sorted(files)


def area_of(path):
    for name, _, pats in AREAS:
        if any(fnmatch.fnmatch(path, p) for p in pats):
            return name
    return None


def nlines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def grouped(repo):
    groups = {name: [] for name, _, _ in AREAS}
    unassigned = []
    for f in md_files(repo):
        a = area_of(f)
        (groups[a] if a else unassigned).append(f)
    return groups, unassigned


def cmd_areas(repo):
    groups, unassigned = grouped(repo)
    for name, what, _ in AREAS:
        fs = groups[name]
        total = sum(nlines(repo / f) for f in fs)
        print(f"## {name} — {what} — {len(fs)} files, {total} lines")
        for f in fs:
            sub = "  (submodule: commit there)" if f.startswith("hw/") else ""
            print(f"   {nlines(repo / f):6d}  {f}{sub}")
    if unassigned:
        print("## UNASSIGNED — give these to an agent or extend AREAS")
        for f in unassigned:
            print(f"   {nlines(repo / f):6d}  {f}")
    return 0


def pytest_counts(repo, where, cwd, run):
    py = repo / "inference-scheduler/.venv/bin/python"
    if not py.exists():
        return {"error": f"{py} missing"}
    args = [str(py), "-m", "pytest", where, "-q", "-p", "no:cacheprovider", "--color=no"]
    rc, out = sh(args + ([] if run else ["--collect-only"]), cwd)
    tail = "\n".join(out.strip().splitlines()[-3:])
    res = {}
    m = re.search(r"(\d+) tests? collected", out)
    if m:
        res["collected"] = int(m.group(1))
    for key in ("passed", "skipped", "failed", "errors?", "deselected"):
        m = re.search(rf"(\d+) {key}\b", tail)
        if m:
            res[key.rstrip("s?")] = int(m.group(1))
    if run and res:
        res["collected"] = sum(res.get(k, 0) for k in ("passed", "skipped", "failed", "error"))
    if not res:
        res["error"] = tail
    return res


def counts(repo, run, build):
    facts = {}
    sched = repo / "inference-scheduler"
    models = sched / "test/models"
    if not models.is_dir() or not any(models.iterdir()):
        print("warning: inference-scheduler/test/models is empty — run test/gen_all_models.py first",
              file=sys.stderr)
    r = pytest_counts(repo, "test/", sched, run)
    facts.update({f"scheduler pytest {k}": v for k, v in r.items()})
    facts["scheduler test modules"] = len(list((sched / "test").glob("test_*.py")))
    r = pytest_counts(repo, "demo/chat/tests", repo, run)
    facts.update({f"chat pytest {k}": v for k, v in r.items()})
    if build and not (build / "CTestTestfile.cmake").exists():
        print(f"warning: {build} is not a configured CMake build dir — ctest skipped", file=sys.stderr)
    elif build:
        rc, out = sh(["ctest", "-N"], build)
        m = re.search(r"Total Tests: (\d+)", out)
        facts["ctest tests"] = int(m.group(1)) if m else out.strip()[-200:]
    total = 0
    for man in sorted((repo / "hw/test_data").glob("*_test_data/manifest.txt")):
        n = sum(1 for ln in man.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
        facts[f"RTL fixtures {man.parent.name.replace('_test_data', '')}"] = n
        total += n
    if total:
        facts["RTL fixtures total"] = total
    for cfg in ("remote_config.json.example", "remote_config_all_models.json"):
        p = sched / cfg
        if p.exists():
            facts[f"board models ({cfg})"] = len(json.loads(p.read_text()).get("models", []))
    p = sched / "perf_config.json.example"
    if p.exists():
        b = json.loads(p.read_text()).get("benchmarks", {})
        per = {k: len(v.get("cases", [])) for k, v in b.items() if isinstance(v, dict)}
        facts.update({f"perf cases {k}": v for k, v in per.items()})
        facts["perf cases total"] = sum(per.values())
    return facts


NUM = r"\d{1,3}(?:[ ,]\d{3})+|\d+"          # 1564, 1,564, 10 000
CLAIM = re.compile(rf"(?<![\d.\-–])({NUM})\s*(?:/\s*(?:{NUM})\s*)?(tests?|models|cases|modules|passed|pass|"
                   r"skip(?:ped|s)?|fixtures|failed|test models)\b")


def cmd_counts(repo, run, build, show_all, briefs):
    facts = counts(repo, run, build)
    print("== measured")
    for k, v in facts.items():
        print(f"   {k:45s} {v}")
    known = {v for v in facts.values() if isinstance(v, int)}
    print("== claims in current-state docs" + ("" if show_all else " whose number matches no measured count"))
    n = 0
    for f in md_files(repo):
        if any(fnmatch.fnmatch(f, p) for p in HISTORY):
            continue
        try:
            lines = (repo / f).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for m in CLAIM.finditer(line):
                num = int(re.sub(r"[ ,]", "", m.group(1)))
                ok = num in known
                if show_all or (not ok and num > 1):
                    n += 1
                    mark = "ok   " if ok else "CHECK"
                    print(f"   {mark} {f}:{i}: …{line[max(0, m.start() - 50):m.end() + 30].strip()}…")
    if not n:
        print("   none")
    print("(CHECK = no measured count has this number: stale, or a count this script does not "
          "measure — dated history, per-module counts; judge each)")
    if briefs:
        write_briefs(repo, facts, briefs)
    return 0


def fill(text, **kv):
    for k, v in kv.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    if left:
        sys.exit(f"unfilled placeholders: {sorted(set(left))}")
    return text


def write_briefs(repo, facts, out):
    tpl = Path(__file__).resolve().parent.parent / "templates"
    area_t = (tpl / "static_area_brief.md").read_text()
    review_t = (tpl / "static_review_brief.md").read_text()
    out.mkdir(parents=True, exist_ok=True)
    common = dict(REPO=str(repo), DATE=datetime.date.today().isoformat(),
                  COUNTS="\n".join(f"{k:45s} {v}" for k, v in facts.items()))
    groups, unassigned = grouped(repo)
    whats = {name: what for name, what, _ in AREAS}
    if unassigned:
        groups["unassigned"] = unassigned
        whats["unassigned"] = "docs no area claims"
    written = []
    for name, fs in groups.items():
        if not fs:
            continue
        files = "\n".join(f"- `{f}` ({nlines(repo / f)} lines)"
                          + (" — hw/ submodule: its own commit" if f.startswith("hw/") else "") for f in fs)
        p = out / f"area_{name}.md"
        p.write_text(fill(area_t, AREA=f"{name} ({whats[name]})", FILES=files, **common))
        written.append(p)
    _, st = sh(["git", "status", "--short", "--", "*.md"], repo)
    pre = ("Doc files already modified BEFORE the audit — review only the audit's hunks in them:\n\n"
           + "\n".join(f"    {ln}" for ln in st.splitlines())) if st.strip() else \
        "The doc tree was clean before the audit: every hunk is the audit's."
    p = out / "review.md"
    p.write_text(fill(review_t, PREEXISTING=pre, **common))
    written.append(p)
    print("== briefs")
    for p in written:
        print(f"   {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["areas", "counts"])
    ap.add_argument("--repo", default=".", help="repository root (default: .)")
    ap.add_argument("--run", action="store_true", help="counts: run the suites (~3.5 min), not only collect")
    ap.add_argument("--build", help="counts: a configured CMake build dir for ctest -N")
    ap.add_argument("--all", action="store_true", help="counts: print every claim, not only mismatches")
    ap.add_argument("--briefs", help="counts: also write the filled agent briefs into this directory")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    if not (repo / ".git").exists():
        sys.exit(f"{repo} is not a repository root")
    if a.cmd == "areas":
        return cmd_areas(repo)
    return cmd_counts(repo, a.run, Path(a.build).resolve() if a.build else None, a.all,
                      Path(a.briefs).resolve() if a.briefs else None)


if __name__ == "__main__":
    sys.exit(main())

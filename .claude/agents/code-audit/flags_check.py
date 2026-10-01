#!/usr/bin/env python3
"""
flags_check.py — command-line flags one script passes to another, against the
flags the other one's argparse accepts.

  --pairs PATH ...          list caller -> callee pairs: files that name another
                            tracked script with an argparse parser (by its file
                            name in a string), with the calling function
  --caller F --callee G     every "--flag" string literal in F (or in one
     [--function NAME]      function of F) that G's argparse does not accept;
                            f-strings "--flag=..." count by their flag part

Only literal flags are checked; flags built at run time are counted as
"dynamic" and must be read by hand.  A flag of G that the caller never passes
is listed for information (with --function only).

usage: python3 .claude/agents/code-audit/flags_check.py --pairs demo/chat
       python3 .claude/agents/code-audit/flags_check.py --caller demo/chat/deploy.py \\
               --function server_argv --callee demo/chat/kv260_chat_server.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FLAG = re.compile(r"^--[A-Za-z0-9][\w-]*$")


def parse(path: Path):
    return ast.parse(path.read_text(errors="replace"))


def accepted_flags(path: Path) -> set:
    flags = set()
    for n in ast.walk(parse(path)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
            flags |= {a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                      and a.value.startswith("-")}
    return flags


def passed_flags(path: Path, function: str | None):
    tree = parse(path)
    roots = [tree]
    if function:
        roots = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == function]
        if not roots:
            raise SystemExit(f"no function {function} in {path}")
    lits, dynamic = {}, 0
    for root in roots:
        for n in ast.walk(root):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                v = n.value.split("=", 1)[0]
                if FLAG.match(v):
                    lits.setdefault(v, n.lineno)
            elif isinstance(n, ast.JoinedStr) and n.values and isinstance(n.values[0], ast.Constant):
                head = str(n.values[0].value)
                m = re.match(r"^(--[A-Za-z0-9][\w-]*)=", head)
                if m:
                    lits.setdefault(m.group(1), n.lineno)
                elif head.startswith("--"):
                    dynamic += 1
    return lits, dynamic


def scripts_with_parsers():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True).stdout.split()
    res = {}
    for f in out:
        if f.startswith("hw/"):
            continue
        p = REPO / f
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if "add_argument(" in text:
            res.setdefault(p.name, []).append(p)
    return res


def pairs(paths):
    parsers = scripts_with_parsers()
    rows = []
    for a in paths:
        p = REPO / a if not Path(a).is_absolute() else Path(a)
        for f in (sorted(p.rglob("*.py")) if p.is_dir() else [p]):
            try:
                tree = parse(f)
            except (SyntaxError, OSError):
                continue
            funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            docs = {id(b.value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef,
                                                                        ast.AsyncFunctionDef))
                    for b in n.body[:1] if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)}
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                    for name in re.findall(r"([\w./-]+\.py)\b", n.value):
                        base = Path(name).name
                        if base in parsers and base != f.name:
                            fn = next((g.name for g in funcs if g.lineno <= n.lineno <= (g.end_lineno or 0)), None)
                            cands = parsers[base]
                            if "/" in name:                           # a path: match its tail
                                cands = [c for c in cands if str(c).endswith(name.lstrip("./"))] or cands
                            if len(cands) > 1:                        # still several: the nearest script
                                near = [c for c in cands if f.parent in c.parents or c.parent == f.parent
                                        or c.parts[:len(REPO.parts) + 2] == f.parts[:len(REPO.parts) + 2]]
                                cands = near or cands
                            for callee in cands:
                                rows.append({"caller": f"{f.relative_to(REPO)}:{n.lineno}", "function": fn,
                                             "callee": str(callee.relative_to(REPO))})
    seen, out = set(), []
    for r in rows:
        k = (r["caller"].split(":")[0], r["function"], r["callee"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--caller")
    ap.add_argument("--callee")
    ap.add_argument("--function")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.pairs:
        rows = pairs(a.pairs)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                print(f"{r['caller']:<52} {r['function'] or '(module)':<24} -> {r['callee']}")
        return 0
    if not (a.caller and a.callee):
        ap.error("--pairs PATH..., or --caller F --callee G [--function NAME]")
    caller, callee = REPO / a.caller, REPO / a.callee
    ok = accepted_flags(callee)
    lits, dynamic = passed_flags(caller, a.function)
    bad = {f: ln for f, ln in lits.items() if f not in ok}
    rep = {"caller": a.caller, "function": a.function, "callee": a.callee,
           "unknown_flags": [{"flag": f, "where": f"{a.caller}:{ln}"} for f, ln in sorted(bad.items())],
           "dynamic_flags": dynamic, "passed": len(lits),
           "never_passed": sorted(ok - set(lits)) if a.function else None}
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        for b in rep["unknown_flags"]:
            print(f"UNKNOWN  {b['flag']:<32} {b['where']}  (not accepted by {a.callee})")
        print(f"{len(lits)} literal flags checked, {len(bad)} unknown, {dynamic} built at run time")
        if a.function and rep["never_passed"]:
            print("accepted but not passed here (info):", " ".join(rep["never_passed"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

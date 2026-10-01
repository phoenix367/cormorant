#!/usr/bin/env python3
"""
config_keys.py — the keys of an example config (JSON) that no code reads, and
the keys code reads from that config that the example does not document.

  keys never read   every key of EXAMPLE (keys starting with "_" are notes)
                    whose name appears as a string literal in none of the code
                    files: a dead option, or one read under another name
  keys not in the   key paths the code reads from the config dict — chains
  example           cfg["a"]["b"], cfg.get("a", {}).get("b"), s["b"] after
                    s = cfg["a"], rooted at a name of --roots — that the example
                    does not have (under a parent it does have): an undocumented
                    option, or a typo.  Include every module that reads the config
                    (e.g. inference-scheduler/src/remote for ssh.*)

usage: python3 .claude/agents/code-audit/config_keys.py EXAMPLE.json --code PATH [PATH ...]
           [--roots cfg,config] [--json]
       e.g.  demo/chat/chat_config.json.example --code demo/chat
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            path = f"{prefix}.{k}" if prefix else k
            yield path, k
            yield from flatten(v, path)


def py_files(paths):
    for a in paths:
        p = REPO / a if not Path(a).is_absolute() else Path(a)
        yield from (sorted(p.rglob("*.py")) if p.is_dir() else [p])


def code_strings(paths):
    """string literal -> [file:line] over the code."""
    lits = defaultdict(list)
    for f in py_files(paths):
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except (SyntaxError, OSError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                lits[n.value].append(f"{f.relative_to(REPO)}:{n.lineno}")
    return lits


def chain(node, aliases):
    """cfg["a"]["b"] / cfg.get("a", {}).get("b") / s["b"] with s = cfg["a"] -> ["a", "b"], else None."""
    keys = []
    while True:
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            keys.append(node.slice.value)
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("get", "setdefault") and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            keys.append(node.args[0].value)
            node = node.func.value
        elif isinstance(node, ast.Name) and node.id in aliases:
            return aliases[node.id] + keys[::-1]
        else:
            return None


def config_reads(paths, roots):
    """Dotted key paths read from the config (chains rooted at a name in roots) -> [file:line]."""
    reads = defaultdict(list)
    for f in py_files(paths):
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = str(f.relative_to(REPO))
        for scope in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            aliases = {r: [] for r in roots}
            body = scope.body
            for stmt in body:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                        c = chain(n.value, aliases)
                        if c:
                            aliases[n.targets[0].id] = c
                    if isinstance(n, (ast.Subscript, ast.Call)):
                        c = chain(n, aliases)
                        if c and not any(k.startswith("_") for k in c):
                            reads[".".join(c)].append(f"{rel}:{n.lineno}")
    return reads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("example")
    ap.add_argument("--code", nargs="+", required=True)
    ap.add_argument("--roots", default="cfg,config",
                    help="names the config dict is bound to in the code (default cfg,config)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    ex = json.loads((REPO / a.example).read_text())
    keys = list(flatten(ex))
    paths = {p for p, _ in keys}
    lits = code_strings(a.code)
    never = [{"key": path} for path, k in keys if k not in lits]
    reads = config_reads(a.code, [r.strip() for r in a.roots.split(",") if r.strip()])
    undoc = []
    for path, where in sorted(reads.items()):
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        if path not in paths and (not parent or parent in paths):
            undoc.append({"key": path, "where": sorted(set(where))[:5]})
    rep = {"example": a.example, "keys": len(keys), "never_read": never, "read_but_not_in_example": undoc}
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        for r in never:
            print(f"NEVER READ      {r['key']}")
        for r in undoc:
            print(f"NOT IN EXAMPLE  {r['key']:<32} {', '.join(r['where'])}")
        print(f"{len(keys)} keys; {len(never)} never read; {len(undoc)} read from {a.roots} but not in the example")
    return 0


if __name__ == "__main__":
    sys.exit(main())

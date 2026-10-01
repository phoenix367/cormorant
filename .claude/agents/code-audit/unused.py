#!/usr/bin/env python3
"""
unused.py — definitions in the given paths that nothing else in the repo names.

Definitions:  Python top-level functions, classes, methods and UPPER_CASE
module constants (AST); C / C++ function definitions (regex).
References:   every other occurrence of the name as a word in any tracked text
file outside hw/ (code, tests, scripts, CMake, TCL, docs, JSON), so a name
used through getattr("name"), a registry dict, a CMake target or a doc still
counts.  Names built at run time (f"cmd_{x}", string concatenation) are NOT
seen: confirm every candidate by hand before removing it.

Reported: refs == 0 (unreferenced), or refs <= --max-refs with their places;
"tests_only": every reference is in a test file (production code kept alive
only by its tests).  Skipped: dunders, test_* / setUp* / tearDown*, do_GET-style
and visit_* hooks, framework overrides (log_message, handle, ...), main, names
in __all__, @pytest.fixture functions.

usage: python3 .claude/agents/code-audit/unused.py PATH [PATH ...] [--max-refs N] [--json]
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TEXT_EXT = (".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".tcl", ".in", ".txt", ".cmake", ".md", ".json", ".sh",
            ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sv", ".v", ".tmpl", ".example", ".xdc")
HOOKS = {"log_message", "handle", "setup", "finish", "run", "default", "emit", "close", "flush", "write", "read",
         "handle_error", "server_bind", "process_request", "get_request", "shutdown", "start", "stop",
         "setUpClass", "tearDownClass", "setUpModule", "tearDownModule", "load_tests", "generic_visit",
         "fileno", "readable", "writable", "seekable", "tell", "seek", "keys", "items", "values", "get",
         "copy", "update", "pop", "clear", "append", "extend", "__init__",
         "log_error", "log_request", "send_error", "version_string", "address_string", "date_time_string",
         "handle_one_request", "parse_request", "verify_request", "service_actions", "server_activate",
         "connection_made", "data_received", "startTest", "stopTest", "addSuccess", "addFailure", "addError"}
C_DEF = re.compile(r"^[ \t]*(?:static\s+|inline\s+|extern\s+)*(?:const\s+)?[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)"
                   r"\s*\(([^;{}()]*)\)\s*(?:const\s*)?\{", re.M)
C_DECL = re.compile(r"^[ \t]*(?:extern\s+)?(?:const\s+)?[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)"
                    r"\s*\(([^;{}()]*)\)\s*;", re.M)
C_SKIP = {"main", "if", "for", "while", "switch", "return", "sizeof", "else"}


def tracked(exts):
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split()
    files = []
    for f in out:
        if f.startswith("hw/") or not f.endswith(exts):
            continue
        p = REPO / f
        try:
            if p.stat().st_size < 3_000_000:
                files.append(p)
        except OSError:
            pass
    return files


def is_test(path: str) -> bool:
    name = Path(path).name
    return "/tests/" in f"/{path}" or "/test/" in f"/{path}" or name.startswith("test_") or name.endswith("_test.py")


def py_defs(path: Path):
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    rel = str(path.relative_to(REPO))
    exported = set()
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            if isinstance(n.value, (ast.List, ast.Tuple)):
                exported |= {e.value for e in n.value.elts if isinstance(e, ast.Constant)}
    out = []

    def skip(name, node=None) -> bool:
        if name.startswith("__") and name.endswith("__"):
            return True
        if name in exported or name in HOOKS or name.startswith(("test", "setUp", "tearDown", "visit_")):
            return True
        if re.match(r"do_[A-Z]+$", name):
            return True
        if node is not None:
            for d in getattr(node, "decorator_list", []):
                src = ast.unparse(d)
                if "fixture" in src or src.endswith(".setter") or src.endswith(".deleter"):
                    return True
        return False

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not skip(n.name, n):
            out.append({"name": n.name, "kind": "function", "where": f"{rel}:{n.lineno}"})
        elif isinstance(n, ast.ClassDef):
            test_case = any("TestCase" in ast.unparse(b) for b in n.bases)
            if not skip(n.name) and not n.name.startswith("Test") and not test_case:
                out.append({"name": n.name, "kind": "class", "where": f"{rel}:{n.lineno}"})
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not skip(m.name, m):
                    out.append({"name": m.name, "kind": f"method of {n.name}", "where": f"{rel}:{m.lineno}"})
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", t.id) and t.id not in exported:
                    out.append({"name": t.id, "kind": "constant", "where": f"{rel}:{n.lineno}"})
    return out


def c_defs(path: Path):
    text = path.read_text(errors="replace")
    blank = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    blank = re.sub(r"//[^\n]*", "", blank)
    rel = str(path.relative_to(REPO))
    out = []
    for m in C_DEF.finditer(blank):
        name = m.group(1)
        if name in C_SKIP:
            continue
        line = blank.count("\n", 0, m.start(1)) + 1
        out.append({"name": name, "kind": "C function", "where": f"{rel}:{line}"})
    return out


def c_decls(path: Path):
    """Names of the function prototypes in a header (declarations, not references)."""
    text = re.sub(r"/\*.*?\*/", " ", path.read_text(errors="replace"), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return [m.group(1) for m in C_DECL.finditer(text) if m.group(1) not in C_SKIP]


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    as_json = "--json" in args
    max_refs = 0
    if "--max-refs" in args:
        i = args.index("--max-refs")
        max_refs = int(args[i + 1])
        del args[i:i + 2]
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print(__doc__.split("\n\n")[-1].strip(), file=sys.stderr)
        return 2
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                            capture_output=True, text=True).stdout.split()
    known = {REPO / f for f in listed if not f.startswith("hw/")}      # tracked + new, not ignored
    scope = []
    for a in paths:
        p = (REPO / a if not Path(a).is_absolute() else Path(a)).resolve()
        scope += [q for q in (sorted(p.rglob("*")) if p.is_dir() else [p])
                  if q in known and q.suffix in (".py", ".c", ".cpp", ".cc", ".h", ".hpp")]
    defs = []
    for f in scope:
        defs += py_defs(f) if f.suffix == ".py" else c_defs(f)
    # every definition site in the repo (a definition is not a reference)
    def_sites = Counter()
    for f in tracked((".py",)):
        for d in py_defs(f):
            def_sites[d["name"]] += 1
    for f in tracked((".c", ".cpp", ".cc", ".h", ".hpp")):
        for d in c_defs(f):
            def_sites[d["name"]] += 1
        if f.suffix in (".h", ".hpp"):
            for name in c_decls(f):
                def_sites[name] += 1
    names = {d["name"] for d in defs}
    total, places = Counter(), defaultdict(list)
    for f in tracked(TEXT_EXT):
        rel = str(f.relative_to(REPO))
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            for w in WORD.findall(line):
                if w in names:
                    total[w] += 1
                    places[w].append(f"{rel}:{ln}")
    rows = []
    for d in defs:
        name = d["name"]
        refs = total[name] - max(def_sites[name], 1)
        if refs > max_refs:
            continue
        own = {d2["where"] for d2 in defs if d2["name"] == name}
        where_refs = [p for p in places[name] if p not in own][:12]
        rows.append({**d, "refs": max(refs, 0), "ref_places": where_refs,
                     "tests_only": bool(where_refs) and all(is_test(p.split(":")[0]) for p in where_refs)})
    rows.sort(key=lambda r: (r["refs"], r["where"]))
    if as_json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            tag = " [tests only]" if r["tests_only"] else ""
            print(f"refs {r['refs']:<2} {r['kind']:<24} {r['name']:<32} {r['where']}{tag}")
            for p in r["ref_places"][:max_refs or 0]:
                print(f"        {p}")
        print(f"summary: {len(defs)} definitions in scope, {sum(r['refs'] == 0 for r in rows)} unreferenced"
              + (f", {len(rows)} with <= {max_refs} references" if max_refs else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

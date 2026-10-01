#!/usr/bin/env python3
"""
ctypes_check.py — every ctypes binding in the given Python files against the
C prototype of the same symbol in the repo's headers / sources.

Recognised binding forms (per function scope, aliases such as
``fp = ctypes.POINTER(ctypes.c_float)`` resolved):
  lib.sym.argtypes = [...]            lib.sym.restype = T
  lib.sym.argtypes, lib.sym.restype = [...], T
  {"sym": ([...], T), ...}            (("sym", T, [...]), ...)   tables, any order
  for n in ("a", "b"): f = getattr(lib, n); f.argtypes = [...]; f.restype = T

Reported per binding: ok | mismatch (arity / argument / return type) |
loose (c_void_p for a typed pointer) | no-prototype (external library, e.g.
espeak-ng) | inconsistent (two files bind the symbol differently).  A symbol
that is called but whose restype is never declared, while its C return type
is not int, is reported as "undeclared-restype" (ctypes would truncate it).

usage: python3 .claude/agents/code-audit/ctypes_check.py [PATH ...] [--json]
       (default: every tracked .py file outside hw/)
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CT = {"c_int": "int", "c_uint": "unsigned", "c_int8": "int8_t", "c_int16": "int16_t", "c_int32": "int32_t",
      "c_int64": "int64_t", "c_uint8": "uint8_t", "c_uint16": "uint16_t", "c_uint32": "uint32_t",
      "c_uint64": "uint64_t", "c_long": "long", "c_ulong": "unsigned long", "c_longlong": "int64_t",
      "c_ulonglong": "uint64_t", "c_short": "short", "c_ushort": "unsigned short", "c_byte": "int8_t",
      "c_ubyte": "uint8_t", "c_char": "char", "c_bool": "bool", "c_float": "float", "c_double": "double",
      "c_size_t": "size_t", "c_ssize_t": "ssize_t", "c_char_p": "char*", "c_wchar_p": "wchar_t*",
      "c_void_p": "void*"}
SAME = {"int32_t": "int", "unsigned": "uint32_t", "unsigned int": "uint32_t", "_Bool": "bool",
        "long long": "int64_t", "unsigned long long": "uint64_t", "signed char": "int8_t",
        "unsigned char": "uint8_t", "long": "int64_t", "unsigned long": "uint64_t"}   # LP64 (aarch64 / x86-64)


BYTES = {"char*", "uint8_t*", "int8_t*"}                        # byte buffers: interchangeable


def norm(t: str) -> str:
    base, stars = t.rstrip("*"), len(t) - len(t.rstrip("*"))
    base = SAME.get(base, base)
    return base + "*" * stars


def git_files(pattern_exts):
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split()
    return [REPO / f for f in out if f.endswith(pattern_exts) and not f.startswith("hw/")]


# ---- C prototypes --------------------------------------------------------------- #

PROTO = re.compile(r"(?:^|[;}\n])\s*((?:[A-Za-z_][\w]*[\s\*]+)+?)([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*([;{])", re.M)
C_WORDS = {"const", "volatile", "extern", "static", "inline", "__inline__", "restrict", "__restrict", "struct",
           "enum", "register", "signed", "EXPORT", "API"}


def c_type(decl: str, is_param: bool) -> str:
    decl = re.sub(r"__attribute__\s*\(\(.*?\)\)", "", decl)
    stars = decl.count("*") + (1 if re.search(r"\[\s*\w*\s*\]\s*$", decl) else 0)
    decl = re.sub(r"\[.*?\]", "", decl).replace("*", " ")
    words = [w for w in decl.split() if w not in C_WORDS]
    if is_param and len(words) >= 2:
        words = words[:-1]                              # the parameter name
    base = " ".join(words) or "int"
    if base in ("unsigned int", "unsigned"):
        base = "unsigned"
    return base + "*" * stars


def keep_lines(m):
    """A comment -> blanks, its newlines kept (so line numbers stay right)."""
    return re.sub(r"[^\n]", " ", m.group(0))


def c_prototypes():
    protos = {}
    for f in git_files((".h", ".c", ".cpp", ".hpp")):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        text = re.sub(r"/\*.*?\*/", keep_lines, text, flags=re.S)
        text = re.sub(r"//[^\n]*", keep_lines, text)
        text = re.sub(r"^[ \t]*#[^\n]*", keep_lines, text, flags=re.M)
        for m in PROTO.finditer(text):
            ret, name, params = m.group(1), m.group(2), m.group(3)
            if name in ("if", "for", "while", "switch", "return", "sizeof") or ret.strip() in ("return", "else"):
                continue
            ps = [p.strip() for p in params.split(",") if p.strip()]
            if ps == ["void"]:
                ps = []
            if any(p == "..." for p in ps):
                continue
            line = text.count("\n", 0, m.start(2)) + 1
            entry = {"ret": c_type(ret, False), "args": [c_type(p, True) for p in ps],
                     "where": f"{f.relative_to(REPO)}:{line}", "header": f.suffix in (".h", ".hpp")}
            old = protos.get(name)
            if old is None or (entry["header"] and not old["header"]):
                protos[name] = entry
    return protos


# ---- Python bindings -------------------------------------------------------------- #

def ct_eval(node, aliases):
    """A ctypes type expression -> C-ish type string, or None when it is not one."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and node.value is None:
        return "void"
    if isinstance(node, ast.Attribute) and node.attr in CT:
        return CT[node.attr]
    if isinstance(node, ast.Name):
        if node.id in CT:
            return CT[node.id]
        if node.id in aliases:
            return aliases[node.id]
        return None
    if isinstance(node, ast.Call):
        fn = node.func
        fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if fname == "POINTER" and node.args:
            inner = ct_eval(node.args[0], aliases)
            return inner + "*" if inner else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        inner = ct_eval(node.left, aliases)
        return inner + "*" if inner else None
    return None


def collect_aliases(body_nodes, base):
    aliases = dict(base)
    for n in body_nodes:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
                t, v = sub.targets[0], sub.value
                if isinstance(t, ast.Name):
                    ty = ct_eval(v, aliases)
                    if ty:
                        aliases[t.id] = ty
                elif isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) and len(t.elts) == len(v.elts):
                    for tt, vv in zip(t.elts, v.elts, strict=True):
                        ty = ct_eval(vv, aliases)
                        if isinstance(tt, ast.Name) and ty:
                            aliases[tt.id] = ty
    return aliases


def sym_of(attr_node):
    """lib.sym.argtypes -> 'sym' (the attribute before argtypes / restype)."""
    v = attr_node.value
    return v.attr if isinstance(v, ast.Attribute) else None


def bindings_in(path: Path):
    src = path.read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    mod_alias = collect_aliases([n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))], {})
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]
    out, seen = [], set()
    rel = str(path.relative_to(REPO))

    def add(sym, args, ret, node):
        key = (sym, node.lineno)
        if sym and key not in seen:
            seen.add(key)
            out.append({"sym": sym, "args": args, "ret": ret, "where": f"{rel}:{node.lineno}"})

    for scope in scopes:
        body = scope.body
        aliases = collect_aliases(body, mod_alias) if scope is not tree else mod_alias
        partial = {}                                           # sym -> {args, ret, node}
        for n in (x for b in body for x in ast.walk(b)) if scope is not tree else ast.walk(tree):
            if isinstance(n, ast.Assign):
                targets = n.targets[0].elts if isinstance(n.targets[0], ast.Tuple) else n.targets
                values = n.value.elts if isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Tuple) \
                    else [n.value] * len(targets)
                for t, v in zip(targets, values, strict=False):
                    if isinstance(t, ast.Attribute) and t.attr in ("argtypes", "restype"):
                        s = sym_of(t)
                        if s:
                            d = partial.setdefault(s, {"node": n})
                            if t.attr == "argtypes" and isinstance(v, (ast.List, ast.Tuple)):
                                d["args"] = [ct_eval(e, aliases) or "?" for e in v.elts]
                            elif t.attr == "restype":
                                d["ret"] = ct_eval(v, aliases) or "?"
            # tables: {"sym": ([args], T)} and (("sym", T, [args]), ...)
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values, strict=True):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and isinstance(v, ast.Tuple):
                        lst = [e for e in v.elts if isinstance(e, (ast.List,))]
                        oth = [e for e in v.elts if not isinstance(e, ast.List)]
                        if len(lst) == 1 and len(oth) == 1 and (ct_eval(oth[0], aliases) or
                                                                   (isinstance(oth[0], ast.Constant) and oth[0].value is None)):
                            add(k.value, [ct_eval(e, aliases) or "?" for e in lst[0].elts],
                                ct_eval(oth[0], aliases) or "void", k)
            if isinstance(n, ast.Tuple) and len(n.elts) == 3 and isinstance(n.elts[0], ast.Constant) \
                    and isinstance(n.elts[0].value, str):
                lst = [e for e in n.elts[1:] if isinstance(e, ast.List)]
                oth = [e for e in n.elts[1:] if not isinstance(e, ast.List)]
                if len(lst) == 1 and len(oth) == 1:
                    ret = ct_eval(oth[0], aliases)
                    if ret:
                        add(n.elts[0].value, [ct_eval(e, aliases) or "?" for e in lst[0].elts], ret, n)
            # for n in ("a", "b"): f = getattr(lib, n); f.argtypes = [...]; f.restype = T
            # ... and for f in (lib.a, lib.b): f.argtypes = [...]
            if isinstance(n, ast.For) and isinstance(n.iter, (ast.Tuple, ast.List)) and all(
                    (isinstance(e, ast.Constant) and isinstance(e.value, str)) or isinstance(e, ast.Attribute)
                    for e in n.iter.elts):
                args = ret = None
                for b in ast.walk(n):
                    if isinstance(b, ast.Assign) and isinstance(b.targets[0], ast.Attribute) and \
                            isinstance(b.targets[0].value, ast.Name):
                        if b.targets[0].attr == "argtypes" and isinstance(b.value, (ast.List, ast.Tuple)):
                            args = [ct_eval(e, aliases) or "?" for e in b.value.elts]
                        elif b.targets[0].attr == "restype":
                            ret = ct_eval(b.value, aliases) or "?"
                if args is not None or ret is not None:
                    for e in n.iter.elts:
                        add(e.attr if isinstance(e, ast.Attribute) else e.value, args, ret, n)
        for s, d in partial.items():
            add(s, d.get("args"), d.get("ret"), d["node"])
    return out


def called_syms(path: Path, names):
    """Symbols of the C map called as <x>.sym(...) in this file."""
    src = path.read_text(errors="replace")
    return {m.group(1) for m in re.finditer(r"\.\s*([A-Za-z_]\w*)\s*\(", src) if m.group(1) in names}


def compare(b, p):
    issues = []
    if b["args"] is not None:
        if len(b["args"]) != len(p["args"]):
            issues.append(f"arity: Python {len(b['args'])} args {b['args']}, C {len(p['args'])} {p['args']}")
        else:
            for i, (a, c) in enumerate(zip(b["args"], p["args"], strict=True)):
                na, nc = norm(a), norm(c)
                if na == nc or a == "?":
                    continue
                if na == "void*" and nc.endswith("*"):
                    issues.append(f"loose: arg {i} c_void_p for C {c}")
                elif {na, nc} <= BYTES:
                    issues.append(f"loose: arg {i} {a} (bytes) for C {c}")
                elif na.endswith("*") and nc == "void*":
                    continue
                elif na == "char*" and nc == "char*":
                    continue
                else:
                    issues.append(f"arg {i}: Python {a}, C {c}")
    if b["ret"] is not None:
        nr, nc = norm(b["ret"]), norm(p["ret"])
        if nr != nc and not (nr == "void*" and nc.endswith("*")) and b["ret"] != "?":
            issues.append(f"return: Python {b['ret']}, C {p['ret']}")
    return issues


def main(argv=None) -> int:
    args = [a for a in (argv if argv is not None else sys.argv[1:]) if a != "--json"]
    as_json = "--json" in (argv if argv is not None else sys.argv[1:])
    files = []
    for a in args:
        p = (REPO / a) if not Path(a).is_absolute() else Path(a)
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    if not args:
        files = git_files((".py",))
    protos = c_prototypes()
    rows, by_sym = [], {}
    for f in files:
        if ".claude/agents/code-audit" in str(f):
            continue
        bs = bindings_in(f)
        declared_ret = {b["sym"] for b in bs if b["ret"] is not None}
        for b in bs:
            p = protos.get(b["sym"])
            if p is None:
                rows.append({**b, "status": "no-prototype", "detail": "no C prototype in the repo (external library?)"})
                continue
            iss = compare(b, p)
            status = "ok" if not iss else ("loose" if all(i.startswith("loose") for i in iss) else "mismatch")
            rows.append({**b, "c": p["where"], "c_sig": f"{p['ret']} {b['sym']}({', '.join(p['args'])})",
                         "status": status, "detail": "; ".join(iss)})
            by_sym.setdefault(b["sym"], []).append(b)
        if bs:
            for s in sorted(called_syms(f, protos) - declared_ret):
                p = protos[s]
                if norm(p["ret"]) not in ("int", "void") and any(b["sym"] == s for b in bs) is False and \
                        re.search(rf"\b(lib|L|_lib|self\.lib|self\._lib)\.{s}\s*\(", f.read_text(errors="replace")):
                    rows.append({"sym": s, "where": str(f.relative_to(REPO)), "status": "undeclared-restype",
                                 "detail": f"called without restype; C returns {p['ret']} ({p['where']})"})
    for s, bl in by_sym.items():
        sigs = {(tuple(map(norm, b["args"])) if b["args"] is not None else None,
                 norm(b["ret"]) if b["ret"] else None) for b in bl}
        full = {x for x in sigs if None not in x}
        if len(full) > 1:
            rows.append({"sym": s, "where": ", ".join(b["where"] for b in bl), "status": "inconsistent",
                         "detail": f"bound differently: {sorted(map(str, full))}"})
    if as_json:
        print(json.dumps(rows, indent=1))
    else:
        order = {"mismatch": 0, "inconsistent": 1, "undeclared-restype": 2, "loose": 3, "no-prototype": 4, "ok": 5}
        for r in sorted(rows, key=lambda r: (order[r["status"]], r["sym"])):
            print(f"{r['status']:<18} {r['sym']:<28} {r['where']:<48} {r.get('c', '')}  {r.get('detail', '')}")
        cnt = {k: sum(r["status"] == k for r in rows) for k in order}
        print("summary:", ", ".join(f"{k} {v}" for k, v in cnt.items() if v))
    return 1 if any(r["status"] in ("mismatch", "inconsistent", "undeclared-restype") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())

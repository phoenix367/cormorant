"""
Check plugins for interface facts (facts.yaml ``check: NAME``, ``args: {...}``).

Each plugin is ``fn(args, ctx) -> [(level, message, where)]`` with level
error | warn | info; facts.py turns them into the fact's findings.

  register_map     a kernel's AXI-Lite registers: the HLS s_axilite ports, the
                   generated driver header (when built), an RTL kernel's driver
                   table (its --check --json), the performance-model key
                   (src/perf_calls.FIELDS), the timeline's decoder table and
                   every C file that writes them through the driver
  cli_flags        a script's argparse flags (its --help usage) against the
                   option lists of its docs
  ctypes_bindings  ctypes bindings against the C prototypes
                   (.claude/agents/code-audit/ctypes_check.py)
  config_keys      an example config's keys against the code that reads it
                   (.claude/agents/code-audit/config_keys.py), with waivers
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

Finding = Tuple[str, str, str]
AUDIT = Path(__file__).resolve().parents[2] / ".claude" / "agents" / "code-audit"   # the helpers of this checkout


def _line_of(text: str, pattern: str) -> int:
    m = re.search(pattern, text, re.M)
    return text.count("\n", 0, m.start()) + 1 if m else 0


# ─────────────────────────────────────────────────────────────────────────────
# register_map
# ─────────────────────────────────────────────────────────────────────────────

def _decode_keys(text: str) -> Dict[str, List[str]]:
    """{kernel: [field, ...]} of the viewer's ``const DECODE = {...};``."""
    m = re.search(r"const DECODE = \{(.*?)\};", text, re.S)
    if not m:
        return {}
    return {k: re.findall(r"(\w+):", body) for k, body in re.findall(r"(\w+): \{([^}]*)\}", m.group(1))}


def register_map(args: dict, ctx) -> List[Finding]:
    out: List[Finding] = []
    kernel, prefix, hls = args["kernel"], args["prefix"], args["hls"]
    src = ctx.read(hls)
    ports = [p for p in re.findall(r"#pragma HLS INTERFACE s_axilite port=(\w+)", src) if p != "return"]
    if not ports:
        return [("error", "no s_axilite ports found", hls)]
    P = set(ports)
    out.append(("info", f"{len(P)} registers: {', '.join(ports)}", hls))

    # the generated driver (a build artifact: checked where it is built)
    headers = ctx.glob(args["driver"]) if args.get("driver") else []
    for h in headers:
        regs = {r.lower() for r in re.findall(r"#define \w+_CTRL_ADDR_(\w+)_DATA\b", ctx.read(h))}
        if regs != P:                       # a local build artifact: warn, the HLS source is the truth
            out.append(("warn", f"driver registers differ from the HLS ports: only in the driver "
                                 f"{sorted(regs - P)}, only in HLS {sorted(P - regs)} (re-export the IP)", h))
    if args.get("driver") and not headers:
        out.append(("info", f"no driver header built here ({args['driver']})", ""))

    # an RTL implementation of the kernel: its driver generator's register table,
    # which the generator itself checks against the RTL address constants
    if args.get("rtl_driver"):
        rel = args["rtl_driver"]
        rtl = f" --rtl {args['rtl']}" if args.get("rtl") else ""
        r = json.loads(ctx.run(f"{{python}} {rel} --check --json{rtl}"))
        regs = r["registers"]
        for e in r["errors"]:
            out.append(("error", f"driver table vs the RTL: {e}", rel))
        if set(regs) != P:
            out.append(("error", f"RTL driver registers differ from the HLS ports: only in the RTL driver "
                                 f"{sorted(set(regs) - P)}, only in HLS {sorted(P - set(regs))}", rel))
        for h in headers:
            offs = {n.lower(): int(v, 16) for n, v in
                    re.findall(r"#define \w+_CTRL_ADDR_(\w+)_DATA\s+(0x[0-9a-fA-F]+)", ctx.read(h))}
            diff = sorted(n for n in set(regs) & set(offs) if offs[n] != regs[n]["offset"])
            if diff:                        # a local build artifact: warn, as above
                out.append(("warn", f"RTL driver offsets differ from this HLS driver for {', '.join(diff)} "
                                     f"(re-export the HLS IP, or fix the RTL)", h))
        if not r["errors"]:
            out.append(("info", f"RTL driver table: {len(regs)} registers, offsets as in the RTL", rel))

    # the performance-model key
    fields = args["fields"] if "fields" in args else \
        ctx.py("src.perf_calls", "FIELDS", "inference-scheduler").get(kernel)
    if fields is None:
        out.append(("error", f"src/perf_calls.FIELDS has no {kernel}", "inference-scheduler/src/perf_calls.py"))
        fields = []
    alias = args.get("fields_alias", {})
    not_keyed = args.get("not_keyed", {})
    keyed = {alias.get(f, f) for f in fields}
    pc = "inference-scheduler/src/perf_calls.py"
    for f in fields:
        if alias.get(f, f) not in P:
            out.append(("error", f"FIELDS[{kernel}] field {f!r} is no register"
                                 + (f" (alias of {alias[f]!r})" if f in alias else ""), pc))
    for p in ports:
        if p not in keyed and p not in not_keyed:
            out.append(("error", f"register {p!r} is neither a FIELDS[{kernel}] field nor listed in not_keyed: "
                                 f"the performance model would not see it (add it to FIELDS, or to not_keyed "
                                 f"with the reason)", pc))
    for p in not_keyed:
        if p not in P:
            out.append(("warn", f"not_keyed lists {p!r}, which is no register any more (stale entry)", "facts.yaml"))

    # the timeline's decoder table
    if args.get("decode"):
        dk = _decode_keys(ctx.read(args["decode"])).get(kernel, [])
        for k in dk:
            if k not in fields:
                out.append(("error", f"DECODE[{kernel}] decodes {k!r}, which is no FIELDS[{kernel}] field",
                            args["decode"]))

    # the C that writes the registers through the driver
    for rel in args.get("writers_all", []) + args.get("writers", []):
        text = ctx.read(rel)
        setters = set(re.findall(rf"\b{prefix}_Set_(\w+)", text))
        for s in sorted(setters - P):
            ln = _line_of(text, prefix + "_Set_" + s + r"\b")
            out.append(("error", f"sets {s!r}, which is no register", f"{rel}:{ln}"))
        if rel in args.get("writers_all", []):
            miss = [p for p in ports if p not in setters]
            if miss:
                out.append(("error", f"never sets {', '.join(miss)}: the register keeps whatever the last "
                                     f"program on the board wrote", rel))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# cli_flags
# ─────────────────────────────────────────────────────────────────────────────

FLAG_AT_START = re.compile(r"^\s*(?:\|\s*)?`?(--[a-z0-9][a-z0-9-]*)")


def _usage_flags(text: str) -> List[str]:
    """The flags of argparse's ``usage:`` block (everything the parser accepts)."""
    block = text.split("\n\n", 1)[0] if text.lstrip().startswith("usage:") else \
        text[text.find("usage:"):].split("\n\n", 1)[0]
    seen: List[str] = []
    for f in re.findall(r"(--[a-z0-9][a-z0-9-]*)", block):
        if f not in seen:
            seen.append(f)
    return seen


def cli_flags(args: dict, ctx) -> List[Finding]:
    out: List[Finding] = []
    script, cwd = args["script"], args.get("cwd", ".")
    helptext = ctx.run(f"{{python}} {os.path.relpath(script, cwd)} --help", cwd)
    ignore = set(args.get("ignore", ["--help"]))
    accepted = [f for f in _usage_flags(helptext) if f not in ignore]
    out.append(("info", f"{len(accepted)} flags accepted", script))
    for doc in args.get("docs", []):
        rel = doc["file"]
        text = ctx.read(rel)
        m = re.search(doc["start"], text, re.M)
        if not m:
            out.append(("error", f"section /{doc['start']}/ not found", rel))
            continue
        rest = text[m.end():]
        e = re.search(doc["end"], rest, re.M) if doc.get("end") else None
        body = rest[:e.start()] if e else rest
        base = text.count("\n", 0, m.end())
        documented: Dict[str, int] = {}
        pat = re.compile(doc["pattern"]) if doc.get("pattern") else None     # e.g. a usage synopsis
        for i, line in enumerate(body.splitlines()):
            found = pat.findall(line) if pat else [fm.group(1)] if (fm := FLAG_AT_START.match(line)) else []
            for flag in found:
                documented.setdefault(flag, base + i + 1)
        for f in accepted:
            if f not in documented:
                out.append(("error", f"{f} is accepted by {script} but not documented here", f"{rel}:{base + 1}"))
        for f, ln in documented.items():
            if f not in accepted and f not in ignore:
                out.append(("error", f"documents {f}, which {script} does not accept", f"{rel}:{ln}"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ctypes_bindings / config_keys: the code-audit helpers
# ─────────────────────────────────────────────────────────────────────────────

def _audit_json(ctx, argv: List[str]):
    r = subprocess.run([sys.executable, str(AUDIT / argv[0]), *argv[1:], "--json"], cwd=ctx.repo,
                       capture_output=True, text=True, timeout=300)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"{argv[0]} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout)


ERROR_STATUSES = ("mismatch", "inconsistent", "undeclared-restype")


def ctypes_bindings(args: dict, ctx) -> List[Finding]:
    rows = _audit_json(ctx, ["ctypes_check.py", *args["files"]])
    out: List[Finding] = []
    external = set(args.get("external", []))
    counts: Dict[str, int] = {}
    for b in rows:
        st = b["status"]
        counts[st] = counts.get(st, 0) + 1
        if st == "ok":
            continue
        where = b["where"]
        what = f"{b['sym']}: {st}" + (f" — {b['detail']}" if b.get("detail") else "") \
            + (f" (C: {b['c_sig']} at {b['c']})" if b.get("c_sig") else "")
        if st in ERROR_STATUSES:
            out.append(("error", what, where))
        elif st == "no-prototype" and b["sym"] not in external:
            out.append(("warn", what + " — not in this fact's external symbols", where))
        else:
            out.append(("info", what, where))
    out.insert(0, ("info", f"{len(rows)} bindings: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())), ""))
    return out


def config_keys(args: dict, ctx) -> List[Finding]:
    argv = ["config_keys.py", args["example"], "--code", *args["code"]]
    if args.get("roots"):
        argv += ["--roots", args["roots"]]
    d = _audit_json(ctx, argv)
    waive: Dict[str, str] = args.get("waive", {})
    out: List[Finding] = [("info", f"{d.get('keys', '?')} keys in the example", args["example"])]
    seen = set()
    for k in d.get("never_read", []):
        key = k if isinstance(k, str) else k.get("key")
        seen.add(key)
        out.append(("info", f"{key}: never read — waived: {waive[key]}", args["example"]) if key in waive
                   else ("error", f"{key}: in the example, read by no code (dead option, or read under another name)",
                         args["example"]))
    for item in d.get("read_but_not_in_example", []):
        key, where = item["key"], ", ".join(item.get("where", [])[:3])
        seen.add(key)
        out.append(("info", f"{key}: not in the example — waived: {waive[key]}", where) if key in waive
                   else ("error", f"{key}: read by the code, missing from the example (undocumented option, "
                                  f"or a typo)", where))
    for key in waive:
        if key not in seen:
            out.append(("warn", f"waiver for {key!r} no longer needed (remove it)", "facts.yaml"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# names_in_docs / script_flags / json_excerpt
# ─────────────────────────────────────────────────────────────────────────────

def _section(text: str, start: str, end: str = None):
    """(body, first line number) of the part of ``text`` after /start/ up to /end/."""
    m = re.search(start, text, re.M)
    if not m:
        return None, 0
    rest = text[m.end():]
    e = re.search(end, rest, re.M) if end else None
    return (rest[:e.start()] if e else rest), text.count("\n", 0, m.start()) + 1


def names_in_docs(args: dict, ctx) -> List[Finding]:
    """A set of names from the code (``names``: a source spec) against the
    names a doc section shows (``pattern``'s group 1 over the section):
    every name must be shown (unless ``forward: false`` or in ``skip``);
    with ``reverse`` every shown name must exist (``extra_ok``: names the
    section may show besides them, e.g. ops it rejects)."""
    import facts as F
    names = sorted(set(F.extract(args["names"], ctx)))
    out: List[Finding] = [("info", f"{len(names)} names: {', '.join(names)}", "")]
    for doc in args["docs"]:
        rel = doc["file"]
        body, line = _section(ctx.read(rel), doc["start"], doc.get("end"))
        if body is None:
            out.append(("error", f"section /{doc['start']}/ not found", rel))
            continue
        shown = set(re.findall(doc.get("pattern", r"`([A-Za-z][\w.]*)"), body))
        only = doc.get("only")                              # the names this doc covers (a glob)
        want = [n for n in names if (not only or any(re.fullmatch(o.replace("*", ".*"), n) for o in only))
                and n not in doc.get("skip", [])]
        miss = [n for n in want if n not in shown] if doc.get("forward", True) else []
        if miss:
            out.append(("error", f"not shown here: {', '.join(miss)}", f"{rel}:{line}"))
        if doc.get("reverse"):
            extra = sorted(shown - set(names) - set(doc.get("extra_ok", [])))
            if extra:
                out.append(("error", f"shown here but {args.get('missing', 'not in the code')}: {', '.join(extra)} "
                                     f"(or add them to extra_ok)", f"{rel}:{line}"))
    return out


def script_flags(args: dict, ctx) -> List[Finding]:
    """Flags a function passes to other scripts (string literals, as
    code-audit/flags_check.py finds them) against the flags the callee(s)
    accept (their argparse ``usage:``, which includes helper-added flags)."""
    sys.path.insert(0, str(AUDIT))
    import flags_check
    out: List[Finding] = []
    for pair in args["pairs"]:
        caller, fn = pair["caller"], pair.get("function")
        callees = pair["callee"] if isinstance(pair["callee"], list) else [pair["callee"]]
        accepted = set()
        for c in callees:
            accepted |= set(_usage_flags(ctx.run(f"{{python}} {c} --help")))
        lits, dynamic = flags_check.passed_flags(ctx.repo / caller, fn)
        label = f"{caller}{':' + fn if fn else ''} -> {', '.join(callees)}"
        bad = sorted((f, ln) for f, ln in lits.items() if f not in accepted)
        for f, ln in bad:
            out.append(("error", f"passes {f}, which {' / '.join(callees)} does not accept", f"{caller}:{ln}"))
        out.append(("info", f"{label}: {len(lits)} flags, {len(bad)} unknown"
                            + (f", {dynamic} built at run time (not checked)" if dynamic else ""), caller))
        if not lits and not dynamic:
            out.append(("warn", f"{label}: no flags found any more (moved? update facts.yaml)", caller))
    return out


def _strip_jsonc(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r'(?m)^((?:[^"\n]|"(?:\\.|[^"\\])*")*?)\s*//.*$', r"\1", text)


def json_excerpt(args: dict, ctx) -> List[Finding]:
    """A doc's copy of a JSON file (the first ```json / ```jsonc block after
    /start/): every value it shows equals the file's; with ``complete`` it
    shows every key (keys starting with ``_`` are notes and skipped)."""
    real = json.loads(ctx.read(args["json"]))
    rel = args["doc"]
    body, line = _section(ctx.read(rel), args["start"])
    if body is None:
        return [("error", f"section /{args['start']}/ not found", rel)]
    m = re.search(r"```jsonc?\n(.*?)```", body, re.S)
    if not m:
        return [("error", "no json code block in the section", f"{rel}:{line}")]
    try:
        shown = json.loads(_strip_jsonc(m.group(1)))
    except ValueError as e:
        return [("error", f"the doc's JSON does not parse: {e}", f"{rel}:{line}")]
    out: List[Finding] = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k, v in a.items():
                if k not in b:
                    out.append(("error", f"{path}{k}: shown, but not in {args['json']}", f"{rel}:{line}"))
                else:
                    walk(v, b[k], f"{path}{k}.")
            if args.get("complete"):
                for k in b:
                    if k not in a and not k.startswith("_"):
                        out.append(("error", f"{path}{k}: in {args['json']} ({b[k]!r}), not shown", f"{rel}:{line}"))
        elif a != b:
            out.append(("error", f"{path.rstrip('.')}: shows {a!r}, {args['json']} has {b!r}", f"{rel}:{line}"))
    walk(shown, real, "")
    return out


PLUGINS = {"register_map": register_map, "cli_flags": cli_flags, "ctypes_bindings": ctypes_bindings,
           "config_keys": config_keys, "names_in_docs": names_in_docs, "script_flags": script_flags,
           "json_excerpt": json_excerpt}

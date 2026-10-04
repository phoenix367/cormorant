#!/usr/bin/env python3
"""
facts.py — the project's fact registry (facts.yaml at the repo root): facts
that code and docs state in several places, where each one is true, and the
places that repeat it.

  value       derived from the code by a ``source`` (a command, a file, a
              glob, a Python expression) and quoted by ``mentions`` (regex
              locators, or ``<!-- fact:ID -->VALUE<!-- /fact -->`` markers
              found anywhere); ``fix: auto`` rewrites a stale mention
  interface   one component and the components that must agree with it: a
              ``producer`` and ``consumers`` extracted to the same mapping
              (an enum and its copies), or a ``check`` plugin (plugins.py:
              register maps, CLI flags, ctypes bindings, config keys)
  recorded    measured, so not derivable: a ``value`` with ``provenance``
              (commit, date, how).  Mentions must quote it; it is stale when
              commits touched its ``depends_on`` since the provenance commit;
              ``verify`` re-reads it from where it can be measured (local builds)

usage: python3 tools/facts/facts.py list
       python3 tools/facts/facts.py check [ID|GLOB ...] [--json] [--strict]
       python3 tools/facts/facts.py fix [ID|GLOB ...] [--dry-run]
       python3 tools/facts/facts.py impact ID|PATH ...
       python3 tools/facts/facts.py changed [--base origin/main | --staged] [--json] [--quiet]
       python3 tools/facts/facts.py verify [ID|GLOB ...]
       python3 tools/facts/facts.py install-hook [--force | --uninstall]

check exits 1 when a fact fails (with --strict also on warnings); facts.yaml's
format: tools/facts/README.md.  install-hook makes `git commit` run
tools/facts/pre-commit: `changed --staged`, the facts the staged files touch.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import glob as globmod
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:                                         # pragma: no cover
    sys.exit("facts.py needs PyYAML: run it with inference-scheduler/.venv/bin/python "
             "(pyyaml is in inference-scheduler/requirements.txt) or pip install pyyaml")

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "facts.yaml"
KINDS = ("value", "interface", "recorded")
MARKER = re.compile(r"<!-- fact:([\w.\-]+) -->(.*?)<!-- /fact -->")
MARKER_SUFFIXES = (".md", ".txt", ".html", ".rst")    # files whose fact markers count (mentions, the hook)
LEVELS = ("error", "warn", "info")


class FactError(Exception):
    """A source or locator that cannot be evaluated (reported, not raised)."""


@dataclass
class Finding:
    level: str                      # error | warn | info
    msg: str
    where: str = ""                 # file[:line]

    def as_dict(self):
        return {"level": self.level, "msg": self.msg, "where": self.where}


@dataclass
class Result:
    id: str
    kind: str
    status: str = "ok"              # ok | fail | warn | skipped
    value: Any = None
    findings: List[Finding] = field(default_factory=list)

    def add(self, level, msg, where=""):
        self.findings.append(Finding(level, msg, where))

    def settle(self):
        levels = {f.level for f in self.findings}
        if self.status != "skipped":
            self.status = "fail" if "error" in levels else "warn" if "warn" in levels else "ok"
        return self

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "status": self.status, "value": self.value,
                "findings": [f.as_dict() for f in self.findings]}


# ─────────────────────────────────────────────────────────────────────────────
# context: the repo, the interpreter for Python sources, caches
# ─────────────────────────────────────────────────────────────────────────────

class Ctx:
    def __init__(self, repo: Path = REPO, registry: Optional[dict] = None):
        self.repo = repo
        self.reg = registry if registry is not None else load(repo / "facts.yaml")
        py = self.reg.get("python")
        self.python = str(repo / py) if py and (repo / py).exists() else sys.executable
        self.facts = {f["id"]: f for f in self.reg.get("facts", [])}
        self._values: Dict[str, Any] = {}
        self._cmd: Dict[tuple, str] = {}
        self._py: Dict[tuple, Any] = {}
        self._tracked: Optional[List[str]] = None

    def path(self, rel: str) -> Path:
        return self.repo / rel

    def read(self, rel: str) -> str:
        p = self.path(rel)
        if not p.exists():
            raise FactError(f"{rel}: no such file")
        return p.read_text(errors="replace")

    def tracked(self) -> List[str]:
        if self._tracked is None:
            try:
                out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=self.repo,
                                     capture_output=True, text=True, check=True).stdout
                self._tracked = [ln for ln in out.splitlines() if ln and (self.repo / ln).is_file()]
            except (OSError, subprocess.CalledProcessError):
                self._tracked = [str(p.relative_to(self.repo)) for p in self.repo.rglob("*") if p.is_file()]
        return self._tracked

    def glob(self, pattern: str) -> List[str]:
        """Repo-relative paths: tracked files matching the pattern, or (for a
        build artifact) the files on disk."""
        hits = sorted(p for p in self.tracked() if fnmatch.fnmatch(p, pattern))
        if not hits:
            hits = sorted(str(Path(p).relative_to(self.repo)) for p in globmod.glob(str(self.repo / pattern),
                                                                                      recursive=True))
        return hits

    def run(self, cmd: str, cwd: str = ".", timeout: int = 600) -> str:
        key = (cmd, cwd)
        if key not in self._cmd:
            env = dict(os.environ, NO_COLOR="1", PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run(cmd.replace("{python}", self.python), shell=True, cwd=self.path(cwd),
                               capture_output=True, text=True, timeout=timeout, env=env)
            if r.returncode not in (0, 5):                      # pytest: 5 = nothing collected
                raise FactError(f"`{cmd}` exited {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
            self._cmd[key] = r.stdout + r.stderr
        return self._cmd[key]

    def py(self, module: str, expr: str, cwd: str = ".") -> Any:
        """``expr`` evaluated in ``module``'s namespace (private names too) by the
        configured interpreter (the scheduler's venv), returned as JSON (memoised)."""
        key = (module, expr, cwd)
        if key not in self._py:
            self._py[key] = self._py_run(module, expr, cwd)
        return self._py[key]

    def _py_run(self, module: str, expr: str, cwd: str) -> Any:
        code = (f"import importlib, json, sys; sys.path.insert(0, '.'); m = importlib.import_module({module!r}); "
                f"print(json.dumps(eval({expr!r}, vars(m)), default=list))")
        r = subprocess.run([self.python, "-c", code], cwd=self.path(cwd), capture_output=True, text=True,
                           timeout=300, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if r.returncode:
            raise FactError(f"python {module}: {r.stderr.strip()[-300:]}")
        return json.loads(r.stdout.strip().splitlines()[-1])

    def value_of(self, fid: str) -> Any:
        """A value / recorded fact's value (memoised), for ``{fact:ID}``."""
        if fid not in self._values:
            f = self.facts.get(fid)
            if f is None:
                raise FactError(f"unknown fact {fid!r}")
            self._values[fid] = f["value"] if f["kind"] == "recorded" else extract(f["source"], self)
        return self._values[fid]

    def subst(self, s: str) -> str:
        return re.sub(r"\{fact:([\w.\-]+)\}", lambda m: str(self.value_of(m.group(1))), s)


# ─────────────────────────────────────────────────────────────────────────────
# the registry
# ─────────────────────────────────────────────────────────────────────────────

def load(path: Path) -> dict:
    reg = yaml.safe_load(path.read_text()) or {}
    seen = set()
    for f in reg.get("facts", []):
        fid = f.get("id")
        if not fid or fid in seen:
            raise SystemExit(f"{path}: fact without an id, or a duplicate id: {fid!r}")
        seen.add(fid)
        if f.get("kind") not in KINDS:
            raise SystemExit(f"{path}: {fid}: kind must be one of {KINDS}")
        need = {"value": ("source",), "recorded": ("value", "provenance"), "interface": ()}[f["kind"]]
        for k in need:
            if k not in f:
                raise SystemExit(f"{path}: {fid}: a {f['kind']} fact needs {k!r}")
        if f["kind"] == "interface" and not (f.get("check") or f.get("producer")):
            raise SystemExit(f"{path}: {fid}: an interface needs a check plugin or a producer")
    return reg


def select(ctx: Ctx, patterns: Iterable[str]) -> List[dict]:
    pats = list(patterns)
    facts = ctx.reg.get("facts", [])
    if not pats:
        return facts
    out = [f for f in facts if any(fnmatch.fnmatch(f["id"], p) for p in pats)]
    unknown = [p for p in pats if not any(fnmatch.fnmatch(f["id"], p) for f in facts)]
    if unknown:
        raise SystemExit(f"no fact matches {', '.join(unknown)} (facts.py list)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# sources: spec -> value
# ─────────────────────────────────────────────────────────────────────────────

def _one(xs):
    xs = list(xs)
    if len(xs) != 1:
        raise FactError(f"expected exactly one item, got {len(xs)}: {xs[:5]}")
    return xs[0]


SAFE = {"len": len, "sorted": sorted, "set": set, "list": list, "dict": dict, "int": int, "float": float,
        "str": str, "round": round, "enumerate": enumerate, "zip": zip, "min": min, "max": max, "sum": sum,
        "any": any, "all": all, "tuple": tuple, "isinstance": isinstance}


def _eval(expr: str, v: Any) -> Any:
    env = {"__builtins__": SAFE, "ast": ast.literal_eval, "json": json, "re": re, "one": _one,
           "basename": os.path.basename, "stem": lambda p: os.path.splitext(os.path.basename(p))[0]}
    try:
        return eval(expr, env, {"v": v})                              # noqa: S307 — trusted registry
    except FactError:
        raise
    except Exception as e:                                            # noqa: BLE001
        raise FactError(f"transform {expr!r}: {type(e).__name__}: {e}") from None


def _regex_value(text: str, spec: dict, where: str) -> Any:
    flags = re.M | (re.S if spec.get("dotall") else 0)
    ms = list(re.finditer(spec["regex"], text, flags))
    if not ms:
        raise FactError(f"{where}: no match for /{spec['regex']}/")
    if spec.get("as") == "dict":
        pairs = [(m.group(2), m.group(1)) if spec.get("swap") else (m.group(1), m.group(2)) for m in ms]
        return {k: (int(v) if re.fullmatch(r"-?\d+", v) else v) for k, v in pairs}
    if spec.get("all"):
        return [m.group(1) if m.re.groups == 1 else m.groups() for m in ms]
    m = ms[0]
    return m.group(1) if m.re.groups == 1 else m.groups()


def extract(spec: Any, ctx: Ctx) -> Any:
    """Evaluate a source spec (see the module docstring and README)."""
    if not isinstance(spec, dict):
        return spec                                                   # a literal
    if "parts" in spec:
        return {k: extract(s, ctx) for k, s in spec["parts"].items()}
    if "cmd" in spec:
        out = ctx.run(ctx.subst(spec["cmd"]), spec.get("cwd", "."), spec.get("timeout", 600))
        v = _regex_value(out, {"regex": spec["parse"]}, f"`{spec['cmd']}`") if "parse" in spec else out.strip()
    elif "py" in spec:
        module, _, expr = spec["py"].partition(":")
        v = ctx.py(module, expr or "None", spec.get("cwd", "."))
    elif "glob" in spec:
        v = ctx.glob(ctx.subst(spec["glob"]))
    elif "file" in spec:
        rel = ctx.subst(spec["file"])
        text = ctx.read(rel)
        if "json" in spec:
            v = json.loads(text)
            for k in [k for k in spec["json"].split(".") if k]:
                v = v[k]
        elif "regex" in spec:
            v = _regex_value(text, spec, rel)
        else:
            v = text
    elif "fact" in spec:
        v = ctx.value_of(spec["fact"])
    else:
        raise FactError(f"unknown source {spec}")
    if "transform" in spec:
        v = _eval(spec["transform"], v)
    if spec.get("type") in ("int", "float", "str"):
        v = {"int": int, "float": float, "str": str}[spec["type"]](v)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# mentions: where a value is quoted
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Hit:
    file: str
    line: int
    key: Optional[str]               # None: the whole value; else a dict key
    start: int                       # span of the quoted text in the file
    end: int
    text: str
    fix: str                         # auto | report
    fmt: Optional[str] = None        # the mention's `format`: how this text renders the value


def _norm(s: Any) -> str:
    s = str(s).strip().replace(" ", "").replace(" ", "").replace(" ", "")
    return re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", s)


def _render(want: Any, fmt: Optional[str]) -> str:
    """The text a mention should hold: ``format`` (an expression on ``v``) or the value."""
    if fmt:
        return str(_eval(fmt, want))
    return str(int(want)) if isinstance(want, float) and want.is_integer() else str(want)


def _same(quoted: str, want: Any, tol: Any = None, fmt: Optional[str] = None) -> bool:
    """``tol``: an absolute tolerance, or {abs: x, rel: y} (numbers only)."""
    if tol is not None and not fmt:
        try:
            q, w = float(_norm(quoted)), float(want)
        except (TypeError, ValueError):                     # text (a commit id): compared exactly
            q = w = None
        if q is not None:
            lim = tol if not isinstance(tol, dict) else max(tol.get("abs", 0), tol.get("rel", 0) * abs(w))
            return abs(q - w) <= lim
    return _norm(quoted).lower() == _norm(_render(want, fmt)).lower()


def _tol_label(tol: Any) -> str:
    if tol is None:
        return ""
    if isinstance(tol, dict):
        return " (± " + " / ".join(f"{tol['rel'] * 100:g} %" if k == "rel" else f"{tol['abs']}" for k in tol) + ")"
    return f" (± {tol})"


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def mention_hits(fact: dict, ctx: Ctx, res: Result) -> List[Hit]:
    """Every quoted occurrence: the fact's regex locators, then the markers."""
    hits: List[Hit] = []
    default_fix = fact.get("fix", "report")
    for loc in fact.get("mentions", []):
        rel = ctx.subst(loc["file"])
        try:
            text = ctx.read(rel)
        except FactError as e:
            res.add("error", f"mention file missing: {e}", rel)
            continue
        ms = list(re.finditer(loc["regex"], text, re.M | (re.S if loc.get("dotall") else 0)))
        if not ms:
            res.add("error", f"locator lost: /{loc['regex']}/ matches nothing (the text changed? "
                             f"update facts.yaml)", rel)
            continue
        if "count" in loc and len(ms) != loc["count"]:
            res.add("error", f"locator /{loc['regex']}/ matches {len(ms)} times, expected {loc['count']}", rel)
        fix = loc.get("fix", default_fix)
        for m in ms:
            names = [n for n in m.re.groupindex if m.group(n) is not None]
            groups = [(n, m.span(n), m.group(n)) for n in names] or [(None, m.span(1), m.group(1))]
            for key, (a, b), txt in groups:
                hits.append(Hit(rel, _line(text, a), key, a, b, txt, fix, loc.get("format")))
    for rel in ctx.tracked():                                     # markers anywhere
        if not rel.endswith(MARKER_SUFFIXES):
            continue
        try:
            text = ctx.read(rel)
        except (FactError, OSError):
            continue
        if "<!-- fact:" not in text:
            continue
        for m in MARKER.finditer(text):
            mid = m.group(1)
            if mid == fact["id"] or mid.startswith(fact["id"] + "."):
                key = mid[len(fact["id"]) + 1:] or None
                hits.append(Hit(rel, _line(text, m.start(2)), key, m.start(2), m.end(2), m.group(2), default_fix))
    return hits


def _want(value: Any, key: Optional[str]) -> Any:
    if key is None:
        return value
    if not isinstance(value, dict) or key not in value:
        raise FactError(f"the value has no key {key!r}")
    return value[key]


def check_mentions(fact: dict, value: Any, ctx: Ctx, res: Result) -> List[Hit]:
    """The stale hits (and findings for them)."""
    stale = []
    tol = fact.get("tolerance")
    for h in mention_hits(fact, ctx, res):
        try:
            want = _want(value, h.key)
        except FactError as e:
            res.add("error", str(e), f"{h.file}:{h.line}")
            continue
        if not _same(h.text, want, tol, h.fmt):
            stale.append(h)
            label = f"{h.key} " if h.key else ""
            shown = repr(want) + (f" = {_render(want, h.fmt)!r} here" if h.fmt else "") \
                + (_tol_label(tol) if isinstance(want, (int, float)) else "")
            res.add("error", f"says {label}{h.text!r}, the fact is {shown}"
                             + (" (fix: auto)" if h.fix == "auto" else ""), f"{h.file}:{h.line}")
    return stale


def apply_fixes(hits: List[Hit], value: Any, ctx: Ctx, dry: bool = False) -> List[str]:
    """Rewrite the ``fix: auto`` stale hits, last span first per file."""
    done = []
    by_file: Dict[str, List[Hit]] = {}
    for h in hits:
        if h.fix == "auto":
            by_file.setdefault(h.file, []).append(h)
    for rel, hs in by_file.items():
        text = ctx.read(rel)
        for h in sorted(hs, key=lambda h: -h.start):
            new = _render(_want(value, h.key), h.fmt)
            text = text[:h.start] + new + text[h.end:]
            done.append(f"{rel}:{h.line}: {h.text!r} -> {new!r}")
        if not dry:
            ctx.path(rel).write_text(text)
    return done


# ─────────────────────────────────────────────────────────────────────────────
# checks per kind
# ─────────────────────────────────────────────────────────────────────────────

def _restrict(d: dict, keys: Optional[str]) -> dict:
    return {k: v for k, v in d.items() if keys is None or fnmatch.fnmatch(str(k), keys)}


def check_fact(fact: dict, ctx: Ctx, fix: bool = False, dry: bool = False) -> Result:
    res = Result(fact["id"], fact["kind"])
    try:
        if fact["kind"] == "value":
            res.value = ctx.value_of(fact["id"])
            for p in fact.get("exists", []):
                rel = ctx.subst(p)
                if not ctx.path(rel).exists():
                    res.add("error", f"required file missing: {rel}", rel)
        elif fact["kind"] == "recorded":
            res.value = fact["value"]
            _staleness(fact, ctx, res)
        if fact["kind"] in ("value", "recorded"):
            stale = check_mentions(fact, res.value, ctx, res)
            if fix and stale:
                for line in apply_fixes(stale, res.value, ctx, dry):
                    res.add("info", ("would fix " if dry else "fixed ") + line)
                if not dry:                                   # re-check what is left
                    left = [f for f in res.findings if f.level != "error" or "(fix: auto)" not in f.msg]
                    res.findings = left
        else:
            _check_interface(fact, ctx, res)
    except FactError as e:
        if fact.get("optional"):                            # its source is not here (a submodule, a build)
            res.status = "skipped"
            res.findings = [Finding("info", f"skipped, source not available here: {e}")]
            return res
        res.add("error", str(e))
    except subprocess.TimeoutExpired as e:
        res.add("error", f"timed out: {e.cmd}")
    return res.settle()


def _check_interface(fact: dict, ctx: Ctx, res: Result) -> None:
    if fact.get("check"):
        from plugins import PLUGINS                           # tools/facts/plugins.py
        fn = PLUGINS.get(fact["check"])
        if fn is None:
            raise FactError(f"unknown check plugin {fact['check']!r} (plugins.PLUGINS)")
        for level, msg, where in fn(fact.get("args", {}), ctx):
            res.add(level, msg, where)
        return
    prod = extract(fact["producer"], ctx)
    res.value = prod
    if not isinstance(prod, dict):
        raise FactError("the producer must extract to a mapping")
    for c in fact.get("consumers", []):
        name = c.get("name", c.get("file", "?"))
        try:
            got = extract(c, ctx)
        except FactError as e:
            res.add("error", f"{name}: {e}", c.get("file", ""))
            continue
        if not isinstance(got, dict):
            res.add("error", f"{name}: extracted {got!r}, not a mapping", c.get("file", ""))
            continue
        drop = set(c.get("exclude", []))                   # keys this consumer does not handle
        want = {k: v for k, v in _restrict(prod, c.get("keys")).items() if k not in drop}
        got = {k: v for k, v in _restrict(got, c.get("keys")).items() if k not in drop}
        if c.get("subset"):                                 # it may name only some of them
            want = {k: v for k, v in want.items() if k in got}
        if got == want:
            continue
        diffs = [f"{k}: {got.get(k, '—')} (producer {want.get(k, '—')})"
                 for k in sorted(set(want) | set(got), key=str) if got.get(k) != want.get(k)]
        res.add("error", f"{name} disagrees with {fact['producer'].get('file', 'the producer')}: "
                         + "; ".join(diffs), c.get("file", ""))


def _staleness(fact: dict, ctx: Ctx, res: Result) -> None:
    """Stale when another fact it was measured under changed (provenance.facts,
    e.g. the bitstream id), or when commits touched its depends_on."""
    prov = fact["provenance"]
    for fid, then in (prov.get("facts") or {}).items():
        try:
            now = ctx.value_of(fid)
        except FactError as e:
            res.add("warn", f"provenance fact {fid}: {e}")
            continue
        if _norm(now) != _norm(then):
            res.add("warn", f"stale: measured under {fid} = {then}, now {now}; re-measure and update value "
                            f"and provenance")
    deps = [ctx.subst(d) for d in fact.get("depends_on", [])]
    commit = prov.get("commit")
    if not (commit and deps):
        return
    try:
        out = subprocess.run(["git", "log", "--format=%h %s", f"{commit}..HEAD", "--", *deps], cwd=ctx.repo,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    except subprocess.CalledProcessError as e:
        res.add("warn", f"provenance commit {commit}: {e.stderr.strip()}")
        return
    if out:
        res.add("warn", f"stale? {len(out)} commit(s) changed its inputs since {commit} ({prov.get('date', '?')}), "
                        f"latest {out[0]!r}; re-measure (facts.py verify {fact['id']}) and update its provenance")


def verify_fact(fact: dict, ctx: Ctx) -> Result:
    """A recorded fact against its ``verify`` sources (missing ones are skipped)."""
    res = Result(fact["id"], fact["kind"], value=fact.get("value"))
    if fact["kind"] != "recorded" or not fact.get("verify"):
        res.status = "skipped"
        res.add("info", "nothing to verify (only recorded facts with `verify` sources)")
        return res
    tol = fact.get("tolerance")
    measured = 0
    for key, spec in fact["verify"].items():
        try:
            got = extract(spec, ctx)
        except FactError as e:
            res.add("info", f"{key}: not measurable here ({e})")
            continue
        measured += 1
        want = _want(fact["value"], key if isinstance(fact["value"], dict) else None)
        if _same(str(got), want, tol):
            res.add("info", f"{key}: measured {got!r} = recorded {want!r}{_tol_label(tol)}")
        else:
            res.add("error", f"{key}: measured {got!r}, recorded {want!r}{_tol_label(tol)}"
                             " — update value and provenance")
    if not measured:
        res.status = "skipped"
    return res.settle()


# ─────────────────────────────────────────────────────────────────────────────
# impact: which files a fact ties together
# ─────────────────────────────────────────────────────────────────────────────

NOT_PATHS = {"regex", "transform", "cmd", "parse", "start", "end", "title", "how", "name", "keys", "py",
             "prefix", "kernel", "id", "kind", "fix", "type", "as", "format", "pattern", "exclude", "extra_ok",
             "function"}
REVIEW_ROLES = ("producer", "consumers", "check", "mentions")     # files that state the fact


def _pathlike(ctx: Ctx, s: str) -> bool:
    """A repo path or glob: an existing file (README.md too), or a path with a
    directory part that exists or holds a glob."""
    if not s or " " in s or s.startswith(("http:", "https:")):
        return False
    if "/" not in s:
        return ctx.path(s).is_file()
    return any(ch in s for ch in "*?[{") or ctx.path(s).exists()


def fact_files(fact: dict, ctx: Ctx) -> Dict[str, List[str]]:
    """{role: [repo paths]} of a fact: source, producer, consumers, mentions,
    plugin args (check), depends_on, watch (globs whose change concerns the
    fact), verify — globs expanded, ``{fact:...}`` resolved when it can be."""
    roles: Dict[str, List[str]] = {}

    def subst(s):
        try:
            return ctx.subst(s)
        except FactError:
            return re.sub(r"\{fact:[^}]*\}", "*", s)

    def add(role, p):
        p = subst(str(p))
        found = ctx.glob(p) if any(ch in p for ch in "*?[") else [p]
        for f in found or [p]:
            if f not in roles.setdefault(role, []):
                roles[role].append(f)

    def walk(role, node, key=None):
        if isinstance(node, dict):
            for k, v in node.items():
                if k not in NOT_PATHS:
                    walk(role, v, k)
        elif isinstance(node, list):
            for x in node:
                walk(role, x, key)
        elif isinstance(node, str) and _pathlike(ctx, subst(node)):
            add(role, node)

    walk("source", fact.get("source", {}))
    walk("producer", fact.get("producer", {}))
    walk("consumers", fact.get("consumers", []))
    walk("check", fact.get("args", {}))
    walk("mentions", fact.get("mentions", []))
    walk("depends_on", fact.get("depends_on", []))
    walk("watch", fact.get("watch", []))
    walk("verify", fact.get("verify", {}))
    for rel in ctx.tracked():                                       # marker mentions
        if rel.endswith(MARKER_SUFFIXES):
            try:
                if f"<!-- fact:{fact['id']}" in ctx.read(rel):
                    add("mentions", rel)
            except (FactError, OSError):
                pass
    return roles


def changed_files(ctx: Ctx, base: str, untracked: bool = False) -> List[str]:
    """Files changed on this branch since ``base`` plus uncommitted ones
    (untracked new files only on request: build artifacts are untracked too)."""
    files = set()
    cmds = [["git", "diff", "--name-only", f"{base}...HEAD"], ["git", "diff", "--name-only", "HEAD"]]
    if untracked:
        cmds.append(["git", "ls-files", "-o", "--exclude-standard"])
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=ctx.repo, capture_output=True, text=True)
        if r.returncode == 0:
            files |= {ln for ln in r.stdout.splitlines() if ln}
    return sorted(files)


def staged_files(ctx: Ctx) -> List[str]:
    """The files of the commit being made (the index against HEAD; git sets
    GIT_INDEX_FILE for ``commit -a`` / ``commit PATH``, which this follows)."""
    r = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ctx.repo, capture_output=True, text=True)
    return sorted(ln for ln in r.stdout.splitlines() if ln) if r.returncode == 0 else []


def unstaged_files(ctx: Ctx) -> set:
    r = subprocess.run(["git", "diff", "--name-only"], cwd=ctx.repo, capture_output=True, text=True)
    return {ln for ln in r.stdout.splitlines() if ln} if r.returncode == 0 else set()


ENGINE = ("facts.yaml", "tools/facts/")       # a change here re-checks every fact


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

MARK = {"ok": "ok", "fail": "FAIL", "warn": "warn", "skipped": "skip"}


def _short(v: Any, n: int = 70) -> str:
    s = json.dumps(v, default=str, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n - 1] + "…"


def print_results(results: List[Result], verbose: bool = False) -> None:
    w = max((len(r.id) for r in results), default=10)
    for r in results:
        print(f"{r.id:{w}s}  {MARK[r.status]:4s}  {_short(r.value) if r.value is not None else ''}")
        for f in r.findings:
            if f.level != "info" or verbose:
                print(f"{'':{w}s}    {f.level:5s} {f.where + ': ' if f.where else ''}{f.msg}")


def cmd_list(ctx: Ctx, args) -> int:
    w = max(len(f["id"]) for f in ctx.reg["facts"])
    for f in ctx.reg["facts"]:
        print(f"{f['id']:{w}s}  {f['kind']:9s}  {f.get('title', '')}")
    return 0


def _results(ctx, facts, fn) -> List[Result]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return [fn(f) for f in facts]


def _exit(results: List[Result], strict: bool) -> int:
    bad = {"fail"} | ({"warn"} if strict else set())
    return 1 if any(r.status in bad for r in results) else 0


def cmd_check(ctx: Ctx, args) -> int:
    results = _results(ctx, select(ctx, args.ids), lambda f: check_fact(f, ctx))
    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=1, default=str))
    else:
        print_results(results, args.verbose)
        n = {s: sum(r.status == s for r in results) for s in MARK}
        print(f"\n{len(results)} facts: {n['ok']} ok, {n['fail']} failed, {n['warn']} with warnings, "
              f"{n['skipped']} skipped")
    return _exit(results, args.strict)


def cmd_fix(ctx: Ctx, args) -> int:
    results = _results(ctx, select(ctx, args.ids), lambda f: check_fact(f, ctx, fix=True, dry=args.dry_run))
    print_results(results, verbose=True)
    return _exit(results, False)


def cmd_verify(ctx: Ctx, args) -> int:
    facts = [f for f in select(ctx, args.ids) if f["kind"] == "recorded"]
    results = _results(ctx, facts, lambda f: verify_fact(f, ctx))
    print_results(results, verbose=True)
    return _exit(results, False)


def cmd_impact(ctx: Ctx, args) -> int:
    for target in args.targets:
        facts = [ctx.facts[target]] if target in ctx.facts else \
            [f for f in ctx.reg["facts"] if any(target == p or fnmatch.fnmatch(p, target)
                                                 for ps in fact_files(f, ctx).values() for p in ps)]
        if not facts:
            print(f"{target}: no fact involves it")
            continue
        for f in facts:
            print(f"{f['id']} ({f['kind']}){' — ' + f['title'] if f.get('title') else ''}")
            for role, paths in fact_files(f, ctx).items():
                for p in paths:
                    print(f"  {role:10s} {p}{'   <- ' if p == target else ''}")
    return 0


def cmd_changed(ctx: Ctx, args) -> int:
    changed = staged_files(ctx) if args.staged else changed_files(ctx, args.base, args.untracked)
    every = any(p == e or p.startswith(e) for p in changed for e in ENGINE)
    touched = []
    for f in ctx.reg["facts"]:
        roles = fact_files(f, ctx)
        files = {p for ps in roles.values() for p in ps}
        hit = sorted(files & set(changed))
        if hit or every:
            touched.append((f, roles, hit))
    results = _results(ctx, [f for f, _, _ in touched], lambda f: check_fact(f, ctx))
    if args.quiet:
        return _report_quiet(ctx, args, changed, touched, results, every)
    if args.json:
        print(json.dumps([{**r.as_dict(), "changed": hit,
                           "unchanged": sorted({p for role, ps in roles.items() if role in REVIEW_ROLES
                                                for p in ps} - set(changed))}
                          for r, (_f, roles, hit) in zip(results, touched, strict=True)], indent=1, default=str))
        return _exit(results, args.strict)
    what = "staged" if args.staged else f"changed against {args.base}"
    print(f"{len(changed)} files {what}; {len(touched)} facts involve them\n")
    for r, (_f, roles, hit) in zip(results, touched, strict=True):
        print_results([r])
        print(f"    changed: {', '.join(hit)}")
        rest = [(role, p) for role, ps in roles.items() for p in ps
                if p not in changed and role in REVIEW_ROLES]
        if rest:
            print("    not changed (review them):")
            for role, p in rest:
                print(f"      {role:10s} {p}")
        print()
    return _exit(results, args.strict)


def _report_quiet(ctx: Ctx, args, changed, touched, results, every) -> int:
    """The pre-commit hook's report: only what is wrong, and how to go on."""
    what = "staged files" if args.staged else "changed files"
    n = {s: sum(r.status == s for r in results) for s in MARK}
    if not touched:
        print(f"facts: no fact involves the {len(changed)} {what}")
        return 0
    head = f"facts: {len(touched)} facts involve the {what}" + (" (the registry changed: all)" if every else "")
    print(f"{head}: {n['ok']} ok" + (f", {n['fail']} failed" if n["fail"] else "")
          + (f", {n['warn']} with warnings" if n["warn"] else ""))
    bad = [r for r in results if r.status in ("fail", "warn")]
    if bad:
        print_results(bad)
    fixable = [r.id for r in results if any("(fix: auto)" in f.msg for f in r.findings)]
    if fixable:
        print(f"  fix the stale counts: python3 tools/facts/facts.py fix {' '.join(fixable)}  (then git add them)")
    dirty = sorted({p for _f, roles, _h in touched for role, ps in roles.items() if role in REVIEW_ROLES
                    for p in ps} & unstaged_files(ctx))
    if dirty and args.staged:
        print(f"  note: the check read the working tree; unstaged changes in {', '.join(dirty)}")
    code = _exit(results, args.strict)
    if code and args.staged:
        print("  to commit anyway: git commit --no-verify  (or FACTS_SKIP=1 git commit)")
    return code


HOOK_MARK = "facts.py pre-commit shim"
SHIM = f"""#!/bin/sh
# {HOOK_MARK}: runs tools/facts/pre-commit of the checked-out tree.
# Installed by `python3 tools/facts/facts.py install-hook`; removed with --uninstall.
script="$(git rev-parse --show-toplevel)/tools/facts/pre-commit"
[ -x "$script" ] || exit 0          # a branch or worktree without the fact registry
exec "$script" "$@"
"""


def cmd_install_hook(ctx: Ctx, args) -> int:
    out = subprocess.run(["git", "rev-parse", "--git-path", "hooks"], cwd=ctx.repo, capture_output=True, text=True)
    if out.returncode:
        print(f"not a git repository: {ctx.repo}", file=sys.stderr)
        return 2
    hooks = Path(out.stdout.strip())
    hooks = hooks if hooks.is_absolute() else ctx.repo / hooks
    hook = hooks / "pre-commit"
    ours = hook.exists() and HOOK_MARK in hook.read_text(errors="replace")
    if args.uninstall:
        if not hook.exists():
            print(f"no pre-commit hook at {hook}")
            return 0
        if not ours:
            print(f"{hook} is not the facts hook; left alone", file=sys.stderr)
            return 1
        hook.unlink()
        print(f"removed {hook}")
        return 0
    if hook.exists() and not ours and not args.force:
        print(f"{hook} exists and is not the facts hook.  Add this line to it instead:\n"
              f'  "$(git rev-parse --show-toplevel)/tools/facts/pre-commit" || exit 1\n'
              f"or replace it: install-hook --force", file=sys.stderr)
        return 1
    hooks.mkdir(parents=True, exist_ok=True)
    hook.write_text(SHIM)
    hook.chmod(0o755)
    print(f"installed {hook}: `git commit` checks the facts the staged files touch "
          f"(skip once: git commit --no-verify)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--registry", default=str(REGISTRY), help="facts.yaml (default: the repo's)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the facts")
    for name, hlp in (("check", "check facts (default: all)"), ("fix", "check and rewrite fix: auto mentions"),
                      ("verify", "re-measure recorded facts")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("ids", nargs="*", metavar="ID|GLOB")
        if name == "check":
            p.add_argument("--json", action="store_true")
            p.add_argument("--strict", action="store_true", help="warnings fail too")
            p.add_argument("-v", "--verbose", action="store_true", help="also the info lines")
        if name == "fix":
            p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("impact", help="the files a fact ties together, or the facts a file is part of")
    p.add_argument("targets", nargs="+", metavar="ID|PATH")
    p = sub.add_parser("changed", help="check the facts a change touches, list their unchanged files")
    p.add_argument("--base", default="origin/main")
    p.add_argument("--untracked", action="store_true", help="count untracked new files as changed too")
    p.add_argument("--staged", action="store_true", help="the staged files instead (the pre-commit hook)")
    p.add_argument("--quiet", action="store_true", help="only what fails, and how to go on (the hook's report)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true")
    p = sub.add_parser("install-hook", help="make git commit run tools/facts/pre-commit")
    p.add_argument("--force", action="store_true", help="replace another pre-commit hook")
    p.add_argument("--uninstall", action="store_true")
    args = ap.parse_args(argv)
    reg_path = Path(args.registry).resolve()
    ctx = Ctx(reg_path.parent, load(reg_path))
    return {"list": cmd_list, "check": cmd_check, "fix": cmd_fix, "verify": cmd_verify,
            "impact": cmd_impact, "changed": cmd_changed, "install-hook": cmd_install_hook}[args.cmd](ctx, args)


if __name__ == "__main__":
    sys.exit(main())

"""Tests of tools/facts: the engine on a throw-away repo (value facts with regex
and marker mentions and their fixes, mention formats, mapping values,
interfaces with excluded keys and subsets, recorded facts with tolerances,
staleness and verify, impact, the register_map, cli_flags, names_in_docs,
json_excerpt and script_flags plugins, registry validation), and the real
facts.yaml: every locator still matches.

    python3 -m unittest tools/facts/test_facts.py -v
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import facts  # noqa: E402


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                            GIT_COMMITTER_EMAIL="t@t"))


class Repo:
    """A temporary git repo with files and a facts.yaml."""

    def __init__(self, files: dict, registry: str):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        for rel, text in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(text))
        (self.root / "facts.yaml").write_text(textwrap.dedent(registry))
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "init")

    def ctx(self):
        return facts.Ctx(self.root, facts.load(self.root / "facts.yaml"))

    def read(self, rel):
        return (self.root / rel).read_text()

    def write(self, rel, text):
        (self.root / rel).write_text(text)

    def check(self, fid, **kw):
        ctx = self.ctx()
        return facts.check_fact(ctx.facts[fid], ctx, **kw)

    def close(self):
        self.td.cleanup()


VALUE_REG = """
    facts:
    - id: n.items
      kind: value
      source: {file: src/items.py, regex: '^ITEMS = (\\[.*\\])', transform: 'len(ast(v))'}
      fix: auto
      mentions:
        - {file: README.md, regex: 'There are (\\d+) items'}
    - id: sizes
      kind: value
      source:
        parts:
          small: {file: src/sizes.json, json: small}
          large: {file: src/sizes.json, json: large}
      fix: auto
      mentions:
        - {file: doc/a.md, regex: 'small (?P<small>\\d+), large (?P<large>\\d+)'}
"""
VALUE_FILES = {
    "src/items.py": "ITEMS = ['a', 'b', 'c']\n",
    "src/sizes.json": '{"small": 2, "large": 9}\n',
    "README.md": "There are 2 items.\nTable: <!-- fact:n.items -->2<!-- /fact --> items\n",
    "doc/a.md": "Sizes: small 2, large 8.\nAgain <!-- fact:sizes.large -->9<!-- /fact -->.\n",
}


class TestValueFacts(unittest.TestCase):

    def setUp(self):
        self.r = Repo(VALUE_FILES, VALUE_REG)

    def tearDown(self):
        self.r.close()

    def test_stale_regex_and_marker_mentions(self):
        res = self.r.check("n.items")
        self.assertEqual((res.status, res.value), ("fail", 3))
        wheres = sorted(f.where for f in res.findings if f.level == "error")
        self.assertEqual(wheres, ["README.md:1", "README.md:2"])

    def test_fix_rewrites_and_rechecks(self):
        res = self.r.check("n.items", fix=True, dry=True)
        self.assertEqual(self.r.read("README.md"), VALUE_FILES["README.md"])     # dry run: untouched
        self.assertEqual(sum("would fix" in f.msg for f in res.findings), 2)
        res = self.r.check("n.items", fix=True)
        self.assertEqual(res.status, "ok")
        self.assertEqual(self.r.read("README.md"),
                         "There are 3 items.\nTable: <!-- fact:n.items -->3<!-- /fact --> items\n")
        self.assertEqual(self.r.check("n.items").status, "ok")

    def test_mapping_value_named_groups_and_key_markers(self):
        res = self.r.check("sizes")
        self.assertEqual(res.value, {"small": 2, "large": 9})
        errs = [f for f in res.findings if f.level == "error"]
        self.assertEqual(len(errs), 1)
        self.assertIn("large '8'", errs[0].msg)
        self.r.check("sizes", fix=True)
        self.assertIn("small 2, large 9", self.r.read("doc/a.md"))

    def test_locator_lost(self):
        self.r.write("README.md", "The item count moved.\n")
        res = self.r.check("n.items")
        self.assertEqual(res.status, "fail")
        self.assertTrue(any("locator lost" in f.msg for f in res.findings))

    def test_mention_format(self):
        r = Repo({"n.txt": "3\n", "doc.md": "We ship two models.\n"},
                 """
                 facts:
                 - id: n
                   kind: value
                   source: {file: n.txt, regex: '(\\d+)', type: int}
                   fix: auto
                   mentions:
                     - {file: doc.md, regex: 'ship (\\w+) models', format: "['zero', 'one', 'two', 'three'][v]"}
                 """)
        try:
            res = r.check("n")
            self.assertEqual(res.status, "fail")
            self.assertIn("= 'three' here", res.findings[0].msg)
            r.check("n", fix=True)
            self.assertEqual(r.read("doc.md"), "We ship three models.\n")
        finally:
            r.close()

    def test_impact(self):
        ctx = self.r.ctx()
        files = facts.fact_files(ctx.facts["n.items"], ctx)
        self.assertEqual(files["source"], ["src/items.py"])
        self.assertEqual(sorted(files["mentions"]), ["README.md"])
        self.assertIn("doc/a.md", facts.fact_files(ctx.facts["sizes"], ctx)["mentions"])


class TestInterface(unittest.TestCase):

    def test_producer_consumers(self):
        r = Repo({"op.h": "enum Op {\n  OP_A = 0,\n  OP_B = 1,\n  ACT_X = 0,\n};\n",
                  "ops.py": "OP_A = 0\nOP_B = 2\n",
                  "names.py": 'NAMES = ["A", "B"]\n'},
                 """
                 facts:
                 - id: codes
                   kind: interface
                   producer: {file: op.h, regex: '^\\s*((?:OP|ACT)_\\w+)\\s*=\\s*(\\d+)', as: dict}
                   consumers:
                     - {name: py, file: ops.py, regex: '^(OP_\\w+) = (\\d+)', as: dict, keys: 'OP_*'}
                     - {name: names, file: names.py, regex: '^NAMES = (\\[.*\\])',
                        transform: "{'OP_' + n: i for i, n in enumerate(ast(v))}", keys: 'OP_*'}
                 """)
        try:
            res = r.check("codes")
            self.assertEqual(res.status, "fail")
            errs = [f.msg for f in res.findings if f.level == "error"]
            self.assertEqual(len(errs), 1)
            self.assertIn("OP_B: 2 (producer 1)", errs[0])
        finally:
            r.close()


class TestInterfaceOptions(unittest.TestCase):

    def test_exclude_and_subset(self):
        r = Repo({"src.py": "emit('a')\nemit('b')\nemit('c')\n",
                  "user.py": "if k == 'a': pass\nif k == 'b': pass\n",
                  "doc.md": "Kinds: `b`.\n"},
                 """
                 facts:
                 - id: kinds
                   kind: interface
                   producer: {file: src.py, regex: 'emit\\(''(\\w+)''\\)', all: true, transform: 'dict.fromkeys(v, 1)'}
                   consumers:
                     - {name: user, file: user.py, regex: 'k == ''(\\w+)''', all: true, transform: 'dict.fromkeys(v, 1)',
                        exclude: [c]}
                     - {name: doc, file: doc.md, regex: '`(\\w+)`', all: true, transform: 'dict.fromkeys(v, 1)', subset: true}
                 """)
        try:
            self.assertEqual(r.check("kinds").status, "ok")
            r.write("doc.md", "Kinds: `b`, `z`.\n")                    # a subset may not name an unknown kind
            self.assertIn("z: 1 (producer —)", r.check("kinds").findings[0].msg)
            r.write("doc.md", "Kinds: `b`.\n")
            r.write("src.py", "emit('a')\nemit('b')\nemit('c')\nemit('d')\n")   # a new kind the user misses
            self.assertIn("d: — (producer 1)", r.check("kinds").findings[0].msg)
        finally:
            r.close()


class TestRecorded(unittest.TestCase):

    def test_relative_tolerance_and_provenance_facts(self):
        r = Repo({"id.txt": "abc\n", "doc.md": "About ~10 tokens/s.\n"},
                 """
                 facts:
                 - id: build.id
                   kind: value
                   source: {file: id.txt, regex: '(\\w+)'}
                 - id: speed
                   kind: recorded
                   value: 10.07
                   tolerance: {rel: 0.03}
                   provenance: {date: 2026-10-02, facts: {build.id: abc}}
                   mentions:
                     - {file: doc.md, regex: '~([\\d.]+) tokens/s'}
                 """)
        try:
            self.assertEqual(r.check("speed").status, "ok")              # 10 is within 3 % of 10.07
            r.write("doc.md", "About ~9 tokens/s.\n")
            self.assertEqual(r.check("speed").status, "fail")
            r.write("doc.md", "About ~10 tokens/s.\n")
            r.write("id.txt", "xyz\n")                                  # measured on another build
            res = r.check("speed")
            self.assertEqual(res.status, "warn")
            self.assertIn("measured under build.id = abc, now xyz", res.findings[0].msg)
        finally:
            r.close()

    def test_staleness_and_verify(self):
        r = Repo({"gen.py": "SIZE = 1\n", "build/out.txt": "pool 41.6 MiB\n", "doc.md": "The pool is 42 MiB.\n"},
                 """
                 facts:
                 - id: pool
                   kind: recorded
                   value: 42
                   tolerance: 1
                   provenance: {commit: HEAD, date: 2026-10-02, how: test}
                   depends_on: [gen.py]
                   verify:
                     pool: {file: build/out.txt, regex: 'pool ([\\d.]+) MiB', type: float}
                   mentions:
                     - {file: doc.md, regex: 'pool is (\\d+) MiB'}
                 """)
        try:
            ctx = r.ctx()
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=r.root, capture_output=True,
                                  text=True).stdout.strip()
            ctx.facts["pool"]["provenance"]["commit"] = head
            self.assertEqual(facts.check_fact(ctx.facts["pool"], ctx).status, "ok")
            self.assertEqual(facts.verify_fact(ctx.facts["pool"], ctx).status, "ok")    # 41.6 within 1
            r.write("gen.py", "SIZE = 2\n")
            git(r.root, "commit", "-qam", "bigger")
            res = facts.check_fact(ctx.facts["pool"], ctx)
            self.assertEqual(res.status, "warn")
            self.assertIn("1 commit(s) changed its inputs", res.findings[0].msg)
            r.write("build/out.txt", "pool 50.0 MiB\n")
            self.assertEqual(facts.verify_fact(ctx.facts["pool"], ctx).status, "fail")
            (r.root / "build/out.txt").unlink()                         # not measurable here: skipped
            self.assertEqual(facts.verify_fact(ctx.facts["pool"], ctx).status, "skipped")
        finally:
            r.close()


class TestPlugins(unittest.TestCase):

    def test_register_map(self):
        r = Repo({"k.cpp": "#pragma HLS INTERFACE s_axilite port=a bundle=ctrl\n"
                           "#pragma HLS INTERFACE s_axilite port=n bundle=ctrl\n"
                           "#pragma HLS INTERFACE s_axilite port=mode bundle=ctrl\n"
                           "#pragma HLS INTERFACE s_axilite port=return bundle=ctrl\n",
                  "gen.c": "XK_Set_a(k, 1); XK_Set_n(k, 2); XK_Set_mod(k, 3);\n",
                  "ui.js": "const DECODE = {\n  K: {mode: [\"x\"], flag: [\"no\", \"yes\"]}};\n"},
                 """
                 facts:
                 - id: regs
                   kind: interface
                   check: register_map
                   args: {kernel: K, prefix: XK, hls: k.cpp, fields: [n, mode], not_keyed: {a: address, gone: x},
                          decode: ui.js, writers_all: [gen.c]}
                 """)
        try:
            sys.path.insert(0, str(HERE))
            msgs = [(f.level, f.msg) for f in r.check("regs").findings if f.level != "info"]
            text = "\n".join(m for _, m in msgs)
            self.assertIn("sets 'mod', which is no register", text)
            self.assertIn("never sets mode", text)
            self.assertIn("DECODE[K] decodes 'flag'", text)
            self.assertIn(("warn", "not_keyed lists 'gone', which is no register any more (stale entry)"), msgs)
        finally:
            r.close()

    def test_cli_flags(self):
        script = ("import argparse\np = argparse.ArgumentParser()\np.add_argument('--alpha')\n"
                  "p.add_argument('--beta', action='store_true')\np.parse_args()\n")
        r = Repo({"tool.py": script,
                  "GUIDE.md": "# Guide\n## Options\n| `--alpha X` | a |\n| `--gamma` | gone |\n## Next\n--beta\n"},
                 """
                 facts:
                 - id: cli
                   kind: interface
                   check: cli_flags
                   args: {script: tool.py, docs: [{file: GUIDE.md, start: '^## Options$', end: '^## '}]}
                 """)
        try:
            msgs = sorted(f.msg for f in r.check("cli").findings if f.level == "error")
            self.assertEqual(len(msgs), 2)
            self.assertIn("--beta is accepted by tool.py but not documented here", msgs[0])
            self.assertIn("documents --gamma, which tool.py does not accept", msgs[1])
        finally:
            r.close()


class TestMorePlugins(unittest.TestCase):

    def test_names_in_docs(self):
        r = Repo({"ops.json": '["Add", "MatMul", "Gather"]\n',
                  "GUIDE.md": "# G\n## Ops\n`Add`, `MatMul`, `ReduceMean` (rejected), `Bogus`\n## Next\n`Gather`\n"},
                 """
                 facts:
                 - id: ops
                   kind: interface
                   check: names_in_docs
                   args:
                     names: {file: ops.json, json: ''}
                     docs:
                       - {file: GUIDE.md, start: '^## Ops', end: '^## ', reverse: true, extra_ok: [ReduceMean]}
                       - {file: GUIDE.md, start: '^## Ops', end: '^## ', skip: [Gather]}
                 """)
        try:
            msgs = [f.msg for f in r.check("ops").findings if f.level == "error"]
            self.assertEqual(msgs, ["not shown here: Gather",
                                    "shown here but not in the code: Bogus (or add them to extra_ok)"])
        finally:
            r.close()

    def test_reverse_only_and_optional(self):
        r = Repo({"targets.txt": "all\nbuild_hw\n", "README.md": "## Build\nmake build_hw, then make flash\n## End\n"},
                 """
                 facts:
                 - id: targets
                   kind: interface
                   check: names_in_docs
                   args:
                     names: {file: targets.txt, transform: 'v.split()'}
                     missing: no make target
                     docs: [{file: README.md, start: '^## Build', end: '^## End', pattern: 'make (\\w+)', forward: false, reverse: true}]
                 - id: hw
                   kind: value
                   optional: true
                   source: {file: hw/project.xpr, regex: 'v(\\d+)'}
                   mentions: [{file: README.md, regex: '## (Build)'}]
                 """)
        try:
            msgs = [f.msg for f in r.check("targets").findings if f.level == "error"]
            self.assertEqual(msgs, ["shown here but no make target: flash (or add them to extra_ok)"])   # `all` not required
            res = r.check("hw")                                         # no hw/ here: skipped, not failed
            self.assertEqual(res.status, "skipped")
            self.assertIn("source not available here", res.findings[0].msg)
        finally:
            r.close()

    def test_json_excerpt(self):
        r = Repo({"p.json": '{"clock": 150, "k": {"a": 1, "b": 2}, "_note": "x"}\n',
                  "DOC.md": "# D\n## Example\n```jsonc\n{\n  \"clock\": 150,  // MHz\n"
                            "  /* tiles */ \"k\": {\"a\": 3}\n}\n```\n"},
                 """
                 facts:
                 - id: ex
                   kind: interface
                   check: json_excerpt
                   args: {json: p.json, doc: DOC.md, start: '^## Example', complete: true}
                 """)
        try:
            msgs = sorted(f.msg for f in r.check("ex").findings if f.level == "error")
            self.assertEqual(msgs, ["k.a: shows 3, p.json has 1", "k.b: in p.json (2), not shown"])
        finally:
            r.close()

    def test_script_flags(self):
        callee = ("import argparse\np = argparse.ArgumentParser()\np.add_argument('--port')\n"
                  "p.add_argument('--fast', action='store_true')\np.parse_args()\n")
        caller = ("import subprocess\ndef launch():\n"
                  "    subprocess.run(['python3', 'server.py', '--port', '1', '--speed', '2'])\n")
        r = Repo({"server.py": callee, "deploy.py": caller},
                 """
                 facts:
                 - id: calls
                   kind: interface
                   check: script_flags
                   args: {pairs: [{caller: deploy.py, function: launch, callee: server.py}]}
                 """)
        try:
            msgs = [(f.msg, f.where) for f in r.check("calls").findings if f.level == "error"]
            self.assertEqual(msgs, [("passes --speed, which server.py does not accept", "deploy.py:3")])
        finally:
            r.close()


class TestPreCommitHook(unittest.TestCase):
    """install-hook + a real ``git commit``: a commit that leaves a fact stale is
    refused, the fixed one goes through, an unrelated one is not slowed down."""

    def test_commit_flow(self):
        import shutil
        r = Repo({"src/items.py": "ITEMS = ['a', 'b']\n", "README.md": "There are 2 items.\n",
                  "other.txt": "x\n"}, VALUE_REG.replace("""    - id: sizes""", """    - id: unused""").split(
                      "    - id: unused")[0])
        try:
            tools = r.root / "tools" / "facts"
            tools.mkdir(parents=True)
            for name in ("facts.py", "plugins.py", "pre-commit"):
                shutil.copy2(HERE / name, tools / name)
            git(r.root, "add", "-A")
            git(r.root, "commit", "-qm", "tools")
            self.assertEqual(facts.main(["--registry", str(r.root / "facts.yaml"), "install-hook"]), 0)
            hook = Path(subprocess.run(["git", "rev-parse", "--git-path", "hooks"], cwd=r.root, capture_output=True,
                                       text=True).stdout.strip())
            self.assertIn(facts.HOOK_MARK, ((r.root / hook) / "pre-commit").read_text())
            env = dict(os.environ, FACTS_PYTHON=sys.executable, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                       GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
            env.pop("FACTS_SKIP", None)

            def commit(*paths):
                return subprocess.run(["git", "commit", "-qm", "c", *paths], cwd=r.root, capture_output=True,
                                      text=True, env=env)
            r.write("src/items.py", "ITEMS = ['a', 'b', 'c']\n")       # README now stale
            c = commit("src/items.py")
            self.assertEqual(c.returncode, 1, c.stdout + c.stderr)
            out = c.stdout + c.stderr                                  # git sends a hook's stdout to stderr
            self.assertIn("says '2', the fact is 3", out)
            self.assertIn("facts.py fix n.items", out)
            self.assertIn("git commit --no-verify", out)
            r.write("other.txt", "y\n")                                # unrelated: goes through
            self.assertEqual(commit("other.txt").returncode, 0)
            r.write("README.md", "There are 3 items.\n")
            c = commit("src/items.py", "README.md")
            self.assertEqual(c.returncode, 0, c.stdout + c.stderr)
            r.write("src/items.py", "ITEMS = ['a']\n")                  # FACTS_SKIP bypasses it
            self.assertEqual(subprocess.run(["git", "commit", "-qam", "skip"], cwd=r.root, capture_output=True,
                                            env={**env, "FACTS_SKIP": "1"}).returncode, 0)
            self.assertEqual(facts.main(["--registry", str(r.root / "facts.yaml"), "install-hook", "--uninstall"]), 0)
            self.assertFalse(((r.root / hook) / "pre-commit").exists())
        finally:
            r.close()

    def test_foreign_hook_is_kept(self):
        r = Repo({"a.txt": "x\n"}, "facts: []\n")
        try:
            hook = r.root / ".git" / "hooks" / "pre-commit"
            hook.parent.mkdir(parents=True, exist_ok=True)
            hook.write_text("#!/bin/sh\nexit 0\n")
            self.assertEqual(facts.main(["--registry", str(r.root / "facts.yaml"), "install-hook"]), 1)
            self.assertEqual(hook.read_text(), "#!/bin/sh\nexit 0\n")
        finally:
            r.close()


class TestRegistry(unittest.TestCase):

    def test_validation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "facts.yaml"
            for bad in ("facts:\n- {id: a, kind: value}\n",                          # no source
                        "facts:\n- {id: a, kind: value, source: 1}\n- {id: a, kind: value, source: 1}\n",
                        "facts:\n- {id: a, kind: other}\n",
                        "facts:\n- {id: a, kind: interface}\n"):
                p.write_text(bad)
                with self.assertRaises(SystemExit):
                    facts.load(p)

    def test_real_registry_locators(self):
        """Every mention / producer / consumer locator of facts.yaml still finds
        its text (a stale value is fine here; a lost locator is not)."""
        ctx = facts.Ctx()
        broken = []
        for f in ctx.reg["facts"]:
            if f["kind"] == "value" and "cmd" in f.get("source", {}):
                ctx._values[f["id"]] = 0                    # no pytest run here: only the locators matter
            res = facts.check_fact(f, ctx)
            broken += [f"{f['id']}: {x.where}: {x.msg}" for x in res.findings
                       if x.level == "error" and ("locator" in x.msg or "no such file" in x.msg
                                                  or "no match" in x.msg or "not found" in x.msg)]
        self.assertEqual(broken, [])


if __name__ == "__main__":
    unittest.main()

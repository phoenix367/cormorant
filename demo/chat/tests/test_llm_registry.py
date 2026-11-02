# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The served-LLM registry (doc/plans/CHAT_PLAN.md §29): kv260_chat_server.py's
--llm-model NAME=DESCRIPTOR.json (one construction path for every kind, the
legacy flag groups through it too) and deploy.py's model list — llm_models
entries derived from a checkpoint and its generated project, the legacy
blocks, the descriptors and the server's command line."""

import json
import os
import shutil
import tempfile
import unittest

from _util import CHAT, fake_llm_lib, sampler_lib, server_mod

SMOLLM2 = os.path.join(CHAT, "assets", "smollm2-135m-instruct")
STORIES = os.path.join(CHAT, "assets", "stories15m")
HAVE = all(os.path.exists(os.path.join(d, "tokenizer.json")) for d in (SMOLLM2, STORIES))

try:
    import deploy
except ImportError:                                  # paramiko, numpy: the scheduler venv
    deploy = None


@unittest.skipUnless(HAVE, "the SmolLM2-135M and stories15m tokenizers are not present")
class TestDescriptors(unittest.TestCase):

    def setUp(self):
        self.lib = fake_llm_lib("smollm2-135m-instruct")
        if self.lib is None:
            self.skipTest("no C compiler")
        self.dir = tempfile.mkdtemp(prefix="llmreg_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def desc(self, name, **d):
        path = os.path.join(self.dir, name + ".json")
        with open(path, "w") as f:
            json.dump(d, f)
        return f"{name}={path}"

    def args(self, *argv):
        return server_mod.parse_args(["--llm-sampler-lib", sampler_lib() or "/nonexistent",
                                      "--llm-temperature", "0.3", *argv])

    def test_kinds_order_and_settings(self):
        from smollm2_backend import Smollm2Backend
        from stories_backend import StoriesBackend
        chat = self.desc("mychat", kind="chat", lib=self.lib, tokenizer=os.path.join(SMOLLM2, "tokenizer.json"),
                         cma_mb=123, sampling={"top_k": 7})
        text = self.desc("mytext", kind="text", lib=fake_llm_lib("stories15m"),
                         tokenizer=os.path.join(STORIES, "tokenizer.json"), cma_mb=45, reserve=64,
                         sampling={"temperature": 0.5}, model_id="tiny-stories")
        bs = server_mod.build_backends(self.args("--llm-model", chat, "--llm-model", text,
                                                 "--backend", "mytext", "--backend", "mychat"))
        self.assertEqual(list(bs), ["tiny-stories", "smollm2-135m-instruct"])     # --backend order
        c, t = bs["smollm2-135m-instruct"], bs["tiny-stories"]
        self.assertIs(type(c), Smollm2Backend)
        self.assertIs(type(t), StoriesBackend)
        self.assertEqual((c.cma_mb, c.defaults.temperature, c.defaults.top_k), (123, 0.3, 7))
        self.assertEqual((t.cma_mb, t.reserve, t.defaults.temperature, t.defaults.top_p), (45, 64, 0.5, 0.9))
        for b in bs.values():
            b.load_host()
        self.assertEqual(c.bos, 1)
        self.assertTrue(t.completions)

    def test_legacy_flags_take_the_same_path(self):
        """--backend smollm2 with --llm-lib: the descriptor the flags make."""
        bs = server_mod.build_backends(self.args("--backend", "smollm2", "--llm-lib", self.lib,
                                                 "--llm-tokenizer", os.path.join(SMOLLM2, "tokenizer.json"),
                                                 "--llm-cma-mb", "321"))
        (b,) = bs.values()
        self.assertEqual((b.model_id, b.cma_mb), ("smollm2-135m-instruct", 321))

    def test_refusals(self):
        bad = self.desc("x", kind="audio", lib=self.lib, tokenizer="t")
        with self.assertRaises(SystemExit):
            server_mod.build_backends(self.args("--llm-model", bad, "--backend", "x"))
        with self.assertRaises(SystemExit):
            server_mod.build_backends(self.args("--llm-model", "no-equals-sign", "--backend", "x"))
        with self.assertRaises(SystemExit):
            server_mod.build_backends(self.args("--backend", "not-a-backend"))


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestDeployEntries(unittest.TestCase):

    def cfg(self, backends, **extra):
        c = {"remote": {"dir": "/r"}, "server": {"backends": list(backends), "host": "0.0.0.0", "port": 8000,
                                                  "max_windows": 8, "doc_stride": 128, "queue_timeout": 120,
                                                  "max_queue": 16, "resident": "auto"},
             "smollm2": {"lib": "/r/lib/libsmollm2.so", "tokenizer": "assets/smollm2-135m-instruct/tokenizer.json",
                         "cma_mb": 330, "temperature": 0.2, "loop_guard": True},
             "stories15m": {"lib": "/r/lib/libstories15m.so", "tokenizer": "assets/stories15m/tokenizer.json",
                            "cma_mb": 50, "temperature": 0.8, "top_p": 0.9, "reserve": 128}}
        c.update(extra)
        c["remote"].setdefault("weights_dir", "/w")
        return c

    def test_legacy_blocks(self):
        c = self.cfg(["bert-squad", "stories15m", "smollm2"])
        es = deploy.llm_entries(c)
        self.assertEqual([(e["backend"], e["kind"], e["sub"]) for e in es],
                         [("stories15m", "text", "stories15m"), ("smollm2", "chat", "smollm2")])
        d = deploy.llm_descriptor(c, es[0])
        self.assertEqual(d, {"kind": "text", "lib": "/r/lib/libstories15m.so", "weights": None,
                             "tokenizer": "/r/stories15m/tokenizer.json", "model_id": None, "cma_mb": 50,
                             "reserve": 128, "sampling": {"temperature": 0.8, "top_p": 0.9}})
        argv = deploy.server_argv(c)
        self.assertEqual([argv[i + 1] for i, a in enumerate(argv) if a == "--llm-model"],
                         ["stories15m=/r/models/stories15m.json", "smollm2=/r/models/smollm2.json"])
        self.assertEqual([argv[i + 1] for i, a in enumerate(argv) if a == "--backend"],
                         ["bert-squad", "stories15m", "smollm2"])
        self.assertNotIn("--llm-lib", argv)
        self.assertEqual(argv[argv.index("--llm-temperature") + 1], "0.2")

    @unittest.skipUnless(HAVE, "the stories15m checkpoint files are not present")
    def test_llm_models_entry_derived(self):
        """A model with only a name: its kind from its checkpoint, its library
        and board names from the name, cma_mb from its project's pool."""
        name = "mystory-test"
        with tempfile.TemporaryDirectory() as td:
            assets, proj = os.path.join(td, name), os.path.join(td, "proj")
            os.makedirs(assets)
            os.makedirs(proj)
            for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
                shutil.copy(os.path.join(STORIES, f), assets)
            with open(os.path.join(proj, "project.json"), "w") as f:
                json.dump({"pool_bytes": 100 * 2**20}, f)
            c = self.cfg([name], llm_models={name: {"assets": assets, "project": proj}})
            (e,) = deploy.llm_entries(c)
            self.assertEqual((e["kind"], e["sub"], e["lib"], e["cma_mb"]),
                             ("text", "mystory_test", "/r/lib/libmystory_test.so", 121))
            self.assertEqual(e["tokenizer"], os.path.join(assets, "tokenizer.json"))
            self.assertIn("--project " + os.path.relpath(proj, CHAT), e["install"])
            c = self.cfg([name], llm_models={name: {"assets": assets, "kind": "chat", "cma_mb": 99,
                                                    "model_id": "m"}})
            (e,) = deploy.llm_entries(c)
            self.assertEqual((e["kind"], e["cma_mb"], e["model_id"]), ("chat", 99, "m"))


if __name__ == "__main__":
    unittest.main()

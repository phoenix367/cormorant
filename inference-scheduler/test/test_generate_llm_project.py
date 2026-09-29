"""demo/chat/scripts/generate_llm_project.py: the model name comes from the
checkpoint directory (it names the project, the library and the board's
weights directory), and an output directory holding another model's project
or other files is not wiped without --force.  Fake checkpoints (a
config.json each) in a temp dir; nothing is generated."""

import json
import os
import sys
import tempfile
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "demo", "chat", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import generate_llm_project as g                                 # noqa: E402

M135, M360, VLM = "smollm2-135m-instruct", "smollm2-360m-instruct", "smolvlm-256m-instruct"


class _Assets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "assets")
        for m in (M135, M360, VLM):
            os.makedirs(os.path.join(self.root, m))
            with open(os.path.join(self.root, m, "config.json"), "w") as f:
                json.dump({"model_type": "llama"}, f)
        os.makedirs(os.path.join(self.root, "copy_of_360m"))
        with open(os.path.join(self.root, "copy_of_360m", "config.json"), "w") as f:
            f.write("{}")
        self._env = os.environ.get("SMOLLM_ASSETS")
        os.environ["SMOLLM_ASSETS"] = os.path.join(self.root, M135)   # study.default_assets()

    def tearDown(self):
        if self._env is None:
            os.environ.pop("SMOLLM_ASSETS", None)
        else:
            os.environ["SMOLLM_ASSETS"] = self._env
        self.tmp.cleanup()

    def a(self, name):
        return os.path.join(self.root, name)


class TestResolveModel(_Assets):
    def test_default_is_135m(self):
        self.assertEqual(g.resolve_model(None, None), (M135, self.a(M135)))
        self.assertEqual(g.resolve_model(M135, None), (M135, self.a(M135)))

    def test_name_from_assets(self):
        self.assertEqual(g.resolve_model(None, self.a(M360)), (M360, self.a(M360)))
        self.assertEqual(g.resolve_model(M360, self.a(M360)), (M360, self.a(M360)))

    def test_name_alone_finds_its_assets(self):
        self.assertEqual(g.resolve_model(M360, None), (M360, self.a(M360)))
        self.assertEqual(g.resolve_model(VLM, None), (VLM, self.a(VLM)))

    def test_mismatch_refused(self):
        # the old failure: --assets of a new model with the 135M default name
        with self.assertRaises(SystemExit) as cm:
            g.resolve_model(M135, self.a(M360))
        self.assertIn("--allow-name-mismatch", str(cm.exception))
        self.assertEqual(g.resolve_model(M360, self.a("copy_of_360m"), allow_mismatch=True),
                         (M360, self.a("copy_of_360m")))

    def test_missing_checkpoint(self):
        with self.assertRaises(SystemExit) as cm:
            g.resolve_model("qwen-0.5b-instruct", None)
        self.assertIn("config.json missing", str(cm.exception))
        with self.assertRaises(SystemExit):
            g.resolve_model(None, os.path.join(self.root, "nowhere"))


class TestOutDir(_Assets):
    def setUp(self):
        super().setUp()
        self.out = os.path.join(self.tmp.name, "out")

    def project(self, model, extra=None):
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "project.json"), "w") as f:
            json.dump({"model": model}, f)
        if extra:
            with open(os.path.join(self.out, extra), "w") as f:
                f.write("keep me")

    def test_missing_or_empty_is_fine(self):
        g.check_out_dir(self.out, M135)
        os.makedirs(self.out)
        g.check_out_dir(self.out, M135)

    def test_same_model_regenerates(self):
        self.project(M135)
        g.check_out_dir(self.out, M135)

    def test_other_model_refused(self):
        self.project(M135)
        with self.assertRaises(SystemExit) as cm:
            g.check_out_dir(self.out, M360)
        self.assertIn(M135, str(cm.exception))
        g.check_out_dir(self.out, M360, force=True)

    def test_not_a_project_refused(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "notes.txt"), "w") as f:
            f.write("x")
        with self.assertRaises(SystemExit) as cm:
            g.check_out_dir(self.out, M135)
        self.assertIn("no project.json", str(cm.exception))

    def test_cli_refuses_before_touching_anything(self):
        self.project(M135, extra="weights_marker.txt")
        before = sorted(os.listdir(self.out))
        for argv in (["--assets", self.a(M360), "--out-dir", self.out],       # other model's dir
                     ["--assets", self.a(M360), "--model-name", M135,         # name mismatch
                      "--out-dir", os.path.join(self.tmp.name, "fresh")]):
            with self.assertRaises(SystemExit):
                g.main(argv)
        self.assertEqual(sorted(os.listdir(self.out)), before)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "fresh")))


if __name__ == "__main__":
    unittest.main()

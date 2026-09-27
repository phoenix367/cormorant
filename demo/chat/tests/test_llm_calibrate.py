"""scripts/llm_calibrate.py without the network or numpy: the tracked
manifest's schema, verified downloads (file:// URLs standing in for the
Hugging Face endpoint), fetch keeping / replacing / copying files, the
input check, the formats diff, and calibrate / check against a fake
llm_study.py (install only on a reproduced or recorded hash, provenance,
check never writing)."""

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from _util import CHAT

sys.path.insert(0, os.path.join(CHAT, "scripts"))
import llm_calibrate as lc  # noqa: E402

HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Writes --out with the content of $FAKE_FORMATS; answers `-c` (the
# environment probe) with a fixed JSON.
FAKE_PY = """#!/bin/sh
if [ "$1" = "-c" ]; then echo '{"python": "fake"}'; exit 0; fi
shift
while [ $# -gt 0 ]; do
  if [ "$1" = "--out" ]; then cp "$FAKE_FORMATS" "$2"; fi
  shift
done
"""


def sha(b):
    return hashlib.sha256(b).hexdigest()


def formats(sink_v=3, exp_d=(7, 7, 7)):
    return {"policy": "pow2+sink+p12", "spec": {"fmt": "pow2"}, "margin": 1.0,
            "exponents": {"x@0": 9, "d@0": list(exp_d)}, "sink_token": 1,
            "sink_k_raw": [[[1, 2], [3, 4]]], "sink_v_raw": [[[sink_v, 5], [6, 7]]]}


class Tree:
    """A temporary assets root, a file:// 'Hugging Face' and a manifest."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="test_llm_calibrate_"))
        self.root = self.dir / "assets"
        self.hub = self.dir / "hub"
        self.files = {n: f"{n} contents\n".encode() for n in ("config.json", "model.safetensors")}
        self.texts = {"heldout.txt": b"held-out text\n", "calib.txt": b"calibration text\n"}
        rev = "a" * 40
        for n, b in self.files.items():
            p = self.hub / "org" / "tiny" / "resolve" / rev / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
        self.manifest_path = self.dir / "llm_models.json"
        self.m = {"models": {"tiny": {"repo": "org/tiny", "revision": rev,
                                      "files": {n: sha(b) for n, b in self.files.items()},
                                      "formats": {"policy": "pow2+sink+p12", "sha256": None}}},
                  "texts": {n: {"sha256": sha(b)} for n, b in self.texts.items()}}
        lc.save_manifest(self.m, self.manifest_path)
        self.url = self.hub.as_uri()

    def model_dir(self, model="tiny"):
        return self.root / model

    def put_texts(self, model):
        d = self.root / model
        d.mkdir(parents=True, exist_ok=True)
        for n, b in self.texts.items():
            (d / n).write_bytes(b)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class Quiet(unittest.TestCase):
    def setUp(self):
        self.t = Tree()
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def tearDown(self):
        sys.stdout.close()
        sys.stdout = self._stdout
        self.t.close()


class TestManifest(unittest.TestCase):
    """The tracked llm_models.json pins complete, well-formed inputs."""

    def test_schema(self):
        m = lc.load_manifest()
        self.assertIn(lc.DEFAULT_MODEL, m["models"])
        for name, spec in m["models"].items():
            with self.subTest(model=name):
                self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
                self.assertEqual(sorted(spec["files"]), sorted(lc.CHECKPOINT_FILES))
                for h in spec["files"].values():
                    self.assertRegex(h, HEX64)
                self.assertEqual(spec["formats"]["policy"], "pow2+sink+p12")
                self.assertRegex(spec["formats"]["sha256"], HEX64)
                if "study" in spec:
                    self.assertEqual(spec["study"]["policy"], lc.SHIPPED)
        self.assertEqual(len(m["texts"]), 2)
        for name, t in m["texts"].items():
            self.assertRegex(t["sha256"], HEX64)
            self.assertGreater(t["min_chars"], 0)

    def test_texts_match_llm_study_fetch(self):
        """The text names are those llm_study.py fetch writes."""
        src = lc.STUDY.read_text()
        for name in lc.load_manifest()["texts"]:
            self.assertIn(f'"{name}"', src)

    def test_study_dir_matches_llm_study(self):
        src = lc.STUDY.read_text()
        self.assertIn(f'DEFAULT_MODEL = "{lc.DEFAULT_MODEL}"', src)
        root = Path("/a")
        self.assertEqual(lc.study_dir(root, lc.DEFAULT_MODEL), root / "study")
        self.assertEqual(lc.study_dir(root, "m2"), root / "study" / "m2")
        for model in (lc.DEFAULT_MODEL, "m2"):
            self.assertEqual(lc.provenance_path(root, model, "p"),
                             root / "study" / model / "formats_p.provenance.json")

    def test_provenance_is_tracked(self):
        """assets/.gitignore ignores everything but the provenance files."""
        import subprocess
        assets = os.path.join(CHAT, "assets")
        for path, ignored in ((f"study/{lc.DEFAULT_MODEL}/formats_p.provenance.json", False),
                              ("study/m2/formats_p.provenance.json", False),
                              ("study/formats_p.json", True), ("study/m2/formats_p.json", True),
                              (f"{lc.DEFAULT_MODEL}/model.safetensors", True)):
            r = subprocess.run(["git", "-C", assets, "check-ignore", "-q", path])
            if r.returncode not in (0, 1):
                self.skipTest("not a git checkout")
            self.assertEqual(r.returncode == 0, ignored, path)


class TestDownload(Quiet):
    def test_verified(self):
        dest = self.t.dir / "out" / "config.json"
        url = f"{self.t.url}/org/tiny/resolve/{'a' * 40}/config.json"
        got = lc.download(url, dest, sha(self.t.files["config.json"]))
        self.assertEqual(dest.read_bytes(), self.t.files["config.json"])
        self.assertEqual(got, sha(self.t.files["config.json"]))
        self.assertFalse(dest.with_name("config.json.part").exists())

    def test_mismatch_leaves_nothing(self):
        dest = self.t.dir / "out" / "config.json"
        url = f"{self.t.url}/org/tiny/resolve/{'a' * 40}/config.json"
        with self.assertRaisesRegex(lc.CalibrationError, "SHA-256"):
            lc.download(url, dest, "0" * 64)
        self.assertEqual(list(dest.parent.iterdir()), [])


class TestFetch(Quiet):
    def fetch(self):
        lc.cmd_fetch(self.t.m, "tiny", self.t.root, None, base_url=self.t.url)

    def test_fresh_with_texts_from_another_model(self):
        self.t.put_texts("other")
        self.fetch()
        d = self.t.model_dir()
        for n, b in {**self.t.files, **self.t.texts}.items():
            self.assertEqual((d / n).read_bytes(), b)
        self.assertEqual(sorted(lc.verify_inputs(self.t.m, "tiny", self.t.root)),
                         sorted({**self.t.files, **self.t.texts}))

    def test_keeps_matching_replaces_corrupt(self):
        self.t.put_texts("tiny")
        d = self.t.model_dir()
        (d / "config.json").write_bytes(self.t.files["config.json"])
        os.utime(d / "config.json", (1, 1))
        (d / "model.safetensors").write_bytes(b"truncated")
        self.fetch()
        self.assertEqual(os.stat(d / "config.json").st_mtime, 1)
        self.assertEqual((d / "model.safetensors").read_bytes(), self.t.files["model.safetensors"])

    def test_changed_upstream_fails(self):
        self.t.put_texts("tiny")
        p = self.t.hub / "org" / "tiny" / "resolve" / ("a" * 40) / "model.safetensors"
        p.write_bytes(b"another checkpoint")
        with self.assertRaisesRegex(lc.CalibrationError, "model.safetensors"):
            self.fetch()
        self.assertFalse((self.t.model_dir() / "model.safetensors").exists())

    def test_texts_need_the_study_python(self):
        with self.assertRaisesRegex(lc.CalibrationError, "study python"):
            lc.cmd_fetch(self.t.m, "tiny", self.t.root, str(self.t.dir / "no-python"),
                         base_url=self.t.url)

    def test_verify_inputs_names_problems(self):
        self.t.put_texts("tiny")
        (self.t.model_dir() / "config.json").write_bytes(b"edited")
        with self.assertRaisesRegex(lc.CalibrationError, r"config.json differs; model.safetensors missing"):
            lc.verify_inputs(self.t.m, "tiny", self.t.root)

    def test_unknown_model(self):
        with self.assertRaisesRegex(lc.CalibrationError, "not in the manifest"):
            lc.model_spec(self.t.m, "nope")


class TestDiffFormats(unittest.TestCase):
    def test_diff(self):
        d = Path(tempfile.mkdtemp())
        try:
            a, b = d / "a.json", d / "b.json"
            a.write_text(json.dumps(formats()))
            b.write_text(json.dumps(formats(sink_v=4, exp_d=(7, 6, 7))))
            self.assertEqual(lc.diff_formats(a, a), [])
            self.assertEqual(lc.diff_formats(a, b), ["exponents[d@0]: 1 of 3 channels differ",
                                                     "sink_v_raw: 1 of 4 values differ (max |diff| 1)"])
        finally:
            shutil.rmtree(d)


class TestCalibrate(Quiet):
    """cmd_calibrate / main() against a fake study interpreter."""

    def setUp(self):
        super().setUp()
        self.t.put_texts("tiny")
        lc.cmd_fetch(self.t.m, "tiny", self.t.root, None, base_url=self.t.url)
        self.py = self.t.dir / "fake-python"
        self.py.write_text(FAKE_PY)
        self.py.chmod(0o755)
        self.fmt = self.t.dir / "formats.json"
        self.set_result(formats())
        self._env = os.environ.get("FAKE_FORMATS")
        os.environ["FAKE_FORMATS"] = str(self.fmt)
        self.installed = lc.study_dir(self.t.root, "tiny") / "formats_pow2+sink+p12.json"

    def tearDown(self):
        if self._env is None:
            os.environ.pop("FAKE_FORMATS", None)
        else:
            os.environ["FAKE_FORMATS"] = self._env
        super().tearDown()

    def set_result(self, f):
        self.fmt.write_text(json.dumps(f))
        return sha(self.fmt.read_bytes())

    def run_main(self, *argv):
        m = self.t.manifest_path
        return lc.main([*argv, "tiny", "--assets-root", str(self.t.root), "--manifest", str(m),
                        "--study-python", str(self.py)])

    def manifest_sha(self):
        return lc.load_manifest(self.t.manifest_path)["models"]["tiny"]["formats"]["sha256"]

    def test_record_then_reproduce(self):
        h = sha(self.fmt.read_bytes())
        self.assertEqual(self.run_main("calibrate", "--record"), 0)
        self.assertEqual(self.manifest_sha(), h)
        self.assertEqual(self.installed.read_bytes(), self.fmt.read_bytes())
        prov_path = lc.provenance_path(self.t.root, "tiny", "pow2+sink+p12")
        prov = json.loads(prov_path.read_text())
        self.assertEqual(prov["status"], "recorded")
        self.assertEqual(prov["formats"], "study/tiny/formats_pow2+sink+p12.json")
        self.assertEqual(prov["formats_sha256"], h)
        self.assertEqual(prov["environment"]["python"], "fake")
        self.assertEqual(sorted(prov["inputs"]), sorted({**self.t.files, **self.t.texts}))
        self.assertEqual(self.run_main("check"), 0)
        self.assertEqual(self.run_main("calibrate"), 0)
        prov = json.loads(prov_path.read_text())
        self.assertEqual(prov["status"], "reproduced")

    def test_differs_is_not_installed(self):
        self.assertEqual(self.run_main("calibrate", "--record"), 0)
        before = self.installed.read_bytes()
        self.set_result(formats(sink_v=4))
        self.assertEqual(self.run_main("check"), 1)
        self.assertEqual(self.run_main("calibrate"), 1)
        self.assertEqual(self.installed.read_bytes(), before)
        self.assertEqual(self.installed.with_name("formats_pow2+sink+p12.new.json").read_bytes(),
                         self.fmt.read_bytes())
        self.assertEqual(self.run_main("calibrate", "--force"), 1)
        self.assertEqual(self.installed.read_bytes(), self.fmt.read_bytes())
        self.assertNotEqual(self.manifest_sha(), sha(self.fmt.read_bytes()))

    def test_check_flags_a_stale_installed_file(self):
        h = self.set_result(formats())
        self.t.m["models"]["tiny"]["formats"]["sha256"] = h
        lc.save_manifest(self.t.m, self.t.manifest_path)
        self.installed.parent.mkdir(parents=True)
        self.installed.write_text(json.dumps(formats(exp_d=(6, 6, 6))))
        self.assertEqual(self.run_main("check"), 1)
        self.assertEqual(json.loads(self.installed.read_text())["exponents"]["d@0"], [6, 6, 6])

    def test_missing_inputs(self):
        (self.t.model_dir() / "config.json").unlink()
        self.assertEqual(self.run_main("calibrate"), 2)
        self.assertFalse(self.installed.exists())


if __name__ == "__main__":
    unittest.main()

"""deploy.py's vocab step: BERT's vocab.txt is downloaded (through
demo/bert_squad/scripts/fetch_assets.py, stubbed here) to the BERT config's
inputs.vocab before any board work; an existing file is left alone; a failed
download stops the deploy.  Needs the scheduler venv (deploy.py imports
numpy and paramiko); skipped in a stdlib-only run."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _util  # noqa: F401  (demo/chat and demo/bert_squad/scripts on sys.path)

try:                                  # src.remote exits (SystemExit) when paramiko is missing
    with contextlib.redirect_stderr(io.StringIO()):
        import deploy
        import fetch_assets
except (ImportError, SystemExit):                              # pragma: no cover
    deploy = fetch_assets = None


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestDeployVocab(unittest.TestCase):

    def test_default_path_is_the_bert_demo_asset(self):
        self.assertEqual(deploy.vocab_path({"_bert": {}}),
                         deploy.bert_common.DEMO_DIR / "assets" / "vocab.txt")

    def test_missing_vocab_is_fetched_to_the_configured_path(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "sub" / "vocab.txt"
            calls = []

            def fake_ensure(name, path, quiet=False, from_zoo=False):
                calls.append((name, Path(path)))
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("[PAD]\n")
                return Path(path)

            with mock.patch.object(fetch_assets, "ensure", fake_ensure):
                got = deploy.ensure_vocab({"_bert": {"inputs": {"vocab": str(dest)}}})
            self.assertEqual(got, dest)
            self.assertEqual(calls, [("vocab", dest)])
            self.assertTrue(dest.exists())

    def test_existing_vocab_is_kept(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "vocab.txt"
            dest.write_text("my own vocabulary\n")
            with mock.patch.object(fetch_assets, "download",
                                   side_effect=AssertionError("must not download")):
                deploy.ensure_vocab({"_bert": {"inputs": {"vocab": str(dest)}}})
            self.assertEqual(dest.read_text(), "my own vocabulary\n")

    def test_failed_download_stops_the_deploy(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = {"_bert": {"inputs": {"vocab": str(Path(td) / "vocab.txt")}}}
            with mock.patch.object(fetch_assets, "download",
                                   side_effect=fetch_assets.FetchError("offline")), \
                    self.assertRaises(SystemExit) as cm:
                deploy.ensure_vocab(cfg)
            self.assertEqual(cm.exception.code, 1)

    def test_main_fetches_before_the_project_and_the_board(self):
        order = []
        cfg = {"_bert": {}}

        def rec(name, result=None):
            def f(*a, **k):
                order.append(name)
                if name == "connect":
                    raise SystemExit(7)             # stop before any board work
                return result
            return f

        with mock.patch.object(deploy, "load_config", return_value=cfg), \
                mock.patch.object(deploy, "ensure_vocab", rec("vocab")), \
                mock.patch.object(deploy, "ensure_project", rec("project", {})), \
                mock.patch.object(deploy, "board_lock", mock.MagicMock()), \
                mock.patch.object(deploy, "lock_path", return_value=None), \
                mock.patch.object(deploy, "connect", rec("connect")), \
                self.assertRaises(SystemExit):
            deploy.main([])
        self.assertEqual(order, ["vocab", "project", "connect"])

    def test_check_only_does_not_download(self):
        order = []
        with mock.patch.object(deploy, "load_config", return_value={"_bert": {}}), \
                mock.patch.object(deploy, "ensure_vocab", lambda cfg: order.append("vocab")), \
                mock.patch.object(deploy, "board_lock", mock.MagicMock()), \
                mock.patch.object(deploy, "lock_path", return_value=None), \
                mock.patch.object(deploy, "connect", side_effect=SystemExit(7)), \
                self.assertRaises(SystemExit):
            deploy.main(["--check-only"])
        self.assertEqual(order, [])


if __name__ == "__main__":
    unittest.main()

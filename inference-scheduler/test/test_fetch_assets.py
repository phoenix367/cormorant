"""demo/bert_squad/scripts/fetch_assets.py against a local HTTP server (no
network): fresh download, resume of a .part file, md5 / size checks, a web
page instead of the file, existing files kept, the default model path."""

import hashlib
import http.server
import os
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

_STUDY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "demo", "bert_squad", "scripts")
if _STUDY not in sys.path:
    sys.path.append(_STUDY)
import fetch_assets as fa                             # noqa: E402

PAYLOAD = bytes(range(256)) * 4096 + b"tail"          # ~1 MiB


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):                        # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/page"):
            body = b"<html>Virus scan warning</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and self.path.startswith("/ranged"):
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(PAYLOAD) - start))
        self.end_headers()
        self.wfile.write(PAYLOAD[start:])


class TestFetchAssets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.asset = fa.Asset("vocab", "vocab.txt", f"{self.base}/ranged", len(PAYLOAD),
                              hashlib.md5(PAYLOAD).hexdigest(), "by hand")

    def tearDown(self):
        self.td.cleanup()

    def _ensure(self, asset, dest):
        with mock.patch.dict(fa.ASSETS, {asset.name: asset}):
            return fa.ensure(asset.name, dest, quiet=True)

    def test_download_checks_and_renames(self):
        dest = self.dir / "sub" / "vocab.txt"
        self.assertEqual(self._ensure(self.asset, dest), dest)
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        self.assertFalse(dest.with_name("vocab.txt.part").exists())

    def test_resume_from_part(self):
        dest = self.dir / "vocab.txt"
        dest.with_name("vocab.txt.part").write_bytes(PAYLOAD[:300_000])
        self._ensure(self.asset, dest)
        self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_restart_when_server_ignores_range(self):
        dest = self.dir / "vocab.txt"
        dest.with_name("vocab.txt.part").write_bytes(b"x" * 1000)
        self._ensure(replace(self.asset, url=f"{self.base}/plain"), dest)
        self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_md5_mismatch_removes_download(self):
        dest = self.dir / "vocab.txt"
        with self.assertRaisesRegex(fa.FetchError, "md5"):
            self._ensure(replace(self.asset, md5="0" * 32), dest)
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_name("vocab.txt.part").exists())

    def test_web_page_is_an_error(self):
        with self.assertRaisesRegex(fa.FetchError, "web page"):
            self._ensure(replace(self.asset, url=f"{self.base}/page"), self.dir / "vocab.txt")

    def test_existing_file_is_kept(self):
        dest = self.dir / "vocab.txt"
        dest.write_bytes(b"a different vocabulary")
        self._ensure(replace(self.asset, url=f"{self.base}/page"), dest)
        self.assertEqual(dest.read_bytes(), b"a different vocabulary")
        with mock.patch.dict(fa.ASSETS, {"vocab": self.asset}):
            self.assertIn("bytes", fa.verify("vocab", dest))

    def test_default_model_path(self):
        with mock.patch.dict(os.environ, {"BERT_SQUAD_MODEL": "/x/m.onnx"}):
            self.assertEqual(fa.default_model_path(), Path("/x/m.onnx"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BERT_SQUAD_MODEL", None)
            got = fa.default_model_path()
            self.assertIn(got, (fa.ASSETS_DIR / fa.ASSETS["model"].rel, fa.LEGACY_MODEL))


if __name__ == "__main__":
    unittest.main()

"""Tests for CLI project directory output (TestCLI)."""

import os
import sys
import subprocess
import tempfile
import unittest

from helpers import _model, _models_exist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestCLI(unittest.TestCase):

    def _run_cli(self, model_name: str, extra_args=None) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "inference_scheduler.py"),
            _model(model_name),
        ]
        if extra_args:
            cmd += extra_args
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_exit_zero_default_outdir(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run_cli("single_add.onnx", ["--out-dir", td])
            self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_unsupported_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run_cli("unsupported.onnx", ["--out-dir", td])
            self.assertNotEqual(r.returncode, 0)

    def test_project_files_created(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run_cli("relu_chain.onnx", ["--out-dir", td])
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertTrue(os.path.isfile(os.path.join(td, "CMakeLists.txt")))
            self.assertTrue(os.path.isfile(os.path.join(td, "include", "inference.h")))
            self.assertTrue(os.path.isfile(os.path.join(td, "src", "inference.c")))
            self.assertTrue(os.path.isfile(os.path.join(td, "src", "inference_buf.c")))
            self.assertTrue(os.path.isfile(os.path.join(td, "test", "test_inference.c")))
            self.assertTrue(os.path.isfile(os.path.join(td, "scripts", "check_inference_setup.sh")))
            self.assertTrue(os.path.isdir(os.path.join(td, "driver")))

    def test_driver_readme_when_no_src_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            self.assertTrue(
                os.path.isfile(os.path.join(td, "driver", "README.md"))
            )

    def test_driver_copied_when_dir_given(self):
        # The HLS driver of `make synthesize_vectorop_kv260` when the top-level
        # build/ has it, else (a fresh clone, CI) a stand-in directory with the
        # same file names: the copy is what is under test.
        hls_src = os.path.join(
            ROOT, "..", "build", "kernels", "vectorop", "kv260",
            "vadd_kv260", "solution1", "impl", "ip",
            "drivers", "VectorOPKernel_v1_0", "src",
        )
        names = ["xvectoropkernel.c", "xvectoropkernel.h",
                 "xvectoropkernel_hw.h",
                 "xvectoropkernel_sinit.c",
                 "xvectoropkernel_linux.c"]
        with tempfile.TemporaryDirectory() as td:
            driver_src = hls_src
            if not os.path.isdir(hls_src):
                driver_src = os.path.join(td, "hls_driver")
                os.makedirs(driver_src)
                for fname in names:
                    with open(os.path.join(driver_src, fname), "w") as f:
                        f.write(f"/* stand-in for the HLS {fname} */\n")
            out = os.path.join(td, "out")
            r = self._run_cli(
                "single_add.onnx",
                ["--out-dir", out, "--driver-dir", driver_src],
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertNotIn("missing driver files", r.stderr)
            # Both bare-metal and Linux driver variants must be present
            for fname in names:
                got = os.path.join(out, "driver", fname)
                self.assertTrue(os.path.isfile(got), msg=f"Missing driver file: {fname}")
                with open(got, "rb") as a, open(os.path.join(driver_src, fname), "rb") as b:
                    self.assertEqual(a.read(), b.read(), msg=fname)

    def test_header_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "include", "inference.h")) as f:
                h = f.read()
            self.assertIn("#pragma once", h)
            self.assertIn("inference_run", h)

    def test_source_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("relu_chain.onnx", ["--out-dir", td])
            with open(os.path.join(td, "src", "inference.c")) as f:
                s = f.read()
            self.assertIn('#include "inference.h"', s)
            self.assertIn("VECTOROP_ADD", s)
            self.assertIn("VECTOROP_RELU", s)

    def test_buf_impl_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "src", "inference_buf.c")) as f:
                b = f.read()
            self.assertIn("inference_buf_alloc(", b)
            self.assertIn("xclAllocBO(", b)

    def test_setup_script_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "scripts", "check_inference_setup.sh")) as f:
                s = f.read()
            self.assertIn("xrt", s)
            self.assertIn("POOL_BYTES=", s)

    def test_test_source_content_in_project(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_cli("single_add.onnx", ["--out-dir", td])
            with open(os.path.join(td, "test", "test_inference.c")) as f:
                t = f.read()
            self.assertIn("inference_init(", t)
            self.assertIn("inference_run(", t)
            self.assertIn("inference_deinit(", t)
            self.assertIn("test_inference PASSED", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

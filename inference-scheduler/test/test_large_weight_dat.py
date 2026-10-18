# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for large-weight external .dat file handling (TestLargeWeightDat)."""

import os
import sys
import subprocess
import tempfile
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestLargeWeightDat(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # Threshold classification
    def test_small_weight_not_large(self):
        # single_add bias has 256 elements — well below threshold
        _, cg = self._gen("single_add.onnx")
        self.assertEqual(cg.large_weight_tensors, [])

    def test_large_tensor_weight_is_large(self):
        # large_tensor bias has 1 048 576 elements — above threshold
        _, cg = self._gen("large_tensor.onnx")
        self.assertEqual(len(cg.large_weight_tensors), 1)
        self.assertEqual(cg.large_weight_tensors[0].c_name, "bias")

    # Source: no ROM array for large weights
    def test_no_rom_array_for_large_weight(self):
        _, cg = self._gen("large_tensor.onnx")
        s = cg.generate_source()
        self.assertNotIn("_rom_bias", s)

    def test_load_weight_helper_in_source(self):
        _, cg = self._gen("large_tensor.onnx")
        s = cg.generate_source()
        self.assertIn("_load_weight(", s)
        self.assertIn("fread(", s)

    def test_stdio_included_for_large_weight(self):
        _, cg = self._gen("large_tensor.onnx")
        self.assertIn("#include <stdio.h>", cg.generate_source())

    def test_stdio_not_included_without_large_weight(self):
        _, cg = self._gen("single_add.onnx")
        self.assertNotIn("#include <stdio.h>", cg.generate_source())

    def test_load_weight_called_in_init(self):
        _, cg = self._gen("large_tensor.onnx")
        s = cg.generate_source()
        self.assertIn('_load_weight(bias, "bias",', s)

    def test_memcpy_not_used_for_large_weight(self):
        _, cg = self._gen("large_tensor.onnx")
        s = cg.generate_source()
        # memcpy for _rom_bias must be absent; memcpy may be present for other reasons
        self.assertNotIn("_rom_bias", s)

    def test_extern_ptr_decl_in_source(self):
        _, cg = self._gen("large_tensor.onnx")
        s = cg.generate_source()
        self.assertIn("External weight 'bias'", s)
        self.assertIn("weights/bias.dat", s)

    def test_weights_dir_in_cmake(self):
        _, cg = self._gen("large_tensor.onnx")
        cm = cg.generate_cmake()
        self.assertIn("INFERENCE_WEIGHTS_DIR", cm)

    def test_dat_bytes_correct_size(self):
        _, cg = self._gen("large_tensor.onnx")
        t = cg.large_weight_tensors[0]
        data = cg.generate_weight_dat(t)
        # 1 048 576 elements × 2 bytes each
        self.assertEqual(len(data), 1048576 * 2)

    def test_dat_bytes_type(self):
        _, cg = self._gen("large_tensor.onnx")
        t = cg.large_weight_tensors[0]
        self.assertIsInstance(cg.generate_weight_dat(t), bytes)

    def test_small_weight_still_inline(self):
        # normalize has mean + std, each 256 elements — must stay as ROM arrays
        _, cg = self._gen("normalize.onnx")
        s = cg.generate_source()
        self.assertIn("_rom_mean", s)
        self.assertIn("_rom_std", s)
        self.assertNotIn("_load_weight", s)

    # CLI: weights/ directory written for large-weight models
    def test_cli_writes_dat_file(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                [sys.executable,
                 os.path.join(ROOT, "inference_scheduler.py"),
                 _model("large_tensor.onnx"), "--out-dir", td],
                check=True, capture_output=True,
            )
            dat = os.path.join(td, "weights", "bias.dat")
            self.assertTrue(os.path.isfile(dat))
            self.assertEqual(os.path.getsize(dat), 1048576 * 2)

    def test_cli_no_weights_dir_for_small_model(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                [sys.executable,
                 os.path.join(ROOT, "inference_scheduler.py"),
                 _model("single_add.onnx"), "--out-dir", td],
                check=True, capture_output=True,
            )
            self.assertFalse(os.path.isdir(os.path.join(td, "weights")))

    # --embed-large-weights flag
    def test_embed_flag_inlines_large_weight(self):
        g  = OnnxGraph(_model("large_tensor.onnx"))
        cg = CodeGenerator(g, model_path=_model("large_tensor.onnx"),
                           embed_large_weights=True)
        self.assertEqual(cg.large_weight_tensors, [])
        s = cg.generate_source()
        self.assertIn("_rom_bias", s)
        self.assertNotIn("_load_weight", s)

    def test_embed_flag_no_stdio(self):
        g  = OnnxGraph(_model("large_tensor.onnx"))
        cg = CodeGenerator(g, model_path=_model("large_tensor.onnx"),
                           embed_large_weights=True)
        self.assertNotIn("#include <stdio.h>", cg.generate_source())

    def test_cli_embed_flag_no_dat_file(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                [sys.executable,
                 os.path.join(ROOT, "inference_scheduler.py"),
                 _model("large_tensor.onnx"),
                 "--out-dir", td, "--embed-large-weights"],
                check=True, capture_output=True,
            )
            self.assertFalse(os.path.isdir(os.path.join(td, "weights")))

    def test_cli_embed_flag_rom_array_in_source(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                [sys.executable,
                 os.path.join(ROOT, "inference_scheduler.py"),
                 _model("large_tensor.onnx"),
                 "--out-dir", td, "--embed-large-weights"],
                check=True, capture_output=True,
            )
            with open(os.path.join(td, "src", "inference.c")) as f:
                s = f.read()
            self.assertIn("_rom_bias", s)
            self.assertNotIn("_load_weight", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)

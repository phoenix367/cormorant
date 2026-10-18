# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for setup script generation (TestSetupScript)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSetupScript(unittest.TestCase):

    def _script(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return cg.generate_setup_script()

    def test_shebang(self):
        s = self._script("single_add.onnx")
        self.assertTrue(s.startswith("#!/bin/sh"))

    def test_xrt_referenced_in_script(self):
        s = self._script("single_add.onnx")
        self.assertIn("xrt", s)

    def test_pool_size_in_script(self):
        from src.graph import OnnxGraph as G
        from src.codegen import CodeGenerator as CG
        g  = G(_model("single_add.onnx"))
        cg = CG(g, model_path=_model("single_add.onnx"))
        pool_bytes = cg._compute_pool_bytes()
        s = self._script("single_add.onnx")
        self.assertIn(str(pool_bytes), s)

    def test_xrt_lib_in_script(self):
        s = self._script("single_add.onnx")
        self.assertIn("libxrt_core.so", s)

    def test_pkg_config_in_script(self):
        s = self._script("single_add.onnx")
        self.assertIn("pkg-config", s)

    def test_usr_include_xrt_candidate_in_script(self):
        # Kria apt-package layout must be in the fallback search list
        s = self._script("single_add.onnx")
        self.assertIn("/usr/include/xrt/xrt.h", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)

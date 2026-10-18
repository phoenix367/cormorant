# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for two-input model handling (TestTwoInputModels)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestTwoInputModels(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # Header: both inputs appear as size macros and in inference_run()
    def test_two_input_add_size_macros(self):
        g, cg = self._gen("two_input_add.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_A_SIZE", h)
        self.assertIn("INFERENCE_B_SIZE", h)
        self.assertIn("INFERENCE_Y_SIZE", h)

    def test_two_input_add_run_signature(self):
        g, cg = self._gen("two_input_add.onnx")
        h = cg.generate_header()
        # inference_run must accept both A and B as buf parameters
        self.assertIn("inference_buf_t *A", h)
        self.assertIn("inference_buf_t *B", h)

    def test_two_input_chain_size_macros(self):
        g, cg = self._gen("two_input_chain.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_A_SIZE", h)
        self.assertIn("INFERENCE_B_SIZE", h)

    # Source: both inputs are passed through to run_op
    def test_two_input_add_op_in_source(self):
        g, cg = self._gen("two_input_add.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)

    def test_two_input_chain_ops_in_source(self):
        g, cg = self._gen("two_input_chain.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("VECTOROP_RELU", s)

    # Generated test program: alloc buffers for both runtime inputs
    def test_two_input_add_test_allocs(self):
        g, cg = self._gen("two_input_add.onnx")
        t = cg.generate_test()
        self.assertIn("inference_buf_alloc(INFERENCE_A_SIZE)", t)
        self.assertIn("inference_buf_alloc(INFERENCE_B_SIZE)", t)
        self.assertIn("inference_buf_alloc(INFERENCE_Y_SIZE)", t)

    def test_two_input_chain_test_allocs(self):
        g, cg = self._gen("two_input_chain.onnx")
        t = cg.generate_test()
        self.assertIn("inference_buf_alloc(INFERENCE_A_SIZE)", t)
        self.assertIn("inference_buf_alloc(INFERENCE_B_SIZE)", t)

    # Graph-level input count
    def test_two_input_add_has_two_inputs(self):
        g, _ = self._gen("two_input_add.onnx")
        self.assertEqual(len(g.input_tensors), 2)

    def test_two_input_chain_has_two_inputs(self):
        g, _ = self._gen("two_input_chain.onnx")
        self.assertEqual(len(g.input_tensors), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

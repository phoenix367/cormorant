# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for two-output model handling (TestTwoOutputModels)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestTwoOutputModels(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # Graph topology
    def test_tap_has_one_input(self):
        g, _ = self._gen("two_output_tap.onnx")
        self.assertEqual(len(g.input_tensors), 1)

    def test_tap_has_two_outputs(self):
        g, _ = self._gen("two_output_tap.onnx")
        self.assertEqual(len(g.output_tensors), 2)

    def test_tap_no_intermediates(self):
        # add_Y is a graph output, so no intermediate buffers are allocated
        g, _ = self._gen("two_output_tap.onnx")
        self.assertEqual(len(g.intermediate_tensors), 0)

    def test_chain_has_one_input(self):
        g, _ = self._gen("two_output_chain.onnx")
        self.assertEqual(len(g.input_tensors), 1)

    def test_chain_has_two_outputs(self):
        g, _ = self._gen("two_output_chain.onnx")
        self.assertEqual(len(g.output_tensors), 2)

    def test_chain_has_intermediate_buffer(self):
        # mul_Y is between the two tapped outputs — must be allocated internally
        g, _ = self._gen("two_output_chain.onnx")
        interm_names = [t.c_name for t in g.intermediate_tensors]
        self.assertIn("mul_Y", interm_names)

    # Header: both outputs appear as size macros and in inference_run()
    def test_tap_size_macros(self):
        _, cg = self._gen("two_output_tap.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_ADD_Y_SIZE", h)
        self.assertIn("INFERENCE_RELU_Y_SIZE", h)

    def test_tap_run_signature(self):
        _, cg = self._gen("two_output_tap.onnx")
        h = cg.generate_header()
        self.assertIn("inference_buf_t *add_Y", h)
        self.assertIn("inference_buf_t *relu_Y", h)

    def test_chain_run_signature(self):
        _, cg = self._gen("two_output_chain.onnx")
        h = cg.generate_header()
        self.assertIn("inference_buf_t *add_Y", h)
        self.assertIn("inference_buf_t *clip_Y", h)

    # Source: ops are scheduled correctly
    def test_tap_ops_in_source(self):
        _, cg = self._gen("two_output_tap.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("VECTOROP_RELU", s)

    def test_chain_ops_in_source(self):
        _, cg = self._gen("two_output_chain.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("VECTOROP_MUL", s)
        self.assertIn("VECTOROP_RELU6", s)

    # Generated test program: allocates buffers for both outputs
    def test_tap_test_allocs_both_outputs(self):
        _, cg = self._gen("two_output_tap.onnx")
        t = cg.generate_test()
        self.assertIn("inference_buf_alloc(INFERENCE_ADD_Y_SIZE)", t)
        self.assertIn("inference_buf_alloc(INFERENCE_RELU_Y_SIZE)", t)

    def test_chain_test_allocs_both_outputs(self):
        _, cg = self._gen("two_output_chain.onnx")
        t = cg.generate_test()
        self.assertIn("inference_buf_alloc(INFERENCE_ADD_Y_SIZE)", t)
        self.assertIn("inference_buf_alloc(INFERENCE_CLIP_Y_SIZE)", t)

    def test_tap_test_prints_both_outputs(self):
        _, cg = self._gen("two_output_tap.onnx")
        t = cg.generate_test()
        self.assertIn("Output 'add_Y'", t)
        self.assertIn("Output 'relu_Y'", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

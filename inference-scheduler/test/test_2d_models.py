# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for 2D tensor model handling (TestTwoDimModels)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestTwoDimModels(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # Shape is correctly parsed and stored
    def test_batch_relu_input_shape(self):
        g, _ = self._gen("batch_relu.onnx")
        self.assertEqual(g.input_tensors[0].shape, [4, 64])

    def test_batch_relu_output_shape(self):
        g, _ = self._gen("batch_relu.onnx")
        self.assertEqual(g.output_tensors[0].shape, [4, 64])

    def test_matrix_ops_input_shape(self):
        g, _ = self._gen("matrix_ops.onnx")
        shapes = [t.shape for t in g.input_tensors]
        self.assertIn([8, 32], shapes)

    # numel is the product of all dimensions
    def test_batch_relu_numel(self):
        g, _ = self._gen("batch_relu.onnx")
        self.assertEqual(g.input_tensors[0].numel, 4 * 64)

    def test_matrix_ops_numel(self):
        g, _ = self._gen("matrix_ops.onnx")
        self.assertEqual(g.input_tensors[0].numel, 8 * 32)

    # Size macros reflect the flattened element count
    def test_batch_relu_size_macro(self):
        _, cg = self._gen("batch_relu.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_X_SIZE", h)
        self.assertIn(str(4 * 64), h)

    def test_matrix_ops_size_macros(self):
        _, cg = self._gen("matrix_ops.onnx")
        h = cg.generate_header()
        self.assertIn(str(8 * 32), h)

    # Shape annotation appears in comments
    def test_batch_relu_shape_in_header_comment(self):
        _, cg = self._gen("batch_relu.onnx")
        h = cg.generate_header()
        self.assertIn("shape=[4, 64]", h)

    def test_matrix_ops_shape_in_header_comment(self):
        _, cg = self._gen("matrix_ops.onnx")
        h = cg.generate_header()
        self.assertIn("shape=[8, 32]", h)

    # Ops are scheduled correctly
    def test_batch_relu_ops(self):
        _, cg = self._gen("batch_relu.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("VECTOROP_RELU", s)

    def test_matrix_ops_ops(self):
        _, cg = self._gen("matrix_ops.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_MUL", s)
        self.assertIn("VECTOROP_RELU6", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)

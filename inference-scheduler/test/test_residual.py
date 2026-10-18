# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for residual connection model handling (TestResidualModels)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestResidualModels(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # ---- residual_add: Relu(X) -> relu_X, X + relu_X -> Y ----------

    def test_add_single_input(self):
        g, _ = self._gen("residual_add.onnx")
        self.assertEqual(len(g.input_tensors), 1)

    def test_add_single_output(self):
        g, _ = self._gen("residual_add.onnx")
        self.assertEqual(len(g.output_tensors), 1)

    def test_add_intermediate_relu_x(self):
        # relu_X is the only intermediate — X and Y are boundary tensors
        g, _ = self._gen("residual_add.onnx")
        names = [t.c_name for t in g.intermediate_tensors]
        self.assertIn("relu_X", names)
        self.assertNotIn("X", names)
        self.assertNotIn("Y", names)

    def test_add_ops(self):
        _, cg = self._gen("residual_add.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_RELU", s)
        self.assertIn("VECTOROP_ADD", s)

    def test_add_skip_in_source(self):
        # The final Add must pass X (the original input) as one operand
        _, cg = self._gen("residual_add.onnx")
        s = cg.generate_source()
        # run_op(relu_X, X, Y, ...) or run_op(X, relu_X, Y, ...)
        self.assertIn("run_op(relu_X, X, Y,", s)

    # ---- residual_chain: X+bias -> Relu -> *scale -> X+result -> Y -

    def test_chain_single_input(self):
        g, _ = self._gen("residual_chain.onnx")
        self.assertEqual(len(g.input_tensors), 1)

    def test_chain_single_output(self):
        g, _ = self._gen("residual_chain.onnx")
        self.assertEqual(len(g.output_tensors), 1)

    def test_chain_intermediates(self):
        g, _ = self._gen("residual_chain.onnx")
        names = [t.c_name for t in g.intermediate_tensors]
        self.assertIn("add_X",  names)
        self.assertIn("relu_X", names)
        self.assertIn("mul_X",  names)

    def test_chain_ops(self):
        _, cg = self._gen("residual_chain.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("VECTOROP_RELU", s)
        self.assertIn("VECTOROP_MUL", s)

    def test_chain_skip_in_source(self):
        # Final Add must use X as one of its operands (the skip connection)
        _, cg = self._gen("residual_chain.onnx")
        s = cg.generate_source()
        self.assertIn("run_op(X, mul_X, Y,", s)

    def test_chain_two_nodes_reference_X(self):
        # X appears as an argument in exactly two run_op calls
        _, cg = self._gen("residual_chain.onnx")
        s = cg.generate_source()
        import re
        calls_with_X = [ln for ln in s.splitlines()
                        if re.search(r'\brun_op\b.*\bX\b', ln)]
        self.assertEqual(len(calls_with_X), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

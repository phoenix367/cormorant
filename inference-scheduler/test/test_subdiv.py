# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for Sub and Div model handling (TestSubDivModels)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSubDivModels(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # sub_bias: X - mean -> Y
    def test_sub_op_in_source(self):
        _, cg = self._gen("sub_bias.onnx")
        self.assertIn("VECTOROP_SUB", cg.generate_source())

    def test_sub_no_other_ops(self):
        _, cg = self._gen("sub_bias.onnx")
        # Only VECTOROP_SUB should appear in run_op() call sites
        import re
        calls = re.findall(r'run_op\([^)]+\)', cg.generate_source())
        ops_used = set(re.findall(r'VECTOROP_\w+', " ".join(calls)))
        self.assertEqual(ops_used, {"VECTOROP_SUB"})

    def test_sub_single_node(self):
        g, _ = self._gen("sub_bias.onnx")
        self.assertEqual(len(g.nodes), 1)

    def test_sub_mean_is_weight(self):
        g, _ = self._gen("sub_bias.onnx")
        weight_names = [t.c_name for t in g.weight_tensors]
        self.assertIn("mean", weight_names)

    # div_scale: X / std -> Y
    def test_div_op_in_source(self):
        _, cg = self._gen("div_scale.onnx")
        self.assertIn("VECTOROP_DIV", cg.generate_source())

    def test_div_no_other_ops(self):
        _, cg = self._gen("div_scale.onnx")
        import re
        calls = re.findall(r'run_op\([^)]+\)', cg.generate_source())
        ops_used = set(re.findall(r'VECTOROP_\w+', " ".join(calls)))
        self.assertEqual(ops_used, {"VECTOROP_DIV"})

    def test_div_single_node(self):
        g, _ = self._gen("div_scale.onnx")
        self.assertEqual(len(g.nodes), 1)

    def test_div_std_is_weight(self):
        g, _ = self._gen("div_scale.onnx")
        weight_names = [t.c_name for t in g.weight_tensors]
        self.assertIn("std", weight_names)

    # normalize: (X - mean) / std -> Y
    def test_normalize_two_nodes(self):
        g, _ = self._gen("normalize.onnx")
        self.assertEqual(len(g.nodes), 2)

    def test_normalize_sub_and_div_in_source(self):
        _, cg = self._gen("normalize.onnx")
        s = cg.generate_source()
        self.assertIn("VECTOROP_SUB", s)
        self.assertIn("VECTOROP_DIV", s)

    def test_normalize_intermediate_sub_y(self):
        g, _ = self._gen("normalize.onnx")
        names = [t.c_name for t in g.intermediate_tensors]
        self.assertIn("sub_Y", names)

    def test_normalize_both_weights_present(self):
        g, _ = self._gen("normalize.onnx")
        weight_names = [t.c_name for t in g.weight_tensors]
        self.assertIn("mean", weight_names)
        self.assertIn("std",  weight_names)

    def test_normalize_call_order(self):
        # Sub run_op call must appear before Div run_op call
        import re
        _, cg = self._gen("normalize.onnx")
        lines = cg.generate_source().splitlines()
        call_lines = [l for l in lines if re.search(r'run_op\(', l)
                      and re.search(r'VECTOROP_', l)]
        ops_in_order = [re.search(r'VECTOROP_\w+', l).group() for l in call_lines]
        self.assertEqual(ops_in_order, ["VECTOROP_SUB", "VECTOROP_DIV"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

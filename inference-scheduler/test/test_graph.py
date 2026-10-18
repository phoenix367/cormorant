# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ONNX graph parsing (TestGraphParsing)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.nodes   import SchedulerError


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestGraphParsing(unittest.TestCase):

    def test_single_add_nodes(self):
        g = OnnxGraph(_model("single_add.onnx"))
        self.assertEqual(len(g.nodes), 1)
        self.assertEqual(g.nodes[0].onnx_node.op_type, "Add")

    def test_single_add_weight(self):
        g = OnnxGraph(_model("single_add.onnx"))
        weights = g.weight_tensors
        self.assertEqual(len(weights), 1)
        self.assertTrue(weights[0].is_weight)

    def test_relu_chain_nodes(self):
        g = OnnxGraph(_model("relu_chain.onnx"))
        ops = [sn.onnx_node.op_type for sn in g.nodes]
        self.assertEqual(ops, ["Add", "Relu"])

    def test_mixed_ops_nodes(self):
        g = OnnxGraph(_model("mixed_ops.onnx"))
        ops = [sn.onnx_node.op_type for sn in g.nodes]
        self.assertEqual(ops, ["Add", "Mul", "Clip"])

    def test_mixed_ops_clip_is_relu6(self):
        from src.nodes import OP_RELU6
        g = OnnxGraph(_model("mixed_ops.onnx"))
        clip_node = g.nodes[2]
        self.assertEqual(clip_node.op_code, OP_RELU6)
        self.assertEqual(clip_node.arity, 1)

    def test_unsupported_raises(self):
        with self.assertRaises(SchedulerError):
            OnnxGraph(_model("unsupported.onnx"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for large tensor model handling (TestLargeTensorModel)."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestLargeTensorModel(unittest.TestCase):

    _NUMEL = 64 * 64 * 16 * 16   # 1 048 576

    def _gen(self):
        g  = OnnxGraph(_model("large_tensor.onnx"))
        cg = CodeGenerator(g, model_path=_model("large_tensor.onnx"))
        return g, cg

    def test_input_shape(self):
        g, _ = self._gen()
        self.assertEqual(g.input_tensors[0].shape, [64, 64, 16, 16])

    def test_output_shapes(self):
        g, _ = self._gen()
        for t in g.output_tensors:
            self.assertEqual(t.shape, [64, 64, 16, 16])

    def test_numel(self):
        g, _ = self._gen()
        self.assertEqual(g.input_tensors[0].numel, self._NUMEL)

    def test_two_outputs(self):
        g, _ = self._gen()
        self.assertEqual(len(g.output_tensors), 2)

    def test_no_intermediates(self):
        # add_Y is a graph output, so no intermediate buffers are needed
        g, _ = self._gen()
        self.assertEqual(len(g.intermediate_tensors), 0)

    def test_size_macros_value(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn(f"{self._NUMEL}u", h)

    def test_shape_in_comment(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("shape=[64, 64, 16, 16]", h)

    def test_pool_size_covers_all_buffers(self):
        # X + add_Y + Y + bias weight, each 2 MB => 8 MB total
        _, cg = self._gen()
        pool = cg._compute_pool_bytes()
        self.assertGreaterEqual(pool, self._NUMEL * 2 * 4)  # at least 4 buffers

    def test_ops_scheduled(self):
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("VECTOROP_RELU", s)

    def test_test_allocs_both_outputs(self):
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("inference_buf_alloc(INFERENCE_ADD_Y_SIZE)", t)
        self.assertIn("inference_buf_alloc(INFERENCE_Y_SIZE)", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for broadcast_chain model — stride propagation through two levels."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBroadcastChain(unittest.TestCase):
    """
    Model: X[4,6] + bias[6] -> add_Y[4,6] -> Relu -> relu_Y[4,6] -> +X -> Y[4,6]

    Node 0 (Add):   broadcast, outer_count=4, chunk=6, aligned_chunk=8
    Node 1 (Relu):  non-broadcast — level-1 stride propagation from add_Y
    Node 2 (Add):   non-broadcast — level-2 stride propagation from relu_Y / X

    All four data buffers (X, add_Y, relu_Y, Y) must have alloc=32.
    """

    def _gen(self):
        g  = OnnxGraph(_model("broadcast_chain.onnx"))
        cg = CodeGenerator(g, model_path=_model("broadcast_chain.onnx"))
        return g, cg

    # ---- graph structure ------------------------------------------- #

    def test_three_nodes(self):
        g, _ = self._gen()
        self.assertEqual(len(g.nodes), 3)

    def test_node0_is_broadcast_add(self):
        g, _ = self._gen()
        sn = g.nodes[0]
        self.assertEqual(sn.onnx_node.op_type, "Add")
        self.assertEqual(sn.outer_count, 4)
        self.assertEqual(sn.chunk_size, 6)
        self.assertEqual(sn.aligned_chunk_size, 8)

    def test_node1_is_non_broadcast_relu(self):
        g, _ = self._gen()
        sn = g.nodes[1]
        self.assertEqual(sn.onnx_node.op_type, "Relu")
        self.assertEqual(sn.outer_count, 1)

    def test_node2_is_non_broadcast_add(self):
        g, _ = self._gen()
        sn = g.nodes[2]
        self.assertEqual(sn.onnx_node.op_type, "Add")
        self.assertEqual(sn.outer_count, 1)

    # ---- alloc sizes: two levels of propagation -------------------- #

    def test_add_y_alloc_is_padded(self):
        """Direct broadcast output: 4 outer × 8 aligned = 32."""
        g, cg = self._gen()
        add_y = g.nodes[0].output
        self.assertEqual(cg._alloc_sizes[add_y.onnx_name], 32)

    def test_x_alloc_is_padded(self):
        """X advances in the broadcast node → 4 × 8 = 32."""
        g, cg = self._gen()
        x = g.input_tensors[0]
        self.assertEqual(cg._alloc_sizes[x.onnx_name], 32)

    def test_relu_y_alloc_inherits_padding(self):
        """Level-1 propagation: Relu output inherits add_Y's padded alloc."""
        g, cg = self._gen()
        relu_y = g.nodes[1].output
        self.assertEqual(cg._alloc_sizes[relu_y.onnx_name], 32)

    def test_y_alloc_inherits_padding(self):
        """Level-2 propagation: final Add output inherits relu_Y's padded alloc."""
        g, cg = self._gen()
        y = g.output_tensors[0]
        self.assertEqual(cg._alloc_sizes[y.onnx_name], 32)

    # ---- source: all three op calls correct ------------------------ #

    def test_only_run_op_helper_defined(self):
        """All nodes use run_op() — no run_op_at() emitted."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("static void run_op(", s)
        self.assertNotIn("static void run_op_at(", s)

    def test_node0_emits_broadcast_run_op(self):
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn(
            "run_op(X, bias, add_Y, INFERENCE_ADD_Y_CHUNK, VECTOROP_ADD, 4u,", s
        )
        self.assertNotIn("run_op_at(", s)

    def test_node1_relu_uses_padded_size(self):
        """Relu must use size=32, not 24, to cover the full strided buffer."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("run_op(add_Y, NULL, relu_Y, 32u,", s)

    def test_node2_add_uses_padded_size(self):
        """Final Add must also use size=32 (level-2 propagation)."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("run_op(relu_Y, X, Y, 32u,", s)

    def test_node1_relu_not_24(self):
        """Regression: 24u would only process 3 of 4 rows."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("run_op(add_Y, NULL, relu_Y, 24u,", s)

    def test_node2_add_not_24(self):
        """Regression: 24u would only process 3 of 4 rows."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("run_op(relu_Y, X, Y, 24u,", s)

    # ---- header: SIZE macro references ADD_Y stride ---------------- #

    def test_add_y_chunk_macro_in_header(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_ADD_Y_CHUNK", h)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    def test_no_relu_y_chunk_macro(self):
        """relu_Y has no CHUNK macro of its own — it reuses ADD_Y's macros."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertNotIn("INFERENCE_RELU_Y_CHUNK ", h)
        self.assertNotIn("INFERENCE_RELU_Y_CHUNK_STRIDE", h)

    def test_no_y_chunk_macro(self):
        """Y has no CHUNK macro of its own — it reuses ADD_Y's macros."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertNotIn("INFERENCE_Y_CHUNK ", h)
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE", h)

    def test_y_size_uses_add_y_chunk_stride(self):
        """Two levels deep: INFERENCE_Y_SIZE still references ADD_Y_CHUNK_STRIDE."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("4u * INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    def test_x_size_uses_add_y_chunk_stride(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_X_SIZE", h)
        self.assertIn("4u * INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    # ---- test_inference.c: chunk-loop fill and print --------------- #

    def test_input_fill_uses_chunk_loop(self):
        """X fill must use chunk-by-chunk loop, not a flat SIZE loop."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("broadcast input, fill data chunks only", t)
        self.assertIn("for (_chunk = 0u; _chunk < 4u; _chunk++)", t)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", t)

    def test_output_print_uses_chunk_loop(self):
        """Y inherits the stride so test printing must skip gap elements."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("broadcast output, print data chunks only", t)
        self.assertIn("for (_chunk = 0u; _chunk < 4u && shown < lim; _chunk++)", t)
        self.assertNotIn("for (i = 0u; i < lim; i++)", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

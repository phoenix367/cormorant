# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for broadcast_tapped model — broadcast output exposed as a graph output."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBroadcastTapped(unittest.TestCase):
    """
    Model: X[4,6] + bias[6] -> add_Y[4,6] (graph output 0)
                             -> Relu -> Y[4,6]  (graph output 1)

    add_Y is both the output of a broadcast node AND a graph output, so it
    must be a parameter of inference_run() rather than a static internal buffer.
    Y is a non-broadcast output (outer_count=1) that inherits the strided alloc.
    There are no intermediate tensors.
    """

    def _gen(self):
        g  = OnnxGraph(_model("broadcast_tapped.onnx"))
        cg = CodeGenerator(g, model_path=_model("broadcast_tapped.onnx"))
        return g, cg

    # ---- graph structure ------------------------------------------- #

    def test_two_nodes(self):
        g, _ = self._gen()
        self.assertEqual(len(g.nodes), 2)

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

    def test_two_outputs_no_intermediates(self):
        """add_Y and Y are both graph outputs; no internal intermediate buffers."""
        g, _ = self._gen()
        self.assertEqual(len(g.output_tensors), 2)
        self.assertEqual(len(g.intermediate_tensors), 0)

    def test_add_y_is_graph_output(self):
        g, _ = self._gen()
        output_names = [t.onnx_name for t in g.output_tensors]
        self.assertIn("add_Y", output_names)

    def test_y_is_graph_output(self):
        g, _ = self._gen()
        output_names = [t.onnx_name for t in g.output_tensors]
        self.assertIn("Y", output_names)

    # ---- alloc sizes ----------------------------------------------- #

    def test_x_alloc(self):
        """X: 4 outer × 8 aligned = 32."""
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["X"], 32)

    def test_add_y_alloc(self):
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["add_Y"], 32)

    def test_y_alloc_inherits_stride(self):
        """Y is a non-broadcast output; inherits strided alloc = 32."""
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["Y"], 32)

    def test_bias_alloc_is_aligned_chunk(self):
        """Broadcast bias [6]: one aligned block of 8."""
        g, cg = self._gen()
        bias_name = g.nodes[0].inputs[1].onnx_name
        self.assertEqual(cg._alloc_sizes[bias_name], 8)

    # ---- source: inference_run signature --------------------------- #

    def test_inference_run_has_x_add_y_y(self):
        """inference_run() takes X (input), add_Y and Y (both outputs)."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("inference_buf_t *X", s)
        self.assertIn("inference_buf_t *add_Y", s)
        self.assertIn("inference_buf_t *Y", s)

    def test_relu_uses_padded_size(self):
        """run_op for Relu must use the full 32-element alloc."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("run_op(add_Y, NULL, Y, 32u,", s)

    def test_relu_not_24(self):
        """24u would miss the last row in the strided layout."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("run_op(add_Y, NULL, Y, 24u,", s)

    # ---- header macros --------------------------------------------- #

    def test_add_y_chunk_macros_present(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_ADD_Y_CHUNK", h)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    def test_add_y_size_macro(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_ADD_Y_SIZE", h)
        self.assertIn("4u * INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    def test_y_size_uses_add_y_stride(self):
        """Y inherits ADD_Y's stride for its SIZE macro."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("4u * INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    def test_no_y_chunk_macro(self):
        """Y is a non-broadcast output — no separate CHUNK macros for it."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertNotIn("INFERENCE_Y_CHUNK ", h)
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE", h)

    # ---- test_inference.c ------------------------------------------ #

    def test_fill_uses_broadcast_chunk_loop(self):
        """X input fill uses the chunk-based strategy (broadcast input)."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("broadcast input, fill data chunks only", t)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", t)

    def test_print_add_y_uses_chunk_loop(self):
        """add_Y output print uses chunk loop (it is a broadcast output)."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("broadcast output, print data chunks only", t)
        self.assertIn("for (_chunk = 0u; _chunk < 4u && shown < lim; _chunk++)", t)

    def test_no_static_add_y_buffer(self):
        """add_Y is a graph output (caller-supplied), never a static buffer."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("static inference_buf_t *add_Y", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)

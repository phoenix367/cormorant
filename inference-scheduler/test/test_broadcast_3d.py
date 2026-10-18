# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for broadcast_3d model — 3-D input shape with outer_count=6."""

import re
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBroadcast3D(unittest.TestCase):
    """
    Model: X[2,3,4] + bias[4] -> Y[2,3,4]

    The trailing dimension is 4 (< align_elems=8) so aligned_chunk_size = 8.
    outer_count = 2*3 = 6 — verifies that the scheduler correctly flattens
    all leading dimensions into the outer loop.
    """

    def _gen(self):
        g  = OnnxGraph(_model("broadcast_3d.onnx"))
        cg = CodeGenerator(g, model_path=_model("broadcast_3d.onnx"))
        return g, cg

    # ---- node properties ------------------------------------------- #

    def test_single_broadcast_node(self):
        g, _ = self._gen()
        self.assertEqual(len(g.nodes), 1)
        sn = g.nodes[0]
        self.assertEqual(sn.onnx_node.op_type, "Add")

    def test_outer_count_six(self):
        """2*3 leading dims flattened to outer_count=6."""
        g, _ = self._gen()
        self.assertEqual(g.nodes[0].outer_count, 6)

    def test_chunk_size_four(self):
        g, _ = self._gen()
        self.assertEqual(g.nodes[0].chunk_size, 4)

    def test_aligned_chunk_is_align_elems(self):
        """chunk=4 rounds up to align_elems=8 (for ap_fixed<16,8>)."""
        g, _ = self._gen()
        sn = g.nodes[0]
        self.assertEqual(sn.aligned_chunk_size, 8)

    # ---- alloc sizes ----------------------------------------------- #

    def test_x_alloc(self):
        """X: 6 outer × 8 aligned = 48."""
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["X"], 48)

    def test_y_alloc(self):
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["Y"], 48)

    def test_bias_alloc_is_aligned_chunk(self):
        """Repeating bias [4]: one aligned block of 8."""
        g, cg = self._gen()
        bias_name = g.nodes[0].inputs[1].onnx_name
        self.assertEqual(cg._alloc_sizes[bias_name], 8)

    # ---- weight ROM ------------------------------------------------ #

    def test_bias_rom_has_four_elements(self):
        """Broadcast bias [4]: ROM holds 4 data values (not padded to 8)."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("_rom_bias[4]", s)
        self.assertNotIn("_rom_bias[8]", s)
        self.assertNotIn("_rom_bias[48]", s)

    def test_bias_not_in_strided_params(self):
        """bias is in a broadcast node — must not appear in strided weight params."""
        _, cg = self._gen()
        self.assertNotIn("bias", cg._strided_weight_params())

    # ---- header macros --------------------------------------------- #

    def test_y_chunk_macro_is_four(self):
        _, cg = self._gen()
        h = cg.generate_header()
        m = re.search(r"#define\s+INFERENCE_Y_CHUNK\s+(\d+)u", h)
        self.assertIsNotNone(m, "INFERENCE_Y_CHUNK not found")
        self.assertEqual(int(m.group(1)), 4)

    def test_y_chunk_stride_macro_present(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE", h)

    def test_x_size_macro_uses_six_strides(self):
        """X spans 6 outer blocks × stride."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("6u * INFERENCE_Y_CHUNK_STRIDE", h)
        self.assertIn("INFERENCE_X_SIZE", h)

    def test_y_size_macro_uses_six_strides(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("6u * INFERENCE_Y_CHUNK_STRIDE", h)

    # ---- source: broadcast loop ------------------------------------ #

    def test_broadcast_run_op_with_outer_six(self):
        """Single run_op() call with outer=6 — no CPU for-loop."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("6u, INFERENCE_Y_CHUNK_STRIDE, 0u);", s)
        self.assertNotIn("for (unsigned _i", s)

    def test_run_op_at_uses_chunk_macro(self):
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK, VECTOROP_ADD", s)
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE, VECTOROP_ADD", s)

    # ---- test_inference.c ------------------------------------------ #

    def test_fill_outer_loop_six(self):
        """Fill loop must iterate all 6 chunks."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("for (_chunk = 0u; _chunk < 6u; _chunk++)", t)

    def test_fill_inner_loop_uses_chunk_macro(self):
        """Inner fill loop iterates INFERENCE_Y_CHUNK (=4) elements."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("for (i = 0u; i < INFERENCE_Y_CHUNK; i++)", t)

    def test_print_numel(self):
        """numel reported as 6 * CHUNK."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("numel = 6u * INFERENCE_Y_CHUNK", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

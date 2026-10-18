# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for chunk_one model — maximum alignment gap (chunk=1, gap=7)."""

import re
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestChunkOne(unittest.TestCase):
    """
    Model: X[16,1] + bias[1] -> Y[16,1]

    chunk_size=1 is the minimum possible — every data element sits in its own
    8-element aligned block with 7 gap elements.  This is the worst-case
    alignment ratio and tests the extreme of the alignment rounding logic.
    """

    def _gen(self):
        g  = OnnxGraph(_model("chunk_one.onnx"))
        cg = CodeGenerator(g, model_path=_model("chunk_one.onnx"))
        return g, cg

    # ---- node properties ------------------------------------------- #

    def test_single_broadcast_node(self):
        g, _ = self._gen()
        self.assertEqual(len(g.nodes), 1)
        sn = g.nodes[0]
        self.assertEqual(sn.onnx_node.op_type, "Add")
        self.assertEqual(sn.outer_count, 16)

    def test_chunk_size_is_one(self):
        g, _ = self._gen()
        self.assertEqual(g.nodes[0].chunk_size, 1)

    def test_aligned_chunk_equals_align_elems(self):
        """aligned_chunk_size must round 1 up to align_elems (8 for ap_fixed<16,8>)."""
        g, _ = self._gen()
        sn = g.nodes[0]
        self.assertEqual(sn.aligned_chunk_size, 8)

    def test_gap_is_seven(self):
        """7 out of every 8 slots are gap padding — maximum gap ratio."""
        g, _ = self._gen()
        sn = g.nodes[0]
        gap = sn.aligned_chunk_size - sn.chunk_size
        self.assertEqual(gap, 7)

    # ---- alloc sizes ----------------------------------------------- #

    def test_x_alloc(self):
        """X advances: 16 outer × 8 aligned = 128."""
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["X"], 128)

    def test_y_alloc(self):
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["Y"], 128)

    def test_bias_alloc_is_aligned_chunk(self):
        """Repeating bias: one aligned block of 8 (only position 0 holds data)."""
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["bias"], 8)

    # ---- weight ROM: NOT padded (broadcast node, not non-broadcast) -- #

    def test_bias_rom_has_one_element(self):
        """
        bias has numel=1 and is a *repeating* input to a broadcast node.
        Its ROM holds just the one data value; the alloc is 8 elements
        but only position 0 is ever read by the kernel.
        No strided zero-padding is needed here (different from broadcast_chain_bias).
        """
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("_rom_bias[1]", s)
        self.assertNotIn("_rom_bias[8]", s)

    def test_bias_not_in_strided_params(self):
        """bias is used in a broadcast node — no strided weight params needed."""
        _, cg = self._gen()
        self.assertNotIn("bias", cg._strided_weight_params())

    # ---- header macros --------------------------------------------- #

    def test_chunk_macro_is_one(self):
        _, cg = self._gen()
        h = cg.generate_header()
        m = re.search(r"#define\s+INFERENCE_Y_CHUNK\s+(\d+)u", h)
        self.assertIsNotNone(m, "INFERENCE_Y_CHUNK not found")
        self.assertEqual(int(m.group(1)), 1)

    def test_x_size_macro(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("16u * INFERENCE_Y_CHUNK_STRIDE", h)
        self.assertIn("INFERENCE_X_SIZE", h)

    # ---- source: broadcast loop with 16 iterations ----------------- #

    def test_broadcast_run_op_no_for_loop(self):
        """Single run_op() call with outer=16 — no CPU for-loop."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("for (unsigned _i", s)
        self.assertIn("16u, INFERENCE_Y_CHUNK_STRIDE, 0u);", s)

    def test_run_op_size_is_chunk_not_stride(self):
        """size argument must be INFERENCE_Y_CHUNK (=1), not CHUNK_STRIDE (=8)."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK, VECTOROP_ADD", s)
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE, VECTOROP_ADD", s)

    def test_only_run_op_in_source(self):
        """Single broadcast node: only run_op() defined, no run_op_at()."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("static void run_op(", s)
        self.assertNotIn("static void run_op_at(", s)
        self.assertNotIn("run_op_at(", s)

    # ---- test_inference.c: fill iterates chunk-by-chunk ------------ #

    def test_fill_outer_loop_16(self):
        """Fill loop must iterate all 16 chunks."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("for (_chunk = 0u; _chunk < 16u; _chunk++)", t)

    def test_fill_inner_loop_uses_chunk_macro(self):
        """Inner fill loop iterates INFERENCE_Y_CHUNK (=1) times."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("for (i = 0u; i < INFERENCE_Y_CHUNK; i++)", t)

    def test_print_outer_loop_16(self):
        """Print loop iterates up to 16 chunks (stops when shown reaches lim=8)."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("for (_chunk = 0u; _chunk < 16u && shown < lim; _chunk++)", t)

    def test_print_logical_numel(self):
        """numel reported as 16 * CHUNK (not from SIZE which includes gaps)."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("numel = 16u * INFERENCE_Y_CHUNK", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for clip_opset10 model — Clip(min=0, max=6) via opset-10 attributes."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestClipOpset10(unittest.TestCase):
    """
    Model: X[1,64] -> Clip(min=0, max=6) -> Y[1,64]

    This tests the opset-10 Clip variant where min/max are node attributes
    (not input tensors).  The scheduler must recognise this as RELU6 regardless
    of how the bounds are represented in the ONNX protobuf.

    The model is non-broadcast (outer_count=1, flat) with no weight tensors.
    """

    def _gen(self):
        g  = OnnxGraph(_model("clip_opset10.onnx"))
        cg = CodeGenerator(g, model_path=_model("clip_opset10.onnx"))
        return g, cg

    # ---- graph structure ------------------------------------------- #

    def test_single_clip_node(self):
        g, _ = self._gen()
        self.assertEqual(len(g.nodes), 1)
        sn = g.nodes[0]
        self.assertEqual(sn.onnx_node.op_type, "Clip")

    def test_non_broadcast(self):
        """Flat [1,64] layout: no broadcast needed, outer_count=1."""
        g, _ = self._gen()
        self.assertEqual(g.nodes[0].outer_count, 1)

    def test_no_weight_tensors(self):
        """Clip has no trainable parameters."""
        g, _ = self._gen()
        self.assertEqual(len(g.weight_tensors), 0)

    def test_no_intermediate_tensors(self):
        g, _ = self._gen()
        self.assertEqual(len(g.intermediate_tensors), 0)

    # ---- alloc sizes ----------------------------------------------- #

    def test_x_alloc(self):
        """X: flat [1,64] → 64 elements."""
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["X"], 64)

    def test_y_alloc(self):
        g, cg = self._gen()
        self.assertEqual(cg._alloc_sizes["Y"], 64)

    # ---- source: correct op code ----------------------------------- #

    def test_emits_relu6(self):
        """Clip(0, 6) must map to VECTOROP_RELU6."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("VECTOROP_RELU6", s)

    def test_run_op_size_64(self):
        """run_op() must be called with the full 64-element buffer."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("run_op(X, NULL, Y, 64u,", s)

    def test_no_run_op_at(self):
        """Non-broadcast: only run_op(), never run_op_at()."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("run_op_at(", s)

    def test_no_weight_rom(self):
        """No weight arrays should appear in source."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("_rom_", s)

    # ---- header macros --------------------------------------------- #

    def test_no_chunk_macros(self):
        """Non-broadcast model: no CHUNK or CHUNK_STRIDE macros."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertNotIn("_CHUNK", h)

    def test_x_size_macro_flat(self):
        """X size is a flat literal, not expressed in terms of a stride."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_X_SIZE", h)
        self.assertIn("64u", h)
        self.assertNotIn("CHUNK_STRIDE", h)

    def test_y_size_macro_flat(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_SIZE", h)

    # ---- test_inference.c ------------------------------------------ #

    def test_fill_flat_loop(self):
        """Flat (non-broadcast) input: fill uses a simple for-i loop."""
        _, cg = self._gen()
        t = cg.generate_test()
        # flat fill should not use chunk-based loops
        self.assertNotIn("broadcast input", t)

    def test_print_flat_loop(self):
        """Flat output: print uses simple iteration, not chunk loop."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertNotIn("broadcast output", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

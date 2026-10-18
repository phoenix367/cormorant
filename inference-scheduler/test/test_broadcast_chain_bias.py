# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for broadcast_chain_bias model — full-shape bias2 in strided ROM."""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBroadcastChainBias(unittest.TestCase):
    """
    Model: X[4,6] + bias[6] -> add_Y[4,6] -> Relu -> relu_Y[4,6] -> +bias2[4,6] -> Y[4,6]

    Node 0 (Add):   broadcast, outer_count=4, chunk=6, aligned_chunk=8
    Node 1 (Relu):  non-broadcast — run_op with padded size 32
    Node 2 (Add):   non-broadcast — run_op with padded size 32; bias2 must be
                    in strided layout (4×8 ROM with 2-element gap zeros per row)

    Key new property vs broadcast_chain: bias2 has numel=24 but must be
    allocated and emitted as 32 elements so run_op(size=32) never overreads.
    """

    def _gen(self):
        g  = OnnxGraph(_model("broadcast_chain_bias.onnx"))
        cg = CodeGenerator(g, model_path=_model("broadcast_chain_bias.onnx"))
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

    def test_bias2_numel_is_24(self):
        """Sanity: bias2 has the full [4,6] shape, not a broadcast [6] shape."""
        g, _ = self._gen()
        bias2 = g.nodes[2].inputs[1]
        self.assertEqual(bias2.numel, 24)

    # ---- alloc sizes: bias2 raised to strided size ----------------- #

    def test_bias2_alloc_raised_to_strided_size(self):
        """
        bias2.numel == 24 but its alloc must be 32 so run_op(size=32)
        never reads past the end of the buffer.
        """
        g, cg = self._gen()
        bias2 = g.nodes[2].inputs[1]
        self.assertEqual(cg._alloc_sizes[bias2.onnx_name], 32)

    def test_all_data_buffers_alloc_32(self):
        """X, add_Y, relu_Y, Y must all have alloc=32."""
        g, cg = self._gen()
        for name in ("X", "add_Y", "relu_Y", "Y"):
            with self.subTest(tensor=name):
                self.assertEqual(cg._alloc_sizes[name], 32)

    def test_bias_alloc_is_aligned_chunk(self):
        """Broadcast bias [6] stays at aligned_chunk_size (8)."""
        g, cg = self._gen()
        bias_name = g.nodes[0].inputs[1].onnx_name
        self.assertEqual(cg._alloc_sizes[bias_name], 8)

    # ---- strided weight params ------------------------------------- #

    def test_strided_weight_params_detects_bias2(self):
        """_strided_weight_params() must identify bias2 with (outer_count=4, stride=8)."""
        g, cg = self._gen()
        bias2_name = g.nodes[2].inputs[1].onnx_name
        params = cg._strided_weight_params()
        self.assertIn(bias2_name, params)
        outer_count, aligned_chunk = params[bias2_name]
        self.assertEqual(outer_count, 4)
        self.assertEqual(aligned_chunk, 8)

    def test_strided_weight_params_excludes_bias(self):
        """broadcast bias [6] is handled by the broadcast node, not strided params."""
        g, cg = self._gen()
        bias_name = g.nodes[0].inputs[1].onnx_name
        params = cg._strided_weight_params()
        self.assertNotIn(bias_name, params)

    # ---- source: ROM layout and op calls --------------------------- #

    def test_bias2_rom_is_strided(self):
        """bias2 ROM must declare 32 elements, not 24."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("_rom_bias2[32]", s)

    def test_bias2_rom_has_gap_zeros(self):
        """Each 8-element block ends with two 0x0000 gap entries."""
        _, cg = self._gen()
        s = cg.generate_source()
        # Every row of the strided layout ends: data..., 0x0000, 0x0000
        self.assertIn("0x0000, 0x0000", s)

    def test_bias2_rom_not_flat_24(self):
        """Flat 24-element ROM would be wrong — verify it is not emitted."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("_rom_bias2[24]", s)

    def test_node1_relu_uses_padded_size(self):
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("run_op(add_Y, NULL, relu_Y, 32u,", s)

    def test_node2_add_uses_padded_size(self):
        """Final Add: both relu_Y and bias2 are 32-element buffers."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("run_op(relu_Y, bias2, Y, 32u,", s)

    def test_node2_add_not_24(self):
        """Regression: 24u would overread bias2 and miss row 3 of relu_Y."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertNotIn("run_op(relu_Y, bias2, Y, 24u,", s)

    # ---- header: only ADD_Y chunk macros, no Y_CHUNK --------------- #

    def test_add_y_chunk_macros_in_header(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_ADD_Y_CHUNK", h)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    def test_no_y_chunk_macro(self):
        """Y and relu_Y are non-broadcast outputs — no separate CHUNK macros."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertNotIn("INFERENCE_Y_CHUNK ", h)
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE", h)

    def test_y_size_uses_add_y_stride(self):
        """Y inherits the stride from ADD_Y (via non-broadcast propagation)."""
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_SIZE", h)
        self.assertIn("4u * INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    # ---- test_inference.c ------------------------------------------ #

    def test_input_fill_uses_chunk_loop(self):
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("broadcast input, fill data chunks only", t)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", t)

    def test_output_print_uses_chunk_loop(self):
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("broadcast output, print data chunks only", t)
        self.assertIn("for (_chunk = 0u; _chunk < 4u && shown < lim; _chunk++)", t)
        self.assertNotIn("for (i = 0u; i < lim; i++)", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

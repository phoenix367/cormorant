# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for broadcasting model handling (TestBroadcast)."""

import os
import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.nodes   import SchedulerError


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBroadcast(unittest.TestCase):

    def _gen(self, model_name: str):
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, model_path=_model(model_name))
        return g, cg

    # ---------------------------------------------------------------- #
    # broadcast_aligned: X[8,16] + bias[16] -> Y[8,16]                 #
    # chunk=16 elems × 2 bytes = 32 B — already a multiple of 16 B     #
    # ---------------------------------------------------------------- #

    def test_aligned_outer_count(self):
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.outer_count, 8)

    def test_aligned_chunk_size(self):
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.chunk_size, 16)

    def test_aligned_no_padding(self):
        # chunk_size already a multiple of align_elems → no gap
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.aligned_chunk_size, sn.chunk_size)
        self.assertEqual(sn.chunk_size % 8, 0)

    def test_aligned_x_advances(self):
        g, _ = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        self.assertTrue(sn.a_advances)   # X is the advancing input
        self.assertFalse(sn.b_advances)  # bias repeats

    def test_aligned_chunk_macro_in_header(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_Y_CHUNK", h)
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE", h)

    def test_aligned_chunk_stride_uses_align_up(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_ALIGN_UP(INFERENCE_Y_CHUNK)", h)

    def test_aligned_x_size_uses_stride_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        # X is an advancing buffer — SIZE must reference CHUNK_STRIDE
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE", h)
        self.assertIn("INFERENCE_X_SIZE", h)
        # The SIZE expression must multiply outer_count by CHUNK_STRIDE
        self.assertIn("8u * INFERENCE_Y_CHUNK_STRIDE", h)

    def test_aligned_y_size_uses_stride_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("8u * INFERENCE_Y_CHUNK_STRIDE", h)

    def test_aligned_broadcast_uses_run_op(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("run_op(X, bias, Y, INFERENCE_Y_CHUNK, VECTOROP_ADD, 8u,", s)

    def test_aligned_no_for_loop_in_source(self):
        # Broadcasting is now handled by the kernel; no CPU for-loop is emitted.
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertNotIn("for (unsigned _i", s)

    def test_aligned_loop_uses_chunk_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK,", s)

    def test_aligned_loop_uses_stride_macro(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE", s)

    def test_aligned_advancing_uses_stride_inc(self):
        # X (advancing) a_inc must be CHUNK_STRIDE; b_inc must be 0u
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK_STRIDE, 0u);", s)

    def test_aligned_repeating_zero_inc(self):
        # bias (repeating) b_inc must be 0u (last argument)
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn(", 0u);", s)

    def test_aligned_explicit_syncs_in_source(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("inference_buf_sync_to_device(X)", s)
        self.assertIn("inference_buf_sync_from_device(Y)", s)

    def test_aligned_broadcast_run_op_has_outer(self):
        # Broadcast node uses run_op() with outer=8u (not the non-broadcast 1u)
        _, cg = self._gen("broadcast_aligned.onnx")
        s = cg.generate_source()
        self.assertIn("run_op(X, bias, Y,", s)
        self.assertNotIn("run_op_at(", s)
        # The non-broadcast form (outer=1) must NOT be used for this node
        self.assertNotIn("run_op(X, bias, Y, INFERENCE_Y_CHUNK, VECTOROP_ADD, 1u,", s)

    def test_aligned_bias_alloc_uses_aligned_chunk(self):
        # bias (repeating) is allocated with aligned_chunk_size elements
        g, cg = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        alloc_sizes = cg._alloc_sizes
        bias_name = sn.inputs[1].onnx_name  # bias is b input
        expected = sn.aligned_chunk_size
        self.assertEqual(alloc_sizes[bias_name], expected)

    def test_aligned_x_alloc_uses_total_padded(self):
        g, cg = self._gen("broadcast_aligned.onnx")
        sn = g.nodes[0]
        x_name = sn.inputs[0].onnx_name
        expected = sn.outer_count * sn.aligned_chunk_size
        self.assertEqual(cg._alloc_sizes[x_name], expected)

    def test_aligned_alignment_macros_in_header(self):
        _, cg = self._gen("broadcast_aligned.onnx")
        h = cg.generate_header()
        self.assertIn("INFERENCE_ALIGN_BYTES", h)
        self.assertIn("INFERENCE_ALIGN_ELEMS", h)
        self.assertIn("INFERENCE_ALIGN_UP", h)

    # ---------------------------------------------------------------- #
    # broadcast_unaligned: X[4,6] + bias[6] -> Y[4,6]                  #
    # chunk=6 elems × 2 bytes = 12 B — NOT a multiple of 16 B          #
    # aligned_chunk = 8 (gap of 2 elements per block)                   #
    # ---------------------------------------------------------------- #

    def test_unaligned_outer_count(self):
        g, _ = self._gen("broadcast_unaligned.onnx")
        self.assertEqual(g.nodes[0].outer_count, 4)

    def test_unaligned_chunk_size(self):
        g, _ = self._gen("broadcast_unaligned.onnx")
        self.assertEqual(g.nodes[0].chunk_size, 6)

    def test_unaligned_aligned_chunk_has_gap(self):
        g, _ = self._gen("broadcast_unaligned.onnx")
        sn = g.nodes[0]
        self.assertEqual(sn.aligned_chunk_size, 8)
        self.assertGreater(sn.aligned_chunk_size, sn.chunk_size)
        self.assertEqual(sn.aligned_chunk_size % 8, 0)

    def test_unaligned_chunk_macro_value(self):
        # CHUNK macro must hold the raw chunk_size (6), not the padded stride
        _, cg = self._gen("broadcast_unaligned.onnx")
        h = cg.generate_header()
        import re
        m = re.search(r'#define\s+INFERENCE_Y_CHUNK\s+(\d+)u', h)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 6)

    def test_unaligned_x_size_larger_than_numel(self):
        # With padding, X needs more than its natural numel
        g, cg = self._gen("broadcast_unaligned.onnx")
        sn = g.nodes[0]
        x = g.input_tensors[0]
        alloc = cg._alloc_sizes[x.onnx_name]
        self.assertGreater(alloc, x.numel)      # 32 > 24
        self.assertEqual(alloc, sn.outer_count * sn.aligned_chunk_size)  # 4*8=32

    def test_unaligned_no_for_loop_in_source(self):
        _, cg = self._gen("broadcast_unaligned.onnx")
        s = cg.generate_source()
        self.assertNotIn("for (unsigned _i", s)
        self.assertIn("run_op(X, bias, Y, INFERENCE_Y_CHUNK, VECTOROP_ADD, 4u,", s)

    def test_unaligned_run_op_size_is_chunk_not_stride(self):
        # The 'size' argument to run_op_at must be INFERENCE_Y_CHUNK (the raw
        # data count), not INFERENCE_Y_CHUNK_STRIDE (the padded stride).
        # In the call: run_op_at(..., INFERENCE_Y_CHUNK, VECTOROP_ADD)
        # so CHUNK is immediately followed by ", VECTOROP_".
        _, cg = self._gen("broadcast_unaligned.onnx")
        s = cg.generate_source()
        self.assertIn("INFERENCE_Y_CHUNK, VECTOROP_", s)
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE, VECTOROP_", s)

    # ---------------------------------------------------------------- #
    # Alignment macros are emitted even for non-broadcast models        #
    # ---------------------------------------------------------------- #

    def test_alignment_macros_always_present(self):
        for model in ("single_add.onnx", "relu_chain.onnx", "two_input_add.onnx"):
            with self.subTest(model=model):
                _, cg = self._gen(model)
                h = cg.generate_header()
                self.assertIn("INFERENCE_ALIGN_BYTES", h)
                self.assertIn("INFERENCE_ALIGN_ELEMS", h)
                self.assertIn("INFERENCE_ALIGN_UP",    h)

    # ---------------------------------------------------------------- #
    # Non-broadcast models still use plain run_op() (no regression)    #
    # ---------------------------------------------------------------- #

    def test_no_broadcast_uses_run_op_not_run_op_at(self):
        for model in ("single_add.onnx", "relu_chain.onnx"):
            with self.subTest(model=model):
                _, cg = self._gen(model)
                s = cg.generate_source()
                self.assertIn("run_op(", s)
                # run_op_at must not appear at all in non-broadcast source
                # (the function is not emitted, eliminating the unused-function warning).
                self.assertNotIn("run_op_at", s)

    # ---------------------------------------------------------------- #
    # SchedulerError for invalid broadcast shapes                       #
    # ---------------------------------------------------------------- #

    def test_interleaved_broadcast_raises(self):
        """[N,1,H,W] against [N,C,H,W]: broadcast dim is flanked by matches."""
        import numpy as np
        import onnx.helper as oh
        import onnx.numpy_helper as nph
        from onnx import TensorProto

        N, C, H, W = 2, 3, 4, 4
        # bias has shape [N,1,H,W] — dim-1 is 1 (broadcasts to C), others match
        bias_data = np.ones([N, 1, H, W], dtype=np.float32)
        X_info    = oh.make_tensor_value_info("X",    TensorProto.FLOAT, [N, C, H, W])
        Y_info    = oh.make_tensor_value_info("Y",    TensorProto.FLOAT, [N, C, H, W])
        bias_init = nph.from_array(bias_data, name="bias")
        graph = oh.make_graph(
            [oh.make_node("Add", ["X", "bias"], ["Y"])],
            "bad_broadcast",
            inputs=[X_info],
            outputs=[Y_info],
            initializer=[bias_init],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        model.ir_version = 8
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fname = f.name
        try:
            import onnx
            onnx.save(model, fname)
            with self.assertRaises(SchedulerError):
                OnnxGraph(fname)
        finally:
            os.unlink(fname)

    def test_both_inputs_smaller_raises(self):
        """Both A and B smaller than the output — not supported."""
        import onnx.helper as oh
        from onnx import TensorProto

        # A[4] + B[4] -> Y[4] is fine (no broadcast).
        # To get both inputs smaller than output we need the output to be larger.
        # Use A[4] of shape [4] and B[4] of shape [4], but declare output [2,4].
        # ONNX broadcast: [4] + [4] -> [4], not [2,4], so we use a different trick:
        # A[1,4] + B[2,1] -> Y[2,4] — both A and B are smaller than output.
        A_info = oh.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        B_info = oh.make_tensor_value_info("B", TensorProto.FLOAT, [2, 1])
        Y_info = oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        graph = oh.make_graph(
            [oh.make_node("Add", ["A", "B"], ["Y"])],
            "both_broadcast",
            inputs=[A_info, B_info],
            outputs=[Y_info],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        model.ir_version = 8
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fname = f.name
        try:
            import onnx
            onnx.save(model, fname)
            with self.assertRaises(SchedulerError):
                OnnxGraph(fname)
        finally:
            os.unlink(fname)

    # ---------------------------------------------------------------- #
    # generate_test() — broadcast-aware fill and print                  #
    # ---------------------------------------------------------------- #

    def test_broadcast_fill_uses_chunk_loop(self):
        """Input fill uses a chunk-by-chunk loop, not a flat SIZE loop."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        # chunk loop present
        self.assertIn("for (_chunk = 0u; _chunk < 4u; _chunk++)", t)
        self.assertIn("_off = _chunk * INFERENCE_Y_CHUNK_STRIDE", t)
        self.assertIn("for (i = 0u; i < INFERENCE_Y_CHUNK; i++)", t)
        # flat SIZE loop must NOT be used for the fill
        self.assertNotIn("for (i = 0u; i < INFERENCE_X_SIZE; i++)", t)

    def test_broadcast_fill_comment(self):
        """Fill block is annotated as broadcast."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        self.assertIn("broadcast input, fill data chunks only", t)

    def test_broadcast_print_uses_chunk_loop(self):
        """Output print uses a chunk-by-chunk loop, not a flat SIZE loop."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        # chunk print loop present
        self.assertIn("for (_chunk = 0u; _chunk < 4u && shown < lim; _chunk++)", t)
        self.assertIn("_off = _chunk * INFERENCE_Y_CHUNK_STRIDE", t)
        self.assertIn("INFERENCE_Y_CHUNK && shown < lim", t)
        # flat SIZE loop must NOT be used for the print
        self.assertNotIn("for (i = 0u; i < lim; i++)", t)

    def test_broadcast_print_comment(self):
        """Print block is annotated as broadcast."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        self.assertIn("broadcast output, print data chunks only", t)

    def test_broadcast_print_logical_numel(self):
        """printf reports outer_count * CHUNK (logical numel), not SIZE."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        # numel computed as 4 * CHUNK, not from SIZE
        self.assertIn("numel = 4u * INFERENCE_Y_CHUNK", t)
        # SIZE must not appear in printf numel context
        self.assertNotIn("(unsigned)INFERENCE_Y_SIZE", t)

    def test_broadcast_print_shown_counter(self):
        """Print loop uses 'shown' to count valid elements printed."""
        _, cg = self._gen("broadcast_unaligned.onnx")
        t = cg.generate_test()
        self.assertIn("shown = 0u", t)
        self.assertIn("i++, shown++", t)
        # 'shown' is passed as the [%u] index argument to printf
        self.assertIn("shown,", t)

    def test_non_broadcast_test_unchanged(self):
        """Non-broadcast models still use the simple flat fill/print."""
        for model in ("single_add.onnx", "relu_chain.onnx"):
            with self.subTest(model=model):
                _, cg = self._gen(model)
                t = cg.generate_test()
                # flat fill
                self.assertIn("for (i = 0u; i < INFERENCE_X_SIZE; i++)", t)
                # flat print
                self.assertIn("for (i = 0u; i < lim; i++)", t)
                # no chunk machinery
                self.assertNotIn("_chunk", t)
                self.assertNotIn("_off", t)


# ------------------------------------------------------------------ #
# Mixed model: broadcast Add + non-broadcast Relu                    #
# X[4,6] + bias[6] -> add_Y[4,6] -> Relu -> Y[4,6]                  #
# ------------------------------------------------------------------ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestBroadcastMixed(unittest.TestCase):
    """
    Model with one broadcast node (Add) and one non-broadcast node (Relu).
    Unique properties not covered by single-node broadcast models:
      - both run_op() and run_op_at() must be defined in the source
      - add_Y (intermediate) is allocated with the padded broadcast size
      - Y (Relu output) uses plain numel — no padding
      - test_inference.c uses chunk-loop fill for X but flat print for Y
    """

    def _gen(self):
        g  = OnnxGraph(_model("broadcast_mixed.onnx"))
        cg = CodeGenerator(g, model_path=_model("broadcast_mixed.onnx"))
        return g, cg

    # ---- graph structure ------------------------------------------- #

    def test_two_nodes(self):
        g, _ = self._gen()
        self.assertEqual(len(g.nodes), 2)

    def test_node0_is_broadcast(self):
        g, _ = self._gen()
        sn = g.nodes[0]
        self.assertEqual(sn.onnx_node.op_type, "Add")
        self.assertEqual(sn.outer_count, 4)
        self.assertEqual(sn.chunk_size, 6)
        self.assertEqual(sn.aligned_chunk_size, 8)

    def test_node1_is_not_broadcast(self):
        g, _ = self._gen()
        sn = g.nodes[1]
        self.assertEqual(sn.onnx_node.op_type, "Relu")
        self.assertEqual(sn.outer_count, 1)

    # ---- source: both helpers present ------------------------------ #

    def test_only_run_op_helper_defined(self):
        """All nodes use run_op() — no run_op_at() emitted."""
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn("static void run_op(", s)
        self.assertNotIn("static void run_op_at(", s)

    def test_add_emits_broadcast_run_op(self):
        _, cg = self._gen()
        s = cg.generate_source()
        self.assertIn(
            "run_op(X, bias, add_Y, INFERENCE_ADD_Y_CHUNK, VECTOROP_ADD, 4u,", s
        )
        self.assertNotIn("run_op_at(", s)

    def test_relu_emits_plain_run_op(self):
        _, cg = self._gen()
        s = cg.generate_source()
        # Relu output inherits the padded alloc (32); run_op must use 32u so
        # it covers the full strided buffer (4 rows × 8-elem stride), not just
        # the logical 24 elements which would miss row 3 and read gap elements.
        self.assertIn("run_op(add_Y, NULL, Y, 32u,", s)

    # ---- header: chunk macros for Add output only ------------------ #

    def test_chunk_macros_named_after_add_y(self):
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("INFERENCE_ADD_Y_CHUNK", h)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", h)

    def test_x_size_uses_add_y_chunk_stride(self):
        # X advances through the broadcast Add node
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("4u * INFERENCE_ADD_Y_CHUNK_STRIDE", h)
        self.assertIn("INFERENCE_X_SIZE", h)

    def test_y_size_uses_add_y_stride(self):
        # Y inherits add_Y's stride; its SIZE must reference the same
        # CHUNK_STRIDE so the caller allocates the full padded buffer.
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertIn("4u * INFERENCE_ADD_Y_CHUNK_STRIDE", h)
        self.assertIn("INFERENCE_Y_SIZE", h)

    def test_no_y_chunk_macro(self):
        # Y has no CHUNK macro of its own — it reuses ADD_Y's macros.
        _, cg = self._gen()
        h = cg.generate_header()
        self.assertNotIn("INFERENCE_Y_CHUNK ", h)   # not a separate definition
        self.assertNotIn("INFERENCE_Y_CHUNK_STRIDE", h)

    # ---- alloc sizes ----------------------------------------------- #

    def test_add_y_alloc_is_padded(self):
        # add_Y intermediate buffer: 4 outer × 8 aligned = 32 elements
        g, cg = self._gen()
        add_y = g.nodes[0].output
        self.assertEqual(cg._alloc_sizes[add_y.onnx_name], 32)

    def test_x_alloc_is_padded(self):
        # X advances → also 4 × 8 = 32
        g, cg = self._gen()
        x = g.input_tensors[0]
        self.assertEqual(cg._alloc_sizes[x.onnx_name], 32)

    def test_y_alloc_inherits_padding(self):
        # Y is Relu output; its alloc inherits the padded size from add_Y
        # so run_op() can process the full strided buffer (4 × 8 = 32).
        g, cg = self._gen()
        y = g.output_tensors[0]
        self.assertEqual(cg._alloc_sizes[y.onnx_name], 32)

    # ---- test_inference.c: mixed fill/print ------------------------ #

    def test_input_fill_uses_chunk_loop(self):
        """X (broadcast advancing input) must use chunk-by-chunk fill."""
        _, cg = self._gen()
        t = cg.generate_test()
        self.assertIn("broadcast input, fill data chunks only", t)
        self.assertIn("for (_chunk = 0u; _chunk < 4u; _chunk++)", t)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", t)

    def test_output_print_uses_chunk_loop(self):
        """Y inherits add_Y's stride so test printing must skip gap elements."""
        _, cg = self._gen()
        t = cg.generate_test()
        # Y print uses the same chunk/stride macros as the broadcast Add node
        self.assertIn("for (_chunk = 0u; _chunk < 4u && shown < lim; _chunk++)", t)
        self.assertIn("INFERENCE_ADD_Y_CHUNK_STRIDE", t)
        # flat loop must NOT be used for Y (would print gap elements)
        self.assertNotIn("for (i = 0u; i < lim; i++)", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

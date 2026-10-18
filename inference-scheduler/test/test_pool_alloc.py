# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for single-pool DMA buffer allocation in generated inference.c (TestPoolAlloc).

Verifies that inference_init() allocates one contiguous pool, carves
sub-buffers via inference_buf_init_view() with correct offsets and counts,
emits static backing structs, and that inference_deinit() frees only the
pool handle — not individual sub-buffers.
"""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator
from src.tensor  import _sanitize_c_name


def _cg(model_name: str) -> CodeGenerator:
    g = OnnxGraph(_model(model_name))
    return CodeGenerator(g, model_path=_model(model_name))


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestPoolAlloc(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # Exactly one inference_buf_alloc() call                              #
    # ------------------------------------------------------------------ #

    def test_single_alloc_single_weight(self):
        s = _cg("single_add.onnx").generate_source()
        self.assertEqual(s.count("inference_buf_alloc("), 1)

    def test_single_alloc_multi_weight(self):
        s = _cg("gemm_with_bias.onnx").generate_source()
        self.assertEqual(s.count("inference_buf_alloc("), 1)

    def test_single_alloc_gemm_chain(self):
        s = _cg("gemm_chain.onnx").generate_source()
        self.assertEqual(s.count("inference_buf_alloc("), 1)

    def test_single_alloc_conv_model(self):
        s = _cg("conv_then_add_flat.onnx").generate_source()
        self.assertEqual(s.count("inference_buf_alloc("), 1)

    # ------------------------------------------------------------------ #
    # Alloc element count matches _compute_pool_layout() total            #
    # ------------------------------------------------------------------ #

    def test_alloc_count_single_weight(self):
        cg = _cg("single_add.onnx")
        _, total = cg._compute_pool_layout()
        self.assertIn(f"inference_buf_alloc({total}u)", cg.generate_source())

    def test_alloc_count_gemm_with_bias(self):
        cg = _cg("gemm_with_bias.onnx")
        _, total = cg._compute_pool_layout()
        self.assertIn(f"inference_buf_alloc({total}u)", cg.generate_source())

    def test_alloc_count_gemm_chain(self):
        cg = _cg("gemm_chain.onnx")
        _, total = cg._compute_pool_layout()
        self.assertIn(f"inference_buf_alloc({total}u)", cg.generate_source())

    # ------------------------------------------------------------------ #
    # inference_buf_init_view() — count matches number of layout slots    #
    # ------------------------------------------------------------------ #

    def test_init_view_count_single_weight(self):
        cg = _cg("single_add.onnx")
        layout, _ = cg._compute_pool_layout()
        self.assertEqual(cg.generate_source().count("inference_buf_init_view("), len(layout))

    def test_init_view_count_gemm_with_bias(self):
        cg = _cg("gemm_with_bias.onnx")
        layout, _ = cg._compute_pool_layout()
        self.assertEqual(cg.generate_source().count("inference_buf_init_view("), len(layout))

    def test_init_view_count_gemm_chain(self):
        cg = _cg("gemm_chain.onnx")
        layout, _ = cg._compute_pool_layout()
        self.assertEqual(cg.generate_source().count("inference_buf_init_view("), len(layout))

    # ------------------------------------------------------------------ #
    # Correct offsets and sizes in each init_view call                    #
    # ------------------------------------------------------------------ #

    def _assert_init_views(self, model_name: str):
        cg = _cg(model_name)
        layout, _ = cg._compute_pool_layout()
        s = cg.generate_source()
        for onnx_name, offset, alloc in layout:
            c = _sanitize_c_name(onnx_name)
            expected = (
                f"inference_buf_init_view(&_s_buf_{c}, s_alloc_pool, {offset}u, {alloc}u)"
            )
            self.assertIn(expected, s, msg=f"init_view wrong for {c}: expected offset={offset}, alloc={alloc}")

    def test_init_view_args_single_weight(self):
        self._assert_init_views("single_add.onnx")

    def test_init_view_args_gemm_with_bias(self):
        self._assert_init_views("gemm_with_bias.onnx")

    def test_init_view_args_gemm_chain(self):
        self._assert_init_views("gemm_chain.onnx")

    def test_init_view_args_conv_model(self):
        self._assert_init_views("conv_then_add_flat.onnx")

    # ------------------------------------------------------------------ #
    # Every slot offset is 64-byte aligned                                #
    # ------------------------------------------------------------------ #

    def _assert_offsets_aligned(self, model_name: str):
        cg = _cg(model_name)
        bpe = cg._dtype.bytes_per_elem
        align_elems = 64 // bpe
        layout, _ = cg._compute_pool_layout()
        for onnx_name, offset, _ in layout:
            self.assertEqual(
                offset % align_elems, 0,
                msg=f"{onnx_name}: offset {offset} not aligned to {align_elems} elements (64 bytes)",
            )

    def test_offsets_aligned_gemm_with_bias(self):
        self._assert_offsets_aligned("gemm_with_bias.onnx")

    def test_offsets_aligned_gemm_chain(self):
        self._assert_offsets_aligned("gemm_chain.onnx")

    def test_offsets_aligned_conv_model(self):
        self._assert_offsets_aligned("conv_then_add_flat.onnx")

    # ------------------------------------------------------------------ #
    # No two slots overlap                                                 #
    # ------------------------------------------------------------------ #

    def _assert_no_overlap(self, model_name: str):
        cg = _cg(model_name)
        layout, _ = cg._compute_pool_layout()
        intervals = cg._compute_live_intervals()
        for i, (n1, off1, sz1) in enumerate(layout):
            for n2, off2, sz2 in layout[i + 1:]:
                pool_overlap = off1 < off2 + sz2 and off2 < off1 + sz1
                if not pool_overlap:
                    continue
                # Pool ranges overlap: only valid when both tensors are
                # intermediates whose live intervals are disjoint.
                iv1 = intervals.get(n1)
                iv2 = intervals.get(n2)
                if iv1 is None or iv2 is None:
                    self.fail(
                        f"Pool overlap for non-intermediate: "
                        f"{n1}@[{off1},{off1+sz1}) vs {n2}@[{off2},{off2+sz2})"
                    )
                live_overlap = iv1[0] <= iv2[1] and iv2[0] <= iv1[1]
                self.assertFalse(
                    live_overlap,
                    msg=(
                        f"Live+pool overlap: {n1} live={iv1} pool=[{off1},{off1+sz1}) "
                        f"vs {n2} live={iv2} pool=[{off2},{off2+sz2})"
                    ),
                )

    def test_no_overlap_gemm_with_bias(self):
        self._assert_no_overlap("gemm_with_bias.onnx")

    def test_no_overlap_gemm_chain(self):
        self._assert_no_overlap("gemm_chain.onnx")

    def test_no_overlap_conv_model(self):
        self._assert_no_overlap("conv_then_add_flat.onnx")

    # ------------------------------------------------------------------ #
    # Static backing struct declarations (_s_buf_*)                       #
    # ------------------------------------------------------------------ #

    def _assert_backing_structs(self, model_name: str):
        cg = _cg(model_name)
        layout, _ = cg._compute_pool_layout()
        s = cg.generate_source()
        for onnx_name, _, _ in layout:
            c = _sanitize_c_name(onnx_name)
            self.assertIn(
                f"static inference_buf_t _s_buf_{c};", s,
                msg=f"Backing struct _s_buf_{c} missing",
            )

    def test_backing_structs_single_weight(self):
        self._assert_backing_structs("single_add.onnx")

    def test_backing_structs_gemm_with_bias(self):
        self._assert_backing_structs("gemm_with_bias.onnx")

    def test_backing_structs_gemm_chain(self):
        self._assert_backing_structs("gemm_chain.onnx")

    # ------------------------------------------------------------------ #
    # Reshape aliases excluded from pool                                   #
    # ------------------------------------------------------------------ #

    def test_reshape_alias_absent_from_layout(self):
        # Z is a reshape alias in reshape_then_matmul — must not get a pool slot
        cg = _cg("reshape_then_matmul.onnx")
        layout, _ = cg._compute_pool_layout()
        onnx_names = {name for name, _, _ in layout}
        self.assertNotIn("Z", onnx_names)

    def test_reshape_alias_has_no_backing_struct(self):
        s = _cg("reshape_then_matmul.onnx").generate_source()
        self.assertNotIn("_s_buf_Z", s)

    def test_reshape_alias_has_no_init_view_call(self):
        s = _cg("reshape_then_matmul.onnx").generate_source()
        self.assertNotIn("inference_buf_init_view(&_s_buf_Z", s)

    def test_reshape_alias_assigned_not_allocated(self):
        s = _cg("reshape_then_matmul.onnx").generate_source()
        # Source is a graph input → assignment is in inference_run(), not inference_init()
        self.assertIn("Z = X;  /* input reshape alias */", s)

    # ------------------------------------------------------------------ #
    # Deinit — pool freed once, sub-buffers never freed individually      #
    # ------------------------------------------------------------------ #

    def test_exactly_one_buf_free_call(self):
        s = _cg("gemm_with_bias.onnx").generate_source()
        self.assertEqual(s.count("inference_buf_free("), 1)

    def test_pool_freed_not_sub_buffers(self):
        cg = _cg("gemm_with_bias.onnx")
        layout, _ = cg._compute_pool_layout()
        s = cg.generate_source()
        self.assertIn("inference_buf_free(s_alloc_pool)", s)
        for onnx_name, _, _ in layout:
            c = _sanitize_c_name(onnx_name)
            self.assertNotIn(f"inference_buf_free({c})", s)

    def test_pool_ptr_nulled_in_deinit(self):
        s = _cg("gemm_with_bias.onnx").generate_source()
        self.assertIn("s_alloc_pool = NULL", s)

    def test_pool_freed_gemm_chain(self):
        s = _cg("gemm_chain.onnx").generate_source()
        self.assertEqual(s.count("inference_buf_free("), 1)
        self.assertIn("inference_buf_free(s_alloc_pool)", s)

    # ------------------------------------------------------------------ #
    # pool_bytes reported in inference.h covers total allocation          #
    # ------------------------------------------------------------------ #

    def test_pool_bytes_covers_alloc_gemm_with_bias(self):
        cg = _cg("gemm_with_bias.onnx")
        _, total = cg._compute_pool_layout()
        self.assertGreaterEqual(cg._compute_pool_bytes(), total * cg._dtype.bytes_per_elem)

    def test_pool_bytes_covers_alloc_gemm_chain(self):
        cg = _cg("gemm_chain.onnx")
        _, total = cg._compute_pool_layout()
        self.assertGreaterEqual(cg._compute_pool_bytes(), total * cg._dtype.bytes_per_elem)


if __name__ == "__main__":
    unittest.main(verbosity=2)

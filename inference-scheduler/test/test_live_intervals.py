# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for live-interval computation and buffer-slot reuse.

Verifies that _compute_live_intervals() returns correct produce/consume
indices, and that _compute_pool_layout() packs non-overlapping intervals
into the same slot to minimise DMA pool size.
"""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator


def _cg(model_name: str) -> CodeGenerator:
    g = OnnxGraph(_model(model_name))
    return CodeGenerator(g, model_path=_model(model_name))


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestLiveIntervals(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # Interval basic properties                                            #
    # ------------------------------------------------------------------ #

    def test_intervals_produce_le_consume(self):
        """produce_idx <= last_consume_idx for every intermediate."""
        for m in ("gemm_chain.onnx", "gemm_with_bias.onnx", "single_add.onnx"):
            ivs = _cg(m)._compute_live_intervals()
            for name, (prod, cons) in ivs.items():
                self.assertLessEqual(
                    prod, cons,
                    msg=f"{m} / {name}: produce={prod} > consume={cons}",
                )

    def test_no_weights_in_intervals(self):
        """Weight tensors must not appear in live intervals."""
        cg = _cg("gemm_chain.onnx")
        ivs = cg._compute_live_intervals()
        weight_names = {t.onnx_name for t in cg._graph.weight_tensors}
        for name in ivs:
            self.assertNotIn(
                name, weight_names,
                msg=f"Weight tensor '{name}' must not appear in live intervals",
            )

    def test_no_reshape_aliases_in_intervals(self):
        """Reshape alias tensors must not appear in live intervals."""
        cg = _cg("reshape_then_matmul.onnx")
        ivs = cg._compute_live_intervals()
        aliases = set(cg._reshape_aliases)
        for name in ivs:
            self.assertNotIn(
                name, aliases,
                msg=f"Reshape alias '{name}' must not appear in live intervals",
            )

    def test_intervals_cover_all_non_alias_intermediates(self):
        """Every non-alias intermediate has an interval entry."""
        for m in ("gemm_chain.onnx", "gemm_with_bias.onnx"):
            cg = _cg(m)
            ivs = cg._compute_live_intervals()
            aliases = set(cg._reshape_aliases)
            for t in cg._graph.intermediate_tensors:
                if t.onnx_name in aliases:
                    continue
                self.assertIn(
                    t.onnx_name, ivs,
                    msg=f"{m}: intermediate '{t.onnx_name}' missing from live intervals",
                )

    # ------------------------------------------------------------------ #
    # Interval index monotonicity across a linear chain                   #
    # ------------------------------------------------------------------ #

    def test_gemm_chain_ordered_intervals(self):
        """In a linear gemm chain each intermediate's interval starts
        strictly after the previous one's."""
        cg = _cg("gemm_chain.onnx")
        ivs = cg._compute_live_intervals()
        # Sort by start index
        ordered = sorted(ivs.values(), key=lambda iv: iv[0])
        for i in range(len(ordered) - 1):
            self.assertLess(
                ordered[i][0], ordered[i + 1][0],
                msg="Consecutive intermediates should have distinct start indices",
            )

    # ------------------------------------------------------------------ #
    # Buffer reuse: shared slots                                           #
    # ------------------------------------------------------------------ #

    def _shared_pairs(self, model_name: str) -> int:
        """Count how many layout-entry pairs share the same pool offset."""
        layout, _ = _cg(model_name)._compute_pool_layout()
        count = 0
        for i, (_, off1, _) in enumerate(layout):
            for _, off2, _ in layout[i + 1:]:
                if off1 == off2:
                    count += 1
        return count

    def test_reuse_occurs_in_gemm_chain(self):
        """A 4-node gemm chain produces 3 intermediates; at least one pair
        should be packed into the same slot."""
        self.assertGreater(
            self._shared_pairs("gemm_chain.onnx"), 0,
            msg="Expected at least one shared slot in gemm_chain",
        )

    def test_reuse_occurs_in_conv_relu_chain(self):
        self.assertGreater(
            self._shared_pairs("conv_relu_chain.onnx"), 0,
            msg="Expected at least one shared slot in conv_relu_chain",
        )

    def test_no_spurious_reuse_in_single_intermediate(self):
        """A model with only one intermediate can never share."""
        self.assertEqual(
            self._shared_pairs("gemm_with_bias.onnx"), 0,
            msg="Single-intermediate model should have no shared slots",
        )

    # ------------------------------------------------------------------ #
    # Pool size reduction                                                  #
    # ------------------------------------------------------------------ #

    def _pool_size_without_reuse(self, model_name: str) -> int:
        """Baseline pool size if all intermediates were sequential."""
        cg = _cg(model_name)
        bpe = cg._dtype.bytes_per_elem
        align_to = 64 // bpe
        def align_up(n):
            return (n + align_to - 1) & ~(align_to - 1)
        aliases = set(cg._reshape_aliases)
        weight_total = sum(
            align_up(cg._alloc_sizes[t.onnx_name])
            for t in cg._graph.weight_tensors
        )
        inter_total = sum(
            align_up(cg._alloc_sizes[t.onnx_name])
            for t in cg._graph.intermediate_tensors
            if t.onnx_name not in aliases
        )
        return weight_total + inter_total

    def test_pool_size_not_larger_than_sequential(self):
        """After reuse optimisation the pool must never grow."""
        for m in ("gemm_chain.onnx", "conv_relu_chain.onnx", "gemm_with_bias.onnx"):
            _, optimised = _cg(m)._compute_pool_layout()
            baseline = self._pool_size_without_reuse(m)
            self.assertLessEqual(
                optimised, baseline,
                msg=f"{m}: optimised={optimised} > baseline={baseline}",
            )

    def test_pool_size_strictly_smaller_for_gemm_chain(self):
        """The gemm_chain has provably reusable intermediates."""
        _, optimised = _cg("gemm_chain.onnx")._compute_pool_layout()
        baseline = self._pool_size_without_reuse("gemm_chain.onnx")
        self.assertLess(
            optimised, baseline,
            msg=f"Expected pool reduction; optimised={optimised} baseline={baseline}",
        )

    # ------------------------------------------------------------------ #
    # Shared slots must not have overlapping live intervals                #
    # ------------------------------------------------------------------ #

    def test_shared_slots_have_disjoint_intervals(self):
        """Any two layout entries at the same offset must have non-overlapping
        live intervals."""
        for m in ("gemm_chain.onnx", "conv_relu_chain.onnx",
                  "mixed_all_conv_matmul_relu.onnx"):
            cg = _cg(m)
            layout, _ = cg._compute_pool_layout()
            ivs = cg._compute_live_intervals()
            for i, (n1, off1, _) in enumerate(layout):
                for n2, off2, _ in layout[i + 1:]:
                    if off1 != off2:
                        continue
                    iv1, iv2 = ivs[n1], ivs[n2]
                    live_overlap = iv1[0] <= iv2[1] and iv2[0] <= iv1[1]
                    self.assertFalse(
                        live_overlap,
                        msg=(
                            f"{m}: {n1} live={iv1} and {n2} live={iv2} "
                            f"share offset {off1} but have overlapping intervals"
                        ),
                    )

    # ------------------------------------------------------------------ #
    # Slot alloc must fit all tensors assigned to it                      #
    # ------------------------------------------------------------------ #

    def test_slot_alloc_fits_all_tenors(self):
        """For each offset that is shared, the slot-level allocation (derived
        from the backing struct count) must be >= every tenant's alloc."""
        for m in ("gemm_chain.onnx", "conv_relu_chain.onnx"):
            cg = _cg(m)
            bpe = cg._dtype.bytes_per_elem
            align_to = 64 // bpe
            def align_up(n, align_to=align_to):
                return (n + align_to - 1) & ~(align_to - 1)
            layout, _ = cg._compute_pool_layout()
            # Group by offset
            by_offset: dict = {}
            for name, off, alloc in layout:
                by_offset.setdefault(off, []).append(alloc)
            # For shared slots, find the next-offset and infer slot_alloc
            offsets = sorted(by_offset)
            for i, off in enumerate(offsets):
                allocs = by_offset[off]
                if len(allocs) < 2:
                    continue
                # slot_alloc must be >= every tenant's alloc
                max_needed = max(align_up(a) for a in allocs)
                # Infer slot_alloc from the gap to the next offset (if any)
                if i + 1 < len(offsets):
                    gap = offsets[i + 1] - off
                    self.assertGreaterEqual(
                        gap, max_needed,
                        msg=f"{m}: slot at offset {off} gap={gap} < needed {max_needed}",
                    )

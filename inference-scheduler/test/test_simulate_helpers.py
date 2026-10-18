# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for src.codegen._simulate helper functions and
edge-case paths.

Targets the 9 previously-uncovered lines (87, 161, 165, 193, 529, 618,
633–637) by exercising:
  * _residual_stats with a zero target + non-zero residual,
  * _depthwise_conv2d_ref boundary skips (ih/iw < 0 with padding),
  * _pool_poly_sqrt with all-non-positive input,
  * _forward_pass defensive ValueError on an unknown op_code,
  * large_expected_tensors short-circuit when embed_large_expected=True,
  * generate_expected_dat end-to-end.
"""

import unittest

import numpy as np

from helpers import _model, _models_exist
from src.codegen._simulate import (
    _residual_stats,
    _depthwise_conv2d_ref,
    _pool_poly_sqrt,
)
from src.codegen import CodeGenerator
from src.graph   import OnnxGraph


# ================================================================ #
# _residual_stats — zero-target / non-zero-residual edge case       #
# ================================================================ #

class TestResidualStatsZeroTarget(unittest.TestCase):
    def test_zero_full_with_nonzero_residual(self):
        # norm_w ≈ 0 but norm_r > 0 → line 87: nrmse=inf, sqnr_db=-inf.
        full  = np.zeros(4, dtype=np.float64)
        quant = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        stats = _residual_stats(full, quant)
        self.assertEqual(stats.nrmse,   float("inf"))
        self.assertEqual(stats.sqnr_db, float("-inf"))
        self.assertGreater(stats.abs_max, 0.0)

    def test_zero_full_with_zero_residual_still_reachable(self):
        # Defensive: keep the symmetric branch exercised too.
        full  = np.zeros(4, dtype=np.float64)
        quant = np.zeros(4, dtype=np.float64)
        stats = _residual_stats(full, quant)
        self.assertEqual(stats.nrmse,   0.0)
        self.assertEqual(stats.sqnr_db, float("inf"))


# ================================================================ #
# _depthwise_conv2d_ref — pad-boundary skip branches                #
# ================================================================ #

class TestDepthwiseConv2dRefBoundary(unittest.TestCase):
    def test_pad_top_and_pad_left_skip_branches(self):
        # 3x3 kernel on 3x3 input with pad_top=1, pad_left=1:
        # - oh=0, khi=0 → ih = 0*1 + 0*1 - 1 = -1  →  line 161 continue
        # - ow=0, kwi=0 (with ih in-range) → iw=-1 →  line 165 continue
        x = np.ones((1, 2, 3, 3), dtype=np.float64)
        w = np.ones((2, 1, 3, 3), dtype=np.float64)
        y = _depthwise_conv2d_ref(
            x, w, bias=None,
            stride_h=1, stride_w=1,
            pad_top=1, pad_left=1,
            dilation_h=1, dilation_w=1,
            out_h=3, out_w=3,
        )
        self.assertEqual(y.shape, (1, 2, 3, 3))
        # Top-left corner sees only a 2x2 sub-window (4 contributions of 1·1).
        self.assertEqual(y[0, 0, 0, 0], 4.0)
        # Centre sees the full 3x3 window (9 contributions).
        self.assertEqual(y[0, 0, 1, 1], 9.0)


# ================================================================ #
# _pool_poly_sqrt — all-non-positive input                          #
# ================================================================ #

class TestPoolPolySqrtAllNonPositive(unittest.TestCase):
    def test_all_zero_short_circuits(self):
        # not np.any(pos) → returns the pre-allocated zeros (line 193).
        acc = np.zeros(8, dtype=np.float64)
        result = _pool_poly_sqrt(acc)
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_all_negative_short_circuits(self):
        acc = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        result = _pool_poly_sqrt(acc)
        np.testing.assert_array_equal(result, np.zeros(3))


# ================================================================ #
# CodeGenerator paths — needs ONNX fixtures                          #
# ================================================================ #

@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSimulateMixinPaths(unittest.TestCase):
    """Exercises the three CodeGenerator-bound uncovered paths."""

    def _cg(self, **kwargs):
        path = _model("single_add.onnx")
        g    = OnnxGraph(path)
        return CodeGenerator(g, model_path=path, **kwargs)

    def test_large_expected_tensors_empty_with_embed_flag(self):
        # Line 618 — short-circuit when embed_large_expected=True.
        cg = self._cg(embed_large_expected=True)
        self.assertEqual(cg.large_expected_tensors, [])

    def test_generate_expected_dat_returns_byte_buffer(self):
        # Lines 633–637 — end-to-end call.
        cg  = self._cg()
        out = cg._graph.output_tensors[0]
        buf = cg.generate_expected_dat(out)
        self.assertIsInstance(buf, (bytes, bytearray))
        self.assertGreater(len(buf), 0)
        # Length is the storage buffer's alloc count × bytes_per_elem.
        self.assertEqual(len(buf) % cg._dtype.bytes_per_elem, 0)

    def test_forward_pass_unknown_op_code_raises(self):
        # Line 529 — defensive raise on impossible op_code.
        cg   = self._cg()
        node = cg._graph.nodes[0]
        # Mutate the ScheduledNode's op_code to something not in OP_*.
        node.op_code = 99
        with self.assertRaisesRegex(ValueError, "unknown op_code"):
            cg._simulate()


if __name__ == "__main__":
    unittest.main()
